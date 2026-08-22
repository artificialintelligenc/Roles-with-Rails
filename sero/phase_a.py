"""
Phase A: Inference Pipeline

Phase A runs at task-solve time (frozen policy):
  1. Role retrieval: rank pool by (cosine + alpha_s * c̄) → top-N active set
  2. Credit-ranked DAG construction
  3. Multi-round message passing in topological order
  4. Aggregator produces final answer
  5. Credit computation (Fast Credit per-round, LOO Precise Credit for one targeted role)

Returns the final answer string + credit update data for the controller.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from sero.role_card import RoleCard
from sero.credit_engine import CreditEngine, compute_round_fast_credits, loo_precise_credit
from sero.dag_builder import build_credit_dag, topological_order, get_incoming_messages, compute_dag_levels
from sero.openrouter_client import OpenRouterClient
from sero.config import SeroConfig
from sero.benchmarks.scoring_utils import normalize_score

# Maximum parallel agent calls within one Phase A round
_MAX_PARALLEL_AGENTS = 4
# Maximum parallel tasks for batch evaluation
_MAX_PARALLEL_TASKS = 5
_TERMINAL_ROLE_NAME_HINTS = ("aggregator", "synthesizer")
_TERMINAL_ROLE_CAPABILITY_HINTS = {"aggregation", "synthesis"}
_FINAL_ROLE_TEXT_HINTS = (
    "you are the final",
    "produce the final",
    "final answer",
    "final plan",
    "final schedule",
)
_STRUCTURED_TASK_HINTS = (
    "trip",
    "travel",
    "flight",
    "meeting",
    "calendar",
    "schedule",
    "availability",
    "constraint",
)
_STRUCTURED_OUTPUT_HINTS = (
    "exactly",
    "output only",
    "only these",
    "hh:mm",
    "day x-y",
    "you travel to",
)
_VALIDATOR_REVISION_NEGATIVE_HINTS = (
    "invalid",
    "conflict",
    "violation",
    "violates",
    "incorrect",
    "not valid",
    "not feasible",
    "not available",
    "unavailable",
    "wrong",
    "fails",
    "issue",
    "problem",
    "error",
    "suggest",
    "recommend",
    "corrected",
    "instead",
)
_VALIDATOR_REVISION_POSITIVE_HINTS = (
    "is valid",
    "are valid",
    "no conflict",
    "no conflicts",
    "meets the duration requirement",
    "meets all constraints",
    "preference is respected",
    "preferences are respected",
    "recommended to schedule",
    "is recommended",
)


def _role_supports_final_output(role: RoleCard) -> bool:
    text = " ".join([
        role.system_prompt,
        role.communication_protocol,
        " ".join(role.capability_tags),
    ]).lower()
    return any(hint in text for hint in _FINAL_ROLE_TEXT_HINTS)


def _role_has_strict_output_format(role: RoleCard) -> bool:
    text = " ".join([
        role.system_prompt,
        role.communication_protocol,
    ]).lower()
    return any(hint in text for hint in _STRUCTURED_OUTPUT_HINTS)


def select_aggregator_name(active_roles: List[RoleCard]) -> str:
    """Select the terminal role for final answer synthesis.

    Prefer explicit role_type="aggregator" semantics. Heuristic fallbacks are
    kept only for backward compatibility with legacy role cards that predate
    the explicit terminal-role field.
    """
    if not active_roles:
        return ""

    explicit_aggregators = [
        role.name for role in active_roles
        if role.role_type == "aggregator"
    ]
    if explicit_aggregators:
        return explicit_aggregators[-1]

    name_candidates = [
        role.name for role in active_roles
        if any(hint in role.name.lower() for hint in _TERMINAL_ROLE_NAME_HINTS)
    ]
    if name_candidates:
        return name_candidates[-1]

    tag_candidates = [
        role.name for role in active_roles
        if any(tag.lower() in _TERMINAL_ROLE_CAPABILITY_HINTS for tag in role.capability_tags)
    ]
    if tag_candidates:
        return tag_candidates[-1]

    final_role_candidates = [
        role.name for role in active_roles
        if role.protected and _role_supports_final_output(role)
    ]
    if final_role_candidates:
        return final_role_candidates[-1]

    return active_roles[-1].name


def select_validator_name(
    pool: List[RoleCard],
    credit_engine: CreditEngine,
    exclude_names: Optional[List[str]] = None,
) -> str:
    exclude = set(exclude_names or [])
    validators = [
        role for role in pool
        if role.role_type == "validator" and role.name not in exclude
    ]
    if not validators:
        return ""
    validators.sort(
        key=lambda role: (credit_engine.get_ema(role.name), role.protected),
        reverse=True,
    )
    return validators[0].name


def _should_run_validator_pass(
    config: SeroConfig,
    task_prompt: str,
    aggregator_role: Optional[RoleCard],
    answer: str,
) -> bool:
    if (
        not config.conditional_validator_pass
        or not config.post_aggregator_validator_check
        or aggregator_role is None
    ):
        return False
    if not answer.strip():
        return True
    if _role_has_strict_output_format(aggregator_role):
        return True
    task_lower = task_prompt.lower()
    return any(hint in task_lower for hint in _STRUCTURED_TASK_HINTS)


def _collect_response_texts(active_names: List[str], all_responses: Dict[str, str]) -> List[str]:
    missing = [name for name in active_names if name not in all_responses]
    if missing:
        raise KeyError(f"Missing responses for active roles: {missing}")
    return [all_responses[name] for name in active_names]


def _fallback_final_answer(
    topo_order: List[str],
    aggregator_name: str,
    all_responses: Dict[str, str],
) -> Optional[str]:
    for role_name in reversed(topo_order):
        if role_name == aggregator_name:
            continue
        response = (all_responses.get(role_name) or "").strip()
        if response:
            return response
    for role_name, response in reversed(list(all_responses.items())):
        if role_name == aggregator_name:
            continue
        if (response or "").strip():
            return response
    return None


def _validator_feedback_requires_revision(draft_answer: str, validator_feedback: str) -> bool:
    feedback = (validator_feedback or "").strip().lower()
    if not feedback:
        return False

    if any(hint in feedback for hint in _VALIDATOR_REVISION_NEGATIVE_HINTS):
        return True

    if any(hint in feedback for hint in _VALIDATOR_REVISION_POSITIVE_HINTS):
        return False

    draft = (draft_answer or "").strip().lower()
    if draft and draft in feedback:
        return False

    return False


def _apply_role_type_fast_credit_priors(
    active_roles: List[RoleCard],
    base_fast_credits: List[float],
    validator_name: Optional[str],
    validator_feedback: Optional[str],
    repair_triggered: bool,
    validator_adopted: bool,
) -> List[float]:
    adjusted: List[float] = []

    for role, base_credit in zip(active_roles, base_fast_credits):
        credit = float(base_credit)
        if role.role_type == "validator":
            validator_signal = _validator_fast_credit_signal(
                role.name,
                validator_name,
                validator_feedback,
                repair_triggered,
                validator_adopted,
            )
            if validator_signal == 0.0:
                validator_signal = 0.35
            credit = max(credit, validator_signal)
        adjusted.append(min(1.0, credit))

    return adjusted


def _validator_fast_credit_signal(
    role_name: str,
    validator_name: Optional[str],
    validator_feedback: Optional[str],
    repair_triggered: bool,
    validator_adopted: bool,
) -> float:
    if role_name != validator_name or not (validator_feedback or "").strip():
        return 0.0
    if validator_adopted:
        return 1.0
    if repair_triggered:
        return 0.75
    return 0.55


# ── Role Retrieval ─────────────────────────────────────────────────────────────

def retrieve_active_roles(
    pool: List[RoleCard],
    query_emb: np.ndarray,
    credit_engine: CreditEngine,
    encoder,                            # SentenceTransformer encoder
    n_active: int,
    specialist_slots: Optional[int] = None,
    exclude_validators: bool = False,
    validator_slots: int = 0,
    alpha_s: float = 0.5,               # cosine vs EMA credit balance
    use_active_set_credit: bool = True,
) -> List[RoleCard]:
    """
    Rank dynamic roles by query-role similarity, optionally mixed with EMA credit.

    The terminal aggregator is fixed separately and does not need to consume the
    dynamic specialist budget when specialist_slots is explicitly configured.
    When conditional validator routing is enabled, validator_slots reserve an
    additional verifier budget so validators do not have to compete directly with
    specialists for the same retrieval seats.
    """
    if not pool:
        return []

    def _rank_roles(candidates: List[RoleCard]) -> List[RoleCard]:
        if not candidates:
            return []

        prompts = [r.system_prompt for r in candidates]
        role_embs = encoder.encode(prompts, normalize_embeddings=True)

        cos_scores = np.dot(role_embs, query_emb / (np.linalg.norm(query_emb) + 1e-9))
        if use_active_set_credit:
            ema_scores = np.array([credit_engine.get_ema(r.name) for r in candidates], dtype=np.float32)
            ema_min, ema_max = ema_scores.min(), ema_scores.max()
            if ema_max > ema_min:
                ema_norm = (ema_scores - ema_min) / (ema_max - ema_min)
            else:
                ema_norm = np.zeros_like(ema_scores)
            combined = alpha_s * cos_scores + (1.0 - alpha_s) * ema_norm
        else:
            combined = cos_scores
        top_idx = np.argsort(combined)[::-1]
        return [candidates[i] for i in top_idx]

    terminal_name = select_aggregator_name(pool)
    terminal_role = next((r for r in pool if r.name == terminal_name), None)
    dynamic_roles = [r for r in pool if terminal_role is None or r.name != terminal_name]

    selected_dynamic_roles: List[RoleCard] = []
    if dynamic_roles:
        if exclude_validators:
            validator_roles = [r for r in dynamic_roles if r.role_type == "validator"]
            specialist_roles = [r for r in dynamic_roles if r.role_type != "validator"]

            if specialist_slots is None:
                specialist_budget = min(
                    max(0, n_active - (1 if terminal_role is not None else 0)),
                    len(specialist_roles),
                )
            else:
                specialist_budget = min(max(0, specialist_slots), len(specialist_roles))

            validator_budget = min(max(0, validator_slots), len(validator_roles))

            ranked_specialists = _rank_roles(specialist_roles)
            ranked_validators = _rank_roles(validator_roles)
            selected_dynamic_roles = (
                ranked_specialists[:specialist_budget]
                + ranked_validators[:validator_budget]
            )
        else:
            if specialist_slots is None:
                dynamic_budget = min(
                    max(0, n_active - (1 if terminal_role is not None else 0)),
                    len(dynamic_roles),
                )
            else:
                dynamic_budget = min(max(0, specialist_slots), len(dynamic_roles))

            ranked_dynamic_roles = _rank_roles(dynamic_roles)
            selected_dynamic_roles = ranked_dynamic_roles[:dynamic_budget]

    if terminal_role is None:
        return selected_dynamic_roles
    return selected_dynamic_roles + [terminal_role]


# ── Message Formatting ─────────────────────────────────────────────────────────

def format_agent_prompt(
    role: RoleCard,
    task_prompt: str,
    incoming_msgs: List[str],
    round_idx: int,
    previous_response: Optional[str] = None,
) -> str:
    """Build the user-facing prompt for an agent in a given round."""
    parts = [f"## Task\n{task_prompt}"]
    if round_idx > 0 and previous_response:
        parts.append(
            "## Your previous-round draft\n"
            "This is your own answer from the previous communication round. "
            "Refine it before producing your updated answer."
        )
        parts.append(previous_response)
    if incoming_msgs:
        parts.append("## Inputs from upstream agents")
        for i, msg in enumerate(incoming_msgs, 1):
            parts.append(f"### Input {i}\n{msg}")
    parts.append(f"\n## Your role\n{role.communication_protocol}")
    return "\n\n".join(parts)


def _invoke_role(
    client: OpenRouterClient,
    model: str,
    role: RoleCard,
    task_prompt: str,
    incoming_msgs: List[str],
    round_idx: int,
    previous_response: Optional[str] = None,
) -> str:
    """Execute one role call with the formatted prompt."""
    user_prompt = format_agent_prompt(
        role,
        task_prompt,
        incoming_msgs,
        round_idx,
        previous_response=previous_response,
    )
    return client.system_user(
        model=model,
        system=role.system_prompt,
        user=user_prompt,
        temperature=getattr(role, "temperature", 0.0),
        max_tokens=4096,
    )


# ── Phase A Core ───────────────────────────────────────────────────────────────

class PhaseA:
    """
    Executes Phase A inference for one task query.

    Args:
        config: SeroConfig instance
        client: OpenRouterClient
        encoder: SentenceTransformer model (for embeddings)
        credit_engine: CreditEngine (read/write)
        pool: current role pool (List[RoleCard])
        eval_fn: callable(answer: str) -> float, returns task score in [0,1]
    """

    def __init__(
        self,
        config: SeroConfig,
        client: OpenRouterClient,
        encoder,
        credit_engine: CreditEngine,
        pool: List[RoleCard],
        eval_fn,
    ):
        self.config = config
        self.client = client
        self.encoder = encoder
        self.credit_engine = credit_engine
        self.pool = pool
        self.eval_fn = eval_fn

    def run(
        self,
        task_prompt: str,
        loo_target: Optional[str] = None,
        update_fast_credits: bool = True,
        bootstrap_credit_dag: bool = False,
    ) -> Dict[str, Any]:
        """
        Run Phase A for one task.

        Args:
            task_prompt: natural language problem statement
            loo_target: role name to compute LOO credit for (optional, triggers
                        a second "without" run using pool minus that role)
            update_fast_credits: when False, compute per-role fast credits but do
                        not write them back into the shared CreditEngine state.
            bootstrap_credit_dag: when True, run one independent bootstrap pass
                        for active non-aggregator roles, compute current-task
                        fast credits from those drafts, and build the DAG from
                        those bootstrap credits instead of stale engine state.

        Returns dict with:
            answer: str
            score: float (from eval_fn)
            fast_credits: dict[role_name -> float]
            loo_phi: float or None (LOO Precise Credit for loo_target)
            pool_mean_emb: np.ndarray (mean of active role embeddings)
        """
        cfg = self.config

        # 1. Embed query
        query_emb = self.encoder.encode([task_prompt], normalize_embeddings=True)[0]  # (d_e,)

        # 2. Retrieve active roles
        active = retrieve_active_roles(
            self.pool, query_emb, self.credit_engine, self.encoder,
            n_active=cfg.n_max,
            specialist_slots=cfg.specialist_slots,
            exclude_validators=cfg.conditional_validator_pass,
            validator_slots=cfg.validator_slots,
            use_active_set_credit=cfg.use_active_set_credit,
        )
        active_names = [r.name for r in active]

        # 3. Detect aggregator: prefer explicit terminal roles over generic protected validators
        role_map: Dict[str, RoleCard] = {r.name: r for r in self.pool}
        agg_name = select_aggregator_name(active)

        # Non-aggregator agents participate in message-passing
        non_agg_names = [n for n in active_names if n != agg_name]

        bootstrap_responses: Dict[str, str] = {}
        dag_fast_credits: Dict[str, float] = {}

        if cfg.use_credit_dag and bootstrap_credit_dag and non_agg_names:
            if len(non_agg_names) == 1:
                role_name = non_agg_names[0]
                bootstrap_responses[role_name] = _invoke_role(
                    self.client,
                    cfg.agent_model,
                    role_map[role_name],
                    task_prompt,
                    [],
                    0,
                )
            else:
                def _bootstrap_call(rn: str) -> tuple:
                    resp = _invoke_role(
                        self.client,
                        cfg.agent_model,
                        role_map[rn],
                        task_prompt,
                        [],
                        0,
                    )
                    return rn, resp

                workers = min(len(non_agg_names), _MAX_PARALLEL_AGENTS)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_bootstrap_call, rn): rn for rn in non_agg_names}
                    for fut in as_completed(futures):
                        rn, resp = fut.result()
                        bootstrap_responses[rn] = resp

            bootstrap_texts = [bootstrap_responses[name] for name in non_agg_names]
            bootstrap_embs = self.encoder.encode(bootstrap_texts, normalize_embeddings=True)
            bootstrap_fc_list = compute_round_fast_credits(
                list(bootstrap_embs), query_emb, cfg.fast_credit_alpha
            )
            dag_fast_credits = {
                name: float(fc) for name, fc in zip(non_agg_names, bootstrap_fc_list)
            }
            fast_c_list = [dag_fast_credits[name] for name in non_agg_names]
        else:
            fast_c_list = [self.credit_engine.get_fast_credit(n) for n in non_agg_names]
            dag_fast_credits = {
                name: float(fc) for name, fc in zip(non_agg_names, fast_c_list)
            }

        # Build execution DAG: credit-ranked for full SERO, randomized-order DAG for A1b.
        if cfg.use_credit_dag:
            edges = build_credit_dag(
                non_agg_names, fast_c_list,
                aggregator_name=agg_name,
                role_types={name: role_map[name].role_type for name in non_agg_names},
                flat=False,
            )
        else:
            # A1b ablation: retain DAG construction, but remove credit-informed ordering.
            edges = build_credit_dag(
                non_agg_names, fast_c_list,
                aggregator_name=agg_name,
                flat=False,
                random_order=True,
                order_seed=cfg.seed,
            )

        topo_order = topological_order(non_agg_names, edges, aggregator_name=agg_name)
        dag_levels = compute_dag_levels(topo_order, edges, aggregator_name=agg_name)

        # 4. Multi-round message passing — parallel within each DAG level
        all_responses: Dict[str, str] = dict(bootstrap_responses)
        round_traces: List[Dict[str, Any]] = []

        for round_idx in range(cfg.t_round):
            round_responses: Dict[str, str] = {}

            for level_nodes in dag_levels:
                # Skip aggregator entries if they crept in
                level_nodes = [n for n in level_nodes if n != agg_name and n in role_map]
                if not level_nodes:
                    continue

                if len(level_nodes) == 1:
                    # Single node: call directly (no threading overhead)
                    role_name = level_nodes[0]
                    role = role_map[role_name]
                    incoming = get_incoming_messages(role_name, edges, all_responses)
                    previous_response = all_responses.get(role_name) if round_idx > 0 else None
                    round_responses[role_name] = _invoke_role(
                        self.client,
                        cfg.agent_model,
                        role,
                        task_prompt,
                        incoming,
                        round_idx,
                        previous_response=previous_response,
                    )
                else:
                    # Multiple independent nodes: call in parallel
                    def _call_agent(rn: str, snapshot: Dict[str, str]) -> tuple:
                        r = role_map[rn]
                        inc = get_incoming_messages(rn, edges, snapshot)
                        prev = snapshot.get(rn) if round_idx > 0 else None
                        resp = _invoke_role(
                            self.client,
                            cfg.agent_model,
                            r,
                            task_prompt,
                            inc,
                            round_idx,
                            previous_response=prev,
                        )
                        return rn, resp

                    # Snapshot all_responses at this level boundary (thread-safe read)
                    snapshot = dict(all_responses)
                    workers = min(len(level_nodes), _MAX_PARALLEL_AGENTS)
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {executor.submit(_call_agent, rn, snapshot): rn
                                   for rn in level_nodes}
                        for fut in as_completed(futures):
                            rn, resp = fut.result()
                            round_responses[rn] = resp

            round_trace = {
                role_name: round_responses[role_name]
                for role_name in topo_order
                if role_name in round_responses
            }
            round_traces.append({
                "round_idx": round_idx,
                "agent_traces": round_trace,
            })
            all_responses.update(round_responses)

        # 5. Aggregator produces final answer
        aggregator_role = role_map.get(agg_name)
        draft_answer = None
        validator_feedback = None
        revised_answer = None
        aggregator_fallback_answer = None
        aggregator_fallback_used = False
        validator_name = ""
        repair_triggered = False
        validator_adopted = False
        if aggregator_role is not None:
            agg_incoming = [all_responses[src] for src, dst in edges
                            if dst == agg_name and src in all_responses]
            draft_answer = _invoke_role(
                self.client,
                cfg.agent_model,
                aggregator_role,
                task_prompt,
                agg_incoming,
                0,
            )
            answer = draft_answer
        else:
            # Fallback: use last response in topological order
            answer = all_responses.get(topo_order[-2] if len(topo_order) >= 2 else topo_order[-1], "")
            draft_answer = answer

        if not (answer or "").strip():
            aggregator_fallback_answer = _fallback_final_answer(topo_order, agg_name, all_responses)
            if aggregator_fallback_answer is not None:
                answer = aggregator_fallback_answer
                aggregator_fallback_used = True

        validator_name = select_validator_name(
            self.pool,
            self.credit_engine,
            exclude_names=active_names,
        )
        validator_role = role_map.get(validator_name) if validator_name else None
        if validator_role is not None and _should_run_validator_pass(
            cfg,
            task_prompt,
            aggregator_role,
            answer,
        ):
            validator_inputs = list(agg_incoming if aggregator_role is not None else [])
            validator_inputs.append(f"Draft final answer from {agg_name}:\n{answer}")
            validator_feedback = _invoke_role(
                self.client,
                cfg.agent_model,
                validator_role,
                task_prompt,
                validator_inputs,
                0,
            )
            all_responses[validator_name] = validator_feedback
            repair_triggered = _validator_feedback_requires_revision(answer, validator_feedback)
            if aggregator_role is not None and repair_triggered:
                repair_inputs = list(agg_incoming)
                repair_inputs.append(f"Draft final answer from {agg_name}:\n{answer}")
                repair_inputs.append(validator_feedback)
                revised_answer = _invoke_role(
                    self.client,
                    cfg.agent_model,
                    aggregator_role,
                    task_prompt,
                    repair_inputs,
                    0,
                )
                validator_adopted = revised_answer.strip() != answer.strip()
                answer = revised_answer

        # Keep traces and downstream fast-credit inputs aligned with the final answer.
        all_responses[agg_name] = answer

        # 6. Score the answer
        raw_score = self.eval_fn(answer)
        score_with = normalize_score(raw_score)

        # 7. Fast Credits (per-round, based on final responses)
        response_texts = _collect_response_texts(active_names, all_responses)
        response_embs = self.encoder.encode(response_texts, normalize_embeddings=True)
        fc_list = compute_round_fast_credits(list(response_embs), query_emb, cfg.fast_credit_alpha)
        fc_list = _apply_role_type_fast_credit_priors(
            active,
            fc_list,
            validator_name if validator_name else None,
            validator_feedback,
            repair_triggered,
            validator_adopted,
        )

        fast_credits_dict = {}
        for name, fc in zip(active_names, fc_list):
            if update_fast_credits:
                self.credit_engine.update_fast_credit(name, fc)
            fast_credits_dict[name] = fc

        posthoc_validator_credit = _validator_fast_credit_signal(
            validator_name if validator_name else "",
            validator_name if validator_name else None,
            validator_feedback,
            repair_triggered,
            validator_adopted,
        )
        if validator_name and validator_name not in fast_credits_dict and posthoc_validator_credit > 0.0:
            if update_fast_credits:
                self.credit_engine.update_fast_credit(validator_name, posthoc_validator_credit)
            fast_credits_dict[validator_name] = posthoc_validator_credit

        # 8. LOO Precise Credit (optional, for one targeted role)
        loo_phi = None
        if (
            loo_target is not None
            and loo_target in active_names
            and len(self.pool) >= cfg.loo_min_pool_size
        ):
            # Run reduced pool (without loo_target)
            reduced_pool = [r for r in self.pool if r.name != loo_target]
            reduced_ph = PhaseA(
                config=self.config,
                client=self.client,
                encoder=self.encoder,
                credit_engine=self.credit_engine,
                pool=reduced_pool,
                eval_fn=self.eval_fn,
            )
            reduced_result = reduced_ph.run(
                task_prompt,
                loo_target=None,
                update_fast_credits=update_fast_credits,
            )
            score_without = reduced_result["score"]
            loo_phi = loo_precise_credit(score_with, score_without)

        # 9. Pool mean embedding (for controller state)
        pool_embs = self.encoder.encode([r.system_prompt for r in active],
                                        normalize_embeddings=True)
        pool_mean_emb = pool_embs.mean(axis=0)

        # Serialize topology for logging
        topology = {
            "edges": [(src, dst) for src, dst in edges],
            "topo_order": list(topo_order),
            "dag_levels": [list(level) for level in dag_levels],
            "aggregator": agg_name,
        }

        return {
            "answer": answer,
            "draft_answer": draft_answer,
            "validator_name": validator_name or None,
            "validator_feedback": validator_feedback,
            "revised_answer": revised_answer,
            "aggregator_fallback_used": aggregator_fallback_used,
            "aggregator_fallback_answer": aggregator_fallback_answer,
            "repair_triggered": repair_triggered,
            "validator_adopted": validator_adopted,
            "score": score_with,
            "raw_score": raw_score,
            "fast_credits": fast_credits_dict,
            "loo_phi": loo_phi,
            "pool_mean_emb": pool_mean_emb,
            "active_roles": active_names,
            "all_responses": dict(all_responses),
            "bootstrap_responses": dict(bootstrap_responses),
            "round_traces": round_traces,
            "dag_fast_credits": dict(dag_fast_credits),
            "topology": topology,
        }


    def run_batch(
        self,
        task_prompts: List[str],
        max_workers: int = _MAX_PARALLEL_TASKS,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of tasks in parallel using ThreadPoolExecutor.

        Returns results in the same order as task_prompts.
        Safe for frozen-pool evaluation (does NOT update credits or pool).
        """
        results: List[Optional[Dict[str, Any]]] = [None] * len(task_prompts)

        def _run_one(idx: int, prompt: str) -> tuple:
            try:
                r = self.run(prompt, update_fast_credits=False)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error("run_batch error idx=%d: %s", idx, e)
                r = {"answer": "", "score": 0.0, "fast_credits": {},
                     "loo_phi": None, "pool_mean_emb": np.zeros(1), "active_roles": [],
                     "dag_fast_credits": {},
                     "topology": {"edges": [], "topo_order": [], "dag_levels": [], "aggregator": ""}}
            return idx, r

        workers = min(max_workers, len(task_prompts))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, i, p): i
                       for i, p in enumerate(task_prompts)}
            for fut in as_completed(futures):
                idx, r = fut.result()
                results[idx] = r

        return results  # type: ignore[return-value]


# ── Batch LOO Refresh ─────────────────────────────────────────────────────────

def full_pool_loo_refresh(
    phase_a: PhaseA,
    task_items: List[Any],
) -> None:
    """
    Run LOO Precise Credit for every role in the current pool, averaging over
    a small sample of tasks. Called every LOO_REFRESH_INTERVAL episodes.
    Parallelizes across roles.

    task_items may be either plain prompts or task dicts with "prompt" and
    task-specific "eval_fn". The latter keeps refresh semantics aligned with
    trainer._loo_refresh, where each benchmark task has its own scorer.
    """
    if len(phase_a.pool) < phase_a.config.loo_min_pool_size:
        return

    pool_names = [r.name for r in phase_a.pool]

    def _resolve_task_item(task_item: Any) -> tuple[str, Any, Optional[PhaseA]]:
        if isinstance(task_item, dict):
            prompt = task_item["prompt"]
            eval_fn = task_item.get("eval_fn", phase_a.eval_fn)
            if eval_fn is phase_a.eval_fn:
                return prompt, eval_fn, phase_a
            task_phase_a = PhaseA(
                config=phase_a.config,
                client=phase_a.client,
                encoder=phase_a.encoder,
                credit_engine=phase_a.credit_engine,
                pool=phase_a.pool,
                eval_fn=eval_fn,
            )
            return prompt, eval_fn, task_phase_a
        return task_item, phase_a.eval_fn, phase_a

    def _loo_for_role(role_name: str) -> tuple:
        phi_samples = []
        for task_item in task_items:
            prompt, _, task_phase_a = _resolve_task_item(task_item)
            try:
                result = task_phase_a.run(
                    prompt,
                    loo_target=role_name,
                    update_fast_credits=False,
                )
                if result["loo_phi"] is not None:
                    phi_samples.append(result["loo_phi"])
            except Exception:
                continue
        avg = float(np.mean(phi_samples)) if phi_samples else 0.0
        return role_name, avg, bool(phi_samples)

    workers = min(len(pool_names), _MAX_PARALLEL_TASKS)
    role_updates = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_loo_for_role, rn): rn for rn in pool_names}
        for fut in as_completed(futures):
            role_updates.append(fut.result())

    for role_name, avg_phi, has_samples in role_updates:
        if has_samples and role_name in pool_names:
            phase_a.credit_engine.update_ema(role_name, avg_phi)
