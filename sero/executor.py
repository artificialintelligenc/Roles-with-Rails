"""
LLM Executor — generates role card edits via OpenRouter.

The executor receives the current role pool state and a controller-chosen
operation (op_type, target_role_name) and asks an LLM to produce the edit.

Operations:
  ADD-anchor : create a new role, using highest-EMA role as anchor/template
  REMOVE     : request confirmation (always approved) — returns None
  NOOP       : no call, returns None

The LLM is asked to produce a valid JSON role card diff. The executor
validates the output and falls back to the anchor card if the JSON is invalid.
"""

import json
import logging
from typing import Optional, Dict, Any

from sero.role_card import RoleCard, validate_role_card
from sero.credit_engine import CreditEngine
from sero.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

_EXECUTOR_SYSTEM = """You are a role card editor for a multi-agent REASONING and PROBLEM-SOLVING system.
The system solves structured tasks (e.g., scheduling, constraint satisfaction, optimization,
mathematical proofs) by having multiple specialist agents collaborate through message passing.

Given a task description and an existing role pool, produce a NEW specialized role card
that fills a REASONING OR ANALYTICAL gap not covered by any existing role.

CRITICAL RULES:
- The new role MUST directly help SOLVE the task (e.g., analyze constraints, verify solutions,
  propose strategies, catch errors, handle edge cases, decompose problems).
- The new role must NOT be about subjective advice, recommendations, tips, experiences,
  lifestyle suggestions, or any content unrelated to solving the core problem.
- The new role should complement existing roles by adding a different analytical perspective
  or verification capability — not by adding tangential domain knowledge.
- NEVER include task-specific data in the system_prompt (e.g., specific city names, dates,
  numbers, day ranges, person names from the task context). The system_prompt must be
  GENERAL-PURPOSE so it works across all instances of the same problem type.

Output ONLY a JSON object with this schema:
{
  "name": "<unique role name, 2-4 words>",
  "system_prompt": "<detailed system prompt for this specialist, 2-5 sentences>",
    "capability_family": "<short hyphenated family label>",
  "capability_tags": ["<tag1>", "<tag2>"],
  "communication_protocol": "<how this role should format its output>",
    "temperature": <0.0 for deterministic/verification roles, 0.7 for creative/solution roles>,
    "role_type": "<specialist|validator|aggregator>"
}

Do NOT include any explanation outside the JSON."""

_EXECUTOR_USER_TEMPLATE = """Task context (this is a PROBLEM TO SOLVE, not a topic for advice):
{task_context}

Existing roles in the pool (do NOT duplicate these capabilities):
{existing_roles_summary}

Current capability families already covered in the pool:
{existing_capability_families}

Missing seed capability families that are currently uncovered:
{missing_capability_families}

Recent failed task examples under the current/nearby pool:
{recent_failed_task_examples}

Recent recurring failure patterns:
{recent_failure_patterns}

Diversity monitor status:
{diversity_status}

Mean prompt overlap among current non-protected roles:
{prompt_similarity_mean}

Dominant capability families to avoid when diversity is low:
{orthogonal_family_blacklist}

Existing roles to stay materially distinct from when diversity is low:
{orthogonal_role_blacklist}

Allowed role types for this ADD operation:
{allowed_new_role_types}

Validator-role policy for this ADD operation:
{validator_role_policy}
{domain_generation_constraints_block}

Anchor role (highest performing — use as inspiration for the new role):
Name: {anchor_name}
Tags: {anchor_tags}
Output protocol: {anchor_protocol}

IMPORTANT CONSTRAINTS:
1. If any existing role specifies a strict output format (e.g., step-by-step format strings),
   the new role MUST also respect that format in its communication_protocol and system_prompt.
2. The new role must help SOLVE the problem — it should reason about constraints, validate
   solutions, propose alternative approaches, handle edge cases, or verify correctness.
3. Do NOT create roles that give subjective recommendations (e.g., restaurant tips, cultural
   advice, budget suggestions) — only roles that contribute to finding the correct answer.
4. The system_prompt MUST be task-agnostic: do NOT mention any specific city, person, date,
   number, or constraint value from the task context above. Write a GENERAL-PURPOSE prompt.
5. DO NOT recreate recently removed roles by exact name: {recently_removed_roles}
6. You MUST set capability_family to a short reusable label describing the new role's core reasoning function.
7. If there are missing seed capability families, prefer covering one of them first.
8. If no family is missing, make the new role materially different from these existing families: {existing_capability_families}.
9. Use the recent failed task examples and recurring failure patterns to target the most likely reasoning bottleneck.
10. When the diversity monitor is LOW, avoid the blacklisted dominant families unless you are explicitly covering a missing seed family.
11. When the diversity monitor is LOW, stay materially distinct from the overlapping roles listed above unless you are explicitly covering a missing seed family.
12. Obey the allowed role types and validator-role policy exactly.
{domain_generation_constraint_rule}

Create a NEW role card that fills a REASONING capability gap NOT already covered above.
Assign temperature=0.7 if this role generates solutions/strategies; temperature=0.0 if it verifies/checks/aggregates."""


class Executor:
    """
    Wraps the LLM-based role card editor.

    Usage:
        ex = Executor(client, config.executor_model)
        new_card = ex.add_anchor(pool, credit_engine, task_context)
    """

    def __init__(self, client: OpenRouterClient, model: str):
        self.client = client
        self.model = model

    # ── ADD-anchor ────────────────────────────────────────────────────────────

    def add_anchor(
        self,
        pool: list,                     # List[RoleCard]
        credit_engine: CreditEngine,
        task_context: str,
        recently_removed_role_names: Optional[list[str]] = None,
        task_conditioning: Optional[Dict[str, Any]] = None,
        max_attempts: int = 2,
        format_inherit: bool = True,
        return_trace: bool = False,
    ) -> "Optional[RoleCard] | dict":
        """
        Generate a new RoleCard anchored on the highest-EMA role in the pool.

        If return_trace=False (default): returns the new RoleCard or None.
        If return_trace=True: returns a dict with keys:
            card: Optional[RoleCard], anchor_name: str, llm_raw_output: str,
            executor_prompt: str, fallback: bool, recently_removed_roles: list[str]
        """
        trace: Dict[str, Any] = {
            "card": None, "anchor_name": "", "llm_raw_output": "",
            "executor_prompt": "", "fallback": False,
            "recently_removed_roles": list(recently_removed_role_names or []),
            "task_conditioning": dict(task_conditioning or {}),
        }
        if not pool:
            return trace if return_trace else None

        # Find anchor: highest EMA credit role
        anchor = max(pool, key=lambda r: credit_engine.get_ema(r.name))
        trace["anchor_name"] = anchor.name

        # Summarize all existing roles to prevent redundant additions
        # Show more context for aggregator/synthesizer roles since they carry format constraints
        existing_summary_lines = []
        for r in pool:
            is_format_critical = any(w in r.name.lower() for w in ("aggregat", "synth", "final"))
            snippet_len = 250 if is_format_critical else 120
            existing_summary_lines.append(
                f"- {r.name} [{', '.join(r.capability_tags)}]: {r.system_prompt[:snippet_len]}..."
            )
        existing_summary = "\n".join(existing_summary_lines)

        anchor_protocol = (anchor.communication_protocol or "(not specified)") if format_inherit else "(not specified)"
        recently_removed_roles = _format_recently_removed_roles(recently_removed_role_names)
        task_conditioning = dict(task_conditioning or {})
        domain_constraints = task_conditioning.get("domain_generation_constraints")
        if domain_constraints and str(domain_constraints).strip() != "None":
            domain_constraints_block = (
                "\nBenchmark/domain-specific role-generation constraints:\n"
                f"{domain_constraints}\n"
            )
            domain_generation_constraint_rule = (
                "13. Obey the benchmark/domain-specific constraints above as general role-design rules. "
                "They are not task answers."
            )
        else:
            domain_constraints_block = ""
            domain_generation_constraint_rule = ""
        user_msg = _EXECUTOR_USER_TEMPLATE.format(
            task_context=task_context[:800],          # truncate for API cost
            existing_roles_summary=existing_summary,
            existing_capability_families=_format_string_list(task_conditioning.get("existing_capability_families")),
            missing_capability_families=_format_string_list(task_conditioning.get("missing_capability_families")),
            recent_failed_task_examples=_format_string_list(task_conditioning.get("recent_failed_task_examples")),
            recent_failure_patterns=_format_string_list(task_conditioning.get("recent_failure_patterns")),
            diversity_status=task_conditioning.get("diversity_status", "healthy"),
            prompt_similarity_mean=task_conditioning.get("prompt_similarity_mean", 0.0),
            orthogonal_family_blacklist=_format_string_list(task_conditioning.get("orthogonal_family_blacklist")),
            orthogonal_role_blacklist=_format_string_list(task_conditioning.get("orthogonal_role_blacklist")),
            allowed_new_role_types=_format_string_list(task_conditioning.get("allowed_new_role_types")),
            validator_role_policy=task_conditioning.get(
                "validator_role_policy",
                "Validator roles are allowed when they are actually needed.",
            ),
            domain_generation_constraints_block=domain_constraints_block,
            domain_generation_constraint_rule=domain_generation_constraint_rule,
            anchor_name=anchor.name,
            anchor_tags=", ".join(anchor.capability_tags),
            anchor_protocol=anchor_protocol,
            recently_removed_roles=recently_removed_roles,
        )
        trace["executor_prompt"] = user_msg

        for attempt in range(max_attempts):
            try:
                raw = self.client.system_user(
                    model=self.model,
                    system=_EXECUTOR_SYSTEM,
                    user=user_msg,
                    temperature=0.7,    # higher diversity for genuinely novel roles
                    max_tokens=4096,
                )
                trace["llm_raw_output"] = raw
                card = _parse_role_card(raw)
                if card is not None:
                    # Ensure name is unique in pool
                    existing_names = {r.name for r in pool}
                    if card.name in existing_names:
                        card = card.copy_with(name=f"{card.name} v{attempt+2}")
                    trace["card"] = card
                    return trace if return_trace else card
            except Exception as e:
                logger.warning("Executor ADD attempt %d failed: %s", attempt + 1, e)

        # Fallback: return a mutated anchor
        logger.warning("Executor falling back to anchor copy for ADD")
        fallback_card = _fallback_from_anchor(anchor, pool)
        trace["card"] = fallback_card
        trace["fallback"] = True
        return trace if return_trace else fallback_card

    # ── REMOVE ────────────────────────────────────────────────────────────────

    def remove(self, role_name: str, pool: list) -> list:
        """
        Remove a role from the pool by name. Returns updated pool.
        Preserves at least 1 role in the pool.
        """
        if len(pool) <= 1:
            logger.warning("Cannot REMOVE: pool would become empty.")
            return pool
        return [r for r in pool if r.name != role_name]

    # ── NOOP ──────────────────────────────────────────────────────────────────

    def noop(self, pool: list) -> list:
        """NOOP: return pool unchanged."""
        return pool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_role_card(raw: str) -> Optional[RoleCard]:
    """Parse LLM output into a RoleCard. Returns None on failure."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Find JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        logger.warning("Executor: no JSON found in response")
        return None

    try:
        d = json.loads(text[start:end])
    except json.JSONDecodeError as e:
        logger.warning("Executor: JSON parse error: %s", e)
        return None

    err = validate_role_card(d)
    if err:
        logger.warning("Executor: invalid role card: %s", err)
        return None

    return RoleCard.from_dict(d)


def _fallback_from_anchor(anchor: RoleCard, pool: list) -> RoleCard:
    """Create a simple variant of the anchor as a last-resort fallback."""
    existing_names = {r.name for r in pool}
    suffix = 2
    new_name = f"{anchor.name} Alt"
    while new_name in existing_names:
        new_name = f"{anchor.name} Alt{suffix}"
        suffix += 1
    return anchor.copy_with(
        name=new_name,
        system_prompt=anchor.system_prompt + " Focus on edge cases and verification.",
        protected=False,
        role_type="specialist",
    )


def _format_recently_removed_roles(role_names: Optional[list[str]]) -> str:
    """Render recent removal history for prompt injection."""
    if not role_names:
        return "None"
    return json.dumps(role_names, ensure_ascii=False)


def _format_string_list(items: Optional[list[str]]) -> str:
    if not items:
        return "None"
    return "\n".join(f"- {item}" for item in items)
