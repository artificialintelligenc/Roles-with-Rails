"""
SERO Trainer — one-edit-per-task REINFORCE with batch-normalized reward.

Training loop:
  For each episode (task):
    1. Run Phase A → get score_before, pool_mean_emb, fast_credits
    2. Controller selects action (op, target)
    3. Executor applies edit to pool
    4. Run Phase A again → get score_after
    5. Reward r_t = score_after - score_before (raw delta)
    6. Accumulate (log_prob, r_t) for batch

  Every BATCH_SIZE episodes:
    - Batch-normalize rewards: r̂ = (r - mean) / (std + eps)
    - Subtract EMA baseline: advantage = r̂ - b
    - REINFORCE loss = -mean(advantage * log_prob) + entropy_beta * H(pi_op)
    - Update EMA baseline
    - Gradient step

  Every LOO_REFRESH_INTERVAL episodes:
    - Full-pool LOO refresh for all roles

Warmup epochs (no credit state): controller trains on Δscore only.
Main epochs: full credit state enabled.
"""

import logging
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from sero.config import SeroConfig
from sero.role_card import RoleCard, capability_family_counts, is_role_removable
from sero.credit_engine import CreditEngine
from sero.controller import SeroController, sample_action, sample_random_action, build_controller_tensors, build_op_action_mask
from sero.executor import Executor
from sero.phase_a import PhaseA, full_pool_loo_refresh
from sero.openrouter_client import OpenRouterClient, OpenRouterSampleSkipError

logger = logging.getLogger(__name__)

RECENT_REMOVED_ROLE_LIMIT = 10
RECENT_FAILED_TASK_LIMIT = 4
RECENT_FAILURE_PATTERN_LIMIT = 6
_FAILURE_PATTERN_HINTS = {
    "route-coverage": ("route", "flight", "travel", "path"),
    "window-reasoning": ("window", "timeslot", "time slot", "interval"),
    "availability-reasoning": ("availability", "available", "attendee", "person"),
    "distance-reasoning": ("distance", "commute", "drive", "travel time"),
    "constraint-validation": ("constraint", "conflict", "verify", "validation"),
    "schedule-construction": ("schedule", "calendar", "meeting", "plan"),
}


@dataclass
class EpisodeRecord:
    """Record of one training episode for logging."""
    episode: int
    task_id: str
    op_type: str
    executed_op_type: Optional[str]
    target: Optional[str]
    score_before: float
    score_after: float
    reward: float
    pool_size: int  # legacy alias for candidate_pool_size
    log_prob: float  # float copy for logging only; tensor is kept in batch buffer
    candidate_pool_size: Optional[int] = None
    committed_pool_size: Optional[int] = None
    pool_before: Optional[List[str]] = None
    pool_after: Optional[List[str]] = None  # legacy alias for candidate_pool_after
    candidate_pool_after: Optional[List[str]] = None
    committed_pool_after: Optional[List[str]] = None
    accepted: Optional[bool] = None
    executor_prompt: Optional[str] = None
    designer_raw_output: Optional[str] = None
    new_role_card: Optional[Dict[str, Any]] = None
    recent_removed_roles: Optional[List[str]] = None
    family_diversity: Optional[float] = None
    family_dominance: Optional[float] = None
    prompt_similarity_mean: Optional[float] = None
    missing_capability_families: Optional[List[str]] = None
    orthogonal_family_blacklist: Optional[List[str]] = None
    orthogonal_role_blacklist: Optional[List[str]] = None
    low_diversity_mode: Optional[bool] = None
    rejection_reason: Optional[str] = None
    fast_credits_before: Optional[Dict[str, float]] = None
    fast_credits_after: Optional[Dict[str, float]] = None
    credit_stats_input: Optional[List[float]] = None
    credit_snapshot: Optional[Dict[str, dict]] = None  # legacy alias for candidate_credit_snapshot
    candidate_credit_snapshot: Optional[Dict[str, dict]] = None
    committed_credit_snapshot: Optional[Dict[str, dict]] = None
    topology_before: Optional[Dict[str, Any]] = None
    topology_after: Optional[Dict[str, Any]] = None  # legacy alias for candidate_topology_after
    candidate_topology_after: Optional[Dict[str, Any]] = None
    committed_topology_after: Optional[Dict[str, Any]] = None
    validator_name_before: Optional[str] = None
    validator_name_after: Optional[str] = None
    repair_triggered_before: Optional[bool] = None
    repair_triggered_after: Optional[bool] = None
    validator_adopted_before: Optional[bool] = None
    validator_adopted_after: Optional[bool] = None


@dataclass
class TrainingStats:
    episodes: int = 0
    total_reward: float = 0.0
    noop_count: int = 0
    skip_count: int = 0
    add_count: int = 0
    remove_count: int = 0
    policy_loss: float = 0.0
    entropy: float = 0.0
    baseline: float = 0.0
    episode_records: List[EpisodeRecord] = field(default_factory=list)
    _recent_ops: List[str] = field(default_factory=list)
    _recent_committed_pool_unchanged: List[bool] = field(default_factory=list)

    def record_op(self, executed_op_type: str, committed_pool_unchanged: bool = False) -> None:
        """Record the executed operation for stats and collapse tracking."""
        if executed_op_type in {"noop", "invalid_add", "invalid_remove"}:
            actual = "noop"
        elif executed_op_type == "skip_error":
            actual = "skip"
        else:
            actual = executed_op_type
        self._recent_ops.append(actual)
        self._recent_committed_pool_unchanged.append(committed_pool_unchanged)
        if actual == "noop":
            self.noop_count += 1
        elif actual == "skip":
            self.skip_count += 1
        elif actual == "add_anchor":
            self.add_count += 1
        else:
            self.remove_count += 1

    def noop_rate(self, window: int = 24) -> float:
        """Windowed noop rate over the last `window` episodes."""
        recent = self._recent_ops[-window:] if self._recent_ops else []
        if not recent:
            return 0.0
        return sum(1 for op in recent if op == "noop") / len(recent)

    def committed_pool_unchanged_rate(
        self,
        window: int = 24,
        require_full_window: bool = False,
    ) -> Optional[float]:
        """Windowed unchanged rate for the committed pool.

        When `require_full_window` is true, return None until the window is full.
        """
        recent = (
            self._recent_committed_pool_unchanged[-window:]
            if self._recent_committed_pool_unchanged
            else []
        )
        if require_full_window and len(recent) < window:
            return None
        if not recent:
            return 0.0
        return sum(1 for unchanged in recent if unchanged) / len(recent)


class SeroTrainer:
    """
    End-to-end SERO training loop.

    Args:
        config: SeroConfig
        seed_pool: initial role pool (List[RoleCard])
        tasks: list of dicts with keys "id", "prompt", "eval_fn"
        client: OpenRouterClient
        encoder: SentenceTransformer
        results_dir: where to save checkpoints and logs
    """

    def __init__(
        self,
        config: SeroConfig,
        seed_pool: List[RoleCard],
        tasks: List[Dict[str, Any]],
        client: OpenRouterClient,
        encoder,
        results_dir: str = "results",
    ):
        self.cfg = config
        self.pool: List[RoleCard] = list(seed_pool)
        self.tasks = tasks
        self.client = client
        self.encoder = encoder
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Credit engine
        self.credit_engine = CreditEngine(mu=config.ema_mu, alpha=config.fast_credit_alpha)
        self.credit_engine.sync_with_pool([r.name for r in self.pool])

        # Controller
        self.controller = SeroController(config)
        self.optimizer = optim.Adam(self.controller.parameters(), lr=config.lr)

        # Executor
        self.executor = Executor(client, config.executor_model)

        # Training state
        self.ema_baseline = 0.0
        self.stats = TrainingStats()
        self.episode_idx = 0
        self.recently_removed_role_names: List[str] = []
        self.recent_failed_task_examples: List[str] = []
        self.recent_failure_patterns: List[str] = []
        self.required_capability_families = {
            role.capability_family
            for role in seed_pool
            if role.capability_family
        }

        # Batch buffers (log_prob tensors keep grad_fn for REINFORCE)
        self._batch_log_probs: List[torch.Tensor] = []
        self._batch_rewards: List[float] = []
        self._batch_op_logits: List[torch.Tensor] = []

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    # ── Training entry point ──────────────────────────────────────────────────

    def train(self) -> TrainingStats:
        """Run full training (warmup + main epochs)."""
        cfg = self.cfg
        collapse_window = 24
        main_committed_pool_unchanged: List[bool] = []
        min_main_episodes_before_collapse = max(len(self.tasks), collapse_window)
        total_episodes = (cfg.warmup_epochs + cfg.main_epochs) * len(self.tasks)

        logger.info("Starting SERO training: %d episodes (%d tasks × %d epochs)",
                    total_episodes, len(self.tasks), cfg.warmup_epochs + cfg.main_epochs)

        for epoch in range(cfg.warmup_epochs + cfg.main_epochs):
            in_main_phase = epoch >= cfg.warmup_epochs
            controller_uses_credit_state = in_main_phase and cfg.use_credit_state
            logger.info("Epoch %d/%d  credit_state=%s", epoch + 1,
                        cfg.warmup_epochs + cfg.main_epochs, controller_uses_credit_state)
            if epoch == cfg.warmup_epochs:
                main_committed_pool_unchanged.clear()
                logger.info(
                    "Reset committed-pool collapse window at main phase start "
                    "(window=%d, min_main_episodes=%d).",
                    collapse_window,
                    min_main_episodes_before_collapse,
                )

            if cfg.shuffle_train_tasks:
                np.random.shuffle(self.tasks)  # legacy behavior: shuffle task order each epoch
            else:
                logger.info("Preserving training task order for epoch %d.", epoch + 1)

            for task in self.tasks:
                self._run_episode(
                    task,
                    controller_uses_credit_state=controller_uses_credit_state,
                    strict_improvement_only=in_main_phase,
                    allow_remove=epoch >= cfg.warmup_epochs,
                )
                self.episode_idx += 1
                self.stats.episodes += 1
                if in_main_phase and self.stats._recent_committed_pool_unchanged:
                    main_committed_pool_unchanged.append(
                        self.stats._recent_committed_pool_unchanged[-1]
                    )

                # Batch gradient step
                if (
                    (not cfg.random_controller)
                    and cfg.controller_reward_training
                    and len(self._batch_log_probs) >= cfg.batch_size
                ):
                    self._gradient_step()

                # LOO refresh
                if self.episode_idx % cfg.loo_refresh_interval == 0:
                    self._loo_refresh(sample_tasks=min(3, len(self.tasks)))

                # Collapse check
                collapse_rate = None
                if (
                    in_main_phase
                    and len(main_committed_pool_unchanged) >= collapse_window
                    and len(main_committed_pool_unchanged) >= min_main_episodes_before_collapse
                ):
                    recent_main_unchanged = main_committed_pool_unchanged[-collapse_window:]
                    collapse_rate = sum(1 for unchanged in recent_main_unchanged if unchanged) / len(recent_main_unchanged)
                if (
                    in_main_phase
                    and collapse_rate is not None
                    and collapse_rate > cfg.noop_collapse_threshold
                ):
                    logger.warning(
                        "Committed-pool unchanged collapse detected (%.1f%% over last %d main episodes after %d main episodes). Stopping early.",
                        collapse_rate * 100,
                        collapse_window,
                        len(main_committed_pool_unchanged),
                    )
                    self._save_checkpoint(epoch, "collapsed")
                    return self.stats

            # End-of-epoch checkpoint
            self._save_checkpoint(epoch)
            self._save_logs()

        # Final flush
        if (not cfg.random_controller) and cfg.controller_reward_training and self._batch_log_probs:
            self._gradient_step()
        self._save_logs()
        return self.stats

    # ── Single episode ────────────────────────────────────────────────────────

    def _run_episode(
        self,
        task: Dict[str, Any],
        controller_uses_credit_state: bool,
        strict_improvement_only: bool,
        allow_remove: bool = True,
    ) -> None:
        cfg = self.cfg
        task_id = task.get("id", str(self.episode_idx))
        task_prompt = task["prompt"]
        eval_fn = task["eval_fn"]
        credit_state_before_episode = self.credit_engine.state_dict()

        # ── Phase A (before edit) ──────────────────────────────────────────
        phase_a = PhaseA(
            config=cfg,
            client=self.client,
            encoder=self.encoder,
            credit_engine=self.credit_engine,
            pool=self.pool,
            eval_fn=eval_fn,
        )
        try:
            result_before = phase_a.run(task_prompt, loo_target=None)
        except OpenRouterSampleSkipError as exc:
            self.credit_engine.load_state_dict(credit_state_before_episode)
            self._record_skipped_episode(task_id, str(exc))
            logger.warning("Skipping training sample %s due to content inspection error.", task_id)
            return
        score_before = result_before["score"]
        pool_mean_emb = result_before["pool_mean_emb"]
        fast_credits_before = result_before.get("fast_credits", {})
        topology_before = result_before.get("topology")
        validator_name_before = result_before.get("validator_name")
        repair_triggered_before = result_before.get("repair_triggered")
        validator_adopted_before = result_before.get("validator_adopted")
        credit_state_before_edit = self.credit_engine.state_dict()

        # ── Build feedback embedding ───────────────────────────────────────
        feedback_text = f"{task_prompt}\n\nAnswer: {result_before['answer'][:200]}"
        feedback_emb = self.encoder.encode([feedback_text], normalize_embeddings=True)[0]

        # ── Build role embeddings ──────────────────────────────────────────
        role_embs_np = self.encoder.encode(
            [r.system_prompt for r in self.pool], normalize_embeddings=True
        )
        family_monitor = self._compute_pool_family_monitor(self.pool)

        # ── Controller forward ─────────────────────────────────────────────
        tensors = build_controller_tensors(
            self.pool, self.credit_engine,
            role_embs_np, feedback_emb, pool_mean_emb,
            use_credit_state=controller_uses_credit_state,
        )
        credit_stats_input = tensors["credit_stats"].tolist()
        required_families = (
            self.required_capability_families
            if cfg.preserve_seed_family_coverage
            else None
        )
        op_mask = build_op_action_mask(
            self.pool,
            cfg,
            allow_remove=allow_remove,
            required_capability_families=required_families,
        )
        if cfg.random_controller:
            sampled_op_logits = torch.zeros(len(SeroController.OPS), dtype=torch.float32)
            sampled_op_logits[~op_mask] = float("-inf")
            op_type, target, log_prob = sample_random_action(
                self.pool,
                op_mask,
                config=cfg,
                required_capability_families=required_families,
            )
        else:
            # For training we need gradients — run the learned controller path.
            self.controller.train()
            op_logits_grad, target_scores_grad = self.controller(**tensors)

            sampled_op_logits = op_logits_grad.clone()
            sampled_op_logits[~op_mask] = float("-inf")

            op_type, target, log_prob = sample_action(
                sampled_op_logits, target_scores_grad,
                self.pool,
                use_credit_state=controller_uses_credit_state,
                config=cfg,
                required_capability_families=required_families,
            )

        # ── Apply edit ────────────────────────────────────────────────────
        pool_before_names = [r.name for r in self.pool]
        pool_after, add_trace, executed_op_type = self._apply_action(op_type, target, task)
        pool_after_names = [r.name for r in pool_after]
        pool_changed = pool_after_names != pool_before_names
        static_pool = executed_op_type in {"noop", "invalid_add", "invalid_remove"} or (not pool_changed)

        # ── Phase A (after edit) ──────────────────────────────────────────
        if static_pool:
            result_after = result_before
            score_after = score_before
            fast_credits_after = fast_credits_before
            topology_after = topology_before
        else:
            phase_a_after = PhaseA(
                config=cfg,
                client=self.client,
                encoder=self.encoder,
                credit_engine=self.credit_engine,
                pool=pool_after,
                eval_fn=eval_fn,
            )
            try:
                result_after = phase_a_after.run(task_prompt, loo_target=None)
            except OpenRouterSampleSkipError as exc:
                self.credit_engine.load_state_dict(credit_state_before_episode)
                self._record_skipped_episode(task_id, str(exc))
                logger.warning("Skipping training sample %s due to content inspection error after edit.", task_id)
                return
            score_after = result_after["score"]
            fast_credits_after = result_after.get("fast_credits", {})
            topology_after = result_after.get("topology")
        validator_name_after = result_after.get("validator_name")
        repair_triggered_after = result_after.get("repair_triggered")
        validator_adopted_after = result_after.get("validator_adopted")

        # ── Reward ────────────────────────────────────────────────────────
        reward = 0.0 if static_pool else score_after - score_before

        candidate_pool_after_names = list(pool_after_names)
        candidate_pool_size = len(pool_after)
        candidate_topology_after = topology_after
        candidate_credit_snapshot = self.credit_engine.all_role_info(candidate_pool_after_names)

        # ── Accumulate ────────────────────────────────────────────────────
        if (not cfg.random_controller) and cfg.controller_reward_training:
            self._batch_log_probs.append(log_prob)
            self._batch_rewards.append(reward)
            self._batch_op_logits.append(sampled_op_logits)

        # ── Update pool if edit was beneficial (or NOOP) ─────────────────
        # During warmup: keep exploratory edits only if they are not harmful
        # (reward >= 0). During main epochs: keep only improving edits.
        strict_add_acceptance = self._uses_strict_add_acceptance(task)
        should_commit = False
        if executed_op_type == "noop":
            should_commit = True
        elif executed_op_type in {"add_anchor", "remove"} and pool_changed:
            if strict_add_acceptance and executed_op_type == "add_anchor":
                should_commit = reward > 0
            else:
                should_commit = (reward > 0) if strict_improvement_only else (reward >= 0)
        if should_commit:
            self.pool = pool_after
            if executed_op_type == "remove" and target is not None and pool_changed:
                self._remember_removed_role(target)

        # Restore exact committed credit state after rejected real edits so a
        # tentative Phase A run cannot leak fast-credit updates into later episodes.
        if (not should_commit) and executed_op_type in {"add_anchor", "remove"} and pool_changed:
            self.credit_engine.load_state_dict(credit_state_before_edit)
        else:
            # Always resync credit membership to the committed pool. This removes
            # tentative role registration from invalid / no-op paths and keeps
            # committed edits aligned with the retained pool.
            self.credit_engine.sync_with_pool(
                [r.name for r in self.pool],
                parent_ema_map={} if executed_op_type in {"noop", "invalid_add", "invalid_remove"} else None,
            )

        committed_pool_after_names = [r.name for r in self.pool]
        committed_pool_size = len(self.pool)
        committed_topology_after = (
            candidate_topology_after
            if committed_pool_after_names == candidate_pool_after_names
            else topology_before
        )
        committed_credit_snapshot = self.credit_engine.all_role_info(committed_pool_after_names)

        # Only a real pool edit that is actually committed counts as accepted.
        edit_accepted = (executed_op_type in {"add_anchor", "remove"}) and pool_changed and should_commit
        committed_pool_unchanged = committed_pool_after_names == pool_before_names

        # ── Stats ─────────────────────────────────────────────────────────
        self.stats.total_reward += reward
        self.stats.record_op(
            executed_op_type,
            committed_pool_unchanged=committed_pool_unchanged,
        )

        rec = EpisodeRecord(
            episode=self.episode_idx,
            task_id=task_id,
            op_type=op_type,
            executed_op_type=executed_op_type,
            target=target,
            score_before=score_before,
            score_after=score_after,
            reward=reward,
            pool_size=candidate_pool_size,
            candidate_pool_size=candidate_pool_size,
            committed_pool_size=committed_pool_size,
            log_prob=float(log_prob.detach()),
            pool_before=pool_before_names,
            pool_after=candidate_pool_after_names,
            candidate_pool_after=candidate_pool_after_names,
            committed_pool_after=committed_pool_after_names,
            accepted=edit_accepted,
            executor_prompt=add_trace["executor_prompt"] if add_trace else None,
            designer_raw_output=add_trace["llm_raw_output"] if add_trace else None,
            new_role_card=add_trace["new_role_card"] if add_trace else None,
            recent_removed_roles=add_trace["recently_removed_roles"] if add_trace else None,
            family_diversity=family_monitor["family_diversity"],
            family_dominance=family_monitor["family_dominance"],
            prompt_similarity_mean=family_monitor["prompt_similarity_mean"],
            missing_capability_families=family_monitor["missing_capability_families"],
            orthogonal_family_blacklist=family_monitor["orthogonal_family_blacklist"],
            orthogonal_role_blacklist=family_monitor["orthogonal_role_blacklist"],
            low_diversity_mode=family_monitor["low_diversity_mode"],
            rejection_reason=add_trace.get("rejection_reason") if add_trace else None,
            fast_credits_before={k: round(v, 5) for k, v in fast_credits_before.items()},
            fast_credits_after={k: round(v, 5) for k, v in fast_credits_after.items()},
            credit_stats_input=[round(v, 5) for v in credit_stats_input],
            credit_snapshot=candidate_credit_snapshot,
            candidate_credit_snapshot=candidate_credit_snapshot,
            committed_credit_snapshot=committed_credit_snapshot,
            topology_before=topology_before,
            topology_after=candidate_topology_after,
            candidate_topology_after=candidate_topology_after,
            committed_topology_after=committed_topology_after,
            validator_name_before=validator_name_before,
            validator_name_after=validator_name_after,
            repair_triggered_before=repair_triggered_before,
            repair_triggered_after=repair_triggered_after,
            validator_adopted_before=validator_adopted_before,
            validator_adopted_after=validator_adopted_after,
        )
        self.stats.episode_records.append(rec)
        self._remember_failed_task_context(task_prompt, score_before)
        logger.debug("Ep %d | sampled=%s executed=%s target=%s | %.3f→%.3f | r=%.3f",
                 self.episode_idx, op_type, executed_op_type, target, score_before, score_after, reward)

    def _record_skipped_episode(self, task_id: str, reason: str) -> None:
        pool_names = [r.name for r in self.pool]
        credit_snapshot = self.credit_engine.all_role_info(pool_names)

        self.stats.record_op("skip_error", committed_pool_unchanged=False)
        self.stats.episode_records.append(
            EpisodeRecord(
                episode=self.episode_idx,
                task_id=task_id,
                op_type="skip_error",
                executed_op_type="skip_error",
                target=None,
                score_before=0.0,
                score_after=0.0,
                reward=0.0,
                pool_size=len(pool_names),
                log_prob=0.0,
                candidate_pool_size=len(pool_names),
                committed_pool_size=len(pool_names),
                pool_before=pool_names,
                pool_after=pool_names,
                candidate_pool_after=pool_names,
                committed_pool_after=pool_names,
                accepted=False,
                rejection_reason=f"sample-skip:{reason}",
                fast_credits_before={},
                fast_credits_after={},
                credit_stats_input=[],
                credit_snapshot=credit_snapshot,
                candidate_credit_snapshot=credit_snapshot,
                committed_credit_snapshot=credit_snapshot,
                topology_before=None,
                topology_after=None,
                candidate_topology_after=None,
                committed_topology_after=None,
                validator_name_before=None,
                validator_name_after=None,
                repair_triggered_before=None,
                repair_triggered_after=None,
                validator_adopted_before=None,
                validator_adopted_after=None,
            )
        )

    # ── Gradient step ─────────────────────────────────────────────────────────

    def _gradient_step(self) -> None:
        cfg = self.cfg
        if not cfg.controller_reward_training:
            self._batch_log_probs.clear()
            self._batch_rewards.clear()
            self._batch_op_logits.clear()
            return
        rewards = np.array(self._batch_rewards, dtype=np.float32)

        # Batch normalize
        r_mean = rewards.mean()
        r_std = rewards.std() if len(rewards) > 1 else 1.0
        rewards_norm = (rewards - r_mean) / (r_std + cfg.reward_eps)

        # Subtract EMA baseline
        advantages = rewards_norm - self.ema_baseline
        self.ema_baseline = (cfg.ema_baseline_decay * self.ema_baseline
                             + (1.0 - cfg.ema_baseline_decay) * float(r_mean))

        # REINFORCE loss — stack Tensors to preserve grad_fn
        log_probs = torch.stack(self._batch_log_probs)          # (B,) with grad_fn
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        policy_loss = -(adv_t * log_probs).mean()

        # Entropy regularization (op head only)
        op_logits_stack = torch.stack(self._batch_op_logits, dim=0)  # (B, n_ops)
        op_probs = F.softmax(op_logits_stack, dim=-1)
        entropy = -(op_probs * torch.log(op_probs + 1e-9)).sum(dim=-1).mean()
        total_loss = policy_loss - cfg.entropy_beta * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.stats.policy_loss = float(total_loss)
        self.stats.entropy = float(entropy)
        self.stats.baseline = self.ema_baseline

        logger.info("Batch grad step | loss=%.4f entropy=%.4f baseline=%.4f",
                    float(policy_loss), float(entropy), self.ema_baseline)

        # Clear batch buffers
        self._batch_log_probs.clear()
        self._batch_rewards.clear()
        self._batch_op_logits.clear()

    # ── Action application ────────────────────────────────────────────────────

    def _apply_action(
        self,
        op_type: str,
        target: Optional[str],
        task_context: Any,
    ) -> tuple:
        """Return (updated_pool, add_trace, executed_op_type) after applying action.

        add_trace is a dict with designer details for add_anchor ops, None otherwise.
        Does not mutate self.pool.
        """
        pool_copy = list(self.pool)
        add_trace = None
        task = task_context if isinstance(task_context, dict) else None
        executor_task_context = self._build_executor_task_context(task_context)

        if op_type == "noop":
            return pool_copy, add_trace, "noop"

        if op_type in {"invalid_add", "invalid_remove"}:
            return pool_copy, add_trace, op_type

        if self.cfg.freeze_role_pool and op_type in {"remove", "add_anchor"}:
            logger.debug("Role-pool freeze ablation active; ignoring controller action '%s'.", op_type)
            return pool_copy, add_trace, "noop"

        if op_type == "remove" and target is not None:
            if len(pool_copy) <= self.cfg.p_min:
                logger.debug("Pool at min size %d; skipping REMOVE.", self.cfg.p_min)
                return pool_copy, add_trace, "invalid_remove"
            required_families = (
                self.required_capability_families
                if self.cfg.preserve_seed_family_coverage
                else None
            )
            if not is_role_removable(
                pool_copy,
                target,
                protect_critical_roles=self.cfg.protect_critical_roles,
                required_capability_families=required_families,
                required_role_type_minima=(
                    {"validator": self.cfg.validator_slots}
                    if self.cfg.conditional_validator_pass and self.cfg.validator_slots > 0
                    else None
                ),
            ):
                logger.debug("Role '%s' is not removable under current pool constraints; skipping REMOVE.", target)
                return pool_copy, add_trace, "invalid_remove"
            return self.executor.remove(target, pool_copy), add_trace, "remove"

        if op_type == "remove":
            return pool_copy, add_trace, "invalid_remove"

        if op_type == "add_anchor":
            if len(pool_copy) >= self.cfg.p_max:
                logger.debug("Pool at max size %d; skipping ADD.", self.cfg.p_max)
                return pool_copy, add_trace, "invalid_add"
            task_conditioning = self._build_add_task_conditioning(pool_copy, task)
            trace_result = self.executor.add_anchor(
                pool_copy, self.credit_engine, executor_task_context,
                task_conditioning=task_conditioning,
                recently_removed_role_names=self.recently_removed_role_names,
                format_inherit=self.cfg.format_inherit,
                return_trace=True,
            )
            new_card = trace_result["card"]
            if (
                new_card is not None
                and self.cfg.conditional_validator_pass
                and new_card.role_type == "validator"
            ):
                logger.debug(
                    "Rejecting ADD for role '%s': validator roles are disabled when conditional_validator_pass is enabled.",
                    new_card.name,
                )
                add_trace = {
                    "anchor_name": trace_result["anchor_name"],
                    "executor_prompt": trace_result["executor_prompt"],
                    "llm_raw_output": trace_result["llm_raw_output"],
                    "fallback": trace_result["fallback"],
                    "recently_removed_roles": trace_result["recently_removed_roles"],
                    "task_conditioning": trace_result.get("task_conditioning"),
                    "new_role_card": new_card.to_dict(),
                    "rejection_reason": "validator-role-disabled",
                }
                return pool_copy, add_trace, "invalid_add"
            domain_rejection = self._domain_role_rejection(new_card, task)
            if domain_rejection is not None:
                logger.debug(
                    "Rejecting ADD for role '%s': %s",
                    new_card.name if new_card else "(none)",
                    domain_rejection,
                )
                add_trace = {
                    "anchor_name": trace_result["anchor_name"],
                    "executor_prompt": trace_result["executor_prompt"],
                    "llm_raw_output": trace_result["llm_raw_output"],
                    "fallback": trace_result["fallback"],
                    "recently_removed_roles": trace_result["recently_removed_roles"],
                    "task_conditioning": trace_result.get("task_conditioning"),
                    "new_role_card": new_card.to_dict() if new_card else None,
                    "rejection_reason": domain_rejection,
                }
                return pool_copy, add_trace, "invalid_add"
            orthogonal_rejection = None
            if new_card is not None:
                orthogonal_rejection = self._violates_orthogonal_add(
                    new_card,
                    pool_copy,
                    task_conditioning,
                )
                if orthogonal_rejection is not None:
                    logger.debug(
                        "Rejecting ADD for role '%s': %s",
                        new_card.name,
                        orthogonal_rejection,
                    )
            add_trace = {
                "anchor_name": trace_result["anchor_name"],
                "executor_prompt": trace_result["executor_prompt"],
                "llm_raw_output": trace_result["llm_raw_output"],
                "fallback": trace_result["fallback"],
                "recently_removed_roles": trace_result["recently_removed_roles"],
                "task_conditioning": trace_result.get("task_conditioning"),
                "new_role_card": new_card.to_dict() if new_card else None,
                "rejection_reason": orthogonal_rejection,
            }
            if orthogonal_rejection is not None:
                return pool_copy, add_trace, "invalid_add"
            if new_card is not None:
                pool_copy.append(new_card)
                self.credit_engine.register_role(
                    new_card.name,
                    init_n_updates=self.cfg.new_role_initial_n_updates,
                )
            return pool_copy, add_trace, "add_anchor"

        return pool_copy, add_trace, "noop"

    def _remember_removed_role(self, role_name: str) -> None:
        """Keep a bounded recency list of actually committed removals."""
        updated = [name for name in self.recently_removed_role_names if name != role_name]
        updated.append(role_name)
        self.recently_removed_role_names = updated[-RECENT_REMOVED_ROLE_LIMIT:]

    def _build_add_task_conditioning(self, pool: List[RoleCard], task: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        family_monitor = self._compute_pool_family_monitor(pool)
        allowed_new_role_types = ["specialist"] if self.cfg.conditional_validator_pass else ["specialist", "validator"]
        validator_role_policy = (
            "The runtime already has a dedicated post-aggregation validator pass. Do NOT create a new validator role; emit role_type='specialist'."
            if self.cfg.conditional_validator_pass
            else "Validator roles are allowed when the task truly needs a new validation function."
        )
        recent_failure_patterns = list(self.recent_failure_patterns)
        for pattern in self._domain_failure_patterns(task):
            if pattern not in recent_failure_patterns:
                recent_failure_patterns.append(pattern)
        domain_generation_constraints = self._domain_generation_constraints(task)
        return {
            "existing_capability_families": family_monitor["existing_capability_families"],
            "missing_capability_families": family_monitor["missing_capability_families"],
            "diversity_status": family_monitor["diversity_status"],
            "low_diversity_mode": family_monitor["low_diversity_mode"],
            "prompt_similarity_mean": family_monitor["prompt_similarity_mean"],
            "orthogonal_family_blacklist": family_monitor["orthogonal_family_blacklist"],
            "orthogonal_role_blacklist": family_monitor["orthogonal_role_blacklist"],
            "allowed_new_role_types": allowed_new_role_types,
            "validator_role_policy": validator_role_policy,
            "recent_failed_task_examples": list(self.recent_failed_task_examples),
            "recent_failure_patterns": recent_failure_patterns[-RECENT_FAILURE_PATTERN_LIMIT:],
            "domain_generation_constraints": domain_generation_constraints,
        }

    def _build_executor_task_context(self, task: Any) -> str:
        if not isinstance(task, dict):
            return str(task)
        prompt = task.get("prompt", "")
        guardrail = self._domain_executor_guardrail(task)
        if not guardrail:
            return prompt
        return f"Task prompt:\n{prompt}\n\n{guardrail}"

    def _task_domain(self, task: Optional[Dict[str, Any]]) -> Optional[str]:
        if not task:
            return None
        explicit_domain = task.get("domain") or task.get("benchmark")
        if explicit_domain:
            return str(explicit_domain)
        prompt = task.get("prompt", "")
        if "TRIP PLANNING" in prompt or "CALENDAR SCHEDULING" in prompt or "MEETING PLANNING" in prompt:
            return "naturalplan"
        return None

    def _uses_strict_add_acceptance(self, task: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(task, dict):
            return False
        if "strict_add_acceptance" in task:
            return bool(task["strict_add_acceptance"])
        return self._task_domain(task) == "naturalplan"

    def _domain_executor_guardrail(self, task: Optional[Dict[str, Any]]) -> str:
        if not isinstance(task, dict):
            return ""
        explicit = task.get("domain_executor_guardrail") or task.get("executor_guardrail")
        if explicit:
            return str(explicit)
        if self._task_domain(task) == "naturalplan":
            return self._naturalplan_executor_guardrail()
        return ""

    def _domain_failure_patterns(self, task: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(task, dict):
            return []
        explicit = task.get("domain_failure_patterns") or task.get("failure_patterns")
        if isinstance(explicit, list):
            return [str(item) for item in explicit]
        if explicit:
            return [str(explicit)]
        if self._task_domain(task) == "naturalplan":
            return self._naturalplan_failure_patterns()
        return []

    def _domain_generation_constraints(self, task: Optional[Dict[str, Any]]) -> str:
        if not isinstance(task, dict):
            return "None"
        explicit = task.get("domain_generation_constraints") or task.get("role_generation_constraints")
        if explicit:
            return str(explicit)
        if self._task_domain(task) == "naturalplan":
            return self._naturalplan_domain_generation_constraints()
        return "None"

    def _domain_role_rejection(self, card: Optional[RoleCard], task: Optional[Dict[str, Any]]) -> Optional[str]:
        if card is None or not isinstance(task, dict):
            return None
        signal_terms = task.get("role_signal_terms")
        if isinstance(signal_terms, dict) and signal_terms:
            domain_name = str(self._task_domain(task) or "domain")
            return self._single_family_role_rejection(card, signal_terms, domain_name)
        if self._task_domain(task) == "naturalplan":
            return self._single_family_role_rejection(
                card,
                self._naturalplan_role_signal_terms(),
                "naturalplan",
            )
        return None

    def _naturalplan_executor_guardrail(self) -> str:
        return (
            "NaturalPlan SERO guardrails: infer the subtask from prompt evidence only. Create a role for exactly one "
            "subtask or for task-classification/format validation, but do not mix subtask semantics. Calendar may say 'meeting' "
            "but uses busy intervals/work hours and no travel matrix; it outputs one proposed-time line only. Meeting "
            "uses locations/travel times/windows and outputs travel/wait/meet steps only. Trip uses cities/days/flights "
            "and outputs Day visit/flight lines only."
        )

    def _naturalplan_failure_patterns(self) -> List[str]:
        return [
            "NaturalPlan task parsing: infer subtask from prompt evidence, not from hidden labels or the word 'meeting' alone.",
            "NaturalPlan format isolation: one final answer must use exactly one subtask parser contract.",
            "Trip planning: use direct flights and shared flight-day arithmetic; never emit meeting route or calendar slot formats.",
            "Calendar scheduling: intersect busy/free intervals, treat preferences as hard constraints, and never emit travel/wait/meet steps.",
            "Meeting planning: use exact travel matrix and availability windows, wait if early, and never emit a calendar proposed-time line.",
        ]

    def _naturalplan_domain_generation_constraints(self) -> str:
        return (
            "NaturalPlan role design contract:\n"
            "- Create a specialist for exactly one observable subtask family, or create a task-classification/format/validation role.\n"
            "- Do not use hidden labels. Infer trip/calendar/meeting from the problem text and format instructions only.\n"
            "- Trip specialists must talk about cities, stays, direct flights, day windows, and shared flight-day arithmetic; they must not mention people, meeting maximization, busy calendars, proposed-time slots, travel/wait/meet route steps, or generic tourism advice.\n"
            "- Calendar specialists must talk about participants' busy blocks, free-window intersections, hard preferences, duration, boundaries, and one 24-hour proposed-time line; they must not emit travel, wait, or 'You meet ...' route steps.\n"
            "- Meeting specialists must talk about locations, exact travel times, availability windows, waiting when early, and travel/wait/meet route steps; they must not emit a calendar proposed-time line.\n"
            "- A specialist prompt that mixes multiple subtask formats is invalid unless the role is explicitly a task parser, validator, formatter, or aggregator."
        )

    def _single_family_role_rejection(
        self,
        card: RoleCard,
        signal_terms: Dict[str, Any],
        domain_name: str,
    ) -> Optional[str]:
        role_text = "\n".join([
            card.name,
            card.system_prompt,
            card.communication_protocol,
            card.capability_family or "",
            " ".join(card.capability_tags or []),
        ]).lower()

        cross_task_roles = {
            "aggregation",
            "validation",
            "formatting",
            "task-classification",
            "task-contract-parsing",
            "task-routing",
            "router",
            "naturalplan",
        }
        if (
            card.role_type in {"aggregator", "validator"}
            or (card.capability_family or "").lower() in cross_task_roles
            or any(tag.lower() in cross_task_roles for tag in (card.capability_tags or []))
        ):
            return None

        signals = self._role_family_signals(role_text, signal_terms)
        if not signals:
            return f"{domain_name}-role: specialist is not tied to a declared domain family or cross-family validation/task-parsing evidence"
        if len(signals) > 1:
            return f"{domain_name}-role: specialist mixes domain-family semantics ({', '.join(sorted(signals))})"
        return None

    def _naturalplan_role_signal_terms(self) -> Dict[str, tuple]:
        return {
            "trip": (
                "trip planning", "city", "cities", "flight", "flights", "direct-flight", "direct flight",
                "itinerary", "day range", "day ranges", "stay duration", "shared flight-day",
            ),
            "calendar": (
                "calendar scheduling", "busy block", "busy blocks", "free window", "free windows",
                "proposed time", "24-hour", "weekday", "weekdays", "time slot", "time slots",
                "participant", "participants", "preference", "preferences",
            ),
            "meeting": (
                "meeting planning", "travel matrix", "wait step", "wait steps",
                "meet step", "meet steps", "person", "people", "availability window", "availability windows",
                "am/pm", "route step", "route steps",
            ),
        }

    def _role_family_signals(self, role_text: str, signal_terms: Dict[str, Any]) -> set:
        signals = set()
        for family, terms in signal_terms.items():
            if isinstance(terms, str):
                terms = (terms,)
            if any(term in role_text for term in terms):
                signals.add(family)
        return signals

    def _role_diversity_text(self, role: RoleCard) -> str:
        parts = [role.system_prompt]
        if role.capability_family:
            parts.append(role.capability_family)
        if role.capability_tags:
            parts.append(" ".join(role.capability_tags))
        return "\n".join(part for part in parts if part)

    def _compute_prompt_overlap_monitor(self, pool: List[RoleCard]) -> Dict[str, Any]:
        diversity_roles = [role for role in pool if not role.protected]
        if len(diversity_roles) < 2:
            return {
                "prompt_similarity_mean": 0.0,
                "orthogonal_role_blacklist": [],
            }

        texts = [self._role_diversity_text(role) for role in diversity_roles]
        try:
            role_embs = self.encoder.encode(texts, normalize_embeddings=True)
        except Exception:
            logger.debug("Prompt overlap monitor failed; skipping similarity snapshot.", exc_info=True)
            return {
                "prompt_similarity_mean": 0.0,
                "orthogonal_role_blacklist": [],
            }

        sim_matrix = np.matmul(role_embs, role_embs.T)
        off_diag_mask = ~np.eye(len(diversity_roles), dtype=bool)
        off_diag_values = sim_matrix[off_diag_mask]
        mean_similarity = float(np.mean(off_diag_values)) if off_diag_values.size else 0.0

        row_means = []
        for idx, role in enumerate(diversity_roles):
            other_scores = np.delete(sim_matrix[idx], idx)
            role_mean = float(np.mean(other_scores)) if other_scores.size else 0.0
            row_means.append((role_mean, role.name))
        row_means.sort(key=lambda item: item[0], reverse=True)

        top_k = max(0, self.cfg.orthogonal_role_blacklist_top_k)
        return {
            "prompt_similarity_mean": round(mean_similarity, 5),
            "orthogonal_role_blacklist": [name for _, name in row_means[:top_k]],
        }

    def _compute_pool_family_monitor(self, pool: List[RoleCard]) -> Dict[str, Any]:
        diversity_roles = [role for role in pool if not role.protected] or list(pool)
        all_counts = capability_family_counts(pool)
        counts = capability_family_counts(diversity_roles)
        existing_families = sorted(all_counts)
        missing_families = sorted(self.required_capability_families - set(existing_families))
        family_diversity = (len(counts) / len(diversity_roles)) if diversity_roles else 0.0
        family_dominance = (max(counts.values()) / len(diversity_roles)) if counts and diversity_roles else 0.0
        dominant_families = sorted(
            family
            for family, count in counts.items()
            if (count / len(diversity_roles)) >= self.cfg.max_capability_family_dominance
        ) if diversity_roles else []
        prompt_monitor = self._compute_prompt_overlap_monitor(diversity_roles)
        low_diversity = (
            family_diversity < self.cfg.min_capability_family_diversity
            or bool(dominant_families)
            or prompt_monitor["prompt_similarity_mean"] >= self.cfg.max_role_prompt_similarity
        )
        return {
            "existing_capability_families": existing_families,
            "missing_capability_families": missing_families,
            "family_diversity": round(family_diversity, 5),
            "family_dominance": round(family_dominance, 5),
            "prompt_similarity_mean": prompt_monitor["prompt_similarity_mean"],
            "low_diversity_mode": low_diversity,
            "diversity_status": "low" if low_diversity else "healthy",
            "orthogonal_family_blacklist": dominant_families if low_diversity else [],
            "orthogonal_role_blacklist": prompt_monitor["orthogonal_role_blacklist"] if low_diversity else [],
        }

    def _violates_orthogonal_add(
        self,
        new_card: RoleCard,
        pool: List[RoleCard],
        family_monitor: Dict[str, Any],
    ) -> Optional[str]:
        if not family_monitor.get("low_diversity_mode"):
            return None

        missing_families = set(family_monitor.get("missing_capability_families") or [])
        if new_card.capability_family and new_card.capability_family in missing_families:
            return None

        dominant_families = set(family_monitor.get("orthogonal_family_blacklist") or [])
        if new_card.capability_family and new_card.capability_family in dominant_families:
            return (
                "orthogonal: capability family "
                f"'{new_card.capability_family}' is already dominant in low-diversity mode"
            )

        compare_names = set(family_monitor.get("orthogonal_role_blacklist") or [])
        compare_roles = [
            role for role in pool
            if not role.protected and (not compare_names or role.name in compare_names)
        ]
        if not compare_roles:
            compare_roles = [role for role in pool if not role.protected]
        if not compare_roles:
            return None

        texts = [self._role_diversity_text(role) for role in compare_roles]
        texts.append(self._role_diversity_text(new_card))
        try:
            role_embs = self.encoder.encode(texts, normalize_embeddings=True)
        except Exception:
            logger.debug("Orthogonal add validation failed; skipping prompt-overlap gate.", exc_info=True)
            return None

        candidate_emb = role_embs[-1]
        existing_embs = role_embs[:-1]
        if existing_embs.size == 0:
            return None

        similarities = np.dot(existing_embs, candidate_emb)
        max_idx = int(np.argmax(similarities))
        max_similarity = float(similarities[max_idx])
        if max_similarity >= self.cfg.max_role_prompt_similarity:
            return (
                "orthogonal: new role overlaps with "
                f"'{compare_roles[max_idx].name}' (cosine={max_similarity:.3f}) in low-diversity mode"
            )
        return None

    def _remember_failed_task_context(self, task_prompt: str, score_before: float) -> None:
        if score_before >= 0.999:
            return
        summary = f"score={score_before:.3f} | {task_prompt[:160].strip()}"
        self.recent_failed_task_examples = [
            item for item in self.recent_failed_task_examples if item != summary
        ]
        self.recent_failed_task_examples.append(summary)
        self.recent_failed_task_examples = self.recent_failed_task_examples[-RECENT_FAILED_TASK_LIMIT:]

        for pattern in self._extract_failure_patterns(task_prompt):
            self.recent_failure_patterns = [
                item for item in self.recent_failure_patterns if item != pattern
            ]
            self.recent_failure_patterns.append(pattern)
        self.recent_failure_patterns = self.recent_failure_patterns[-RECENT_FAILURE_PATTERN_LIMIT:]

    def _extract_failure_patterns(self, task_prompt: str) -> List[str]:
        lowered = task_prompt.lower()
        matched = [
            name
            for name, hints in _FAILURE_PATTERN_HINTS.items()
            if any(hint in lowered for hint in hints)
        ]
        return matched or ["general-reasoning-gap"]

    # ── LOO refresh ───────────────────────────────────────────────────────────

    def _loo_refresh(self, sample_tasks: int = 3) -> None:
        if len(self.pool) < self.cfg.loo_min_pool_size:
            logger.info(
                "Skipping LOO refresh: pool size %d < min %d.",
                len(self.pool),
                self.cfg.loo_min_pool_size,
            )
            return

        sample = self.tasks[:sample_tasks]
        if not sample:
            logger.info("Skipping LOO refresh: no sampled tasks.")
            return

        phase_a = PhaseA(
            config=self.cfg,
            client=self.client,
            encoder=self.encoder,
            credit_engine=self.credit_engine,
            pool=self.pool,
            eval_fn=sample[0]["eval_fn"],
        )
        full_pool_loo_refresh(phase_a, sample)
        logger.info("LOO refresh done for %d roles.", len(self.pool))

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, tag: str = "checkpoint") -> None:
        pool_dicts = [r.to_dict() for r in self.pool]
        path = os.path.join(self.results_dir, f"controller_{tag}_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "tag": tag,
            "model_state": self.controller.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "ema_baseline": self.ema_baseline,
            "pool": pool_dicts,
            "recently_removed_role_names": list(self.recently_removed_role_names),
            "credit_engine": self.credit_engine.state_dict(),
        }, path)
        logger.info("Checkpoint saved: %s", path)

        # Also persist pool as human-readable JSON (always overwritten to latest)
        pool_json_path = os.path.join(self.results_dir, "final_pool.json")
        with open(pool_json_path, "w") as f:
            json.dump(pool_dicts, f, indent=2, ensure_ascii=False)
        logger.info("Pool snapshot saved: %s (%d roles)", pool_json_path, len(pool_dicts))

        checkpoint_log_path = os.path.join(
            self.results_dir,
            f"episode_log_{tag}_epoch{epoch}.json",
        )
        self._save_logs(checkpoint_log_path)
        logger.info("Episode log snapshot saved: %s", checkpoint_log_path)

    def _save_logs(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.results_dir, "episode_log.json")
        records = [asdict(r) for r in self.stats.episode_records]
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
