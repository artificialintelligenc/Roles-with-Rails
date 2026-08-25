"""Random-evolution baseline: apply uniformly random ADD/REMOVE/NOOP actions
without a controller and without performance gating.

Each episode samples an action uniformly from {ADD, REMOVE, NOOP} and a random
target role, then executes unconditionally (no before/after comparison).

This ablation tests whether SERO's performance gains come from the learned
controller policy or just from having some form of role evolution (A3 ablation).
"""

import json
import os
import logging
import random
from typing import List, Dict, Any, Optional

import numpy as np

from sero.role_card import RoleCard
from sero.credit_engine import CreditEngine
from sero.phase_a import PhaseA
from sero.executor import Executor
from sero.config import SeroConfig
from sero.openrouter_client import OpenRouterClient
from sero.controller import SeroController

logger = logging.getLogger(__name__)


def run_random_role_evolution(
    config: SeroConfig,
    seed_pool: List[RoleCard],
    tasks: List[Dict[str, Any]],
    client: OpenRouterClient,
    encoder,
    results_dir: str = "results",
    tag: str = "random_role_evolution",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Random Role Evolution baseline: for each task, uniformly sample an action from
    {ADD, REMOVE, NOOP} and a random target role, then execute unconditionally.
    No before/after scoring, no performance gating.

    Returns summary dict with per-task evolution log and aggregate stats.
    """
    os.makedirs(results_dir, exist_ok=True)
    rng = random.Random(seed)
    ops = SeroController.OPS

    pool = list(seed_pool)
    executor = Executor(client, config.executor_model)
    credit_engine = CreditEngine(mu=config.ema_mu, alpha=config.fast_credit_alpha)
    credit_engine.sync_with_pool([r.name for r in pool])

    records = []
    for task in tasks:
        # Random action
        op = rng.choice(ops)
        target: Optional[str] = None
        if pool and op != "noop":
            target = rng.choice(pool).name

        # Apply unconditionally — no performance comparison
        if op == "remove" and target is not None:
            # L1 fix: Protect protected roles from removal (consistent with SERO trainer)
            target_role = next((r for r in pool if r.name == target), None)
            if config.protect_critical_roles and target_role is not None and target_role.protected:
                logger.info("[Random] Role '%s' is protected; skipping REMOVE.", target)
            else:
                pool = executor.remove(target, list(pool))
        elif op == "add_anchor" and len(pool) < config.p_max:
            new_card = executor.add_anchor(list(pool), credit_engine, task["prompt"][:200])
            if new_card:
                pool = list(pool) + [new_card]
        # noop or fallback: pool unchanged

        credit_engine.sync_with_pool([r.name for r in pool])

        records.append({
            "task_id": task["id"],
            "op": op,
            "target": target,
            "pool_size": len(pool),
        })
        logger.info("[Random] task=%s op=%s pool_size=%d",
                    task["id"], op, len(pool))

    scores = []  # no per-task scores during evolution phase
    summary = {
        "baseline": tag,
        "n_tasks": len(records),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "records": records,
    }

    out_path = os.path.join(results_dir, f"{tag}_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Random evo results saved: %s  mean=%.3f", out_path, summary["mean_score"])

    return summary
