"""
Static-seed baseline: use the fixed seed role pool without evolution.

Evaluates performance using only the initial seed roles (no ADD/REMOVE/NOOP
controller actions). Serves as the "no evolution" ablation / lower-bound baseline.
"""

import json
import os
import logging
from typing import List, Dict, Any

import numpy as np

from sero.role_card import RoleCard
from sero.credit_engine import CreditEngine
from sero.phase_a import PhaseA
from sero.config import SeroConfig
from sero.openrouter_client import OpenRouterClient
from sero.benchmarks.scoring_utils import is_dual_score

logger = logging.getLogger(__name__)


def run_static_baseline(
    config: SeroConfig,
    seed_pool: List[RoleCard],
    tasks: List[Dict[str, Any]],
    client: OpenRouterClient,
    encoder,
    results_dir: str = "results",
    tag: str = "static",
) -> Dict[str, Any]:
    """
    Evaluate the static seed pool on all tasks without any evolution.

    Returns summary dict with per-task scores and aggregate stats.
    """
    os.makedirs(results_dir, exist_ok=True)

    credit_engine = CreditEngine(mu=config.ema_mu, alpha=config.fast_credit_alpha)
    credit_engine.sync_with_pool([r.name for r in seed_pool])

    phase_a = PhaseA(
        config=config,
        client=client,
        encoder=encoder,
        credit_engine=credit_engine,
        pool=seed_pool,
        eval_fn=None,       # overridden per-task below
    )

    records = []
    for task in tasks:
        phase_a.eval_fn = task["eval_fn"]
        try:
            result = phase_a.run(task["prompt"], loo_target=None)
            score = result["score"]
            raw_score = result.get("raw_score", score)
        except Exception as e:
            logger.error("Static baseline error on task %s: %s", task["id"], e)
            score = 0.0
            raw_score = 0.0

        records.append({
            "task_id": task["id"],
            "score": score,
            "raw_score": raw_score,
            "pool_size": len(seed_pool),
            "roles": [r.name for r in seed_pool],
        })
        logger.info("[Static] task=%s score=%.3f", task["id"], score)

    scores = [r["score"] for r in records]
    summary = {
        "baseline": tag,
        "n_tasks": len(records),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "records": records,
    }

    sample = records[0].get("raw_score") if records else None
    if sample is not None and is_dual_score(sample):
        vals = [r["raw_score"].get("exact_score", 0.0)
                if isinstance(r.get("raw_score"), dict) else r["score"]
                for r in records]
        summary["mean_exact_score"] = float(np.mean(vals)) if vals else 0.0
        summary["std_exact_score"] = float(np.std(vals)) if vals else 0.0

    out_path = os.path.join(results_dir, f"{tag}_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Static baseline results saved: %s  mean=%.3f", out_path, summary["mean_score"])

    return summary
