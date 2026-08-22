"""
Train the SERO controller on a planning benchmark.

Usage:
    export OPENROUTER_API_KEY="your-key"
    python scripts/train.py --benchmark trip --tasks 30 --seed 42

Benchmarks: naturalplan | trip | meeting | calendar | olympiadbench | tablebench

Output:
    results/main/controller_checkpoint_epochN.pt  — model checkpoints
    results/main/episode_log.json                 — training trajectory
    results/main/eval_results.json                — final evaluation scores
    results/main/sero_summary.json                — aggregate summary
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results/main"
BENCHMARK_DIR = os.environ.get("BENCHMARK_DIR", "Benchmark")
BENCHMARK_CHOICES = ["naturalplan", "trip", "meeting", "calendar", "olympiadbench", "tablebench"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=30, help="Training tasks")
    p.add_argument("--eval_tasks", type=int, default=20, help="Evaluation tasks (held-out)")
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--main_epochs", type=int, default=5)
    p.add_argument("--t_round", type=int, default=None, help="Override T_ROUND (collaboration rounds)")
    p.add_argument("--p_min", type=int, default=2, help="Min pool size (prevents collapse)")
    p.add_argument("--loo_min_pool_size", type=int, default=None,
                   help="Skip precise LOO refresh when pool size is below this threshold")
    p.add_argument("--new_role_initial_n_updates", type=int, default=None,
                   help="Cold-start n_updates assigned to newly added roles")
    p.add_argument("--benchmark", choices=BENCHMARK_CHOICES, default="naturalplan")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; load checkpoint for eval")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to controller checkpoint for eval_only mode")
    return p.parse_args()


def load_components(args):
    from sero.config import SeroConfig
    from sero.openrouter_client import OpenRouterClient
    from sentence_transformers import SentenceTransformer

    kwargs = dict(seed=args.seed, warmup_epochs=args.warmup_epochs, main_epochs=args.main_epochs,
                  p_min=args.p_min)
    if args.t_round is not None:
        kwargs["t_round"] = args.t_round
    if args.loo_min_pool_size is not None:
        kwargs["loo_min_pool_size"] = args.loo_min_pool_size
    if args.new_role_initial_n_updates is not None:
        kwargs["new_role_initial_n_updates"] = args.new_role_initial_n_updates
    config = SeroConfig(**kwargs)
    if not config.openrouter_api_key:
        logger.error("OPENROUTER_API_KEY not set.")
        sys.exit(1)

    client = OpenRouterClient(config.openrouter_api_key, base_url=config.openrouter_base_url)
    logger.info("Loading encoder: %s", config.encoder_model)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(config.encoder_model, trust_remote_code=True)
    return config, client, encoder


def evaluate_sero(config, pool, credit_engine, client, encoder, eval_tasks, results_dir, tag):
    """Run Phase A (no evolution) on eval tasks with the final trained pool."""
    import numpy as np
    from sero.phase_a import PhaseA

    records = []
    for task in eval_tasks:
        phase_a = PhaseA(
            config=config,
            client=client,
            encoder=encoder,
            credit_engine=credit_engine,
            pool=pool,
            eval_fn=task["eval_fn"],
        )
        try:
            result = phase_a.run(task["prompt"])
            score = result["score"]
            active_roles = result.get("active_roles", [])
        except Exception as e:
            logger.error("Eval error on %s: %s", task["id"], e)
            score = 0.0
            active_roles = []

        records.append({
            "task_id": task["id"],
            "score": score,
            "pool_size": len(pool),
            "active_roles": active_roles,
            "n_active": len(active_roles),
        })
        logger.info("[Eval] task=%s score=%.3f active=%d", task["id"], score, len(active_roles))

    scores = [r["score"] for r in records]
    avg_active = float(np.mean([r["n_active"] for r in records])) if records else 0.0
    summary = {
        "system": tag,
        "n_tasks": len(records),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "avg_active_roles": avg_active,
        "final_pool": [p.name for p in pool],
        "records": records,
    }
    out = os.path.join(results_dir, f"{tag}_eval.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Eval (%s): mean=%.3f std=%.3f", tag, summary["mean_score"], summary["std_score"])
    return summary


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    config, client, encoder = load_components(args)

    # Load tasks with train/eval split
    all_tasks = []
    seed_pool_cls = None

    if args.benchmark == "naturalplan":
        from sero.benchmarks.natural_plan_adapter import load_naturalplan_tasks
        from sero.role_card import NATURALPLAN_SEED_ROLES
        naturalplan_tasks = load_naturalplan_tasks(
            BENCHMARK_DIR,
            max_tasks=args.tasks + args.eval_tasks,
            seed=args.seed,
        )
        all_tasks.extend(naturalplan_tasks)
        seed_pool_cls = NATURALPLAN_SEED_ROLES
        logger.info("Loaded %d combined NaturalPlan tasks", len(naturalplan_tasks))

    elif args.benchmark == "trip":
        from sero.benchmarks.trip_adapter import load_trip_planning_tasks
        from sero.role_card import TRIP_PLANNING_SEED_ROLES
        trip_tasks = load_trip_planning_tasks(BENCHMARK_DIR,
                                               max_tasks=args.tasks + args.eval_tasks,
                                               seed=args.seed)
        all_tasks.extend(trip_tasks)
        seed_pool_cls = TRIP_PLANNING_SEED_ROLES
        logger.info("Loaded %d trip tasks", len(trip_tasks))

    elif args.benchmark == "meeting":
        from sero.benchmarks.meeting_plan_adapter import load_meeting_planning_tasks
        from sero.role_card import MEETING_PLANNING_SEED_ROLES
        meeting_tasks = load_meeting_planning_tasks(
            BENCHMARK_DIR,
            max_tasks=args.tasks + args.eval_tasks,
            seed=args.seed,
        )
        all_tasks.extend(meeting_tasks)
        seed_pool_cls = MEETING_PLANNING_SEED_ROLES
        logger.info("Loaded %d meeting tasks", len(meeting_tasks))

    elif args.benchmark == "calendar":
        from sero.benchmarks.calendar_scheduling_adapter import load_calendar_scheduling_tasks
        from sero.role_card import CALENDAR_SCHEDULING_SEED_ROLES
        calendar_tasks = load_calendar_scheduling_tasks(
            BENCHMARK_DIR,
            max_tasks=args.tasks + args.eval_tasks,
            seed=args.seed,
        )
        all_tasks.extend(calendar_tasks)
        seed_pool_cls = CALENDAR_SCHEDULING_SEED_ROLES
        logger.info("Loaded %d calendar tasks", len(calendar_tasks))

    elif args.benchmark == "olympiadbench":
        from sero.benchmarks.olympiad_adapter import load_olympiad_tasks
        from sero.role_card import OLYMPIAD_SEED_ROLES
        olympiad_tasks = load_olympiad_tasks(
            BENCHMARK_DIR,
            max_tasks=args.tasks + args.eval_tasks,
            seed=args.seed,
        )
        all_tasks.extend(olympiad_tasks)
        seed_pool_cls = OLYMPIAD_SEED_ROLES
        logger.info("Loaded %d OlympiadBench tasks", len(olympiad_tasks))

    elif args.benchmark == "tablebench":
        from sero.benchmarks.tablebench_adapter import load_tablebench_tasks
        from sero.role_card import TABLEBENCH_SEED_ROLES
        tablebench_tasks = load_tablebench_tasks(
            BENCHMARK_DIR,
            max_tasks=args.tasks + args.eval_tasks,
            seed=args.seed,
        )
        all_tasks.extend(tablebench_tasks)
        seed_pool_cls = TABLEBENCH_SEED_ROLES
        logger.info("Loaded %d TableBench tasks", len(tablebench_tasks))

    if not all_tasks:
        logger.error("No tasks loaded.")
        sys.exit(1)

    # Train/eval split
    train_tasks = all_tasks[:args.tasks]
    eval_tasks = all_tasks[args.tasks:args.tasks + args.eval_tasks]
    seed_pool = list(seed_pool_cls)
    logger.info("Train: %d  Eval: %d  Pool: %d roles", len(train_tasks), len(eval_tasks), len(seed_pool))

    if not args.eval_only:
        # ── Training ──────────────────────────────────────────────────────────
        from sero.trainer import SeroTrainer

        trainer = SeroTrainer(
            config=config,
            seed_pool=seed_pool,
            tasks=train_tasks,
            client=client,
            encoder=encoder,
            results_dir=RESULTS_DIR,
        )
        stats = trainer.train()
        final_pool = trainer.pool
        credit_engine = trainer.credit_engine

        # Save training summary
        train_summary = {
            "episodes": stats.episodes,
            "mean_reward": stats.total_reward / max(1, stats.episodes),
            "noop_rate": stats.noop_rate(),
            "add_count": stats.add_count,
            "remove_count": stats.remove_count,
            "final_pool": [r.name for r in final_pool],
            "policy_loss": stats.policy_loss,
        }
        with open(os.path.join(RESULTS_DIR, "training_summary.json"), "w") as f:
            json.dump(train_summary, f, indent=2)
        logger.info("Training complete: %s", train_summary)

    else:
        # ── Load checkpoint ───────────────────────────────────────────────────
        import torch
        from sero.controller import SeroController
        from sero.credit_engine import CreditEngine

        if args.checkpoint is None:
            # Find latest checkpoint
            import glob as _glob
            ckpts = sorted(_glob.glob(os.path.join(RESULTS_DIR, "controller_*.pt")))
            if not ckpts:
                logger.error("No checkpoint found in %s", RESULTS_DIR)
                sys.exit(1)
            args.checkpoint = ckpts[-1]

        logger.info("Loading checkpoint: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        final_pool = [_rc_from_dict(d) for d in ckpt["pool"]]
        credit_engine = CreditEngine()
        credit_engine.sync_with_pool([r.name for r in final_pool])

    # ── Evaluation ────────────────────────────────────────────────────────────
    if eval_tasks:
        eval_summary = evaluate_sero(
            config, final_pool, credit_engine, client, encoder,
            eval_tasks, RESULTS_DIR, tag="sero_main",
        )
        print("\n=== SERO Main Results ===")
        print(f"Mean score: {eval_summary['mean_score']:.3f} ± {eval_summary['std_score']:.3f}")
        print(f"Final pool ({len(final_pool)} roles): {eval_summary['final_pool']}")
    else:
        logger.warning("No eval tasks available.")


def _rc_from_dict(d):
    from sero.role_card import RoleCard
    return RoleCard.from_dict(d)


if __name__ == "__main__":
    main()
