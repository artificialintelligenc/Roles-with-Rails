"""
Ablation studies: isolate the contribution of each SERO component.

Ablations:
  A1a: --no_credit_state     controller gets zero credit stats
  A1b: --no_credit_dag       random flat topology instead of credit-ranked DAG
  A1c: both A1a + A1b combined
  A3:  --random_controller   no learned policy (random action selection)
  L3:  --no_format_inherit   disable Level-3 format inheritance fix

Usage:
    export OPENROUTER_API_KEY="your-key"
    python scripts/ablation.py --benchmark trip --tasks 20 --ablations A1a A1b A3
    python scripts/ablation.py --benchmark trip --no_format_inherit

Output:
  results/ablations/<ablation>_results.json
  results/ablations/ablation_summary.json
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results/ablations"
BENCHMARK_DIR = os.environ.get("BENCHMARK_DIR", "Benchmark")
BENCHMARK_CHOICES = ["naturalplan", "trip", "meeting", "calendar", "olympiadbench", "tablebench"]

ABLATION_CONFIGS = {
    "A1a": {"use_credit_state": False, "use_credit_dag": True,  "random_controller": False},
    "A1b": {"use_credit_state": True,  "use_credit_dag": False, "random_controller": False},
    "A1c": {"use_credit_state": False, "use_credit_dag": False, "random_controller": False},
    "A3":  {"use_credit_state": True,  "use_credit_dag": True,  "random_controller": True},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--main_epochs", type=int, default=3)
    p.add_argument("--ablations", nargs="+", default=list(ABLATION_CONFIGS.keys()))
    p.add_argument("--benchmark", choices=BENCHMARK_CHOICES, default="naturalplan")
    p.add_argument("--t_round", type=int, default=None)
    p.add_argument("--p_min", type=int, default=2, help="Min pool size")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_ablation(ablation_name, config, seed_pool, train_tasks, eval_tasks,
                  client, encoder, results_dir):
    import numpy as np
    from sero.trainer import SeroTrainer
    from sero.phase_a import PhaseA
    from sero.credit_engine import CreditEngine

    trainer = SeroTrainer(
        config=config,
        seed_pool=list(seed_pool),
        tasks=train_tasks,
        client=client,
        encoder=encoder,
        results_dir=os.path.join(results_dir, ablation_name),
    )
    stats = trainer.train()
    final_pool = trainer.pool
    credit_engine = trainer.credit_engine

    # Evaluate
    records = []
    for task in eval_tasks:
        phase_a = PhaseA(
            config=config,
            client=client,
            encoder=encoder,
            credit_engine=credit_engine,
            pool=final_pool,
            eval_fn=task["eval_fn"],
        )
        try:
            result = phase_a.run(task["prompt"])
            score = result["score"]
        except Exception as e:
            logger.error("[%s] Eval error on %s: %s", ablation_name, task["id"], e)
            score = 0.0
        records.append({"task_id": task["id"], "score": score})
        logger.info("[%s] task=%s score=%.3f", ablation_name, task["id"], score)

    scores = [r["score"] for r in records]
    summary = {
        "ablation": ablation_name,
        "config": ABLATION_CONFIGS[ablation_name],
        "n_tasks": len(records),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "noop_rate": stats.noop_rate(),
        "final_pool": [r.name for r in final_pool],
        "records": records,
    }

    out = os.path.join(results_dir, f"{ablation_name}_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[%s] mean=%.3f std=%.3f", ablation_name, summary["mean_score"], summary["std_score"])
    return summary


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    from sero.config import SeroConfig
    from sero.openrouter_client import OpenRouterClient
    from sentence_transformers import SentenceTransformer

    base_config = SeroConfig(seed=args.seed,
                              warmup_epochs=args.warmup_epochs,
                              main_epochs=args.main_epochs)
    if not base_config.openrouter_api_key:
        logger.error("OPENROUTER_API_KEY not set.")
        sys.exit(1)

    client = OpenRouterClient(base_config.openrouter_api_key, base_url=base_config.openrouter_base_url)
    encoder = SentenceTransformer(base_config.encoder_model, trust_remote_code=True)

    # Load tasks
    if args.benchmark == "naturalplan":
        from sero.benchmarks.natural_plan_adapter import load_naturalplan_tasks
        from sero.role_card import NATURALPLAN_SEED_ROLES
        tasks = load_naturalplan_tasks(BENCHMARK_DIR, max_tasks=args.tasks * 2, seed=args.seed)
        seed_pool = list(NATURALPLAN_SEED_ROLES)
    elif args.benchmark == "trip":
        from sero.benchmarks.trip_adapter import load_trip_planning_tasks
        from sero.role_card import TRIP_PLANNING_SEED_ROLES
        tasks = load_trip_planning_tasks(BENCHMARK_DIR, max_tasks=args.tasks * 2, seed=args.seed)
        seed_pool = list(TRIP_PLANNING_SEED_ROLES)
    elif args.benchmark == "meeting":
        from sero.benchmarks.meeting_plan_adapter import load_meeting_planning_tasks
        from sero.role_card import MEETING_PLANNING_SEED_ROLES
        tasks = load_meeting_planning_tasks(BENCHMARK_DIR, max_tasks=args.tasks * 2, seed=args.seed)
        seed_pool = list(MEETING_PLANNING_SEED_ROLES)
    elif args.benchmark == "calendar":
        from sero.benchmarks.calendar_scheduling_adapter import load_calendar_scheduling_tasks
        from sero.role_card import CALENDAR_SCHEDULING_SEED_ROLES
        tasks = load_calendar_scheduling_tasks(BENCHMARK_DIR, max_tasks=args.tasks * 2, seed=args.seed)
        seed_pool = list(CALENDAR_SCHEDULING_SEED_ROLES)
    elif args.benchmark == "olympiadbench":
        from sero.benchmarks.olympiad_adapter import load_olympiad_tasks
        from sero.role_card import OLYMPIAD_SEED_ROLES
        tasks = load_olympiad_tasks(BENCHMARK_DIR, max_tasks=args.tasks * 2, seed=args.seed)
        seed_pool = list(OLYMPIAD_SEED_ROLES)
    elif args.benchmark == "tablebench":
        from sero.benchmarks.tablebench_adapter import load_tablebench_tasks
        from sero.role_card import TABLEBENCH_SEED_ROLES
        tasks = load_tablebench_tasks(BENCHMARK_DIR, max_tasks=args.tasks * 2, seed=args.seed)
        seed_pool = list(TABLEBENCH_SEED_ROLES)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    train_tasks = tasks[:args.tasks]
    eval_tasks = tasks[args.tasks:args.tasks * 2]

    all_summaries = {}
    for abl_name in args.ablations:
        if abl_name not in ABLATION_CONFIGS:
            logger.warning("Unknown ablation: %s, skipping", abl_name)
            continue

        logger.info("Running ablation: %s — %s", abl_name, ABLATION_CONFIGS[abl_name])

        # Build config with ablation overrides
        abl_kwargs = dict(seed=args.seed,
                          warmup_epochs=args.warmup_epochs,
                          main_epochs=args.main_epochs,
                          p_min=args.p_min,
                          **ABLATION_CONFIGS[abl_name])
        if args.t_round is not None:
            abl_kwargs["t_round"] = args.t_round
        abl_config = SeroConfig(**abl_kwargs)

        summary = run_ablation(
            abl_name, abl_config, seed_pool, train_tasks, eval_tasks,
            client, encoder, RESULTS_DIR,
        )
        all_summaries[abl_name] = summary

    # Save aggregate
    agg_path = os.path.join(RESULTS_DIR, "ablation_summary.json")
    with open(agg_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n=== Ablation Summary ===")
    print(f"{'Ablation':<10} {'use_cstate':>10} {'use_cdag':>10} {'rand_ctrl':>10} {'Mean':>8} {'Std':>8}")
    print("-" * 62)
    for name, s in all_summaries.items():
        cfg = s["config"]
        print(f"{name:<10} {str(cfg['use_credit_state']):>10} {str(cfg['use_credit_dag']):>10} "
              f"{str(cfg['random_controller']):>10} {s['mean_score']:>8.3f} {s['std_score']:>8.3f}")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
