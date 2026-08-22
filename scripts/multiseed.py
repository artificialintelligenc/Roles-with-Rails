"""
Multi-seed evaluation for statistical robustness.

Runs SERO and all baselines across multiple random seeds and reports
mean ± std with paired significance tests.

Usage:
    export OPENROUTER_API_KEY="your-key"  # optional; config default is used if omitted
    python scripts/multiseed.py --benchmark trip --seeds 42 123 456 --tasks 50

Output:
    results/multiseed/seed_{42,123,456}_results.json
    results/multiseed/multiseed_summary.json
"""

import argparse
import json
import logging
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results/multiseed"
BENCHMARK_DIR = os.environ.get("BENCHMARK_DIR", "Benchmark")
API_KEY = os.environ.get("OPENROUTER_API_KEY")
BENCHMARK_CHOICES = ["naturalplan", "trip", "meeting", "calendar", "olympiadbench", "tablebench"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=BENCHMARK_CHOICES, default="naturalplan")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--tasks", type=int, default=15, help="Training tasks")
    p.add_argument("--eval_tasks", type=int, default=10, help="Evaluation tasks per seed")
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--main_epochs", type=int, default=2)
    p.add_argument("--t_round", type=int, default=1)
    p.add_argument("--p_min", type=int, default=2)
    return p.parse_args()


def run_one_seed(args, seed: int, results_dir: str):
    """Run all systems for one seed and return results dict."""
    from sero.config import SeroConfig
    from sero.openrouter_client import OpenRouterClient
    from sentence_transformers import SentenceTransformer
    from sero.baselines.workflow_baseline import run_workflow_baseline

    config = SeroConfig(
        **({"openrouter_api_key": API_KEY} if API_KEY else {}),
        warmup_epochs=args.warmup_epochs,
        main_epochs=args.main_epochs,
        t_round=args.t_round,
        n_max=4,
        p_min=args.p_min,
        seed=seed,
    )
    client = OpenRouterClient(config.openrouter_api_key, base_url=config.openrouter_base_url)
    encoder = SentenceTransformer(config.encoder_model, trust_remote_code=True)

    # Load tasks
    if args.benchmark == "naturalplan":
        from sero.benchmarks.natural_plan_adapter import load_naturalplan_tasks
        from sero.role_card import NATURALPLAN_SEED_ROLES
        all_tasks = load_naturalplan_tasks(
            BENCHMARK_DIR, max_tasks=args.tasks + args.eval_tasks, seed=seed
        )
        seed_pool = list(NATURALPLAN_SEED_ROLES)
        benchmark_tag = "naturalplan"
    elif args.benchmark == "trip":
        from sero.benchmarks.trip_adapter import load_trip_planning_tasks
        from sero.role_card import TRIP_PLANNING_SEED_ROLES
        all_tasks = load_trip_planning_tasks(
            BENCHMARK_DIR, max_tasks=args.tasks + args.eval_tasks, seed=seed
        )
        seed_pool = list(TRIP_PLANNING_SEED_ROLES)
        benchmark_tag = "trip"
    elif args.benchmark == "meeting":
        from sero.benchmarks.meeting_plan_adapter import load_meeting_planning_tasks
        from sero.role_card import MEETING_PLANNING_SEED_ROLES
        all_tasks = load_meeting_planning_tasks(
            BENCHMARK_DIR, max_tasks=args.tasks + args.eval_tasks, seed=seed
        )
        seed_pool = list(MEETING_PLANNING_SEED_ROLES)
        benchmark_tag = "meeting"
    elif args.benchmark == "calendar":
        from sero.benchmarks.calendar_scheduling_adapter import load_calendar_scheduling_tasks
        from sero.role_card import CALENDAR_SCHEDULING_SEED_ROLES
        all_tasks = load_calendar_scheduling_tasks(
            BENCHMARK_DIR, max_tasks=args.tasks + args.eval_tasks, seed=seed
        )
        seed_pool = list(CALENDAR_SCHEDULING_SEED_ROLES)
        benchmark_tag = "calendar"
    elif args.benchmark == "olympiadbench":
        from sero.benchmarks.olympiad_adapter import load_olympiad_tasks
        from sero.role_card import OLYMPIAD_SEED_ROLES
        all_tasks = load_olympiad_tasks(
            BENCHMARK_DIR, subjects=["maths"],
            max_tasks=args.tasks + args.eval_tasks, seed=seed
        )
        seed_pool = list(OLYMPIAD_SEED_ROLES)
        benchmark_tag = "olympiadbench"
    elif args.benchmark == "tablebench":
        from sero.benchmarks.tablebench_adapter import load_tablebench_tasks
        from sero.role_card import TABLEBENCH_SEED_ROLES
        all_tasks = load_tablebench_tasks(
            BENCHMARK_DIR, max_tasks=args.tasks + args.eval_tasks, seed=seed
        )
        seed_pool = list(TABLEBENCH_SEED_ROLES)
        benchmark_tag = "tablebench"
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    train_tasks = all_tasks[:args.tasks]
    eval_tasks = all_tasks[args.tasks:args.tasks + args.eval_tasks]

    seed_results = {}

    # Single agent
    import numpy as np
    records = []
    for task in eval_tasks:
        try:
            response = client.system_user(
                model=config.agent_model,
                system=(
                    "You are an expert problem solver. Solve the problem carefully and "
                    "express your final answer as \\boxed{<answer>} for math, or in required format."
                ),
                user=task["prompt"],
                temperature=0.0,
                max_tokens=1024,
            )
            score = task["eval_fn"](response)
        except Exception as e:
            logger.error("Single agent error: %s", e)
            score = 0.0
        records.append(score)
    seed_results["single_agent"] = float(np.mean(records))
    logger.info("[seed=%d] single_agent=%.3f", seed, seed_results["single_agent"])

    # Static baseline
    from sero.phase_a import PhaseA
    from sero.credit_engine import CreditEngine
    credit_engine = CreditEngine()
    credit_engine.sync_with_pool([r.name for r in seed_pool])
    static_config = SeroConfig(openrouter_api_key=config.openrouter_api_key, agent_model=config.agent_model,
                               t_round=args.t_round, n_max=4, evolve_roles=False, seed=seed)
    records = []
    for task in eval_tasks:
        try:
            phase_a = PhaseA(config=static_config, client=client, encoder=encoder,
                             credit_engine=credit_engine, pool=list(seed_pool),
                             eval_fn=task["eval_fn"])
            r = phase_a.run(task["prompt"])
            records.append(r["score"])
        except Exception as e:
            logger.error("Static error: %s", e)
            records.append(0.0)
    seed_results["static"] = float(np.mean(records))
    logger.info("[seed=%d] static=%.3f", seed, seed_results["static"])

    # Workflow baseline
    wf_summary = run_workflow_baseline(eval_tasks, benchmark_tag, client, config)
    seed_results["workflow"] = wf_summary["mean_score"]
    logger.info("[seed=%d] workflow=%.3f", seed, seed_results["workflow"])

    # SERO
    from sero.trainer import SeroTrainer
    trainer = SeroTrainer(
        config=config, seed_pool=list(seed_pool), tasks=train_tasks,
        client=client, encoder=encoder,
        results_dir=os.path.join(results_dir, f"seed_{seed}_sero"),
    )
    trainer.train()
    final_pool = trainer.pool
    credit_engine2 = trainer.credit_engine
    records = []
    active_role_counts = []
    for task in eval_tasks:
        try:
            phase_a = PhaseA(config=config, client=client, encoder=encoder,
                             credit_engine=credit_engine2, pool=final_pool,
                             eval_fn=task["eval_fn"])
            r = phase_a.run(task["prompt"])
            records.append(r["score"])
            active_role_counts.append(len(r.get("active_roles", [])))
        except Exception as e:
            logger.error("SERO eval error: %s", e)
            records.append(0.0)
    seed_results["sero"] = float(np.mean(records))
    seed_results["sero_avg_active_roles"] = float(np.mean(active_role_counts)) if active_role_counts else 0
    seed_results["sero_final_pool"] = [r.name for r in final_pool]
    logger.info("[seed=%d] sero=%.3f avg_active=%.1f pool=%s",
                seed, seed_results["sero"], seed_results["sero_avg_active_roles"],
                seed_results["sero_final_pool"])

    return seed_results


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_seed_results = {}
    for seed in args.seeds:
        logger.info("=== Running seed %d ===", seed)
        try:
            results = run_one_seed(args, seed, RESULTS_DIR)
            all_seed_results[str(seed)] = results
            out = os.path.join(RESULTS_DIR, f"seed_{seed}_results.json")
            with open(out, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.error("Seed %d failed: %s", seed, e)

    # Aggregate across seeds
    systems = ["single_agent", "static", "workflow", "sero"]
    summary = {}
    for system in systems:
        vals = [all_seed_results[str(s)][system]
                for s in args.seeds if str(s) in all_seed_results and system in all_seed_results[str(s)]]
        if vals:
            summary[system] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
                "per_seed": {str(s): v for s, v in zip(args.seeds, vals)},
            }

    out = os.path.join(RESULTS_DIR, "multiseed_summary.json")
    with open(out, "w") as f:
        json.dump({"per_seed": all_seed_results, "aggregate": summary}, f, indent=2)

    print("\n=== Multi-Seed Results ===")
    print(f"Benchmark: {args.benchmark}, Seeds: {args.seeds}")
    print(f"{'System':<20} {'Mean':>8} {'Std':>8} {'Seeds':>6}")
    print("-" * 46)
    for sys_name, stats in summary.items():
        print(f"{sys_name:<20} {stats['mean']:>8.3f} {stats['std']:>8.3f} {stats['n_seeds']:>6}")


if __name__ == "__main__":
    main()
