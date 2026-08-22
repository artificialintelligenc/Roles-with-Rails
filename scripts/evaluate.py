"""
Single-system evaluation for SERO benchmarks.

Evaluate one system at a time on a specified benchmark.
Systems: cot | sc | static | static_dag | workflow | random_evo | sero

Usage:
    export OPENROUTER_API_KEY="your-key"
    python scripts/evaluate.py --system cot --benchmark trip --eval_tasks 15
    python scripts/evaluate.py --system sc --benchmark trip --eval_tasks 15
    python scripts/evaluate.py --system static --benchmark trip --eval_tasks 15
    python scripts/evaluate.py --system static_dag --benchmark trip --eval_tasks 15
    python scripts/evaluate.py --system workflow --benchmark trip --eval_tasks 15
    python scripts/evaluate.py --system random_evo --benchmark trip --tasks 20 --eval_tasks 15
    python scripts/evaluate.py --system sero --benchmark trip --tasks 20 --eval_tasks 15
    python scripts/evaluate.py --system sero --benchmark trip --eval_tasks 15 --checkpoint path/to/ckpt.pt

Benchmarks: naturalplan | trip | meeting | calendar | olympiadbench | tablebench

Output:
    results/evaluation/{benchmark}_{system}{suffix}.json
"""

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from sero.benchmarks.scoring_utils import normalize_score, is_dual_score
from sero.config import SeroConfig

EVAL_WORKERS = 5
RESULTS_DIR = "results/evaluation"
BENCHMARK_DIR = os.environ.get("BENCHMARK_DIR", "Benchmark")
LOAD_ALL_TASKS = 10 ** 9
BENCHMARK_CHOICES = ["naturalplan", "trip", "meeting", "calendar", "olympiadbench", "tablebench"]

SPLIT_KEY_MAP = {
    "naturalplan": "naturalplan",
    "trip": "trip_planning",
    "meeting": "meeting_planning",
    "calendar": "calendar_scheduling",
    "olympiadbench": "olympiad",
    "tablebench": "tablebench",
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a single system on a SERO benchmark.")
    default_config = SeroConfig()
    p.add_argument("--system", required=True,
                   choices=["cot", "sc", "static", "static_dag", "workflow", "random_evo", "sero"],
                   help="System to evaluate")
    p.add_argument("--benchmark", required=True,
                   choices=BENCHMARK_CHOICES)
    p.add_argument("--tasks", type=int, default=20,
                   help="Training tasks (used by random_evo and sero)")
    p.add_argument("--eval_tasks", type=int, default=15, help="Evaluation tasks")
    p.add_argument("--sc_k", type=int, default=3, help="SC sampling count")
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--main_epochs", type=int, default=2)
    p.add_argument("--t_round", type=int, default=1)
    p.add_argument("--n_max", type=int, default=default_config.n_max)
    p.add_argument("--p_max", type=int, default=default_config.p_max,
                   help="Maximum role-pool size before add_anchor is masked")
    p.add_argument("--p_min", type=int, default=2)
    p.add_argument("--specialist_slots", type=int, default=None,
                   help="Explicit specialist retrieval budget for Phase A (default: config)")
    p.add_argument("--validator_slots", type=int, default=None,
                   help="Reserved validator retrieval budget when conditional validator routing is enabled")
    p.add_argument("--batch_size", type=int, default=None,
                   help="REINFORCE batch size (default: config default)")
    p.add_argument("--loo_refresh", type=int, default=None,
                   help="LOO refresh interval in episodes (default: config default)")
    p.add_argument("--entropy_beta", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ema_mu", type=float, default=None,
                   help="EMA decay for historical credit (default: config)")
    p.add_argument("--fast_credit_alpha", type=float, default=None,
                   help="Fast Credit: sim_query weight (default: config)")
    p.add_argument("--exploration_gamma", type=float, default=None,
                   help="Credit-maintenance exploration bonus weight (default: config)")
    p.add_argument("--loo_min_pool_size", type=int, default=None,
                   help="Skip precise LOO refresh when pool size is below this threshold")
    p.add_argument("--new_role_initial_n_updates", type=int, default=None,
                   help="Cold-start n_updates assigned to newly added roles")
    p.add_argument("--controller_hidden_dim", type=int, default=default_config.d_h,
                   help="Controller hidden width d_h (default: config)")
    p.add_argument("--controller_encoder_layers", type=int, default=default_config.controller_encoder_layers,
                   help="Number of hidden layers in the controller state encoder")
    p.add_argument("--controller_target_layers", type=int, default=default_config.controller_target_layers,
                   help="Number of hidden layers in the controller target head before the scalar output")
    p.add_argument("--controller_scale_label", type=str, default="",
                   help="Optional scale label recorded in run metadata")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="SERO checkpoint path for eval-only mode")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Directory for SERO training checkpoints and logs")
    p.add_argument("--noop_collapse_threshold", type=float, default=None,
                   help="Early-stop threshold for the collapse detector (default: config)")
    p.add_argument("--disable_conditional_validator_pass", action="store_true",
                   help="Let validator roles compete inside the main retrieval budget")
    p.add_argument("--disable_seed_family_coverage", action="store_true",
                   help="Disable capability-family coverage constraints when removing roles")
    p.add_argument("--min_capability_family_diversity", type=float, default=None,
                   help="Minimum diversity ratio across capability families before removals are blocked")
    p.add_argument("--max_capability_family_dominance", type=float, default=None,
                   help="Maximum allowed single-family dominance ratio before removals are blocked")
    p.add_argument("--max_role_prompt_similarity", type=float, default=None,
                   help="Similarity threshold above which near-duplicate roles are blocked")
    p.add_argument("--orthogonal_role_blacklist_top_k", type=int, default=None,
                   help="How many nearest existing roles to blacklist when proposing a new role")
    p.add_argument("--preserve_train_order", action="store_true",
                   help="Disable training-task shuffling for trainable systems")
    p.add_argument("--shuffle_train_tasks", action="store_true",
                   help="Force training-task shuffling even for benchmarks that preserve split order by default")
    p.add_argument("--suffix", type=str, default="")
    p.add_argument("--no_format_inherit", action="store_true",
                   help="Disable format inheritance (ablation)")
    p.add_argument("--no_credit_state", action="store_true",
                   help="Disable controller-visible credit-memory features while keeping role evolution enabled")
    p.add_argument("--no_active_set_credit", action="store_true",
                   help="Select active roles using query-role similarity only")
    p.add_argument("--no_credit_dag", action="store_true",
                   help="Disable credit-informed DAG ordering while keeping DAG construction enabled")
    p.add_argument("--random_controller", action="store_true",
                   help="Replace the learned controller with uniform random legal actions")
    p.add_argument("--no_protect", action="store_true",
                   help="Disable critical-role removal protection while keeping other SERO mechanisms enabled")
    p.add_argument("--joint_ablate_credit", action="store_true",
                   help="Jointly disable controller credit state, active-set credit, and credit-DAG ordering")
    p.add_argument("--joint_ablate_evolution", action="store_true",
                   help="Freeze role-pool evolution: controller still runs, but add/remove actions do not change the committed pool")
    p.add_argument("--joint_ablate_protect", action="store_true",
                   help="Jointly disable protected-role removal guards and the post-aggregator validator check")
    p.add_argument("--joint_ablate_controller_reward", action="store_true",
                   help="Disable controller REINFORCE loss / backward while keeping the rest of SERO active")
    p.add_argument("--use_split", action="store_true",
                    help="Use train_split.json for train/eval separation. "
                        "Training systems (sero, random_evo) get the benchmark's fixed train keys; "
                        "supported benchmarks' eval uses the split-defined heldout/subset keys.")
    p.add_argument("--eval_subset", action="store_true",
                   help="Use the pre-sampled 100-task stratified eval subset instead of "
                        "full test set. Requires --use_split and train_split.json with "
                        "eval_subset_keys. Overrides --eval_tasks.")
    p.add_argument("--split_file", type=str, default=None,
                   help="Path to train_split.json (default: Benchmark/train_split.json, fallback Benchmark/natural-plan/data/train_split.json)")
    p.add_argument("--eval_set", type=str, default=None,
                   choices=["legacy", "heldout", "subset", "natural_full"],
                   help="Explicitly choose the evaluation set. Split-supported benchmarks: "
                        "heldout = exclude fixed train keys, subset = fixed 100-task subset, "
                        "natural_full = original full benchmark without split filtering. "
                        "If omitted, resolve from legacy flags (--eval_subset -> subset, "
                        "--use_split -> heldout, otherwise legacy).")
    return p.parse_args()


def _supports_split_benchmark(benchmark):
    return benchmark in SPLIT_KEY_MAP


def _default_split_path():
    canonical = os.path.join(BENCHMARK_DIR, "train_split.json")
    legacy = os.path.join(BENCHMARK_DIR, "natural-plan", "data", "train_split.json")
    if os.path.exists(canonical):
        return canonical
    return legacy


def _resolve_eval_set(args):
    if args.eval_set is not None:
        if args.eval_subset and args.eval_set != "subset":
            raise ValueError("--eval_subset conflicts with --eval_set unless eval_set=subset")
        if args.use_split and args.eval_set == "legacy":
            raise ValueError("--use_split conflicts with --eval_set=legacy")
        return args.eval_set

    if args.eval_subset:
        return "subset"
    if args.use_split:
        return "heldout"
    return "legacy"


def _resolve_task_limit(task_count):
    return task_count if task_count > 0 else LOAD_ALL_TASKS


def _result_output_path(benchmark, system, suffix=""):
    return os.path.join(RESULTS_DIR, f"{benchmark}_{system}{suffix}.json")


def _mask_secret(value):
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _get_git_commit(repo_root):
    try:
        full = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return full, short
    except Exception:
        return None, None


def _estimate_controller_param_count(config):
    from sero.controller import SeroController

    return SeroController(config).param_count


def _build_run_manifest(args, config, eval_set, result_path, checkpoint_dir=None):
    script_path = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(script_path))
    git_commit, git_commit_short = _get_git_commit(repo_root)

    effective_config = asdict(config)
    effective_config["openrouter_api_key"] = _mask_secret(
        effective_config.get("openrouter_api_key", "")
    )

    architecture_switches = {
        "controller_credit_state_enabled": config.use_credit_state,
        "active_set_credit_enabled": config.use_active_set_credit,
        "credit_dag_enabled": config.use_credit_dag,
        "random_controller_enabled": config.random_controller,
        "role_evolution_enabled": config.evolve_roles and not config.freeze_role_pool,
        "role_pool_frozen": config.freeze_role_pool,
        "protect_critical_roles_enabled": config.protect_critical_roles,
        "post_aggregator_validator_check_enabled": config.post_aggregator_validator_check,
        "controller_reward_training_enabled": config.controller_reward_training,
    }
    joint_ablation_groups = []
    if getattr(args, "joint_ablate_credit", False):
        joint_ablation_groups.append("credit mechanisms")
    if getattr(args, "joint_ablate_evolution", False):
        joint_ablation_groups.append("evolution mechanisms")
    if getattr(args, "joint_ablate_protect", False):
        joint_ablation_groups.append("protect mechanisms")
    if getattr(args, "joint_ablate_controller_reward", False):
        joint_ablation_groups.append("controller reward")
    architecture_ablation = []
    if not config.use_credit_state:
        architecture_ablation.append("w/o credit state")
    if not config.use_active_set_credit:
        architecture_ablation.append("w/o active-set credit")
    if not config.use_credit_dag:
        architecture_ablation.append("w/o credit DAG")
    if config.random_controller:
        architecture_ablation.append("random controller")
    if not config.protect_critical_roles:
        architecture_ablation.append("w/o protect")
    if config.freeze_role_pool:
        architecture_ablation.append("frozen role pool")
    if not config.post_aggregator_validator_check:
        architecture_ablation.append("w/o post-aggregator validator check")
    if not config.controller_reward_training:
        architecture_ablation.append("w/o controller reward training")
    if not architecture_ablation:
        architecture_ablation.append("full SERO")

    controller_scale_label = getattr(args, "controller_scale_label", "") or None
    controller_param_count = None
    if args.system == "sero":
        controller_param_count = _estimate_controller_param_count(config)

    resolved_run_args = {
        "system": args.system,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "tasks": args.tasks,
        "eval_tasks": args.eval_tasks,
        "eval_set": eval_set,
        "use_split": args.use_split,
        "eval_subset": args.eval_subset,
        "checkpoint": args.checkpoint,
        "format_inherit": not args.no_format_inherit,
        "warmup_epochs": config.warmup_epochs,
        "main_epochs": config.main_epochs,
        "t_round": config.t_round,
        "n_max": config.n_max,
        "p_max": config.p_max,
        "p_min": config.p_min,
        "specialist_slots": config.specialist_slots,
        "conditional_validator_pass": config.conditional_validator_pass,
        "validator_slots": config.validator_slots,
        "batch_size": config.batch_size,
        "loo_refresh_interval": config.loo_refresh_interval,
        "entropy_beta": config.entropy_beta,
        "lr": config.lr,
        "ema_mu": config.ema_mu,
        "fast_credit_alpha": config.fast_credit_alpha,
        "exploration_gamma": config.exploration_gamma,
        "loo_min_pool_size": config.loo_min_pool_size,
        "new_role_initial_n_updates": config.new_role_initial_n_updates,
        "controller_hidden_dim": config.d_h,
        "controller_encoder_layers": config.controller_encoder_layers,
        "controller_target_layers": config.controller_target_layers,
        "controller_scale_label": controller_scale_label,
        "controller_param_count": controller_param_count,
        "shuffle_train_tasks": config.shuffle_train_tasks,
        "use_credit_state": config.use_credit_state,
        "use_active_set_credit": config.use_active_set_credit,
        "use_credit_dag": config.use_credit_dag,
        "random_controller": config.random_controller,
        "evolve_roles": config.evolve_roles,
        "freeze_role_pool": config.freeze_role_pool,
        "protect_critical_roles": config.protect_critical_roles,
        "post_aggregator_validator_check": config.post_aggregator_validator_check,
        "controller_reward_training": config.controller_reward_training,
        "preserve_seed_family_coverage": config.preserve_seed_family_coverage,
        "min_capability_family_diversity": config.min_capability_family_diversity,
        "max_capability_family_dominance": config.max_capability_family_dominance,
        "max_role_prompt_similarity": config.max_role_prompt_similarity,
        "orthogonal_role_blacklist_top_k": config.orthogonal_role_blacklist_top_k,
        "architecture_switches": architecture_switches,
        "architecture_ablation": architecture_ablation,
        "joint_ablation_groups": joint_ablation_groups,
    }

    run_metadata = {
        "script_path": script_path,
        "repo_root": repo_root,
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "argv": list(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_commit_short": git_commit_short,
        "result_path": os.path.abspath(result_path),
        "checkpoint_dir": os.path.abspath(checkpoint_dir) if checkpoint_dir else None,
    }

    return {
        "cli_args": vars(args).copy(),
        "effective_config": effective_config,
        "resolved_run_args": resolved_run_args,
        "architecture_switches": architecture_switches,
        "architecture_ablation": architecture_ablation,
        "joint_ablation_groups": joint_ablation_groups,
        "run_metadata": run_metadata,
    }


def _write_run_config(checkpoint_dir, run_manifest):
    if not checkpoint_dir:
        return None
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_config_path = os.path.join(checkpoint_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(run_manifest, f, indent=2, ensure_ascii=False)
    return run_config_path


def _checkpoint_episode_log_path(checkpoint_dir, epoch, tag="checkpoint"):
    if checkpoint_dir is None or epoch is None:
        return None
    return os.path.join(checkpoint_dir, f"episode_log_{tag}_epoch{epoch}.json")


def _load_checkpoint_episode_records(checkpoint_path, epoch=None, tag="checkpoint", expected_count=None):
    ckpt_dir = os.path.dirname(checkpoint_path)
    candidates = []
    epoch_log_path = _checkpoint_episode_log_path(ckpt_dir, epoch, tag)
    if epoch_log_path is not None:
        candidates.append((epoch_log_path, False))
    candidates.append((os.path.join(ckpt_dir, "episode_log.json"), True))

    for path, is_fallback in candidates:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            records = json.load(f)
        truncated = False
        if is_fallback and expected_count is not None and len(records) > expected_count:
            records = records[:expected_count]
            truncated = True
        return records, path, truncated

    return [], None, False


def _mean(values):
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 5) if vals else None


def _rate(records, key):
    if not records:
        return 0.0
    return round(sum(1 for record in records if record.get(key)) / len(records), 5)


def _build_monitoring_summary(records, final_pool_detail, train_episode_records=None, encoder=None):
    """Summarize role usage, repair, diversity, and dead-role diagnostics."""
    train_episode_records = train_episode_records or []
    role_details = {role.get("name"): role for role in final_pool_detail if role.get("name")}
    active_sets = [tuple(record.get("active_roles") or []) for record in records]
    active_roles = set()
    for active in active_sets:
        active_roles.update(active)

    dead_roles = [role.get("name") for role in final_pool_detail if role.get("name") not in active_roles]
    role_count = len(final_pool_detail)
    all_families = {role.get("capability_family") for role in final_pool_detail if role.get("capability_family")}
    active_families = {
        role_details[name].get("capability_family")
        for name in active_roles
        if name in role_details and role_details[name].get("capability_family")
    }
    validator_roles = {
        role.get("name")
        for role in final_pool_detail
        if role.get("role_type") == "validator"
        or role.get("capability_family") == "validation"
        or "validation" in (role.get("capability_tags") or [])
    }
    validator_active = [any(name in validator_roles for name in active) for active in active_sets]

    prompt_similarity_mean = None
    if encoder is not None and len(final_pool_detail) > 1:
        try:
            import numpy as np
            prompts = [role.get("system_prompt", "") for role in final_pool_detail]
            embeddings = encoder.encode(prompts, normalize_embeddings=True)
            sims = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sims.append(float(np.dot(embeddings[i], embeddings[j])))
            prompt_similarity_mean = round(float(np.mean(sims)), 5) if sims else None
        except Exception as exc:
            logger.warning("Failed to compute pool prompt similarity: %s", exc)

    return {
        "eval": {
            "unique_active_set_count": len(set(active_sets)),
            "dead_role_ratio": round(len(dead_roles) / role_count, 5) if role_count else 0.0,
            "dead_roles": dead_roles,
            "capability_family_coverage_ratio": (
                round(len(active_families) / len(all_families), 5) if all_families else 0.0
            ),
            "validator_active_task_rate": (
                round(sum(validator_active) / len(validator_active), 5) if validator_active else 0.0
            ),
            "repair_trigger_rate": _rate(records, "repair_triggered"),
            "validator_adoption_rate": _rate(records, "validator_adopted"),
        },
        "train": {
            "episode_count": len(train_episode_records),
            "add_anchor_count": sum(1 for record in train_episode_records if record.get("op_type") == "add_anchor"),
            "orthogonal_invalid_add_count": sum(
                1 for record in train_episode_records
                if "orthogonal" in str(record.get("rejection_reason") or "").lower()
            ),
            "low_diversity_episode_count": sum(
                1 for record in train_episode_records if record.get("low_diversity_mode")
            ),
            "mean_family_diversity": _mean(record.get("family_diversity") for record in train_episode_records),
            "repair_trigger_rate": _rate(train_episode_records, "repair_triggered_after"),
            "validator_adoption_rate": _rate(train_episode_records, "validator_adopted_after"),
        },
        "pool": {
            "role_count": role_count,
            "capability_family_count": len(all_families),
            "prompt_similarity_mean": prompt_similarity_mean,
        },
    }


# ── Benchmark Loading ──────────────────────────────────────────────────────────


def _load_benchmark(benchmark, total_tasks, seed,
                    include_keys=None, exclude_keys=None,
                    meeting_min_people=None):
    """Load tasks, seed pool, and single-agent system prompt for a benchmark.

    Args:
        include_keys: Load ONLY these task keys (for training split).
        exclude_keys: Exclude these task keys before sampling (for eval split).
    """
    if benchmark == "naturalplan":
        from sero.benchmarks.natural_plan_adapter import load_naturalplan_tasks
        from sero.role_card import NATURALPLAN_SEED_ROLES
        tasks = load_naturalplan_tasks(
            BENCHMARK_DIR, max_tasks=total_tasks, seed=seed,
            include_keys=include_keys, exclude_keys=exclude_keys)
        seed_pool = list(NATURALPLAN_SEED_ROLES)
        sa_system = (
            "You are an expert NaturalPlan solver. First classify the task as exactly one of: "
            "trip planning, calendar scheduling, or meeting planning. Do not let the word 'meeting' "
            "alone choose the meeting-planning format: calendar scheduling tasks also ask to schedule "
            "a meeting, but they have participants' busy intervals and no travel matrix. If the prompt contains "
            "an appended IMPORTANT marker such as CALENDAR SCHEDULING, TRIP PLANNING, or MEETING PLANNING, "
            "treat that marker as authoritative. For calendar "
            "scheduling, output ONLY 'Here is the proposed time: <Day>, HH:MM - HH:MM' and never output travel, "
            "wait, or 'You meet ...' steps. For meeting planning, output only travel/wait/meet route steps. "
            "For trip planning, output only day-by-day visit and flight itinerary lines."
        )
    elif benchmark == "trip":
        from sero.benchmarks.trip_adapter import load_trip_planning_tasks
        from sero.role_card import TRIP_PLANNING_SEED_ROLES
        tasks = load_trip_planning_tasks(
            BENCHMARK_DIR, max_tasks=total_tasks, seed=seed,
            include_keys=include_keys, exclude_keys=exclude_keys)
        seed_pool = list(TRIP_PLANNING_SEED_ROLES)
        sa_system = (
            "You are an expert travel planner. Produce a day-by-day trip itinerary "
            "satisfying all constraints."
        )
    elif benchmark == "meeting":
        from sero.benchmarks.meeting_plan_adapter import load_meeting_planning_tasks
        from sero.role_card import MEETING_PLANNING_SEED_ROLES
        tasks = load_meeting_planning_tasks(
            BENCHMARK_DIR, max_tasks=total_tasks, seed=seed,
            include_keys=include_keys, exclude_keys=exclude_keys,
            **({"min_people": meeting_min_people} if meeting_min_people is not None else {}))
        seed_pool = list(MEETING_PLANNING_SEED_ROLES)
        sa_system = (
            "You are an expert scheduler. Find a schedule to meet as many people as possible "
            "respecting all constraints. Output your plan step by step using this format: "
            "'You travel to [Location] in [N] minutes and arrive at [Time].' "
            "'You wait until [Time].' "
            "'You meet [Person] for [N] minutes from [Start] to [End].'"
        )
    elif benchmark in ("olympiadbench", "olympiad"):
        from sero.benchmarks.olympiad_adapter import load_olympiad_tasks
        from sero.role_card import OLYMPIAD_SEED_ROLES
        tasks = load_olympiad_tasks(
            BENCHMARK_DIR, max_tasks=total_tasks, seed=seed,
            include_keys=include_keys, exclude_keys=exclude_keys,
        )
        seed_pool = list(OLYMPIAD_SEED_ROLES)
        sa_system = (
            "You are an expert olympiad problem solver for mathematics and physics. "
            "Solve the problem step by step. "
            "Express final answer as \\boxed{<answer>}."
        )
    elif benchmark == "tablebench":
        from sero.benchmarks.tablebench_adapter import load_tablebench_tasks
        from sero.role_card import TABLEBENCH_SEED_ROLES
        tasks = load_tablebench_tasks(
            BENCHMARK_DIR,
            max_tasks=total_tasks,
            seed=seed,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
        seed_pool = list(TABLEBENCH_SEED_ROLES)
        sa_system = (
            "You are an expert table question-answering solver. Use only the given table and question. "
            "Do not solve visualization, chart-generation, plotting, or code-execution tasks. "
            "First identify the answer target column or label. If the question asks which date, time, team, district, "
            "row, or entity has the highest/lowest value, return that requested label/entity, not the numeric value itself. "
            "For extreme-row questions, use the primary semantic row identifier, usually the first meaningful column such as date, year, name, district, team, or episode no.; ignore generic index columns like Unnamed: 0 unless explicitly requested. "
            "If a table has both date and clock/origin-time columns, use the date/event row label for broad 'which time' extreme questions unless clock time is explicitly requested. "
            "For TV episode anomaly questions, answer with Episode <no> from the episode/no column, not the generic index or title alone. "
            "For ranking questions, compute extrema differences only when the question asks for a difference; for ordinal/domain-specific questions, "
            "return the exact period or column label. For statistical, correlation, trend, and causal questions, "
            "answer the table-supported statistic or association instead of refusing causal wording, preserving the requested unit/scale and avoiding percent signs unless the final value is truly a percentage. "
            "For difference-from-average questions, compute average over all numeric rows, then output value minus average on the same column scale unless a relative percent is unambiguously required. "
            "If the requested derived column already exists, such as pop density (per km2), use that column directly and average every numeric row in it; do not recompute from raw population/area unless the derived column is absent. "
            "For highest-density versus average-density questions, output '<entity>, <highest density - average density>' on the density scale, not a relative percent. "
            "For anomaly questions, give only the minimal dominant anomalies, normally at most two when no count is specified, and no weaker extras; include the semantic row identifier, or a 1-indexed data-row number if no row label exists, plus abnormal column/value. "
            "Prefer anomalies where multiple related columns are jointly extreme; unknown/range formatting alone is weaker than rows with all relevant numeric columns extremely high or extremely low. "
            "For casualty/death tables, choose rows whose military, civilian, total deaths, wounded, and total casualties are collectively extreme high or collectively tiny; do not select rows merely because values are unknown or ranges. "
            "Row numbers are 1-indexed over data rows after the header; do not use zero-based positions or generic index-column values as row numbers. "
            "For descriptive-analysis questions, cover the column meanings, range, notable extrema/trends, and missing or unknown values. "
            "Reason over columns, rows, units, and values, then output exactly one line: "
            "Final Answer: <answer>."
        )
    elif benchmark == "calendar":
        from sero.benchmarks.calendar_scheduling_adapter import load_calendar_scheduling_tasks
        from sero.role_card import CALENDAR_SCHEDULING_SEED_ROLES
        tasks = load_calendar_scheduling_tasks(
            BENCHMARK_DIR, max_tasks=total_tasks, seed=seed,
            include_keys=include_keys, exclude_keys=exclude_keys)
        seed_pool = list(CALENDAR_SCHEDULING_SEED_ROLES)
        sa_system = (
            "You are an expert calendar scheduler. Given participants' existing schedules "
            "and a required meeting duration, find a time when all participants are free. "
            "Output ONLY: 'Here is the proposed time: <Day>, HH:MM - HH:MM' using 24-hour format."
        )
    elif benchmark == "jssp":
        from sero.benchmarks.realm_adapter import load_j1_tasks
        from sero.role_card import JSSP_SEED_ROLES
        tasks = load_j1_tasks(BENCHMARK_DIR, max_tasks=total_tasks, seed=seed)
        seed_pool = list(JSSP_SEED_ROLES)
        sa_system = (
            "You are an expert job-shop scheduler. Produce an optimal schedule "
            "minimizing makespan. Respect all precedence and machine constraints."
        )
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return tasks, seed_pool, sa_system


def _save_result(result, benchmark, system, suffix=""):
    """Save result JSON and print summary table."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = _result_output_path(benchmark, system, suffix)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Dual-metric header when available
    has_dual = "mean_exact_score" in result or any(
        isinstance(v, dict) and "mean_exact_score" in v for v in result.values()
    )
    if has_dual:
        print(f"\n{'System':<25} {'Partial':>8} {'Exact':>8} {'N':>5}")
        print("-" * 55)
    else:
        print(f"\n{'System':<25} {'Mean':>8} {'Std':>8} {'N':>5}")
        print("-" * 50)

    def _print_entry(sub):
        if "mean_exact_score" in sub:
            print(f"{sub['system']:<25} {sub['mean_score']:>8.3f} "
                  f"{sub['mean_exact_score']:>8.3f} {sub['n_tasks']:>5}")
        elif "mean_score" in sub:
            print(f"{sub['system']:<25} {sub['mean_score']:>8.3f} "
                  f"{sub['std_score']:>8.3f} {sub['n_tasks']:>5}")

    if "mean_score" in result:
        _print_entry(result)
    else:
        for key, sub in result.items():
            if isinstance(sub, dict) and "mean_score" in sub:
                _print_entry(sub)
    print(f"\nResults saved to {out}")
    return out


def _add_dual_metrics(result, records):
    """Add exact and per-sub-benchmark metrics when records expose them."""
    import numpy as np
    sample = records[0].get("raw_score") if records else None
    if sample is not None and is_dual_score(sample):
        for key in ("exact_score",):
            vals = [r["raw_score"].get(key, 0.0) if isinstance(r.get("raw_score"), dict) else r["score"]
                    for r in records]
            result[f"mean_{key}"] = float(np.mean(vals))
            result[f"std_{key}"] = float(np.std(vals))

    grouped = {}
    for record in records:
        sub_benchmark = record.get("sub_benchmark")
        if sub_benchmark:
            grouped.setdefault(sub_benchmark, []).append(record)
    if grouped:
        result["per_sub_benchmark"] = {}
        for sub_benchmark, sub_records in sorted(grouped.items()):
            scores = [float(record.get("score", 0.0)) for record in sub_records]
            sub_result = {
                "n_tasks": len(sub_records),
                "mean_score": float(np.mean(scores)) if scores else 0.0,
                "std_score": float(np.std(scores)) if scores else 0.0,
            }
            sub_sample = sub_records[0].get("raw_score") if sub_records else None
            if sub_sample is not None and is_dual_score(sub_sample):
                exact_scores = [
                    record["raw_score"].get("exact_score", 0.0)
                    if isinstance(record.get("raw_score"), dict)
                    else record.get("score", 0.0)
                    for record in sub_records
                ]
                sub_result["mean_exact_score"] = float(np.mean(exact_scores))
                sub_result["std_exact_score"] = float(np.std(exact_scores))
            result["per_sub_benchmark"][sub_benchmark] = sub_result


# ── Answer Extraction ──────────────────────────────────────────────────────────

# Adapter-level canonical extractors — each returns a deterministic string
# from a response, suitable for SC majority voting.
_CANONICAL_EXTRACTORS = {}


def _init_extractors():
    """Lazy-load canonical extractors from adapter modules."""
    if _CANONICAL_EXTRACTORS:
        return
    from sero.benchmarks.trip_adapter import extract_canonical_answer as trip_extract
    from sero.benchmarks.natural_plan_adapter import extract_canonical_answer as naturalplan_extract
    from sero.benchmarks.meeting_plan_adapter import extract_canonical_answer as meeting_extract
    from sero.benchmarks.calendar_scheduling_adapter import extract_canonical_answer as cal_extract
    from sero.benchmarks.olympiad_adapter import extract_canonical_answer as oly_extract
    from sero.benchmarks.tablebench_adapter import extract_canonical_answer as tablebench_extract

    _CANONICAL_EXTRACTORS.update({
        "naturalplan": naturalplan_extract,
        "trip": trip_extract,
        "meeting": meeting_extract,
        "calendar": cal_extract,
        "olympiadbench": oly_extract,
        "tablebench": tablebench_extract,
    })


def _extract_answer(response, benchmark, task=None):
    """Extract the key structured answer for majority voting.

    Dispatches to benchmark-specific canonical extractors that understand
    the response format (flight sequences, meeting steps, time slots,
    boxed answers). Returns empty string for unknown benchmarks.
    """
    _init_extractors()
    extractor = _CANONICAL_EXTRACTORS.get(benchmark)
    if extractor is not None:
        if benchmark == "naturalplan":
            sub_benchmark = task.get("sub_benchmark") if isinstance(task, dict) else None
            return extractor(response, sub_benchmark=sub_benchmark)
        return extractor(response)
    return ""


def _extract_role_answers(responses, benchmark, exclude=None, task=None):
    """Return raw and parsed answers for a role-response mapping."""
    excluded = set(exclude or [])
    raw_answers = {}
    parsed_answers = {}
    for name, text in responses.items():
        if name in excluded:
            continue
        raw_answers[name] = text
        parsed_answers[name] = _extract_answer(text, benchmark, task) if benchmark else ""
    return raw_answers, parsed_answers


def _get_golden_answer_fields(task, benchmark):
    """Return raw and canonical gold answers for logging/debugging.

    Olympiad gold answers are stored as lists, but the canonical extractor expects
    answer-like text rather than a JSON list string such as ["12"].
    """
    golden_value = task.get("golden_plan")
    if golden_value is None:
        golden_value = task.get("gold_answer") or ""

    if isinstance(golden_value, list):
        golden_raw = json.dumps(golden_value, ensure_ascii=False)
        if benchmark == "olympiadbench":
            primary_gold = str(golden_value[0]) if golden_value else ""
            golden_source = "\\boxed{" + primary_gold + "}"
        else:
            golden_source = golden_raw
    else:
        golden_raw = golden_value
        golden_source = golden_value

    golden_extracted = _extract_answer(golden_source, benchmark, task) if benchmark else ""
    return golden_raw, golden_extracted


# ── System Evaluators ──────────────────────────────────────────────────────────


def eval_cot(eval_tasks, client, config, sa_system, benchmark):
    """Chain-of-Thought baseline — deterministic single response (temperature=0).

    Handles both single-float and dual-metric (dict) eval_fn returns.
    """
    import numpy as np

    def _eval_task(task):
        golden, golden_extracted = _get_golden_answer_fields(task, benchmark)
        try:
            resp = client.system_user(
                model=config.agent_model, system=sa_system,
                user=task["prompt"], temperature=0, max_tokens=4096,
            )
            raw_score = task["eval_fn"](resp)
        except Exception as e:
            logger.error("CoT error on %s: %s", task["id"], e)
            resp = ""
            raw_score = 0.0

        score = normalize_score(raw_score)
        extracted = _extract_answer(resp, benchmark, task)
        logger.info("[CoT] task=%s score=%.3f", task["id"], score)
        return {
            "task_id": task["id"],
            "sub_benchmark": task.get("sub_benchmark"),
            "score": score,
            "raw_score": raw_score,
            "response": resp,
            "extracted_answer": extracted,
            "golden_answer": golden,
            "golden_answer_extracted": golden_extracted,
        }

    records = [None] * len(eval_tasks)
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        fmap = {ex.submit(_eval_task, t): i for i, t in enumerate(eval_tasks)}
        for fut in as_completed(fmap):
            records[fmap[fut]] = fut.result()

    scores = [r["score"] for r in records]
    result = {
        "system": "cot", "n_tasks": len(scores),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "records": records,
    }
    _add_dual_metrics(result, records)
    return result


def eval_sc(eval_tasks, client, config, sa_system, benchmark, k=3):
    """Self-Consistency Majority Vote — k diverse samples (temperature=0.7).

    Generate k responses per task at temperature=0.7, then majority-vote
    on canonicalized structured answers.

    Handles both single-float and dual-metric (dict) eval_fn returns.
    """
    import numpy as np

    def _eval_task(task):
        golden, golden_extracted = _get_golden_answer_fields(task, benchmark)
        responses, raw_scores = [], []
        try:
            for _ in range(k):
                resp = client.system_user(
                    model=config.agent_model, system=sa_system,
                    user=task["prompt"], temperature=0.7, max_tokens=4096,
                )
                responses.append(resp)
                raw_scores.append(task["eval_fn"](resp))
        except Exception as e:
            logger.error("SC error on %s: %s", task["id"], e)
            while len(responses) < k:
                responses.append("")
                raw_scores.append(0.0)

        # Majority vote on extracted answers
        answers = [_extract_answer(r, benchmark, task) for r in responses]
        vote_counts = Counter(a for a in answers if a)
        if vote_counts:
            best_answer = vote_counts.most_common(1)[0][0]
            # Score from the first response matching the majority answer
            sc_raw = raw_scores[0]  # fallback
            for r, rs in zip(responses, raw_scores):
                if _extract_answer(r, benchmark, task) == best_answer:
                    sc_raw = rs
                    break
            sc_score = normalize_score(sc_raw)
        else:
            # No valid extraction — fall back to first response
            sc_score = normalize_score(raw_scores[0])
            sc_raw = raw_scores[0]
            best_answer = ""

        logger.info("[SC] task=%s score=%.3f voted='%s'", task["id"], sc_score,
                    best_answer[:60] if best_answer else "(empty)")
        return {
            "task_id": task["id"],
            "sub_benchmark": task.get("sub_benchmark"),
            "score": sc_score,
            "raw_score": sc_raw,
            "responses": responses,
            "extracted_answers": answers,
            "voted_answer": best_answer,
            "golden_answer": golden,
            "golden_answer_extracted": golden_extracted,
        }

    records = [None] * len(eval_tasks)
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        fmap = {ex.submit(_eval_task, t): i for i, t in enumerate(eval_tasks)}
        for fut in as_completed(fmap):
            records[fmap[fut]] = fut.result()

    scores = [r["score"] for r in records]
    result = {
        "system": f"sc_majority_vote_k{k}", "n_tasks": len(scores),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "k": k,
        "records": records,
    }
    _add_dual_metrics(result, records)
    return result

def eval_static(eval_tasks, benchmark, client, config):
    """Static baseline — benchmark-specific fixed-topology DAG, no credit, no evolution."""
    from sero.baselines.static_dag_free import run_static_dag_free
    return run_static_dag_free(eval_tasks, benchmark, client, config)


def eval_static_dag(eval_tasks, seed_pool, client, encoder, config,
                    benchmark=None):
    """Static-DAG ablation — credit mechanism + DAG, no role pool evolution."""
    import numpy as np
    from sero.credit_engine import CreditEngine
    from sero.phase_a import PhaseA
    from sero.config import SeroConfig

    static_cfg = SeroConfig(
        openrouter_api_key=config.openrouter_api_key,
        agent_model=config.agent_model,
        t_round=config.t_round, n_max=config.n_max,
        evolve_roles=False, seed=config.seed,
    )
    ce = CreditEngine()
    ce.sync_with_pool([r.name for r in seed_pool])

    def _eval_task(task):
        golden_raw, golden_parsed = _get_golden_answer_fields(task, benchmark)
        try:
            pa = PhaseA(config=static_cfg, client=client, encoder=encoder,
                       credit_engine=ce, pool=list(seed_pool), eval_fn=task["eval_fn"])
            r = pa.run(task["prompt"], update_fast_credits=False, bootstrap_credit_dag=True)
            score = r["score"]
            raw_score = r.get("raw_score", score)
            active = r.get("active_roles", [])
            answer = r.get("answer", "")
            all_responses = r.get("all_responses", {})
            answer_parsed = _extract_answer(answer, benchmark, task) if benchmark else ""
            topology = r.get("topology", {})
            aggregator_name = topology.get("aggregator", "")
            intermediate_answers, intermediate_answers_parsed = _extract_role_answers(
                all_responses,
                benchmark,
                exclude={aggregator_name} if aggregator_name else None,
                task=task,
            )
            _, role_answers_parsed = _extract_role_answers(all_responses, benchmark, task=task)
            dag_fast_credits = r.get("dag_fast_credits", {})
            draft_answer = r.get("draft_answer", "")
            validator_name = r.get("validator_name", "")
            validator_feedback = r.get("validator_feedback", "")
            revised_answer = r.get("revised_answer", "")
            bootstrap_traces = r.get("bootstrap_responses", {})
            round_traces = r.get("round_traces", [])
            aggregator_fallback_used = bool(r.get("aggregator_fallback_used", False))
            aggregator_fallback_answer = r.get("aggregator_fallback_answer", "")
            repair_triggered = bool(r.get("repair_triggered", False))
            validator_adopted = bool(r.get("validator_adopted", False))
        except Exception as e:
            logger.error("Static-DAG error on %s: %s", task["id"], e)
            score, raw_score, active = 0.0, 0.0, []
            answer, all_responses, answer_parsed = "", {}, ""
            topology = {}
            intermediate_answers, intermediate_answers_parsed = {}, {}
            role_answers_parsed = {}
            dag_fast_credits = {}
            draft_answer = ""
            validator_name = ""
            validator_feedback = ""
            revised_answer = ""
            bootstrap_traces = {}
            round_traces = []
            aggregator_fallback_used = False
            aggregator_fallback_answer = ""
            repair_triggered = False
            validator_adopted = False
        logger.info("[Static-DAG] task=%s score=%.3f", task["id"], score)
        return {
            "task_id": task["id"],
            "sub_benchmark": task.get("sub_benchmark"),
            "prompt": task["prompt"],
            "score": score,
            "raw_score": raw_score,
            "active_roles": active,
            "topology": topology,
            "agent_traces": all_responses,
            "bootstrap_traces": bootstrap_traces,
            "round_traces": round_traces,
            "intermediate_answers": intermediate_answers,
            "intermediate_answers_parsed": intermediate_answers_parsed,
            "role_answers_parsed": role_answers_parsed,
            "dag_fast_credits": dag_fast_credits,
            "draft_answer": draft_answer,
            "validator_name": validator_name,
            "validator_feedback": validator_feedback,
            "revised_answer": revised_answer,
            "aggregator_fallback_used": aggregator_fallback_used,
            "aggregator_fallback_answer": aggregator_fallback_answer,
            "repair_triggered": repair_triggered,
            "validator_adopted": validator_adopted,
            "final_answer": answer,
            "final_answer_parsed": answer_parsed,
            "golden_answer_raw": golden_raw,
            "golden_answer_parsed": golden_parsed,
        }

    records = [None] * len(eval_tasks)
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        fmap = {ex.submit(_eval_task, t): i for i, t in enumerate(eval_tasks)}
        for fut in as_completed(fmap):
            records[fmap[fut]] = fut.result()

    scores = [r["score"] for r in records]
    result = {
        "system": "static_dag", "n_tasks": len(scores),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "pool": [r.to_dict() for r in seed_pool],
        "records": records,
    }
    _add_dual_metrics(result, records)
    return result

# NaturalPlan benchmarks — the only ones where the hand-engineered Workflow
# baseline is defined (expert-designed upper-bound reference).
_WORKFLOW_BENCHMARKS = ("naturalplan", "trip", "meeting", "calendar", "olympiadbench", "tablebench")


def eval_workflow(eval_tasks, benchmark, client, encoder, config):
    """Workflow baseline — expert-designed sequential pipeline (NaturalPlan only)."""
    if benchmark not in _WORKFLOW_BENCHMARKS:
        raise ValueError(
            f"Workflow baseline is only defined for NaturalPlan benchmarks "
            f"({', '.join(_WORKFLOW_BENCHMARKS)}), got '{benchmark}'"
        )
    from sero.baselines.workflow_baseline import run_workflow_baseline
    return run_workflow_baseline(eval_tasks, benchmark, client, config)


def eval_random_evo(train_tasks, eval_tasks, seed_pool, client, encoder, config,
                    benchmark=None):
    """Random evolution: unconditional random actions on train_tasks, then frozen-eval.

    Produces comprehensive trace data for every step of training and evaluation.
    """
    import numpy as np
    from sero.credit_engine import CreditEngine
    from sero.executor import Executor
    from sero.phase_a import PhaseA
    from sero.controller import SeroController

    rng = random.Random(config.seed)
    ops = SeroController.OPS
    pool = list(seed_pool)
    executor = Executor(client, config.executor_model)
    credit_engine = CreditEngine(mu=config.ema_mu, alpha=config.fast_credit_alpha)
    credit_engine.sync_with_pool([r.name for r in pool])

    # Helper: serialize pool snapshot
    def _pool_snapshot(p):
        return [r.to_dict() for r in p]

    # Helper: get golden answer (raw + parsed)
    # ── Phase 1: unconditional random evolution on training tasks ─────────────
    initial_pool = _pool_snapshot(pool)
    train_records = []
    for task in train_tasks:
        op = rng.choice(ops)
        target = rng.choice(pool).name if pool and op != "noop" else None
        pool_before = _pool_snapshot(pool)

        # Action-specific trace
        add_trace = None
        removed_role = None

        if op == "remove" and target is not None:
            # L1 fix: Protect protected roles from removal (same as SERO trainer logic)
            target_role = next((r for r in pool if r.name == target), None)
            if config.protect_critical_roles and target_role is not None and target_role.protected:
                logger.info("[RandomEvo-train] Role '%s' is protected; skipping REMOVE.", target)
                op = "noop (protected skip)"  # reclassify for logging
                removed_role = None
            else:
                removed_role = next((r.to_dict() for r in pool if r.name == target), None)
                pool = executor.remove(target, list(pool))

        elif op == "add_anchor" and len(pool) < config.p_max:
            trace_result = executor.add_anchor(
                list(pool), credit_engine, task["prompt"][:200],
                return_trace=True,
            )
            add_trace = {
                "anchor_name": trace_result["anchor_name"],
                "executor_prompt": trace_result["executor_prompt"],
                "llm_raw_output": trace_result["llm_raw_output"],
                "fallback": trace_result["fallback"],
                "new_role_card": trace_result["card"].to_dict() if trace_result["card"] else None,
            }
            new_card = trace_result["card"]
            if new_card:
                pool = list(pool) + [new_card]
        # noop or fallback: pool unchanged

        credit_engine.sync_with_pool([r.name for r in pool])

        # Run PhaseA on this training task with current pool (for visibility only)
        train_inference = None
        try:
            pa = PhaseA(config=config, client=client, encoder=encoder,
                       credit_engine=credit_engine, pool=pool, eval_fn=task["eval_fn"])
            r = pa.run(
                task["prompt"],
                update_fast_credits=False,
                bootstrap_credit_dag=True,
            )
            golden_raw, golden_parsed = _get_golden_answer_fields(task, benchmark)
            answer_parsed = _extract_answer(r["answer"], benchmark, task) if benchmark else ""
            topology = r.get("topology", {})
            aggregator_name = topology.get("aggregator", "")
            intermediate_answers, intermediate_answers_parsed = _extract_role_answers(
                r.get("all_responses", {}),
                benchmark,
                exclude={aggregator_name} if aggregator_name else None,
                task=task,
            )
            _, role_answers_parsed = _extract_role_answers(
                r.get("all_responses", {}),
                benchmark,
                task=task,
            )
            train_inference = {
                "answer": r["answer"],
                "answer_parsed": answer_parsed,
                "score": r["score"],
                "raw_score": r.get("raw_score", r["score"]),
                "active_roles": r.get("active_roles", []),
                "topology": topology,
                "agent_traces": r.get("all_responses", {}),
                "bootstrap_traces": r.get("bootstrap_responses", {}),
                "round_traces": r.get("round_traces", []),
                "intermediate_answers": intermediate_answers,
                "intermediate_answers_parsed": intermediate_answers_parsed,
                "role_answers_parsed": role_answers_parsed,
                "dag_fast_credits": r.get("dag_fast_credits", {}),
                "draft_answer": r.get("draft_answer", ""),
                "validator_name": r.get("validator_name", ""),
                "validator_feedback": r.get("validator_feedback", ""),
                "revised_answer": r.get("revised_answer", ""),
                "aggregator_fallback_used": bool(r.get("aggregator_fallback_used", False)),
                "aggregator_fallback_answer": r.get("aggregator_fallback_answer", ""),
                "repair_triggered": bool(r.get("repair_triggered", False)),
                "validator_adopted": bool(r.get("validator_adopted", False)),
                "golden_answer_raw": golden_raw,
                "golden_answer_parsed": golden_parsed,
            }
        except Exception as e:
            logger.warning("[RandomEvo-train] inference failed for %s: %s", task["id"], e)

        train_records.append({
            "task_id": task["id"],
            "sub_benchmark": task.get("sub_benchmark"),
            "prompt": task["prompt"],
            "op": op,
            "target": target,
            "removed_role": removed_role,
            "add_trace": add_trace,
            "pool_before": [r["name"] for r in pool_before],
            "pool_after": [r.name for r in pool],
            "pool_size": len(pool),
            "inference": train_inference,
        })
        logger.info("[RandomEvo-train] task=%s op=%s pool_size=%d",
                    task["id"], op, len(pool))

    # ── Phase 2: frozen eval on eval tasks with final pool ────────────────────
    final_pool_detail = _pool_snapshot(pool)
    logger.info("RandomEvo: evolved pool has %d roles: %s",
                len(pool), [r.name for r in pool])

    def _eval_task(task):
        golden_raw, golden_parsed = _get_golden_answer_fields(task, benchmark)
        topology = {}
        try:
            pa = PhaseA(config=config, client=client, encoder=encoder,
                       credit_engine=credit_engine, pool=pool, eval_fn=task["eval_fn"])
            r = pa.run(
                task["prompt"],
                update_fast_credits=False,
                bootstrap_credit_dag=True,
            )
            score = r["score"]
            raw_score = r.get("raw_score", score)
            active = r.get("active_roles", [])
            answer = r.get("answer", "")
            all_responses = r.get("all_responses", {})
            answer_parsed = _extract_answer(answer, benchmark, task) if benchmark else ""
            topology = r.get("topology", {})
            aggregator_name = topology.get("aggregator", "")
            intermediate_answers, intermediate_answers_parsed = _extract_role_answers(
                all_responses,
                benchmark,
                exclude={aggregator_name} if aggregator_name else None,
                task=task,
            )
            _, role_answers_parsed = _extract_role_answers(all_responses, benchmark, task=task)
            dag_fast_credits = r.get("dag_fast_credits", {})
            draft_answer = r.get("draft_answer", "")
            validator_name = r.get("validator_name", "")
            validator_feedback = r.get("validator_feedback", "")
            revised_answer = r.get("revised_answer", "")
            bootstrap_traces = r.get("bootstrap_responses", {})
            round_traces = r.get("round_traces", [])
            aggregator_fallback_used = bool(r.get("aggregator_fallback_used", False))
            aggregator_fallback_answer = r.get("aggregator_fallback_answer", "")
            repair_triggered = bool(r.get("repair_triggered", False))
            validator_adopted = bool(r.get("validator_adopted", False))
        except Exception as e:
            logger.error("RandomEvo eval error on %s: %s", task["id"], e)
            score, raw_score, active = 0.0, 0.0, []
            answer, all_responses, answer_parsed = "", {}, ""
            topology = {}
            intermediate_answers, intermediate_answers_parsed = {}, {}
            role_answers_parsed = {}
            dag_fast_credits = {}
            draft_answer = ""
            validator_name = ""
            validator_feedback = ""
            revised_answer = ""
            bootstrap_traces = {}
            round_traces = []
            aggregator_fallback_used = False
            aggregator_fallback_answer = ""
            repair_triggered = False
            validator_adopted = False

        logger.info("[RandomEvo-eval] task=%s score=%.3f", task["id"], score)
        return {
            "task_id": task["id"],
            "sub_benchmark": task.get("sub_benchmark"),
            "prompt": task["prompt"],
            "score": score,
            "raw_score": raw_score,
            "active_roles": active,
            "topology": topology,
            "agent_traces": all_responses,
            "bootstrap_traces": bootstrap_traces,
            "round_traces": round_traces,
            "intermediate_answers": intermediate_answers,
            "intermediate_answers_parsed": intermediate_answers_parsed,
            "role_answers_parsed": role_answers_parsed,
            "dag_fast_credits": dag_fast_credits,
            "draft_answer": draft_answer,
            "validator_name": validator_name,
            "validator_feedback": validator_feedback,
            "revised_answer": revised_answer,
            "aggregator_fallback_used": aggregator_fallback_used,
            "aggregator_fallback_answer": aggregator_fallback_answer,
            "repair_triggered": repair_triggered,
            "validator_adopted": validator_adopted,
            "final_answer": answer,
            "final_answer_parsed": answer_parsed,
            "golden_answer_raw": golden_raw,
            "golden_answer_parsed": golden_parsed,
        }

    eval_records = [None] * len(eval_tasks)
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        fmap = {ex.submit(_eval_task, t): i for i, t in enumerate(eval_tasks)}
        for fut in as_completed(fmap):
            eval_records[fmap[fut]] = fut.result()

    eval_scores = [r["score"] for r in eval_records]
    result = {
        "system": "random_evo", "n_tasks": len(eval_scores),
        "mean_score": float(np.mean(eval_scores)),
        "std_score": float(np.std(eval_scores)),
        "seed_pool": [r.to_dict() for r in seed_pool],
        "initial_pool": initial_pool,
        "final_pool": [r.name for r in pool],
        "final_pool_detail": final_pool_detail,
        "train_records": train_records,
        "records": eval_records,
    }
    _add_dual_metrics(result, eval_records)
    return result


_SERO_NATURALPLAN_ROLE_SUPPLEMENTS = {
    "NaturalPlan Task & Contract Parser": (
        "SERO guardrail: identify the task type by evidence, not by isolated keywords. Calendar scheduling often says "
        "'meeting' but has busy intervals and work hours, with no locations or travel matrix. Meeting planning "
        "has locations, travel times, availability windows, and route steps. Trip planning has cities, day counts, "
        "events, and direct flights. If the prompt has an appended IMPORTANT marker such as CALENDAR SCHEDULING, "
        "TRIP PLANNING, or MEETING PLANNING, that marker is authoritative. Never let one subtask borrow another "
        "subtask's final format."
    ),
    "Trip Constraint Extractor": (
        "SERO guardrail: stay inside trip planning. Do not reason about attendees, meeting routes, busy calendars, "
        "or proposed-time lines; extract only cities, stays, day windows, total duration, and direct flights. "
        "An appended IMPORTANT TRIP PLANNING marker makes this role applicable; other NaturalPlan markers do not. "
        "Use shared-flight arithmetic exactly: sum(stays) - number_of_flights = total days; adjacent city ranges "
        "sharing a flight day are valid, not conflicts."
    ),
    "Trip Flight Route Planner": (
        "SERO guardrail: trip feasibility is a city-flight graph problem. Do not introduce meeting travel steps, "
        "wait steps, people, or calendar slots. An appended IMPORTANT TRIP PLANNING marker makes this role applicable; "
        "other NaturalPlan markers do not. Choose the city order that satisfies directed flights and event windows; "
        "do not assume prompt order is route order."
    ),
    "Trip Day Logistics Formatter": (
        "SERO guardrail: output Day visit and Day flight itinerary drafts only. Calendar proposed-time lines and "
        "meeting travel/wait/meet lines are always wrong for trip tasks. An appended IMPORTANT TRIP PLANNING marker "
        "makes this role applicable; other NaturalPlan markers do not. If no route candidate is supplied, choose the "
        "city order from the direct-flight graph and event windows before assigning days. Flight days are shared: "
        "the next city starts on the flight day."
    ),
    "Calendar Parser and Preference Extractor": (
        "SERO guardrail: calendar tasks may contain the word meeting, but they are not route-planning tasks. Never "
        "create travel, wait, or 'You meet ...' steps; keep only busy blocks, free windows, duration, and hard preferences. "
        "An appended IMPORTANT CALENDAR SCHEDULING marker makes this role applicable even when the wording says meeting."
    ),
    "Calendar Slot and Conflict Checker": (
        "SERO guardrail: validate one shared free interval in 24-hour time. Treat preferences as hard filters. Use "
        "half-open intervals [start, end): intersection_start=max(free starts), intersection_end=min(free ends), and "
        "the overlap must fit the duration. A slot may end exactly when a busy block starts or start exactly when a busy "
        "block ends. Reject any candidate that turns the calendar slot into travel or route steps."
    ),
    "Meeting Window and Distance Analyzer": (
        "SERO guardrail: use this role only when the prompt has locations, travel times, availability windows, and people. "
        "An appended IMPORTANT MEETING PLANNING marker makes this role applicable; other NaturalPlan markers do not. "
        "Do not convert it into generic calendar proposed-time scheduling."
    ),
    "Meeting Route Scheduler": (
        "SERO guardrail: meeting planning output is a chronological route with exact travel, optional wait, and meet steps. "
        "An appended IMPORTANT MEETING PLANNING marker makes this role applicable; other NaturalPlan markers do not. "
        "Never output the calendar sentence 'Here is the proposed time'."
    ),
    "Cross-Task Constraint Validator": (
        "SERO guardrail: wrong subtask format is an invalid answer even when some constraints look plausible. Audit both "
        "constraint satisfaction and whether the candidate uses the selected subtask's parser-facing final format. For "
        "calendar tasks, use half-open interval overlap and choose the earliest valid shared slot unless an explicit "
        "preference says otherwise. For trip tasks, adjacent city ranges sharing a flight day are valid and should not "
        "be rejected as double-counting."
    ),
    "NaturalPlan Aggregator": (
        "SERO guardrail: before emitting the final answer, discard candidates from the wrong subtask format. If the prompt "
        "has an appended IMPORTANT marker such as CALENDAR SCHEDULING, TRIP PLANNING, or MEETING PLANNING, use that "
        "marker as the authoritative final subtask. The final answer must match exactly one NaturalPlan parser contract "
        "and contain no cross-task residue. For calendar tasks, choose the earliest valid shared slot unless an explicit "
        "preference says otherwise. For trip tasks, keep shared flight-day overlaps when they satisfy the route."
    ),
}


def _prepare_sero_seed_pool(seed_pool, benchmark):
    """Apply SERO-only NaturalPlan seed prompt supplements without changing baselines."""
    if benchmark != "naturalplan":
        return list(seed_pool)
    prepared = []
    for role in seed_pool:
        supplement = _SERO_NATURALPLAN_ROLE_SUPPLEMENTS.get(role.name)
        if not supplement or supplement in role.system_prompt:
            prepared.append(role)
            continue
        tags = list(role.capability_tags)
        if "sero-naturalplan-guardrail" not in tags:
            tags.append("sero-naturalplan-guardrail")
        prepared.append(role.copy_with(
            system_prompt=f"{role.system_prompt}\n\n{supplement}",
            capability_tags=tags,
        ))
    return prepared


def eval_sero(train_tasks, eval_tasks, seed_pool, client, encoder, config,
              checkpoint=None, benchmark=None, results_dir=None):
    """SERO: train + eval, or checkpoint-only eval.

    Produces comprehensive trace data including seed pool, training episode
    records, final pool detail, and per-task agent traces / golden answers.
    """
    import numpy as np
    from sero.credit_engine import CreditEngine
    from sero.phase_a import PhaseA

    # Helper: serialize pool snapshot
    def _pool_snapshot(p):
        return [r.to_dict() for r in p]

    # Helper: get golden answer (raw + parsed)
    seed_pool = _prepare_sero_seed_pool(seed_pool, benchmark)
    initial_pool = _pool_snapshot(seed_pool)
    train_episode_records = []

    if checkpoint is not None:
        # Eval-only: load checkpoint
        import torch
        from sero.controller import SeroController
        from sero.role_card import RoleCard

        ckpt = torch.load(checkpoint, map_location="cpu")
        controller = SeroController(config)
        controller.load_state_dict(ckpt["model_state"])
        controller.eval()
        logger.info("Loaded SERO checkpoint from %s (epoch %s)", checkpoint,
                    ckpt.get("epoch", "?"))

        # Restore pool: .pt "pool" key is authoritative (matches this epoch's weights)
        #             > final_pool.json (fallback — may be from a later epoch)
        #             > seed pool (last resort)
        if "pool" in ckpt and ckpt["pool"]:
            pool = [RoleCard.from_dict(d) for d in ckpt["pool"]]
            logger.info("Restored pool from checkpoint .pt (%d roles).", len(pool))
        else:
            ckpt_dir_pool = os.path.dirname(checkpoint)
            pool_path = os.path.join(ckpt_dir_pool, "final_pool.json")
            if os.path.exists(pool_path):
                with open(pool_path) as f:
                    pool_data = json.load(f)
                pool = [RoleCard.from_dict(d) for d in pool_data]
                logger.info("Restored pool from %s (%d roles).", pool_path, len(pool))
            else:
                logger.warning("No pool in checkpoint or final_pool.json; using seed pool")
                pool = list(seed_pool)

        # Restore credit engine state if available; otherwise cold-start
        credit_engine = CreditEngine(mu=config.ema_mu, alpha=config.fast_credit_alpha)
        if "credit_engine" in ckpt:
            credit_engine.load_state_dict(ckpt["credit_engine"])
            logger.info("Restored credit engine state (%d roles tracked).",
                        len(ckpt["credit_engine"].get("credits", {})))
        else:
            credit_engine.sync_with_pool([r.name for r in pool])
            logger.warning("No credit engine state in checkpoint; cold-started.")

        expected_episode_records = None
        if isinstance(ckpt.get("epoch"), int) and train_tasks:
            expected_episode_records = (ckpt["epoch"] + 1) * len(train_tasks)

        train_episode_records, loaded_log_path, truncated_log = _load_checkpoint_episode_records(
            checkpoint,
            epoch=ckpt.get("epoch"),
            tag=ckpt.get("tag", "checkpoint"),
            expected_count=expected_episode_records,
        )
        if loaded_log_path is not None:
            logger.info(
                "Loaded %d training episode records from %s.",
                len(train_episode_records),
                loaded_log_path,
            )
            if truncated_log:
                logger.warning(
                    "Fallback episode log had later records; truncated to %d entries for checkpoint epoch %s.",
                    len(train_episode_records),
                    ckpt.get("epoch", "?"),
                )
    else:
        # Train + eval
        from sero.trainer import SeroTrainer

        trainer = SeroTrainer(
            config=config, seed_pool=list(seed_pool), tasks=train_tasks,
            client=client, encoder=encoder,
            results_dir=results_dir or os.path.join(RESULTS_DIR, f"sero_ckpt_{config.seed}"),
        )
        trainer.train()
        pool = trainer.pool
        credit_engine = trainer.credit_engine
        train_episode_records = [asdict(r) for r in trainer.stats.episode_records]
        logger.info("SERO trained. Pool: %s", [r.name for r in pool])

    final_pool_detail = _pool_snapshot(pool)

    # Frozen evaluation on eval tasks
    def _eval_task(task):
        golden_raw, golden_parsed = _get_golden_answer_fields(task, benchmark)
        topology = {}
        try:
            pa = PhaseA(config=config, client=client, encoder=encoder,
                       credit_engine=credit_engine, pool=pool, eval_fn=task["eval_fn"])
            r = pa.run(task["prompt"], update_fast_credits=False)
            score = r["score"]
            raw_score = r.get("raw_score", score)
            active = r.get("active_roles", [])
            answer = r.get("answer", "")
            all_responses = r.get("all_responses", {})
            answer_parsed = _extract_answer(answer, benchmark, task) if benchmark else ""
            topology = r.get("topology", {})
            aggregator_name = topology.get("aggregator", "")
            intermediate_answers, intermediate_answers_parsed = _extract_role_answers(
                all_responses,
                benchmark,
                exclude={aggregator_name} if aggregator_name else None,
                task=task,
            )
            _, role_answers_parsed = _extract_role_answers(all_responses, benchmark, task=task)
            dag_fast_credits = r.get("dag_fast_credits", {})
            draft_answer = r.get("draft_answer", "")
            validator_name = r.get("validator_name", "")
            validator_feedback = r.get("validator_feedback", "")
            revised_answer = r.get("revised_answer", "")
            bootstrap_traces = r.get("bootstrap_responses", {})
            round_traces = r.get("round_traces", [])
            aggregator_fallback_used = bool(r.get("aggregator_fallback_used", False))
            aggregator_fallback_answer = r.get("aggregator_fallback_answer", "")
            repair_triggered = bool(r.get("repair_triggered", False))
            validator_adopted = bool(r.get("validator_adopted", False))
        except Exception as e:
            logger.error("SERO error on %s: %s", task["id"], e)
            score, raw_score, active = 0.0, 0.0, []
            answer, all_responses, answer_parsed = "", {}, ""
            topology = {}
            intermediate_answers, intermediate_answers_parsed = {}, {}
            role_answers_parsed = {}
            dag_fast_credits = {}
            draft_answer = ""
            validator_name = ""
            validator_feedback = ""
            revised_answer = ""
            bootstrap_traces = {}
            round_traces = []
            aggregator_fallback_used = False
            aggregator_fallback_answer = ""
            repair_triggered = False
            validator_adopted = False

        logger.info("[SERO] task=%s score=%.3f", task["id"], score)
        return {
            "task_id": task["id"],
            "sub_benchmark": task.get("sub_benchmark"),
            "prompt": task["prompt"],
            "score": score,
            "raw_score": raw_score,
            "active_roles": active,
            "topology": topology,
            "agent_traces": all_responses,
            "bootstrap_traces": bootstrap_traces,
            "round_traces": round_traces,
            "intermediate_answers": intermediate_answers,
            "intermediate_answers_parsed": intermediate_answers_parsed,
            "role_answers_parsed": role_answers_parsed,
            "dag_fast_credits": dag_fast_credits,
            "draft_answer": draft_answer,
            "validator_name": validator_name,
            "validator_feedback": validator_feedback,
            "revised_answer": revised_answer,
            "aggregator_fallback_used": aggregator_fallback_used,
            "aggregator_fallback_answer": aggregator_fallback_answer,
            "repair_triggered": repair_triggered,
            "validator_adopted": validator_adopted,
            "final_answer": answer,
            "final_answer_parsed": answer_parsed,
            "golden_answer_raw": golden_raw,
            "golden_answer_parsed": golden_parsed,
        }

    records = [None] * len(eval_tasks)
    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as ex:
        fmap = {ex.submit(_eval_task, t): i for i, t in enumerate(eval_tasks)}
        for fut in as_completed(fmap):
            records[fmap[fut]] = fut.result()

    scores = [r["score"] for r in records]
    result = {
        "system": "sero", "n_tasks": len(scores),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "seed_pool": initial_pool,
        "final_pool": [r.name for r in pool],
        "final_pool_detail": final_pool_detail,
        "train_episode_records": train_episode_records,
        "records": records,
    }
    result["monitoring"] = _build_monitoring_summary(
        records,
        final_pool_detail,
        train_episode_records=train_episode_records,
        encoder=encoder,
    )
    _add_dual_metrics(result, records)
    return result


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    from sero.config import SeroConfig
    from sero.openrouter_client import OpenRouterClient

    if getattr(args, "preserve_train_order", False) and getattr(args, "shuffle_train_tasks", False):
        logger.error("--preserve_train_order conflicts with --shuffle_train_tasks")
        sys.exit(1)

    default_config = SeroConfig()
    if getattr(args, "preserve_train_order", False):
        shuffle_train_tasks = False
    elif getattr(args, "shuffle_train_tasks", False):
        shuffle_train_tasks = True
    elif args.benchmark in ("naturalplan", "tablebench"):
        shuffle_train_tasks = False
    else:
        shuffle_train_tasks = default_config.shuffle_train_tasks

    joint_ablate_credit = getattr(args, "joint_ablate_credit", False)
    joint_ablate_evolution = getattr(args, "joint_ablate_evolution", False)
    joint_ablate_protect = getattr(args, "joint_ablate_protect", False)
    joint_ablate_controller_reward = getattr(args, "joint_ablate_controller_reward", False)

    config = SeroConfig(
        **({"openrouter_api_key": os.environ["OPENROUTER_API_KEY"]}
           if os.environ.get("OPENROUTER_API_KEY") else {}),
        warmup_epochs=args.warmup_epochs,
        main_epochs=args.main_epochs,
        t_round=args.t_round,
          d_h=getattr(args, "controller_hidden_dim", default_config.d_h),
          controller_encoder_layers=getattr(args, "controller_encoder_layers", default_config.controller_encoder_layers),
          controller_target_layers=getattr(args, "controller_target_layers", default_config.controller_target_layers),
        n_max=args.n_max,
        p_max=args.p_max,
        p_min=args.p_min,
        seed=args.seed,
        shuffle_train_tasks=shuffle_train_tasks,
        format_inherit=not args.no_format_inherit,
        use_credit_state=not (getattr(args, "no_credit_state", False) or joint_ablate_credit),
        use_active_set_credit=not (getattr(args, "no_active_set_credit", False) or joint_ablate_credit),
        use_credit_dag=not (getattr(args, "no_credit_dag", False) or joint_ablate_credit),
        random_controller=getattr(args, "random_controller", False),
        protect_critical_roles=not (getattr(args, "no_protect", False) or joint_ablate_protect),
        freeze_role_pool=joint_ablate_evolution,
        post_aggregator_validator_check=not joint_ablate_protect,
        controller_reward_training=not joint_ablate_controller_reward,
        conditional_validator_pass=not getattr(args, "disable_conditional_validator_pass", False),
        preserve_seed_family_coverage=not getattr(args, "disable_seed_family_coverage", False),
        **({"batch_size": args.batch_size} if getattr(args, "batch_size", None) is not None else {}),
        **({"specialist_slots": args.specialist_slots} if getattr(args, "specialist_slots", None) is not None else {}),
        **({"validator_slots": args.validator_slots} if getattr(args, "validator_slots", None) is not None else {}),
        **({"loo_refresh_interval": args.loo_refresh} if getattr(args, "loo_refresh", None) is not None else {}),
        **({"entropy_beta": args.entropy_beta} if getattr(args, "entropy_beta", None) is not None else {}),
        **({"lr": args.lr} if getattr(args, "lr", None) is not None else {}),
        **({"ema_mu": args.ema_mu} if getattr(args, "ema_mu", None) is not None else {}),
        **({"fast_credit_alpha": args.fast_credit_alpha} if getattr(args, "fast_credit_alpha", None) is not None else {}),
        **({"exploration_gamma": args.exploration_gamma} if getattr(args, "exploration_gamma", None) is not None else {}),
        **({"loo_min_pool_size": args.loo_min_pool_size} if getattr(args, "loo_min_pool_size", None) is not None else {}),
        **({"new_role_initial_n_updates": args.new_role_initial_n_updates} if getattr(args, "new_role_initial_n_updates", None) is not None else {}),
        **({"noop_collapse_threshold": args.noop_collapse_threshold} if getattr(args, "noop_collapse_threshold", None) is not None else {}),
        **({"min_capability_family_diversity": args.min_capability_family_diversity} if getattr(args, "min_capability_family_diversity", None) is not None else {}),
        **({"max_capability_family_dominance": args.max_capability_family_dominance} if getattr(args, "max_capability_family_dominance", None) is not None else {}),
        **({"max_role_prompt_similarity": args.max_role_prompt_similarity} if getattr(args, "max_role_prompt_similarity", None) is not None else {}),
        **({"orthogonal_role_blacklist_top_k": args.orthogonal_role_blacklist_top_k} if getattr(args, "orthogonal_role_blacklist_top_k", None) is not None else {}),
    )
    if not config.openrouter_api_key:
        logger.error("OPENROUTER_API_KEY not set.")
        sys.exit(1)

    client = OpenRouterClient(config.openrouter_api_key, base_url=config.openrouter_base_url)
    result_path = _result_output_path(args.benchmark, args.system, args.suffix)
    sero_results_dir = None
    if args.system == "sero":
        sero_results_dir = (
            os.path.abspath(args.results_dir)
            if args.results_dir is not None
            else (
            os.path.dirname(args.checkpoint)
            if args.checkpoint is not None
            else os.path.join(RESULTS_DIR, f"sero_ckpt_{config.seed}")
            )
        )

    needs_train = args.system in ("random_evo", "sero") and args.checkpoint is None
    try:
        eval_set = _resolve_eval_set(args)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    run_manifest = _build_run_manifest(
        args,
        config,
        eval_set=eval_set,
        result_path=result_path,
        checkpoint_dir=sero_results_dir,
    )
    if args.system == "sero" and args.checkpoint is None and sero_results_dir is not None:
        run_config_path = _write_run_config(sero_results_dir, run_manifest)
        if run_config_path is not None:
            run_manifest["run_metadata"]["run_config_path"] = os.path.abspath(run_config_path)
            logger.info("Run config saved: %s", run_config_path)

    split_path = args.split_file or _default_split_path()
    split_key = SPLIT_KEY_MAP.get(args.benchmark)
    train_keys = None
    heldout_keys = None
    eval_subset_keys = None

    if eval_set != "legacy" and not _supports_split_benchmark(args.benchmark):
        logger.error(
            "Eval set '%s' is only supported for split-enabled benchmarks (%s).",
            eval_set,
            ", ".join(BENCHMARK_CHOICES),
        )
        sys.exit(1)

    split_required = eval_set in ("heldout", "subset") or (
        eval_set == "natural_full" and needs_train
    )
    meeting_full_eval_min_people = 1 if (
        args.benchmark == "meeting" and eval_set in ("heldout", "natural_full")
    ) else None

    if split_required:
        if not os.path.exists(split_path):
            logger.error("train_split.json not found at %s. "
                         "Run scripts/build_train_split.py first.", split_path)
            sys.exit(1)
        with open(split_path) as f:
            split_data = json.load(f)
        if split_key not in split_data:
            logger.error("No split definition for benchmark '%s' in %s.",
                         args.benchmark, split_path)
            sys.exit(1)

        train_keys = split_data[split_key]["train_keys"]
        heldout_keys = split_data[split_key].get("heldout_keys")
        logger.info("Using split: %d fixed train keys for '%s'",
                    len(train_keys), args.benchmark)
        if heldout_keys is not None:
            logger.info("Using split-defined heldout pool: %d keys for '%s'",
                        len(heldout_keys), args.benchmark)
        if eval_set == "subset":
            eval_subset_keys = split_data[split_key].get("eval_subset_keys")
            if not eval_subset_keys:
                logger.error("No eval_subset_keys in train_split.json for '%s'. "
                             "Re-run scripts/build_train_split.py to generate.", args.benchmark)
                sys.exit(1)
            logger.info("Using eval subset: %d fixed keys for '%s'",
                        len(eval_subset_keys), args.benchmark)

    if eval_set in ("heldout", "subset", "natural_full") and _supports_split_benchmark(args.benchmark):
        logger.info("Loading %s benchmark (eval_set=%s)...", args.benchmark, eval_set)
        if needs_train:
            if train_keys is None:
                logger.error("Eval set '%s' requires split-defined train keys for trainable systems.",
                             eval_set)
                sys.exit(1)
            train_tasks, seed_pool, sa_system = _load_benchmark(
                args.benchmark,
                total_tasks=len(train_keys),
                seed=args.seed,
                include_keys=train_keys,
            )
        else:
            train_tasks = []

        if eval_set == "subset":
            eval_tasks_result = _load_benchmark(
                args.benchmark,
                total_tasks=len(eval_subset_keys),
                seed=args.seed,
                include_keys=eval_subset_keys,
            )
        elif eval_set == "heldout":
            eval_limit = _resolve_task_limit(args.eval_tasks)
            if heldout_keys is not None:
                heldout_key_list = heldout_keys if eval_limit == LOAD_ALL_TASKS else heldout_keys[:eval_limit]
                eval_tasks_result = _load_benchmark(
                    args.benchmark,
                    total_tasks=len(heldout_key_list),
                    seed=args.seed,
                    include_keys=heldout_key_list,
                    meeting_min_people=meeting_full_eval_min_people,
                )
            else:
                eval_tasks_result = _load_benchmark(
                    args.benchmark,
                    total_tasks=eval_limit,
                    seed=args.seed,
                    exclude_keys=train_keys,
                    meeting_min_people=meeting_full_eval_min_people,
                )
        else:
            eval_tasks_result = _load_benchmark(
                args.benchmark,
                total_tasks=_resolve_task_limit(args.eval_tasks),
                seed=args.seed,
                meeting_min_people=meeting_full_eval_min_people,
            )

        if needs_train:
            eval_tasks = eval_tasks_result[0]
        else:
            eval_tasks, seed_pool, sa_system = eval_tasks_result

        if eval_set == "natural_full" and needs_train:
            logger.warning(
                "Eval set 'natural_full' includes the fixed train keys for trainable systems; "
                "scores are not held-out."
            )
    else:
        eval_limit = _resolve_task_limit(args.eval_tasks)
        total_tasks = (args.tasks + eval_limit) if needs_train else eval_limit
        logger.info("Loading %s benchmark (%s mode, requested_eval_tasks=%s)...",
                    args.benchmark, eval_set, args.eval_tasks)
        all_tasks, seed_pool, sa_system = _load_benchmark(
            args.benchmark, total_tasks, args.seed)

        if needs_train:
            train_tasks = all_tasks[:args.tasks]
            eval_tasks = all_tasks[args.tasks:args.tasks + eval_limit]
        else:
            train_tasks = []
            eval_tasks = all_tasks[:eval_limit]

    logger.info("Train: %d  Eval: %d", len(train_tasks), len(eval_tasks))

    # Load encoder only for systems that need it
    encoder = None
    if args.system in ("static_dag", "random_evo", "sero"):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading encoder: %s", config.encoder_model)
        encoder = SentenceTransformer(config.encoder_model, trust_remote_code=True)

    # Dispatch to the selected system
    system = args.system
    if system == "cot":
        result = eval_cot(eval_tasks, client, config, sa_system, args.benchmark)
    elif system == "sc":
        result = eval_sc(eval_tasks, client, config, sa_system,
                         args.benchmark, k=args.sc_k)
    elif system == "static":
        result = eval_static(eval_tasks, args.benchmark, client, config)
    elif system == "static_dag":
        result = eval_static_dag(eval_tasks, seed_pool, client, encoder, config,
                                benchmark=args.benchmark)
    elif system == "workflow":
        result = eval_workflow(eval_tasks, args.benchmark, client, encoder, config)
    elif system == "random_evo":
        result = eval_random_evo(train_tasks, eval_tasks, seed_pool,
                                 client, encoder, config, benchmark=args.benchmark)
    elif system == "sero":
        result = eval_sero(train_tasks, eval_tasks, seed_pool, client, encoder,
                           config, checkpoint=args.checkpoint,
                           benchmark=args.benchmark, results_dir=sero_results_dir)
    else:
        logger.error("Unknown system: %s", system)
        sys.exit(1)

    result["benchmark"] = args.benchmark
    result["eval_set"] = eval_set
    result["seed"] = args.seed
    result["requested_eval_tasks"] = args.eval_tasks
    result["train_tasks_loaded"] = len(train_tasks)
    result["eval_tasks_loaded"] = len(eval_tasks)
    result["uses_split_train_keys"] = train_keys is not None
    result["cli_args"] = run_manifest["cli_args"]
    result["effective_config"] = run_manifest["effective_config"]
    result["resolved_run_args"] = run_manifest["resolved_run_args"]
    result["run_metadata"] = run_manifest["run_metadata"]
    if train_keys is not None:
        result["split_file"] = split_path
    if eval_set == "natural_full" and needs_train:
        result["eval_includes_train_keys"] = True

    _save_result(result, args.benchmark, system, args.suffix)


if __name__ == "__main__":
    main()
