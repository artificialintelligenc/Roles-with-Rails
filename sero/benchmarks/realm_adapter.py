"""
REALM-Bench J1 (Job Shop Scheduling) adapter.

JSSP format (standard OR-Lib):
  First line: n_jobs n_machines
  Next n_jobs lines: pairs (machine_id processing_time) for each operation in order

Evaluation:
  - Agent produces a schedule as text listing start times
  - We parse the makespan from the response
  - Score = best_known_makespan / parsed_makespan (capped at 1.0)
    → 1.0 if optimal, lower if worse schedule

Fallback: if no parseable makespan found, score = 0.0
"""

import os
import re
import glob
import random
from typing import List, Dict, Any, Tuple, Optional


# ── JSSP Instance Parsing ─────────────────────────────────────────────────────

def parse_jssp_instance(filepath: str) -> Optional[Dict]:
    """
    Parse a standard JSSP text file.
    Returns dict with n_jobs, n_machines, jobs (list of lists of (machine, time)).
    """
    try:
        with open(filepath) as f:
            lines = [l.strip() for l in f if l.strip()]

        header = lines[0].split()
        n_jobs, n_machines = int(header[0]), int(header[1])
        jobs = []
        for i in range(1, n_jobs + 1):
            if i >= len(lines):
                break
            tokens = lines[i].split()
            ops = []
            for j in range(0, len(tokens) - 1, 2):
                machine = int(tokens[j])
                duration = int(tokens[j + 1])
                ops.append((machine, duration))
            jobs.append(ops)

        return {
            "n_jobs": n_jobs,
            "n_machines": n_machines,
            "jobs": jobs,
            "source_file": os.path.basename(filepath),
        }
    except Exception as e:
        return None


# ── Known Optimal/Best Makespans ──────────────────────────────────────────────

# Best known makespans for common JSSP benchmarks (from literature)
KNOWN_BEST = {
    "abz07.txt": 656,
    "abz08.txt": 665,
    "abz09.txt": 679,
    "swv01.txt": 1407,
    "swv02.txt": 1475,
    "swv03.txt": 1398,
    "swv04.txt": 1470,
    "swv05.txt": 1424,
    "yn1.txt": 884,
    "yn2.txt": 904,
    "yn3.txt": 892,
    "yn4.txt": 968,
    # TA instances
    "ta01.txt": 1231,
    "ta02.txt": 1244,
    "ta03.txt": 1218,
    "ta04.txt": 1175,
    "ta05.txt": 1224,
}


# ── Prompt Construction ───────────────────────────────────────────────────────

def instance_to_prompt(instance: Dict) -> str:
    """Convert a parsed JSSP instance to a natural language prompt."""
    n_jobs = instance["n_jobs"]
    n_machines = instance["n_machines"]
    jobs = instance["jobs"]

    lines = [
        f"Job Shop Scheduling Problem: {n_jobs} jobs, {n_machines} machines.",
        "",
        "Each job has a fixed sequence of operations (machine, processing time).",
        "Each machine can process at most one operation at a time.",
        "Operations within a job must be processed in the given order.",
        "Goal: minimize the makespan (time when all jobs complete).",
        "",
        "Jobs (operation sequence as [machine, duration]):",
    ]

    for i, ops in enumerate(jobs):
        op_str = " → ".join(f"[M{m},{t}]" for m, t in ops)
        lines.append(f"  Job {i}: {op_str}")

    lines += [
        "",
        "Please provide:",
        "1. A schedule assigning start times to each operation",
        "2. The resulting makespan",
        "",
        "Format your answer as:",
        "SCHEDULE:",
        "Job X, Op Y: Machine M, Start=T, Duration=D",
        "...",
        "MAKESPAN: <number>",
    ]
    return "\n".join(lines)


# ── Score Extraction ──────────────────────────────────────────────────────────

def extract_makespan(response: str) -> Optional[int]:
    """
    Extract the makespan number from an agent's response.
    Uses multiple patterns from most to least specific.
    """
    patterns = [
        r"MAKESPAN[:\s=*]+(\d+)",
        r"makespan[:\s=*]+(\d+)",
        r"total\s+(?:time|duration|makespan|completion)[:\s=*]+(\d+)",
        r"(?:minimum|optimal|best)\s+makespan[:\s=*]+(\d+)",
        r"(?:schedule|finishing|completion)\s+time[:\s=*]+(\d+)",
        r"(?:C_max|Cmax|C-max)[:\s=*]+(\d+)",
        r"time[:\s=]+(\d{3,5})",
        r"=\s*(\d{3,5})\s*(?:time|units?|steps?)?",
        r"\*\*(\d{3,5})\*\*",              # bold number (markdown)
        r"(?:is|are|equals?)\s+(\d{3,5})",
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 100 <= val <= 999999:
                return val

    # Last-resort: scan last 5 lines for a standalone number in range
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    for line in reversed(lines[-5:]):
        nums = re.findall(r"\b(\d{3,6})\b", line)
        for n in reversed(nums):
            val = int(n)
            if 100 <= val <= 99999:
                return val
    return None


def score_response(response: str, instance: Dict, best_known: Optional[int] = None) -> float:
    """
    Score an agent's schedule response.
    Returns float in [0, 1]: 1.0 = optimal or better, 0.0 = failed/invalid.
    """
    parsed_makespan = extract_makespan(response)
    if parsed_makespan is None:
        return 0.0

    if best_known is not None and best_known > 0:
        ratio = best_known / parsed_makespan
        return min(1.0, max(0.0, ratio))

    # No best known: compute a lower bound (sum of critical path)
    # and use a relative score based on the computed makespan vs lower bound
    lb = _compute_lower_bound(instance)
    if lb > 0:
        return min(1.0, lb / parsed_makespan)
    return 0.5  # neutral score if we can't evaluate


def _compute_lower_bound(instance: Dict) -> int:
    """
    Compute a simple lower bound: max(max_job_time, max_machine_load).
    """
    jobs = instance["jobs"]
    n_machines = instance["n_machines"]

    # Minimum job completion: each job takes at least sum of its operation times
    max_job_time = max(sum(t for _, t in ops) for ops in jobs)

    # Minimum machine completion: each machine's total load
    machine_load = [0] * n_machines
    for ops in jobs:
        for m, t in ops:
            machine_load[m] += t
    max_machine_load = max(machine_load)

    return max(max_job_time, max_machine_load)


# ── Dataset Loader ────────────────────────────────────────────────────────────

def load_j1_tasks(
    benchmark_dir: str,
    max_tasks: int = 30,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load JSSP J1 tasks from REALM-Bench dataset directory.

    Returns list of task dicts with keys: id, prompt, eval_fn.
    """
    j1_dir = os.path.join(benchmark_dir, "REALM-Bench", "datasets", "J1")
    if not os.path.exists(j1_dir):
        raise FileNotFoundError(f"J1 dataset not found at {j1_dir}")

    # Collect all .txt files recursively
    txt_files = glob.glob(os.path.join(j1_dir, "**", "*.txt"), recursive=True)
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {j1_dir}")

    rng = random.Random(seed)
    rng.shuffle(txt_files)
    txt_files = txt_files[:max_tasks]

    tasks = []
    for fpath in txt_files:
        instance = parse_jssp_instance(fpath)
        if instance is None:
            continue

        fname = os.path.basename(fpath)
        best_known = KNOWN_BEST.get(fname)
        prompt = instance_to_prompt(instance)

        def make_eval_fn(inst, bk):
            def eval_fn(response: str) -> float:
                return score_response(response, inst, best_known=bk)
            return eval_fn

        tasks.append({
            "id": f"jssp_{fname.replace('.txt', '')}",
            "prompt": prompt,
            "eval_fn": make_eval_fn(instance, best_known),
            "instance": instance,
            "best_known": best_known,
        })

    return tasks
