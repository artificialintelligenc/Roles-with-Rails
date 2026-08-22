"""
NaturalPlan Calendar Scheduling adapter.

Data format: calendar_scheduling.json
    Each example has:
        - num_people: int
        - num_days: int
        - duration: int (minutes)
        - prompt_0shot: zero-shot natural language prompt
        - golden_plan: e.g. "Here is the proposed time: Monday, 14:30 - 15:30"

Evaluation:
    - Parse the agent's response for a final (day, start, end) slot tuple
    - Accept equivalent punctuation / separator variants when extracting the slot
    - Binary exact match: day, start time, and end time must all match golden plan
    - Score = 1.0 if exact match, else 0.0
"""

import json
import re
import os
import random
from typing import List, Dict, Any, Optional, Tuple


_DAY_NAME_PATTERN = r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_TIME_VALUE_PATTERN = r"\d{1,2}:\d{2}"
_SLOT_SEPARATOR_PATTERN = r"(?:-|–|—|to)"


def _find_last_time_slot(
    text: str,
    pattern: str,
    flags: int = 0,
) -> Optional[Tuple[str, str, str]]:
    """Return the last matched (day, start, end) tuple for a regex pattern."""
    matches = list(re.finditer(pattern, text, flags))
    if not matches:
        return None

    match = matches[-1]
    return match.group(1), match.group(2), match.group(3)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_calendar_scheduling_tasks(
    benchmark_dir: str,
    max_tasks: int = 200,
    seed: int = 42,
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load calendar scheduling tasks from NaturalPlan dataset.

    Args:
        include_keys: If set, load ONLY these task keys (ignores max_tasks/seed shuffle).
        exclude_keys: If set, remove these keys before sampling.

    Returns list of task dicts with keys: id, prompt, eval_fn.
    """
    data_path = os.path.join(benchmark_dir, "natural-plan", "data", "calendar_scheduling.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Calendar scheduling data not found at {data_path}")

    with open(data_path) as f:
        raw = json.load(f)

    if include_keys is not None:
        examples = [(k, raw[k]) for k in include_keys if k in raw]
    else:
        examples = list(raw.items())
        if exclude_keys:
            ex_set = set(exclude_keys)
            examples = [(k, v) for k, v in examples if k not in ex_set]
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_tasks]

    tasks = []
    for key, ex in examples:
        prompt = ex["prompt_0shot"]
        golden = ex.get("golden_plan", "").strip()
        num_people = ex.get("num_people", 0)
        num_days = ex.get("num_days", 1)
        duration = ex.get("duration", 60)

        golden_slot = _parse_time_slot(golden)
        if golden_slot is None:
            continue  # skip malformed examples

        def make_eval_fn(gold_slot: Tuple[str, str, str]):
            def eval_fn(response: str) -> float:
                return score_calendar_response(response, gold_slot)
            return eval_fn

        tasks.append({
            "id": key,
            "prompt": prompt,
            "eval_fn": make_eval_fn(golden_slot),
            "golden_plan": golden,
            "num_people": num_people,
            "num_days": num_days,
            "duration": duration,
        })

    return tasks


# ── Evaluation ────────────────────────────────────────────────────────────────

def score_calendar_response(
    response: str,
    golden_slot: Tuple[str, str, str],  # (day, start_time, end_time)
) -> float:
    """
    Score a calendar scheduling response.

    Binary: 1.0 if the predicted slot exactly matches the golden slot, else 0.0.
    The day name, start time, and end time must all match (case-insensitive for day).
    """
    if not response.strip():
        return 0.0

    pred_slot = _parse_time_slot(response)
    if pred_slot is None:
        return 0.0

    pred_day, pred_start, pred_end = pred_slot
    gold_day, gold_start, gold_end = golden_slot

    return 1.0 if (
        pred_day.lower() == gold_day.lower()
        and _normalize_time(pred_start) == _normalize_time(gold_start)
        and _normalize_time(pred_end) == _normalize_time(gold_end)
    ) else 0.0


def _parse_time_slot(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract (day, start_time, end_time) from a string like:
      "Here is the proposed time: Monday, 14:30 - 15:30"
    or equivalent variants such as:
      - "Monday 14:30 - 15:30"
      - "Monday: 14:30 to 15:30"
      - "Day: Monday ... Start Time: 14:30 ... End Time: 15:30"

    Returns None if no match found.
    """
    if not text:
        return None

    # Prefer explicit final-answer phrasing when present.
    proposed_pattern = (
        rf"Here is the proposed time\s*:\s*"
        rf"({_DAY_NAME_PATTERN})\s*(?:,|:)?\s*"
        rf"({_TIME_VALUE_PATTERN})\s*{_SLOT_SEPARATOR_PATTERN}\s*({_TIME_VALUE_PATTERN})"
    )
    slot = _find_last_time_slot(text, proposed_pattern, re.IGNORECASE)
    if slot is not None:
        return slot

    # Handle structured recommendations such as:
    # Day: Monday / Start Time: 13:00 / End Time: 13:30
    structured_pattern = (
        rf"Day\s*:\s*({_DAY_NAME_PATTERN}).{{0,200}}?"
        rf"Start\s*Time\s*:\s*({_TIME_VALUE_PATTERN}).{{0,200}}?"
        rf"End\s*Time\s*:\s*({_TIME_VALUE_PATTERN})"
    )
    slot = _find_last_time_slot(text, structured_pattern, re.IGNORECASE | re.DOTALL)
    if slot is not None:
        return slot

    # Fallback: accept any day/time slot mention and take the last one, which
    # best reflects verbose responses that reason first and answer last.
    generic_pattern = (
        rf"\b({_DAY_NAME_PATTERN})\b\s*(?:,|:)?\s*"
        rf"({_TIME_VALUE_PATTERN})\s*{_SLOT_SEPARATOR_PATTERN}\s*({_TIME_VALUE_PATTERN})"
    )
    slot = _find_last_time_slot(text, generic_pattern, re.IGNORECASE)
    if slot is not None:
        return slot

    return None


def _normalize_time(t: str) -> str:
    """Normalize time string to HH:MM format (zero-pad hours)."""
    parts = t.split(":")
    if len(parts) == 2:
        hour = parts[0].zfill(2)
        minute = parts[1].zfill(2)
        return f"{hour}:{minute}"
    return t


# ── SC Canonical Answer Extraction ────────────────────────────────────────────

def extract_canonical_answer(response: str) -> str:
    """Extract a canonical time slot string for SC majority voting.

    Uses _parse_time_slot to parse and _normalize_time to normalize,
    producing a deterministic "day|HH:MM|HH:MM" string.
    """
    slot = _parse_time_slot(response)
    if slot:
        day, start, end = slot
        return f"{day.lower()}|{_normalize_time(start)}|{_normalize_time(end)}"
    return ""
