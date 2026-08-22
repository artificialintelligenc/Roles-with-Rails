"""
NaturalPlan Meeting Planning adapter.

Data format: meeting_planning.json
  Each example has:
    - num_people: int
    - constraints: [[start_loc, start_time], [person, location, time_range, min_duration], ...]
    - dist_matrix: {loc: {loc: minutes}} or similar
    - prompt_0shot: zero-shot natural language prompt
    - golden_plan: list of steps (ground truth solution)

Evaluation returns a dict with two metrics:
    - partial_score: valid_meetings / golden_valid (continuous [0,1], continue-on-error)
    - exact_score:   strict binary {0,1} (break-on-error)

Both modes use real travel times from dist_matrix and meeting durations parsed
from the response text.
"""

import json
import re
import os
import random
import datetime
from typing import List, Dict, Any, Optional


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_meeting_planning_tasks(
    benchmark_dir: str,
    max_tasks: int = 200,
    min_people: int = 3,
    seed: int = 42,
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load meeting planning tasks from NaturalPlan dataset.

    Args:
        include_keys: If set, load ONLY these task keys (ignores max_tasks/seed shuffle).
        exclude_keys: If set, remove these keys before sampling.

    Returns list of task dicts with keys: id, prompt, eval_fn.
    eval_fn returns a dict: {"partial_score": float, "exact_score": float}
    """
    data_path = os.path.join(benchmark_dir, "natural-plan", "data", "meeting_planning.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Meeting planning data not found at {data_path}")

    with open(data_path) as f:
        raw = json.load(f)

    if include_keys is not None:
        examples = [(k, raw[k]) for k in include_keys if k in raw]
    else:
        examples = [(k, v) for k, v in raw.items() if v.get("num_people", 0) >= min_people]
        if exclude_keys:
            ex_set = set(exclude_keys)
            examples = [(k, v) for k, v in examples if k not in ex_set]
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_tasks]

    tasks = []
    for key, ex in examples:
        constraints = ex.get("constraints", [])
        if len(constraints) < 2:
            continue

        start_info = constraints[0]
        person_constraints = constraints[1:]

        prompt = ex.get("prompt_0shot", "")
        if not prompt:
            continue

        golden = ex.get("golden_plan", [])
        if isinstance(golden, list):
            golden_steps = golden
        else:
            golden_steps = [golden]

        dist_matrix = ex.get("dist_matrix", {})
        people = [c[0] for c in person_constraints if len(c) >= 4]

        # Pre-compute golden valid count using the NaturalPlan-style exact
        # validator so the denominator is consistent with the original benchmark.
        golden_valid = _count_valid_meetings_exact(
            golden_steps,
            person_constraints,
            start_info[0],
            start_info[1],
            dist_matrix,
        )

        def make_eval_fn(
            start_: list,
            person_constraints_: list,
            dist_matrix_: dict,
            golden_valid_: int,
        ):
            def eval_fn(response: str) -> Dict[str, float]:
                return score_meeting_response(
                    response, start_, person_constraints_, dist_matrix_, golden_valid_
                )
            return eval_fn

        tasks.append({
            "id": key,
            "prompt": prompt,
            "eval_fn": make_eval_fn(
                start_info, person_constraints, dist_matrix, golden_valid
            ),
            "people": people,
            "num_people": ex.get("num_people", len(people)),
            "golden_plan": " ".join(golden_steps),
        })

    return tasks


# ── Shared helpers ────────────────────────────────────────────────────────────

def _to_time(time_str: str) -> datetime.datetime:
    """Parse time string like '9:00AM', '10:30AM', '1:30PM', '9:30pm', '14:30', '09:05'."""
    s = time_str.strip().upper()
    s = re.sub(r'\s+', '', s)
    for fmt in ("%I:%M%p", "%H:%M"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"Cannot parse time: {time_str!r}")


def _parse_plan_to_steps_np(response: str) -> List[str]:
    """Parse text response using the exact same logic as NaturalPlan's parse_text_plan.

    This mirrors evaluate_meeting_planning.py::parse_text_plan exactly:
    split on ".", strip, keep non-empty.  No number-prefix stripping,
    no "you" filtering — the validator itself handles unknown formats.
    """
    if "SOLUTION:" in response:
        response = response[response.find("SOLUTION:") + len("SOLUTION:"):].strip()
    parts = response.split(".")
    return [p.strip() for p in parts if p.strip()]


def _parse_plan_to_steps(response: str) -> List[str]:
    """Parse text response into individual plan steps (lenient, for partial scoring)."""
    if "SOLUTION:" in response:
        response = response[response.find("SOLUTION:") + len("SOLUTION:"):].strip()

    # Strip markdown bold/italic markers before parsing
    response = re.sub(r'\*{1,3}', '', response)

    raw = re.split(r'\.\s+|\.\s*$', response)
    steps = []
    for s in raw:
        s = s.strip()
        # Strip leading list markers like "1. ", "2) ", "- "
        s = re.sub(r'^(?:\d+[\.\)]\s*|-\s*)', '', s).strip()
        if s and s.lower().startswith("you "):
            steps.append(s)
    return steps


def _build_processed_constraints(person_constraints: list) -> Dict[str, Any]:
    """Build processed constraints dict from raw person_constraints."""
    processed: Dict[str, Any] = {}
    for c in person_constraints:
        if len(c) < 4:
            continue
        person, location, time_range, meeting_time = c[0], c[1], c[2], c[3]
        parts = time_range.split("to")
        if len(parts) != 2:
            continue
        try:
            start = _to_time(parts[0].strip())
            end = _to_time(parts[1].strip())
        except ValueError:
            continue
        processed[person] = {
            "location": location,
            "start_time": start,
            "end_time": end,
            "meeting_time": int(meeting_time),
        }
    return processed


def _lookup_travel(dist_matrix: dict, src: str, dst: str) -> int:
    """Lookup travel time between two locations."""
    if src in dist_matrix and dst in dist_matrix[src]:
        return int(dist_matrix[src][dst])
    raise ValueError(f"No dist_matrix entry: {src} -> {dst}")


def _parse_travel_step_info(step: str) -> tuple[str, Optional[int], Optional[str]]:
    """Parse destination plus optional stated minutes/arrival from a travel step."""
    m = re.search(
        r"travel to (.+?)(?:\s+in\s+(\d+)\s+minutes?)?(?:\s+and\s+arrive\s+at\s+(.+?))?\.?$",
        step.strip(),
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Cannot parse travel step: {step}")

    destination = m.group(1).strip()
    stated_minutes = int(m.group(2)) if m.group(2) else None
    stated_arrival = m.group(3).strip() if m.group(3) else None
    return destination, stated_minutes, stated_arrival


def _parse_meet_step_info(step: str) -> tuple[str, int, Optional[str], Optional[str]]:
    """Parse person, duration, and optional explicit start/end times from a meet step."""
    time_token = r"(\d{1,2}:\d{2}\s*[APap][Mm]?|\d{1,2}:\d{2})"
    m = re.search(
        rf"meet (.+?) for (\d+) minutes?(?: from {time_token} to {time_token})?",
        step,
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Cannot parse meet step: {step}")

    person = m.group(1).strip()
    duration = int(m.group(2))
    stated_start = m.group(3).strip() if m.group(3) else None
    stated_end = m.group(4).strip() if m.group(4) else None
    return person, duration, stated_start, stated_end


def _handle_travel_step(step, cur_time, cur_location, dist_matrix, validate_stated: bool = False):
    """Parse a travel step and return (new_time, new_location).
    Raises ValueError if unparseable."""
    destination, stated_minutes, stated_arrival = _parse_travel_step_info(step)
    travel_mins = _lookup_travel(dist_matrix, cur_location, destination)
    new_time = cur_time + datetime.timedelta(minutes=travel_mins)

    if validate_stated and stated_minutes is not None and stated_minutes != travel_mins:
        raise ValueError("Stated travel minutes do not match dist_matrix")
    if validate_stated and stated_arrival is not None and _to_time(stated_arrival) != new_time:
        raise ValueError("Stated arrival time does not match actual travel")

    return new_time, destination


def _handle_wait_step(step, cur_time):
    """Parse a wait step and return new_time. Raises ValueError if unparseable."""
    # Match 12h format (e.g. '2:15PM') or 24h format (e.g. '14:15')
    m = re.search(r"wait until (\d+:\d+\s*[APap][Mm]?)", step, re.IGNORECASE)
    if not m:
        # Fallback: bare 24h time like '14:15', '21:30'
        m = re.search(r"wait until (\d{1,2}:\d{2})", step, re.IGNORECASE)
    if m:
        end_wait = _to_time(m.group(1))
        if end_wait < cur_time:
            raise ValueError("Cannot go backwards in time")
        return end_wait
    raise ValueError(f"Cannot parse wait step: {step}")


def _handle_meet_step(
    step,
    cur_time,
    cur_location,
    processed,
    met_with,
    allow_implicit_wait: bool = False,
    validate_stated: bool = False,
):
    """Parse a meet step and return (score_delta, new_time, person_met).
    Raises ValueError on constraint violation."""
    person, meeting_dur, stated_start, stated_end = _parse_meet_step_info(step)

    if person in met_with:
        raise ValueError(f"Already met {person}")
    if person not in processed:
        raise ValueError(f"Unknown person: {person}")

    if stated_start is not None:
        parsed_start = _to_time(stated_start)
        if allow_implicit_wait:
            if parsed_start > cur_time:
                cur_time = parsed_start
        elif parsed_start != cur_time:
            raise ValueError("Stated meeting start does not match current time")

    c_info = processed[person]
    new_time = cur_time + datetime.timedelta(minutes=meeting_dur)
    if validate_stated and stated_end is not None and _to_time(stated_end) != new_time:
        raise ValueError("Stated meeting end does not match duration")
    if (
        cur_location == c_info["location"]
        and cur_time >= c_info["start_time"]
        and new_time <= c_info["end_time"]
        and meeting_dur >= c_info["meeting_time"]
    ):
        return 1, new_time, person
    raise ValueError("Invalid meeting time or location")


# ── Partial-credit scoring (SERO-style: continue on error) ───────────────────

def _count_valid_meetings_partial(
    steps: List[str],
    person_constraints: list,
    start_location: str,
    start_time: str,
    dist_matrix: dict,
) -> int:
    """Count valid meetings, skipping steps that fail (continue on error).

    Failed steps do not immediately zero out later opportunities, but time still
    advances according to the real travel graph and the response's stated wait /
    meeting durations.
    """
    processed = _build_processed_constraints(person_constraints)
    if not processed:
        return 0
    try:
        cur_location = start_location
        cur_time = _to_time(start_time)
    except ValueError:
        return 0

    met_with: Dict[str, bool] = {}
    score = 0

    for step in steps:
        try:
            if step.lower().startswith("you start"):
                continue
            elif step.lower().startswith("you travel"):
                cur_time, cur_location = _handle_travel_step(
                    step, cur_time, cur_location, dist_matrix)
            elif step.lower().startswith("you wait"):
                cur_time = _handle_wait_step(step, cur_time)
            elif step.lower().startswith("you meet"):
                delta, cur_time, person = _handle_meet_step(
                    step,
                    cur_time,
                    cur_location,
                    processed,
                    met_with,
                    allow_implicit_wait=True,
                    validate_stated=False,
                )
                score += delta
                met_with[person] = True
        except Exception:
            # On failure, still advance cur_time using stated durations so
            # subsequent steps are evaluated at the time the LLM intended.
            sl = step.lower()
            if sl.startswith("you meet"):
                try:
                    _, dur, stated_start, _ = _parse_meet_step_info(step)
                except ValueError:
                    continue
                if stated_start is not None:
                    try:
                        parsed_start = _to_time(stated_start)
                        if parsed_start > cur_time:
                            cur_time = parsed_start
                    except ValueError:
                        pass
                cur_time = cur_time + datetime.timedelta(minutes=dur)
            elif sl.startswith("you travel"):
                try:
                    destination, _, _ = _parse_travel_step_info(step)
                    travel_mins = _lookup_travel(dist_matrix, cur_location, destination)
                    cur_location = destination
                    cur_time = cur_time + datetime.timedelta(minutes=travel_mins)
                except Exception:
                    pass
            elif sl.startswith("you wait"):
                m = re.search(r"wait until (\d+:\d+\s*[APap][Mm]?)", step, re.IGNORECASE)
                if not m:
                    m = re.search(r"wait until (\d{1,2}:\d{2})", step, re.IGNORECASE)
                if m:
                    try:
                        new_t = _to_time(m.group(1))
                        if new_t > cur_time:
                            cur_time = new_t
                    except ValueError:
                        pass
            continue

    return score


def _evaluate_exact_meetings(
    steps: List[str],
    person_constraints: list,
    start_location: str,
    start_time: str,
    dist_matrix: dict,
) -> tuple[int, bool]:
    """Evaluate a plan in strict mode and return (valid_count, had_error)."""
    processed = _build_processed_constraints(person_constraints)
    if not processed:
        return 0, True
    try:
        cur_location = start_location
        cur_time = _to_time(start_time)
    except ValueError:
        return 0, True

    met_with: Dict[str, bool] = {}
    score = 0

    for step in steps:
        sl = step.lower().strip()
        try:
            if sl.startswith("you start"):
                continue
            elif sl.startswith("you travel"):
                cur_time, cur_location = _handle_travel_step(
                    step,
                    cur_time,
                    cur_location,
                    dist_matrix,
                    validate_stated=True,
                )

            elif sl.startswith("you wait"):
                # --- wait: end_time <= cur_time is an error ---
                m = re.search(
                    r"wait until (\d+:\d+\s*[APap][Mm]?)", step, re.IGNORECASE
                )
                if not m:
                    m = re.search(
                        r"wait until (\d{1,2}:\d{2})", step, re.IGNORECASE
                    )
                if not m:
                    raise ValueError(f"Cannot parse wait step: {step}")
                end_wait = _to_time(m.group(1))
                if end_wait <= cur_time:
                    raise ValueError("Cannot go backwards in time")
                cur_time = end_wait

            elif sl.startswith("you meet"):
                delta, cur_time, person = _handle_meet_step(
                    step,
                    cur_time,
                    cur_location,
                    processed,
                    met_with,
                    allow_implicit_wait=False,
                    validate_stated=True,
                )
                score += delta
                met_with[person] = True
            else:
                raise ValueError("Unknown plan format")

        except (ValueError, KeyError, TypeError):
            return score, True

    return score, False


# ── Exact-match scoring (NaturalPlan-style: break on error) ──────────────────

def _count_valid_meetings_exact(
    steps: List[str],
    person_constraints: list,
    start_location: str,
    start_time: str,
    dist_matrix: dict,
) -> int:
    """Count valid meetings in strict mode, ignoring any suffix after the first error."""
    score, _ = _evaluate_exact_meetings(
        steps, person_constraints, start_location, start_time, dist_matrix
    )
    return score


# ── Combined scoring entry point ─────────────────────────────────────────────

def score_meeting_response(
    response: str,
    start_info: list,
    person_constraints: list,
    dist_matrix: dict,
    golden_valid: int,
) -> Dict[str, float]:
    """
    Score a meeting planning response with two metrics.

    partial_score: SERO-style — valid_meetings / golden_valid.
        Skips failed steps but still advances time, counting all
        independently valid meetings regardless of earlier errors.

    exact_score: strict binary {0, 1}.
        Uses real travel times from dist_matrix and the response's stated
        meeting duration, with break-on-first-error semantics.
        1.0 iff valid_count == golden_valid.

    Returns:
        {"partial_score": float in [0,1], "exact_score": float in {0,1}}
    """
    if not response.strip():
        return {"partial_score": 0.0, "exact_score": 0.0}

    if golden_valid <= 0:
        has_meet = bool(re.search(r"\bmeet\b", response, re.IGNORECASE))
        v = 1.0 if has_meet else 0.0
        return {"partial_score": v, "exact_score": v}

    start_location = start_info[0] if start_info else "Unknown"
    start_time = start_info[1] if len(start_info) > 1 else "9:00AM"

    steps = _parse_plan_to_steps(response)
    steps_np = _parse_plan_to_steps_np(response)

    partial_valid = _count_valid_meetings_partial(
        steps, person_constraints, start_location, start_time, dist_matrix
    )
    exact_valid = _count_valid_meetings_exact(
        steps_np, person_constraints, start_location, start_time, dist_matrix
    )

    partial_score = min(1.0, partial_valid / golden_valid)
    exact_score = 1.0 if exact_valid == golden_valid else 0.0

    return {"partial_score": partial_score, "exact_score": exact_score}


# ── SC Canonical Answer Extraction ────────────────────────────────────────────

# Words that are clearly not person names
_NOISE_NAMES = {
    "informal", "catch-up", "an additional", "everyone", "all",
    "friends", "them", "no set person", "for",
}


def _is_valid_person_name(name: str) -> bool:
    """Check if the extracted name looks like a real person name."""
    low = name.lower()
    # Reject if it contains noise fragments
    for noise in _NOISE_NAMES:
        if noise in low:
            return False
    # Reject names that are too long (likely garbage text) or too short
    if len(name) > 40 or len(name) < 2:
        return False
    # Reject names that look like descriptions (contain multiple common words)
    if re.search(r'\b(?:time|person|set|informal|minutes|hours)\b', low):
        return False
    return True


def extract_canonical_answer(response: str) -> str:
    """Extract a canonical meeting plan string for SC majority voting.

        Parses the response into a sequence of meeting steps, extracting four fields
    that correspond to the original NaturalPlan validator's scoring conditions:
      - person: who to meet (determines constraint lookup)
      - location: where (must match constraint's location)
      - start_time: when meeting starts (must be within time window)
      - duration: how long (determines if meeting ends within window)

    Format: "person@location@start_time:duration|..."

    Deduplicates: keeps only the first occurrence of each person.
    Filters invalid names and 0-minute meetings.
    """
    steps = _parse_plan_to_steps(response)
    meetings = []
    seen_persons: set = set()
    cur_location = ""  # Track current location from the start / travel steps

    for step in steps:
        sl = step.lower()

        if sl.startswith("you start"):
            m = re.search(r"start at (.+?) at", step, re.IGNORECASE)
            if m:
                cur_location = m.group(1).strip()
            continue

        # Update location from travel steps
        if sl.startswith("you travel"):
            try:
                cur_location, _, _ = _parse_travel_step_info(step)
            except ValueError:
                pass
            continue

        # Extract meeting info
        if sl.startswith("you meet"):
            m = re.search(r"meet (.+?) for (\d+) minutes?", step, re.IGNORECASE)
            if not m:
                continue

            person = m.group(1).strip()
            duration = int(m.group(2))

            # Strip trailing "again" or "for an additional ..." artifacts
            person = re.sub(r'\s+again$', '', person, flags=re.IGNORECASE).strip()
            person = re.sub(r'\s+for an additional.*$', '', person, flags=re.IGNORECASE).strip()

            # Skip 0-minute meetings
            if duration <= 0:
                continue

            # Skip garbage names
            if not _is_valid_person_name(person):
                continue

            # Deduplicate: keep first occurrence only
            person_key = person.lower()
            if person_key in seen_persons:
                continue
            seen_persons.add(person_key)

            # Extract start time from "from HH:MM AM/PM to ..."
            start_time = ""
            m_time = re.search(r"from\s+(\d{1,2}:\d{2}\s*[APap][Mm]?)", step, re.IGNORECASE)
            if m_time:
                # Normalize: remove spaces, colons, uppercase AM/PM → e.g. "930AM"
                start_time = m_time.group(1).strip().replace(" ", "").replace(":", "").upper()

            # Format: person@location@start_time:duration
            meetings.append(f"{person}@{cur_location}@{start_time}:{duration}")

    if meetings:
        return "|".join(meetings)

    return ""
