"""
Combined NaturalPlan adapter.

This benchmark mixes the three NaturalPlan sub-tasks that SERO previously ran
as separate benchmarks:

- trip planning
- calendar scheduling
- meeting planning

Every returned task has a `sub_benchmark` field and a dual-score eval_fn:
`{"partial_score": float, "exact_score": float}`. Trip and meeting reuse
their existing partial/exact scorers. Calendar has only exact matching, so its
partial score is defined to be the same as exact.
"""

import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sero.benchmarks.calendar_scheduling_adapter import load_calendar_scheduling_tasks
from sero.benchmarks.meeting_plan_adapter import load_meeting_planning_tasks
from sero.benchmarks.trip_adapter import load_trip_planning_tasks


SUB_BENCHMARKS: Tuple[str, ...] = ("trip", "calendar", "meeting")
KEY_SEPARATOR = "::"

SUBTASK_FORMAT_INSTRUCTIONS: Dict[str, str] = {
    "trip": (
        "IMPORTANT: This is a TRIP PLANNING task. Your final answer must contain only itinerary lines "
        "of the forms '**Day X-Y:** Visit CityName for N days.' and '**Day Z:** Fly from CityA to CityB.'. "
        "Use the shared flight-day rule exactly: if a city visit starts on Day S and lasts D days, it ends on "
        "Day S+D-1; fly on that same end day, and the next city also starts on that flight day. The last day "
        "number must equal the total trip duration. Overlapping adjacent city ranges on a flight day are correct, "
        "not a double-counting error. Use only listed direct flights and satisfy all day-window events. Choose the "
        "city order that satisfies directed flights and event windows; do not assume the prompt order is the route. "
        "Do not drop or shorten required stays to force the total. Do not output calendar proposed-time "
        "lines or meeting travel/wait/meet lines."
    ),
    "calendar": (
        "IMPORTANT: This is a CALENDAR SCHEDULING task, even though the text asks to schedule a meeting. "
        "Find one shared free time slot from the intersection of every participant's free windows. Treat all "
        "preferences as hard constraints. Before finalizing, verify the chosen interval against each participant's "
        "original busy blocks; any overlap makes it invalid. The meeting must start at or after work begins and end "
        "at or before work closes. Treat intervals as half-open [start, end): a slot overlaps a busy block iff "
        "slot_start < busy_end AND slot_end > busy_start. Therefore a busy block ending at 14:00 allows a slot "
        "starting at 14:00, and a busy block starting at 16:30 allows a slot ending at 16:30; but a busy block "
        "11:00-17:00 rejects 12:00-12:30. Your final "
        "answer must be exactly one line: "
        "'Here is the proposed time: <Day>, HH:MM - HH:MM' using 24-hour time. "
        "Do not output travel steps, wait steps, or 'You meet ...' lines."
    ),
    "meeting": (
        "IMPORTANT: This is a MEETING PLANNING task with locations, travel times, and availability windows. "
        "Build a chronological feasible route from the stated start location/time. Every travel time must match "
        "the matrix, every meeting must start after arrival and within that person's window, and meeting_end must "
        "be no later than the window end. If arrival is early, insert a wait step; never wait backwards or schedule "
        "a meeting after the window closes. Prefer maximizing the number of valid meetings, but omit any invalid "
        "meeting rather than listing it. "
        "Your final answer must contain only step lines: "
        "'You travel to [Location] in [N] minutes and arrive at [H:MMAM/PM].', "
        "'You wait until [H:MMAM/PM].', and "
        "'You meet [Person] for [N] minutes from [H:MMAM/PM] to [H:MMAM/PM].'. "
        "Do not output a calendar proposed-time line."
    ),
}


def _split_key(key: str) -> Tuple[Optional[str], str]:
    if KEY_SEPARATOR in key:
        prefix, raw_key = key.split(KEY_SEPARATOR, 1)
        if prefix in SUB_BENCHMARKS:
            return prefix, raw_key
    if key.startswith("trip_planning"):
        return "trip", key
    if key.startswith("calendar_scheduling"):
        return "calendar", key
    if key.startswith("meeting_planning"):
        return "meeting", key
    return None, key


def _group_keys(keys: Optional[Iterable[str]]) -> Optional[Dict[str, List[str]]]:
    if keys is None:
        return None
    grouped: Dict[str, List[str]] = {name: [] for name in SUB_BENCHMARKS}
    ambiguous: List[str] = []
    for key in keys:
        sub_benchmark, raw_key = _split_key(key)
        if sub_benchmark is None:
            ambiguous.append(raw_key)
        else:
            grouped[sub_benchmark].append(raw_key)
    if ambiguous:
        for sub_benchmark in SUB_BENCHMARKS:
            grouped[sub_benchmark].extend(ambiguous)
    return grouped


def _dualize_eval_fn(eval_fn: Callable[[str], Any], sub_benchmark: str) -> Callable[[str], Dict[str, float]]:
    def wrapped(response: str) -> Dict[str, float]:
        raw = eval_fn(response)
        if isinstance(raw, dict):
            partial = float(raw.get("partial_score", 0.0))
            exact = float(raw.get("exact_score", partial))
            return {"partial_score": partial, "exact_score": exact}
        exact = float(raw)
        return {"partial_score": exact, "exact_score": exact}

    wrapped.__name__ = f"naturalplan_{sub_benchmark}_eval_fn"
    return wrapped


def _decorate_task(task: Dict[str, Any], sub_benchmark: str) -> Dict[str, Any]:
    raw_id = str(task["id"])
    task_prompt = str(task["prompt"]).rstrip()
    format_instruction = SUBTASK_FORMAT_INSTRUCTIONS[sub_benchmark]
    combined = dict(task)
    combined["id"] = f"{sub_benchmark}{KEY_SEPARATOR}{raw_id}"
    combined["prompt"] = f"{format_instruction}\n\n{task_prompt}\n\n{format_instruction}"
    combined["original_id"] = raw_id
    combined["benchmark"] = "naturalplan"
    combined["sub_benchmark"] = sub_benchmark
    combined["task_type"] = sub_benchmark
    combined["eval_fn"] = _dualize_eval_fn(task["eval_fn"], sub_benchmark)
    return combined


def _load_subtasks(
    benchmark_dir: str,
    sub_benchmark: str,
    max_tasks: int,
    seed: int,
    include_keys: Optional[List[str]],
    exclude_keys: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if sub_benchmark == "trip":
        tasks = load_trip_planning_tasks(
            benchmark_dir,
            max_tasks=max_tasks,
            seed=seed,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
    elif sub_benchmark == "calendar":
        tasks = load_calendar_scheduling_tasks(
            benchmark_dir,
            max_tasks=max_tasks,
            seed=seed,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
    elif sub_benchmark == "meeting":
        tasks = load_meeting_planning_tasks(
            benchmark_dir,
            max_tasks=max_tasks,
            min_people=1,
            seed=seed,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
    else:
        raise ValueError(f"Unknown NaturalPlan sub-benchmark: {sub_benchmark}")

    return [_decorate_task(task, sub_benchmark) for task in tasks]


def _round_robin(tasks_by_sub_benchmark: Dict[str, List[Dict[str, Any]]], max_tasks: int) -> List[Dict[str, Any]]:
    if max_tasks is None:
        max_tasks = sum(len(tasks) for tasks in tasks_by_sub_benchmark.values())

    selected: List[Dict[str, Any]] = []
    cursors = {name: 0 for name in SUB_BENCHMARKS}
    while len(selected) < max_tasks:
        progressed = False
        for sub_benchmark in SUB_BENCHMARKS:
            cursor = cursors[sub_benchmark]
            tasks = tasks_by_sub_benchmark[sub_benchmark]
            if cursor >= len(tasks):
                continue
            selected.append(tasks[cursor])
            cursors[sub_benchmark] += 1
            progressed = True
            if len(selected) >= max_tasks:
                break
        if not progressed:
            break
    return selected


def _ordered_include_result(
    tasks_by_sub_benchmark: Dict[str, List[Dict[str, Any]]],
    include_keys: Iterable[str],
) -> List[Dict[str, Any]]:
    by_combined_id: Dict[str, Dict[str, Any]] = {}
    by_raw_id: Dict[str, Dict[str, Any]] = {}
    for tasks in tasks_by_sub_benchmark.values():
        for task in tasks:
            by_combined_id[task["id"]] = task
            by_raw_id[task["original_id"]] = task

    ordered: List[Dict[str, Any]] = []
    seen = set()
    for key in include_keys:
        task = by_combined_id.get(key)
        if task is None:
            _, raw_key = _split_key(key)
            task = by_raw_id.get(raw_key)
        if task is not None and task["id"] not in seen:
            ordered.append(task)
            seen.add(task["id"])
    return ordered


def load_naturalplan_tasks(
    benchmark_dir: str,
    max_tasks: int = 200,
    seed: int = 42,
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load mixed NaturalPlan tasks from trip, calendar, and meeting planning.

    If `include_keys` is provided, keys may be either prefixed combined keys such
    as `trip::trip_planning_example_1` or raw legacy keys. Without include keys,
    the loader samples tasks in round-robin order across the three sub-tasks so
    small smoke runs still exercise mixed routing.
    """
    include_by_sub = _group_keys(include_keys)
    exclude_by_sub = _group_keys(exclude_keys) or {name: [] for name in SUB_BENCHMARKS}

    tasks_by_sub_benchmark: Dict[str, List[Dict[str, Any]]] = {}
    for offset, sub_benchmark in enumerate(SUB_BENCHMARKS):
        sub_include = include_by_sub[sub_benchmark] if include_by_sub is not None else None
        sub_exclude = exclude_by_sub[sub_benchmark]
        sub_max = max_tasks if include_keys is None else max(1, len(sub_include or []))
        tasks_by_sub_benchmark[sub_benchmark] = _load_subtasks(
            benchmark_dir,
            sub_benchmark,
            max_tasks=sub_max,
            seed=seed + offset,
            include_keys=sub_include,
            exclude_keys=sub_exclude,
        )

    if include_keys is not None:
        return _ordered_include_result(tasks_by_sub_benchmark, include_keys)

    rng = random.Random(seed)
    for tasks in tasks_by_sub_benchmark.values():
        rng.shuffle(tasks)
    return _round_robin(tasks_by_sub_benchmark, max_tasks)


def extract_canonical_answer(response: str, sub_benchmark: Optional[str] = None) -> str:
    """Extract a canonical answer for a NaturalPlan response.

    The combined benchmark needs the sub-task type to choose the right parser.
    Unknown sub-task types return an empty string instead of guessing.
    """
    if sub_benchmark == "trip":
        from sero.benchmarks.trip_adapter import extract_canonical_answer as extract_trip
        return extract_trip(response)
    if sub_benchmark == "calendar":
        from sero.benchmarks.calendar_scheduling_adapter import extract_canonical_answer as extract_calendar
        return extract_calendar(response)
    if sub_benchmark == "meeting":
        from sero.benchmarks.meeting_plan_adapter import extract_canonical_answer as extract_meeting
        return extract_meeting(response)
    return ""
