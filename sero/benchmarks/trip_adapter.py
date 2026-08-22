"""
NaturalPlan Trip Planning adapter.

Data format: trip_planning.json
  Each example has:
    - num_cities: str (e.g., "3")
    - cities: city names separated by "**"
    - durations: durations separated by "**"
    - prompt_0shot: zero-shot natural language prompt
    - golden_plan: ground truth trip plan

Evaluation returns a dict with two metrics:
  - partial_score: SERO-style partial credit [0..1]
  - exact_score:   NaturalPlan-style binary exact match {0, 1}
"""

import json
import re
import os
import random
from typing import List, Dict, Any, Optional, Tuple


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_trip_planning_tasks(
    benchmark_dir: str,
    max_tasks: int = 200,
    seed: int = 42,
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load trip planning tasks from NaturalPlan dataset.

    Args:
        include_keys: If set, load ONLY these task keys (ignores max_tasks/seed shuffle).
        exclude_keys: If set, remove these keys before sampling.

    Returns list of task dicts with keys: id, prompt, eval_fn.
    eval_fn returns a dict: {"partial_score": float, "exact_score": float}
    """
    data_path = os.path.join(benchmark_dir, "natural-plan", "data", "trip_planning.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Trip planning data not found at {data_path}")

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
        cities = ex["cities"].split("**")
        durations_str = ex["durations"].split("**")
        try:
            durations = [int(d) for d in durations_str]
        except ValueError:
            continue

        prompt = ex["prompt_0shot"]
        golden = ex.get("golden_plan", "")
        constraints = list(zip(cities, durations))

        def make_eval_fn(city_durations: List[Tuple[str, int]], gold: str,
                         raw_cities: str, raw_durations: str):
            def eval_fn(response: str) -> Dict[str, float]:
                return score_trip_response(response, city_durations, gold,
                                           raw_cities, raw_durations)
            return eval_fn

        tasks.append({
            "id": key,
            "prompt": prompt,
            "eval_fn": make_eval_fn(constraints, golden,
                                    ex["cities"], ex["durations"]),
            "cities": cities,
            "durations": durations,
            "golden_plan": golden,
        })

    return tasks


# ── Noise-word blocklist for city name validation ─────────────────────────────

_NOISE_WORDS: set = {
    "your", "the", "a", "an", "my", "our", "this", "that",
    "attend", "visit", "explore", "fly", "depart", "arrive",
    "travel", "return", "start", "end", "enjoy", "day", "days",
    "morning", "afternoon", "evening", "night", "free",
    "final", "additional", "optional", "alternative", "bonus",
    "summary", "overview", "itinerary", "notes", "total",
    "flight", "direct", "connecting", "hotel", "accommodation",
    "departure", "check", "prepare", "leisure", "relaxation",
    "transition", "connection", "stopover",
    "spend", "if", "include", "continue", "activities",
    "activity", "finally", "structured", "detailed",
    "arrival", "stay", "relax", "head", "go", "begin",
    "conference", "conferences", "workshop", "workshops",
    "wedding", "weddings", "meeting", "meetings", "show",
    "shows", "event", "events", "friend", "friends",
    "relative", "relatives", "tour", "touring", "post",
}

_CITY_PHRASE_PATTERN = r"[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+){0,2}"


def _normalize_city_candidate(name: str) -> str:
    """Normalize itinerary header text into a plausible city candidate."""
    text = re.sub(r"[*_`#]", " ", name)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^]]*\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,:;.-")
    if not text:
        return ""

    leading = re.match(rf"^({_CITY_PHRASE_PATTERN})\b", text)
    if leading:
        candidate = leading.group(1).strip()
        if _is_valid_city(candidate):
            return candidate

    patterns = [
        rf"(?:Conference|Workshop|Wedding|Meeting|Meet(?:ing)?|Show|Event|Friends?|Relatives?)\s+(?:in|at)\s+({_CITY_PHRASE_PATTERN})",
        rf"(?:Explore|Visit|Tour|Stay(?:\s+in)?|Relax(?:\s+in)?|Continue\s+(?:exploring|visiting|touring)|Begin\s+exploring)\s+({_CITY_PHRASE_PATTERN})",
        rf"(?:Arrive|Arrival)\s+in\s+({_CITY_PHRASE_PATTERN})",
        rf"(?:Travel|Head|Go|Return)\s+to\s+({_CITY_PHRASE_PATTERN})",
        rf"(?:Fly|Flight)\s+(?:from\s+{_CITY_PHRASE_PATTERN}\s+)?to\s+({_CITY_PHRASE_PATTERN})",
        rf"(?:Depart(?:ure)?(?:\s+from\s+{_CITY_PHRASE_PATTERN})?\s+(?:for|to))\s+({_CITY_PHRASE_PATTERN})",
        rf"\b(?:in|to|for)\s+({_CITY_PHRASE_PATTERN})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = match.group(match.lastindex).strip()
            if _is_valid_city(candidate):
                return candidate

    for phrase in re.findall(rf"{_CITY_PHRASE_PATTERN}", text):
        words = phrase.split()
        for start in range(len(words)):
            candidate = " ".join(words[start:]).strip()
            if _is_valid_city(candidate):
                return candidate

    return text


def _is_valid_city(name: str) -> bool:
    """Return True if *name* looks like a real city (not a noise word)."""
    tokens = [token.lower() for token in re.findall(r"[A-Za-z]+", name)]
    return (
        len(name) >= 2
        and name.lower() not in _NOISE_WORDS
        and tokens != []
        and all(token not in _NOISE_WORDS for token in tokens)
        and name[0].isupper()
        and not name.isupper()          # reject all-caps like "THE"
        and not name.endswith("ing")     # reject gerunds like "Connecting"
    )


# ── Exact-match scoring (NaturalPlan original) ───────────────────────────────

def _parse_response_exact(response: str) -> List[Tuple[str, int]]:
    """Parse response using the original NaturalPlan logic (evaluate_trip_planning.py).

    Returns a list of (city, stay_days) tuples derived from flight sequences.
    """
    pattern_visit = r'\d+-\d+'
    pattern_flight = r'.*Day (\d+).*from (\w+) to (\w+)'
    pattern_days = r'European cities for (\d+) days'

    days, flights, flight_days = [], [], []
    total_days = None
    for piece in response.split('\n'):
        days_match = re.findall(pattern_days, piece)
        if days_match:
            total_days = int(days_match[0])

        visit_match = re.findall(pattern_visit, piece)
        if visit_match:
            days.append(visit_match[0])
            end_day = int(visit_match[0].split('-')[1])
            if end_day == total_days:
                break
        flight_match = re.findall(pattern_flight, piece)
        if flight_match:
            flights.append(flight_match[0])

    visit_cities: List[str] = []
    for flight_day, begin_city, end_city in flights:
        flight_days.append(int(flight_day))
        if not visit_cities:
            if _is_valid_city(begin_city):
                visit_cities.append(begin_city)
            if _is_valid_city(end_city):
                visit_cities.append(end_city)
        else:
            if _is_valid_city(end_city):
                visit_cities.append(end_city)

    if not days or not flights or not visit_cities:
        return []
    last_day = int(days[-1].split('-')[1])
    flight_days = [1] + flight_days + [last_day]
    parsed_plan: List[Tuple[str, int]] = []
    for i, visit_city in enumerate(visit_cities):
        city_stay = flight_days[i + 1] - flight_days[i] + 1
        if city_stay > 0:
            parsed_plan.append((visit_city, city_stay))

    return parsed_plan


def _exact_score(response: str, raw_cities: str, raw_durations: str) -> float:
    """Compute NaturalPlan-style exact-match binary score (0 or 1).

    Uses ONLY the flight-sequence parser (_parse_response_exact) to match
    the original NaturalPlan evaluate_trip_planning.py logic exactly.

    Strict ordered prefix matching: first mismatch → 0.
    """
    parsed_plan = _parse_response_exact(response)
    if not parsed_plan:
        return 0.0

    stays = [x for x in raw_cities.split('**') if x]
    days = [int(x) for x in raw_durations.split('**') if x]

    num_stays = min(len(stays), len(parsed_plan))
    num_match = 0
    for i in range(num_stays):
        if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
            num_match += 1
        else:
            break
    return 1.0 if num_match == len(stays) else 0.0


# ── Partial-credit scoring (SERO-style) ──────────────────────────────────────

def _extract_stay_duration(response: str, city: str) -> Optional[int]:
    """
    Try to extract how many days the city is visited from the response text.
    Looks for patterns like "Day X-Y: Visit <city>" or "<city> for N days".
    Duration = Y - X + 1 (inclusive of both start and end day).

    Uses findall to locate ALL "Day X-Y" segments, then picks the one(s)
    whose trailing text (up to the next Day header) contains the target city.
    This avoids the greedy/lazy .*? bug where a distant Day X-Y header
    could be matched instead of the correct one adjacent to the city name.
    """
    escaped = re.escape(city)

    # Strategy 1: Split response at each "Day" header (both ranges like "Day X-Y"
    # and singles like "Day N") and check which range-segment contains the city.
    # We use ALL Day headers as segment boundaries so that "Day 7: Fly to X"
    # doesn't leak into the previous range segment's text.
    boundary_pattern = re.compile(
        r"[Dd]ays?\s+(\d+)\s*(?:[–\-]\s*(\d+))?\s*\*{0,2}\s*[:\s]",
    )
    boundaries = list(boundary_pattern.finditer(response))
    for i, m in enumerate(boundaries):
        start_day = int(m.group(1))
        end_day_str = m.group(2)
        # Only consider range headers (Day X-Y), skip single-day headers
        if end_day_str is None:
            continue
        end_day = int(end_day_str)
        # The text belonging to this segment: from match end to next boundary start
        seg_start = m.end()
        seg_end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(response)
        seg_text = response[seg_start:seg_end]
        if re.search(r'\b' + escaped + r'\b', seg_text, re.IGNORECASE):
            if end_day >= start_day and (end_day - start_day + 1) <= 30:
                return end_day - start_day + 1

    # Strategy 2: "visit CityName for N days"
    for_days_pattern = rf"{escaped}\s+for\s+(\d+)\s+day"
    m = re.search(for_days_pattern, response, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Strategy 3: "N days in CityName"
    in_days_pattern = rf"(\d+)\s+days?\s+in\s+{escaped}"
    m = re.search(in_days_pattern, response, re.IGNORECASE)
    if m:
        return int(m.group(1))

    return None


def _build_day_assignment(sequence: List[Tuple[str, int]]) -> Dict[int, str]:
    """Expand (city, duration) sequence into {day_number: city} mapping."""
    assignment: Dict[int, str] = {}
    day = 1
    for city, duration in sequence:
        for d in range(duration):
            assignment[day + d] = city
        day += duration
    return assignment


def _parse_day_range_headers(
    response: str,
    golden_cities: List[str],
) -> List[Tuple[str, int]]:
    """Parse Day X-Y headers in response, match golden cities, return (city, duration) sequence."""
    boundary_pattern = re.compile(
        r"[Dd]ays?\s+(\d+)\s*[–\-]\s*(\d+)\s*\*{0,2}\s*[:\s]",
    )
    boundaries = list(boundary_pattern.finditer(response))
    if not boundaries:
        return []

    city_patterns = {
        city: re.compile(r'\b' + re.escape(city) + r'\b', re.IGNORECASE)
        for city in golden_cities
    }

    segments: List[Tuple[int, str, int]] = []
    for i, m in enumerate(boundaries):
        start_day = int(m.group(1))
        end_day = int(m.group(2))
        if end_day < start_day or (end_day - start_day + 1) > 30:
            continue
        seg_start = m.end()
        seg_end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(response)
        seg_text = response[seg_start:seg_end]

        for city, pat in city_patterns.items():
            if pat.search(seg_text):
                segments.append((start_day, city, end_day - start_day + 1))
                break

    if not segments:
        return []

    segments.sort(key=lambda x: x[0])
    return [(city, dur) for _, city, dur in segments]


def _partial_score(
    response: str,
    constraints: List[Tuple[str, int]],
) -> float:
    """Day-slot matching partial score.

    Expands golden constraints into day-level city assignments:
        day t → city c*(t)
    Parses response into a (city, duration) sequence, expands likewise.
    Score = fraction of days with correct city assignment.

    This single metric implicitly captures presence, duration, ordering,
    and absolute day positioning.  Any error in an early city cascades to
    shift all subsequent day assignments, matching real-world semantics.

        Parsing strategies (all attempted):
            1. LLM itinerary parser (_extract_city_days_from_llm)
            2. Day X-Y range headers matched against golden cities
            3. Flight-sequence parser (_parse_response_exact)
            4. Per-city duration extraction ordered by text position

        The final partial score uses the best non-empty parsed sequence among
        these strategies, measured by day-slot matching against the golden plan.
    """
    if not response.strip():
        return 0.0

    n = len(constraints)
    if n == 0:
        return 0.0

    total_days = sum(d for _, d in constraints)
    if total_days == 0:
        return 0.0

    # Golden day-level assignment
    golden_assign = _build_day_assignment(constraints)

    golden_cities = [c for c, _ in constraints]

    candidates: List[List[Tuple[str, int]]] = []

    # Strategy 1: LLM itinerary parser
    parsed = _extract_city_days_from_llm(response)
    if parsed:
        candidates.append(parsed)

    # Strategy 2: Day X-Y range headers
    parsed = _parse_day_range_headers(response, golden_cities)
    if parsed:
        candidates.append(parsed)

    # Strategy 3: flight-sequence parser
    parsed = _parse_response_exact(response)
    if parsed:
        candidates.append(parsed)

    # Strategy 4: per-city duration extraction, ordered by text position
    city_info: List[Tuple[int, str, int]] = []
    for city, _ in constraints:
        m = re.search(r'\b' + re.escape(city) + r'\b', response, re.IGNORECASE)
        if m:
            dur = _extract_stay_duration(response, city)
            if dur is not None:
                city_info.append((m.start(), city, dur))
    city_info.sort(key=lambda x: x[0])
    if city_info:
        candidates.append([(city, dur) for _, city, dur in city_info])

    if not candidates:
        return 0.0

    best_score = 0.0
    for parsed in candidates:
        parsed_assign = _build_day_assignment(parsed)
        correct = sum(
            1 for t in range(1, total_days + 1)
            if parsed_assign.get(t) == golden_assign.get(t)
        )
        best_score = max(best_score, correct / total_days)

    return best_score


# ── Combined scoring entry point ─────────────────────────────────────────────

def score_trip_response(
    response: str,
    constraints: List[Tuple[str, int]],
    golden_plan: str,
    raw_cities: str,
    raw_durations: str,
) -> Dict[str, float]:
    """
    Score a trip planning response with two metrics.

    Returns:
        {"partial_score": float in [0,1], "exact_score": float in {0,1}}
    """
    if not response.strip():
        return {"partial_score": 0.0, "exact_score": 0.0}

    return {
        "partial_score": _partial_score(response, constraints),
        "exact_score": _exact_score(response, raw_cities, raw_durations),
    }


# ── SC Canonical Answer Extraction ────────────────────────────────────────────

def _extract_city_days_from_llm(response: str) -> List[Tuple[str, int]]:
    """Parse (city, days) from typical LLM itinerary output formats.

    Handles patterns commonly found in LLM-generated trip plans:
      - "**Day 1-4: Milan**" / "Day 1-4: Milan, Italy"
      - "**Days 1-6: Reykjavik, Iceland**"
      - "- **Milan:** Days 1-4" / "- **Bucharest:** 4 days (Day 1-4)"
      - "**Milan**: 4 days"

    Merges consecutive entries for the same city.
    """
    results: List[Tuple[str, int]] = []

    # Pattern A: "Day(s) X-Y: CityName" — handles markdown bold (**) around
    # headers in various positions:
    #   "Day 1-4: Milan"  /  "**Day 1-4: Milan**"  /  "**Days 1-6**: Munich"
    #   Also matches: "Day 4-6: Fly from Brussels to Krakow" → extracts Krakow
    #   Also matches: "Day 14-18: Visit relatives in Oslo" → extracts Oslo
    #   Also matches: "Day 5-8: Attend annual show in Split" → extracts Split
    pat_a = re.finditer(
        r"[Dd]ays?\s+(\d+)\s*[–\-]\s*(\d+)\s*\*{0,2}\s*[:\s]\s*\*{0,2}\s*"
        r"(?:(?:Visit|Arriving?\s+in|Stay\s+in|Explore|Attend|Meet|Spend(?:\s+time)?)\b[^,\n]*?\bin\s+)?"
        r"(?:(?:Visit|Stay\s+in|Explore)\s+)?"
        r"(?:Fly\s+from\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s+to\s+)?"
        r"([A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+)?)",
        response,
    )
    seen_ranges: set[Tuple[int, int, str]] = set()
    for m in pat_a:
        start, end = int(m.group(1)), int(m.group(2))
        city = _normalize_city_candidate(m.group(3).strip().rstrip(",").strip())
        key = (start, end, city.lower())
        if (
            _is_valid_city(city)
            and end >= start
            and (end - start + 1) <= 30
            and key not in seen_ranges
        ):
            seen_ranges.add(key)
            results.append((city, end - start + 1))

    if results:
        return _merge_city_days(results)

    # Pattern B: "**CityName:** Days X-Y" or "**CityName:** N days"
    # Note: colon may be inside or outside the bold markers (**)
    pat_b = re.finditer(
        r"\*{1,2}([A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+)?):?\*{1,2}\s*:?\s*"
        r"(?:[Dd]ays?\s+(\d+)\s*[–\-]\s*(\d+)|(\d+)\s+days?)",
        response,
    )
    for m in pat_b:
        city = _normalize_city_candidate(m.group(1).strip().rstrip(",").strip())
        if not _is_valid_city(city):
            continue
        if m.group(2) and m.group(3):
            start, end = int(m.group(2)), int(m.group(3))
            days = end - start + 1
        elif m.group(4):
            days = int(m.group(4))
        else:
            continue
        if 0 < days <= 30:
            results.append((city, days))

    if results:
        return _merge_city_days(results)

    # Pattern C: Single-day headers like "### Day 7: Frankfurt" or "Day 7: Frankfurt"
    # Also handles: "Day 7: Fly from X to Frankfurt"
    # These appear when LLMs list day-by-day without day-range headers.
    pat_c = re.finditer(
        r"#{0,4}\s*[Dd]ay\s+\d+\s*\*{0,2}\s*[:\s]\s*\*{0,2}\s*([^\n]+)",
        response,
    )
    for m in pat_c:
        city = _normalize_city_candidate(m.group(1).strip())
        if _is_valid_city(city):
            results.append((city, 1))  # each single-day entry = 1 day

    if results:
        return _merge_city_days(results)

    return []


def _merge_city_days(entries: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Merge consecutive entries for the same city, deduplicate repeated sequences,
    and consolidate non-adjacent duplicates.

    LLM outputs often contain both an "Overview" and "Detailed Itinerary" section
    that produce duplicate city sequences.  If the merged list is an exact repeat
    of its first half, return only the first half.
    After structural dedup, also consolidate any remaining non-adjacent duplicates
    (keep first occurrence, use max days across occurrences).
    """
    if not entries:
        return []
    merged: List[Tuple[str, int]] = [entries[0]]
    for city, days in entries[1:]:
        if city == merged[-1][0]:
            merged[-1] = (city, merged[-1][1] + days)
        else:
            merged.append((city, days))

    # Deduplicate: if first half == second half, keep first half only
    n = len(merged)
    if n >= 2 and n % 2 == 0:
        half = n // 2
        if merged[:half] == merged[half:]:
            merged = merged[:half]

    # Consolidate non-adjacent duplicates: keep first occurrence, use max days
    seen: Dict[str, int] = {}   # city -> index in result
    consolidated: List[Tuple[str, int]] = []
    for city, days in merged:
        if city in seen:
            idx = seen[city]
            old_city, old_days = consolidated[idx]
            consolidated[idx] = (old_city, max(old_days, days))
        else:
            seen[city] = len(consolidated)
            consolidated.append((city, days))

    return consolidated


def extract_canonical_answer(response: str) -> str:
    """Extract a canonical trip plan string for SC majority voting.

    Tries multiple strategies in order:
    1. LLM itinerary parser (handles Day X-Y: City, City: N days, etc.)
       — preferred because it reads the actual visit lines the LLM wrote.
    2. NaturalPlan flight-sequence parser (works on golden-plan-like format)
       — fallback for golden-plan format where no explicit visit lines exist.
    3. Empty string if nothing found.
    """
    def _postprocess(entries: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Filter invalid entries + deduplicate non-adjacent cities."""
        clean = [(c, d) for c, d in entries
                 if 0 < d <= 20 and _is_valid_city(c)]
        # Consolidate non-adjacent duplicates: keep first, use max days
        seen: Dict[str, int] = {}
        result: List[Tuple[str, int]] = []
        for city, days in clean:
            if city in seen:
                idx = seen[city]
                result[idx] = (result[idx][0], max(result[idx][1], days))
            else:
                seen[city] = len(result)
                result.append((city, days))
        return result

    # Strategy 1: LLM output parser (robust to various formats)
    # Preferred — reads "Day X-Y: Visit CityName" lines directly.
    parsed = _extract_city_days_from_llm(response)
    if parsed:
        parsed = _postprocess(parsed)
        if parsed:
            return "|".join(f"{city}:{days}" for city, days in parsed)

    # Strategy 2: Original flight-sequence parser (golden-plan format fallback)
    # Only trust it when it finds ≥2 cities — single-city results are almost
    # always artefacts of greedy regex on single-line comma-separated input.
    parsed = _parse_response_exact(response)
    if parsed and len(parsed) >= 2:
        parsed = _postprocess(parsed)
        if parsed:
            return "|".join(f"{city}:{days}" for city, days in parsed)

    return ""


# ── Constraint Parsing Helper ─────────────────────────────────────────────────

def parse_golden_plan_constraints(golden_plan: str) -> List[Tuple[str, int]]:
    """
    Extract (city, days) pairs from a golden plan string.
    Used for debugging / verification.
    """
    constraints = []
    for m in re.finditer(r"[Dd]ay\s+(\d+)[–\-](\d+)[:\s].*?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", golden_plan):
        start, end = int(m.group(1)), int(m.group(2))
        city = m.group(3).strip()
        days = end - start + 1
        if days > 0:
            constraints.append((city, days))
    return constraints
