"""
Static baseline: benchmark-specific fixed-topology DAG with dedicated roles.

The simplest non-trivial multi-agent baseline with a *structured* topology:
  - Fixed role pool per benchmark (dedicated agents, not the shared seed pool)
  - Fixed DAG topology per benchmark (hand-crafted, non-linear)
  - No credit scoring, no role retrieval, no evolution

Each benchmark has a unique topology reflecting its problem structure:
    naturalplan — task/contract parser → trip/calendar/meeting specialist branches → audit → final format
    tablebench — contract/schema → evidence → parallel numeric/lookup-analysis branches → arithmetic audit → audit-synthesis → final answer
  trip      — parallel constraint / flight-graph extraction → route solving → auditing
  meeting   — parallel time-window / distance analysis → greedy scheduling → verification
  calendar  — parallel per-person parsing / preference extraction → 2×2 bipartite → decision
    olympiadbench — concept analysis → parallel dual-strategy solving → cross-checking

Compared to other baselines:
  Workflow   — expert-designed LINEAR chain, NaturalPlan only
  Static-DAG — uses credit mechanism + DAG ranking, no evolution
  Static     — NO credit, NO ranking, benchmark-specific fixed DAG   ← this file
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

import numpy as np

from sero.benchmarks.scoring_utils import normalize_score, is_dual_score
from sero.role_card import RoleCard
from sero.openrouter_client import OpenRouterClient
from sero.config import SeroConfig

logger = logging.getLogger(__name__)

_EVAL_WORKERS = 5
_MAX_PARALLEL_AGENTS = 4


# ═══════════════════════════════════════════════════════════════════════════════
# Per-benchmark DAG definitions
#
# Each factory returns (roles_dict, edges, terminal_name):
#   roles_dict : Dict[str, RoleCard]  — name → role
#   edges      : List[(src, dst)]     — directed edges (message flow)
#   terminal   : str                  — name of the terminal node
# ═══════════════════════════════════════════════════════════════════════════════

_DagSpec = Tuple[Dict[str, RoleCard], List[Tuple[str, str]], str]


def _trip_planning_dag() -> _DagSpec:
    """
    Trip planning DAG — constraint satisfaction + Hamiltonian path.

    Topology (4 levels):
        L1: [Constraint Extractor] ∥ [Flight Graph Analyzer]   (parallel)
        L2: [Route Solver]            (needs both L1 outputs)
        L3: [Constraint Auditor]      (verifies route; skip-edge from CE)
        L4: [Plan Formatter]          (terminal)
    """
    roles = {
        "Constraint Extractor": RoleCard(
            name="Constraint Extractor",
            system_prompt=(
                "You are a travel constraint extraction expert. From the problem description, "
                "extract ALL constraints into a structured list:\n"
                "1. Each city and its required stay duration (in days)\n"
                "2. Time-window constraints (e.g. 'visit X between day A and day B')\n"
                "3. The total trip duration\n"
                "Output a clear numbered list of every constraint."
            ),
            capability_tags=["extraction", "constraints", "parsing"],
        ),
        "Flight Graph Analyzer": RoleCard(
            name="Flight Graph Analyzer",
            system_prompt=(
                "You are a flight connectivity expert. From the problem description, "
                "extract the complete directed flight graph:\n"
                "1. List every direct flight connection (city A → city B)\n"
                "2. Identify which city pairs have NO direct flights\n"
                "3. Determine possible starting cities and ending cities\n"
                "4. Note any one-way-only connections\n"
                "Output the adjacency list and key connectivity observations."
            ),
            capability_tags=["graph", "connectivity", "flights"],
        ),
        "Route Solver": RoleCard(
            name="Route Solver",
            system_prompt=(
                "You are a route planning solver. Given the extracted constraints and flight "
                "graph, find a valid city ordering that satisfies ALL constraints:\n"
                "1. Each consecutive pair must have a direct flight\n"
                "2. Each city gets exactly its required number of days\n"
                "3. Time-window constraints are respected\n"
                "4. Total days must sum to EXACTLY the trip duration\n\n"
                "CRITICAL day-counting rule: the flight day is the LAST day of the current "
                "city AND the FIRST day of the next city (they overlap). Example for a "
                "10-day trip visiting A(4 days) then B(3 days) then C(4 days):\n"
                "  Day 1-4: A (4 days). Fly on Day 4. Day 4-6: B (3 days). Fly on Day 6. "
                "Day 6-9: C (4 days). Total = 4+3+4 - 2 overlaps = 9 != 10, so adjust.\n"
                "  Sum of all city days - (number_of_cities - 1) = total trip days.\n"
                "Show your reasoning step by step. Output the complete day-by-day itinerary."
            ),
            capability_tags=["solving", "routing", "scheduling"],
        ),
        "Constraint Auditor": RoleCard(
            name="Constraint Auditor",
            system_prompt=(
                "You are a strict constraint auditor. Given the original constraints and the "
                "proposed route, verify EVERY constraint:\n"
                "1. Check each city's stay duration matches the requirement\n"
                "2. Check every flight connection exists in the flight graph\n"
                "3. Check time-window constraints are satisfied\n"
                "4. CRITICAL: total days = sum_of_city_days - (num_cities - 1). "
                "Flight days overlap: the last day in city A is also the first day in city B.\n"
                "If ANY violation is found, explain it and provide the corrected itinerary "
                "preserving EVERY city with its required days (do NOT drop any city). "
                "If all constraints pass, confirm the itinerary is valid."
            ),
            capability_tags=["verification", "auditing", "constraints"],
        ),
        "Plan Formatter": RoleCard(
            name="Plan Formatter",
            system_prompt=(
                "You are the final trip plan formatter. Given the audited itinerary, "
                "produce the plan in EXACTLY this format.\n\n"
                "CRITICAL RULE: The flight day is SHARED. When you fly on Day X, "
                "the visit range for the NEXT city must START at Day X (not Day X+1).\n\n"
                "CORRECT example (Milan 4d, Paris 3d, Nice 3d = 9 days total):\n"
                "Arriving in Milan\n"
                "Day 1-4: Visit Milan for 4 days.\n"
                "Day 4: Fly from Milan to Paris.\n"
                "Day 4-6: Visit Paris for 3 days.\n"
                "Day 6: Fly from Paris to Nice.\n"
                "Day 6-8: Visit Nice for 3 days.\n\n"
                "WRONG (DO NOT DO THIS — next city starts at Day X+1):\n"
                "Day 1-4: Visit Milan for 4 days.\n"
                "Day 4: Fly from Milan to Paris.\n"
                "Day 5-7: Visit Paris for 3 days.  <-- WRONG! Must be Day 4-6\n\n"
                "Rules:\n"
                "- The visit range after a flight on Day X MUST start at Day X.\n"
                "- Total: sum_of_city_days - (num_cities - 1) = trip duration.\n"
                "- The last day number must equal the total trip duration.\n"
                "- Include ALL cities. Do NOT drop any city."
            ),
            capability_tags=["formatting", "synthesis"],
            communication_protocol="produce the final plan in the exact Day X-Y format",
            protected=True,
        ),
    }
    edges = [
        ("Constraint Extractor", "Route Solver"),
        ("Flight Graph Analyzer", "Route Solver"),
        ("Route Solver", "Constraint Auditor"),
        ("Constraint Extractor", "Constraint Auditor"),  # skip-edge
        ("Constraint Auditor", "Plan Formatter"),
    ]
    return roles, edges, "Plan Formatter"


def _meeting_planning_dag() -> _DagSpec:
    """
    Meeting planning DAG — TSPTW (traveling salesman with time windows).

    Topology (4 levels):
        L1: [Window Analyzer] ∥ [Distance Calculator]   (parallel)
        L2: [Greedy Scheduler]     (needs both L1 outputs)
        L3: [Time Verifier]        (skip-edges from WA + DC for verification)
        L4: [Format Synthesizer]   (terminal)
    """
    roles = {
        "Window Analyzer": RoleCard(
            name="Window Analyzer",
            system_prompt=(
                "You are a time-window analysis expert. Be CONCISE. For each person:\n"
                "1. List: Person | Location | Window (start-end) | Min Duration | Slack\n"
                "2. Sort by slack (tightest first)\n"
                "3. Flag conflicting pairs that cannot both be met in sequence\n"
                "Output ONLY the structured table and conflicts. No prose."
            ),
            capability_tags=["analysis", "time-windows", "constraints"],
        ),
        "Distance Calculator": RoleCard(
            name="Distance Calculator",
            system_prompt=(
                "You are a travel distance expert. Be CONCISE.\n"
                "1. List ONLY the travel times between the starting location and "
                "each meeting location, and between each pair of meeting locations\n"
                "2. Note the starting location\n"
                "Do NOT repeat the full distance matrix from the problem. "
                "Output a compact summary of relevant distances only.\n\n"
                "IMPORTANT: Do NOT propose a schedule or recommend meeting order. "
                "Your job is ONLY to provide distances. The Scheduler agent will "
                "use these distances to build the schedule."
            ),
            capability_tags=["distances", "routing", "travel-time"],
        ),
        "Greedy Scheduler": RoleCard(
            name="Greedy Scheduler",
            system_prompt=(
                "You are a meeting schedule optimizer. Given time-window analysis and distances, "
                "construct a feasible schedule that meets AS MANY people as possible.\n\n"
                "CRITICAL RULES:\n"
                "1. Start from the given starting location and time\n"
                "2. You may ONLY travel to a location to meet someone there. "
                "Do NOT travel to locations where you have no meeting.\n"
                "3. For each step: compute arrival_time = depart_time + travel_minutes\n"
                "4. **A meeting can ONLY start when the person is available.** "
                "If arrival_time < person's window_start, you MUST wait until window_start. "
                "meeting_start = max(arrival_time, window_start).\n"
                "5. meeting_end = meeting_start + min_duration. "
                "You MUST verify: meeting_end <= window_end. If not, this person cannot be met.\n"
                "6. Prefer people with tight windows first. Maximize total meetings.\n\n"
                "WRONG EXAMPLE: If Kevin is available 1:30PM-5:00PM and you arrive at 10:48AM, "
                "you MUST wait until 1:30PM. You CANNOT meet Kevin at 10:48AM!\n\n"
                "For each step output: travel(dest, minutes, arrive_time), "
                "wait(until_time) if needed, meet(person, duration, start-end). Be CONCISE."
            ),
            capability_tags=["scheduling", "optimization", "greedy"],
        ),
        "Time Verifier": RoleCard(
            name="Time Verifier",
            system_prompt=(
                "You are a strict schedule time verifier. You MUST check the proposed schedule "
                "against the ORIGINAL problem constraints (not the scheduler's claims).\n\n"
                "For EACH meeting step, verify ALL of these:\n"
                "1. Travel time: look up the EXACT value in the distance matrix. "
                "If stated travel time differs, use the matrix value and recompute arrival.\n"
                "2. Arrival time = previous_end_time + travel_time_from_matrix\n"
                "3. **WINDOW CHECK (MOST IMPORTANT)**: Is the person actually available? "
                "Read their window from the ORIGINAL constraints. "
                "meeting_start MUST be >= person's window_start. "
                "If arrival < window_start, a WAIT step is required.\n"
                "4. meeting_end = meeting_start + duration. meeting_end MUST be <= window_end.\n"
                "5. Duration >= minimum required.\n\n"
                "COMMON ERROR: The scheduler may place meetings BEFORE the person is available. "
                "For example, meeting Joshua at 9:15AM when he's available 6:15PM-7:15PM is INVALID. "
                "You MUST catch and fix such errors by removing the invalid meeting "
                "or rescheduling it to the correct time window.\n\n"
                "Output the corrected step-by-step schedule. Be CONCISE."
            ),
            capability_tags=["verification", "time-checking", "arithmetic"],
        ),
        "Format Synthesizer": RoleCard(
            name="Format Synthesizer",
            system_prompt=(
                "You are the final meeting plan formatter. Given the verified schedule, "
                "produce the plan using EXACTLY these step formats (one per line):\n"
                "'You start at [Location] at [Time].'\n"
                "'You travel to [Location] in [N] minutes and arrive at [HH:MMAM/PM].'\n"
                "'You wait until [HH:MMAM/PM].'\n"
                "'You meet [Person] for [N] minutes from [HH:MMAM/PM] to [HH:MMAM/PM].'\n\n"
                "Rules:\n"
                "- Use 12-hour time with AM/PM, no spaces (e.g. 9:07AM, 1:30PM)\n"
                "- Travel time N must match the distance matrix exactly\n"
                "- If arrival_time < meeting_start_time, you MUST insert a 'You wait until' step. "
                "This is essential — do NOT skip it!\n"
                "- Do NOT include travel to locations where no meeting happens\n"
                "- Output ONLY the formatted steps, no commentary"
            ),
            capability_tags=["formatting", "synthesis"],
            communication_protocol=(
                "produce the schedule using ONLY these formats: "
                "'You travel to [Location] in [N] minutes and arrive at [HH:MMAM/PM].' "
                "'You meet [Person] for [N] minutes from [HH:MMAM/PM] to [HH:MMAM/PM].'"
            ),
            protected=True,
        ),
    }
    edges = [
        ("Window Analyzer", "Greedy Scheduler"),
        ("Distance Calculator", "Greedy Scheduler"),
        ("Greedy Scheduler", "Time Verifier"),
        ("Window Analyzer", "Time Verifier"),      # skip-edge
        ("Distance Calculator", "Time Verifier"),  # skip-edge
        ("Time Verifier", "Format Synthesizer"),
    ]
    return roles, edges, "Format Synthesizer"


def _calendar_scheduling_dag() -> _DagSpec:
    """
    Calendar scheduling DAG — interval intersection + preference filtering.

    Topology (3 levels, 2×2 bipartite middle):
        L1: [Per-Person Parser] ∥ [Preference Extractor]   (parallel)
        L2: [Slot Finder] ∥ [Conflict Checker]             (parallel, both need L1)
        L3: [Decision Maker]   (terminal)
    """
    roles = {
        "Per-Person Parser": RoleCard(
            name="Per-Person Parser",
            system_prompt=(
                "You are a calendar parsing expert. Your ONLY job is to parse each person's schedule.\n\n"
                "For each participant:\n"
                "1. List their name and ALL existing appointments (day, start time, end time)\n"
                "2. Compute their FREE time slots within working hours (9:00-17:00)\n"
                "3. Present each person's free slots as a clear table\n\n"
                "BOUNDARY RULE (CRITICAL):\n"
                "- If someone is busy 9:00-11:00, they become FREE at exactly 11:00.\n"
                "  So 11:00-11:30 IS a free slot. The busy period ENDS at 11:00.\n"
                "- If someone is busy 13:00-14:00, they become FREE at exactly 14:00.\n"
                "- Always list the free gap BETWEEN consecutive busy periods.\n\n"
                "OTHER RULES:\n"
                "- Use 24-hour format (e.g. 13:00, not 1:00 PM).\n"
                "- Do NOT compute overlaps between people.\n"
                "- Do NOT recommend any meeting time.\n"
                "- ONLY output each person's individual free slots."
            ),
            capability_tags=["parsing", "availability", "calendar"],
        ),
        "Preference Extractor": RoleCard(
            name="Preference Extractor",
            system_prompt=(
                "You are a meeting preference expert. Your ONLY job is to extract constraints.\n\n"
                "ALL preferences in this task are HARD CONSTRAINTS (not soft). "
                "Any slot that violates a preference MUST be excluded entirely.\n\n"
                "From the problem description:\n"
                "1. Extract the required meeting duration (e.g., 30 min or 60 min)\n"
                "2. List the days being considered\n"
                "3. Identify ALL constraints. PAY CLOSE ATTENTION to sentences like:\n"
                "   'X would like to avoid more meetings on Monday after 14:30. Tuesday. Wednesday.'\n"
                "   This means X wants to avoid: (a) Monday after 14:30, (b) ALL of Tuesday, (c) ALL of Wednesday.\n"
                "   The periods separate DISTINCT day constraints.\n"
                "4. 'X would like to meet at their earliest availability' = pick the earliest valid slot.\n"
                "5. 'X would rather not meet on Day after HH:MM' = EXCLUDE all slots starting at or after HH:MM on Day.\n\n"
                "Output format: a numbered list of HARD CONSTRAINTS, each stating who, what day(s), "
                "and what time range is excluded. Do NOT recommend any meeting time."
            ),
            capability_tags=["extraction", "preferences", "constraints"],
        ),
        "Slot Finder": RoleCard(
            name="Slot Finder",
            system_prompt=(
                "You are a time-slot intersection expert. Your ONLY job is to find ALL valid candidate slots.\n\n"
                "Step-by-step process:\n"
                "1. List each person's free slots (from Per-Person Parser).\n"
                "2. For each day, find ALL time ranges where EVERY participant is simultaneously free.\n"
                "3. From each overlapping range, extract slots of the required meeting duration.\n"
                "4. Sort by day, then earliest start time.\n\n"
                "BOUNDARY RULES (CRITICAL — most common source of errors):\n"
                "- Busy 9:00-11:00 means FREE from 11:00 onward. 11:00 is the START of free time.\n"
                "- If A is free 11:00-12:00 and B is free 10:30-12:30, overlap = 11:00-12:00.\n"
                "- overlap_start = max(A_start, B_start); overlap_end = min(A_end, B_end)\n"
                "- If overlap_end - overlap_start >= duration, it is VALID.\n\n"
                "EXHAUSTIVENESS: Compare EVERY pair of free windows across ALL participants.\n"
                "Do NOT skip any valid overlap. Do NOT recommend a best slot."
            ),
            capability_tags=["intersection", "scheduling", "time-slots"],
        ),
        "Conflict Checker": RoleCard(
            name="Conflict Checker",
            system_prompt=(
                "You are a scheduling conflict detector. Your ONLY job is to validate and filter.\n\n"
                "ALL preferences are HARD CONSTRAINTS. Any slot violating a preference is INVALID.\n\n"
                "For each candidate time slot:\n"
                "1. Check EACH participant against their ORIGINAL busy schedule from the task.\n"
                "   Boundary rule: busy 9:00-11:00 means free AT 11:00. So 11:00-11:30 has NO conflict.\n"
                "2. If ANY participant has a conflicting appointment → INVALID.\n"
                "3. Then check against ALL preference constraints:\n"
                "   - 'avoid Monday after 14:30' → any slot starting >= 14:30 on Monday is INVALID.\n"
                "   - 'avoid Tuesday' → ALL Tuesday slots are INVALID.\n"
                "4. Slots passing BOTH checks → VALID.\n\n"
                "Output a table: Day | Time Slot | Status (VALID or INVALID) | Reason if invalid.\n"
                "Do NOT recommend or pick a best slot."
            ),
            capability_tags=["validation", "conflicts", "preferences"],
        ),
        "Decision Maker": RoleCard(
            name="Decision Maker",
            system_prompt=(
                "You are the final scheduling decision maker.\n\n"
                "You receive candidate slots from Slot Finder and validation from Conflict Checker.\n"
                "IMPORTANT: The upstream agents may contain errors. You MUST independently verify.\n\n"
                "Your process:\n"
                "1. Collect all slots marked VALID by Conflict Checker.\n"
                "2. Also check: did Slot Finder list any slots that Conflict Checker missed?\n"
                "   If so, verify those slots yourself against the original schedules.\n"
                "3. Apply ALL preference constraints as HARD exclusions:\n"
                "   - 'avoid day X' → exclude ALL slots on day X\n"
                "   - 'avoid day X after HH:MM' → exclude slots starting >= HH:MM on day X\n"
                "4. From remaining VALID slots:\n"
                "   - If 'earliest availability' preference → pick earliest\n"
                "   - Otherwise → pick the LAST (latest) valid slot that satisfies all constraints\n\n"
                "Output EXACTLY one line: 'Here is the proposed time: <Day>, HH:MM - HH:MM'\n"
                "using 24-hour format. Output ONLY this one line, nothing else."
            ),
            capability_tags=["decision", "synthesis", "formatting"],
            communication_protocol="output ONLY 'Here is the proposed time: <Day>, HH:MM - HH:MM'",
            protected=True,
        ),
    }
    edges = [
        ("Per-Person Parser", "Slot Finder"),
        ("Per-Person Parser", "Conflict Checker"),
        ("Preference Extractor", "Slot Finder"),
        ("Preference Extractor", "Conflict Checker"),
        ("Slot Finder", "Decision Maker"),
        ("Conflict Checker", "Decision Maker"),
    ]
    return roles, edges, "Decision Maker"
def _olympiad_dag() -> _DagSpec:
    """
    Olympiad math/physics DAG — dual-strategy solving + cross-checking.

    Topology (4 levels):
        L1: [Concept Identifier]                          (single analysis)
        L2: [Algebraic Solver] ∥ [Intuitive Solver]       (parallel, diverse)
        L3: [Cross-Checker]                                (compares both)
        L4: [Answer Extractor]  (terminal)
    """
    roles = {
        "Concept Identifier": RoleCard(
            name="Concept Identifier",
            system_prompt=(
                "You are an olympiad concept analysis expert. For the given problem:\n"
                "1. Identify the subject and the exact target quantity / answer type\n"
                "2. List only the relevant theorems, lemmas, formulas, or provided context results that are directly useful\n"
                "3. Identify key variables, invariants, conserved quantities, and what must be solved for\n"
                "4. Note special conditions or edge cases: domain restrictions, threshold terms, equality conditions, required constructions, frame/sign conventions, and whether the target is a signed scalar or a magnitude\n"
                "5. For least-number or extremal problems, identify the lower-bound certificate before proposing a construction\n"
                "Output a clear decomposition with applicable techniques. Do NOT solve the problem."
            ),
            capability_tags=["analysis", "decomposition", "math", "physics"],
        ),
        "Algebraic Solver": RoleCard(
            name="Algebraic Solver",
            system_prompt=(
                "You are a rigorous olympiad solver. Given the problem analysis, solve the problem "
                "using a formal derivation:\n"
                "1. Start from the explicit givens and any provided context results; do NOT introduce unrelated formulas\n"
                "2. Show the critical intermediate derivation steps needed to justify the answer\n"
                "3. For logarithmic or exponential equations, derive an equivalent equation first and check every candidate root in the ORIGINAL equation\n"
                "4. For least-number or extremal problems, prove a lower/upper bound and then give the construction or equality case that attains it\n"
                "5. For trigonometric, periodic, or multi-answer problems, report ALL solutions/angles in the required domain or one full cycle, not just one representative solution\n"
                "6. For physics problems, define the frame/sign convention before substituting, keep threshold or additive terms, and preserve geometric factors such as projected area and pi when they arise from the derivation\n"
                "7. Prefer an exact symbolic answer; do NOT switch to decimals unless the task explicitly asks for them\n"
                "Conclude with a clear final candidate answer."
            ),
            capability_tags=["algebra", "formal", "calculation"],
        ),
        "Intuitive Solver": RoleCard(
            name="Intuitive Solver",
            system_prompt=(
                "You are a creative but disciplined olympiad solver. Given the problem analysis, "
                "try a genuinely different approach:\n"
                "1. Look for patterns, symmetries, invariants, special cases, bounding arguments, or dimensional reasoning\n"
                "2. Keep the alternative path tied to the original task; do NOT hand-wave or rely on vague plausibility\n"
                "3. For physics, use dimensional analysis and sanity checks only as support, not as a substitute for the decisive derivation\n"
                "4. For trigonometric, periodic, or multi-answer problems, verify that the answer set is complete over the required domain or one full cycle\n"
                "5. Verify the final candidate against the original equation, constraints, units, or frame/sign convention\n"
                "6. Prefer an exact symbolic answer when possible\n"
                "Conclude with your answer."
            ),
            capability_tags=["intuition", "creative", "estimation"],
            temperature=0.4,
        ),
        "Cross-Checker": RoleCard(
            name="Cross-Checker",
            system_prompt=(
                "You are a strict olympiad cross-checker. You receive a concept analysis and TWO independent solutions. Your job:\n"
                "1. Extract each solver's final candidate answer explicitly\n"
                "2. Even if both solvers agree, do NOT trust agreement alone; verify by substituting into the original equation/constraints or by re-deriving the decisive step\n"
                "3. If the candidates disagree, identify the first invalid step and explain why\n"
                "4. For logarithmic or exponential equations, test candidates in the ORIGINAL equation\n"
                "5. For least-number or extremal problems, check both the bound and the attainment/construction\n"
                "6. For trigonometric, periodic, or multi-answer problems, verify completeness over the relevant domain; include every angle/solution that achieves the same extremal value or satisfies the equation\n"
                "7. For physics, check frame/sign conventions, threshold/additive terms, dimensional consistency, and geometric factors such as projected area and pi\n"
                "8. Determine one correct final answer in exact form when possible\n"
                "Show your verification work. End with exactly one line: Final verified answer: <answer>"
            ),
            capability_tags=["verification", "cross-checking", "error-detection"],
        ),
        "Answer Extractor": RoleCard(
            name="Answer Extractor",
            system_prompt=(
                "You are the final answer formatter for mathematical and scientific problems. "
                "Use the Cross-Checker's final line 'Final verified answer: <answer>' as the source of truth.\n"
                "1. State the answer in \\boxed{<answer>} format\n"
                "2. Preserve exact symbolic form; do NOT convert to decimals unless explicitly requested\n"
                "3. Do NOT drop geometric factors such as pi, powers, or units that are part of the verified answer\n"
                "4. If the problem involves units, keep the value/expression inside \\boxed{} and place the unit outside the box\n"
                "5. For multiple answers, separate with commas inside \\boxed{}\n"
                "Output ONLY the boxed answer."
            ),
            capability_tags=["formatting", "extraction"],
            communication_protocol="provide the final answer in \\boxed{<answer>} format",
            protected=True,
        ),
    }
    edges = [
        ("Concept Identifier", "Algebraic Solver"),
        ("Concept Identifier", "Intuitive Solver"),
        ("Concept Identifier", "Cross-Checker"),
        ("Algebraic Solver", "Cross-Checker"),
        ("Intuitive Solver", "Cross-Checker"),
        ("Cross-Checker", "Answer Extractor"),
    ]
    return roles, edges, "Answer Extractor"


def _jssp_dag() -> _DagSpec:
    """
    Job-shop scheduling DAG — optimization with conflict repair.

    Topology (4 levels):
        L1: [Bottleneck Analyzer] ∥ [Priority Ranker]   (parallel)
        L2: [Schedule Constructor]      (needs both)
        L3: [Conflict Repairer]         (skip-edge from BA for bottleneck info)
        L4: [Schedule Formatter]        (terminal)
    """
    roles = {
        "Bottleneck Analyzer": RoleCard(
            name="Bottleneck Analyzer",
            system_prompt=(
                "You are a manufacturing bottleneck expert. Analyze the job-shop problem:\n"
                "1. Identify the bottleneck machine(s) with highest total processing load\n"
                "2. Compute the critical path length for each job\n"
                "3. Identify operations that compete for the same machine\n"
                "4. Estimate a lower bound for the makespan\n"
                "Output the bottleneck analysis with machine utilization estimates."
            ),
            capability_tags=["analysis", "bottleneck", "scheduling"],
        ),
        "Priority Ranker": RoleCard(
            name="Priority Ranker",
            system_prompt=(
                "You are a job priority expert. For each job in the problem:\n"
                "1. Compute total processing time across all operations\n"
                "2. Compute remaining work from each operation onward\n"
                "3. Rank jobs by longest remaining processing time (LPT rule)\n"
                "4. Consider due dates if provided\n"
                "Output a priority-ordered job list with justification."
            ),
            capability_tags=["priority", "ranking", "heuristics"],
        ),
        "Schedule Constructor": RoleCard(
            name="Schedule Constructor",
            system_prompt=(
                "You are a schedule construction expert. Given bottleneck analysis and "
                "job priorities:\n"
                "1. Use a priority-based dispatch rule to schedule operations\n"
                "2. For each operation, find the earliest feasible start time respecting "
                "both job precedence and machine availability\n"
                "3. Build a complete Gantt chart (job, operation, machine, start, end)\n"
                "4. Report the makespan\n"
                "Show the scheduling decisions step by step."
            ),
            capability_tags=["construction", "dispatch", "scheduling"],
        ),
        "Conflict Repairer": RoleCard(
            name="Conflict Repairer",
            system_prompt=(
                "You are a schedule conflict repair expert. Given the constructed schedule "
                "and bottleneck analysis:\n"
                "1. Check for machine conflicts (two ops on same machine at same time)\n"
                "2. Check precedence violations (op starts before predecessor finishes)\n"
                "3. For each conflict, shift the later operation forward\n"
                "4. Re-check for cascading conflicts\n"
                "5. Report the final conflict-free makespan\n"
                "Output the corrected schedule."
            ),
            capability_tags=["repair", "conflict-resolution", "constraints"],
        ),
        "Schedule Formatter": RoleCard(
            name="Schedule Formatter",
            system_prompt=(
                "You are the final schedule formatter. Given the repaired schedule, "
                "produce the complete job-shop schedule in the required format. "
                "Report the final makespan and verify all constraints are satisfied. "
                "List each operation with: Job, Operation, Machine, Start Time, End Time."
            ),
            capability_tags=["formatting", "synthesis", "validation"],
            communication_protocol="produce a final complete schedule in the exact required format",
            protected=True,
        ),
    }
    edges = [
        ("Bottleneck Analyzer", "Schedule Constructor"),
        ("Priority Ranker", "Schedule Constructor"),
        ("Schedule Constructor", "Conflict Repairer"),
        ("Bottleneck Analyzer", "Conflict Repairer"),  # skip-edge
        ("Conflict Repairer", "Schedule Formatter"),
    ]
    return roles, edges, "Schedule Formatter"


def _naturalplan_dag() -> _DagSpec:
    """
    Combined NaturalPlan DAG — task/contract parser + subtask-specialist branches.

    Topology:
        L1: [Task & Contract Parser]
        L2: [Trip Specialist] ∥ [Calendar Specialist] ∥ [Meeting Specialist]
        L3: [Candidate Synthesizer]
        L4: [Cross-Task Auditor]
        L5: [NaturalPlan Format Finalizer]

    The specialist branches are intentionally parallel. Non-matching specialists
    must say that their layer is not applicable, so the finalizer can still rely
    on a single task-type decision while preserving the detailed single-task
    prompt constraints that the parsers depend on.
    """
    roles = {
        "Task & Contract Parser": RoleCard(
            name="Task & Contract Parser",
            system_prompt=(
                "You are the NaturalPlan task-and-contract parser. Identify the task as exactly one of: trip planning, "
                "calendar scheduling, or meeting planning. Extract the final-format contract and hard constraints.\n\n"
                "Trip final format: only itinerary lines: **Day X-Y:** Visit CityName for N days. and "
                "**Day Z:** Fly from CityA to CityB. Flight days are shared: next visit starts on the flight day.\n"
                "Calendar final format: exactly one line, Here is the proposed time: <Day>, HH:MM - HH:MM, using 24-hour time.\n"
                "Meeting final format: only these step templates with 12-hour AM/PM and no spaces: "
                "You travel to [Location] in [N] minutes and arrive at [H:MMAM/PM]. "
                "You wait until [H:MMAM/PM]. "
                "You meet [Person] for [N] minutes from [H:MMAM/PM] to [H:MMAM/PM].\n\n"
                "Do not solve yet. Output subtask, entities, times, durations, locations, and hard constraints."
            ),
            capability_tags=["task-classification", "constraint-extraction", "format-contract"],
        ),
        "Trip Specialist": RoleCard(
            name="Trip Specialist",
            system_prompt=(
                "If the parser did not classify this as trip planning, output only 'NOT_APPLICABLE: not a trip task'. "
                "Otherwise solve the trip layer with the original trip constraints:\n"
                "1. Extract required cities, stay durations, total trip days, direct flights, and day-window anchors\n"
                "2. Build the flight adjacency list and propose a city order using only direct flights\n"
                "3. Preserve every required city exactly once; do not invent flights\n"
                "4. Assign day ranges with shared flight-day arithmetic: sum(stays) - (cities - 1) = total days\n"
                "5. If you fly on Day X, the next city visit starts on Day X, not Day X+1\n\n"
                "Draft the trip answer using the exact lines: **Day X-Y:** Visit CityName for N days. and "
                "**Day Z:** Fly from CityA to CityB."
            ),
            capability_tags=["trip", "routing", "day-arithmetic", "formatting"],
        ),
        "Calendar Specialist": RoleCard(
            name="Calendar Specialist",
            system_prompt=(
                "If the parser did not classify this as calendar scheduling, output only 'NOT_APPLICABLE: not a calendar task'. "
                "Otherwise solve the calendar layer with the original calendar constraints:\n"
                "1. For each participant, list busy blocks and derived free windows within working hours\n"
                "2. Treat all preferences as HARD constraints, including avoid-day and avoid-after-time rules\n"
                "3. Boundary rule: busy 9:00-11:00 means free at exactly 11:00\n"
                "4. Intersect all participants' free windows per day and keep every candidate of required duration\n"
                "5. Sort valid slots by day/time; if earliest availability is requested, choose earliest; otherwise follow task preference\n\n"
                "Draft exactly one proposed slot in 24-hour format: Here is the proposed time: <Day>, HH:MM - HH:MM"
            ),
            capability_tags=["calendar", "availability", "hard-constraints", "formatting"],
        ),
        "Meeting Specialist": RoleCard(
            name="Meeting Specialist",
            system_prompt=(
                "If the parser did not classify this as meeting planning, output only 'NOT_APPLICABLE: not a meeting task'. "
                "Otherwise solve the meeting layer with the original meeting constraints:\n"
                "1. Start at the stated location and time\n"
                "2. Use exact travel times from the distance matrix\n"
                "3. Travel only to locations where a meeting happens\n"
                "4. If arrival is before a person's availability window, insert a wait step until the window start\n"
                "5. Each meeting must start within the window and last at least the required minimum duration\n"
                "6. Maximize feasible meetings without illegal transitions or overlaps\n\n"
                "Draft only step lines in 12-hour AM/PM format: travel, wait, and meet templates."
            ),
            capability_tags=["meeting", "travel-time", "time-windows", "formatting"],
        ),
        "Candidate Synthesizer": RoleCard(
            name="Candidate Synthesizer",
            system_prompt=(
                "Use the parser's subtask decision. Select the applicable specialist output and ignore NOT_APPLICABLE branches. "
                "If the applicable output is incomplete, repair it from the original task. Produce one candidate answer only, "
                "already close to the required final format. Do not mix trip, calendar, and meeting formats."
            ),
            capability_tags=["selection", "construction", "repair"],
        ),
        "Cross-Task Auditor": RoleCard(
            name="Cross-Task Auditor",
            system_prompt=(
                "Strictly audit the candidate against the original task and parser decision.\n"
                "Trip: check every city, inclusive stay duration, direct flight edge, day window, shared flight-day overlap, and final day.\n"
                "Calendar: check every participant's busy blocks, duration, hard preferences, boundary times, and 24-hour proposed-time format.\n"
                "Meeting: check exact travel times, wait steps, availability windows, minimum durations, overlaps, and 12-hour step format.\n"
                "If any violation exists, output a corrected candidate in the same subtask format. If valid, reproduce the candidate."
            ),
            capability_tags=["validation", "constraints", "repair"],
        ),
        "NaturalPlan Format Finalizer": RoleCard(
            name="NaturalPlan Format Finalizer",
            system_prompt=(
                "You are the terminal NaturalPlan formatter. Output ONLY the final answer for the parsed subtask.\n\n"
                "Trip: only these itinerary lines, no heading or prose:\n"
                "**Day X-Y:** Visit CityName for N days.\n"
                "**Day Z:** Fly from CityA to CityB.\n\n"
                "Calendar: exactly one line, no quotes, no prose:\n"
                "Here is the proposed time: <Day>, HH:MM - HH:MM\n\n"
                "Meeting: only these step lines, using 12-hour AM/PM with no spaces:\n"
                "You travel to [Location] in [N] minutes and arrive at [H:MMAM/PM].\n"
                "You wait until [H:MMAM/PM].\n"
                "You meet [Person] for [N] minutes from [H:MMAM/PM] to [H:MMAM/PM].\n\n"
                "Never output analysis, labels, alternatives, or mixed formats."
            ),
            capability_tags=["aggregation", "formatting", "structured-output"],
            communication_protocol="produce only the exact final NaturalPlan subtask format",
            protected=True,
        ),
    }
    edges = [
        ("Task & Contract Parser", "Trip Specialist"),
        ("Task & Contract Parser", "Calendar Specialist"),
        ("Task & Contract Parser", "Meeting Specialist"),
        ("Task & Contract Parser", "Candidate Synthesizer"),
        ("Trip Specialist", "Candidate Synthesizer"),
        ("Calendar Specialist", "Candidate Synthesizer"),
        ("Meeting Specialist", "Candidate Synthesizer"),
        ("Candidate Synthesizer", "Cross-Task Auditor"),
        ("Task & Contract Parser", "Cross-Task Auditor"),
        ("Task & Contract Parser", "NaturalPlan Format Finalizer"),
        ("Candidate Synthesizer", "NaturalPlan Format Finalizer"),
        ("Cross-Task Auditor", "NaturalPlan Format Finalizer"),
    ]
    return roles, edges, "NaturalPlan Format Finalizer"


def _tablebench_dag() -> _DagSpec:
    """
    TableBench DAG — table contract parsing, evidence lookup, branch reasoning, verification.

    Topology:
        L1: [Question Schema Mapper]
        L2: [Evidence Retriever]
        L3: [Numerical Reasoner] || [Lookup Analysis Specialist]
        L4: [Arithmetic Auditor]
        L5: [Answer Auditor Synthesizer]
        L6: [Final Answer Formatter]
    """
    roles = {
        "Question Schema Mapper": RoleCard(
            name="Question Schema Mapper",
            system_prompt=(
                "You are a TableBench question and schema mapper. Classify the requested answer path before anyone solves: "
                "fact lookup, multi-hop fact checking, numerical reasoning, ranking, time calculation, descriptive analysis, "
                "correlation/trend/statistical reasoning, impact/causal analysis, or anomaly detection. "
                "State the exact final answer contract: entity/label, date/time/event row, numeric value, ordered list, anomaly row plus abnormal fields, or explanation. "
                "Flag whether the final answer should be a semantic identifier rather than the comparison number. Map the question to relevant columns, row identifiers, "
                "units, percentages, date fields, ranks, aggregation targets, ambiguous column names, and repeated headers. "
                "Explicitly identify the final answer target column or label: date, time, team, district, row, entity, or numeric value. "
                "Identify the primary semantic row identifier and mark generic index columns like Unnamed: 0 as non-answer columns unless explicitly requested. "
                "If both date and clock/origin-time columns exist, treat the date/event row label as the primary identifier for broad which-time extreme questions unless clock time is explicitly requested. "
                "For ordinal/domain-specific questions, identify whether period labels are column headers. "
                "Do not solve the task, do not produce charts, and do not reinterpret it as visualization or chart generation."
            ),
            capability_tags=["question-analysis", "table-schema", "answer-contract"],
            communication_protocol="output a compact answer contract and schema map without solving",
        ),
        "Evidence Retriever": RoleCard(
            name="Evidence Retriever",
            system_prompt=(
                "You are a table evidence retriever. Use the answer contract and schema map to select the exact rows and cells needed to answer the question. "
                "For multi-hop questions, trace each intermediate lookup from table evidence. "
                "For data-analysis questions, identify the variables or factors requested by the wording. "
                "For threshold filters such as greater than 2%, less than 10, or at least 5, scan every data row and include every row satisfying the condition; parse percentages numerically, so 2.356% is greater than 2%. "
                "For ranking questions, retrieve the extreme row, comparison value, and requested output label; retrieve both extrema only for a requested difference or gap. "
                "For TV episode anomaly questions, retrieve Episode <no> from the episode/no column. "
                "For anomaly questions, retrieve only the minimal dominant anomalies, normally at most two when no count is specified, with semantic row identifier or 1-indexed data-row number, abnormal columns, and abnormal values. "
                "For aggregation over an avg/average column, output only the raw values and COUNT in this form: VALUES=[...]; COUNT=n; do not calculate a sum or average in the evidence role. "
                "Do not invent facts outside the table, do not emit Final Answer, do not calculate the final answer, and keep the evidence list compact."
            ),
            capability_tags=["evidence", "row-selection", "grounding"],
        ),
        "Numerical Reasoner": RoleCard(
            name="Numerical Reasoner",
            system_prompt=(
                "You are the numerical and ranking branch of a TableBench static team. Solve only when the answer contract involves arithmetic, counting, aggregation, ranking, sorting, median, percentage, density, or time calculation. "
                "If the contract is mainly fact lookup or open-ended data analysis, state 'Not primary numeric path' and give only numeric checks that may help the synthesizer. "
                "For highest-lowest ranking questions, return the requested entity/label for which or what questions and compute the extrema difference only when asked. For period/ordinal questions, "
                "return the exact column label. Preserve requested precision, units, percentages, rankings, and comma-separated answer order. "
                "When the requested column itself is an average column (for example avg daily flts) and the question asks for total average, overall average, or average over all rows, compute the mean of that column across rows, not the sum; only output a sum when the wording asks for total/sum/combined total without asking for an average. For any sum over more than three values, recompute by pairwise chunks from the raw values instead of copying an upstream total. "
                "For filtered aggregations, first enumerate every row satisfying the filter before calculating; for percentage filters compare the numeric percent values, so values like 2.356% satisfy > 2%. "
                "For difference-from-average questions, compute the average over all numeric rows, then output value minus average on the same numeric scale unless a relative percentage is unambiguously required. "
                "If the requested derived column already exists, such as pop density (per km2), use that column directly and average every numeric row in it; do not recompute from raw population/area unless the derived column is absent. "
                "For highest-density versus average-density questions, output '<entity>, <highest density - average density>' on the density scale, not a relative percent. "
                "Do not produce charts, plots, Python code, or visualization instructions."
            ),
            capability_tags=["calculation", "ranking", "time-reasoning"],
            communication_protocol="solve the numeric/ranking/time branch when applicable, otherwise provide only useful numeric checks",
        ),
        "Lookup Analysis Specialist": RoleCard(
            name="Lookup Analysis Specialist",
            system_prompt=(
                "You are the lookup and data-analysis branch of a TableBench static team. Resolve direct lookup, multi-hop fact checking, descriptive analysis, trend, correlation, statistical, causal, impact, and anomaly questions using only visible table cells. "
                "Follow references across rows, columns, editions, years, categories, notes, titles, teams, districts, or labels. "
                "Return the exact requested entity, value, award, label, statement, or qualifier. "
                "For correlation, statistical, trend, and causal wording, compute or describe the table-supported statistic or association instead of refusing the wording. "
                "For impact questions, preserve exact factor lists. For descriptive analysis, cover column meanings, range, notable extrema/trends, and missing or unknown values in a concise evidence-backed explanation. "
                "For anomaly questions, include only the minimal dominant anomalies, normally at most two when no count is specified, with no weaker extras and give the semantic row identifier or 1-indexed data-row number plus abnormal column/value, using Episode <no> for TV episode anomalies. "
                "Prefer anomalies where multiple related columns are jointly extreme; unknown/range formatting alone is weaker than rows with all relevant numeric columns extremely high or extremely low. "
                "For casualty/death tables, choose rows whose military, civilian, total deaths, wounded, and total casualties are collectively extreme high or collectively tiny; do not select rows merely because values are unknown or ranges. "
                "If the answer contract is mainly numerical derivation, state 'Not primary lookup-analysis path' and list only exact facts or analysis checks that constrain the final answer. "
                "Never use outside knowledge, never answer with a generic index column when a semantic row identifier exists, and never produce charts, plots, Python code, or visualization instructions."
            ),
            capability_tags=["fact-checking", "multi-hop", "data-analysis"],
            communication_protocol="solve lookup/fact/data-analysis paths when applicable, otherwise provide constraining facts and checks",
        ),
        "Arithmetic Auditor": RoleCard(
            name="Arithmetic Auditor",
            system_prompt=(
                "You are a strict arithmetic auditor for TableBench. Recompute only arithmetic that appears in the evidence or specialist outputs: filters, row counts, sums, means, differences, medians, percentages, thresholds, and unit scale. "
                "Do not solve from scratch unless arithmetic is involved. Do not trust upstream totals; recompute from the listed raw values and verify every qualifying row is included exactly once. "
                "For averages, state the count, the exact sum, and sum/count. For sums over more than three values, recompute with pairwise chunks from the raw list, then add the chunks; never copy a total written by an upstream role. "
                "For filtered aggregations, verify the filter row by row, including percentage thresholds such as 2.356% > 2%. "
                "If a column header itself says avg or average and the question asks for total average, overall average, or average over all rows, audit that the result is the mean of that column, not the raw sum. "
                "If upstream arithmetic is correct, reproduce the concise candidate. If it is wrong, output the corrected concise candidate with the minimal arithmetic check. Never output charts, code, or markdown tables."
            ),
            capability_tags=["arithmetic-audit", "aggregation-check", "unit-scale"],
            communication_protocol="recompute and correct arithmetic candidates, or confirm no arithmetic correction is needed",
            temperature=0.0,
        ),
        "Answer Auditor Synthesizer": RoleCard(
            name="Answer Auditor Synthesizer",
            system_prompt=(
                "You are a TableBench answer auditor and synthesizer. Read the answer contract, schema map, retrieved evidence, both specialist branch outputs, and the arithmetic audit. "
                "Choose the branch that matches the contract. For any arithmetic, aggregation, threshold, or unit-scale disagreement, the Arithmetic Auditor is the authoritative upstream source; adopt its corrected candidate when it shows a row count, sum, mean, filter, or chunked recomputation grounded in the listed values. "
                "Reconcile non-arithmetic disagreements using visible table evidence, then audit the candidate before final formatting. "
                "Verify selected rows, columns, arithmetic, units, percentage handling, rank direction, multi-part answer order, extrema differences, sorted medians, ordinal column labels, factor-list names, and missing qualifiers such as year or edition. For sums over listed values, recompute by grouping adjacent values before accepting any upstream total. "
                "For filtered aggregations, reject candidates that skip any row satisfying the filter, especially percentage thresholds such as 2.356% > 2%. "
                "For avg/average columns, reject a raw sum when the question asks for total average, overall average, or average over all rows; the corrected answer should be sum divided by row count. "
                "If the question asks which, what, date, time, row, team, district, or entity, output the requested label/entity rather than the comparison number. "
                "Reject answers that return an extreme numeric value when the question asks for a date, time, row, or entity. "
                "Reject answers that use generic index columns such as Unnamed: 0 when a semantic identifier exists. "
                "Reject broad which-time answers that return only clock/origin time when the date/event row label is the benchmark target. "
                "Reject TV episode anomaly answers that omit Episode <no>. "
                "Reject anomaly answers that omit the row identifier/row number or abnormal column/value, use zero-based row numbers, or add weaker extra anomalies. "
                "For open-ended data-analysis answers, ensure the answer is supported by visible table evidence. Output one corrected concise answer, not a reasoning trace."
            ),
            capability_tags=["synthesis", "verification", "table-grounding"],
            communication_protocol="select, audit, and output one corrected concise answer",
            role_type="validator",
        ),
        "Final Answer Formatter": RoleCard(
            name="Final Answer Formatter",
            system_prompt=(
                "You are the final TableBench answer formatter. Use the verifier's corrected answer if present. "
                "Output exactly one line and nothing else:\n"
                "Final Answer: <answer>\n"
                "Do not include reasoning, markdown tables, code, chart instructions, labels other than Final Answer, or alternatives."
            ),
            capability_tags=["aggregation", "formatting", "structured-output"],
            communication_protocol="output exactly one line: Final Answer: <answer>",
            protected=True,
            role_type="aggregator",
        ),
    }
    edges = [
        ("Question Schema Mapper", "Evidence Retriever"),
        ("Question Schema Mapper", "Numerical Reasoner"),
        ("Evidence Retriever", "Numerical Reasoner"),
        ("Question Schema Mapper", "Lookup Analysis Specialist"),
        ("Evidence Retriever", "Lookup Analysis Specialist"),
        ("Question Schema Mapper", "Answer Auditor Synthesizer"),
        ("Numerical Reasoner", "Arithmetic Auditor"),
        ("Lookup Analysis Specialist", "Arithmetic Auditor"),
        ("Evidence Retriever", "Arithmetic Auditor"),
        ("Question Schema Mapper", "Arithmetic Auditor"),
        ("Lookup Analysis Specialist", "Answer Auditor Synthesizer"),
        ("Arithmetic Auditor", "Answer Auditor Synthesizer"),
        ("Answer Auditor Synthesizer", "Final Answer Formatter"),
    ]
    return roles, edges, "Final Answer Formatter"


_DAG_REGISTRY: Dict[str, Any] = {
    "trip": _trip_planning_dag,
    "naturalplan": _naturalplan_dag,
    "meeting": _meeting_planning_dag,
    "calendar": _calendar_scheduling_dag,
    "olympiadbench": _olympiad_dag,
    "tablebench": _tablebench_dag,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight DAG execution engine
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_levels(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    terminal: str,
) -> List[List[str]]:
    """Compute parallel execution levels (Kahn's algorithm), excluding terminal."""
    non_terminal = [n for n in nodes if n != terminal]
    node_set = set(non_terminal)

    predecessors: Dict[str, set] = {n: set() for n in non_terminal}
    for src, dst in edges:
        if src in node_set and dst in node_set:
            predecessors[dst].add(src)

    levels: List[List[str]] = []
    completed: set = set()
    remaining = list(non_terminal)

    while remaining:
        ready = [n for n in remaining if predecessors[n].issubset(completed)]
        if not ready:
            ready = [remaining[0]]  # safety fallback
        levels.append(ready)
        completed.update(ready)
        remaining = [n for n in remaining if n not in completed]

    return levels


def _format_prompt(
    role: RoleCard,
    task_prompt: str,
    incoming_msgs: List[str],
) -> str:
    """Build the user prompt for an agent, including upstream messages."""
    parts = [f"## Task\n{task_prompt}"]
    if incoming_msgs:
        parts.append("## Inputs from upstream agents")
        for i, msg in enumerate(incoming_msgs, 1):
            parts.append(f"### Input {i}\n{msg}")
    if role.communication_protocol:
        parts.append(f"\n## Your role\n{role.communication_protocol}")
    return "\n\n".join(parts)


def _get_incoming(
    role_name: str,
    edges: List[Tuple[str, str]],
    responses: Dict[str, str],
) -> List[str]:
    """Return responses from all predecessors that have an edge to role_name."""
    return [responses[src] for src, dst in edges
            if dst == role_name and src in responses]


def _execute_dag(
    task_prompt: str,
    roles: Dict[str, RoleCard],
    edges: List[Tuple[str, str]],
    terminal: str,
    client: OpenRouterClient,
    config: SeroConfig,
) -> Tuple[str, Dict[str, str]]:
    """
    Execute a fixed-topology DAG for one task.

    Runs non-terminal nodes level-by-level (parallel within each level),
    then runs the terminal node with all its incoming messages.

    Returns (final_answer, all_agent_responses).
    """
    node_names = list(roles.keys())
    levels = _compute_levels(node_names, edges, terminal)

    all_responses: Dict[str, str] = {}

    for level in levels:
        if len(level) == 1:
            name = level[0]
            role = roles[name]
            incoming = _get_incoming(name, edges, all_responses)
            prompt = _format_prompt(role, task_prompt, incoming)
            all_responses[name] = client.system_user(
                model=config.agent_model,
                system=role.system_prompt,
                user=prompt,
                temperature=role.temperature,
                max_tokens=4096,
            )
        else:
            snapshot = dict(all_responses)

            def _call(rn: str, snap: Dict[str, str]) -> Tuple[str, str]:
                r = roles[rn]
                inc = _get_incoming(rn, edges, snap)
                p = _format_prompt(r, task_prompt, inc)
                resp = client.system_user(
                    model=config.agent_model,
                    system=r.system_prompt,
                    user=p,
                    temperature=r.temperature,
                    max_tokens=4096,
                )
                return rn, resp

            workers = min(len(level), _MAX_PARALLEL_AGENTS)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_call, n, snapshot): n for n in level}
                for fut in as_completed(futures):
                    name, resp = fut.result()
                    all_responses[name] = resp

    # Terminal node
    terminal_role = roles[terminal]
    incoming = _get_incoming(terminal, edges, all_responses)
    prompt = _format_prompt(terminal_role, task_prompt, incoming)
    final_answer = client.system_user(
        model=config.agent_model,
        system=terminal_role.system_prompt,
        user=prompt,
        temperature=0.0,
        max_tokens=4096,
    )
    all_responses[terminal] = final_answer
    return final_answer, dict(all_responses)


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth & answer extraction helpers (benchmark-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_ground_truth(task: Dict[str, Any]) -> str:
    """Return the raw ground-truth string from a task dict."""
    gt = task.get("golden_plan") or task.get("gold_answer") or task.get("best_known")
    if gt is None:
        return ""
    if isinstance(gt, list):
        return ", ".join(str(x) for x in gt)
    return str(gt)


def _effective_benchmark(benchmark: str, task: Dict[str, Any] = None) -> str:
    if benchmark == "naturalplan" and task is not None:
        return task.get("sub_benchmark") or task.get("task_type") or benchmark
    return benchmark


def _extract_answer_for_benchmark(response: str, benchmark: str, task: Dict[str, Any] = None) -> str:
    """Extract the key structured answer from a response using benchmark-specific logic."""
    effective_benchmark = _effective_benchmark(benchmark, task)
    try:
        if effective_benchmark == "trip":
            from sero.benchmarks.trip_adapter import extract_canonical_answer
            return extract_canonical_answer(response)
        elif effective_benchmark == "meeting":
            from sero.benchmarks.meeting_plan_adapter import extract_canonical_answer
            return extract_canonical_answer(response)
        elif effective_benchmark == "calendar":
            from sero.benchmarks.calendar_scheduling_adapter import extract_canonical_answer
            return extract_canonical_answer(response)
        elif effective_benchmark == "olympiadbench":
            from sero.benchmarks.olympiad_adapter import extract_canonical_answer
            return extract_canonical_answer(response)
        elif effective_benchmark == "tablebench":
            from sero.benchmarks.tablebench_adapter import extract_canonical_answer
            return extract_canonical_answer(response)
        elif effective_benchmark == "jssp":
            from sero.benchmarks.realm_adapter import extract_makespan
            ms = extract_makespan(response)
            return str(ms) if ms is not None else ""
    except Exception:
        pass
    return ""


def _extract_ground_truth_for_benchmark(task: Dict[str, Any], benchmark: str) -> str:
    """Extract the comparison-ready ground truth (same format as extracted answer)."""
    gt = _get_ground_truth(task)
    if not gt:
        return ""
    effective_benchmark = _effective_benchmark(benchmark, task)
    try:
        if effective_benchmark in ("trip", "meeting", "calendar"):
            # For these benchmarks, apply the same canonical extractor to golden_plan
            if effective_benchmark == "trip":
                from sero.benchmarks.trip_adapter import extract_canonical_answer
                return extract_canonical_answer(gt)
            elif effective_benchmark == "meeting":
                from sero.benchmarks.meeting_plan_adapter import extract_canonical_answer
                return extract_canonical_answer(gt)
            elif effective_benchmark == "calendar":
                from sero.benchmarks.calendar_scheduling_adapter import extract_canonical_answer
                return extract_canonical_answer(gt)
        elif effective_benchmark == "olympiadbench":
            # gold_answer is already the reference expression
            return gt
        elif effective_benchmark == "tablebench":
            from sero.benchmarks.tablebench_adapter import extract_canonical_answer
            return extract_canonical_answer(f"Final Answer: {gt}")
        elif effective_benchmark == "jssp":
            # best_known is already the integer makespan
            return gt
    except Exception:
        pass
    return gt


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_static_dag_free(
    tasks: List[Dict[str, Any]],
    benchmark: str,
    client: OpenRouterClient,
    config: SeroConfig,
) -> Dict[str, Any]:
    """
    Evaluate the static fixed-topology DAG baseline on a list of tasks.

    Each benchmark uses a hand-crafted non-linear DAG with dedicated roles.
    No credit mechanism, no role retrieval, no evolution.
    Returns summary dict compatible with _save_result().
    """
    if benchmark not in _DAG_REGISTRY:
        raise ValueError(
            f"No static DAG defined for benchmark '{benchmark}'. "
            f"Available: {', '.join(_DAG_REGISTRY)}"
        )

    roles, edges, terminal = _DAG_REGISTRY[benchmark]()

    def _eval_task(task: Dict[str, Any]) -> Dict[str, Any]:
        agent_traces: Dict[str, str] = {}
        answer = ""
        try:
            answer, agent_traces = _execute_dag(
                task["prompt"], roles, edges, terminal, client, config)
            raw_score = task["eval_fn"](answer)
            score = normalize_score(raw_score)
        except Exception as e:
            logger.error("Static error on %s: %s", task["id"], e)
            score, raw_score = 0.0, 0.0
        logger.info("[Static-%s] task=%s score=%.3f", benchmark, task["id"], score)

        extracted_answer = _extract_answer_for_benchmark(answer, benchmark, task)
        ground_truth_raw = _get_ground_truth(task)
        ground_truth_extracted = _extract_ground_truth_for_benchmark(task, benchmark)

        return {
            "task_id": task["id"],
            "sub_benchmark": task.get("sub_benchmark"),
            "prompt": task["prompt"],
            "ground_truth": ground_truth_raw,
            "ground_truth_extracted": ground_truth_extracted,
            "response": answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "raw_score": raw_score,
            "agent_traces": agent_traces,
        }

    records = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=_EVAL_WORKERS) as executor:
        fmap = {executor.submit(_eval_task, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(fmap):
            records[fmap[fut]] = fut.result()

    scores = [r["score"] for r in records]
    result = {
        "system": "static",
        "benchmark": benchmark,
        "n_tasks": len(scores),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "pool": list(roles.keys()),
        "dag_edges": [(s, d) for s, d in edges],
        "records": records,
    }

    # Dual metrics (trip / meeting)
    sample = records[0]["raw_score"] if records else None
    if sample is not None and is_dual_score(sample):
        for key in ("exact_score",):
            vals = [r["raw_score"].get(key, 0.0) if isinstance(r["raw_score"], dict)
                    else r["score"] for r in records]
            result[f"mean_{key}"] = float(np.mean(vals))
            result[f"std_{key}"] = float(np.std(vals))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        sub_benchmark = record.get("sub_benchmark")
        if sub_benchmark:
            grouped.setdefault(sub_benchmark, []).append(record)
    if grouped:
        result["per_sub_benchmark"] = {}
        for sub_benchmark, group_records in sorted(grouped.items()):
            partial_vals = [
                r["raw_score"].get("partial_score", r.get("score", 0.0))
                if isinstance(r.get("raw_score"), dict)
                else r.get("score", 0.0)
                for r in group_records
            ]
            exact_vals = [
                r["raw_score"].get("exact_score", r.get("score", 0.0))
                if isinstance(r.get("raw_score"), dict)
                else r.get("score", 0.0)
                for r in group_records
            ]
            result["per_sub_benchmark"][sub_benchmark] = {
                "n_tasks": len(group_records),
                "mean_score": float(np.mean(partial_vals)) if partial_vals else 0.0,
                "std_score": float(np.std(partial_vals)) if partial_vals else 0.0,
                "mean_exact_score": float(np.mean(exact_vals)) if exact_vals else 0.0,
                "std_exact_score": float(np.std(exact_vals)) if exact_vals else 0.0,
            }

    return result
