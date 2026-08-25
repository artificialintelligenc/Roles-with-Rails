"""
Workflow-Based Multi-Agent Baseline.

An expert-designed, hand-engineered sequential pipeline for the supported
benchmarks (naturalplan / trip / meeting / calendar / olympiadbench / tablebench).

Fixed topology (linear chain), fixed per-benchmark role pool, no evolution,
no credit computation, no DAG construction, no role selection.

Decoupled from PhaseA — reuses the lightweight DAG execution engine from
static_dag_mas.py directly.

The key difference from SERO:
- Fixed topology (linear chain, not credit-ranked DAG)
- Fixed role pool (no evolution)
- No controller (no RL)
- No credit signals

Compared to static_dag_mas:
- LINEAR chain topology (not non-linear benchmark-specific DAGs)
- Expert-designed prompts that explicitly reference upstream agents
- Fixed workflow definitions for naturalplan / trip / meeting / calendar / olympiadbench / tablebench
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

import numpy as np

from sero.benchmarks.scoring_utils import normalize_score, is_dual_score
from sero.role_card import RoleCard
from sero.openrouter_client import OpenRouterClient
from sero.config import SeroConfig
from sero.baselines.static_dag_mas import (
    _execute_dag,
    _get_ground_truth,
    _extract_answer_for_benchmark,
    _extract_ground_truth_for_benchmark,
)

logger = logging.getLogger(__name__)

_EVAL_WORKERS = 5

_DagSpec = Tuple[Dict[str, RoleCard], List[Tuple[str, str]], str]

# Benchmarks for which hand-engineered workflows are defined.
_NATURALPLAN_BENCHMARKS = ("naturalplan", "trip", "meeting", "calendar", "olympiadbench", "tablebench")



# ═══════════════════════════════════════════════════════════════════════════════
# Per-benchmark workflow definitions (linear pipeline)
#
# Each factory returns (roles_dict, edges, terminal_name).
# Edges form a strict linear chain: A → B → C → D (no skip-edges).
# ═══════════════════════════════════════════════════════════════════════════════

def _trip_workflow_dag() -> _DagSpec:
    """
    Trip workflow — deterministic linear pipeline:
      Route Planner → Constraint Checker → Logistics Optimizer → Plan Aggregator
    """
    roles = {
        "WF Route Planner": RoleCard(
            name="WF Route Planner",
            system_prompt=(
                "You are a travel routing expert. Given a trip planning problem with cities, "
                "required stay durations, direct flight connections, and time-window constraints, "
                "propose a valid city ordering.\n\n"
                "IMPORTANT RULES:\n"
                "- Each consecutive city pair must have a direct flight.\n"
                "- Flight days are SHARED: when you fly from city A to city B on Day X, that day "
                "is the LAST day in city A AND the FIRST day in city B. Therefore, the total trip "
                "days equals the sum of all city stay durations (NOT sum + flight days).\n"
                "- Time-window constraints (e.g. 'visit X between day A and day B') must be "
                "satisfied.\n\n"
                "Output a day-by-day itinerary in multi-line format, one line per segment:\n"
                "Day X-Y: Visit CityName\n"
                "Day Z: Fly from CityA to CityB"
            ),
            capability_tags=["routing", "travel"],
        ),
        "WF Constraint Checker": RoleCard(
            name="WF Constraint Checker",
            system_prompt=(
                "You are a travel constraint validator. The Route Planner above has proposed "
                "an itinerary. Review it and check ALL constraints:\n"
                "1. Every required city is visited for the correct number of days\n"
                "2. Every consecutive flight connection exists in the direct-flight list\n"
                "3. All time-window constraints are satisfied\n"
                "4. Total days equal the trip duration (remember: flight days are SHARED "
                "between departure and arrival cities)\n\n"
                "If ANY violation is found, explain it and provide a CORRECTED itinerary.\n"
                "If all constraints pass, confirm the itinerary is valid and reproduce it "
                "in the same multi-line format."
            ),
            capability_tags=["validation", "constraints"],
        ),
        "WF Logistics Optimizer": RoleCard(
            name="WF Logistics Optimizer",
            system_prompt=(
                "You are a travel logistics optimizer. The Constraint Checker above has "
                "verified (and possibly corrected) the itinerary. Your job:\n"
                "1. If the Constraint Checker corrected the plan, verify the correction is valid\n"
                "2. Confirm every constraint is satisfied in the final version\n"
                "3. Output the definitive itinerary\n\n"
                "Output the itinerary in multi-line format, one line per segment:\n"
                "Day X-Y: Visit CityName\n"
                "Day Z: Fly from CityA to CityB"
            ),
            capability_tags=["optimization", "logistics"],
        ),
        "WF Plan Aggregator": RoleCard(
            name="WF Plan Aggregator",
            system_prompt=(
                "You are the final plan synthesizer. Given all upstream analyses, produce "
                "the final trip plan.\n\n"
                "You MUST output the plan with EXACTLY one line per segment, using this format:\n"
                "**Day X-Y:** Visit CityName for N days.\n"
                "**Day Z:** Fly from CityA to CityB.\n\n"
                "RULES:\n"
                "- Start the first city at Day 1.\n"
                "- Flight days overlap: the departure day is the last day at the origin AND "
                "the first day at the destination.\n"
                "- Do NOT output anything else. Only the day-by-day plan lines."
            ),
            capability_tags=["synthesis", "formatting"],
            communication_protocol=(
                "produce ONLY the final plan in multi-line Day X-Y format, one line per segment"
            ),
        ),
    }
    edges = [
        ("WF Route Planner", "WF Constraint Checker"),
        ("WF Constraint Checker", "WF Logistics Optimizer"),
        ("WF Logistics Optimizer", "WF Plan Aggregator"),
    ]
    return roles, edges, "WF Plan Aggregator"


def _meeting_workflow_dag() -> _DagSpec:
    """
    Meeting workflow — deterministic linear pipeline:
      Schedule Analyzer → Route Planner → Schedule Verifier → Meeting Aggregator
    """
    roles = {
        "WF Schedule Analyzer": RoleCard(
            name="WF Schedule Analyzer",
            system_prompt=(
                "You are a meeting schedule analysis expert. Given the list of people, their "
                "available time windows, locations, and travel distances, analyze constraints.\n"
                "For each person output ONE line: Person | Location | Window | MinDuration | LatestStart | Slack\n"
                "Then list conflicts (overlapping windows that force a choice).\n"
                "Be CONCISE — use a table or compact list, no prose."
            ),
            capability_tags=["analysis", "scheduling"],
        ),
        "WF Route Planner": RoleCard(
            name="WF Route Planner",
            system_prompt=(
                "You are a travel route optimization expert. Using the Schedule Analyzer's "
                "constraint analysis above, propose an efficient meeting route:\n"
                "1. Start from the given starting location and time\n"
                "2. Pick next person: prefer tight windows, minimize travel time\n"
                "3. For each step, compute: departure time + travel time = arrival time\n"
                "4. Schedule meeting within the person's available window\n\n"
                "Use 12-hour AM/PM time format (e.g. 2:30PM, not 14:30).\n"
                "Output ONLY the step-by-step schedule: travel, wait (if needed), meet.\n"
                "For each meet step, verify: start+duration=end, and end<=window_end.\n"
                "Be CONCISE — show arithmetic inline, no separate analysis paragraphs."
            ),
            capability_tags=["routing", "optimization"],
        ),
        "WF Schedule Verifier": RoleCard(
            name="WF Schedule Verifier",
            system_prompt=(
                "You are a strict schedule constraint verifier. Review the Route Planner's "
                "proposed schedule above and check EVERY step:\n"
                "1. Travel times match the distance matrix (use exact values from the task)\n"
                "2. Arrival time = departure time + travel time (arithmetic must be correct)\n"
                "3. Meeting start >= person's window start (not early)\n"
                "4. Meeting end <= person's window end (not late)\n"
                "5. Meeting duration >= minimum required\n"
                "6. stated duration must equal (end_time - start_time)\n\n"
                "Use 12-hour AM/PM time format (e.g. 2:30PM, not 14:30).\n"
                "If ANY error is found, FIX it and output the CORRECTED complete schedule.\n"
                "Output ONLY the corrected step-by-step schedule, no analysis prose."
            ),
            capability_tags=["verification", "constraints"],
        ),
        "WF Meeting Aggregator": RoleCard(
            name="WF Meeting Aggregator",
            system_prompt=(
                "You are the final meeting plan synthesizer. Given all upstream analyses, "
                "produce the complete meeting schedule.\n\n"
                "RULES:\n"
                "1. Use 12-hour AM/PM time (e.g. 2:30PM, NOT 14:30).\n"
                "2. For 'meet' steps, duration MUST equal (end_time - start_time).\n"
                "3. Travel times MUST match the distance matrix in the task.\n"
                "4. Start with 'SOLUTION:'\n\n"
                "Format every step using EXACTLY one of these templates:\n"
                "'You travel to [Location] in [N] minutes and arrive at [H:MMAM/PM].'\n"
                "'You wait until [H:MMAM/PM].'\n"
                "'You meet [Person] for [N] minutes from [H:MMAM/PM] to [H:MMAM/PM].'\n\n"
                "Do NOT deviate from these formats. Cover every feasible meeting."
            ),
            capability_tags=["synthesis", "formatting"],
            communication_protocol=(
                "Start with 'SOLUTION:'. "
                "Produce the complete schedule using ONLY these step formats with 12-hour AM/PM time: "
                "'You travel to [Location] in [N] minutes and arrive at [H:MMAM/PM].' "
                "'You wait until [H:MMAM/PM].' "
                "'You meet [Person] for [N] minutes from [H:MMAM/PM] to [H:MMAM/PM].'"
            ),
        ),
    }
    edges = [
        ("WF Schedule Analyzer", "WF Route Planner"),
        ("WF Route Planner", "WF Schedule Verifier"),
        ("WF Schedule Verifier", "WF Meeting Aggregator"),
    ]
    return roles, edges, "WF Meeting Aggregator"


def _calendar_workflow_dag() -> _DagSpec:
    """
    Calendar workflow — deterministic linear pipeline:
      Availability Analyzer → Time Slot Proposer → Constraint Validator → Schedule Aggregator
    """
    roles = {
        "WF Availability Analyzer": RoleCard(
            name="WF Availability Analyzer",
            system_prompt=(
                "You are a calendar availability expert. Given all participants' existing "
                "schedules and the required meeting duration:\n"
                "1. List each participant's name and ALL existing appointments (day, times)\n"
                "2. Compute each participant's FREE time slots within working hours (9:00-17:00)\n"
                "3. Find the intersection of all participants' free slots per day\n"
                "4. Filter intersections to those >= the required meeting duration\n"
                "5. List ALL valid candidate slots (day, start, end)\n\n"
                "Be precise with times. Use 24-hour format."
            ),
            capability_tags=["analysis", "availability", "scheduling"],
        ),
        "WF Time Slot Proposer": RoleCard(
            name="WF Time Slot Proposer",
            system_prompt=(
                "You are a meeting time slot expert. Using the Availability Analyzer's "
                "list of free windows above:\n"
                "1. Filter to slots that are >= the required meeting duration\n"
                "2. Rank candidates: prefer the earliest valid slot\n"
                "3. If multiple options on the same day, prefer morning or early afternoon\n"
                "4. Propose the single best meeting slot with day, start time, and end time\n\n"
                "Output your proposed slot clearly."
            ),
            capability_tags=["scheduling", "proposal", "time-slots"],
        ),
        "WF Constraint Validator": RoleCard(
            name="WF Constraint Validator",
            system_prompt=(
                "You are a scheduling constraint validator. Review the Time Slot Proposer's "
                "recommendation above:\n"
                "1. Verify the proposed slot does NOT conflict with any participant's appointments\n"
                "2. Verify the duration exactly matches the requirement\n"
                "3. Check all soft preferences are respected if mentioned\n"
                "4. If the slot is invalid, pick the next best candidate from the Availability "
                "Analyzer's list\n\n"
                "Output the validated recommendation."
            ),
            capability_tags=["validation", "constraints"],
        ),
        "WF Schedule Aggregator": RoleCard(
            name="WF Schedule Aggregator",
            system_prompt=(
                "You are the final schedule decision maker. Given the availability analysis, "
                "proposed slot, and constraint validation, select the best valid meeting time.\n\n"
                "Output EXACTLY one line in this format:\n"
                "'Here is the proposed time: <Day>, HH:MM - HH:MM'\n"
                "using 24-hour format. Output ONLY this one line, nothing else."
            ),
            capability_tags=["synthesis", "formatting"],
            communication_protocol=(
                "output ONLY 'Here is the proposed time: <Day>, HH:MM - HH:MM'"
            ),
        ),
    }
    edges = [
        ("WF Availability Analyzer", "WF Time Slot Proposer"),
        ("WF Time Slot Proposer", "WF Constraint Validator"),
        ("WF Constraint Validator", "WF Schedule Aggregator"),
    ]
    return roles, edges, "WF Schedule Aggregator"


def _olympiad_workflow_dag() -> _DagSpec:
    """
    Olympiad workflow — deterministic linear pipeline:
      Problem Decomposer → Solution Strategist → Verification Expert → Answer Synthesizer
    """
    roles = {
        "WF Problem Decomposer": RoleCard(
            name="WF Problem Decomposer",
            system_prompt=(
                "You are a mathematical problem analysis expert. Break down the given "
                "olympiad problem into its key sub-problems:\n"
                "1. Identify the subject and the core domain(s): number theory, combinatorics, "
                "algebra, geometry, calculus, mechanics, thermodynamics, electromagnetism, relativity, etc.\n"
                "2. List only the relevant theorems, lemmas, formulas, or provided context results that are directly useful\n"
                "3. Identify key variables and what needs to be found\n"
                "4. Note any special conditions or edge cases\n"
                "5. Suggest the most promising solution approach(es)\n\n"
                "Extra checks:\n"
                "- For Mathematics: call out invariants, legality of intended inequality tools, and whether an extremum or minimum requires both a bound and a matching construction\n"
                "- For discrete operation/counting problems: identify the conserved quantity or lower-bound certificate before choosing a target configuration\n"
                "- For Physics: note the frame/sign convention, whether the target is a signed scalar or a magnitude, and any provided context formula that should be substituted directly\n\n"
                "Output a clear decomposition with key observations. Do NOT solve the problem."
            ),
            capability_tags=["decomposition", "analysis", "math", "physics"],
        ),
        "WF Solution Strategist": RoleCard(
            name="WF Solution Strategist",
            system_prompt=(
                "You are a rigorous olympiad solver. Use the task statement and the Problem "
                "Decomposer's notes, but trust the task statement over the decomposition if they conflict.\n"
                "1. Set up equations or relationships from the identified concepts or provided context results\n"
                "2. Show the critical intermediate calculation or derivation steps\n"
                "3. For Mathematics: justify each inequality, invariant, extremal, or counting step; if the task asks for a least number, prove a lower bound and then give a construction/count that attains it\n"
                "4. For Physics: define the frame/sign convention before substituting, track units throughout, and distinguish a signed scalar quantity from a magnitude\n"
                "5. For relativity, rotating-frame, or simultaneity problems, write the transformation/effective-field equation explicitly before inferring the final distance/frequency; do NOT reuse an Earth-frame distance or a vector norm as the moving-frame signed answer without derivation\n"
                "6. For flux/geometry/radiation problems, explicitly derive the emitting or intercepting area used; do NOT introduce an extra length/area factor without derivation\n"
                "7. Prefer an exact symbolic answer. Do NOT switch to a decimal approximation unless the problem explicitly asks for one\n"
                "8. If you get stuck, try one brief alternative approach\n\n"
                "Conclude with exactly one line in this format:\n"
                "Final candidate answer: <answer>"
            ),
            capability_tags=["solving", "calculation", "proof", "derivation"],
            temperature=0.7,
        ),
        "WF Verification Expert": RoleCard(
            name="WF Verification Expert",
            system_prompt=(
                "You are an adversarial verifier. Do NOT merely paraphrase the proposed solution. "
                "Independently stress-test it against the task statement.\n"
                "1. Re-check the target quantity and required answer type\n"
                "2. Re-derive the key step(s) independently, especially the final transformation to the answer\n"
                "3. Look for omitted terms, hidden assumptions, sign errors, branch mistakes, domain restrictions, and failed edge cases\n"
                "4. For Mathematics, explicitly check that each inequality, invariant, lower-bound argument, and attainability claim is legal\n"
                "5. For Physics, verify frame-specific quantities, sign conventions, units, dimensional consistency, threshold conditions, and whether any geometric factor or reference term was dropped or invented\n"
                "6. If the task asks for a signed scalar quantity around a specified axis or frame, do NOT replace it with a magnitude unless the problem explicitly asks for magnitude\n"
                "7. If the proposed answer is decimal but an exact form is available, recover the exact form\n"
                "8. Do NOT issue a correction unless you can point to the first invalid step and replace it with a concrete corrected derivation that reaches the new answer\n\n"
                "Output in exactly this format:\n"
                "VERIFIED: <exact answer>\n"
                "or\n"
                "CORRECTED: <exact corrected answer>\n"
                "Then give a concise justification."
            ),
            capability_tags=["verification", "error-checking", "validation"],
        ),
        "WF Answer Synthesizer": RoleCard(
            name="WF Answer Synthesizer",
            system_prompt=(
                "You are the final answer synthesizer. Read the task statement and the verifier output. "
                "Treat the verifier's leading VERIFIED/CORRECTED answer as the primary source for the final answer.\n\n"
                "Rules:\n"
                "1. Output exactly one final boxed answer\n"
                "2. Preserve exact symbolic form whenever possible; avoid decimal approximations unless explicitly requested\n"
                "3. Match the answer type required by the task: numerical value, expression, equation, interval, tuple, or comma-separated list\n"
                "4. If the task specifies a unit, put the value/expression inside \\boxed{} and place the unit outside the box\n"
                "5. If no unit is specified, do NOT invent one\n"
                "6. Do NOT include explanation, labels, or multiple candidate answers\n\n"
                "Output ONLY one of these forms:\n"
                "\\boxed{<answer>}\n"
                "or\n"
                "\\boxed{<answer>} <unit>"
            ),
            capability_tags=["synthesis", "aggregation", "formatting"],
            communication_protocol="provide the final answer in \\boxed{<answer>} format",
        ),
    }
    edges = [
        ("WF Problem Decomposer", "WF Solution Strategist"),
        ("WF Solution Strategist", "WF Verification Expert"),
        ("WF Verification Expert", "WF Answer Synthesizer"),
    ]
    return roles, edges, "WF Answer Synthesizer"


def _naturalplan_workflow_dag() -> _DagSpec:
    """Combined NaturalPlan workflow — task/contract parser → constraints → candidate → validator → finalizer."""
    roles = {
        "WF NaturalPlan Task & Contract Parser": RoleCard(
            name="WF NaturalPlan Task & Contract Parser",
            system_prompt=(
                "Identify the task as exactly one NaturalPlan subtask: trip planning, calendar scheduling, or meeting planning. "
                "Then state the required final answer contract.\n\n"
                "The appended IMPORTANT marker in the task prompt is authoritative: if it says CALENDAR SCHEDULING, "
                "TRIP PLANNING, or MEETING PLANNING, use that subtask label even if the natural language mentions scheduling a meeting.\n"
                "Trip contract: only itinerary lines: **Day X-Y:** Visit CityName for N days. and **Day Z:** Fly from CityA to CityB. "
                "Flight days are shared; the next city starts on the flight day.\n"
                "Calendar contract: exactly one line: Here is the proposed time: <Day>, HH:MM - HH:MM, with 24-hour time.\n"
                "Meeting contract: only step lines: You travel to [Location] in [N] minutes and arrive at [H:MMAM/PM]. "
                "You wait until [H:MMAM/PM]. You meet [Person] for [N] minutes from [H:MMAM/PM] to [H:MMAM/PM].\n\n"
                "Do not solve. Output subtask and final-format contract first."
            ),
            capability_tags=["task-classification", "format-contract"],
        ),
        "WF NaturalPlan Constraint Sheet": RoleCard(
            name="WF NaturalPlan Constraint Sheet",
            system_prompt=(
                "Using the parser decision and original task, extract the hard constraints for the selected subtask only.\n\n"
                "Trip: cities, stay durations, total trip days, direct flights, time windows, day anchors, and shared-flight arithmetic.\n"
                "Calendar: participants, busy blocks, working hours, required duration, day/time preferences, earliest/latest rules, and boundary rule that busy end time is free.\n"
                "Meeting: starting location/time, each person/location/window/minimum duration, exact travel times, tight windows, and waiting requirements.\n\n"
                "Preferences in calendar tasks are hard constraints. Keep all numbers and names exact. Do not output a final answer yet. "
                "Never output SOLUTION, travel/wait/meet steps, a calendar proposed-time line, or a trip itinerary in this constraint-sheet role. "
                "Output only a structured constraint table/checklist."
            ),
            capability_tags=["constraint-extraction", "naturalplan"],
        ),
        "WF NaturalPlan Candidate Planner": RoleCard(
            name="WF NaturalPlan Candidate Planner",
            system_prompt=(
                "Construct one candidate answer for the selected subtask from the constraint sheet.\n\n"
                "Trip planning:\n"
                "- Choose a city order using only direct flights\n"
                "- Preserve every city exactly once\n"
                "- Assign inclusive day ranges with shared flight-day overlap\n"
                "- Use this arithmetic exactly: if a city starts Day S and lasts D days, it ends Day S+D-1; fly on Day S+D-1; the next city starts on that same day\n"
                "- Verify the final day equals the total trip duration before drafting\n"
                "- Draft in the exact Day X-Y / Day Z itinerary style\n\n"
                "Calendar scheduling:\n"
                "- Compute simultaneous free windows across all participants\n"
                "- Apply all preferences as hard filters\n"
                "- Respect boundary times exactly with half-open intervals [start, end): overlap iff slot_start < busy_end AND slot_end > busy_start\n"
                "- A busy block ending at 14:00 allows 14:00-14:30; a busy block starting at 16:30 allows 16:00-16:30; a busy block 11:00-17:00 rejects 12:00-12:30\n"
                "- Verify the chosen slot against every original busy block and ensure it ends no later than the work-hour close\n"
                "- Draft exactly one proposed-time line in 24-hour format\n\n"
                "Meeting planning:\n"
                "- Build a feasible travel/wait/meet schedule from the stated start\n"
                "- Use exact matrix travel times\n"
                "- Insert wait steps before availability windows\n"
                "- Never wait backwards; meeting_end must be <= that person's window_end\n"
                "- Maximize valid meetings, but omit invalid meetings rather than listing them\n"
                "- Draft only the allowed step templates in 12-hour AM/PM format"
            ),
            capability_tags=["planning", "construction", "format-aware"],
        ),
        "WF NaturalPlan Validator": RoleCard(
            name="WF NaturalPlan Validator",
            system_prompt=(
                "Audit the candidate against the original task, not just upstream notes. Recompute the decisive constraints yourself.\n"
                "Trip: city coverage, inclusive stay durations, direct flights, day windows, shared flight-day overlap using S+D-1 arithmetic, total final day.\n"
                "Calendar: no conflicts with each original busy block for every participant, exact duration, all hard preferences, exact boundary behavior, work-hour close, 24-hour line format. Use half-open intervals [start, end): slot overlaps busy iff slot_start < busy_end AND slot_end > busy_start. Starts exactly at busy_end are valid; ends exactly at busy_start are valid. Examples: 14:00-14:30 is valid after busy 9:00-14:00; 16:00-16:30 is valid before busy 16:30-17:00; 12:00-12:30 is invalid during busy 11:00-17:00.\n"
                "Meeting: exact travel times, chronological arrival/end times, required wait steps, availability windows, minimum durations, no impossible transitions or overlaps, 12-hour line format. If a step is outside a window or time goes backwards, remove or reschedule it; never pass it through as final.\n"
                "If invalid, output only a corrected candidate in the same subtask format. If valid, reproduce it."
            ),
            capability_tags=["validation", "repair", "constraints"],
        ),
        "WF NaturalPlan Finalizer": RoleCard(
            name="WF NaturalPlan Finalizer",
            system_prompt=(
                "Output only the final answer for the selected NaturalPlan subtask.\n\n"
                "Use the validator's corrected candidate as source of truth. Do not copy earlier invalid candidate steps, "
                "slots, or itinerary lines that the validator flagged. Trust the original task constraints over upstream text.\n\n"
                "Trip: only lines of these forms, no heading/prose:\n"
                "**Day X-Y:** Visit CityName for N days.\n"
                "**Day Z:** Fly from CityA to CityB.\n\n"
                "Calendar: exactly one line, no quotes/prose:\n"
                "Here is the proposed time: <Day>, HH:MM - HH:MM\n\n"
                "Meeting: only lines of these forms, 12-hour AM/PM with no spaces:\n"
                "You travel to [Location] in [N] minutes and arrive at [H:MMAM/PM].\n"
                "You wait until [H:MMAM/PM].\n"
                "You meet [Person] for [N] minutes from [H:MMAM/PM] to [H:MMAM/PM].\n\n"
                "Do not include SOLUTION:, explanations, labels, alternatives, or mixed formats."
            ),
            capability_tags=["aggregation", "formatting", "structured-output"],
            communication_protocol="produce only the exact final NaturalPlan subtask answer",
            protected=True,
        ),
    }
    edges = [
        ("WF NaturalPlan Task & Contract Parser", "WF NaturalPlan Constraint Sheet"),
        ("WF NaturalPlan Constraint Sheet", "WF NaturalPlan Candidate Planner"),
        ("WF NaturalPlan Candidate Planner", "WF NaturalPlan Validator"),
        ("WF NaturalPlan Validator", "WF NaturalPlan Finalizer"),
    ]
    return roles, edges, "WF NaturalPlan Finalizer"


def _tablebench_workflow_dag() -> _DagSpec:
    """TableBench workflow — parse table -> retrieve evidence -> solve -> verify -> format."""
    roles = {
        "WF Table Parser": RoleCard(
            name="WF Table Parser",
            system_prompt=(
                "You are a table parser. Read the table and question, then identify the relevant columns, row keys, "
                "units, percentages, date fields, and answer type. Do not solve yet. "
                "Explicitly identify the final answer target column or label: date, time, team, district, row, entity, or numeric value. "
                "Identify the primary semantic row identifier and mark generic index columns like Unnamed: 0 as non-answer columns unless explicitly requested. "
                "If both date and clock/origin-time columns exist, treat the date/event row label as the primary identifier for broad which-time extreme questions unless clock time is explicitly requested. "
                "For ordinal/domain-specific questions, identify whether period labels are column headers. "
                "Ignore visualization and chart-generation interpretations."
            ),
            capability_tags=["table-parsing", "schema", "answer-type"],
        ),
        "WF Evidence Selector": RoleCard(
            name="WF Evidence Selector",
            system_prompt=(
                "You are a table evidence selector. Using the parser's schema map, list the exact rows and cells needed. "
                "For multi-hop questions, show the chain of table lookups. For data-analysis questions, identify the visible "
                "variables, factors, trends, anomalies, or correlations that the answer must discuss. For ranking questions, "
                "retrieve the extreme row, comparison value, and requested output label; retrieve both extrema only for a requested difference or gap. "
                "For TV episode anomaly questions, retrieve Episode <no> from the episode/no column. "
                "For anomaly questions, retrieve only the minimal dominant anomalies, normally at most two when no count is specified, with semantic row identifier or 1-indexed data-row number, abnormal columns, and abnormal values. "
                "Do not invent outside facts. Keep the evidence compact."
            ),
            capability_tags=["evidence", "row-selection", "grounding"],
        ),
        "WF Table Solver": RoleCard(
            name="WF Table Solver",
            system_prompt=(
                "You are a TableBench solver. Use the selected evidence to compute or infer the answer. "
                "For numerical tasks, perform the decisive arithmetic and preserve precision, units, rankings, percentages, "
                "and comma-separated answer order. For fact checking, return the exact requested fact. For data analysis, "
                "write the concise factor list or explanation requested by the question. For highest-lowest ranking questions, "
                "return the requested entity/label for which or what questions and compute the extrema difference only when asked. For period/ordinal questions, return the exact column label. For "
                "correlation, statistical, trend, and causal questions, compute the table-supported statistic or association "
                "instead of refusing causal wording, preserving unit/scale and avoiding percent signs unless the final value is truly a percentage. For difference-from-average questions, compute the average over all numeric rows, then output value minus average on the same numeric scale unless a relative percentage is unambiguously required. If the requested derived column already exists, such as pop density (per km2), use that column directly and average every numeric row in it; do not recompute from raw population/area unless the derived column is absent. For highest-density versus average-density questions, output '<entity>, <highest density - average density>' on the density scale, not a relative percent. For anomaly questions, include only the minimal dominant anomalies, normally at most two when no count is specified, "
                "with no weaker extras and give semantic row identifier or 1-indexed data-row number plus abnormal column/value, using Episode <no> for TV episode anomalies, not generic index values or merely high/low secondary rows. "
                "Prefer anomalies where multiple related columns are jointly extreme; unknown/range formatting alone is weaker than rows with all relevant numeric columns extremely high or extremely low. "
                "For casualty/death tables, choose rows whose military, civilian, total deaths, wounded, and total casualties are collectively extreme high or collectively tiny; do not select rows merely because values are unknown or ranges. "
                "For descriptive analysis, cover column meanings, range, notable extrema/trends, and missing or unknown values. Do not generate charts, plots, or code."
            ),
            capability_tags=["solving", "calculation", "data-analysis"],
        ),
        "WF Answer Auditor": RoleCard(
            name="WF Answer Auditor",
            system_prompt=(
                "You are a TableBench answer auditor. Verify the solver's answer against the original table and question. "
                "Check row/column selection, arithmetic, units, percentage interpretation, ranking direction, and multi-part order. "
                "Check extrema differences, sorted medians, ordinal column labels, factor-list names, and missing qualifiers such as year or edition. "
                "Reject answers that return an extreme numeric value when the question asks for a date, time, row, or entity. "
                "Reject answers that use generic index columns such as Unnamed: 0 when a semantic identifier exists. "
                "Reject broad which-time answers that return only clock/origin time when the date/event row label is the benchmark target. "
                "Reject TV episode anomaly answers that omit Episode <no>. "
                "Reject anomaly answers that omit the row identifier/row number or abnormal column/value, use zero-based row numbers, or add weaker extra anomalies. "
                "If the answer is unsupported or malformed, output a corrected concise answer."
            ),
            capability_tags=["verification", "validation", "table-grounding"],
            role_type="validator",
        ),
        "WF Final Answer Formatter": RoleCard(
            name="WF Final Answer Formatter",
            system_prompt=(
                "You are the final TableBench formatter. Use the auditor's corrected answer when available. "
                "Output exactly one line and nothing else:\n"
                "Final Answer: <answer>\n"
                "Do not include reasoning, markdown, code, chart instructions, alternatives, or extra labels."
            ),
            capability_tags=["synthesis", "formatting", "structured-output"],
            communication_protocol="output exactly one line: Final Answer: <answer>",
            protected=True,
            role_type="aggregator",
        ),
    }
    edges = [
        ("WF Table Parser", "WF Evidence Selector"),
        ("WF Evidence Selector", "WF Table Solver"),
        ("WF Table Solver", "WF Answer Auditor"),
        ("WF Answer Auditor", "WF Final Answer Formatter"),
    ]
    return roles, edges, "WF Final Answer Formatter"


_WORKFLOW_REGISTRY: Dict[str, Any] = {
    "trip": _trip_workflow_dag,
    "meeting": _meeting_workflow_dag,
    "calendar": _calendar_workflow_dag,
    "naturalplan": _naturalplan_workflow_dag,
    "olympiadbench": _olympiad_workflow_dag,
    "tablebench": _tablebench_workflow_dag,
}



# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_workflow_baseline(
    tasks: List[Dict[str, Any]],
    benchmark: str,
    client: OpenRouterClient,
    config: SeroConfig,
) -> Dict[str, Any]:
    """
    Run the workflow baseline on a list of tasks.

    Uses a fixed-order linear chain (no PhaseA, no CreditEngine, no shuffle).
    Reuses the DAG execution engine from static_dag_mas.

    Returns summary dict with mean_score, std_score, records.
    """
    if benchmark not in _WORKFLOW_REGISTRY:
        raise ValueError(
            f"Workflow baseline is defined only for the configured workflow benchmarks "
            f"({', '.join(_NATURALPLAN_BENCHMARKS)}), got '{benchmark}'"
        )

    roles, edges, terminal = _WORKFLOW_REGISTRY[benchmark]()

    def _eval_task(task: Dict[str, Any]) -> Dict[str, Any]:
        agent_traces: Dict[str, str] = {}
        answer = ""
        try:
            answer, agent_traces = _execute_dag(
                task["prompt"], roles, edges, terminal, client, config,
            )
            raw_score = task["eval_fn"](answer)
            score = normalize_score(raw_score)
        except Exception as e:
            logger.error("Workflow error on %s: %s", task["id"], e)
            score, raw_score = 0.0, 0.0
        logger.info("[WF-%s] task=%s score=%.3f", benchmark, task["id"], score)

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
        "system": f"workflow_{benchmark}",
        "benchmark": benchmark,
        "pool": list(roles.keys()),
        "dag_edges": [(s, d) for s, d in edges],
        "n_tasks": len(scores),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
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
