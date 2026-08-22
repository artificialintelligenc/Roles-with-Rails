"""
OlympiadBench Adapter — text-only (TO) subset.

Loads open-ended text-only math and physics olympiad problems.
Uses the OlympiadBench AutoScoringJudge (sympy-based) for answer matching.

Data files used (text-only, open-ended, English):
  - OE_TO_maths_en_COMP.json   (competition math, English)
  - OE_TO_physics_en_COMP.json (competition physics, English)

Each item has:
  id, subfield, context, question, solution, final_answer,
  is_multiple_answer, unit, answer_type, error

Score:
  - Single-answer: 1.0 if correct, 0.0 otherwise.
  - Multi-answer (is_multiple_answer=True): partial credit = correct_count / total.
"""

import json
import os
import re
import random
import sys
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Attempt to import OlympiadBench scoring ────────────────────────────────────

def _get_judge():
    """Return an AutoScoringJudge instance, or None if unavailable."""
    olympiad_eval = os.path.join(
        os.path.dirname(__file__), "..", "..", "Benchmark", "OlympiadBench", "eval"
    )
    olympiad_eval = os.path.normpath(olympiad_eval)
    if olympiad_eval not in sys.path:
        sys.path.insert(0, olympiad_eval)
    try:
        from auto_scoring_judge import AutoScoringJudge
        return AutoScoringJudge()
    except Exception as e:
        logger.warning(
            "OlympiadBench judge unavailable: %s. "
            "Install sympy and antlr4-python3-runtime for accurate scoring. "
            "Falling back to string matching which may under-score correct answers.",
            e,
        )
        return None


def _normalize_for_fallback(s: str) -> str:
    """Normalize a math string for fallback comparison."""
    s = s.strip()
    # Strip surrounding dollar signs (gold answers are often $...$)
    s = s.strip("$")
    # Remove LaTeX formatting that doesn't affect value
    for tok in ("\\left", "\\right", "\\,", "\\;", "\\!", "\\quad", "\\qquad"):
        s = s.replace(tok, "")
    # Normalize whitespace
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _fallback_score(pred: str, gold_list: List[str]) -> float:
    """Simple string-match fallback when sympy judge is unavailable."""
    pred_clean = _normalize_for_fallback(pred)
    for g in gold_list:
        g_clean = _normalize_for_fallback(str(g))
        if not g_clean:
            continue
        if pred_clean == g_clean or pred_clean.endswith(g_clean):
            return 1.0
    return 0.0


def _strip_math_wrappers(text: str) -> str:
    """Strip lightweight LaTeX wrappers from a single math expression."""
    text = text.strip()
    if "\\boxed{" in text:
        text = _extract_boxed_content(text)
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1]
    return text.strip()


def _wrap_like(template: str, expr: str) -> str:
    """Wrap a generated equivalent expression in the same math delimiters as template."""
    template = template.strip()
    expr = expr.strip()
    if template.startswith("$") and template.endswith("$"):
        return f"${expr}$"
    return expr


def _extract_prompt_aliases(prompt: str) -> List[tuple[str, str]]:
    """Extract simple symbol definitions from prompt context answers.

    Example supported pattern: Context answer: \boxed{$\omega_{0}=\gamma B_{0}$}
    """
    aliases: List[tuple[str, str]] = []
    answer_blocks = re.findall(r"Context answer:\s*(.+?)(?=\n\s*\n|\Z)", prompt, flags=re.S)
    eq_re = re.compile(r"(\\?[A-Za-z]+(?:_\{[^}]+\})?)\s*=\s*([^\n$]+)")
    for block in answer_blocks:
        cleaned = _strip_math_wrappers(block)
        match = eq_re.search(cleaned)
        if not match:
            continue
        lhs = match.group(1).strip()
        rhs = match.group(2).strip().rstrip(".,;")
        if lhs and rhs and len(lhs) <= 32 and len(rhs) <= 120:
            aliases.append((lhs, rhs))
    return aliases


def _augment_gold_with_aliases(gold_answers: List[str], prompt: str) -> List[str]:
    """Augment gold answers with prompt-defined equivalent aliases.

    This fixes cases where the prompt explicitly defines a shorthand such as
    \omega_0 = \gamma B_0 but the stored gold answer only uses the expanded form.
    """
    aliases = _extract_prompt_aliases(prompt)
    if not aliases:
        return gold_answers

    expanded: List[str] = []
    seen = set()
    for gold in gold_answers:
        gold_str = str(gold).strip()
        if gold_str and gold_str not in seen:
            expanded.append(gold_str)
            seen.add(gold_str)

        core = _strip_math_wrappers(gold_str)
        for lhs, rhs in aliases:
            variants = []
            if rhs in core:
                variants.append(core.replace(rhs, lhs))
            if lhs in core:
                variants.append(core.replace(lhs, rhs))
            for variant in variants:
                wrapped = _wrap_like(gold_str, variant)
                if wrapped and wrapped not in seen:
                    expanded.append(wrapped)
                    seen.add(wrapped)

    return expanded or gold_answers


def _extract_boxed_content(latex_str: str) -> str:
    """Extract content from \\boxed{...} using stack-based brace matching.

    Handles arbitrarily nested braces (e.g. \\boxed{\\frac{1}{2}}).
    If multiple \\boxed{} are found, joins them with commas.
    Falls back to dollar-sign expressions or the whole string.
    """
    boxed_matches = list(re.finditer(r'\\boxed\{', latex_str))
    results = []

    for match in boxed_matches:
        start_index = match.end()
        end_index = start_index
        stack = 1

        while stack > 0 and end_index < len(latex_str):
            if latex_str[end_index] == '{':
                stack += 1
            elif latex_str[end_index] == '}':
                stack -= 1
            end_index += 1

        if stack == 0:
            content = latex_str[start_index:end_index - 1]
            results.append(content)

    if results:
        return ",".join(results)

    # Fallback: extract dollar-sign expressions from last non-empty line
    last_line = latex_str.strip().split("\n")[-1]
    dollar_answers = re.findall(r"\$(.*?)\$", last_line)
    if dollar_answers:
        return ",".join(dollar_answers)

    return latex_str


def _extract_final_answer(response: str) -> str:
    """
    Extract the final answer from an LLM response.
    Uses stack-based \\boxed{} extraction, then fallback heuristics.
    Does NOT strip \\boxed{} for judge — returns raw extracted content.
    """
    # Try \\boxed{} with proper nesting support
    boxed_matches = list(re.finditer(r'\\boxed\{', response))
    if boxed_matches:
        # Use the last \\boxed{} (most likely the final answer)
        match = boxed_matches[-1]
        start_index = match.end()
        end_index = start_index
        stack = 1
        while stack > 0 and end_index < len(response):
            if response[end_index] == '{':
                stack += 1
            elif response[end_index] == '}':
                stack -= 1
            end_index += 1
        if stack == 0:
            return response[start_index:end_index - 1].strip()

    # "Final answer: ..." or "Answer: ..."
    m = re.search(r'(?:final\s+answer|answer)\s*[:=]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # "The answer is ..."
    m = re.search(r'the answer is\s+(.+?)(?:\.|$)', response, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Last non-empty line (fallback)
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else response.strip()[:100]


def _unwrap_multi_answer_container(text: str) -> str:
    """Strip one outer tuple/list-like wrapper for multi-answer matching."""
    text = text.strip()
    if len(text) < 2:
        return text
    wrappers = {
        '(': ')',
        '[': ']',
        '{': '}',
    }
    closing = wrappers.get(text[0])
    if closing is None or text[-1] != closing:
        return text

    inner = text[1:-1].strip()
    if ',' not in inner:
        return text
    return inner


def _split_multi_answer_text(text: str, judge) -> List[str]:
    """Split a multiple-answer string conservatively, tolerating tuple wrappers."""
    text = text.strip()
    if not text:
        return []

    for candidate in (text, _unwrap_multi_answer_container(text)):
        parts = [part.strip() for part in judge.split_by_comma(candidate) if part.strip()]
        if len(parts) > 1:
            return parts
    return [text]


# ── Answer type description for prompts ───────────────────────────────────────

_ANSWER_TYPE_HINTS = {
    "Numerical": "a numerical value",
    "Expression": "a mathematical expression",
    "Equation": "an equation",
    "Interval": "an interval (e.g. [a, b) or (a, b])",
    "Tuple": "a tuple of values",
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def _olympiad_task_key(item: dict, subject: str) -> str:
    """Return the stable task key used across loading, splits, and results."""
    return f"olympiad_{subject}_{item.get('id', hash(item.get('question', '')) % 100000)}"

def load_olympiad_tasks(
    benchmark_dir: str,
    subjects: Optional[List[str]] = None,
    max_tasks: Optional[int] = 40,
    seed: int = 42,
    text_only: bool = True,
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load olympiad tasks from OlympiadBench.

    Returns list of task dicts: {id, prompt, eval_fn, subject, subfield, gold_answer}.
    Only loads problems without images (text-only) by default.
    """
    data_dir = os.path.join(benchmark_dir, "OlympiadBench", "OlympiadBench_Dataset", "data")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"OlympiadBench data not found at {data_dir}")

    if subjects is None:
        subjects = ["maths", "physics"]

    prefix = "OE_TO" if text_only else "OE"

    all_items = []
    for subj in subjects:
        fname = f"{prefix}_{subj}_en_COMP.json"
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            logger.warning("File not found: %s", fpath)
            continue
        with open(fpath) as f:
            items = json.load(f)
        for item in items:
            item["_subject"] = subj
        all_items.extend(items)

    if not all_items:
        raise FileNotFoundError(f"No OlympiadBench data loaded from {data_dir}")

    # Filter: skip items with images referenced in question
    text_items = []
    for item in all_items:
        q = item.get("question", "")
        ctx = item.get("context", "") or ""
        if any(x in (q + ctx).lower() for x in ["figure", "diagram", "image", "shown below", "shown above"]):
            continue
        text_items.append(item)

    include_key_list = list(include_keys) if include_keys is not None else None
    include_key_set = set(include_key_list) if include_key_list is not None else None
    exclude_key_set = set(exclude_keys or [])

    keyed_items = []
    for item in text_items:
        subject = item.get("_subject", "maths")
        task_key = _olympiad_task_key(item, subject)
        if include_key_set is not None:
            if task_key in include_key_set:
                keyed_items.append((task_key, item))
            continue
        if task_key in exclude_key_set:
            continue
        keyed_items.append((task_key, item))

    if include_key_list is not None:
        keyed_by_key = {task_key: item for task_key, item in keyed_items}
        text_items = [keyed_by_key[key] for key in include_key_list if key in keyed_by_key]
    else:
        rng = random.Random(seed)
        rng.shuffle(keyed_items)
        text_items = [item for _, item in keyed_items]
        if max_tasks is not None:
            text_items = text_items[:max_tasks]

    judge = _get_judge()
    tasks = []
    for item in text_items:
        task = _make_task(item, judge)
        if task is not None:
            tasks.append(task)

    subject_desc = "+".join(subjects)
    mode_desc = "text-only" if text_only else "multimodal"
    logger.info("Loaded %d olympiad tasks (%s, %s)", len(tasks), subject_desc, mode_desc)
    return tasks


def _parse_error_field(error_val) -> Any:
    """Parse the error field from data, handling comma-separated multi-precision."""
    if not error_val:
        return 1e-8
    error_str = str(error_val)
    if ',' in error_str:
        parts = error_str.split(',')
        return [float(p) if p.strip() else 1e-8 for p in parts]
    return float(error_str)


def _make_task(item: dict, judge) -> Optional[Dict[str, Any]]:
    """Build a task dict from one OlympiadBench item."""
    question = item.get("question", "").strip()
    context = (item.get("context") or "").strip()
    final_answer = item.get("final_answer", [])
    answer_type = item.get("answer_type", "Numerical")
    unit = item.get("unit") or ""
    error = _parse_error_field(item.get("error"))
    is_multiple = item.get("is_multiple_answer", False)
    subject = item.get("_subject", "maths")

    if not question or not final_answer:
        return None

    # Build prompt with answer type hints (matching original OlympiadBench)
    subject_label = "Mathematics" if subject == "maths" else "Physics"
    prompt_parts = []
    prompt_parts.append(
        f"The following is an open-ended problem from an International {subject_label} competition."
    )

    # Answer type hint
    type_hint = _ANSWER_TYPE_HINTS.get(answer_type, "")
    if type_hint and answer_type != "Need_human_evaluate":
        prompt_parts.append(f"The answer is {type_hint}.")

    if context:
        prompt_parts.append(f"\nContext: {context}")
    prompt_parts.append(f"\nProblem: {question}")
    if unit:
        prompt_parts.append(f"(Express your answer in: {unit}. Do NOT include the unit inside \\boxed{{}}.)")

    # Multi-answer format guidance
    if is_multiple:
        prompt_parts.append(
            "\nThis problem has multiple answers. Please provide all answers "
            "separated by commas inside a single \\boxed{}, e.g. \\boxed{a, b, c}."
        )
    prompt_parts.append(
        "\nPlease solve this problem step by step. "
        "Express your final answer as \\boxed{<your answer>}."
    )
    prompt = "\n".join(prompt_parts)
    final_answer = _augment_gold_with_aliases(final_answer, prompt)

    item_id = _olympiad_task_key(item, subject)

    def make_eval_fn(gold: List[str], ans_type: str, err, is_mult: bool, j):
        def eval_fn(response: str) -> float:
            return _score_olympiad(response, gold, ans_type, err, is_mult, j)
        return eval_fn

    return {
        "id": item_id,
        "prompt": prompt,
        "eval_fn": make_eval_fn(final_answer, answer_type, error, is_multiple, judge),
        "subject": subject,
        "subfield": item.get("subfield", ""),
        "gold_answer": final_answer,
        "answer_type": answer_type,
    }


def _score_olympiad(
    response: str,
    gold: List[str],
    answer_type: str,
    error,
    is_multiple: bool,
    judge,
) -> float:
    """Score an olympiad response.

    For single-answer problems: binary 1.0/0.0.
    For multi-answer (is_multiple_answer=True): partial credit = matched/total.

    The full response is passed to the judge which internally handles
    \\boxed{} extraction via its preprocess() method.
    """
    if judge is None:
        pred = _extract_final_answer(response)
        return _fallback_score(pred, gold)

    try:
        # gold is typically a one-element list containing all answers
        # (possibly comma-separated inside the string).
        # Pass the full response as expression2 so the judge's preprocess()
        # can extract \\boxed{} content with proper nesting support.
        gold_str = str(gold[0])
        gold_parts = [str(g).strip() for g in gold if str(g).strip()]
        if is_multiple and len(gold_parts) == 1:
            split_gold = _split_multi_answer_text(gold_parts[0], judge)
            if len(split_gold) > 1:
                gold_parts = split_gold

        # Handle Tuple type: no precision
        if 'Tuple' in answer_type:
            result = judge.judge(response, gold_str)
            return 1.0 if result else 0.0

        # For multi-answer problems with multiple gold entries,
        # compute partial credit: correct_count / total.
        #
        # Strategy:
        #   1. Try full match: join golds with comma → let judge do its
        #      internal split_by_comma + pairing against the response.
        #      If that passes, all answers match → score = 1.0.
        #   2. Fallback to element-wise partial credit: split pred by
        #      top-level commas, then greedily pair each gold with a pred
        #      element via judge.is_equal (bypassing the length check).
        if is_multiple and len(gold_parts) > 1:
            # Attempt 1: full match (gold as comma-joined string)
            gold_combined = ", ".join(gold_parts)
            for gold_candidate in dict.fromkeys([gold_str, gold_combined]):
                try:
                    if judge.judge(response, gold_candidate, precision=error):
                        return 1.0
                except Exception:
                    continue

            # Attempt 2: element-wise partial credit
            pred_raw = _extract_final_answer(response)
            pred_parts = _split_multi_answer_text(pred_raw, judge)
            remaining = list(pred_parts)  # mutable copy for greedy removal
            correct = 0
            per_error = error if isinstance(error, list) else [error] * len(gold_parts)
            for i, g in enumerate(gold_parts):
                g_str = str(g).strip()
                judge.precision = per_error[min(i, len(per_error) - 1)]
                matched = False
                for p in remaining:
                    try:
                        if judge.is_equal(g_str, p.strip()):
                            remaining.remove(p)
                            matched = True
                            break
                    except Exception:
                        continue
                if matched:
                    correct += 1
            return correct / len(gold_parts)

        # Standard single-answer: accept any equivalent gold variant.
        for gold_candidate in gold:
            try:
                if judge.judge(response, str(gold_candidate), precision=error):
                    return 1.0
            except Exception:
                continue
        return 0.0

    except Exception as e:
        logger.debug("Judge exception: %s, falling back to string match", e)
        pred = _extract_final_answer(response)
        return _fallback_score(pred, gold)


# ── SC Canonical Answer Extraction ────────────────────────────────────────────

def extract_canonical_answer(response: str) -> str:
    """Extract a canonical answer string for SC majority voting.

    Delegates to _extract_final_answer (stack-based \\boxed{} extraction with
    fallback heuristics). Already deterministic.
    """
    return _extract_final_answer(response)


# ── Seed role pools for olympiad reasoning ────────────────────────────────────

OLYMPIAD_SEED_ROLES_MATHS = None   # imported from role_card to avoid circular import
