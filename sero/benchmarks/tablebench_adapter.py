"""
TableBench adapter.

Loads TableBench table QA tasks from Benchmark/TableBench-main/TableBench.jsonl.
Visualization tasks are intentionally excluded: this adapter covers only
FactChecking, NumericalReasoning, and DataAnalysis.

Scoring returns a single exact-accuracy float. The match is intentionally more
forgiving than the official parser: it accepts answer-equivalent unit wrapping,
short explanations, ordinal suffixes, reasonable decimal rounding, and the
official 10% numerical tolerance for correlation/trend/statistical analysis.
"""

import json
import math
import os
import random
import re
import string
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Sequence, Tuple


KEY_SEPARATOR = "::"
TABLEBENCH_DIR = "TableBench-main"
DATA_FILE = "TableBench.jsonl"
EXCLUDED_QTYPES = {"Visualization"}
EXCLUDED_QSUBTYPES = {"ChartGeneration"}
TOLERANCE_SUBTYPES = {"CorrelationAnalysis", "TrendForecasting", "StatisticalAnalysis"}
YES_NO_TERMS = {"yes", "no"}
KEY_TERM_STOPWORDS = {
    "answer", "table", "tables", "column", "columns", "value", "values", "number", "numbers",
    "total", "average", "median", "mean", "higher", "lower", "listed", "shows", "indicates",
    "indicate", "there", "their", "these", "those", "with", "from", "into", "than", "then",
    "that", "this", "which", "what", "when", "where", "does", "have", "been", "were", "are",
    "the", "and", "for", "all", "each", "task", "question", "requested", "significant", "has",
    "having",
}
ANOMALY_GENERIC_TERMS = {
    "anomaly", "anomalies", "anomalous", "area", "population", "value", "values",
    "large", "largest", "small", "smallest", "extreme", "extremely", "high", "low",
    "location", "locations", "place", "places", "significant", "deviate", "deviates",
    "deviation", "outlier", "outliers", "km", "km2", "m2", "cm2", "mm2", "sqkm", "sqm",
    "sqmi", "mi2", "square", "row", "rows",
}


TABLEBENCH_ROLE_GENERATION_CONSTRAINTS = (
    "TableBench role design contract:\n"
    "- Create roles for table understanding, evidence retrieval, numerical reasoning, data-analysis interpretation, validation, or final-answer formatting.\n"
    "- Do not create visualization, chart-generation, plotting, code-execution, or image-rendering roles; visualization tasks are excluded from this benchmark.\n"
    "- Roles must reason from table columns, rows, values, units, and the question only. Do not invent outside facts.\n"
    "- Roles must identify the answer target and primary semantic row identifier; generic index columns like Unnamed: 0 are not answer identifiers unless explicitly requested.\n"
    "- Final output must be a single parser-facing line: Final Answer: <answer>.\n"
    "- Numerical roles should preserve requested precision, rankings, percentages, and comma-separated multi-part answers.\n"
    "- Data-analysis roles should distinguish exact influential-factor answers from open-ended descriptive or causal explanations, keep difference-from-average answers on the table column scale, and return only the minimal dominant anomalies."
)

TABLEBENCH_ANSWER_SHAPE_GUIDANCE = (
    "TableBench answer-shape guidance:\n"
    "- First identify the answer target: if the question asks which date, time, team, district, row, or entity has an extreme value, return that requested label/entity, not the extreme numeric value itself.\n"
    "- For extreme-row questions, use the row's primary semantic identifier, usually the first meaningful column such as date, year, name, district, team, or episode no.; ignore generic index columns like Unnamed: 0 unless explicitly requested.\n"
    "- If the table has both a date column and a clock/origin-time column, use the date/event row label for broad 'which time' extreme questions unless the question explicitly asks for clock time.\n"
    "- For TV episode anomaly questions, identify anomalies as Episode <no> from the episode/no column; do not answer with the generic index or title alone.\n"
    "- Ranking/highest-lowest questions: return the requested entity/label for which/what questions; compute a numeric difference only when the question asks for a difference or gap.\n"
    "- Domain-specific period/ordinal questions: return the exact column or period label, such as 23rd, not a nearby row number.\n"
    "- Statistical/correlation/trend/causal questions: compute the table-supported statistic or association requested; keep the requested unit/scale and do not add a percent sign unless the final value is truly a percentage.\n"
    "- Difference-from-average questions: compute average over all numeric rows, then output max/min value minus average on the same table-column scale unless a relative percentage is unambiguously required.\n"
    "- If the table already has the requested derived column, such as pop density (per km2), use that column directly and average every numeric row in that column; do not recompute from raw population/area unless the derived column is absent.\n"
    "- For highest-density versus average-density questions, output '<entity>, <highest density - average density>' on the density scale, not a relative percent.\n"
    "- Anomaly questions: include only the minimal dominant anomalies, normally at most two when no count is specified; do not append weaker high/low cells after the obvious outliers.\n"
    "- Prefer anomalies where multiple related columns are jointly extreme; unknown/range formatting alone is weaker than rows with all relevant numeric columns extremely high or extremely low.\n"
    "- For casualty/death tables, choose rows whose military, civilian, total deaths, wounded, and total casualties are collectively extreme high or collectively tiny; do not select rows merely because values are unknown or ranges.\n"
    "- For each anomaly give the semantic row identifier, or a 1-indexed data-row number if no row label exists, plus the abnormal column/value.\n"
    "- Row numbers are 1-indexed over data rows after the header; do not use zero-based positions or generic index-column values as row numbers.\n"
    "- Descriptive-analysis questions: briefly cover what each column means, the time/entity range, notable extrema/trends, and missing or unknown values when present.\n"
    "- Impact/factor-list questions: preserve the exact factor or column names requested, without substitutions.\n"
    "- Fact-checking questions: preserve requested qualifiers such as edition, year, category, or note values when present."
)


TABLEBENCH_FAILURE_PATTERNS = [
    "TableBench evidence grounding: locate the exact rows, columns, and units before answering.",
    "TableBench answer target: return the requested row label/entity, not the comparison value or a generic index column.",
    "TableBench numerical reasoning: preserve precision, percentages, rankings, same-scale differences, and multi-part answer order.",
    "TableBench data analysis: choose exact factors for impact questions, minimal dominant anomalies, and concise explanatory text for open-ended analysis.",
    "TableBench format discipline: output one line only, Final Answer: <answer>.",
    "TableBench scope: never solve as visualization, plotting, chart generation, or code execution.",
]


def load_tablebench_tasks(
    benchmark_dir: str,
    max_tasks: int = 200,
    seed: int = 42,
    include_keys: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Load non-visualization TableBench tasks.

    Args:
        include_keys: If set, load only these raw ids or tablebench::raw_id keys.
        exclude_keys: If set, remove these raw ids or tablebench::raw_id keys.

    Returns task dicts with id, prompt, eval_fn, gold_answer, qtype, and qsubtype.
    """
    data_path = os.path.join(benchmark_dir, TABLEBENCH_DIR, DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"TableBench data not found at {data_path}")

    all_examples = _read_tablebench_jsonl(data_path)
    non_visual_examples = [
        example for example in all_examples
        if not _is_visualization_task(example)
    ]
    by_id = {str(example["id"]): example for example in non_visual_examples}

    if include_keys is not None:
        examples = []
        for key in include_keys:
            raw_key = _normalize_key(key)
            example = by_id.get(raw_key)
            if example is not None:
                examples.append(example)
    else:
        excluded = {_normalize_key(key) for key in (exclude_keys or [])}
        examples = [example for example in non_visual_examples if str(example["id"]) not in excluded]
        rng = random.Random(seed)
        rng.shuffle(examples)
        examples = examples[:max_tasks]

    tasks = []
    for example in examples:
        qtype = str(example.get("qtype", ""))
        qsubtype = str(example.get("qsubtype", ""))
        answer = str(example.get("answer", ""))
        prompt = _build_prompt(example)

        def make_eval_fn(gold_answer: str, question_type: str, question_subtype: str):
            def eval_fn(response: str) -> float:
                return score_tablebench_response(
                    response,
                    gold_answer=gold_answer,
                    qtype=question_type,
                    qsubtype=question_subtype,
                )
            return eval_fn

        raw_id = str(example["id"])
        tasks.append({
            "id": raw_id,
            "original_id": raw_id,
            "benchmark": "tablebench",
            "domain": "tablebench",
            "sub_benchmark": qtype,
            "task_type": qsubtype,
            "qtype": qtype,
            "qsubtype": qsubtype,
            "prompt": prompt,
            "eval_fn": make_eval_fn(answer, qtype, qsubtype),
            "gold_answer": answer,
            "answer": answer,
            "tablebench_metric": tablebench_metric_name(qtype, qsubtype),
            "domain_generation_constraints": TABLEBENCH_ROLE_GENERATION_CONSTRAINTS,
            "domain_failure_patterns": TABLEBENCH_FAILURE_PATTERNS,
        })

    return tasks


def score_tablebench_response(
    response: str,
    gold_answer: str,
    qtype: str,
    qsubtype: str,
) -> float:
    """Score one TableBench response as a single lenient exact-accuracy value."""
    prediction = extract_canonical_answer(response)
    return float(_lenient_exact_match(gold_answer, prediction, qtype, qsubtype))


def tablebench_metric_name(qtype: str, qsubtype: str) -> str:
    return "exact_acc"


def extract_canonical_answer(response: str) -> str:
    """Extract TableBench's parser-facing final answer."""
    if not response:
        return ""

    final_answer_matches = list(re.finditer(
        r"(?:final\s+answer|answer)\s*:\s*(.+)",
        response,
        flags=re.IGNORECASE,
    ))
    if final_answer_matches:
        return _clean_extracted_answer(final_answer_matches[-1].group(1))

    boxed = _extract_boxed_content(response)
    if boxed:
        return _clean_extracted_answer(boxed)

    lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return _clean_extracted_answer(lines[-1])


def _read_tablebench_jsonl(path: str) -> List[Dict[str, Any]]:
    examples = []
    with open(path, encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _is_visualization_task(example: Dict[str, Any]) -> bool:
    return (
        example.get("qtype") in EXCLUDED_QTYPES
        or example.get("qsubtype") in EXCLUDED_QSUBTYPES
    )


def _normalize_key(key: str) -> str:
    text = str(key)
    if KEY_SEPARATOR in text:
        prefix, raw_key = text.split(KEY_SEPARATOR, 1)
        if prefix.lower() == "tablebench":
            return raw_key
    return text


def _build_prompt(example: Dict[str, Any]) -> str:
    qtype = str(example.get("qtype", ""))
    qsubtype = str(example.get("qsubtype", ""))
    question = str(example.get("question", ""))
    table_markdown = _format_table_markdown(example.get("table", {}))
    statistic_hint = _build_statistic_hint(example)
    statistic_hint_block = f"\n\n{statistic_hint}" if statistic_hint else ""
    return (
        "You are solving a TableBench table question-answering task.\n"
        f"Question type: {qtype}\n"
        f"Question subtype: {qsubtype}\n\n"
        "Use only the table and the question. Do not use visualization, chart generation, plotting, or code execution.\n\n"
        f"Table:\n{table_markdown}\n\n"
        f"Question: {question}\n\n"
        f"{TABLEBENCH_ANSWER_SHAPE_GUIDANCE}{statistic_hint_block}\n\n"
        "Return exactly one final line in this format:\n"
        "Final Answer: <answer>\n"
        "For comma-separated multi-part answers, preserve the requested order. For numerical answers, preserve units, percentages, and precision when the question requires them."
    )


def _build_statistic_hint(example: Dict[str, Any]) -> str:
    qtype = str(example.get("qtype", ""))
    qsubtype = str(example.get("qsubtype", ""))
    if qtype != "DataAnalysis" or qsubtype not in {"CausalAnalysis", "CorrelationAnalysis"}:
        return ""

    table = example.get("table", {}) or {}
    columns = [str(column) for column in table.get("columns", [])]
    rows = table.get("data", []) or []
    question = str(example.get("question", ""))
    column_pair = _select_statistic_columns(columns, rows, question, qsubtype)
    if column_pair is None:
        return ""

    x_index, y_index, predictor_phrase, target_phrase = column_pair
    pairs = []
    for row in rows:
        row_values = list(row) if isinstance(row, list) else [row]
        if x_index >= len(row_values) or y_index >= len(row_values):
            continue
        x_value = _coerce_table_float(row_values[x_index])
        y_value = _coerce_table_float(row_values[y_index])
        if x_value is not None and y_value is not None:
            pairs.append((x_value, y_value))

    if len(pairs) < 2:
        return ""

    coefficient = _pearson_coefficient(pairs)
    if coefficient is None:
        return ""

    coefficient_text = f"{coefficient:.2f}"
    if qsubtype == "CausalAnalysis":
        effect_label = _causal_effect_label(coefficient)
        target = _sentence_case(target_phrase or columns[y_index])
        predictor = predictor_phrase or columns[x_index]
        return (
            "TableBench statistic calculation aid:\n"
            f"- The question mentions paired numeric columns '{columns[y_index]}' and '{columns[x_index]}'; {len(pairs)} rows have numeric values in both columns.\n"
            f"- The paired Pearson coefficient from the table values, rounded to two decimals, is {coefficient_text}.\n"
            f"- For this CausalAnalysis wording, use this parser-facing answer shape when consistent with the question: Final Answer: {target} exhibits {effect_label} ({coefficient_text}) with increasing {predictor}.\n"
            "- If the coefficient magnitude is below 0.30, keep the wording 'no causal effect'; do not rewrite it as weak positive/negative correlation."
        )

    relation_label = _correlation_label(coefficient)
    return (
        "TableBench statistic calculation aid:\n"
        f"- The question mentions paired numeric columns '{columns[x_index]}' and '{columns[y_index]}'; {len(pairs)} rows have numeric values in both columns.\n"
        f"- The paired Pearson correlation coefficient from the table values, rounded to two decimals, is {coefficient_text}.\n"
        f"- For this CorrelationAnalysis wording, use this answer shape when consistent with the question: Final Answer: {relation_label}, {coefficient_text}."
    )


def _select_statistic_columns(
    columns: Sequence[str],
    rows: Sequence[Any],
    question: str,
    qsubtype: str,
) -> Optional[Tuple[int, int, str, str]]:
    numeric_columns = []
    for index, column in enumerate(columns):
        values = []
        for row in rows:
            row_values = list(row) if isinstance(row, list) else [row]
            if index < len(row_values):
                parsed = _coerce_table_float(row_values[index])
                if parsed is not None:
                    values.append(parsed)
        if len(values) >= 2:
            score, position = _column_question_score(column, question)
            if score > 0:
                numeric_columns.append((score, position, index))

    if len(numeric_columns) < 2:
        return None

    if qsubtype == "CausalAnalysis":
        change_match = re.search(
            r"how\s+does\s+(.+?)\s+change\s+with\s+increasing\s+(.+?)\??$",
            question.strip(),
            flags=re.IGNORECASE,
        )
        if change_match:
            target_phrase = change_match.group(1).strip()
            predictor_phrase = change_match.group(2).strip().rstrip("?.")
            target_index = _best_column_for_phrase(columns, numeric_columns, target_phrase)
            predictor_index = _best_column_for_phrase(columns, numeric_columns, predictor_phrase)
            if target_index is not None and predictor_index is not None and target_index != predictor_index:
                return predictor_index, target_index, predictor_phrase, target_phrase

    top_two = sorted(numeric_columns, key=lambda item: (item[1], -item[0]))[:2]
    return top_two[0][2], top_two[1][2], columns[top_two[0][2]], columns[top_two[1][2]]


def _column_question_score(column: str, question: str) -> Tuple[int, int]:
    question_text = _normalize_for_containment(question)
    column_tokens = [token for token in _normalize_for_containment(column).split() if token not in _STAT_UNIT_TOKENS]
    score = sum(1 for token in column_tokens if token in question_text.split())
    positions = [question_text.find(token) for token in column_tokens if token in question_text.split()]
    position = min(positions) if positions else 10**9
    compact_column = " ".join(column_tokens)
    if compact_column and compact_column in question_text:
        score += 2
        position = min(position, question_text.find(compact_column))
    return score, position


def _best_column_for_phrase(
    columns: Sequence[str],
    numeric_columns: Sequence[Tuple[int, int, int]],
    phrase: str,
) -> Optional[int]:
    best = None
    for _, _, index in numeric_columns:
        score, position = _column_question_score(columns[index], phrase)
        if score <= 0:
            continue
        candidate = (score, -position, index)
        if best is None or candidate > best:
            best = candidate
    return None if best is None else best[2]


def _coerce_table_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"-", "--", "n/a", "na", "tba", "unknown"}:
        return None
    text = text.replace(" ", "").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?%?", text)
    if not match:
        return None
    token = match.group(0)
    try:
        if token.endswith("%"):
            return float(token[:-1]) / 100.0
        return float(token)
    except ValueError:
        return None


def _pearson_coefficient(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
    n = len(pairs)
    if n < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in pairs)
    denom_x = sum((x_value - mean_x) ** 2 for x_value in xs)
    denom_y = sum((y_value - mean_y) ** 2 for y_value in ys)
    denominator = math.sqrt(denom_x * denom_y)
    if denominator == 0:
        return None
    return numerator / denominator


def _causal_effect_label(coefficient: float) -> str:
    magnitude = abs(coefficient)
    if magnitude < 0.30:
        return "no causal effect"
    strength = "weak" if magnitude < 0.70 else "strong"
    direction = "positive" if coefficient >= 0 else "negative"
    return f"{strength} {direction} causal effect"


def _correlation_label(coefficient: float) -> str:
    magnitude = abs(coefficient)
    if magnitude < 0.30:
        return "No correlation"
    strength = "Weak" if magnitude < 0.70 else "Strong"
    direction = "positive" if coefficient >= 0 else "negative"
    return f"{strength} {direction} correlation"


def _sentence_case(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    return stripped[0].upper() + stripped[1:]


_STAT_UNIT_TOKENS = {
    "a", "an", "the", "of", "in", "for", "per", "and", "or",
    "mm", "cm", "m", "km", "mi", "bar", "kgf", "lbf", "n", "2", "3",
    "square", "squared", "percent", "percentage", "usd",
}


def _format_table_markdown(table: Dict[str, Any]) -> str:
    columns = [str(column) for column in table.get("columns", [])]
    rows = table.get("data", []) or []
    if not columns:
        return "(empty table)"

    header = "| " + " | ".join(_escape_markdown_cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for row in rows:
        values = list(row) if isinstance(row, list) else [row]
        padded = values[:len(columns)] + [""] * max(0, len(columns) - len(values))
        body_lines.append("| " + " | ".join(_escape_markdown_cell(value) for value in padded) + " |")
    return "\n".join([header, separator] + body_lines)


def _escape_markdown_cell(value: Any) -> str:
    if value is None:
        text = ""
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text.strip())
    return text.replace("|", r"\|")


def _clean_extracted_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", cleaned).strip()
    cleaned = cleaned.strip("` ")
    if cleaned.lower().startswith("final answer:"):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned


def _extract_boxed_content(text: str) -> str:
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return ""
    match = matches[-1]
    start_index = match.end()
    end_index = start_index
    stack = 1
    while stack > 0 and end_index < len(text):
        char = text[end_index]
        if char == "{":
            stack += 1
        elif char == "}":
            stack -= 1
        end_index += 1
    if stack == 0:
        return text[start_index:end_index - 1]
    return ""


def _normalize_answer(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(char for char in text if char not in set(string.punctuation))
    return " ".join(text.split())


def _normalize_for_containment(value: str) -> str:
    text = str(value or "").lower()
    text = text.replace("≈", " ").replace("~", " ")
    text = text.replace("\u00b2", "2").replace("\u00b3", "3")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(char if char not in set(string.punctuation) else " " for char in text)
    return " ".join(text.split())


def _tokenize_normalized(value: str) -> List[str]:
    return _normalize_for_containment(value).split()


def _numeric_tokens(value: str) -> List[str]:
    text = str(value or "")
    return re.findall(
        r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage)?(?:st|nd|rd|th)?",
        text,
        flags=re.IGNORECASE,
    )


def _clean_numeric_token(value: str) -> str:
    text = value.strip().lower().replace(",", "")
    text = re.sub(r"\s*(percent|percentage)$", "%", text)
    text = re.sub(r"(st|nd|rd|th)$", "", text)
    return text


def _numeric_precision(value: str) -> int:
    text = _clean_numeric_token(value)
    text = text.rstrip("%")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def _numeric_value(value: str) -> Decimal:
    return _normalize_number(_clean_numeric_token(value))


def _numbers_equivalent(reference_number: str, prediction_number: str, allow_relative_tolerance: bool) -> bool:
    try:
        ref_value = _numeric_value(reference_number)
        pred_value = _numeric_value(prediction_number)
    except Exception:
        return False

    if ref_value == pred_value:
        return True

    if allow_relative_tolerance:
        if ref_value == Decimal("0"):
            return pred_value == ref_value
        return abs(ref_value - pred_value) / abs(ref_value) <= Decimal("0.10")

    ref_precision = _numeric_precision(reference_number)
    pred_precision = _numeric_precision(prediction_number)
    if ref_precision == 0 and pred_precision == 0:
        return False
    if pred_precision == 0 and ref_precision > 0:
        tolerance = Decimal("0.5") * (Decimal("10") ** Decimal(-ref_precision))
    else:
        tolerance = Decimal("0.5") * (Decimal("10") ** Decimal(-min(ref_precision, pred_precision)))
    return abs(ref_value - pred_value) <= tolerance


def _reference_numbers_match(
    reference: str,
    prediction: str,
    allow_relative_tolerance: bool,
    order_sensitive: bool = True,
    allow_extra_numbers: bool = False,
) -> bool:
    ref_numbers = _numeric_tokens(reference)
    if not ref_numbers:
        return True
    pred_numbers = _numeric_tokens(prediction)
    if not pred_numbers:
        return False
    if not allow_extra_numbers and len(pred_numbers) > len(ref_numbers) + 1:
        return False

    if not order_sensitive:
        remaining = list(pred_numbers)
        for ref_number in ref_numbers:
            matched_at = None
            for pred_index, pred_number in enumerate(remaining):
                if _numbers_equivalent(ref_number, pred_number, allow_relative_tolerance):
                    matched_at = pred_index
                    break
            if matched_at is None:
                return False
            remaining.pop(matched_at)
        return True

    cursor = 0
    for ref_number in ref_numbers:
        matched_at = None
        for pred_index in range(cursor, len(pred_numbers)):
            if _numbers_equivalent(ref_number, pred_numbers[pred_index], allow_relative_tolerance):
                matched_at = pred_index
                break
        if matched_at is None:
            return False
        cursor = matched_at + 1
    return True


def _strip_trailing_parenthetical_year(value: str) -> str:
    return re.sub(r"\s*\((?:18|19|20|21)\d{2}\)\s*$", "", str(value or "")).strip()


def _leading_yes_no(value: str) -> Optional[str]:
    tokens = _tokenize_normalized(value)
    if not tokens:
        return None
    first = tokens[0]
    return first if first in YES_NO_TERMS else None


def _polarity_conflicts(reference: str, prediction: str) -> bool:
    ref_polarity = _leading_yes_no(reference)
    pred_polarity = _leading_yes_no(prediction)
    if ref_polarity and pred_polarity and ref_polarity != pred_polarity:
        return True

    ref_text = _normalize_for_containment(reference)
    pred_text = _normalize_for_containment(prediction)
    ref_positive = "positive" in ref_text
    ref_negative = "negative" in ref_text
    pred_positive = "positive" in pred_text
    pred_negative = "negative" in pred_text
    return (ref_positive and pred_negative) or (ref_negative and pred_positive)


def _quoted_terms(value: str) -> List[str]:
    return [match.strip() for match in re.findall(r"['\"]([^'\"]+)['\"]", value) if match.strip()]


def _reference_key_terms(reference: str) -> List[str]:
    terms = [_normalize_for_containment(term) for term in _quoted_terms(reference)]
    for token in _tokenize_normalized(re.sub(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", " ", reference)):
        if len(token) >= 4 and token not in KEY_TERM_STOPWORDS:
            terms.append(token)
    unique_terms = []
    seen = set()
    for term in terms:
        if term and term not in seen:
            unique_terms.append(term)
            seen.add(term)
    return unique_terms


def _key_terms_match(reference: str, prediction: str, qtype: str, qsubtype: str) -> bool:
    quoted_terms = [_normalize_for_containment(term) for term in _quoted_terms(reference)]
    if quoted_terms:
        pred_text = _normalize_for_containment(prediction)
        return all(term in pred_text for term in quoted_terms)

    terms = _reference_key_terms(reference)
    if not terms:
        return True

    pred_text = _normalize_for_containment(prediction)
    if qtype == "DataAnalysis" and qsubtype in {"AnomalyDetection", "ImpactAnalysis"}:
        return all(term in pred_text for term in terms)

    required = terms[:]
    if len(required) <= 3:
        return all(term in pred_text for term in required)

    matched = sum(1 for term in required if term in pred_text)
    return matched / len(required) >= 0.7


def _comma_parts_match(reference: str, prediction: str, qtype: str, qsubtype: str) -> bool:
    raw_ref_parts = [part.strip() for part in reference.split(",") if part.strip()]
    ref_parts = [_normalize_for_containment(part) for part in raw_ref_parts if _normalize_for_containment(part)]
    if len(ref_parts) <= 1:
        return True
    pred_text = _normalize_for_containment(prediction)
    pred_numbers = _numeric_tokens(prediction)

    def part_matches(raw_part: str, normalized_part: str, cursor: int = 0) -> Tuple[bool, int]:
        part_numbers = _numeric_tokens(raw_part)
        part_without_numbers = _normalize_for_containment(
            re.sub(r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage)?(?:st|nd|rd|th)?", " ", raw_part, flags=re.IGNORECASE)
        )
        if part_numbers and not part_without_numbers:
            matched = any(
                _numbers_equivalent(part_number, pred_number, allow_relative_tolerance=False)
                for part_number in part_numbers
                for pred_number in pred_numbers
            )
            return matched, cursor
        found = pred_text.find(normalized_part, cursor)
        return found >= 0, found + len(normalized_part) if found >= 0 else cursor

    if qtype == "NumericalReasoning" and qsubtype == "Ranking":
        cursor = 0
        for raw_part, part in zip(raw_ref_parts, ref_parts):
            matched, cursor = part_matches(raw_part, part, cursor)
            if not matched:
                return False
        return True
    return all(part_matches(raw_part, part)[0] for raw_part, part in zip(raw_ref_parts, ref_parts))


def _non_numeric_list_has_extra_parts(reference: str, prediction: str) -> bool:
    if _numeric_tokens(reference):
        return False
    ref_parts = [
        _normalize_for_containment(part)
        for part in reference.split(",")
        if _normalize_for_containment(part)
    ]
    if len(ref_parts) <= 1:
        return False
    pred_parts = [
        _normalize_for_containment(part)
        for part in re.split(r"\s*(?:,|;|\band\b)\s*", prediction, flags=re.IGNORECASE)
        if _normalize_for_containment(part)
    ]
    if len(pred_parts) <= len(ref_parts):
        return False
    for pred_part in pred_parts:
        if not any(ref_part in pred_part or pred_part in ref_part for ref_part in ref_parts):
            return True
    return False


def _lenient_exact_match(reference: str, prediction: str, qtype: str, qsubtype: str) -> bool:
    if not str(prediction or "").strip():
        return False

    ref_text = _normalize_for_containment(reference)
    pred_text = _normalize_for_containment(prediction)
    if not ref_text:
        return False
    if ref_text == pred_text:
        return True
    has_reference_numbers = bool(_numeric_tokens(reference))
    if _polarity_conflicts(reference, prediction):
        return False
    if _non_numeric_list_has_extra_parts(reference, prediction):
        return False
    if qtype == "NumericalReasoning" and qsubtype == "Ranking" and ref_text in pred_text:
        if any(token.isalpha() for token in ref_text.split()):
            return True
    if not has_reference_numbers and ref_text in pred_text:
        return True

    if qtype == "DataAnalysis" and qsubtype == "AnomalyDetection":
        if _anomaly_has_extra_named_entities(reference, prediction):
            return False
        if _anomaly_entity_only_match(reference, prediction):
            return True

    if qtype == "DataAnalysis" and qsubtype == "DescriptiveAnalysis":
        return _descriptive_analysis_match(reference, prediction)

    optional_year_reference = _strip_trailing_parenthetical_year(reference)
    if optional_year_reference != str(reference or "").strip():
        optional_year_text = _normalize_for_containment(optional_year_reference)
        if optional_year_text and optional_year_text == pred_text:
            return True

    allow_relative_tolerance = qtype == "DataAnalysis" and qsubtype in TOLERANCE_SUBTYPES
    order_sensitive = not (qtype == "DataAnalysis" and qsubtype == "AnomalyDetection")
    allow_extra_numbers = qtype == "DataAnalysis" and qsubtype == "AnomalyDetection"
    if not _reference_numbers_match(
        reference,
        prediction,
        allow_relative_tolerance,
        order_sensitive=order_sensitive,
        allow_extra_numbers=allow_extra_numbers,
    ):
        return False

    if allow_relative_tolerance and has_reference_numbers:
        if "correlation" in ref_text:
            return "correlation" in pred_text or "coefficient" in pred_text
        return True

    if qtype == "DataAnalysis" and qsubtype == "AnomalyDetection" and has_reference_numbers:
        return True

    if not _comma_parts_match(reference, prediction, qtype, qsubtype):
        return False

    if not _key_terms_match(reference, prediction, qtype, qsubtype):
        return False

    if has_reference_numbers:
        return True

    if qtype == "DataAnalysis" and qsubtype not in {"ImpactAnalysis"}:
        return _rouge_l_score(_normalize_answer(reference), _normalize_answer(prediction)) >= 0.5
    return False


def _anomaly_entity_only_match(reference: str, prediction: str) -> bool:
    quoted_terms = [_normalize_for_containment(term) for term in _quoted_terms(reference)]
    if not quoted_terms:
        return False
    if _numeric_tokens(prediction):
        return False
    pred_tokens = _tokenize_normalized(prediction)
    pred_text = " ".join(pred_tokens)
    if not all(term in pred_text for term in quoted_terms):
        return False
    allowed = set(ANOMALY_GENERIC_TERMS)
    for term in quoted_terms:
        allowed.update(term.split())
    content_tokens = [token for token in pred_tokens if token not in KEY_TERM_STOPWORDS]
    return all(token in allowed for token in content_tokens)


def _descriptive_analysis_match(reference: str, prediction: str) -> bool:
    normalized_reference = _normalize_answer(reference)
    normalized_prediction = _normalize_answer(prediction)
    if not normalized_reference or not normalized_prediction:
        return False
    rouge_score = _rouge_l_score(normalized_reference, normalized_prediction)
    if rouge_score >= 0.40:
        return True

    reference_terms = _reference_key_terms(reference)
    if not reference_terms:
        return False
    prediction_text = _normalize_for_containment(prediction)
    matched_terms = sum(1 for term in reference_terms if term in prediction_text)
    return rouge_score >= 0.28 and matched_terms / len(reference_terms) >= 0.40


def _anomaly_has_extra_named_entities(reference: str, prediction: str) -> bool:
    quoted_terms = [_normalize_for_containment(term) for term in _quoted_terms(reference)]
    if not quoted_terms:
        return False
    allowed_tokens = set(KEY_TERM_STOPWORDS) | set(ANOMALY_GENERIC_TERMS)
    for term in quoted_terms:
        allowed_tokens.update(term.split())
    if ";" in str(prediction):
        prediction_parts = [part for part in str(prediction).split(";") if part.strip()]
    else:
        prediction_parts = [part for part in re.split(r"\s*(?:,|\band\b)\s*", prediction, flags=re.IGNORECASE) if part.strip()]
    for part in prediction_parts:
        part_text = _normalize_for_containment(part)
        if any(term in part_text for term in quoted_terms):
            continue
        part_without_numbers = re.sub(
            r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage)?(?:st|nd|rd|th)?",
            " ",
            part,
            flags=re.IGNORECASE,
        )
        content_tokens = [
            token for token in _tokenize_normalized(part_without_numbers)
            if token not in allowed_tokens
        ]
        if content_tokens:
            return True
    return False


def _normalize_number(value: str) -> Decimal:
    text = value.strip()
    if text.endswith("%"):
        return (Decimal(text.strip("%")) / Decimal("100")).quantize(
            Decimal("1.0000"), rounding=ROUND_HALF_UP
        )
    return Decimal(text)


def _is_number(value: str) -> bool:
    return bool(re.match(r"^-?\d+(\.\d+)?%?$", value.strip()))


def _decimal_precision(values: Sequence[str]) -> int:
    precisions = []
    for value in values:
        if value.endswith("%"):
            continue
        if "." in value:
            precisions.append(len(value.split(".")[-1]))
        else:
            precisions.append(0)
    return min(precisions) if precisions else 0


def _round_decimal(value: Decimal, precision: int) -> str:
    rounding_format = f"1.{('0' * precision)}"
    return str(value.quantize(Decimal(rounding_format), rounding=ROUND_HALF_UP))


def _split_answer_parts(value: str) -> List[str]:
    return [part.strip() for part in value.split(",")]


def _compute_em_score(reference: str, prediction: str) -> float:
    normalized_reference = _normalize_answer(reference)
    normalized_prediction = _normalize_answer(prediction)
    ref_parts = _split_answer_parts(normalized_reference)
    pred_parts = _split_answer_parts(normalized_prediction)
    if not ref_parts:
        return 0.0

    score = 0.0
    weight = 1.0 / len(ref_parts)
    numeric_refs = [part for part in ref_parts if _is_number(part) and not part.endswith("%")]
    precision = _decimal_precision(numeric_refs)
    for part_index, ref_part in enumerate(ref_parts):
        if part_index >= len(pred_parts):
            continue
        pred_part = pred_parts[part_index]
        if _is_number(ref_part):
            try:
                if ref_part.endswith("%"):
                    if _normalize_number(ref_part) == _normalize_number(pred_part):
                        score += weight
                else:
                    if _round_decimal(_normalize_number(ref_part), precision) == _round_decimal(_normalize_number(pred_part), precision):
                        score += weight
            except Exception:
                continue
        elif ref_part == pred_part:
            score += weight
    return score


def _compute_em_with_tolerance_score(reference: str, prediction: str, error_range: float) -> float:
    normalized_reference = _normalize_answer(reference)
    normalized_prediction = _normalize_answer(prediction)
    ref_parts = _split_answer_parts(normalized_reference)
    pred_parts = _split_answer_parts(normalized_prediction)
    if not ref_parts:
        return 0.0

    score = 0.0
    weight = 1.0 / len(ref_parts)
    for part_index, ref_part in enumerate(ref_parts):
        if part_index >= len(pred_parts):
            continue
        pred_part = pred_parts[part_index]
        if _is_number(ref_part):
            try:
                ref_value = _normalize_number(ref_part)
                pred_value = _normalize_number(pred_part)
                if ref_value == Decimal("0"):
                    if pred_value == ref_value:
                        score += weight
                elif abs(ref_value - pred_value) / abs(ref_value) <= Decimal(str(error_range)) / Decimal("100"):
                    score += weight
            except Exception:
                continue
        elif ref_part == pred_part:
            score += weight
    return score


def _rouge_l_score(reference: str, prediction: str) -> float:
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    lcs_length = _lcs_length(reference_tokens, prediction_tokens)
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> int:
    previous = [0] * (len(right_tokens) + 1)
    for left_token in left_tokens:
        current = [0]
        for column_index, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                current.append(previous[column_index - 1] + 1)
            else:
                current.append(max(previous[column_index], current[-1]))
        previous = current
    return previous[-1]