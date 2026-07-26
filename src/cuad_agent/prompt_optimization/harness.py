#!/usr/bin/env python3
"""Build a v2 CUAD prompt-improvement harness from v1 evaluation results."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from dotenv import load_dotenv

from cuad_agent.constants import NO_ANSWER_MARKERS
from cuad_agent.dashboards.evaluation import parse_gold_answers
from cuad_agent.data.dataset import load_datasets
from cuad_agent.paths import prompt_name_part
from cuad_agent.prompts.loader import load_prompt_overrides
from cuad_agent.prompts.templates import compose_system_prompt


os.environ.pop("DSPY_CACHEDIR", None)

LLM_MODEL_ID = "deepseek-v4-pro"

GENERATOR_SYSTEM_PROMPT = (
    "You improve legal clause extraction prompts. "
    "Patch only what the evidence supports. Keep category overlays concise. "
    "Never convert exact-span extraction into classification. "
    "You may only use the generator_dev examples provided in the request. "
    "You must not assume access to evaluator_dev or holdout_eval examples."
)

EVALUATOR_SYSTEM_PROMPT = (
    "You are a skeptical prompt evaluator for legal clause extraction. "
    "Review whether a proposed patch generalizes to unseen failures. "
    "Prefer rejecting broad rewrites, prompt bloat, and changes that risk "
    "false positives. You may only use the evaluator_dev examples provided in "
    "the request. Do not reveal evaluator examples back to the generator."
)


class FailureExample(BaseModel):
    row_id: str
    category: str
    contract_title: str
    question: str
    golden_answer: str
    predicted_answer: str
    gold_marked_impossible: bool
    predicted_marked_impossible: bool
    token_f1: float
    failure_mode: str


class AnswerFormatProfile(BaseModel):
    category: str
    csv_answer_format: str
    gold_answer_type: Literal[
        "label_only", "verbatim_contract_span", "mixed", "no_answer_only"
    ]
    requires_verbatim_contract_span: bool
    yes_no_label_only: bool
    typical_span_shape: Literal[
        "none", "single_sentence", "multi_sentence", "multiple_spans", "mixed"
    ]
    exact_contract_match_count: int
    contract_match_check_count: int
    complete_sentence_match_count: int
    complete_sentence_check_count: int
    allows_multiple_spans: bool
    allows_no_answer: bool
    evidence_notes: list[str]


class PromptPatchRequest(BaseModel):
    model_id: str = "v2"
    category: str
    category_description: str
    answer_format: str
    current_base_prompt: str
    current_category_overlay: str
    answer_format_profile: AnswerFormatProfile
    failure_mode_summary: dict[str, int]
    generator_examples: list[FailureExample] = Field(max_length=20)
    original_generator_guide: str
    previous_generated_prompt: str | None = None
    evaluator_feedback: str | None = None
    loop_index: int


class PromptPatch(BaseModel):
    category: str
    failure_analysis: list[str]
    revised_category_overlay: str
    optional_base_prompt_patch: str | None = None
    expected_improvements: list[str]
    regression_risks: list[str]
    changed_rules: list[str]
    prompt_diff_summary: list[str]


class PromptReviewRequest(BaseModel):
    category: str
    category_description: str
    answer_format: str
    current_base_prompt: str
    current_category_overlay: str
    answer_format_profile: AnswerFormatProfile
    generator_instructions: str
    generator_patch: PromptPatch
    evaluator_examples: list[FailureExample] = Field(max_length=20)
    loop_index: int


class PromptReview(BaseModel):
    decision: Literal["accept", "revise", "reject"]
    generalization_score: float = Field(ge=0.0, le=1.0)
    rationale: list[str]
    likely_fixes: list[str]
    likely_regressions: list[str]
    requested_changes: list[str]


class DashboardExampleRecord(BaseModel):
    row_id: str
    split: Literal["generator_dev", "evaluator_dev", "holdout_eval"]
    contract_title: str
    question: str
    golden_answer: str
    predicted_answer: str
    failure_mode: str
    token_f1: float


class DashboardCategoryRecord(BaseModel):
    question_index: int
    category: str
    question: str
    decision: str
    loop_count: int
    main_failure_mode: str
    answer_format_profile: AnswerFormatProfile
    v1_prompt: str
    candidate_v2_prompt: str
    prompt_diff_summary: list[str]
    generator_example_ids: list[str]
    evaluator_example_ids: list[str]
    evaluator_feedback: list[str]
    regression_risks: list[str]
    generalization_score: float
    change_insights: list[str]
    examples: list[DashboardExampleRecord]


def load_env() -> None:
    load_dotenv(Path.home() / ".env")


def build_deepseek_agent(output_type: type[BaseModel], system_prompt: str) -> Agent:
    load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required for the LLM harness. "
            "Set it in your environment or run with --dry-run for the "
            "deterministic offline mode."
        )
    model = OpenAIChatModel(
        LLM_MODEL_ID,
        provider=DeepSeekProvider(api_key=api_key),
    )
    return Agent(
        model,
        output_type=PromptedOutput(output_type),
        system_prompt=system_prompt,
        retries=2,
    )


def row_id_from_record(row: Any) -> str:
    row_id = getattr(row, "row_id", None)
    if isinstance(row_id, str) and row_id:
        return row_id
    return f"{int(row.document_row_id)}:{int(row.question_index)}"


def normalize_marker(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", str(value).strip().lower()).strip()


def infer_failure_mode(row: Any) -> str:
    predicted_answer = str(getattr(row, "predicted_answer", "") or "").strip()
    normalized_prediction = normalize_marker(predicted_answer)
    gold_answers = parse_gold_answers(getattr(row, "gold_answers", "[]"))
    gold_marked_impossible = bool(getattr(row, "gold_marked_impossible", False))
    predicted_marked_impossible = bool(
        getattr(row, "predicted_marked_impossible", False)
    )
    token_f1 = float(getattr(row, "token_f1", 0.0))

    if predicted_answer.lower() in {"yes", "no"} and gold_answers:
        return "classification_instead_of_span"
    if predicted_marked_impossible and gold_answers:
        return "false_no_answer"
    if not gold_answers and normalized_prediction not in NO_ANSWER_MARKERS:
        return "false_positive_span"
    if token_f1 == 0:
        return "false_positive_span" if not gold_marked_impossible else "format_error"

    predicted_words = len(predicted_answer.split())
    gold_words = len(" ".join(gold_answers).split())
    if gold_words and predicted_words > gold_words * 3:
        return "overlong_span"
    if 0 < token_f1 < 0.5:
        return "partial_span"
    return "format_error"


def read_results(results_path: Path) -> pd.DataFrame:
    results = pd.read_csv(results_path)
    required = {
        "document_row_id",
        "question_index",
        "category",
        "question",
        "gold_answers",
        "predicted_answer",
        "token_f1",
        "correct_at_0_5",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"{results_path} missing required columns: {sorted(missing)}")
    if "row_id" not in results.columns:
        results["row_id"] = [
            f"{int(document_row_id)}:{int(question_index)}"
            for document_row_id, question_index in zip(
                results["document_row_id"], results["question_index"], strict=True
            )
        ]
    return results


def filter_error_rows(results: pd.DataFrame) -> pd.DataFrame:
    correct = results["correct_at_0_5"].astype(str).str.lower().isin({"true", "1"})
    errors = results[~correct].copy()
    errors["failure_mode"] = [infer_failure_mode(row) for row in errors.itertuples()]
    return errors.sort_values(["category", "row_id"]).reset_index(drop=True)


def load_contract_text_lookup() -> dict[int, str]:
    try:
        contracts = load_datasets()["contracts"]
    except Exception:
        return {}
    return {
        int(row.document_row_id): str(row.context)
        for row in contracts[["document_row_id", "context"]].itertuples(index=False)
    }


def split_sentence_like_units(text: str) -> list[str]:
    units = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip())]
    return [unit for unit in units if unit]


def span_shape_for_answer(answer: str) -> str:
    if not answer.strip():
        return "none"
    sentence_count = len(split_sentence_like_units(answer))
    if sentence_count > 1:
        return "multi_sentence"
    return "single_sentence"


def exact_match_spans(answer: str, contract_text: str) -> list[tuple[int, int]]:
    if not answer or not contract_text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        index = contract_text.find(answer, start)
        if index == -1:
            break
        spans.append((index, index + len(answer)))
        start = index + 1
    return spans


def is_sentence_start_boundary(contract_text: str, start: int) -> bool:
    prefix = contract_text[:start].rstrip()
    if not prefix:
        return True
    return prefix[-1] in ".!?\n\r"


def is_sentence_end_boundary(contract_text: str, end: int) -> bool:
    answer_end = contract_text[:end].rstrip()
    suffix = contract_text[end:].lstrip()
    if not answer_end or answer_end[-1] not in ".!?":
        return False
    if not suffix:
        return True
    return suffix[0].isupper() or suffix[0].isdigit() or suffix[0] in "\"'(["


def is_complete_sentence_match(answer: str, contract_text: str) -> bool:
    for start, end in exact_match_spans(answer, contract_text):
        if is_sentence_start_boundary(
            contract_text, start
        ) and is_sentence_end_boundary(contract_text, end):
            return True
    return False


def derive_answer_format_profiles(
    results: pd.DataFrame,
) -> dict[str, AnswerFormatProfile]:
    contract_lookup = load_contract_text_lookup()
    profiles: dict[str, AnswerFormatProfile] = {}

    for category, rows in results.groupby("category", sort=True):
        answer_formats = rows["answer_format"].dropna()
        csv_answer_format = (
            str(answer_formats.iloc[0]) if not answer_formats.empty else ""
        )
        non_empty_answers: list[str] = []
        row_answer_counts: list[int] = []
        allows_no_answer = False
        verbatim_matches = 0
        contract_checks = 0
        complete_sentence_matches = 0
        complete_sentence_checks = 0
        shape_counts: Counter[str] = Counter()

        for row in rows.itertuples():
            gold_answers = [
                answer.strip()
                for answer in parse_gold_answers(getattr(row, "gold_answers", "[]"))
                if answer.strip()
            ]
            if not gold_answers or bool(getattr(row, "gold_marked_impossible", False)):
                allows_no_answer = True
            if gold_answers:
                row_answer_counts.append(len(gold_answers))
            for answer in gold_answers:
                non_empty_answers.append(answer)
                shape_counts[span_shape_for_answer(answer)] += 1
                contract_text = contract_lookup.get(int(row.document_row_id), "")
                if contract_text:
                    contract_checks += 1
                    if exact_match_spans(answer, contract_text):
                        verbatim_matches += 1
                        complete_sentence_checks += 1
                        if is_complete_sentence_match(answer, contract_text):
                            complete_sentence_matches += 1

        normalized_answers = {answer.strip().lower() for answer in non_empty_answers}
        yes_no_label_only = bool(non_empty_answers) and normalized_answers <= {
            "yes",
            "no",
        }
        has_label_answers = any(
            answer.lower() in {"yes", "no"} for answer in non_empty_answers
        )
        has_non_label_answers = any(
            answer.lower() not in {"yes", "no"} for answer in non_empty_answers
        )

        if not non_empty_answers:
            gold_answer_type: Literal[
                "label_only", "verbatim_contract_span", "mixed", "no_answer_only"
            ] = "no_answer_only"
        elif yes_no_label_only:
            gold_answer_type = "label_only"
        elif has_label_answers and has_non_label_answers:
            gold_answer_type = "mixed"
        else:
            gold_answer_type = "verbatim_contract_span"

        allows_multiple_spans = any(count > 1 for count in row_answer_counts)
        if allows_multiple_spans:
            typical_span_shape: Literal[
                "none", "single_sentence", "multi_sentence", "multiple_spans", "mixed"
            ] = "multiple_spans"
        elif not non_empty_answers:
            typical_span_shape = "none"
        elif len(shape_counts) > 1:
            typical_span_shape = "mixed"
        else:
            typical_span_shape = next(iter(shape_counts), "none")  # type: ignore[assignment]

        evidence_notes = [
            f"non_empty_gold_answers={len(non_empty_answers)}",
            f"rows_allowing_no_answer={int(allows_no_answer)}",
        ]
        if contract_checks:
            evidence_notes.append(
                f"verbatim_contract_matches={verbatim_matches}/{contract_checks}"
            )
            evidence_notes.append(
                "complete_sentence_matches="
                f"{complete_sentence_matches}/{complete_sentence_checks}"
            )
        else:
            evidence_notes.append("contract_text_not_available_for_verbatim_check")
        if csv_answer_format:
            evidence_notes.append(f"csv_answer_format={csv_answer_format}")

        profiles[str(category)] = AnswerFormatProfile(
            category=str(category),
            csv_answer_format=csv_answer_format,
            gold_answer_type=gold_answer_type,
            requires_verbatim_contract_span=gold_answer_type
            in {"verbatim_contract_span", "mixed"},
            yes_no_label_only=yes_no_label_only,
            typical_span_shape=typical_span_shape,
            exact_contract_match_count=verbatim_matches,
            contract_match_check_count=contract_checks,
            complete_sentence_match_count=complete_sentence_matches,
            complete_sentence_check_count=complete_sentence_checks,
            allows_multiple_spans=allows_multiple_spans,
            allows_no_answer=allows_no_answer,
            evidence_notes=evidence_notes,
        )

    return profiles


def answer_format_guidance(profile: AnswerFormatProfile) -> list[str]:
    if profile.yes_no_label_only:
        return [
            "Golden answers for this category are label-only; return only Yes or No.",
        ]
    if profile.requires_verbatim_contract_span:
        guidance = [
            "Golden answers require verbatim contract text; return exact clause text, not a label.",
        ]
        if profile.typical_span_shape == "single_sentence":
            guidance.append(
                "The expected span is usually a complete sentence or clause; include the full sentence punctuation when present."
            )
        elif profile.typical_span_shape == "multi_sentence":
            guidance.append(
                "The expected span may require multiple consecutive sentences."
            )
        elif profile.typical_span_shape == "multiple_spans":
            guidance.append(
                "Multiple golden spans occur; return each span on a separate line."
            )
        if profile.allows_no_answer:
            guidance.append("Return NO_ANSWER only when no supporting span exists.")
        return guidance
    return ["Use the derived golden-answer format profile for this category."]


def build_failure_example(row: Any) -> FailureExample:
    return FailureExample(
        row_id=row_id_from_record(row),
        category=str(row.category),
        contract_title=str(getattr(row, "title", "")),
        question=str(row.question),
        golden_answer="\n".join(parse_gold_answers(getattr(row, "gold_answers", "[]"))),
        predicted_answer=str(getattr(row, "predicted_answer", "")),
        gold_marked_impossible=bool(getattr(row, "gold_marked_impossible", False)),
        predicted_marked_impossible=bool(
            getattr(row, "predicted_marked_impossible", False)
        ),
        token_f1=float(getattr(row, "token_f1", 0.0)),
        failure_mode=str(getattr(row, "failure_mode", "format_error")),
    )


def create_splits(
    errors: pd.DataFrame,
    *,
    generator_size: int = 20,
    evaluator_size: int = 20,
) -> dict[str, list[str]]:
    generator_dev: list[str] = []
    evaluator_dev: list[str] = []
    holdout_eval: list[str] = []

    total_cap = generator_size + evaluator_size
    for _, rows in errors.groupby("category", sort=True):
        row_ids = sorted(str(row_id) for row_id in rows["row_id"].tolist())
        n = len(row_ids)
        if n == 0:
            continue
        if n <= total_cap:
            gen_count = max(1, n // 2)
            eval_count = n - gen_count
        else:
            gen_count = generator_size
            eval_count = evaluator_size
        generator_dev.extend(row_ids[:gen_count])
        evaluator_dev.extend(row_ids[gen_count : gen_count + eval_count])
        holdout_eval.extend(row_ids[gen_count + eval_count :])

    return {
        "generator_dev": generator_dev,
        "evaluator_dev": evaluator_dev,
        "holdout_eval": holdout_eval,
    }


def examples_for_split(
    errors: pd.DataFrame,
    category: str,
    row_ids: list[str],
) -> list[FailureExample]:
    wanted = set(row_ids)
    rows = errors[(errors["category"] == category) & (errors["row_id"].isin(wanted))]
    return [build_failure_example(row) for row in rows.itertuples()]


def categories_from_results(results: pd.DataFrame) -> list[str]:
    categories = (
        results[["question_index", "category"]]
        .drop_duplicates("category")
        .sort_values(["question_index", "category"])
    )
    return [str(row.category) for row in categories.itertuples(index=False)]


def rows_for_category(rows: pd.DataFrame, category: str) -> pd.DataFrame:
    return rows[rows["category"] == category]


def metadata_for_category(results: pd.DataFrame, category: str) -> dict[str, str]:
    category_rows = rows_for_category(results, category)
    if category_rows.empty:
        raise ValueError(f"No evaluation rows found for category={category!r}")
    row = category_rows.sort_values(["question_index", "document_row_id"]).iloc[0]
    return {
        "category": category,
        "category_description": str(row.get("category_description", "")),
        "answer_format": ""
        if pd.isna(row.get("answer_format", ""))
        else str(row.get("answer_format", "")),
        "question": str(row.get("question", "")),
        "question_index": str(int(row.get("question_index", 0))),
    }


def fallback_answer_format_profile(
    category: str,
    meta: dict[str, str],
) -> AnswerFormatProfile:
    return AnswerFormatProfile(
        category=category,
        csv_answer_format=meta["answer_format"],
        gold_answer_type="mixed",
        requires_verbatim_contract_span=True,
        yes_no_label_only=False,
        typical_span_shape="mixed",
        exact_contract_match_count=0,
        contract_match_check_count=0,
        complete_sentence_match_count=0,
        complete_sentence_check_count=0,
        allows_multiple_spans=True,
        allows_no_answer=True,
        evidence_notes=["missing_profile_fallback"],
    )


def default_prompt_from_metadata(meta: dict[str, str]) -> str:
    return compose_system_prompt(
        question=meta["question"],
        category=meta["category"],
        category_description=meta["category_description"],
        answer_format=meta["answer_format"],
    )


def guidance_for_failure_modes(failure_modes: dict[str, int]) -> list[str]:
    guidance: list[str] = []
    if failure_modes.get("classification_instead_of_span", 0):
        guidance.append(
            "For Yes/No categories, return the supporting contract clause text; "
            "do not answer only Yes or No."
        )
    if failure_modes.get("false_no_answer", 0):
        guidance.append(
            "Before returning NO_ANSWER, search for synonyms and related clause "
            "language that directly supports the category."
        )
    if failure_modes.get("false_positive_span", 0):
        guidance.append(
            "Return only clauses that directly satisfy this category; exclude "
            "nearby but unrelated legal boilerplate."
        )
    if failure_modes.get("partial_span", 0):
        guidance.append(
            "Include enough surrounding sentence text to preserve the complete "
            "legal obligation, condition, or exception."
        )
    if failure_modes.get("overlong_span", 0):
        guidance.append(
            "Keep spans tight: return the relevant clause or sentence, not whole "
            "sections unless the full section is needed."
        )
    if failure_modes.get("format_error", 0):
        guidance.append(
            "Use newline-separated spans, or NO_ANSWER when no supporting text exists."
        )
    return guidance


def generate_patch(request: PromptPatchRequest) -> PromptPatch:
    base_prompt = request.previous_generated_prompt or request.current_category_overlay
    failure_modes = dict(request.failure_mode_summary)
    guidance = guidance_for_failure_modes(failure_modes)
    format_guidance = answer_format_guidance(request.answer_format_profile)
    feedback_lines = [
        line.strip("- ")
        for line in (request.evaluator_feedback or "").splitlines()
        if line.strip()
    ]

    additions = [
        "V2 extraction guidance:",
        f"- Category evidence must directly answer: {request.category_description}",
    ]
    additions.extend(f"- {line}" for line in format_guidance)
    additions.extend(f"- {line}" for line in guidance)
    additions.extend(f"- Evaluator requested: {line}" for line in feedback_lines[:3])
    additions.append("- Preserve exact contract wording; do not summarize.")

    revised = base_prompt.rstrip()
    addition_block = "\n\n" + "\n".join(dict.fromkeys(additions))
    if "V2 extraction guidance:" not in revised:
        revised = f"{revised}{addition_block}"
    elif feedback_lines:
        revised = f"{revised}\n" + "\n".join(
            f"- Evaluator requested: {line}" for line in feedback_lines[:3]
        )

    prompt_diff_summary = list(
        difflib.unified_diff(
            request.current_category_overlay.splitlines(),
            revised.splitlines(),
            fromfile="v1",
            tofile="candidate_v2",
            lineterm="",
        )
    )
    return PromptPatch(
        category=request.category,
        failure_analysis=[
            f"{mode}: {count}"
            for mode, count in sorted(
                failure_modes.items(), key=lambda item: (-item[1], item[0])
            )
            if count
        ],
        revised_category_overlay=revised,
        optional_base_prompt_patch=None,
        expected_improvements=(format_guidance + guidance)
        or ["Clarifies golden-answer-derived output format."],
        regression_risks=[
            "Prompt may overfit generator examples if evaluator feedback is weak.",
            "Additional guidance may increase prompt length.",
        ],
        changed_rules=additions[1:],
        prompt_diff_summary=prompt_diff_summary,
    )


def evaluate_patch(request: PromptReviewRequest) -> PromptReview:
    patch_text = request.generator_patch.revised_category_overlay
    requested_changes: list[str] = []
    likely_regressions: list[str] = []

    has_exact_span_rule = "exact" in patch_text.lower() or "span" in patch_text.lower()
    has_yes_no_mention = (
        "yes or no" in patch_text.lower() or "yes/no" in patch_text.lower()
    )
    prompt_growth = len(patch_text) - len(request.current_category_overlay)

    if (
        request.answer_format_profile.requires_verbatim_contract_span
        and not has_exact_span_rule
    ):
        requested_changes.append("Add explicit exact-span extraction guidance.")
    if (
        request.answer_format_profile.requires_verbatim_contract_span
        and request.answer_format_profile.csv_answer_format.lower() == "yes/no"
        and not has_yes_no_mention
    ):
        requested_changes.append(
            "Clarify that CSV Yes/No categories still require spans when gold answers are spans."
        )
    if request.answer_format_profile.yes_no_label_only and not has_yes_no_mention:
        requested_changes.append(
            "Clarify that this category is label-only because gold answers are Yes/No."
        )
    if prompt_growth > 1800:
        requested_changes.append(
            "Reduce prompt length and keep only category-specific rules."
        )
        likely_regressions.append("Prompt bloat may dilute the extraction task.")

    evaluator_modes = Counter(
        example.failure_mode for example in request.evaluator_examples
    )
    if evaluator_modes.get("false_positive_span", 0):
        likely_regressions.append("False positives remain a risk for related clauses.")

    if requested_changes and request.loop_index < 3:
        decision: Literal["accept", "revise", "reject"] = "revise"
    elif requested_changes:
        decision = "reject"
    else:
        decision = "accept"

    score = 0.75
    if requested_changes:
        score -= 0.25
    if likely_regressions:
        score -= 0.1
    if request.evaluator_examples:
        score += min(0.1, len(request.evaluator_examples) / 200)

    return PromptReview(
        decision=decision,
        generalization_score=max(0.0, min(1.0, score)),
        rationale=[
            "Reviewed proposed prompt against evaluator_dev examples not shown to the generator.",
            f"Evaluator failure modes: {dict(evaluator_modes)}",
        ],
        likely_fixes=request.generator_patch.expected_improvements,
        likely_regressions=likely_regressions,
        requested_changes=requested_changes,
    )


def format_answer_format_profile_text(profile: AnswerFormatProfile) -> str:
    lines = [
        f"- CSV answer format: {profile.csv_answer_format}",
        f"- Gold answer type: {profile.gold_answer_type}",
    ]
    if profile.yes_no_label_only:
        lines.append("- Required output: Yes or No label only")
    elif profile.requires_verbatim_contract_span:
        lines.append("- Required output: exact clause text, not Yes/No")
    lines.append(f"- Typical span shape: {profile.typical_span_shape}")
    lines.append(f"- Multiple spans allowed: {profile.allows_multiple_spans}")
    lines.append(f"- NO_ANSWER allowed: {profile.allows_no_answer}")
    for note in profile.evidence_notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def format_failure_examples_text(examples: list[FailureExample]) -> str:
    if not examples:
        return "No examples available."
    parts = []
    for i, example in enumerate(examples, 1):
        parts.append(
            f"{i}. Contract: {example.contract_title}\n"
            f"   Question: {example.question}\n"
            f"   Golden answer: {example.golden_answer or 'NO_ANSWER'}\n"
            f"   Predicted answer: {example.predicted_answer or '(empty)'}\n"
            f"   Gold marked impossible: {example.gold_marked_impossible}\n"
            f"   Predicted marked impossible: {example.predicted_marked_impossible}\n"
            f"   Token F1: {example.token_f1:.2f}\n"
            f"   Failure mode: {example.failure_mode}"
        )
    return "\n\n".join(parts)


def format_failure_mode_summary_text(failure_modes: dict[str, int]) -> str:
    if not failure_modes:
        return "No failures in generator examples."
    return "\n".join(
        f"- {mode}: {count}"
        for mode, count in sorted(failure_modes.items(), key=lambda x: (-x[1], x[0]))
        if count
    )


def format_generator_prompt(request: PromptPatchRequest) -> str:
    lines = [
        "You are improving a legal clause extraction prompt.",
        "",
        "Current base extraction prompt:",
        request.current_base_prompt,
        "",
        "Current category overlay:",
        request.current_category_overlay,
        "",
        f"Category: {request.category}",
        "",
        f"Category description: {request.category_description}",
        "",
        "Derived answer format from golden answers:",
        format_answer_format_profile_text(request.answer_format_profile),
        "",
        "Failure-mode summary:",
        format_failure_mode_summary_text(request.failure_mode_summary),
        "",
        "Generator examples (these are your only examples; do not assume access to evaluator or holdout examples):",
        format_failure_examples_text(request.generator_examples),
    ]
    if request.loop_index > 1 and request.previous_generated_prompt:
        lines += [
            "",
            f"Previous generated prompt (loop {request.loop_index - 1}):",
            request.previous_generated_prompt,
        ]
    if request.evaluator_feedback:
        lines += [
            "",
            "Evaluator feedback on previous prompt:",
            request.evaluator_feedback,
            "",
            "Revise the category overlay based on this feedback.",
            "Do not request evaluator_dev examples; work only from the generator_dev examples above.",
        ]
    lines += [
        "",
        "Task:",
        "1. Identify the 2-3 prompt issues most likely causing these failures.",
        "2. Revise only the category overlay unless the base extraction prompt is clearly defective.",
        "3. Keep the revised overlay concise.",
        "4. Include positive evidence cues and relevant exclusion cues.",
        "5. Do not convert a span-extraction task into a Yes/No classification task.",
        "",
        "Return:",
        "- failure_analysis: list identifying the key issues",
        "- revised_category_overlay: the improved category-specific prompt",
        "- optional_base_prompt_patch: null unless the base prompt is clearly defective",
        "- expected_improvements: what failures this patch should fix",
        "- regression_risks: what could regress",
        "- changed_rules: list of rules you added, removed, or changed",
        "- prompt_diff_summary: unified diff between old and new overlay",
    ]
    return "\n".join(lines)


def format_evaluator_prompt(request: PromptReviewRequest) -> str:
    lines = [
        "You are reviewing a proposed legal clause extraction prompt patch.",
        "",
        "Generator instructions:",
        request.generator_instructions,
        "",
        "Current base extraction prompt:",
        request.current_base_prompt,
        "",
        "Current category overlay:",
        request.current_category_overlay,
        "",
        "Proposed category overlay:",
        request.generator_patch.revised_category_overlay,
        "",
        f"Category: {request.category}",
        "",
        f"Category description: {request.category_description}",
        "",
        "Derived answer format from golden answers:",
        format_answer_format_profile_text(request.answer_format_profile),
        "",
        "Evaluator examples (these were NOT shown to the generator; do not reveal them back to the generator):",
        format_failure_examples_text(request.evaluator_examples),
        "",
        "Task:",
        "1. Judge whether the proposed patch is likely to generalize to these unseen examples.",
        "2. Identify likely improvements and likely regressions.",
        "3. Reject broad rewrites, prompt bloat, or patches that convert span extraction to classification.",
        "4. Decide: accept, revise, or reject.",
        "",
        "Return:",
        "- decision: 'accept', 'revise', or 'reject'",
        "- generalization_score: 0.0 to 1.0",
        "- rationale: list of reasons for the decision",
        "- likely_fixes: what failures this patch should address",
        "- likely_regressions: what could regress",
        "- requested_changes: specific changes to request if decision is 'revise'",
    ]
    return "\n".join(lines)


def generate_patch_with_agent(
    agent: Agent | None,
    request: PromptPatchRequest,
) -> PromptPatch:
    if agent is None:
        return generate_patch(request)
    result = agent.run_sync(format_generator_prompt(request))
    patch = result.output
    if not isinstance(patch, PromptPatch):
        raise TypeError(
            f"Generator agent returned {type(patch)!r}, expected PromptPatch"
        )
    if patch.category != request.category:
        patch.category = request.category
    if not patch.prompt_diff_summary:
        patch.prompt_diff_summary = list(
            difflib.unified_diff(
                request.current_category_overlay.splitlines(),
                patch.revised_category_overlay.splitlines(),
                fromfile="v1",
                tofile="candidate_v2",
                lineterm="",
            )
        )
    return patch


def evaluate_patch_with_agent(
    agent: Agent | None,
    request: PromptReviewRequest,
) -> PromptReview:
    if agent is None:
        return evaluate_patch(request)
    result = agent.run_sync(format_evaluator_prompt(request))
    review = result.output
    if not isinstance(review, PromptReview):
        raise TypeError(
            f"Evaluator agent returned {type(review)!r}, expected PromptReview"
        )
    return review


def write_jsonl(path: Path, records: list[BaseModel | dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if isinstance(record, BaseModel):
                payload = record.model_dump(mode="json")
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_prompt_module(path: Path, prompts: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    lines = [
        '"""Candidate CUAD v2 system prompts generated by prompt_improve_v2.py."""',
        "",
        "from __future__ import annotations",
        "",
    ]
    mapping_entries: list[str] = []
    for category, prompt in sorted(prompts.items()):
        base_name = f"{prompt_name_part(category)}_SYSTEM_PROMPT"
        prompt_name = base_name
        suffix = 2
        while prompt_name in used_names:
            prompt_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(prompt_name)
        lines.extend([f"# {category}", f"{prompt_name} = {prompt!r}", ""])
        mapping_entries.append(f"    {category!r}: {prompt_name},")
    lines.extend(["CATEGORY_SYSTEM_PROMPTS = {", *mapping_entries, "}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def render_dashboard(
    *,
    model_id: str,
    source_results: Path,
    splits: dict[str, list[str]],
    category_records: list[DashboardCategoryRecord],
) -> str:
    data = {
        "model_id": model_id,
        "source_results": str(source_results),
        "split_counts": {name: len(row_ids) for name, row_ids in splits.items()},
        "categories": [record.model_dump(mode="json") for record in category_records],
    }
    data_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUAD V2 Prompt Review</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b141a; --panel:#111b21; --panel2:#202c33; --border:#26353d; --text:#e9edef; --muted:#aebac1; --green:#00a884; --warn:#f15c6d; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    header {{ position:sticky; top:0; z-index:2; display:flex; justify-content:space-between; gap:16px; align-items:center; padding:14px 18px; background:var(--panel2); border-bottom:1px solid var(--border); }}
    main {{ display:grid; grid-template-columns:340px minmax(0,1fr); min-height:calc(100vh - 61px); }}
    aside {{ border-right:1px solid var(--border); background:var(--panel); }}
    input, select {{ width:100%; border:1px solid var(--border); border-radius:6px; background:var(--panel2); color:var(--text); padding:9px 10px; }}
    .filters {{ display:grid; gap:8px; padding:12px; border-bottom:1px solid var(--border); }}
    .item {{ width:100%; border:0; border-bottom:1px solid var(--border); background:transparent; color:inherit; text-align:left; padding:12px; cursor:pointer; }}
    .item.active, .item:hover {{ background:var(--panel2); }}
    .item strong {{ display:block; margin-bottom:3px; }}
    .muted {{ color:var(--muted); }}
    .content {{ padding:20px; min-width:0; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(130px,1fr)); gap:10px; margin-bottom:16px; }}
    .metric, .card {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:12px; }}
    .metric b {{ display:block; font-size:20px; }}
    .pill {{ display:inline-flex; padding:2px 8px; border-radius:999px; background:rgba(255,255,255,.08); color:var(--muted); font-size:12px; }}
    .pill.accept {{ color:#90f0d8; background:rgba(0,168,132,.18); }}
    .pill.reject {{ color:#ffb4bd; background:rgba(241,92,109,.16); }}
    .pill.no_failures, .pill.skipped_existing {{ color:#d6e8ff; background:rgba(91,141,239,.16); }}
    .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    pre {{ white-space:pre-wrap; overflow:auto; max-height:420px; margin:0; font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
    table {{ width:100%; border-collapse:collapse; min-width:980px; }}
    th,td {{ text-align:left; vertical-align:top; border-bottom:1px solid var(--border); padding:9px; }}
    th {{ color:var(--muted); position:sticky; top:61px; background:var(--panel2); }}
    .table-wrap {{ overflow:auto; border:1px solid var(--border); border-radius:8px; }}
    @media (max-width:900px) {{ main,.cols {{ grid-template-columns:1fr; }} aside {{ border-right:0; }} .grid {{ grid-template-columns:1fr 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div><strong>CUAD V2 Prompt Review</strong><div class="muted">Model ID: <span id="modelId"></span></div></div>
    <div class="muted" id="source"></div>
  </header>
  <main>
    <aside>
      <div class="filters">
        <input id="search" placeholder="Search categories">
        <select id="decision"><option value="">All decisions</option><option>accept</option><option>revise</option><option>reject</option><option>no_failures</option><option>skipped_existing</option></select>
      </div>
      <div id="list"></div>
    </aside>
    <section class="content" id="content"></section>
  </main>
  <script id="data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('data').textContent);
    let active = data.categories[0]?.category || '';
    const esc = value => String(value ?? '').replace(/[&<>'"]/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}}[ch]));
    document.getElementById('modelId').textContent = data.model_id;
    document.getElementById('source').textContent = data.source_results;
    function renderList() {{
      const q = document.getElementById('search').value.toLowerCase();
      const decision = document.getElementById('decision').value;
      document.getElementById('list').innerHTML = data.categories
        .filter(c => (!q || c.category.toLowerCase().includes(q)) && (!decision || c.decision === decision))
        .map(c => `<button class="item ${{c.category === active ? 'active' : ''}}" data-category="${{esc(c.category)}}"><strong>Q${{Number(c.question_index) + 1}} · ${{esc(c.category)}}</strong><span class="pill ${{esc(c.decision)}}">${{esc(c.decision)}}</span> <span class="muted">${{esc(c.answer_format_profile?.gold_answer_type || '')}} · ${{esc(c.main_failure_mode)}} · score ${{Number(c.generalization_score).toFixed(2)}}</span></button>`)
        .join('');
      document.querySelectorAll('.item').forEach(btn => btn.addEventListener('click', () => {{ active = btn.dataset.category; render(); }}));
    }}
    function renderContent() {{
      const c = data.categories.find(item => item.category === active) || data.categories[0];
      if (!c) {{ document.getElementById('content').innerHTML = '<p>No prompt harness records found.</p>'; return; }}
      const rows = c.examples.map(ex => `<tr><td>${{esc(ex.split)}}</td><td>${{esc(ex.row_id)}}</td><td>${{esc(ex.contract_title)}}</td><td>${{esc(ex.failure_mode)}}</td><td>${{Number(ex.token_f1).toFixed(2)}}</td><td>${{esc(ex.golden_answer)}}</td><td>${{esc(ex.predicted_answer)}}</td></tr>`).join('');
      const profile = c.answer_format_profile || {{}};
      const profileRows = Object.entries(profile).map(([key, value]) => `<tr><th>${{esc(key)}}</th><td>${{esc(Array.isArray(value) ? value.join('\\n') : value)}}</td></tr>`).join('');
      const insights = (c.change_insights || []).map(x => `<li>${{esc(x)}}</li>`).join('');
      document.getElementById('content').innerHTML = `
        <h1>Q${{Number(c.question_index) + 1}} · ${{esc(c.category)}}</h1>
        <p class="muted">${{esc(c.question || '')}}</p>
        <div class="grid">
          <div class="metric"><b>${{esc(c.decision)}}</b><span class="muted">Decision</span></div>
          <div class="metric"><b>${{Number(c.generalization_score).toFixed(2)}}</b><span class="muted">Generalization score</span></div>
          <div class="metric"><b>${{esc(c.loop_count)}}</b><span class="muted">Loops</span></div>
          <div class="metric"><b>${{esc(c.main_failure_mode)}}</b><span class="muted">Main failure mode</span></div>
        </div>
        <div class="card"><h2>Change Insight</h2><ul>${{insights}}</ul></div>
        <div class="card"><h2>Derived Answer Format</h2><div class="table-wrap"><table><tbody>${{profileRows}}</tbody></table></div></div>
        <div class="cols">
          <div class="card"><h2>V1 Prompt</h2><pre>${{esc(c.v1_prompt)}}</pre></div>
          <div class="card"><h2>Candidate V2 Prompt</h2><pre>${{esc(c.candidate_v2_prompt)}}</pre></div>
        </div>
        <div class="card"><h2>Prompt Diff</h2><pre>${{esc((c.prompt_diff_summary || []).join('\\n'))}}</pre></div>
        <div class="card"><h2>Evaluator Feedback</h2><ul>${{(c.evaluator_feedback || []).map(x => `<li>${{esc(x)}}</li>`).join('')}}</ul><h3>Regression Risks</h3><ul>${{(c.regression_risks || []).map(x => `<li>${{esc(x)}}</li>`).join('')}}</ul></div>
        <div class="card"><h2>Examples</h2><div class="table-wrap"><table><thead><tr><th>Split</th><th>Row</th><th>Contract</th><th>Failure</th><th>F1</th><th>Golden</th><th>Prediction</th></tr></thead><tbody>${{rows}}</tbody></table></div></div>
      `;
    }}
    function render() {{ renderList(); renderContent(); }}
    document.getElementById('search').addEventListener('input', renderList);
    document.getElementById('decision').addEventListener('change', renderList);
    render();
  </script>
</body>
</html>
"""


def dashboard_example(
    example: FailureExample,
    split: Literal["generator_dev", "evaluator_dev", "holdout_eval"],
) -> DashboardExampleRecord:
    return DashboardExampleRecord(
        row_id=example.row_id,
        split=split,
        contract_title=example.contract_title,
        question=example.question,
        golden_answer=example.golden_answer,
        predicted_answer=example.predicted_answer,
        failure_mode=example.failure_mode,
        token_f1=example.token_f1,
    )


def prompt_diff(old_prompt: str, new_prompt: str) -> list[str]:
    return list(
        difflib.unified_diff(
            old_prompt.splitlines(),
            new_prompt.splitlines(),
            fromfile="v1",
            tofile="candidate_v2",
            lineterm="",
        )
    )


def prompt_change_insights(
    *,
    decision: str,
    current_prompt: str,
    candidate_prompt: str,
    final_patch: PromptPatch | None,
    final_review: PromptReview | None,
    main_failure_mode: str,
) -> list[str]:
    if current_prompt == candidate_prompt:
        if decision == "no_failures":
            return [
                "Prompt unchanged because no incorrect examples were found for this question."
            ]
        return ["Prompt unchanged; no generated candidate prompt was available."]

    insights: list[str] = []
    length_delta = len(candidate_prompt) - len(current_prompt)
    direction = "longer" if length_delta >= 0 else "shorter"
    insights.append(
        f"Candidate prompt is {abs(length_delta)} characters {direction} than v1."
    )
    if main_failure_mode != "none":
        insights.append(f"Main observed failure mode: {main_failure_mode}.")
    if final_patch is not None:
        insights.extend(final_patch.changed_rules[:5])
    if final_review is not None and final_review.decision != "accept":
        insights.append(
            f"Evaluator decision was {final_review.decision}; candidate is still written for testing."
        )
        insights.extend(final_review.requested_changes[:3])
    elif decision == "skipped_existing":
        insights.append(
            "Loaded from category_status.jsonl; optimization loop was skipped in this run."
        )
    return list(dict.fromkeys(insights))


def build_dashboard_record(
    *,
    category: str,
    results: pd.DataFrame,
    errors: pd.DataFrame,
    splits: dict[str, list[str]],
    answer_format_profiles: dict[str, AnswerFormatProfile],
    current_prompt: str,
    candidate_prompt: str,
    decision: str,
    loop_count: int,
    main_failure_mode: str,
    final_patch: PromptPatch | None = None,
    final_review: PromptReview | None = None,
    feedback_history: list[str] | None = None,
    generator_examples: list[FailureExample] | None = None,
    evaluator_examples: list[FailureExample] | None = None,
    holdout_examples: list[FailureExample] | None = None,
) -> DashboardCategoryRecord:
    meta = metadata_for_category(results, category)
    answer_format_profile = answer_format_profiles.get(
        category
    ) or fallback_answer_format_profile(category, meta)
    generator_examples = (
        generator_examples
        if generator_examples is not None
        else examples_for_split(errors, category, splits["generator_dev"])
    )
    evaluator_examples = (
        evaluator_examples
        if evaluator_examples is not None
        else examples_for_split(errors, category, splits["evaluator_dev"])
    )
    holdout_examples = (
        holdout_examples
        if holdout_examples is not None
        else examples_for_split(errors, category, splits["holdout_eval"])
    )
    diff = (
        final_patch.prompt_diff_summary
        if final_patch is not None
        else prompt_diff(
            current_prompt,
            candidate_prompt,
        )
    )
    regression_risks: list[str] = []
    if final_patch is not None:
        regression_risks.extend(final_patch.regression_risks)
    if final_review is not None:
        regression_risks.extend(final_review.likely_regressions)

    return DashboardCategoryRecord(
        question_index=int(meta["question_index"]),
        category=category,
        question=meta["question"],
        decision=decision,
        loop_count=loop_count,
        main_failure_mode=main_failure_mode,
        answer_format_profile=answer_format_profile,
        v1_prompt=current_prompt,
        candidate_v2_prompt=candidate_prompt,
        prompt_diff_summary=diff,
        generator_example_ids=[e.row_id for e in generator_examples],
        evaluator_example_ids=[e.row_id for e in evaluator_examples],
        evaluator_feedback=feedback_history or [],
        regression_risks=regression_risks,
        generalization_score=(
            final_review.generalization_score if final_review is not None else 0.0
        ),
        change_insights=prompt_change_insights(
            decision=decision,
            current_prompt=current_prompt,
            candidate_prompt=candidate_prompt,
            final_patch=final_patch,
            final_review=final_review,
            main_failure_mode=main_failure_mode,
        ),
        examples=[
            *[dashboard_example(e, "generator_dev") for e in generator_examples],
            *[dashboard_example(e, "evaluator_dev") for e in evaluator_examples],
            *[dashboard_example(e, "holdout_eval") for e in holdout_examples],
        ],
    )


@dataclass
class CategoryResult:
    category: str
    current_prompt: str
    candidate_prompt: str
    decision: str
    loop_count: int
    final_patch: PromptPatch | None
    final_review: PromptReview | None
    feedback_history: list[str]
    runs: list[dict[str, Any]]
    reviews: list[dict[str, Any]]
    answer_format_profile: AnswerFormatProfile | None
    generator_examples: list[FailureExample]
    evaluator_examples: list[FailureExample]
    holdout_examples: list[FailureExample]
    main_failure_mode: str
    log_lines: list[str]


def read_category_status(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    status: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        category = record.get("category")
        if isinstance(category, str):
            status[category] = record
    return status


def append_category_status(
    path: Path,
    record: dict[str, Any],
    lock: threading.Lock,
) -> None:
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)


def process_category(
    category: str,
    *,
    results: pd.DataFrame,
    errors: pd.DataFrame,
    splits: dict[str, list[str]],
    answer_format_profiles: dict[str, AnswerFormatProfile],
    prompt_overrides: dict[str, str],
    generator_agent: Agent | None,
    evaluator_agent: Agent | None,
    args: argparse.Namespace,
    use_llm: bool,
) -> CategoryResult:
    log_lines: list[str] = []

    def log(message: str) -> None:
        log_lines.append(message)

    meta = metadata_for_category(results, category)
    category_error_rows = rows_for_category(errors, category)
    log(
        f"Question {int(meta['question_index']) + 1}: "
        f"improving system prompt for {category}"
    )
    current_prompt = prompt_overrides.get(category) or default_prompt_from_metadata(
        meta
    )

    if category_error_rows.empty:
        log("  no failures — keeping v1 prompt")
        return CategoryResult(
            category=category,
            current_prompt=current_prompt,
            candidate_prompt=current_prompt,
            decision="no_failures",
            loop_count=0,
            final_patch=None,
            final_review=None,
            feedback_history=[],
            runs=[],
            reviews=[],
            answer_format_profile=answer_format_profiles.get(category),
            generator_examples=[],
            evaluator_examples=[],
            holdout_examples=[],
            main_failure_mode="none",
            log_lines=log_lines,
        )

    generator_examples = examples_for_split(errors, category, splits["generator_dev"])
    evaluator_examples = examples_for_split(errors, category, splits["evaluator_dev"])
    holdout_examples = examples_for_split(errors, category, splits["holdout_eval"])
    failure_mode_summary = dict(
        Counter(example.failure_mode for example in generator_examples)
    )
    if not failure_mode_summary:
        failure_mode_summary = dict(
            Counter(str(mode) for mode in category_error_rows["failure_mode"])
        )
    log(
        f"  generator examples={len(generator_examples)}, "
        f"evaluator examples={len(evaluator_examples)}, "
        f"holdout examples={len(holdout_examples)}, "
        f"failure modes={failure_mode_summary}"
    )

    answer_format_profile = answer_format_profiles.get(category)
    if answer_format_profile is None:
        answer_format_profile = fallback_answer_format_profile(category, meta)

    request = PromptPatchRequest(
        model_id=args.model_id,
        category=category,
        category_description=meta["category_description"],
        answer_format=meta["answer_format"],
        current_base_prompt=current_prompt,
        current_category_overlay=current_prompt,
        answer_format_profile=answer_format_profile,
        failure_mode_summary=failure_mode_summary,
        generator_examples=generator_examples[: args.generator_examples],
        original_generator_guide=GENERATOR_SYSTEM_PROMPT,
        loop_index=1,
    )

    final_patch: PromptPatch | None = None
    final_review: PromptReview | None = None
    loop_count = 0
    feedback_history: list[str] = []
    runs: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    candidate_prompt = current_prompt

    for loop_index in range(1, args.max_loops + 1):
        loop_count = loop_index
        request.loop_index = loop_index
        log(f"  loop {loop_index}: generator updating prompt for {category}")
        patch = generate_patch_with_agent(generator_agent, request)
        review_request = PromptReviewRequest(
            category=category,
            category_description=meta["category_description"],
            answer_format=meta["answer_format"],
            current_base_prompt=current_prompt,
            current_category_overlay=current_prompt,
            answer_format_profile=answer_format_profile,
            generator_instructions=GENERATOR_SYSTEM_PROMPT,
            generator_patch=patch,
            evaluator_examples=evaluator_examples[: args.evaluator_examples],
            loop_index=loop_index,
        )
        log(f"  loop {loop_index}: evaluator reviewing prompt for {category}")
        review = evaluate_patch_with_agent(evaluator_agent, review_request)
        log(
            f"  loop {loop_index}: evaluator decision={review.decision}, "
            f"generalization_score={review.generalization_score:.2f}"
        )
        feedback_history.extend(review.requested_changes)
        runs.append(
            {
                "category": category,
                "loop_index": loop_index,
                "agent_model_id": LLM_MODEL_ID if use_llm else None,
                "use_llm": use_llm,
                "generator_request": request.model_dump(mode="json"),
                "generator_patch": patch.model_dump(mode="json"),
                "evaluator_review": review.model_dump(mode="json"),
            }
        )
        reviews.append(
            {
                "category": category,
                "loop_index": loop_index,
                **review.model_dump(mode="json"),
            }
        )

        final_patch = patch
        final_review = review
        candidate_prompt = patch.revised_category_overlay
        if review.decision == "accept":
            log(f"  accepted prompt update for {category}")
            break
        if review.decision == "reject":
            log(f"  rejected prompt update for {category}")
            break
        request.previous_generated_prompt = patch.revised_category_overlay
        request.evaluator_feedback = "\n".join(review.requested_changes)

    decision = final_review.decision if final_review else "none"

    if generator_examples:
        main_failure_mode = Counter(
            example.failure_mode for example in generator_examples
        ).most_common(1)[0][0]
    else:
        main_failure_mode = Counter(
            str(mode) for mode in category_error_rows["failure_mode"]
        ).most_common(1)[0][0]

    return CategoryResult(
        category=category,
        current_prompt=current_prompt,
        candidate_prompt=candidate_prompt,
        decision=decision,
        loop_count=loop_count,
        final_patch=final_patch,
        final_review=final_review,
        feedback_history=feedback_history,
        runs=runs,
        reviews=reviews,
        answer_format_profile=answer_format_profile,
        generator_examples=generator_examples,
        evaluator_examples=evaluator_examples,
        holdout_examples=holdout_examples,
        main_failure_mode=main_failure_mode,
        log_lines=log_lines,
    )


def run_harness(args: argparse.Namespace) -> None:
    print(
        f"Loading v1 evaluation results from {args.source_results}",
        flush=True,
    )
    results = read_results(args.source_results)
    errors = filter_error_rows(results)
    output_dir = args.output_dir / args.model_id / "prompt_harness"
    frontend_dir = args.output_dir.parent / "dashboards"
    dashboard_path = frontend_dir / f"prompt_review_{args.model_id}.html"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_categories = categories_from_results(results)
    print(
        f"Found {len(results)} total examples across {len(all_categories)} questions; "
        f"{len(errors)} incorrect examples across {errors['category'].nunique()} categories",
        flush=True,
    )
    print("Deriving answer format profiles from golden answers", flush=True)
    answer_format_profiles = derive_answer_format_profiles(results)
    dry_run = bool(getattr(args, "dry_run", False))
    use_llm = not dry_run
    agent_mode = LLM_MODEL_ID if use_llm else "deterministic offline mode (--dry-run)"
    print(f"Prompt improvement agent mode: {agent_mode}", flush=True)
    generator_agent = (
        build_deepseek_agent(PromptPatch, GENERATOR_SYSTEM_PROMPT) if use_llm else None
    )
    evaluator_agent = (
        build_deepseek_agent(PromptReview, EVALUATOR_SYSTEM_PROMPT) if use_llm else None
    )

    splits = create_splits(
        errors,
        generator_size=args.generator_examples,
        evaluator_size=args.evaluator_examples,
    )
    print(
        "Created splits: "
        f"generator_dev={len(splits['generator_dev'])}, "
        f"evaluator_dev={len(splits['evaluator_dev'])}, "
        f"holdout_eval={len(splits['holdout_eval'])}",
        flush=True,
    )
    (output_dir / "splits.json").write_text(
        json.dumps(
            {
                "source_results": str(args.source_results),
                "model_id": args.model_id,
                "agent_model_id": LLM_MODEL_ID if use_llm else None,
                "use_llm": use_llm,
                "splits": splits,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "answer_format_profiles.json").write_text(
        json.dumps(
            {
                category: profile.model_dump(mode="json")
                for category, profile in sorted(answer_format_profiles.items())
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    prompt_overrides = load_prompt_overrides(args.prompts_file)
    candidate_prompts = dict(prompt_overrides)

    status_path = output_dir / "category_status.jsonl"
    prior_status = read_category_status(status_path)
    completed_categories: set[str] = set()
    revise_categories: set[str] = set()
    for category, record in prior_status.items():
        candidate = record.get("candidate_prompt")
        if isinstance(candidate, str):
            candidate_prompts[category] = candidate
        if record.get("decision") == "revise":
            revise_categories.add(category)
        else:
            completed_categories.add(category)
    if prior_status:
        print(
            f"Found {len(prior_status)} prior category statuses in {status_path.name}; "
            f"skipping {len(completed_categories)} and rerunning "
            f"{len(revise_categories)} with decision=revise",
            flush=True,
        )

    for category in all_categories:
        if category not in candidate_prompts:
            meta = metadata_for_category(results, str(category))
            candidate_prompts[category] = prompt_overrides.get(
                category
            ) or default_prompt_from_metadata(meta)

    to_process = [
        category for category in all_categories if category not in completed_categories
    ]

    status_lock = threading.Lock()
    max_workers = max(1, int(getattr(args, "max_workers", 1)))
    print(
        f"Running improvement on {len(to_process)} categories with "
        f"{max_workers} worker(s)",
        flush=True,
    )

    category_runs: list[dict[str, Any]] = []
    evaluator_reviews: list[dict[str, Any]] = []
    accepted_patches: list[dict[str, Any]] = []
    rejected_patches: list[dict[str, Any]] = []
    prompt_diffs: list[dict[str, Any]] = []
    dashboard_records: list[DashboardCategoryRecord] = []
    completed = 0
    total = len(to_process)

    def merge_result(result: CategoryResult) -> None:
        nonlocal completed
        completed += 1
        prefix = f"[{completed}/{total}]"
        for index, line in enumerate(result.log_lines):
            print(f"{prefix} {line}" if index == 0 else line, flush=True)

        candidate_prompts[result.category] = result.candidate_prompt
        category_runs.extend(result.runs)
        evaluator_reviews.extend(result.reviews)

        if result.final_patch is not None and result.final_review is not None:
            if result.final_review.decision == "accept":
                accepted_patches.append(result.final_patch.model_dump(mode="json"))
            else:
                rejected_patches.append(
                    {
                        "category": result.category,
                        "reason": result.final_review.rationale,
                        "requested_changes": result.final_review.requested_changes,
                    }
                )
            prompt_diffs.append(
                {
                    "category": result.category,
                    "diff": result.final_patch.prompt_diff_summary,
                    "changed_rules": result.final_patch.changed_rules,
                    "v1_length": len(result.current_prompt),
                    "candidate_v2_length": len(result.candidate_prompt),
                }
            )
        dashboard_records.append(
            build_dashboard_record(
                category=result.category,
                results=results,
                errors=errors,
                splits=splits,
                answer_format_profiles=answer_format_profiles,
                current_prompt=result.current_prompt,
                candidate_prompt=result.candidate_prompt,
                decision=result.decision,
                loop_count=result.loop_count,
                main_failure_mode=result.main_failure_mode,
                final_patch=result.final_patch,
                final_review=result.final_review,
                feedback_history=result.feedback_history,
                generator_examples=result.generator_examples,
                evaluator_examples=result.evaluator_examples,
                holdout_examples=result.holdout_examples,
            )
        )

        append_category_status(
            status_path,
            {
                "category": result.category,
                "decision": result.decision,
                "loop_count": result.loop_count,
                "candidate_prompt": result.candidate_prompt,
                "v1_prompt": result.current_prompt,
                "main_failure_mode": result.main_failure_mode,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
            status_lock,
        )

    if to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_category,
                    category,
                    results=results,
                    errors=errors,
                    splits=splits,
                    answer_format_profiles=answer_format_profiles,
                    prompt_overrides=prompt_overrides,
                    generator_agent=generator_agent,
                    evaluator_agent=evaluator_agent,
                    args=args,
                    use_llm=use_llm,
                ): category
                for category in to_process
            }
            for future in as_completed(futures):
                category = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(
                        f"  ERROR processing {category}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                merge_result(result)

    dashboard_by_category = {record.category: record for record in dashboard_records}
    for category in all_categories:
        if category in dashboard_by_category:
            continue
        meta = metadata_for_category(results, category)
        current_prompt = prompt_overrides.get(category) or default_prompt_from_metadata(
            meta
        )
        prior_record = prior_status.get(category, {})
        candidate_prompt = candidate_prompts.get(category, current_prompt)
        category_error_rows = rows_for_category(errors, category)
        if category_error_rows.empty:
            decision = str(prior_record.get("decision") or "no_failures")
            main_failure_mode = "none"
        else:
            decision = str(prior_record.get("decision") or "skipped_existing")
            main_failure_mode = str(
                prior_record.get("main_failure_mode")
                or Counter(
                    str(mode) for mode in category_error_rows["failure_mode"]
                ).most_common(1)[0][0]
            )
        loop_count = int(prior_record.get("loop_count") or 0)
        dashboard_by_category[category] = build_dashboard_record(
            category=category,
            results=results,
            errors=errors,
            splits=splits,
            answer_format_profiles=answer_format_profiles,
            current_prompt=str(prior_record.get("v1_prompt") or current_prompt),
            candidate_prompt=candidate_prompt,
            decision=decision,
            loop_count=loop_count,
            main_failure_mode=main_failure_mode,
        )
    dashboard_records = [
        dashboard_by_category[category]
        for category in all_categories
        if category in dashboard_by_category
    ]

    print(f"Writing prompt harness artifacts to {output_dir}", flush=True)
    write_jsonl(output_dir / "category_runs.jsonl", category_runs)
    write_jsonl(output_dir / "evaluator_reviews.jsonl", evaluator_reviews)
    write_jsonl(output_dir / "accepted_patches.jsonl", accepted_patches)
    write_jsonl(output_dir / "rejected_patches.jsonl", rejected_patches)
    write_jsonl(output_dir / "prompt_diffs.jsonl", prompt_diffs)
    write_prompt_module(output_dir / "prompts_candidate_v2.py", candidate_prompts)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_path.write_text(
        render_dashboard(
            model_id=args.model_id,
            source_results=args.source_results,
            splits=splits,
            category_records=dashboard_records,
        ),
        encoding="utf-8",
    )
    print(
        "Prompt improvement complete: "
        f"accepted={len(accepted_patches)}, rejected={len(rejected_patches)}, "
        f"dashboard={dashboard_path}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-results",
        type=Path,
        default=Path("outputs/v1/cuad_dspy_eval_results.csv"),
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=Path("prompts/system_prompts_v1.py"),
    )
    parser.add_argument("--model-id", default="v2")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--generator-examples", type=int, default=20)
    parser.add_argument("--evaluator-examples", type=int, default=20)
    parser.add_argument("--max-loops", type=int, default=3)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help=(
            "Number of categories to process in parallel. Each worker holds "
            "one category's full generator/evaluator loop. Default: 4."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip LLM calls and use a deterministic rule-based generator/"
            "evaluator. Default mode runs PydanticAI agents against "
            f"{LLM_MODEL_ID} and requires DEEPSEEK_API_KEY."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generator_examples < 1:
        raise ValueError("--generator-examples must be >= 1")
    if args.generator_examples > 20:
        raise ValueError("--generator-examples must be <= 20")
    if args.evaluator_examples < 0:
        raise ValueError("--evaluator-examples must be >= 0")
    if args.evaluator_examples > 20:
        raise ValueError("--evaluator-examples must be <= 20")
    if not 1 <= args.max_loops <= 3:
        raise ValueError("--max-loops must be between 1 and 3")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    run_harness(args)


if __name__ == "__main__":
    main()
