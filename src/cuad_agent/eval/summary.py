"""Evaluation summary helpers."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from cuad_agent.constants import (
    EVAL_QUESTION_COUNT,
    EVAL_QUESTION_INDICES,
    NO_ANSWER_MARKERS,
)
from cuad_agent.eval.metrics import normalize_answer

__all__ = [
    "detection_metrics",
    "predicted_no_answer_mask",
    "summarize_results",
]


def predicted_no_answer_mask(results: pd.DataFrame) -> pd.Series:
    """Per-row model decision that the clause is absent.

    Combines the explicit ``predicted_marked_impossible`` flag with the answer
    text normalizing to a NO_ANSWER marker, so it agrees with how
    ``token_overlap_f1`` scores the no-answer class.
    """
    if results.empty:
        return pd.Series([], dtype=bool)
    marked = results["predicted_marked_impossible"].astype(bool)
    normalized = (
        results["predicted_answer"]
        .fillna("")
        .map(lambda text: normalize_answer(str(text)) in NO_ANSWER_MARKERS)
    )
    return marked | normalized


def detection_metrics(results: pd.DataFrame) -> dict[str, Any]:
    """No-answer vs answer detection accuracy.

    Many CUAD gold answers are "no answer", so aggregate F1 is dominated by the
    no-answer class. These metrics split detection accuracy by gold class to
    show how well the model identifies whether a clause is present at all.
    """
    if results.empty:
        return {
            "gold_no_answer_count": 0,
            "gold_answer_count": 0,
            "predicted_no_answer_count": 0,
            "no_answer_detection_accuracy": None,
            "answer_detection_accuracy": None,
            "detection_accuracy": None,
        }
    gold_no = results["gold_marked_impossible"].astype(bool)
    pred_no = predicted_no_answer_mask(results)
    gold_no_count = int(gold_no.sum())
    gold_answer_count = int((~gold_no).sum())
    return {
        "gold_no_answer_count": gold_no_count,
        "gold_answer_count": gold_answer_count,
        "predicted_no_answer_count": int(pred_no.sum()),
        "no_answer_detection_accuracy": (
            float((pred_no & gold_no).sum() / gold_no_count * 100)
            if gold_no_count
            else None
        ),
        "answer_detection_accuracy": (
            float((~pred_no & ~gold_no).sum() / gold_answer_count * 100)
            if gold_answer_count
            else None
        ),
        "detection_accuracy": float((pred_no == gold_no).mean() * 100),
    }


def summarize_results(
    results: pd.DataFrame,
    *,
    args: argparse.Namespace,
    selected_document_row_ids: list[int],
) -> dict[str, Any]:
    per_category_df = (
        results.groupby(["question_index", "category"], as_index=False)
        .agg(
            mean_token_f1=("token_f1", "mean"),
            correct_at_0_5=("correct_at_0_5", "mean"),
            count=("token_f1", "size"),
        )
        .sort_values(["question_index"])
    )
    per_category = [
        {
            "question_index": int(row.question_index),
            "category": str(row.category),
            "mean_token_f1": float(row.mean_token_f1 * 100),
            "correct_at_0_5": float(row.correct_at_0_5 * 100),
            "count": int(row.count),
        }
        for row in per_category_df.itertuples(index=False)
    ]
    return {
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "model_id": str(args.model_id),
        "selected_document_row_ids": selected_document_row_ids,
        "total_examples": int(len(results)),
        "questions_per_contract": EVAL_QUESTION_COUNT,
        "evaluated_question_numbers": [index + 1 for index in EVAL_QUESTION_INDICES],
        "evaluated_question_indices": list(EVAL_QUESTION_INDICES),
        "agent_count": int(results["question_index"].nunique()),
        "model": args.model,
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "num_threads": int(args.num_threads),
        "dry_run": bool(args.dry_run),
        "prompts_file": str(args.prompts_file)
        if getattr(args, "prompts_file", None)
        else None,
        "eval_split": str(args.eval_split)
        if getattr(args, "eval_split", None)
        else None,
        "context_mode": str(getattr(args, "context_mode", "raw")),
        "overlap_accuracy_mean_f1": float(results["token_f1"].mean() * 100),
        "correct_at_0_5": float(results["correct_at_0_5"].mean() * 100),
        **detection_metrics(results),
        "per_category": per_category,
    }
