"""Evaluation example and split helpers."""

from __future__ import annotations

from typing import Any

from cuad_agent.data.sampling import (
    build_eval_sample,
    evaluation_row_id,
    filter_eval_rows_by_split,
    load_eval_split_ids,
    select_evaluation_set,
)

__all__ = [
    "answer_texts",
    "build_eval_sample",
    "evaluation_row_id",
    "filter_eval_rows_by_split",
    "load_eval_split_ids",
    "select_evaluation_set",
]


def answer_texts(answers: Any) -> list[str]:
    if not isinstance(answers, list):
        return []
    return [
        str(answer.get("text", "")) for answer in answers if isinstance(answer, dict)
    ]
