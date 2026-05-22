"""Evaluation example and split helpers."""

from __future__ import annotations

from cuad_agent.data.sampling import (
    build_eval_sample,
    evaluation_row_id,
    filter_eval_rows_by_split,
    load_eval_split_ids,
    select_evaluation_set,
)
from cuad_agent.evaluators.dspy_runner import (
    answer_texts,
    build_devset,
)

__all__ = [
    "answer_texts",
    "build_devset",
    "build_eval_sample",
    "evaluation_row_id",
    "filter_eval_rows_by_split",
    "load_eval_split_ids",
    "select_evaluation_set",
]
