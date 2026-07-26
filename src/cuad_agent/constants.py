"""Shared CUAD evaluation constants."""

from __future__ import annotations

QUESTION_COUNT = 41
EVAL_QUESTION_COUNT = QUESTION_COUNT
EVAL_QUESTION_INDICES = tuple(range(EVAL_QUESTION_COUNT))

NO_ANSWER = "NO_ANSWER"
NO_ANSWER_MARKERS = {
    "",
    "no answer",
    "none",
    "not found",
    "n/a",
    "na",
    "not applicable",
    "no_answer",
    "noanswer",
}

OUTPUT_STEM = "cuad_langchain_eval"

__all__ = [
    "EVAL_QUESTION_COUNT",
    "EVAL_QUESTION_INDICES",
    "NO_ANSWER",
    "NO_ANSWER_MARKERS",
    "OUTPUT_STEM",
    "QUESTION_COUNT",
]
