"""Framework-neutral evaluation metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

from cuad_agent.constants import NO_ANSWER_MARKERS

__all__ = [
    "cuad_overlap_metric",
    "normalize_answer",
    "parse_bool",
    "token_overlap_f1",
    "tokens",
]


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization."""
    text = str(text or "").lower()
    punctuation = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    normalized = normalize_answer(str(value))
    return normalized in {"true", "yes", "y", "1", "marked impossible", "impossible"}


def token_overlap_f1(prediction: str, gold_answers: list[str]) -> float:
    """CUAD-adapted SQuAD token F1 over all required gold spans."""
    pred_norm = normalize_answer(prediction or "")
    gold_toks = tokens(" ".join(gold_answers or []))
    pred_toks = pred_norm.split() if pred_norm else []

    if not gold_toks:
        return 1.0 if pred_norm in NO_ANSWER_MARKERS else 0.0
    if not pred_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    same_count = sum(common.values())
    if same_count == 0:
        return 0.0
    precision = same_count / len(pred_toks)
    recall = same_count / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def cuad_overlap_metric(example: Any, pred: Any, trace: Any = None) -> float:
    return token_overlap_f1(
        str(getattr(pred, "answer", "")), list(example.gold_answers)
    )
