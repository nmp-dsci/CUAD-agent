"""Golden-answer sentence matching for CUAD RAG eligibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cuad_agent.rag.sentences import (
    SentenceSpan,
    normalize_sentence_text,
    split_sentence_segments,
)


@dataclass(frozen=True)
class GoldenSentence:
    text: str
    normalized_text: str
    matched_sentence_id: str | None = None
    raw_contract_match: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "normalized_text": self.normalized_text,
            "matched_sentence_id": self.matched_sentence_id,
            "raw_contract_match": self.raw_contract_match,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EligibilityRecord:
    row_id: str
    document_row_id: int
    question_index: int
    category: str
    is_eligible: bool
    gold_sentence_count: int
    matched_sentence_count: int
    raw_contract_sentence_match_count: int
    per_row_gold_sentence_contract_coverage: float
    per_row_raw_contract_sentence_coverage: float
    reason: str | None = None
    golden_sentences: list[GoldenSentence] = field(default_factory=list)

    @property
    def matched_sentence_ids(self) -> list[str]:
        return [
            sentence.matched_sentence_id
            for sentence in self.golden_sentences
            if sentence.matched_sentence_id
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "row_id": self.row_id,
            "document_row_id": self.document_row_id,
            "question_index": self.question_index,
            "category": self.category,
            "is_eligible": self.is_eligible,
            "gold_sentence_count": self.gold_sentence_count,
            "matched_sentence_count": self.matched_sentence_count,
            "raw_contract_sentence_match_count": self.raw_contract_sentence_match_count,
            "per_row_gold_sentence_contract_coverage": (
                self.per_row_gold_sentence_contract_coverage
            ),
            "per_row_raw_contract_sentence_coverage": (
                self.per_row_raw_contract_sentence_coverage
            ),
            "reason": self.reason,
            "matched_sentence_ids": self.matched_sentence_ids,
            "golden_sentences": [
                sentence.to_dict() for sentence in self.golden_sentences
            ],
        }


def answer_texts(answers: Any) -> list[str]:
    if not isinstance(answers, list):
        return []
    return [
        str(answer.get("text", "")).strip()
        for answer in answers
        if isinstance(answer, dict) and str(answer.get("text", "")).strip()
    ]


def split_golden_answer_sentences(answer_values: list[str]) -> list[str]:
    sentences: list[str] = []
    for answer in answer_values:
        span = str(answer).strip()
        if not span:
            continue
        segments = split_sentence_segments(span)
        if segments:
            sentences.extend(text.strip() for _, _, text in segments if text.strip())
        else:
            sentences.append(span)
    return sentences


def build_sentence_lookup(
    contract_sentences: list[SentenceSpan],
) -> dict[str, SentenceSpan]:
    lookup: dict[str, SentenceSpan] = {}
    for sentence in contract_sentences:
        lookup.setdefault(sentence.normalized_text, sentence)
    return lookup


def classify_unmatched_sentence(
    normalized_gold: str,
    contract_sentences: list[SentenceSpan],
) -> str:
    if not normalized_gold:
        return "gold_answer_not_found"
    for sentence in contract_sentences:
        if normalized_gold in sentence.normalized_text:
            return "partial_span_inside_sentence"
    return "gold_answer_not_found"


def evaluate_row_eligibility(
    *,
    row_id: str,
    document_row_id: int,
    question_index: int,
    category: str,
    answers: Any,
    is_impossible: bool,
    contract_sentences: list[SentenceSpan],
    contract_text: str = "",
) -> EligibilityRecord:
    if is_impossible:
        return EligibilityRecord(
            row_id=row_id,
            document_row_id=document_row_id,
            question_index=question_index,
            category=category,
            is_eligible=False,
            gold_sentence_count=0,
            matched_sentence_count=0,
            raw_contract_sentence_match_count=0,
            per_row_gold_sentence_contract_coverage=0.0,
            per_row_raw_contract_sentence_coverage=0.0,
            reason="no_answer_row",
        )

    gold_texts = answer_texts(answers)
    gold_sentences = split_golden_answer_sentences(gold_texts)
    if not gold_sentences:
        return EligibilityRecord(
            row_id=row_id,
            document_row_id=document_row_id,
            question_index=question_index,
            category=category,
            is_eligible=False,
            gold_sentence_count=0,
            matched_sentence_count=0,
            raw_contract_sentence_match_count=0,
            per_row_gold_sentence_contract_coverage=0.0,
            per_row_raw_contract_sentence_coverage=0.0,
            reason="gold_answer_not_found",
        )

    lookup = build_sentence_lookup(contract_sentences)
    normalized_contract_text = normalize_sentence_text(contract_text)
    records: list[GoldenSentence] = []
    for gold in gold_sentences:
        normalized = normalize_sentence_text(gold)
        matched = lookup.get(normalized)
        raw_contract_match = bool(normalized and normalized in normalized_contract_text)
        if matched:
            records.append(
                GoldenSentence(
                    text=gold,
                    normalized_text=normalized,
                    matched_sentence_id=matched.sentence_id,
                    raw_contract_match=raw_contract_match,
                )
            )
        else:
            records.append(
                GoldenSentence(
                    text=gold,
                    normalized_text=normalized,
                    raw_contract_match=raw_contract_match,
                    reason=classify_unmatched_sentence(normalized, contract_sentences),
                )
            )

    matched_count = sum(1 for record in records if record.matched_sentence_id)
    raw_contract_match_count = sum(1 for record in records if record.raw_contract_match)
    coverage = matched_count / len(records) if records else 0.0
    raw_contract_coverage = raw_contract_match_count / len(records) if records else 0.0
    is_eligible = bool(records) and matched_count == len(records)
    reason = (
        None
        if is_eligible
        else next(
            (record.reason for record in records if record.reason),
            "gold_answer_not_found",
        )
    )
    return EligibilityRecord(
        row_id=row_id,
        document_row_id=document_row_id,
        question_index=question_index,
        category=category,
        is_eligible=is_eligible,
        gold_sentence_count=len(records),
        matched_sentence_count=matched_count,
        raw_contract_sentence_match_count=raw_contract_match_count,
        per_row_gold_sentence_contract_coverage=coverage,
        per_row_raw_contract_sentence_coverage=raw_contract_coverage,
        reason=reason,
        golden_sentences=records,
    )
