from __future__ import annotations

from cuad_agent.rag.coverage import coverage_at_k
from cuad_agent.rag.gold_answers import evaluate_row_eligibility, split_golden_answer_sentences
from cuad_agent.rag.sentences import build_sentence_spans


def test_exact_sentence_eligibility_requires_full_sentence_match() -> None:
    contract = (
        "The buyer may inspect records. "
        "Neither party may assign this Agreement without consent."
    )
    spans = build_sentence_spans(1, contract)
    answers = [{"text": "Neither party may assign this Agreement without consent."}]

    record = evaluate_row_eligibility(
        row_id="1:0",
        document_row_id=1,
        question_index=0,
        category="Anti-Assignment",
        answers=answers,
        is_impossible=False,
        contract_sentences=spans,
        contract_text=contract,
    )

    assert record.is_eligible
    assert record.per_row_gold_sentence_contract_coverage == 1.0
    assert record.per_row_raw_contract_sentence_coverage == 1.0
    assert record.matched_sentence_ids == ["1:s:1"]


def test_partial_span_is_not_sentence_extraction_eligible() -> None:
    spans = build_sentence_spans(
        1,
        "Neither party may assign this Agreement without consent.",
    )
    record = evaluate_row_eligibility(
        row_id="1:0",
        document_row_id=1,
        question_index=0,
        category="Anti-Assignment",
        answers=[{"text": "party may assign this Agreement"}],
        is_impossible=False,
        contract_sentences=spans,
        contract_text="Neither party may assign this Agreement without consent.",
    )

    assert not record.is_eligible
    assert record.reason == "partial_span_inside_sentence"
    assert record.per_row_gold_sentence_contract_coverage == 0.0
    assert record.per_row_raw_contract_sentence_coverage == 1.0


def test_golden_answer_newline_does_not_force_chunk_split() -> None:
    chunks = split_golden_answer_sentences(
        [
            (
                "The buyer shall maintain insurance\n"
                "and comply with all applicable laws."
            )
        ]
    )

    assert chunks == [
        "The buyer shall maintain insurance\nand comply with all applicable laws."
    ]


def test_coverage_at_k_uses_sentence_ids_only() -> None:
    result = coverage_at_k(["1:s:2", "1:s:4"], ["1:s:0", "1:s:2", "1:s:3"], 3)

    assert result["covered_sentence_count"] == 1
    assert result["gold_sentence_coverage"] == 0.5
    assert not result["all_gold_sentences_covered"]
    assert result["first_covering_rank"] == 2
