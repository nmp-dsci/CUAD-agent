from __future__ import annotations

from cuad_agent.rag.sentences import build_sentence_spans, normalize_sentence_text


def test_sentence_splitter_preserves_offsets_and_legal_abbreviations() -> None:
    text = (
        "Acme Inc. may assign this Agreement only with consent. "
        "Notices must be in writing."
    )
    spans = build_sentence_spans(7, text)

    assert len(spans) == 2
    assert spans[0].sentence_id == "7:s:0"
    assert spans[0].raw_text == "Acme Inc. may assign this Agreement only with consent."
    assert text[spans[0].start_char : spans[0].end_char] == spans[0].raw_text
    assert spans[1].raw_text == "Notices must be in writing."


def test_normalize_sentence_text_keeps_exact_matching_conservative() -> None:
    assert (
        normalize_sentence_text(" A\u00a0sentence   , with text. ")
        == "A sentence, with text."
    )


def test_sentence_splitter_keeps_list_markers_with_sentence_text() -> None:
    text = (
        "SERVICES PROVIDED\n\n"
        "The supplier shall provide support.\n\n"
        "1. First numbered obligation applies.\n"
        "2. Second numbered obligation applies.\n\n"
        "(a) Alpha list item applies.\n"
        "(b) Beta list item applies.\n"
        "(1) Numeric parenthetical item applies.\n"
        "(h) Alphabetic parenthetical item applies.\n"
        "• Bullet point item applies."
    )
    spans = build_sentence_spans(8, text)

    assert [span.raw_text for span in spans] == [
        "SERVICES PROVIDED",
        "The supplier shall provide support.",
        "1. First numbered obligation applies.",
        "2. Second numbered obligation applies.",
        "(a) Alpha list item applies.",
        "(b) Beta list item applies.",
        "(1) Numeric parenthetical item applies.",
        "(h) Alphabetic parenthetical item applies.",
        "• Bullet point item applies.",
    ]
    for span in spans:
        assert text[span.start_char : span.end_char] == span.raw_text


def test_sentence_splitter_does_not_chunk_inline_parenthesized_points() -> None:
    text = (
        "The licensee shall (a) maintain insurance; "
        "(b) notify the licensor promptly; and "
        "(c) return confidential information. Section 2(a) remains unchanged."
    )
    spans = build_sentence_spans(9, text)

    assert [span.raw_text for span in spans] == [
        "The licensee shall (a) maintain insurance; (b) notify the licensor promptly; and (c) return confidential information.",
        "Section 2(a) remains unchanged.",
    ]
    for span in spans:
        assert text[span.start_char : span.end_char] == span.raw_text


def test_sentence_splitter_does_not_chunk_roman_or_numbered_parentheticals() -> None:
    text = (
        "The buyer shall (i) pay fees, (ii) keep records, (iii) notify seller, "
        "(1) maintain insurance, (2) comply with laws, and (30) preserve data. "
        "The seller may audit records."
    )
    spans = build_sentence_spans(10, text)

    assert [span.raw_text for span in spans] == [
        "The buyer shall (i) pay fees, (ii) keep records, (iii) notify seller, (1) maintain insurance, (2) comply with laws, and (30) preserve data.",
        "The seller may audit records.",
    ]
    for span in spans:
        assert text[span.start_char : span.end_char] == span.raw_text
