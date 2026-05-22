"""Sentence splitting with stable offsets for CUAD RAG experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


LEGAL_ABBREVIATIONS = {
    "inc.",
    "corp.",
    "ltd.",
    "llc.",
    "co.",
    "no.",
    "sec.",
    "art.",
    "ex.",
    "u.s.",
    "u.s.a.",
    "e.g.",
    "i.e.",
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
}

@dataclass(frozen=True)
class SentenceSpan:
    document_row_id: int
    sentence_id: str
    sentence_index: int
    raw_text: str
    normalized_text: str
    start_char: int
    end_char: int
    page_number: int | None = None
    section_number: str | None = None
    section_title: str | None = None
    clause_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "document_row_id": self.document_row_id,
            "sentence_id": self.sentence_id,
            "sentence_index": self.sentence_index,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_number": self.page_number,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "clause_path": self.clause_path,
        }


def normalize_sentence_text(value: str) -> str:
    """Normalize text for exact sentence matching while preserving meaning."""
    text = str(value or "")
    replacements = {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def _previous_token(text: str, end_index: int) -> str:
    cursor = end_index
    while cursor >= 0 and not text[cursor].isspace():
        cursor -= 1
    return text[cursor + 1 : end_index + 1].lower()


def _next_nonspace(text: str, start_index: int) -> str:
    match = re.search(r"\S", text[start_index:])
    return text[start_index + match.start()] if match else ""


def _is_sentence_boundary(text: str, index: int) -> bool:
    char = text[index]
    if char not in ".?!":
        return False
    token = _previous_token(text, index)
    if token in LEGAL_ABBREVIATIONS:
        return False
    if re.fullmatch(r"\d{1,3}\.", token):
        return False
    if re.fullmatch(r"[A-Z]\.", token):
        return False

    lookahead = index + 1
    while lookahead < len(text) and text[lookahead] in "\"')]}":
        lookahead += 1
    if lookahead >= len(text):
        return True
    if not text[lookahead].isspace():
        return False
    next_char = _next_nonspace(text, lookahead)
    return (
        not next_char
        or next_char.isupper()
        or next_char.isdigit()
        or next_char in "\"'([•‣▪▫◦●○"
    )


def fallback_sentence_segments(text: str) -> list[tuple[int, int, str]]:
    """Split text with deterministic legal-aware rules and exact offsets."""
    segments: list[tuple[int, int, str]] = []
    start = 0
    length = len(text)
    while start < length and text[start].isspace():
        start += 1

    index = start
    while index < length:
        if _is_sentence_boundary(text, index):
            end = index + 1
            while end < length and text[end] in "\"')]}":
                end += 1
            raw = text[start:end].strip()
            if raw:
                raw_start = start + len(text[start:end]) - len(text[start:end].lstrip())
                raw_end = end - len(text[start:end]) + len(text[start:end].rstrip())
                segments.append((raw_start, raw_end, text[raw_start:raw_end]))
            start = end
            while start < length and text[start].isspace():
                start += 1
            index = start
            continue
        index += 1

    if start < length:
        raw = text[start:].strip()
        if raw:
            raw_start = start + len(text[start:]) - len(text[start:].lstrip())
            raw_end = length - len(text[start:]) + len(text[start:].rstrip())
            segments.append((raw_start, raw_end, text[raw_start:raw_end]))
    return segments


def _trimmed_segment(text: str, start: int, end: int) -> tuple[int, int, str] | None:
    raw = text[start:end]
    stripped = raw.strip()
    if not stripped:
        return None
    raw_start = start + len(raw) - len(raw.lstrip())
    raw_end = end - len(raw) + len(raw.rstrip())
    return raw_start, raw_end, text[raw_start:raw_end]


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 120 or len(stripped.split()) > 14:
        return False
    if re.search(r"[.!?;:]$", stripped):
        return False
    letters = [character for character in stripped if character.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for character in letters if character.isupper()) / len(letters)
    return uppercase_ratio >= 0.65 or stripped.istitle()


def paragraph_blocks(text: str) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"\n[ \t]*\n+", text):
        trimmed = _trimmed_segment(text, start, match.start())
        if trimmed:
            blocks.append(trimmed)
        start = match.end()
    trimmed = _trimmed_segment(text, start, len(text))
    if trimmed:
        blocks.append(trimmed)
    return blocks


def heading_blocks(start: int, block: str) -> list[tuple[int, int, str]]:
    parts: list[tuple[int, int, str]] = []
    cursor = 0
    lines = block.splitlines(keepends=True)
    while cursor < len(lines):
        line = lines[cursor]
        line_start = start + sum(len(value) for value in lines[:cursor])
        line_end = line_start + len(line)
        if cursor < len(lines) - 1 and _is_heading_line(line):
            trimmed = _trimmed_segment(block, line_start - start, line_end - start)
            if trimmed:
                block_start, block_end, text = trimmed
                parts.append((start + block_start, start + block_end, text))
            cursor += 1
            continue

        remainder_start = line_start
        remainder = "".join(lines[cursor:])
        trimmed = _trimmed_segment(block, remainder_start - start, len(block))
        if trimmed:
            block_start, block_end, text = trimmed
            parts.append((start + block_start, start + block_end, text))
        break
    return parts or [(start, start + len(block), block)]


def structural_text_blocks(text: str) -> list[tuple[int, int, str]]:
    """Split headings and paragraph breaks before sentence-boundary chunking."""
    blocks: list[tuple[int, int, str]] = []
    for paragraph_start, _, paragraph in paragraph_blocks(text):
        for heading_start, _, heading_block in heading_blocks(paragraph_start, paragraph):
            blocks.append((heading_start, heading_start + len(heading_block), heading_block))
    return blocks


def package_sentence_segments(text: str) -> list[tuple[int, int, str]] | None:
    """Try a package-backed splitter and reconstruct exact offsets."""
    try:
        import pysbd  # type: ignore[import-not-found]
    except Exception:
        return None

    try:
        segmenter = pysbd.Segmenter(language="en", clean=False)
        pieces = [piece for piece in segmenter.segment(text) if piece.strip()]
    except Exception:
        return None

    segments: list[tuple[int, int, str]] = []
    cursor = 0
    for piece in pieces:
        stripped = piece.strip()
        start = text.find(stripped, cursor)
        if start < 0:
            return None
        end = start + len(stripped)
        segments.append((start, end, text[start:end]))
        cursor = end
    return segments


def split_sentence_segments(text: str) -> list[tuple[int, int, str]]:
    """Return `(start, end, text)` sentence segments with exact source offsets."""
    segments: list[tuple[int, int, str]] = []
    for block_start, _, block_text in structural_text_blocks(text):
        block_segments = fallback_sentence_segments(block_text)
        for start, end, raw in block_segments:
            trimmed = _trimmed_segment(block_text, start, end)
            if trimmed:
                segment_start, segment_end, segment_text = trimmed
                segments.append(
                    (
                        block_start + segment_start,
                        block_start + segment_end,
                        segment_text,
                    )
                )
    return segments


def build_sentence_spans(
    document_row_id: int,
    text: str,
    *,
    section_metadata: dict[int, tuple[str | None, str | None, list[str]]] | None = None,
) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    metadata = section_metadata or {}
    for index, (start, end, raw) in enumerate(split_sentence_segments(text)):
        section_number, section_title, clause_path = metadata.get(start, (None, None, []))
        spans.append(
            SentenceSpan(
                document_row_id=int(document_row_id),
                sentence_id=f"{int(document_row_id)}:s:{index}",
                sentence_index=index,
                raw_text=raw,
                normalized_text=normalize_sentence_text(raw),
                start_char=start,
                end_char=end,
                page_number=None,
                section_number=section_number,
                section_title=section_title,
                clause_path=list(clause_path),
            )
        )
    return spans


def sentence_dicts(spans: Iterable[SentenceSpan]) -> list[dict[str, object]]:
    return [span.to_dict() for span in spans]
