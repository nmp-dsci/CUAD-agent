"""Chunk schema for RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cuad_agent.rag.sentences import SentenceSpan


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    document_row_id: int
    text: str
    normalized_text: str
    chunk_type: Literal["sentence", "legal_recursive"]
    sentence_ids: list[str]
    start_char: int
    end_char: int
    page_number: int | None
    section_number: str | None
    section_title: str | None
    clause_path: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_row_id": self.document_row_id,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "chunk_type": self.chunk_type,
            "sentence_ids": self.sentence_ids,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_number": self.page_number,
            "section_number": self.section_number,
            "section_title": self.section_title,
            "clause_path": self.clause_path,
        }


def chunk_from_sentence(span: SentenceSpan) -> RagChunk:
    return RagChunk(
        chunk_id=span.sentence_id,
        document_row_id=span.document_row_id,
        text=span.raw_text,
        normalized_text=span.normalized_text,
        chunk_type="sentence",
        sentence_ids=[span.sentence_id],
        start_char=span.start_char,
        end_char=span.end_char,
        page_number=span.page_number,
        section_number=span.section_number,
        section_title=span.section_title,
        clause_path=list(span.clause_path),
    )


def chunks_from_sentences(spans: list[SentenceSpan]) -> list[RagChunk]:
    return [chunk_from_sentence(span) for span in spans]


def chunk_from_dict(row: dict[str, object]) -> RagChunk:
    return RagChunk(
        chunk_id=str(row["chunk_id"]),
        document_row_id=int(row["document_row_id"]),
        text=str(row["text"]),
        normalized_text=str(row["normalized_text"]),
        chunk_type=row.get("chunk_type", "sentence"),  # type: ignore[arg-type]
        sentence_ids=[str(value) for value in row.get("sentence_ids", [])],
        start_char=int(row["start_char"]),
        end_char=int(row["end_char"]),
        page_number=(
            int(row["page_number"]) if row.get("page_number") is not None else None
        ),
        section_number=(
            str(row["section_number"]) if row.get("section_number") is not None else None
        ),
        section_title=(
            str(row["section_title"]) if row.get("section_title") is not None else None
        ),
        clause_path=[str(value) for value in row.get("clause_path", [])],
    )
