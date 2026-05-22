"""LangChain recursive legal-hierarchy chunking for CUAD contracts."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from cuad_agent.rag.chunks import RagChunk
from cuad_agent.rag.sentences import SentenceSpan, normalize_sentence_text


LEGAL_RECURSIVE_CHUNKING_VERSION = "legal-recursive-v1"
LEGAL_RECURSIVE_CHUNK_SIZE = 1200
LEGAL_RECURSIVE_CHUNK_OVERLAP = 150
LEGAL_RECURSIVE_SEPARATORS = [
    r"\n\s*(?:ARTICLE|Article)\s+[IVXLC\d]+[^\n]*\n",
    r"\n\s*(?:SECTION|Section)\s+\d+(?:\.\d+)*[^\n]*\n",
    r"\n\s*\d{1,3}\.\s+",
    r"\n\s*\d{1,3}\)\s+",
    r"\n\s*\([A-Za-z0-9]{1,4}\)\s+",
    r"\n\s*[A-Za-z]\)\s+",
    r"\n\s*[•‣▪▫◦●○*-]\s+",
    r"\n\s*\n+",
    r"\n",
    r"(?<=[.;:!?])\s+",
    r"\s+",
    "",
]


@dataclass(frozen=True)
class LegalRecursiveConfig:
    chunk_size: int = LEGAL_RECURSIVE_CHUNK_SIZE
    chunk_overlap: int = LEGAL_RECURSIVE_CHUNK_OVERLAP
    separators: tuple[str, ...] = tuple(LEGAL_RECURSIVE_SEPARATORS)
    keep_separator: str = "start"
    is_separator_regex: bool = True
    add_start_index: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": list(self.separators),
            "keep_separator": self.keep_separator,
            "is_separator_regex": self.is_separator_regex,
            "add_start_index": self.add_start_index,
        }


def build_legal_recursive_splitter(
    config: LegalRecursiveConfig | None = None,
) -> RecursiveCharacterTextSplitter:
    cfg = config or LegalRecursiveConfig()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=list(cfg.separators),
        keep_separator=cfg.keep_separator,
        is_separator_regex=cfg.is_separator_regex,
        add_start_index=cfg.add_start_index,
        strip_whitespace=True,
    )


def sentence_ids_for_chunk(
    *,
    chunk_start: int,
    chunk_end: int,
    sentence_spans: list[SentenceSpan],
) -> list[str]:
    ids: list[str] = []
    for sentence in sentence_spans:
        if sentence.start_char < chunk_end and sentence.end_char > chunk_start:
            ids.append(sentence.sentence_id)
    return ids


def section_metadata_for_chunk(
    sentence_spans: list[SentenceSpan],
    sentence_ids: list[str],
) -> tuple[str | None, str | None, list[str]]:
    if not sentence_ids:
        return None, None, []
    by_id = {sentence.sentence_id: sentence for sentence in sentence_spans}
    first = by_id.get(sentence_ids[0])
    if first is None:
        return None, None, []
    return first.section_number, first.section_title, list(first.clause_path)


def build_legal_recursive_chunks_for_contract(
    *,
    document_row_id: int,
    text: str,
    sentence_spans: list[SentenceSpan],
    config: LegalRecursiveConfig | None = None,
) -> list[RagChunk]:
    splitter = build_legal_recursive_splitter(config)
    documents = splitter.create_documents(
        [text],
        metadatas=[{"document_row_id": int(document_row_id)}],
    )
    chunks: list[RagChunk] = []
    fallback_cursor = 0
    for index, document in enumerate(documents):
        chunk_text = document.page_content.strip()
        if not chunk_text:
            continue
        raw_start = document.metadata.get("start_index")
        if isinstance(raw_start, int):
            start = raw_start
        else:
            start = text.find(chunk_text, fallback_cursor)
            if start < 0:
                start = text.find(chunk_text)
            if start < 0:
                start = fallback_cursor
        end = start + len(chunk_text)
        fallback_cursor = max(fallback_cursor, end)
        sentence_ids = sentence_ids_for_chunk(
            chunk_start=start,
            chunk_end=end,
            sentence_spans=sentence_spans,
        )
        section_number, section_title, clause_path = section_metadata_for_chunk(
            sentence_spans,
            sentence_ids,
        )
        chunks.append(
            RagChunk(
                chunk_id=f"{int(document_row_id)}:lr:{index}",
                document_row_id=int(document_row_id),
                text=chunk_text,
                normalized_text=normalize_sentence_text(chunk_text),
                chunk_type="legal_recursive",
                sentence_ids=sentence_ids,
                start_char=start,
                end_char=end,
                page_number=None,
                section_number=section_number,
                section_title=section_title,
                clause_path=clause_path,
            )
        )
    return chunks


def build_legal_recursive_chunks(
    *,
    text_by_document_id: dict[int, str],
    sentence_lookup: dict[int, list[SentenceSpan]],
    config: LegalRecursiveConfig | None = None,
) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for document_row_id, text in sorted(text_by_document_id.items()):
        chunks.extend(
            build_legal_recursive_chunks_for_contract(
                document_row_id=document_row_id,
                text=text,
                sentence_spans=sentence_lookup.get(document_row_id, []),
                config=config,
            )
        )
    return chunks
