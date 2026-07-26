"""Sentence retrieval orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from cuad_agent.rag.chunks import RagChunk
from cuad_agent.rag.hierarchy import HierarchicalRetriever, SectionNode
from cuad_agent.rag.indexes import BM25SentenceIndex, DenseSentenceIndex, SearchResult


@dataclass
class SentenceRetriever:
    method: str
    index: BM25SentenceIndex | DenseSentenceIndex

    def search(
        self, query: str, *, document_row_id: int, top_k: int
    ) -> list[SearchResult]:
        return self.index.search(query, document_row_id=document_row_id, top_k=top_k)


def build_retriever(
    method: str,
    chunks: list[RagChunk],
    *,
    embedding_model: str,
) -> SentenceRetriever:
    if method in {"bm25_sentence", "bm25_legal_recursive"}:
        return SentenceRetriever(method=method, index=BM25SentenceIndex(chunks))
    if method in {"dense_sentence", "dense_legal_recursive"}:
        return SentenceRetriever(
            method=method,
            index=DenseSentenceIndex(chunks, embedding_model=embedding_model),
        )
    raise ValueError(f"Unsupported sentence retriever: {method}")


def build_hierarchical_retriever(
    method: str,
    index: BM25SentenceIndex | DenseSentenceIndex,
    section_index: dict[int, list[SectionNode]],
    *,
    leaf_k: int = 50,
    top_sections: int = 5,
) -> HierarchicalRetriever:
    if method not in {"bm25_hierarchical", "dense_hierarchical"}:
        raise ValueError(f"Unsupported hierarchical retriever: {method}")
    return HierarchicalRetriever(
        method=method,  # type: ignore[arg-type]
        index=index,
        section_index=section_index,
        leaf_k=leaf_k,
        top_sections=top_sections,
    )
