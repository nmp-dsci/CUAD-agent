"""Hierarchical RAG: leaf search, section expansion, and candidate re-ranking."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from cuad_agent.rag.chunks import RagChunk
from cuad_agent.rag.indexes import BM25SentenceIndex, DenseSentenceIndex, SearchResult
from cuad_agent.rag.sentences import SentenceSpan


HIERARCHICAL_RETRIEVERS = {"bm25_hierarchical", "dense_hierarchical"}


@dataclass(frozen=True)
class SectionNode:
    section_key: tuple[str, ...]
    document_row_id: int
    section_number: str | None
    section_title: str | None
    sentence_ids: list[str]


def section_key_for(item: SentenceSpan | RagChunk) -> tuple[str, ...]:
    clause_path = item.clause_path
    if clause_path:
        return tuple(clause_path)
    return ("_unsectioned",)


def build_section_index(
    sentence_spans: list[SentenceSpan],
) -> dict[int, list[SectionNode]]:
    """Group sentence spans into ordered section nodes by document."""
    grouped: dict[int, OrderedDict[tuple[str, ...], list[SentenceSpan]]] = {}
    for span in sentence_spans:
        document_sections = grouped.setdefault(span.document_row_id, OrderedDict())
        document_sections.setdefault(section_key_for(span), []).append(span)

    section_index: dict[int, list[SectionNode]] = {}
    for document_row_id, sections in grouped.items():
        nodes: list[SectionNode] = []
        for section_key, spans in sections.items():
            first = spans[0]
            nodes.append(
                SectionNode(
                    section_key=section_key,
                    document_row_id=document_row_id,
                    section_number=first.section_number,
                    section_title=first.section_title,
                    sentence_ids=[span.sentence_id for span in spans],
                )
            )
        section_index[document_row_id] = nodes
    return section_index


class HierarchicalRetriever:
    """Wrap an existing sentence-level index with section expansion."""

    def __init__(
        self,
        *,
        method: Literal["bm25_hierarchical", "dense_hierarchical"],
        index: BM25SentenceIndex | DenseSentenceIndex,
        section_index: dict[int, list[SectionNode]],
        leaf_k: int = 50,
        top_sections: int = 5,
    ) -> None:
        self.method = method
        self.index = index
        self.section_index = section_index
        self.leaf_k = leaf_k
        self.top_sections = top_sections
        self._chunk_positions = {
            chunk.chunk_id: idx for idx, chunk in enumerate(index.chunks)
        }

    def search(
        self,
        query: str,
        *,
        document_row_id: int,
        top_k: int,
    ) -> list[SearchResult]:
        if document_row_id not in self.section_index:
            return []

        leaf_results = self.index.search(
            query,
            document_row_id=document_row_id,
            top_k=self.leaf_k,
        )
        if not leaf_results:
            return []

        section_scores: dict[tuple[str, ...], float] = {}
        for result in leaf_results:
            key = section_key_for(result.chunk)
            section_scores[key] = section_scores.get(key, 0.0) + result.score

        top_keys = set(
            sorted(section_scores, key=section_scores.__getitem__, reverse=True)[
                : self.top_sections
            ]
        )
        if not top_keys:
            return []

        expanded_ids: set[str] = set()
        for node in self.section_index.get(document_row_id, []):
            if node.section_key in top_keys:
                expanded_ids.update(node.sentence_ids)
        if not expanded_ids:
            return []

        candidate_chunks = [
            chunk
            for chunk in self.index.chunks
            if chunk.document_row_id == int(document_row_id)
            and any(sentence_id in expanded_ids for sentence_id in chunk.sentence_ids)
        ]
        scored = self._score_candidates(query, candidate_chunks)
        ranked = sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]
        return [
            SearchResult(chunk=result.chunk, score=result.score, rank=rank)
            for rank, result in enumerate(ranked, start=1)
        ]

    def _score_candidates(
        self,
        query: str,
        candidate_chunks: list[RagChunk],
    ) -> list[SearchResult]:
        if isinstance(self.index, BM25SentenceIndex):
            return [
                SearchResult(
                    chunk=chunk,
                    score=self.index.score(
                        query, self._chunk_positions[chunk.chunk_id]
                    ),
                    rank=0,
                )
                for chunk in candidate_chunks
            ]

        if not candidate_chunks:
            return []

        candidate_indices = [
            self._chunk_positions[chunk.chunk_id] for chunk in candidate_chunks
        ]
        if (
            self.index.backend == "sentence_transformers"
            and self.index.model is not None
        ):
            query_vector = np.asarray(
                self.index.model.encode([query], normalize_embeddings=True)
            )
            embeddings = np.asarray(self.index.embeddings)
            scores = np.dot(embeddings[candidate_indices], query_vector[0])
        else:
            assert self.index.vectorizer is not None
            query_vector = self.index.vectorizer.transform([query])
            embeddings = self.index.embeddings
            if sparse.issparse(embeddings):
                selected = embeddings[candidate_indices]
            else:
                selected = np.asarray(embeddings)[candidate_indices]
            scores = cosine_similarity(selected, query_vector).ravel()

        return [
            SearchResult(chunk=chunk, score=float(score), rank=0)
            for chunk, score in zip(candidate_chunks, scores)
        ]


def _section_header(chunk: RagChunk) -> str:
    if chunk.section_number and chunk.section_title:
        return f"SECTION {chunk.section_number} - {chunk.section_title}"
    if chunk.section_number:
        return f"SECTION {chunk.section_number}"
    if chunk.section_title:
        return f"SECTION {chunk.section_title}"
    if chunk.clause_path:
        return "SECTION " + " - ".join(chunk.clause_path)
    return "SECTION _unsectioned"


def format_hierarchical_context(results: list[SearchResult]) -> str:
    """Group retrieved chunks by section with one header per section."""
    sections: OrderedDict[tuple[str, ...], list[RagChunk]] = OrderedDict()
    for result in results:
        sections.setdefault(section_key_for(result.chunk), []).append(result.chunk)

    blocks: list[str] = []
    for chunks in sections.values():
        header = _section_header(chunks[0])
        body = "\n".join(chunk.text for chunk in chunks)
        blocks.append(f"[{header}]\n{body}")
    return "\n\n".join(blocks)
