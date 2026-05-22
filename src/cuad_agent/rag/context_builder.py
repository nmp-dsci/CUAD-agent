"""Build RAG context strings for single-question evaluation.

Single responsibility: turn (document_row_id, query, method) into a context
string the LLM receives in place of the full contract transcript.

Requires the sentence/chunk cache built by rag_eval.py. Run the preflight step
before using RAG context modes:

    uv run python rag_eval.py --preflight-golden-sentences-only \\
        --contract-scope all --run-id s6-preflight
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from cuad_agent.rag.cache import (
    index_cache_path,
    load_all_cached_sentence_spans,
    load_or_build_dense_sentence_encoder,
    load_or_build_retriever,
    sentence_cache_paths,
)
from cuad_agent.rag.chunks import chunks_from_sentences
from cuad_agent.rag.hierarchy import (
    HierarchicalRetriever,
    build_section_index,
    format_hierarchical_context,
)
from cuad_agent.rag.indexes import load_pickle
from cuad_agent.rag.query_enrichment import hybrid_fuse_results
from cuad_agent.rag.retrievers import SentenceRetriever


_HIERARCHICAL_CONTEXT_CACHE: dict[
    tuple[Path, str, str, str, int, int],
    HierarchicalRetriever,
] = {}
_SENTENCE_RETRIEVER_CACHE: dict[
    tuple[Path, str, str, str],
    SentenceRetriever,
] = {}


def _require_spans(output_dir: Path, chunking_version: str):  # type: ignore[return]
    """Load all sentence spans from cache or raise with a clear remediation message."""
    spans = load_all_cached_sentence_spans(
        output_dir=output_dir,
        chunking_version=chunking_version,
    )
    if spans is None:
        cache_path = sentence_cache_paths(output_dir, chunking_version)["sentences"]
        raise FileNotFoundError(
            f"Sentence span cache not found: {cache_path}\n"
            "Run the preflight step first:\n"
            "  uv run python rag_eval.py --preflight-golden-sentences-only "
            "--contract-scope all --run-id s6-preflight"
        )
    return spans


def build_rag_context(
    *,
    document_row_id: int,
    query: str,
    method: Literal["rag-dense", "rag-hybrid"],
    top_k: int,
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
) -> tuple[str, list[str]]:
    """Return (context_text, retrieved_chunk_ids).

    Loads retrievers from the on-disk pickle cache when available. On a cache
    miss the function rebuilds from the sentence span cache (also on disk).
    If neither exists, raises FileNotFoundError with the preflight command.

    For rag-hybrid: runs both dense and BM25 retrievers, fuses with RRF, and
    returns top_k fused results.

    context_text is chunks joined by "\\n\\n---\\n\\n".
    """
    # --- Dense retriever (required by both rag-dense and rag-hybrid) ---
    dense_key = (output_dir.resolve(), "dense_sentence", chunking_version, embedding_model)
    dense_retriever = _SENTENCE_RETRIEVER_CACHE.get(dense_key)

    if dense_retriever is None:
        dense_pickle = index_cache_path(output_dir, "dense_sentence", chunking_version, embedding_model)
        cached = load_pickle(dense_pickle)
        if isinstance(cached, SentenceRetriever):
            dense_retriever = cached
        else:
            spans = _require_spans(output_dir, chunking_version)
            chunks = chunks_from_sentences(spans)
            dense_retriever, _ = load_or_build_dense_sentence_encoder(
                chunks=chunks,
                method="dense_sentence",
                output_dir=output_dir,
                chunking_version=chunking_version,
                embedding_model=embedding_model,
                rebuild=False,
            )
        _SENTENCE_RETRIEVER_CACHE[dense_key] = dense_retriever

    if not isinstance(dense_retriever, SentenceRetriever):
        spans = _require_spans(output_dir, chunking_version)
        chunks = chunks_from_sentences(spans)
        dense_retriever, _ = load_or_build_dense_sentence_encoder(
            chunks=chunks,
            method="dense_sentence",
            output_dir=output_dir,
            chunking_version=chunking_version,
            embedding_model=embedding_model,
            rebuild=False,
        )

    dense_results = dense_retriever.search(query, document_row_id=document_row_id, top_k=top_k)

    if method == "rag-dense":
        results = dense_results
    else:
        # --- BM25 retriever (hybrid only) ---
        bm25_key = (output_dir.resolve(), "bm25_sentence", chunking_version, embedding_model)
        bm25_retriever = _SENTENCE_RETRIEVER_CACHE.get(bm25_key)

        if bm25_retriever is None:
            bm25_pickle = index_cache_path(
                output_dir, "bm25_sentence", chunking_version, embedding_model
            )
            cached = load_pickle(bm25_pickle)
            if isinstance(cached, SentenceRetriever):
                bm25_retriever = cached
            else:
                spans = _require_spans(output_dir, chunking_version)
                chunks = chunks_from_sentences(spans)
                bm25_retriever, _ = load_or_build_retriever(
                    method="bm25_sentence",
                    chunks=chunks,
                    output_dir=output_dir,
                    chunking_version=chunking_version,
                    embedding_model=embedding_model,
                    rebuild=False,
                )
            _SENTENCE_RETRIEVER_CACHE[bm25_key] = bm25_retriever

        if not isinstance(bm25_retriever, SentenceRetriever):
            spans = _require_spans(output_dir, chunking_version)
            chunks = chunks_from_sentences(spans)
            bm25_retriever, _ = load_or_build_retriever(
                method="bm25_sentence",
                chunks=chunks,
                output_dir=output_dir,
                chunking_version=chunking_version,
                embedding_model=embedding_model,
                rebuild=False,
            )

        bm25_results = bm25_retriever.search(query, document_row_id=document_row_id, top_k=top_k)
        results = hybrid_fuse_results(dense_results, bm25_results, top_k=top_k)

    chunk_ids = [result.chunk.chunk_id for result in results]
    context_text = "\n\n---\n\n".join(result.chunk.text for result in results)
    return context_text, chunk_ids


def build_hierarchical_rag_context(
    *,
    document_row_id: int,
    query: str,
    method: Literal["rag-hierarchical-bm25", "rag-hierarchical-dense"],
    leaf_k: int = 50,
    top_sections: int = 5,
    top_k: int,
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
) -> tuple[str, list[str]]:
    """Return section-grouped hierarchical RAG context and retrieved chunk ids."""
    cache_key = (
        output_dir.resolve(),
        chunking_version,
        embedding_model,
        method,
        leaf_k,
        top_sections,
    )
    h_retriever = _HIERARCHICAL_CONTEXT_CACHE.get(cache_key)
    if h_retriever is None:
        spans = _require_spans(output_dir, chunking_version)
        chunks = chunks_from_sentences(spans)

        if method == "rag-hierarchical-bm25":
            retriever, _ = load_or_build_retriever(
                method="bm25_sentence",
                chunks=chunks,
                output_dir=output_dir,
                chunking_version=chunking_version,
                embedding_model=embedding_model,
                rebuild=False,
            )
            hierarchical_method = "bm25_hierarchical"
        else:
            retriever, _ = load_or_build_dense_sentence_encoder(
                chunks=chunks,
                method="dense_sentence",
                output_dir=output_dir,
                chunking_version=chunking_version,
                embedding_model=embedding_model,
                rebuild=False,
            )
            hierarchical_method = "dense_hierarchical"

        h_retriever = HierarchicalRetriever(
            method=hierarchical_method,
            index=retriever.index,
            section_index=build_section_index(spans),
            leaf_k=leaf_k,
            top_sections=top_sections,
        )
        _HIERARCHICAL_CONTEXT_CACHE[cache_key] = h_retriever

    results = h_retriever.search(query, document_row_id=document_row_id, top_k=top_k)
    return format_hierarchical_context(results), [result.chunk.chunk_id for result in results]
