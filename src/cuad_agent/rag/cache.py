"""Cache I/O helpers for RAG sentence/chunk indexes and embeddings.

Shared by experiments.py (bulk orchestration) and context_builder.py (single-question
retrieval). Keeps lower-level cache helpers out of the orchestration layer so that
context_builder can import them without creating a circular dependency.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Callable

from cuad_agent.rag.chunks import RagChunk
from cuad_agent.rag.indexes import DenseSentenceIndex, load_pickle, write_pickle
from cuad_agent.rag.outputs import read_jsonl
from cuad_agent.rag.retrievers import SentenceRetriever, build_retriever
from cuad_agent.rag.sentences import SentenceSpan


ProgressLogger = Callable[[str], None]

DEFAULT_EMBEDDING_MODEL: str = "tfidf"


def emit_progress(progress: ProgressLogger | None, message: str) -> None:
    if progress is not None:
        progress(message)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value).strip().lower())
    return re.sub(r"-+", "-", cleaned).strip("-._") or "value"


def sentence_cache_paths(output_dir: Path, chunking_version: str) -> dict[str, Path]:
    cache_dir = output_dir / "rag_cache" / "chunking" / chunking_version
    return {
        "dir": cache_dir,
        "manifest": cache_dir / "contracts_manifest.json",
        "sentences": cache_dir / "sentence_spans.jsonl",
        "config": cache_dir / "chunking_config.json",
    }


def load_cached_sentence_spans_for_version(
    *,
    output_dir: Path,
    chunking_version: str,
    document_ids: set[int],
) -> list[SentenceSpan] | None:
    paths = sentence_cache_paths(output_dir, chunking_version)
    if not paths["sentences"].exists():
        return None
    try:
        spans = [SentenceSpan(**row) for row in read_jsonl(paths["sentences"])]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return [span for span in spans if int(span.document_row_id) in document_ids]


def load_all_cached_sentence_spans(
    *,
    output_dir: Path,
    chunking_version: str,
) -> list[SentenceSpan] | None:
    """Load all sentence spans from cache (no document_ids filter)."""
    paths = sentence_cache_paths(output_dir, chunking_version)
    if not paths["sentences"].exists():
        return None
    try:
        return [SentenceSpan(**row) for row in read_jsonl(paths["sentences"])]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def dense_encoding_cache_dir(
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
) -> Path:
    return (
        output_dir
        / "rag_cache"
        / "chunking"
        / chunking_version
        / "encodings"
        / slugify(embedding_model)
    )


def index_cache_path(
    output_dir: Path,
    method: str,
    chunking_version: str,
    embedding_model: str,
) -> Path:
    if method.startswith("bm25_"):
        return output_dir / "rag_cache" / "sparse" / chunking_version / method / "bm25_index.pkl"
    return dense_encoding_cache_dir(output_dir, chunking_version, embedding_model) / "dense_index.pkl"


def load_or_build_retriever(
    *,
    method: str,
    chunks: list[RagChunk],
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
    rebuild: bool,
    progress: ProgressLogger | None = None,
) -> tuple[SentenceRetriever, bool]:
    path = index_cache_path(output_dir, method, chunking_version, embedding_model)
    if not rebuild:
        cached = load_pickle(path)
        if isinstance(cached, SentenceRetriever):
            emit_progress(progress, f"Loaded retriever cache: method={method}")
            return cached, True
    emit_progress(progress, f"Building retriever: method={method}, chunks={len(chunks)}")
    retriever = build_retriever(method, chunks, embedding_model=embedding_model)
    try:
        write_pickle(path, retriever)
    except (pickle.PickleError, TypeError, AttributeError):
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "embedding_manifest.json").write_text(
            json.dumps({"pickle_skipped": True, "method": method}, indent=2),
            encoding="utf-8",
        )
    emit_progress(progress, f"Built retriever: method={method}")
    return retriever, False


def load_or_build_dense_sentence_encoder(
    *,
    chunks: list[RagChunk],
    method: str = "dense_sentence",
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
    rebuild: bool,
    progress: ProgressLogger | None = None,
) -> tuple[SentenceRetriever, dict[str, Any]]:
    cache_dir = dense_encoding_cache_dir(output_dir, chunking_version, embedding_model)
    dense_index_path = cache_dir / "dense_index.pkl"
    manifest_path = cache_dir / "embedding_manifest.json"
    if not rebuild:
        cached = load_pickle(dense_index_path)
        if isinstance(cached, SentenceRetriever) and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            emit_progress(
                progress,
                f"Loaded dense sentence encoding cache: sentences={manifest.get('sentence_count')}",
            )
            encoded_count = int(
                manifest.get("chunk_count", manifest.get("sentence_count", 0))
            )
            info: dict[str, Any] = {
                f"{method}_encoding_cache_hit": True,
                f"{method}_encoded_chunk_count": encoded_count,
                "embedding_backend": manifest.get("backend"),
                "embedding_cache_dir": str(cache_dir),
            }
            if method == "dense_sentence":
                info["encoded_sentence_count"] = encoded_count
            return cached, info

    emit_progress(
        progress,
        f"Encoding sentence chunks: sentences={len(chunks)}, embedding_model={embedding_model}",
    )
    dense_index = DenseSentenceIndex(chunks, embedding_model=embedding_model)
    dense_index.write_encoded_artifacts(cache_dir)
    retriever = SentenceRetriever(method=method, index=dense_index)
    try:
        write_pickle(dense_index_path, retriever)
    except (pickle.PickleError, TypeError, AttributeError):
        pass
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    emit_progress(progress, f"Wrote dense sentence encoding cache: {cache_dir}")
    encoded_count = int(manifest.get("chunk_count", manifest.get("sentence_count", len(chunks))))
    info = {
        f"{method}_encoding_cache_hit": False,
        f"{method}_encoded_chunk_count": encoded_count,
        "embedding_backend": manifest.get("backend"),
        "embedding_cache_dir": str(cache_dir),
    }
    if method == "dense_sentence":
        info["encoded_sentence_count"] = encoded_count
    return retriever, info
