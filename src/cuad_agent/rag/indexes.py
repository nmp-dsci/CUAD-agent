"""Sentence indexes for BM25 and dense-style retrieval."""

from __future__ import annotations

import math
import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cuad_agent.rag.chunks import RagChunk
from cuad_agent.rag.sentences import normalize_sentence_text


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(normalize_sentence_text(value).lower())


@dataclass
class SearchResult:
    chunk: RagChunk
    score: float
    rank: int

    def to_dict(self) -> dict[str, object]:
        data = self.chunk.to_dict()
        data.update({"score": self.score, "rank": self.rank})
        return data


class BM25SentenceIndex:
    def __init__(self, chunks: list[RagChunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.normalized_text) for chunk in chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freqs: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freqs[token] += 1

    def score(self, query: str, index: int) -> float:
        tokens = tokenize(query)
        if not tokens or not self.chunks:
            return 0.0
        score = 0.0
        doc_len = self.doc_lengths[index] or 1
        term_freq = self.term_freqs[index]
        total_docs = len(self.chunks)
        for token in tokens:
            freq = term_freq.get(token, 0)
            if freq == 0:
                continue
            doc_freq = self.doc_freqs.get(token, 0)
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
            score += idf * (freq * (self.k1 + 1)) / denominator
        return float(score)

    def search(self, query: str, *, document_row_id: int, top_k: int) -> list[SearchResult]:
        scored = [
            (self.score(query, index), chunk)
            for index, chunk in enumerate(self.chunks)
            if chunk.document_row_id == int(document_row_id)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


class DenseSentenceIndex:
    def __init__(self, chunks: list[RagChunk], *, embedding_model: str = "tfidf"):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.backend = "tfidf"
        self.model: Any = None
        self.vectorizer: TfidfVectorizer | None = None
        self.embeddings: Any = None
        texts = [chunk.normalized_text for chunk in chunks]

        if embedding_model not in {"tfidf", "local-tfidf"}:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

                self.model = SentenceTransformer(embedding_model)
                self.embeddings = np.asarray(
                    self.model.encode(texts, normalize_embeddings=True)
                )
                self.backend = "sentence_transformers"
                return
            except Exception:
                self.model = None

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.embeddings = self.vectorizer.fit_transform(texts)

    def write_encoded_artifacts(self, cache_dir: Path) -> dict[str, object]:
        cache_dir.mkdir(parents=True, exist_ok=True)
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        (cache_dir / "chunk_ids.json").write_text(
            json.dumps(chunk_ids, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        embedding_file: str
        if sparse.issparse(self.embeddings):
            embedding_file = "embeddings.npz"
            sparse.save_npz(cache_dir / embedding_file, self.embeddings)
        else:
            embedding_file = "embeddings.npy"
            np.save(cache_dir / embedding_file, np.asarray(self.embeddings))

        vectorizer_file = None
        if self.vectorizer is not None:
            vectorizer_file = "vectorizer.pkl"
            write_pickle(cache_dir / vectorizer_file, self.vectorizer)

        manifest = {
            "embedding_model": self.embedding_model,
            "backend": self.backend,
            "sentence_count": len(self.chunks),
            "chunk_count": len(self.chunks),
            "embedding_file": embedding_file,
            "vectorizer_file": vectorizer_file,
            "chunk_ids_file": "chunk_ids.json",
        }
        (cache_dir / "embedding_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest

    def search(self, query: str, *, document_row_id: int, top_k: int) -> list[SearchResult]:
        if not self.chunks:
            return []
        if self.backend == "sentence_transformers" and self.model is not None:
            query_vector = np.asarray(
                self.model.encode([query], normalize_embeddings=True)
            )
            scores = np.dot(self.embeddings, query_vector[0])
        else:
            assert self.vectorizer is not None
            query_vector = self.vectorizer.transform([query])
            scores = cosine_similarity(self.embeddings, query_vector).ravel()

        scored = [
            (float(scores[index]), chunk)
            for index, chunk in enumerate(self.chunks)
            if chunk.document_row_id == int(document_row_id)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


def load_pickle(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)
