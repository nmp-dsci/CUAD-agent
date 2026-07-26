"""Question enrichment and retrieval diagnostics for sentence RAG."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cuad_agent.data.sampling import evaluation_row_id
from cuad_agent.rag.cache import slugify
from cuad_agent.rag.coverage import (
    coverage_by_top_chunks,
    retrieved_sentence_ids_from_results,
)
from cuad_agent.rag.gold_answers import EligibilityRecord
from cuad_agent.rag.indexes import SearchResult
from cuad_agent.rag.retrievers import SentenceRetriever
from cuad_agent.rag.sentences import normalize_sentence_text


ProgressLogger = Callable[[str], None]

RAG_DEFAULT_TOP_K: int = 30

LEGAL_KEYWORD_FALLBACKS: dict[str, list[str]] = {
    "non-compete": [
        "non-compete",
        "noncompetition",
        "covenant not to compete",
        "competing business",
        "competitive activities",
        "directly or indirectly compete",
        "restricted business",
        "restricted territory",
        "same or similar business",
        "engage in business",
    ],
    "anti-assignment": [
        "assign",
        "assignment",
        "transfer",
        "delegate",
        "written consent",
        "prior consent",
        "without consent",
        "successors and assigns",
    ],
    "governing law": ["governed by", "laws of", "jurisdiction", "venue"],
    "audit rights": [
        "audit",
        "inspect",
        "books",
        "records",
        "accountant",
        "compliance",
    ],
    "exclusivity": ["exclusive", "sole", "only", "not appoint", "not engage"],
    "license": ["license", "licensed", "non-transferable", "sublicense", "scope"],
}


def query_for_row(row: Any) -> str:
    return " ".join(
        str(value).strip()
        for value in (row.category, row.category_description, row.question)
        if str(value).strip()
    )


def hybrid_fuse_results(
    dense_results: list[SearchResult],
    bm25_results: list[SearchResult],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """Fuse dense and BM25 results with reciprocal-rank fusion."""
    fused_scores: dict[str, float] = {}
    chunks: dict[str, Any] = {}
    for results in (dense_results, bm25_results):
        for result in results:
            chunk_id = result.chunk.chunk_id
            chunks[chunk_id] = result.chunk
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + result.rank)
            )
    ranked = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
    return [
        SearchResult(chunk=chunks[chunk_id], score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(ranked, start=1)
    ]


@dataclass(frozen=True)
class QuestionEnrichment:
    question_index: int
    category: str
    question: str
    category_description: str
    enrichment_terms: str
    enriched_query: str
    provider: str
    status: str
    cache_key: str

    def to_dict(self) -> dict[str, object]:
        return {
            "question_index": self.question_index,
            "category": self.category,
            "question": self.question,
            "category_description": self.category_description,
            "enrichment_terms": self.enrichment_terms,
            "enriched_query": self.enriched_query,
            "provider": self.provider,
            "status": self.status,
            "cache_key": self.cache_key,
        }


def cache_key_for_question(
    *,
    question_index: int,
    category: str,
    question: str,
    category_description: str,
) -> str:
    payload = json.dumps(
        {
            "question_index": question_index,
            "category": category,
            "question": question,
            "category_description": category_description,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enrichment_cache_path(output_dir: Path, provider: str) -> Path:
    return (
        output_dir
        / "rag_cache"
        / "query_enrichment"
        / slugify(provider)
        / "enriched_questions.jsonl"
    )


def load_enrichment_cache(path: Path) -> dict[str, QuestionEnrichment]:
    if not path.exists():
        return {}
    cache: dict[str, QuestionEnrichment] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            enrichment = QuestionEnrichment(
                question_index=int(row["question_index"]),
                category=str(row["category"]),
                question=str(row["question"]),
                category_description=str(row.get("category_description", "")),
                enrichment_terms=str(row.get("enrichment_terms", "")),
                enriched_query=str(row.get("enriched_query", "")),
                provider=str(row.get("provider", "")),
                status=str(row.get("status", "cached")),
                cache_key=str(row["cache_key"]),
            )
            cache[enrichment.cache_key] = enrichment
    return cache


def write_enrichment_cache(path: Path, values: list[QuestionEnrichment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    deduped = {value.cache_key: value for value in values}
    with tmp_path.open("w", encoding="utf-8") as handle:
        for value in sorted(deduped.values(), key=lambda item: item.question_index):
            handle.write(json.dumps(value.to_dict(), ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def offline_enrichment_terms(category: str, question: str, description: str) -> str:
    text = " ".join([category, question, description]).lower()
    terms: list[str] = []
    for key, values in LEGAL_KEYWORD_FALLBACKS.items():
        if key in text:
            terms.extend(values)
    if not terms:
        tokens = [
            token
            for token in normalize_sentence_text(
                " ".join([category, description])
            ).split()
            if len(token) > 4
        ]
        terms.extend(tokens[:12])
    return "; ".join(dict.fromkeys(terms))


def build_enriched_query(question: str, enrichment_terms: str) -> str:
    """Build the retrieval query from the raw question plus expansion terms."""
    return " ".join(
        part
        for part in [
            str(question).strip(),
            "Contract words to look for:",
            str(enrichment_terms).strip(),
        ]
        if part
    )


def deepseek_enrichment_terms(
    *,
    category: str,
    question: str,
    category_description: str,
    model: str,
    timeout: int = 60,
) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    prompt = (
        "What are words to look out for in a contract related to this legal "
        "clause / question? Return only concise comma-separated words and "
        "phrases. Do not explain.\n\n"
        f'Legal clause: "{category}"\n'
        f"Question: {question}\n"
        f"Details: {category_description}"
    )
    response = requests.post(
        os.environ.get(
            "DEEPSEEK_API_BASE", "https://api.deepseek.com/chat/completions"
        ),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You identify contract review search terms.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 300,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()


def build_question_enrichments(
    *,
    eval_rows: Any,
    output_dir: Path,
    provider: str,
    model: str,
    progress: ProgressLogger | None = None,
) -> tuple[dict[int, QuestionEnrichment], dict[str, Any]]:
    cache_path = enrichment_cache_path(output_dir, provider)
    cached = load_enrichment_cache(cache_path)
    values = list(cached.values())
    by_question: dict[int, QuestionEnrichment] = {}
    unique_rows = (
        eval_rows.sort_values(["question_index", "document_row_id"])
        .drop_duplicates("question_index")
        .itertuples(index=False)
    )
    llm_enabled = provider == "llm" or (
        provider == "auto" and bool(os.environ.get("DEEPSEEK_API_KEY"))
    )
    status_counts: dict[str, int] = {}
    for row in unique_rows:
        cache_key = cache_key_for_question(
            question_index=int(row.question_index),
            category=str(row.category),
            question=str(row.question),
            category_description=str(row.category_description),
        )
        enrichment = cached.get(cache_key)
        if enrichment is not None:
            refreshed_query = build_enriched_query(
                str(row.question),
                enrichment.enrichment_terms,
            )
            if enrichment.enriched_query != refreshed_query:
                enrichment = QuestionEnrichment(
                    question_index=enrichment.question_index,
                    category=enrichment.category,
                    question=enrichment.question,
                    category_description=enrichment.category_description,
                    enrichment_terms=enrichment.enrichment_terms,
                    enriched_query=refreshed_query,
                    provider=enrichment.provider,
                    status="cache_query_refreshed",
                    cache_key=enrichment.cache_key,
                )
                values.append(enrichment)
            by_question[int(row.question_index)] = enrichment
            status_counts["cache_hit"] = status_counts.get("cache_hit", 0) + 1
            continue
        status = "offline_fallback"
        provider_used = "offline"
        try:
            if llm_enabled:
                terms = deepseek_enrichment_terms(
                    category=str(row.category),
                    question=str(row.question),
                    category_description=str(row.category_description),
                    model=model,
                )
                status = "llm_generated"
                provider_used = "llm"
            else:
                terms = offline_enrichment_terms(
                    str(row.category),
                    str(row.question),
                    str(row.category_description),
                )
        except Exception as exc:
            terms = offline_enrichment_terms(
                str(row.category),
                str(row.question),
                str(row.category_description),
            )
            status = f"llm_failed_offline_fallback:{type(exc).__name__}"
            provider_used = "offline"
        query = build_enriched_query(str(row.question), terms)
        enrichment = QuestionEnrichment(
            question_index=int(row.question_index),
            category=str(row.category),
            question=str(row.question),
            category_description=str(row.category_description),
            enrichment_terms=terms,
            enriched_query=query,
            provider=provider_used,
            status=status,
            cache_key=cache_key,
        )
        values.append(enrichment)
        by_question[int(row.question_index)] = enrichment
        status_counts[status] = status_counts.get(status, 0) + 1
        if progress is not None:
            progress(
                f"Prepared query enrichment: question_index={row.question_index}, status={status}"
            )
    write_enrichment_cache(cache_path, values)
    return by_question, {
        "query_enrichment_provider": provider,
        "query_enrichment_model": model,
        "query_enrichment_cache_path": str(cache_path),
        "query_enrichment_status_counts": status_counts,
    }


def save_enriched_question_files(
    enrichments: dict[int, QuestionEnrichment],
    output_dir: Path,
    *,
    provider: str,
) -> None:
    """Write one JSON file per question for human review and offline editing.

    These files are NOT the primary cache (the JSONL is). They exist so
    engineers can inspect and hand-edit enrichment terms between runs.
    """
    base = output_dir / "enriched_questions" / slugify(provider)
    base.mkdir(parents=True, exist_ok=True)
    for question_index, enrichment in sorted(enrichments.items()):
        slug = slugify(enrichment.category)
        path = base / f"q{question_index:02d}_{slug}.json"
        path.write_text(
            json.dumps(enrichment.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def text_similarity(query: str, gold_texts: list[str]) -> dict[str, float]:
    normalized_gold = [normalize_sentence_text(value) for value in gold_texts if value]
    if not query.strip() or not normalized_gold:
        return {"mean": 0.0, "max": 0.0}
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([query, *normalized_gold])
    scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
    return {
        "mean": float(np.mean(scores)) if len(scores) else 0.0,
        "max": float(np.max(scores)) if len(scores) else 0.0,
    }


def run_query_enrichment_eval(
    *,
    eval_rows: Any,
    eligibility_by_row_id: dict[str, EligibilityRecord],
    dense_retriever: SentenceRetriever,
    bm25_retriever: SentenceRetriever | None = None,
    enrichments: dict[int, QuestionEnrichment],
    top_ks: tuple[int, ...] = (10, 20, 30),
    progress: ProgressLogger | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rows_by_id = {
        evaluation_row_id(int(row.document_row_id), int(row.question_index)): row
        for row in eval_rows.itertuples(index=False)
    }
    eligible_items = [
        (row_id, record)
        for row_id, record in eligibility_by_row_id.items()
        if record.is_eligible
    ]
    max_k = max(top_ks)
    for index, (row_id, eligibility) in enumerate(eligible_items, start=1):
        row = rows_by_id[row_id]
        baseline_query = query_for_row(row)
        enrichment = enrichments.get(int(row.question_index))
        enriched_query = enrichment.enriched_query if enrichment else baseline_query
        gold_texts = [sentence.text for sentence in eligibility.golden_sentences]
        baseline_similarity = text_similarity(baseline_query, gold_texts)
        enriched_similarity = text_similarity(enriched_query, gold_texts)
        baseline_results = dense_retriever.search(
            baseline_query,
            document_row_id=int(row.document_row_id),
            top_k=max_k,
        )
        enriched_results = dense_retriever.search(
            enriched_query,
            document_row_id=int(row.document_row_id),
            top_k=max_k,
        )
        baseline_hybrid_results: list[SearchResult] = []
        enriched_hybrid_results: list[SearchResult] = []
        if bm25_retriever is not None:
            baseline_bm25_results = bm25_retriever.search(
                baseline_query,
                document_row_id=int(row.document_row_id),
                top_k=max_k,
            )
            enriched_bm25_results = bm25_retriever.search(
                enriched_query,
                document_row_id=int(row.document_row_id),
                top_k=max_k,
            )
            baseline_hybrid_results = hybrid_fuse_results(
                baseline_results,
                baseline_bm25_results,
                top_k=max_k,
            )
            enriched_hybrid_results = hybrid_fuse_results(
                enriched_results,
                enriched_bm25_results,
                top_k=max_k,
            )
        baseline_coverage = {
            k: coverage_by_top_chunks(
                eligibility.matched_sentence_ids,
                baseline_results,
                top_chunks=k,
            )
            for k in top_ks
        }
        enriched_coverage = {
            k: coverage_by_top_chunks(
                eligibility.matched_sentence_ids,
                enriched_results,
                top_chunks=k,
            )
            for k in top_ks
        }
        baseline_hybrid_coverage = {
            k: coverage_by_top_chunks(
                eligibility.matched_sentence_ids,
                baseline_hybrid_results,
                top_chunks=k,
            )
            for k in top_ks
        }
        enriched_hybrid_coverage = {
            k: coverage_by_top_chunks(
                eligibility.matched_sentence_ids,
                enriched_hybrid_results,
                top_chunks=k,
            )
            for k in top_ks
        }
        output: dict[str, Any] = {
            "row_id": row_id,
            "document_row_id": int(row.document_row_id),
            "question_index": int(row.question_index),
            "category": str(row.category),
            "gold_sentence_count": eligibility.gold_sentence_count,
            "baseline_question_gold_similarity_mean": baseline_similarity["mean"],
            "baseline_question_gold_similarity_max": baseline_similarity["max"],
            "enriched_question_gold_similarity_mean": enriched_similarity["mean"],
            "enriched_question_gold_similarity_max": enriched_similarity["max"],
            "similarity_mean_delta": enriched_similarity["mean"]
            - baseline_similarity["mean"],
            "similarity_max_delta": enriched_similarity["max"]
            - baseline_similarity["max"],
            "enrichment_terms": enrichment.enrichment_terms if enrichment else "",
            "enrichment_status": enrichment.status if enrichment else "missing",
            "baseline_query": baseline_query,
            "enriched_query": enriched_query,
            "gold_sentences": gold_texts,
            "baseline_retrieved_sentence_ids_top30": retrieved_sentence_ids_from_results(
                baseline_results,
                top_chunks=max_k,
            ),
            "enriched_retrieved_sentence_ids_top30": retrieved_sentence_ids_from_results(
                enriched_results,
                top_chunks=max_k,
            ),
            "baseline_hybrid_retrieved_sentence_ids_top30": (
                retrieved_sentence_ids_from_results(
                    baseline_hybrid_results,
                    top_chunks=max_k,
                )
                if baseline_hybrid_results
                else []
            ),
            "enriched_hybrid_retrieved_sentence_ids_top30": (
                retrieved_sentence_ids_from_results(
                    enriched_hybrid_results,
                    top_chunks=max_k,
                )
                if enriched_hybrid_results
                else []
            ),
        }
        for k in top_ks:
            output[f"baseline_gold_sentence_coverage_at_{k}"] = baseline_coverage[k][
                "gold_sentence_coverage"
            ]
            output[f"enriched_gold_sentence_coverage_at_{k}"] = enriched_coverage[k][
                "gold_sentence_coverage"
            ]
            output[f"coverage_delta_at_{k}"] = float(
                enriched_coverage[k]["gold_sentence_coverage"]
            ) - float(baseline_coverage[k]["gold_sentence_coverage"])
            output[f"baseline_hybrid_gold_sentence_coverage_at_{k}"] = (
                baseline_hybrid_coverage[k]["gold_sentence_coverage"]
            )
            output[f"enriched_hybrid_gold_sentence_coverage_at_{k}"] = (
                enriched_hybrid_coverage[k]["gold_sentence_coverage"]
            )
            output[f"hybrid_coverage_delta_at_{k}"] = float(
                enriched_hybrid_coverage[k]["gold_sentence_coverage"]
            ) - float(baseline_hybrid_coverage[k]["gold_sentence_coverage"])
        rows.append(output)
        if progress is not None and (
            index == 1 or index % 25 == 0 or index == len(eligible_items)
        ):
            progress(f"Evaluated query enrichment row {index}/{len(eligible_items)}")
    summary = summarize_query_enrichment(rows, top_ks=top_ks)
    return rows, summary


def summarize_query_enrichment(
    rows: list[dict[str, Any]],
    *,
    top_ks: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["question_index"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for question_index, question_rows in sorted(grouped.items()):
        first = question_rows[0]
        output: dict[str, Any] = {
            "question_index": question_index,
            "category": first["category"],
            "rows": len(question_rows),
            "baseline_similarity_mean": sum(
                float(row["baseline_question_gold_similarity_mean"])
                for row in question_rows
            )
            / len(question_rows),
            "enriched_similarity_mean": sum(
                float(row["enriched_question_gold_similarity_mean"])
                for row in question_rows
            )
            / len(question_rows),
            "similarity_mean_delta": sum(
                float(row["similarity_mean_delta"]) for row in question_rows
            )
            / len(question_rows),
        }
        for k in top_ks:
            baseline_key = f"baseline_gold_sentence_coverage_at_{k}"
            enriched_key = f"enriched_gold_sentence_coverage_at_{k}"
            delta_key = f"coverage_delta_at_{k}"
            baseline_hybrid_key = f"baseline_hybrid_gold_sentence_coverage_at_{k}"
            enriched_hybrid_key = f"enriched_hybrid_gold_sentence_coverage_at_{k}"
            hybrid_delta_key = f"hybrid_coverage_delta_at_{k}"
            output[baseline_key] = sum(
                float(row[baseline_key]) for row in question_rows
            ) / len(question_rows)
            output[enriched_key] = sum(
                float(row[enriched_key]) for row in question_rows
            ) / len(question_rows)
            output[delta_key] = sum(
                float(row[delta_key]) for row in question_rows
            ) / len(question_rows)
            output[baseline_hybrid_key] = sum(
                float(row.get(baseline_hybrid_key, 0.0)) for row in question_rows
            ) / len(question_rows)
            output[enriched_hybrid_key] = sum(
                float(row.get(enriched_hybrid_key, 0.0)) for row in question_rows
            ) / len(question_rows)
            output[hybrid_delta_key] = sum(
                float(row.get(hybrid_delta_key, 0.0)) for row in question_rows
            ) / len(question_rows)
        summary.append(output)
    return summary
