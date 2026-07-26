"""Sentence-level RAG experiment orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cuad_agent.data.dataset import load_datasets
from cuad_agent.data.sampling import (
    all_contract_lookup,
    evaluation_row_id,
    select_evaluation_set,
)
from cuad_agent.rag.cache import (
    ProgressLogger,
    emit_progress,
    load_cached_sentence_spans_for_version,
    load_or_build_dense_sentence_encoder,
    load_or_build_retriever,
    sentence_cache_paths,
    slugify,
)
from cuad_agent.rag.chunks import RagChunk, chunk_from_dict, chunks_from_sentences
from cuad_agent.rag.clauses import build_section_metadata
from cuad_agent.rag.contracts import ContractDocument, contracts_from_lookup
from cuad_agent.rag.coverage import (
    coverage_by_top_chunks,
    retrieved_sentence_ids_from_results,
)
from cuad_agent.rag.gold_answers import (
    EligibilityRecord,
    answer_texts,
    evaluate_row_eligibility,
)
from cuad_agent.rag.hierarchy import (
    HIERARCHICAL_RETRIEVERS,
    HierarchicalRetriever,
    build_section_index,
)
from cuad_agent.rag.legal_recursive import (
    LEGAL_RECURSIVE_CHUNKING_VERSION,
    LegalRecursiveConfig,
    build_legal_recursive_chunks,
)
from cuad_agent.rag.outputs import (
    rag_output_paths,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
    write_pipeline_html,
)
from cuad_agent.rag.query_enrichment import (
    build_question_enrichments,
    query_for_row,
    run_query_enrichment_eval,
)
from cuad_agent.rag.retrievers import SentenceRetriever, build_hierarchical_retriever
from cuad_agent.rag.sentences import SentenceSpan, build_sentence_spans


DEFAULT_CHUNKING_VERSION = "sentence-v3"
DEFAULT_RETRIEVERS = ("bm25_sentence", "dense_sentence")
LEGAL_RECURSIVE_RETRIEVERS = {"bm25_legal_recursive", "dense_legal_recursive"}


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def chunk_cache_paths(output_dir: Path, chunking_version: str) -> dict[str, Path]:
    cache_dir = output_dir / "rag_cache" / "chunking" / chunking_version
    return {
        "dir": cache_dir,
        "manifest": cache_dir / "contracts_manifest.json",
        "chunks": cache_dir / "rag_chunks.jsonl",
        "config": cache_dir / "chunking_config.json",
    }


def sentence_manifest(
    contracts: dict[int, ContractDocument], chunking_version: str
) -> dict[str, Any]:
    return {
        "chunking_version": chunking_version,
        "sentence_splitter": "pysbd-if-available-with-deterministic-fallback",
        "adapter": "cuad-json-docling-like-v1",
        "contracts": {
            str(document_row_id): hash_text(contract.text)
            for document_row_id, contract in sorted(contracts.items())
        },
    }


def legal_recursive_manifest(
    contracts: dict[int, ContractDocument],
    chunking_version: str,
) -> dict[str, Any]:
    config = LegalRecursiveConfig()
    return {
        "chunking_version": chunking_version,
        "chunker": "langchain-recursive-character-text-splitter",
        "sentence_source": DEFAULT_CHUNKING_VERSION,
        "config": config.to_dict(),
        "contracts": {
            str(document_row_id): hash_text(contract.text)
            for document_row_id, contract in sorted(contracts.items())
        },
    }


def load_sentence_cache(
    paths: dict[str, Path], manifest: dict[str, Any]
) -> list[SentenceSpan] | None:
    if not paths["manifest"].exists() or not paths["sentences"].exists():
        return None
    try:
        cached_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached_manifest != manifest:
        return None
    try:
        return [SentenceSpan(**row) for row in read_jsonl(paths["sentences"])]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def sentence_lookup_from_spans(
    spans: list[SentenceSpan],
) -> dict[int, list[SentenceSpan]]:
    lookup: dict[int, list[SentenceSpan]] = {}
    for span in spans:
        lookup.setdefault(span.document_row_id, []).append(span)
    return lookup


def load_chunk_cache(
    paths: dict[str, Path], manifest: dict[str, Any]
) -> list[RagChunk] | None:
    if not paths["manifest"].exists() or not paths["chunks"].exists():
        return None
    try:
        cached_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached_manifest != manifest:
        return None
    try:
        return [chunk_from_dict(row) for row in read_jsonl(paths["chunks"])]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def build_sentence_store(
    contracts: dict[int, ContractDocument],
    *,
    progress: ProgressLogger | None = None,
) -> list[SentenceSpan]:
    spans: list[SentenceSpan] = []
    ordered_contracts = sorted(contracts.items())
    total = len(ordered_contracts)
    for contract_number, (document_row_id, contract) in enumerate(
        ordered_contracts, start=1
    ):
        emit_progress(
            progress,
            f"Chunking contract {contract_number}/{total}: "
            f"document_row_id={document_row_id}",
        )
        metadata = build_section_metadata(contract.text)
        contract_spans = build_sentence_spans(
            document_row_id,
            contract.text,
            section_metadata=metadata,
        )
        spans.extend(contract_spans)
        emit_progress(
            progress,
            f"Chunked contract {contract_number}/{total}: "
            f"document_row_id={document_row_id}, sentences={len(contract_spans)}",
        )
    emit_progress(progress, f"Chunked {total} contracts into {len(spans)} sentences")
    return spans


def load_or_build_sentence_cache(
    *,
    contracts: dict[int, ContractDocument],
    output_dir: Path,
    chunking_version: str,
    rebuild: bool,
    progress: ProgressLogger | None = None,
) -> tuple[list[SentenceSpan], dict[str, Any]]:
    paths = sentence_cache_paths(output_dir, chunking_version)
    manifest = sentence_manifest(contracts, chunking_version)
    cache_hit = False
    spans = None if rebuild else load_sentence_cache(paths, manifest)
    if spans is not None:
        cache_hit = True
        emit_progress(
            progress,
            f"Loaded sentence cache: contracts={len(contracts)}, sentences={len(spans)}",
        )
    else:
        emit_progress(
            progress,
            f"Building sentence cache: contracts={len(contracts)}, rebuild={rebuild}",
        )
        spans = build_sentence_store(contracts, progress=progress)
        write_jsonl(paths["sentences"], [span.to_dict() for span in spans])
        write_json(paths["manifest"], manifest)
        write_json(paths["config"], manifest)
        emit_progress(progress, f"Wrote sentence cache: {paths['sentences']}")
    return spans, {"sentence_cache_hit": cache_hit, "sentence_count": len(spans)}


def load_or_build_legal_recursive_cache(
    *,
    contracts: dict[int, ContractDocument],
    sentence_lookup: dict[int, list[SentenceSpan]],
    output_dir: Path,
    chunking_version: str,
    rebuild: bool,
    progress: ProgressLogger | None = None,
) -> tuple[list[RagChunk], dict[str, Any]]:
    paths = chunk_cache_paths(output_dir, chunking_version)
    manifest = legal_recursive_manifest(contracts, chunking_version)
    cache_hit = False
    chunks = None if rebuild else load_chunk_cache(paths, manifest)
    if chunks is not None:
        cache_hit = True
        emit_progress(
            progress,
            f"Loaded legal recursive chunk cache: contracts={len(contracts)}, chunks={len(chunks)}",
        )
    else:
        emit_progress(
            progress,
            "Building legal recursive chunk cache with LangChain "
            f"RecursiveCharacterTextSplitter: contracts={len(contracts)}, rebuild={rebuild}",
        )
        chunks = build_legal_recursive_chunks(
            text_by_document_id={
                document_row_id: contract.text
                for document_row_id, contract in contracts.items()
            },
            sentence_lookup=sentence_lookup,
            config=LegalRecursiveConfig(),
        )
        write_jsonl(paths["chunks"], [chunk.to_dict() for chunk in chunks])
        write_json(paths["manifest"], manifest)
        write_json(paths["config"], manifest)
        emit_progress(progress, f"Wrote legal recursive chunk cache: {paths['chunks']}")
    return chunks, {
        "legal_recursive_chunk_cache_hit": cache_hit,
        "legal_recursive_chunk_count": len(chunks),
        "legal_recursive_chunking_version": chunking_version,
    }


def build_legal_recursive_documents(
    legal_recursive_chunks: list[RagChunk],
    contracts: dict[int, ContractDocument],
    review_document_ids: set[int],
) -> dict[str, dict[str, Any]]:
    by_doc = chunks_by_document(legal_recursive_chunks)
    documents: dict[str, dict[str, Any]] = {}
    for document_row_id in sorted(review_document_ids):
        contract = contracts.get(
            document_row_id,
            ContractDocument(
                document_row_id=document_row_id, title=str(document_row_id), text=""
            ),
        )
        documents[str(document_row_id)] = {
            "document_row_id": document_row_id,
            "title": contract.title,
            "raw_text": contract.text,
            "sentences": [
                {
                    "sentence_id": chunk.chunk_id,
                    "sentence_index": index,
                    "raw_text": chunk.text,
                    "contained_sentence_ids": chunk.sentence_ids,
                    "section_number": chunk.section_number or "",
                    "section_title": chunk.section_title or "",
                    "clause_path": chunk.clause_path,
                }
                for index, chunk in enumerate(by_doc.get(document_row_id, []))
            ],
        }
    return documents


def chunks_by_document(chunks: list[RagChunk]) -> dict[int, list[RagChunk]]:
    grouped: dict[int, list[RagChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_row_id, []).append(chunk)
    return grouped


def eligibility_records(
    eval_rows: pd.DataFrame,
    sentence_lookup: dict[int, list[SentenceSpan]],
    *,
    contracts: dict[int, ContractDocument] | None = None,
    progress: ProgressLogger | None = None,
) -> list[EligibilityRecord]:
    records: list[EligibilityRecord] = []
    total = len(eval_rows)
    emit_progress(progress, f"Checking golden-answer sentence coverage: rows={total}")
    for row_number, row in enumerate(eval_rows.itertuples(index=False), start=1):
        row_id = evaluation_row_id(int(row.document_row_id), int(row.question_index))
        contract = (contracts or {}).get(int(row.document_row_id))
        records.append(
            evaluate_row_eligibility(
                row_id=row_id,
                document_row_id=int(row.document_row_id),
                question_index=int(row.question_index),
                category=str(row.category),
                answers=row.answers,
                is_impossible=bool(row.is_impossible),
                contract_sentences=sentence_lookup.get(int(row.document_row_id), []),
                contract_text=contract.text if contract else "",
            )
        )
        if row_number == total or row_number % 100 == 0:
            emit_progress(
                progress,
                f"Checked golden-answer coverage row {row_number}/{total}",
            )
    emit_progress(progress, f"Finished golden-answer coverage: rows={total}")
    return records


def all_contract_question_rows(
    question_indices: list[int] | None = None,
) -> pd.DataFrame:
    questions = load_datasets()["questions"].copy()
    if question_indices:
        questions = questions[questions["question_index"].isin(question_indices)].copy()
    return questions.sort_values(["document_row_id", "question_index"]).reset_index(
        drop=True
    )


def summarize_eligibility(records: list[EligibilityRecord]) -> dict[str, Any]:
    extraction_records = [
        record for record in records if record.gold_sentence_count > 0
    ]
    eligible = [record for record in records if record.is_eligible]
    return {
        "rows_total": len(records),
        "rows_with_gold_sentences": len(extraction_records),
        "eligible_rows": len(eligible),
        "non_eligible_rows": len(records) - len(eligible),
        "eligible_rate": len(eligible) / len(extraction_records)
        if extraction_records
        else 0.0,
        "gold_sentence_count": sum(record.gold_sentence_count for record in records),
        "matched_sentence_count": sum(
            record.matched_sentence_count for record in records
        ),
        "raw_contract_sentence_match_count": sum(
            record.raw_contract_sentence_match_count for record in records
        ),
        "raw_contract_sentence_match_rate": (
            sum(record.raw_contract_sentence_match_count for record in records)
            / sum(record.gold_sentence_count for record in extraction_records)
            if extraction_records
            else 0.0
        ),
    }


def summarize_eligibility_by_question(
    records: list[EligibilityRecord],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[EligibilityRecord]] = {}
    for record in records:
        grouped.setdefault(record.question_index, []).append(record)

    rows: list[dict[str, Any]] = []
    for question_index, question_records in sorted(grouped.items()):
        with_gold = [
            record for record in question_records if record.gold_sentence_count > 0
        ]
        eligible = [record for record in with_gold if record.is_eligible]
        total_gold_sentences = sum(record.gold_sentence_count for record in with_gold)
        total_matched_sentences = sum(
            record.matched_sentence_count for record in with_gold
        )
        total_raw_contract_matches = sum(
            record.raw_contract_sentence_match_count for record in with_gold
        )
        full_match_rate = len(eligible) / len(with_gold) if with_gold else 0.0
        split_sentence_match_rate = (
            total_matched_sentences / total_gold_sentences
            if total_gold_sentences
            else 0.0
        )
        raw_contract_sentence_match_rate = (
            total_raw_contract_matches / total_gold_sentences
            if total_gold_sentences
            else 0.0
        )
        rows.append(
            {
                "question_index": question_index,
                "category": question_records[0].category if question_records else "",
                "evaluated_contract_rows": len(question_records),
                "rows_with_split_golden_answer_sentences": len(with_gold),
                "full_match_rows": len(eligible),
                "full_match_rate": round(full_match_rate, 4),
                "split_golden_sentence_count": total_gold_sentences,
                "raw_contract_matched_split_golden_sentence_count": total_raw_contract_matches,
                "raw_contract_sentence_match_rate": round(
                    raw_contract_sentence_match_rate, 4
                ),
                "matched_split_golden_sentence_count": total_matched_sentences,
                "split_sentence_match_rate": round(split_sentence_match_rate, 4),
                "strict_sentence_extraction_question": (
                    bool(with_gold) and len(eligible) == len(with_gold)
                ),
                "rag_suitability": classify_question_rag_suitability(full_match_rate),
            }
        )
    return rows


def classify_question_rag_suitability(full_match_rate: float) -> str:
    if full_match_rate >= 1.0:
        return "strict_sentence_extraction"
    if full_match_rate >= 0.9:
        return "strong_sentence_rag_candidate"
    if full_match_rate >= 0.5:
        return "mixed_sentence_rag_candidate"
    return "not_sentence_rag_suitable"


def chunking_summary(
    *,
    contracts: dict[int, ContractDocument],
    sentence_count: int,
    contract_scope: str,
) -> dict[str, Any]:
    contract_count = len(contracts)
    return {
        "contract_scope": contract_scope,
        "chunk_all_contracts": contract_scope == "all",
        "chunked_contract_count": contract_count,
        "sentence_count": sentence_count,
        "average_sentences_per_contract": (
            sentence_count / contract_count if contract_count else 0.0
        ),
    }


def run_sentence_retrieval(
    *,
    eval_rows: pd.DataFrame,
    eligibility_by_row_id: dict[str, EligibilityRecord],
    chunks: list[RagChunk],
    chunks_by_method: dict[str, list[RagChunk]] | None = None,
    chunking_version_by_method: dict[str, str] | None = None,
    retriever_methods: list[str],
    top_k: int,
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
    rebuild_embeddings: bool,
    prebuilt_retrievers: dict[
        str, tuple[SentenceRetriever | HierarchicalRetriever, bool]
    ]
    | None = None,
    progress: ProgressLogger | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    rows_by_id = {
        evaluation_row_id(int(row.document_row_id), int(row.question_index)): row
        for row in eval_rows.itertuples(index=False)
    }
    results: list[dict[str, Any]] = []
    ranking_summary: list[dict[str, Any]] = []
    cache_hits: dict[str, bool] = {}
    eligible_count = sum(
        1 for eligibility in eligibility_by_row_id.values() if eligibility.is_eligible
    )
    prebuilt_retrievers = prebuilt_retrievers or {}
    chunks_by_method = chunks_by_method or {}
    chunking_version_by_method = chunking_version_by_method or {}
    retrieval_top_k = max(top_k, 30)
    retrieval_cutoffs = tuple(sorted({1, 3, 5, 10, 20, 30, top_k}))
    for method in retriever_methods:
        method_chunks = chunks_by_method.get(method, chunks)
        method_chunking_version = chunking_version_by_method.get(
            method, chunking_version
        )
        emit_progress(
            progress,
            f"Running retrieval method {method}: chunks={len(method_chunks)}, eligible rows={eligible_count}, top_k={retrieval_top_k}",
        )
        if method in prebuilt_retrievers:
            retriever, cache_hit = prebuilt_retrievers[method]
        else:
            retriever, cache_hit = load_or_build_retriever(
                method=method,
                chunks=method_chunks,
                output_dir=output_dir,
                chunking_version=method_chunking_version,
                embedding_model=embedding_model,
                rebuild=rebuild_embeddings,
                progress=progress,
            )
        cache_hits[f"{method}_cache_hit"] = cache_hit
        method_results: list[dict[str, Any]] = []
        processed = 0
        for row_id, eligibility in eligibility_by_row_id.items():
            if not eligibility.is_eligible:
                continue
            processed += 1
            if processed == 1 or processed % 25 == 0 or processed == eligible_count:
                emit_progress(
                    progress,
                    f"Retrieving {method} row {processed}/{eligible_count}: row_id={row_id}",
                )
            row = rows_by_id[row_id]
            query = query_for_row(row)
            retrieved = retriever.search(
                query,
                document_row_id=int(row.document_row_id),
                top_k=retrieval_top_k,
            )
            retrieved_sentence_ids = retrieved_sentence_ids_from_results(
                retrieved,
                top_chunks=retrieval_top_k,
            )
            coverage = {
                k: coverage_by_top_chunks(
                    eligibility.matched_sentence_ids,
                    retrieved,
                    top_chunks=k,
                )
                for k in retrieval_cutoffs
            }
            result_row = {
                "retriever": method,
                "chunking_version": method_chunking_version,
                "row_id": row_id,
                "document_row_id": int(row.document_row_id),
                "question_index": int(row.question_index),
                "category": str(row.category),
                "question": str(row.question),
                "answer_format": str(row.answer_format),
                "gold_sentence_ids": eligibility.matched_sentence_ids,
                "retrieved_sentence_ids": retrieved_sentence_ids,
                "retrieved_chunk_count": len(retrieved),
                "retrieved_top_n": [result.to_dict() for result in retrieved],
                "first_covering_rank": coverage[retrieval_top_k]["first_covering_rank"],
            }
            for cutoff in retrieval_cutoffs:
                result_row[f"covered_at_{cutoff}"] = coverage[cutoff][
                    "all_gold_sentences_covered"
                ]
                result_row[f"gold_sentence_coverage_at_{cutoff}"] = coverage[cutoff][
                    "gold_sentence_coverage"
                ]
            method_results.append(result_row)
            results.append(result_row)
        ranking_summary.append(
            summarize_retrieval_method(method, method_results, retrieval_top_k)
        )
        emit_progress(
            progress,
            f"Finished retrieval method {method}: rows={len(method_results)}",
        )
    return results, ranking_summary, cache_hits


def summarize_retrieval_method(
    method: str, rows: list[dict[str, Any]], top_k: int
) -> dict[str, Any]:
    cutoffs = tuple(sorted({1, 3, 5, 10, 20, 30, top_k}))
    if not rows:
        output: dict[str, Any] = {
            "retriever": method,
            "rows": 0,
        }
        for cutoff in cutoffs:
            output[f"gold_sentence_coverage_at_{cutoff}"] = 0.0
            output[f"all_gold_covered_rate_at_{cutoff}"] = 0.0
        return output
    output = {
        "retriever": method,
        "rows": len(rows),
    }
    for cutoff in cutoffs:
        key = f"gold_sentence_coverage_at_{cutoff}"
        output[key] = sum(float(row[key]) for row in rows) / len(rows)
        covered_key = f"covered_at_{cutoff}"
        output[f"all_gold_covered_rate_at_{cutoff}"] = sum(
            1.0 if row.get(covered_key) else 0.0 for row in rows
        ) / len(rows)
    return output


def summarize_retrieval_by_document_question(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["row_id"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    cutoffs = tuple(sorted({10, 20, 30, top_k}))
    for row_id, group in sorted(
        grouped.items(),
        key=lambda item: (
            int(item[1][0]["document_row_id"]),
            int(item[1][0]["question_index"]),
        ),
    ):
        first = group[0]
        output_row: dict[str, Any] = {
            "row_id": row_id,
            "document_row_id": int(first["document_row_id"]),
            "question_index": int(first["question_index"]),
            "category": first["category"],
            "question": first["question"],
            "gold_sentence_count": len(first.get("gold_sentence_ids", [])),
            "retriever_count": len(group),
        }
        for cutoff in cutoffs:
            coverage_key = f"gold_sentence_coverage_at_{cutoff}"
            covered_key = f"covered_at_{cutoff}"
            best = max(group, key=lambda item: float(item.get(coverage_key, 0.0)))
            output_row[f"best_gold_sentence_coverage_at_{cutoff}"] = best.get(
                coverage_key,
                0.0,
            )
            output_row[f"any_covered_at_{cutoff}"] = any(
                bool(item.get(covered_key)) for item in group
            )
            output_row[f"best_retriever_at_{cutoff}"] = best["retriever"]
        output_row["best_first_covering_rank"] = min(
            (
                int(item["first_covering_rank"])
                for item in group
                if item.get("first_covering_rank") is not None
            ),
            default=None,
        )
        for item in sorted(group, key=lambda value: str(value["retriever"])):
            retriever_key = slugify(str(item["retriever"])).replace("-", "_")
            for cutoff in cutoffs:
                coverage_key = f"gold_sentence_coverage_at_{cutoff}"
                covered_key = f"covered_at_{cutoff}"
                output_row[f"{retriever_key}_coverage_at_{cutoff}"] = item.get(
                    coverage_key,
                    0.0,
                )
                output_row[f"{retriever_key}_covered_at_{cutoff}"] = item.get(
                    covered_key,
                    False,
                )
            output_row[f"{retriever_key}_first_covering_rank"] = item.get(
                "first_covering_rank"
            )
        summary_rows.append(output_row)
    return summary_rows


def summarize_chunk_matching_by_question(
    records: list[EligibilityRecord],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[EligibilityRecord]] = {}
    for record in records:
        if record.gold_sentence_count <= 0:
            continue
        grouped.setdefault(record.question_index, []).append(record)

    rows: list[dict[str, Any]] = []
    for question_index, question_records in sorted(grouped.items()):
        full_match = [
            record
            for record in question_records
            if record.matched_sentence_count == record.gold_sentence_count
        ]
        partial_match = [
            record
            for record in question_records
            if 0 < record.matched_sentence_count < record.gold_sentence_count
        ]
        no_match = [
            record for record in question_records if record.matched_sentence_count == 0
        ]
        total_gold_sentences = sum(
            record.gold_sentence_count for record in question_records
        )
        matched_sentences = sum(
            record.matched_sentence_count for record in question_records
        )
        raw_contract_matched = sum(
            record.raw_contract_sentence_match_count for record in question_records
        )
        rows.append(
            {
                "question_index": question_index,
                "category": question_records[0].category,
                "rows_with_split_golden_answer_sentences": len(question_records),
                "full_match_rows": len(full_match),
                "partial_match_rows": len(partial_match),
                "no_match_rows": len(no_match),
                "full_match_rate": round(len(full_match) / len(question_records), 4),
                "partial_match_rate": round(
                    len(partial_match) / len(question_records), 4
                ),
                "no_match_rate": round(len(no_match) / len(question_records), 4),
                "split_golden_sentence_count": total_gold_sentences,
                "matched_split_golden_sentence_count": matched_sentences,
                "split_sentence_match_rate": round(
                    matched_sentences / total_gold_sentences
                    if total_gold_sentences
                    else 0.0,
                    4,
                ),
                "raw_contract_matched_sentence_count": raw_contract_matched,
                "raw_contract_match_rate": round(
                    raw_contract_matched / total_gold_sentences
                    if total_gold_sentences
                    else 0.0,
                    4,
                ),
            }
        )
    return rows


def chunk_match_counts(records: list[EligibilityRecord]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[EligibilityRecord]] = {}
    for record in records:
        if record.gold_sentence_count <= 0:
            continue
        grouped.setdefault(record.question_index, []).append(record)

    counts: dict[int, dict[str, Any]] = {}
    for question_index, question_records in grouped.items():
        full_match = [
            record
            for record in question_records
            if record.matched_sentence_count == record.gold_sentence_count
        ]
        partial_match = [
            record
            for record in question_records
            if 0 < record.matched_sentence_count < record.gold_sentence_count
        ]
        no_match = [
            record for record in question_records if record.matched_sentence_count == 0
        ]
        total_gold_sentences = sum(
            record.gold_sentence_count for record in question_records
        )
        matched_sentences = sum(
            record.matched_sentence_count for record in question_records
        )
        counts[question_index] = {
            "category": question_records[0].category,
            "rows_with_split_golden_answer_sentences": len(question_records),
            "full_match_rows": len(full_match),
            "partial_match_rows": len(partial_match),
            "no_match_rows": len(no_match),
            "split_golden_sentence_count": total_gold_sentences,
            "matched_split_golden_sentence_count": matched_sentences,
            "split_sentence_match_rate": (
                matched_sentences / total_gold_sentences
                if total_gold_sentences
                else 0.0
            ),
            "full_match_rate": len(full_match) / len(question_records)
            if question_records
            else 0.0,
        }
    return counts


def compare_chunking_versions_by_question(
    *,
    baseline_version: str,
    baseline_records: list[EligibilityRecord],
    comparison_version: str,
    comparison_records: list[EligibilityRecord],
) -> list[dict[str, Any]]:
    baseline = chunk_match_counts(baseline_records)
    comparison = chunk_match_counts(comparison_records)
    question_indices = sorted(set(baseline) | set(comparison))
    rows: list[dict[str, Any]] = []
    for question_index in question_indices:
        base = baseline.get(question_index, {})
        other = comparison.get(question_index, {})
        base_rate = float(base.get("split_sentence_match_rate", 0.0))
        other_rate = float(other.get("split_sentence_match_rate", 0.0))
        rows.append(
            {
                "question_index": question_index,
                "category": base.get("category") or other.get("category") or "",
                "baseline_version": baseline_version,
                "comparison_version": comparison_version,
                "baseline_rows": base.get("rows_with_split_golden_answer_sentences", 0),
                "comparison_rows": other.get(
                    "rows_with_split_golden_answer_sentences", 0
                ),
                "baseline_full_match_rows": base.get("full_match_rows", 0),
                "comparison_full_match_rows": other.get("full_match_rows", 0),
                "full_match_row_delta": int(base.get("full_match_rows", 0))
                - int(other.get("full_match_rows", 0)),
                "baseline_partial_match_rows": base.get("partial_match_rows", 0),
                "comparison_partial_match_rows": other.get("partial_match_rows", 0),
                "baseline_no_match_rows": base.get("no_match_rows", 0),
                "comparison_no_match_rows": other.get("no_match_rows", 0),
                "baseline_split_sentence_match_rate": round(base_rate, 4),
                "comparison_split_sentence_match_rate": round(other_rate, 4),
                "split_sentence_match_rate_delta": round(base_rate - other_rate, 4),
                "baseline_matched_split_golden_sentence_count": base.get(
                    "matched_split_golden_sentence_count",
                    0,
                ),
                "comparison_matched_split_golden_sentence_count": other.get(
                    "matched_split_golden_sentence_count",
                    0,
                ),
            }
        )
    return rows


def cached_chunking_version_comparison(
    *,
    eval_rows: pd.DataFrame,
    baseline_version: str,
    baseline_records: list[EligibilityRecord],
    comparison_version: str,
    output_dir: Path,
    contracts: dict[int, ContractDocument],
    progress: ProgressLogger | None = None,
) -> list[dict[str, Any]]:
    comparison_spans = load_cached_sentence_spans_for_version(
        output_dir=output_dir,
        chunking_version=comparison_version,
        document_ids=set(contracts),
    )
    if comparison_spans is None:
        return [
            {
                "baseline_version": baseline_version,
                "comparison_version": comparison_version,
                "status": "comparison_cache_missing",
                "cache_path": str(
                    sentence_cache_paths(output_dir, comparison_version)["sentences"]
                ),
            }
        ]
    emit_progress(
        progress,
        f"Comparing chunking versions for golden matching: {baseline_version} vs {comparison_version}",
    )
    comparison_records = eligibility_records(
        eval_rows,
        sentence_lookup_from_spans(comparison_spans),
        contracts=contracts,
        progress=progress,
    )
    return compare_chunking_versions_by_question(
        baseline_version=baseline_version,
        baseline_records=baseline_records,
        comparison_version=comparison_version,
        comparison_records=comparison_records,
    )


def cached_chunking_review_payload(
    *,
    eval_rows: pd.DataFrame,
    output_dir: Path,
    source_chunking_version: str,
    display_version: str,
    contracts: dict[int, ContractDocument],
    embedding_model: str,
    enrichments: dict[int, Any] | None = None,
    progress: ProgressLogger | None = None,
) -> tuple[dict[str, Any] | None, list[EligibilityRecord] | None]:
    spans = load_cached_sentence_spans_for_version(
        output_dir=output_dir,
        chunking_version=source_chunking_version,
        document_ids=set(contracts),
    )
    if spans is None:
        return None, None
    emit_progress(
        progress,
        f"Building chunking review payload: {display_version} from {source_chunking_version}",
    )
    lookup = sentence_lookup_from_spans(spans)
    records = eligibility_records(
        eval_rows,
        lookup,
        contracts=contracts,
        progress=progress,
    )
    summary_rows, documents, reviews = build_chunking_review_payload(
        eval_rows=eval_rows,
        records=records,
        sentence_lookup=lookup,
        contracts=contracts,
        embedding_model=embedding_model,
        chunking_version=display_version,
        dense_encoding_info={},
        enrichments=enrichments,
    )
    for row in summary_rows:
        row["source_chunking_version"] = source_chunking_version
        row["source_sentence_count"] = len(spans)
    return (
        {
            "label": display_version,
            "source_chunking_version": source_chunking_version,
            "summary_rows": summary_rows,
            "match_distribution": summarize_chunk_matching_by_question(records),
            "documents": documents,
            "reviews": reviews,
        },
        records,
    )


def build_chunking_review_payload(
    *,
    eval_rows: pd.DataFrame,
    records: list[EligibilityRecord],
    sentence_lookup: dict[int, list[SentenceSpan]],
    contracts: dict[int, ContractDocument],
    embedding_model: str,
    chunking_version: str,
    dense_encoding_info: dict[str, Any],
    enrichments: dict[int, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records_by_row_id = {record.row_id: record for record in records}
    answerable_rows = eval_rows[~eval_rows["is_impossible"].astype(bool)].copy()
    review_rows: list[dict[str, Any]] = []
    review_document_ids: set[int] = set()

    for row in answerable_rows.itertuples(index=False):
        row_id = evaluation_row_id(int(row.document_row_id), int(row.question_index))
        record = records_by_row_id.get(row_id)
        if record is None or record.gold_sentence_count == 0:
            continue
        enrichment = (enrichments or {}).get(int(row.question_index))
        review_document_ids.add(int(row.document_row_id))
        review_rows.append(
            {
                "row_id": row_id,
                "document_row_id": int(row.document_row_id),
                "document_title": contracts.get(
                    int(row.document_row_id),
                    ContractDocument(
                        document_row_id=int(row.document_row_id),
                        title=str(row.document_row_id),
                        text="",
                    ),
                ).title,
                "question_index": int(row.question_index),
                "category": str(row.category),
                "question": str(row.question),
                "raw_question": str(row.question),
                "enriched_question": (
                    str(enrichment.enriched_query)
                    if enrichment is not None
                    else str(row.question)
                ),
                "enrichment_terms": (
                    str(enrichment.enrichment_terms) if enrichment is not None else ""
                ),
                "answer_format": str(row.answer_format),
                "raw_golden_answers": answer_texts(row.answers),
                "gold_sentence_count": record.gold_sentence_count,
                "matched_sentence_count": record.matched_sentence_count,
                "match_rate": round(record.per_row_gold_sentence_contract_coverage, 4),
                "match_status": "full_match"
                if record.is_eligible
                else "partial_or_no_match",
                "reason": record.reason,
                "matched_sentence_ids": record.matched_sentence_ids,
                "golden_sentences": [
                    sentence.to_dict() for sentence in record.golden_sentences
                ],
            }
        )

    documents: dict[str, dict[str, Any]] = {}
    for document_row_id in sorted(review_document_ids):
        contract = contracts.get(
            document_row_id,
            ContractDocument(
                document_row_id=document_row_id, title=str(document_row_id), text=""
            ),
        )
        documents[str(document_row_id)] = {
            "document_row_id": document_row_id,
            "title": contract.title,
            "raw_text": contract.text,
            "sentences": [
                {
                    "sentence_id": sentence.sentence_id,
                    "sentence_index": sentence.sentence_index,
                    "raw_text": sentence.raw_text,
                    "normalized_text": sentence.normalized_text,
                    "start_char": sentence.start_char,
                    "end_char": sentence.end_char,
                    "page_number": sentence.page_number,
                    "section_number": sentence.section_number,
                    "section_title": sentence.section_title,
                    "clause_path": sentence.clause_path,
                }
                for sentence in sentence_lookup.get(document_row_id, [])
            ],
        }

    summary_rows = [
        {
            "chunking_version": chunking_version,
            "review_answerable_question_rows": len(review_rows),
            "review_contracts": len(documents),
            "review_contract_sentences": sum(
                len(value.get("sentences", [])) for value in documents.values()
            ),
            "embedding_model": embedding_model,
            "embedding_backend": dense_encoding_info.get("embedding_backend"),
            "encoded_sentence_count": dense_encoding_info.get("encoded_sentence_count"),
            "embedding_cache_dir": dense_encoding_info.get("embedding_cache_dir"),
        }
    ]
    return summary_rows, documents, review_rows


def compact_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        compact.append(
            {
                key: json.dumps(value, ensure_ascii=False, default=json_default)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
        )
    return compact


def run_rag_eval(
    *,
    run_id: str,
    sample_size: int,
    seed: int,
    retrievers: list[str],
    top_k: int,
    output_dir: Path,
    preflight_golden_sentences_only: bool = False,
    eval_split: str | None = None,
    question_indices: list[int] | None = None,
    contract_ids: list[int] | None = None,
    embedding_model: str = "tfidf",
    chunking_version: str = DEFAULT_CHUNKING_VERSION,
    contract_scope: str = "all",
    rebuild_chunks: bool = False,
    rebuild_embeddings: bool = False,
    query_enrichment_provider: str = "auto",
    query_enrichment_model: str = "deepseek-chat",
    hierarchical_leaf_k: int = 50,
    hierarchical_top_sections: int = 5,
    progress: ProgressLogger | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    paths = rag_output_paths(output_dir, run_id)
    if contract_scope not in {"all", "eval-set"}:
        raise ValueError("--contract-scope must be 'all' or 'eval-set'")
    emit_progress(progress, f"Starting RAG run: run_id={run_id}")
    emit_progress(progress, "Loading CUAD dataset and evaluation rows")
    selection = select_evaluation_set(
        sample_size=sample_size,
        seed=seed,
        contract_ids=contract_ids,
        question_indices=question_indices,
        eval_split=eval_split,
    )
    selected_ids = selection.selected_ids
    contract_lookup = selection.contract_lookup
    eval_rows = selection.eval_rows
    emit_progress(
        progress,
        f"Loaded evaluation rows: contracts={len(selected_ids)}, rows={len(eval_rows)}",
    )
    if contract_scope == "all":
        emit_progress(progress, "Loading all CUAD contracts for sentence chunking")
        contracts = contracts_from_lookup(all_contract_lookup())
    else:
        selected_lookup = {
            int(document_row_id): contract_lookup[int(document_row_id)]
            for document_row_id in selected_ids
            if int(document_row_id) in contract_lookup
        }
        contracts = contracts_from_lookup(selected_lookup)
    emit_progress(
        progress,
        f"Contracts selected for chunking: {len(contracts)} "
        f"(contract_scope={contract_scope})",
    )
    sentence_chunking_version = (
        DEFAULT_CHUNKING_VERSION
        if chunking_version == LEGAL_RECURSIVE_CHUNKING_VERSION
        or any(method in LEGAL_RECURSIVE_RETRIEVERS for method in retrievers)
        else chunking_version
    )
    legal_recursive_chunking_version = (
        chunking_version
        if chunking_version == LEGAL_RECURSIVE_CHUNKING_VERSION
        else LEGAL_RECURSIVE_CHUNKING_VERSION
    )

    sentence_spans, sentence_cache_info = load_or_build_sentence_cache(
        contracts=contracts,
        output_dir=output_dir,
        chunking_version=sentence_chunking_version,
        rebuild=rebuild_chunks,
        progress=progress,
    )
    sentence_lookup: dict[int, list[SentenceSpan]] = {}
    for span in sentence_spans:
        sentence_lookup.setdefault(span.document_row_id, []).append(span)

    chunks = chunks_from_sentences(sentence_spans)
    legal_recursive_chunks: list[RagChunk] = []
    legal_recursive_cache_info: dict[str, Any] = {}
    if any(method in LEGAL_RECURSIVE_RETRIEVERS for method in retrievers):
        legal_recursive_chunks, legal_recursive_cache_info = (
            load_or_build_legal_recursive_cache(
                contracts=contracts,
                sentence_lookup=sentence_lookup,
                output_dir=output_dir,
                chunking_version=legal_recursive_chunking_version,
                rebuild=rebuild_chunks,
                progress=progress,
            )
        )

    dense_retriever, dense_encoding_info = load_or_build_dense_sentence_encoder(
        chunks=chunks,
        method="dense_sentence",
        output_dir=output_dir,
        chunking_version=sentence_chunking_version,
        embedding_model=embedding_model,
        rebuild=rebuild_embeddings,
        progress=progress,
    )
    dense_encoding_info.setdefault(
        "dense_sentence_encoding_cache_hit",
        dense_encoding_info.get("dense_sentence_encoding_cache_hit", False),
    )
    bm25_retriever, bm25_cache_hit = load_or_build_retriever(
        method="bm25_sentence",
        chunks=chunks,
        output_dir=output_dir,
        chunking_version=sentence_chunking_version,
        embedding_model=embedding_model,
        rebuild=rebuild_embeddings,
        progress=progress,
    )
    section_index = (
        build_section_index(sentence_spans)
        if any(method in HIERARCHICAL_RETRIEVERS for method in retrievers)
        else {}
    )
    bm25_hierarchical_retriever: HierarchicalRetriever | None = None
    dense_hierarchical_retriever: HierarchicalRetriever | None = None
    if "bm25_hierarchical" in retrievers:
        bm25_hierarchical_retriever = build_hierarchical_retriever(
            "bm25_hierarchical",
            index=bm25_retriever.index,
            section_index=section_index,
            leaf_k=hierarchical_leaf_k,
            top_sections=hierarchical_top_sections,
        )
    if "dense_hierarchical" in retrievers:
        dense_hierarchical_retriever = build_hierarchical_retriever(
            "dense_hierarchical",
            index=dense_retriever.index,
            section_index=section_index,
            leaf_k=hierarchical_leaf_k,
            top_sections=hierarchical_top_sections,
        )
    dense_legal_retriever: SentenceRetriever | None = None
    dense_legal_encoding_info: dict[str, Any] = {}
    if "dense_legal_recursive" in retrievers:
        dense_legal_retriever, dense_legal_encoding_info = (
            load_or_build_dense_sentence_encoder(
                chunks=legal_recursive_chunks,
                method="dense_legal_recursive",
                output_dir=output_dir,
                chunking_version=legal_recursive_chunking_version,
                embedding_model=embedding_model,
                rebuild=rebuild_embeddings,
                progress=progress,
            )
        )

    records = eligibility_records(
        eval_rows,
        sentence_lookup,
        contracts=contracts,
        progress=progress,
    )
    eligibility_by_row_id = {record.row_id: record for record in records}
    enrichments, enrichment_info = build_question_enrichments(
        eval_rows=eval_rows,
        output_dir=output_dir,
        provider=query_enrichment_provider,
        model=query_enrichment_model,
        progress=progress,
    )
    query_enrichment_rows, query_enrichment_summary = run_query_enrichment_eval(
        eval_rows=eval_rows,
        eligibility_by_row_id=eligibility_by_row_id,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        enrichments=enrichments,
        top_ks=(10, 20, 30),
        progress=progress,
    )
    eligibility_rows = [record.to_dict() for record in records]
    gold_summary = summarize_eligibility(records)
    if contract_scope == "all":
        emit_progress(
            progress,
            "Building all-contract question-level golden eligibility summary",
        )
        question_summary_records = eligibility_records(
            all_contract_question_rows(question_indices),
            sentence_lookup,
            contracts=contracts,
            progress=progress,
        )
        eligibility_question_summary_scope = "all_contracts"
    else:
        question_summary_records = records
        eligibility_question_summary_scope = "evaluated_contracts"
    eligibility_question_summary = summarize_eligibility_by_question(
        question_summary_records
    )
    chunking_match_distribution = summarize_chunk_matching_by_question(records)
    chunking_summary_rows, chunking_documents, chunking_reviews = (
        build_chunking_review_payload(
            eval_rows=eval_rows,
            records=records,
            sentence_lookup=sentence_lookup,
            contracts=contracts,
            embedding_model=embedding_model,
            chunking_version=chunking_version,
            dense_encoding_info=dense_encoding_info,
            enrichments=enrichments,
        )
    )
    chunking_versions: dict[str, dict[str, Any]] = {
        "sentence-v3": {
            "label": "sentence-v3",
            "source_chunking_version": sentence_chunking_version,
            "summary_rows": chunking_summary_rows,
            "match_distribution": chunking_match_distribution,
            "documents": chunking_documents,
            "reviews": chunking_reviews,
        }
    }
    if legal_recursive_chunks:
        lr_review_doc_ids = {int(doc_id) for doc_id in chunking_documents}
        lr_documents = build_legal_recursive_documents(
            legal_recursive_chunks, contracts, lr_review_doc_ids
        )
        lr_summary_rows = (
            [
                {
                    **chunking_summary_rows[0],
                    "chunking_version": "legal-recursive-v1",
                    "review_contract_sentences": sum(
                        len(d.get("sentences", [])) for d in lr_documents.values()
                    ),
                }
            ]
            if chunking_summary_rows
            else []
        )
        lr_match_distribution = chunking_match_distribution
        lr_reviews = chunking_reviews
    else:
        lr_documents = {}
        lr_summary_rows = []
        lr_match_distribution = []
        lr_reviews = []
    chunking_versions["legal-recursive-v1"] = {
        "label": "legal-recursive-v1",
        "source_chunking_version": legal_recursive_chunking_version,
        "summary_rows": lr_summary_rows,
        "match_distribution": lr_match_distribution,
        "documents": lr_documents,
        "reviews": lr_reviews,
    }
    chunking_version_comparison: list[dict[str, Any]] = []
    chunk_summary = chunking_summary(
        contracts=contracts,
        sentence_count=int(sentence_cache_info["sentence_count"]),
        contract_scope=contract_scope,
    )

    emit_progress(progress, "Writing golden-answer coverage outputs")
    write_csv(paths["gold_csv"], compact_csv_rows(eligibility_rows))
    write_csv(paths["gold_question_summary"], eligibility_question_summary)
    write_json(paths["gold_summary"], gold_summary)
    write_jsonl(paths["sentences"], [chunk.to_dict() for chunk in chunks])
    write_csv(
        paths["query_enrichment_results"], compact_csv_rows(query_enrichment_rows)
    )
    write_csv(paths["query_enrichment_summary"], query_enrichment_summary)

    config = {
        "run_id": run_id,
        "sample_size": sample_size,
        "seed": seed,
        "selected_contract_ids": selected_ids,
        "retrievers": retrievers,
        "top_k": top_k,
        "embedding_model": embedding_model,
        "chunking_version": chunking_version,
        "sentence_chunking_version": sentence_chunking_version,
        "legal_recursive_chunking_version": legal_recursive_chunking_version,
        **chunk_summary,
        **dense_encoding_info,
        "query_enrichment_bm25_cache_hit": bm25_cache_hit,
        **legal_recursive_cache_info,
        **dense_legal_encoding_info,
        **enrichment_info,
        "hierarchical_leaf_k": (
            hierarchical_leaf_k
            if any(method in HIERARCHICAL_RETRIEVERS for method in retrievers)
            else None
        ),
        "hierarchical_top_sections": (
            hierarchical_top_sections
            if any(method in HIERARCHICAL_RETRIEVERS for method in retrievers)
            else None
        ),
        "eligibility_question_summary_scope": eligibility_question_summary_scope,
        "preflight_golden_sentences_only": preflight_golden_sentences_only,
    }
    write_json(paths["config"], config)

    if preflight_golden_sentences_only:
        summary = {
            **gold_summary,
            **sentence_cache_info,
            **chunk_summary,
            **dense_encoding_info,
            "query_enrichment_bm25_cache_hit": bm25_cache_hit,
            **legal_recursive_cache_info,
            **dense_legal_encoding_info,
            **enrichment_info,
            "run_id": run_id,
            "eligibility_question_summary_scope": eligibility_question_summary_scope,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "retrieval_skipped": True,
        }
        write_json(paths["summary"], summary)
        write_pipeline_html(
            paths["pipeline_html"],
            run_id=run_id,
            summary=summary,
            chunking_summary_rows=chunking_summary_rows,
            chunking_match_distribution=chunking_match_distribution,
            chunking_version_comparison=chunking_version_comparison,
            chunking_versions=chunking_versions,
            chunking_documents=chunking_documents,
            chunking_reviews=chunking_reviews,
            query_enrichment_summary=query_enrichment_summary,
            query_enrichment_rows=compact_csv_rows(query_enrichment_rows),
            eligibility_question_summary=eligibility_question_summary,
            eligibility_rows=compact_csv_rows(eligibility_rows),
            retrieval_doc_question_summary=[],
            retrieval_rows=[],
            ranking_summary=[],
            artifact_paths=paths,
            hierarchical_config=None,
        )
        emit_progress(progress, "RAG preflight complete")
        return summary

    retrieval_rows, ranking_summary, index_cache_hits = run_sentence_retrieval(
        eval_rows=eval_rows,
        eligibility_by_row_id=eligibility_by_row_id,
        chunks=chunks,
        chunks_by_method={
            "bm25_sentence": chunks,
            "dense_sentence": chunks,
            "bm25_legal_recursive": legal_recursive_chunks,
            "dense_legal_recursive": legal_recursive_chunks,
            "bm25_hierarchical": chunks,
            "dense_hierarchical": chunks,
        },
        chunking_version_by_method={
            "bm25_sentence": sentence_chunking_version,
            "dense_sentence": sentence_chunking_version,
            "bm25_legal_recursive": legal_recursive_chunking_version,
            "dense_legal_recursive": legal_recursive_chunking_version,
            "bm25_hierarchical": sentence_chunking_version,
            "dense_hierarchical": sentence_chunking_version,
        },
        retriever_methods=retrievers,
        top_k=top_k,
        output_dir=output_dir,
        chunking_version=sentence_chunking_version,
        embedding_model=embedding_model,
        rebuild_embeddings=rebuild_embeddings,
        prebuilt_retrievers={
            "dense_sentence": (
                dense_retriever,
                bool(dense_encoding_info["dense_sentence_encoding_cache_hit"]),
            ),
            "bm25_sentence": (
                bm25_retriever,
                bool(bm25_cache_hit),
            ),
            **(
                {"bm25_hierarchical": (bm25_hierarchical_retriever, False)}
                if bm25_hierarchical_retriever is not None
                else {}
            ),
            **(
                {"dense_hierarchical": (dense_hierarchical_retriever, False)}
                if dense_hierarchical_retriever is not None
                else {}
            ),
            **(
                {
                    "dense_legal_recursive": (
                        dense_legal_retriever,
                        bool(
                            dense_legal_encoding_info.get(
                                "dense_legal_recursive_encoding_cache_hit",
                                False,
                            )
                        ),
                    )
                }
                if dense_legal_retriever is not None
                else {}
            ),
        },
        progress=progress,
    )
    retrieval_doc_question_summary = summarize_retrieval_by_document_question(
        retrieval_rows,
        top_k=top_k,
    )

    emit_progress(progress, "Writing retrieval outputs")
    write_jsonl(paths["results_jsonl"], retrieval_rows)
    write_csv(paths["results_csv"], compact_csv_rows(retrieval_rows))
    write_csv(
        paths["results_doc_question_summary"],
        compact_csv_rows(retrieval_doc_question_summary),
    )
    write_csv(paths["ranking_summary"], ranking_summary)

    summary = {
        **gold_summary,
        **sentence_cache_info,
        **chunk_summary,
        **dense_encoding_info,
        "query_enrichment_bm25_cache_hit": bm25_cache_hit,
        **legal_recursive_cache_info,
        **dense_legal_encoding_info,
        **enrichment_info,
        **index_cache_hits,
        "run_id": run_id,
        "eligibility_question_summary_scope": eligibility_question_summary_scope,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "retrievers": retrievers,
        "top_k": top_k,
        "hierarchical_leaf_k": (
            hierarchical_leaf_k
            if any(method in HIERARCHICAL_RETRIEVERS for method in retrievers)
            else None
        ),
        "hierarchical_top_sections": (
            hierarchical_top_sections
            if any(method in HIERARCHICAL_RETRIEVERS for method in retrievers)
            else None
        ),
        "retrieval_rows": len(retrieval_rows),
        "ranking_summary": ranking_summary,
    }
    write_json(paths["summary"], summary)
    write_pipeline_html(
        paths["pipeline_html"],
        run_id=run_id,
        summary=summary,
        chunking_summary_rows=chunking_summary_rows,
        chunking_match_distribution=chunking_match_distribution,
        chunking_version_comparison=chunking_version_comparison,
        chunking_versions=chunking_versions,
        chunking_documents=chunking_documents,
        chunking_reviews=chunking_reviews,
        query_enrichment_summary=query_enrichment_summary,
        query_enrichment_rows=compact_csv_rows(query_enrichment_rows),
        eligibility_question_summary=eligibility_question_summary,
        eligibility_rows=compact_csv_rows(eligibility_rows),
        retrieval_doc_question_summary=compact_csv_rows(retrieval_doc_question_summary),
        retrieval_rows=compact_csv_rows(retrieval_rows),
        ranking_summary=ranking_summary,
        artifact_paths=paths,
        hierarchical_config=(
            {
                "leaf_k": hierarchical_leaf_k,
                "top_sections": hierarchical_top_sections,
            }
            if any(method in HIERARCHICAL_RETRIEVERS for method in retrievers)
            else None
        ),
    )
    emit_progress(progress, "RAG run complete")
    return summary
