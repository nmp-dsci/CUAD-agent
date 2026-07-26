"""Exact retrieved sentence-id coverage metrics."""

from __future__ import annotations

from typing import Any


def retrieved_sentence_ids_from_results(
    retrieved: list[Any], *, top_chunks: int
) -> list[str]:
    sentence_ids: list[str] = []
    seen: set[str] = set()
    for result in retrieved[:top_chunks]:
        for sentence_id in result.chunk.sentence_ids:
            if sentence_id not in seen:
                seen.add(sentence_id)
                sentence_ids.append(sentence_id)
    return sentence_ids


def coverage_by_top_chunks(
    gold_sentence_ids: list[str],
    retrieved: list[Any],
    *,
    top_chunks: int,
) -> dict[str, object]:
    gold = list(dict.fromkeys(gold_sentence_ids))
    retrieved_ids = retrieved_sentence_ids_from_results(
        retrieved, top_chunks=top_chunks
    )
    retrieved_set = set(retrieved_ids)
    covered = [sentence_id for sentence_id in gold if sentence_id in retrieved_set]
    first_ranks: list[int] = []
    for sentence_id in covered:
        for result in retrieved[:top_chunks]:
            if sentence_id in result.chunk.sentence_ids:
                first_ranks.append(result.rank)
                break
    return {
        "k": top_chunks,
        "gold_sentence_count": len(gold),
        "covered_sentence_count": len(covered),
        "gold_sentence_coverage": len(covered) / len(gold) if gold else 0.0,
        "all_gold_sentences_covered": bool(gold) and len(covered) == len(gold),
        "covered_sentence_ids": covered,
        "first_covering_rank": min(first_ranks) if first_ranks else None,
    }


def coverage_at_k(
    gold_sentence_ids: list[str], retrieved_sentence_ids: list[str], k: int
) -> dict[str, object]:
    gold = list(dict.fromkeys(gold_sentence_ids))
    retrieved = retrieved_sentence_ids[:k]
    retrieved_set = set(retrieved)
    covered = [sentence_id for sentence_id in gold if sentence_id in retrieved_set]
    first_ranks = [
        retrieved.index(sentence_id) + 1
        for sentence_id in covered
        if sentence_id in retrieved
    ]
    return {
        "k": k,
        "gold_sentence_count": len(gold),
        "covered_sentence_count": len(covered),
        "gold_sentence_coverage": len(covered) / len(gold) if gold else 0.0,
        "all_gold_sentences_covered": bool(gold) and len(covered) == len(gold),
        "covered_sentence_ids": covered,
        "first_covering_rank": min(first_ranks) if first_ranks else None,
    }
