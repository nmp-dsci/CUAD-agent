from __future__ import annotations

from cuad_agent.rag.chunks import chunks_from_sentences
from cuad_agent.rag.coverage import coverage_by_top_chunks
from cuad_agent.rag.hierarchy import (
    HierarchicalRetriever,
    build_section_index,
    format_hierarchical_context,
    section_key_for,
)
from cuad_agent.rag.indexes import BM25SentenceIndex
from cuad_agent.rag.sentences import SentenceSpan, normalize_sentence_text


def span(
    document_row_id: int,
    index: int,
    text: str,
    *,
    clause_path: list[str] | None = None,
    section_number: str | None = None,
    section_title: str | None = None,
) -> SentenceSpan:
    return SentenceSpan(
        document_row_id=document_row_id,
        sentence_id=f"{document_row_id}:s:{index}",
        sentence_index=index,
        raw_text=text,
        normalized_text=normalize_sentence_text(text),
        start_char=index * 100,
        end_char=index * 100 + len(text),
        section_number=section_number,
        section_title=section_title,
        clause_path=clause_path or [],
    )


def test_build_section_index_groups_by_clause_path() -> None:
    spans = [
        span(1, 0, "First sentence.", clause_path=["3.1", "Confidentiality"]),
        span(1, 1, "Second sentence.", clause_path=["3.1", "Confidentiality"]),
    ]

    index = build_section_index(spans)

    assert len(index[1]) == 1
    assert index[1][0].section_key == ("3.1", "Confidentiality")
    assert index[1][0].sentence_ids == ["1:s:0", "1:s:1"]


def test_build_section_index_unsectioned_bucket() -> None:
    spans = [span(1, 0, "Preamble."), span(1, 1, "Background.")]

    index = build_section_index(spans)

    assert len(index[1]) == 1
    assert index[1][0].section_key == ("_unsectioned",)
    assert index[1][0].sentence_ids == ["1:s:0", "1:s:1"]


def test_section_key_stability() -> None:
    left = span(1, 0, "Left.", clause_path=["7", "Law"])
    right = span(1, 1, "Right.", clause_path=["7", "Law"])

    assert section_key_for(left) == section_key_for(right)


def test_hierarchical_retriever_expands_section() -> None:
    spans = [
        span(1, 0, "Assignment requires consent.", clause_path=["2", "Assignment"]),
        span(1, 1, "Consent must be written.", clause_path=["2", "Assignment"]),
        span(1, 2, "Notices are delivered by mail.", clause_path=["9", "Notices"]),
    ]
    retriever = HierarchicalRetriever(
        method="bm25_hierarchical",
        index=BM25SentenceIndex(chunks_from_sentences(spans)),
        section_index=build_section_index(spans),
        leaf_k=1,
        top_sections=1,
    )

    results = retriever.search("assignment", document_row_id=1, top_k=5)

    assert {result.chunk.chunk_id for result in results} == {"1:s:0", "1:s:1"}


def test_hierarchical_retriever_expands_full_clause_when_siblings_rank_low() -> None:
    spans = [
        span(1, 0, "Assignment requires consent.", clause_path=["2", "Assignment"]),
        span(1, 1, "This approval must be written.", clause_path=["2", "Assignment"]),
        span(1, 2, "The transfer is void otherwise.", clause_path=["2", "Assignment"]),
        span(1, 3, "No delegation releases liability.", clause_path=["2", "Assignment"]),
        span(1, 4, "Assignment appears in another clause.", clause_path=["5", "Other"]),
    ]
    chunks = chunks_from_sentences(spans)
    flat = BM25SentenceIndex(chunks)
    leaf_results = flat.search("assignment consent", document_row_id=1, top_k=1)
    retriever = HierarchicalRetriever(
        method="bm25_hierarchical",
        index=flat,
        section_index=build_section_index(spans),
        leaf_k=1,
        top_sections=1,
    )

    results = retriever.search("assignment consent", document_row_id=1, top_k=10)

    leaf_ids = {result.chunk.chunk_id for result in leaf_results}
    assert {"1:s:1", "1:s:2", "1:s:3"}.isdisjoint(leaf_ids)
    assert {"1:s:0", "1:s:1", "1:s:2", "1:s:3"}.issubset(
        {result.chunk.chunk_id for result in results}
    )


def test_hierarchical_expansion_uses_sections_not_leaf_results_only() -> None:
    spans = [
        span(1, 0, "Audit rights apply.", clause_path=["4", "Audit"]),
        span(1, 1, "Records must be retained.", clause_path=["4", "Audit"]),
    ]
    flat = BM25SentenceIndex(chunks_from_sentences(spans))
    leaf_results = flat.search("audit", document_row_id=1, top_k=1)
    retriever = HierarchicalRetriever(
        method="bm25_hierarchical",
        index=flat,
        section_index=build_section_index(spans),
        leaf_k=1,
        top_sections=1,
    )

    results = retriever.search("audit", document_row_id=1, top_k=5)

    assert "1:s:1" not in {result.chunk.chunk_id for result in leaf_results}
    assert "1:s:1" in {result.chunk.chunk_id for result in results}


def test_hierarchical_retriever_respects_top_k() -> None:
    spans = [
        span(1, 0, "Audit rights apply.", clause_path=["4", "Audit"]),
        span(1, 1, "Records must be retained.", clause_path=["4", "Audit"]),
        span(1, 2, "Records must be supplied.", clause_path=["4", "Audit"]),
    ]
    retriever = HierarchicalRetriever(
        method="bm25_hierarchical",
        index=BM25SentenceIndex(chunks_from_sentences(spans)),
        section_index=build_section_index(spans),
        leaf_k=1,
        top_sections=1,
    )

    assert len(retriever.search("audit records", document_row_id=1, top_k=2)) == 2


def test_hierarchical_retriever_top_sections_limit() -> None:
    spans = [
        span(1, 0, "Assignment assignment.", clause_path=["2", "Assignment"]),
        span(1, 1, "Consent follows.", clause_path=["2", "Assignment"]),
        span(1, 2, "Audit rights apply.", clause_path=["4", "Audit"]),
        span(1, 3, "Records follow.", clause_path=["4", "Audit"]),
    ]
    retriever = HierarchicalRetriever(
        method="bm25_hierarchical",
        index=BM25SentenceIndex(chunks_from_sentences(spans)),
        section_index=build_section_index(spans),
        leaf_k=4,
        top_sections=1,
    )

    results = retriever.search("assignment audit", document_row_id=1, top_k=10)

    assert {tuple(result.chunk.clause_path) for result in results} == {
        ("2", "Assignment")
    }


def test_hierarchical_retriever_filters_requested_document() -> None:
    spans = [
        span(1, 0, "Assignment requires consent.", clause_path=["2"]),
        span(2, 0, "Assignment is unrestricted.", clause_path=["2"]),
    ]
    retriever = HierarchicalRetriever(
        method="bm25_hierarchical",
        index=BM25SentenceIndex(chunks_from_sentences(spans)),
        section_index=build_section_index(spans),
        leaf_k=5,
        top_sections=1,
    )

    results = retriever.search("assignment", document_row_id=1, top_k=10)

    assert results
    assert all(result.chunk.document_row_id == 1 for result in results)


def test_hierarchical_retriever_missing_document_returns_empty() -> None:
    spans = [span(1, 0, "Assignment requires consent.", clause_path=["2"])]
    retriever = HierarchicalRetriever(
        method="bm25_hierarchical",
        index=BM25SentenceIndex(chunks_from_sentences(spans)),
        section_index=build_section_index(spans),
        leaf_k=5,
        top_sections=1,
    )

    assert retriever.search("assignment", document_row_id=999, top_k=10) == []


def test_format_hierarchical_context_groups_by_section() -> None:
    spans = [
        span(
            1,
            0,
            "Confidential information means data.",
            clause_path=["3.1", "Confidentiality"],
            section_number="3.1",
            section_title="Confidentiality",
        ),
        span(
            1,
            1,
            "It excludes public data.",
            clause_path=["3.1", "Confidentiality"],
            section_number="3.1",
            section_title="Confidentiality",
        ),
    ]
    chunks = chunks_from_sentences(spans)

    context = format_hierarchical_context(
        [
            type("R", (), {"chunk": chunks[0], "score": 1.0, "rank": 1})(),
            type("R", (), {"chunk": chunks[1], "score": 0.5, "rank": 2})(),
        ]
    )

    assert context.count("[SECTION 3.1 - Confidentiality]") == 1
    assert "Confidential information means data." in context
    assert "It excludes public data." in context


def test_format_hierarchical_context_empty() -> None:
    assert format_hierarchical_context([]) == ""


def test_coverage_works_on_hierarchical_results() -> None:
    spans = [
        span(1, 0, "Assignment requires consent.", clause_path=["2"]),
        span(1, 1, "Consent must be written.", clause_path=["2"]),
    ]
    retriever = HierarchicalRetriever(
        method="bm25_hierarchical",
        index=BM25SentenceIndex(chunks_from_sentences(spans)),
        section_index=build_section_index(spans),
        leaf_k=1,
        top_sections=1,
    )

    results = retriever.search("assignment", document_row_id=1, top_k=5)
    coverage = coverage_by_top_chunks(["1:s:0", "1:s:1"], results, top_chunks=5)

    assert coverage["all_gold_sentences_covered"]
    assert coverage["gold_sentence_coverage"] == 1.0
