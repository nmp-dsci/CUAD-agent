from __future__ import annotations

from cuad_agent.rag.chunks import chunks_from_sentences
from cuad_agent.rag.query_enrichment import hybrid_fuse_results
from cuad_agent.rag.retrievers import build_retriever
from cuad_agent.rag.sentences import build_sentence_spans


def test_bm25_sentence_retrieval_stays_within_contract() -> None:
    spans = [
        *build_sentence_spans(1, "Assignment requires consent. Notices are written."),
        *build_sentence_spans(2, "Assignment is freely permitted."),
    ]
    chunks = chunks_from_sentences(spans)
    retriever = build_retriever("bm25_sentence", chunks, embedding_model="tfidf")

    results = retriever.search("assignment consent", document_row_id=1, top_k=5)

    assert results
    assert all(result.chunk.document_row_id == 1 for result in results)
    assert results[0].chunk.text == "Assignment requires consent."


def test_hybrid_fuse_combines_bm25_and_dense_rankings() -> None:
    spans = build_sentence_spans(
        1,
        "Assignment requires consent. Notices are written. Audit records yearly.",
    )
    chunks = chunks_from_sentences(spans)
    bm25 = build_retriever("bm25_sentence", chunks, embedding_model="tfidf")
    dense = build_retriever("dense_sentence", chunks, embedding_model="tfidf")

    bm25_results = bm25.search("assignment consent", document_row_id=1, top_k=3)
    dense_results = dense.search("audit records", document_row_id=1, top_k=3)
    hybrid = hybrid_fuse_results(dense_results, bm25_results, top_k=3)

    assert len(hybrid) == 3
    assert {result.chunk.chunk_id for result in hybrid} == {
        result.chunk.chunk_id for result in bm25_results
    }
    assert [result.rank for result in hybrid] == [1, 2, 3]
