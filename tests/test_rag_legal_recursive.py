from __future__ import annotations

from cuad_agent.rag.legal_recursive import (
    LegalRecursiveConfig,
    build_legal_recursive_chunks_for_contract,
)
from cuad_agent.rag.retrievers import build_retriever
from cuad_agent.rag.sentences import build_sentence_spans


def test_legal_recursive_chunks_map_to_sentence_ids() -> None:
    text = (
        "SECTION 1 Assignment\n"
        "(a) Assignment requires prior written consent. Notices are written.\n"
        "(b) Confidentiality survives termination."
    )
    spans = build_sentence_spans(7, text)

    chunks = build_legal_recursive_chunks_for_contract(
        document_row_id=7,
        text=text,
        sentence_spans=spans,
        config=LegalRecursiveConfig(chunk_size=80, chunk_overlap=0),
    )

    assert chunks
    assert all(chunk.chunk_type == "legal_recursive" for chunk in chunks)
    assert any(chunk.sentence_ids for chunk in chunks)
    assert set().union(*(set(chunk.sentence_ids) for chunk in chunks)) == {
        span.sentence_id for span in spans
    }


def test_bm25_legal_recursive_retrieval_stays_within_contract() -> None:
    spans_1 = build_sentence_spans(
        1,
        "SECTION 1 Assignment\n(a) Assignment requires consent. Notices are written.",
    )
    spans_2 = build_sentence_spans(2, "Assignment is freely permitted.")
    chunks = [
        *build_legal_recursive_chunks_for_contract(
            document_row_id=1,
            text="SECTION 1 Assignment\n(a) Assignment requires consent. Notices are written.",
            sentence_spans=spans_1,
            config=LegalRecursiveConfig(chunk_size=120, chunk_overlap=0),
        ),
        *build_legal_recursive_chunks_for_contract(
            document_row_id=2,
            text="Assignment is freely permitted.",
            sentence_spans=spans_2,
            config=LegalRecursiveConfig(chunk_size=120, chunk_overlap=0),
        ),
    ]
    retriever = build_retriever("bm25_legal_recursive", chunks, embedding_model="tfidf")

    results = retriever.search("assignment consent", document_row_id=1, top_k=5)

    assert results
    assert all(result.chunk.document_row_id == 1 for result in results)
    assert any("consent" in result.chunk.text.lower() for result in results)
