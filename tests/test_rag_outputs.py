from __future__ import annotations

from pathlib import Path

from cuad_agent.rag.experiments import run_rag_eval


def test_rag_preflight_writes_outputs_and_frontend(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"

    summary = run_rag_eval(
        run_id="rag-test",
        sample_size=1,
        seed=42,
        retrievers=["bm25_sentence"],
        top_k=3,
        output_dir=output_dir,
        preflight_golden_sentences_only=True,
    )

    assert summary["run_id"] == "rag-test"
    assert summary["contract_scope"] == "all"
    assert summary["chunk_all_contracts"] is True
    assert summary["chunked_contract_count"] == 510
    assert (output_dir / "rag-test/rag/golden_sentence_coverage.csv").exists()
    assert (output_dir / "rag-test/rag/rag_summary.json").exists()
    assert (output_dir / "rag_cache/chunking/sentence-v3/sentence_spans.jsonl").exists()
    assert (
        output_dir
        / "rag_cache/chunking/sentence-v3/encodings/tfidf/embedding_manifest.json"
    ).exists()
    assert (
        output_dir / "rag_cache/chunking/sentence-v3/encodings/tfidf/embeddings.npz"
    ).exists()
    html_path = tmp_path / "frontend/rag_pipeline_eval.html"
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Summary" in html
    assert "Chunking" in html
    assert "Chunked Golden Answer" in html
    assert "Enriched question" in html
    assert "Enriched Query" in html
    assert "Retrieval Technique Comparison" in html
    assert "query-enrichment-technique-table" in html
    assert "v3-review-detail" in html
    assert "query-enrichment-summary-table" in html
    assert "Chunking rag" not in html
    assert "rag-review-detail" not in html


def test_rag_can_chunk_eval_contracts_only_for_debug_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"

    summary = run_rag_eval(
        run_id="rag-all-cache-test",
        sample_size=1,
        seed=42,
        retrievers=["bm25_sentence"],
        top_k=3,
        output_dir=output_dir,
        preflight_golden_sentences_only=True,
        contract_scope="eval-set",
    )

    assert summary["contract_scope"] == "eval-set"
    assert summary["chunk_all_contracts"] is False
    assert summary["chunked_contract_count"] == 1
    assert summary["sentence_count"] > 1
    assert summary["average_sentences_per_contract"] > 1
