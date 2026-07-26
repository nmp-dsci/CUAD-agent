"""Tests for the single-question variant comparison mode (S6).

All tests use --dry-run and --context-mode raw so no LLM calls or
cached RAG indexes are required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuad_agent.evaluators.langchain_runner import (
    _DEFAULT_SINGLE_Q_MODEL_ID,
    _VARIANT_CTX_LABELS,
    print_variant_table,
    run_single_question_variants,
)
from cuad_agent.rag.query_enrichment import (
    QuestionEnrichment,
    RAG_DEFAULT_TOP_K,
    save_enriched_question_files,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GOVERNING_LAW_IDX = 7  # question_index for "Governing Law" in the CUAD dataset
SAMPLE_CONTRACT_ID = 327  # first contract in seed=42 sample


def _fake_prompt_overrides(category: str = "Governing Law") -> dict[str, str]:
    """Minimal prompt_overrides covering the Governing Law category."""
    return {category: "You are a legal AI reviewing contracts."}


# ---------------------------------------------------------------------------
# RAG_DEFAULT_TOP_K
# ---------------------------------------------------------------------------


def test_rag_default_top_k_value() -> None:
    assert RAG_DEFAULT_TOP_K == 30


# ---------------------------------------------------------------------------
# save_enriched_question_files
# ---------------------------------------------------------------------------


def test_save_enriched_question_files_writes_per_question_json(tmp_path: Path) -> None:
    enrichment = QuestionEnrichment(
        question_index=5,
        category="Governing Law",
        question="What law governs the contract?",
        category_description="Identifies the governing law.",
        enrichment_terms="governed by; laws of; jurisdiction",
        enriched_query="What law governs the contract? Contract words to look for: governed by; laws of",
        provider="offline",
        status="offline_fallback",
        cache_key="abc123",
    )
    save_enriched_question_files({5: enrichment}, tmp_path, provider="offline")

    expected = tmp_path / "enriched_questions" / "offline" / "q05_governing-law.json"
    assert expected.exists(), f"Expected file not found: {expected}"
    data = json.loads(expected.read_text(encoding="utf-8"))
    assert data["question_index"] == 5
    assert data["category"] == "Governing Law"
    assert data["enrichment_terms"] == "governed by; laws of; jurisdiction"


def test_save_enriched_question_files_multiple_questions(tmp_path: Path) -> None:
    enrichments = {
        0: QuestionEnrichment(
            question_index=0,
            category="Anti-Assignment",
            question="Is assignment restricted?",
            category_description="Assignment clause.",
            enrichment_terms="assign; transfer",
            enriched_query="Is assignment restricted? Contract words to look for: assign",
            provider="offline",
            status="offline_fallback",
            cache_key="key0",
        ),
        5: QuestionEnrichment(
            question_index=5,
            category="Governing Law",
            question="What law governs?",
            category_description="Governing law clause.",
            enrichment_terms="governed by; laws of",
            enriched_query="What law governs? Contract words to look for: governed by",
            provider="offline",
            status="offline_fallback",
            cache_key="key5",
        ),
    }
    save_enriched_question_files(enrichments, tmp_path, provider="offline")

    base = tmp_path / "enriched_questions" / "offline"
    assert (base / "q00_anti-assignment.json").exists()
    assert (base / "q05_governing-law.json").exists()


# ---------------------------------------------------------------------------
# print_variant_table
# ---------------------------------------------------------------------------


def _sample_variants_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant_name": "raw_q / raw_ctx",
                "question_mode": "raw",
                "context_mode": "raw",
                "retrieval_query": "Governing Law ...",
                "hint_used": "",
                "predicted_answer": "governed by the laws of the State of Delaware",
                "gold_answers": json.dumps(
                    ["governed by the laws of the State of Delaware"]
                ),
                "token_f1": 1.0,
                "correct_at_0_5": True,
                "enrichment_terms": "governed by; laws of",
                "document_row_id": 327,
                "question_index": 5,
                "category": "Governing Law",
            },
            {
                "variant_name": "enriched_v1 / raw_ctx",
                "question_mode": "enriched",
                "context_mode": "raw",
                "retrieval_query": "Governing Law ... governed by; laws of",
                "hint_used": "governed by; laws of",
                "predicted_answer": "Delaware",
                "gold_answers": json.dumps(
                    ["governed by the laws of the State of Delaware"]
                ),
                "token_f1": 0.25,
                "correct_at_0_5": False,
                "enrichment_terms": "governed by; laws of",
                "document_row_id": 327,
                "question_index": 5,
                "category": "Governing Law",
            },
        ]
    )


def test_print_variant_table_does_not_crash(capsys: pytest.CaptureFixture) -> None:
    print_variant_table(_sample_variants_df())
    captured = capsys.readouterr()
    assert "raw_q / raw_ctx" in captured.out
    assert "enriched_v1 / raw_ctx" in captured.out
    assert "Governing Law" in captured.out


def test_print_variant_table_empty_df_does_not_crash(
    capsys: pytest.CaptureFixture,
) -> None:
    print_variant_table(pd.DataFrame())
    captured = capsys.readouterr()
    assert "no variants" in captured.out.lower()


def test_print_variant_table_single_row(capsys: pytest.CaptureFixture) -> None:
    df = _sample_variants_df().iloc[:1].reset_index(drop=True)
    print_variant_table(df)
    captured = capsys.readouterr()
    assert "raw_q / raw_ctx" in captured.out


# ---------------------------------------------------------------------------
# run_single_question_variants — dry-run, context-mode raw
# ---------------------------------------------------------------------------


def test_single_variant_raw_q_raw_ctx_returns_one_row() -> None:
    prompt_overrides = _fake_prompt_overrides("Governing Law")
    df = run_single_question_variants(
        contract_id=SAMPLE_CONTRACT_ID,
        question_index=GOVERNING_LAW_IDX,
        llm=None,
        dry_run=True,
        question_modes=["raw"],
        context_modes=["raw"],
        top_k=30,
        output_dir=Path("outputs"),
        model_id="test-dry",
        prompt_overrides=prompt_overrides,
        query_enrichment_provider="offline",
        query_enrichment_model="deepseek-chat",
        embedding_model="tfidf",
        chunking_version="sentence-v3",
    )
    assert len(df) == 1
    assert df.iloc[0]["variant_name"] == "raw_q / raw_ctx"
    assert df.iloc[0]["question_mode"] == "raw"
    assert df.iloc[0]["context_mode"] == "raw"


def test_compare_variants_raw_context_only_produces_two_rows() -> None:
    """When question modes are [raw, enriched] and context mode is raw → 2 rows."""
    prompt_overrides = _fake_prompt_overrides("Governing Law")
    df = run_single_question_variants(
        contract_id=SAMPLE_CONTRACT_ID,
        question_index=GOVERNING_LAW_IDX,
        llm=None,
        dry_run=True,
        question_modes=["raw", "enriched"],
        context_modes=["raw"],
        top_k=30,
        output_dir=Path("outputs"),
        model_id="test-dry",
        prompt_overrides=prompt_overrides,
        query_enrichment_provider="offline",
        query_enrichment_model="deepseek-chat",
        embedding_model="tfidf",
        chunking_version="sentence-v3",
    )
    assert len(df) == 2
    assert set(df["variant_name"]) == {"raw_q / raw_ctx", "enriched_v1 / raw_ctx"}


def test_compare_all_variants_raw_context_produces_two_rows() -> None:
    """Simulates --compare-variants with context_mode fixed to raw (no cache needed)."""
    prompt_overrides = _fake_prompt_overrides("Governing Law")
    df = run_single_question_variants(
        contract_id=SAMPLE_CONTRACT_ID,
        question_index=GOVERNING_LAW_IDX,
        llm=None,
        dry_run=True,
        question_modes=["raw", "enriched"],
        context_modes=["raw"],
        top_k=30,
        output_dir=Path("outputs"),
        model_id="test-dry",
        prompt_overrides=prompt_overrides,
        query_enrichment_provider="offline",
        query_enrichment_model="deepseek-chat",
        embedding_model="tfidf",
        chunking_version="sentence-v3",
    )
    assert len(df) == 2
    assert "variant_name" in df.columns
    assert "token_f1" in df.columns
    assert "correct_at_0_5" in df.columns
    assert "predicted_answer" in df.columns
    assert "gold_answers" in df.columns
    assert "enrichment_terms" in df.columns
    assert "document_row_id" in df.columns
    assert "question_index" in df.columns
    assert "category" in df.columns


def test_variant_name_construction_covers_all_labels() -> None:
    assert _VARIANT_CTX_LABELS == {
        "raw": "raw_ctx",
        "rag-dense": "rag_dense",
        "rag-hybrid": "rag_hybrid",
        "rag-hierarchical-bm25": "rag_hier_bm25",
        "rag-hierarchical-dense": "rag_hier_dense",
    }


def test_run_single_q_missing_category_in_prompt_overrides_raises() -> None:
    with pytest.raises(ValueError, match="not found in prompt_overrides"):
        run_single_question_variants(
            contract_id=SAMPLE_CONTRACT_ID,
            question_index=GOVERNING_LAW_IDX,
            llm=None,
            dry_run=True,
            question_modes=["raw"],
            context_modes=["raw"],
            top_k=30,
            output_dir=Path("outputs"),
            model_id="test",
            prompt_overrides={"Anti-Assignment": "some prompt"},  # wrong category
            query_enrichment_provider="offline",
            query_enrichment_model="deepseek-chat",
            embedding_model="tfidf",
            chunking_version="sentence-v3",
        )


def test_dry_run_score_is_1_for_gold_answer() -> None:
    """In dry-run mode the model echoes the gold answer, so F1 should be 1.0."""
    prompt_overrides = _fake_prompt_overrides("Governing Law")
    df = run_single_question_variants(
        contract_id=SAMPLE_CONTRACT_ID,
        question_index=GOVERNING_LAW_IDX,
        llm=None,
        dry_run=True,
        question_modes=["raw"],
        context_modes=["raw"],
        top_k=30,
        output_dir=Path("outputs"),
        model_id="test",
        prompt_overrides=prompt_overrides,
        query_enrichment_provider="offline",
        query_enrichment_model="deepseek-chat",
        embedding_model="tfidf",
        chunking_version="sentence-v3",
    )
    assert float(df.iloc[0]["token_f1"]) == 1.0
    assert bool(df.iloc[0]["correct_at_0_5"]) is True


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------


def test_default_single_q_model_id() -> None:
    assert _DEFAULT_SINGLE_Q_MODEL_ID == "s6"
