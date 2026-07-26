"""Tests for src/cuad_agent/autoresearch/ — all use dry_run=True or in-memory fixtures.

No API keys required; no real LLM calls are made.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from glob import glob
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so imports resolve correctly when pytest is
# invoked from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cuad_agent.autoresearch.report import write_iter_report, write_progress_report
from cuad_agent.autoresearch.results import (
    SynthesisResult,
    TriageDiagnosis,
    write_prompt_module,
    write_tsv_row,
)
from cuad_agent.autoresearch.synthesis import synthesise
from cuad_agent.autoresearch.triage import triage
from cuad_agent.data.dataset import load_datasets

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_csv(tmp_path: Path) -> Path:
    """Create a minimal eval-results CSV with one wrong-answer row."""
    csv_path = tmp_path / "eval_results.csv"
    pd.DataFrame(
        [
            {
                "model_id": "eval-raw",
                "row_id": "327:7",
                "document_row_id": 327,
                "title": "Test",
                "question_index": 7,
                "category": "Governing Law",
                "category_description": "Governing law clause.",
                "answer_format": "State/Country",
                "question": "What is the governing law?",
                "gold_answers": '["New York"]',
                "golden_answer": '["New York"]',
                "predicted_answer": "N/A",
                "predicted_marked_impossible": False,
                "gold_marked_impossible": False,
                "token_f1": 0.0,
                "correct_at_0_5": 0.0,
                "is_impossible": False,
                "answers_len": 1,
            }
        ]
    ).to_csv(csv_path, index=False)
    return csv_path


_PROMPTS_FILE = _PROJECT_ROOT / "prompts" / "system_prompts_v2.py"


# ---------------------------------------------------------------------------
# 1. test_triage_dry_run
# ---------------------------------------------------------------------------


def test_triage_dry_run(tmp_path: Path) -> None:
    csv_path = _make_stub_csv(tmp_path)
    output_path = tmp_path / "triage_outputs.jsonl"

    result = triage(
        eval_results_path=csv_path,
        question_index=7,
        category="Governing Law",
        prompts_file=_PROMPTS_FILE,
        model_id="eval-raw",
        model="deepseek/deepseek-v4-flash",
        output_path=output_path,
        dry_run=True,
    )

    assert isinstance(result, list), "triage() should return a list"
    assert len(result) > 0, "expected at least one TriageDiagnosis stub"
    for item in result:
        assert isinstance(item, TriageDiagnosis)
        # All required fields must be present (Pydantic would raise if missing)
        assert item.contract_id is not None
        assert item.golden_answer_location != ""
        assert item.failure_reason != ""
        assert item.proposed_rule != ""
        assert item.confidence in {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# 2. test_triage_contract_text
# ---------------------------------------------------------------------------


def test_triage_contract_text() -> None:
    """load_datasets()["contracts"] row 0 must have non-empty context."""
    datasets = load_datasets()
    context = str(datasets["contracts"].iloc[0]["context"])
    assert len(context) > 0, "contract context for row 0 should be non-empty"


# ---------------------------------------------------------------------------
# 3. test_synthesis_dry_run
# ---------------------------------------------------------------------------


def test_synthesis_dry_run() -> None:
    result = synthesise(
        category="Governing Law",
        current_prompt="Extract the governing law clause.",
        diagnoses=[
            TriageDiagnosis(
                contract_id=327,
                golden_answer_location="This Agreement shall be governed by New York law.",
                failure_reason="model returned N/A",
                proposed_rule="look for jurisdiction keywords",
                confidence="low",
            )
        ],
        history=[],
        model_id="eval-raw_ar_r1_i1",
        model="deepseek/deepseek-v4-pro",
        dry_run=True,
    )

    assert isinstance(result, SynthesisResult)
    assert len(result.prompt_text) > 0, "prompt_text must be non-empty"
    assert len(result.notes) > 0, "notes must be non-empty"


# ---------------------------------------------------------------------------
# 4. test_question_index_resolution
# ---------------------------------------------------------------------------


def test_question_index_resolution() -> None:
    """question_index 7 must resolve to 'Governing Law'."""
    datasets = load_datasets()
    questions = datasets["questions"]
    matching = questions[questions["question_index"] == 7]
    assert not matching.empty, "question_index 7 should exist in the dataset"
    category = str(matching.iloc[0]["category"])
    assert category == "Governing Law"


# ---------------------------------------------------------------------------
# 5. test_results_tsv_write
# ---------------------------------------------------------------------------


def test_results_tsv_write(tmp_path: Path) -> None:
    tsv_path = tmp_path / "results.tsv"
    row = {
        "iter": 1,
        "question_index": 7,
        "category": "Governing Law",
        "model_id": "eval-raw",
        "prompt_file": "prompts/system_prompts_v2.py",
        "correct_at_0_5": 0.8,
        "n_wrong": 2,
        "n_diagnosed": 2,
        "status": "keep",
        "notes": "+rule: jurisdiction test",
    }
    write_tsv_row(tsv_path, row)

    assert tsv_path.exists(), "TSV file should be created"
    lines = tsv_path.read_text().splitlines()
    assert len(lines) >= 2, "TSV should have a header row and at least one data row"
    # Header row
    assert "category" in lines[0], "header should contain 'category'"
    # Data row contains our category
    assert "Governing Law" in lines[1]


# ---------------------------------------------------------------------------
# 6. test_loop_dry_run
# ---------------------------------------------------------------------------


def test_loop_dry_run() -> None:
    """Run the full autoresearch loop as a subprocess and verify exit code + outputs."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "autoresearch.py",
            "--model-id",
            "eval-raw",
            "--prompts-file",
            "prompts/system_prompts_v2.py",
            "--question-index",
            "7",
            "--dry-run",
            "--max-iterations",
            "2",
            "--sample-size",
            "5",
        ],
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"autoresearch.py exited with code {result.returncode}\n"
        f"stdout: {result.stdout[-2000:]}\n"
        f"stderr: {result.stderr[-2000:]}"
    )

    # Verify output files exist using glob
    tsv_files = glob(
        str(_PROJECT_ROOT / "outputs" / "autoresearch" / "q07" / "*" / "results.tsv")
    )
    assert len(tsv_files) > 0, (
        "expected outputs/autoresearch/*/q07/results.tsv to exist"
    )

    html_files = glob(
        str(_PROJECT_ROOT / "dashboards" / "autoresearch-eval-raw-q07.html")
    )
    assert len(html_files) > 0, (
        "expected dashboards/autoresearch-eval-raw-q07.html to exist"
    )


# ---------------------------------------------------------------------------
# 7. test_prompt_module_write
# ---------------------------------------------------------------------------


def test_prompt_module_write(tmp_path: Path) -> None:
    """write_prompt_module should produce a valid importable Python file."""
    out_path = tmp_path / "candidate.py"
    write_prompt_module(out_path, "Governing Law", "test prompt text")

    assert out_path.exists(), "prompt module file should be created"
    source = out_path.read_text(encoding="utf-8")

    # Content checks
    assert "Governing Law" in source
    assert "test prompt text" in source

    # Must be valid Python
    compile(source, str(out_path), "exec")

    # Import and verify CATEGORY_SYSTEM_PROMPTS
    spec = importlib.util.spec_from_file_location("_test_candidate", out_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    prompts = getattr(module, "CATEGORY_SYSTEM_PROMPTS", None)
    assert prompts is not None, "CATEGORY_SYSTEM_PROMPTS not found in module"
    assert prompts["Governing Law"] == "test prompt text"


# ---------------------------------------------------------------------------
# 8. test_report_html_write (iter report)
# ---------------------------------------------------------------------------


def test_report_html_write(tmp_path: Path) -> None:
    """write_iter_report should produce an HTML file with expected content."""
    _STUB_COLS = [
        "document_row_id",
        "contract_title",
        "predicted_answer",
        "golden_answer",
        "correct_at_0_5",
    ]
    current_df = pd.DataFrame(
        [
            {
                "document_row_id": 327,
                "contract_title": "Test Contract",
                "predicted_answer": "N/A",
                "golden_answer": '["New York"]',
                "correct_at_0_5": 0.0,
            }
        ],
        columns=_STUB_COLS,
    )
    candidate_df = pd.DataFrame(
        [
            {
                "document_row_id": 327,
                "contract_title": "Test Contract",
                "predicted_answer": "New York",
                "golden_answer": '["New York"]',
                "correct_at_0_5": 1.0,
            }
        ],
        columns=_STUB_COLS,
    )

    output_path = tmp_path / "iter_1" / "report.html"
    write_iter_report(
        iter_n=1,
        category="Governing Law",
        question_index=7,
        date_str="20260606",
        status="keep",
        notes="test prompt",
        current_accuracy=0.4,
        candidate_accuracy=0.8,
        current_eval_df=current_df,
        candidate_eval_df=candidate_df,
        triage_diagnoses=[],
        candidate_prompt_text="test prompt text",
        output_path=output_path,
    )

    assert output_path.exists(), "iter report HTML should be created"
    html = output_path.read_text(encoding="utf-8")
    assert "Governing Law" in html
    assert "0.4" in html
    assert "test prompt" in html


# ---------------------------------------------------------------------------
# 9. test_progress_html_write
# ---------------------------------------------------------------------------


def test_progress_html_write(tmp_path: Path) -> None:
    """write_progress_report should produce an HTML file with expected content."""
    rows = [
        {
            "iter": 0,
            "status": "baseline",
            "correct_at_0_5": 0.5,
            "notes": "baseline",
            "prompt_file": "prompts/system_prompts_v2.py",
        },
        {
            "iter": 1,
            "status": "keep",
            "correct_at_0_5": 0.7,
            "notes": "+rule: jurisdiction test",
            "prompt_file": "prompts/autoresearch/20260606/q07/v2_r1_i1.py",
        },
    ]

    output_path = tmp_path / "progress.html"
    write_progress_report(
        rows=rows,
        category="Governing Law",
        question_index=7,
        output_path=output_path,
    )

    assert output_path.exists(), "progress HTML should be created"
    html = output_path.read_text(encoding="utf-8")
    assert "Governing Law" in html
    assert "Kept Improvements" in html
    assert "+rule: jurisdiction test" in html
