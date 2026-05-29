from __future__ import annotations

from pathlib import Path

import pandas as pd

from cuad_agent.dashboards.model_comparison import (
    build_model_comparison_data,
    write_model_comparison_html,
)
from cuad_agent.evaluators.dspy_runner import detection_metrics


def _frame() -> pd.DataFrame:
    # 2 gold no-answer rows, 2 gold answer rows.
    # Model correctly flags 1 of 2 no-answer rows, and correctly finds 2 of 2 answers.
    return pd.DataFrame(
        [
            # gold no-answer, predicted no-answer (correct)
            {
                "question_index": 0,
                "category": "Governing Law",
                "token_f1": 1.0,
                "correct_at_0_5": True,
                "gold_marked_impossible": True,
                "predicted_marked_impossible": True,
                "predicted_answer": "NO_ANSWER",
            },
            # gold no-answer, predicted an answer (miss)
            {
                "question_index": 0,
                "category": "Governing Law",
                "token_f1": 0.0,
                "correct_at_0_5": False,
                "gold_marked_impossible": True,
                "predicted_marked_impossible": False,
                "predicted_answer": "New York",
            },
            # gold answer, predicted an answer (correct presence)
            {
                "question_index": 1,
                "category": "Anti-Assignment",
                "token_f1": 0.8,
                "correct_at_0_5": True,
                "gold_marked_impossible": False,
                "predicted_marked_impossible": False,
                "predicted_answer": "no assignment",
            },
            # gold answer, predicted an answer (correct presence)
            {
                "question_index": 1,
                "category": "Anti-Assignment",
                "token_f1": 0.4,
                "correct_at_0_5": False,
                "gold_marked_impossible": False,
                "predicted_marked_impossible": False,
                "predicted_answer": "consent required",
            },
        ]
    )


def test_detection_metrics_split_by_gold_class() -> None:
    metrics = detection_metrics(_frame())
    assert metrics["gold_no_answer_count"] == 2
    assert metrics["gold_answer_count"] == 2
    assert metrics["no_answer_detection_accuracy"] == 50.0
    assert metrics["answer_detection_accuracy"] == 100.0
    assert metrics["detection_accuracy"] == 75.0


def test_detection_metrics_uses_answer_text_when_flag_unset() -> None:
    # Flag says answerable but text is a no-answer marker -> counts as no-answer.
    frame = pd.DataFrame(
        [
            {
                "question_index": 0,
                "category": "Governing Law",
                "token_f1": 1.0,
                "correct_at_0_5": True,
                "gold_marked_impossible": True,
                "predicted_marked_impossible": False,
                "predicted_answer": "none",
            },
        ]
    )
    metrics = detection_metrics(frame)
    assert metrics["no_answer_detection_accuracy"] == 100.0


def test_build_and_write_comparison_html(tmp_path: Path) -> None:
    models = [
        {
            "label": "raw",
            "model_id": "eval-raw",
            "context_mode": "raw",
            "summary": {},
            "results": _frame(),
        },
        {
            "label": "rag-dense",
            "model_id": "eval-rag-dense",
            "context_mode": "rag-dense",
            "summary": {},
            "results": _frame(),
        },
    ]
    page_data = build_model_comparison_data(models)
    assert len(page_data["models"]) == 2
    assert {c["question_index"] for c in page_data["categories"]} == {0, 1}
    assert page_data["models"][0]["no_answer_detection_accuracy"] == 50.0

    out = tmp_path / "dashboards" / "eval_comparison_eval.html"
    write_model_comparison_html(models, out)
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "No-answer acc" in html
    assert "Answer acc" in html
    assert "evaluation_eval-raw.html" in html
