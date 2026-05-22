"""Static evaluation dashboard renderer."""

from __future__ import annotations

from cuad_agent.evaluators.dspy_runner import (
    build_evaluation_page_data,
    parse_gold_answers,
    render_evaluation_html,
    write_evaluation_html,
)

__all__ = [
    "build_evaluation_page_data",
    "parse_gold_answers",
    "render_evaluation_html",
    "write_evaluation_html",
]
