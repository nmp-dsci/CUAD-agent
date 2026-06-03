"""Filesystem and identifier helpers."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from cuad_agent.constants import OUTPUT_STEM

__all__ = [
    "class_name_part",
    "output_paths",
    "prompt_name_part",
    "resolve_model_id",
    "slugify_model_id",
]


def class_name_part(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", value).title().replace(" ", "")
    return cleaned or "Category"


def prompt_name_part(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").upper()
    return cleaned or "CATEGORY"


def slugify_model_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value).strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-._")
    return cleaned or "model"


def resolve_model_id(args: argparse.Namespace) -> str:
    if getattr(args, "model_id", None):
        return slugify_model_id(args.model_id)
    return slugify_model_id(args.model)


def output_paths(
    output_dir: Path, model_id: str, html_output: Path | None
) -> dict[str, Path]:
    model_dir = output_dir / model_id
    frontend_dir = output_dir.parent / "dashboards"
    resolved_html_output = html_output
    if resolved_html_output is not None and not resolved_html_output.is_absolute():
        if resolved_html_output.parent == Path("."):
            resolved_html_output = frontend_dir / resolved_html_output
    return {
        "model_dir": model_dir,
        "results": model_dir / f"{OUTPUT_STEM}_results.csv",
        "summary": model_dir / f"{OUTPUT_STEM}_summary.json",
        "html": resolved_html_output or frontend_dir / f"evaluation_{model_id}.html",
        "system_prompts": model_dir / "system_prompts.py",
    }
