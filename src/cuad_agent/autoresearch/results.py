"""Shared Pydantic models, TSV logger, and prompt writer for autoresearch."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

__all__ = [
    "SynthesisResult",
    "TriageDiagnosis",
    "write_prompt_module",
    "write_tsv_row",
]

# TSV columns in order
_TSV_COLUMNS = [
    "iter",
    "question_index",
    "category",
    "model_id",
    "prompt_file",
    "correct_at_0_5",
    "n_wrong",
    "n_diagnosed",
    "status",
    "notes",
]


class TriageDiagnosis(BaseModel):
    contract_id: int  # = document_row_id
    golden_answer_location: str  # verbatim sentences surrounding golden answer
    failure_reason: str  # why the system prompt led to the wrong answer
    proposed_rule: str  # concrete rule referencing actual clause structure
    confidence: Literal["high", "medium", "low"]


class SynthesisResult(BaseModel):
    prompt_text: str  # full new system prompt text
    notes: str  # one-line description: what changed and why


def write_tsv_row(path: Path, row: dict) -> None:
    """Append one tab-separated row; creates file with header if it doesn't exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "\t".join(_TSV_COLUMNS) + "\n"
        path.write_text(header)
    line = "\t".join(str(row.get(col, "")) for col in _TSV_COLUMNS) + "\n"
    with path.open("a") as fh:
        fh.write(line)


def write_prompt_module(path: Path, category: str, prompt_text: str) -> None:
    """Write a candidate.py with CATEGORY_SYSTEM_PROMPTS = {category: prompt_text}."""
    content = (
        f'"""Autoresearch candidate — category: {category}."""\n\n'
        f"CATEGORY_SYSTEM_PROMPTS = {{\n"
        f'    "{category}": """{prompt_text}""",\n'
        f"}}\n"
    )
    path.write_text(content)
