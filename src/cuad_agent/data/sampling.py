"""Deterministic evaluation set selection helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cuad_agent.data.dataset import load_datasets


QUESTION_COUNT = 41
EVAL_QUESTION_INDICES = tuple(range(QUESTION_COUNT))


@dataclass(frozen=True)
class EvaluationSelection:
    selected_ids: list[int]
    contract_lookup: dict[int, dict[str, Any]]
    eval_rows: pd.DataFrame


def evaluation_row_id(document_row_id: int, question_index: int) -> str:
    return f"{int(document_row_id)}:{int(question_index)}"


def load_eval_split_ids(eval_split: str | None) -> set[str] | None:
    if not eval_split:
        return None
    path_text, sep, split_name = eval_split.partition(":")
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Eval split file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if sep:
        values = data.get(split_name)
        if values is None:
            raise KeyError(f"Split {split_name!r} not found in {path}")
    else:
        values = data
    return {str(value) for value in values}


def filter_eval_rows_by_split(
    eval_rows: pd.DataFrame,
    split_row_ids: set[str] | None,
) -> pd.DataFrame:
    if not split_row_ids:
        rows = eval_rows.copy()
    else:
        rows = eval_rows.copy()
        rows["row_id"] = [
            evaluation_row_id(document_row_id, question_index)
            for document_row_id, question_index in zip(
                rows["document_row_id"], rows["question_index"]
            )
        ]
        rows = rows[rows["row_id"].isin(split_row_ids)]
    if "row_id" not in rows.columns:
        rows["row_id"] = [
            evaluation_row_id(document_row_id, question_index)
            for document_row_id, question_index in zip(
                rows["document_row_id"], rows["question_index"]
            )
        ]
    return rows.sort_values(["document_row_id", "question_index"]).reset_index(drop=True)


def all_contract_lookup() -> dict[int, dict[str, Any]]:
    contracts = load_datasets()["contracts"]
    return {
        int(key): value
        for key, value in contracts.set_index("document_row_id").to_dict("index").items()
    }


def select_evaluation_set(
    *,
    sample_size: int = 50,
    seed: int = 42,
    contract_ids: list[int] | None = None,
    question_indices: list[int] | None = None,
    eval_split: str | None = None,
) -> EvaluationSelection:
    datasets = load_datasets()
    contracts = datasets["contracts"]
    questions = datasets["questions"]
    full_contract_lookup = {
        int(key): value
        for key, value in contracts.set_index("document_row_id").to_dict("index").items()
    }

    if contract_ids:
        selected_ids = [int(value) for value in contract_ids]
    else:
        candidate_ids = sorted(contracts["document_row_id"].astype(int).tolist())
        if sample_size > len(candidate_ids):
            raise ValueError(
                f"sample_size={sample_size} exceeds available contracts={len(candidate_ids)}"
            )
        selected_ids = random.Random(seed).sample(candidate_ids, k=sample_size)

    selected_set = set(selected_ids)
    eval_rows = questions[
        questions["document_row_id"].isin(selected_set)
        & questions["question_index"].isin(EVAL_QUESTION_INDICES)
    ].copy()
    if question_indices:
        eval_rows = eval_rows[eval_rows["question_index"].isin(question_indices)].copy()

    split_row_ids = load_eval_split_ids(eval_split)
    eval_rows = filter_eval_rows_by_split(eval_rows, split_row_ids)

    selected_contract_lookup = {
        document_row_id: full_contract_lookup[document_row_id]
        for document_row_id in selected_ids
        if document_row_id in full_contract_lookup
    }
    return EvaluationSelection(
        selected_ids=selected_ids,
        contract_lookup=selected_contract_lookup,
        eval_rows=eval_rows,
    )


def build_eval_sample(
    sample_size: int = 50,
    seed: int = 42,
) -> tuple[list[int], dict[int, dict[str, Any]], pd.DataFrame]:
    selection = select_evaluation_set(sample_size=sample_size, seed=seed)
    expected_rows = sample_size * QUESTION_COUNT
    if selection.eval_rows.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} eval rows, found {selection.eval_rows.shape[0]}"
        )
    return selection.selected_ids, selection.contract_lookup, selection.eval_rows

__all__ = [
    "EVAL_QUESTION_INDICES",
    "EvaluationSelection",
    "all_contract_lookup",
    "build_eval_sample",
    "evaluation_row_id",
    "filter_eval_rows_by_split",
    "load_eval_split_ids",
    "select_evaluation_set",
]
