#!/usr/bin/env python3
"""Explore CUADv1 as contract-level and question-level pandas datasets."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_SOURCE = ("cuadv1", "data/CUADv1.json", ("CUADv1.json",))
CATEGORY_SOURCE = (
    "category_descriptions",
    "data/category_descriptions.csv",
    ("category_descriptions.csv",),
)
CATEGORY_COLUMN = "Category (incl. context and answer)"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(primary: str, aliases: tuple[str, ...]) -> Path | None:
    for name in (primary, *aliases):
        for path in (Path(name), SCRIPT_DIR / name):
            if path.exists():
                return path
    return None


def strip_prefix(value: Any, prefix: str) -> Any:
    if not isinstance(value, str):
        return value
    return value.removeprefix(prefix).strip()


def normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).casefold()


def extract_question_category(question: Any) -> str:
    if not isinstance(question, str):
        return ""
    match = re.search(r'related to "([^"]+)"', question)
    return match.group(1) if match else ""


def get_records(data: Any) -> list[Any]:
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def make_contracts_df(records: list[Any]) -> pd.DataFrame:
    df = pd.DataFrame([record for record in records if isinstance(record, dict)])
    df.insert(0, "document_row_id", range(len(df)))
    if "title" in df.columns:
        df.insert(1, "document_id", df["title"])
    #
    if "paragraphs" not in df.columns:
        return df
    #
    df["paragraphs_len"] = df["paragraphs"].apply(
        lambda paragraphs: len(paragraphs) if isinstance(paragraphs, list) else 0
    )
    first_paragraph = df["paragraphs"].apply(
        lambda paragraphs: paragraphs[0]
        if isinstance(paragraphs, list)
        and paragraphs
        and isinstance(paragraphs[0], dict)
        else {}
    )
    paragraph_fields = pd.json_normalize(first_paragraph).add_prefix("paragraphs.")
    df = pd.concat([df, paragraph_fields], axis=1)
    # Convenience aliases for interactive exploration, e.g. contracts.context.
    if "paragraphs.context" in df.columns:
        df["context"] = df["paragraphs.context"]
    if "paragraphs.qas" in df.columns:
        df["qas"] = df["paragraphs.qas"]
    return df


def make_paragraphs_df(contracts: pd.DataFrame) -> pd.DataFrame:
    if "paragraphs" not in contracts.columns:
        return pd.DataFrame()
    #
    paragraph_rows: list[dict[str, Any]] = []
    for _, row in contracts.iterrows():
        paragraphs = row["paragraphs"]
        if not isinstance(paragraphs, list):
            continue
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue
            paragraph_rows.append(
                {
                    "document_row_id": row.get("document_row_id"),
                    "document_id": row.get("document_id"),
                    "title": row.get("title"),
                    "paragraph_index": paragraph_index,
                    **paragraph,
                }
            )
    return pd.DataFrame(paragraph_rows)


def make_questions_df(paragraphs: pd.DataFrame) -> pd.DataFrame:
    if "qas" not in paragraphs.columns:
        return pd.DataFrame()
    #
    question_rows: list[dict[str, Any]] = []
    for _, row in paragraphs.iterrows():
        qas = row["qas"]
        if not isinstance(qas, list):
            continue
        for question_index, question in enumerate(qas):
            if not isinstance(question, dict):
                continue
            question_rows.append(
                {
                    "document_row_id": row.get("document_row_id"),
                    "document_id": row.get("document_id"),
                    "title": row.get("title"),
                    "paragraph_index": row.get("paragraph_index"),
                    "question_index": question_index,
                    **question,
                }
            )
    #
    questions = pd.DataFrame(question_rows)
    if "answers" in questions.columns:
        questions["answers_len"] = questions["answers"].apply(
            lambda answers: len(answers) if isinstance(answers, list) else 0
        )
    return questions


def load_category_descriptions() -> pd.DataFrame:
    datasource, primary, aliases = CATEGORY_SOURCE
    path = resolve_path(primary, aliases)
    if path is None:
        names = ", ".join((primary, *aliases))
        raise FileNotFoundError(f"Missing file for {datasource}: none of {names} found")

    categories = pd.read_csv(path, encoding="utf-8-sig")
    categories = categories.rename(
        columns={
            CATEGORY_COLUMN: "category_source",
            "Description": "category_description_source",
            "Answer Format": "answer_format_source",
            "Group": "category_group_source",
        }
    )
    categories.insert(0, "category_index", range(len(categories)))
    categories["category"] = categories["category_source"].apply(
        lambda value: strip_prefix(value, "Category:")
    )
    categories["category_description"] = categories[
        "category_description_source"
    ].apply(lambda value: strip_prefix(value, "Description:"))
    categories["answer_format"] = categories["answer_format_source"].apply(
        lambda value: strip_prefix(value, "Answer Format:")
    )
    categories["category_group"] = categories["category_group_source"].apply(
        lambda value: strip_prefix(value, "Group:")
    )
    return categories[
        [
            "category_index",
            "category",
            "category_description",
            "answer_format",
            "category_group",
        ]
    ]


def validate_question_category_order(
    questions: pd.DataFrame, categories: pd.DataFrame
) -> None:
    if questions.empty:
        return

    category_count = len(categories)
    question_counts = questions.groupby("document_row_id")["question"].size()
    bad_counts = question_counts[question_counts != category_count]
    if not bad_counts.empty:
        examples = bad_counts.head().astype(int).to_dict()
        raise ValueError(
            f"Expected {category_count} questions per document; mismatches: {examples}"
        )

    ordered_questions = questions.sort_values(
        ["document_row_id", "paragraph_index", "question_index"]
    )
    question_sequences = ordered_questions.groupby("document_row_id")["question"].agg(
        tuple
    )
    if question_sequences.nunique() != 1:
        raise ValueError("Question order is not identical for every document")

    question_categories = [
        normalize_category(extract_question_category(question))
        for question in question_sequences.iloc[0]
    ]
    category_names = [
        normalize_category(category) for category in categories["category"]
    ]
    mismatches = [
        (index, question_category, category_name)
        for index, (question_category, category_name) in enumerate(
            zip(question_categories, category_names)
        )
        if question_category != category_name
    ]
    if mismatches:
        raise ValueError(
            f"Question order does not match category_descriptions.csv: {mismatches[:5]}"
        )


def join_category_descriptions(questions: pd.DataFrame) -> pd.DataFrame:
    categories = load_category_descriptions()
    validate_question_category_order(questions, categories)
    return questions.merge(
        categories,
        how="left",
        left_on="question_index",
        right_on="category_index",
        validate="many_to_one",
    )


def load_datasets() -> dict[str, pd.DataFrame]:
    datasource, primary, aliases = DATA_SOURCE
    path = resolve_path(primary, aliases)
    if path is None:
        names = ", ".join((primary, *aliases))
        raise FileNotFoundError(f"Missing file for {datasource}: none of {names} found")
    #
    records = get_records(load_json(path))
    contracts = make_contracts_df(records)
    paragraphs = make_paragraphs_df(contracts)
    questions = make_questions_df(paragraphs)
    questions = join_category_descriptions(questions)
    #
    for table_name, table in {
        "contracts": contracts,
        "paragraphs": paragraphs,
        "questions": questions,
    }.items():
        table.attrs["datasource"] = datasource
        table.attrs["path"] = path
        table.attrs["table_name"] = table_name
    #
    return {
        "contracts": contracts,
        "paragraphs": paragraphs,
        "questions": questions,
    }


def load_datasource_tables() -> list[pd.DataFrame]:
    """Backward-compatible helper for interactive contract-level exploration."""
    return [load_datasets()["contracts"]]


def summarize_contracts(table: pd.DataFrame) -> dict[str, int]:
    summary = table.count().astype(int).to_dict()
    paragraphs = make_paragraphs_df(table)
    #
    if not paragraphs.empty:
        paragraph_counts = paragraphs.count().astype(int).to_dict()
        summary.update(
            {
                f"paragraphs.{key}": count
                for key, count in paragraph_counts.items()
                if key not in {"title", "paragraph_index"}
            }
        )
    #
    summary["paragraphs_len_sum"] = int(table.get("paragraphs_len", pd.Series()).sum())
    summary["overall_record_count"] = len(table)
    return summary


def summarize_questions(table: pd.DataFrame) -> dict[str, int]:
    summary = table.count().astype(int).to_dict()
    summary["overall_record_count"] = len(table)
    return summary


def ordered_columns(rows: list[dict[str, Any]]) -> list[str]:
    metric_columns = sorted(
        {column for row in rows for column in row if column != "datasource"}
    )
    trailing_columns = ["paragraphs_len_sum", "overall_record_count"]
    return [
        *[column for column in metric_columns if column not in trailing_columns],
        *[column for column in trailing_columns if column in metric_columns],
    ]


def print_summary(rows: list[dict[str, Any]], columns: list[str]) -> None:
    df = pd.DataFrame(rows).fillna(0).set_index("datasource").astype(int)
    print(df.reindex(columns, axis=1).to_string())


def main() -> None:
    datasets = load_datasets()
    contract_rows = [
        {
            "datasource": datasets["contracts"].attrs["datasource"],
            **summarize_contracts(datasets["contracts"]),
        }
    ]
    print("contract_level")
    print_summary(contract_rows, ordered_columns(contract_rows))

    question_rows = [
        {
            "datasource": datasets["questions"].attrs["datasource"],
            **summarize_questions(datasets["questions"]),
        }
    ]
    print("\nquestion_level")
    print_summary(question_rows, ordered_columns(question_rows))


def run_adhoc_examples() -> None:
    ##adhoc: dont remove

    datasets = load_datasets()
    categories = load_category_descriptions()
    questions = datasets["questions"]

    question_counts = questions.groupby("document_id")["question"].size()
    question_sequences = (
        questions.sort_values(["document_row_id", "paragraph_index", "question_index"])
        .groupby("document_row_id")["question"]
        .agg(tuple)
    )
    print(f"category_descriptions rows: {len(categories)}")
    print(f"documents: {question_counts.size}")
    print(
        "questions per document:",
        question_counts.value_counts().sort_index().astype(int).to_dict(),
    )
    print(f"unique question orders: {question_sequences.nunique()}")
    print(
        questions[
            [
                "question_index",
                "category",
                "category_description",
                "answer_format",
                "category_group",
                "question",
            ]
        ]
        .drop_duplicates("question_index")
        .sort_values("question_index")
        .to_string(index=False)
    )

    datasets["contracts"]["context"].loc[0]
    datasets["paragraphs"]
    questions

    questions.head()
    questions.loc[0, "answers"]

    questions["document_id"].value_counts()
    questions["question"].value_counts()

    if os.environ.get("EXPLORE_PRINT_ANSWERS"):
        query_df = questions.query("is_impossible==False")
        for i in query_df.to_dict(orient="records"):
            print("#" * 20)
            print(i.get("question"))
            print("#" * 10 + "ANSWER")
            print(i.get("answers"))


if __name__ == "__main__":
    main()
    if os.environ.get("EXPLORE_ADHOC"):
        run_adhoc_examples()
