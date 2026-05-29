#!/usr/bin/env python3
"""Evaluate all CUAD questions with DSPy over a contract sample."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

os.environ.setdefault(
    "DSPY_CACHEDIR", str(Path(__file__).resolve().parent / ".dspy_cache")
)
os.environ.setdefault("LITELLM_MERGE_REASONING_CONTENT_IN_CHOICES", "true")

import dspy
from dotenv import load_dotenv

from cuad_agent.data.dataset import load_datasets
from cuad_agent.rag.cache import DEFAULT_EMBEDDING_MODEL
from cuad_agent.rag.experiments import DEFAULT_CHUNKING_VERSION
from cuad_agent.rag.query_enrichment import RAG_DEFAULT_TOP_K, query_for_row


QUESTION_COUNT = 41
EVAL_QUESTION_COUNT = QUESTION_COUNT
EVAL_QUESTION_INDICES = tuple(range(EVAL_QUESTION_COUNT))
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
NO_ANSWER = "NO_ANSWER"
NO_ANSWER_MARKERS = {
    "",
    "no answer",
    "none",
    "not found",
    "n/a",
    "na",
    "not applicable",
    "no_answer",
    "noanswer",
}
OUTPUT_STEM = "cuad_dspy_eval"


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


def evaluation_row_id(document_row_id: int, question_index: int) -> str:
    return f"{int(document_row_id)}:{int(question_index)}"


def load_prompt_overrides(prompts_file: Path | None) -> dict[str, str]:
    if prompts_file is None:
        return {}
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_file}")

    spec = importlib.util.spec_from_file_location(
        f"cuad_prompt_overrides_{slugify_model_id(str(prompts_file))}",
        prompts_file,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load prompt file: {prompts_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = getattr(module, "CATEGORY_SYSTEM_PROMPTS", None)
    if not isinstance(prompts, dict):
        raise ValueError(
            f"{prompts_file} must define CATEGORY_SYSTEM_PROMPTS as a dict"
        )
    return {str(category): str(prompt) for category, prompt in prompts.items()}


def resolve_prompts_file_for_model_id(
    prompts_file: Path | None,
    model_id: str,
    *,
    prompts_dir: Path = Path("prompts"),
) -> Path | None:
    if prompts_file is not None:
        return prompts_file
    candidate = prompts_dir / f"system_prompts_{model_id}.py"
    return candidate if candidate.exists() else None


def load_eval_split_ids(eval_split: str | None) -> set[str] | None:
    if not eval_split:
        return None
    path_text, sep, split_name = eval_split.partition(":")
    if not sep or not split_name:
        raise ValueError("--eval-split must use PATH:SPLIT_NAME format")
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"Eval split file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    row_ids: Any
    if isinstance(payload, dict) and isinstance(payload.get("splits"), dict):
        row_ids = payload["splits"].get(split_name)
    elif isinstance(payload, dict):
        row_ids = payload.get(split_name)
    else:
        row_ids = None
    if row_ids is None:
        raise ValueError(f"Split {split_name!r} not found in {path}")
    if not isinstance(row_ids, list):
        raise ValueError(f"Split {split_name!r} in {path} must be a list")
    return {str(row_id) for row_id in row_ids}


def filter_eval_rows_by_split(
    eval_rows: pd.DataFrame,
    split_row_ids: set[str] | None,
) -> pd.DataFrame:
    if split_row_ids is None:
        return eval_rows
    filtered = eval_rows.copy()
    filtered["row_id"] = [
        evaluation_row_id(document_row_id, question_index)
        for document_row_id, question_index in zip(
            filtered["document_row_id"], filtered["question_index"], strict=True
        )
    ]
    filtered = filtered[filtered["row_id"].isin(split_row_ids)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("Eval split did not match any evaluation rows")
    return filtered


def make_question_signature_class(
    question_index: int,
    category: str,
    prompt: str,
) -> type[dspy.Signature]:
    name = f"CuadQuestion{question_index + 1:02d}{class_name_part(category)}Signature"
    namespace = {
        "__module__": __name__,
        "__doc__": prompt,
        "__annotations__": {
            "contract_title": str,
            "contract_text": str,
            "category": str,
            "category_description": str,
            "answer_format": str,
            "answer": str,
            "marked_impossible": bool,
        },
        "contract_title": dspy.InputField(),
        "contract_text": dspy.InputField(),
        "category": dspy.InputField(),
        "category_description": dspy.InputField(),
        "answer_format": dspy.InputField(),
        "answer": dspy.OutputField(
            desc=(
                "Exact answer text span(s) from the contract, separated by newlines "
                "if multiple; or NO_ANSWER."
            )
        ),
        "marked_impossible": dspy.OutputField(
            desc=(
                "Boolean. Return true when the contract does not contain an answer "
                "for this category/question; otherwise false."
            )
        ),
    }
    return type(name, (dspy.Signature,), namespace)


class ContractQuestionAgentBase(dspy.Module):
    """Base class for one fixed CUAD question/category agent."""

    def __init__(
        self,
        question_index: int,
        category: str,
        question: str,
        category_description: str,
        answer_format: str,
        signature_class: type[dspy.Signature],
        *,
        system_prompt: str | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.question_index = question_index
        self.category = category
        self.question = question
        self.category_description = category_description
        self.answer_format = answer_format
        self.signature_class = signature_class
        self.system_prompt = system_prompt
        self.dry_run = dry_run
        self.predict = dspy.ChainOfThought(signature_class)

    def forward(
        self,
        contract_title: str,
        contract_text: str,
        question: str,
        gold_answers: list[str] | None = None,
    ) -> dspy.Prediction:
        if self.dry_run:
            answer = "\n".join(gold_answers or []) if gold_answers else NO_ANSWER
            return dspy.Prediction(
                answer=answer,
                marked_impossible=not bool(gold_answers),
                reasoning="dry_run",
            )

        pred = self.predict(
            contract_title=contract_title,
            contract_text=contract_text,
            category=self.category,
            category_description=self.category_description,
            answer_format=self.answer_format,
        )
        return dspy.Prediction(
            answer=str(pred.answer),
            marked_impossible=parse_bool(getattr(pred, "marked_impossible", False)),
            reasoning=getattr(pred, "reasoning", ""),
        )


ContractQuestionAgent = ContractQuestionAgentBase


def make_question_agent_class(
    question_index: int,
    category: str,
    question: str,
) -> type[ContractQuestionAgentBase]:
    name = f"CuadQuestion{question_index + 1:02d}{class_name_part(category)}Agent"
    return type(
        name,
        (ContractQuestionAgentBase,),
        {"__module__": __name__, "__doc__": question},
    )


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization."""
    text = str(text or "").lower()
    punctuation = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    normalized = normalize_answer(str(value))
    return normalized in {"true", "yes", "y", "1", "marked impossible", "impossible"}


def token_overlap_f1(prediction: str, gold_answers: list[str]) -> float:
    """CUAD-adapted SQuAD token F1 over all required gold spans."""
    pred_norm = normalize_answer(prediction or "")
    gold_toks = tokens(" ".join(gold_answers or []))
    pred_toks = pred_norm.split() if pred_norm else []

    if not gold_toks:
        return 1.0 if pred_norm in NO_ANSWER_MARKERS else 0.0
    if not pred_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    same_count = sum(common.values())
    if same_count == 0:
        return 0.0
    precision = same_count / len(pred_toks)
    recall = same_count / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def build_eval_sample(
    sample_size: int = 50, seed: int = 42
) -> tuple[list[int], dict[int, dict[str, Any]], pd.DataFrame]:
    datasets = load_datasets()
    contracts = datasets["contracts"]
    questions = datasets["questions"]

    candidate_ids = sorted(contracts["document_row_id"].astype(int).tolist())
    if sample_size > len(candidate_ids):
        raise ValueError(
            f"sample_size={sample_size} exceeds available contracts={len(candidate_ids)}"
        )

    rng = random.Random(seed)
    selected_ids = rng.sample(candidate_ids, k=sample_size)

    contract_lookup = {
        int(key): value
        for key, value in contracts.set_index("document_row_id")
        .to_dict("index")
        .items()
    }
    eval_rows = questions[
        questions["document_row_id"].isin(selected_ids)
        & (questions["question_index"].isin(EVAL_QUESTION_INDICES))
    ]
    eval_rows = eval_rows.sort_values(
        ["document_row_id", "question_index"]
    ).reset_index(drop=True)

    expected_rows = sample_size * EVAL_QUESTION_COUNT
    if eval_rows.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} eval rows, found {eval_rows.shape[0]}"
        )
    return selected_ids, contract_lookup, eval_rows


def build_agents(
    questions_df: pd.DataFrame,
    *,
    dry_run: bool = False,
    prompt_overrides: dict[str, str] | None = None,
) -> dict[int, ContractQuestionAgent]:
    categories = (
        questions_df[
            [
                "question_index",
                "category",
                "question",
                "category_description",
                "answer_format",
            ]
        ]
        .drop_duplicates("question_index")
        .sort_values("question_index")
    )
    if categories.empty:
        raise ValueError("No categories found for agent construction")

    agents: dict[int, ContractQuestionAgent] = {}
    prompt_overrides = prompt_overrides or {}
    for row in categories.itertuples(index=False):
        question_index = int(row.question_index)
        if question_index not in EVAL_QUESTION_INDICES:
            raise ValueError(f"Unexpected question_index={question_index}")
        category = str(row.category)
        question = str(row.question)
        system_prompt = prompt_overrides.get(category)
        signature_prompt = system_prompt or question
        signature_class = make_question_signature_class(
            question_index=question_index,
            category=category,
            prompt=signature_prompt,
        )
        agent_class = make_question_agent_class(
            question_index=question_index,
            category=category,
            question=question,
        )
        agents[question_index] = agent_class(
            question_index=question_index,
            category=category,
            question=question,
            category_description=str(row.category_description),
            answer_format="" if pd.isna(row.answer_format) else str(row.answer_format),
            signature_class=signature_class,
            system_prompt=system_prompt,
            dry_run=dry_run,
        )

    for agent in agents.values():
        expected_prompt = agent.system_prompt or agent.question
        if agent.signature_class.__doc__ != expected_prompt:
            raise ValueError(
                f"Signature docstring mismatch for question_index={agent.question_index}"
            )
    return agents


def compose_system_prompt(
    *,
    question: str,
    category: str,
    category_description: str,
    answer_format: str,
) -> str:
    answer_format = answer_format or "No fixed answer format"
    return "\n".join(
        [
            "You are a legal contract review assistant evaluating CUAD clauses.",
            "",
            "Task:",
            question,
            "",
            "Category:",
            category,
            "",
            "Category description:",
            category_description,
            "",
            "Answer format:",
            answer_format,
            "",
            "Instructions:",
            "- Read the provided contract title and contract text.",
            "- Return exact answer text span(s) from the contract when present.",
            "- Separate multiple answer spans with newlines.",
            f"- Return {NO_ANSWER} when the contract does not contain an answer.",
            "- Set marked_impossible to true only when no answer is present.",
        ]
    )


def build_agent_system_prompt(agent: ContractQuestionAgentBase) -> str:
    if agent.system_prompt:
        return agent.system_prompt
    return compose_system_prompt(
        question=agent.question,
        category=agent.category,
        category_description=agent.category_description,
        answer_format=agent.answer_format,
    )


def write_system_prompts(
    agents: dict[int, ContractQuestionAgentBase],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    lines = [
        '"""Generated CUAD v1 system prompts by category.',
        "",
        "Edit these strings to iterate on prompt variants, then wire the chosen",
        "prompt text back into the evaluator before running a new model_id.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    mapping_entries: list[str] = []
    for question_index, agent in sorted(agents.items()):
        base_name = f"{prompt_name_part(agent.category)}_SYSTEM_PROMPT"
        prompt_name = base_name
        suffix = 2
        while prompt_name in used_names:
            prompt_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(prompt_name)
        lines.extend(
            [
                f"# Question {question_index + 1}: {agent.category}",
                f"{prompt_name} = {build_agent_system_prompt(agent)!r}",
                "",
            ]
        )
        mapping_entries.append(f"    {agent.category!r}: {prompt_name},")

    lines.extend(["CATEGORY_SYSTEM_PROMPTS = {", *mapping_entries, "}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def answer_texts(answers: Any) -> list[str]:
    if not isinstance(answers, list):
        return []
    return [
        str(answer.get("text", "")) for answer in answers if isinstance(answer, dict)
    ]


def build_devset(
    contract_lookup: dict[int, dict[str, Any]],
    eval_rows: pd.DataFrame,
    *,
    dry_run: bool = False,
) -> list[dspy.Example]:
    input_fields = ["contract_title", "contract_text", "question"]
    if dry_run:
        input_fields.append("gold_answers")

    examples: list[dspy.Example] = []
    ordered_rows = eval_rows.sort_values(["document_row_id", "question_index"])
    for row in ordered_rows.itertuples(index=False):
        contract = contract_lookup[int(row.document_row_id)]
        gold_answers = answer_texts(row.answers)
        row_id = evaluation_row_id(int(row.document_row_id), int(row.question_index))
        example = dspy.Example(
            row_id=row_id,
            document_row_id=int(row.document_row_id),
            title=str(contract.get("title", "")),
            contract_title=str(contract.get("title", "")),
            contract_text=str(contract.get("context", "")),
            question_index=int(row.question_index),
            category=str(row.category),
            category_description=str(row.category_description),
            answer_format="" if pd.isna(row.answer_format) else str(row.answer_format),
            question=str(row.question),
            gold_answers=gold_answers,
            gold_marked_impossible=bool(row.is_impossible),
            answers_len=int(row.answers_len),
        ).with_inputs(*input_fields)
        examples.append(example)
    return examples


def cuad_overlap_metric(
    example: dspy.Example, pred: dspy.Prediction, trace: Any = None
) -> float:
    return token_overlap_f1(
        str(getattr(pred, "answer", "")), list(example.gold_answers)
    )


def eval_results_to_dataframe(
    eval_results: list[Any], *, model_id: str | None = None
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for eval_result in eval_results:
        for example, pred, score in eval_result.results:
            predicted_answer = str(getattr(pred, "answer", ""))
            predicted_marked_impossible = parse_bool(
                getattr(pred, "marked_impossible", False)
            )
            gold_answers = list(example.gold_answers)
            records.append(
                {
                    "model_id": model_id or "",
                    "row_id": str(getattr(example, "row_id", "")),
                    "document_row_id": int(example.document_row_id),
                    "title": str(example.title),
                    "question_index": int(example.question_index),
                    "category": str(example.category),
                    "category_description": str(example.category_description),
                    "answer_format": str(example.answer_format),
                    "question": str(example.question),
                    "gold_answers": json.dumps(gold_answers, ensure_ascii=False),
                    "predicted_answer": predicted_answer,
                    "predicted_marked_impossible": predicted_marked_impossible,
                    "gold_marked_impossible": bool(example.gold_marked_impossible),
                    "token_f1": float(score),
                    "correct_at_0_5": float(score) >= 0.5,
                    "is_impossible": bool(example.gold_marked_impossible),
                    "answers_len": int(example.answers_len),
                }
            )
    return (
        pd.DataFrame(records)
        .sort_values(["document_row_id", "question_index"])
        .reset_index(drop=True)
    )


def parse_gold_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def build_evaluation_page_data(
    results: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    per_category = {
        int(row["question_index"]): row for row in summary.get("per_category", [])
    }
    comparison = summary.get("baseline_comparison")
    comparison_examples: dict[str, dict[str, Any]] = {}
    comparison_categories: dict[int, dict[str, Any]] = {}
    if isinstance(comparison, dict):
        comparison_examples = {
            str(row.get("row_id")): row
            for row in comparison.get("examples", [])
            if isinstance(row, dict) and row.get("row_id") is not None
        }
        comparison_categories = {
            int(row["question_index"]): row
            for row in comparison.get("per_category", [])
            if isinstance(row, dict) and row.get("question_index") is not None
        }
    ordered_results = results.copy()
    if "row_id" not in ordered_results.columns:
        ordered_results["row_id"] = [
            evaluation_row_id(document_row_id, question_index)
            for document_row_id, question_index in zip(
                ordered_results["document_row_id"],
                ordered_results["question_index"],
                strict=True,
            )
        ]
    ordered_results = ordered_results.sort_values(["question_index", "document_row_id"])
    for question_index, question_rows in ordered_results.groupby("question_index"):
        first = question_rows.iloc[0]
        metrics = per_category.get(int(question_index), {})
        comparison_metrics = comparison_categories.get(int(question_index), {})
        rows: list[dict[str, Any]] = []
        for row in question_rows.to_dict("records"):
            row_comparison = comparison_examples.get(str(row["row_id"]))
            rows.append(
                {
                    "row_id": str(row["row_id"]),
                    "document_row_id": int(row["document_row_id"]),
                    "title": str(row["title"]),
                    "gold_answers": parse_gold_answers(row["gold_answers"]),
                    "predicted_answer": str(row["predicted_answer"]),
                    "predicted_marked_impossible": bool(
                        row["predicted_marked_impossible"]
                    ),
                    "gold_marked_impossible": bool(row["gold_marked_impossible"]),
                    "token_f1": float(row["token_f1"]),
                    "correct_at_0_5": bool(row["correct_at_0_5"]),
                    "comparison": row_comparison,
                }
            )
        questions.append(
            {
                "question_index": int(question_index),
                "category": str(first.category),
                "category_description": str(first.category_description),
                "answer_format": str(first.answer_format),
                "question": str(first.question),
                "mean_token_f1": float(metrics.get("mean_token_f1", 0.0)),
                "correct_at_0_5": float(metrics.get("correct_at_0_5", 0.0)),
                "count": int(metrics.get("count", len(question_rows))),
                "comparison": comparison_metrics,
                "results": rows,
            }
        )
    category_context = {
        question["question_index"]: {
            "question_index": question["question_index"],
            "category": question["category"],
            "category_description": question["category_description"],
            "answer_format": question["answer_format"],
            "question": question["question"],
        }
        for question in questions
    }
    per_category = [
        {
            **category_context.get(int(row.get("question_index", -1)), {}),
            **row,
            "comparison": comparison_categories.get(
                int(row.get("question_index", -1)), {}
            ),
        }
        for row in summary.get("per_category", [])
    ]
    return {
        "summary": {
            key: summary[key]
            for key in [
                "sample_size",
                "seed",
                "model_id",
                "total_examples",
                "questions_per_contract",
                "agent_count",
                "model",
                "temperature",
                "max_tokens",
                "num_threads",
                "dry_run",
                "overlap_accuracy_mean_f1",
                "correct_at_0_5",
            ]
            if key in summary
        },
        "comparison": comparison if isinstance(comparison, dict) else None,
        "per_category": per_category,
        "questions": questions,
    }


def render_evaluation_html(page_data: dict[str, Any]) -> str:
    data_json = json.dumps(page_data, ensure_ascii=False).replace("</", "<\\/")
    model_id = str(page_data.get("summary", {}).get("model_id", "model"))
    evaluation_href = f"evaluation_{model_id}.html"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUAD Evaluation</title>
  <style>
    :root {{
      color-scheme: dark;
      --app-bg: #0b141a;
      --panel: #111b21;
      --panel-2: #202c33;
      --panel-3: #182229;
      --border: #26353d;
      --text: #e9edef;
      --muted: #8696a0;
      --muted-2: #aebac1;
      --green: #00a884;
      --bubble-in: #202c33;
      --bubble-out: #005c4b;
      --danger: #f15c6d;
      --shadow: rgba(0, 0, 0, 0.28);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; }}
    body {{
      margin: 0;
      background: var(--app-bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
    }}
    button {{ font: inherit; }}
    .global-header {{ height: 56px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 18px; background: var(--panel-2); border-bottom: 1px solid var(--border); }}
    .brand {{ min-width: 0; display: flex; align-items: center; gap: 10px; font-weight: 700; }}
    .brand-mark {{ width: 30px; height: 30px; border-radius: 8px; background: linear-gradient(135deg, #00a884, #3b82f6); display: grid; place-items: center; color: white; font-size: 13px; }}
    .tabs {{ display: inline-flex; align-items: center; gap: 4px; padding: 4px; border: 1px solid var(--border); border-radius: 8px; background: rgba(0,0,0,0.16); }}
    .tab {{ color: var(--muted-2); text-decoration: none; padding: 7px 12px; border-radius: 6px; font-size: 13px; line-height: 1; white-space: nowrap; }}
    .tab:hover {{ color: var(--text); background: rgba(255,255,255,0.06); }}
    .tab.active {{ color: var(--text); background: var(--green); }}
    .app {{ display: grid; grid-template-columns: minmax(300px, 380px) minmax(0, 1fr); height: calc(100vh - 56px); height: calc(100dvh - 56px); min-height: 0; overflow: hidden; }}
    .sidebar {{ background: var(--panel); border-right: 1px solid var(--border); min-width: 0; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }}
    .side-top, .chat-top {{ height: 64px; flex: 0 0 64px; background: var(--panel-2); display: flex; align-items: center; padding: 0 16px; gap: 12px; }}
    .avatar {{ width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #00a884, #3b82f6); display: grid; place-items: center; color: white; font-weight: 700; flex: 0 0 auto; }}
    .side-title {{ font-size: 16px; font-weight: 650; }}
    .side-subtitle, .chat-subtitle {{ color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .search {{ flex: 0 0 auto; padding: 10px 12px; border-bottom: 1px solid var(--border); }}
    .search input {{ width: 100%; border: 0; outline: 0; border-radius: 8px; background: var(--panel-2); color: var(--text); padding: 11px 14px; }}
    .conversations {{ flex: 1 1 auto; min-height: 0; overflow: auto; }}
    .conversation {{ width: 100%; border: 0; border-bottom: 1px solid var(--border); color: inherit; background: transparent; display: grid; grid-template-columns: 48px 1fr; gap: 12px; padding: 12px 14px; text-align: left; cursor: pointer; }}
    .conversation:hover, .conversation.active {{ background: var(--panel-2); }}
    .conversation h3 {{ margin: 0 0 3px; font-size: 15px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .conversation p {{ margin: 0; color: var(--muted); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .chat {{ min-width: 0; min-height: 0; display: flex; flex-direction: column; background: #0b141a; overflow: hidden; }}
    .chat-title {{ min-width: 0; }}
    .chat-title h1 {{ margin: 0 0 2px; font-size: 16px; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .messages {{ flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden; overscroll-behavior: contain; -webkit-overflow-scrolling: touch; padding: 22px clamp(12px, 3vw, 42px); background-image: radial-gradient(circle at 12px 12px, rgba(255,255,255,0.035) 1px, transparent 1px); background-size: 24px 24px; }}
    .message {{ max-width: min(1440px, 100%); margin: 0 0 14px; padding: 12px 14px; border-radius: 8px; box-shadow: 0 1px 0 var(--shadow); }}
    .incoming {{ background: var(--bubble-in); }}
    .outgoing {{ background: var(--bubble-out); margin-left: auto; }}
    .message h2 {{ margin: 0 0 8px; font-size: 17px; }}
    .message p {{ margin: 0 0 8px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(4, minmax(110px, 1fr)); gap: 10px; }}
    .metric {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px; }}
    .metric b {{ display: block; font-size: 18px; }}
    .metric span {{ color: var(--muted-2); font-size: 12px; }}
    .table-wrap {{ width: 100%; overflow: auto; border: 1px solid var(--border); border-radius: 8px; background: rgba(0,0,0,0.14); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1120px; }}
    .result-table {{ min-width: 1320px; table-layout: fixed; }}
    .result-table.compare {{ min-width: 1820px; }}
    .summary-table {{ min-width: 1180px; table-layout: fixed; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; text-align: left; }}
    th {{ color: var(--muted-2); background: rgba(0,0,0,0.18); font-size: 12px; font-weight: 650; position: sticky; top: 0; z-index: 1; }}
    tr:last-child td {{ border-bottom: 0; }}
    .idx {{ color: var(--muted); width: 56px; }}
    .category {{ font-weight: 650; min-width: 190px; }}
    .answer {{ white-space: pre-wrap; min-width: 300px; overflow-wrap: anywhere; }}
    .contract-col {{ width: 260px; overflow-wrap: anywhere; }}
    .gold-col, .prediction-col {{ width: 360px; }}
    .score-col, .status-col {{ width: 88px; }}
    .empty {{ color: var(--muted); font-style: italic; }}
    .pill {{ display: inline-flex; align-items: center; min-height: 22px; padding: 2px 8px; border-radius: 999px; background: rgba(255,255,255,0.08); color: var(--muted-2); font-size: 12px; white-space: nowrap; }}
    .pill.good {{ background: rgba(0,168,132,0.16); color: #90f0d8; }}
    .pill.warn {{ background: rgba(241,92,109,0.14); color: #ffb4bd; }}
    .score.good {{ color: #90f0d8; font-weight: 700; }}
    .score.bad {{ color: #ffb4bd; font-weight: 700; }}
    .delta.good {{ color: #90f0d8; font-weight: 700; }}
    .delta.bad {{ color: #ffb4bd; font-weight: 700; }}
    .delta.neutral {{ color: var(--muted-2); font-weight: 700; }}
    @media (max-width: 760px) {{
      body {{ overflow: auto; }}
      .global-header {{ align-items: stretch; flex-direction: column; height: auto; padding: 10px 12px; }}
      .brand {{ width: 100%; }}
      .tabs {{ width: 100%; display: grid; grid-template-columns: 1fr 1fr; }}
      .tab {{ text-align: center; }}
      .app {{ grid-template-columns: 1fr; height: calc(100vh - 103px); height: calc(100dvh - 103px); min-height: 0; }}
      .sidebar {{ max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--border); }}
      .chat {{ min-height: 0; }}
      .meta-grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header class="global-header">
    <div class="brand"><div class="brand-mark">C</div><span>CUAD Review</span></div>
    <nav class="tabs" aria-label="Views">
      <a class="tab" href="explore.html">Explorer</a>
      <a class="tab active" href="{evaluation_href}" aria-current="page">Evaluation</a>
    </nav>
  </header>
  <div class="app">
    <aside class="sidebar" aria-label="Conversations">
      <div class="side-top">
        <div class="avatar">E</div>
        <div>
          <div class="side-title">CUAD Evaluation</div>
          <div class="side-subtitle">Summary and question results</div>
        </div>
      </div>
      <div class="search"><input id="search" type="search" placeholder="Search questions"></div>
      <div id="conversationList" class="conversations"></div>
    </aside>
    <main class="chat">
      <header class="chat-top">
        <div id="chatAvatar" class="avatar">S</div>
        <div class="chat-title">
          <h1 id="chatTitle"></h1>
          <div id="chatSubtitle" class="chat-subtitle"></div>
        </div>
      </header>
      <section id="messages" class="messages" aria-live="polite"></section>
    </main>
  </div>
  <script id="app-data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('app-data').textContent);
    const conversations = [
      {{ id: 'summary', kind: 'summary', title: 'Evaluation summary', subtitle: `${{data.summary.total_examples}} examples · ${{data.summary.questions_per_contract}} questions`, avatar: 'S' }},
      ...data.questions.map(question => ({{
        id: `question-${{question.question_index}}`,
        kind: 'question',
        title: `${{question.question_index + 1}}. ${{question.category}}`,
        subtitle: `${{pct(question.mean_token_f1 / 100)}} mean F1 · ${{pct(question.correct_at_0_5 / 100)}} correct@0.5`,
        avatar: String(question.question_index + 1),
        question,
      }})),
    ];
    let activeId = 'summary';

    const listEl = document.getElementById('conversationList');
    const messagesEl = document.getElementById('messages');
    const titleEl = document.getElementById('chatTitle');
    const subtitleEl = document.getElementById('chatSubtitle');
    const avatarEl = document.getElementById('chatAvatar');
    const searchEl = document.getElementById('search');

    function esc(value) {{
      return String(value ?? '').replace(/[&<>'"]/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
      }}[ch]));
    }}

    function pct(value) {{
      return `${{(Number(value || 0) * 100).toFixed(1)}}%`;
    }}

    function pctPoints(value) {{
      const number = Number(value || 0);
      const sign = number > 0 ? '+' : '';
      return `${{sign}}${{(number * 100).toFixed(1)}} pts`;
    }}

    function pctValue(value) {{
      return `${{Number(value || 0).toFixed(1)}}%`;
    }}

    function deltaClass(value) {{
      const number = Number(value || 0);
      if (number > 0.0001) return 'good';
      if (number < -0.0001) return 'bad';
      return 'neutral';
    }}

    function answerHtml(answers) {{
      if (!answers || answers.length === 0) return '<span class="empty">No golden answer</span>';
      return answers.map(esc).join('\\n\\n');
    }}

    function predictionHtml(answer, markedImpossible) {{
      const text = String(answer || '').trim();
      const body = text ? esc(text) : '<span class="empty">Empty prediction</span>';
      return `${{body}}${{markedImpossible ? '<br><span class="pill warn">Predicted no answer</span>' : ''}}`;
    }}

    function modelLabel(kind) {{
      const comparison = data.comparison || {{}};
      if (kind === 'baseline') return esc(comparison.baseline_model_id || 'Baseline');
      return esc(data.summary.model_id || 'Latest');
    }}

    function renderList() {{
      const needle = searchEl.value.trim().toLowerCase();
      listEl.innerHTML = conversations
        .filter(c => !needle || c.title.toLowerCase().includes(needle) || c.subtitle.toLowerCase().includes(needle))
        .map(c => `
          <button class="conversation ${{c.id === activeId ? 'active' : ''}}" data-id="${{esc(c.id)}}">
            <div class="avatar">${{esc(c.avatar)}}</div>
            <div><h3>${{esc(c.title)}}</h3><p>${{esc(c.subtitle)}}</p></div>
          </button>
        `).join('');
      listEl.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {{
        activeId = btn.dataset.id;
        render();
      }}));
    }}

    function renderSummary() {{
      const rows = data.per_category.map(row => `
        <tr>
          <td class="idx">${{row.question_index + 1}}</td>
          <td class="category">${{esc(row.category)}}<br><span class="pill">${{esc(row.answer_format || 'No answer format')}}</span></td>
          <td>${{esc(row.category_description || '')}}<br><span class="pill">Question ${{row.question_index + 1}}</span></td>
          <td><span class="score ${{row.mean_token_f1 >= 50 ? 'good' : 'bad'}}">${{pctValue(row.mean_token_f1)}}</span></td>
          <td>${{pctValue(row.correct_at_0_5)}}</td>
          <td>${{row.comparison && row.comparison.baseline_mean_token_f1 !== undefined ? pctValue(row.comparison.baseline_mean_token_f1) : '<span class="empty">n/a</span>'}}</td>
          <td>${{row.comparison && row.comparison.mean_token_f1_delta !== undefined ? `<span class="delta ${{deltaClass(row.comparison.mean_token_f1_delta / 100)}}">${{pctPoints(row.comparison.mean_token_f1_delta / 100)}}</span>` : '<span class="empty">n/a</span>'}}</td>
          <td>${{row.count}}</td>
        </tr>
      `).join('');
      const comparison = data.comparison;
      const comparisonHtml = comparison ? `
        <article class="message incoming">
          <h2>${{modelLabel('baseline')}} vs ${{modelLabel('latest')}}</h2>
          <div class="meta-grid">
            <div class="metric"><b>${{pctValue(comparison.baseline_mean_token_f1)}}</b><span>${{modelLabel('baseline')}} mean F1</span></div>
            <div class="metric"><b>${{pctValue(comparison.candidate_mean_token_f1)}}</b><span>${{modelLabel('latest')}} mean F1</span></div>
            <div class="metric"><b><span class="delta ${{deltaClass(comparison.mean_token_f1_delta / 100)}}">${{pctPoints(comparison.mean_token_f1_delta / 100)}}</span></b><span>Mean F1 change</span></div>
            <div class="metric"><b>${{comparison.matched_examples}}</b><span>Matched examples</span></div>
            <div class="metric"><b>${{pctValue(comparison.baseline_correct_at_0_5)}}</b><span>${{modelLabel('baseline')}} correct@0.5</span></div>
            <div class="metric"><b>${{pctValue(comparison.candidate_correct_at_0_5)}}</b><span>${{modelLabel('latest')}} correct@0.5</span></div>
            <div class="metric"><b><span class="delta ${{deltaClass(comparison.correct_at_0_5_delta / 100)}}">${{pctPoints(comparison.correct_at_0_5_delta / 100)}}</span></b><span>Correct@0.5 change</span></div>
            <div class="metric"><b>${{esc(comparison.baseline_results_path || '')}}</b><span>Baseline source</span></div>
          </div>
        </article>
      ` : '';
      messagesEl.innerHTML = `
        <article class="message incoming">
          <h2>Evaluation Run</h2>
          <div class="meta-grid">
            <div class="metric"><b>${{data.summary.total_examples}}</b><span>Examples</span></div>
            <div class="metric"><b>${{data.summary.sample_size}}</b><span>Contracts</span></div>
            <div class="metric"><b>${{data.summary.agent_count}}</b><span>Question agents</span></div>
            <div class="metric"><b>${{data.summary.dry_run ? 'Yes' : 'No'}}</b><span>Dry run</span></div>
          </div>
        </article>
        <article class="message outgoing"><p>Model ID: ${{esc(data.summary.model_id)}} · model: ${{esc(data.summary.model)}} · seed ${{esc(data.summary.seed)}} · threads ${{esc(data.summary.num_threads)}}</p></article>
        <article class="message incoming">
          <div class="meta-grid">
            <div class="metric"><b>${{data.summary.overlap_accuracy_mean_f1.toFixed(1)}}%</b><span>Mean token F1</span></div>
            <div class="metric"><b>${{data.summary.correct_at_0_5.toFixed(1)}}%</b><span>Correct at 0.5</span></div>
            <div class="metric"><b>${{data.summary.temperature}}</b><span>Temperature</span></div>
            <div class="metric"><b>${{data.summary.max_tokens}}</b><span>Max tokens</span></div>
          </div>
        </article>
        ${{comparisonHtml}}
        <article class="message incoming">
          <div class="table-wrap">
            <table class="summary-table">
              <thead><tr><th>#</th><th>Category</th><th>category_descriptions.csv context</th><th>${{modelLabel('latest')}} F1</th><th>${{modelLabel('latest')}} correct</th><th>${{modelLabel('baseline')}} F1</th><th>F1 change</th><th>Examples</th></tr></thead>
              <tbody>${{rows}}</tbody>
            </table>
          </div>
        </article>
      `;
    }}

    function renderQuestion(question) {{
      const hasComparison = question.results.some(row => row.comparison);
      const rows = question.results.map(row => {{
        if (!hasComparison) return `
          <tr>
            <td class="idx">${{row.document_row_id}}</td>
            <td>${{esc(row.title)}}<br><span class="pill">${{row.gold_marked_impossible ? 'Gold no answer' : 'Gold answer'}}</span></td>
            <td class="answer">${{answerHtml(row.gold_answers)}}</td>
            <td class="answer">${{predictionHtml(row.predicted_answer, row.predicted_marked_impossible)}}</td>
            <td><span class="score ${{row.correct_at_0_5 ? 'good' : 'bad'}}">${{pct(row.token_f1)}}</span></td>
            <td>${{row.correct_at_0_5 ? '<span class="pill good">Pass</span>' : '<span class="pill warn">Miss</span>'}}</td>
          </tr>
        `;
        const comparison = row.comparison || {{}};
        const baselineAnswer = comparison.baseline_predicted_answer !== undefined
          ? predictionHtml(comparison.baseline_predicted_answer, comparison.baseline_predicted_marked_impossible)
          : '<span class="empty">No baseline row</span>';
        const baselineF1 = comparison.baseline_token_f1 !== undefined
          ? `<span class="score ${{comparison.baseline_correct_at_0_5 ? 'good' : 'bad'}}">${{pct(comparison.baseline_token_f1)}}</span>`
          : '<span class="empty">n/a</span>';
        const delta = comparison.token_f1_delta !== undefined
          ? `<span class="delta ${{deltaClass(comparison.token_f1_delta)}}">${{pctPoints(comparison.token_f1_delta)}}</span>`
          : '<span class="empty">n/a</span>';
        return `
          <tr>
            <td class="idx">${{row.document_row_id}}</td>
            <td>${{esc(row.title)}}<br><span class="pill">${{row.gold_marked_impossible ? 'Gold no answer' : 'Gold answer'}}</span></td>
            <td class="answer">${{answerHtml(row.gold_answers)}}</td>
            <td class="answer">${{baselineAnswer}}</td>
            <td class="answer">${{predictionHtml(row.predicted_answer, row.predicted_marked_impossible)}}</td>
            <td>${{baselineF1}}</td>
            <td><span class="score ${{row.correct_at_0_5 ? 'good' : 'bad'}}">${{pct(row.token_f1)}}</span></td>
            <td>${{delta}}</td>
            <td>${{row.correct_at_0_5 ? '<span class="pill good">Pass</span>' : '<span class="pill warn">Miss</span>'}}</td>
          </tr>
        `;
      }}).join('');
      const comparisonMetrics = question.comparison && question.comparison.baseline_mean_token_f1 !== undefined ? `
        <article class="message incoming">
          <div class="meta-grid">
            <div class="metric"><b>${{pctValue(question.comparison.baseline_mean_token_f1)}}</b><span>${{modelLabel('baseline')}} mean F1</span></div>
            <div class="metric"><b>${{pctValue(question.comparison.candidate_mean_token_f1)}}</b><span>${{modelLabel('latest')}} mean F1</span></div>
            <div class="metric"><b><span class="delta ${{deltaClass(question.comparison.mean_token_f1_delta / 100)}}">${{pctPoints(question.comparison.mean_token_f1_delta / 100)}}</span></b><span>Mean F1 change</span></div>
            <div class="metric"><b>${{pctPoints(question.comparison.correct_at_0_5_delta / 100)}}</b><span>Correct@0.5 change</span></div>
          </div>
        </article>
      ` : '';
      const tableHead = hasComparison
        ? `<tr><th>Doc</th><th>Contract</th><th>Gold</th><th>${{modelLabel('baseline')}} answer</th><th>${{modelLabel('latest')}} answer</th><th>${{modelLabel('baseline')}} F1</th><th>${{modelLabel('latest')}} F1</th><th>F1 change</th><th>Status</th></tr>`
        : '<tr><th>Doc</th><th>Contract</th><th>Gold</th><th>Prediction</th><th>F1</th><th>Status</th></tr>';
      const tableCols = hasComparison
        ? `
                <col style="width: 64px">
                <col class="contract-col">
                <col class="gold-col">
                <col class="prediction-col">
                <col class="prediction-col">
                <col class="score-col">
                <col class="score-col">
                <col class="score-col">
                <col class="status-col">
        `
        : `
                <col style="width: 64px">
                <col class="contract-col">
                <col class="gold-col">
                <col class="prediction-col">
                <col class="score-col">
                <col class="status-col">
        `;
      messagesEl.innerHTML = `
        <article class="message outgoing"><p>${{esc(question.question)}}</p></article>
        <article class="message incoming">
          <h2>${{esc(question.category)}}</h2>
          <p>${{esc(question.category_description)}}</p>
          <span class="pill">${{esc(question.answer_format || 'No answer format')}}</span>
        </article>
        <article class="message incoming">
          <div class="meta-grid">
            <div class="metric"><b>${{question.mean_token_f1.toFixed(1)}}%</b><span>Mean token F1</span></div>
            <div class="metric"><b>${{question.correct_at_0_5.toFixed(1)}}%</b><span>Correct at 0.5</span></div>
            <div class="metric"><b>${{question.count}}</b><span>Contracts</span></div>
            <div class="metric"><b>${{question.question_index + 1}}</b><span>Question number</span></div>
          </div>
        </article>
        ${{comparisonMetrics}}
        <article class="message incoming">
          <div class="table-wrap">
            <table class="result-table ${{hasComparison ? 'compare' : ''}}">
              <colgroup>
                ${{tableCols}}
              </colgroup>
              <thead>${{tableHead}}</thead>
              <tbody>${{rows}}</tbody>
            </table>
          </div>
        </article>
      `;
    }}

    function render() {{
      const active = conversations.find(c => c.id === activeId) || conversations[0];
      titleEl.textContent = active.title;
      subtitleEl.textContent = active.subtitle;
      avatarEl.textContent = active.avatar;
      if (active.kind === 'summary') renderSummary();
      else renderQuestion(active.question);
      renderList();
      messagesEl.scrollTop = 0;
    }}

    searchEl.addEventListener('input', renderList);
    render();
  </script>
</body>
</html>
"""


def write_evaluation_html(
    results: pd.DataFrame,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    page_data = build_evaluation_page_data(results, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_evaluation_html(page_data), encoding="utf-8")


def configure_lm(args: argparse.Namespace) -> None:
    load_dotenv(Path.home() / ".env")
    api_key: str | None = None
    if args.model.startswith("deepseek/"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek DSPy models")
    elif args.model.startswith("openai/"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI DSPy models")

    lm = dspy.LM(
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())


def predicted_no_answer_mask(results: pd.DataFrame) -> pd.Series:
    """Per-row model decision that the clause is absent.

    Combines the explicit ``predicted_marked_impossible`` flag with the answer
    text normalizing to a NO_ANSWER marker, so it agrees with how
    ``token_overlap_f1`` scores the no-answer class.
    """
    if results.empty:
        return pd.Series([], dtype=bool)
    marked = results["predicted_marked_impossible"].astype(bool)
    normalized = (
        results["predicted_answer"]
        .fillna("")
        .map(lambda text: normalize_answer(str(text)) in NO_ANSWER_MARKERS)
    )
    return marked | normalized


def detection_metrics(results: pd.DataFrame) -> dict[str, Any]:
    """No-answer vs answer detection accuracy.

    Many CUAD gold answers are "no answer", so aggregate F1 is dominated by the
    no-answer class. These metrics split detection accuracy by gold class to
    show how well the model identifies whether a clause is present at all.
    """
    if results.empty:
        return {
            "gold_no_answer_count": 0,
            "gold_answer_count": 0,
            "predicted_no_answer_count": 0,
            "no_answer_detection_accuracy": None,
            "answer_detection_accuracy": None,
            "detection_accuracy": None,
        }
    gold_no = results["gold_marked_impossible"].astype(bool)
    pred_no = predicted_no_answer_mask(results)
    gold_no_count = int(gold_no.sum())
    gold_answer_count = int((~gold_no).sum())
    return {
        "gold_no_answer_count": gold_no_count,
        "gold_answer_count": gold_answer_count,
        "predicted_no_answer_count": int(pred_no.sum()),
        "no_answer_detection_accuracy": (
            float((pred_no & gold_no).sum() / gold_no_count * 100)
            if gold_no_count
            else None
        ),
        "answer_detection_accuracy": (
            float((~pred_no & ~gold_no).sum() / gold_answer_count * 100)
            if gold_answer_count
            else None
        ),
        "detection_accuracy": float((pred_no == gold_no).mean() * 100),
    }


def summarize_results(
    results: pd.DataFrame,
    *,
    args: argparse.Namespace,
    selected_document_row_ids: list[int],
) -> dict[str, Any]:
    per_category_df = (
        results.groupby(["question_index", "category"], as_index=False)
        .agg(
            mean_token_f1=("token_f1", "mean"),
            correct_at_0_5=("correct_at_0_5", "mean"),
            count=("token_f1", "size"),
        )
        .sort_values(["question_index"])
    )
    per_category = [
        {
            "question_index": int(row.question_index),
            "category": str(row.category),
            "mean_token_f1": float(row.mean_token_f1 * 100),
            "correct_at_0_5": float(row.correct_at_0_5 * 100),
            "count": int(row.count),
        }
        for row in per_category_df.itertuples(index=False)
    ]
    return {
        "sample_size": int(args.sample_size),
        "seed": int(args.seed),
        "model_id": str(args.model_id),
        "selected_document_row_ids": selected_document_row_ids,
        "total_examples": int(len(results)),
        "questions_per_contract": EVAL_QUESTION_COUNT,
        "evaluated_question_numbers": [index + 1 for index in EVAL_QUESTION_INDICES],
        "evaluated_question_indices": list(EVAL_QUESTION_INDICES),
        "agent_count": int(results["question_index"].nunique()),
        "model": args.model,
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "num_threads": int(args.num_threads),
        "dry_run": bool(args.dry_run),
        "prompts_file": str(args.prompts_file)
        if getattr(args, "prompts_file", None)
        else None,
        "eval_split": str(args.eval_split)
        if getattr(args, "eval_split", None)
        else None,
        "context_mode": str(getattr(args, "context_mode", "raw")),
        "overlap_accuracy_mean_f1": float(results["token_f1"].mean() * 100),
        "correct_at_0_5": float(results["correct_at_0_5"].mean() * 100),
        **detection_metrics(results),
        "per_category": per_category,
    }


def write_outputs(
    results: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    html_output: Path | None,
    agents: dict[int, ContractQuestionAgentBase] | None = None,
) -> None:
    model_id = str(summary["model_id"])
    paths = output_paths(output_dir, model_id, html_output)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["html"].parent.mkdir(parents=True, exist_ok=True)
    if agents is not None:
        write_system_prompts(agents, paths["system_prompts"])
        summary["system_prompts_path"] = str(paths["system_prompts"])
    results.to_csv(paths["results"], index=False)
    paths["summary"].write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_evaluation_html(results, summary, paths["html"])


def _apply_rag_context_to_devset(
    devset: list[dspy.Example],
    *,
    context_mode: str,
    top_k: int,
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
) -> list[dspy.Example]:
    from cuad_agent.rag.context_builder import build_rag_context

    print(
        f"[RAG] Replacing contract_text with {context_mode} context "
        f"(top_k={top_k}) for {len(devset)} examples...",
        flush=True,
    )
    updated: list[dspy.Example] = []
    for ex in devset:
        query = query_for_row(ex)
        rag_context, _ = build_rag_context(
            document_row_id=ex.document_row_id,
            query=query,
            method=context_mode,
            top_k=top_k,
            output_dir=output_dir,
            chunking_version=chunking_version,
            embedding_model=embedding_model,
        )
        updated.append(ex.copy(contract_text=rag_context))
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "Stable identifier for this model/config run. Defaults to a slug "
            "derived from --model."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help=(
            "Optional explicit HTML output path. Defaults to "
            "dashboards/evaluation_MODEL_ID.html. Bare relative filenames are "
            "written under dashboards/."
        ),
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help=(
            "Optional Python prompt module defining CATEGORY_SYSTEM_PROMPTS. "
            "Category prompts override generated question docstrings."
        ),
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help=(
            "Optional split selector in PATH:SPLIT_NAME format. The split file "
            "must contain row ids like document_row_id:question_index."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--context-mode",
        choices=["raw", "rag-dense", "rag-hybrid"],
        default="raw",
        help=(
            "Context supplied to the agent. 'raw' uses the full contract text; "
            "'rag-dense' and 'rag-hybrid' use retrieved sentence chunks. "
            "RAG modes require a prebuilt sentence cache (run rag-eval first)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RAG_DEFAULT_TOP_K,
        help="Number of chunks to retrieve for RAG context modes.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model key for dense retrieval cache.",
    )
    parser.add_argument(
        "--chunking-version",
        default=DEFAULT_CHUNKING_VERSION,
        help="Chunking version key for sentence cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_id = resolve_model_id(args)
    args.prompts_file = resolve_prompts_file_for_model_id(
        args.prompts_file,
        args.model_id,
    )
    if args.num_threads < 1:
        raise ValueError("--num-threads must be >= 1")
    if not args.dry_run:
        configure_lm(args)

    selected_ids, contract_lookup, eval_rows = build_eval_sample(
        sample_size=args.sample_size,
        seed=args.seed,
    )
    check_ids, _, _ = build_eval_sample(sample_size=args.sample_size, seed=args.seed)
    if selected_ids != check_ids:
        raise AssertionError(
            "Deterministic sampling failed for repeated build_eval_sample"
        )
    prompt_overrides = load_prompt_overrides(args.prompts_file)
    split_row_ids = load_eval_split_ids(args.eval_split)
    eval_rows = filter_eval_rows_by_split(eval_rows, split_row_ids)

    agents = build_agents(
        eval_rows,
        dry_run=args.dry_run,
        prompt_overrides=prompt_overrides,
    )
    devset = build_devset(contract_lookup, eval_rows, dry_run=args.dry_run)
    if args.context_mode != "raw":
        devset = _apply_rag_context_to_devset(
            devset,
            context_mode=args.context_mode,
            top_k=args.top_k,
            output_dir=args.output_dir,
            chunking_version=args.chunking_version,
            embedding_model=args.embedding_model,
        )
    eval_results = []
    for question_index in sorted(agents):
        agent = agents[question_index]
        print(
            f"Evaluating question {question_index + 1}: "
            f"{agent.category} with {agent.__class__.__name__}",
            flush=True,
        )
        question_devset = [
            example for example in devset if example.question_index == question_index
        ]
        evaluator = dspy.Evaluate(
            devset=question_devset,
            metric=cuad_overlap_metric,
            num_threads=args.num_threads,
            display_progress=True,
            display_table=False,
        )
        eval_results.append(evaluator(agent))
    results = eval_results_to_dataframe(eval_results, model_id=args.model_id)
    summary = summarize_results(
        results,
        args=args,
        selected_document_row_ids=selected_ids,
    )
    write_outputs(results, summary, args.output_dir, args.html_output, agents=agents)


if __name__ == "__main__":
    main()
