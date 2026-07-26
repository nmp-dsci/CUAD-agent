#!/usr/bin/env python3
"""Evaluate all CUAD questions with LangChain over a contract sample.

LangChain port of dspy_eval_v1.py. The CLI, output paths, summary JSON,
results CSV, HTML dashboard, and token-F1 metric are identical. Only the
per-category agent + LLM execution layer is swapped:

| dspy_eval_v1.py                  | langchain_eval_v2.py                  |
|----------------------------------|---------------------------------------|
| dspy.Signature (per category)    | CuadAnswer Pydantic schema (shared)   |
| dspy.ChainOfThought              | reasoning field on CuadAnswer         |
| dspy.LM + dspy.configure         | langchain ChatDeepSeek/ChatOpenAI     |
| dspy.Module / Predict            | LangChain Runnable (prompt | LLM)     |
| dspy.Evaluate(..., num_threads)  | chain.batch(..., max_concurrency)     |
| dspy.Example                     | plain dict[str, Any]                  |

Requires `langchain-deepseek` for DeepSeek models or `langchain-openai` for
OpenAI models.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from cuad_agent.constants import EVAL_QUESTION_INDICES, NO_ANSWER, NO_ANSWER_MARKERS
from cuad_agent.dashboards.evaluation import write_evaluation_html
from cuad_agent.dashboards.model_comparison import write_model_comparison_html
from cuad_agent.eval.examples import answer_texts, evaluation_row_id
from cuad_agent.data.sampling import select_evaluation_set
from cuad_agent.eval.metrics import token_overlap_f1
from cuad_agent.eval.summary import summarize_results
from cuad_agent.paths import output_paths, prompt_name_part, resolve_model_id
from cuad_agent.prompts.loader import load_prompt_overrides, resolve_prompts_file
from cuad_agent.prompts.templates import compose_system_prompt
from cuad_agent.rag.cache import DEFAULT_EMBEDDING_MODEL
from cuad_agent.rag.context_builder import (
    build_hierarchical_rag_context,
    build_rag_context,
)
from cuad_agent.rag.experiments import DEFAULT_CHUNKING_VERSION
from cuad_agent.rag.query_enrichment import (
    RAG_DEFAULT_TOP_K,
    build_question_enrichments,
    query_for_row,
    save_enriched_question_files,
)


DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
_DEFAULT_SINGLE_Q_MODEL_ID = "s6"
DEFAULT_BASELINE_RESULTS = Path("outputs/v1/cuad_dspy_eval_results.csv")
FULL_EVAL_CONTEXT_MODES = (
    "raw",
    "rag-dense",
    "rag-hybrid",
    "rag-hierarchical-bm25",
    "rag-hierarchical-dense",
)
RESULT_SORT_COLUMNS = ["document_row_id", "question_index"]
RESULT_REQUIRED_COLUMNS = {
    "model_id",
    "row_id",
    "document_row_id",
    "title",
    "question_index",
    "category",
    "category_description",
    "answer_format",
    "question",
    "gold_answers",
    "predicted_answer",
    "predicted_marked_impossible",
    "gold_marked_impossible",
    "token_f1",
    "correct_at_0_5",
    "is_impossible",
    "answers_len",
}
RESULT_JSONL_NAME = "cuad_dspy_eval_results.jsonl"


class CuadAnswer(BaseModel):
    """Structured output schema for CUAD clause extraction."""

    reasoning: str = Field(
        description=(
            "Brief step-by-step reasoning that identifies the supporting "
            "clause(s) in the contract before producing the final answer."
        )
    )
    answer: str = Field(
        description=(
            "Exact answer text span(s) from the contract, separated by "
            "newlines if multiple; or NO_ANSWER if no supporting clause exists."
        )
    )
    marked_impossible: bool = Field(
        description=(
            "True only when the contract does not contain an answer for this "
            "category/question; otherwise false."
        )
    )


def build_chain_for_agent(llm: BaseChatModel, system_prompt: str) -> Runnable:
    """Build a per-category chain that turns input fields into a typed CuadAnswer.

    Uses a PydanticOutputParser instead of `with_structured_output()` because the
    latter defaults to OpenAI tool calling, which DeepSeek's reasoner endpoints
    (e.g. deepseek-v4-flash) reject. The parser approach embeds the JSON schema
    in the system prompt and validates the response client-side, so it works on
    any chat model regardless of tool/function-calling support.

    Messages are constructed manually instead of via ChatPromptTemplate so that
    contract text containing literal curly braces does not break f-string-style
    template substitution.
    """
    parser = PydanticOutputParser(pydantic_object=CuadAnswer)
    full_system = f"{system_prompt}\n\n{parser.get_format_instructions()}"

    def make_messages(inputs: dict[str, Any]) -> list[Any]:
        user_content = (
            f"Contract title:\n{inputs['contract_title']}\n\n"
            f"Contract text:\n{inputs['contract_text']}\n\n"
            f"Category:\n{inputs['category']}\n\n"
            f"Category description:\n{inputs['category_description']}\n\n"
            f"Answer format:\n{inputs['answer_format']}"
        )
        return [
            SystemMessage(content=full_system),
            HumanMessage(content=user_content),
        ]

    return RunnableLambda(make_messages) | llm


def message_text(output: Any) -> str:
    content = getattr(output, "content", output)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def parse_cuad_answer(output: Any) -> CuadAnswer:
    text = message_text(output).strip()
    parser = PydanticOutputParser(pydantic_object=CuadAnswer)
    try:
        return parser.parse(text)
    except Exception:
        normalized = text.strip().lower()
        marked_impossible = normalized in NO_ANSWER_MARKERS
        return CuadAnswer(
            reasoning="unstructured_model_output",
            answer=NO_ANSWER if marked_impossible else text,
            marked_impossible=marked_impossible,
        )


class ContractQuestionAgent:
    """One per-category agent wrapping a LangChain chain."""

    def __init__(
        self,
        *,
        question_index: int,
        category: str,
        question: str,
        category_description: str,
        answer_format: str,
        system_prompt: str,
        chain: Runnable | None,
        dry_run: bool,
    ) -> None:
        self.question_index = question_index
        self.category = category
        self.question = question
        self.category_description = category_description
        self.answer_format = answer_format
        self.system_prompt = system_prompt
        self.chain = chain
        self.dry_run = dry_run

    def _dry_run_prediction(self, gold_answers: list[str]) -> CuadAnswer:
        return CuadAnswer(
            reasoning="dry_run",
            answer="\n".join(gold_answers) if gold_answers else NO_ANSWER,
            marked_impossible=not bool(gold_answers),
        )

    def predict_batch(
        self,
        examples: list[dict[str, Any]],
        *,
        max_concurrency: int = 4,
    ) -> list[CuadAnswer]:
        if self.dry_run:
            return [self._dry_run_prediction(ex["gold_answers"]) for ex in examples]
        if self.chain is None:
            raise RuntimeError("LangChain chain not configured for non-dry-run agent")
        chain_inputs = [
            {
                "contract_title": ex["contract_title"],
                "contract_text": ex["contract_text"],
                "category": self.category,
                "category_description": self.category_description,
                "answer_format": self.answer_format,
            }
            for ex in examples
        ]
        raw_outputs = self.chain.batch(
            chain_inputs,
            config={"max_concurrency": max_concurrency},
        )
        return [parse_cuad_answer(output) for output in raw_outputs]


def build_agents(
    questions_df: pd.DataFrame,
    *,
    llm: BaseChatModel | None,
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
        answer_format = "" if pd.isna(row.answer_format) else str(row.answer_format)
        category_description = str(row.category_description)

        system_prompt = prompt_overrides.get(category) or compose_system_prompt(
            question=question,
            category=category,
            category_description=category_description,
            answer_format=answer_format,
        )

        chain: Runnable | None = None
        if not dry_run and llm is not None:
            chain = build_chain_for_agent(llm, system_prompt)

        agents[question_index] = ContractQuestionAgent(
            question_index=question_index,
            category=category,
            question=question,
            category_description=category_description,
            answer_format=answer_format,
            system_prompt=system_prompt,
            chain=chain,
            dry_run=dry_run,
        )
    return agents


def resolve_prompt_harness_paths(args: argparse.Namespace) -> None:
    """Resolve prompt_improve_v2.py harness artifacts into evaluator inputs."""
    harness_dir = getattr(args, "prompt_harness_dir", None)
    if harness_dir is None:
        return

    prompts_file = harness_dir / "prompts_candidate_v2.py"
    if args.prompts_file is None:
        if not prompts_file.exists():
            raise FileNotFoundError(
                f"Prompt harness candidate file not found: {prompts_file}"
            )
        args.prompts_file = prompts_file

    splits_file = harness_dir / "splits.json"
    if (
        args.eval_split is None
        and getattr(args, "use_harness_holdout", True)
        and splits_file.exists()
    ):
        args.eval_split = f"{splits_file}:holdout_eval"


def _apply_rag_context_to_devset(
    devset: list[dict[str, Any]],
    *,
    context_mode: str,
    top_k: int,
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
    hierarchical_leaf_k: int = 50,
    hierarchical_top_sections: int = 5,
) -> list[dict[str, Any]]:
    from cuad_agent.rag.query_enrichment import query_for_row
    import types

    print(
        f"[RAG] Replacing contract_text with {context_mode} context "
        f"(top_k={top_k}) for {len(devset)} examples...",
        flush=True,
    )
    updated: list[dict[str, Any]] = []
    total = len(devset)
    for index, ex in enumerate(devset, start=1):
        row_obj = types.SimpleNamespace(
            category=ex["category"],
            category_description=ex["category_description"],
            question=ex["question"],
        )
        query = query_for_row(row_obj)
        if context_mode in {"rag-hierarchical-bm25", "rag-hierarchical-dense"}:
            rag_context, _ = build_hierarchical_rag_context(
                document_row_id=ex["document_row_id"],
                query=query,
                method=context_mode,  # type: ignore[arg-type]
                leaf_k=hierarchical_leaf_k,
                top_sections=hierarchical_top_sections,
                top_k=top_k,
                output_dir=output_dir,
                chunking_version=chunking_version,
                embedding_model=embedding_model,
            )
        else:
            rag_context, _ = build_rag_context(
                document_row_id=ex["document_row_id"],
                query=query,
                method=context_mode,  # type: ignore[arg-type]
                top_k=top_k,
                output_dir=output_dir,
                chunking_version=chunking_version,
                embedding_model=embedding_model,
            )
        updated.append({**ex, "contract_text": rag_context})
        if index == 1 or index % 50 == 0 or index == total:
            print(
                f"[RAG] Built {context_mode} context {index}/{total}",
                flush=True,
            )
    return updated


def build_devset(
    contract_lookup: dict[int, dict[str, Any]],
    eval_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    ordered = eval_rows.sort_values(["document_row_id", "question_index"])
    for row in ordered.itertuples(index=False):
        contract = contract_lookup[int(row.document_row_id)]
        gold_answers = answer_texts(row.answers)
        row_id = evaluation_row_id(int(row.document_row_id), int(row.question_index))
        examples.append(
            {
                "row_id": row_id,
                "document_row_id": int(row.document_row_id),
                "title": str(contract.get("title", "")),
                "contract_title": str(contract.get("title", "")),
                "contract_text": str(contract.get("context", "")),
                "question_index": int(row.question_index),
                "category": str(row.category),
                "category_description": str(row.category_description),
                "answer_format": (
                    "" if pd.isna(row.answer_format) else str(row.answer_format)
                ),
                "question": str(row.question),
                "gold_answers": gold_answers,
                "gold_marked_impossible": bool(row.is_impossible),
                "answers_len": int(row.answers_len),
            }
        )
    return examples


def evaluate_agent(
    agent: ContractQuestionAgent,
    devset: list[dict[str, Any]],
    *,
    max_concurrency: int = 4,
) -> list[tuple[dict[str, Any], CuadAnswer, float]]:
    examples = [ex for ex in devset if ex["question_index"] == agent.question_index]
    if not examples:
        return []
    predictions = agent.predict_batch(examples, max_concurrency=max_concurrency)
    return [
        (ex, pred, float(token_overlap_f1(pred.answer, ex["gold_answers"])))
        for ex, pred in zip(examples, predictions, strict=True)
    ]


def result_record(
    example: dict[str, Any],
    pred: CuadAnswer,
    score: float,
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    return {
        "model_id": model_id or "",
        "row_id": str(example.get("row_id", "")),
        "document_row_id": int(example["document_row_id"]),
        "title": str(example["title"]),
        "question_index": int(example["question_index"]),
        "category": str(example["category"]),
        "category_description": str(example["category_description"]),
        "answer_format": str(example["answer_format"]),
        "question": str(example["question"]),
        "gold_answers": json.dumps(example["gold_answers"], ensure_ascii=False),
        "predicted_answer": str(pred.answer),
        "predicted_marked_impossible": bool(pred.marked_impossible),
        "gold_marked_impossible": bool(example["gold_marked_impossible"]),
        "token_f1": float(score),
        "correct_at_0_5": float(score) >= 0.5,
        "is_impossible": bool(example["gold_marked_impossible"]),
        "answers_len": int(example["answers_len"]),
    }


def empty_results_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=sorted(RESULT_REQUIRED_COLUMNS))


def eval_results_to_dataframe(
    eval_results: list[list[tuple[dict[str, Any], CuadAnswer, float]]],
    *,
    model_id: str | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for category_results in eval_results:
        for example, pred, score in category_results:
            records.append(result_record(example, pred, score, model_id=model_id))
    if not records:
        return empty_results_dataframe()
    return pd.DataFrame(records).sort_values(RESULT_SORT_COLUMNS).reset_index(drop=True)


def ensure_row_id(results: pd.DataFrame) -> pd.DataFrame:
    if "row_id" in results.columns:
        return results
    results = results.copy()
    results["row_id"] = [
        evaluation_row_id(document_row_id, question_index)
        for document_row_id, question_index in zip(
            results["document_row_id"], results["question_index"], strict=True
        )
    ]
    return results


def eval_row_ids(eval_rows: pd.DataFrame) -> set[str]:
    return {
        evaluation_row_id(document_row_id, question_index)
        for document_row_id, question_index in zip(
            eval_rows["document_row_id"], eval_rows["question_index"], strict=True
        )
    }


def sort_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.reset_index(drop=True)
    return results.sort_values(RESULT_SORT_COLUMNS).reset_index(drop=True)


def load_cached_results(
    results_path: Path,
    *,
    eval_rows: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    if not results_path.exists():
        return empty_results_dataframe()

    cached = ensure_row_id(pd.read_csv(results_path))
    missing = RESULT_REQUIRED_COLUMNS - set(cached.columns)
    if missing:
        raise ValueError(
            f"Existing cache {results_path} is missing required columns: "
            f"{sorted(missing)}"
        )
    cached = cached[cached["model_id"].astype(str) == str(model_id)]
    cached = cached[cached["row_id"].isin(eval_row_ids(eval_rows))]
    if cached.empty:
        return empty_results_dataframe()
    cached = cached.drop_duplicates("row_id", keep="last")
    return sort_results(cached)


def load_jsonl_results(
    jsonl_path: Path,
    *,
    eval_rows: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    if not jsonl_path.exists():
        return empty_results_dataframe()

    records: list[dict[str, Any]] = []
    valid_row_ids = eval_row_ids(eval_rows)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(record.get("model_id", "")) != str(model_id):
                continue
            if str(record.get("row_id", "")) not in valid_row_ids:
                continue
            records.append(record)
    if not records:
        return empty_results_dataframe()

    results = pd.DataFrame(records)
    missing = RESULT_REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(
            f"Existing JSONL cache {jsonl_path} is missing required columns: "
            f"{sorted(missing)}"
        )
    return sort_results(results.drop_duplicates("row_id", keep="last"))


def merge_result_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return empty_results_dataframe()
    merged = pd.concat(non_empty, ignore_index=True)
    merged = ensure_row_id(merged)
    merged = merged.drop_duplicates("row_id", keep="last")
    return sort_results(merged)


def write_results_cache(results: pd.DataFrame, results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    sort_results(results).to_csv(results_path, index=False)


def append_result_jsonl(
    jsonl_path: Path,
    record: dict[str, Any],
    lock: threading.Lock,
) -> None:
    payload = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(payload)


def build_baseline_comparison(
    current_results: pd.DataFrame,
    baseline_results_path: Path | None,
) -> dict[str, Any] | None:
    if baseline_results_path is None:
        return None
    if not baseline_results_path.exists():
        return None

    baseline = ensure_row_id(pd.read_csv(baseline_results_path))
    current = ensure_row_id(current_results)
    required = {"row_id", "question_index", "category", "token_f1", "correct_at_0_5"}
    missing_baseline = required - set(baseline.columns)
    missing_current = required - set(current.columns)
    if missing_baseline or missing_current:
        raise ValueError(
            "Cannot compare baseline results; missing columns: "
            f"baseline={sorted(missing_baseline)}, current={sorted(missing_current)}"
        )

    joined = current.merge(
        baseline[
            [
                "row_id",
                "token_f1",
                "correct_at_0_5",
                "predicted_answer",
                "predicted_marked_impossible",
            ]
        ].rename(
            columns={
                "token_f1": "baseline_token_f1",
                "correct_at_0_5": "baseline_correct_at_0_5",
                "predicted_answer": "baseline_predicted_answer",
                "predicted_marked_impossible": "baseline_predicted_marked_impossible",
            }
        ),
        on="row_id",
        how="inner",
    )
    if joined.empty:
        raise ValueError(
            f"Baseline results {baseline_results_path} did not overlap current eval rows"
        )

    joined["token_f1_delta"] = joined["token_f1"] - joined["baseline_token_f1"]
    joined["correct_at_0_5_delta"] = joined["correct_at_0_5"].astype(float) - joined[
        "baseline_correct_at_0_5"
    ].astype(float)
    baseline_model_id = None
    if "model_id" in baseline.columns and not baseline["model_id"].dropna().empty:
        baseline_model_id = str(baseline["model_id"].dropna().iloc[0])
    per_category_df = (
        joined.groupby(["question_index", "category"], as_index=False)
        .agg(
            baseline_mean_token_f1=("baseline_token_f1", "mean"),
            candidate_mean_token_f1=("token_f1", "mean"),
            mean_token_f1_delta=("token_f1_delta", "mean"),
            baseline_correct_at_0_5=("baseline_correct_at_0_5", "mean"),
            candidate_correct_at_0_5=("correct_at_0_5", "mean"),
            correct_at_0_5_delta=("correct_at_0_5_delta", "mean"),
            count=("row_id", "size"),
        )
        .sort_values(["question_index"])
    )
    return {
        "baseline_results_path": str(baseline_results_path),
        "baseline_model_id": baseline_model_id,
        "matched_examples": int(len(joined)),
        "baseline_mean_token_f1": float(joined["baseline_token_f1"].mean() * 100),
        "candidate_mean_token_f1": float(joined["token_f1"].mean() * 100),
        "mean_token_f1_delta": float(joined["token_f1_delta"].mean() * 100),
        "baseline_correct_at_0_5": float(
            joined["baseline_correct_at_0_5"].mean() * 100
        ),
        "candidate_correct_at_0_5": float(joined["correct_at_0_5"].mean() * 100),
        "correct_at_0_5_delta": float(joined["correct_at_0_5_delta"].mean() * 100),
        "per_category": [
            {
                "question_index": int(row.question_index),
                "category": str(row.category),
                "baseline_mean_token_f1": float(row.baseline_mean_token_f1 * 100),
                "candidate_mean_token_f1": float(row.candidate_mean_token_f1 * 100),
                "mean_token_f1_delta": float(row.mean_token_f1_delta * 100),
                "baseline_correct_at_0_5": float(row.baseline_correct_at_0_5 * 100),
                "candidate_correct_at_0_5": float(row.candidate_correct_at_0_5 * 100),
                "correct_at_0_5_delta": float(row.correct_at_0_5_delta * 100),
                "count": int(row.count),
            }
            for row in per_category_df.itertuples(index=False)
        ],
        "examples": [
            {
                "row_id": str(row.row_id),
                "baseline_predicted_answer": str(row.baseline_predicted_answer),
                "baseline_predicted_marked_impossible": bool(
                    row.baseline_predicted_marked_impossible
                ),
                "baseline_token_f1": float(row.baseline_token_f1),
                "baseline_correct_at_0_5": bool(row.baseline_correct_at_0_5),
                "candidate_token_f1": float(row.token_f1),
                "candidate_correct_at_0_5": bool(row.correct_at_0_5),
                "token_f1_delta": float(row.token_f1_delta),
                "correct_at_0_5_delta": float(row.correct_at_0_5_delta),
            }
            for row in joined.itertuples(index=False)
        ],
    }


def configure_llm(args: argparse.Namespace) -> BaseChatModel:
    load_dotenv(Path.home() / ".env")
    model_name = args.model

    if model_name.startswith("deepseek/") or model_name.startswith("deepseek-"):
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError as exc:
            raise RuntimeError(
                "langchain-deepseek is required for DeepSeek models. "
                "Install with: uv add langchain-deepseek"
            ) from exc
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek models")
        actual_name = model_name.removeprefix("deepseek/")
        return ChatDeepSeek(
            model=actual_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=api_key,
        )

    if model_name.startswith("openai/"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-openai is required for OpenAI models. "
                "Install with: uv add langchain-openai"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI models")
        actual_name = model_name.removeprefix("openai/")
        return ChatOpenAI(
            model=actual_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=api_key,
        )

    raise RuntimeError(
        f"Unknown model provider for: {model_name}. "
        "Use 'deepseek/...' or 'openai/...' prefix."
    )


def write_system_prompts(
    agents: dict[int, ContractQuestionAgent],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    lines = [
        '"""Generated CUAD system prompts by category (LangChain).',
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
                f"{prompt_name} = {agent.system_prompt!r}",
                "",
            ]
        )
        mapping_entries.append(f"    {agent.category!r}: {prompt_name},")
    lines.extend(["CATEGORY_SYSTEM_PROMPTS = {", *mapping_entries, "}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    results: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    html_output: Path | None,
    agents: dict[int, ContractQuestionAgent] | None = None,
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
    if "baseline_comparison" in summary:
        (paths["model_dir"] / "baseline_comparison.json").write_text(
            json.dumps(summary["baseline_comparison"], indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
    write_evaluation_html(results, summary, paths["html"])


def print_run_comparison(summaries: list[dict[str, Any]], labels: list[str]) -> None:
    col_w = 22
    separator = "-" * 80
    header = f"{'Category':<35}" + "".join(f"{label:>{col_w}}" for label in labels)
    print(separator)
    print(header)
    print(separator)
    print(
        f"{'OVERALL mean F1':<35}"
        + "".join(
            f"{_format_percent(summary.get('overlap_accuracy_mean_f1')):>{col_w}}"
            for summary in summaries
        )
    )
    print(
        f"{'OVERALL correct@0.5':<35}"
        + "".join(
            f"{_format_percent(summary.get('correct_at_0_5')):>{col_w}}"
            for summary in summaries
        )
    )
    print(separator)

    category_order: list[tuple[int, str]] = []
    seen: set[str] = set()
    for summary in summaries:
        for entry in summary.get("per_category", []):
            category = str(entry["category"])
            if category not in seen:
                seen.add(category)
                category_order.append((int(entry["question_index"]), category))
    category_order.sort()

    lookups = [
        {
            str(entry["category"]): float(entry["mean_token_f1"])
            for entry in summary.get("per_category", [])
        }
        for summary in summaries
    ]
    for _idx, category in category_order:
        print(
            f"{category:<35}"
            + "".join(
                f"{_format_percent(lookup.get(category)):>{col_w}}"
                for lookup in lookups
            )
        )

    print(separator)
    print("Run details:")
    for label, summary in zip(labels, summaries):
        print(
            f"  {label}: model_id={summary.get('model_id', '?')}, "
            f"model={summary.get('model', '?')}, "
            f"context={summary.get('context_mode', 'raw')}, "
            f"n={summary.get('total_examples', '?')}"
            f"{' [dry-run]' if summary.get('dry_run') else ''}"
        )
    print(separator)


def _format_percent(value: Any) -> str:
    if value is None:
        return "   n/a"
    return f"{float(value):6.1f}%"


def make_messages_with_hint(
    inputs: dict[str, Any],
    hint: str,
    full_system: str,
) -> list[Any]:
    """Like the make_messages closure but appends an enrichment hint to the user turn."""
    user_content = (
        f"Contract title:\n{inputs['contract_title']}\n\n"
        f"Contract text:\n{inputs['contract_text']}\n\n"
        f"Category:\n{inputs['category']}\n\n"
        f"Category description:\n{inputs['category_description']}\n\n"
        f"Answer format:\n{inputs['answer_format']}"
        f"\n\nKey terms to look for: {hint}"
    )
    return [SystemMessage(content=full_system), HumanMessage(content=user_content)]


_VARIANT_CTX_LABELS: dict[str, str] = {
    "raw": "raw_ctx",
    "rag-dense": "rag_dense",
    "rag-hybrid": "rag_hybrid",
    "rag-hierarchical-bm25": "rag_hier_bm25",
    "rag-hierarchical-dense": "rag_hier_dense",
}


def run_single_question_variants(
    *,
    contract_id: int,
    question_index: int,
    llm: BaseChatModel | None,
    dry_run: bool,
    question_modes: list[str],
    context_modes: list[str],
    top_k: int,
    output_dir: Path,
    model_id: str,
    prompt_overrides: dict[str, str],
    query_enrichment_provider: str,
    query_enrichment_model: str,
    embedding_model: str,
    chunking_version: str,
    hierarchical_leaf_k: int = 50,
    hierarchical_top_sections: int = 5,
) -> pd.DataFrame:
    """Run all requested (question_mode × context_mode) variants for one (contract, question).

    Returns a DataFrame with one row per variant.
    """
    # Load the single (contract, question) row from the dataset
    selection = select_evaluation_set(
        contract_ids=[contract_id],
        question_indices=[question_index],
    )
    eval_rows = selection.eval_rows
    if eval_rows.empty:
        raise ValueError(
            f"No data found for contract_id={contract_id}, question_index={question_index}"
        )

    row = eval_rows.iloc[0]
    contract = selection.contract_lookup.get(contract_id)
    if contract is None:
        raise ValueError(f"contract_id={contract_id} not found in the dataset")

    contract_title = str(contract.get("title", ""))
    contract_text = str(contract.get("context", ""))
    category = str(row.category)
    category_description = str(row.category_description)
    answer_format = "" if pd.isna(row.answer_format) else str(row.answer_format)
    gold_answers = answer_texts(row.answers)

    # Validate v2 prompt coverage before doing any LLM work
    if category not in prompt_overrides:
        raise ValueError(
            f"Category {category!r} not found in prompt_overrides. "
            "Ensure system_prompts_v2.py covers all 41 categories. "
            f"Available categories: {sorted(prompt_overrides)}"
        )
    system_prompt = prompt_overrides[category]
    parser = PydanticOutputParser(pydantic_object=CuadAnswer)
    full_system = f"{system_prompt}\n\n{parser.get_format_instructions()}"

    # Obtain enrichment (offline fallback when no API key)
    enrichments, _ = build_question_enrichments(
        eval_rows=eval_rows,
        output_dir=output_dir,
        provider=query_enrichment_provider,
        model=query_enrichment_model,
    )
    enrichment = enrichments.get(question_index)
    save_enriched_question_files(
        enrichments, output_dir, provider=query_enrichment_provider
    )

    records: list[dict[str, Any]] = []

    for question_mode in question_modes:
        for context_mode in context_modes:
            # Resolve retrieval query
            if question_mode == "raw":
                retrieval_query = query_for_row(row)
            else:
                retrieval_query = (
                    enrichment.enriched_query if enrichment else query_for_row(row)
                )

            # Resolve contract context (runs even in dry-run for RAG modes)
            if context_mode == "raw":
                resolved_context = contract_text
            elif context_mode in {"rag-hierarchical-bm25", "rag-hierarchical-dense"}:
                resolved_context, _ = build_hierarchical_rag_context(
                    document_row_id=contract_id,
                    query=retrieval_query,
                    method=context_mode,  # type: ignore[arg-type]
                    leaf_k=hierarchical_leaf_k,
                    top_sections=hierarchical_top_sections,
                    top_k=top_k,
                    output_dir=output_dir,
                    chunking_version=chunking_version,
                    embedding_model=embedding_model,
                )
            else:
                resolved_context, _ = build_rag_context(
                    document_row_id=contract_id,
                    query=retrieval_query,
                    method=context_mode,  # type: ignore[arg-type]
                    top_k=top_k,
                    output_dir=output_dir,
                    chunking_version=chunking_version,
                    embedding_model=embedding_model,
                )

            # Enrichment hint: only for enriched-question + raw-context variant
            hint = ""
            if (
                question_mode == "enriched"
                and context_mode == "raw"
                and enrichment is not None
            ):
                hint = enrichment.enrichment_terms

            inputs = {
                "contract_title": contract_title,
                "contract_text": resolved_context,
                "category": category,
                "category_description": category_description,
                "answer_format": answer_format,
            }

            # LLM prediction or dry-run echo
            if dry_run:
                pred = CuadAnswer(
                    reasoning="dry_run",
                    answer="\n".join(gold_answers) if gold_answers else NO_ANSWER,
                    marked_impossible=not bool(gold_answers),
                )
            else:
                if llm is None:
                    raise RuntimeError(
                        "LLM not configured for non-dry-run single-q mode"
                    )
                if hint:
                    messages = make_messages_with_hint(inputs, hint, full_system)
                else:
                    user_content = (
                        f"Contract title:\n{inputs['contract_title']}\n\n"
                        f"Contract text:\n{inputs['contract_text']}\n\n"
                        f"Category:\n{inputs['category']}\n\n"
                        f"Category description:\n{inputs['category_description']}\n\n"
                        f"Answer format:\n{inputs['answer_format']}"
                    )
                    messages = [
                        SystemMessage(content=full_system),
                        HumanMessage(content=user_content),
                    ]
                raw_output = llm.invoke(messages)
                pred = parse_cuad_answer(raw_output)

            score = float(token_overlap_f1(pred.answer, gold_answers))
            q_label = "raw_q" if question_mode == "raw" else "enriched_v1"
            ctx_label = _VARIANT_CTX_LABELS[context_mode]

            records.append(
                {
                    "variant_name": f"{q_label} / {ctx_label}",
                    "question_mode": question_mode,
                    "context_mode": context_mode,
                    "retrieval_query": retrieval_query,
                    "hint_used": hint,
                    "predicted_answer": pred.answer,
                    "gold_answers": json.dumps(gold_answers, ensure_ascii=False),
                    "token_f1": score,
                    "correct_at_0_5": score >= 0.5,
                    "enrichment_terms": enrichment.enrichment_terms
                    if enrichment
                    else "",
                    "document_row_id": contract_id,
                    "question_index": question_index,
                    "category": category,
                }
            )

    return pd.DataFrame(records)


def print_variant_table(df: pd.DataFrame) -> None:
    """Print a side-by-side comparison table for single-question variant runs."""
    if df.empty:
        print("(no variants to display)")
        return

    col_variant = 26
    col_f1 = 10
    col_correct = 13
    col_pred = 28

    header = (
        f"{'Variant':<{col_variant}} | {'Token F1':^{col_f1}} | "
        f"{'Correct@0.5':^{col_correct}} | {'Predicted (first 80 chars)'}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    for row in df.itertuples(index=False):
        pred_preview = str(row.predicted_answer).replace("\n", " ")[:80]
        if len(str(row.predicted_answer).replace("\n", " ")) > 80:
            pred_preview += "..."
        f1_str = f"{float(row.token_f1):.2f}"
        correct_str = str(bool(row.correct_at_0_5))
        print(
            f"{str(row.variant_name):<{col_variant}} | {f1_str:^{col_f1}} | "
            f"{correct_str:^{col_correct}} | {pred_preview}"
        )

    print()
    # Footer
    gold = json.loads(df.iloc[0]["gold_answers"])
    gold_text = "; ".join(gold) if gold else "(impossible)"
    category = str(df.iloc[0]["category"])
    doc_id = int(df.iloc[0]["document_row_id"])
    q_idx = int(df.iloc[0]["question_index"])
    print(f"Golden answer: {gold_text}")
    print(f"Category: {category}  |  Contract: {doc_id}  |  Question index: {q_idx}")


def run_full_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke_test:
        args.sample_size = 1
        if args.model_id is None:
            args.model_id = "smoke-test"
        print(
            "=== SMOKE TEST MODE: 1 contract × 41 questions ===",
            flush=True,
        )
    if args.model_id is None:
        args.model_id = "v2"
    args.model_id = resolve_model_id(args)
    if args.num_threads < 1:
        raise ValueError("--num-threads must be >= 1")
    resolve_prompt_harness_paths(args)
    args.prompts_file = resolve_prompts_file(args.prompts_file, args.model_id)

    llm: BaseChatModel | None = None
    if not args.dry_run:
        llm = configure_llm(args)

    selection = select_evaluation_set(
        sample_size=args.sample_size,
        seed=args.seed,
        eval_split=args.eval_split,
    )
    selected_ids = selection.selected_ids
    contract_lookup = selection.contract_lookup
    eval_rows = selection.eval_rows

    prompt_overrides = load_prompt_overrides(args.prompts_file)

    agents = build_agents(
        eval_rows,
        llm=llm,
        dry_run=args.dry_run,
        prompt_overrides=prompt_overrides,
    )
    devset = build_devset(contract_lookup, eval_rows)
    paths = output_paths(args.output_dir, args.model_id, args.html_output)
    jsonl_results_path = paths["model_dir"] / RESULT_JSONL_NAME
    cached_results = (
        merge_result_frames(
            load_cached_results(
                paths["results"],
                eval_rows=eval_rows,
                model_id=args.model_id,
            ),
            load_jsonl_results(
                jsonl_results_path,
                eval_rows=eval_rows,
                model_id=args.model_id,
            ),
        )
        if args.resume_existing
        else empty_results_dataframe()
    )
    completed_row_ids = set(cached_results["row_id"].astype(str))
    if completed_row_ids:
        print(
            f"Loaded {len(completed_row_ids)} cached evaluation row(s) from "
            f"{paths['results']} and {jsonl_results_path}",
            flush=True,
        )
    if getattr(args, "context_mode", "raw") != "raw":
        missing_devset = [
            ex for ex in devset if str(ex["row_id"]) not in completed_row_ids
        ]
        updated_missing_devset = _apply_rag_context_to_devset(
            missing_devset,
            context_mode=args.context_mode,
            top_k=args.top_k,
            output_dir=args.output_dir,
            chunking_version=args.chunking_version,
            embedding_model=args.embedding_model,
            hierarchical_leaf_k=args.hierarchical_leaf_k,
            hierarchical_top_sections=args.hierarchical_top_sections,
        )
        updated_by_row_id = {str(ex["row_id"]): ex for ex in updated_missing_devset}
        devset = [updated_by_row_id.get(str(ex["row_id"]), ex) for ex in devset]

    new_results = empty_results_dataframe()
    jsonl_lock = threading.Lock()
    for question_index in sorted(agents):
        agent = agents[question_index]
        question_examples = [
            ex for ex in devset if ex["question_index"] == agent.question_index
        ]
        missing_examples = [
            ex for ex in question_examples if str(ex["row_id"]) not in completed_row_ids
        ]
        if not missing_examples:
            print(
                f"Skipping question {question_index + 1}: {agent.category} "
                f"({len(question_examples)} cached)",
                flush=True,
            )
            continue
        print(
            f"Evaluating question {question_index + 1}: {agent.category} "
            f"({len(missing_examples)} missing, "
            f"{len(question_examples) - len(missing_examples)} cached)",
            flush=True,
        )
        missing_row_ids = {str(ex["row_id"]) for ex in missing_examples}
        total_contracts = len(question_examples)
        for contract_number, example in enumerate(question_examples, 1):
            if str(example["row_id"]) not in missing_row_ids:
                continue
            print(
                f"  question {question_index + 1}/{len(agents)} "
                f"contract {contract_number}/{total_contracts} "
                f"document_row_id={example['document_row_id']}: "
                f"{example['contract_title']}",
                flush=True,
            )
        question_results = evaluate_agent(
            agent,
            missing_examples,
            max_concurrency=args.num_threads,
        )
        for example, pred, score in question_results:
            append_result_jsonl(
                jsonl_results_path,
                result_record(example, pred, score, model_id=args.model_id),
                jsonl_lock,
            )
        question_frame = eval_results_to_dataframe(
            [question_results],
            model_id=args.model_id,
        )
        new_results = merge_result_frames(new_results, question_frame)
        completed_row_ids.update(question_frame["row_id"].astype(str))
        write_results_cache(
            merge_result_frames(cached_results, new_results),
            paths["results"],
        )

    results = merge_result_frames(cached_results, new_results)
    expected_row_count = len(eval_rows)
    if len(results) != expected_row_count:
        raise RuntimeError(
            f"Evaluation incomplete: have {len(results)} rows, expected "
            f"{expected_row_count}. Re-run with the same model_id to continue."
        )
    summary = summarize_results(
        results,
        args=args,
        selected_document_row_ids=selected_ids,
    )
    if getattr(args, "prompt_harness_dir", None) is not None:
        summary["prompt_harness_dir"] = str(args.prompt_harness_dir)
    if not args.no_baseline_comparison:
        comparison = build_baseline_comparison(results, args.baseline_results)
        if comparison is not None:
            summary["baseline_comparison"] = comparison
    write_outputs(results, summary, args.output_dir, args.html_output, agents=agents)
    return summary


def run_all_context_modes(args: argparse.Namespace) -> None:
    if args.single_q:
        raise ValueError("--all-context-modes cannot be combined with --single-q")
    prefix = args.model_id or "eval"
    summaries: list[dict[str, Any]] = []
    labels: list[str] = []
    models_payload: list[dict[str, Any]] = []
    for context_mode in FULL_EVAL_CONTEXT_MODES:
        mode_args = copy.copy(args)
        mode_args.all_context_modes = False
        mode_args.context_mode = context_mode
        mode_args.model_id = f"{prefix}-{context_mode}"
        mode_args.html_output = None
        print(
            f"\n=== Running context mode {context_mode} "
            f"(model_id={mode_args.model_id}) ===",
            flush=True,
        )
        summary = run_full_evaluation(mode_args)
        summaries.append(summary)
        label = {
            "raw": "raw",
            "rag-dense": "rag-dense",
            "rag-hybrid": "rag-hybrid",
            "rag-hierarchical-bm25": "hier-bm25",
            "rag-hierarchical-dense": "hier-dense",
        }[context_mode]
        labels.append(label)
        results_path = output_paths(args.output_dir, mode_args.model_id, None)[
            "results"
        ]
        models_payload.append(
            {
                "label": label,
                "model_id": mode_args.model_id,
                "context_mode": context_mode,
                "summary": summary,
                "results": pd.read_csv(results_path),
            }
        )
    print_run_comparison(summaries, labels)

    comparison_path = (
        args.output_dir.parent / "dashboards" / f"eval_comparison_{prefix}.html"
    )
    write_model_comparison_html(models_payload, comparison_path)
    print(f"\nSaved model comparison dashboard to {comparison_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="max_concurrency passed to chain.batch (parallel example processing).",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "Stable identifier for this model/config run. Defaults to 'v2' "
            "for this LangChain evaluator. With --all-context-modes, this is "
            "used as the model-id prefix."
        ),
    )
    parser.add_argument(
        "--all-context-modes",
        action="store_true",
        help=(
            "Run the full evaluation set for raw, rag-dense, rag-hybrid, "
            "rag-hierarchical-bm25, and rag-hierarchical-dense, then print a "
            "comparison table."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help=(
            "Optional explicit HTML output path. Defaults to "
            "dashboards/evaluation_MODEL_ID.html for cross-comparability "
            "with dspy_eval_v1.py output. Bare relative filenames are "
            "written under dashboards/."
        ),
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help=(
            "Optional Python prompt module defining CATEGORY_SYSTEM_PROMPTS. "
            "Category prompts override the composed default system prompt."
        ),
    )
    parser.add_argument(
        "--prompt-harness-dir",
        type=Path,
        default=None,
        help=(
            "Optional prompt_improve_v2.py harness directory, e.g. "
            "outputs/v2/prompt_harness. When set, loads prompts_candidate_v2.py "
            "as --prompts-file unless --prompts-file is explicitly provided."
        ),
    )
    parser.add_argument(
        "--use-harness-holdout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --prompt-harness-dir is set and --eval-split is omitted, use "
            "splits.json:holdout_eval from the harness directory. Default: true."
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
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=DEFAULT_BASELINE_RESULTS,
        help=(
            "Optional v1 results CSV to compare against by row_id. Defaults to "
            "outputs/v1/cuad_dspy_eval_results.csv when present."
        ),
    )
    parser.add_argument(
        "--no-baseline-comparison",
        action="store_true",
        help="Disable baseline comparison even if --baseline-results exists.",
    )
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse completed row_id evaluations from the current model_id output "
            "CSV and write incremental progress after each question. Default: true."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run all 41 questions on a single contract. Forces "
            "--sample-size 1 and defaults --model-id to 'smoke-test' if not "
            "set. Use to verify the harness end-to-end before a full run."
        ),
    )

    # Single-question variant comparison mode
    parser.add_argument(
        "--single-q",
        action="store_true",
        help="Run one question for one contract across all requested variants.",
    )
    parser.add_argument("--contract-id", type=int, default=None)
    parser.add_argument("--question-index", type=int, default=None)
    parser.add_argument(
        "--question-mode",
        choices=("raw", "enriched"),
        default="raw",
        help="Whether to use the raw CUAD question or an enriched retrieval query.",
    )
    parser.add_argument(
        "--context-mode",
        choices=FULL_EVAL_CONTEXT_MODES,
        default="raw",
        help="What contract context to pass to the LLM.",
    )
    parser.add_argument(
        "--compare-variants",
        action="store_true",
        help="Run all 10 (question_mode × context_mode) combinations; overrides --question-mode and --context-mode.",
    )
    parser.add_argument("--top-k", type=int, default=RAG_DEFAULT_TOP_K)
    parser.add_argument(
        "--query-enrichment-provider",
        choices=("auto", "llm", "offline"),
        default="auto",
    )
    parser.add_argument("--query-enrichment-model", default="deepseek-chat")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunking-version", default=DEFAULT_CHUNKING_VERSION)
    parser.add_argument("--hierarchical-leaf-k", type=int, default=50)
    parser.add_argument("--hierarchical-top-sections", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_context_modes:
        run_all_context_modes(args)
        return

    if args.single_q:
        if args.contract_id is None or args.question_index is None:
            raise ValueError(
                "--single-q requires both --contract-id and --question-index"
            )
        prompts_file = args.prompts_file or Path("prompts/system_prompts_v2.py")
        if not prompts_file.exists():
            raise FileNotFoundError(
                f"v2 prompts file not found: {prompts_file}\n"
                "Single-question mode requires system_prompts_v2.py. "
                "Run the v2 prompt harness first, or pass --prompts-file."
            )
        prompt_overrides = load_prompt_overrides(prompts_file)

        llm: BaseChatModel | None = None
        if not args.dry_run:
            llm = configure_llm(args)

        question_modes = (
            ["raw", "enriched"] if args.compare_variants else [args.question_mode]
        )
        context_modes = (
            [
                "raw",
                "rag-dense",
                "rag-hybrid",
                "rag-hierarchical-bm25",
                "rag-hierarchical-dense",
            ]
            if args.compare_variants
            else [args.context_mode]
        )
        if args.compare_variants and (
            args.question_mode != "raw" or args.context_mode != "raw"
        ):
            print(
                "Note: --compare-variants overrides --question-mode and --context-mode; "
                "running all 10 combinations.",
                flush=True,
            )
        if args.compare_variants and not args.dry_run:
            print(
                "Running 10 live single-question variants "
                "(2 question modes x 5 context modes).",
                flush=True,
            )

        model_id = args.model_id or _DEFAULT_SINGLE_Q_MODEL_ID
        results_df = run_single_question_variants(
            contract_id=args.contract_id,
            question_index=args.question_index,
            llm=llm,
            dry_run=args.dry_run,
            question_modes=question_modes,
            context_modes=context_modes,
            top_k=args.top_k,
            output_dir=args.output_dir,
            model_id=model_id,
            prompt_overrides=prompt_overrides,
            query_enrichment_provider=args.query_enrichment_provider,
            query_enrichment_model=args.query_enrichment_model,
            embedding_model=args.embedding_model,
            chunking_version=args.chunking_version,
            hierarchical_leaf_k=args.hierarchical_leaf_k,
            hierarchical_top_sections=args.hierarchical_top_sections,
        )
        print_variant_table(results_df)
        out_dir = args.output_dir / model_id / "single_q_variants"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (
            out_dir / f"c{args.contract_id}_q{args.question_index:02d}_variants.csv"
        )
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
        return

    run_full_evaluation(args)


if __name__ == "__main__":
    main()
