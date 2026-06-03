#!/usr/bin/env python3
"""Evaluate all CUAD questions with DSPy over a contract sample."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

os.environ.setdefault(
    "DSPY_CACHEDIR", str(Path(__file__).resolve().parent / ".dspy_cache")
)
os.environ.setdefault("LITELLM_MERGE_REASONING_CONTENT_IN_CHOICES", "true")

import dspy

from cuad_agent.constants import (
    EVAL_QUESTION_COUNT,
    EVAL_QUESTION_INDICES,
    NO_ANSWER,
    NO_ANSWER_MARKERS,
    OUTPUT_STEM,
    QUESTION_COUNT,
)
from cuad_agent.dashboards.evaluation import (
    build_evaluation_page_data,
    parse_gold_answers,
    render_evaluation_html,
    write_evaluation_html,
)
from cuad_agent.data.sampling import (
    build_eval_sample,
    evaluation_row_id,
    filter_eval_rows_by_split,
    load_eval_split_ids,
)
from cuad_agent.eval.examples import answer_texts
from cuad_agent.eval.metrics import (
    cuad_overlap_metric,
    normalize_answer,
    parse_bool,
    token_overlap_f1,
    tokens,
)
from cuad_agent.eval.summary import (
    detection_metrics,
    predicted_no_answer_mask,
    summarize_results,
)
from cuad_agent.evaluators.cli_common import (
    add_common_eval_args,
    add_rag_context_args,
    resolve_rag_context_for_row,
)
from cuad_agent.paths import (
    class_name_part,
    output_paths,
    prompt_name_part,
    resolve_model_id,
    slugify_model_id,
)
from cuad_agent.prompts.loader import (
    load_prompt_overrides,
    resolve_prompts_file_for_model_id,
)
from cuad_agent.prompts.templates import (
    build_agent_system_prompt,
    compose_system_prompt,
)
from cuad_agent.prompts.writer import write_system_prompts
from cuad_agent.rag.query_enrichment import query_for_row
from cuad_agent.agents.dspy_agent import (
    ContractQuestionAgent,
    ContractQuestionAgentBase,
    build_agents,
    configure_lm,
    make_question_agent_class,
    make_question_signature_class,
)

# The framework-neutral helpers now live in the service layer (cuad_agent.paths,
# cuad_agent.eval.*, cuad_agent.prompts.*, cuad_agent.dashboards.evaluation,
# cuad_agent.data.sampling). They are re-exported here for backwards
# compatibility with code that historically imported them from this runner.
__all__ = [
    # DSPy-specific (defined in this module)
    "ContractQuestionAgent",
    "ContractQuestionAgentBase",
    "build_agents",
    "build_devset",
    "configure_lm",
    "eval_results_to_dataframe",
    "main",
    "make_question_agent_class",
    "make_question_signature_class",
    "parse_args",
    "write_outputs",
    # Re-exported constants (back-compat)
    "EVAL_QUESTION_COUNT",
    "EVAL_QUESTION_INDICES",
    "NO_ANSWER",
    "NO_ANSWER_MARKERS",
    "OUTPUT_STEM",
    "QUESTION_COUNT",
    # Re-exported service-layer helpers (back-compat)
    "answer_texts",
    "build_agent_system_prompt",
    "build_eval_sample",
    "build_evaluation_page_data",
    "class_name_part",
    "compose_system_prompt",
    "cuad_overlap_metric",
    "detection_metrics",
    "evaluation_row_id",
    "filter_eval_rows_by_split",
    "load_eval_split_ids",
    "load_prompt_overrides",
    "normalize_answer",
    "output_paths",
    "parse_gold_answers",
    "parse_bool",
    "predicted_no_answer_mask",
    "prompt_name_part",
    "render_evaluation_html",
    "resolve_model_id",
    "resolve_prompts_file_for_model_id",
    "slugify_model_id",
    "summarize_results",
    "token_overlap_f1",
    "tokens",
    "write_evaluation_html",
    "write_system_prompts",
]

DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


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
    print(
        f"[RAG] Replacing contract_text with {context_mode} context "
        f"(top_k={top_k}) for {len(devset)} examples...",
        flush=True,
    )
    updated: list[dspy.Example] = []
    for ex in devset:
        rag_context = resolve_rag_context_for_row(
            document_row_id=ex.document_row_id,
            query=query_for_row(ex),
            context_mode=context_mode,
            top_k=top_k,
            output_dir=output_dir,
            chunking_version=chunking_version,
            embedding_model=embedding_model,
        )
        updated.append(ex.copy(contract_text=rag_context))
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_eval_args(
        parser, default_model=DEFAULT_MODEL, default_max_tokens=64000
    )
    add_rag_context_args(parser, context_modes=["raw", "rag-dense", "rag-hybrid"])
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
