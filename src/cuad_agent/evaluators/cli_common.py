"""Shared CLI argument definitions and RAG-context plumbing for evaluators.

The DSPy and LangChain runners orchestrate different frameworks but share the
same evaluation surface: identical CLI flags for sampling/model/output and the
same retrieval-context construction. Those mechanics live here so a change is
made once and both runners stay in sync.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cuad_agent.rag.cache import DEFAULT_EMBEDDING_MODEL
from cuad_agent.rag.experiments import DEFAULT_CHUNKING_VERSION
from cuad_agent.rag.query_enrichment import RAG_DEFAULT_TOP_K

__all__ = [
    "add_common_eval_args",
    "add_rag_context_args",
    "resolve_rag_context_for_row",
]

# Context modes that retrieve sentence chunks instead of using the full
# contract transcript. The hierarchical modes additionally expand parent
# sections around the top leaf sentences.
HIERARCHICAL_CONTEXT_MODES = ("rag-hierarchical-bm25", "rag-hierarchical-dense")


def add_common_eval_args(
    parser: argparse.ArgumentParser,
    *,
    default_model: str,
    default_max_tokens: int,
) -> None:
    """Add the CLI flags shared by every CUAD evaluation runner."""
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=default_max_tokens)
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


def add_rag_context_args(
    parser: argparse.ArgumentParser,
    *,
    context_modes: list[str],
) -> None:
    """Add the retrieval-context flags shared by RAG-capable runners."""
    parser.add_argument(
        "--context-mode",
        choices=context_modes,
        default="raw",
        help=(
            "Context supplied to the agent. 'raw' uses the full contract text; "
            "the rag-* modes use retrieved sentence chunks and require a "
            "prebuilt sentence cache (run rag-eval first)."
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


def resolve_rag_context_for_row(
    *,
    document_row_id: int,
    query: str,
    context_mode: str,
    top_k: int,
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
    hierarchical_leaf_k: int = 50,
    hierarchical_top_sections: int = 5,
) -> str:
    """Return the retrieval-augmented context string for one (doc, query).

    Dispatches to flat or hierarchical retrieval based on ``context_mode``.
    Callers apply the returned string to their own devset row shape.
    """
    from cuad_agent.rag.context_builder import (
        build_hierarchical_rag_context,
        build_rag_context,
    )

    if context_mode in HIERARCHICAL_CONTEXT_MODES:
        context, _ = build_hierarchical_rag_context(
            document_row_id=document_row_id,
            query=query,
            method=context_mode,  # type: ignore[arg-type]
            leaf_k=hierarchical_leaf_k,
            top_sections=hierarchical_top_sections,
            output_dir=output_dir,
            chunking_version=chunking_version,
            embedding_model=embedding_model,
        )
        return context

    method = "dense_sentence" if context_mode == "rag-dense" else context_mode
    context, _ = build_rag_context(
        document_row_id=document_row_id,
        query=query,
        method=method,  # type: ignore[arg-type]
        top_k=top_k,
        output_dir=output_dir,
        chunking_version=chunking_version,
        embedding_model=embedding_model,
    )
    return context
