"""CLI for sentence-level CUAD RAG evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

from cuad_agent.rag.experiments import DEFAULT_CHUNKING_VERSION, DEFAULT_RETRIEVERS, run_rag_eval


def parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_RETRIEVERS)
    return [part.strip() for part in value.split(",") if part.strip()]


def format_float(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "0.00"


def print_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate sentence-level RAG coverage for CUAD contracts.",
    )
    parser.add_argument("--run-id", default="rag-sentence-v1")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--retrievers",
        default=",".join(DEFAULT_RETRIEVERS),
        help=(
            "Comma-separated retrievers, e.g. bm25_sentence,dense_sentence,"
            "bm25_legal_recursive,dense_legal_recursive,bm25_hierarchical,"
            "dense_hierarchical."
        ),
    )
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--preflight-golden-sentences-only", action="store_true")
    parser.add_argument("--eval-split")
    parser.add_argument("--question-indices")
    parser.add_argument("--contract-ids")
    parser.add_argument("--embedding-model", default="tfidf")
    parser.add_argument(
        "--query-enrichment-provider",
        choices=("auto", "llm", "offline"),
        default="auto",
        help=(
            "Question enrichment provider. 'auto' uses DeepSeek when "
            "DEEPSEEK_API_KEY is set, otherwise deterministic offline terms."
        ),
    )
    parser.add_argument("--query-enrichment-model", default="deepseek-chat")
    parser.add_argument(
        "--chunking-version",
        default=DEFAULT_CHUNKING_VERSION,
        help=(
            "Primary chunking version label. Use legal-recursive-v1 with "
            "bm25_legal_recursive/dense_legal_recursive to compare LangChain "
            "RecursiveCharacterTextSplitter chunks against sentence-v3."
        ),
    )
    parser.add_argument(
        "--contract-scope",
        choices=("all", "eval-set"),
        default="all",
        help=(
            "Contracts to sentence-chunk/cache: 'all' chunks every CUAD "
            "contract; 'eval-set' chunks only the selected evaluation contracts."
        ),
    )
    legacy_chunk_group = parser.add_mutually_exclusive_group()
    legacy_chunk_group.add_argument(
        "--chunk-all-contracts",
        dest="legacy_chunk_all_contracts",
        action="store_true",
        default=None,
        help="Deprecated alias for --contract-scope all.",
    )
    legacy_chunk_group.add_argument(
        "--chunk-eval-contracts",
        dest="legacy_chunk_all_contracts",
        action="store_false",
        help="Deprecated alias for --contract-scope eval-set.",
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--rebuild-chunks", action="store_true")
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument(
        "--hierarchical-leaf-k",
        type=int,
        default=50,
        help="Number of leaf sentences to retrieve before section expansion.",
    )
    parser.add_argument(
        "--hierarchical-top-sections",
        type=int,
        default=5,
        help="Number of top sections to expand into for hierarchical retrieval.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging and print only the final summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    contract_scope = args.contract_scope
    if args.legacy_chunk_all_contracts is True:
        contract_scope = "all"
    elif args.legacy_chunk_all_contracts is False:
        contract_scope = "eval-set"
    summary = run_rag_eval(
        run_id=args.run_id,
        sample_size=args.sample_size,
        seed=args.seed,
        retrievers=parse_str_list(args.retrievers),
        top_k=args.top_k,
        output_dir=args.output_dir,
        preflight_golden_sentences_only=args.preflight_golden_sentences_only,
        eval_split=args.eval_split,
        question_indices=parse_int_list(args.question_indices),
        contract_ids=parse_int_list(args.contract_ids),
        embedding_model=args.embedding_model,
        query_enrichment_provider=args.query_enrichment_provider,
        query_enrichment_model=args.query_enrichment_model,
        chunking_version=args.chunking_version,
        contract_scope=contract_scope,
        rebuild_chunks=args.rebuild_chunks,
        rebuild_embeddings=args.rebuild_embeddings,
        hierarchical_leaf_k=args.hierarchical_leaf_k,
        hierarchical_top_sections=args.hierarchical_top_sections,
        progress=None if args.quiet else print_progress,
    )
    print("RAG run complete")
    print(f"  run_id: {summary.get('run_id')}")
    print(f"  contracts chunked: {summary.get('chunked_contract_count', 0)}")
    print(f"  total sentences: {summary.get('sentence_count', 0)}")
    print(
        "  average sentences per contract: "
        f"{format_float(summary.get('average_sentences_per_contract'))}"
    )
    print(f"  eligible rows: {summary.get('eligible_rows', 0)}")
    print(f"  retrieval rows: {summary.get('retrieval_rows', 0)}")
    print(f"  elapsed seconds: {summary.get('elapsed_seconds')}")


if __name__ == "__main__":
    main()
