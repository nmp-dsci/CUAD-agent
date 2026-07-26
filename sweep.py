"""Run autoresearch.py for every question index whose baseline accuracy is below a threshold.

Usage example (mirrors your autoresearch.py command):

    uv run python sweep.py \
        --model-id eval-raw \
        --prompts-file prompts/system_prompts_v2.py \
        --sample-size 50 \
        --max-iterations 3

Optional flags:
    --accuracy-threshold  Stop iterating question indices at or above this value (default: 0.9)
    --output-dir          Where model outputs live (default: outputs)
    --dry-run             Pass --dry-run through to autoresearch.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-id", required=True, help="Baseline model_id (e.g. eval-raw)"
    )
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.9,
        help="Run autoresearch for question indices strictly below this accuracy (default: 0.9)",
    )
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--triage-model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--synthesis-model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--context-mode", default="raw")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run autoresearch even for questions that already have a results.tsv",
    )
    return parser.parse_args(argv)


def _load_per_question_accuracy(output_dir: Path, model_id: str) -> dict[int, float]:
    """Return {question_index: mean_correct_at_0_5} from the baseline results CSV."""
    results_csv = output_dir / model_id / "cuad_langchain_eval_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Baseline results not found: {results_csv}\n"
            "Run a full evaluation first:\n"
            f"  uv run python agent.py --model-id {model_id} --sample-size <N>"
        )
    df = pd.read_csv(results_csv)
    return df.groupby("question_index")["correct_at_0_5"].mean().to_dict()


def _find_existing_run(output_dir: Path, question_index: int) -> Path | None:
    """Return the results.tsv of the most recent completed autoresearch run, or None."""
    base = output_dir / "autoresearch" / f"q{question_index:02d}"
    if not base.is_dir():
        return None
    candidates = sorted(
        p for p in base.iterdir() if p.is_dir() and (p / "results.tsv").exists()
    )
    return (candidates[-1] / "results.tsv") if candidates else None


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    accuracy_by_q = _load_per_question_accuracy(args.output_dir, args.model_id)

    total_questions = 41  # CUAD has 41 question categories
    missing = sorted(set(range(total_questions)) - set(accuracy_by_q))
    if missing:
        print(
            f"WARNING: {len(missing)} question index/indices have no baseline data "
            f"and will be skipped: {missing}",
            flush=True,
        )

    targets = sorted(
        q_idx for q_idx, acc in accuracy_by_q.items() if acc < args.accuracy_threshold
    )

    if not targets:
        print(
            f"No question indices below accuracy threshold {args.accuracy_threshold:.2f}. Nothing to do.",
            flush=True,
        )
        return

    print(
        f"Found {len(targets)} question index/indices with accuracy < {args.accuracy_threshold:.2f}:",
        flush=True,
    )
    for q in targets:
        print(f"  q{q:02d}  accuracy={accuracy_by_q[q]:.3f}", flush=True)
    print(flush=True)

    failed: list[int] = []
    skipped: list[int] = []
    for i, q_idx in enumerate(targets, 1):
        acc = accuracy_by_q[q_idx]
        print(
            f"[sweep {i}/{len(targets)}] question_index={q_idx}  accuracy={acc:.3f}",
            flush=True,
        )
        if not args.force:
            existing = _find_existing_run(args.output_dir, q_idx)
            if existing is not None:
                print(
                    f"[sweep] Skipping q{q_idx:02d} — already processed ({existing}). "
                    "Use --force to re-run.",
                    flush=True,
                )
                skipped.append(q_idx)
                continue
        cmd = [
            "uv",
            "run",
            "python",
            "autoresearch.py",
            "--model-id",
            args.model_id,
            "--prompts-file",
            str(args.prompts_file),
            "--question-index",
            str(q_idx),
            "--sample-size",
            str(args.sample_size),
            "--seed",
            str(args.seed),
            "--max-iterations",
            str(args.max_iterations),
            "--output-dir",
            str(args.output_dir),
            "--model",
            args.model,
            "--triage-model",
            args.triage_model,
            "--synthesis-model",
            args.synthesis_model,
            "--context-mode",
            args.context_mode,
            "--round",
            str(args.round),
        ]
        if args.dry_run:
            cmd.append("--dry-run")

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(
                f"[sweep] autoresearch.py failed for question_index={q_idx} "
                f"(rc={proc.returncode}) — continuing with remaining questions.",
                flush=True,
            )
            failed.append(q_idx)

    print(flush=True)
    ran = len(targets) - len(skipped)
    if skipped:
        print(
            f"[sweep] Skipped {len(skipped)} already-processed question(s): {[f'q{q:02d}' for q in skipped]}",
            flush=True,
        )
    if failed:
        print(f"[sweep] DONE — {len(failed)} question(s) failed: {failed}", flush=True)
        sys.exit(1)
    else:
        print(
            f"[sweep] DONE — {ran} question(s) ran, {len(skipped)} skipped.", flush=True
        )


if __name__ == "__main__":
    main()
