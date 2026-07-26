"""Orchestration loop for autoresearch: iterative prompt optimisation for CUAD."""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from cuad_agent.autoresearch.report import write_iter_report, write_progress_report
from cuad_agent.autoresearch.results import (
    SynthesisResult,
    TriageDiagnosis,
    write_prompt_module,
    write_tsv_row,
)
from cuad_agent.autoresearch.synthesis import synthesise
from cuad_agent.autoresearch.triage import triage
from cuad_agent.constants import OUTPUT_STEM
from cuad_agent.data.dataset import load_datasets
from cuad_agent.data.sampling import select_evaluation_set
from cuad_agent.prompts.loader import load_prompt_overrides

__all__ = ["main"]

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

_R = "\033[0m"  # reset
_B = "\033[1m"  # bold
_EVAL = "\033[96m"  # bright cyan  — eval / measurement
_TRIAGE = "\033[93m"  # bright yellow — triage / diagnosis
_SYNTHESIS = "\033[95m"  # bright magenta — synthesis / generation
_KEEP = "\033[92m"  # bright green  — keep / improvement
_DISCARD = "\033[91m"  # bright red    — discard / crash
_HEADER = "\033[1m\033[94m"  # bold bright blue — section header


def _section(label: str) -> None:
    print(f"\n{_HEADER}{'─' * 60}{_R}", flush=True)
    print(f"{_HEADER}  {label}{_R}", flush=True)
    print(f"{_HEADER}{'─' * 60}{_R}", flush=True)


def _log(colour: str, msg: str) -> None:
    print(f"{colour}{msg}{_R}", flush=True)


def _print_eval_contracts(csv_path: Path, question_index: int) -> None:
    """Print per-contract correct/incorrect for *question_index* from a results CSV."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    q_df = df[df["question_index"] == question_index].reset_index(drop=True)
    if q_df.empty:
        return
    n = len(q_df)
    for i, row in enumerate(q_df.itertuples(), 1):
        correct = bool(row.correct_at_0_5)
        tick = "✓" if correct else "✗"
        colour = _KEEP if correct else _DISCARD
        title_col = (
            "title"
            if hasattr(row, "title")
            else "contract_title"
            if hasattr(row, "contract_title")
            else None
        )
        title = str(getattr(row, title_col, ""))[:55] if title_col else ""
        doc_id = getattr(row, "document_row_id", "?")
        _log(colour, f"  {tick} contract {i}/{n}  id={doc_id}  {title}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_accuracy(csv_path: Path, question_index: int) -> float:
    """Return mean correct_at_0_5 for *question_index* from a results CSV."""
    df = pd.read_csv(csv_path)
    rows = df[df["question_index"] == question_index]
    if rows.empty:
        return 0.0
    return float(rows["correct_at_0_5"].mean())


def _run_agent(
    *,
    model_id: str,
    prompts_file: Path,
    question_index: int,
    sample_size: int,
    seed: int,
    context_mode: str,
    output_dir: Path,
    dry_run: bool,
    model: str,
) -> subprocess.CompletedProcess:
    """Run agent.py as a subprocess and return the CompletedProcess."""
    cmd = [
        "uv",
        "run",
        "python",
        "agent.py",
        "--context-mode",
        context_mode,
        "--model-id",
        model_id,
        "--question-index",
        str(question_index),
        "--prompts-file",
        str(prompts_file),
        "--sample-size",
        str(sample_size),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--no-baseline-comparison",
        "--no-resume-existing",
        "--model",
        model,
    ]
    if dry_run:
        cmd.append("--dry-run")
    return subprocess.run(cmd)  # check=False — caller handles returncode


def _copy_eval_outputs(src_model_dir: Path, dest_dir: Path) -> None:
    """Copy eval results CSV and summary JSON from a model_dir to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    results_src = src_model_dir / f"{OUTPUT_STEM}_results.csv"
    summary_src = src_model_dir / f"{OUTPUT_STEM}_summary.json"
    if results_src.exists():
        shutil.copy2(results_src, dest_dir / "eval_results.csv")
    if summary_src.exists():
        shutil.copy2(summary_src, dest_dir / "eval_summary.json")


def _copy_candidate_outputs(src_model_dir: Path, dest_dir: Path) -> None:
    """Copy eval results CSV and summary JSON from a candidate model_dir to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    results_src = src_model_dir / f"{OUTPUT_STEM}_results.csv"
    summary_src = src_model_dir / f"{OUTPUT_STEM}_summary.json"
    if results_src.exists():
        shutil.copy2(results_src, dest_dir / "candidate_results.csv")
    if summary_src.exists():
        shutil.copy2(summary_src, dest_dir / "candidate_summary.json")


def _load_history_from_tsv(tsv_path: Path) -> list[dict]:
    """Load non-baseline TSV rows as history dicts (including prompt_text)."""
    if not tsv_path.exists():
        return []
    df = pd.read_csv(tsv_path, sep="\t")
    history: list[dict] = []
    for _, row in df.iterrows():
        if str(row.get("status", "")) == "baseline":
            continue
        prompt_file = str(row.get("prompt_file", ""))
        prompt_text = ""
        if prompt_file:
            p = Path(prompt_file)
            if p.exists():
                try:
                    category = str(row.get("category", ""))
                    overrides = load_prompt_overrides(p)
                    prompt_text = overrides.get(category, "")
                except Exception:
                    pass
        history.append(
            {
                "iter": int(row.get("iter", 0)),
                "status": str(row.get("status", "")),
                "notes": str(row.get("notes", "")),
                "prompt_text": prompt_text,
            }
        )
    return history


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-index", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--triage-model", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--synthesis-model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-mode", default="raw")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=10)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Resolve category from question_index
    # ------------------------------------------------------------------
    datasets = load_datasets()
    questions = datasets["questions"]
    q_row = questions[questions["question_index"] == args.question_index]
    if q_row.empty:
        raise ValueError(f"No question found for question_index={args.question_index}")
    category = str(q_row.iloc[0]["category"])
    question_text = str(q_row.iloc[0].get("question", "")) if not q_row.empty else ""

    # ------------------------------------------------------------------
    # Output directory layout
    # ------------------------------------------------------------------
    date_str = datetime.date.today().strftime("%Y%m%d")
    run_dir = (
        args.output_dir / "autoresearch" / f"q{args.question_index:02d}" / date_str
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = run_dir / "results.tsv"
    tsv_path.unlink(missing_ok=True)  # start fresh — don't append across runs

    # Clear stale triage outputs from any previous run in this date directory.
    # Triage caching is only valid within a single run; across runs the eval CSV
    # (and therefore which contracts are wrong) may differ.
    for _stale in run_dir.glob("iter_*/triage_outputs.jsonl"):
        _stale.unlink(missing_ok=True)
    dashboards_dir = Path("dashboards")
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    progress_html_path = (
        dashboards_dir / f"autoresearch-{args.model_id}-q{args.question_index:02d}.html"
    )

    # ------------------------------------------------------------------
    # Derive the "seed prefix" from the prompts-file stem
    # e.g. system_prompts_v2.py -> "v2"
    # ------------------------------------------------------------------
    stem = args.prompts_file.stem  # e.g. "system_prompts_v2"
    parts = stem.split("_")
    # Use the last part if it starts with a letter (version prefix like "v2")
    if parts and parts[-1] and parts[-1][0].isalpha():
        version_prefix = parts[-1]
    else:
        version_prefix = stem

    # ------------------------------------------------------------------
    # Iter-0: baseline eval
    # ------------------------------------------------------------------
    iter0_dir = run_dir / "iter_0"
    iter0_dir.mkdir(parents=True, exist_ok=True)

    baseline_model_dir = args.output_dir / args.model_id
    baseline_results_src = baseline_model_dir / f"{OUTPUT_STEM}_results.csv"

    _section(f"ITER 0 — BASELINE EVAL  (model_id={args.model_id})")

    # Determine the exact contract IDs this run requires (question × contract level).
    # Only reuse the cache if every required (question_index, document_row_id) pair is present.
    _selection = select_evaluation_set(sample_size=args.sample_size, seed=args.seed)
    _required_ids = set(_selection.selected_ids)

    _needs_fresh_baseline = True
    if baseline_results_src.exists():
        # Reject cache if it was produced by a dry-run but we're doing a real run
        _summary_src = baseline_model_dir / f"{OUTPUT_STEM}_summary.json"
        _cached_is_dry = False
        if _summary_src.exists():
            try:
                _cached_is_dry = bool(
                    json.loads(_summary_src.read_text()).get("dry_run")
                )
            except Exception:
                pass
        if _cached_is_dry and not args.dry_run:
            _log(
                _DISCARD,
                "[eval] Cached baseline is from a dry-run — running fresh real eval",
            )
        else:
            _cached_df = pd.read_csv(baseline_results_src)
            _cached_ids = set(
                _cached_df[_cached_df["question_index"] == args.question_index][
                    "document_row_id"
                ].astype(int)
            )
            _missing = _required_ids - _cached_ids
            if not _missing:
                _needs_fresh_baseline = False
            else:
                _log(
                    _DISCARD,
                    f"[eval] Cached baseline missing {len(_missing)} contract(s) for "
                    f"q{args.question_index} — running fresh baseline eval",
                )

    if not _needs_fresh_baseline:
        _copy_eval_outputs(baseline_model_dir, iter0_dir)
        baseline_notes = "reused existing eval"
        _log(_EVAL, f"[eval] Reusing cached baseline from {baseline_results_src}")
        _print_eval_contracts(iter0_dir / "eval_results.csv", args.question_index)
    else:
        _log(_EVAL, f"[eval] Running fresh baseline  model_id={args.model_id}")
        proc = _run_agent(
            model_id=args.model_id,
            prompts_file=args.prompts_file,
            question_index=args.question_index,
            sample_size=args.sample_size,
            seed=args.seed,
            context_mode=args.context_mode,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            model=args.model,
        )
        if proc.returncode != 0:
            _log(
                _DISCARD,
                f"[eval] Baseline agent.py failed (rc={proc.returncode}) — continuing with accuracy=0.0",
            )
        _copy_eval_outputs(baseline_model_dir, iter0_dir)
        _print_eval_contracts(iter0_dir / "eval_results.csv", args.question_index)
        baseline_notes = "ran fresh eval"

    # Read baseline accuracy
    iter0_results_csv = iter0_dir / "eval_results.csv"
    if iter0_results_csv.exists():
        baseline_accuracy = _read_accuracy(iter0_results_csv, args.question_index)
    else:
        # Dry-run or crash: stub with 0.0
        baseline_accuracy = 0.0

    # Current state — updated as improvements are accepted
    current_prompt_path: Path = args.prompts_file
    current_prompt_text: str = load_prompt_overrides(current_prompt_path).get(
        category, ""
    )
    current_model_id: str = args.model_id
    current_correct_at_0_5: float = baseline_accuracy
    current_eval_results_path: Path = iter0_results_csv

    # TSV row tracker (for progress reports)
    tsv_rows: list[dict] = []
    # Eval DataFrames keyed by iter number (0 = baseline)
    eval_dfs: dict[int, pd.DataFrame] = {}
    # (before_prompt_text, candidate_prompt_text) keyed by iter number
    iter_prompts: dict[int, tuple[str, str]] = {}

    _STUB_COLUMNS = [
        "document_row_id",
        "contract_title",
        "predicted_answer",
        "golden_answer",
        "correct_at_0_5",
    ]

    def _load_filtered_df(csv_path: Path) -> pd.DataFrame:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df[df["question_index"] == args.question_index]
            if not df.empty:
                if "contract_title" not in df.columns and "title" in df.columns:
                    df = df.rename(columns={"title": "contract_title"})
                if "golden_answer" not in df.columns and "gold_answers" in df.columns:
                    df = df.rename(columns={"gold_answers": "golden_answer"})
                return df
        return pd.DataFrame(columns=_STUB_COLUMNS)

    baseline_row: dict = {
        "iter": 0,
        "question_index": args.question_index,
        "category": category,
        "model_id": current_model_id,
        "prompt_file": str(current_prompt_path),
        "correct_at_0_5": current_correct_at_0_5,
        "n_wrong": 0,
        "n_diagnosed": 0,
        "status": "baseline",
        "notes": baseline_notes,
    }
    write_tsv_row(tsv_path, baseline_row)
    tsv_rows.append(baseline_row)

    # Stash baseline eval df for progress report
    if iter0_results_csv.exists():
        eval_dfs[0] = _load_filtered_df(iter0_results_csv)

    # Write baseline-only progress report so the dashboard exists even if we stop early
    write_progress_report(
        rows=tsv_rows,
        category=category,
        question_index=args.question_index,
        output_path=progress_html_path,
        eval_dfs=eval_dfs,
        question=question_text,
        iter_prompts=iter_prompts,
        run_dir=run_dir,
    )

    # ------------------------------------------------------------------
    # Iteration loop
    # ------------------------------------------------------------------
    for iter_n in range(1, args.max_iterations + 1):
        iter_dir = run_dir / f"iter_{iter_n}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        candidate_model_id = f"{args.model_id}_ar_r{args.round}_i{iter_n}"

        # Snapshot pre-iteration state for reports
        pre_iter_accuracy: float = current_correct_at_0_5
        pre_iter_eval_path: Path = current_eval_results_path

        if current_correct_at_0_5 >= 1.0:
            _log(
                _KEEP,
                f"\n[autoresearch] Accuracy is already 1.0 — stopping after iter {iter_n - 1}.",
            )
            break

        _section(f"ITER {iter_n}/{args.max_iterations} — {candidate_model_id}")

        # ------------------------------------------------------------------
        # 1. TRIAGE
        # ------------------------------------------------------------------
        _log(
            _TRIAGE,
            f"[triage] Diagnosing wrong answers in {current_eval_results_path.name} ...",
        )
        diagnoses: list[TriageDiagnosis] = triage(
            eval_results_path=current_eval_results_path,
            question_index=args.question_index,
            category=category,
            prompts_file=current_prompt_path,
            model_id=current_model_id,
            model=args.triage_model,
            output_path=iter_dir / "triage_outputs.jsonl",
            dry_run=args.dry_run,
        )
        _log(_TRIAGE, f"[triage] {len(diagnoses)} diagnosis(es) produced.")
        for d in diagnoses:
            _log(
                _TRIAGE,
                f"  · contract {d.contract_id}  [{d.confidence}]  {d.failure_reason[:80]}",
            )

        # ------------------------------------------------------------------
        # 2. SYNTHESISE
        # ------------------------------------------------------------------
        history = _load_history_from_tsv(tsv_path)
        current_prompt_text = load_prompt_overrides(current_prompt_path).get(
            category, ""
        )

        _log(_SYNTHESIS, "[synthesis] Generating candidate prompt ...")
        result: SynthesisResult = synthesise(
            category=category,
            current_prompt=current_prompt_text,
            diagnoses=diagnoses,
            history=history,
            model_id=candidate_model_id,
            model=args.synthesis_model,
            dry_run=args.dry_run,
        )
        _log(_SYNTHESIS, f"[synthesis] {result.notes}")

        # Capture before/after for progress report diff
        iter_prompts[iter_n] = (current_prompt_text, result.prompt_text)

        # ------------------------------------------------------------------
        # 3. WRITE candidate.py
        # ------------------------------------------------------------------
        candidate_py = iter_dir / "candidate.py"
        write_prompt_module(candidate_py, category, result.prompt_text)

        # ------------------------------------------------------------------
        # 4. VALIDATE: run agent.py with candidate prompt (retry once)
        # ------------------------------------------------------------------
        status = "discard"
        candidate_accuracy = 0.0
        candidate_results_csv: Path | None = None
        crash_notes = ""

        def _run_candidate() -> tuple[bool, Path | None]:
            """Run agent.py for candidate; return (success, results_csv_path)."""
            nonlocal candidate_accuracy
            _log(_EVAL, f"[eval] Running candidate  model_id={candidate_model_id} ...")
            proc = _run_agent(
                model_id=candidate_model_id,
                prompts_file=candidate_py,
                question_index=args.question_index,
                sample_size=args.sample_size,
                seed=args.seed,
                context_mode=args.context_mode,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
                model=args.model,
            )
            if proc.returncode != 0:
                return False, None
            cand_dir = args.output_dir / candidate_model_id
            csv = cand_dir / f"{OUTPUT_STEM}_results.csv"
            if not csv.exists():
                return False, None
            return True, csv

        success, cand_csv = _run_candidate()
        if not success:
            _log(_DISCARD, "[eval] First attempt failed — retrying ...")
            success, cand_csv = _run_candidate()

        if not success:
            _log(
                _DISCARD,
                "[eval] Candidate eval failed on both attempts — marking crash.",
            )
            status = "crash"
            crash_notes = "agent crash"
            candidate_accuracy = 0.0
        else:
            candidate_results_csv = cand_csv
            candidate_accuracy = _read_accuracy(
                candidate_results_csv, args.question_index
            )
            _print_eval_contracts(candidate_results_csv, args.question_index)
            _log(
                _EVAL,
                f"[eval] candidate correct_at_0.5={candidate_accuracy:.3f}  baseline={current_correct_at_0_5:.3f}",
            )

        # ------------------------------------------------------------------
        # 5 & 6. COMPARE and KEEP/DISCARD
        # ------------------------------------------------------------------
        accepted_prompt_path: Path | None = None

        if status != "crash":
            if candidate_accuracy > current_correct_at_0_5:
                status = "keep"
                accepted_dir = (
                    Path("prompts")
                    / "autoresearch"
                    / f"q{args.question_index:02d}"
                    / date_str
                )
                accepted_dir.mkdir(parents=True, exist_ok=True)
                accepted_filename = f"{version_prefix}_r{args.round}_i{iter_n}.py"
                accepted_prompt_path = accepted_dir / accepted_filename

                accepted_prompt_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_py, accepted_prompt_path)
                shutil.copy2(candidate_py, iter_dir / "accepted.py")

                current_prompt_path = accepted_prompt_path
                current_prompt_text = result.prompt_text
                current_model_id = candidate_model_id
                current_correct_at_0_5 = candidate_accuracy
                current_eval_results_path = iter_dir / "candidate_results.csv"

                _log(
                    _KEEP,
                    f"[decision] ✓ KEEP — correct_at_0.5 improved to {candidate_accuracy:.3f}",
                )
            else:
                status = "discard"
                _log(
                    _DISCARD,
                    f"[decision] ✗ DISCARD — no improvement ({candidate_accuracy:.3f} ≤ {current_correct_at_0_5:.3f})",
                )

        # ------------------------------------------------------------------
        # 7. LOG TSV row
        # ------------------------------------------------------------------
        notes_text = crash_notes if status == "crash" else result.notes
        tsv_row: dict = {
            "iter": iter_n,
            "question_index": args.question_index,
            "category": category,
            "model_id": current_model_id,
            "prompt_file": str(current_prompt_path),
            "correct_at_0_5": current_correct_at_0_5,
            "n_wrong": len(diagnoses),
            "n_diagnosed": len(diagnoses),
            "status": status,
            "notes": notes_text,
        }
        write_tsv_row(tsv_path, tsv_row)
        tsv_rows.append(tsv_row)

        # ------------------------------------------------------------------
        # 8. REPORT
        # ------------------------------------------------------------------
        # Copy candidate outputs to iter_dir
        if candidate_results_csv is not None and candidate_results_csv.exists():
            shutil.copy2(candidate_results_csv, iter_dir / "candidate_results.csv")
            cand_summary_src = (
                args.output_dir / candidate_model_id / f"{OUTPUT_STEM}_summary.json"
            )
            if cand_summary_src.exists():
                shutil.copy2(cand_summary_src, iter_dir / "candidate_summary.json")

        # Load candidate DataFrame
        iter_candidate_csv = iter_dir / "candidate_results.csv"
        candidate_df = _load_filtered_df(iter_candidate_csv)

        # Load "before" DataFrame — the eval that was current before this iter
        # (iter0 baseline, or the last accepted candidate)
        if pre_iter_eval_path.exists():
            current_df_for_report = _load_filtered_df(pre_iter_eval_path)
        else:
            current_df_for_report = pd.DataFrame(columns=_STUB_COLUMNS)

        # Accuracy shown as "before" in the report is the pre-iter baseline
        # Stash candidate df for progress report
        if not candidate_df.empty:
            eval_dfs[iter_n] = candidate_df

        write_iter_report(
            iter_n=iter_n,
            category=category,
            question_index=args.question_index,
            date_str=date_str,
            status=status,
            notes=notes_text,
            current_accuracy=pre_iter_accuracy,
            candidate_accuracy=candidate_accuracy,
            current_eval_df=current_df_for_report,
            candidate_eval_df=candidate_df,
            triage_diagnoses=diagnoses,
            candidate_prompt_text=result.prompt_text,
            output_path=iter_dir / "report.html",
        )

        write_progress_report(
            rows=tsv_rows,
            category=category,
            question_index=args.question_index,
            output_path=progress_html_path,
            eval_dfs=eval_dfs,
            question=question_text,
            iter_prompts=iter_prompts,
            run_dir=run_dir,
        )

    _section("DONE")
    _log(_HEADER, f"  Results TSV : {tsv_path}")
    _log(_HEADER, f"  Dashboard   : {progress_html_path}")
