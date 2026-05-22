"""Compare evaluation summaries across multiple runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_SUMMARY_STEM = "cuad_dspy_eval_summary.json"
_SEP = "-" * 80


def _load_summary(path: Path) -> dict[str, Any]:
    if path.is_dir():
        candidate = path / _SUMMARY_STEM
        if not candidate.exists():
            summaries = sorted(path.rglob(_SUMMARY_STEM))
            if not summaries:
                raise FileNotFoundError(f"No {_SUMMARY_STEM} found under {path}")
            candidate = summaries[0]
        path = candidate
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(val: float | None) -> str:
    if val is None:
        return "   n/a"
    return f"{val:6.1f}%"


def _print_header(summaries: list[dict[str, Any]], labels: list[str]) -> None:
    col_w = 22
    header = f"{'Category':<35}" + "".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(_SEP)
    print(header)
    print(_SEP)


def _print_comparison(summaries: list[dict[str, Any]], labels: list[str]) -> None:
    # Overall row
    col_w = 22
    overall_row = f"{'OVERALL mean F1':<35}" + "".join(
        f"{_fmt(s.get('overlap_accuracy_mean_f1')):>{col_w}}" for s in summaries
    )
    print(overall_row)
    correct_row = f"{'OVERALL correct@0.5':<35}" + "".join(
        f"{_fmt(s.get('correct_at_0_5')):>{col_w}}" for s in summaries
    )
    print(correct_row)
    print(_SEP)

    # Per-category rows — union of all categories in order
    category_order: list[tuple[int, str]] = []
    seen: set[str] = set()
    for summary in summaries:
        for entry in summary.get("per_category", []):
            key = entry["category"]
            if key not in seen:
                seen.add(key)
                category_order.append((entry["question_index"], key))
    category_order.sort()

    # Build lookup per summary: category → mean_token_f1
    lookups: list[dict[str, float]] = [
        {entry["category"]: entry["mean_token_f1"] for entry in s.get("per_category", [])}
        for s in summaries
    ]

    for _idx, category in category_order:
        row = f"{category:<35}" + "".join(
            f"{_fmt(lu.get(category)):>{col_w}}" for lu in lookups
        )
        print(row)


def _print_run_info(summaries: list[dict[str, Any]], labels: list[str]) -> None:
    print(_SEP)
    print("Run details:")
    for label, s in zip(labels, summaries):
        model = s.get("model", "?")
        model_id = s.get("model_id", "?")
        context_mode = s.get("context_mode", "raw")
        n = s.get("total_examples", "?")
        dry = " [dry-run]" if s.get("dry_run") else ""
        print(f"  {label}: model_id={model_id}, model={model}, context={context_mode}, n={n}{dry}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help=(
            "Path to a cuad_dspy_eval_summary.json file or an output directory "
            "containing one. Repeat to compare multiple runs."
        ),
    )
    parser.add_argument(
        "--label",
        nargs="*",
        default=None,
        metavar="LABEL",
        help="Short labels for each run (same order as paths). Defaults to model_id.",
    )
    args = parser.parse_args()

    summaries = [_load_summary(p) for p in args.paths]
    if args.label:
        if len(args.label) != len(summaries):
            parser.error(f"--label count ({len(args.label)}) must match path count ({len(summaries)})")
        labels = args.label
    else:
        labels = [s.get("model_id", str(p)) for s, p in zip(summaries, args.paths)]

    _print_header(summaries, labels)
    _print_comparison(summaries, labels)
    _print_run_info(summaries, labels)
    print(_SEP)


if __name__ == "__main__":
    main()
