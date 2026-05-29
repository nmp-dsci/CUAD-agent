#!/usr/bin/env python3
"""CUAD dataset exploration — prints summary stats and generates chunking_analysis.html."""

from __future__ import annotations

from pathlib import Path

from cuad_agent.data.dataset import *  # noqa: F403
from cuad_agent.data.dataset import load_datasets, main
from cuad_agent.data.chunking_analysis import write_chunking_analysis_html

_OUTPUT_PATH = Path(__file__).resolve().parent / "dashboards" / "chunking_analysis.html"


if __name__ == "__main__":
    main()
    datasets = load_datasets()
    write_chunking_analysis_html(_OUTPUT_PATH, datasets["contracts"], n_contracts=50)
    print(f"\nChunking analysis written → {_OUTPUT_PATH}")
