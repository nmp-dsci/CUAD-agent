#!/usr/bin/env python3
"""Compatibility wrapper for the DSPy CUAD evaluator."""

from __future__ import annotations

from cuad_agent.evaluators.dspy_runner import *  # noqa: F403
from cuad_agent.evaluators.dspy_runner import main


if __name__ == "__main__":
    main()
