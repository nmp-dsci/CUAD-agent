#!/usr/bin/env python3
"""Compatibility wrapper for the LangChain CUAD evaluator."""

from __future__ import annotations

from cuad_agent.evaluators.langchain_runner import *  # noqa: F403
from cuad_agent.evaluators.langchain_runner import main


if __name__ == "__main__":
    main()
