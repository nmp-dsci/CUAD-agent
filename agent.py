#!/usr/bin/env python3
"""Primary CUAD legal-agent evaluator entrypoint."""

from __future__ import annotations

from cuad_agent.evaluators.langchain_runner import *  # noqa: F403
from cuad_agent.evaluators.langchain_runner import main


if __name__ == "__main__":
    main()
