#!/usr/bin/env python3
"""Compatibility wrapper for the CUAD dataset exploration CLI."""

from __future__ import annotations

from cuad_agent.data.dataset import *  # noqa: F403
from cuad_agent.data.dataset import main


if __name__ == "__main__":
    main()
