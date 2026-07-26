#!/usr/bin/env python3
"""Compatibility wrapper for the v2 prompt-improvement harness."""

from __future__ import annotations

import sys

from cuad_agent.prompt_optimization import harness as _harness

if __name__ == "__main__":
    _harness.main()
else:
    sys.modules[__name__] = _harness
