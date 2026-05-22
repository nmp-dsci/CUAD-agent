"""Shared dashboard helpers."""

from __future__ import annotations

import json
from typing import Any


def embedded_json(payload: dict[str, Any]) -> str:
    """Serialize JSON for safe embedding in a static HTML script tag."""
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


__all__ = ["embedded_json"]
