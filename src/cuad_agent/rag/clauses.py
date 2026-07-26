"""Lightweight clause and section metadata detection."""

from __future__ import annotations

import re


SECTION_RE = re.compile(
    r"^\s*(?:(Section)\s+)?(?P<number>\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?|ARTICLE\s+[IVXLC]+)\.?\s+(?P<title>.{1,120})$",
    re.IGNORECASE,
)


def detect_line_section(line: str) -> tuple[str | None, str | None] | None:
    stripped = line.strip()
    if not stripped:
        return None
    match = SECTION_RE.match(stripped)
    if match:
        return match.group("number"), match.group("title").strip()
    if (
        len(stripped) <= 80
        and stripped.upper() == stripped
        and any(char.isalpha() for char in stripped)
        and not stripped.endswith((".", ";", ","))
    ):
        return None, stripped.title()
    return None


def build_section_metadata(
    text: str,
) -> dict[int, tuple[str | None, str | None, list[str]]]:
    """Map line start offsets to the current section metadata."""
    metadata: dict[int, tuple[str | None, str | None, list[str]]] = {}
    current_number: str | None = None
    current_title: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        detected = detect_line_section(line)
        if detected is not None:
            current_number, current_title = detected
        if line.strip():
            metadata[offset] = (
                current_number,
                current_title,
                [part for part in (current_number, current_title) if part],
            )
        offset += len(line)
    return metadata
