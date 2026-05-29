"""Contract chunking analysis — hierarchy pattern detection and HTML dashboard."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from unstructured.partition.text import partition_text as _partition_text  # type: ignore[import]
    _HAS_UNSTRUCTURED = True
except ImportError:
    _HAS_UNSTRUCTURED = False


# ---------------------------------------------------------------------------
# Hierarchy pattern definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HierarchyPattern:
    name: str
    label: str
    level: int        # 0=doc-root, 1=major, 2=section, 3=subsection, 4=item, 5=sub-item
    css_class: str
    description: str
    example: str
    regex: re.Pattern[str] | None


HIERARCHY_PATTERNS: list[HierarchyPattern] = [
    HierarchyPattern(
        name="exhibit",
        label="Exhibit / Schedule Header",
        level=0,
        css_class="hl-exhibit",
        description="SEC filing exhibit/schedule marker at document root.",
        example="EXHIBIT 10.6 · Exhibit 10.26 · Schedule A",
        regex=re.compile(r"^(?:EXHIBIT|Exhibit|SCHEDULE|Schedule|ANNEX|Annex)\s+[\d.\w]"),
    ),
    HierarchyPattern(
        name="article",
        label="ARTICLE (Roman / Arabic)",
        level=1,
        css_class="hl-article",
        description="Top-level article grouping, often with Roman numerals.",
        example="ARTICLE I — DEFINITIONS · Article 2",
        regex=re.compile(r"^(?:ARTICLE|Article)\s+(?:[IVXLCDM]+|\d+)"),
    ),
    HierarchyPattern(
        name="whereas",
        label="WHEREAS / Recital Opening",
        level=1,
        css_class="hl-whereas",
        description="Recital clause openers — WHEREAS, WITNESSETH, RECITALS, NOW THEREFORE.",
        example="WHEREAS · WITNESSETH · RECITALS · NOW, THEREFORE",
        regex=re.compile(r"^(?:WHEREAS|WITNESSETH|RECITALS?|NOW,?\s+THEREFORE)\b"),
    ),
    HierarchyPattern(
        name="allcaps",
        label="ALL CAPS Heading",
        level=1,
        css_class="hl-allcaps",
        description="Short line in ALL CAPS used as a major heading (≤80 chars, no trailing punctuation).",
        example="RECITALS · REPRESENTATIONS AND WARRANTIES · DEFINITIONS",
        regex=None,  # handled by special logic
    ),
    HierarchyPattern(
        name="section",
        label="Section N.N (keyword)",
        level=2,
        css_class="hl-section",
        description="Explicit SECTION or Section keyword followed by a dotted number.",
        example="Section 3.1 · SECTION 12 · Section 1.2(a)",
        regex=re.compile(r"^(?:SECTION|Section)\s+\d+(?:\.\d+)*(?:\([a-z]\))?\.?\s+\S"),
    ),
    HierarchyPattern(
        name="top_numbered",
        label="Top-Level Number  (N.)",
        level=2,
        css_class="hl-section",
        description="Single integer followed by period then a capital-letter title.",
        example="1. Definitions · 12. Termination",
        regex=re.compile(r"^\d+\.\s+[A-Z]"),
    ),
    HierarchyPattern(
        name="subsection",
        label="Subsection (N.N)",
        level=3,
        css_class="hl-subsection",
        description="Dotted numeric subsection (two or more parts), with optional capital-letter title.",
        example="1.1 · 2.3.4 Payments",
        regex=re.compile(r"^\d+\.\d+"),
    ),
    # roman_paren must come before alpha_paren so (i)(ii)(v)(x) are not misclassified as letters
    HierarchyPattern(
        name="roman_paren",
        label="Roman Paren  (i)",
        level=5,
        css_class="hl-subitem",
        description="Roman numeral parenthetical sub-list — typically the deepest nesting level.",
        example="(i) · (ii) · (iii) · (iv) · (viii)",
        regex=re.compile(r"^\([ivxlIVXL]+\)\s+\S"),
    ),
    HierarchyPattern(
        name="alpha_paren_lower",
        label="Lower Alpha Paren  (a)",
        level=4,
        css_class="hl-item",
        description="Lowercase lettered parenthetical — most common list style in US contracts.",
        example="(a) · (b) · (c) · (d)",
        regex=re.compile(r"^\([a-z]\)\s+\S"),
    ),
    HierarchyPattern(
        name="alpha_paren_upper",
        label="Upper Alpha Paren  (A)",
        level=5,
        css_class="hl-item-upper",
        description="Uppercase lettered parenthetical — deeper nesting than lowercase (a)(b).",
        example="(A) · (B) · (C) · (D)",
        regex=re.compile(r"^\([A-Z]\)\s+\S"),
    ),
    HierarchyPattern(
        name="num_paren",
        label="Numeric Paren  (1)",
        level=4,
        css_class="hl-item",
        description="Numbered parenthetical list item.",
        example="(1) · (2) · (15)",
        regex=re.compile(r"^\(\d+\)\s+\S"),
    ),
    HierarchyPattern(
        name="bare_letter",
        label="Bare Letter  a)",
        level=4,
        css_class="hl-item",
        description="Letter followed by closing parenthesis only (no opening paren).",
        example="a) · b) · c)",
        regex=re.compile(r"^[a-z]\)\s+\S"),
    ),
]

_PATTERN_INDEX: dict[str, HierarchyPattern] = {p.name: p for p in HIERARCHY_PATTERNS}

# ---------------------------------------------------------------------------
# Prefix extraction — candidate new-pattern detection on unclassified lines
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"""^(
        \([a-zA-Z0-9]+\)      # parenthetical: (a) (A) (1) (iv)
        | [A-Za-z]{1,4}\.     # letter/abbrev + period: a. B. ii. Art.
        | \d+\.               # integer + period: 1.  12.
        | [A-Za-z0-9]{1,3}\)  # letter/number + close-paren: a) B) 1)
        | [•\-–—*]            # bullet or dash
    )""",
    re.VERBOSE,
)


def _extract_prefix(stripped: str) -> str | None:
    """Return the leading sequence token if the line looks like a potential structural element."""
    m = _PREFIX_RE.match(stripped)
    return m.group(1).rstrip() if m else None


# ---------------------------------------------------------------------------
# Unstructured integration (optional — used for independent label comparison)
# ---------------------------------------------------------------------------

def _unstructured_line_labels(lines: list[str]) -> dict[int, str]:
    """Map 0-based line index → unstructured element type (Title/ListItem/NarrativeText/…)."""
    text = "\n".join(lines)
    try:
        elements = _partition_text(text=text)
    except Exception:
        return {}
    labels: dict[int, str] = {}
    cursor = 0
    for el in elements:
        el_text = el.text.strip()
        if not el_text:
            continue
        el_type = type(el).__name__
        search_key = el_text[:35]
        for j in range(cursor, min(cursor + 30, len(lines))):
            if search_key in lines[j]:
                labels[j] = el_type
                cursor = j
                break
    return labels


def detect_line_type(line: str) -> HierarchyPattern | None:
    """Return the best-matching HierarchyPattern for a stripped line, or None for body text."""
    stripped = line.strip()
    if not stripped:
        return None
    for pattern in HIERARCHY_PATTERNS:
        if pattern.name == "allcaps":
            if (
                len(stripped) <= 80
                and stripped.upper() == stripped
                and any(c.isalpha() for c in stripped)
                and not stripped.endswith((".", ";", ","))
            ):
                return pattern
            continue
        if pattern.regex is not None and pattern.regex.match(stripped):
            return pattern
    return None


# ---------------------------------------------------------------------------
# Contract annotation
# ---------------------------------------------------------------------------

MAX_LINES_SHOWN = 200


def annotate_contract(document_row_id: int, title: str, text: str) -> dict[str, Any]:
    all_lines = text.splitlines()
    lines = all_lines[:MAX_LINES_SHOWN]
    annotated: list[dict[str, Any]] = []
    pattern_counts: dict[str, int] = {}

    total_words = 0
    classified_words = 0
    total_nonempty = 0
    classified_lines = 0
    prefix_counts: dict[str, int] = {}

    u_labels = _unstructured_line_labels(lines) if _HAS_UNSTRUCTURED else {}

    for i, line in enumerate(lines):
        detection = detect_line_type(line)
        words = len(line.split())
        total_words += words
        stripped = line.strip()
        if stripped:
            total_nonempty += 1

        if detection:
            pattern_counts[detection.name] = pattern_counts.get(detection.name, 0) + 1
            classified_words += words
            classified_lines += 1
        elif stripped:
            prefix = _extract_prefix(stripped)
            if prefix:
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

        entry: dict[str, Any] = {
            "n": i + 1,
            "t": line,
            "p": detection.name if detection else None,
            "l": detection.level if detection else None,
            "w": words,
        }
        if u_labels:
            entry["u"] = u_labels.get(i)
        annotated.append(entry)

    word_coverage = round(classified_words / total_words * 100, 1) if total_words else 0
    line_coverage = round(classified_lines / total_nonempty * 100, 1) if total_nonempty else 0
    top_prefixes = sorted(prefix_counts.items(), key=lambda x: -x[1])[:25]

    first = all_lines[0].strip() if all_lines else ""
    exhibit_match = re.match(r"^(?:EXHIBIT|Exhibit|exhibit|SCHEDULE|Schedule)\s+[\d.]+", first)

    return {
        "id": document_row_id,
        "title": title,
        "lines": annotated,
        "total_lines": len(all_lines),
        "starts_with_exhibit": bool(exhibit_match),
        "exhibit_header": first[:80] if exhibit_match else None,
        "pattern_counts": pattern_counts,
        "total_words": total_words,
        "classified_words": classified_words,
        "word_coverage": word_coverage,
        "line_coverage": line_coverage,
        "top_prefixes": top_prefixes,
        "has_unstructured": bool(u_labels),
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_CSS_COLORS = """
    --c-exhibit:    #f59e0b;
    --c-article:    #22d3ee;
    --c-whereas:    #fb923c;
    --c-allcaps:    #38bdf8;
    --c-section:    #818cf8;
    --c-subsection: #c084fc;
    --c-item:       #4ade80;
    --c-item-upper: #86efac;
    --c-subitem:    #f472b6;
    --c-body:       #94a3b8;
"""

_LEVEL_COLORS = {
    "hl-exhibit":    "var(--c-exhibit)",
    "hl-article":    "var(--c-article)",
    "hl-whereas":    "var(--c-whereas)",
    "hl-allcaps":    "var(--c-allcaps)",
    "hl-section":    "var(--c-section)",
    "hl-subsection": "var(--c-subsection)",
    "hl-item":       "var(--c-item)",
    "hl-item-upper": "var(--c-item-upper)",
    "hl-subitem":    "var(--c-subitem)",
}


def _legend_html() -> str:
    rows = ""
    for p in HIERARCHY_PATTERNS:
        color = _LEVEL_COLORS.get(p.css_class, "var(--c-body)")
        rows += (
            f'<tr><td><span class="swatch" style="background:{color}"></span></td>'
            f'<td><strong style="color:{color}">{p.label}</strong><br>'
            f'<span class="ex">{p.example}</span></td>'
            f'<td class="desc">{p.description}</td></tr>'
        )
    return rows


def _tool_table_html() -> str:
    tools = [
        {
            "Tool": "regex (current — clauses.py)",
            "Install": "built-in",
            "Strength": "Zero deps, fast, works on plain text",
            "Weakness": "Misses inline sections, no indentation depth, no bold/font info",
            "Best for": "Quick section detection on already-extracted plain text",
        },
        {
            "Tool": "docling (IBM, 2024)",
            "Install": "pip install docling",
            "Strength": "Reads PDF/DOCX using font size + bold to assign heading levels. Returns structured Markdown with # / ## / ###.",
            "Weakness": "Needs original PDF/DOCX file; slower than regex; heavy deps",
            "Best for": "If you have original contract PDFs — gold-standard hierarchy extraction",
        },
        {
            "Tool": "unstructured",
            "Install": "pip install unstructured",
            "Strength": "partition_text() classifies lines as Title / NarrativeText / ListItem. Works on plain text without regex.",
            "Weakness": "NLP heuristics can misfire on dense legal text; no dotted-number awareness",
            "Best for": "Mixed-format corpora where regex breaks; fast prototype without PDFs",
        },
        {
            "Tool": "pdfplumber / pdfminer",
            "Install": "pip install pdfplumber",
            "Strength": "Extracts font size, weight, and position per character — definitive heading detection from PDF layout",
            "Weakness": "Requires PDF source; complex column/table layout handling",
            "Best for": "Font-based heading extraction when PDFs are available",
        },
        {
            "Tool": "Indentation analysis",
            "Install": "built-in",
            "Strength": "Leading-whitespace depth is a reliable proxy for nesting in OCR'd contracts",
            "Weakness": "Inconsistent indentation in many contracts; not semantic",
            "Best for": "Supplementing regex — add indent_level field to SentenceSpan",
        },
        {
            "Tool": "LLM-based (Claude API)",
            "Install": "anthropic SDK",
            "Strength": "Handles any format, edge cases, mixed numbering styles; structured output via tool_use",
            "Weakness": "Expensive at scale (510 contracts); latency per document",
            "Best for": "One-time pre-processing pass to build a canonical section tree per contract",
        },
    ]
    cols = ["Tool", "Install", "Strength", "Weakness", "Best for"]
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows = ""
    for t in tools:
        rows += "<tr>" + "".join(f"<td>{t[c]}</td>" for c in cols) + "</tr>"
    return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"


def _enhanced_regex_html() -> str:
    patterns = [
        ("Exhibit header", r"^(?:EXHIBIT|Exhibit|SCHEDULE|Schedule|ANNEX|Annex)\\s+[\\d.\\w]",
         "Not in clauses.py — strips SEC exhibit prefix before chunking"),
        ("ARTICLE (Roman)", r"^(?:ARTICLE|Article)\\s+(?:[IVXLCDM]+|\\d+)",
         "clauses.py catches this via SECTION_RE fallback"),
        ("Section keyword", r"^(?:SECTION|Section)\\s+\\d+(?:\\.\\d+)*(?:\\([a-z]\\))?\\.?\\s+\\S",
         "clauses.py SECTION_RE — works"),
        ("Top-level N.", r"^\\d+\\.\\s+[A-Z]",
         "Not in clauses.py — missed single-integer sections"),
        ("Subsection N.N", r"^\\d+\\.\\d+",
         "Caught by clauses.py SECTION_RE number group \\d+(?:\\.\\d+)*"),
        ("ALL CAPS heading", r"short line, upper() == stripped, len ≤ 80",
         "clauses.py detects this as section_title with number=None"),
        ("(a) alpha paren", r"^\\([a-zA-Z]\\)\\s+\\S",
         "Not in clauses.py — missed list items"),
        ("(i) roman paren", r"^\\([ivxlIVXL]+\\)\\s+\\S",
         "Not in clauses.py — missed sub-list items"),
        ("(1) numeric paren", r"^\\(\\d+\\)\\s+\\S",
         "Not in clauses.py — missed numeric list items"),
        ("a) bare letter", r"^[a-z]\\)\\s+\\S",
         "Not in clauses.py — missed bare-letter list items"),
        ("WHEREAS / RECITALS", r"^(?:RECITALS?|WITNESSETH|WHEREAS|NOW,?\\s+THEREFORE)\\b",
         "Partially caught as ALL CAPS heading"),
        ("Indentation depth", "count leading spaces / 4",
         "Not tracked anywhere — would improve hierarchical retrieval accuracy"),
    ]
    rows = "".join(
        f"<tr><td><code>{p[0]}</code></td><td><code>{p[1]}</code></td><td>{p[2]}</td></tr>"
        for p in patterns
    )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Pattern</th><th>Regex</th><th>Current coverage in clauses.py</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def write_chunking_analysis_html(
    output_path: Path,
    contracts_df: pd.DataFrame,
    n_contracts: int = 10,
) -> None:
    """Generate chunking_analysis.html showing hierarchy pattern detection across N contracts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample = contracts_df.head(n_contracts)
    annotated = []
    for row in sample.itertuples(index=False):
        text = getattr(row, "context", "") or ""
        annotated.append(annotate_contract(int(row.document_row_id), str(row.title), text))

    exhibit_count = sum(1 for a in annotated if a["starts_with_exhibit"])
    payload = json.dumps(
        {"contracts": annotated, "patterns": [p.name for p in HIERARCHY_PATTERNS]},
        ensure_ascii=False,
    ).replace("</", "<\\/")

    css_level_vars = "\n".join(
        f"    .{p.css_class} {{ color: {_LEVEL_COLORS.get(p.css_class, 'inherit')}; }}"
        for p in HIERARCHY_PATTERNS
    )

    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Contract Chunking Analysis — CUAD</title>
  <style>
    :root {{
      color-scheme: dark;
      {_CSS_COLORS}
      --bg: #0b141a; --panel: #111b21; --panel-2: #202c33;
      --border: #26353d; --text: #e9edef; --muted: #8696a0; --muted-2: #aebac1;
      --green: #00a884;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; background: var(--bg); color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    a {{ color: inherit; }}
    .global-header {{ height: 56px; display: flex; align-items: center; gap: 16px;
      padding: 0 18px; background: var(--panel-2); border-bottom: 1px solid var(--border); }}
    .brand {{ font-weight: 700; display: flex; align-items: center; gap: 10px; }}
    .brand-mark {{ width: 30px; height: 30px; border-radius: 8px;
      background: linear-gradient(135deg,#00a884,#3b82f6);
      display: grid; place-items: center; color: #fff; font-size: 13px; }}
    .tabs {{ display: inline-flex; align-items: center; gap: 4px; padding: 4px;
      border: 1px solid var(--border); border-radius: 8px; background: rgba(0,0,0,.16); }}
    .tab {{ color: var(--muted-2); text-decoration: none; padding: 7px 12px;
      border-radius: 6px; font-size: 13px; white-space: nowrap; }}
    .tab:hover {{ color: var(--text); background: rgba(255,255,255,.06); }}
    .tab.active {{ color: #fff; background: var(--green); }}
    .layout {{ display: grid; grid-template-columns: 300px 1fr; height: calc(100vh - 56px); overflow: hidden; }}
    .sidebar {{ background: var(--panel); border-right: 1px solid var(--border);
      display: flex; flex-direction: column; overflow: hidden; }}
    .sidebar-header {{ padding: 14px 16px; background: var(--panel-2);
      border-bottom: 1px solid var(--border); font-weight: 650; font-size: 15px; }}
    .contract-list {{ flex: 1; overflow-y: auto; }}
    .contract-btn {{ width: 100%; border: 0; border-bottom: 1px solid var(--border);
      background: transparent; color: var(--text); text-align: left;
      padding: 11px 14px; cursor: pointer; font: inherit; }}
    .contract-btn:hover, .contract-btn.active {{ background: var(--panel-2); }}
    .contract-btn .exhibit-tag {{ color: var(--c-exhibit); font-size: 11px; }}
    .contract-btn .doc-title {{ font-size: 12px; color: var(--muted); white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis; }}
    .legend-section {{ padding: 12px; border-top: 1px solid var(--border);
      overflow-y: auto; max-height: 40vh; }}
    .legend-section h3 {{ margin: 0 0 8px; font-size: 12px; color: var(--muted); text-transform: uppercase; }}
    .legend-section table {{ font-size: 11px; border-collapse: collapse; width: 100%; }}
    .legend-section td {{ padding: 3px 4px; vertical-align: top; }}
    .swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
    .ex {{ color: var(--muted); font-style: italic; }}
    .desc {{ color: var(--muted-2); }}
    .main {{ display: flex; flex-direction: column; overflow: hidden; }}
    .main-header {{ padding: 14px 20px; background: var(--panel-2);
      border-bottom: 1px solid var(--border); }}
    .main-header h2 {{ margin: 0 0 4px; font-size: 16px; }}
    .main-header .stats {{ color: var(--muted); font-size: 12px; }}
    .contract-view {{ flex: 1; overflow-y: auto; padding: 0; }}
    .code-line {{ display: flex; gap: 0; font-family: ui-monospace,SFMono-Regular,Menlo,monospace;
      font-size: 12px; line-height: 1.55; padding: 0 16px; }}
    .code-line:hover {{ background: rgba(255,255,255,.04); }}
    .line-num {{ color: var(--muted); width: 36px; flex: 0 0 36px; text-align: right;
      padding-right: 12px; user-select: none; }}
    .line-text {{ white-space: pre-wrap; overflow-wrap: anywhere; color: var(--c-body); }}
    {css_level_vars}
    .section-divider {{ border-top: 1px solid var(--border); margin: 16px 0; }}
    .tools-section {{ padding: 24px; overflow-y: auto; }}
    .tools-section h2 {{ color: var(--text); margin-top: 28px; }}
    .tools-section p {{ color: var(--muted-2); max-width: 820px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 8px;
      background: rgba(0,0,0,.14); margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; text-align: left; }}
    th {{ background: rgba(0,0,0,.18); color: var(--muted-2); font-weight: 650;
      position: sticky; top: 0; white-space: nowrap; }}
    code {{ background: rgba(255,255,255,.08); padding: 1px 5px; border-radius: 3px; font-size: 11px; }}
    .badge {{ display: inline-block; padding: 2px 7px; border-radius: 99px;
      font-size: 11px; background: rgba(255,255,255,.08); }}
    .tabs-inner {{ display: flex; border-bottom: 1px solid var(--border); background: var(--panel-2); }}
    .inner-tab {{ padding: 10px 18px; border: 0; background: transparent; color: var(--muted-2);
      cursor: pointer; font: 14px/1 inherit; border-bottom: 2px solid transparent; }}
    .inner-tab.active {{ color: var(--text); border-bottom-color: var(--green); }}
    .inner-panel {{ display: none; }}
    .inner-panel.active {{ display: block; }}
    .filter-bar {{ display: flex; gap: 8px; padding: 8px 16px;
      background: var(--panel-2); border-bottom: 1px solid var(--border); flex-shrink: 0; }}
    .filter-btn {{ padding: 4px 12px; border-radius: 6px; border: 1px solid var(--border);
      background: transparent; color: var(--muted-2); cursor: pointer; font: 12px/1.4 inherit; }}
    .filter-btn.active {{ background: var(--green); color: #fff; border-color: var(--green); }}
    .u-badge {{ font-size: 9px; padding: 1px 4px; border-radius: 3px;
      background: rgba(255,255,255,.08); color: var(--muted); margin-left: 5px;
      vertical-align: middle; font-family: inherit; }}
    .u-badge.Title {{ color: #22d3ee; }}
    .u-badge.ListItem {{ color: #4ade80; }}
    .cov-section {{ padding: 16px 20px; overflow-y: auto; flex: 1; }}
    .cov-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
    .cov-card {{ background: var(--panel-2); border: 1px solid var(--border);
      border-radius: 8px; padding: 14px 16px; }}
    .cov-card-title {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
      letter-spacing: .04em; margin: 0 0 6px; }}
    .cov-card-val {{ font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 6px; }}
    .cov-bar-outer {{ height: 6px; background: rgba(255,255,255,.06);
      border-radius: 3px; overflow: hidden; margin-bottom: 6px; }}
    .cov-bar-fill {{ height: 100%; border-radius: 3px; transition: width .4s; }}
    .cov-sub {{ font-size: 11px; color: var(--muted); }}
    .cov-h3 {{ font-size: 13px; font-weight: 600; margin: 20px 0 4px; }}
    .cov-hint {{ font-size: 11px; color: var(--muted-2); margin: 0 0 10px; }}
    .cov-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    .cov-table th {{ background: rgba(0,0,0,.18); color: var(--muted-2); padding: 6px 10px;
      text-align: left; font-weight: 600; position: sticky; top: 0; border-bottom: 1px solid var(--border); }}
    .cov-table td {{ padding: 5px 10px; border-bottom: 1px solid rgba(255,255,255,.04); }}
    .cov-table tr:hover td {{ background: rgba(255,255,255,.03); }}
    .cov-sidebar-bar {{ height: 3px; margin-top: 4px; border-radius: 2px; background: var(--border); overflow: hidden; }}
    .cov-sidebar-fill {{ height: 100%; border-radius: 2px; }}
  </style>
</head>
<body>
<header class="global-header">
  <div class="brand">
    <div class="brand-mark">C</div>
    CUAD Explorer
  </div>
  <nav class="tabs">
    <a class="tab" href="explore.html">Explorer</a>
    <a class="tab active" href="chunking_analysis.html">Chunking</a>
    <a class="tab" href="rag_pipeline_eval.html">RAG Pipeline</a>
  </nav>
</header>
<div class="layout">
  <aside class="sidebar">
    <div class="sidebar-header">{n_contracts} Sampled Contracts</div>
    <div class="contract-list" id="contract-list"></div>
    <div class="legend-section">
      <h3>Pattern Legend</h3>
      <table>
        <tbody id="legend-body">
          {_legend_html()}
        </tbody>
      </table>
    </div>
  </aside>
  <div class="main" id="main">
    <div class="tabs-inner">
      <button class="inner-tab active" data-panel="structure">Structure View</button>
      <button class="inner-tab" data-panel="coverage">Coverage</button>
      <button class="inner-tab" data-panel="tools">Tools &amp; Research</button>
    </div>
    <div class="inner-panel active" id="panel-structure" style="display:flex;flex-direction:column;overflow:hidden;flex:1;">
      <div class="main-header" id="contract-header">
        <h2>Select a contract</h2>
        <div class="stats" id="contract-stats"></div>
      </div>
      <div class="filter-bar">
        <button class="filter-btn active" id="btn-all" onclick="setFilter('all')">All lines</button>
        <button class="filter-btn" id="btn-unclassified" onclick="setFilter('unclassified')">Unclassified only</button>
        <button class="filter-btn" id="btn-patterns" onclick="setFilter('patterns')">Patterns only</button>
      </div>
      <div class="contract-view" id="contract-view">
        <p style="padding:24px;color:var(--muted)">Choose a contract from the left panel.</p>
      </div>
    </div>
    <div class="inner-panel" id="panel-coverage" style="display:flex;flex-direction:column;overflow:hidden;flex:1;">
      <div class="cov-section" id="cov-content">
        <p style="color:var(--muted)">Select a contract to see coverage stats.</p>
      </div>
    </div>
    <div class="inner-panel" id="panel-tools" style="overflow-y:auto;flex:1;">
      <div class="tools-section">
        <h2>Findings: Exhibit Header Pattern</h2>
        <p>{exhibit_count}/{n_contracts} of the sampled contracts start with
        <code>EXHIBIT N.N</code> or <code>Exhibit N.N</code>. The one exception starts with
        <strong>REDACTED COPY</strong> followed by the agreement title —
        no exhibit number in the contract text itself (it appears in the SEC filename only).</p>

        <h2>Hierarchy Patterns Found in CUAD Contracts</h2>
        <p>Contracts show up to 6 nesting levels. The hierarchy is implicit in typography
        (ALL CAPS, numbered sections, parenthetical letters/romans) since plain-text extraction
        loses bold/italic/font-size cues from the original PDF.</p>
        {_enhanced_regex_html()}

        <h2>Tools for Hierarchy Discovery</h2>
        <p>From fastest/lightest to most accurate:</p>
        {_tool_table_html()}

        <h2>Recommended Next Steps</h2>
        <p>
          <strong>Short term</strong> — enhance <code>clauses.py</code> with the missing patterns
          above (top-level <code>N.</code>, <code>(a)</code>, <code>(i)</code>, indentation depth).
          This gives better <code>clause_path</code> breadcrumbs for the hierarchical retriever
          at zero extra dependencies.<br><br>
          <strong>Medium term</strong> — run <code>docling</code> or <code>unstructured</code>
          as a one-time pre-processing pass over the 510 contracts and cache the resulting section
          trees alongside the existing sentence span JSONL files. This would give heading-level
          metadata that regex cannot recover from plain text.<br><br>
          <strong>Long term</strong> — if original PDFs are available, <code>pdfplumber</code>
          font-size extraction gives definitive heading levels that map directly to
          <code>clause_path</code> depth without any regex heuristics.
        </p>
      </div>
    </div>
  </div>
</div>
<script type="application/json" id="payload">{payload}</script>
<script>
  const data = JSON.parse(document.getElementById('payload').textContent);
  const PATTERN_CSS = {{
    exhibit: 'hl-exhibit', article: 'hl-article', whereas: 'hl-whereas',
    allcaps: 'hl-allcaps', section: 'hl-section', top_numbered: 'hl-section',
    subsection: 'hl-subsection', roman_paren: 'hl-subitem',
    alpha_paren_lower: 'hl-item', alpha_paren_upper: 'hl-item-upper',
    alpha_paren: 'hl-item', num_paren: 'hl-item', bare_letter: 'hl-item',
  }};
  const PATTERN_INDENT = {{
    exhibit: 0, article: 0, whereas: 0, allcaps: 0, section: 1, top_numbered: 1,
    subsection: 2, roman_paren: 4,
    alpha_paren_lower: 3, alpha_paren_upper: 4,
    alpha_paren: 3, num_paren: 3, bare_letter: 3,
  }};

  let currentIdx = 0;
  let currentFilter = 'all';

  function detectPattern(text) {{
    const s = text.trim();
    if (!s) return null;
    if (/^(?:EXHIBIT|Exhibit|SCHEDULE|Schedule|ANNEX|Annex)\\s+[\\d.\\w]/.test(s)) return 'exhibit';
    if (/^(?:ARTICLE|Article)\\s+(?:[IVXLCDM]+|\\d+)/.test(s)) return 'article';
    if (/^(?:WHEREAS|WITNESSETH|RECITALS?|NOW,?\\s+THEREFORE)\\b/.test(s)) return 'whereas';
    if (s.length <= 80 && s === s.toUpperCase() && /[A-Z]/.test(s) && !/[.;,]$/.test(s)) return 'allcaps';
    if (/^(?:SECTION|Section)\\s+\\d+(?:\\.\\d+)*(?:\\([a-z]\\))?\\.?\\s+\\S/.test(s)) return 'section';
    if (/^\\d+\\.\\s+[A-Z]/.test(s)) return 'top_numbered';
    if (/^\\d+\\.\\d+/.test(s)) return 'subsection';
    if (/^\\([ivxlIVXL]+\\)\\s+\\S/.test(s)) return 'roman_paren';
    if (/^\\([a-z]\\)\\s+\\S/.test(s)) return 'alpha_paren_lower';
    if (/^\\([A-Z]\\)\\s+\\S/.test(s)) return 'alpha_paren_upper';
    if (/^\\(\\d+\\)\\s+\\S/.test(s)) return 'num_paren';
    if (/^[a-z]\\)\\s+\\S/.test(s)) return 'bare_letter';
    return null;
  }}

  function esc(s) {{
    return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }}

  function setFilter(f) {{
    currentFilter = f;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-' + f).classList.add('active');
    renderStructureLines(data.contracts[currentIdx]);
  }}

  function renderStructureLines(c) {{
    const view = document.getElementById('contract-view');
    view.innerHTML = '';
    const frag = document.createDocumentFragment();
    for (const line of c.lines) {{
      const p = detectPattern(line.t);
      if (currentFilter === 'unclassified' && p) continue;
      if (currentFilter === 'patterns' && !p) continue;
      const div = document.createElement('div');
      div.className = 'code-line';
      const indent = p ? (PATTERN_INDENT[p] || 0) * 12 : 0;
      const cls = p ? (PATTERN_CSS[p] || '') : '';
      const uBadge = line.u ? '<span class="u-badge ' + esc(line.u) + '">' + esc(line.u) + '</span>' : '';
      div.innerHTML =
        '<span class="line-num">' + line.n + '</span>' +
        '<span class="line-text ' + cls + '" style="padding-left:' + indent + 'px">' + esc(line.t) + uBadge + '</span>';
      frag.appendChild(div);
    }}
    view.appendChild(frag);
  }}

  function guessPatternType(prefix) {{
    if (/^\\([a-z]\\)/.test(prefix)) return '<span style="color:var(--c-item)">lower alpha paren</span> — already covered';
    if (/^\\([A-Z]\\)/.test(prefix)) return '<span style="color:var(--c-item-upper)">upper alpha paren</span> — already covered';
    if (/^\\([ivxIVX]+\\)/.test(prefix)) return '<span style="color:var(--c-subitem)">roman paren</span> — already covered';
    if (/^\\(\\d+\\)/.test(prefix)) return '<span style="color:var(--c-item)">numeric paren</span> — already covered';
    if (/^[a-z]\\)/.test(prefix)) return '<span style="color:var(--c-item)">bare lower paren</span> — already covered';
    if (/^[a-z]\\./.test(prefix)) return '<span style="color:#fde047">bare lower period</span> — <strong>potential new pattern</strong>';
    if (/^[A-Z]\\./.test(prefix)) return '<span style="color:#fde047">bare upper period</span> — <strong>potential new pattern</strong>';
    if (/^[ivxIVX]{{1,4}}\\./.test(prefix)) return '<span style="color:#fde047">roman period</span> — <strong>potential new pattern</strong>';
    if (/^\\d+\\./.test(prefix)) return '<span style="color:var(--c-section)">top-level number</span> — already covered';
    if (/^[•\\-–—*]/.test(prefix)) return '<span style="color:#fb7185">bullet/dash</span> — <strong>potential new pattern</strong>';
    return '<span style="color:var(--muted)">unknown</span>';
  }}

  function coverageColor(pct) {{
    return pct > 15 ? 'var(--green)' : pct > 7 ? '#f59e0b' : '#f87171';
  }}

  function renderCoverage(c) {{
    const wPct = c.word_coverage || 0;
    const lPct = c.line_coverage || 0;
    const wCol = coverageColor(wPct);
    const lCol = coverageColor(lPct);

    let html = '<div class="cov-grid">'
      + '<div class="cov-card">'
      +   '<div class="cov-card-title">Word Coverage</div>'
      +   '<div class="cov-card-val" style="color:' + wCol + '">' + wPct + '%</div>'
      +   '<div class="cov-bar-outer"><div class="cov-bar-fill" style="width:' + wPct + '%;background:' + wCol + '"></div></div>'
      +   '<div class="cov-sub">' + (c.classified_words||0) + ' of ' + (c.total_words||0) + ' words on pattern-matched lines</div>'
      + '</div>'
      + '<div class="cov-card">'
      +   '<div class="cov-card-title">Line Coverage</div>'
      +   '<div class="cov-card-val" style="color:' + lCol + '">' + lPct + '%</div>'
      +   '<div class="cov-bar-outer"><div class="cov-bar-fill" style="width:' + lPct + '%;background:' + lCol + '"></div></div>'
      +   '<div class="cov-sub">% of non-empty lines matched by a pattern</div>'
      + '</div>'
      + '</div>';

    if (c.has_unstructured) {{
      html += '<p class="cov-sub" style="margin-bottom:16px">&#x2713; unstructured labels included — see <span class="u-badge Title">Title</span> / <span class="u-badge ListItem">ListItem</span> badges in Structure View</p>';
    }}

    if (c.top_prefixes && c.top_prefixes.length) {{
      html += '<div class="cov-h3">Unclassified Line Prefixes — Candidate New Patterns</div>'
        + '<p class="cov-hint">First token of lines not matched by any current regex. '
        + 'High-frequency entries with "potential new pattern" status are worth adding to <code>HIERARCHY_PATTERNS</code>.</p>'
        + '<div class="table-wrap"><table class="cov-table">'
        + '<thead><tr><th>Prefix</th><th>Count</th><th>Pattern type</th></tr></thead><tbody>';
      for (const [prefix, count] of c.top_prefixes) {{
        html += '<tr><td><code>' + esc(prefix) + '</code></td><td>' + count + '</td><td>' + guessPatternType(prefix) + '</td></tr>';
      }}
      html += '</tbody></table></div>';
    }} else {{
      html += '<p class="cov-hint">No unclassified sequence prefixes found in first ' + c.lines.length + ' lines.</p>';
    }}

    document.getElementById('cov-content').innerHTML = html;
  }}

  function selectContract(idx) {{
    currentIdx = idx;
    const c = data.contracts[idx];

    const patternCounts = {{}};
    c.lines.forEach(line => {{
      const p = detectPattern(line.t);
      if (p) patternCounts[p] = (patternCounts[p] || 0) + 1;
    }});

    document.getElementById('contract-header').innerHTML =
      '<h2>' + esc(c.title.slice(0,90)) + '</h2>' +
      '<div class="stats">' +
        (c.starts_with_exhibit
          ? '<span class="badge" style="color:var(--c-exhibit)">' + esc(c.exhibit_header) + '</span> &nbsp; '
          : '<span class="badge">No exhibit header</span> &nbsp; ') +
        'First ' + c.lines.length + ' of ' + c.total_lines + ' lines &nbsp;|&nbsp; ' +
        '<span style="color:' + coverageColor(c.word_coverage||0) + '">' + (c.word_coverage||0) + '% word coverage</span>' +
      '</div>';

    renderStructureLines(c);
    renderCoverage(c);

    document.querySelectorAll('.contract-btn').forEach((btn, i) => {{
      btn.classList.toggle('active', i === idx);
    }});
  }}

  function buildSidebar() {{
    const list = document.getElementById('contract-list');
    data.contracts.forEach((c, i) => {{
      const btn = document.createElement('button');
      btn.className = 'contract-btn';
      const wPct = c.word_coverage || 0;
      const col = coverageColor(wPct);
      btn.innerHTML =
        '<div class="exhibit-tag">' + (c.starts_with_exhibit ? esc(c.exhibit_header) : 'No exhibit') + '</div>' +
        '<div class="doc-title">' + esc(c.title.slice(0,55)) + '</div>' +
        '<div class="cov-sidebar-bar"><div class="cov-sidebar-fill" style="width:' + Math.min(wPct*4,100) + '%;background:' + col + '"></div></div>';
      btn.addEventListener('click', () => selectContract(i));
      list.appendChild(btn);
    }});
  }}

  document.querySelectorAll('.inner-tab').forEach(btn => {{
    btn.addEventListener('click', () => {{
      const panel = btn.dataset.panel;
      document.querySelectorAll('.inner-tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.inner-panel').forEach(p => {{
        p.classList.remove('active');
        p.style.display = 'none';
      }});
      btn.classList.add('active');
      const el = document.getElementById('panel-' + panel);
      el.classList.add('active');
      el.style.display = (panel === 'structure' || panel === 'coverage') ? 'flex' : 'block';
    }});
  }});

  buildSidebar();
  selectContract(0);
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
