"""Output writers for sentence-level RAG evaluation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def rag_output_paths(output_dir: Path, run_id: str) -> dict[str, Path]:
    run_dir = output_dir / run_id / "rag"
    frontend_dir = output_dir.parent / "dashboards"
    return {
        "run_dir": run_dir,
        "frontend_dir": frontend_dir,
        "gold_csv": run_dir / "golden_sentence_coverage.csv",
        "gold_summary": run_dir / "golden_sentence_coverage_summary.json",
        "gold_question_summary": run_dir / "golden_sentence_question_summary.csv",
        "sentences": run_dir / "rag_sentences.jsonl",
        "results_jsonl": run_dir / "rag_retrieval_results.jsonl",
        "results_csv": run_dir / "rag_retrieval_results.csv",
        "results_doc_question_summary": run_dir
        / "rag_retrieval_doc_question_summary.csv",
        "summary": run_dir / "rag_summary.json",
        "ranking_summary": run_dir / "rag_ranking_summary.csv",
        "query_enrichment_results": run_dir / "rag_query_enrichment_results.csv",
        "query_enrichment_summary": run_dir / "rag_query_enrichment_summary.csv",
        "config": run_dir / "rag_config.json",
        "pipeline_html": frontend_dir / "rag_pipeline_eval.html",
    }


def write_simple_html(
    path: Path,
    *,
    title: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_items = "\n".join(
        f"<div><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</div>"
        for key, value in summary.items()
    )
    table_rows = []
    columns = list(rows[0].keys()) if rows else []
    for row in rows[:500]:
        cells = "".join(
            f"<td>{html.escape(str(row.get(column, ''))[:1000])}</td>"
            for column in columns
        )
        table_rows.append(f"<tr>{cells}</tr>")
    header = "".join(
        f'<th><span class="column-title">{html.escape(column)}</span></th>'
        for column in columns
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; color: #17202a; }}
    h1 {{ font-size: 22px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin: 16px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; table-layout: fixed; }}
    th, td {{ border: 1px solid #d5d8dc; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #f4f6f7; text-align: left; position: sticky; top: 0; white-space: normal; overflow-wrap: anywhere; word-break: break-word; hyphens: auto; line-height: 1.25; min-width: 0; }}
    th .column-title {{ display: block; max-width: 100%; white-space: normal; overflow-wrap: anywhere; word-break: break-word; hyphens: auto; }}
    tr:nth-child(even) {{ background: #fbfcfc; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <section class="summary">{summary_items}</section>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{"".join(table_rows)}</tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def _summary_cards(summary: dict[str, Any]) -> str:
    preferred = [
        "run_id",
        "contract_scope",
        "chunked_contract_count",
        "sentence_count",
        "average_sentences_per_contract",
        "encoded_sentence_count",
        "embedding_backend",
        "eligible_rows",
        "retrieval_rows",
        "elapsed_seconds",
    ]
    items = [(key, summary.get(key)) for key in preferred if key in summary]
    items.extend(
        (key, value)
        for key, value in summary.items()
        if key not in {key for key, _ in items} and not isinstance(value, (list, dict))
    )
    return "\n".join(
        '<div class="metric">'
        f"<span>{html.escape(str(key).replace('_', ' ').title())}</span>"
        f"<strong>{html.escape(_format_display_value(key, value))}</strong>"
        "</div>"
        for key, value in items[:24]
    )


def _is_percentage_metric(column: str) -> bool:
    lowered = column.lower()
    if any(
        blocked in lowered for blocked in ("count", "rows", "seconds", "id", "index")
    ):
        return False
    return any(
        marker in lowered
        for marker in (
            "rate",
            "coverage",
            "similarity",
            "f1",
            "accuracy",
            "precision",
            "recall",
        )
    )


def _format_display_value(column: str, value: Any) -> str:
    if _is_percentage_metric(column):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        sign = "+" if "delta" in column.lower() and numeric > 0 else ""
        return f"{sign}{numeric * 100:.1f}%"
    return str(value)


def _table(rows: list[dict[str, Any]], *, limit: int = 500) -> str:
    if not rows:
        return '<p class="empty">No rows for this tab.</p>'
    columns = list(rows[0].keys())
    header = "".join(
        f'<th><span class="column-title">{html.escape(column)}</span></th>'
        for column in columns
    )
    body_rows: list[str] = []
    for row in rows[:limit]:
        cells = "".join(
            f"<td>{html.escape(_format_display_value(column, row.get(column, ''))[:1200])}</td>"
            for column in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _json_script_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def write_pipeline_html(
    path: Path,
    *,
    run_id: str,
    summary: dict[str, Any],
    chunking_summary_rows: list[dict[str, Any]],
    chunking_match_distribution: list[dict[str, Any]],
    chunking_version_comparison: list[dict[str, Any]],
    chunking_versions: dict[str, dict[str, Any]],
    chunking_documents: dict[str, dict[str, Any]],
    chunking_reviews: list[dict[str, Any]],
    query_enrichment_summary: list[dict[str, Any]],
    query_enrichment_rows: list[dict[str, Any]],
    eligibility_question_summary: list[dict[str, Any]],
    eligibility_rows: list[dict[str, Any]],
    retrieval_doc_question_summary: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    ranking_summary: list[dict[str, Any]],
    artifact_paths: dict[str, Path],
    hierarchical_config: dict[str, Any] | None = None,
) -> None:
    """Write one tabbed RAG dashboard for the latest pipeline run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "chunking_versions": chunking_versions,
        "query_enrichment_summary": query_enrichment_summary,
        "query_enrichment_rows": query_enrichment_rows[:500],
        "ranking_summary": ranking_summary,
        "hierarchical_config": hierarchical_config,
    }
    script_payload = _json_script_payload(payload)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RAG Pipeline Evaluation</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1120;
      --surface: #111827;
      --surface-2: #172033;
      --surface-3: #1f2937;
      --border: #334155;
      --border-soft: #243145;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --muted-2: #cbd5e1;
      --accent: #38bdf8;
      --accent-strong: #0ea5e9;
      --green-bg: #052e1a;
      --green-border: #22c55e;
      --green-text: #86efac;
      --yellow-bg: #3b2f0b;
      --yellow-border: #f59e0b;
      --yellow-text: #fde68a;
      --red-bg: #3b1014;
      --red-border: #ef4444;
      --red-text: #fca5a5;
    }}
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: var(--text); background: var(--bg); }}
    header {{ padding: 20px 24px 12px; background: var(--surface); border-bottom: 1px solid var(--border); }}
    h1 {{ margin: 0; font-size: 22px; }}
    h2 {{ color: var(--text); }}
    .subtle {{ color: var(--muted); margin-top: 4px; }}
    .tabs {{ display: flex; gap: 4px; padding: 12px 24px 0; background: var(--surface); border-bottom: 1px solid var(--border); }}
    .tab-button {{ border: 1px solid var(--border); background: var(--surface-2); color: var(--muted-2); padding: 8px 12px; cursor: pointer; border-radius: 6px 6px 0 0; font-size: 14px; }}
    .tab-button:hover {{ background: var(--surface-3); color: var(--text); }}
    .tab-button.active {{ background: var(--accent-strong); color: #f8fafc; border-color: var(--accent-strong); }}
    main {{ padding: 20px 24px; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; margin-bottom: 18px; }}
    .metric {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; }}
    .metric span {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
    .metric strong {{ font-size: 16px; }}
    .review-controls {{ display: flex; gap: 10px; align-items: center; margin: 12px 0 16px; flex-wrap: wrap; }}
    .review-controls label {{ color: var(--muted-2); }}
    .review-controls select {{ max-width: min(100%, 900px); padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px; background: var(--surface); color: var(--text); }}
    .review-grid {{ display: grid; gap: 14px; align-items: start; }}
    .review-result {{ display: grid; grid-template-columns: minmax(280px, 0.85fr) minmax(360px, 1.4fr); gap: 14px; align-items: start; }}
    .review-pane {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
    .review-pane h3 {{ margin: 0; padding: 10px 12px; font-size: 15px; background: var(--surface-2); border-bottom: 1px solid var(--border); color: var(--text); }}
    .review-list {{ max-height: 68vh; overflow: auto; }}
    .review-item {{ padding: 9px 12px; border-bottom: 1px solid var(--border-soft); }}
    .review-item.match {{ background: var(--green-bg); border-left: 4px solid var(--green-border); }}
    .review-item.partial {{ background: var(--yellow-bg); border-left: 4px solid var(--yellow-border); }}
    .review-item.miss {{ background: var(--red-bg); border-left: 4px solid var(--red-border); }}
    .review-meta {{ color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .review-text {{ white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.35; }}
    details.raw-contract {{ margin: 14px 0; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; }}
    details.raw-contract summary {{ cursor: pointer; padding: 10px 12px; font-weight: 600; background: var(--surface-2); }}
    .raw-contract pre {{ max-height: 54vh; margin: 0; border: 0; border-top: 1px solid var(--border); border-radius: 0; white-space: pre-wrap; overflow-wrap: anywhere; }}
    .table-wrap {{ overflow-y: auto; overflow-x: hidden; max-height: 72vh; border: 1px solid var(--border); background: var(--surface); }}
    table {{ border-collapse: collapse; width: 100%; min-width: 0; table-layout: fixed; font-size: 12px; }}
    th, td {{ border-bottom: 1px solid var(--border-soft); border-right: 1px solid var(--border-soft); padding: 5px 6px; vertical-align: top; min-width: 0; max-width: 1px; }}
    th {{ background: var(--surface-2); color: var(--muted-2); text-align: left; position: sticky; top: 0; z-index: 1; white-space: normal; overflow-wrap: anywhere; word-break: break-word; hyphens: auto; line-height: 1.25; min-width: 0; }}
    th .column-title {{ display: block; max-width: 100%; white-space: normal; overflow-wrap: anywhere; word-break: break-word; hyphens: auto; }}
    td {{ color: var(--text); white-space: normal; overflow-wrap: anywhere; word-break: break-word; hyphens: auto; }}
    td.text-cell {{ max-height: 140px; overflow: auto; white-space: pre-wrap; }}
    td.score-cell {{ font-variant-numeric: tabular-nums; font-weight: 700; white-space: normal; }}
    td.score-high {{ color: var(--green-text); background: rgba(34, 197, 94, 0.12); }}
    td.score-mid {{ color: var(--yellow-text); background: rgba(245, 158, 11, 0.12); }}
    td.score-low {{ color: var(--red-text); background: rgba(239, 68, 68, 0.12); }}
    td.score-positive {{ color: var(--green-text); background: rgba(34, 197, 94, 0.12); }}
    td.score-negative {{ color: var(--red-text); background: rgba(239, 68, 68, 0.12); }}
    td.score-neutral {{ color: var(--yellow-text); background: rgba(245, 158, 11, 0.10); }}
    tr:nth-child(even) {{ background: #0f172a; }}
    tr.hierarchical-row {{ background: rgba(56, 189, 248, 0.10); }}
    tr:hover {{ background: #1e293b; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(160px, 240px) 1fr 70px; gap: 10px; align-items: center; margin: 8px 0; }}
    .bar-track {{ height: 14px; background: var(--surface-3); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--accent); }}
    .bar-fill.hierarchical {{ background: var(--green-border); }}
    .empty {{ color: var(--muted); }}
    pre {{ background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 12px; overflow: auto; }}
    ::selection {{ background: rgba(56, 189, 248, 0.35); }}
  </style>
</head>
<body>
  <header>
    <h1>RAG Pipeline Evaluation</h1>
    <div class="subtle">Latest run: {html.escape(run_id)}</div>
  </header>
  <nav class="tabs" aria-label="RAG dashboard tabs">
    <button class="tab-button active" data-tab="summary">Summary</button>
    <button class="tab-button" data-tab="chunking-v3">Chunking — Sentence</button>
    <button class="tab-button" data-tab="chunking-lr">Chunking — Recursive</button>
    <button class="tab-button" data-tab="enrichment">Enriched Query</button>
    <button class="tab-button" data-tab="hierarchical">Hierarchical RAG</button>
  </nav>
  <main>
    <section id="summary" class="tab-panel active">
      <div class="metrics">{_summary_cards(summary)}</div>
      <h2>Chunking Summary</h2>
      <p class="subtle">Golden-answer sentence matching summary for contract chunks.</p>
      <div id="chunking-overview-table"></div>
    </section>
    <section id="chunking-v3" class="tab-panel">
      <h2>Chunking — Sentence</h2>
      <p class="subtle">Legal-aware sentence boundary detection (one sentence per chunk). Review how golden-answer sentences match against individual sentence chunks by document and question.</p>
      <div id="v3-summary-table"></div>
      <h2>Question Match Rates</h2>
      <p class="subtle">Chunk sentence matching rates per question. <strong>raw_contract_match_rate</strong> is the fraction of golden-answer sentences found verbatim in the raw contract text — the baseline extractability rate before any chunking.</p>
      <div id="v3-match-table"></div>
      <div class="review-controls">
        <label for="v3-document-select">Document</label>
        <select id="v3-document-select"></select>
        <label for="v3-question-select">Question</label>
        <select id="v3-question-select"></select>
      </div>
      <div id="v3-review-detail" class="review-grid"></div>
    </section>
    <section id="chunking-lr" class="tab-panel">
      <h2>Chunking — Recursive</h2>
      <p class="subtle">LangChain RecursiveCharacterTextSplitter chunks (~1200 chars, 150 overlap) using legal separators (ARTICLE/SECTION/lists). A chunk is highlighted as matched if any of its contained sentence IDs appear in the golden answer. Run with <code>--retrievers bm25_legal_recursive</code> or <code>dense_legal_recursive</code> to populate this tab.</p>
      <div id="lr-summary-table"></div>
      <h2>Question Match Rates</h2>
      <p class="subtle">Same sentence-level golden-answer match rates as the Chunking tab — shown here for reference against the LR chunks on the right.</p>
      <div id="lr-match-table"></div>
      <div class="review-controls">
        <label for="lr-document-select">Document</label>
        <select id="lr-document-select"></select>
        <label for="lr-question-select">Question</label>
        <select id="lr-question-select"></select>
      </div>
      <div id="lr-review-detail" class="review-grid"></div>
    </section>
    <section id="enrichment" class="tab-panel">
      <h2>Enriched Query Evaluation</h2>
      <p class="subtle">Compares baseline question retrieval against enriched-question retrieval. Similarity is TF-IDF cosine between the question text and split golden-answer sentences; coverage is the percentage of golden sentence IDs found in the top 10, 20, or 30 retrieved sentence chunks.</p>
      <h2>Retrieval Technique Comparison</h2>
      <p class="subtle">Average golden-answer sentence coverage across questions for each query and retrieval method combination.</p>
      <div id="query-enrichment-technique-table"></div>
      <h2>Question-Level Summary</h2>
      <div id="query-enrichment-summary-table"></div>
      <h2>Eligible Row Details</h2>
      <div id="query-enrichment-rows-table"></div>
    </section>
    <section id="hierarchical" class="tab-panel">
      <h2>Hierarchical RAG Performance</h2>
      <p class="subtle">Hierarchical rows retrieve leaves, expand only the selected top sections, then re-rank sentences inside those sections. High-cutoff coverage such as @30 is intentionally bounded by the top_sections setting.</p>
      <h2>Coverage Comparison</h2>
      <div id="hierarchical-coverage-table"></div>
      <h2>Coverage at 20</h2>
      <div id="hierarchical-bar-chart"></div>
      <h2>Parameters</h2>
      <div id="hierarchical-parameter-table"></div>
    </section>
  </main>
  <script type="application/json" id="rag-data">{script_payload}</script>
  <script>
    const ragData = JSON.parse(document.getElementById('rag-data').textContent);
    const buttons = document.querySelectorAll('.tab-button');
    const panels = document.querySelectorAll('.tab-panel');
    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}
    function isPercentageColumn(column) {{
      const lowered = String(column || '').toLowerCase();
      if (/(count|rows|seconds|_id|^id$|index|rank)/.test(lowered)) return false;
      return /(rate|coverage|similarity|f1|accuracy|precision|recall)/.test(lowered);
    }}
    function numericValue(value) {{
      if (value === null || value === undefined || value === '') return null;
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : null;
    }}
    function formatPercentage(column, value) {{
      const numeric = numericValue(value);
      if (numeric === null) return escapeHtml(String(value ?? '').slice(0, 1200));
      const sign = String(column || '').toLowerCase().includes('delta') && numeric > 0 ? '+' : '';
      return `${{sign}}${{(numeric * 100).toFixed(1)}}%`;
    }}
    function scoreClass(column, value) {{
      const numeric = numericValue(value);
      if (numeric === null) return '';
      const lowered = String(column || '').toLowerCase();
      if (lowered.includes('delta')) {{
        if (numeric > 0.0001) return 'score-positive';
        if (numeric < -0.0001) return 'score-negative';
        return 'score-neutral';
      }}
      if (numeric >= 0.75) return 'score-high';
      if (numeric >= 0.4) return 'score-mid';
      return 'score-low';
    }}
    function tableCell(column, value) {{
      if (!isPercentageColumn(column)) {{
        const textClass = /(?:question|query|answer|sentences|terms|path|ids|title)$/i.test(String(column || '')) ? ' text-cell' : '';
        return `<td class="${{textClass.trim()}}">${{escapeHtml(String(value ?? '').slice(0, 1200))}}</td>`;
      }}
      return `<td class="score-cell ${{scoreClass(column, value)}}">${{formatPercentage(column, value)}}</td>`;
    }}
    function tableFromRows(rows) {{
      if (!rows || !rows.length) {{
        return '<p class="empty">No rows for this table.</p>';
      }}
      const columns = Object.keys(rows[0]);
      const header = columns.map((column) => `<th><span class="column-title">${{escapeHtml(column)}}</span></th>`).join('');
      const body = rows.slice(0, 500).map((row) => {{
        return `<tr>${{columns.map((column) => tableCell(column, row[column])).join('')}}</tr>`;
      }}).join('');
      return `<div class="table-wrap"><table><thead><tr>${{header}}</tr></thead><tbody>${{body}}</tbody></table></div>`;
    }}
    function versionData(versionKey) {{
      return (ragData.chunking_versions || {{}})[versionKey] || {{}};
    }}
    function summaryRowsForVersions() {{
      return Object.entries(ragData.chunking_versions || {{}}).map(([key, data]) => {{
        const summary = (data.summary_rows || [{{}}])[0] || {{}};
        return {{
          version: data.label || key,
          source_chunking_version: data.source_chunking_version || '',
          review_answerable_question_rows: summary.review_answerable_question_rows || 0,
          review_contracts: summary.review_contracts || 0,
          review_contract_sentences: summary.review_contract_sentences || 0,
          source_sentence_count: summary.source_sentence_count || summary.encoded_sentence_count || '',
        }};
      }});
    }}
    function averageColumn(rows, column) {{
      const values = (rows || [])
        .map((row) => numericValue(row[column]))
        .filter((value) => value !== null);
      if (!values.length) return 0;
      return values.reduce((total, value) => total + value, 0) / values.length;
    }}
    function retrievalTechniqueRows() {{
      const rows = ragData.query_enrichment_summary || [];
      return [
        {{
          query_variant: 'baseline',
          retrieval_method: 'dense_vector',
          coverage_at_top_10_retrieved_sentences: averageColumn(rows, 'baseline_gold_sentence_coverage_at_10'),
          coverage_at_top_20_retrieved_sentences: averageColumn(rows, 'baseline_gold_sentence_coverage_at_20'),
          coverage_at_top_30_retrieved_sentences: averageColumn(rows, 'baseline_gold_sentence_coverage_at_30'),
        }},
        {{
          query_variant: 'enriched',
          retrieval_method: 'dense_vector',
          coverage_at_top_10_retrieved_sentences: averageColumn(rows, 'enriched_gold_sentence_coverage_at_10'),
          coverage_at_top_20_retrieved_sentences: averageColumn(rows, 'enriched_gold_sentence_coverage_at_20'),
          coverage_at_top_30_retrieved_sentences: averageColumn(rows, 'enriched_gold_sentence_coverage_at_30'),
        }},
        {{
          query_variant: 'baseline',
          retrieval_method: 'hybrid_bm25_dense',
          coverage_at_top_10_retrieved_sentences: averageColumn(rows, 'baseline_hybrid_gold_sentence_coverage_at_10'),
          coverage_at_top_20_retrieved_sentences: averageColumn(rows, 'baseline_hybrid_gold_sentence_coverage_at_20'),
          coverage_at_top_30_retrieved_sentences: averageColumn(rows, 'baseline_hybrid_gold_sentence_coverage_at_30'),
        }},
        {{
          query_variant: 'enriched',
          retrieval_method: 'hybrid_bm25_dense',
          coverage_at_top_10_retrieved_sentences: averageColumn(rows, 'enriched_hybrid_gold_sentence_coverage_at_10'),
          coverage_at_top_20_retrieved_sentences: averageColumn(rows, 'enriched_hybrid_gold_sentence_coverage_at_20'),
          coverage_at_top_30_retrieved_sentences: averageColumn(rows, 'enriched_hybrid_gold_sentence_coverage_at_30'),
        }},
      ];
    }}
    function hierarchicalCoverageRows() {{
      const rows = ragData.ranking_summary || [];
      const byMethod = Object.fromEntries(rows.map((row) => [row.retriever, row]));
      const bm25Base = byMethod.bm25_sentence || {{}};
      const denseBase = byMethod.dense_sentence || {{}};
      return rows.map((row) => {{
        const flatBase = row.retriever === 'bm25_hierarchical'
          ? bm25Base
          : row.retriever === 'dense_hierarchical'
            ? denseBase
            : null;
        const output = {{
          retriever: row.retriever,
          hierarchical: String(row.retriever || '').includes('hierarchical'),
          coverage_at_1: row.gold_sentence_coverage_at_1 ?? 0,
          coverage_at_3: row.gold_sentence_coverage_at_3 ?? 0,
          coverage_at_5: row.gold_sentence_coverage_at_5 ?? 0,
          coverage_at_10: row.gold_sentence_coverage_at_10 ?? 0,
          coverage_at_20: row.gold_sentence_coverage_at_20 ?? 0,
          coverage_at_30: row.gold_sentence_coverage_at_30 ?? 0,
          all_gold_covered_rate_at_10: row.all_gold_covered_rate_at_10 ?? 0,
          all_gold_covered_rate_at_20: row.all_gold_covered_rate_at_20 ?? 0,
          all_gold_covered_rate_at_30: row.all_gold_covered_rate_at_30 ?? 0,
          delta_vs_bm25_at_20: numericValue(row.gold_sentence_coverage_at_20) - numericValue(bm25Base.gold_sentence_coverage_at_20),
        }};
        if (flatBase) {{
          output.delta_vs_flat_counterpart_at_20 = numericValue(row.gold_sentence_coverage_at_20) - numericValue(flatBase.gold_sentence_coverage_at_20);
        }} else {{
          output.delta_vs_flat_counterpart_at_20 = '';
        }}
        return output;
      }});
    }}
    function hierarchicalParameterRows() {{
      const config = ragData.hierarchical_config;
      if (!config) return [];
      return [{{
        leaf_k: config.leaf_k,
        top_sections: config.top_sections,
        top_k: ragData.summary?.top_k ?? '',
      }}];
    }}
    function renderHierarchicalBars() {{
      const target = document.getElementById('hierarchical-bar-chart');
      if (!target) return;
      const rows = ragData.ranking_summary || [];
      if (!rows.length) {{
        target.innerHTML = '<p class="empty">No retrieval ranking summary for this run.</p>';
        return;
      }}
      target.innerHTML = rows.map((row) => {{
        const value = numericValue(row.gold_sentence_coverage_at_20) || 0;
        const isHierarchical = String(row.retriever || '').includes('hierarchical');
        return `<div class="bar-row">
          <div>${{escapeHtml(row.retriever)}}</div>
          <div class="bar-track"><div class="bar-fill ${{isHierarchical ? 'hierarchical' : ''}}" style="width: ${{Math.max(0, Math.min(100, value * 100))}}%"></div></div>
          <div>${{(value * 100).toFixed(1)}}%</div>
        </div>`;
      }}).join('');
    }}
    function renderReviewRows(prefix, version, reviews) {{
      const detail = document.getElementById(`${{prefix}}-review-detail`);
      if (!reviews.length) {{
        detail.innerHTML = '<p class="empty">No answerable chunking review rows match the selected filters.</p>';
        return;
      }}
      const blocks = reviews.slice(0, 100).map((review) => {{
        const matchedIds = new Set((review.matched_sentence_ids || []).map(String));
        const matchedCount = Number(review.matched_sentence_count || 0);
        const goldCount = Number(review.gold_sentence_count || 0);
        const rowMatchClass = matchedCount >= goldCount && goldCount > 0 ? 'match' : matchedCount > 0 ? 'partial' : 'miss';
        const rowMatchLabel = rowMatchClass === 'match' ? 'full_match' : rowMatchClass === 'partial' ? 'partial_match' : 'no_match';
        const documentInfo = (version.documents || {{}})[String(review.document_row_id)] || {{}};
        const contractSentences = documentInfo.sentences || [];
        const gold = (review.golden_sentences || []).map((item) => {{
          const klass = item.matched_sentence_id ? 'match' : rowMatchClass === 'partial' ? 'partial' : 'miss';
          const status = item.matched_sentence_id
            ? `green · chunk match: ${{item.matched_sentence_id}}`
            : rowMatchClass === 'partial'
              ? `yellow · partial row match, this split sentence unmatched: ${{item.reason || 'not matched'}}`
              : `red · no chunk match: ${{item.reason || 'not matched'}}`;
          return `<div class="review-item ${{klass}}">
            <div class="review-meta">${{escapeHtml(status)}}</div>
            <div class="review-text">${{escapeHtml(item.text)}}</div>
          </div>`;
        }}).join('');
        const rawGold = (review.raw_golden_answers || []).map((answer, index) => {{
          return `<div class="review-item">
            <div class="review-meta">Raw golden answer ${{index + 1}}</div>
            <div class="review-text">${{escapeHtml(answer)}}</div>
          </div>`;
        }}).join('');
        const sentences = contractSentences.map((item) => {{
          const containedIds = item.contained_sentence_ids || [];
          const klass = matchedIds.has(String(item.sentence_id)) || containedIds.some(id => matchedIds.has(String(id))) ? 'match' : '';
          const chunkMeta = containedIds.length > 0
            ? `${{escapeHtml(item.sentence_id)}} · ${{containedIds.length}} sentences · ${{escapeHtml(item.section_title || item.section_number || '')}}`
            : `${{escapeHtml(item.sentence_id)}} · index ${{escapeHtml(item.sentence_index)}} · chars ${{escapeHtml(item.start_char)}}-${{escapeHtml(item.end_char)}}`;
          return `<div class="review-item ${{klass}}">
            <div class="review-meta">${{chunkMeta}}</div>
            <div class="review-text">${{escapeHtml(item.raw_text)}}</div>
          </div>`;
        }}).join('');
        const documentTitle = review.document_title || documentInfo.title || review.document_row_id;
        const rawContract = documentInfo.raw_text || '';
        const rawQuestion = review.raw_question || review.question || '';
        const enrichedQuestion = review.enriched_question || rawQuestion;
        const enrichmentTerms = review.enrichment_terms || '';
        return `<section class="review-result">
          <section class="review-pane">
            <h3>Q${{escapeHtml(review.question_index)}} · ${{escapeHtml(review.category)}} · ${{escapeHtml(rowMatchLabel)}}</h3>
            <div class="review-item">
              <div class="review-meta">Document</div>
              <div class="review-text">${{escapeHtml(documentTitle)}}</div>
            </div>
            <div class="review-item">
              <div class="review-meta">Raw question</div>
              <div class="review-text">${{escapeHtml(rawQuestion)}}</div>
            </div>
            <div class="review-item">
              <div class="review-meta">Enriched question</div>
              <div class="review-text">${{escapeHtml(enrichedQuestion)}}</div>
            </div>
            ${{enrichmentTerms ? `<div class="review-item">
              <div class="review-meta">Enrichment terms</div>
              <div class="review-text">${{escapeHtml(enrichmentTerms)}}</div>
            </div>` : ''}}
            ${{rawGold || '<p class="empty review-item">No raw golden answer.</p>'}}
            <h3>Chunked Golden Answer</h3>
            <div class="review-list">${{gold || '<p class="empty review-item">No split golden answers.</p>'}}</div>
          </section>
          <section class="review-pane">
            <h3>Contract Chunks: ${{escapeHtml(documentTitle)}}</h3>
            <details class="raw-contract">
              <summary>Full Raw Contract Document</summary>
              <pre>${{escapeHtml(rawContract)}}</pre>
            </details>
            <div class="review-list">${{sentences || '<p class="empty review-item">No contract sentences.</p>'}}</div>
          </section>
        </section>`;
      }}).join('');
      const limitNote = reviews.length > 100 ? `<p class="subtle">Showing first 100 of ${{reviews.length}} matching rows.</p>` : '';
      detail.innerHTML = `<p class="subtle">Matching rows: ${{reviews.length}}</p>${{limitNote}}${{blocks}}`;
    }}
    function applyChunkingFilters(prefix, versionKey) {{
      const documentSelect = document.getElementById(`${{prefix}}-document-select`);
      const questionSelect = document.getElementById(`${{prefix}}-question-select`);
      const documentValue = documentSelect?.value || 'all';
      const questionValue = questionSelect?.value || 'all';
      const version = versionData(versionKey);
      const reviews = (version.reviews || []).filter((review) => {{
        const documentMatch = documentValue === 'all' || String(review.document_row_id) === documentValue;
        const questionMatch = questionValue === 'all' || String(review.question_index) === questionValue;
        return documentMatch && questionMatch;
      }});
      renderReviewRows(prefix, version, reviews);
    }}
    function renderChunkingTables(prefix, versionKey) {{
      const version = versionData(versionKey);
      const summary = document.getElementById(`${{prefix}}-summary-table`);
      const match = document.getElementById(`${{prefix}}-match-table`);
      if (summary) summary.innerHTML = tableFromRows(version.summary_rows || []);
      if (match) match.innerHTML = tableFromRows(version.match_distribution || []);
    }}
    function populateChunkingReviewFilters(prefix, versionKey) {{
      const documentSelect = document.getElementById(`${{prefix}}-document-select`);
      const questionSelect = document.getElementById(`${{prefix}}-question-select`);
      if (!documentSelect || !questionSelect) return;
      const version = versionData(versionKey);
      const reviews = version.reviews || [];
      const documents = Array.from(new Map(reviews.map((review) => [
        String(review.document_row_id),
        review.document_title || review.document_row_id,
      ])).entries()).sort((left, right) => String(left[1]).localeCompare(String(right[1])));
      const questions = Array.from(new Map(reviews.map((review) => [
        String(review.question_index),
        `Q${{review.question_index}} · ${{review.category}}`,
      ])).entries()).sort((left, right) => Number(left[0]) - Number(right[0]));
      documentSelect.innerHTML = '<option value="all">All documents</option>' + documents.map(([value, label]) => {{
        return `<option value="${{escapeHtml(value)}}">${{escapeHtml(label)}}</option>`;
      }}).join('');
      questionSelect.innerHTML = '<option value="all">All questions</option>' + questions.map(([value, label]) => {{
        return `<option value="${{escapeHtml(value)}}">${{escapeHtml(label)}}</option>`;
      }}).join('');
      if (documents.length) documentSelect.value = documents[0][0];
      questionSelect.value = 'all';
      applyChunkingFilters(prefix, versionKey);
    }}
    function initChunkingPanel(prefix, versionKey) {{
      const documentSelect = document.getElementById(`${{prefix}}-document-select`);
      const questionSelect = document.getElementById(`${{prefix}}-question-select`);
      if (!documentSelect || !questionSelect) return;
      renderChunkingTables(prefix, versionKey);
      populateChunkingReviewFilters(prefix, versionKey);
      documentSelect.addEventListener('change', () => applyChunkingFilters(prefix, versionKey));
      questionSelect.addEventListener('change', () => applyChunkingFilters(prefix, versionKey));
    }}
    function initDashboard() {{
      const overview = document.getElementById('chunking-overview-table');
      if (overview) overview.innerHTML = tableFromRows(summaryRowsForVersions());
      const techniqueComparison = document.getElementById('query-enrichment-technique-table');
      if (techniqueComparison) techniqueComparison.innerHTML = tableFromRows(retrievalTechniqueRows());
      const enrichmentSummary = document.getElementById('query-enrichment-summary-table');
      if (enrichmentSummary) enrichmentSummary.innerHTML = tableFromRows(ragData.query_enrichment_summary || []);
      const enrichmentRows = document.getElementById('query-enrichment-rows-table');
      if (enrichmentRows) enrichmentRows.innerHTML = tableFromRows(ragData.query_enrichment_rows || []);
      const hierarchicalCoverage = document.getElementById('hierarchical-coverage-table');
      if (hierarchicalCoverage) hierarchicalCoverage.innerHTML = tableFromRows(hierarchicalCoverageRows());
      const hierarchicalParameters = document.getElementById('hierarchical-parameter-table');
      if (hierarchicalParameters) hierarchicalParameters.innerHTML = tableFromRows(hierarchicalParameterRows());
      renderHierarchicalBars();
      initChunkingPanel('v3', 'sentence-v3');
      initChunkingPanel('lr', 'legal-recursive-v1');
    }}
    buttons.forEach((button) => {{
      button.addEventListener('click', () => {{
        buttons.forEach((item) => item.classList.remove('active'));
        panels.forEach((panel) => panel.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.tab).classList.add('active');
      }});
    }});
    initDashboard();
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
