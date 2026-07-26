"""HTML report writers for autoresearch iterations and progress tracking."""

from __future__ import annotations

import difflib
import html as html_module
from pathlib import Path

import pandas as pd

from cuad_agent.autoresearch.results import TriageDiagnosis

__all__ = ["write_iter_report", "write_progress_report"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STATUS_COLOUR = {
    "keep": "#22c55e",
    "discard": "#f59e0b",
    "crash": "#ef4444",
}

_BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 14px; color: #111; background: #f8fafc; }
.page { max-width: 1100px; margin: 24px auto; padding: 0 16px 48px; }
h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
h2 { font-size: 1.05rem; font-weight: 600; margin: 24px 0 8px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; color: #fff; font-size: 0.78rem; font-weight: 700; vertical-align: middle; margin-left: 8px; }
.date { font-size: 0.8rem; color: #64748b; margin-top: 4px; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; }
th { text-align: left; padding: 6px 8px; background: #f1f5f9; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .04em; }
td { padding: 6px 8px; vertical-align: top; border-bottom: 1px solid #e2e8f0; }
.pos { color: #15803d; font-weight: 700; }
.neg { color: #dc2626; font-weight: 700; }
pre { background: #1e293b; color: #e2e8f0; padding: 16px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; font-size: 0.82rem; line-height: 1.5; margin-top: 8px; }
.notes { font-style: italic; color: #475569; margin-top: 8px; }
details { margin: 4px 0; }
summary { cursor: pointer; padding: 4px 0; }
.detail-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 10px 12px; margin-top: 6px; font-size: 0.82rem; line-height: 1.6; }
.detail-box strong { display: inline-block; min-width: 130px; color: #475569; }
.correct { background: #f0fdf4; }
.incorrect { background: #fff1f2; }
"""


def _esc(text: object) -> str:
    return html_module.escape(str(text))


def _render_diff(before: str, after: str) -> str:
    """Return an HTML <pre> block showing a unified diff between two prompt strings."""
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="current prompt",
            tofile="candidate prompt",
            lineterm="",
        )
    )
    if not diff_lines:
        return '<pre style="background:#f1f5f9;padding:12px;border-radius:6px;font-size:0.8rem;">(no change)</pre>'

    parts = []
    for line in diff_lines:
        escaped = html_module.escape(line)
        if line.startswith("+++") or line.startswith("---"):
            parts.append(
                f'<span style="color:#94a3b8;font-weight:600;">{escaped}</span>'
            )
        elif line.startswith("@@"):
            parts.append(
                f'<span style="background:#dbeafe;color:#1d4ed8;display:block;padding:0 4px;">{escaped}</span>'
            )
        elif line.startswith("+"):
            parts.append(
                f'<span style="background:#dcfce7;color:#15803d;display:block;padding:0 4px;">{escaped}</span>'
            )
        elif line.startswith("-"):
            parts.append(
                f'<span style="background:#fee2e2;color:#dc2626;display:block;padding:0 4px;">{escaped}</span>'
            )
        else:
            parts.append(f'<span style="display:block;padding:0 4px;">{escaped}</span>')

    inner = "\n".join(parts)
    return (
        '<pre style="background:#0f172a;color:#e2e8f0;padding:14px;border-radius:6px;'
        'font-size:0.78rem;line-height:1.5;overflow-x:auto;white-space:pre-wrap;">'
        f"{inner}</pre>"
    )


def _delta_cell(delta: float) -> str:
    sign = "+" if delta > 0 else ""
    cls = "pos" if delta > 0 else "neg"
    return f'<td class="{cls}">{sign}{delta:.3f}</td>'


# ---------------------------------------------------------------------------
# write_iter_report
# ---------------------------------------------------------------------------


def write_iter_report(
    *,
    iter_n: int,
    category: str,
    question_index: int,
    date_str: str,
    status: str,
    notes: str,
    current_accuracy: float,
    candidate_accuracy: float,
    current_eval_df: pd.DataFrame,
    candidate_eval_df: pd.DataFrame,
    triage_diagnoses: list[TriageDiagnosis],
    candidate_prompt_text: str,
    output_path: Path,
) -> None:
    """Write a self-contained HTML iteration report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    badge_colour = _STATUS_COLOUR.get(status, "#94a3b8")
    delta = candidate_accuracy - current_accuracy

    # Build triage lookup by contract_id
    triage_by_id: dict[int, TriageDiagnosis] = {
        t.contract_id: t for t in triage_diagnoses
    }

    # ---- Section 1: header ------------------------------------------------
    header_html = f"""
<h1>Iter {iter_n} — {_esc(category)} (q{question_index:02d})
  <span class="badge" style="background:{badge_colour};">{_esc(status)}</span>
</h1>
<div class="date">{_esc(date_str)}</div>
"""

    # ---- Section 2: accuracy strip ----------------------------------------
    accuracy_html = f"""
<h2>Accuracy</h2>
<table>
  <thead><tr><th>Before</th><th>After</th><th>Δ</th></tr></thead>
  <tbody>
    <tr>
      <td>{current_accuracy:.3f}</td>
      <td>{candidate_accuracy:.3f}</td>
      {_delta_cell(delta)}
    </tr>
  </tbody>
</table>
"""

    # ---- Section 3: candidate prompt --------------------------------------
    prompt_html = f"""
<h2>Candidate System Prompt</h2>
<pre>{_esc(candidate_prompt_text)}</pre>
<p class="notes">{_esc(notes)}</p>
"""

    # ---- Section 4: changed answers ---------------------------------------
    cur = current_eval_df.set_index("document_row_id")
    cand = candidate_eval_df.set_index("document_row_id")
    shared_ids = cur.index.intersection(cand.index)
    changed_rows = [
        rid
        for rid in shared_ids
        if cur.at[rid, "correct_at_0_5"] != cand.at[rid, "correct_at_0_5"]
    ]

    if changed_rows:
        rows_html_parts = []
        for rid in changed_rows:
            c_row = cur.loc[rid]
            k_row = cand.loc[rid]
            td = triage_by_id.get(rid)
            loc_html = (
                f"<div><strong>Golden location:</strong> {_esc(td.golden_answer_location)}</div>"
                if td
                else ""
            )
            rows_html_parts.append(f"""
<details>
  <summary>
    <strong>{_esc(k_row["contract_title"])}</strong>
    (id={rid})
    — before: {"correct" if c_row["correct_at_0_5"] == 1.0 else "incorrect"}
    → after: {"correct" if k_row["correct_at_0_5"] == 1.0 else "incorrect"}
  </summary>
  <div class="detail-box">
    <div><strong>Golden answer:</strong> {_esc(k_row["golden_answer"])}</div>
    <div><strong>Before predicted:</strong> {_esc(c_row["predicted_answer"])}</div>
    <div><strong>After predicted:</strong> {_esc(k_row["predicted_answer"])}</div>
    {loc_html}
  </div>
</details>""")
        changed_html = f"""
<h2>Changed Answers ({len(changed_rows)})</h2>
{"".join(rows_html_parts)}
"""
    else:
        changed_html = "<h2>Changed Answers</h2><p style='color:#64748b;margin-top:8px;'>No answers changed between current and candidate.</p>"

    # ---- Section 5: all results -------------------------------------------
    # Sort: incorrect first
    cand_reset = candidate_eval_df.copy()
    cand_reset = cand_reset.sort_values("correct_at_0_5", ascending=True)

    all_rows_parts = []
    for _, row in cand_reset.iterrows():
        rid = int(row["document_row_id"])
        is_correct = row["correct_at_0_5"] == 1.0
        td = triage_by_id.get(rid)

        if is_correct:
            all_rows_parts.append(f"""
<tr class="correct">
  <td>{_esc(rid)}</td>
  <td>{_esc(row["contract_title"])}</td>
  <td>{_esc(row["predicted_answer"])}</td>
  <td>{_esc(row["golden_answer"])}</td>
  <td>✓</td>
</tr>""")
        else:
            triage_html = ""
            if td:
                triage_html = f"""
  <div class="detail-box">
    <div><strong>Golden location:</strong> {_esc(td.golden_answer_location)}</div>
    <div><strong>Failure reason:</strong> {_esc(td.failure_reason)}</div>
    <div><strong>Proposed rule:</strong> {_esc(td.proposed_rule)}</div>
    <div><strong>Confidence:</strong> {_esc(td.confidence)}</div>
  </div>"""
            all_rows_parts.append(f"""
<tr class="incorrect">
  <td colspan="5" style="padding:0;">
    <details>
      <summary style="padding:6px 8px;">
        <strong>{_esc(row["contract_title"])}</strong>
        (id={_esc(rid)})
        — predicted: {_esc(row["predicted_answer"])}
        | golden: {_esc(row["golden_answer"])}
      </summary>
      {triage_html}
    </details>
  </td>
</tr>""")

    all_results_html = f"""
<h2>All Results — Candidate (incorrect first)</h2>
<table>
  <thead>
    <tr>
      <th>ID</th><th>Contract</th><th>Predicted</th><th>Golden</th><th>Correct</th>
    </tr>
  </thead>
  <tbody>
    {"".join(all_rows_parts)}
  </tbody>
</table>
"""

    # ---- Assemble page ----------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Iter {iter_n} — {_esc(category)} (q{question_index:02d})</title>
  <style>
{_BASE_CSS}
  </style>
</head>
<body>
  <div class="page">
    {header_html}
    {accuracy_html}
    {prompt_html}
    {changed_html}
    {all_results_html}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# write_progress_report
# ---------------------------------------------------------------------------


def write_progress_report(
    *,
    rows: list[dict],
    category: str,
    question_index: int,
    output_path: Path,
    eval_dfs: dict[int, pd.DataFrame] | None = None,
    question: str = "",
    iter_prompts: dict[int, tuple[str, str]] | None = None,
    run_dir: Path | None = None,
) -> None:
    """Write a self-contained HTML progress report with SVG chart and per-iter contract results."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(rows)
    n_kept = sum(1 for r in rows if r.get("status") == "keep")

    title = f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements"

    # ---- SVG chart --------------------------------------------------------
    # Layout constants
    SVG_W, SVG_H = 880, 380
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 40, 50, 60
    chart_w = SVG_W - PAD_L - PAD_R
    chart_h = SVG_H - PAD_T - PAD_B

    def cx(iter_val: float) -> float:
        if n_total <= 1:
            return PAD_L + chart_w / 2
        return PAD_L + (iter_val / (n_total - 1)) * chart_w

    def cy(acc: float) -> float:
        return PAD_T + chart_h - acc * chart_h

    # Build axis ticks
    x_ticks = "".join(
        f'<text x="{cx(i):.1f}" y="{PAD_T + chart_h + 20}" text-anchor="middle" font-size="12" fill="#64748b">{r.get("iter", i)}</text>'
        f'<line x1="{cx(i):.1f}" y1="{PAD_T + chart_h}" x2="{cx(i):.1f}" y2="{PAD_T + chart_h + 5}" stroke="#94a3b8" />'
        for i, r in enumerate(rows)
    )

    y_tick_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_ticks = "".join(
        f'<text x="{PAD_L - 8}" y="{cy(v) + 4:.1f}" text-anchor="end" font-size="11" fill="#64748b">{v:.1f}</text>'
        f'<line x1="{PAD_L - 4}" y1="{cy(v):.1f}" x2="{PAD_L + chart_w}" y2="{cy(v):.1f}" stroke="#e2e8f0" />'
        for v in y_tick_vals
    )

    # Green stepped line connecting baseline → kept dots in order
    kept_points: list[tuple[float, float]] = []
    for i, r in enumerate(rows):
        if r.get("status") in ("baseline", "keep"):
            kept_points.append((cx(i), cy(float(r.get("correct_at_0_5", 0)))))

    step_path = ""
    if len(kept_points) >= 2:
        parts = [f"M {kept_points[0][0]:.1f} {kept_points[0][1]:.1f}"]
        for j in range(1, len(kept_points)):
            prev_x, prev_y = kept_points[j - 1]
            curr_x, curr_y = kept_points[j]
            # Staircase: horizontal then vertical
            parts.append(f"H {curr_x:.1f} V {curr_y:.1f}")
        step_path = (
            f'<path d="{" ".join(parts)}" fill="none" stroke="#22c55e" '
            f'stroke-width="2.5" stroke-linejoin="round" />'
        )

    # F1 lookup from eval DataFrames
    f1_by_iter: dict[int, float] = {}
    if eval_dfs:
        for iter_key, df in eval_dfs.items():
            if df is not None and not df.empty and "token_f1" in df.columns:
                f1_by_iter[iter_key] = float(df["token_f1"].mean())

    # Build circle + tooltip data elements
    circles_html_parts = []
    for i, r in enumerate(rows):
        status = r.get("status", "")
        acc = float(r.get("correct_at_0_5", 0))
        notes_text = r.get("notes", "")
        iter_val = r.get("iter", i)

        colour = "#22c55e" if status == "keep" else "#94a3b8"
        x = cx(i)
        y = cy(acc)

        f1_val = f1_by_iter.get(iter_val, None)
        f1_str = f" | f1={f1_val:.3f}" if f1_val is not None else ""
        tooltip = _esc(
            f"Iter {iter_val} | {status} | correct_at_0.5={acc:.3f}{f1_str} | {notes_text}"
        )

        # Inline label for kept points
        label_html = ""
        if status == "keep" and notes_text:
            label_html = (
                f'<text x="{x:.1f}" y="{y - 10:.1f}" font-size="10" fill="#15803d" '
                f'transform="rotate(-30,{x:.1f},{y - 10:.1f})" '
                f'text-anchor="start">{_esc(notes_text[:40])}</text>'
            )

        circles_html_parts.append(
            f'<g class="dot" data-tip="{tooltip}">'
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colour}" stroke="#fff" stroke-width="1.5" />'
            f"  {label_html}"
            f"</g>"
        )

    circles_html = "\n".join(circles_html_parts)

    svg = f"""<svg viewBox="0 0 {SVG_W} {SVG_H}" width="{SVG_W}" height="{SVG_H}" xmlns="http://www.w3.org/2000/svg">
  <!-- grid -->
  {y_ticks}
  <!-- axes -->
  <line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T + chart_h}" stroke="#94a3b8" />
  <line x1="{PAD_L}" y1="{PAD_T + chart_h}" x2="{PAD_L + chart_w}" y2="{PAD_T + chart_h}" stroke="#94a3b8" />
  <!-- axis labels -->
  <text x="{PAD_L + chart_w / 2:.1f}" y="{SVG_H - 8}" text-anchor="middle" font-size="12" fill="#475569">Iteration</text>
  <text x="14" y="{PAD_T + chart_h / 2:.1f}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90,14,{PAD_T + chart_h / 2:.1f})">correct_at_0.5</text>
  <!-- chart title -->
  <text x="{PAD_L + chart_w / 2:.1f}" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#1e293b">{_esc(title)}</text>
  <text x="{PAD_L + chart_w / 2:.1f}" y="38" text-anchor="middle" font-size="12" fill="#475569">{_esc(category)} — q{question_index:02d}</text>
  <!-- x ticks -->
  {x_ticks}
  <!-- step line -->
  {step_path}
  <!-- dots -->
  {circles_html}
</svg>"""

    # ---- JavaScript tooltip -----------------------------------------------
    js = """
const tooltip = document.getElementById('tt');
document.querySelectorAll('.dot').forEach(g => {
  g.addEventListener('mouseover', e => {
    tooltip.textContent = g.dataset.tip;
    tooltip.style.display = 'block';
  });
  g.addEventListener('mousemove', e => {
    tooltip.style.left = (e.pageX + 14) + 'px';
    tooltip.style.top  = (e.pageY - 28) + 'px';
  });
  g.addEventListener('mouseout', () => {
    tooltip.style.display = 'none';
  });
});
"""

    # ---- Contract results by iteration ------------------------------------
    contract_section_html = ""
    if eval_dfs:
        # Build a status lookup from rows
        status_by_iter = {
            r.get("iter", i): r.get("status", "") for i, r in enumerate(rows)
        }
        acc_by_iter = {
            r.get("iter", i): float(r.get("correct_at_0_5", 0))
            for i, r in enumerate(rows)
        }

        iter_blocks = []
        for iter_key in sorted(eval_dfs):
            df = eval_dfs[iter_key]
            if df is None or df.empty:
                continue
            iter_status = status_by_iter.get(iter_key, "")
            iter_acc = acc_by_iter.get(iter_key, 0.0)
            badge_col = _STATUS_COLOUR.get(iter_status, "#94a3b8")

            # Sort: incorrect first
            df_sorted = df.copy().sort_values("correct_at_0_5", ascending=True)

            contract_rows_html = []
            for _, row in df_sorted.iterrows():
                is_correct = row.get("correct_at_0_5", 0) == 1.0
                row_cls = "correct" if is_correct else "incorrect"
                tick = "✓" if is_correct else "✗"
                contract_rows_html.append(
                    f'<tr class="{row_cls}">'
                    f"<td>{_esc(row.get('contract_title', row.get('title', '')))}</td>"
                    f'<td style="white-space:pre-wrap;max-width:340px;">{_esc(row.get("predicted_answer", ""))}</td>'
                    f'<td style="white-space:pre-wrap;max-width:340px;">{_esc(row.get("golden_answer", row.get("gold_answers", "")))}</td>'
                    f'<td style="text-align:center;font-weight:700;">{tick}</td>'
                    f"</tr>"
                )

            n_correct = int((df["correct_at_0_5"] == 1.0).sum())
            n_total = len(df)
            iter_notes = next(
                (r.get("notes", "") for r in rows if r.get("iter") == iter_key), ""
            )
            summary_label = f"Iter {iter_key} — {iter_status} — {n_correct}/{n_total} correct ({iter_acc:.0%})"

            # Prompt diff block
            diff_html = ""
            if iter_prompts and iter_key in iter_prompts and iter_key != 0:
                before_text, after_text = iter_prompts[iter_key]
                diff_html = f"""
  <details style="margin:12px 0 8px;">
    <summary style="cursor:pointer;font-size:0.82rem;font-weight:600;color:#475569;padding:4px 0;">
      ▶ Prompt diff
    </summary>
    <div style="margin-top:6px;">
      {_render_diff(before_text, after_text)}
    </div>
  </details>"""

            iter_blocks.append(f"""
<details {"open" if iter_key == 0 else ""}>
  <summary style="cursor:pointer;padding:8px 0;font-weight:600;">
    {_esc(summary_label)}
    <span class="badge" style="background:{badge_col};">{_esc(iter_status)}</span>
  </summary>
  {f'<p style="margin:8px 0 4px;font-style:italic;color:#475569;font-size:0.85rem;">{_esc(iter_notes)}</p>' if iter_notes and iter_key != 0 else ""}
  {diff_html}
  <table style="margin-top:8px;">
    <thead>
      <tr><th>Contract</th><th>Predicted Answer</th><th>Golden Answer</th><th>✓</th></tr>
    </thead>
    <tbody>
      {"".join(contract_rows_html)}
    </tbody>
  </table>
</details>""")

        if iter_blocks:
            contract_section_html = (
                '<h2 style="margin-top:32px;">Contract Results by Iteration</h2>\n'
                + "\n".join(iter_blocks)
            )

    # ---- Rows table -------------------------------------------------------
    def _f1_cell(iter_val: int) -> str:
        f1 = f1_by_iter.get(iter_val)
        return f"{f1:.3f}" if f1 is not None else "—"

    table_rows = "".join(
        f"<tr>"
        f"<td>{_esc(r.get('iter', ''))}</td>"
        f"<td>{_esc(r.get('status', ''))}</td>"
        f"<td>{float(r.get('correct_at_0_5', 0)):.3f}</td>"
        f"<td>{_f1_cell(int(r.get('iter', -1)))}</td>"
        f"<td>{_esc(r.get('notes', ''))}</td>"
        f"<td style='font-size:0.75rem;color:#64748b;'>{_esc(r.get('prompt_file', ''))}</td>"
        f"</tr>"
        for r in rows
    )

    progress_css = (
        _BASE_CSS
        + """
#tt { position:fixed; background:#1e293b; color:#e2e8f0; padding:6px 10px; border-radius:4px;
      font-size:0.8rem; pointer-events:none; display:none; z-index:999; max-width:320px; white-space:pre-wrap; }
.dot circle { cursor:pointer; }
.dot:hover circle { r: 8; }
"""
    )

    question_html = ""
    if question:
        question_html = f"""
    <div style="margin-top:12px;background:#f1f5f9;border-left:4px solid #6366f1;padding:10px 14px;border-radius:0 6px 6px 0;">
      <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:.05em;color:#6366f1;font-weight:700;margin-bottom:4px;">Question asked to LLM</div>
      <div style="font-weight:600;color:#1e293b;">{_esc(question)}</div>
    </div>"""

    run_dir_html = ""
    if run_dir is not None:
        run_dir_html = f"""
    <div style="margin-top:8px;font-size:0.8rem;color:#64748b;">
      Run outputs:
      <code style="background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:0.78rem;">{_esc(str(run_dir))}</code>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_esc(title)}</title>
  <style>
{progress_css}
  </style>
</head>
<body>
  <div id="tt"></div>
  <div class="page">
    <h1>{_esc(category)} — Autoresearch Progress</h1>
    {question_html}
    {run_dir_html}
    <div style="margin-top:20px; overflow-x:auto;">
      {svg}
    </div>
    <h2 style="margin-top:32px;">Experiment Log</h2>
    <table>
      <thead>
        <tr><th>Iter</th><th>Status</th><th>correct_at_0.5</th><th>Mean F1</th><th>Notes</th><th>Prompt File</th></tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
    {contract_section_html}
  </div>
  <script>
{js}
  </script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
