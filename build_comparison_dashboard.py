#!/usr/bin/env python3
"""
Generate a three-way comparison dashboard: prompt_v1 → prompt_v2 → autoresearch best.

Output: dashboards/ar_comparison.html

Usage:
    uv run python build_comparison_dashboard.py
"""

from __future__ import annotations

import csv
import difflib
import html as html_module
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(text: object) -> str:
    return html_module.escape(str(text))


def _load_prompt_module(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_prompts", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return {}
    return getattr(mod, "CATEGORY_SYSTEM_PROMPTS", {})


def _render_diff(
    before: str, after: str, from_label: str = "before", to_label: str = "after"
) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=from_label,
            tofile=to_label,
            lineterm="",
        )
    )
    if not diff_lines:
        return (
            '<p style="color:#8696a0;font-style:italic;margin:6px 0;">(no change)</p>'
        )

    parts = []
    for line in diff_lines:
        escaped = html_module.escape(line)
        if line.startswith("+++") or line.startswith("---"):
            parts.append(
                f'<span style="color:#94a3b8;font-weight:600;">{escaped}</span>\n'
            )
        elif line.startswith("@@"):
            parts.append(
                f'<span style="background:#1e3a5f;color:#7dd3fc;display:block;padding:0 4px;">{escaped}</span>\n'
            )
        elif line.startswith("+"):
            parts.append(
                f'<span style="background:#14302a;color:#86efac;display:block;padding:0 4px;">{escaped}</span>\n'
            )
        elif line.startswith("-"):
            parts.append(
                f'<span style="background:#3b1219;color:#fca5a5;display:block;padding:0 4px;">{escaped}</span>\n'
            )
        else:
            parts.append(
                f'<span style="display:block;padding:0 4px;">{escaped}</span>\n'
            )

    inner = "".join(parts)
    return (
        '<pre style="background:#0b141a;color:#e9edef;padding:14px;border-radius:6px;'
        "border:1px solid #26353d;font-size:0.78rem;line-height:1.5;overflow-x:auto;"
        'white-space:pre-wrap;margin-top:6px;">'
        f"{inner}</pre>"
    )


def _delta_html(delta: float) -> str:
    if abs(delta) < 0.001:
        return '<span style="color:#8696a0;font-weight:700;">—</span>'
    sign = "+" if delta > 0 else ""
    color = "#90f0d8" if delta > 0 else "#ffb4bd"
    return f'<b style="color:{color};">{sign}{delta:.0%}</b>'


def _acc_bar(acc: float, label: str, color: str) -> str:
    pct = acc * 100
    return (
        f'<div style="margin-bottom:8px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        f'<span style="font-size:12px;color:#8696a0;">{_esc(label)}</span>'
        f'<span style="font-size:13px;font-weight:700;color:#e9edef;">{pct:.0f}%</span>'
        f"</div>"
        f'<div style="background:#26353d;border-radius:4px;height:8px;overflow:hidden;">'
        f'<div style="background:{color};width:{pct:.1f}%;height:100%;border-radius:4px;"></div>'
        f"</div>"
        f"</div>"
    )


def _score_color(acc: float) -> str:
    if acc >= 0.9:
        return "#90f0d8"
    if acc < 0.65:
        return "#ffb4bd"
    return "#e9edef"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_ar_data(output_dir: Path) -> dict[int, dict]:
    ar_base = output_dir / "autoresearch"
    if not ar_base.exists():
        return {}

    results: dict[int, dict] = {}
    for q_dir in sorted(ar_base.iterdir()):
        if not q_dir.is_dir() or not q_dir.name.startswith("q"):
            continue
        date_dirs = sorted(
            d for d in q_dir.iterdir() if d.is_dir() and not d.name.startswith("__")
        )
        if not date_dirs:
            continue
        run_dir = date_dirs[-1]
        tsv_path = run_dir / "results.tsv"
        if not tsv_path.exists():
            continue

        with open(tsv_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if not rows:
            continue

        q_idx = int(rows[0]["question_index"])
        category = rows[0]["category"]
        baseline_acc = float(rows[0]["correct_at_0_5"])
        best_acc = max(float(r["correct_at_0_5"]) for r in rows)
        kept_rows = [r for r in rows if r["status"] == "keep"]
        best_iter = max(int(r["iter"]) for r in kept_rows) if kept_rows else 0

        results[q_idx] = {
            "category": category,
            "run_dir": run_dir,
            "run_date": run_dir.name,
            "baseline_acc": baseline_acc,
            "best_acc": best_acc,
            "kept": len(kept_rows),
            "best_iter": best_iter,
            "rows": rows,
        }

    return results


def _get_best_ar_prompt(q_idx: int, ar_data: dict, prompts_root: Path) -> str | None:
    if q_idx not in ar_data:
        return None
    info = ar_data[q_idx]
    if info["kept"] == 0:
        return None

    run_date = info["run_date"]
    best_iter = info["best_iter"]
    category = info["category"]

    # Primary: prompts/autoresearch/q{idx:02d}/{date}/v2_r1_i{iter}.py
    prompt_path = (
        prompts_root
        / "autoresearch"
        / f"q{q_idx:02d}"
        / run_date
        / f"v2_r1_i{best_iter}.py"
    )
    if prompt_path.exists():
        return _load_prompt_module(prompt_path).get(category)

    # Fallback: accepted.py in iteration dir
    accepted_path = info["run_dir"] / f"iter_{best_iter}" / "accepted.py"
    if accepted_path.exists():
        return _load_prompt_module(accepted_path).get(category)

    return None


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


_STATUS_COLORS = {
    "keep": "#22c55e",
    "discard": "#f59e0b",
    "crash": "#ef4444",
    "baseline": "#94a3b8",
}

_BASE_CSS = """
    :root {
      color-scheme: dark;
      --app-bg: #0b141a;
      --panel: #111b21;
      --panel-2: #202c33;
      --border: #26353d;
      --text: #e9edef;
      --muted: #8696a0;
      --muted-2: #aebac1;
      --green: #00a884;
      --bubble-in: #202c33;
      --shadow: rgba(0, 0, 0, 0.28);
    }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; }
    body {
      margin: 0;
      background: var(--app-bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
    }
    button { font: inherit; cursor: pointer; }
    .global-header {
      height: 56px;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 0 18px;
      background: var(--panel-2);
      border-bottom: 1px solid var(--border);
    }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; }
    .brand-mark {
      width: 30px; height: 30px; border-radius: 8px;
      background: linear-gradient(135deg, #00a884, #3b82f6);
      display: grid; place-items: center; color: white; font-size: 13px;
    }
    .app {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      height: calc(100vh - 56px);
      min-height: 0;
      overflow: hidden;
    }
    .sidebar {
      background: var(--panel);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }
    .side-top {
      height: 64px;
      flex: 0 0 64px;
      background: var(--panel-2);
      display: flex;
      align-items: center;
      padding: 0 16px;
      gap: 12px;
      border-bottom: 1px solid var(--border);
    }
    .side-title { font-size: 16px; font-weight: 650; }
    .side-subtitle { color: var(--muted); font-size: 12px; }
    .search { padding: 10px 12px; border-bottom: 1px solid var(--border); }
    .search input {
      width: 100%; border: 0; outline: 0; border-radius: 8px;
      background: var(--panel-2); color: var(--text); padding: 9px 12px;
    }
    .conversations { flex: 1 1 auto; min-height: 0; overflow-y: auto; }
    .conv-btn {
      width: 100%; border: 0; border-bottom: 1px solid var(--border);
      color: inherit; background: transparent;
      display: grid; grid-template-columns: 48px 1fr;
      gap: 12px; padding: 12px 14px; text-align: left;
    }
    .conv-btn:hover, .conv-btn.active { background: var(--panel-2); }
    .conv-btn h3 { margin: 0 0 3px; font-size: 15px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .conv-btn p { margin: 0; color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .avatar {
      width: 40px; height: 40px; border-radius: 50%;
      background: linear-gradient(135deg, #00a884, #3b82f6);
      display: grid; place-items: center; color: white;
      font-weight: 700; flex: 0 0 auto; font-size: 11px;
    }
    .main-area { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
    .panel { display: none; flex-direction: column; min-height: 0; flex: 1; overflow: hidden; }
    .panel.active { display: flex; }
    .chat-top {
      height: 64px; flex: 0 0 64px;
      background: var(--panel-2);
      display: flex; align-items: center;
      padding: 0 16px; gap: 12px;
      border-bottom: 1px solid var(--border);
    }
    .chat-title h1 { margin: 0 0 2px; font-size: 16px; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .chat-subtitle { color: var(--muted); font-size: 12px; }
    .messages {
      flex: 1 1 auto; min-height: 0; overflow-y: auto;
      padding: 20px clamp(12px, 3vw, 40px);
      background-image: radial-gradient(circle at 12px 12px, rgba(255,255,255,0.03) 1px, transparent 1px);
      background-size: 24px 24px;
    }
    .message {
      max-width: min(1440px, 100%);
      margin: 0 0 14px;
      padding: 16px 18px;
      border-radius: 8px;
      box-shadow: 0 1px 0 var(--shadow);
      background: var(--bubble-in);
    }
    .message h2 { margin: 0 0 10px; font-size: 15px; font-weight: 700; color: var(--muted-2); }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(100px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      background: rgba(255,255,255,0.05);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .metric-val { display: block; font-size: 20px; font-weight: 700; margin-bottom: 2px; }
    .metric-lbl { color: var(--muted-2); font-size: 11px; }
    .table-wrap {
      width: 100%; overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: rgba(0,0,0,0.14);
    }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 9px 12px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
    th {
      color: var(--muted-2); background: rgba(0,0,0,0.2);
      font-size: 12px; font-weight: 650;
      position: sticky; top: 0; z-index: 1;
    }
    tr:last-child td { border-bottom: 0; }
    .summary-table { min-width: 680px; }
    .col-q { width: 46px; color: var(--muted); }
    .col-cat { font-weight: 600; min-width: 200px; }
    .col-num { width: 80px; text-align: center; }
    .col-delta { width: 90px; text-align: center; }
    .pill {
      display: inline-flex; align-items: center;
      padding: 2px 8px; border-radius: 999px;
      font-size: 12px; font-weight: 600;
    }
    details { margin: 0; }
    details summary { cursor: pointer; list-style: none; }
    details summary::-webkit-details-marker { display: none; }
    @media (max-width: 760px) {
      body { overflow: auto; }
      .app { grid-template-columns: 1fr; height: auto; }
      .sidebar { max-height: 40vh; border-right: 0; border-bottom: 1px solid var(--border); }
      .meta-grid { grid-template-columns: repeat(2, 1fr); }
    }
"""

_JS = """
function showPanel(id, btn) {
  document.querySelectorAll('.panel').forEach(p => {
    p.style.display = 'none';
    p.classList.remove('active');
  });
  document.querySelectorAll('.conv-btn').forEach(b => b.classList.remove('active'));
  const panel = document.getElementById('panel-' + id);
  if (panel) { panel.style.display = 'flex'; panel.classList.add('active'); }
  if (btn) btn.classList.add('active');
}

function filterQuestions(val) {
  const q = val.toLowerCase();
  document.querySelectorAll('.conv-btn').forEach(btn => {
    btn.style.display = btn.textContent.toLowerCase().includes(q) ? '' : 'none';
  });
}

// Activate first item on load
(function() {
  const first = document.querySelector('.conv-btn.active');
  if (first) {
    const onclick = first.getAttribute('onclick');
    if (onclick) {
      const m = onclick.match(/showPanel[(]'([^']+)'/);
      if (m) showPanel(m[1], first);
    }
  }
})();
"""


def _build_summary_panel(questions: list[dict]) -> str:
    rows_html = []
    for q in questions:
        v1 = q["v1_acc"]
        v2 = q["v2_acc"]
        ar = q["ar_acc"]
        q_idx = q["q_idx"]

        delta_v1_v2 = v2 - v1
        v1_color = _score_color(v1)
        v2_color = _score_color(v2)

        if ar is not None:
            delta_v2_ar = ar - v2
            ar_cell = f'<b style="color:{_score_color(ar)};">{ar:.0%}</b>'
            ar_delta_cell = _delta_html(delta_v2_ar)
            link = (
                f'<a href="#" onclick="showPanel(\'q{q_idx:02d}\', '
                f'document.querySelector(\'[data-q=\\"{q_idx:02d}\\"]\'));return false;" '
                f'style="color:inherit;text-decoration:none;">'
                f"{_esc(q['category'])}</a>"
            )
        else:
            ar_cell = '<span style="color:#8696a0;">—</span>'
            ar_delta_cell = '<span style="color:#8696a0;">—</span>'
            link = _esc(q["category"])

        rows_html.append(
            f"<tr>"
            f'<td class="col-q">q{q_idx:02d}</td>'
            f'<td class="col-cat">{link}</td>'
            f'<td class="col-num"><b style="color:{v1_color};">{v1:.0%}</b></td>'
            f'<td class="col-num"><b style="color:{v2_color};">{v2:.0%}</b></td>'
            f'<td class="col-num">{ar_cell}</td>'
            f'<td class="col-delta">{_delta_html(delta_v1_v2)}</td>'
            f'<td class="col-delta">{ar_delta_cell}</td>'
            f"</tr>"
        )

    ar_count = sum(1 for q in questions if q["ar_acc"] is not None)
    improved = sum(
        1 for q in questions if q["ar_acc"] is not None and q["ar_acc"] > q["v2_acc"]
    )

    return f"""
<div id="panel-summary" class="panel active">
  <div class="chat-top">
    <div class="chat-title">
      <h1>Prompt Evolution Summary</h1>
      <div class="chat-subtitle">
        {len(questions)} questions · {ar_count} autoresearched · {improved} improved over v2
      </div>
    </div>
  </div>
  <div class="messages">
    <div class="message">
      <h2>Accuracy by Prompt Version</h2>
      <div class="table-wrap">
        <table class="summary-table">
          <thead>
            <tr>
              <th class="col-q">Q#</th>
              <th class="col-cat">Category</th>
              <th class="col-num">v1</th>
              <th class="col-num">v2</th>
              <th class="col-num">AR Best</th>
              <th class="col-delta">v1 → v2</th>
              <th class="col-delta">v2 → AR</th>
            </tr>
          </thead>
          <tbody>
            {"".join(rows_html)}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>"""


def _build_question_panel(q: dict) -> str:
    q_idx = q["q_idx"]
    category = q["category"]
    v1_acc = q["v1_acc"]
    v2_acc = q["v2_acc"]
    ar_acc = q["ar_acc"]
    kept = q["ar_kept"]
    run_date = q["ar_run_date"]

    delta_v1_v2 = v2_acc - v1_acc
    delta_v2_ar = ar_acc - v2_acc

    # Accuracy bars
    bars = (
        _acc_bar(v1_acc, "Prompt v1", "#94a3b8")
        + _acc_bar(v2_acc, "Prompt v2", "#3b82f6")
        + _acc_bar(ar_acc, "Autoresearch Best", "#00a884")
    )

    # Metric cards
    def _metric(val_html: str, label: str) -> str:
        return (
            f'<div class="metric">'
            f'<span class="metric-val">{val_html}</span>'
            f'<span class="metric-lbl">{_esc(label)}</span>'
            f"</div>"
        )

    metrics = (
        _metric(
            f'<span style="color:{_score_color(v1_acc)};">{v1_acc:.0%}</span>',
            "Prompt v1",
        )
        + _metric(
            f'<span style="color:{_score_color(v2_acc)};">{v2_acc:.0%}</span>',
            "Prompt v2",
        )
        + _metric(
            f'<span style="color:{_score_color(ar_acc)};">{ar_acc:.0%}</span>',
            f"AR Best ({kept} kept)",
        )
        + _metric(
            _delta_html(delta_v2_ar),
            "v2 → AR Δ",
        )
    )

    # Prompt comparison
    v1_text = q.get("v1_prompt", "")
    v2_text = q.get("v2_prompt", "")
    ar_text = q.get("ar_prompt") or v2_text
    ar_is_same = not q.get("ar_prompt")

    ar_note = " (same as v2 — no improvement found)" if ar_is_same else ""
    v1_v2_diff = _render_diff(v1_text, v2_text, "prompt_v1", "prompt_v2")
    v2_ar_diff = _render_diff(v2_text, ar_text, "prompt_v2", "autoresearch_best")

    prompt_section = f"""
<details style="margin-bottom:14px;background:rgba(255,255,255,0.03);border:1px solid var(--border);border-radius:8px;overflow:hidden;">
  <summary style="padding:12px 16px;font-weight:600;color:var(--muted-2);display:flex;align-items:center;gap:8px;">
    <span style="font-size:16px;user-select:none;">▶</span>
    Prompt Evolution: v1 → v2 → Autoresearch{_esc(ar_note)}
  </summary>
  <div style="padding:0 16px 16px;">
    <div style="margin-top:12px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:4px;">v1 → v2 diff</div>
      {v1_v2_diff}
    </div>
    <div style="margin-top:12px;">
      <div style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:4px;">v2 → autoresearch best{_esc(ar_note)}</div>
      {v2_ar_diff}
    </div>
    <div style="margin-top:12px;display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">
      <details style="background:rgba(0,0,0,0.2);border:1px solid var(--border);border-radius:6px;">
        <summary style="padding:8px 10px;font-size:12px;font-weight:600;color:#94a3b8;">v1 prompt</summary>
        <pre style="margin:0;padding:10px;background:#0b141a;color:#e9edef;font-size:0.75rem;line-height:1.5;overflow-x:auto;white-space:pre-wrap;">{_esc(v1_text)}</pre>
      </details>
      <details style="background:rgba(0,0,0,0.2);border:1px solid var(--border);border-radius:6px;">
        <summary style="padding:8px 10px;font-size:12px;font-weight:600;color:#7dd3fc;">v2 prompt</summary>
        <pre style="margin:0;padding:10px;background:#0b141a;color:#e9edef;font-size:0.75rem;line-height:1.5;overflow-x:auto;white-space:pre-wrap;">{_esc(v2_text)}</pre>
      </details>
      <details style="background:rgba(0,0,0,0.2);border:1px solid var(--border);border-radius:6px;">
        <summary style="padding:8px 10px;font-size:12px;font-weight:600;color:#86efac;">Autoresearch best{_esc(ar_note)}</summary>
        <pre style="margin:0;padding:10px;background:#0b141a;color:#e9edef;font-size:0.75rem;line-height:1.5;overflow-x:auto;white-space:pre-wrap;">{_esc(ar_text)}</pre>
      </details>
    </div>
  </div>
</details>"""

    # Iteration log table
    ar_rows = q.get("ar_rows", [])
    iter_rows = []
    for row in ar_rows:
        status = row["status"]
        acc = float(row["correct_at_0_5"])
        notes = row.get("notes", "")
        s_color = _STATUS_COLORS.get(status, "#94a3b8")
        iter_rows.append(
            f"<tr>"
            f'<td style="color:var(--muted);width:36px;">{_esc(row["iter"])}</td>'
            f'<td style="width:90px;"><span class="pill" style="background:rgba(0,0,0,0.3);color:{s_color};">{_esc(status)}</span></td>'
            f'<td style="width:80px;text-align:center;"><b style="color:{_score_color(acc)};">{acc:.0%}</b></td>'
            f'<td style="font-size:12px;color:#8696a0;max-width:500px;">{_esc(notes)}</td>'
            f"</tr>"
        )

    return f"""
<div id="panel-q{q_idx:02d}" class="panel">
  <div class="chat-top">
    <div class="chat-title">
      <h1>q{q_idx:02d} — {_esc(category)}</h1>
      <div class="chat-subtitle">Run: {_esc(run_date)} · {kept} improvement(s) kept</div>
    </div>
  </div>
  <div class="messages">
    <div class="message">
      {prompt_section}
      <div class="meta-grid">{metrics}</div>
      <div style="margin-top:4px;">{bars}</div>
    </div>
    <div class="message">
      <h2>Autoresearch Iteration Log</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th style="width:36px;">Iter</th>
              <th style="width:90px;">Status</th>
              <th style="width:80px;text-align:center;">Accuracy</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody>{"".join(iter_rows)}</tbody>
        </table>
      </div>
    </div>
  </div>
</div>"""


def generate_html(questions: list[dict]) -> str:
    ar_questions = [q for q in questions if q["ar_acc"] is not None]
    ar_count = len(ar_questions)

    # Sidebar
    summary_btn = (
        '<button class="conv-btn active" onclick="showPanel(\'summary\', this)">'
        '<div class="avatar">Σ</div>'
        "<div>"
        f"<h3>Summary</h3>"
        f"<p>All {len(questions)} questions · v1 → v2 → AR</p>"
        "</div>"
        "</button>"
    )

    q_buttons = []
    for q in ar_questions:
        q_idx = q["q_idx"]
        ar_acc = q["ar_acc"]
        delta = ar_acc - q["v2_acc"]
        delta_str = f"{delta:+.0%}" if abs(delta) > 0.001 else "no change"
        improved_marker = " ▲" if delta > 0.001 else (" ▼" if delta < -0.001 else "")
        q_buttons.append(
            f'<button class="conv-btn" data-q="{q_idx:02d}" '
            f"onclick=\"showPanel('q{q_idx:02d}', this)\">"
            f'<div class="avatar">q{q_idx:02d}</div>'
            f"<div>"
            f"<h3>{_esc(q['category'])}</h3>"
            f"<p>AR: {ar_acc:.0%}{improved_marker} · v2→AR {delta_str}</p>"
            f"</div>"
            f"</button>"
        )

    sidebar_html = summary_btn + "\n".join(q_buttons)

    # Panels
    summary_panel = _build_summary_panel(questions)
    q_panels = "\n".join(_build_question_panel(q) for q in ar_questions)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUAD Prompt Evolution: v1 → v2 → AR</title>
  <style>
{_BASE_CSS}
  </style>
</head>
<body>
  <div class="global-header">
    <div class="brand">
      <div class="brand-mark">AR</div>
      <span>CUAD Prompt Evolution</span>
    </div>
    <span style="color:var(--muted);font-size:13px;">
      prompt_v1 → prompt_v2 → autoresearch best · {ar_count} questions researched
    </span>
  </div>
  <div class="app">
    <div class="sidebar">
      <div class="side-top">
        <div class="avatar">Q</div>
        <div>
          <div class="side-title">Questions</div>
          <div class="side-subtitle">{ar_count} with autoresearch data</div>
        </div>
      </div>
      <div class="search">
        <input type="search" placeholder="Filter questions…" oninput="filterQuestions(this.value)">
      </div>
      <div class="conversations">
        {sidebar_html}
      </div>
    </div>
    <div class="main-area">
      {summary_panel}
      {q_panels}
    </div>
  </div>
  <script>
{_JS}
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    output_dir = ROOT / "outputs"
    prompts_root = ROOT / "prompts"
    dashboard_dir = ROOT / "dashboards"

    bc_path = output_dir / "eval-raw" / "baseline_comparison.json"
    with open(bc_path) as f:
        bc = json.load(f)

    # v1 and v2 per-category accuracy (stored as percentages 0–100)
    cat_data: dict[int, dict] = {}
    for entry in bc["per_category"]:
        q_idx = entry["question_index"]
        cat_data[q_idx] = {
            "category": entry["category"],
            "v1_acc": entry["baseline_correct_at_0_5"] / 100.0,
            "v2_acc": entry["candidate_correct_at_0_5"] / 100.0,
        }

    ar_data = _load_ar_data(output_dir)
    v1_prompts = _load_prompt_module(prompts_root / "system_prompts_v1.py")
    v2_prompts = _load_prompt_module(prompts_root / "system_prompts_v2.py")

    questions = []
    for q_idx in sorted(cat_data.keys()):
        info = cat_data[q_idx].copy()
        info["q_idx"] = q_idx

        if q_idx in ar_data:
            ar = ar_data[q_idx]
            info.update(
                {
                    "ar_acc": ar["best_acc"],
                    "ar_run_dir": ar["run_dir"],
                    "ar_kept": ar["kept"],
                    "ar_best_iter": ar["best_iter"],
                    "ar_rows": ar["rows"],
                    "ar_run_date": ar["run_date"],
                }
            )
        else:
            info["ar_acc"] = None

        category = info["category"]
        info["v1_prompt"] = v1_prompts.get(category, "")
        info["v2_prompt"] = v2_prompts.get(category, "")
        info["ar_prompt"] = _get_best_ar_prompt(q_idx, ar_data, prompts_root)

        questions.append(info)

    html = generate_html(questions)
    out_path = dashboard_dir / "autoresearch_comparison.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Written: {out_path}")
    improved = sum(
        1 for q in questions if q["ar_acc"] is not None and q["ar_acc"] > q["v2_acc"]
    )
    print(
        f"  {len([q for q in questions if q['ar_acc'] is not None])} questions with AR data"
    )
    print(f"  {improved} improved over v2")


if __name__ == "__main__":
    main()
