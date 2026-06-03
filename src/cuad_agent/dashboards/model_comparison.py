"""Static multi-model evaluation comparison dashboard.

Compares several evaluation runs (e.g. the context-mode variants produced by
``agent.py --all-context-modes``) on two axes:

1. Token-F1 accuracy by model.
2. No-answer vs answer detection accuracy by model — most CUAD gold answers are
   "no answer", so aggregate F1 is dominated by that class. This view shows how
   well each model identifies whether a clause is present at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cuad_agent.eval.summary import (
    detection_metrics,
    predicted_no_answer_mask,
)


def _per_category_rows(results: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if results.empty:
        return {}
    frame = results.copy()
    frame["_gold_no"] = frame["gold_marked_impossible"].astype(bool)
    frame["_pred_no"] = predicted_no_answer_mask(frame)
    rows: dict[int, dict[str, Any]] = {}
    for (question_index, category), group in frame.groupby(
        ["question_index", "category"]
    ):
        gold_no = group["_gold_no"]
        pred_no = group["_pred_no"]
        gold_no_count = int(gold_no.sum())
        gold_answer_count = int((~gold_no).sum())
        rows[int(question_index)] = {
            "question_index": int(question_index),
            "category": str(category),
            "count": int(len(group)),
            "mean_f1": float(group["token_f1"].mean() * 100),
            "correct_at_0_5": float(group["correct_at_0_5"].astype(float).mean() * 100),
            "gold_no_answer_count": gold_no_count,
            "gold_answer_count": gold_answer_count,
            "no_answer_detection_accuracy": (
                float((pred_no & gold_no).sum() / gold_no_count * 100)
                if gold_no_count
                else None
            ),
            "answer_detection_accuracy": (
                float((~pred_no & ~gold_no).sum() / gold_answer_count * 100)
                if gold_answer_count
                else None
            ),
        }
    return rows


def build_model_comparison_data(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn a list of model runs into the JSON payload consumed by the page.

    Each entry in ``models`` is a dict with keys ``label``, ``model_id``,
    ``context_mode``, ``summary`` (the summary dict) and ``results`` (a
    DataFrame of per-row results).
    """
    model_payloads: list[dict[str, Any]] = []
    category_order: list[dict[str, Any]] = []
    seen: set[int] = set()

    for model in models:
        results = model["results"]
        summary = model.get("summary", {})
        metrics = detection_metrics(results)
        per_category = _per_category_rows(results)
        for question_index in sorted(per_category):
            if question_index not in seen:
                seen.add(question_index)
                category_order.append(
                    {
                        "question_index": question_index,
                        "category": per_category[question_index]["category"],
                    }
                )
        model_payloads.append(
            {
                "label": str(model.get("label", model.get("model_id", "model"))),
                "model_id": str(model.get("model_id", "")),
                "context_mode": str(
                    model.get("context_mode", summary.get("context_mode", ""))
                ),
                "evaluation_href": f"evaluation_{model.get('model_id', '')}.html",
                "total": int(len(results)),
                "mean_f1": float(results["token_f1"].mean() * 100)
                if len(results)
                else 0.0,
                "correct_at_0_5": (
                    float(results["correct_at_0_5"].astype(float).mean() * 100)
                    if len(results)
                    else 0.0
                ),
                **metrics,
                "per_category": per_category,
            }
        )

    category_order.sort(key=lambda row: row["question_index"])
    return {"models": model_payloads, "categories": category_order}


def render_model_comparison_html(page_data: dict[str, Any]) -> str:
    data_json = json.dumps(page_data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUAD Eval Comparison</title>
  <style>
    :root {{
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
      --bubble-out: #005c4b;
      --shadow: rgba(0, 0, 0, 0.28);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ width: 100%; height: 100%; }}
    body {{
      margin: 0;
      background: var(--app-bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
    }}
    button {{ font: inherit; }}
    .global-header {{ height: 56px; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 0 18px; background: var(--panel-2); border-bottom: 1px solid var(--border); }}
    .brand {{ min-width: 0; display: flex; align-items: center; gap: 10px; font-weight: 700; }}
    .brand-mark {{ width: 30px; height: 30px; border-radius: 8px; background: linear-gradient(135deg, #00a884, #3b82f6); display: grid; place-items: center; color: white; font-size: 13px; }}
    .tabs {{ display: inline-flex; align-items: center; gap: 4px; padding: 4px; border: 1px solid var(--border); border-radius: 8px; background: rgba(0,0,0,0.16); }}
    .tab {{ color: var(--muted-2); text-decoration: none; padding: 7px 12px; border-radius: 6px; font-size: 13px; line-height: 1; white-space: nowrap; }}
    .tab:hover {{ color: var(--text); background: rgba(255,255,255,0.06); }}
    .tab.active {{ color: var(--text); background: var(--green); }}
    .app {{ display: grid; grid-template-columns: minmax(260px, 320px) minmax(0, 1fr); height: calc(100vh - 56px); height: calc(100dvh - 56px); min-height: 0; overflow: hidden; }}
    .sidebar {{ background: var(--panel); border-right: 1px solid var(--border); min-width: 0; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }}
    .side-top {{ height: 64px; flex: 0 0 64px; background: var(--panel-2); display: flex; align-items: center; padding: 0 16px; gap: 12px; }}
    .avatar {{ width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #00a884, #3b82f6); display: grid; place-items: center; color: white; font-weight: 700; flex: 0 0 auto; }}
    .side-title {{ font-size: 16px; font-weight: 650; }}
    .side-subtitle {{ color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .search {{ flex: 0 0 auto; padding: 10px 12px; border-bottom: 1px solid var(--border); }}
    .search input {{ width: 100%; border: 0; outline: 0; border-radius: 8px; background: var(--panel-2); color: var(--text); padding: 11px 14px; }}
    .conversations {{ flex: 1 1 auto; min-height: 0; overflow: auto; }}
    .conversation {{ width: 100%; border: 0; border-bottom: 1px solid var(--border); color: inherit; background: transparent; display: grid; grid-template-columns: 40px 1fr; gap: 12px; padding: 12px 14px; text-align: left; cursor: pointer; }}
    .conversation:hover, .conversation.active {{ background: var(--panel-2); }}
    .conversation h3 {{ margin: 0 0 3px; font-size: 14px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .conversation p {{ margin: 0; color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .chat {{ min-width: 0; min-height: 0; display: flex; flex-direction: column; background: #0b141a; overflow: hidden; }}
    .chat-top {{ height: 64px; flex: 0 0 64px; background: var(--panel-2); display: flex; align-items: center; padding: 0 16px; gap: 12px; }}
    .chat-title h1 {{ margin: 0 0 2px; font-size: 16px; font-weight: 650; }}
    .chat-subtitle {{ color: var(--muted); font-size: 12px; }}
    .messages {{ flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden; padding: 22px clamp(12px, 3vw, 42px); background-image: radial-gradient(circle at 12px 12px, rgba(255,255,255,0.035) 1px, transparent 1px); background-size: 24px 24px; }}
    .message {{ max-width: min(1440px, 100%); margin: 0 0 14px; padding: 12px 14px; border-radius: 8px; box-shadow: 0 1px 0 var(--shadow); }}
    .incoming {{ background: var(--bubble-in); }}
    .outgoing {{ background: var(--bubble-out); }}
    .message h2 {{ margin: 0 0 8px; font-size: 17px; }}
    .message p {{ margin: 0 0 8px; color: var(--muted-2); }}
    .legend {{ color: var(--muted); font-size: 12px; margin: 0; }}
    .table-wrap {{ width: 100%; overflow: auto; border: 1px solid var(--border); border-radius: 8px; background: rgba(0,0,0,0.14); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; text-align: left; white-space: nowrap; }}
    th {{ color: var(--muted-2); background: rgba(0,0,0,0.18); font-size: 12px; font-weight: 650; position: sticky; top: 0; z-index: 1; }}
    tr:last-child td {{ border-bottom: 0; }}
    td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .model-name {{ font-weight: 650; }}
    .model-name a {{ color: inherit; text-decoration: none; border-bottom: 1px dotted var(--muted); }}
    .model-name a:hover {{ color: var(--green); }}
    .pill {{ display: inline-flex; align-items: center; min-height: 20px; padding: 2px 8px; border-radius: 999px; background: rgba(255,255,255,0.08); color: var(--muted-2); font-size: 11px; }}
    .bar {{ position: relative; min-width: 120px; }}
    .bar > span {{ display: inline-block; }}
    .bar-track {{ display: block; height: 6px; margin-top: 4px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; }}
    .bar-fill {{ display: block; height: 100%; border-radius: 999px; background: var(--green); }}
    .good {{ color: #90f0d8; font-weight: 700; }}
    .mid {{ color: #ffd479; font-weight: 700; }}
    .bad {{ color: #ffb4bd; font-weight: 700; }}
    .best {{ box-shadow: inset 3px 0 0 var(--green); }}
  </style>
</head>
<body>
  <header class="global-header">
    <div class="brand"><div class="brand-mark">C</div><span>CUAD Review</span></div>
    <nav class="tabs" aria-label="Views">
      <a class="tab" href="explore.html">Explorer</a>
      <a class="tab active" aria-current="page">Comparison</a>
    </nav>
  </header>
  <div class="app">
    <aside class="sidebar">
      <div class="side-top">
        <div class="avatar">C</div>
        <div>
          <div class="side-title">Model Comparison</div>
          <div class="side-subtitle">F1 + no-answer / answer detection</div>
        </div>
      </div>
      <div class="search"><input id="search" type="search" placeholder="Search categories"></div>
      <div id="conversationList" class="conversations"></div>
    </aside>
    <main class="chat">
      <header class="chat-top">
        <div id="chatAvatar" class="avatar">O</div>
        <div>
          <h1 id="chatTitle" class="chat-title"></h1>
          <div id="chatSubtitle" class="chat-subtitle"></div>
        </div>
      </header>
      <section id="messages" class="messages" aria-live="polite"></section>
    </main>
  </div>
  <script id="app-data" type="application/json">{data_json}</script>
  <script>
    const data = JSON.parse(document.getElementById('app-data').textContent);
    const conversations = [
      {{ id: 'overview', kind: 'overview', title: 'Overview', subtitle: `${{data.models.length}} models compared`, avatar: 'O' }},
      ...data.categories.map(cat => ({{
        id: `cat-${{cat.question_index}}`,
        kind: 'category',
        title: `${{cat.question_index + 1}}. ${{cat.category}}`,
        subtitle: 'Per-category breakdown',
        avatar: String(cat.question_index + 1),
        category: cat,
      }})),
    ];
    let activeId = 'overview';

    const listEl = document.getElementById('conversationList');
    const messagesEl = document.getElementById('messages');
    const titleEl = document.getElementById('chatTitle');
    const subtitleEl = document.getElementById('chatSubtitle');
    const avatarEl = document.getElementById('chatAvatar');
    const searchEl = document.getElementById('search');

    function esc(value) {{
      return String(value ?? '').replace(/[&<>'"]/g, ch => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
      }}[ch]));
    }}

    function pct(value) {{
      if (value === null || value === undefined) return '<span class="pill">n/a</span>';
      return `${{Number(value).toFixed(1)}}%`;
    }}

    function cls(value) {{
      if (value === null || value === undefined) return '';
      const n = Number(value);
      if (n >= 75) return 'good';
      if (n >= 50) return 'mid';
      return 'bad';
    }}

    function scored(value) {{
      if (value === null || value === undefined) return '<span class="pill">n/a</span>';
      return `<span class="${{cls(value)}}">${{Number(value).toFixed(1)}}%</span>`;
    }}

    function bar(value) {{
      if (value === null || value === undefined) return '<span class="pill">n/a</span>';
      const n = Math.max(0, Math.min(100, Number(value)));
      return `<span class="bar"><span class="${{cls(value)}}">${{n.toFixed(1)}}%</span><span class="bar-track"><span class="bar-fill" style="width:${{n}}%"></span></span></span>`;
    }}

    function bestIndex(values) {{
      let best = -1, bestVal = -Infinity;
      values.forEach((v, i) => {{
        if (v !== null && v !== undefined && Number(v) > bestVal) {{ bestVal = Number(v); best = i; }}
      }});
      return best;
    }}

    function renderList() {{
      const needle = searchEl.value.trim().toLowerCase();
      listEl.innerHTML = conversations
        .filter(c => !needle || c.title.toLowerCase().includes(needle))
        .map(c => `
          <button class="conversation ${{c.id === activeId ? 'active' : ''}}" data-id="${{esc(c.id)}}">
            <div class="avatar">${{esc(c.avatar)}}</div>
            <div><h3>${{esc(c.title)}}</h3><p>${{esc(c.subtitle)}}</p></div>
          </button>
        `).join('');
      listEl.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {{
        activeId = btn.dataset.id;
        render();
      }}));
    }}

    function renderOverview() {{
      const f1Best = bestIndex(data.models.map(m => m.mean_f1));
      const naBest = bestIndex(data.models.map(m => m.no_answer_detection_accuracy));
      const ansBest = bestIndex(data.models.map(m => m.answer_detection_accuracy));
      const rows = data.models.map((m, i) => `
        <tr>
          <td class="model-name">${{m.evaluation_href ? `<a href="${{esc(m.evaluation_href)}}">${{esc(m.label)}}</a>` : esc(m.label)}}<br><span class="pill">${{esc(m.context_mode || m.model_id)}}</span></td>
          <td class="num">${{m.total}}</td>
          <td class="num ${{i === f1Best ? 'best' : ''}}">${{scored(m.mean_f1)}}</td>
          <td class="num">${{scored(m.correct_at_0_5)}}</td>
          <td class="num ${{i === naBest ? 'best' : ''}}">${{scored(m.no_answer_detection_accuracy)}}<br><span class="pill">${{m.gold_no_answer_count}} gold</span></td>
          <td class="num ${{i === ansBest ? 'best' : ''}}">${{scored(m.answer_detection_accuracy)}}<br><span class="pill">${{m.gold_answer_count}} gold</span></td>
          <td class="num">${{scored(m.detection_accuracy)}}</td>
        </tr>
      `).join('');
      const first = data.models[0] || {{}};
      messagesEl.innerHTML = `
        <article class="message incoming">
          <h2>Accuracy by model</h2>
          <p>Token-F1 next to no-answer / answer detection accuracy. Most gold answers are "no answer"
          (${{first.gold_no_answer_count || 0}} of ${{first.total || 0}} examples), so high mean F1 can hide
          weak detection of clauses that <em>are</em> present.</p>
          <div class="table-wrap">
            <table>
              <thead><tr>
                <th>Model</th><th class="num">Examples</th><th class="num">Mean F1</th><th class="num">Correct@0.5</th>
                <th class="num">No-answer acc</th><th class="num">Answer acc</th><th class="num">Detection acc</th>
              </tr></thead>
              <tbody>${{rows}}</tbody>
            </table>
          </div>
          <p class="legend">No-answer acc = share of gold "no answer" rows the model correctly flagged absent.
          Answer acc = share of gold answer rows the model correctly identified as present.
          Detection acc = overall present/absent classification accuracy. Green outline marks the best model per column.</p>
        </article>
      `;
    }}

    function renderCategory(category) {{
      const qi = category.question_index;
      const rows = data.models.map(m => {{
        const c = m.per_category[qi] || m.per_category[String(qi)] || {{}};
        return `
          <tr>
            <td class="model-name">${{esc(m.label)}}<br><span class="pill">${{esc(m.context_mode || m.model_id)}}</span></td>
            <td>${{bar(c.mean_f1)}}</td>
            <td class="num">${{scored(c.correct_at_0_5)}}</td>
            <td class="num">${{scored(c.no_answer_detection_accuracy)}}${{c.gold_no_answer_count !== undefined ? `<br><span class="pill">${{c.gold_no_answer_count}} gold</span>` : ''}}</td>
            <td class="num">${{scored(c.answer_detection_accuracy)}}${{c.gold_answer_count !== undefined ? `<br><span class="pill">${{c.gold_answer_count}} gold</span>` : ''}}</td>
          </tr>
        `;
      }}).join('');
      const sample = data.models.map(m => m.per_category[qi] || m.per_category[String(qi)]).find(Boolean) || {{}};
      messagesEl.innerHTML = `
        <article class="message outgoing"><p>Question ${{qi + 1}} · ${{esc(category.category)}} · ${{sample.count || 0}} contracts (${{sample.gold_no_answer_count || 0}} gold no-answer)</p></article>
        <article class="message incoming">
          <h2>${{esc(category.category)}}</h2>
          <div class="table-wrap">
            <table>
              <thead><tr>
                <th>Model</th><th>Mean F1</th><th class="num">Correct@0.5</th>
                <th class="num">No-answer acc</th><th class="num">Answer acc</th>
              </tr></thead>
              <tbody>${{rows}}</tbody>
            </table>
          </div>
        </article>
      `;
    }}

    function render() {{
      const active = conversations.find(c => c.id === activeId) || conversations[0];
      titleEl.textContent = active.title;
      subtitleEl.textContent = active.subtitle;
      avatarEl.textContent = active.avatar;
      if (active.kind === 'overview') renderOverview();
      else renderCategory(active.category);
      renderList();
      messagesEl.scrollTop = 0;
    }}

    searchEl.addEventListener('input', renderList);
    render();
  </script>
</body>
</html>
"""


def write_model_comparison_html(
    models: list[dict[str, Any]],
    output_path: Path,
) -> None:
    page_data = build_model_comparison_data(models)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_model_comparison_html(page_data), encoding="utf-8")
