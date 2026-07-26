"""Static evaluation dashboard renderer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cuad_agent.data.sampling import evaluation_row_id

__all__ = [
    "build_evaluation_page_data",
    "parse_gold_answers",
    "render_evaluation_html",
    "write_evaluation_html",
]


def parse_gold_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def build_evaluation_page_data(
    results: pd.DataFrame,
    summary: dict[str, Any],
) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    per_category = {
        int(row["question_index"]): row for row in summary.get("per_category", [])
    }
    comparison = summary.get("baseline_comparison")
    comparison_examples: dict[str, dict[str, Any]] = {}
    comparison_categories: dict[int, dict[str, Any]] = {}
    if isinstance(comparison, dict):
        comparison_examples = {
            str(row.get("row_id")): row
            for row in comparison.get("examples", [])
            if isinstance(row, dict) and row.get("row_id") is not None
        }
        comparison_categories = {
            int(row["question_index"]): row
            for row in comparison.get("per_category", [])
            if isinstance(row, dict) and row.get("question_index") is not None
        }
    ordered_results = results.copy()
    if "row_id" not in ordered_results.columns:
        ordered_results["row_id"] = [
            evaluation_row_id(document_row_id, question_index)
            for document_row_id, question_index in zip(
                ordered_results["document_row_id"],
                ordered_results["question_index"],
                strict=True,
            )
        ]
    ordered_results = ordered_results.sort_values(["question_index", "document_row_id"])
    for question_index, question_rows in ordered_results.groupby("question_index"):
        first = question_rows.iloc[0]
        metrics = per_category.get(int(question_index), {})
        comparison_metrics = comparison_categories.get(int(question_index), {})
        rows: list[dict[str, Any]] = []
        for row in question_rows.to_dict("records"):
            row_comparison = comparison_examples.get(str(row["row_id"]))
            rows.append(
                {
                    "row_id": str(row["row_id"]),
                    "document_row_id": int(row["document_row_id"]),
                    "title": str(row["title"]),
                    "gold_answers": parse_gold_answers(row["gold_answers"]),
                    "predicted_answer": str(row["predicted_answer"]),
                    "predicted_marked_impossible": bool(
                        row["predicted_marked_impossible"]
                    ),
                    "gold_marked_impossible": bool(row["gold_marked_impossible"]),
                    "token_f1": float(row["token_f1"]),
                    "correct_at_0_5": bool(row["correct_at_0_5"]),
                    "comparison": row_comparison,
                }
            )
        questions.append(
            {
                "question_index": int(question_index),
                "category": str(first.category),
                "category_description": str(first.category_description),
                "answer_format": str(first.answer_format),
                "question": str(first.question),
                "mean_token_f1": float(metrics.get("mean_token_f1", 0.0)),
                "correct_at_0_5": float(metrics.get("correct_at_0_5", 0.0)),
                "count": int(metrics.get("count", len(question_rows))),
                "comparison": comparison_metrics,
                "results": rows,
            }
        )
    category_context = {
        question["question_index"]: {
            "question_index": question["question_index"],
            "category": question["category"],
            "category_description": question["category_description"],
            "answer_format": question["answer_format"],
            "question": question["question"],
        }
        for question in questions
    }
    per_category = [
        {
            **category_context.get(int(row.get("question_index", -1)), {}),
            **row,
            "comparison": comparison_categories.get(
                int(row.get("question_index", -1)), {}
            ),
        }
        for row in summary.get("per_category", [])
    ]
    return {
        "summary": {
            key: summary[key]
            for key in [
                "sample_size",
                "seed",
                "model_id",
                "total_examples",
                "questions_per_contract",
                "agent_count",
                "model",
                "temperature",
                "max_tokens",
                "num_threads",
                "dry_run",
                "overlap_accuracy_mean_f1",
                "correct_at_0_5",
            ]
            if key in summary
        },
        "comparison": comparison if isinstance(comparison, dict) else None,
        "per_category": per_category,
        "questions": questions,
    }


def render_evaluation_html(page_data: dict[str, Any]) -> str:
    data_json = json.dumps(page_data, ensure_ascii=False).replace("</", "<\\/")
    model_id = str(page_data.get("summary", {}).get("model_id", "model"))
    evaluation_href = f"evaluation_{model_id}.html"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CUAD Evaluation</title>
  <style>
    :root {{
      color-scheme: dark;
      --app-bg: #0b141a;
      --panel: #111b21;
      --panel-2: #202c33;
      --panel-3: #182229;
      --border: #26353d;
      --text: #e9edef;
      --muted: #8696a0;
      --muted-2: #aebac1;
      --green: #00a884;
      --bubble-in: #202c33;
      --bubble-out: #005c4b;
      --danger: #f15c6d;
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
    .app {{ display: grid; grid-template-columns: minmax(300px, 380px) minmax(0, 1fr); height: calc(100vh - 56px); height: calc(100dvh - 56px); min-height: 0; overflow: hidden; }}
    .sidebar {{ background: var(--panel); border-right: 1px solid var(--border); min-width: 0; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }}
    .side-top, .chat-top {{ height: 64px; flex: 0 0 64px; background: var(--panel-2); display: flex; align-items: center; padding: 0 16px; gap: 12px; }}
    .avatar {{ width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #00a884, #3b82f6); display: grid; place-items: center; color: white; font-weight: 700; flex: 0 0 auto; }}
    .side-title {{ font-size: 16px; font-weight: 650; }}
    .side-subtitle, .chat-subtitle {{ color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .search {{ flex: 0 0 auto; padding: 10px 12px; border-bottom: 1px solid var(--border); }}
    .search input {{ width: 100%; border: 0; outline: 0; border-radius: 8px; background: var(--panel-2); color: var(--text); padding: 11px 14px; }}
    .conversations {{ flex: 1 1 auto; min-height: 0; overflow: auto; }}
    .conversation {{ width: 100%; border: 0; border-bottom: 1px solid var(--border); color: inherit; background: transparent; display: grid; grid-template-columns: 48px 1fr; gap: 12px; padding: 12px 14px; text-align: left; cursor: pointer; }}
    .conversation:hover, .conversation.active {{ background: var(--panel-2); }}
    .conversation h3 {{ margin: 0 0 3px; font-size: 15px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .conversation p {{ margin: 0; color: var(--muted); font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .chat {{ min-width: 0; min-height: 0; display: flex; flex-direction: column; background: #0b141a; overflow: hidden; }}
    .chat-title {{ min-width: 0; }}
    .chat-title h1 {{ margin: 0 0 2px; font-size: 16px; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .messages {{ flex: 1 1 auto; min-height: 0; overflow-y: auto; overflow-x: hidden; overscroll-behavior: contain; -webkit-overflow-scrolling: touch; padding: 22px clamp(12px, 3vw, 42px); background-image: radial-gradient(circle at 12px 12px, rgba(255,255,255,0.035) 1px, transparent 1px); background-size: 24px 24px; }}
    .message {{ max-width: min(1440px, 100%); margin: 0 0 14px; padding: 12px 14px; border-radius: 8px; box-shadow: 0 1px 0 var(--shadow); }}
    .incoming {{ background: var(--bubble-in); }}
    .outgoing {{ background: var(--bubble-out); margin-left: auto; }}
    .message h2 {{ margin: 0 0 8px; font-size: 17px; }}
    .message p {{ margin: 0 0 8px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(4, minmax(110px, 1fr)); gap: 10px; }}
    .metric {{ background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px; }}
    .metric b {{ display: block; font-size: 18px; }}
    .metric span {{ color: var(--muted-2); font-size: 12px; }}
    .table-wrap {{ width: 100%; overflow: auto; border: 1px solid var(--border); border-radius: 8px; background: rgba(0,0,0,0.14); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 1120px; }}
    .result-table {{ min-width: 1320px; table-layout: fixed; }}
    .result-table.compare {{ min-width: 1820px; }}
    .summary-table {{ min-width: 1180px; table-layout: fixed; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: top; text-align: left; }}
    th {{ color: var(--muted-2); background: rgba(0,0,0,0.18); font-size: 12px; font-weight: 650; position: sticky; top: 0; z-index: 1; }}
    tr:last-child td {{ border-bottom: 0; }}
    .idx {{ color: var(--muted); width: 56px; }}
    .category {{ font-weight: 650; min-width: 190px; }}
    .answer {{ white-space: pre-wrap; min-width: 300px; overflow-wrap: anywhere; }}
    .contract-col {{ width: 260px; overflow-wrap: anywhere; }}
    .gold-col, .prediction-col {{ width: 360px; }}
    .score-col, .status-col {{ width: 88px; }}
    .empty {{ color: var(--muted); font-style: italic; }}
    .pill {{ display: inline-flex; align-items: center; min-height: 22px; padding: 2px 8px; border-radius: 999px; background: rgba(255,255,255,0.08); color: var(--muted-2); font-size: 12px; white-space: nowrap; }}
    .pill.good {{ background: rgba(0,168,132,0.16); color: #90f0d8; }}
    .pill.warn {{ background: rgba(241,92,109,0.14); color: #ffb4bd; }}
    .score.good {{ color: #90f0d8; font-weight: 700; }}
    .score.bad {{ color: #ffb4bd; font-weight: 700; }}
    .delta.good {{ color: #90f0d8; font-weight: 700; }}
    .delta.bad {{ color: #ffb4bd; font-weight: 700; }}
    .delta.neutral {{ color: var(--muted-2); font-weight: 700; }}
    @media (max-width: 760px) {{
      body {{ overflow: auto; }}
      .global-header {{ align-items: stretch; flex-direction: column; height: auto; padding: 10px 12px; }}
      .brand {{ width: 100%; }}
      .tabs {{ width: 100%; display: grid; grid-template-columns: 1fr 1fr; }}
      .tab {{ text-align: center; }}
      .app {{ grid-template-columns: 1fr; height: calc(100vh - 103px); height: calc(100dvh - 103px); min-height: 0; }}
      .sidebar {{ max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--border); }}
      .chat {{ min-height: 0; }}
      .meta-grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <header class="global-header">
    <div class="brand"><div class="brand-mark">C</div><span>CUAD Review</span></div>
    <nav class="tabs" aria-label="Views">
      <a class="tab" href="explore.html">Explorer</a>
      <a class="tab active" href="{evaluation_href}" aria-current="page">Evaluation</a>
    </nav>
  </header>
  <div class="app">
    <aside class="sidebar" aria-label="Conversations">
      <div class="side-top">
        <div class="avatar">E</div>
        <div>
          <div class="side-title">CUAD Evaluation</div>
          <div class="side-subtitle">Summary and question results</div>
        </div>
      </div>
      <div class="search"><input id="search" type="search" placeholder="Search questions"></div>
      <div id="conversationList" class="conversations"></div>
    </aside>
    <main class="chat">
      <header class="chat-top">
        <div id="chatAvatar" class="avatar">S</div>
        <div class="chat-title">
          <h1 id="chatTitle"></h1>
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
      {{ id: 'summary', kind: 'summary', title: 'Evaluation summary', subtitle: `${{data.summary.total_examples}} examples · ${{data.summary.questions_per_contract}} questions`, avatar: 'S' }},
      ...data.questions.map(question => ({{
        id: `question-${{question.question_index}}`,
        kind: 'question',
        title: `${{question.question_index + 1}}. ${{question.category}}`,
        subtitle: `${{pct(question.mean_token_f1 / 100)}} mean F1 · ${{pct(question.correct_at_0_5 / 100)}} correct@0.5`,
        avatar: String(question.question_index + 1),
        question,
      }})),
    ];
    let activeId = 'summary';

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
      return `${{(Number(value || 0) * 100).toFixed(1)}}%`;
    }}

    function pctPoints(value) {{
      const number = Number(value || 0);
      const sign = number > 0 ? '+' : '';
      return `${{sign}}${{(number * 100).toFixed(1)}} pts`;
    }}

    function pctValue(value) {{
      return `${{Number(value || 0).toFixed(1)}}%`;
    }}

    function deltaClass(value) {{
      const number = Number(value || 0);
      if (number > 0.0001) return 'good';
      if (number < -0.0001) return 'bad';
      return 'neutral';
    }}

    function answerHtml(answers) {{
      if (!answers || answers.length === 0) return '<span class="empty">No golden answer</span>';
      return answers.map(esc).join('\\n\\n');
    }}

    function predictionHtml(answer, markedImpossible) {{
      const text = String(answer || '').trim();
      const body = text ? esc(text) : '<span class="empty">Empty prediction</span>';
      return `${{body}}${{markedImpossible ? '<br><span class="pill warn">Predicted no answer</span>' : ''}}`;
    }}

    function modelLabel(kind) {{
      const comparison = data.comparison || {{}};
      if (kind === 'baseline') return esc(comparison.baseline_model_id || 'Baseline');
      return esc(data.summary.model_id || 'Latest');
    }}

    function renderList() {{
      const needle = searchEl.value.trim().toLowerCase();
      listEl.innerHTML = conversations
        .filter(c => !needle || c.title.toLowerCase().includes(needle) || c.subtitle.toLowerCase().includes(needle))
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

    function renderSummary() {{
      const rows = data.per_category.map(row => `
        <tr>
          <td class="idx">${{row.question_index + 1}}</td>
          <td class="category">${{esc(row.category)}}<br><span class="pill">${{esc(row.answer_format || 'No answer format')}}</span></td>
          <td>${{esc(row.category_description || '')}}<br><span class="pill">Question ${{row.question_index + 1}}</span></td>
          <td><span class="score ${{row.mean_token_f1 >= 50 ? 'good' : 'bad'}}">${{pctValue(row.mean_token_f1)}}</span></td>
          <td>${{pctValue(row.correct_at_0_5)}}</td>
          <td>${{row.comparison && row.comparison.baseline_mean_token_f1 !== undefined ? pctValue(row.comparison.baseline_mean_token_f1) : '<span class="empty">n/a</span>'}}</td>
          <td>${{row.comparison && row.comparison.mean_token_f1_delta !== undefined ? `<span class="delta ${{deltaClass(row.comparison.mean_token_f1_delta / 100)}}">${{pctPoints(row.comparison.mean_token_f1_delta / 100)}}</span>` : '<span class="empty">n/a</span>'}}</td>
          <td>${{row.count}}</td>
        </tr>
      `).join('');
      const comparison = data.comparison;
      const comparisonHtml = comparison ? `
        <article class="message incoming">
          <h2>${{modelLabel('baseline')}} vs ${{modelLabel('latest')}}</h2>
          <div class="meta-grid">
            <div class="metric"><b>${{pctValue(comparison.baseline_mean_token_f1)}}</b><span>${{modelLabel('baseline')}} mean F1</span></div>
            <div class="metric"><b>${{pctValue(comparison.candidate_mean_token_f1)}}</b><span>${{modelLabel('latest')}} mean F1</span></div>
            <div class="metric"><b><span class="delta ${{deltaClass(comparison.mean_token_f1_delta / 100)}}">${{pctPoints(comparison.mean_token_f1_delta / 100)}}</span></b><span>Mean F1 change</span></div>
            <div class="metric"><b>${{comparison.matched_examples}}</b><span>Matched examples</span></div>
            <div class="metric"><b>${{pctValue(comparison.baseline_correct_at_0_5)}}</b><span>${{modelLabel('baseline')}} correct@0.5</span></div>
            <div class="metric"><b>${{pctValue(comparison.candidate_correct_at_0_5)}}</b><span>${{modelLabel('latest')}} correct@0.5</span></div>
            <div class="metric"><b><span class="delta ${{deltaClass(comparison.correct_at_0_5_delta / 100)}}">${{pctPoints(comparison.correct_at_0_5_delta / 100)}}</span></b><span>Correct@0.5 change</span></div>
            <div class="metric"><b>${{esc(comparison.baseline_results_path || '')}}</b><span>Baseline source</span></div>
          </div>
        </article>
      ` : '';
      messagesEl.innerHTML = `
        <article class="message incoming">
          <h2>Evaluation Run</h2>
          <div class="meta-grid">
            <div class="metric"><b>${{data.summary.total_examples}}</b><span>Examples</span></div>
            <div class="metric"><b>${{data.summary.sample_size}}</b><span>Contracts</span></div>
            <div class="metric"><b>${{data.summary.agent_count}}</b><span>Question agents</span></div>
            <div class="metric"><b>${{data.summary.dry_run ? 'Yes' : 'No'}}</b><span>Dry run</span></div>
          </div>
        </article>
        <article class="message outgoing"><p>Model ID: ${{esc(data.summary.model_id)}} · model: ${{esc(data.summary.model)}} · seed ${{esc(data.summary.seed)}} · threads ${{esc(data.summary.num_threads)}}</p></article>
        <article class="message incoming">
          <div class="meta-grid">
            <div class="metric"><b>${{data.summary.overlap_accuracy_mean_f1.toFixed(1)}}%</b><span>Mean token F1</span></div>
            <div class="metric"><b>${{data.summary.correct_at_0_5.toFixed(1)}}%</b><span>Correct at 0.5</span></div>
            <div class="metric"><b>${{data.summary.temperature}}</b><span>Temperature</span></div>
            <div class="metric"><b>${{data.summary.max_tokens}}</b><span>Max tokens</span></div>
          </div>
        </article>
        ${{comparisonHtml}}
        <article class="message incoming">
          <div class="table-wrap">
            <table class="summary-table">
              <thead><tr><th>#</th><th>Category</th><th>category_descriptions.csv context</th><th>${{modelLabel('latest')}} F1</th><th>${{modelLabel('latest')}} correct</th><th>${{modelLabel('baseline')}} F1</th><th>F1 change</th><th>Examples</th></tr></thead>
              <tbody>${{rows}}</tbody>
            </table>
          </div>
        </article>
      `;
    }}

    function renderQuestion(question) {{
      const hasComparison = question.results.some(row => row.comparison);
      const rows = question.results.map(row => {{
        if (!hasComparison) return `
          <tr>
            <td class="idx">${{row.document_row_id}}</td>
            <td>${{esc(row.title)}}<br><span class="pill">${{row.gold_marked_impossible ? 'Gold no answer' : 'Gold answer'}}</span></td>
            <td class="answer">${{answerHtml(row.gold_answers)}}</td>
            <td class="answer">${{predictionHtml(row.predicted_answer, row.predicted_marked_impossible)}}</td>
            <td><span class="score ${{row.correct_at_0_5 ? 'good' : 'bad'}}">${{pct(row.token_f1)}}</span></td>
            <td>${{row.correct_at_0_5 ? '<span class="pill good">Pass</span>' : '<span class="pill warn">Miss</span>'}}</td>
          </tr>
        `;
        const comparison = row.comparison || {{}};
        const baselineAnswer = comparison.baseline_predicted_answer !== undefined
          ? predictionHtml(comparison.baseline_predicted_answer, comparison.baseline_predicted_marked_impossible)
          : '<span class="empty">No baseline row</span>';
        const baselineF1 = comparison.baseline_token_f1 !== undefined
          ? `<span class="score ${{comparison.baseline_correct_at_0_5 ? 'good' : 'bad'}}">${{pct(comparison.baseline_token_f1)}}</span>`
          : '<span class="empty">n/a</span>';
        const delta = comparison.token_f1_delta !== undefined
          ? `<span class="delta ${{deltaClass(comparison.token_f1_delta)}}">${{pctPoints(comparison.token_f1_delta)}}</span>`
          : '<span class="empty">n/a</span>';
        return `
          <tr>
            <td class="idx">${{row.document_row_id}}</td>
            <td>${{esc(row.title)}}<br><span class="pill">${{row.gold_marked_impossible ? 'Gold no answer' : 'Gold answer'}}</span></td>
            <td class="answer">${{answerHtml(row.gold_answers)}}</td>
            <td class="answer">${{baselineAnswer}}</td>
            <td class="answer">${{predictionHtml(row.predicted_answer, row.predicted_marked_impossible)}}</td>
            <td>${{baselineF1}}</td>
            <td><span class="score ${{row.correct_at_0_5 ? 'good' : 'bad'}}">${{pct(row.token_f1)}}</span></td>
            <td>${{delta}}</td>
            <td>${{row.correct_at_0_5 ? '<span class="pill good">Pass</span>' : '<span class="pill warn">Miss</span>'}}</td>
          </tr>
        `;
      }}).join('');
      const comparisonMetrics = question.comparison && question.comparison.baseline_mean_token_f1 !== undefined ? `
        <article class="message incoming">
          <div class="meta-grid">
            <div class="metric"><b>${{pctValue(question.comparison.baseline_mean_token_f1)}}</b><span>${{modelLabel('baseline')}} mean F1</span></div>
            <div class="metric"><b>${{pctValue(question.comparison.candidate_mean_token_f1)}}</b><span>${{modelLabel('latest')}} mean F1</span></div>
            <div class="metric"><b><span class="delta ${{deltaClass(question.comparison.mean_token_f1_delta / 100)}}">${{pctPoints(question.comparison.mean_token_f1_delta / 100)}}</span></b><span>Mean F1 change</span></div>
            <div class="metric"><b>${{pctPoints(question.comparison.correct_at_0_5_delta / 100)}}</b><span>Correct@0.5 change</span></div>
          </div>
        </article>
      ` : '';
      const tableHead = hasComparison
        ? `<tr><th>Doc</th><th>Contract</th><th>Gold</th><th>${{modelLabel('baseline')}} answer</th><th>${{modelLabel('latest')}} answer</th><th>${{modelLabel('baseline')}} F1</th><th>${{modelLabel('latest')}} F1</th><th>F1 change</th><th>Status</th></tr>`
        : '<tr><th>Doc</th><th>Contract</th><th>Gold</th><th>Prediction</th><th>F1</th><th>Status</th></tr>';
      const tableCols = hasComparison
        ? `
                <col style="width: 64px">
                <col class="contract-col">
                <col class="gold-col">
                <col class="prediction-col">
                <col class="prediction-col">
                <col class="score-col">
                <col class="score-col">
                <col class="score-col">
                <col class="status-col">
        `
        : `
                <col style="width: 64px">
                <col class="contract-col">
                <col class="gold-col">
                <col class="prediction-col">
                <col class="score-col">
                <col class="status-col">
        `;
      messagesEl.innerHTML = `
        <article class="message outgoing"><p>${{esc(question.question)}}</p></article>
        <article class="message incoming">
          <h2>${{esc(question.category)}}</h2>
          <p>${{esc(question.category_description)}}</p>
          <span class="pill">${{esc(question.answer_format || 'No answer format')}}</span>
        </article>
        <article class="message incoming">
          <div class="meta-grid">
            <div class="metric"><b>${{question.mean_token_f1.toFixed(1)}}%</b><span>Mean token F1</span></div>
            <div class="metric"><b>${{question.correct_at_0_5.toFixed(1)}}%</b><span>Correct at 0.5</span></div>
            <div class="metric"><b>${{question.count}}</b><span>Contracts</span></div>
            <div class="metric"><b>${{question.question_index + 1}}</b><span>Question number</span></div>
          </div>
        </article>
        ${{comparisonMetrics}}
        <article class="message incoming">
          <div class="table-wrap">
            <table class="result-table ${{hasComparison ? 'compare' : ''}}">
              <colgroup>
                ${{tableCols}}
              </colgroup>
              <thead>${{tableHead}}</thead>
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
      if (active.kind === 'summary') renderSummary();
      else renderQuestion(active.question);
      renderList();
      messagesEl.scrollTop = 0;
    }}

    searchEl.addEventListener('input', renderList);
    render();
  </script>
</body>
</html>
"""


def write_evaluation_html(
    results: pd.DataFrame,
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    page_data = build_evaluation_page_data(results, summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_evaluation_html(page_data), encoding="utf-8")
