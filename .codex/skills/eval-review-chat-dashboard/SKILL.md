---
name: eval-review-chat-dashboard
description: Build static review dashboards for model or agent evaluations using a WhatsApp-like conversation UI. Use when a user wants to inspect evaluation results, golden answers, predictions, rubrics, document-level conversations, question-level review, prompt-review output, or baseline-vs-candidate comparisons in a browser. The skill covers data shaping, summary views, document/conversation sidebars, question panes, golden answer review, metrics, failure status, prompt diffs, and portable HTML/CSS/JS layout patterns.
metadata:
  short-description: WhatsApp-style eval review dashboards
---

# Eval Review Chat Dashboard

Create a static HTML dashboard for reviewing model/agent evaluations as conversations. The core UX is:

- Left sidebar = conversations, documents, questions, categories, or review units.
- Right pane = a chat-like thread containing summary messages, user questions, model predictions, golden answers, metrics, reviewer comments, and tables.
- A top summary view gives run-level metrics.
- Detail views let a reviewer compare input, golden answer, prediction, status, and evidence without opening spreadsheets.

This skill is based on the reusable layout patterns from `cuad_dspy_eval.html` and `prompt_review_dashboard.html`, generalized for any evaluation project.

## When To Use

Use for:

- Evaluation dashboards for LLM extraction, QA, classification, routing, tool-use, grading, or agent workflows.
- Reviewing model predictions against golden answers.
- Comparing baseline vs candidate runs.
- Reviewing prompt-improvement outputs, prompt diffs, evaluator feedback, and accepted/rejected patches.
- Turning CSV/JSONL eval logs into a static artifact that can be opened locally in a browser.

Do not build a marketing page. The first screen should be the review tool.

## Load More Detail

For full implementation guidance, read [references/dashboard_pattern.md](references/dashboard_pattern.md). It includes:

- Recommended page-data schemas.
- Renderer skeletons.
- UX rules for summary, conversation, document, and question views.
- CSS tokens and responsive layout rules.
- Review affordances for golden answers, predictions, diffs, rubrics, and failure modes.

## Core Workflow

1. Normalize eval data into a browser-friendly JSON object.
2. Embed the JSON in the HTML inside `<script type="application/json">`.
3. Render a two-pane chat UI: searchable sidebar plus detail pane.
4. Include a run summary conversation.
5. Include one conversation per document/question/review unit.
6. Show golden answer and prediction side-by-side.
7. Surface pass/fail, score, failure mode, and reviewer notes as pills/metrics.
8. Keep it fully static: no server and no external assets required.

## Minimal Data Contract

At minimum, construct:

```python
page_data = {
    "summary": {
        "model_id": "...",
        "total_examples": 100,
        "mean_score": 72.3,
        "accuracy": 68.0,
    },
    "conversations": [
        {
            "id": "doc-123",
            "title": "Document 123",
            "subtitle": "8 questions · 5 pass · 3 fail",
            "avatar": "D",
            "kind": "document",
            "messages": [
                {"role": "user", "title": "Question", "body": "..."},
                {"role": "assistant", "title": "Prediction", "body": "...", "score": 0.71},
                {"role": "gold", "title": "Golden answer", "body": "..."},
            ],
            "rows": [],
        }
    ],
}
```

Adapt names to the project, but keep `summary` and `conversations` as stable top-level concepts.

## UX Requirements

- Use dark WhatsApp-inspired colors: app background, panel, chat bubble in/out, muted text, green accent, danger/warn accent.
- Sidebar search should filter conversation titles and subtitles.
- Header should show the current conversation title and subtitle.
- Summary should include metric cards and an overview table.
- Detail views should show the prompt/question as an outgoing message and the review/evaluation result as incoming messages.
- Tables should be horizontally scrollable and use sticky headers.
- Long gold/predicted text must preserve whitespace and wrap safely.
- Mobile view should stack sidebar above chat.

## Artifact Pattern

The dashboard should be written next to evaluation outputs:

```text
outputs/<run_id>/
  eval_results.csv
  eval_summary.json
  eval_review.html
```

For prompt review:

```text
outputs/<run_id>/prompt_harness/
  prompt_review_dashboard.html
```

## Practical Defaults

- One static `eval_review.html` file.
- Inline CSS and JS.
- Escape all user/model/gold text before inserting into HTML.
- Replace `</` in embedded JSON as `<\/` to avoid closing script tags.
- Use compact cards, not decorative nested cards.
- Keep repeated row data in tables; use chat bubbles for narrative context and metrics.

## Adaptation Checklist

- Identify review unit: document, conversation, question, category, prompt key, or eval example.
- Decide sidebar grouping: one conversation per document or one per question/category.
- Build a summary conversation.
- Define row schema for gold vs prediction comparison.
- Add failure labels and score thresholds.
- Add baseline deltas if available.
- Add prompt diffs and reviewer comments if this is prompt optimization.
- Verify the output by opening the HTML locally and checking a small/mobile viewport.

