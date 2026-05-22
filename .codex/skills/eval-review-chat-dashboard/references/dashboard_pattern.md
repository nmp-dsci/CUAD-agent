# Dashboard Pattern

Use this reference when building a WhatsApp-style evaluation review dashboard in another project.

## Purpose

The dashboard is a static browser artifact for reviewing agent output. It should make it easy to answer:

- What did this run evaluate?
- Which documents/conversations/questions failed?
- What was the user/input/question?
- What did the model predict?
- What was the golden answer?
- How was it scored?
- What changed between baseline and candidate?
- What prompt/reviewer insight explains the result?

The dashboard should feel like a review inbox: pick a conversation on the left, inspect the evidence and result on the right.

## Information Architecture

### Recommended Top-Level Views

Always include:

- `summary`: run-level metrics, counts, model/config metadata, and per-unit overview table.
- `conversation` detail pages: one per review unit.

Optional:

- `failures`: filtered view of failed examples.
- `baseline`: baseline-vs-candidate comparison.
- `prompts`: prompt diff and prompt review output.
- `rubric`: scoring rubric and grader notes.

### Choosing the Conversation Unit

Use one of these:

- **Document-centric**: one sidebar item per document/conversation. Inside it, show all questions, gold answers, predictions, and scores for that document.
- **Question-centric**: one sidebar item per question/category. Inside it, show all documents/examples for that question. This is good for benchmark suites such as CUAD.
- **Prompt-centric**: one sidebar item per prompt key/category. Inside it, show v1 vs v2 prompt, diff, reviewer feedback, failures, and examples.
- **Example-centric**: one sidebar item per failed example. Good for small, high-touch error analysis.

Pick the unit that best matches the user's review workflow. If they ask to "review evaluations and golden answers by document", prefer document-centric.

## Data Shape

### Generic Page Data

```python
page_data = {
    "summary": {
        "run_id": "candidate-v2",
        "model": "deepseek/deepseek-v4-flash",
        "total_examples": 2050,
        "document_count": 50,
        "question_count": 41,
        "mean_score": 40.3,
        "accuracy": 39.9,
        "baseline_mean_score": 40.0,
        "score_delta": 0.3,
    },
    "overview_rows": [
        {
            "id": "question-18",
            "title": "Anti-Assignment",
            "subtitle": "50 examples",
            "score": 61.0,
            "accuracy": 58.0,
            "count": 50,
            "status": "mixed",
        }
    ],
    "conversations": [
        {
            "id": "question-18",
            "kind": "question",
            "title": "19. Anti-Assignment",
            "subtitle": "61.0% F1 · 58.0% correct",
            "avatar": "19",
            "metadata": {},
            "messages": [],
            "rows": [],
        }
    ],
}
```

### Evaluation Row Shape

```python
row = {
    "row_id": "327:18",
    "document_id": "327",
    "document_title": "Contract title",
    "input": "Question or user message",
    "gold_answers": ["exact gold span", "another gold span"],
    "prediction": "model output",
    "score": 0.72,
    "passed": True,
    "failure_mode": "partial_answer",
    "gold_no_answer": False,
    "predicted_no_answer": False,
    "baseline_prediction": "...",
    "baseline_score": 0.41,
    "score_delta": 0.31,
    "notes": [],
}
```

### Prompt Review Conversation Shape

```python
conversation = {
    "id": "prompt-anti-assignment",
    "kind": "prompt_review",
    "title": "Anti-Assignment",
    "subtitle": "decision revise · score 0.62",
    "avatar": "P",
    "metadata": {
        "decision": "revise",
        "generalization_score": 0.62,
        "main_failure_mode": "false_no_answer",
    },
    "messages": [
        {"role": "incoming", "title": "Change insight", "body": "..."},
        {"role": "incoming", "title": "Evaluator feedback", "items": ["..."]},
    ],
    "prompt": {
        "v1": "...",
        "candidate": "...",
        "diff": ["--- v1", "+++ candidate", "@@ ..."],
    },
    "rows": [],
}
```

## Rendering Rules

### Static HTML

Build a single file:

```python
def render_review_html(page_data: dict[str, Any]) -> str:
    data_json = json.dumps(page_data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
    ...
    <script id="app-data" type="application/json">{data_json}</script>
    ...
    """
```

Never interpolate raw model/gold/user text directly into HTML. Put it in JSON, then escape in JS before inserting.

### Escape Function

Use this in the client:

```javascript
function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[ch]));
}
```

### Layout

Use:

```css
.app {
  display: grid;
  grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
  height: calc(100vh - 56px);
  overflow: hidden;
}
.sidebar {
  background: var(--panel);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.chat {
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 22px clamp(12px, 3vw, 42px);
}
```

Mobile:

```css
@media (max-width: 760px) {
  .app { grid-template-columns: 1fr; }
  .sidebar { max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--border); }
  .meta-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
}
```

### Theme Tokens

Use these as stable defaults:

```css
:root {
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
}
```

## Sidebar Pattern

Render conversations from `data.conversations` plus synthetic summary:

```javascript
const conversations = [
  {
    id: 'summary',
    kind: 'summary',
    title: 'Evaluation summary',
    subtitle: `${data.summary.total_examples} examples`,
    avatar: 'S'
  },
  ...data.conversations
];
```

Search should match title and subtitle:

```javascript
function renderList() {
  const needle = searchEl.value.trim().toLowerCase();
  listEl.innerHTML = conversations
    .filter(c => !needle || c.title.toLowerCase().includes(needle) || c.subtitle.toLowerCase().includes(needle))
    .map(c => `<button class="conversation ${c.id === activeId ? 'active' : ''}" data-id="${esc(c.id)}">
      <div class="avatar">${esc(c.avatar)}</div>
      <div><h3>${esc(c.title)}</h3><p>${esc(c.subtitle)}</p></div>
    </button>`)
    .join('');
}
```

## Summary View

Summary should include:

- Metric cards: examples, documents, prompts/questions, pass rate, mean score.
- Run metadata: model, run id, seed, temperature, date, source files.
- Overview table: one row per conversation unit with score/status/count.
- Optional baseline comparison: baseline score, candidate score, delta.

Use chat bubbles:

- Incoming bubble: "Evaluation Run" and metric cards.
- Outgoing bubble: run configuration.
- Incoming bubble: overview table.

## Detail View

### Evaluation Detail

Recommended order:

1. Outgoing message: input/question/user task.
2. Incoming message: metadata and rubric/category/context.
3. Incoming message: metric cards for this unit.
4. Incoming table: rows with document, gold, prediction, score, status.

Table columns:

- Document/conversation id.
- Input or contract title.
- Golden answer.
- Prediction.
- Score.
- Status/failure mode.
- Optional baseline and delta.

### Document Conversation Detail

If grouping by document, show:

- Document title and metadata.
- Each question as an outgoing message.
- Under each question, incoming gold answer and prediction bubbles.
- A compact table at the bottom for all question scores.

### Prompt Review Detail

Recommended order:

1. Heading: prompt key/category and question/task.
2. Metrics: evaluator decision, generalization score, loops, failure mode.
3. Change insight bubble.
4. Output profile/rubric bubble.
5. Two columns: v1 prompt and candidate prompt.
6. Diff block.
7. Evaluator feedback and regression risks.
8. Example table with split, row id, gold, prediction, failure mode, score.

## Components

### Message Bubbles

```css
.message {
  max-width: min(1440px, 100%);
  margin: 0 0 14px;
  padding: 12px 14px;
  border-radius: 8px;
  box-shadow: 0 1px 0 var(--shadow);
}
.incoming { background: var(--bubble-in); }
.outgoing { background: var(--bubble-out); margin-left: auto; }
```

### Metrics

```css
.meta-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(110px, 1fr));
  gap: 10px;
}
.metric {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  padding: 10px;
}
.metric b { display: block; font-size: 18px; }
.metric span { color: var(--muted-2); font-size: 12px; }
```

### Tables

```css
.table-wrap {
  width: 100%;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: rgba(0,0,0,0.14);
}
table {
  width: 100%;
  border-collapse: collapse;
  min-width: 1120px;
}
th, td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
  text-align: left;
}
th {
  color: var(--muted-2);
  background: rgba(0,0,0,0.18);
  font-size: 12px;
  font-weight: 650;
  position: sticky;
  top: 0;
  z-index: 1;
}
.answer {
  white-space: pre-wrap;
  min-width: 300px;
  overflow-wrap: anywhere;
}
```

### Status Pills

```css
.pill {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  color: var(--muted-2);
  font-size: 12px;
  white-space: nowrap;
}
.pill.good { background: rgba(0,168,132,0.16); color: #90f0d8; }
.pill.warn { background: rgba(241,92,109,0.14); color: #ffb4bd; }
.score.good { color: #90f0d8; font-weight: 700; }
.score.bad { color: #ffb4bd; font-weight: 700; }
```

## HTML Skeleton

```html
<header class="global-header">
  <div class="brand"><div class="brand-mark">R</div><span>Eval Review</span></div>
  <nav class="tabs" aria-label="Views">
    <a class="tab active" href="eval_review.html">Evaluation</a>
  </nav>
</header>
<div class="app">
  <aside class="sidebar" aria-label="Conversations">
    <div class="side-top">
      <div class="avatar">E</div>
      <div>
        <div class="side-title">Evaluation</div>
        <div class="side-subtitle">Summary and results</div>
      </div>
    </div>
    <div class="search"><input id="search" type="search" placeholder="Search"></div>
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
<script id="app-data" type="application/json">...</script>
```

## Python Build Helpers

### JSON Embedding

```python
def safe_json_for_html(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
```

### Gold Answer Parser

If gold answers arrive as JSON strings:

```python
def parse_gold_answers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return [str(parsed)]
```

### Conversation Builder

```python
def build_conversations(results: pd.DataFrame) -> list[dict[str, Any]]:
    conversations = []
    for document_id, rows in results.groupby("document_id", sort=True):
        first = rows.iloc[0]
        passed = int(rows["passed"].sum())
        conversations.append({
            "id": f"doc-{document_id}",
            "kind": "document",
            "title": str(first["document_title"]),
            "subtitle": f"{passed}/{len(rows)} pass · mean {rows['score'].mean():.1%}",
            "avatar": str(document_id)[:2],
            "rows": [
                {
                    "row_id": str(row.row_id),
                    "input": str(row.input),
                    "gold_answers": parse_gold_answers(row.gold_answers),
                    "prediction": str(row.prediction),
                    "score": float(row.score),
                    "passed": bool(row.passed),
                    "failure_mode": str(getattr(row, "failure_mode", "")),
                }
                for row in rows.itertuples(index=False)
            ],
        })
    return conversations
```

## Review Quality Checklist

Before delivering:

- Open the HTML and verify the summary renders.
- Verify at least one pass and one fail row render correctly.
- Verify long gold and prediction text wraps without breaking layout.
- Verify search works.
- Verify mobile width stacks sidebar over chat.
- Verify JSON escaping with strings containing `<`, `>`, `&`, quotes, and `</script>`.
- Verify no external network dependencies are needed.
- Verify source artifacts next to the HTML can reproduce it.

## Anti-Patterns

- Showing only aggregate score with no gold/prediction inspection.
- Making the first screen a landing page.
- Hiding failures behind collapsed panels by default.
- Loading external JS/CSS libraries for a simple static artifact.
- Inserting raw model output into `innerHTML` without escaping.
- Letting long contract/model text expand the page horizontally without a scroll container.
- Making one giant table with no sidebar navigation.
- Omitting run configuration and source file paths.

