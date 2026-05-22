# PRP: DSPy CUAD 41-Agent Evaluation

## Goal

Build a Python DSPy evaluation runner that answers the same 41 CUAD contract-review questions for a deterministic sample of 10 contracts. Each question/category pair must be answered by its own DSPy agent, so the runner instantiates exactly 41 total question-specialized agents and reuses those same 41 agents across each sampled contract. For example, there must be a `Document Name` agent, a `Parties` agent, an `Agreement Date` agent, and so on through all 41 CUAD categories.

The final output must report evaluation accuracy across 10 contracts x 41 questions = 410 evaluated examples, plus per-question/category accuracy. The primary metric is normalized token-overlap F1 between the model answer text and the CUAD golden answer text.

## User Requirements

- Use Python and DSPy.
- Use the local CUAD data already explored in `explore.py` and visualized in `frontend/explore.html`.
- There are 41 questions per contract; the questions are the same for every document.
- Each of the 41 question/category pairs must be answered by a separate agent keyed by `question_index` and category name.
- There must be exactly 41 total agents for the evaluation run, not 410 agents. Those 41 agents are reused across the 10 selected contracts.
- Evaluate against a sample of 10 contracts.
- The 10 contracts must be selected after setting the seed, and the selected `document_row_id` values must be identical between runs with the same seed.
- Compare predicted answer text to golden answer text using an overlap metric.
- Output evaluation accuracy across the 10 contracts for all 41 questions.

## Codebase Context

Important existing files:

- `explore.py`
  - Loads `data/CUADv1.json`.
  - Loads and normalizes `data/category_descriptions.csv`.
  - Validates every document has 41 questions and that question order is identical across all documents.
  - Joins question-level rows to category metadata by `question_index`.
  - `load_datasets()` returns:
    - `contracts`: 510 rows, columns include `document_row_id`, `document_id`, `title`, `context`.
    - `questions`: 20,910 rows, 41 per contract, columns include `document_row_id`, `question_index`, `question`, `answers`, `answers_len`, `is_impossible`, `category`, `category_description`, `answer_format`, `category_group`.
- `frontend/explore.html`
  - Static UI generated from the same data.
  - Confirms the intended UX/data concept: summary conversation plus 10 sampled contracts, each with 41 question/golden-answer rows and expandable contract text.
- `planning/egs/dspy_agent.py`
  - Local DSPy example with:
    - `os.environ.setdefault("DSPY_CACHEDIR", ...)` before importing DSPy.
    - `dspy.LM(...)` plus `dspy.configure(...)`.
    - `dspy.Signature` classes with `InputField` and `OutputField`.
    - `dspy.Module` agent class with `forward(...)`.
    - Evaluation logic over deterministic sampled records.
- `pyproject.toml`
  - Currently lacks `dspy` and `python-dotenv`. Add them.

Data facts verified locally:

- `contracts.shape == (510, 9)`
- `questions.shape == (20910, 15)`
- `questions.groupby("document_row_id").size().value_counts().to_dict() == {41: 510}`
- Unique question order across documents is `1`.
- `category_descriptions.csv` has 41 rows.

## External Research

DSPy docs:

- https://dspy.ai/
  - DSPy is a declarative framework for modular AI software. The docs show configuring an LM via `dspy.LM(...)` and `dspy.configure(lm=lm)`.
- https://dspy.ai/learn/programming/signatures/
  - DSPy Signatures declare the input/output behavior of modules. Field names matter semantically.
- https://dspy.ai/learn/evaluation/metrics/
  - A DSPy metric is a Python function taking `example`, `pred`, and optional `trace`, returning a score. Docs explicitly cite exact match and F1 as simple task metrics.
  - Evaluation can be a plain Python loop or `dspy.evaluate.Evaluate`.
- https://dspy.ai/api/evaluation/Evaluate/
  - `Evaluate` accepts `devset`, `metric`, `num_threads`, `display_progress`, `display_table`, `save_as_csv`, and `save_as_json`.
  - Result includes `score` and per-example `(example, prediction, score)` results.
- https://dspy.ai/api/modules/Parallel/
  - `dspy.Parallel` executes `(module, example)` pairs with `num_threads`; useful if concurrently dispatching contract-question jobs to the 41 fixed category agents.

Metric research:

- https://docs.allennlp.org/models/main/models/rc/tools/squad/
  - AllenNLP documents functions taken from the official SQuAD 2.0 evaluation script, including `normalize_answer`, `compute_exact`, `compute_f1`, and max-over-ground-truth helpers.
  - SQuAD normalization lowers text, removes punctuation/articles, and fixes whitespace.
- https://docs.allennlp.org/models/main/models/rc/metrics/squad_em_and_f1/
  - AllenNLP’s `SquadEmAndF1` computes exact match and F1 using official SQuAD scripts and returns average exact match and F1 over all inputs.

Recommended metric for this project:

- Use a CUAD-adapted SQuAD-style normalized token F1.
- Rationale: CUAD golden answers are extractive text spans. Exact match is too brittle because models may include surrounding words or formatting. Token F1 gives partial credit based on overlap and penalizes both over-long predictions and incomplete predictions.
- CUAD adaptation: a single question may have multiple required golden spans, not merely multiple alternative answers. Therefore concatenate all golden answer texts for the row into one gold token multiset before computing precision/recall/F1 against the predicted answer. This rewards returning all relevant spans and penalizes missing any span.
- No-answer rows: if the gold answer list is empty, score `1.0` only when the prediction is empty or a normalized no-answer marker such as `no answer`, `none`, `not found`, `n/a`, or `not applicable`; otherwise score `0.0`.

Metric formula:

```python
precision = overlap_token_count / predicted_token_count
recall = overlap_token_count / gold_token_count
f1 = 2 * precision * recall / (precision + recall)
```

Primary reported `overlap_accuracy_mean_f1` should be `mean(token_f1) * 100` over 410 examples. Also report `correct_at_0_5 = mean(token_f1 >= 0.5) * 100` as a secondary thresholded accuracy for quick inspection.

## Implementation Blueprint

Create a new script, suggested path: `cuad_dspy_eval.py`.

### Dependency Changes

Add dependencies:

```bash
uv add dspy python-dotenv
```

If using OpenAI through DSPy, require `OPENAI_API_KEY`. The script should also support any LiteLLM/DSPy-compatible model string via CLI, defaulting to a small/cheap model.

### Data Preparation

Reuse `explore.load_datasets()` rather than reparsing CUAD manually.

Sampling must be deterministic. Do not rely on implicit global RNG state or dataframe row order that may change across pandas versions. Sort the candidate IDs, create an explicit seeded RNG, sample from that sorted list, and persist the selected IDs in the summary JSON so repeated runs can prove they evaluated the same 10 contracts.

Pseudocode:

```python
import random

from explore import load_datasets

def build_eval_sample(sample_size: int = 10, seed: int = 42):
    datasets = load_datasets()
    contracts = datasets["contracts"]
    questions = datasets["questions"]

    candidate_ids = sorted(contracts["document_row_id"].astype(int).tolist())
    rng = random.Random(seed)
    selected_ids = rng.sample(candidate_ids, k=sample_size)

    contract_lookup = contracts.set_index("document_row_id").to_dict("index")
    eval_rows = questions[questions["document_row_id"].isin(selected_ids)]
    eval_rows = eval_rows.sort_values(["document_row_id", "question_index"])

    assert eval_rows.shape[0] == sample_size * 41
    return selected_ids, contract_lookup, eval_rows
```

The implementation must include a deterministic sampling test or assertion equivalent to:

```python
ids_a, _, _ = build_eval_sample(sample_size=10, seed=42)
ids_b, _, _ = build_eval_sample(sample_size=10, seed=42)
assert ids_a == ids_b
assert len(ids_a) == 10
assert len(set(ids_a)) == 10
```

### DSPy Signature

Use one shared signature class but instantiate 41 separate agents with different question/category metadata. The agent identity is the CUAD `question_index` plus `category`, e.g. `(0, "Document Name")`, `(1, "Parties")`, `(2, "Agreement Date")`.

```python
class ContractQuestionSignature(dspy.Signature):
    """Answer one contract-review question using only the supplied contract text.

    Return exact text spans from the contract when the answer exists. If no answer
    exists, return an empty string or `NO_ANSWER`. Do not explain.
    """

    contract_title: str = dspy.InputField()
    contract_text: str = dspy.InputField()
    question: str = dspy.InputField()
    category: str = dspy.InputField()
    category_description: str = dspy.InputField()
    answer_format: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="Exact answer text span(s) from the contract, separated by newlines if multiple; or NO_ANSWER."
    )
```

### 41 Separate Agents

Implement each question agent as a DSPy module. The key requirement is that the runner creates 41 separate instances total, one per question/category, before evaluation starts. These 41 instances are then reused for all 10 contracts.

```python
class ContractQuestionAgent(dspy.Module):
    def __init__(self, question_index: int, category: str, category_description: str, answer_format: str):
        super().__init__()
        self.question_index = question_index
        self.category = category
        self.category_description = category_description
        self.answer_format = answer_format
        self.predict = dspy.ChainOfThought(ContractQuestionSignature)

    def forward(self, contract_title: str, contract_text: str, question: str):
        pred = self.predict(
            contract_title=contract_title,
            contract_text=contract_text,
            question=question,
            category=self.category,
            category_description=self.category_description,
            answer_format=self.answer_format,
        )
        return dspy.Prediction(answer=str(pred.answer), reasoning=getattr(pred, "reasoning", ""))
```

Build agents:

```python
def build_agents(questions_df):
    categories = (
        questions_df[["question_index", "category", "category_description", "answer_format"]]
        .drop_duplicates("question_index")
        .sort_values("question_index")
    )
    assert len(categories) == 41
    return {
        int(row.question_index): ContractQuestionAgent(
            question_index=int(row.question_index),
            category=row.category,
            category_description=row.category_description,
            answer_format=row.answer_format,
        )
        for row in categories.itertuples(index=False)
    }
```

The implementation must preserve the question/category capture on every agent. Add an assertion or diagnostic equivalent to:

```python
agents = build_agents(questions)
assert len(agents) == 41
assert agents[0].category == "Document Name"
assert agents[1].category == "Parties"
assert agents[2].category == "Agreement Date"
```

### Evaluation Loop

Do not make a single agent answer all 41 questions, and do not create new agents per contract. Build the 41 category agents once, then dispatch every evaluation row to `agents[question_index]`.

```python
def run_evaluation(agents, contract_lookup, eval_rows):
    records = []
    for row in eval_rows.itertuples(index=False):
        contract = contract_lookup[int(row.document_row_id)]
        agent = agents[int(row.question_index)]
        pred = agent(
            contract_title=str(contract["title"]),
            contract_text=str(contract["context"]),
            question=str(row.question),
        )
        gold_texts = [a["text"] for a in row.answers] if isinstance(row.answers, list) else []
        score = token_overlap_f1(pred.answer, gold_texts)
        records.append({
            "document_row_id": int(row.document_row_id),
            "title": contract["title"],
            "question_index": int(row.question_index),
            "category": row.category,
            "question": row.question,
            "gold_answers": gold_texts,
            "predicted_answer": pred.answer,
            "token_f1": score,
            "correct_at_0_5": score >= 0.5,
        })
    return pd.DataFrame(records)
```

### Parallelization Strategy

The evaluation can be parallelized across either axis of the 10 x 41 grid:

- Contract axis: split the 10 selected contracts across workers. Each worker evaluates all 41 prebuilt category agents for one or more contracts.
- Question axis: split the 41 question/category agents across workers. Each worker evaluates one or more fixed category agents across all 10 contracts.
- Flattened contract-question axis: create 410 jobs shaped as `(document_row_id, question_index)` and dispatch each job to `agents[question_index]`.

The preferred implementation is the flattened job list because it naturally supports both contract and question parallelism and keeps the mapping simple:

```python
jobs = [
    (int(row.document_row_id), int(row.question_index), row)
    for row in eval_rows.sort_values(["document_row_id", "question_index"]).itertuples(index=False)
]

def run_job(job):
    document_row_id, question_index, row = job
    contract = contract_lookup[document_row_id]
    agent = agents[question_index]
    return evaluate_one_row(agent, contract, row)
```

Run serially when `--num-threads 1`. When `--num-threads > 1`, use `dspy.Parallel` or `concurrent.futures.ThreadPoolExecutor` to run the same jobs concurrently. Regardless of execution order, sort the final result dataframe by `document_row_id, question_index` before writing CSV/JSON so repeated runs remain comparable.

Do not create one agent per job. The parallel jobs must reuse the same 41 prebuilt agents keyed by `question_index`.

Add CLI support for a split hint:

```text
--parallel-axis flat
```

Allowed values should be `flat`, `contract`, and `question`. `flat` is the default and may be the only fully implemented path initially; if `contract` or `question` are accepted, they must still produce the same output schema and final sorted order.

### Metric Implementation

Implement this locally in the script and unit test it. Avoid importing heavyweight QA libraries just for the metric.

```python
import re
import string
from collections import Counter

NO_ANSWER_MARKERS = {"", "no answer", "none", "not found", "n/a", "na", "not applicable", "no_answer"}

def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def tokens(text: str) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []

def token_overlap_f1(prediction: str, gold_answers: list[str]) -> float:
    pred_norm = normalize_answer(prediction or "")
    gold_text = " ".join(gold_answers or [])
    gold_toks = tokens(gold_text)
    pred_toks = pred_norm.split() if pred_norm else []

    if not gold_toks:
        return 1.0 if pred_norm in NO_ANSWER_MARKERS else 0.0
    if not pred_toks:
        return 0.0

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)
```

### Output Requirements

Create an `outputs/` directory if needed.

The script should write:

- `outputs/cuad_dspy_eval_results.csv`
  - One row per document-question pair, 410 rows for default run.
- `outputs/cuad_dspy_eval_summary.json`
  - Overall:
    - sample size
    - seed
    - selected contract IDs as `selected_document_row_ids`
    - total examples
    - mean token F1 as `overlap_accuracy_mean_f1`
    - thresholded `correct_at_0_5`
    - model name
  - Per-category:
    - `question_index`
    - `category`
    - `mean_token_f1`
    - `correct_at_0_5`
    - count
- Also print a concise terminal summary:

```text
CUAD DSPy evaluation
model: openai/gpt-4o-mini
contracts: 10
questions per contract: 41
examples: 410
overlap_accuracy_mean_f1: 63.42%
correct_at_0_5: 58.29%

Worst categories:
...
Best categories:
...
```

### CLI

Use `argparse`.

Required/default options:

```text
--sample-size 10
--seed 42
--model openai/gpt-4o-mini
--temperature 0
--max-tokens 1200
--num-threads 1
--parallel-axis flat
--output-dir outputs
--dry-run
```

`--dry-run` should build data, instantiate agents, and write a small fake prediction result without calling the LM. This makes validation possible without API keys.

### LM Setup

Follow `planning/egs/dspy_agent.py` pattern:

```python
os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parent / ".dspy_cache"))
from dotenv import load_dotenv
load_dotenv(Path.home() / ".env")
import dspy

lm = dspy.LM(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
```

If not `--dry-run`, fail early with a clear message when required credentials are missing for the selected provider. For OpenAI model strings, check `OPENAI_API_KEY`.

## Error Handling and Gotchas

- Do not infer category descriptions by string matching at evaluation time; `explore.py` already validates and joins category metadata.
- Some CUAD rows have multiple gold answer spans. Treat them as multiple required spans by concatenating them for the metric.
- Many rows are impossible/no-answer (`answers_len == 0`). The agent should return `NO_ANSWER` or empty answer for these.
- Long contract contexts can be large. Start with direct full-context prompting because the user requested answering using a contract. If model context limits fail, add a later retrieval/chunking PRP; do not silently truncate without recording it.
- Set `temperature=0` by default for evaluation repeatability.
- Keep the 10-contract sample deterministic by sorting candidate `document_row_id` values, creating `random.Random(args.seed)`, and sampling only after that seed is set.
- Persist `selected_document_row_ids` in the summary JSON. Two runs with the same seed must produce the exact same ordered list of 10 IDs.
- Ensure there are exactly 41 agent instances. Add an assertion.
- Ensure each agent stores its `question_index`, `category`, `category_description`, and `answer_format` so outputs can be grouped by category, e.g. `Document Name`.
- Parallelization may split by contract, by question/category, or by flattened `(document_row_id, question_index)` jobs, but it must always reuse the same 41 agents and must sort final outputs deterministically.
- Save raw predictions so failures can be inspected.

## Implementation Tasks

1. Add dependencies with `uv add dspy python-dotenv`.
2. Create `cuad_dspy_eval.py`.
3. Import and reuse `load_datasets()` from `explore.py`.
4. Implement seeded deterministic sample selection, persist `selected_document_row_ids`, and assert default evaluation size is 410.
5. Implement `normalize_answer`, `tokens`, and `token_overlap_f1`.
6. Implement `ContractQuestionSignature`.
7. Implement `ContractQuestionAgent`.
8. Implement `build_agents()` and assert exactly 41 total agents, including category checks such as `agents[0].category == "Document Name"`.
9. Implement evaluation dispatch over prebuilt `agents[row.question_index]`; do not instantiate agents inside the contract loop or per job.
10. Implement `--num-threads` and `--parallel-axis` so evaluation can be split by contract, by question/category, or by flattened contract-question jobs. Default to serial behavior when `--num-threads 1`.
11. Implement `--dry-run` mode for API-free validation.
12. Implement CSV and JSON output files.
13. Implement terminal summary with overall and per-category scores.
14. Add focused tests for the metric and data prep. Suggested path: `tests/test_cuad_dspy_eval.py`.
15. Run validation gates and fix all failures.

## Validation Gates

These must pass without an API key:

```bash
uv run python -m py_compile explore.py cuad_dspy_eval.py
uv run python cuad_dspy_eval.py --dry-run --sample-size 10 --seed 42 --output-dir outputs
uv run python cuad_dspy_eval.py --dry-run --sample-size 10 --seed 42 --num-threads 4 --parallel-axis flat --output-dir outputs_parallel
uv run python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("outputs/cuad_dspy_eval_summary.json").read_text())
parallel = json.loads(Path("outputs_parallel/cuad_dspy_eval_summary.json").read_text())
assert summary["sample_size"] == 10
assert summary["seed"] == 42
assert len(summary["selected_document_row_ids"]) == 10
assert len(set(summary["selected_document_row_ids"])) == 10
assert summary["total_examples"] == 410
assert "overlap_accuracy_mean_f1" in summary
assert len(summary["per_category"]) == 41
assert summary["selected_document_row_ids"] == parallel["selected_document_row_ids"]
assert summary["total_examples"] == parallel["total_examples"]
print("summary ok")
PY
uv run python cuad_dspy_eval.py --dry-run --sample-size 10 --seed 42 --output-dir outputs_repeat
uv run python - <<'PY'
import json
from pathlib import Path
first = json.loads(Path("outputs/cuad_dspy_eval_summary.json").read_text())
second = json.loads(Path("outputs_repeat/cuad_dspy_eval_summary.json").read_text())
assert first["selected_document_row_ids"] == second["selected_document_row_ids"]
print("deterministic sample ok")
PY
```

If tests are added:

```bash
uv run pytest -q
```

Full LM evaluation gate, requires credentials:

```bash
OPENAI_API_KEY=... uv run python cuad_dspy_eval.py \
  --sample-size 10 \
  --seed 42 \
  --model openai/gpt-4o-mini \
  --temperature 0 \
  --output-dir outputs
```

Expected success criteria:

- 41 DSPy agents instantiated.
- The 41 agents are keyed by `question_index` and retain category metadata such as `Document Name`, `Parties`, and `Agreement Date`.
- 10 contracts selected deterministically after setting the seed; repeated runs with the same seed produce identical `selected_document_row_ids`.
- Evaluation supports splitting work by contract, by question/category, or by flattened contract-question jobs while preserving deterministic final output ordering.
- 410 predictions evaluated.
- Output CSV has 410 rows.
- Summary JSON has overall `overlap_accuracy_mean_f1` and per-category results for 41 categories.
- Terminal output includes overall accuracy across the 10-contract evaluation.

## Confidence Score

8/10.

The data loading and category alignment are already solved in `explore.py`, and DSPy usage patterns exist locally in `planning/egs/dspy_agent.py`. Main implementation risks are LM context-window limits on full contracts and provider-specific DSPy/LiteLLM credential configuration. The PRP mitigates this with deterministic dry-run validation, cached DSPy setup, and explicit full-run credential gates.
