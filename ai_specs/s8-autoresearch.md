# S8 — Autoresearch: Autonomous System Prompt Optimisation Loop

## Goal

Mirror the `opensrc/autoresearch` loop, but instead of modifying `train.py` to improve
`val_bpb`, modify the system prompt to improve `correct_at_0_5` for one CUAD question
category.

| autoresearch | S8 |
|---|---|
| `train.py` | system prompt (`candidate.py`) |
| `evaluate_bpb()` | `agent.py --question-index N --prompts-file F` |
| `val_bpb` (lower = better) | `correct_at_0_5` (higher = better) |
| `results.tsv` | `results.tsv` |
| agent edits `train.py` directly | triage + synthesis produce the edit |

**One script, one command:**

```bash
DEEPSEEK_API_KEY=... uv run python autoresearch.py --question-index 7
```

That is the entire interface. `autoresearch.py` runs the full loop — baseline eval,
then N iterations of triage → synthesise → validate → keep/discard — and stops. No
human interaction required between iterations.

The seed prompt is `prompts/system_prompts_v2.py`. Accepted improvements are written to
`prompts/autoresearch/{YYYYMMDD}/q{question_index}/` so each of the 41 CUAD categories
can be optimised independently.

---

## Definition of complete

### Proof-of-concept live run

```bash
DEEPSEEK_API_KEY=... uv run python autoresearch.py \
    --question-index 7 \
    --sample-size 5 \
    --max-iterations 3
```

**5 contracts, 1 question (Governing Law), 3 optimisation iterations** plus a baseline
eval = 4 rows in `results.tsv`. Each iteration completes the full
triage → synthesise → validate cycle. The run must write:

- `outputs/autoresearch/{YYYYMMDD}/q07/results.tsv` — 4 rows
- `outputs/autoresearch/{YYYYMMDD}/q07/iter_{1,2,3}/` — eval CSVs, triage JSONL,
  candidate.py, report.html for each iteration
- At least one accepted prompt under `prompts/autoresearch/{YYYYMMDD}/q07/`
  (or all discarded with valid discard logs — both outcomes are valid)

### Invariants

1. `autoresearch.py` is the single entry point. One command runs everything.
2. `agent.py` is never modified. It is called as a subprocess with `--prompts-file`
   pointing to the current or candidate prompt — exactly as the user would run it
   manually.
3. `triage.py` and `synthesis.py` are internal modules called by `cli.py`. They are
   not standalone scripts and are not called as subprocesses.
4. `triage.py` reuses `add_common_eval_args` from `cuad_agent.evaluators.cli_common`,
   giving it the same flag vocabulary as `agent.py`.
5. `--question-index` integer is resolved to a category name string **once** at startup
   in `cli.py`. All internal logic uses the category name string.
6. `--dry-run` stubs all LLM calls. `uv run pytest -q` passes with no API keys.
7. No modification to `agent.py` or any existing module.

---

## How the loop works

### Setup (runs once at startup)

1. Resolve `question_index → category` from the dataset.
2. Create output directories; initialise `results.tsv` with the header row.
3. **Baseline eval** — check whether `outputs/{model_id}/cuad_langchain_eval_results.csv`
   already exists:
   - **Exists:** copy it to `iter_0/eval_results.csv`; read `correct_at_0_5` for the
     target question directly — skip the `agent.py` subprocess entirely.
   - **Missing:** run `agent.py --model-id {model_id} --prompts-file {prompts_file}
     --sample-size {sample_size} --seed {seed} --question-index {question_index}`;
     then copy output to `iter_0/`.

   Either way, log a row with `status = baseline`. Set
   `current_prompt = args.prompts_file` and
   `current_eval_results = outputs/{model_id}/cuad_langchain_eval_results.csv`.

### Iteration loop (runs N times, default 10)

Each iteration receives `current_prompt` and `current_eval_results` from the previous
step (baseline or last kept candidate).

```
LOOP iter = 1 to --max-iterations:

  1. TRIAGE    — call triage(current_eval_results, current_prompt)
                 → iter_{N}/triage_outputs.jsonl

  2. SYNTHESISE — call synthesise(diagnoses, current_prompt, history)
                 → iter_{N}/candidate.py

  3. VALIDATE  — call agent.py with --prompts-file iter_{N}/candidate.py
                 (same --sample-size, --seed, --question-index as baseline)
                 → iter_{N}/candidate_results.csv + candidate_summary.json

  4. COMPARE   — read correct_at_0_5 from candidate_summary.json
                 vs current correct_at_0_5

  5. KEEP      if candidate accuracy > current accuracy:
                 copy candidate.py → prompts/autoresearch/.../
                 set current_prompt = accepted file
                 set current_eval_results = candidate_results.csv
                 log status = keep

     DISCARD   otherwise:
                 current_prompt unchanged
                 current_eval_results unchanged
                 log status = discard

  6. LOG       — append row to results.tsv

  7. REPORT    — write iter_{N}/report.html

STOP after --max-iterations. Print summary.
```

**After a keep:** next iteration triages the NEW eval results (the kept candidate's
eval). New wrong answers may appear.

**After a discard:** next iteration triages the SAME eval results again — but the
history list now includes the discard, so synthesis must try a different approach.

**Crash handling:** if `agent.py` exits non-zero, retry once. On second failure log
`status = crash`, leave `current_prompt` and `current_eval_results` unchanged, continue.

---

## What each step does

### Triage

`triage(eval_results_path, current_prompt)` is a Python function in `triage.py`.
`cli.py` calls it directly — not as a subprocess.

It reads `current_eval_results` (already on disk from the previous agent.py run),
filters to rows where `correct_at_0_5 == 0` and `question_index == target`, then calls
the triage LLM once per wrong-answer row.

**Each triage LLM call receives:**

```
system_prompt    — current system prompt for this category
contract_title   — contract title
contract_text    — full raw contract text (loaded from dataset by document_row_id)
question         — CUAD question text for this category
provided_answer  — what agent.py predicted
golden_answer    — ground-truth answer span(s)
```

**Output per wrong answer — `TriageDiagnosis`:**

```python
class TriageDiagnosis(BaseModel):
    contract_id: int          # = document_row_id
    golden_answer_location: str  # verbatim sentences surrounding golden answer
    failure_reason: str          # why the system prompt led to the wrong answer
    proposed_rule: str           # concrete rule referencing actual clause structure
    confidence: Literal["high", "medium", "low"]
```

Diagnoses are written to `iter_{N}/triage_outputs.jsonl` (one JSON object per line).

**`triage.py` accepts the same flags as `agent.py`** via `add_common_eval_args`:
`--question-index`, `--prompts-file`, `--model-id`, `--output-dir`, `--dry-run`.
It also accepts `--model` (the triage LLM) and `--output` (JSONL destination).
These shared flags exist so `cli.py` can pass them through unchanged.

`model_id` carries the identity of the eval run being triaged — it directly determines
which scored contracts are being diagnosed. Triage uses it in two ways:
1. **Locate eval results:** `outputs/{model_id}/cuad_langchain_eval_results.csv`
2. **Cache triage output:** if `output_path` already exists and is non-empty, triage
   loads and returns the cached `TriageDiagnosis` records without making any LLM call.
   This prevents re-diagnosing the same wrong answers if the loop is interrupted and
   restarted between triage and synthesis.

---

### Synthesise

`synthesise(diagnoses, current_prompt, history)` is a Python function in `synthesis.py`.
`cli.py` calls it directly — not as a subprocess.

It makes **one LLM call**. Python assembles all context before the call:

- `category` — resolved at startup
- `current_prompt` — the system prompt in use this iteration
- `diagnoses` — all `TriageDiagnosis` records from this iteration's `triage_outputs.jsonl`
- `history` — list of `{iter, status, notes, prompt_text}` assembled by `cli.py`
  by reading `results.tsv` and each prior `candidate.py`

No tools, no agent loop, no file reads by the LLM. Python reads the files; the LLM
receives the assembled text in a single prompt.

**Output — `SynthesisResult`:**

```python
class SynthesisResult(BaseModel):
    prompt_text: str   # full new system prompt text
    notes: str         # one-line description: what changed and why
```

Parsed via `PydanticOutputParser` — same pattern as `CuadAnswer` in `langchain_agent.py`.

**Signature:**

```python
def synthesise(
    *,
    category: str,
    current_prompt: str,
    diagnoses: list[TriageDiagnosis],
    history: list[dict],   # {iter, status, notes, prompt_text} — assembled by cli.py
    model_id: str,         # candidate model_id for this iteration — carried for traceability
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 64000,
    dry_run: bool = False,
) -> SynthesisResult
```

`cli.py` writes the result to `iter_{N}/candidate.py` using
`results.py:write_prompt_module(path, category, prompt_text)`.

---

### model_id construction

`--model-id` is the base model_id passed to autoresearch. It must match the model_id of
an existing (or intended) `agent.py` eval run — e.g. `eval-raw` from:

```bash
uv run python agent.py \
    --context-mode raw \
    --model-id eval-raw \
    --question-index 7 \
    --prompts-file prompts/system_prompts_v2.py \
    --sample-size 50 \
    --seed 42
```

`cli.py` uses `--model-id` directly as the baseline and derives candidate model_ids from it:

| Step | model_id |
|------|----------|
| Baseline | `{model_id}` — e.g. `eval-raw` |
| Candidate iter N | `{model_id}_ar_r{round}_i{N}` — e.g. `eval-raw_ar_r1_i1` |

`cli.py` passes `model_id` explicitly to `triage()` (the model_id of the eval being
diagnosed, i.e. `current_model_id`) and to `synthesise()` (the candidate model_id for
this iteration).

---

### Validate

`cli.py` calls `agent.py` as a subprocess with the candidate prompt:

```bash
uv run python agent.py \
    --context-mode raw \
    --model-id {base_model_id}_ar_r{round}_i{N} \
    --question-index {question_index} \
    --prompts-file outputs/autoresearch/{YYYYMMDD}/q{question_index}/iter_{N}/candidate.py \
    --sample-size {same as baseline} \
    --seed {same as baseline}
```

Same `--sample-size` and `--seed` as the baseline so results are directly comparable.
`agent.py` writes its output to `outputs/{model_id}/`. `cli.py` reads back:

- `outputs/{model_id}/cuad_langchain_eval_results.csv` → `iter_{N}/candidate_results.csv`
- `outputs/{model_id}/cuad_langchain_eval_summary.json` → `iter_{N}/candidate_summary.json`

**Metric:** `correct_at_0_5` for the target `question_index`.

---

## Architecture

### Module layout

```
src/cuad_agent/autoresearch/
├── __init__.py
├── program.md                    # human-readable description of what autoresearch.py does
├── cli.py                        # loop orchestrator; resolves question_index → category
├── triage.py                     # triage() function; reuses add_common_eval_args
├── synthesis.py                  # synthesise() function; single LLM call
├── prompts/
│   ├── triage_system_prompt.py
│   └── synthesis_system_prompt.py
├── results.py                    # write_tsv_row(), write_prompt_module()
└── report.py                     # write_iter_report()

autoresearch.py                   # root entry point → cli.main()
prompts/autoresearch/             # accepted prompts, organised by date / question-index
```

### Output layout

```
outputs/autoresearch/
└── {YYYYMMDD}/
    └── q{question_index}/
        ├── results.tsv
        ├── progress.html                # updated after every iteration
        ├── iter_0/                      # baseline
        │   ├── eval_results.csv         # copied from outputs/{model_id}/
        │   └── eval_summary.json
        ├── iter_1/
        │   ├── triage_outputs.jsonl     # one TriageDiagnosis per wrong answer
        │   ├── candidate.py             # synthesised candidate prompt
        │   ├── candidate_results.csv    # agent.py output for candidate
        │   ├── candidate_summary.json
        │   ├── accepted.py              # copy written only if kept
        │   └── report.html
        └── …
```

---

## question-index resolution

`--question-index` accepts an integer (0–40). `cli.py` resolves it once at startup:

```python
_, questions = load_datasets()
category = str(questions[questions["question_index"] == args.question_index].iloc[0]["category"])
```

`category` (e.g. `"Governing Law"`) is used everywhere internally. The integer is only
passed back to `agent.py --question-index`. Never hardcode the mapping — always derive
from the dataset at runtime.

---

## Prompt naming convention

```
prompts/autoresearch/
└── {YYYYMMDD}/
    └── q{question_index}/        # zero-padded, e.g. q07
        └── {seed}_r{round}_i{N}.py
```

Example: `prompts/autoresearch/20260607/q07/v2_r1_i1.py`

| Part | Meaning |
|------|---------|
| `YYYYMMDD` | date the run started |
| `q{question_index}` | zero-padded index, 00–40 |
| `{seed}` | seed version (`v2`) |
| `r{round}` | round — incremented when a fresh seed is chosen |
| `i{N}` | accepted improvement number within the round |

Only **accepted** (kept) prompts are written here. Discarded candidates stay in
`outputs/…/iter_{N}/candidate.py`. The current live prompt is always the highest `i{N}`
file in the round directory.

---

## Results TSV

Tab-separated. Never commas — they appear in prompt text.

```
iter  question_index  category       model_id                    prompt_file                                              correct_at_0_5  n_wrong  n_diagnosed  status    notes
0     7               Governing Law  eval-raw                    prompts/system_prompts_v2.py                             0.4200          29       0            baseline  reused existing eval
1     7               Governing Law  eval-raw                    prompts/system_prompts_v2.py                             0.4600          27       29           keep      +rule: jurisdiction in subordinate clause
2     7               Governing Law  eval-raw_ar_r1_i1           prompts/autoresearch/20260607/q07/eval-raw_ar_r1_i1.py  0.4400          27       27           discard   tried N/A guard — overcautious; reverted to i1
3     7               Governing Law  eval-raw_ar_r1_i1           prompts/autoresearch/20260607/q07/eval-raw_ar_r1_i1.py  0.4800          25       27           keep      +rule: extract full sentence not just jurisdiction name
```

| Column | Description |
|--------|-------------|
| `iter` | 0-based integer |
| `question_index` | integer passed to `--question-index` |
| `category` | resolved at startup; never changes within a run |
| `model_id` | model_id of the eval being compared against (baseline = `--model-id`; candidates = `{model_id}_ar_r{round}_i{N}`) |
| `prompt_file` | `current_prompt` used as the baseline for this iteration's triage |
| `correct_at_0_5` | accuracy of `current_prompt` (the baseline being compared against) |
| `n_wrong` | wrong answers available for triage |
| `n_diagnosed` | diagnoses produced (0 for baseline row) |
| `status` | `baseline`, `keep`, `discard`, `crash` |
| `notes` | precise description of what changed or why discarded |

---

## CLI

```
uv run python autoresearch.py [OPTIONS]
```

| Flag | Default | Notes |
|------|---------|-------|
| `--question-index` | required | Resolved to category name at startup; passed unchanged to `agent.py` |
| `--model-id` | required | Base model_id — must match an existing or intended `agent.py` eval run; reused as baseline if results already exist |
| `--prompts-file` | required | Prompt used in the base eval run; passed to `agent.py` for baseline and updated to accepted candidate on each keep |
| `--output-dir` | `outputs` | Root output directory; passed unchanged to `agent.py` |
| `--dry-run` | False | Stubs all LLM calls in triage and synthesis; passes `--dry-run` to `agent.py` |
| `--model` | `deepseek/deepseek-v4-flash` | LLM for contract eval — passed to `agent.py` |
| `--triage-model` | `deepseek/deepseek-v4-flash` | LLM for triage |
| `--synthesis-model` | `deepseek/deepseek-v4-pro` | LLM for synthesis (teacher agent) |
| `--sample-size` | `50` | Contracts per eval — passed to `agent.py` |
| `--seed` | `42` | RNG seed — passed to `agent.py` |
| `--context-mode` | `raw` | Contract context mode — passed to `agent.py` |
| `--round` | `1` | Encoded into accepted prompt filename only |
| `--max-iterations` | `10` | Iterations after baseline |

```bash
# Standard run — reuse existing eval-raw results as baseline
DEEPSEEK_API_KEY=... uv run python autoresearch.py \
    --model-id eval-raw \
    --prompts-file prompts/system_prompts_v2.py \
    --question-index 7

# Proof-of-concept — 5 contracts, 3 iterations (runs baseline eval if not cached)
DEEPSEEK_API_KEY=... uv run python autoresearch.py \
    --model-id eval-raw \
    --prompts-file prompts/system_prompts_v2.py \
    --question-index 7 \
    --sample-size 5 \
    --max-iterations 3

# Round 2 seeded from best round-1 result
DEEPSEEK_API_KEY=... uv run python autoresearch.py \
    --model-id eval-raw_ar_r1_i3 \
    --prompts-file prompts/autoresearch/20260607/q07/eval-raw_ar_r1_i3.py \
    --question-index 7 \
    --round 2

# Dry-run wiring check (no API key needed)
uv run python autoresearch.py \
    --model-id eval-raw \
    --prompts-file prompts/system_prompts_v2.py \
    --question-index 7 \
    --dry-run
```

**Environment variables**

| Variable | Required for |
|----------|-------------|
| `DEEPSEEK_API_KEY` | All LLM calls (eval via agent.py, triage, synthesis) |

---

## Implementation tasks

Tasks are grouped into four waves. Waves 1–3 write code; Wave 4 writes tests and
verifies the whole implementation. Within Wave 2 all three tasks are independent and
can be executed by separate subagents in parallel.

### Dependency graph

```
Wave 1 ── Task A  (foundation + shared models)
               ↓
Wave 2 ── Task B  (triage)    ─┐
          Task C  (synthesis)  ├── parallel, all depend on A only
          Task D  (report)    ─┘
               ↓
Wave 3 ── Task E  (cli + entry point)
               ↓
Wave 4 ── Task F  (tests + done gate)
```

---

### Task A — Foundation  *(Wave 1, run first)*

**Files to create:**
- `src/cuad_agent/autoresearch/__init__.py` — empty, makes the package importable
- `src/cuad_agent/autoresearch/results.py` — TSV logger, prompt writer, **and the shared Pydantic models**
- `src/cuad_agent/autoresearch/llm.py` — LLM factory shared by triage and synthesis
- `src/cuad_agent/autoresearch/program.md` — human-readable description of what `autoresearch.py` does (not executable instructions)
- `src/cuad_agent/autoresearch/prompts/__init__.py` — empty

**`results.py` must export:**
```python
from typing import Literal

class TriageDiagnosis(BaseModel):
    contract_id: int
    golden_answer_location: str
    failure_reason: str
    proposed_rule: str
    confidence: Literal["high", "medium", "low"]

class SynthesisResult(BaseModel):
    prompt_text: str
    notes: str

def write_tsv_row(path: Path, row: dict) -> None: ...
    # appends one tab-separated row; creates file with header if it doesn't exist

def write_prompt_module(path: Path, category: str, prompt_text: str) -> None: ...
    # writes a candidate.py with CATEGORY_SYSTEM_PROMPTS = {category: prompt_text}
```

**`llm.py` must export:**
```python
from langchain_core.language_models import BaseChatModel

def make_llm(model: str, temperature: float = 0.0, max_tokens: int = 64000) -> BaseChatModel:
    # builds argparse.Namespace(model=model, temperature=temperature, max_tokens=max_tokens)
    # and delegates to configure_llm() from cuad_agent.agents.langchain_agent
    ...
```

`triage.py` and `synthesis.py` import `make_llm` from here — never call `configure_llm` directly with a hand-rolled Namespace.

`TriageDiagnosis` and `SynthesisResult` live in `results.py` — **not** in `triage.py`
or `synthesis.py` — so Tasks B, C, D can all import them without depending on each other.

**Done when:**
```bash
uv run python -m py_compile src/cuad_agent/autoresearch/results.py src/cuad_agent/autoresearch/llm.py
uv run python -c "from cuad_agent.autoresearch.results import TriageDiagnosis, SynthesisResult, write_tsv_row, write_prompt_module"
uv run python -c "from cuad_agent.autoresearch.llm import make_llm"
uv run python -c "from cuad_agent.autoresearch import cli"  # resolves (cli.py doesn't exist yet — __init__ must not import it eagerly)
```

---

### Task B — Triage  *(Wave 2, parallel with C and D)*

**Files to create:**
- `src/cuad_agent/autoresearch/triage.py`
- `src/cuad_agent/autoresearch/prompts/triage_system_prompt.py`

**Read before writing:**
- `src/cuad_agent/evaluators/cli_common.py` — understand `add_common_eval_args`; note it does **not** include `--question-index`, triage adds that flag separately
- `src/cuad_agent/agents/langchain_agent.py` — read `configure_llm` to understand the Namespace shape; do **not** call it directly — use `make_llm` from `cuad_agent.autoresearch.llm` instead
- `src/cuad_agent/data/dataset.py` — `load_datasets()` returns `(contracts_df, questions_df)`
- `src/cuad_agent/prompts/loader.py` — how to load a prompt module and read `CATEGORY_SYSTEM_PROMPTS`

**Message construction**: `contract_text` is raw legal text and may contain literal `{` and `}` characters. Build triage messages with `SystemMessage` and `HumanMessage` directly — **never** `ChatPromptTemplate` or f-string template substitution on the contract body. This matches the pattern in `langchain_agent.py:build_chain_for_agent`.

**`triage.py` must export:**
```python
def triage(
    *,
    eval_results_path: Path,      # outputs/{model_id}/cuad_langchain_eval_results.csv
    question_index: int,
    category: str,
    prompts_file: Path,
    model_id: str,                # identity of the eval being triaged; used to locate results + cache
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 64000,
    output_path: Path,            # iter_{N}/triage_outputs.jsonl
    dry_run: bool = False,
) -> list[TriageDiagnosis]:
```

`dry_run=True` — return two stub `TriageDiagnosis` objects; write them to `output_path`; make no LLM call.

**Cache check (first thing triage does):** if `output_path` exists and is non-empty, parse
each line as `TriageDiagnosis` and return the list immediately — no LLM calls, no CSV read.

**What triage does:**
1. Loads `eval_results_path` as a DataFrame; filters `correct_at_0_5 == 0` and `question_index == target`
2. Loads contract text per row from `load_datasets()[0]` via `document_row_id`
3. Loads the system prompt from `prompts_file` under the `category` key
4. Calls the triage LLM once per wrong-answer row using the triage system prompt, passing `model`, `temperature`, `max_tokens` to `make_llm()`
5. Parses each response: `TriageDiagnosis.model_validate(json.loads(raw_text))`; writes all to `output_path` as JSONL.
   Do **not** use `PydanticOutputParser` — the triage system prompt already encodes the exact JSON structure; adding format instructions is redundant and risks confusing the model.

**Done when:**
```bash
uv run python -m py_compile src/cuad_agent/autoresearch/triage.py
uv run python -c "
from pathlib import Path
from cuad_agent.autoresearch.triage import triage
# dry-run needs a real CSV — create a minimal stub
import pandas as pd, tempfile, json
with tempfile.TemporaryDirectory() as d:
    csv_path = Path(d) / 'eval_results.csv'
    out_path = Path(d) / 'triage.jsonl'
    pd.DataFrame({'correct_at_0_5':[0.0],'question_index':[7],'document_row_id':[327],'predicted_answer':['N/A'],'golden_answer':['New York'],'contract_title':['Test']}).to_csv(csv_path, index=False)
    results = triage(eval_results_path=csv_path, question_index=7, category='Governing Law', prompts_file=Path('prompts/system_prompts_v2.py'), model_id='eval-raw', model='deepseek/deepseek-v4-flash', output_path=out_path, dry_run=True)
    assert len(results) == 1
    assert out_path.exists()
    print('triage dry-run OK')
"
```

---

### Task C — Synthesis  *(Wave 2, parallel with B and D)*

**Files to create:**
- `src/cuad_agent/autoresearch/synthesis.py`
- `src/cuad_agent/autoresearch/prompts/synthesis_system_prompt.py`

**`synthesis.py` must export:**
```python
def synthesise(
    *,
    category: str,
    current_prompt: str,
    diagnoses: list[TriageDiagnosis],
    history: list[dict],   # [{iter, status, notes, prompt_text}, ...]
    model_id: str,         # candidate model_id for this iteration — carried for traceability
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 64000,
    dry_run: bool = False,
) -> SynthesisResult:
```

**Read before writing:**
- `src/cuad_agent/agents/langchain_agent.py` — read `configure_llm` and `PydanticOutputParser` usage; use `make_llm` from `cuad_agent.autoresearch.llm` for LLM construction; use `PydanticOutputParser(pydantic_object=SynthesisResult)` to parse the response (same pattern as `CuadAnswer`)
- `src/cuad_agent/autoresearch/results.py` — import `TriageDiagnosis`, `SynthesisResult`
- `src/cuad_agent/autoresearch/llm.py` — import `make_llm`

`dry_run=True` — return a stub `SynthesisResult` with non-empty `prompt_text` and `notes`; make no LLM call.

**Done when:**
```bash
uv run python -m py_compile src/cuad_agent/autoresearch/synthesis.py
uv run python -c "
from cuad_agent.autoresearch.synthesis import synthesise
result = synthesise(category='Governing Law', current_prompt='test', diagnoses=[], history=[], model_id='eval-raw_ar_r1_i1', model='deepseek/deepseek-v4-pro', dry_run=True)
assert result.prompt_text and result.notes
print('synthesis dry-run OK')
"
```

---

### Task D — Report  *(Wave 2, parallel with B and C)*

**Files to create:**
- `src/cuad_agent/autoresearch/report.py`

**Read before writing:**
- `src/cuad_agent/autoresearch/results.py` — import `TriageDiagnosis`
- `src/cuad_agent/dashboards/` — reference for existing HTML dashboard patterns

**`report.py` must export:**
```python
def write_iter_report(
    *,
    iter_n: int,
    category: str,
    question_index: int,
    date_str: str,
    status: str,                     # "keep" | "discard" | "crash"
    notes: str,
    current_accuracy: float,
    candidate_accuracy: float,
    current_eval_df: pd.DataFrame,
    candidate_eval_df: pd.DataFrame,
    triage_diagnoses: list[TriageDiagnosis],
    candidate_prompt_text: str,
    output_path: Path,
) -> None:

def write_progress_report(
    *,
    rows: list[dict],         # all TSV rows so far; keys: iter, status, correct_at_0_5, notes, prompt_file
    category: str,
    question_index: int,
    output_path: Path,
) -> None:
```

`write_iter_report` — self-contained HTML with inline CSS, no JavaScript required.
`write_progress_report` — self-contained HTML with inline CSS and vanilla JavaScript for hover tooltips.

**Done when:**
```bash
uv run python -m py_compile src/cuad_agent/autoresearch/report.py
uv run python -c "
from pathlib import Path
from cuad_agent.autoresearch.report import write_iter_report, write_progress_report
import pandas as pd, tempfile
cols = ['document_row_id','contract_title','predicted_answer','golden_answer','correct_at_0_5']
df = pd.DataFrame([[327,'Test Ltd','N/A','New York',0.0]], columns=cols)
rows = [
    {'iter': 0, 'status': 'baseline', 'correct_at_0_5': 0.4, 'notes': 'seed: v2', 'prompt_file': 'prompts/system_prompts_v2.py'},
    {'iter': 1, 'status': 'keep',     'correct_at_0_5': 0.5, 'notes': '+rule: jurisdiction test', 'prompt_file': 'prompts/system_prompts_v2.py'},
]
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / 'report.html'
    write_iter_report(iter_n=1, category='Governing Law', question_index=7, date_str='20260607', status='keep', notes='test', current_accuracy=0.4, candidate_accuracy=0.5, current_eval_df=df, candidate_eval_df=df, triage_diagnoses=[], candidate_prompt_text='test prompt', output_path=out)
    html = out.read_text()
    assert 'Governing Law' in html and '0.4' in html and 'test prompt' in html
    prog = Path(d) / 'progress.html'
    write_progress_report(rows=rows, category='Governing Law', question_index=7, output_path=prog)
    phtml = prog.read_text()
    assert 'Governing Law' in phtml and 'Kept Improvements' in phtml and 'jurisdiction test' in phtml
    print('report + progress OK')
"
```

---

### Task E — Orchestration  *(Wave 3, after B, C, D complete)*

**Files to create:**
- `src/cuad_agent/autoresearch/cli.py`
- `autoresearch.py` (root entry point)

**Read before writing:**
- `src/cuad_agent/autoresearch/triage.py` — `triage()` signature
- `src/cuad_agent/autoresearch/synthesis.py` — `synthesise()` signature
- `src/cuad_agent/autoresearch/report.py` — `write_iter_report()` signature
- `src/cuad_agent/autoresearch/results.py` — `write_tsv_row()`, `write_prompt_module()`
- `src/cuad_agent/data/dataset.py` — `load_datasets()` for category resolution
- `agent.py` — confirm available flags before building the subprocess call

**`cli.py` responsibilities:**
1. Parse args (see CLI table in §CLI)
2. Resolve `question_index → category` via `load_datasets()` — once at startup
3. Run baseline eval: `subprocess.run(["uv", "run", "python", "agent.py", ...], check=True)`
4. Copy baseline eval output to `outputs/autoresearch/{date}/q{idx}/iter_0/`
5. Loop `max_iterations` times: call `triage()` → `synthesise()` → run validate subprocess → compare → keep/discard → `write_tsv_row()` → `write_iter_report()` → `write_progress_report()`
6. On keep: copy `candidate.py` to `prompts/autoresearch/.../{seed}_r{round}_i{N}.py`

**`autoresearch.py`** is a three-line shim:
```python
from cuad_agent.autoresearch.cli import main
if __name__ == "__main__":
    main()
```

**Done when:**
```bash
uv run python -m py_compile src/cuad_agent/autoresearch/cli.py autoresearch.py
uv run python autoresearch.py --question-index 7 --dry-run
# must exit 0 and write outputs/autoresearch/{date}/q07/results.tsv
```

---

### Task F — Tests + Done Gate  *(Wave 4, after all code is complete)*

**File to create:**
- `tests/test_autoresearch.py`

Write all 9 tests listed in §Testing. Every test must use `dry_run=True` or in-memory
fixtures — no API keys, no subprocess calls to `agent.py`.

**Implementation is complete when ALL of the following pass:**

```bash
# 1. All tests — new and existing — pass
uv run pytest -q

# 2. Dry-run entry point works end-to-end
uv run python autoresearch.py --question-index 7 --dry-run

# 3. Output layout exists after dry-run
ls outputs/autoresearch/*/q07/results.tsv
ls outputs/autoresearch/*/q07/progress.html
ls outputs/autoresearch/*/q07/iter_*/report.html

# 4. No existing module was modified
git diff --name-only src/cuad_agent/evaluators/ src/cuad_agent/agents/ agent.py
# must print nothing

# 5. All new Python files compile cleanly
uv run python -m py_compile \
    src/cuad_agent/autoresearch/__init__.py \
    src/cuad_agent/autoresearch/results.py \
    src/cuad_agent/autoresearch/llm.py \
    src/cuad_agent/autoresearch/triage.py \
    src/cuad_agent/autoresearch/synthesis.py \
    src/cuad_agent/autoresearch/report.py \
    src/cuad_agent/autoresearch/cli.py \
    autoresearch.py
```

If any of the five checks fail, the implementation is not complete.

---

## Triage system prompt

```python
TRIAGE_SYSTEM_PROMPT = """\
You are a legal AI evaluation analyst. A legal contract review agent answered a
question incorrectly. Your job is to read the raw contract, find where the correct
answer actually appears, and explain exactly why the system prompt failed to guide
the agent to it — then propose a specific rule to fix that gap.

You will receive:
- system_prompt:    the system prompt that was used
- question:         the legal review question
- contract_title:   the contract name
- contract_text:    the full raw contract text
- provided_answer:  what the agent predicted
- golden_answer:    the correct answer span(s) from the contract

Follow these steps in order:

STEP 1 — Locate the golden answer in the contract.
Find the exact sentence(s) in contract_text that contain or immediately surround
the golden_answer span. Quote them verbatim. Note the clause type (subordinate clause,
numbered section, definition, recital, boilerplate) and any section heading.

STEP 2 — Diagnose the failure.
Compare what the agent provided against where the golden answer actually sits.
Classify the error:
  over_refusal  — agent answered N/A but the span exists in the contract
  span_mismatch — right section, wrong sentence or wrong amount of text
  hallucination — agent invented text not present in the contract
  omission      — missed the clause; system prompt didn't steer toward it

STEP 3 — Propose a grounded rule.
Write one concrete rule referencing the clause structure found in STEP 1.
It must be specific enough to produce the golden answer on this contract and
general enough to transfer to similar contracts.
Bad:  "look more carefully at the contract"
Good: "When the governing law clause appears inside a 'General' or 'Miscellaneous'
      section and uses the phrase 'construed under the laws of', extract that full
      sentence even when it does not contain the word 'govern'."

Return ONLY a JSON object — no text outside it:
{
  "golden_answer_location": "verbatim sentences surrounding the golden answer",
  "failure_reason": "why the system prompt led to the wrong answer",
  "proposed_rule": "the concrete rule from STEP 3",
  "confidence": "high | medium | low"
}
"""
```

---

## Synthesis system prompt

```python
SYNTHESIS_SYSTEM_PROMPT = """\
You are a legal AI prompt engineer. Your job is to improve a system prompt for a
contract-review agent so it correctly answers a specific type of legal question.

You will receive:
- category:       the legal question category (e.g. "Governing Law")
- current_prompt: the system prompt currently in use
- diagnoses:      TriageDiagnosis records for every wrong answer this iteration
- history:        prior iterations — each entry has iter number, status (keep/discard),
                  notes (one-line summary of what was tried), and the full prompt_text
                  of the candidate that was evaluated

REASONING PROTOCOL — follow this order:

STEP 1 — Understand the current failures.
Read all diagnoses. Look for shared patterns across multiple wrong answers — error type,
clause structure, section location. A pattern in 3+ diagnoses is a strong signal. Note
the failure_reason classifications: over_refusal, span_mismatch, hallucination, omission.

STEP 2 — Check history before proposing anything.
For every discard entry in history, read its prompt_text — not just the notes. Understand
exactly what rule text was tried so you do not repeat it. If triage proposes the same
rule that already failed in a discard, skip it.

STEP 3 — Decide your approach.
- First iteration: apply the highest-confidence proposed_rules from diagnoses.
- After a keep: build on what worked; address new failures from the updated triage.
- After a discard: choose a different angle — combine smaller changes, or remove an
  overly restrictive rule rather than adding another.
- After consecutive discards: try one minimal isolated change to identify the regression
  source rather than a broad rewrite.
- Never repeat rule text that appears in any prior history entry with status=discard.

STEP 4 — Write the candidate prompt.
Make the minimum change that addresses the diagnosed pattern. Prefer one grounded rule
over multiple speculative changes. A prompt that regresses is worse than no change.

Return ONLY a JSON object — no text outside it:
{
  "prompt_text": "the full new system prompt text",
  "notes": "one precise line: what changed and why"
}
Good notes: "+rule: extract full sentence when 'construed under the laws of' appears in General section (iter 2 N/A guard overcautious — reverted)"
Bad notes:  "updated system prompt"
"""
```

---

## Prompt module helper

```python
def write_prompt_module(path: Path, category: str, prompt_text: str) -> None:
    content = (
        f'"""Autoresearch candidate — category: {category}."""\n\n'
        f"CATEGORY_SYSTEM_PROMPTS = {{\n"
        f'    "{category}": """{prompt_text}""",\n'
        f"}}\n"
    )
    path.write_text(content)
```

The dict key is always the **category name string**, never the integer question_index.
This is the format `agent.py` already expects when loading prompts via `--prompts-file`.

---

## HTML Iteration Report

Written to `outputs/autoresearch/{YYYYMMDD}/q{question_index}/iter_{N}/report.html`
after every iteration. Always written — whether kept, discarded, or crashed.
Self-contained: inline CSS, no external dependencies.

### Page structure

```
┌──────────────────────────────────────────────────────────┐
│  Iter N — Governing Law (q07)       [KEPT] / [DISCARDED] │
│  2026-06-07                                               │
├─────────────┬──────────────┬─────────────────────────────┤
│  Before     │  After       │  Δ                          │
│  0.42       │  0.46        │  +0.04 ▲                    │
├──────────────────────────────────────────────────────────┤
│  Candidate System Prompt                                  │
│  ┌─────────────────────────────────────────────────┐    │
│  │  <full prompt text, scrollable pre block>       │    │
│  └─────────────────────────────────────────────────┘    │
│  Change: +rule: jurisdiction in subordinate clause        │
├──────────────────────────────────────────────────────────┤
│  Changed Answers (N contracts)                            │
│  Contract              Before  After  Direction           │
│  Master Services Agr…  ✗       ✓      improved           │
│  Lease Agreement 2019  ✓       ✗      regressed          │
├──────────────────────────────────────────────────────────┤
│  All Results (50 contracts)  [correct: 23  wrong: 27]     │
│  ✓  Software License Agreement   New York                 │
│  ✗  ▶ Master Services Agreement  (expand for triage)     │
└──────────────────────────────────────────────────────────┘
```

**Header:** iteration, category, question index, date, status badge (green/amber/red).

**Accuracy strip:** Before / After / Δ. Delta coloured green if positive, red if negative.

**Candidate prompt:** `<pre>` block with full `candidate.py` text. Change notes in italics below it.

**Changed answers table:** join `current_eval_df` and `candidate_eval_df` on
`document_row_id`; rows where `correct_at_0_5` differs. Each row is a `<details>`
element; expanding shows golden answer, both predicted answers, and the triage
`golden_answer_location` if a diagnosis exists.

**All results table:** every row from `candidate_eval_df`, incorrect first. Incorrect
rows are `<details>` elements; expanding shows the full `TriageDiagnosis` (matched on
`contract_id = document_row_id`). Correct rows are plain `<tr>` elements.

### `write_iter_report` signature

```python
def write_iter_report(
    *,
    iter_n: int,
    category: str,
    question_index: int,
    date_str: str,
    status: str,                     # "keep" | "discard" | "crash"
    notes: str,
    current_accuracy: float,
    candidate_accuracy: float,
    current_eval_df: pd.DataFrame,
    candidate_eval_df: pd.DataFrame,
    triage_diagnoses: list[TriageDiagnosis],
    candidate_prompt_text: str,
    output_path: Path,
) -> None:
```

Build as a single f-string. Write with `output_path.write_text(html, encoding="utf-8")`.

### DataFrame columns expected

| Column | Type | Source |
|--------|------|--------|
| `document_row_id` | int | join key |
| `contract_title` | str | display |
| `predicted_answer` | str | agent output |
| `golden_answer` | str | ground truth |
| `correct_at_0_5` | float | 0.0 or 1.0 |

`TriageDiagnosis.contract_id` equals `document_row_id` — use it to join diagnoses to rows.

---

## Progress Chart

Written to `outputs/autoresearch/{YYYYMMDD}/q{question_index}/progress.html` and
re-written after every iteration so it always shows the full run to date.

Self-contained HTML — inline CSS and JavaScript only, no external dependencies.

### Visual design (mirrors `opensrc/autoresearch/progress.png`)

- **X axis:** Iteration number (0 = baseline, 1, 2, …)
- **Y axis:** `correct_at_0_5` (0.0–1.0, higher is better)
- **Grey dot:** baseline or discarded iteration
- **Green dot:** kept iteration
- **Green stepped line:** running best — connects baseline to each kept dot in order; steps
  horizontally then vertically (same staircase shape as the reference chart)
- **Hover tooltip:** floating `<div>` that appears on mouseover showing iter number, status,
  accuracy, and the synthesis `notes` for that iteration
- **Inline labels:** kept dots display their `notes` text inline, rotated ~30°, same style
  as the labels in the reference chart
- **Chart title:** `Autoresearch Progress: {n} Experiments, {k} Kept Improvements`

### `write_progress_report` signature

```python
def write_progress_report(
    *,
    rows: list[dict],         # all TSV rows so far; keys: iter, status, correct_at_0_5, notes, prompt_file
    category: str,
    question_index: int,
    output_path: Path,
) -> None:
```

`rows` is assembled by `cli.py` from `results.tsv` after each iteration (baseline row
included). Each dict has keys matching the TSV columns.

Build as a single f-string — inline SVG scatter plot with vanilla JavaScript `mouseover`
handlers for the tooltip. Write with `output_path.write_text(html, encoding="utf-8")`.

---

## Testing

| Test | Covers |
|------|--------|
| `test_triage_dry_run` | `triage()` returns list of valid `TriageDiagnosis` with dry-run stubs |
| `test_triage_contract_text` | contract text loaded from dataset by `document_row_id` |
| `test_synthesis_dry_run` | `synthesise(..., dry_run=True)` returns valid `SynthesisResult` without LLM call |
| `test_question_index_resolution` | `cli.py` resolves integer to category name string |
| `test_results_tsv_write` | TSV row written correctly; `category` populated from resolved name |
| `test_loop_dry_run` | full loop runs 2 iterations with `--dry-run`; writes expected files in output layout |
| `test_prompt_naming` | accepted prompt written to correct `{YYYYMMDD}/q{idx}/` path |
| `test_prompt_module_write` | `write_prompt_module` produces importable Python with correct category key |
| `test_report_html_write` | `write_iter_report` with stub DataFrames writes parseable HTML containing prompt text, accuracy values, and at least one changed-answers row |
| `test_progress_html_write` | `write_progress_report` with stub rows writes HTML containing the chart title, at least one kept-dot data point, and the synthesis notes text for that kept iteration |

All tests pass with `uv run pytest -q` — no API keys required.

---

## Scaling to 41 questions

The loop is already parameterised by `--question-index`. Each question index is fully
independent: its own output directory, its own TSV, its own accepted prompts path. Once
the proof-of-concept (1 question, 3 iterations) confirms the loop improves accuracy:

1. Add `--all-questions` flag to `cli.py` that iterates `range(41)` sequentially.
2. Each question resolves its own category, writes to its own path.

After a full run, the best accepted prompt per category lives in
`prompts/autoresearch/{YYYYMMDD}/q{idx}/`. A one-shot assembly script promotes these
into `prompts/system_prompts_v3.py`.

---

## Appendix — Relationship to existing harness

The v2 harness (`prompt_improve_v2.py`) reads a static results CSV and proposes patches
offline — it never re-runs the evaluator to validate improvement.

S8 is a live loop: it runs `agent.py` every iteration, diagnoses failures against the
raw contract, synthesises a new prompt, and immediately validates it with `agent.py`
again on the same contract sample. The v2 harness produced `system_prompts_v2.py`;
S8 takes v2 as its seed.

---

## Open questions / future sprints

- **All 41 question indices**: add `--all-questions` flag once POC confirms improvement.
- **Meta-triage**: when a candidate is discarded, note which rule addition caused the regression.
- **Cost ceiling**: `--max-usd` stop condition alongside `--max-iterations`.
- **Round promotion**: assemble best autoresearch prompts into `prompts/system_prompts_v3.py`.
- **Parallel runs**: `--parallel N` to run N questions concurrently (each is independent).
