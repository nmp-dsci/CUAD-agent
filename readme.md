# CUAD Legal AI Agent

Experiment in building and evaluating a legal AI agent on the
[CUAD](https://www.atticusprojectai.org/cuad) contract-understanding dataset.
The agent answers 41 standard legal-review questions across 50 contracts and is
evaluated against human-labelled golden answers using token-overlap F1.

The primary experiment has two evaluation layers:

- **RAG retrieval coverage** — compare every retriever on whether it surfaces
  the human-labelled golden-answer spans in the top 10/20/30 retrieved chunks.
- **Agent answer accuracy** — run the legal agent over the same
  contract/question evaluation set with each context mode and compare token F1
  plus `correct_at_0_5`.

Supported full-agent context modes are `raw`, `rag-dense`, `rag-hybrid`,
`rag-hierarchical-bm25`, and `rag-hierarchical-dense`.

---

## Evaluation

The latest all-context evaluation used `deepseek/deepseek-v4-flash` with
`prompts/system_prompts_v2.py` on the deterministic 50-contract, seed-42 sample
for 2,050 total contract/question examples. Each version below changes only the
context supplied to the same legal-review agent:

| Version | Description |
|---|---|
| `v1 baseline` | Earlier DeepSeek run with the original prompt set; included as the comparison baseline in saved summaries. |
| `eval-raw` | V2 prompts with the full contract transcript as context. |
| `eval-rag-dense` | V2 prompts with dense sentence retrieval, using the top-30 retrieved chunks. |
| `eval-rag-hybrid` | V2 prompts with reciprocal-rank fusion over BM25 and dense sentence retrieval, using top-30 chunks. |
| `eval-rag-hierarchical-bm25` | V2 prompts with BM25 leaf-sentence retrieval, top-section expansion, then reranking. |
| `eval-rag-hierarchical-dense` | V2 prompts with dense leaf-sentence retrieval, top-section expansion, then reranking. |

| Run | Mean token F1 | Correct at 0.5 | F1 delta vs v1 | Correct delta vs v1 |
|---|---:|---:|---:|---:|
| `v1 baseline` | 40.03% | 39.85% | - | - |
| `eval-raw` | 83.65% | 85.07% | +43.62 pts | +45.22 pts |
| `eval-rag-dense` | 79.52% | 80.05% | +39.49 pts | +40.20 pts |
| `eval-rag-hybrid` | 78.32% | 78.93% | +38.29 pts | +39.07 pts |
| `eval-rag-hierarchical-bm25` | 76.34% | 76.49% | +36.31 pts | +36.63 pts |
| `eval-rag-hierarchical-dense` | 79.88% | 80.29% | +39.85 pts | +40.44 pts |

The strongest run in this set is `eval-raw`: the V2 prompt set with full-contract
context. Among RAG modes, `eval-rag-hierarchical-dense` performs best, slightly
ahead of flat dense retrieval.

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Environment variables

API keys are loaded automatically from `~/.env`. Add your key there:

```text
DEEPSEEK_API_KEY=your-key-here
```

### 3. Verify the dataset loads

```bash
uv run python explore.py
```

Expected output confirms 510 contracts × 41 questions. Open
`frontend/explore.html` in a browser for a visual walkthrough.

---

## End-to-End Evaluation: RAG Coverage → Agent Accuracy

The standard evaluation set is the deterministic 50-contract sample selected by
`--sample-size 50 --seed 42`. This produces 50 contracts × 41 questions =
2,050 contract/question examples. Run the workflow in two stages:

1. Run the RAG pipeline for all retriever types and compare golden-span
   coverage at top 10/20/30.
2. Score the legal agent over the same evaluation set for every supported
   context mode and compare answer accuracy.

Quick command map:

```bash
# 1. Retrieval-only benchmark, no LLM calls. Full command below.
uv run python rag_eval.py --sample-size 50 --seed 42 ...

# 2. LLM answer-accuracy benchmark across all context modes. Full command below.
uv run python agent.py --all-context-modes --sample-size 50 --seed 42 ...
```

### Step 1 — Score RAG Retrieval for All Retriever Types

`rag_eval.py` chunks the 50 evaluation contracts, builds/loads BM25 and dense
indexes, runs every retrieval technique, and writes retrieval coverage metrics.
This step does not call an LLM. It answers: “How often does each retriever put
the golden-answer sentences in the top 10/20/30 retrieved chunks?”

```bash
uv run python rag_eval.py \
  --run-id s7-all-rag-techniques-eval50 \
  --sample-size 50 \
  --seed 42 \
  --retrievers bm25_sentence,dense_sentence,bm25_legal_recursive,dense_legal_recursive,bm25_hierarchical,dense_hierarchical \
  --chunking-version legal-recursive-v1 \
  --hierarchical-leaf-k 50 \
  --hierarchical-top-sections 5 \
  --top-k 30 \
  --contract-scope eval-set \
  --query-enrichment-provider offline
```

Outputs:

```text
frontend/rag_pipeline_eval.html
outputs/s7-all-rag-techniques-eval50/rag/rag_ranking_summary.csv
outputs/s7-all-rag-techniques-eval50/rag/rag_retrieval_results.csv
outputs/s7-all-rag-techniques-eval50/rag/rag_summary.json
```

Open `frontend/rag_pipeline_eval.html` and use the **Hierarchical RAG
Performance** tab to compare all retrievers on the same eval set. The ranking
CSV includes:

```text
gold_sentence_coverage_at_10 / 20 / 30
all_gold_covered_rate_at_10 / 20 / 30
```

Current retrieval comparison on the 50-contract seed-42 eval set:

```text
Retriever                 Avg coverage@10  @20    @30    All covered@10  @20    @30
bm25_sentence             12.2%            17.3%  22.1%  10.3%           15.1%  19.5%
dense_sentence            50.5%            62.4%  71.3%  44.1%           54.4%  64.0%
bm25_legal_recursive      41.5%            54.5%  63.7%  38.2%           50.0%  59.2%
dense_legal_recursive     40.8%            55.2%  63.9%  37.9%           51.5%  60.3%
bm25_hierarchical         13.7%            20.0%  24.7%  11.8%           17.6%  21.3%
dense_hierarchical        50.7%            63.3%  72.1%  44.1%           55.5%  64.7%
```

Cache is written under:

```text
outputs/rag_cache/chunking/sentence-v3/sentence_spans.jsonl
outputs/rag_cache/sparse/sentence-v3/bm25_sentence/bm25_index.pkl
outputs/rag_cache/chunking/sentence-v3/encodings/tfidf/dense_index.pkl
outputs/rag_cache/chunking/legal-recursive-v1/rag_chunks.jsonl
```

---

### Step 2 — Score Agent Accuracy for All Retriever Context Modes

After the retrieval pipeline cache exists, score the agent on the same 50
contracts for each context mode. This is the LLM evaluation: 41 questions × 50
contracts = 2,050 predictions per context mode, or 10,250 predictions for all
five modes.

Supported full-eval context modes:

| Context mode | Retrieval used for LLM context |
|---|---|
| `raw` | Full contract transcript |
| `rag-dense` | Dense sentence top-30 chunks |
| `rag-hybrid` | RRF-fused dense + BM25 sentence top-30 chunks |
| `rag-hierarchical-bm25` | BM25 sentence leaf search → top-section expansion |
| `rag-hierarchical-dense` | Dense sentence leaf search → top-section expansion |

`bm25_legal_recursive` and `dense_legal_recursive` are currently retrieval
pipeline benchmarks only; they are not full-agent `--context-mode` options.

```bash
export DEEPSEEK_API_KEY=...

uv run python agent.py \
  --all-context-modes \
  --model deepseek/deepseek-v4-flash \
  --model-id eval \
  --prompts-file prompts/system_prompts_v2.py \
  --top-k 30 \
  --chunking-version sentence-v3 \
  --hierarchical-leaf-k 50 \
  --hierarchical-top-sections 5 \
  --sample-size 50 \
  --seed 42 \
  --temperature 0 \
  --max-tokens 64000 \
  --num-threads 4
```

You can also replace `uv run python agent.py` with the console-script alias
`uv run cuad-agent` in the same command.

`--all-context-modes` runs `raw`, `rag-dense`, `rag-hybrid`,
`rag-hierarchical-bm25`, and `rag-hierarchical-dense` sequentially. With
`--model-id eval`, the output run IDs are `eval-raw`, `eval-rag-dense`,
`eval-rag-hybrid`, `eval-rag-hierarchical-bm25`, and
`eval-rag-hierarchical-dense`. It prints the comparison table when all runs
finish.

Each run writes:

```text
frontend/evaluation_eval-{mode}.html
outputs/eval-{mode}/cuad_dspy_eval_results.csv
outputs/eval-{mode}/cuad_dspy_eval_summary.json
outputs/eval-{mode}/cuad_dspy_eval_results.jsonl
```

The command prints a summary table when all modes finish. To regenerate the
comparison later from saved outputs:

```bash
uv run cuad-compare-runs \
  outputs/eval-raw \
  outputs/eval-rag-dense \
  outputs/eval-rag-hybrid \
  outputs/eval-rag-hierarchical-bm25 \
  outputs/eval-rag-hierarchical-dense \
  --label raw rag-dense rag-hybrid hier-bm25 hier-dense
```

Open the dashboards side by side for per-question and per-contract inspection:

```text
frontend/evaluation_eval-raw.html
frontend/evaluation_eval-rag-dense.html
frontend/evaluation_eval-rag-hybrid.html
frontend/evaluation_eval-rag-hierarchical-bm25.html
frontend/evaluation_eval-rag-hierarchical-dense.html
```

Use the contract drilldown to see cases where RAG improves or degrades the
answer (e.g. RAG misses the relevant chunk → lower F1; RAG focuses the LLM →
higher F1 on long contracts).

Read the outputs in this order:

1. `frontend/rag_pipeline_eval.html` — which retriever finds the gold spans.
2. `outputs/eval-*/cuad_dspy_eval_summary.json` — aggregate answer accuracy for
   each context mode.
3. `frontend/evaluation_eval-*.html` — per-question and per-contract failure
   review.

---

## RAG Pipeline Reference

### Key flags for `rag_eval.py`

| Flag | Default | Purpose |
|------|---------|---------|
| `--contract-scope` | `all` | `all` (510 contracts) or `eval-set` (50 contracts, faster debug) |
| `--retrievers` | `bm25_sentence,dense_sentence` | Comma-separated retriever list; all current methods are `bm25_sentence`, `dense_sentence`, `bm25_legal_recursive`, `dense_legal_recursive`, `bm25_hierarchical`, `dense_hierarchical` |
| `--chunking-version` | `sentence-v3` | Canonical sentence chunking version; use `legal-recursive-v1` when benchmarking legal-recursive retrievers |
| `--embedding-model` | `tfidf` | `tfidf` or a SentenceTransformers model name |
| `--top-k` | `30` | Retrieved chunks for coverage scoring |
| `--run-id` | `rag-sentence-v1` | Output directory name under `outputs/` |
| `--hierarchical-leaf-k` | `50` | Leaf sentences retrieved before hierarchical section expansion |
| `--hierarchical-top-sections` | `5` | Parent sections expanded for hierarchical retrieval |
| `--rebuild-chunks` | off | Force re-chunk even if cache exists |
| `--rebuild-embeddings` | off | Force re-encode even if cache exists |
| `--quiet` | off | Suppress progress logs, print only final summary |

### Coverage metrics

The RAG pipeline scores retrieval before any LLM call:

```text
gold_sentence_coverage_at_10 =
  count(golden sentence IDs found in top-10 retrieved chunks)
  / count(golden sentence IDs)

all_gold_covered_rate_at_10 =
  count(rows where every golden sentence ID appears in top-10 retrieved chunks)
  / count(sentence-eligible rows)
```

Current all-techniques comparison on the 50-contract eval set is produced by:

```bash
uv run python rag_eval.py \
  --run-id s7-all-rag-techniques-eval50 \
  --sample-size 50 \
  --seed 42 \
  --retrievers bm25_sentence,dense_sentence,bm25_legal_recursive,dense_legal_recursive,bm25_hierarchical,dense_hierarchical \
  --chunking-version legal-recursive-v1 \
  --hierarchical-leaf-k 50 \
  --hierarchical-top-sections 5 \
  --top-k 30 \
  --contract-scope eval-set \
  --query-enrichment-provider offline
```

Useful debug runs:

```bash
# Faster debug — only chunk the 50 eval contracts
uv run python rag_eval.py --contract-scope eval-set --run-id rag-debug

# BM25 only (skips dense index build)
uv run python rag_eval.py --retrievers bm25_sentence

# Force full rebuild
uv run python rag_eval.py --rebuild-chunks --rebuild-embeddings
```

### Key flags for `agent.py` full eval

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `deepseek/deepseek-v4-flash` | LLM model string |
| `--model-id` | slug from `--model` | Stable run identifier; used in output paths and HTML |
| `--all-context-modes` | off | Run all supported context modes and print a comparison table |
| `--prompts-file` | auto from `model_id` | Path to system prompts module |
| `--context-mode` | `raw` | `raw`, `rag-dense`, `rag-hybrid`, `rag-hierarchical-bm25`, or `rag-hierarchical-dense` |
| `--top-k` | `30` | Chunks retrieved per question in RAG modes |
| `--hierarchical-leaf-k` | `50` | Leaf sentences retrieved before hierarchical section expansion |
| `--hierarchical-top-sections` | `5` | Parent sections expanded for hierarchical context modes |
| `--temperature` | `0` | LLM temperature |
| `--max-tokens` | `64000` | Max tokens per LLM response |
| `--num-threads` | `4` | Concurrent agent calls |
| `--sample-size` | `50` | Number of contracts to evaluate (50 = standard eval set, max 510) |
| `--seed` | `42` | Deterministic contract selection seed — must match across runs to compare results |
| `--resume-existing` | off | Resume from cached partial results |
| `--dry-run` | off | Skip LLM calls, echo gold answers (pipeline check) |

---

## Single-Question Spot Check

Before running a full 2,050-call eval, test one contract × one question across
all 10 variants (raw/enriched × raw/rag-dense/rag-hybrid/hierarchical-bm25/hierarchical-dense):

```bash
# Dry-run — no LLM, no RAG cache required
uv run python agent.py \
  --single-q --contract-id 327 --question-index 7 \
  --compare-variants --dry-run --model-id spot-check

# Live run — all 10 variants
uv run python agent.py \
  --single-q --contract-id 327 --question-index 7 \
  --compare-variants \
  --model deepseek/deepseek-v4-flash \
  --model-id spot-check
```

Question index 7 = "Governing Law", contract 327 is the first in the seed=42
eval set.

Example output:

```text
Variant                    |  Token F1  |  Correct@0.5  | Predicted (first 80 chars)
---------------------------|------------|---------------|----------------------------
raw_q / raw_ctx            |    0.42    |     False     | This Agreement shall be go...
raw_q / rag_dense          |    0.61    |     True      | governed by the laws of th...
raw_q / rag_hybrid         |    0.58    |     True      | governed by the laws of th...
raw_q / rag_hier_bm25      |    0.59    |     True      | governed by the laws of th...
raw_q / rag_hier_dense     |    0.62    |     True      | governed by the laws of th...
enriched_v1 / raw_ctx      |    0.44    |     False     | This Agreement shall be go...
enriched_v1 / rag_dense    |    0.74    |     True      | governed by the laws of th...
enriched_v1 / rag_hybrid   |    0.69    |     True      | governed by the laws of th...
enriched_v1 / rag_hier_bm25 |   0.70    |     True      | governed by the laws of th...
enriched_v1 / rag_hier_dense |  0.75    |     True      | governed by the laws of th...

Golden answer: governed by the laws of the State of Delaware
Category: Governing Law  |  Contract: 327  |  Question index: 7
```

### Question index reference

| Index | Category | Index | Category |
|-------|----------|-------|----------|
| 0 | Document Name | 21 | License Grant |
| 1 | Parties | 22 | IP Ownership Assignment |
| 2 | Agreement Date | 23 | Joint IP Ownership |
| 3 | Effective Date | 24 | Liquidated Damages |
| 4 | Expiration Date | 25 | Warranty Duration |
| 5 | Renewal Term | 26 | Insurance |
| 6 | Notice Period to Terminate Renewal | 27 | Covenant Not to Sue |
| 7 | Governing Law | 28 | Third Party Beneficiary |
| 8 | Most Favored Nation | 29 | Post-Agreement Term |
| 9 | Non-Compete | 30 | Audit Rights |
| 10 | Exclusivity | 31 | Uncapped Liability |
| 11 | No-Solicit of Customers | 32 | Cap on Liability |
| 12 | Competitive Restriction Exception | 33 | Liquidated Damages |
| 13 | No-Solicit of Employees | 34 | Unilateral Termination |
| 14 | Non-Disparagement | 35 | Termination for Insolvency |
| 15 | Termination for Convenience | 36 | Change of Control |
| 16 | Rofr/Rofo/Rofn | 37 | Anti-Assignment |
| 17 | Change of Control | 38 | Revenue/Profit Sharing |
| 18 | Anti-Assignment | 39 | Price Restrictions |
| 19 | Revenue/Profit Sharing | 40 | Minimum Commitment |
| 20 | Price Restrictions | | |

---

## Output Files Reference

| Path | Contents |
|------|----------|
| `frontend/rag_pipeline_eval.html` | RAG pipeline dashboard (chunking, coverage, retrieval) |
| `frontend/evaluation_{model_id}.html` | Agent evaluation dashboard (accuracy by question and contract) |
| `outputs/rag_cache/` | Persistent sentence chunks, BM25 indexes, dense encodings |
| `outputs/{run_id}/rag/` | Per-run RAG retrieval results, coverage CSV, summary JSON |
| `outputs/{model_id}/cuad_dspy_eval_results.csv` | Per-row predictions and scores |
| `outputs/{model_id}/cuad_dspy_eval_summary.json` | Overall + per-category mean F1 and correct@0.5 |
| `outputs/{model_id}/cuad_dspy_eval_results.jsonl` | Incremental results (safe to interrupt and resume) |

---

## Prompt Versions

| Version | File | How created |
|---------|------|-------------|
| `v2` | `prompts/system_prompts_v2.py` | LLM-optimised from v1 failures via `prompt_improve_v2.py` |
| `v1` | `prompts/system_prompts_v1.py` | Baseline prompts generated from CUAD questions |

Pass `--prompts-file prompts/system_prompts_v2.py` explicitly, or let the
runner auto-resolve by `model_id` convention (`model_id=v2` → `system_prompts_v2.py`).

---

## Data

Required local files:

```text
data/CUADv1.json
data/category_descriptions.csv
data/train_separate_questions.json
data/test.json
```

The deterministic 50-contract eval set for `seed=42`:

```text
[327, 57, 12, 379, 140, 125, 114, 71, 377, 52, 346, 456, 279, 44, 302, 216,
 16, 15, 47, 111, 119, 258, 308, 13, 287, 101, 366, 332, 359, 214, 112, 229,
 301, 142, 414, 445, 3, 388, 412, 81, 357, 174, 79, 110, 490, 390, 172, 194,
 49, 183]
```

---

## Evaluation Run Log

| Model ID | Context | Mean F1 | Correct@0.5 | Notes |
|----------|---------|---------|-------------|-------|
| `v1` | raw | 40.03% | 39.85% | DSPy baseline |
| `v2` | raw | 72.8% | 72.9% | Optimised prompts |

---

## Validation

```bash
uv run python -m py_compile explore.py dspy_eval_v1.py langchain_agent.py agent.py
uv run python -m py_compile prompt_improve_v2.py rag_eval.py
uv run python -m py_compile $(rg --files src -g '*.py')
uv run python -m pytest -q
```

## Project Structure

```text
src/cuad_agent/
├── data/                  # CUAD loading, validation, sampling
├── eval/                  # metrics, examples, summaries
├── evaluators/            # DSPy and LangChain runner implementations
├── prompts/               # prompt loading, templates, serialization
├── prompt_optimization/   # prompt-improvement harness
├── dashboards/            # static evaluation HTML renderers
├── rag/                   # sentence chunking, retrieval, coverage
│   ├── cache.py           # shared cache I/O (slugify, load_or_build_*)
│   ├── context_builder.py # build_rag_context() → context string for agent
│   ├── coverage.py        # coverage_at_k, coverage_by_top_chunks
│   ├── experiments.py     # run_rag_eval() orchestrator
│   └── query_enrichment.py # build_question_enrichments(), hybrid_fuse_results()
└── cli/                   # cuad-* console script entrypoints
```

Console commands (equivalent to the root scripts):

```bash
uv run cuad-explore
uv run cuad-agent            # agent.py / langchain_agent.py
uv run cuad-eval-langchain   # compatibility alias
uv run cuad-eval-rag         # rag_eval.py
uv run cuad-compare-runs     # cross-run summary table
uv run cuad-eval-dspy        # dspy_eval_v1.py
uv run cuad-improve-prompts  # prompt_improve_v2.py
```
