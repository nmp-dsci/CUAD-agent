# CUAD-Agent — Agent Guide

Experiment in building and evaluating a legal AI agent on the [CUAD](https://www.atticusprojectai.org/cuad) contract-understanding dataset. The pipeline covers dataset exploration, DSPy-based evaluation of 41 legal-review questions across 510 contracts, and an LLM-driven prompt-improvement harness.

---

## Project layout

```
CUAD-agent/
├── src/
│   └── cuad_agent/
│       ├── data/              # Dataset loader, validator, sampling facades
│       ├── eval/              # Metrics, examples, summaries, cache, comparisons
│       ├── evaluators/        # DSPy and LangChain runner implementations
│       ├── prompts/           # Prompt loading, templates, serialization
│       ├── prompt_optimization/ # PydanticAI prompt-improvement harness
│       ├── dashboards/        # Static evaluation/prompt-review HTML renderers
│       ├── rag/               # Sentence-level RAG pipeline (chunking, indexing, retrieval, evaluation)
│       └── cli/               # Package console command entrypoints
├── explore.py                 # Compatibility wrapper
├── dspy_eval_v1.py            # Compatibility wrapper
├── agent.py                   # Primary LangChain agent evaluator wrapper
├── langchain_agent.py         # Compatibility wrapper
├── prompt_improve_v2.py       # Compatibility wrapper
├── rag_eval.py                # Compatibility wrapper → cuad_agent.rag.cli:main
├── prompts/
│   ├── system_prompts_v1.py   # Baseline per-category system prompts
│   └── system_prompts_v2.py   # Optimized per-category system prompts
├── data/
│   ├── CUADv1.json            # Full dataset (510 contracts × 41 questions)
│   ├── category_descriptions.csv
│   ├── train_separate_questions.json
│   └── test.json
├── outputs/                   # Evaluation run artefacts (gitignored)
│   └── {model_id}/
│       ├── cuad_dspy_eval_results.csv
│       ├── cuad_dspy_eval_summary.json
│       ├── system_prompts.py
│       └── prompt_harness/    # v2 harness artefacts
├── frontend/
│   ├── explore.html
│   ├── evaluation_{model_id}.html
│   └── prompt_review_{model_id}.html
├── readme.md
├── planning/                  # PRP planning documents
├── tests/
│   └── test_cuad_dspy_eval.py
├── pyproject.toml
└── uv.lock
```

---

## Architecture

### Dataset (`src/cuad_agent/data/dataset.py`)

`load_datasets()` reads `CUADv1.json` and `category_descriptions.csv` and returns two dataframes:
- `contracts` — one row per contract (510 total), columns include `document_row_id`, `title`, `context`
- `questions` — one row per (contract, question) pair (20,910 total), columns include `question_index` (0–40), `category`, `question`, `category_description`, `answer_format`, `answers`, `is_impossible`

The 41 CUAD categories are consistent across all contracts and are identified by `question_index`.

The root `explore.py` file is a compatibility wrapper around this package module.

### DSPy evaluation (`src/cuad_agent/evaluators/dspy_runner.py`)

One `dspy.Signature` and one `dspy.Module` (subclass of `ContractQuestionAgentBase`) are created per category. Each signature's docstring is set to either the raw question text or a custom system prompt loaded from a prompts file. At inference time each agent calls `dspy.ChainOfThought` with the signature.

**Key objects:**
| Symbol | Description |
|--------|-------------|
| `make_question_signature_class` | Dynamically creates a typed `dspy.Signature` with `contract_title`, `contract_text`, `category`, `category_description`, `answer_format` inputs and `answer`, `marked_impossible` outputs |
| `ContractQuestionAgentBase` | Base DSPy module; `dry_run=True` mode echoes gold answers without LLM calls |
| `build_eval_sample` | Deterministic 50-contract sample using `random.Random(seed)` |
| `build_agents` | Instantiates 41 agents from the question dataframe, applying any prompt overrides |
| `cuad_overlap_metric` | SQuAD-style token overlap F1 between prediction and gold spans |
| `build_devset` | Converts eval rows into `dspy.Example` objects |
| `write_system_prompts` | Serialises per-category prompts to a `.py` file as `CATEGORY_SYSTEM_PROMPTS` dict |

**LLM routing:** DeepSeek models via `deepseek/` prefix → `DEEPSEEK_API_KEY`. OpenAI models via `openai/` prefix → `OPENAI_API_KEY`. DSPy cache stored in `.dspy_cache/`.

The root `dspy_eval_v1.py` file is a compatibility wrapper around this package module.

### LangChain evaluation (`src/cuad_agent/evaluators/langchain_runner.py`)

LangChain evaluation is the v2 runner. It shares framework-neutral paths, metrics, prompt loading, summaries, and dashboard rendering through `src/cuad_agent/eval/`, `src/cuad_agent/prompts/`, `src/cuad_agent/paths.py`, and `src/cuad_agent/dashboards/`.

The root `agent.py` file is the primary wrapper for this evaluator. The root
`langchain_agent.py` file remains as a compatibility wrapper.

### RAG pipeline (`src/cuad_agent/rag/`)

Sentence-level retrieval-augmented generation evaluation over the CUAD dataset. The pipeline measures how well different retrieval strategies can surface the gold-answer spans for each (contract, question) pair before any LLM call.

**Key modules:**

| Module | Description |
|--------|-------------|
| `sentences.py` | `SentenceSpan` dataclass; `build_sentence_spans()` splits contract text into sentence spans with section metadata |
| `chunks.py` | `RagChunk` dataclass wrapping one or more sentence spans; `chunks_from_sentences()` |
| `legal_recursive.py` | `LegalRecursiveConfig`; `build_legal_recursive_chunks()` uses LangChain `RecursiveCharacterTextSplitter` to produce an alternative chunking strategy |
| `hierarchy.py` | `SectionNode`; `build_section_index()`; `HierarchicalRetriever` for leaf → section → sentence retrieval |
| `retrievers.py` | `SentenceRetriever`; `build_retriever(method, chunks)` dispatches to BM25 or dense (TF-IDF/SentenceTransformer) |
| `indexes.py` | `DenseSentenceIndex` — encodes chunks, writes embedding artefacts; `load_pickle` / `write_pickle` |
| `gold_answers.py` | `EligibilityRecord`; `evaluate_row_eligibility()` checks whether all gold spans align to at least one sentence |
| `coverage.py` | Canonical coverage metrics: `coverage_at_k`, `coverage_by_top_chunks`, `retrieved_sentence_ids_from_results` |
| `cache.py` | Shared cache I/O helpers: `slugify`, `sentence_cache_paths`, `load_or_build_retriever`, `load_or_build_dense_sentence_encoder`, `DEFAULT_EMBEDDING_MODEL` |
| `context_builder.py` | `build_rag_context()` — turns `(document_row_id, query, method)` into a context string for agent consumption |
| `query_enrichment.py` | `build_question_enrichments()` — optionally augments BM25/dense queries via DeepSeek LLM or deterministic offline terms; `RAG_DEFAULT_TOP_K = 30` |
| `clauses.py` | `build_section_metadata()` — extracts section numbers and titles for sentence span metadata |
| `contracts.py` | `ContractDocument`; `contracts_from_lookup()` |
| `experiments.py` | `run_rag_eval()` — top-level orchestrator; caches sentence spans, chunk indexes, and dense embeddings under `outputs/rag_cache/` |
| `outputs.py` | `rag_output_paths()` and all file writers (CSV, JSONL, JSON, HTML) |
| `cli.py` | `main()` — CLI entry point consumed by `rag_eval.py` and `cuad-eval-rag` |

**Retriever methods:**

| Method | Chunking |
|--------|----------|
| `bm25_sentence` | Sentence-level (default `sentence-v3`) |
| `dense_sentence` | Sentence-level (default `sentence-v3`); TF-IDF or SentenceTransformer |
| `bm25_hierarchical` | Sentence-level leaf search, section expansion, BM25 re-rank |
| `dense_hierarchical` | Sentence-level leaf search, section expansion, dense re-rank |
| `bm25_legal_recursive` | LangChain RecursiveCharacterTextSplitter (`legal-recursive-v1`) |
| `dense_legal_recursive` | LangChain RecursiveCharacterTextSplitter (`legal-recursive-v1`) |

Hierarchical retrieval reuses sentence chunks and their `clause_path` metadata.
It first retrieves leaf sentences, scores parent sections, expands the top sections,
then re-ranks only those expanded sentences. Tune with `--hierarchical-leaf-k`
and `--hierarchical-top-sections`.

**Caching:** All intermediate artefacts (sentence spans, dense embeddings, BM25/dense indexes) are written to `outputs/rag_cache/` keyed by chunking version and embedding model. Use `--rebuild-chunks` or `--rebuild-embeddings` to force regeneration.

---

### Prompt versions (`prompts/`)

Two prompt versions are tested by the CUAD agents:

| Model ID | Prompt file |
|----------|-------------|
| `v1` | `prompts/system_prompts_v1.py` |
| `v2` | `prompts/system_prompts_v2.py` |

When `--prompts-file` is omitted, evaluators load `prompts/system_prompts_{model_id}.py` if that file exists. Use `--prompts-file` only to test an alternate prompt module.

### Prompt-improvement harness (`src/cuad_agent/prompt_optimization/harness.py`)

Reads a v1 results CSV, splits rows into `generator_dev` / `evaluator_dev` / `holdout_eval`, then loops over failing categories:

1. **Generator agent** (PydanticAI + DeepSeek) proposes a prompt patch from `generator_dev` failures.
2. **Evaluator agent** (PydanticAI + DeepSeek) reviews the patch against `evaluator_dev` failures and accepts or rejects it.
3. Accepted patches are written to `prompts_candidate_v2.py`; diffs and review logs are preserved.

Requires `DEEPSEEK_API_KEY` by default. Pass `--dry-run` to skip LLM calls and use the deterministic offline mode (tests and sanity checks).

The root `prompt_improve_v2.py` file is a compatibility wrapper around this package module.

---

## Common commands

```bash
# Install dependencies
uv sync

# Explore the dataset
uv run python explore.py
EXPLORE_ADHOC=1 uv run python explore.py
uv run cuad-explore

# Dry-run evaluation (no LLM, uses gold answers)
uv run python dspy_eval_v1.py --dry-run --sample-size 50 --seed 42 --model-id v1
uv run cuad-eval-dspy --dry-run --sample-size 50 --seed 42 --model-id v1

# Full LLM evaluation (DeepSeek)
DEEPSEEK_API_KEY=... uv run python dspy_eval_v1.py \
  --sample-size 50 --seed 42 \
  --model deepseek/deepseek-v4-flash \
  --model-id v1 --temperature 0 --max-tokens 2400 --num-threads 4

# Re-evaluate with custom prompts
uv run python dspy_eval_v1.py \
  --model-id v2

# LangChain v2 evaluation
uv run python agent.py
uv run cuad-eval-langchain
uv run cuad-agent

# Evaluate a holdout split only
uv run python dspy_eval_v1.py \
  --model-id v2 \
  --eval-split outputs/v2/prompt_harness/splits.json:holdout_eval

# Run the v2 prompt-improvement harness (LLM by default)
DEEPSEEK_API_KEY=... uv run python prompt_improve_v2.py \
  --source-results outputs/v1/cuad_dspy_eval_results.csv \
  --prompts-file prompts/system_prompts_v1.py \
  --model-id v2

# Run the v2 harness in deterministic offline mode (no LLM)
uv run python prompt_improve_v2.py \
  --source-results outputs/v1/cuad_dspy_eval_results.csv \
  --prompts-file prompts/system_prompts_v1.py \
  --model-id v2 --dry-run
uv run cuad-improve-prompts \
  --source-results outputs/v1/cuad_dspy_eval_results.csv \
  --prompts-file prompts/system_prompts_v1.py \
  --model-id v2 --dry-run

# RAG evaluation — sentence-level retrieval (no LLM required by default)
uv run python rag_eval.py --sample-size 50 --seed 42 --run-id rag-sentence-v1
uv run cuad-eval-rag --sample-size 50 --seed 42 --run-id rag-sentence-v1

# RAG — golden-answer sentence coverage preflight only (skips retrieval)
uv run python rag_eval.py --preflight-golden-sentences-only --run-id rag-preflight

# RAG — compare legal-recursive chunking against sentence-v3
uv run python rag_eval.py \
  --retrievers bm25_legal_recursive,dense_legal_recursive \
  --chunking-version legal-recursive-v1 \
  --run-id rag-legal-recursive-v1

# RAG — compare flat sentence retrieval with hierarchical section expansion
uv run python rag_eval.py \
  --retrievers bm25_sentence,dense_sentence,bm25_hierarchical,dense_hierarchical \
  --hierarchical-leaf-k 50 \
  --hierarchical-top-sections 5 \
  --run-id rag-hierarchical-v1

# RAG — LLM query enrichment (requires DEEPSEEK_API_KEY)
DEEPSEEK_API_KEY=... uv run python rag_eval.py \
  --query-enrichment-provider llm \
  --run-id rag-enriched-v1

# RAG — chunk only evaluation-set contracts (faster, less RAM)
uv run python rag_eval.py --contract-scope eval-set --run-id rag-eval-set-v1

# RAG — force rebuild of cached chunks and embeddings
uv run python rag_eval.py --rebuild-chunks --rebuild-embeddings --run-id rag-rebuild

# Full eval with RAG context (requires prebuilt rag_cache from cuad-eval-rag)
uv run cuad-eval-dspy \
  --model-id v2-rag-dense \
  --context-mode rag-dense \
  --top-k 30 \
  --chunking-version sentence-v3

uv run cuad-eval-dspy \
  --model-id v2-rag-hybrid \
  --context-mode rag-hybrid \
  --top-k 30

uv run python agent.py \
  --model-id v2-rag-hierarchical-dense \
  --context-mode rag-hierarchical-dense \
  --hierarchical-leaf-k 50 \
  --hierarchical-top-sections 5 \
  --top-k 30

# Compare runs side-by-side
uv run cuad-compare-runs outputs/v1 outputs/v2 --label v1-raw v2-optimised
uv run cuad-compare-runs outputs/v1 outputs/v2-rag-dense --label v1-raw v2-rag-dense

# Validate and run tests
uv run python -m py_compile explore.py dspy_eval_v1.py langchain_agent.py agent.py
uv run python -m py_compile prompt_improve_v2.py rag_eval.py
uv run python -m py_compile $(rg --files src -g '*.py')
uv run pytest -q
```

---

## Environment variables

| Variable | Required for |
|----------|-------------|
| `DEEPSEEK_API_KEY` | LLM evaluation with `deepseek/*` models; v2 harness (default mode, skip with `--dry-run`) |
| `OPENAI_API_KEY` | LLM evaluation with `openai/*` models |

Store in a `.env` file at the project root (never commit). `dspy_eval_v1.py` auto-loads `~/.env` via `python-dotenv`.

---

## Evaluation outputs

Each run writes to `outputs/{model_id}/`:

| File | Contents |
|------|----------|
| `cuad_dspy_eval_results.csv` | Per-(contract, question) scores: `token_f1`, `correct_at_0_5`, predictions, gold answers |
| `cuad_dspy_eval_summary.json` | Aggregate metrics, run config, per-category breakdown |
| `system_prompts.py` | Generated `CATEGORY_SYSTEM_PROMPTS` dict — edit and re-run to iterate |

Evaluation HTML dashboards write to `frontend/evaluation_{model_id}.html`.

The RAG evaluation writes to `outputs/{run_id}/rag/`:

| File | Contents |
|------|----------|
| `golden_sentence_coverage.csv` | Per-(contract, question) sentence alignment results |
| `golden_sentence_coverage_summary.json` | Aggregate eligibility metrics |
| `golden_sentence_question_summary.csv` | Per-question RAG-suitability classification |
| `rag_sentences.jsonl` | Sentence chunks used in the run |
| `rag_retrieval_results.jsonl` / `.csv` | Per-(retriever, contract, question) retrieval results with coverage@k |
| `rag_retrieval_doc_question_summary.csv` | Best-retriever summary per (contract, question) |
| `rag_ranking_summary.csv` | Mean gold-sentence coverage@k per retriever |
| `rag_query_enrichment_results.csv` | Enriched-query vs baseline retrieval comparison |
| `rag_query_enrichment_summary.csv` | Aggregate enrichment lift metrics |
| `rag_summary.json` | Full run summary including config and all aggregate metrics |
| `rag_config.json` | Exact run configuration snapshot |

RAG pipeline HTML dashboard writes to `frontend/rag_pipeline_eval.html`; when
hierarchical retrievers are included, it adds a "Hierarchical RAG Performance"
tab with flat-vs-hierarchical coverage deltas and parameter settings.

Intermediate artefacts are cached under `outputs/rag_cache/`:

| Path | Contents |
|------|----------|
| `rag_cache/chunking/{version}/sentence_spans.jsonl` | Sentence-level spans per chunking version |
| `rag_cache/chunking/{version}/contracts_manifest.json` | Contract text hashes for cache invalidation |
| `rag_cache/chunking/{version}/encodings/{model}/` | Dense embedding artefacts |
| `rag_cache/sparse/{version}/{method}/bm25_index.pkl` | Serialised BM25 index |

The v2 harness writes to `outputs/{model_id}/prompt_harness/`:

| File | Contents |
|------|----------|
| `splits.json` | `generator_dev`, `evaluator_dev`, `holdout_eval` row id lists |
| `answer_format_profiles.json` | Per-category answer-format statistics |
| `category_runs.jsonl` | Per-category generator/evaluator run logs |
| `evaluator_reviews.jsonl` | Detailed evaluator reasoning |
| `accepted_patches.jsonl` | Patches accepted by the evaluator |
| `rejected_patches.jsonl` | Patches rejected by the evaluator |
| `prompt_diffs.jsonl` | Unified diffs of accepted changes |
| `prompts_candidate_v2.py` | Candidate prompt module — review before promoting |

Prompt review HTML dashboards write to `frontend/prompt_review_{model_id}.html`.

---

## Evaluation run log

| Model ID | Model | Sample | Seed | Max tokens | Threads | Mean token F1 | Correct at 0.5 |
|----------|-------|-------:|-----:|-----------:|--------:|--------------:|---------------:|
| `v1` | `deepseek/deepseek-v4-flash` | 50 | 42 | 64000 | 4 | 40.03% | 39.85% |

Deterministic 50-contract sample for `seed=42`:
```
[327, 57, 12, 379, 140, 125, 114, 71, 377, 52, 346, 456, 279, 44, 302, 216,
 16, 15, 47, 111, 119, 258, 308, 13, 287, 101, 366, 332, 359, 214, 112, 229,
 301, 142, 414, 445, 3, 388, 412, 81, 357, 174, 79, 110, 490, 390, 172, 194,
 49, 183]
```

---

## Testing

| Test file | Covers |
|-----------|--------|
| `tests/test_cuad_dspy_eval.py` | Token overlap F1, `parse_bool`, deterministic sampling, agent construction, prompt overrides, eval split filtering, dry-run shape, HTML output, v2 harness artefacts |
| `tests/test_rag_sentences.py` | `SentenceSpan` construction, `build_sentence_spans()`, normalisation |
| `tests/test_rag_outputs.py` | `rag_output_paths()`, file writers (CSV, JSONL, JSON) |
| `tests/test_rag_retrieval.py` | BM25 and dense retriever round-trips, `build_retriever()` dispatch, coverage metrics |
| `tests/test_rag_hierarchical.py` | Section indexing, hierarchical expansion/re-ranking, document filtering, context formatting |
| `tests/test_rag_coverage.py` | `evaluate_row_eligibility()`, eligibility summarisation, RAG-suitability classification |
| `tests/test_rag_legal_recursive.py` | `build_legal_recursive_chunks()`, `LegalRecursiveConfig`, chunk-to-sentence mapping |

Run all tests with:
```bash
uv run pytest -q
```

No API keys are required — all tests use dry-run mode or in-memory fixtures.

---

## Single-question variant comparison (S6)

`agent.py --single-q` runs one legal-clause question for one contract across all combinations of question enrichment × retrieval context and prints a side-by-side accuracy table.

**Prerequisite (one-time, ~5 min first run):**

```bash
uv run python rag_eval.py \
  --preflight-golden-sentences-only \
  --contract-scope all \
  --run-id s6-preflight
```

This builds the sentence/embedding cache used by RAG context modes. Subsequent runs are fast cache hits.

**Key question_index values** (CUAD dataset, same across all contracts):

| Index | Category |
|-------|----------|
| 0 | Document Name |
| 7 | Governing Law |
| 18 | Anti-Assignment |

Run `uv run python -c "from cuad_agent.data.sampling import select_evaluation_set; sel = select_evaluation_set(contract_ids=[327]); print(sel.eval_rows[['question_index','category']].drop_duplicates().sort_values('question_index').to_string())"` to see all 41 categories.

**Example commands:**

```bash
# Dry-run: verify harness wiring without LLM or RAG (fast)
uv run python agent.py \
  --single-q --contract-id 327 --question-index 7 \
  --compare-variants \
  --dry-run --model-id s6-test

# Live: all 10 variants for one question (requires DEEPSEEK_API_KEY)
DEEPSEEK_API_KEY=... uv run python agent.py \
  --single-q --contract-id 327 --question-index 7 \
  --compare-variants \
  --query-enrichment-provider auto \
  --model deepseek/deepseek-v4-flash \
  --model-id s6-live

# Live: single variant — enriched question, hybrid RAG context
DEEPSEEK_API_KEY=... uv run python agent.py \
  --single-q --contract-id 327 --question-index 7 \
  --question-mode enriched --context-mode rag-hybrid \
  --model deepseek/deepseek-v4-flash \
  --model-id s6-hybrid

# Live: single variant — enriched question, hierarchical dense context
DEEPSEEK_API_KEY=... uv run python agent.py \
  --single-q --contract-id 327 --question-index 7 \
  --question-mode enriched --context-mode rag-hierarchical-dense \
  --hierarchical-leaf-k 50 --hierarchical-top-sections 5 \
  --model deepseek/deepseek-v4-flash \
  --model-id s7-hier-dense
```

**Output folders:**

| Path | Contents |
|------|----------|
| `outputs/{model_id}/single_q_variants/c{id}_q{idx:02d}_variants.csv` | Per-variant results CSV |
| `outputs/enriched_questions/{provider}/q{idx:02d}_{category-slug}.json` | Human-readable enrichment files for review/editing |

The JSONL at `outputs/rag_cache/query_enrichment/{provider}/enriched_questions.jsonl` remains the primary enrichment cache; the per-question JSON files are for human review only.

**Variant naming convention:**

| Variant name | Question | Context |
|---|---|---|
| `raw_q / raw_ctx` | Raw CUAD question | Full contract transcript |
| `raw_q / rag_dense` | Raw CUAD question | Dense-vector top-30 chunks |
| `raw_q / rag_hybrid` | Raw CUAD question | RRF-fused BM25+dense top-30 chunks |
| `raw_q / rag_hier_bm25` | Raw CUAD question | BM25 leaf search with top-section expansion |
| `raw_q / rag_hier_dense` | Raw CUAD question | Dense leaf search with top-section expansion |
| `enriched_v1 / raw_ctx` | Enriched query (terms appended to user message) | Full contract transcript |
| `enriched_v1 / rag_dense` | Enriched query used for retrieval | Dense-vector top-30 chunks |
| `enriched_v1 / rag_hybrid` | Enriched query used for retrieval | RRF-fused BM25+dense top-30 chunks |
| `enriched_v1 / rag_hier_bm25` | Enriched query used for retrieval | BM25 leaf search with top-section expansion |
| `enriched_v1 / rag_hier_dense` | Enriched query used for retrieval | Dense leaf search with top-section expansion |

The `enriched_v1` label reserves room for `enriched_v2_prompt` (where enrichment terms are baked into the system prompt itself) in a future sprint.

---

## Planning documents

Implementation was designed through PRP files in `planning/`:

| File | Covers |
|------|--------|
| `planning/generate-prp.md` | PRP generation process |
| `planning/execute-prp.md` | PRP execution process |
| `planning/s1-prp-dspy-eval.md` | DSPy evaluation runner design |
| `planning/s2-prp-dspy-rag.md` | RAG extension (initial design) |
| `planning/s3-v2-prompt-opt.md` | V2 prompt optimisation design |
| `planning/s4-project-refactor.md` | Package refactor into `src/cuad_agent/` |
| `planning/s5-rag.md` | Sentence-level RAG pipeline design |
| `planning/s6-rag-agent.md` | Single-question RAG agent variant comparison |
| `planning/s7-rag-hierarchical.md` | Hierarchical leaf → section → sentence retrieval |
