# S6 — Single-Question RAG Agent with Variant Comparison

## Goal

Add a `--single-q` mode to `langchain_agent.py` that answers one legal clause question for one contract across every combination of question type and retrieval context, then prints a side-by-side accuracy table.

**Definition of success:** one command produces one table showing token F1 and correct@0.5 for all variants vs the golden answer.

---

## Fixed baseline: `system_prompts_v2.py`

All variants in this experiment use the **v2 system prompt** loaded from
`prompts/system_prompts_v2.py` via `load_prompt_overrides("prompts/system_prompts_v2.py")`.
The experiment axis is *question enrichment × context retrieval*, not prompt version.
`compose_system_prompt()` (the v1 default) is NOT used here.

Future sprint: once enrichment terms are baked into the system prompt itself, that
becomes `enriched_v2_prompt` — a third question-mode variant added to this table.
That is the "continue to build in prompts" direction, but it is out of scope for S6.

---

## Variants

| Variant name | Question / retrieval query | User message addition | Contract context to LLM |
|---|---|---|---|
| `raw_q / raw_ctx` | Raw CUAD question | — | Full contract transcript |
| `raw_q / rag_dense` | Raw CUAD question | — | Dense-vector top-30 chunks |
| `raw_q / rag_hybrid` | Raw CUAD question | — | RRF-fused BM25+dense top-30 chunks |
| `enriched_v1 / raw_ctx` | Enriched query terms appended | `\n\nKey terms: {enrichment_terms}` | Full contract transcript |
| `enriched_v1 / rag_dense` | Enriched query for retrieval | — | Dense-vector top-30 chunks |
| `enriched_v1 / rag_hybrid` | Enriched query for retrieval | — | RRF-fused BM25+dense top-30 chunks |

**Where the enriched question appears in the LLM call:**

- `enriched_v1 / rag_*`: the enriched query is the RAG search string only. The user
  message is unchanged; the retrieved chunks replace the full transcript as `contract_text`.
- `enriched_v1 / raw_ctx`: the full transcript is passed as `contract_text`. The enriched
  terms are appended to the user message as: `\n\nKey terms to look for: {enrichment_terms}`
  This gives the LLM a hint without changing the system prompt or context.

The system prompt (v2) is identical across all six variants.

**CLI flags:**

- `--question-mode raw | enriched` (default `raw`)
- `--context-mode raw | rag-dense | rag-hybrid` (default `raw`)
- `--compare-variants`: runs all 6 combinations; overrides `--question-mode` and `--context-mode`

---

## Shared State Between `rag_eval.py` and `langchain_agent.py`

### Duplicated code — must consolidate

`query_for_row()` is defined identically in two places:

- `src/cuad_agent/rag/query_enrichment.py:63`
- `src/cuad_agent/rag/experiments.py:434`

**Fix:** canonical copy stays in `query_enrichment.py`. Remove from `experiments.py`,
add import there.

### Enrichment cache — already shared, extend for human readability

The JSONL at `outputs/rag_cache/query_enrichment/{provider}/enriched_questions.jsonl`
is already the shared cache for both entry points. No change needed there.

**Add:** `save_enriched_question_files()` in `query_enrichment.py` writes one JSON file
per question to `outputs/enriched_questions/{provider}/q{idx:02d}_{category_slug}.json`
after each enrichment run. These are for human review and offline editing — they are
NOT the primary cache (the JSONL is).

File naming: `q01_anti-assignment.json`, `q05_governing-law.json`, etc.
Each file contains: `question_index`, `category`, `question`, `enrichment_terms`,
`enriched_query`, `provider`, `cache_key`.

### Default constants — no new module

Do not create `shared.py` for three constants. Define them where they are most
naturally owned and import:

```python
# In src/cuad_agent/rag/query_enrichment.py (already the right home)
RAG_DEFAULT_TOP_K: int = 30
```

`experiments.py` and `langchain_runner.py` import `RAG_DEFAULT_TOP_K` from
`query_enrichment` when needed. `DEFAULT_EMBEDDING_MODEL` and
`DEFAULT_CHUNKING_VERSION` already exist in `experiments.py` — import from there.

---

## New Files

### `src/cuad_agent/rag/context_builder.py`

Turns `(document_row_id, query, method, top_k, output_dir, ...)` into a context
string for the LLM. Single responsibility, no new dependencies.

```python
def build_rag_context(
    *,
    document_row_id: int,
    query: str,
    method: Literal["rag-dense", "rag-hybrid"],
    top_k: int,
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
) -> tuple[str, list[str]]:
    """Return (context_text, retrieved_chunk_ids).

    Requires the sentence/chunk cache to exist under output_dir.
    Build it first with:  uv run python rag_eval.py --preflight-golden-sentences-only

    For rag-hybrid: runs both dense and BM25 retrievers, fuses with
    hybrid_fuse_results() (RRF), returns top_k fused results.
    Returns context_text = chunks joined by "\\n\\n---\\n\\n".
    """
```

Internally uses existing helpers imported from `experiments.py`:
`load_or_build_retriever`, `load_or_build_dense_sentence_encoder`,
`load_cached_sentence_spans_for_version`, `chunks_from_sentences`.
And from `query_enrichment.py`: `hybrid_fuse_results`.

**Cold cache note:** on a first run with no prior `rag_eval.py` output, building
retrievers for all 510 contracts can take several minutes. The example commands
below include a prerequisite preflight step. After that first build, all subsequent
single-question runs are fast cache hits.

**Dry-run note:** `--dry-run` skips the LLM call (echoes gold answer) but RAG
retrieval still runs against cached indexes. If the cache does not exist,
`--dry-run` with a RAG context mode will fail. Tests use `context-mode raw` to
avoid this dependency.

### `tests/test_single_q_variants.py`

Dry-run tests — no LLM, no embeddings, `--context-mode raw` only:

- All 6 variant names appear in the output DataFrame
- CSV columns match expected schema
- `save_enriched_question_files()` writes `q{idx:02d}_{slug}.json` files
- `print_variant_table()` does not crash on empty or single-row DataFrames
- `--compare-variants` produces 6 rows; explicit `--question-mode / --context-mode`
  produces 1 row

---

## Changes to Existing Files

### `src/cuad_agent/evaluators/langchain_runner.py`

#### New CLI arguments

```python
parser.add_argument("--single-q", action="store_true")
parser.add_argument("--contract-id", type=int, default=None)
parser.add_argument("--question-index", type=int, default=None)
parser.add_argument(
    "--question-mode",
    choices=("raw", "enriched"),
    default="raw",
)
parser.add_argument(
    "--context-mode",
    choices=("raw", "rag-dense", "rag-hybrid"),
    default="raw",
)
parser.add_argument("--compare-variants", action="store_true")
parser.add_argument("--top-k", type=int, default=30)
parser.add_argument(
    "--query-enrichment-provider",
    choices=("auto", "llm", "offline"),
    default="auto",
)
parser.add_argument("--query-enrichment-model", default="deepseek-chat")
parser.add_argument("--embedding-model", default="tfidf")
parser.add_argument("--chunking-version", default="sentence-v3")
```

#### Updated `make_messages()` (inside `build_chain_for_agent`)

The existing function is not changed. Instead a new helper is added:

```python
def make_messages_with_hint(inputs: dict[str, Any], hint: str) -> list[Any]:
    """Like make_messages but appends enrichment hint to the user message."""
    user_content = (
        f"Contract title:\n{inputs['contract_title']}\n\n"
        f"Contract text:\n{inputs['contract_text']}\n\n"
        f"Category:\n{inputs['category']}\n\n"
        f"Category description:\n{inputs['category_description']}\n\n"
        f"Answer format:\n{inputs['answer_format']}"
        f"\n\nKey terms to look for: {hint}"
    )
    return [SystemMessage(content=full_system), HumanMessage(content=user_content)]
```

Used only when `question_mode == "enriched"` and `context_mode == "raw"`.

#### New function: `run_single_question_variants()`

No `args` pass-through — all parameters explicit:

```python
def run_single_question_variants(
    *,
    contract_id: int,
    question_index: int,
    llm: BaseChatModel | None,
    dry_run: bool,
    question_modes: list[str],
    context_modes: list[str],
    top_k: int,
    output_dir: Path,
    model_id: str,
    prompt_overrides: dict[str, str],   # loaded from system_prompts_v2.py
    query_enrichment_provider: str,
    query_enrichment_model: str,
    embedding_model: str,
    chunking_version: str,
) -> pd.DataFrame:
```

Steps:

1. Load the single `(contract_id, question_index)` row from the dataset.
2. Load the contract text from `contract_lookup`.
3. Call `build_question_enrichments()` for this question row; write per-question
   JSON file via `save_enriched_question_files()`.
4. For each `(question_mode, context_mode)` pair:
   a. **Resolve retrieval query:**
      - `raw`: `category + " " + category_description + " " + question` (via `query_for_row()`)
      - `enriched`: `enrichment.enriched_query`
   b. **Resolve contract context:**
      - `raw`: full `contract_text`
      - `rag-dense` / `rag-hybrid`: call `build_rag_context(query=retrieval_query, method=context_mode, ...)`
   c. **Resolve user message hint:**
      - Only set when `question_mode == "enriched"` and `context_mode == "raw"`:
        `hint = enrichment.enrichment_terms`
      - Otherwise: `hint = ""`
   d. **Get system prompt** from `prompt_overrides[category]` (v2). No fallback to
      `compose_system_prompt()` — v2 must cover all 41 categories.
   e. Call LLM (or dry-run echo) with the resolved inputs.
   f. Score with `token_overlap_f1(pred.answer, gold_answers)`.
   g. Append result row: `variant_name, question_mode, context_mode, retrieval_query,
      hint_used, predicted_answer, gold_answers, token_f1, correct_at_0_5,
      enrichment_terms, document_row_id, question_index, category`.
5. Return DataFrame.

**Variant name construction:**
```python
q_label = "raw_q" if question_mode == "raw" else "enriched_v1"
ctx_label = {"raw": "raw_ctx", "rag-dense": "rag_dense", "rag-hybrid": "rag_hybrid"}[context_mode]
variant_name = f"{q_label} / {ctx_label}"
```

#### New function: `print_variant_table()`

```
Variant                    | Token F1 | Correct@0.5 | Predicted (first 80 chars)
---------------------------|----------|-------------|---------------------------
raw_q / raw_ctx            |   0.42   |    False    | This Agreement shall be go...
raw_q / rag_dense          |   0.61   |    True     | governed by the laws of th...
raw_q / rag_hybrid         |   0.58   |    True     | governed by the laws of th...
enriched_v1 / raw_ctx      |   0.44   |    False    | This Agreement shall be go...
enriched_v1 / rag_dense    |   0.74   |    True     | governed by the laws of th...
enriched_v1 / rag_hybrid   |   0.69   |    True     | governed by the laws of th...

Golden answer: governed by the laws of the State of Delaware
Category: Governing Law  |  Contract: 327  |  Question index: 5
```

#### Changes to `main()`

```python
def main() -> None:
    args = parse_args()

    if args.single_q:
        if args.contract_id is None or args.question_index is None:
            raise ValueError("--single-q requires --contract-id and --question-index")
        # v2 prompts are mandatory for this mode
        prompts_file = args.prompts_file or Path("prompts/system_prompts_v2.py")
        prompt_overrides = load_prompt_overrides(prompts_file)

        llm = None if args.dry_run else configure_llm(args)
        question_modes = ["raw", "enriched"] if args.compare_variants else [args.question_mode]
        context_modes = (
            ["raw", "rag-dense", "rag-hybrid"] if args.compare_variants else [args.context_mode]
        )
        results_df = run_single_question_variants(
            contract_id=args.contract_id,
            question_index=args.question_index,
            llm=llm,
            dry_run=args.dry_run,
            question_modes=question_modes,
            context_modes=context_modes,
            top_k=args.top_k,
            output_dir=args.output_dir,
            model_id=args.model_id or "s6",
            prompt_overrides=prompt_overrides,
            query_enrichment_provider=args.query_enrichment_provider,
            query_enrichment_model=args.query_enrichment_model,
            embedding_model=args.embedding_model,
            chunking_version=args.chunking_version,
        )
        print_variant_table(results_df)
        out_dir = args.output_dir / (args.model_id or "s6") / "single_q_variants"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"c{args.contract_id}_q{args.question_index:02d}_variants.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
        return
    # ... existing full-eval path unchanged ...
```

### `src/cuad_agent/rag/query_enrichment.py`

Add `RAG_DEFAULT_TOP_K = 30` at module level.

Add `save_enriched_question_files()`:

```python
def save_enriched_question_files(
    enrichments: dict[int, QuestionEnrichment],
    output_dir: Path,
    provider: str,
) -> None:
    base = output_dir / "enriched_questions" / slugify(provider)
    base.mkdir(parents=True, exist_ok=True)
    for question_index, enrichment in sorted(enrichments.items()):
        path = base / f"q{question_index:02d}_{slugify(enrichment.category)}.json"
        path.write_text(
            json.dumps(enrichment.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
```

### `src/cuad_agent/rag/experiments.py`

- Remove local `query_for_row()` (line 434–439).
- Add import: `from cuad_agent.rag.query_enrichment import query_for_row, RAG_DEFAULT_TOP_K`

---

## Output Folder Structure

```
outputs/
  enriched_questions/
    {provider}/
      q01_anti-assignment.json      # human-readable, labeled _q{idx}
      q05_governing-law.json
      ...
  rag_cache/
    query_enrichment/
      {provider}/
        enriched_questions.jsonl    # primary cache — unchanged
  {model_id}/
    single_q_variants/
      c{contract_id}_q{question_index:02d}_variants.csv
```

---

## Example Commands

```bash
# Prerequisite: build sentence/embedding cache (one-time, ~5 min first run)
uv run python rag_eval.py \
  --preflight-golden-sentences-only \
  --contract-scope all \
  --run-id s6-preflight

# Dry-run: verify harness wiring without LLM or RAG (fast)
uv run python langchain_agent.py \
  --single-q --contract-id 327 --question-index 5 \
  --compare-variants \
  --dry-run --model-id s6-test

# Live: all 6 variants for one question (requires DEEPSEEK_API_KEY)
DEEPSEEK_API_KEY=... uv run python langchain_agent.py \
  --single-q --contract-id 327 --question-index 5 \
  --compare-variants \
  --query-enrichment-provider auto \
  --model deepseek/deepseek-v4-flash \
  --model-id s6-live

# Live: single variant — enriched question, hybrid RAG context
DEEPSEEK_API_KEY=... uv run python langchain_agent.py \
  --single-q --contract-id 327 --question-index 5 \
  --question-mode enriched --context-mode rag-hybrid \
  --model deepseek/deepseek-v4-flash \
  --model-id s6-hybrid
```

---

## Implementation Order

1. **Dedup `query_for_row`** — import from `query_enrichment` in `experiments.py`. Run
   existing tests to confirm nothing breaks.
2. **Add `RAG_DEFAULT_TOP_K` and `save_enriched_question_files()`** to
   `query_enrichment.py`. No test changes needed; add a unit test for the file writer.
3. **Create `src/cuad_agent/rag/context_builder.py`** with `build_rag_context()`. Test
   with a tiny in-memory fixture (no real retriever needed — mock
   `load_or_build_retriever` to return a stub).
4. **Add new CLI arguments** to `langchain_runner.py` `parse_args()`. No behaviour
   change yet; run existing tests.
5. **Add `make_messages_with_hint()`** to `langchain_runner.py`.
6. **Implement `run_single_question_variants()`** — dry-run path first (`context-mode raw`
   only), then wire in `build_rag_context()` for RAG modes.
7. **Implement `print_variant_table()`**.
8. **Wire `main()`** — add `if args.single_q:` branch.
9. **Write `tests/test_single_q_variants.py`** — all tests use `--dry-run` and
   `--context-mode raw` so no cache or LLM is needed.
10. **Update `AGENTS.md`** — add the prerequisite preflight step, new command examples,
    and describe the `enriched_questions/` folder and variant naming convention.

---

## Constraints

- The existing full-evaluation path in `langchain_runner.py` is unchanged in behaviour.
- All existing tests pass without modification.
- `--single-q` always loads v2 prompts; if `--prompts-file` is not set, it defaults to
  `prompts/system_prompts_v2.py`. If that file is missing, raise a clear error.
- `--compare-variants` overrides `--question-mode` and `--context-mode` (runs all 6).
- RAG context modes require a pre-built cache. Dry-run tests avoid this by using
  `context-mode raw` only. Live tests require the preflight step.
- No new pip dependencies beyond what already exists in the project.
- The `enriched_v1` label in variant names reserves room for `enriched_v2_prompt`
  (where enrichment terms are baked into the system prompt) in a future sprint.
