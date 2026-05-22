# S5 PRP: Sentence-Level Legal Clause RAG Pipeline

## Goal

Build a reusable RAG evaluation pipeline for CUAD contracts that answers one
question before integrating RAG into the agent:

> For eligible contract-question pairs, what percentage of exact golden-answer
> sentences are present in the top N retrieved contract sentences?

If RAG cannot retrieve the golden clause sentence in the top N, it is unlikely
to help the downstream legal extraction agent.

## Non-Negotiables

- Chunking must respect sentence boundaries. Do not use arbitrary token chunks
  as the scored retrieval unit.
- The first implementation only retrieves individual contract sentences. Do
  not add sentence windows, clause windows, hybrid retrieval, or reranking until
  sentence-only retrieval has been measured.
- Use Docling as the primary parsing/chunking abstraction. CUAD JSON should use
  a Docling-like adapter until original PDFs/page metadata are available.
- Use a Python sentence-splitting package if it can preserve offsets and handle
  legal abbreviations reliably. Keep a deterministic local fallback for cases
  where package boundaries are unsuitable.
- Split golden answers by newline (`\n`) and then by sentence.
- Apply the RAG benchmark only to rows where every golden-answer sentence is an
  exact normalized match to a sentence in the source contract.
- Exclude non-eligible rows from retrieval scoring and report them separately.
- Score coverage by retrieved sentence id, not fuzzy token overlap.
- Store all generated HTML in `frontend/`.
- Store run artifacts in `outputs/{run_id}/rag/`.
- Store reusable sentence/chunk/encoding caches in `outputs/rag_cache/`.

## Primary Metric

```text
gold_sentence_coverage_at_N =
  count(eligible golden answer sentences present in top N retrieved sentences)
  / count(eligible golden answer sentences)
```

Secondary metrics:

- `gold_sentence_coverage_at_1`
- `gold_sentence_coverage_at_3`
- `gold_sentence_coverage_at_5`
- `gold_sentence_coverage_at_10`
- per-category coverage
- per-contract coverage
- mean reciprocal rank of first covering sentence
- no-answer retrieval behavior, tracked separately
- latency, index size, and cache hit/miss rates

Minimum target before feeding RAG into the agent:

```text
100% of scored rows are exact golden-sentence-match eligible
gold_sentence_coverage_at_10 >= 0.85 overall
gold_sentence_coverage_at_10 >= 0.70 for at least 35 of 41 categories
```

## Existing Code To Reuse

- `src/cuad_agent/data/dataset.py`
  - `load_datasets()` for CUAD contracts/questions.
- `src/cuad_agent/eval/examples.py`
  - deterministic sampling, `row_id = document_row_id:question_index`, split
    helpers.
- `src/cuad_agent/eval/metrics.py`
  - existing normalization patterns.
- `src/cuad_agent/evaluators/langchain_runner.py`
  - resumable run/cache/output patterns.
- `src/cuad_agent/dashboards/evaluation.py`
  - static dashboard generation pattern.

Preserve existing project conventions:

- deterministic `sample_size` and `seed`
- JSON/CSV/JSONL outputs for auditability
- browser-review HTML under `frontend/`
- CLI wrapper at repo root plus package console command

## Package Structure

```text
src/cuad_agent/rag/
├── __init__.py
├── contracts.py      # contract records and CUAD/Docling adapters
├── sentences.py      # package-backed sentence splitting and offsets
├── gold_answers.py   # newline + sentence splitting for CUAD gold spans
├── clauses.py        # section/clause metadata detection
├── chunks.py         # sentence-only chunk schema
├── indexes.py        # sentence index interfaces
├── retrievers.py     # sentence retrieval
├── coverage.py       # exact sentence-id coverage metrics
├── experiments.py    # sentence ranking orchestration
├── outputs.py        # CSV/JSONL/summary/dashboard writers
└── cli.py            # command entrypoint
```

Also add:

```text
rag_eval.py
tests/test_rag_sentences.py
tests/test_rag_coverage.py
tests/test_rag_retrieval.py
tests/test_rag_outputs.py
```

`pyproject.toml` console command:

```toml
cuad-eval-rag = "cuad_agent.rag.cli:main"
```

## Pipeline

### 1. Load Evaluation Rows

Use `load_datasets()` and existing sample/split helpers. Inputs needed:

- `document_row_id`
- `title`
- `context`
- `row_id`
- `question_index`
- `category`
- `question`
- `category_description`
- `answer_format`
- `answers`
- `is_impossible`

### 2. Build Sentence Store

Use Docling when structured input is available. For current CUAD JSON, build a
Docling-like text adapter:

- `page_number=None`
- stable `start_char` and `end_char`
- parent section/clause metadata when detectable
- same downstream schema as future PDF parsing

Sentence splitting rules:

- Prefer a package-backed splitter if it meets the contract:
  - preserves character offsets, or allows offsets to be reconstructed exactly
  - can be configured for legal/entity abbreviations
  - produces deterministic boundaries across runs
- Candidate packages:
  - `pysbd` for deterministic rule-based sentence boundaries
  - `spaCy` sentencizer if offset behavior and abbreviation handling pass tests
  - NLTK Punkt only if legal abbreviation tests pass
- If no package passes the tests, use a small deterministic fallback splitter
  with explicit legal abbreviation handling.
- preserve raw source text
- normalize only for matching/indexing
- keep legal abbreviations intact: `Inc.`, `Corp.`, `Ltd.`, `LLC.`, `No.`,
  `Sec.`, `Art.`, `U.S.`
- preserve semicolon-heavy legal sentences unless a reliable sentence boundary
  exists
- store stable ids: `{document_row_id}:s:{sentence_index}`

Core schema:

```python
class SentenceSpan(BaseModel):
    document_row_id: int
    sentence_id: str
    sentence_index: int
    raw_text: str
    normalized_text: str
    start_char: int
    end_char: int
    page_number: int | None
    section_number: str | None
    section_title: str | None
    clause_path: list[str]
```

### 3. Validate Golden Answer Eligibility

Preflight every sampled row before retrieval:

1. Keep extraction rows where `is_impossible == False` and `answers` has text.
2. Split each answer on newline (`\n`).
3. Split each answer span into sentences using the same sentence splitter.
4. Normalize each golden sentence and contract sentence.
5. Mark the row eligible only if every golden sentence exactly equals a
   contract sentence.
6. Report non-eligible rows with reason:
   - `partial_span_inside_sentence`
   - `cross_sentence_span_without_exact_sentence_match`
   - `normalization_mismatch`
   - `gold_answer_not_found`

Acceptance for preflight:

```text
eligible rows have per_row_gold_sentence_contract_coverage == 100%
non-eligible rows are excluded from retrieval scoring and reported
```

### 4. Build Sentence Retrieval Chunks

The retrieval unit and scored unit are the same object: one contract sentence.
Section and clause data are metadata only; they should not expand the retrieved
text in the first implementation.

Chunk schema:

```python
class RagChunk(BaseModel):
    chunk_id: str
    document_row_id: int
    text: str
    normalized_text: str
    chunk_type: Literal["sentence"]
    sentence_ids: list[str]
    start_char: int
    end_char: int
    page_number: int | None
    section_number: str | None
    section_title: str | None
    clause_path: list[str]
```

### 5. Persist Encoded Sentence Cache

Encoded sentences are transformed once and stored permanently for reuse. Later
runs should load them from cache unless a manifest key changes.

```text
outputs/rag_cache/
├── chunking/{chunking_version}/
│   ├── contracts_manifest.json
│   ├── sentence_spans.jsonl
│   ├── chunking_config.json
│   └── encodings/{embedding_model_slug}/
│       ├── chunk_ids.json
│       ├── dense_index.pkl
│       ├── embedding_manifest.json
│       ├── embeddings.npy or embeddings.npz
│       └── vectorizer.pkl when using TF-IDF
├── sparse/{chunking_version}/bm25/
│   ├── bm25_corpus.jsonl
│   └── bm25_index.pkl
```

Cache keys:

- `document_row_id`
- contract text hash
- `chunking_version`
- sentence splitter config hash
- Docling/text-adapter config hash
- embedding model id
- embedding dimension

Required behavior:

- load existing cache when manifest hashes match
- rebuild only missing/stale chunks or embeddings
- support `--rebuild-chunks` and `--rebuild-embeddings`
- write cache hits/misses to `rag_summary.json`

### 6. Encode, Retrieve, Rank Sentences

Encoding is a required stage immediately after sentence chunking. The first
implementation compares sentence-only ranking methods over the same sentence
store and encoded sentence cache:

| Method | Unit | Query Form | Notes |
| --- | --- | --- | --- |
| `bm25_sentence` | sentence | category_description | lexical baseline |
| `dense_sentence` | sentence | category_description | encoded sentence baseline |

Retrieval constraints:

- retrieve only within the same `document_row_id`
- evaluate only eligible rows
- skip no-answer rows for primary coverage
- retrieve top `k` for `k in [1, 3, 5, 10]`
- coverage is true only when the retrieved sentence id equals the golden
  sentence id

Recommended libraries:

- Docling for document model/chunking abstraction
- `rank_bm25` for first BM25 baseline
- SentenceTransformers or BGE for local dense embeddings

## Outputs

Run outputs:

```text
outputs/{run_id}/rag/
├── golden_sentence_coverage.csv
├── golden_sentence_coverage_summary.json
├── rag_sentences.jsonl
├── rag_retrieval_results.jsonl
├── rag_retrieval_results.csv
├── rag_summary.json
├── rag_ranking_summary.csv
└── rag_config.json
```

HTML output:

```text
frontend/rag_pipeline_eval.html
```

The dashboard is a single tabbed page for the latest RAG run. It should show:

- eligible vs non-eligible rows
- unmatched golden sentences and reason codes
- ranking-method summary and coverage@1/3/5/10
- per-category coverage
- top-N retrieved sentences with metadata
- highlighted matched golden sentence when covered
- failed examples where top N does not cover the golden sentence

## CLI

Preflight only:

```bash
uv run python rag_eval.py \
  --run-id rag-golden-preflight \
  --sample-size 50 \
  --seed 42 \
  --preflight-golden-sentences-only \
  --output-dir outputs
```

Smoke test:

```bash
uv run python rag_eval.py \
  --run-id rag-smoke \
  --sample-size 1 \
  --seed 42 \
  --retrievers bm25_sentence \
  --top-k 10 \
  --output-dir /tmp/cuad-rag-smoke/outputs
```

Full local evaluation:

```bash
uv run python rag_eval.py \
  --run-id rag-sentence-v1 \
  --sample-size 50 \
  --seed 42 \
  --retrievers bm25_sentence,dense_sentence \
  --top-k 10 \
  --output-dir outputs
```

Equivalent package command:

```bash
uv run cuad-eval-rag --run-id rag-sentence-v1 --sample-size 50 --seed 42
```

Useful flags:

```text
--preflight-golden-sentences-only
--eval-split PATH:SPLIT_NAME
--question-indices 0,1,18
--contract-ids 327,57,12
--embedding-model MODEL_NAME
--resume-existing
--rebuild-chunks
--rebuild-embeddings
```

## Implementation Order

1. Add package structure, root wrapper, and console command.
2. Implement CUAD-to-Docling-like document adapter.
3. Choose and wrap a Python sentence splitter package, with deterministic
   fallback if package output fails offset/legal-abbreviation tests.
4. Chunk contracts into sentence spans with offsets, stable ids, and optional
   clause/section metadata.
5. Persist the contract sentence store under `outputs/rag_cache/`.
6. Chunk golden answers by newline and sentence using the same splitter.
7. Match golden-answer sentences back to contract sentences exactly.
8. Identify which rows/questions are sentence-extraction eligible and report
   non-eligible rows separately.
9. Encode all chunked contract sentences and persist encoded arrays in the
   chunking cache.
10. Implement sentence-only BM25 retrieval.
11. Implement sentence-only dense retrieval behind an optional dependency
    boundary.
12. Rank top N retrieved sentences for each eligible contract-question row.
13. Implement exact sentence-id coverage metrics for `% of golden sentences in
    top N`.
14. Write CSV/JSONL/summary/dashboard outputs.
15. Add tests for sentence splitting, offsets, golden-answer matching,
    sentence-extraction eligibility, cache reuse, retrieval, and output paths.
16. Run preflight, smoke test, then 50-contract sentence-retrieval comparison.

## Validation

Before implementation:

```bash
uv run python -m pytest -q
```

After implementation:

```bash
uv run python -m py_compile $(rg --files src -g '*.py') rag_eval.py
uv run python -m pytest -q
```

Required smoke outputs:

```text
/tmp/cuad-rag-smoke/outputs/rag-smoke/rag/golden_sentence_coverage.csv
/tmp/cuad-rag-smoke/outputs/rag-smoke/rag/golden_sentence_coverage_summary.json
/tmp/cuad-rag-smoke/outputs/rag-smoke/rag/rag_retrieval_results.csv
/tmp/cuad-rag-smoke/outputs/rag-smoke/rag/rag_summary.json
/tmp/cuad-rag-smoke/frontend/rag_pipeline_eval.html
```

## Gotchas

- CUAD JSON has no real page numbers. Use `page_number=None` until PDF parsing
  is added.
- Some CUAD answers are partial spans inside longer sentences. They are useful
  diagnostics but excluded from the primary sentence-level benchmark unless
  they exactly match a source sentence after splitting.
- Yes/no categories may still need supporting clause retrieval. Track the label
  separately from the golden span text.
- BM25 should be a strong legal baseline because clause names often use exact
  legal terms. Dense-only retrieval may miss exact phrases.
- Do not retrieve across contracts for this metric.
- Do not write HTML outside `frontend/`.
