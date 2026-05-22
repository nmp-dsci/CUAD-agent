# S7 — Hierarchical RAG Retrieval

## Goal

Add a hierarchical RAG retrieval strategy that improves golden-answer coverage by
searching at the sentence level (leaves), expanding to their parent sections, then
returning the top N sentences from the selected sections.

**Definition of complete:**

1. `rag_eval.py` runs sentence chunking (unchanged), uses those sentence chunks with
   their existing `clause_path` metadata for both classic and hierarchical retrieval,
   and writes all retrieval variants — including hierarchical — to `rag_pipeline_eval.html`
   with a dedicated comparison section.
2. `langchain_agent.py` supports `rag-hierarchical-bm25` and `rag-hierarchical-dense`
   as selectable context modes and runs them alongside all existing single-question
   variants.

**Out of scope for S7:** token-consumption tracking. That is a separate schema
migration because it must update the CSV cache, JSONL resume stream, summary
aggregation, and provider-specific usage parsing. Do not add token fields in this
ticket.

---

## Retrieval Algorithm

### Problem with flat retrieval

End-of-sentence chunking fragments clause text. A single clause spanning four sentences
may only have one sentence near the top of BM25/dense results. The golden answer is
captured in the chunk list but not in the top-K cutoff.

### Hierarchical solution: Leaf → Section → Sentence

```
retrieve_hierarchical(query, document_row_id,
                      leaf_k=50, top_sections=5, top_k=20):

  1. Leaf search
     Use BM25 or dense index (same sentence-level chunks as existing retrievers).
     Retrieve top leaf_k sentences with scores.

  2. Section scoring
     For each retrieved sentence, read its clause_path (already populated on
     every SentenceSpan from clauses.py).
     section_key = tuple(clause_path)   # e.g. ("3.1", "Confidential Information")
     score each unique key by the SUM of scores of its retrieved sentences.

  3. Section selection
     Take top_sections unique section keys by score.

  4. Section expansion
     Collect ALL sentences in document_row_id whose clause_path key is in the
     selected top sections — not just the ones originally retrieved.

  5. Re-ranking
     Score every expanded candidate chunk directly with the same index used in
     step 1. Sort by score, return top top_k as SearchResult objects.

     Do not call index.search(..., top_k=top_k * N) and filter afterward. That
     drops expanded sentences whenever the global document top-N does not contain
     enough candidates from the selected sections.

  6. Context formatting
     Group results by section, prepend section header to each group:

     [SECTION 3.1 — Confidential Information]
     Sentence A...
     Sentence B...

     [SECTION 9.2 — Effect of Termination]
     Sentence C...
```

### Why this works for CUAD

Golden answers in CUAD are clause-level spans (one to four sentences). A clause lives
within one section. When even one of its sentences ranks in the leaf top-K, the section
gets scored and expanded. The remaining clause sentences — previously pushed below the
cutoff — are pulled back through expansion.

---

## What Already Exists (no changes needed)

| Component | Status |
|-----------|--------|
| Sentence chunking (`sentences.py`) | Complete. Unchanged. |
| Section header parsing (`clauses.py:build_section_metadata`) | Complete. Populates `SentenceSpan.clause_path` already. |
| `SentenceSpan.clause_path`, `section_number`, `section_title` | Complete. Populated for every span via `experiments.py:build_sentence_store`. |
| `RagChunk.clause_path` | Complete. Copied 1:1 from `SentenceSpan` in `chunk_from_sentence`. |
| BM25 and dense indexes (`indexes.py`) | Complete. Reused as-is. |
| Coverage measurement (`coverage.py`) | Complete. Works on any `SearchResult` list. Unchanged. |
| `run_sentence_retrieval()` (`experiments.py`) | Existing orchestration loop. Extended to accept new methods. |
| `write_pipeline_html()` (`outputs.py`) | Extended with a new section. |

---

## New File: `src/cuad_agent/rag/hierarchy.py`

Single responsibility: section grouping and hierarchical retrieval.

```python
"""Hierarchical RAG: leaf-search → section expansion → re-rank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cuad_agent.rag.chunks import RagChunk
from cuad_agent.rag.indexes import BM25SentenceIndex, DenseSentenceIndex, SearchResult
from cuad_agent.rag.sentences import SentenceSpan


HIERARCHICAL_RETRIEVERS = {"bm25_hierarchical", "dense_hierarchical"}


@dataclass(frozen=True)
class SectionNode:
    section_key: tuple[str, ...]    # normalised tuple(clause_path)
    document_row_id: int
    section_number: str | None
    section_title: str | None
    sentence_ids: list[str]         # ordered list of sentence IDs in this section


def build_section_index(
    sentence_spans: list[SentenceSpan],
) -> dict[int, list[SectionNode]]:
    """Group SentenceSpans into SectionNodes keyed by document_row_id.

    Two spans belong to the same section when their tuple(clause_path) is
    identical. Spans with an empty clause_path are grouped under a single
    '_unsectioned' node per document.
    """
    ...


def section_key_for(item: SentenceSpan | RagChunk) -> tuple[str, ...]:
    clause_path = item.clause_path
    if clause_path:
        return tuple(clause_path)
    return ("_unsectioned",)


class HierarchicalRetriever:
    """Wraps an existing sentence-level index with section expansion.

    Compatible with the SentenceRetriever.search() signature so it can be
    passed to run_sentence_retrieval() as a prebuilt retriever.
    """

    def __init__(
        self,
        *,
        method: Literal["bm25_hierarchical", "dense_hierarchical"],
        index: BM25SentenceIndex | DenseSentenceIndex,
        section_index: dict[int, list[SectionNode]],
        leaf_k: int = 50,
        top_sections: int = 5,
    ) -> None:
        self.method = method
        self.index = index
        self.section_index = section_index
        self.leaf_k = leaf_k
        self.top_sections = top_sections

    def search(
        self,
        query: str,
        *,
        document_row_id: int,
        top_k: int,
    ) -> list[SearchResult]:
        # Step 1: leaf search
        leaf_results = self.index.search(
            query, document_row_id=document_row_id, top_k=self.leaf_k
        )

        # Step 2: section scoring
        section_scores: dict[tuple[str, ...], float] = {}
        for result in leaf_results:
            key = section_key_for(result.chunk)
            section_scores[key] = section_scores.get(key, 0.0) + result.score

        # Step 3: select top sections
        top_keys = set(
            sorted(section_scores, key=section_scores.__getitem__, reverse=True)[
                : self.top_sections
            ]
        )

        # Step 4: expansion — collect all sentence_ids in those sections
        expanded_ids: set[str] = set()
        for node in self.section_index.get(document_row_id, []):
            if node.section_key in top_keys:
                expanded_ids.update(node.sentence_ids)

        # Step 5: re-rank expanded candidates by index score
        candidate_chunks = [
            chunk
            for chunk in self.index.chunks
            if chunk.document_row_id == document_row_id
            and any(s in expanded_ids for s in chunk.sentence_ids)
        ]
        scored = self._score_candidates(query, candidate_chunks)
        ranked = sorted(scored, key=lambda r: r.score, reverse=True)[:top_k]
        return [
            SearchResult(chunk=result.chunk, score=result.score, rank=rank)
            for rank, result in enumerate(ranked, start=1)
        ]

    def _score_candidates(
        self, query: str, candidate_chunks: list[RagChunk]
    ) -> list[SearchResult]:
        """Score exactly candidate_chunks with the wrapped index.

        Implementation requirement:
        - BM25 must call `BM25SentenceIndex.score(query, index)` against each
          candidate chunk's index position.
        - Dense must encode the query once and dot-product/cosine against the
          cached candidate embedding rows, matching `DenseSentenceIndex.search()`
          behavior for `sentence_transformers` and TF-IDF backends.
        - Do not rebuild embeddings and do not call search() over the whole
          document followed by filtering.
        """
        ...


def format_hierarchical_context(results: list[SearchResult]) -> str:
    """Group results by section, prepend section header, join with blank lines."""
    ...
```

**Key constraint on `HierarchicalRetriever.search()`:** it must return `list[SearchResult]`
exactly as `SentenceRetriever.search()` does, with `result.chunk.sentence_ids` populated.
This ensures coverage measurement in `coverage.py` works identically.

**`top_k` semantics:** `run_sentence_retrieval()` passes
`top_k=retrieval_top_k`, where `retrieval_top_k = max(top_k, 30)`. For hierarchical
retrievers, that means `hierarchical@30` is the best 30 sentences from only the
selected top sections. Coverage at high cutoffs is intentionally capped by
`top_sections`; dashboards and comments must state this so readers compare
hierarchical rows against the matching flat retriever with that design in mind.

---

## Changes to Existing Files

### `src/cuad_agent/rag/retrievers.py`

Add `build_hierarchical_retriever()`:

```python
def build_hierarchical_retriever(
    method: Literal["bm25_hierarchical", "dense_hierarchical"],
    index: BM25SentenceIndex | DenseSentenceIndex,
    section_index: dict[int, list[SectionNode]],
    *,
    leaf_k: int = 50,
    top_sections: int = 5,
) -> HierarchicalRetriever:
    from cuad_agent.rag.hierarchy import HierarchicalRetriever
    return HierarchicalRetriever(
        method=method,
        index=index,
        section_index=section_index,
        leaf_k=leaf_k,
        top_sections=top_sections,
    )
```

The caller must pass an already-built flat sentence index. Do not construct a
fresh `DenseSentenceIndex` here; with sentence-transformers that would trigger a
second full embedding pass. Reuse the same `BM25SentenceIndex` or
`DenseSentenceIndex` already loaded for `bm25_sentence` / `dense_sentence`.

### `src/cuad_agent/rag/experiments.py`

**Constants:**

```python
from cuad_agent.rag.hierarchy import HIERARCHICAL_RETRIEVERS

# existing sets unchanged
HIERARCHICAL_RETRIEVERS  # imported, not redeclared
```

**`run_rag_eval()` — new parameters:**

```python
def run_rag_eval(
    *,
    ...,
    hierarchical_leaf_k: int = 50,
    hierarchical_top_sections: int = 5,
) -> dict[str, Any]:
```

**Inside `run_rag_eval()`** — after building sentence cache and sentence lookup:

```python
from cuad_agent.rag.hierarchy import build_section_index, HIERARCHICAL_RETRIEVERS
from cuad_agent.rag.retrievers import build_hierarchical_retriever

section_index: dict[int, list[SectionNode]] = {}
bm25_hierarchical_retriever: HierarchicalRetriever | None = None
dense_hierarchical_retriever: HierarchicalRetriever | None = None

if any(method in HIERARCHICAL_RETRIEVERS for method in retrievers):
    section_index = build_section_index(sentence_spans)
    if "bm25_hierarchical" in retrievers:
        bm25_retriever, _ = load_or_build_retriever(
            method="bm25_sentence",
            chunks=chunks,
            output_dir=output_dir,
            chunking_version=chunking_version,
            embedding_model=embedding_model,
            rebuild=rebuild_embeddings,
        )
        bm25_hierarchical_retriever = build_hierarchical_retriever(
            "bm25_hierarchical",
            index=bm25_retriever.index,
            section_index=section_index,
            leaf_k=hierarchical_leaf_k,
            top_sections=hierarchical_top_sections,
        )
    if "dense_hierarchical" in retrievers:
        dense_retriever, _ = load_or_build_dense_sentence_encoder(
            chunks=chunks,
            method="dense_sentence",
            output_dir=output_dir,
            chunking_version=chunking_version,
            embedding_model=embedding_model,
            rebuild=rebuild_embeddings,
        )
        dense_hierarchical_retriever = build_hierarchical_retriever(
            "dense_hierarchical",
            index=dense_retriever.index,
            section_index=section_index,
            leaf_k=hierarchical_leaf_k,
            top_sections=hierarchical_top_sections,
        )
```

If `bm25_sentence` or `dense_sentence` is already being run in the same
`run_rag_eval()` invocation, reuse that existing retriever object instead of
calling `load_or_build_*` again. The point is one index object per method per run,
shared by flat and hierarchical retrieval.

**Extend `prebuilt_retrievers` dict** passed to `run_sentence_retrieval()`:

```python
prebuilt_retrievers = {
    "dense_sentence": (dense_retriever, ...),
    **({"dense_legal_recursive": ...} if dense_legal_retriever else {}),
    **({"bm25_hierarchical": (bm25_hierarchical_retriever, False)}
       if bm25_hierarchical_retriever else {}),
    **({"dense_hierarchical": (dense_hierarchical_retriever, False)}
       if dense_hierarchical_retriever else {}),
}
```

**Extend `chunks_by_method`:**

```python
chunks_by_method = {
    "bm25_sentence": chunks,
    "dense_sentence": chunks,
    "bm25_legal_recursive": legal_recursive_chunks,
    "dense_legal_recursive": legal_recursive_chunks,
    "bm25_hierarchical": chunks,      # same sentence chunks
    "dense_hierarchical": chunks,     # same sentence chunks
}
```

**Add to config output:**

```python
config = {
    ...,
    "hierarchical_leaf_k": hierarchical_leaf_k if any(m in HIERARCHICAL_RETRIEVERS for m in retrievers) else None,
    "hierarchical_top_sections": hierarchical_top_sections if any(m in HIERARCHICAL_RETRIEVERS for m in retrievers) else None,
}
```

### `src/cuad_agent/rag/cli.py` (rag_eval.py entrypoint)

Add CLI flags:

```python
parser.add_argument("--hierarchical-leaf-k", type=int, default=50,
    help="Number of leaf sentences to retrieve before section expansion.")
parser.add_argument("--hierarchical-top-sections", type=int, default=5,
    help="Number of top sections to expand into.")
```

Pass through to `run_rag_eval()`.

Also extend retriever validation/help text to include `bm25_hierarchical` and
`dense_hierarchical`. The current flag is comma-separated, so examples should use
`--retrievers bm25_sentence,dense_sentence,bm25_hierarchical,dense_hierarchical`.
If not specified explicitly, the default retriever set remains unchanged (no
auto-inclusion of hierarchical variants).

### `src/cuad_agent/rag/context_builder.py`

Add `build_hierarchical_rag_context()`:

```python
from cuad_agent.rag.hierarchy import (
    HierarchicalRetriever,
    build_section_index,
    format_hierarchical_context,
)

_HIERARCHICAL_CONTEXT_CACHE: dict[
    tuple[Path, str, str, str, int, int],
    HierarchicalRetriever,
] = {}

def build_hierarchical_rag_context(
    *,
    document_row_id: int,
    query: str,
    method: Literal["rag-hierarchical-bm25", "rag-hierarchical-dense"],
    leaf_k: int = 50,
    top_sections: int = 5,
    top_k: int = 20,
    output_dir: Path,
    chunking_version: str,
    embedding_model: str,
) -> tuple[str, list[str]]:
    """Return (context_text, retrieved_chunk_ids).

    context_text groups retrieved sentences under their section headers:

        [SECTION 3.1 — Confidential Information]
        The term "Confidential Information" means...
        This obligation shall not apply to...

        [SECTION 9.2 — Effect of Termination]
        Upon any expiration or termination...

    Requires the sentence span cache (same preflight as build_rag_context).
    """
    cache_key = (
        output_dir.resolve(),
        chunking_version,
        embedding_model,
        method,
        leaf_k,
        top_sections,
    )
    h_retriever = _HIERARCHICAL_CONTEXT_CACHE.get(cache_key)
    if h_retriever is None:
        spans = _require_spans(output_dir, chunking_version)
        chunks = chunks_from_sentences(spans)
        section_index = build_section_index(spans)

        underlying_method = (
            "bm25_sentence" if method == "rag-hierarchical-bm25" else "dense_sentence"
        )
        if underlying_method == "bm25_sentence":
            retriever, _ = load_or_build_retriever(
                method="bm25_sentence",
                chunks=chunks,
                output_dir=output_dir,
                chunking_version=chunking_version,
                embedding_model=embedding_model,
                rebuild=False,
            )
        else:
            retriever, _ = load_or_build_dense_sentence_encoder(
                chunks=chunks,
                method="dense_sentence",
                output_dir=output_dir,
                chunking_version=chunking_version,
                embedding_model=embedding_model,
                rebuild=False,
            )

        h_retriever = HierarchicalRetriever(
            method="bm25_hierarchical" if method == "rag-hierarchical-bm25" else "dense_hierarchical",
            index=retriever.index,
            section_index=section_index,
            leaf_k=leaf_k,
            top_sections=top_sections,
        )
        _HIERARCHICAL_CONTEXT_CACHE[cache_key] = h_retriever

    results = h_retriever.search(query, document_row_id=document_row_id, top_k=top_k)
    chunk_ids = [r.chunk.chunk_id for r in results]
    context_text = format_hierarchical_context(results)
    return context_text, chunk_ids
```

`build_hierarchical_rag_context()` is called once per eval row by
`_apply_rag_context_to_devset()`. It must cache the span load, section index, and
wrapped retriever by `(output_dir, chunking_version, embedding_model, method,
leaf_k, top_sections)` so full evaluations do not rebuild the section index 2,050
times for a 50-contract sample. A future optimization may hoist the retriever into
`_apply_rag_context_to_devset()`, but the S7 implementation must at least include
the module-level cache above.

### `src/cuad_agent/evaluators/langchain_runner.py`

#### New `--context-mode` choices

Extend the existing `choices` list:

```python
parser.add_argument(
    "--context-mode",
    choices=("raw", "rag-dense", "rag-hybrid",
             "rag-hierarchical-bm25", "rag-hierarchical-dense"),
    default="raw",
)
```

#### New CLI flags

```python
parser.add_argument("--hierarchical-leaf-k", type=int, default=50)
parser.add_argument("--hierarchical-top-sections", type=int, default=5)
```

#### `run_single_question_variants()` — extend context resolution

In step 4b (Resolve contract context), add the hierarchical branches:

```python
if context_mode in {"rag-hierarchical-bm25", "rag-hierarchical-dense"}:
    contract_text, chunk_ids = build_hierarchical_rag_context(
        document_row_id=contract_id,
        query=retrieval_query,
        method=context_mode,
        leaf_k=hierarchical_leaf_k,
        top_sections=hierarchical_top_sections,
        top_k=top_k,
        output_dir=output_dir,
        chunking_version=chunking_version,
        embedding_model=embedding_model,
    )
```

#### Variant name mapping — extend module-level `_VARIANT_CTX_LABELS`

`_VARIANT_CTX_LABELS` already exists at module scope. Update that dict in place;
do not introduce a shadow local mapping.

```python
_VARIANT_CTX_LABELS = {
    "raw": "raw_ctx",
    "rag-dense": "rag_dense",
    "rag-hybrid": "rag_hybrid",
    "rag-hierarchical-bm25": "rag_hier_bm25",
    "rag-hierarchical-dense": "rag_hier_dense",
}
```

#### `--compare-variants` expansion

When `--compare-variants` is set, context modes expand to all five:

```python
context_modes = (
    ["raw", "rag-dense", "rag-hybrid",
     "rag-hierarchical-bm25", "rag-hierarchical-dense"]
    if args.compare_variants
    else [args.context_mode]
)
```

This gives 2 question modes × 5 context modes = **10 variants** in compare mode.

#### Live compare warning

When `--compare-variants` is set and `--dry-run` is not set, print one warning line
before model calls start:

```text
Running 10 live single-question variants (2 question modes x 5 context modes).
```

The existing 6-variant probe becomes 10 variants once hierarchical contexts are
included, so this warning prevents accidental live spend during ad-hoc probes.

### `src/cuad_agent/rag/outputs.py` — HTML dashboard

#### New section: "Hierarchical RAG Performance"

`write_pipeline_html()` receives the existing `ranking_summary` list (one dict per
retriever method). The hierarchical methods appear in it automatically once they are
added to the retriever list.

Add a **dedicated section** to the rendered HTML with:

1. **Coverage comparison table** — rows are retrieval methods, columns are
   `@1, @3, @5, @10, @20, @30, @top_k`. Mark hierarchical rows with a distinct
   background color. Show delta vs. `bm25_sentence` baseline and delta vs. the
   matching flat counterpart:
   - `bm25_hierarchical` vs. `bm25_sentence`
   - `dense_hierarchical` vs. `dense_sentence`

   Add a short note near the table: hierarchical `@30` is measured after expanding
   only `top_sections` sections, so high-cutoff coverage is intentionally bounded
   by the selected-section design.

2. **Bar chart** (Chart.js, already used) — `gold_sentence_coverage_at_20` for all
   methods including hierarchical. One bar per method, hierarchical bars in a distinct
   color series.

3. **Parameter table** — `leaf_k`, `top_sections`, `top_k` used in this run (from config).

`write_pipeline_html()` signature gains two optional parameters:

```python
def write_pipeline_html(
    path: Path,
    *,
    ...,
    hierarchical_config: dict[str, Any] | None = None,
) -> None:
```

`hierarchical_config` is `{"leaf_k": 50, "top_sections": 5}` when hierarchical methods
were run; `None` otherwise. Controls whether the dedicated section renders.

---

## New Tests: `tests/test_rag_hierarchical.py`

All tests are pure in-memory — no disk I/O, no LLM, no embeddings.

| Test | What it checks |
|------|----------------|
| `test_build_section_index_groups_by_clause_path` | Two spans with same clause_path → one SectionNode |
| `test_build_section_index_unsectioned_bucket` | Spans with empty clause_path → `_unsectioned` node |
| `test_section_key_stability` | Same clause_path content → same key regardless of object identity |
| `test_hierarchical_retriever_expands_section` | Leaf search returns sentence_A, section contains sentence_B; both appear in output |
| `test_hierarchical_retriever_expands_full_clause_when_siblings_rank_low` | Motivating case: leaf search finds sentence_A in a 4-sentence clause while the other 3 sentences would rank outside `leaf_k`; assert all 4 appear in hierarchical output |
| `test_hierarchical_expansion_uses_sections_not_leaf_results_only` | Assert an expanded sentence was not present in `leaf_results` but still appears in final output |
| `test_hierarchical_retriever_respects_top_k` | Output length ≤ top_k |
| `test_hierarchical_retriever_top_sections_limit` | With `top_sections=1`, only sentences from top-scoring section returned |
| `test_hierarchical_retriever_filters_requested_document` | No result has `chunk.document_row_id != requested_document_row_id` |
| `test_hierarchical_retriever_missing_document_returns_empty` | Requesting a document absent from `section_index` returns `[]` and does not raise |
| `test_format_hierarchical_context_groups_by_section` | Section header appears once per section; sentences appear under it |
| `test_format_hierarchical_context_empty` | No crash on empty results |
| `test_coverage_works_on_hierarchical_results` | `coverage_by_top_chunks()` on hierarchical results returns correct coverage |

---

## Parameters and Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `leaf_k` | 50 | Leaf sentences retrieved before expansion. Higher = more section coverage, slower. |
| `top_sections` | 5 | Sections to expand into. 3–7 is the practical range for CUAD-length contracts. |
| `top_k` | 20 | Final sentences returned. Passed from caller (same as other retrievers). |

These defaults are tunable at the CLI:
- `rag_eval.py --hierarchical-leaf-k 50 --hierarchical-top-sections 5`
- `langchain_agent.py --hierarchical-leaf-k 50 --hierarchical-top-sections 5`

---

## Output Folder Structure

No new cache directories. Hierarchical retrieval is computed at query time from the
existing sentence span cache. The extra cost vs. flat BM25 is negligible (in-memory
grouping over the already-loaded span list).

```
outputs/
  {run_id}/
    rag/
      rag_ranking_summary.csv          # now includes bm25_hierarchical, dense_hierarchical rows
      rag_retrieval_results.csv        # now includes hierarchical method rows
      rag_config.json                  # now includes hierarchical_leaf_k, hierarchical_top_sections
      rag_summary.json                 # unchanged structure; new methods auto-appear
  {model_id}/
    single_q_variants/
      c{contract_id}_q{idx:02d}_variants.csv   # 10 rows when --compare-variants
frontend/
  rag_pipeline_eval.html               # new "Hierarchical RAG Performance" section
```

---

## Example Commands

```bash
# Run rag_eval with both flat and hierarchical retrievers (coverage benchmark)
uv run python rag_eval.py \
  --run-id s7-hierarchical \
  --retrievers bm25_sentence,dense_sentence,bm25_hierarchical,dense_hierarchical \
  --hierarchical-leaf-k 50 \
  --hierarchical-top-sections 5 \
  --top-k 20 \
  --contract-scope all

# Single-question: compare all 10 variants (dry-run, no LLM or cache needed for raw)
uv run python langchain_agent.py \
  --single-q --contract-id 327 --question-index 7 \
  --compare-variants \
  --dry-run --model-id s7-test

# Single-question: hierarchical only, live
DEEPSEEK_API_KEY=... uv run python langchain_agent.py \
  --single-q --contract-id 327 --question-index 7 \
  --question-mode enriched \
  --context-mode rag-hierarchical-dense \
  --hierarchical-leaf-k 50 --hierarchical-top-sections 5 \
  --model deepseek/deepseek-v4-flash \
  --model-id s7-hier-dense

# Full eval with hierarchical context (all 41 questions × sample)
DEEPSEEK_API_KEY=... uv run python langchain_agent.py \
  --context-mode rag-hierarchical-bm25 \
  --sample-size 10 \
  --model deepseek/deepseek-v4-flash \
  --model-id s7-full-hier
```

---

## Implementation Order

1. **`src/cuad_agent/rag/hierarchy.py`** — `SectionNode`, `build_section_index()`,
   `section_key_for()`, `HierarchicalRetriever`, `format_hierarchical_context()`.
   Wire nothing else yet. Write unit tests for these in isolation.

2. **`tests/test_rag_hierarchical.py`** — all hierarchy tests against the hierarchy module.
   Run `uv run pytest tests/test_rag_hierarchical.py -q` — all pass before continuing.

3. **`src/cuad_agent/rag/retrievers.py`** — add `build_hierarchical_retriever()`.

4. **`src/cuad_agent/rag/experiments.py`** — import and wire hierarchical retriever.
   Add `hierarchical_leaf_k`, `hierarchical_top_sections` params to `run_rag_eval()`.
   Extend `chunks_by_method`, `prebuilt_retrievers`, `config` outputs.

5. **`src/cuad_agent/rag/cli.py`** — add `--hierarchical-leaf-k`, `--hierarchical-top-sections`
   flags; extend `--retrievers` validation/help text.

6. **`src/cuad_agent/rag/outputs.py`** — add hierarchical comparison section to
   `write_pipeline_html()`. Add `hierarchical_config` parameter.

7. **`src/cuad_agent/rag/context_builder.py`** — add `build_hierarchical_rag_context()`.

8. **`src/cuad_agent/evaluators/langchain_runner.py`** — extend `--context-mode` choices;
   add `--hierarchical-leaf-k`, `--hierarchical-top-sections` flags; wire new modes into
   `run_single_question_variants()`; update `_VARIANT_CTX_LABELS`; add the live
   compare warning.

9. **Run full test suite** — `uv run pytest -q`. All 50+ existing tests must pass.

10. **Smoke test coverage benchmark** — run rag_eval with all 6 retrievers; check
    `rag_pipeline_eval.html` renders the hierarchical section.

11. **`AGENTS.md`** — document new retriever methods, CLI flags, parameter tuning guide,
    example commands, and how hierarchical context appears in the HTML dashboard.

---

## Constraints

- The existing sentence chunking pipeline is **unchanged**. No new chunk type is added.
  The hierarchical strategy operates entirely within retrieval, not chunking.
- The existing `--retrievers` default set is unchanged; hierarchical methods must be
  opted in explicitly.
- All existing tests pass without modification.
- `HierarchicalRetriever.search()` returns `list[SearchResult]` compatible with the
  existing `coverage.py` functions — no special-casing in coverage or experiments.
- The full-eval path in `langchain_runner.py` (non `--single-q`) is unchanged in
  structure; the new context modes are additive.
- No new pip dependencies. BM25 hierarchical uses the existing `BM25SentenceIndex`.
  Dense hierarchical uses the existing `DenseSentenceIndex` (TF-IDF or sentence-transformers).
- `format_hierarchical_context()` is also called from `build_hierarchical_rag_context()`
  in `context_builder.py`; it lives in `hierarchy.py` to avoid circular imports.
