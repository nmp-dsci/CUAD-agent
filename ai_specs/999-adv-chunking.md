# S8 — Advanced Hierarchical Chunking

**GitHub Issue:** [#1 — Smart Chunking](https://github.com/nmp-dsci/CUAD-agent/issues/1)

## Goal

Add Alternative chunking strategy  (sentence-level, legal-recursive token
windows) with a **tree-structured chunking** strategy that mirrors the actual nesting
hierarchy of each contract.

A contract is parsed into a `ContractTree` where each node is a heading (exhibit,
article, section, subsection, list item) with its direct body text and child nodes.
RAG retrieves at the leaf level but can traverse the tree upward to pull sibling clauses
or the full parent section into context.

As a complementary deliverable, raw contract whitespace is normalised at ingestion time
— OCR/PDF extraction artifacts (multi-space padding, irregular line breaks) are stripped
before any chunking or indexing.

**Definition of complete:**

1. `src/cuad_agent/rag/tree.py` exists with `ContractNode`, `build_contract_tree()`,
   and `tree_to_chunks()`. `tree_to_chunks()` produces a flat `list[RagChunk]` with
   `chunk_type="hierarchical"` and a fully-populated `clause_path` reflecting the
   complete ancestry of each node.
2. `src/cuad_agent/rag/contracts.py` exposes `normalize_contract_whitespace(text) -> str`
   that collapses multi-space artifacts while preserving paragraph structure.
3. `rag_eval.py` supports `--chunking-strategy tree` alongside existing strategies. A
   run with `--chunking-strategy tree --retrievers bm25_sentence,dense_sentence` produces
   valid coverage output in `rag_pipeline_eval.html`.
4. All existing tests pass unchanged. New tests in `tests/test_adv_chunking.py` cover
   tree building, whitespace normalization, and tree-to-chunk conversion.

**Out of scope for S8:** changes to the `HierarchicalRetriever` (S7). S8 changes what
gets chunked; S7 changes how chunks are retrieved. Both are independently composable —
S7's `HierarchicalRetriever` can run on either sentence chunks or S8 tree chunks.

---

## Problem: Why Flat Chunking Loses Structure

Current sentence chunking splits at sentence boundaries. The clause:

```
2. APPOINTMENT OF RESELLER.

     2.1  NONEXCLUSIVE RESELLER.  Subject to applicable Legal Requirements:

            (a) TouchStar hereby appoints Reseller as its nonexclusive value-
                added reseller for the limited purposes of...
```

becomes three or more sentence-level chunks, each carrying `clause_path = ["2.1
NONEXCLUSIVE RESELLER"]` (the nearest section header detected by `clauses.py`). The
parent article (`2. APPOINTMENT OF RESELLER`) and the list membership (`(a)`) are both
lost.

With tree chunking the same passage becomes:

```python
ContractNode(
    level=2, pattern="top_numbered", heading="2. APPOINTMENT OF RESELLER.",
    children=[
        ContractNode(
            level=3, pattern="subsection", heading="2.1  NONEXCLUSIVE RESELLER.",
            children=[
                ContractNode(
                    level=4, pattern="alpha_paren_lower", heading="(a) TouchStar hereby...",
                    children=[]
                ),
            ]
        )
    ]
)
```

Each leaf chunk carries `clause_path = ["2. APPOINTMENT OF RESELLER",
"2.1 NONEXCLUSIVE RESELLER", "(a)"]`. Retrieval on any leaf finds the full ancestry,
and expansion to a parent node returns all sibling clauses automatically.

---

## Whitespace Normalization

### Problem

CUAD contracts are extracted from PDFs via SEC EDGAR. The raw text contains:
- Multi-space padding inside lines (e.g. `"TouchStar  hereby  appoints"`)
- Trailing spaces and mixed indentation that is not structurally meaningful
- Long sequences of spaces used as visual separators (`"     2.1  NONEXCLUSIVE..."`)

### Solution

`normalize_contract_whitespace(text: str) -> str` in `contracts.py`:

1. Split on lines.
2. For each line: collapse runs of 2+ interior spaces to one space; strip trailing
   whitespace; preserve leading indentation by collapsing only interior runs (not the
   leading whitespace that signals list-item depth).
3. Collapse runs of 3+ consecutive blank lines to exactly two blank lines (preserve
   paragraph breaks; remove excessive vertical space).
4. Return joined result.

This function is called once per contract during `build_contract_tree()` before any
pattern matching. It does **not** modify `SentenceSpan.raw_text` for existing sentence
chunking — normalization is opt-in per chunking strategy.

```python
def normalize_contract_whitespace(text: str) -> str:
    """Collapse multi-space OCR artifacts while preserving paragraph structure."""
    ...
```

---

## Data Structures

### `ContractNode`

New file: `src/cuad_agent/rag/tree.py`

```python
@dataclass
class ContractNode:
    level: int                    # 0-5 from HierarchyPattern.level
    pattern_name: str             # name from HIERARCHY_PATTERNS, or "body" for uncategorised
    heading: str                  # the line that opened this node (stripped)
    body_lines: list[str]         # non-heading lines directly under this node
    children: list[ContractNode]
    line_start: int               # 0-based index in the normalised line list
    line_end: int                 # exclusive; line_start == line_end means heading only
```

`heading` is the full stripped text of the line that triggered the node (e.g.
`"(a) TouchStar hereby appoints..."` — the list item IS the heading since list items
carry their own content on the same line).

`body_lines` holds lines that appear after the heading and before the first child node
opens — i.e. prose that belongs directly to this node rather than to a sub-heading.

### `ContractTree`

```python
@dataclass
class ContractTree:
    document_row_id: int
    title: str
    normalized_text: str          # output of normalize_contract_whitespace()
    roots: list[ContractNode]     # top-level nodes (level 0 or 1)
```

### Extended `RagChunk.chunk_type`

`chunks.py` gains one new Literal value:

```python
chunk_type: Literal["sentence", "legal_recursive", "hierarchical"]
```

No other fields change.

---

## Build Algorithm

`build_contract_tree(document_row_id, title, text) -> ContractTree`

```
1. Normalise text via normalize_contract_whitespace().
2. Split into lines.
3. For each line, call detect_line_type() from chunking_analysis.py.
   - Returns a HierarchyPattern (level 0-5) or None (body text).
4. Build the tree with a stack:
   - stack: list of open ContractNode objects, ordered from root to current.
   - For a heading line at level L:
       a. Pop all stack entries with level >= L (they are now closed).
       b. Create a new ContractNode(level=L, ...) with line_start = current line.
       c. Append as a child of the current top-of-stack (or to roots if stack empty).
       d. Push the new node onto the stack.
   - For a body line:
       b. Append to body_lines of the top-of-stack node (or to a synthetic
          level=-1 preamble node if the stack is empty — text before the first heading).
5. Close all remaining stack entries (set line_end).
6. Return ContractTree.
```

**Edge cases:**
- Text before the first heading: collected into `roots` as a synthetic node with
  `pattern_name="preamble"`, `level=-1`. This captures the parties / recitals block.
- Headings that jump levels (e.g. level 4 after level 1 with no level 2/3 in between):
  accepted as-is. The tree does not enforce level continuity.
- Empty document: returns `ContractTree(roots=[])`.

---

## Tree-to-Chunks Conversion

`tree_to_chunks(tree: ContractTree, *, include_intermediate: bool = False) -> list[RagChunk]`

Each `ContractNode` may become one `RagChunk`:

- **Leaf nodes** (`node.children == []`): always included. `text` = `node.heading +
  " " + " ".join(node.body_lines)` (stripped). `clause_path` = full ancestry headings
  from root to this node.
- **Intermediate nodes** (`node.children != []`): included only when
  `include_intermediate=True`. `text` = `node.heading` + direct `body_lines` only (not
  children's text). Useful for building a section-level retrieval layer.

`chunk_id` format: `"tree_{document_row_id}_{line_start}_{line_end}"`.

`clause_path` is constructed by walking from the root to the current node and collecting
`node.heading` values. This gives a list like:

```python
["2. APPOINTMENT OF RESELLER.", "2.1  NONEXCLUSIVE RESELLER.", "(a) TouchStar hereby..."]
```

The last element of `clause_path` is the node's own heading (its `section_title` is the
first heading element that looks like a title, `section_number` is the dotted number if
present). These fields are populated using the same extraction logic as `clauses.py`.

```python
def tree_to_chunks(
    tree: ContractTree,
    *,
    include_intermediate: bool = False,
) -> list[RagChunk]:
    """Walk ContractTree depth-first and emit RagChunk objects."""
    ...
```

---

## Changes to Existing Files

### `src/cuad_agent/rag/contracts.py`

Add `normalize_contract_whitespace()`. This file already exists. Do not change any
existing functions.

```python
def normalize_contract_whitespace(text: str) -> str:
    """Collapse multi-space OCR artifacts; preserve leading indent and paragraph breaks."""
    ...
```

### `src/cuad_agent/rag/chunks.py`

Extend `chunk_type` Literal:

```python
chunk_type: Literal["sentence", "legal_recursive", "hierarchical"]
```

Update `chunk_from_dict()` to accept `"hierarchical"` without raising.

### `src/cuad_agent/rag/experiments.py`

Add `"tree"` as a valid `chunking_strategy`:

```python
CHUNKING_STRATEGIES = {"sentence", "legal_recursive", "tree"}
```

Inside `build_sentence_store()` (or the equivalent chunk-building path), add a branch:

```python
if chunking_strategy == "tree":
    from cuad_agent.rag.tree import build_contract_tree, tree_to_chunks
    tree = build_contract_tree(
        document_row_id=row.document_row_id,
        title=row.title,
        text=row.context,
    )
    chunks = tree_to_chunks(tree, include_intermediate=False)
```

The resulting `chunks` list is passed into the existing BM25/dense index builders
unchanged — they consume `list[RagChunk]` and do not inspect `chunk_type`.

Add `chunking_strategy` to the `config` dict emitted by `run_rag_eval()`.

### `src/cuad_agent/rag/cli.py`

Add `--chunking-strategy` flag:

```python
parser.add_argument(
    "--chunking-strategy",
    choices=("sentence", "legal_recursive", "tree"),
    default="sentence",
    help="Chunking strategy. 'tree' builds a hierarchy-preserving tree per contract.",
)
```

Pass through to `run_rag_eval()`.

### `src/cuad_agent/rag/outputs.py`

No new section required for S8. The existing `write_pipeline_html()` already renders
coverage for any retriever method. If `chunking_strategy == "tree"` is present in the
config dict, add a one-line note in the run-config panel:
`"Chunking: hierarchical tree (S8)"`.

---

## New File: `src/cuad_agent/rag/tree.py`

```python
"""Hierarchy-preserving contract tree chunking."""

from __future__ import annotations

from dataclasses import dataclass, field

from cuad_agent.data.chunking_analysis import detect_line_type, HIERARCHY_PATTERNS
from cuad_agent.rag.chunks import RagChunk
from cuad_agent.rag.contracts import normalize_contract_whitespace


@dataclass
class ContractNode:
    level: int
    pattern_name: str
    heading: str
    body_lines: list[str] = field(default_factory=list)
    children: list[ContractNode] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0


@dataclass
class ContractTree:
    document_row_id: int
    title: str
    normalized_text: str
    roots: list[ContractNode] = field(default_factory=list)


def build_contract_tree(
    document_row_id: int,
    title: str,
    text: str,
) -> ContractTree:
    """Parse a contract into a ContractTree using HIERARCHY_PATTERNS."""
    ...


def tree_to_chunks(
    tree: ContractTree,
    *,
    include_intermediate: bool = False,
) -> list[RagChunk]:
    """Walk ContractTree depth-first and emit RagChunk objects.

    Leaves are always emitted. Intermediate nodes only when include_intermediate=True.
    clause_path contains all ancestor headings, root-first, ending with the node heading.
    """
    ...
```

`detect_line_type()` is imported directly from `chunking_analysis.py` — do not
duplicate the pattern definitions. This is the only file outside `data/` that imports
from `cuad_agent.data.chunking_analysis`.

---

## New Tests: `tests/test_adv_chunking.py`

All tests are pure in-memory — no disk I/O, no LLM, no embeddings.

| Test | What it checks |
|------|----------------|
| `test_normalize_collapses_interior_spaces` | `"foo  bar  baz"` → `"foo bar baz"` |
| `test_normalize_preserves_leading_indent` | `"    (a) foo  bar"` → `"    (a) foo bar"` |
| `test_normalize_collapses_blank_lines` | 4+ consecutive blank lines → 2 |
| `test_normalize_preserves_paragraph_breaks` | Two blank lines kept as two blank lines |
| `test_build_tree_top_level_section` | `"2. DEFINITIONS.\n..."` → root node with `level=2, pattern_name="top_numbered"` |
| `test_build_tree_nested_subsection` | Section + subsection → parent.children has one child with correct level |
| `test_build_tree_list_items_under_subsection` | `(a)`, `(b)`, `(c)` become leaf children of their parent |
| `test_build_tree_level_jump` | Heading at level 4 directly after level 1 accepted without error |
| `test_build_tree_preamble_before_first_heading` | Text before any heading → preamble node at level=-1 in roots |
| `test_build_tree_empty_text` | `build_contract_tree(0, "", "")` returns ContractTree with empty roots |
| `test_tree_to_chunks_leaves_only` | Default `include_intermediate=False` — no intermediate-level chunk emitted |
| `test_tree_to_chunks_include_intermediate` | `include_intermediate=True` — parent nodes also appear in output |
| `test_tree_to_chunks_clause_path_full_ancestry` | Leaf chunk has clause_path containing all ancestor headings |
| `test_tree_to_chunks_chunk_id_unique` | All chunk_ids in output are distinct |
| `test_tree_to_chunks_chunk_type` | Every chunk has `chunk_type == "hierarchical"` |
| `test_tree_chunk_text_includes_body_lines` | body_lines content appears in leaf chunk text |
| `test_three_level_contract` | Full 3-level contract (article → section → list items) produces correct tree and chunk count |

---

## Relationship to Existing Chunking Strategies

| Strategy | Where defined | Chunk unit | clause_path |
|----------|--------------|------------|-------------|
| `sentence` | `sentences.py` | One sentence | nearest section heading only |
| `legal_recursive` | `legal_recursive.py` | Token window ≤1200 chars | nearest section heading only |
| `tree` (S8) | `tree.py` | One structural node (leaf by default) | full ancestry, root → leaf |

S7's `HierarchicalRetriever` operates on any `list[RagChunk]` — it groups chunks by
`clause_path` tuple. Running S7 retrieval on S8 tree chunks gives richer section
grouping because `clause_path` is now the full ancestor chain rather than just the
nearest heading.

---

## Parameters and Defaults

| Parameter | Default | CLI flag | Notes |
|-----------|---------|----------|-------|
| `chunking_strategy` | `"sentence"` | `--chunking-strategy` | Existing default unchanged |
| `include_intermediate` | `False` | `--tree-include-intermediate` | Emit parent nodes as chunks too |

When `--chunking-strategy tree` is set, `--chunking-version` is automatically set to
`"tree-v1"` unless overridden. This ensures tree chunks are cached separately from
sentence chunks.

---

## Example Commands

```bash
# Coverage benchmark: tree chunking with BM25 and dense retrieval
uv run python rag_eval.py \
  --run-id s8-tree \
  --chunking-strategy tree \
  --retrievers bm25_sentence,dense_sentence \
  --top-k 20 \
  --contract-scope all

# Compare tree vs sentence chunking on the same retrievers
uv run python rag_eval.py \
  --run-id s8-compare \
  --chunking-strategy tree \
  --retrievers bm25_sentence,dense_sentence,bm25_hierarchical,dense_hierarchical \
  --top-k 20 \
  --contract-scope sample-50

# Inspect whitespace normalization on one contract (REPL)
from cuad_agent.data.dataset import load_datasets
from cuad_agent.rag.contracts import normalize_contract_whitespace
ds = load_datasets()
row = ds["contracts"].iloc[0]
print(normalize_contract_whitespace(row["context"])[:2000])
```

---

## Implementation Order

1. **`src/cuad_agent/rag/contracts.py`** — add `normalize_contract_whitespace()`.
   Write `test_normalize_*` tests. Run `uv run pytest tests/test_adv_chunking.py -k normalize -q`.

2. **`src/cuad_agent/rag/tree.py`** — `ContractNode`, `ContractTree`, `build_contract_tree()`.
   Write `test_build_tree_*` tests. All pass before continuing.

3. **`tree_to_chunks()`** in `tree.py`. Write `test_tree_to_chunks_*` and `test_three_level_contract`.
   All tests pass.

4. **`src/cuad_agent/rag/chunks.py`** — extend `chunk_type` Literal to include `"hierarchical"`.

5. **`src/cuad_agent/rag/experiments.py`** — add `"tree"` branch to chunk-building path;
   add `chunking_strategy` to config output.

6. **`src/cuad_agent/rag/cli.py`** — add `--chunking-strategy` and `--tree-include-intermediate` flags.

7. **Run full test suite** — `uv run pytest -q`. All existing tests pass.

8. **Smoke test** — `uv run python rag_eval.py --run-id s8-smoke --chunking-strategy tree
   --retrievers bm25_sentence --contract-scope sample-10`. Check `rag_pipeline_eval.html`
   renders without error and coverage numbers appear.

9. **`AGENTS.md`** — document `--chunking-strategy tree`, `normalize_contract_whitespace`,
   `ContractTree`, and example commands.

---

## Constraints

- `detect_line_type()` and `HIERARCHY_PATTERNS` are imported from `chunking_analysis.py`
  without modification. Do not duplicate pattern definitions.
- Existing `--chunking-strategy sentence` (default) is unchanged in behaviour.
- `normalize_contract_whitespace()` is not applied to existing sentence or legal-recursive
  paths — it is called only inside `build_contract_tree()`.
- `tree_to_chunks()` with `include_intermediate=False` must emit at least one chunk per
  contract even if the tree has only a root node (i.e. no heading was detected).
- No new pip dependencies. The tree builder uses only stdlib `re` and `dataclasses`,
  plus the existing `chunking_analysis.py` patterns.
- `chunk_from_dict()` in `chunks.py` must not raise on `chunk_type="hierarchical"` after
  this change — existing cached JSONL files with `"sentence"` or `"legal_recursive"` are
  unaffected.
