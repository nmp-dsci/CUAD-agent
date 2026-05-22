# S4 Project Refactor Plan

## Goal

Restructure the CUAD-agent project from a root-level script prototype into a
maintainable evaluation platform for legal-contract agents. The refactor should
keep the existing commands working while moving reusable logic into a coherent
Python package with clear boundaries for data loading, evaluation, prompt
optimization, dashboards, and future RAG work.

This plan is intentionally implementation-ready, but should be reviewed before
execution. The first refactor pass should preserve behavior and outputs rather
than redesign the evaluation approach.

## Current State

The current project works, but most production logic lives in large root
scripts:

- `explore.py` mixes CUAD loading, validation, summary printing, and ad hoc
  diagnostics.
- `dspy_eval_v1.py` contains shared constants, sampling, prompt loading,
  metrics, DSPy agent construction, HTML rendering, summary writing, and CLI
  orchestration.
- `langchain_agent.py` imports many shared utilities from `dspy_eval_v1.py`,
  which makes the LangChain evaluator depend on the DSPy script as a utility
  module.
- `prompt_improve_v2.py` combines prompt patch schemas, split management,
  failure analysis, PydanticAI agent calls, dashboard rendering, and CLI
  orchestration.
- Generated HTML is centralized under `frontend/`.
- Tests cover important behavior, but test imports are tied to root module
  names and monolithic files.

The main issue is not correctness; it is change isolation. A small change to
HTML, prompt handling, or caching currently requires touching broad files that
also own unrelated behavior.

## Refactor Principles

1. Preserve working CLI commands during migration.
2. Move shared behavior before changing behavior.
3. Keep root scripts as compatibility wrappers until docs and tests are fully
   migrated.
4. Separate framework-specific evaluator code from framework-neutral evaluation
   contracts, metrics, output paths, and dashboard rendering.
5. Treat generated artifacts as run outputs, not source modules.
6. Keep prompt versions explicit: `v1`, `v2`, and future variants should map
   cleanly to `outputs/{model_id}` and `frontend/*_{model_id}.html`.
7. Add RAG and contract chunking under new modules without entangling them with
   evaluation runners.

## Proposed Project Structure

```text
CUAD-agent/
├── src/
│   └── cuad_agent/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── paths.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── schemas.py
│       │   └── sampling.py
│       ├── prompts/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── writer.py
│       │   └── templates.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── examples.py
│       │   ├── metrics.py
│       │   ├── results.py
│       │   ├── summary.py
│       │   ├── cache.py
│       │   └── comparison.py
│       ├── evaluators/
│       │   ├── __init__.py
│       │   ├── dspy_runner.py
│       │   └── langchain_runner.py
│       ├── prompt_optimization/
│       │   ├── __init__.py
│       │   ├── schemas.py
│       │   ├── splits.py
│       │   ├── failure_analysis.py
│       │   ├── generator.py
│       │   ├── reviewer.py
│       │   ├── harness.py
│       │   └── artifacts.py
│       ├── dashboards/
│       │   ├── __init__.py
│       │   ├── evaluation.py
│       │   ├── prompt_review.py
│       │   └── static_assets.py
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── parsing.py
│       │   ├── chunking.py
│       │   ├── indexing.py
│       │   └── retrieval.py
│       └── cli/
│           ├── __init__.py
│           ├── explore.py
│           ├── dspy_eval.py
│           ├── langchain_eval.py
│           └── prompt_improve.py
├── scripts/
│   ├── explore.py
│   ├── dspy_eval_v1.py
│   ├── langchain_agent.py
│   ├── langchain_eval_v2.py
│   └── prompt_improve_v2.py
├── prompts/
│   ├── system_prompts_v1.py
│   └── system_prompts_v2.py
├── data/
│   ├── CUADv1.json
│   ├── category_descriptions.csv
│   ├── train_separate_questions.json
│   └── test.json
├── frontend/
│   ├── evaluation_v1.html
│   ├── evaluation_v2.html
│   └── prompt_review_v2.html
├── outputs/
│   └── {model_id}/
│       ├── cuad_dspy_eval_results.csv
│       ├── cuad_dspy_eval_results.jsonl
│       ├── cuad_dspy_eval_summary.json
│       ├── baseline_comparison.json
│       ├── system_prompts.py
│       └── prompt_harness/
├── tests/
│   ├── test_data.py
│   ├── test_eval_metrics.py
│   ├── test_eval_outputs.py
│   ├── test_langchain_eval.py
│   ├── test_prompt_optimization.py
│   └── test_dashboards.py
├── planning/
│   ├── s1-prp-dspy-eval.md
│   ├── s2-prp-dspy-rag.md
│   ├── s3-v2-prompt-opt.md
│   └── s4-project-refactor.md
├── pyproject.toml
├── readme.md
└── AGENTS.md
```

## Compatibility Strategy

Keep the current root commands working during the migration:

```bash
uv run python explore.py
uv run python dspy_eval_v1.py
uv run python langchain_agent.py
uv run python langchain_eval_v2.py
uv run python prompt_improve_v2.py
```

The root files should become thin wrappers that import from
`cuad_agent.cli.*`. For example:

```python
from cuad_agent.cli.langchain_eval import main

if __name__ == "__main__":
    main()
```

After wrappers are in place, add optional console scripts in `pyproject.toml`:

```toml
[project.scripts]
cuad-explore = "cuad_agent.cli.explore:main"
cuad-eval-dspy = "cuad_agent.cli.dspy_eval:main"
cuad-eval-langchain = "cuad_agent.cli.langchain_eval:main"
cuad-improve-prompts = "cuad_agent.cli.prompt_improve:main"
```

## Module Ownership

### `cuad_agent.data`

Owns dataset loading and validation only.

- Move `load_json`, `resolve_path`, `make_contracts_df`,
  `make_questions_df`, `load_category_descriptions`,
  `validate_question_category_order`, `join_category_descriptions`, and
  `load_datasets` from `explore.py`.
- Move deterministic sampling from `dspy_eval_v1.py` into `sampling.py`.
- Keep dataframe output shape stable.

### `cuad_agent.eval`

Owns framework-neutral evaluation contracts.

- `metrics.py`: `normalize_answer`, `tokens`, `parse_bool`,
  `token_overlap_f1`, `cuad_overlap_metric` if it can be generalized.
- `examples.py`: `answer_texts`, `evaluation_row_id`, `build_devset`,
  eval split filtering.
- `results.py`: result dataframe schemas, row construction, sorting.
- `summary.py`: `summarize_results`.
- `cache.py`: CSV/JSONL resume logic currently in `langchain_agent.py`.
- `comparison.py`: v1/v2 baseline comparison, row-level answer deltas.

### `cuad_agent.prompts`

Owns prompt composition, loading, and writing.

- `templates.py`: `compose_system_prompt`, default CUAD system prompt shape.
- `loader.py`: `load_prompt_overrides`.
- `writer.py`: `write_system_prompts`, `prompt_name_part`.

Generated prompt modules remain under `outputs/{model_id}/system_prompts.py`
and reviewed source prompts remain under `prompts/`.

### `cuad_agent.evaluators`

Owns framework-specific runners.

- `dspy_runner.py`: DSPy signatures, modules, LM setup, DSPy evaluation loop.
- `langchain_runner.py`: LangChain schema, chain construction, parsing, batch
  execution, and incremental cache writes.

Both runners should depend on `cuad_agent.eval`, `cuad_agent.data`, and
`cuad_agent.prompts`, not on each other.

### `cuad_agent.prompt_optimization`

Owns the prompt improvement harness.

- `schemas.py`: Pydantic models such as `FailureExample`, `PromptPatch`,
  `PromptReview`, dashboard records.
- `splits.py`: deterministic split creation and loading.
- `failure_analysis.py`: error mode inference, answer format profiles, span
  shape analysis.
- `generator.py`: patch generation agent and deterministic fallback.
- `reviewer.py`: patch review agent and deterministic fallback.
- `harness.py`: category processing orchestration, resume semantics,
  revise/reject/accept handling.
- `artifacts.py`: JSONL writers, prompt diffs, status logs.

### `cuad_agent.dashboards`

Owns all static HTML generation.

- `evaluation.py`: current `build_evaluation_page_data`,
  `render_evaluation_html`, and `write_evaluation_html`.
- `prompt_review.py`: current prompt review dashboard rendering.
- `static_assets.py`: shared colors, layout CSS, escaping helpers, JSON
  embedding helpers.

This keeps `frontend/evaluation_{model_id}.html` and
`frontend/prompt_review_{model_id}.html` as the only default HTML outputs.

### `cuad_agent.rag`

Prepared for future contract RAG work without blocking this refactor.

- `parsing.py`: adapters for Docling, Unstructured, PyMuPDF4LLM, Marker, or
  MinerU.
- `chunking.py`: clause-aware, section-aware, semantic, and parent-child
  chunking.
- `indexing.py`: vector/BM25 index construction.
- `retrieval.py`: hybrid retrieval and reranking interfaces.

The RAG package should not be used by the current evaluation loop until there
is a dedicated evaluation experiment.

## Artifact Policy

Source-controlled or source-like:

- `src/cuad_agent/**`
- `tests/**`
- `prompts/system_prompts_v*.py`
- `.codex/skills/**`
- `planning/**`
- `readme.md`
- `AGENTS.md`

Generated and reviewable:

- `outputs/{model_id}/**`
- `frontend/evaluation_{model_id}.html`
- `frontend/prompt_review_{model_id}.html`

Legacy root generated HTML was moved or removed:

- `explore.html` moved to `frontend/explore.html`.
- `evaluation.html` was a duplicate of `frontend/evaluation_v1.html` and was
  removed.

## Execution Plan

### Phase 0: Safety Baseline

1. Run the current test suite:

   ```bash
   uv run python -m pytest -q
   ```

2. Compile current entrypoints:

   ```bash
   uv run python -m py_compile explore.py dspy_eval_v1.py langchain_agent.py langchain_eval_v2.py prompt_improve_v2.py
   ```

3. Record current generated outputs:

   ```text
   frontend/evaluation_v1.html
   frontend/evaluation_v2.html
   frontend/prompt_review_v2.html
   outputs/v1/**
   outputs/v2/**
   ```

### Phase 1: Package Scaffold

1. Add `src/cuad_agent/` and empty package modules.
2. Update `pyproject.toml` for src-layout packaging:

   ```toml
   [tool.uv]
   package = true

   [tool.setuptools.packages.find]
   where = ["src"]
   ```

3. Do not move behavior yet; verify imports and tests still pass.

### Phase 2: Move Framework-Neutral Utilities

Move stable utilities first:

- constants and path helpers
- model id slugging and output path generation
- prompt loading/writing helpers
- token overlap metric helpers
- eval split loading/filtering
- summary generation
- dashboard rendering

Update root scripts to import these utilities from `cuad_agent.*`.

Success criteria:

- Existing CLI commands still work.
- `uv run python -m pytest -q` passes.
- `frontend/evaluation_v2.html` can be regenerated unchanged in shape.

### Phase 3: Decouple LangChain From DSPy Script

Remove `from dspy_eval_v1 import ...` from `langchain_agent.py`.

`langchain_agent.py` should import only:

- `cuad_agent.data.*`
- `cuad_agent.eval.*`
- `cuad_agent.prompts.*`
- `cuad_agent.dashboards.evaluation`

Then move LangChain-specific classes/functions into
`cuad_agent.evaluators.langchain_runner`.

Keep `langchain_agent.py` and `langchain_eval_v2.py` as wrappers.

Success criteria:

- `uv run python langchain_agent.py --dry-run --smoke-test --no-baseline-comparison`
  writes a smoke output and HTML.
- Resume cache behavior still works with CSV + JSONL.
- v1/v2 comparison still appears in `frontend/evaluation_v2.html`.

### Phase 4: Move DSPy Runner

Move DSPy-specific signature/module/evaluation code into
`cuad_agent.evaluators.dspy_runner`.

Keep `dspy_eval_v1.py` as a wrapper.

Success criteria:

- Dry-run DSPy evaluation still writes the same result schema.
- Prompt override tests still validate signature docstrings.
- `outputs/{model_id}/system_prompts.py` still writes valid
  `CATEGORY_SYSTEM_PROMPTS`.

### Phase 5: Split Prompt Optimization Harness

Move `prompt_improve_v2.py` into the prompt optimization package in small
chunks:

1. Pydantic schemas.
2. Failure analysis.
3. Split management.
4. Generator/reviewer agents.
5. Artifact writers.
6. Dashboard renderer.
7. CLI orchestration.

Keep `prompt_improve_v2.py` as a wrapper.

Success criteria:

- Dry-run harness writes:
  - `splits.json`
  - `answer_format_profiles.json`
  - `category_runs.jsonl`
  - accepted/rejected/revise logs
  - `prompts_candidate_v2.py`
  - `frontend/prompt_review_v2.html`
- Revise decisions remain rerunnable.
- Latest candidate prompt still updates even when decision is not accept.
- Dashboard still includes all questions, including skipped categories.

### Phase 6: Test Restructure

Split `tests/test_cuad_dspy_eval.py` into focused files:

- `test_data.py`
- `test_eval_metrics.py`
- `test_eval_outputs.py`
- `test_dashboards.py`
- `test_langchain_eval.py`
- `test_prompt_optimization.py`

Keep tests behavior-oriented, not file-location-oriented.

Success criteria:

- Tests import package modules directly.
- Wrappers are covered by at least smoke-level compile tests.

### Phase 7: Docs And Cleanup

Update:

- `readme.md`
- `AGENTS.md`
- skill docs if command paths change

Document both old and new commands during transition:

```bash
uv run python langchain_agent.py
uv run cuad-eval-langchain
```

Cleanup candidates after migration:

- remove stale `__pycache__` directories from the working tree if desired
- keep all generated `.html` artifacts under `frontend/`

Do not delete generated artifacts until their replacements have been verified.

## Suggested First Implementation Slice

The first execution slice should be deliberately small:

1. Create `src/cuad_agent/constants.py`, `paths.py`, `eval/metrics.py`,
   `eval/examples.py`, `prompts/loader.py`, `prompts/writer.py`, and
   `dashboards/evaluation.py`.
2. Move only framework-neutral functions from `dspy_eval_v1.py`.
3. Update `dspy_eval_v1.py` and `langchain_agent.py` imports.
4. Run tests.

This produces immediate value by breaking the DSPy-to-LangChain utility
dependency without touching the risky LLM execution paths.

## Risk Register

| Risk | Mitigation |
| --- | --- |
| Import path breakage under `uv run python script.py` | Keep root wrappers and add src-layout packaging before moving CLIs. |
| Dashboard regression due to embedded JS f-strings | Move rendering to `dashboards/`, add tests for comparison labels and row-level v1/v2 answer columns. |
| Resume cache schema drift | Centralize result schema and JSONL append/load behavior in `eval/cache.py`. |
| Prompt harness behavior changes during split | Move schemas and pure functions first; keep orchestration intact until the final step. |
| Generated artifact churn | Do not regenerate full LLM outputs during refactor; use dry-run and existing CSV/summary files. |
| Root command muscle memory | Preserve wrappers through at least one stable release of the refactor. |

## Definition Of Done

The refactor is complete when:

- Root commands still work.
- Package console commands work.
- `langchain_agent.py` no longer imports from `dspy_eval_v1.py`.
- DSPy and LangChain runners share framework-neutral utilities from
  `cuad_agent.eval`, not from each other.
- Dashboard rendering lives under `cuad_agent.dashboards`.
- Prompt optimization logic lives under `cuad_agent.prompt_optimization`.
- Tests are split by concern and all pass.
- Docs describe the new structure and output locations.
- Existing v1/v2 evaluation artifacts remain reviewable.

## Execution Notes

Implemented on 2026-05-12 as a behavior-preserving package migration:

- Added `src/cuad_agent/` package structure with `data`, `eval`,
  `evaluators`, `prompts`, `prompt_optimization`, `dashboards`, `rag`, and
  `cli` packages.
- Moved active runnable logic behind package modules:
  - `src/cuad_agent/data/dataset.py`
  - `src/cuad_agent/evaluators/dspy_runner.py`
  - `src/cuad_agent/evaluators/langchain_runner.py`
  - `src/cuad_agent/prompt_optimization/harness.py`
- Kept root entrypoints as compatibility wrappers:
  - `explore.py`
  - `dspy_eval_v1.py`
  - `langchain_agent.py`
  - `langchain_eval_v2.py`
  - `prompt_improve_v2.py`
- Added package console commands:
  - `cuad-explore`
  - `cuad-eval-dspy`
  - `cuad-eval-langchain`
  - `cuad-improve-prompts`
- Added façade modules for shared concerns:
  - `cuad_agent.constants`
  - `cuad_agent.paths`
  - `cuad_agent.eval.*`
  - `cuad_agent.prompts.*`
  - `cuad_agent.dashboards.*`
- Added `scripts/` snapshots of the pre-refactor root scripts for reference.
- Narrowed the PydanticAI dependency from the umbrella `pydantic-ai` package to
  `pydantic-ai-slim[openai]` to avoid pulling unused provider extras.
- Added `tool.uv.environments` to avoid resolving unsupported Windows
  environments for the current dependency set.
- Updated `readme.md` and `AGENTS.md` for the new structure.
- Moved remaining root HTML artifacts into the frontend policy:
  - `explore.html` -> `frontend/explore.html`
  - removed duplicate root `evaluation.html`

Verification:

```bash
uv run python -m py_compile $(rg --files src -g '*.py') \
  explore.py dspy_eval_v1.py langchain_agent.py langchain_eval_v2.py \
  prompt_improve_v2.py
uv run python -m pytest -q
```

Result:

```text
21 passed, 11 warnings
```

Remaining technical debt:

- `cuad_agent.eval.*`, `cuad_agent.prompts.*`, and
  `cuad_agent.dashboards.*` are façade modules over the behavior-preserving
  runner modules. A future cleanup can physically move each function into its
  owning module now that imports and wrappers are stable.
- Tests are still concentrated in `tests/test_cuad_dspy_eval.py`; split them by
  concern in a later low-risk pass.
- Update readme.md and agents.md with project structure, how to run codes etc 
