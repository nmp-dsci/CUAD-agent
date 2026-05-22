---
name: prompt-eval-opt-harness
description: Build or adapt a generator/evaluator harness that improves system prompts from evaluation results and golden answers. Use when a user has per-example eval outputs, such as gold vs predicted answers plus scores or correctness, and wants an auditable prompt-improvement loop for one prompt, many task prompts, category prompts, routing prompts, extraction prompts, classification prompts, or agent instructions. The harness splits failures into disjoint generator/evaluator/holdout sets, proposes prompt patches, reviews them, writes candidate prompts and diffs, supports resume and parallel execution, and validates final quality by rerunning evals.
metadata:
  short-description: Improve system prompts from eval results
---

# Prompt Eval Opt Harness

Build a reusable evaluation-driven harness for improving system prompts from failures, without turning a pile of wrong examples into an unreviewed prompt rewrite.

## When To Use

Use when the user has:

- System prompts or instruction prompts that drive a model, agent, classifier, extractor, router, grader, or tool caller.
- Evaluation results with per-example gold answers, model outputs, and a metric or correctness flag.
- A need to improve prompts systematically, preserve an audit trail, and test candidates against held-out examples.

Trigger phrases include: "improve prompts from eval results", "gold vs predicted prompt loop", "prompt eval harness", "system prompt optimization", "generator evaluator prompt loop", "use failures to improve prompts".

## Core Pattern

### 1. Define the optimization unit

Choose the smallest prompt ownership boundary that can be improved independently:

- One global system prompt.
- Per-task, per-category, per-label, per-route, per-tool, per-document-type, or per-agent prompts.
- Prompt sections inside a larger composed prompt, such as base instructions plus task-specific overlays.

Call this unit `prompt_key` in the harness. Do not hard-code domain terms like `category` unless the project already uses them.

### 2. Normalize eval records

Create a common internal row shape:

```python
class EvalFailure(BaseModel):
    row_id: str
    prompt_key: str
    input_summary: str
    gold_answer: str
    predicted_answer: str
    score: float
    correct: bool
    failure_mode: str
    metadata: dict[str, Any] = {}
```

If source evals use `(document_id, question_id)` or another composite key, derive a stable `row_id`. Keep enough metadata to show useful examples in audit artifacts, but do not pass full private data to LLMs unless needed.

### 3. Split failures into disjoint sets

The generator and evaluator must see different failures:

- `generator_dev`: visible only to the generator.
- `evaluator_dev`: visible only to the evaluator.
- `holdout_eval`: never seen in the prompt-development loop; used for final measurement.

For each `prompt_key`, sort row ids deterministically and split with a fixed size or seed. If there are too few failures for all three sets, keep generator/evaluator disjoint and let holdout be empty.

### 4. Classify failure modes before patching

Do not ask an LLM to rewrite from raw failures alone. Create a domain-specific failure taxonomy and count modes per `prompt_key`.

Common starting buckets:

- `false_negative`: gold answer exists but model missed it.
- `false_positive`: model produced an answer when it should not.
- `wrong_label`: classification/routing label is wrong.
- `partial_answer`: output is close but incomplete.
- `overbroad_answer`: output includes too much irrelevant content.
- `format_error`: output violates schema or expected form.
- `reasoning_error`: conclusion does not follow from evidence.
- `tool_or_policy_error`: model used the wrong tool, policy, or procedure.
- `gold_or_metric_issue`: eval data or scorer appears wrong.

The taxonomy should reflect the task. Extraction tasks may need span-specific buckets; tool agents may need tool-selection and argument-shape buckets.

### 5. Derive answer/output profiles from gold

Infer expected output shape from the gold data, not only from labels or docs:

- Label only, free text, JSON/schema, exact span, multiple answers, ranked list, tool call, or no-answer.
- Typical length and structure.
- Whether no-answer/null/refusal is allowed.
- Required evidence style or citation format.
- For extraction tasks, whether gold answers are verbatim substrings of source documents.

Pass this profile to both generator and evaluator.

### 6. Use typed generator and evaluator agents

Use structured outputs with Pydantic or equivalent.

```python
class PromptPatch(BaseModel):
    prompt_key: str
    failure_analysis: list[str]
    revised_prompt: str
    expected_improvements: list[str]
    regression_risks: list[str]
    changed_rules: list[str]
    prompt_diff_summary: list[str]


class PromptReview(BaseModel):
    decision: Literal["accept", "revise", "reject"]
    generalization_score: float = Field(ge=0.0, le=1.0)
    rationale: list[str]
    likely_fixes: list[str]
    likely_regressions: list[str]
    requested_changes: list[str]
```

If the model/provider does not support tool calling, use JSON schema/json mode or a parser with format instructions. With pydantic-ai, prefer `PromptedOutput(SchemaModel)` for broad provider compatibility.

### 7. Keep examples disjoint during revise loops

Run up to a small fixed number of loops, usually 2 or 3:

1. Generator sees current prompt, output profile, failure-mode summary, and `generator_dev` examples.
2. Evaluator sees proposed prompt and separate `evaluator_dev` examples.
3. On `revise`, pass only evaluator feedback/requested changes back to the generator, not evaluator examples.
4. On `accept` or `reject`, stop.

For practical candidate generation, still write the latest generated prompt for `accept`, `revise`, and `reject`; preserve the decision as a confidence/review signal. On reruns, treat prior `revise` records as incomplete so they are processed again.

### 8. Resume safely

Write one status record per `prompt_key` as each unit completes:

```json
{"prompt_key": "termination_clause", "decision": "revise", "candidate_prompt": "...", "completed_at": "..."}
```

On startup:

- Skip prior `accept`, `reject`, and `no_failures` records unless the user requests a retry.
- Rerun prior `revise` records.
- Do not append status for failed exceptions; let those units retry next run.
- If processing in parallel, append under a lock and merge results in the main thread.

### 9. Write auditable artifacts

At minimum:

- `splits.json`: generator/evaluator/holdout row ids.
- `status.jsonl`: resume state per `prompt_key`.
- `runs.jsonl`: full per-loop requests, patches, and reviews.
- `reviews.jsonl`: evaluator decisions and rationale.
- `accepted_patches.jsonl` and `rejected_patches.jsonl` or a unified `patches.jsonl` with decisions.
- `prompt_diffs.jsonl`: v1 vs candidate prompt diffs.
- Candidate prompt artifact in the project-native format, such as `.py`, `.json`, `.yaml`, or `.md`.
- Optional static HTML dashboard showing all prompt keys, including unchanged/skipped ones, with v1 vs candidate prompts and change insights.

### 10. Validate by rerunning evals

The loop decision is not the final quality metric. After generating candidate prompts:

1. Rerun the original evaluation on `holdout_eval`, or on a fresh eval set if holdout is small.
2. Compare baseline vs candidate by overall metric and per-`prompt_key` deltas.
3. Inspect regressions on previously strong prompts.
4. Accept only if measured eval results justify promotion.

## Generator Prompt Shape

Include:

1. Current system prompt or prompt section.
2. Prompt key and task metadata.
3. Derived output profile from gold answers.
4. Failure-mode counts.
5. Generator-only examples: input summary, gold, predicted, score, failure mode.
6. Previous generated prompt and evaluator feedback on revise loops.
7. Instructions to patch only evidence-supported behavior and keep prompt length under control.
8. Structured return schema.

System prompt template:

```text
You improve system prompts using evaluation evidence.
Patch only what the failures support.
Keep the prompt concise and preserve working behavior.
Use only the generator_dev examples provided.
Do not assume access to evaluator_dev or holdout_eval examples.
```

## Evaluator Prompt Shape

Include:

1. Current prompt and proposed prompt.
2. Generator instructions.
3. Prompt key and task metadata.
4. Output profile.
5. Evaluator-only examples.
6. Review criteria: likely generalization, regressions, unsupported broad rewrites, prompt bloat, format drift.
7. Structured return schema with `accept | revise | reject`.

System prompt template:

```text
You are a skeptical prompt evaluator.
Review whether a proposed prompt patch is likely to generalize to unseen failures.
Prefer requesting revision or rejection for broad rewrites, prompt bloat, unsupported assumptions, or changes that risk regressions.
Use only evaluator_dev examples and do not reveal them back to the generator.
```

## Anti-Patterns

- Sharing examples between generator and evaluator.
- Treating evaluator confidence as a measured quality score.
- Optimizing only aggregate metrics and hiding per-prompt regressions.
- Rewriting the whole prompt when a narrow rule would fix the failure mode.
- Trusting declared answer format when gold answers show a different output shape.
- Skipping holdout or fresh-set validation.
- Failing to write candidate prompts for lower-confidence `revise` outputs when the user wants prompts to test.
- Skipping `revise` statuses forever on rerun.

## Adaptation Notes

For extraction: profile exact spans, no-answer behavior, and span shape.

For classification/routing: profile label set, confusable labels, abstain behavior, and decision boundaries.

For structured JSON: profile schema violations, missing fields, invalid enum values, and extra fields.

For tool agents: profile wrong tool, missing tool, bad arguments, premature final answers, and policy/procedure failures.

For graders/evaluators: profile false pass, false fail, rubric mismatch, calibration drift, and missing evidence.

## Reference Pattern

The CUAD-agent project contains a concrete implementation in `prompt_improve_v2.py`. Treat it as an example of the pattern, not as the canonical schema:

- It uses `question_index/category` where this generalized skill uses `prompt_key`.
- It derives answer-format profiles from gold answers.
- It writes candidate prompts even when the evaluator decision is `revise` or `reject`.
- It reruns prior `revise` statuses.
- It writes a dashboard showing all prompt keys, including skipped and unchanged ones.

When adapting it, rename domain fields, failure modes, prompt artifact format, and evaluator commands to match the target project.
