# autoresearch — Autonomous System Prompt Optimisation Loop

`autoresearch.py` implements an autonomous loop that iteratively improves the
system prompts used by the CUAD contract-review agent without human intervention.

## What it does

1. **Baseline eval** — runs the current system prompt against a held-out slice of
   the CUAD evaluation set and records per-question accuracy.

2. **Triage** — for each wrongly-answered contract, an LLM diagnoses *why* the
   current system prompt failed: it locates the golden-answer text in the contract,
   identifies the structural failure (missed clause type, wrong granularity, etc.),
   and proposes a concrete new rule tied to the actual clause language.

3. **Synthesise** — a second LLM call aggregates the triage diagnoses for a single
   category into a revised system prompt, incorporating the proposed rules while
   preserving rules that are already working.

4. **Validate** — the candidate prompt is evaluated on the same slice.  Accuracy
   is compared against the baseline.

5. **Keep / discard** — if accuracy improves (or holds within tolerance), the
   candidate prompt replaces the baseline for the next iteration.  Otherwise it is
   discarded and the loop continues with the next category.

Each iteration writes a row to a TSV results log and, on acceptance, saves the
winning prompt as a `candidate.py` module under
`src/cuad_agent/autoresearch/prompts/`.
