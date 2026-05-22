# CUAD-Agent

Legal AI agent and evaluation harness built on the CUAD contract-understanding dataset.

**Read [`AGENTS.md`](AGENTS.md) before working in this project.** It covers project layout, architecture, common commands, environment variables, evaluation outputs, and testing.

## Quick reference

- **Package manager:** `uv` — use `uv sync`, `uv add`, `uv run`
- **Linting/formatting:** Ruff — `uv run ruff format . && uv run ruff check . --fix`
- **Tests:** `uv run pytest -q` (no API keys needed — all tests use dry-run mode)
- **Entry points:** `explore.py`, `dspy_eval_v1.py`, `prompt_improve_v2.py`, `rag_eval.py`
- **API keys:** `DEEPSEEK_API_KEY` (DeepSeek models), `OPENAI_API_KEY` (OpenAI models) — store in `.env`, never commit
