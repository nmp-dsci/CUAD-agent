"""Prompt module loading helpers."""

from __future__ import annotations

from pathlib import Path

from cuad_agent.evaluators.dspy_runner import load_prompt_overrides


def default_prompts_file_for_model_id(
    model_id: str,
    *,
    prompts_dir: Path = Path("prompts"),
) -> Path | None:
    """Return the conventional prompt file for a model id when it exists."""
    candidate = prompts_dir / f"system_prompts_{model_id}.py"
    return candidate if candidate.exists() else None


def resolve_prompts_file(
    prompts_file: Path | None,
    model_id: str,
    *,
    prompts_dir: Path = Path("prompts"),
) -> Path | None:
    """Prefer an explicit prompt file, otherwise load prompts by model id."""
    if prompts_file is not None:
        return prompts_file
    return default_prompts_file_for_model_id(model_id, prompts_dir=prompts_dir)


__all__ = [
    "default_prompts_file_for_model_id",
    "load_prompt_overrides",
    "resolve_prompts_file",
]
