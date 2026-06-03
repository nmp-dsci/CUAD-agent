"""Prompt override loading helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from cuad_agent.paths import slugify_model_id

__all__ = [
    "load_prompt_overrides",
    "resolve_prompts_file",
    "resolve_prompts_file_for_model_id",
]


def load_prompt_overrides(prompts_file: Path | None) -> dict[str, str]:
    if prompts_file is None:
        return {}
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_file}")

    spec = importlib.util.spec_from_file_location(
        f"cuad_prompt_overrides_{slugify_model_id(str(prompts_file))}",
        prompts_file,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load prompt file: {prompts_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = getattr(module, "CATEGORY_SYSTEM_PROMPTS", None)
    if not isinstance(prompts, dict):
        raise ValueError(
            f"{prompts_file} must define CATEGORY_SYSTEM_PROMPTS as a dict"
        )
    return {str(category): str(prompt) for category, prompt in prompts.items()}


def resolve_prompts_file_for_model_id(
    prompts_file: Path | None,
    model_id: str,
    *,
    prompts_dir: Path = Path("prompts"),
) -> Path | None:
    if prompts_file is not None:
        return prompts_file
    candidate = prompts_dir / f"system_prompts_{model_id}.py"
    return candidate if candidate.exists() else None


resolve_prompts_file = resolve_prompts_file_for_model_id
