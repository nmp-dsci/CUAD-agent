"""System prompt serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cuad_agent.paths import prompt_name_part
from cuad_agent.prompts.templates import build_agent_system_prompt

__all__ = ["write_system_prompts"]


def write_system_prompts(
    agents: dict[int, Any],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    lines = [
        '"""Generated CUAD v1 system prompts by category.',
        "",
        "Edit these strings to iterate on prompt variants, then wire the chosen",
        "prompt text back into the evaluator before running a new model_id.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
    ]
    mapping_entries: list[str] = []
    for question_index, agent in sorted(agents.items()):
        base_name = f"{prompt_name_part(agent.category)}_SYSTEM_PROMPT"
        prompt_name = base_name
        suffix = 2
        while prompt_name in used_names:
            prompt_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(prompt_name)
        lines.extend(
            [
                f"# Question {question_index + 1}: {agent.category}",
                f"{prompt_name} = {build_agent_system_prompt(agent)!r}",
                "",
            ]
        )
        mapping_entries.append(f"    {agent.category!r}: {prompt_name},")

    lines.extend(["CATEGORY_SYSTEM_PROMPTS = {", *mapping_entries, "}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
