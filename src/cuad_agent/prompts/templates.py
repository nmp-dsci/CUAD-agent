"""Prompt composition templates."""

from __future__ import annotations

from typing import Any

from cuad_agent.constants import NO_ANSWER

__all__ = [
    "build_agent_system_prompt",
    "compose_system_prompt",
]


def compose_system_prompt(
    *,
    question: str,
    category: str,
    category_description: str,
    answer_format: str,
) -> str:
    answer_format = answer_format or "No fixed answer format"
    return "\n".join(
        [
            "You are a legal contract review assistant evaluating CUAD clauses.",
            "",
            "Task:",
            question,
            "",
            "Category:",
            category,
            "",
            "Category description:",
            category_description,
            "",
            "Answer format:",
            answer_format,
            "",
            "Instructions:",
            "- Read the provided contract title and contract text.",
            "- Return exact answer text span(s) from the contract when present.",
            "- Separate multiple answer spans with newlines.",
            f"- Return {NO_ANSWER} when the contract does not contain an answer.",
            "- Set marked_impossible to true only when no answer is present.",
        ]
    )


def build_agent_system_prompt(agent: Any) -> str:
    if agent.system_prompt:
        return agent.system_prompt
    return compose_system_prompt(
        question=agent.question,
        category=agent.category,
        category_description=agent.category_description,
        answer_format=agent.answer_format,
    )
