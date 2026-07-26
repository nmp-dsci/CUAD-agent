"""Synthesis step for autoresearch: generate an improved system prompt candidate."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from cuad_agent.autoresearch.llm import make_llm
from cuad_agent.autoresearch.prompts.synthesis_system_prompt import (
    SYNTHESIS_SYSTEM_PROMPT,
)
from cuad_agent.autoresearch.results import SynthesisResult, TriageDiagnosis

__all__ = ["synthesise"]


def _message_text(output: Any) -> str:
    content = getattr(output, "content", output)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def synthesise(
    *,
    category: str,
    current_prompt: str,
    diagnoses: list[TriageDiagnosis],
    history: list[dict],  # [{iter, status, notes, prompt_text}, ...]
    model_id: str,  # candidate model_id for this iteration — carried for traceability
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 64000,
    dry_run: bool = False,
) -> SynthesisResult:
    """Generate an improved system prompt candidate via one LLM call.

    Parameters
    ----------
    category:       the legal question category being improved
    current_prompt: the system prompt currently in use
    diagnoses:      TriageDiagnosis records for every wrong answer this iteration
    history:        prior iterations with iter, status, notes, and prompt_text
    model_id:       candidate model_id for traceability (not used for the LLM call)
    model:          LLM model string passed to make_llm (e.g. "deepseek/deepseek-v4-pro")
    temperature:    sampling temperature (default 0.0 for determinism)
    max_tokens:     maximum tokens for the LLM response
    dry_run:        if True, return a stub result without making an LLM call
    """
    if dry_run:
        return SynthesisResult(prompt_text="DRY RUN PROMPT", notes="dry run")

    parser = PydanticOutputParser(pydantic_object=SynthesisResult)
    system_content = SYNTHESIS_SYSTEM_PROMPT + "\n\n" + parser.get_format_instructions()

    diagnoses_json = json.dumps(
        [d.model_dump() for d in diagnoses],
        indent=2,
    )
    history_json = json.dumps(history, indent=2)

    human_content = (
        f"category: {category}\n\n"
        f"current_prompt:\n{current_prompt}\n\n"
        f"diagnoses:\n{diagnoses_json}\n\n"
        f"history:\n{history_json}"
    )

    llm = make_llm(model=model, temperature=temperature, max_tokens=max_tokens)
    output = llm.invoke(
        [SystemMessage(content=system_content), HumanMessage(content=human_content)]
    )
    return parser.parse(_message_text(output))
