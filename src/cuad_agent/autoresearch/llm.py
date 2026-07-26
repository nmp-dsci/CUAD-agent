"""LLM factory shared by triage and synthesis steps in autoresearch."""

from __future__ import annotations

import argparse

from langchain_core.language_models import BaseChatModel

from cuad_agent.agents.langchain_agent import configure_llm

__all__ = ["make_llm"]


def make_llm(
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 64000,
) -> BaseChatModel:
    """Build a LangChain chat model from a model string.

    Constructs an ``argparse.Namespace`` with *model*, *temperature*, and
    *max_tokens* attributes and delegates to ``configure_llm``.
    """
    args = argparse.Namespace(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return configure_llm(args)
