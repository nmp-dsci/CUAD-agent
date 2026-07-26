"""LangChain per-category CUAD agent.

Defines the structured-output schema (``CuadAnswer``), the chain factory, the
``ContractQuestionAgent`` execution unit, the agent-construction helper, and
LLM configuration. This is the LangChain-specific layer; the surrounding
orchestration, caching, and CLI live in ``evaluators.langchain_runner``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from cuad_agent.constants import EVAL_QUESTION_INDICES, NO_ANSWER, NO_ANSWER_MARKERS
from cuad_agent.prompts.templates import compose_system_prompt

__all__ = [
    "CuadAnswer",
    "ContractQuestionAgent",
    "build_agents",
    "build_chain_for_agent",
    "configure_llm",
    "make_messages_with_hint",
    "message_text",
    "parse_cuad_answer",
]


class CuadAnswer(BaseModel):
    """Structured output schema for CUAD clause extraction."""

    reasoning: str = Field(
        description=(
            "Brief step-by-step reasoning that identifies the supporting "
            "clause(s) in the contract before producing the final answer."
        )
    )
    answer: str = Field(
        description=(
            "Exact answer text span(s) from the contract, separated by "
            "newlines if multiple; or NO_ANSWER if no supporting clause exists."
        )
    )
    marked_impossible: bool = Field(
        description=(
            "True only when the contract does not contain an answer for this "
            "category/question; otherwise false."
        )
    )


def build_chain_for_agent(llm: BaseChatModel, system_prompt: str) -> Runnable:
    """Build a per-category chain that turns input fields into a typed CuadAnswer.

    Uses a PydanticOutputParser instead of `with_structured_output()` because the
    latter defaults to OpenAI tool calling, which DeepSeek's reasoner endpoints
    (e.g. deepseek-v4-flash) reject. The parser approach embeds the JSON schema
    in the system prompt and validates the response client-side, so it works on
    any chat model regardless of tool/function-calling support.

    Messages are constructed manually instead of via ChatPromptTemplate so that
    contract text containing literal curly braces does not break f-string-style
    template substitution.
    """
    parser = PydanticOutputParser(pydantic_object=CuadAnswer)
    full_system = f"{system_prompt}\n\n{parser.get_format_instructions()}"

    def make_messages(inputs: dict[str, Any]) -> list[Any]:
        user_content = (
            f"Contract title:\n{inputs['contract_title']}\n\n"
            f"Contract text:\n{inputs['contract_text']}\n\n"
            f"Category:\n{inputs['category']}\n\n"
            f"Category description:\n{inputs['category_description']}\n\n"
            f"Answer format:\n{inputs['answer_format']}"
        )
        return [
            SystemMessage(content=full_system),
            HumanMessage(content=user_content),
        ]

    return RunnableLambda(make_messages) | llm


def message_text(output: Any) -> str:
    content = getattr(output, "content", output)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def parse_cuad_answer(output: Any) -> CuadAnswer:
    text = message_text(output).strip()
    parser = PydanticOutputParser(pydantic_object=CuadAnswer)
    try:
        return parser.parse(text)
    except Exception:
        normalized = text.strip().lower()
        marked_impossible = normalized in NO_ANSWER_MARKERS
        return CuadAnswer(
            reasoning="unstructured_model_output",
            answer=NO_ANSWER if marked_impossible else text,
            marked_impossible=marked_impossible,
        )


class ContractQuestionAgent:
    """One per-category agent wrapping a LangChain chain."""

    def __init__(
        self,
        *,
        question_index: int,
        category: str,
        question: str,
        category_description: str,
        answer_format: str,
        system_prompt: str,
        chain: Runnable | None,
        dry_run: bool,
    ) -> None:
        self.question_index = question_index
        self.category = category
        self.question = question
        self.category_description = category_description
        self.answer_format = answer_format
        self.system_prompt = system_prompt
        self.chain = chain
        self.dry_run = dry_run

    def _dry_run_prediction(self, gold_answers: list[str]) -> CuadAnswer:
        return CuadAnswer(
            reasoning="dry_run",
            answer="\n".join(gold_answers) if gold_answers else NO_ANSWER,
            marked_impossible=not bool(gold_answers),
        )

    def predict_batch(
        self,
        examples: list[dict[str, Any]],
        *,
        max_concurrency: int = 4,
    ) -> list[CuadAnswer]:
        if self.dry_run:
            return [self._dry_run_prediction(ex["gold_answers"]) for ex in examples]
        if self.chain is None:
            raise RuntimeError("LangChain chain not configured for non-dry-run agent")
        chain_inputs = [
            {
                "contract_title": ex["contract_title"],
                "contract_text": ex["contract_text"],
                "category": self.category,
                "category_description": self.category_description,
                "answer_format": self.answer_format,
            }
            for ex in examples
        ]
        raw_outputs = self.chain.batch(
            chain_inputs,
            config={"max_concurrency": max_concurrency},
        )
        return [parse_cuad_answer(output) for output in raw_outputs]


def build_agents(
    questions_df: pd.DataFrame,
    *,
    llm: BaseChatModel | None,
    dry_run: bool = False,
    prompt_overrides: dict[str, str] | None = None,
) -> dict[int, ContractQuestionAgent]:
    categories = (
        questions_df[
            [
                "question_index",
                "category",
                "question",
                "category_description",
                "answer_format",
            ]
        ]
        .drop_duplicates("question_index")
        .sort_values("question_index")
    )
    if categories.empty:
        raise ValueError("No categories found for agent construction")

    agents: dict[int, ContractQuestionAgent] = {}
    prompt_overrides = prompt_overrides or {}
    for row in categories.itertuples(index=False):
        question_index = int(row.question_index)
        if question_index not in EVAL_QUESTION_INDICES:
            raise ValueError(f"Unexpected question_index={question_index}")
        category = str(row.category)
        question = str(row.question)
        answer_format = "" if pd.isna(row.answer_format) else str(row.answer_format)
        category_description = str(row.category_description)

        system_prompt = prompt_overrides.get(category) or compose_system_prompt(
            question=question,
            category=category,
            category_description=category_description,
            answer_format=answer_format,
        )

        chain: Runnable | None = None
        if not dry_run and llm is not None:
            chain = build_chain_for_agent(llm, system_prompt)

        agents[question_index] = ContractQuestionAgent(
            question_index=question_index,
            category=category,
            question=question,
            category_description=category_description,
            answer_format=answer_format,
            system_prompt=system_prompt,
            chain=chain,
            dry_run=dry_run,
        )
    return agents


def configure_llm(args: argparse.Namespace) -> BaseChatModel:
    load_dotenv(Path.home() / ".env")
    model_name = args.model

    if model_name.startswith("deepseek/") or model_name.startswith("deepseek-"):
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError as exc:
            raise RuntimeError(
                "langchain-deepseek is required for DeepSeek models. "
                "Install with: uv add langchain-deepseek"
            ) from exc
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek models")
        actual_name = model_name.removeprefix("deepseek/")
        return ChatDeepSeek(
            model=actual_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=api_key,
        )

    if model_name.startswith("openai/"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-openai is required for OpenAI models. "
                "Install with: uv add langchain-openai"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI models")
        actual_name = model_name.removeprefix("openai/")
        return ChatOpenAI(
            model=actual_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=api_key,
        )

    raise RuntimeError(
        f"Unknown model provider for: {model_name}. "
        "Use 'deepseek/...' or 'openai/...' prefix."
    )


def make_messages_with_hint(
    inputs: dict[str, Any],
    hint: str,
    full_system: str,
) -> list[Any]:
    """Like the make_messages closure but appends an enrichment hint to the user turn."""
    user_content = (
        f"Contract title:\n{inputs['contract_title']}\n\n"
        f"Contract text:\n{inputs['contract_text']}\n\n"
        f"Category:\n{inputs['category']}\n\n"
        f"Category description:\n{inputs['category_description']}\n\n"
        f"Answer format:\n{inputs['answer_format']}"
        f"\n\nKey terms to look for: {hint}"
    )
    return [SystemMessage(content=full_system), HumanMessage(content=user_content)]
