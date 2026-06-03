"""DSPy per-category CUAD agent.

Defines the per-question signature factory, the ``ContractQuestionAgent``
DSPy module, the agent-construction helper, and LM configuration. The
surrounding orchestration, evaluation loop, and CLI live in
``evaluators.dspy_runner``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# DSPy reads these at import time, so they must be set before ``import dspy``.
# The cache dir is kept under the evaluators package so an existing
# ``.dspy_cache`` directory is reused after this module was split out.
os.environ.setdefault(
    "DSPY_CACHEDIR",
    str(Path(__file__).resolve().parent.parent / "evaluators" / ".dspy_cache"),
)
os.environ.setdefault("LITELLM_MERGE_REASONING_CONTENT_IN_CHOICES", "true")

import dspy
import pandas as pd
from dotenv import load_dotenv

from cuad_agent.constants import EVAL_QUESTION_INDICES, NO_ANSWER
from cuad_agent.eval.metrics import parse_bool
from cuad_agent.paths import class_name_part

__all__ = [
    "ContractQuestionAgent",
    "ContractQuestionAgentBase",
    "build_agents",
    "configure_lm",
    "make_question_agent_class",
    "make_question_signature_class",
]


def make_question_signature_class(
    question_index: int,
    category: str,
    prompt: str,
) -> type[dspy.Signature]:
    name = f"CuadQuestion{question_index + 1:02d}{class_name_part(category)}Signature"
    namespace = {
        "__module__": __name__,
        "__doc__": prompt,
        "__annotations__": {
            "contract_title": str,
            "contract_text": str,
            "category": str,
            "category_description": str,
            "answer_format": str,
            "answer": str,
            "marked_impossible": bool,
        },
        "contract_title": dspy.InputField(),
        "contract_text": dspy.InputField(),
        "category": dspy.InputField(),
        "category_description": dspy.InputField(),
        "answer_format": dspy.InputField(),
        "answer": dspy.OutputField(
            desc=(
                "Exact answer text span(s) from the contract, separated by newlines "
                "if multiple; or NO_ANSWER."
            )
        ),
        "marked_impossible": dspy.OutputField(
            desc=(
                "Boolean. Return true when the contract does not contain an answer "
                "for this category/question; otherwise false."
            )
        ),
    }
    return type(name, (dspy.Signature,), namespace)


class ContractQuestionAgentBase(dspy.Module):
    """Base class for one fixed CUAD question/category agent."""

    def __init__(
        self,
        question_index: int,
        category: str,
        question: str,
        category_description: str,
        answer_format: str,
        signature_class: type[dspy.Signature],
        *,
        system_prompt: str | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.question_index = question_index
        self.category = category
        self.question = question
        self.category_description = category_description
        self.answer_format = answer_format
        self.signature_class = signature_class
        self.system_prompt = system_prompt
        self.dry_run = dry_run
        self.predict = dspy.ChainOfThought(signature_class)

    def forward(
        self,
        contract_title: str,
        contract_text: str,
        question: str,
        gold_answers: list[str] | None = None,
    ) -> dspy.Prediction:
        if self.dry_run:
            answer = "\n".join(gold_answers or []) if gold_answers else NO_ANSWER
            return dspy.Prediction(
                answer=answer,
                marked_impossible=not bool(gold_answers),
                reasoning="dry_run",
            )

        pred = self.predict(
            contract_title=contract_title,
            contract_text=contract_text,
            category=self.category,
            category_description=self.category_description,
            answer_format=self.answer_format,
        )
        return dspy.Prediction(
            answer=str(pred.answer),
            marked_impossible=parse_bool(getattr(pred, "marked_impossible", False)),
            reasoning=getattr(pred, "reasoning", ""),
        )


ContractQuestionAgent = ContractQuestionAgentBase


def make_question_agent_class(
    question_index: int,
    category: str,
    question: str,
) -> type[ContractQuestionAgentBase]:
    name = f"CuadQuestion{question_index + 1:02d}{class_name_part(category)}Agent"
    return type(
        name,
        (ContractQuestionAgentBase,),
        {"__module__": __name__, "__doc__": question},
    )


def build_agents(
    questions_df: pd.DataFrame,
    *,
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
        system_prompt = prompt_overrides.get(category)
        signature_prompt = system_prompt or question
        signature_class = make_question_signature_class(
            question_index=question_index,
            category=category,
            prompt=signature_prompt,
        )
        agent_class = make_question_agent_class(
            question_index=question_index,
            category=category,
            question=question,
        )
        agents[question_index] = agent_class(
            question_index=question_index,
            category=category,
            question=question,
            category_description=str(row.category_description),
            answer_format="" if pd.isna(row.answer_format) else str(row.answer_format),
            signature_class=signature_class,
            system_prompt=system_prompt,
            dry_run=dry_run,
        )

    for agent in agents.values():
        expected_prompt = agent.system_prompt or agent.question
        if agent.signature_class.__doc__ != expected_prompt:
            raise ValueError(
                f"Signature docstring mismatch for question_index={agent.question_index}"
            )
    return agents


def configure_lm(args: argparse.Namespace) -> None:
    load_dotenv(Path.home() / ".env")
    api_key: str | None = None
    if args.model.startswith("deepseek/"):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for DeepSeek DSPy models")
    elif args.model.startswith("openai/"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI DSPy models")

    lm = dspy.LM(
        model=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())
