"""DSPy multi-agent baseline for ConvFinQA.

End-to-end implementation in a single file:
  Triage -> Preprocess -> Retriever -> Calculation
  Loads the `dev` split of `data/convfinqa_dataset.json`, picks 2 random
  records (seed=42), evaluates all turns with teacher-forced conversation
  history, and reports per-record + overall execution accuracy.

Run with:  uv run python agent.py
"""

# ruff: noqa: T201

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Literal

# Pin DSPy's LM cache to a repo-local dir so it's portable across machines
# (rsync the repo, you take the cache with you). Must be set before `import dspy`
# — DSPy reads DSPY_CACHEDIR at import time. `.dspy_cache` is already excluded
# by the `.*cache` rule in .gitignore.
os.environ.setdefault("DSPY_CACHEDIR", str(Path(__file__).resolve().parent / ".dspy_cache"))

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field

##
# import mlflow

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("ConvFinQA-dev")
# mlflow.dspy.autolog()

## models

load_dotenv(Path.home() / ".env")

# DeepSeek's reasoning models (v4-flash / v4-pro) split structured output into
# `reasoning_content` and leave `text` empty. The default JSONAdapter only reads
# `text` and crashes with AdapterParseError on every ReAct/CoT call. Telling
# LiteLLM to merge reasoning_content into the choice text fixes this.
os.environ.setdefault("LITELLM_MERGE_REASONING_CONTENT_IN_CHOICES", "true")
os.environ.setdefault("RUN_GEPA", "RUN")
# GEPA_NAME
# GEPA_MODE



lm_mini = dspy.LM(model="deepseek/deepseek-v4-flash",
     api_key=os.environ["DEEPSEEK_API_KEY"],
      max_tokens=64000,        # cap each step
      temperature=1)
lm_max = dspy.LM(model="deepseek/deepseek-v4-pro",
    api_key=os.environ["DEEPSEEK_API_KEY"],
      max_tokens=64000,        # cap each step
      temperature=1)

# ChatAdapter parses markdown-tagged outputs (e.g. `[[ ## field ## ]]`), which
# is more permissive than the default JSONAdapter — important when the model's
# response is contaminated with leading reasoning prose.
dspy.configure(lm=lm_mini, adapter=dspy.ChatAdapter())


# ---------------------------------------------------------------------------
# 1. data
# ---------------------------------------------------------------------------

import pandas as pd

from data import training_data

with open( "data/convfinqa_dataset.json") as f:
    data = json.load(f)


qa_data = training_data()
qa_data = qa_data.query('data_key=="train"')

sampled_report_ids = (
    qa_data["report_id"]
    .drop_duplicates()
    .sample(n=200, random_state=42)
    .tolist()
)

# Additive test-set expansion: pick 60 more report_ids that aren't in the
# original 100, append them to test only. Keeps `train_report_ids` byte-identical
# to prior runs (so prior GEPA-optimized programs remain fairly evaluated) while
# growing the test set from 40 → 100. Deterministic via random_state=42 on the
# disjoint population, so the original 40 test ids are preserved exactly.
additional_test_ids = (
    qa_data.loc[~qa_data["report_id"].isin(sampled_report_ids), "report_id"]
    .drop_duplicates()
    .sample(n=60, random_state=42)
    .tolist()
)
all_report_ids = sampled_report_ids + additional_test_ids

train_report_ids = pd.Series(sampled_report_ids).sample(frac=0.6, random_state=42).tolist()
test_report_ids  = [r for r in sampled_report_ids if r not in train_report_ids] + additional_test_ids

qa_data = qa_data[qa_data["report_id"].isin(all_report_ids)].reset_index(drop=True)

dspy_data = [
    dspy.Example(
        question=x.get("conv_questions"),
        report_id=x.get("report_id"),
        answer=x.get("conv_answers"),
        turn_type=x.get("turn_type"),
        turn_program=x.get("turn_program"),
        conv_type="Type II" if x.get("qa_split")==True else "Type I"
    )
    for x in qa_data.to_dict(orient="records")
]

train_set = [x for x in dspy_data if x.report_id in train_report_ids ]
test_set = [x for x in dspy_data if x.report_id in test_report_ids ]


# ---------------------------------------------------------------------------
# 1. data models
# ---------------------------------------------------------------------------


class QAPair(BaseModel):
    """A question paired with its answer (used for retrieval / calculation hand-off)."""
    question: str
    answer: str


class HistoryTurn(BaseModel):
    """One turn of conversation history, tagged with the report it answered against."""
    question: str
    answer: str
    report_id: str


class ConversationHistory(BaseModel):
    """Multi-turn history for a session. May span multiple documents (report_ids)."""

    pairs: list[HistoryTurn] = Field(default_factory=list)

    def append(self, question: str, answer: str, report_id: str) -> None:
        """Append a question/answer/report_id triple to the history."""
        self.pairs.append(HistoryTurn(question=question, answer=answer, report_id=report_id))

    def as_text(self) -> str:
        """Format the history as a flat text block for inclusion in agent prompts."""
        if not self.pairs:
            return "(no prior turns)"
        return "\n".join(
            f"Q{i + 1} [report={p.report_id}]: {p.question}\nA{i + 1}: {p.answer}"
            for i, p in enumerate(self.pairs)
        )

TurnType = Literal["number", "program"]
ConvType = Literal["Type I", "Type II"]


class AgentResponse(BaseModel):
    """Final response surfaced to the evaluator/caller.

    Per-stage `*_reasoning` fields capture the ChainOfThought rationale from each
    predictor, and `calc_trajectory` captures the ReAct tool-call trace from the
    calculator. Both are useful during evaluation and as optimization targets
    (e.g. by GEPA).
    """
    question: str
    report_id: str
    answer: str
    turn_type: TurnType
    conv_type: ConvType
    turn_program: str | None = None
    triage_reasoning: str | None = None
    preprocess_reasoning: str | None = None
    retriever_reasoning: str | None = None
    calc_trajectory: dict[str, Any] | None = None


# Dataset models (per CLAUDE.md spec)


class Document(BaseModel):
    """Financial document: pre/post text plus structured table."""
    pre_text: str
    post_text: str
    table: dict[str, dict[str, float | str | int]]


class Dialogue(BaseModel):
    """Multi-turn dialogue with gold programs and executed answers."""
    conv_questions: list[str]
    conv_answers: list[str]
    turn_program: list[str]
    executed_answers: list[float | str]
    qa_split: list[bool] = Field(default_factory=list)


class Features(BaseModel):
    """Helper features computed from the dialogue."""
    num_dialogue_turns: int
    has_type2_question: bool
    has_duplicate_columns: bool
    has_non_numeric_values: bool


class ConvFinQARecord(BaseModel):
    """One record from the ConvFinQA dataset."""
    id: str
    doc: Document
    dialogue: Dialogue
    features: Features


# ---------------------------------------------------------------------------
# 2. Calculator tools (mirrors mcp/server_calculator.py as plain Python fns)
# ---------------------------------------------------------------------------


def add(a: float, b: float) -> float:
    """Return a + b."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Return a - b."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Return a * b."""
    return a * b


def divide(a: float, b: float) -> float:
    """Return a / b. Raises ZeroDivisionError if b == 0."""
    return a / b


def exp(a: float, b: float) -> float:
    """Return a raised to the power b."""
    return float(a**b)


def greater(a: float, b: float) -> bool:
    """Return True iff a is strictly greater than b."""
    return a > b


CALCULATOR_TOOLS: list[Any] = [add, subtract, multiply, divide, exp, greater]


# ---------------------------------------------------------------------------
# 3. DSPy Signatures
# ---------------------------------------------------------------------------


class TriageSignature(dspy.Signature):
    """Classify the current turn using the question plus prior conversation history.

    You must predict two labels:

    1. `turn_type`
       - `number`: the answer is a direct value lookup from the document or a
         previously answered value in history. No arithmetic or multi-value
         composition is required.
       - `program`: the answer requires arithmetic, comparison, a rate/change,
         a percentage, a difference between periods, a sum across values, or
         any multi-step reasoning over multiple values.

    2. `conv_type`
       - `Type I`: the question continues the current reasoning thread from the
         same decomposed multi-hop problem.
       - `Type II`: the question switches to a different aspect / sub-problem
         of the same report, even if it still references prior turns.

    Use `history` aggressively. Follow-up questions with references like
    "that", "this", "the change", "the difference", "what about 2010", or
    "what percentage" are often continuations of prior reasoning and are more
    likely to be `program` turns than isolated one-shot lookups. If answering
    the current turn would require combining a value from history with a new
    value, or transforming a prior answer, label it `program`.
    """
    question: str = dspy.InputField()
    history: str = dspy.InputField(
        desc=(
            "Prior Q&A pairs in this session. Use this to resolve follow-up "
            "references and determine whether the current turn is a direct "
            "lookup or a continuation that requires computation."
        )
    )

    turn_type: Literal["number", "program"] = dspy.OutputField(
        desc=(
            "`number` only when the final answer is a single directly retrievable "
            "value. Use `program` when the turn needs arithmetic, comparison, "
            "change-over-time reasoning, percentages, aggregation, or reuse of "
            "a prior answer in a computation."
        ),
    )
    conv_type: Literal["Type I", "Type II"] = dspy.OutputField(
        desc=(
            "`Type I` when the turn continues the current reasoning chain. "
            "Use `Type II` when the turn pivots to a different aspect or a "
            "second decomposed problem about the same report."
        ),
    )


class PreprocessSignature(dspy.Signature):
    """Decompose a program-type question into sub-questions and a calculation program.

    You are given:
      - `question`: the current user question
      - `history`: prior turns with their questions and answers
      - `conv_type`: whether this turn continues the current reasoning chain (Type I)
        or switches to a different aspect of the report (Type II)

    Your job is to produce:
      - `reasoning`: a brief explanation of the decomposition and which cached values
        from `history` can be reused
      - `sub_questions`: value lookups only, not computations
      - `program`: an arithmetic expression over A, B, C, ... using add, subtract,
        multiply, and divide

    Use `conv_type` to guide decomposition:
      - Type I: heavily lean on `history`. Follow-up questions often depend on prior
        answers, so reuse the exact phrasing from relevant earlier turns whenever a
        needed value is already available there.
      - Type II: re-anchor sub-questions on the document because the conversation has
        shifted to a different aspect of the report, but still reuse a cached value
        from `history` if it is clearly the same quantity.

    Reuse `history` whenever possible: if a value needed by the program already appears
    as the answer to a prior turn, restate that earlier question as closely as possible
    so the retriever can return the cached value instead of re-reading the document.
    This reduces drift across turns and is especially important in long conversations.

    Program design rules:
      - If the question asks for a "growth rate" or "percent change", compute the ratio
        and then multiply by 100 so the downstream answer becomes a percentage result.
      - If the question asks what "percentage change this represents", return the raw
        ratio without multiplying by 100.
      - Keep the program numeric only. Do not include units, currency markers, or other
        formatting in the program itself.

    The distinction between `divide(...)` and `multiply(divide(...), 100)` matters and
    should be chosen deliberately based on whether the target answer is a plain ratio or
    a percentage-style result.
    """
    question: str = dspy.InputField()
    history: str = dspy.InputField(desc="Prior Q&A pairs in this session — reuse answers when applicable")
    conv_type: Literal["Type I", "Type II"] = dspy.InputField(
        desc="From triage: 'Type I' continues the prior chain; 'Type II' switches aspect",
    )
    sub_questions: list[str] = dspy.OutputField(
        desc=(
            "Self-contained value lookups only, not computations. "
            "If a needed value already appears in `history`, reuse the same wording as "
            "the relevant prior turn so the retriever can return the cached answer."
        ),
    )
    program: str = dspy.OutputField(
        desc=(
            "Arithmetic DSL such as 'subtract(A, B)' or 'divide(subtract(A, B), B)', "
            "where A, B, C... map positionally to `sub_questions`. Use "
            "'multiply(divide(...), 100)' for percentage-style outputs and "
            "'divide(...)' for raw ratios."
        ),
    )


class RetrieverSignature(dspy.Signature):
    """Answer one or more value-lookup questions from the financial document.

    Behavior depends on `turn_type`:
      - `number`: there is exactly one question and it is the user's final question.
        Return the single value that answers it. No downstream calculator stage will
        run, so you may need to do simple arithmetic here when the question asks for
        a change, net increase/decrease, or a percentage.
      - `program`: the questions are sub-questions from Preprocess. Return the raw
        retrieved value for each one, with no arithmetic or aggregation. These values
        are passed to the Calculation stage.

    In both modes, prefer reusing values already present in `history` over re-reading
    the document when the same value has already been answered in a prior turn.

    Retrieval and arithmetic rules:
      - Match both the entity and the date/year carefully. If the document contains
        multiple values from the same year, use the one the question actually refers to.
      - In `number` mode, change questions should use signed arithmetic:
        later minus earlier. Do not take absolute values unless the question asks for
        magnitude explicitly.
      - Percentage change / return-rate questions should use
        `((new - old) / old) * 100` and return a `%` suffixed answer string.
      - Raw factual lookups and all `program` mode outputs should preserve the source
        numeric string as closely as possible, including meaningful trailing zeroes.
      - Computed numeric answers should use sensible precision based on the operands,
        and should never include extraneous units such as `$`, `million`, or `billion`.
    """
    turn_type: Literal["number", "program"] = dspy.InputField(
        desc=(
            "From triage. 'number' = single question, return the final answer. "
            "'program' = sub-questions from preprocess, return raw values for the calculator."
        ),
    )
    questions: list[str] = dspy.InputField(desc="One or more self-contained value-lookup questions")
    document: Document = dspy.InputField(
        desc="The financial report: pre_text, post_text, and a structured `table` (column -> row -> value)",
    )
    history: str = dspy.InputField(desc="Prior Q&A pairs — reuse cached answers when applicable")
    answers: list[QAPair] = dspy.OutputField(
        desc=(
            "One QAPair per input question, same order as `questions`. "
            "`question` echoes the input question verbatim; `answer` is the retrieved "
            "or computed answer string. In `program` mode, return raw values only. "
            "In `number` mode, return the final answer string, including `%` only when "
            "the question explicitly asks for a percentage-style result."
        ),
    )


class CalculationSignature(dspy.Signature):
    """Execute a DSL program over retrieved values using calculator tools.

    You receive the original `question`, the retrieved sub-question answers, and a
    candidate `program`. The program is a strong hint, but it is not infallible:
    if it conflicts with the question's intent, you should correct the operation,
    argument order, or semantics before finishing.

    Rules for execution:
      - Map placeholders positionally: first retrieved answer = A, second = B, etc.
      - Strip non-numeric decoration from retrieved answers as needed (`%`, `$`,
        commas, units) while preserving the intended numeric value.
      - Treat percentage answers in `retrieved` as whole numbers unless the question
        explicitly asks for a decimal fraction.
      - Sanity-check directionality for changes and differences. If the question asks
        for decline/decrease/change from earlier to later, make sure the subtraction
        order matches that intent.
      - Trust the user's question over the program if the two disagree.
      - The final answer must be a plain numeric string with no units or symbols.
    """
    question: str = dspy.InputField(desc="The user's original question (context only — do not re-answer from it)")
    retrieved: list[QAPair] = dspy.InputField(
        desc=(
            "Sub-questions paired with their retrieved values, in placeholder order: "
            "first entry = A, second = B, etc."
        ),
    )
    program: str = dspy.InputField(
        desc=(
            "Candidate DSL to execute, e.g. 'subtract(A, B)' or "
            "'divide(subtract(A, B), B)'. Correct it if it does not match the question."
        ),
    )
    answer: str = dspy.OutputField(
        desc="Final plain numeric result as a string from the calculator workflow, with no units or symbols",
    )


# ---------------------------------------------------------------------------
# 4. Sequential Agent
# ---------------------------------------------------------------------------



class ConvFinQASequentialAgent(dspy.Module):
    """Sequential pipeline: triage -> preprocess -> retrieve -> calculate."""
    # 
    def __init__(self) -> None:
        """Instantiate the four predictors, build the doc lookup, and start a fresh history."""
        super().__init__()
        self.triage = dspy.ChainOfThought(TriageSignature)
        self.preprocess = dspy.ChainOfThought(PreprocessSignature)
        self.retriever = dspy.ChainOfThought(RetrieverSignature)
        self.calculator = dspy.ReAct(
            CalculationSignature,
            tools=CALCULATOR_TOOLS,
            max_iters=8,
        )
        # Document store: O(1) retrieval by report_id, kept as structured Document
        # so the retriever sees the table as a dict rather than a flattened string.
        self._docs: dict[str, Document] = {
            rec["id"]: Document.model_validate(rec["doc"])
            for split_records in data.values()
            for rec in split_records
        }
        # Conversation history is owned by the agent instance and persists across
        # forward() calls so multi-turn dependencies resolve correctly. Call
        # `reset_conversation()` to start a new session.
        self.conversation: ConversationHistory = ConversationHistory()

    def reset_conversation(self) -> None:
        """Clear conversation history (e.g. when starting a new session)."""
        self.conversation = ConversationHistory()

    def _retrieve_document(self, report_id: str) -> Document:
        """Look up the financial document for a given report_id."""
        try:
            return self._docs[report_id]
        except KeyError as e:
            msg = f"Unknown report_id: {report_id!r}"
            raise KeyError(msg) from e

    def forward(
        self,
        question: str,
        report_id: str,
    ) -> AgentResponse:
        """Run a single turn end-to-end and return an AgentResponse.

        Reads from and appends to `self.conversation` — call `reset_conversation()` between
        unrelated sessions to avoid context bleed.
        """
        hist_text = self.conversation.as_text()
        triage = self.triage(question=question, history=hist_text)
        document = self._retrieve_document(report_id)

        if triage.turn_type == "number":
            r = self.retriever(
                turn_type="number",
                questions=[question],
                document=document,
                history=hist_text,
            )
            answer = str(r.answers[0].answer)
            self.conversation.append(question=question, answer=answer, report_id=report_id)
            return AgentResponse(
                question=question,
                report_id=report_id,
                answer=answer,
                turn_type="number",
                conv_type=triage.conv_type,
                triage_reasoning=getattr(triage, "reasoning", None),
                retriever_reasoning=getattr(r, "reasoning", None),
            )

        pp = self.preprocess(question=question, history=hist_text, conv_type=triage.conv_type)
        r = self.retriever(
            turn_type="program",
            questions=list(pp.sub_questions),
            document=document,
            history=hist_text,
        )
        calc = self.calculator(
            question=question,
            retrieved=list(r.answers),
            program=pp.program,
        )
        answer = str(calc.answer)
        self.conversation.append(question=question, answer=answer, report_id=report_id)
        return AgentResponse(
            question=question,
            report_id=report_id,
            answer=answer,
            turn_type="program",
            conv_type=triage.conv_type,
            turn_program=str(pp.program),
            triage_reasoning=getattr(triage, "reasoning", None),
            preprocess_reasoning=getattr(pp, "reasoning", None),
            retriever_reasoning=getattr(r, "reasoning", None),
            calc_trajectory=getattr(calc, "trajectory", None),
        )


##

# ---------------------------------------------------------------------------
# 5. Multi-turn evaluation
# ---------------------------------------------------------------------------


def numeric_match(pred: Any, gold: Any) -> bool:
    """Loose numeric/string match used by the evaluator."""
    try:
        return abs(float(pred) - float(gold)) < 1e-3
    except (ValueError, TypeError):
        return str(pred).strip().lower() == str(gold).strip().lower()


# 60/40 split of the original 100 sampled report_ids — train stays at 60,
# the original 40 test ids are preserved by-position in the seeded shuffle.
# `additional_test_ids` (60 more, sampled disjoint above) are then appended to
# test only, growing the test set to 100 without altering train.
rng = random.Random(42)
_shuffled_ids = sampled_report_ids[:]
rng.shuffle(_shuffled_ids)
_split = int(len(_shuffled_ids) * 0.6)
train_report_ids = _shuffled_ids[:_split]
test_report_ids = _shuffled_ids[_split:] #+ additional_test_ids


def build_conv_examples(report_ids: list[str]) -> list[dspy.Example]:
    """One dspy.Example per conversation, with all turns in q_order."""
    examples: list[dspy.Example] = []
    for rid in report_ids:
        g = qa_data[qa_data["report_id"] == rid].sort_values("q_order")
        examples.append(
            dspy.Example(
                report_id=rid,
                questions=g["conv_questions"].tolist(),
                gold_answers=g["conv_answers"].tolist(),
            ).with_inputs("report_id", "questions")
        )
    return examples


conv_examples_train = build_conv_examples(train_report_ids)
conv_examples_test = build_conv_examples(test_report_ids)


# Module-level doc store: shared across all ConversationRunner instances. Keeping
# it off `self` means dspy.Module.deepcopy() (used by GEPA between trials) only
# clones the four predictors, not this large dict.
_DOCS: dict[str, Document] = {
    rec["id"]: Document.model_validate(rec["doc"])
    for split_records in data.values()
    for rec in split_records
}


class ConversationRunner(dspy.Module):
    """Walks all turns of one conversation, with predictors owned directly.

    Predictors live on `self` so GEPA can introspect (`runner.named_predictors()`)
    and optimize each one's instructions/demos. Conversation history is a *local*
    in `forward()` — never on `self` — so the runner is safe under parallel
    evaluation and stateless across GEPA trials.
    """

    def __init__(self) -> None:
        super().__init__()
        self.triage = dspy.ChainOfThought(TriageSignature)
        self.preprocess = dspy.ChainOfThought(PreprocessSignature)
        self.retriever = dspy.ChainOfThought(RetrieverSignature)
        self.calculator = dspy.ReAct(
            CalculationSignature,
            tools=CALCULATOR_TOOLS,
            max_iters=8,
        )

    def _run_turn(
        self,
        question: str,
        report_id: str,
        document: Document,
        conversation: ConversationHistory,
    ) -> AgentResponse:
        """Run one turn (triage -> [preprocess -> retrieve -> calc] / [retrieve])."""
        hist_text = conversation.as_text()
        triage = self.triage(question=question, history=hist_text)

        if triage.turn_type == "number":
            r = self.retriever(
                turn_type="number",
                questions=[question],
                document=document,
                history=hist_text,
            )
            answer = str(r.answers[0].answer)
            conversation.append(question=question, answer=answer, report_id=report_id)
            return AgentResponse(
                question=question,
                report_id=report_id,
                answer=answer,
                turn_type="number",
                conv_type=triage.conv_type,
                triage_reasoning=getattr(triage, "reasoning", None),
                retriever_reasoning=getattr(r, "reasoning", None),
            )

        pp = self.preprocess(question=question, history=hist_text, conv_type=triage.conv_type)
        r = self.retriever(
            turn_type="program",
            questions=list(pp.sub_questions),
            document=document,
            history=hist_text,
        )
        calc = self.calculator(
            question=question,
            retrieved=list(r.answers),
            program=pp.program,
        )
        answer = str(calc.answer)
        conversation.append(question=question, answer=answer, report_id=report_id)
        return AgentResponse(
            question=question,
            report_id=report_id,
            answer=answer,
            turn_type="program",
            conv_type=triage.conv_type,
            turn_program=str(pp.program),
            triage_reasoning=getattr(triage, "reasoning", None),
            preprocess_reasoning=getattr(pp, "reasoning", None),
            retriever_reasoning=getattr(r, "reasoning", None),
            calc_trajectory=getattr(calc, "trajectory", None),
        )

    def forward(self, report_id: str, questions: list[str]) -> dspy.Prediction:
        document = _DOCS[report_id]
        conversation = ConversationHistory()  # local, per-call — NOT self.*
        responses = [
            self._run_turn(q, report_id, document, conversation) for q in questions
        ]
        return dspy.Prediction(
            predictions=[r.answer for r in responses],
            responses=responses,
            conversation=conversation,
        )


def conv_turn_accuracy(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
    """Fraction of turns in this conversation where pred matches gold."""
    golds = example.gold_answers
    preds = getattr(pred, "predictions", None) or []
    if not golds:
        return 0.0
    return sum(numeric_match(p, g) for p, g in zip(preds, golds)) / len(golds)


def conv_metric_with_feedback(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> dspy.Prediction:
    """GEPA-style metric: returns a score plus per-turn feedback for reflection."""
    golds = example.gold_answers
    preds = getattr(prediction, "predictions", None) or []
    score = (
        sum(numeric_match(p, g) for p, g in zip(preds, golds)) / len(golds)
        if golds else 0.0
    )

    if not preds:
        feedback = (
            "The runner returned no predictions for this conversation, likely due to "
            "an LM/adapter parsing failure. The downstream agents could not produce "
            "structured outputs. Consider tightening output format instructions."
        )
        return dspy.Prediction(score=score, feedback=feedback)

    lines = [f"Conversation on report {example.report_id}:"]
    for i, (q, g) in enumerate(zip(example.questions, golds), start=1):
        p = preds[i - 1] if i <= len(preds) else "<missing>"
        ok = numeric_match(p, g)
        tag = "PASS" if ok else "FAIL"
        lines.append(f"  T{i} {tag}  Q: {q}")
        lines.append(f"        pred={p!r}  gold={g!r}")
    lines.append(
        "FAIL turns indicate either: wrong value retrieved, wrong DSL program, "
        "answer formatted with extraneous units (e.g. '$3.0 billion' instead of "
        "'3.0'), or unrounded float vs gold percent. Aim for plain numeric strings."
    )
    return dspy.Prediction(score=score, feedback="\n".join(lines))


def analyze_predictions(predictions_path: Path) -> pd.DataFrame:
    """Inner-join a predictions CSV to qa_data and print accuracy by slice.

    Returns the joined DataFrame and writes it to a sibling `*_joined.csv`
    for downstream analysis. Slices reported: turn_type (number/program),
    conv_type (Type I/II), q_order (turn index in conversation).
    """
    preds = pd.read_csv(predictions_path)
    qa = qa_data.sort_values(["report_id", "q_order"]).copy()
    qa["turn_index"] = qa.groupby("report_id").cumcount()
    joined = preds.merge(
        qa[["report_id", "turn_index", "q_order", "turn_type", "qa_split"]],
        on=["report_id", "turn_index"],
        how="inner",
    )
    joined["conv_type"] = joined["qa_split"].map({True: "Type II", False: "Type I"})

    overall = joined["correct"].mean()
    print(f"\nAccuracy breakdowns (n={len(joined)} turns, overall={overall:.1%})")
    for col in ("turn_type", "conv_type", "q_order"):
        cut = joined.groupby(col)["correct"].agg(["mean", "count"])
        cut["mean"] = cut["mean"].map(lambda v: f"{v:.1%}")
        print(f"\nBy {col}:")
        print(cut.to_string())

    for gold_col, pred_col in (
        ("turn_type", "pred_turn_type"),
        ("conv_type", "pred_conv_type"),
    ):
        if pred_col not in joined.columns:
            continue
        cut = joined.groupby([gold_col, pred_col])["correct"].agg(["mean", "count"])
        cut["mean"] = cut["mean"].map(lambda v: f"{v:.1%}")
        print(f"\nBy {gold_col} × {pred_col}:")
        print(cut.to_string())

    out_path = predictions_path.with_name(f"{predictions_path.stem}_joined.csv")
    joined.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    return joined


def _eval_result_to_joined(
    eval_result: Any,
    *,
    model_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ex, pred, _ in eval_result.results:
        preds = getattr(pred, "predictions", None) or []
        for i, (q, g) in enumerate(zip(ex.questions, ex.gold_answers)):
            p = preds[i] if i < len(preds) else None
            rows.append(
                {
                    "report_id": ex.report_id,
                    "turn_index": i,
                    "question": q,
                    "gold_answer": g,
                    "pred_answer": p,
                    "correct": numeric_match(p, g) if p is not None else False,
                    "model": model_label,
                }
            )
    preds = pd.DataFrame(rows)
    qa = qa_data.sort_values(["report_id", "q_order"]).copy()
    qa["turn_index"] = qa.groupby("report_id").cumcount()
    joined = preds.merge(
        qa[["report_id", "turn_index", "turn_type", "qa_split"]],
        on=["report_id", "turn_index"],
        how="inner",
    )
    joined["conv_type"] = joined["qa_split"].map({True: "Type II", False: "Type I"})
    return joined


def print_model_accuracy_table(
    joined_frames: list[pd.DataFrame],
    *,
    slice_col: str,
    title: str,
) -> None:
    combined = pd.concat(joined_frames, ignore_index=True)
    rows: list[dict[str, Any]] = []

    overall = {"bucket": "overall"}
    for model, frame in combined.groupby("model"):
        overall[f"{model}_acc"] = frame["correct"].mean()
    rows.append(overall)

    for bucket in sorted(combined[slice_col].dropna().unique()):
        row = {"bucket": bucket}
        for model, frame in combined.groupby("model"):
            cut = frame[frame[slice_col] == bucket]
            row[f"{model}_acc"] = cut["correct"].mean() if not cut.empty else None
        rows.append(row)

    out = pd.DataFrame(rows)
    printable = out.copy()
    for col in [c for c in printable.columns if c.endswith("_acc")]:
        printable[col] = printable[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "-")
    print(f"\n{title}:")
    print(printable.to_string(index=False))


def load_artifact_instructions(program_path: Path) -> dict[str, str]:
    """Load per-predictor instructions from a saved DSPy program artifact."""
    raw = json.loads(program_path.read_text())
    return {
        key: raw[key]["signature"]["instructions"].rstrip()
        for key in (
            "triage.predict",
            "preprocess.predict",
            "retriever.predict",
            "calculator.react",
        )
    }


def compare_runner_instructions(
    runner: "ConversationRunner",
    program_path: Path,
) -> dict[str, bool]:
    """Compare loaded predictor instructions against a saved program artifact."""
    expected = load_artifact_instructions(program_path)
    results: dict[str, bool] = {}
    print(f"\nInstruction comparison vs {program_path.name}:")
    for name, predictor in runner.named_predictors():
        if name not in expected:
            continue
        loaded = predictor.signature.instructions.rstrip()
        ok = loaded == expected[name]
        results[name] = ok
        status = "MATCH" if ok else "MISMATCH"
        print(f"  - {name:<10} {status}")
    return results


def write_predictions_csv(
    predictions_path: Path,
    eval_results: list[tuple[dspy.Example, dspy.Prediction, Any]],
) -> None:
    """Write per-turn predictions plus predicted turn labels for inspection."""
    import csv

    with predictions_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "report_id",
            "turn_index",
            "question",
            "gold_answer",
            "pred_answer",
            "correct",
            "pred_turn_type",
            "pred_conv_type",
        ])
        for ex, pred, _ in eval_results:
            preds = getattr(pred, "predictions", None) or []
            responses = getattr(pred, "responses", None) or []
            for i, (q, g) in enumerate(zip(ex.questions, ex.gold_answers)):
                p = preds[i] if i < len(preds) else None
                response = responses[i] if i < len(responses) else None
                w.writerow([
                    ex.report_id,
                    i,
                    q,
                    g,
                    p,
                    numeric_match(p, g) if p is not None else False,
                    getattr(response, "turn_type", None),
                    getattr(response, "conv_type", None),
                ])


# ---------------------------------------------------------------------------
# 6. __main__: evaluate on full test set
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    test_set = conv_examples_test
    total_turns = sum(len(ex.questions) for ex in test_set)
    print(f"Test set: {len(test_set)} conversations, {total_turns} turns total")

    # ConversationRunner builds a fresh agent per forward(), so each conversation
    # has its own self.conversation — safe to parallelize across conversations.
    evaluator = dspy.Evaluate(
        devset=test_set,
        metric=conv_turn_accuracy,
        num_threads=8,
        display_progress=True,
    )
    eval_result = evaluator(ConversationRunner())

    # DSPy returns score as a 0–100 percent already, not a 0–1 fraction.
    print(f"\nOverall turn accuracy: {eval_result.score:.1f}%")
    print("\nPer-conversation:")
    n_errored = 0
    for ex, pred, s in eval_result.results:
        n_turns = len(ex.questions)
        preds = getattr(pred, "predictions", None)
        if preds is None:
            n_errored += 1
            print(f"  {ex.report_id:<45}  ERRORED ({n_turns} turns skipped)")
            continue
        n_pass = sum(numeric_match(p, g) for p, g in zip(preds, ex.gold_answers))
        print(f"  {ex.report_id:<45}  {n_pass}/{n_turns} turns  ({s:.0%})")
    if n_errored:
        print(f"\n{n_errored} conversation(s) errored (LM adapter failures); they count as 0 in the overall score.")

    # -----------------------------------------------------------------------
    # 7. GEPA optimization
    # -----------------------------------------------------------------------
    # GEPA is gated behind an env var because compile() is expensive
    # (many LM calls × many trials × reflection). Set RUN_GEPA=1 to enable.

    if os.environ.get("RUN_GEPA"):
        from datetime import datetime

        # Mode selection.
        # ---------------
        # GEPA_MODE controls cost/quality tradeoff:
        #   smoke (default): ~30 min, valset=5, max_metric_calls=120 — for
        #     verifying wiring, sanity-checking proposed prompts, fast iteration.
        #     Output is NOT a transferable optimization (overfit to 5 examples).
        #   real:            5–9 hr,  valset=12, auto="light"           — actual
        #     optimization run that can be evaluated on the held-out test set.
        gepa_mode = os.environ.get("GEPA_MODE", "smoke").lower()
        if gepa_mode not in {"smoke", "real"}:
            raise RuntimeError(f"GEPA_MODE must be 'smoke' or 'real', got {gepa_mode!r}")

        if gepa_mode == "smoke":
            n_val = 5
            gepa_kwargs: dict[str, Any] = {"max_metric_calls": 120}
        else:  # real
            n_val = 12
            gepa_kwargs = {"auto": "light"}

        gepa_name = os.environ.get("GEPA_NAME")
        resume_target = os.environ.get("RESUME_GEPA") or os.environ.get("GEPA_RESUME")
        if gepa_name and resume_target:
            raise RuntimeError("GEPA_NAME and RESUME_GEPA are mutually exclusive")

        # Load-and-evaluate shortcut.
        # ---------------------------
        # If GEPA_NAME points at a run dir that already contains
        # `dspy_optimized_runner.json`, skip optimization entirely and just load +
        # evaluate. Useful for re-scoring a prior run without paying for GEPA.
        run_dir: Path | None = None
        existing_program: Path | None = None
        if not resume_target:
            if gepa_name:
                run_dir = Path("runs") / gepa_name
                existing_program = run_dir / "dspy_optimized_runner.json"
                if not existing_program.exists():
                    existing_program = run_dir / "optimized_runner.json"
            else:
                candidate_dirs = sorted(Path("runs").glob(f"gepa_{gepa_mode}_*"), key=lambda p: p.name)
                for candidate in reversed(candidate_dirs):
                    candidate_program = candidate / "dspy_optimized_runner.json"
                    if not candidate_program.exists():
                        candidate_program = candidate / "optimized_runner.json"
                    if candidate_program.exists():
                        run_dir = candidate
                        existing_program = candidate_program
                        break
        if existing_program and run_dir:
            print(f"\nFound {existing_program} — skipping GEPA, loading and evaluating.")
            optimized_runner = ConversationRunner()
            optimized_runner.load(str(existing_program))
            compare_runner_instructions(optimized_runner, existing_program)
            opt_eval_result = dspy.Evaluate(
                devset=test_set,
                metric=conv_turn_accuracy,
                num_threads=8,
                display_progress=True,
            )(optimized_runner)
            print(f"\nBaseline turn accuracy:  {eval_result.score:.1f}%")
            print(f"Optimized turn accuracy: {opt_eval_result.score:.1f}%")
            print(f"Δ = {opt_eval_result.score - eval_result.score:+.1f} pts")
            baseline_joined = _eval_result_to_joined(eval_result, model_label="baseline")
            optimized_joined = _eval_result_to_joined(
                opt_eval_result,
                model_label="optimized",
            )
            print_model_accuracy_table(
                [baseline_joined, optimized_joined],
                slice_col="turn_type",
                title="Turn Type Accuracy by Model",
            )
            print_model_accuracy_table(
                [baseline_joined, optimized_joined],
                slice_col="conv_type",
                title="Conv Type Accuracy by Model",
            )

            # Per-turn predictions for offline error analysis.
            # Joins to qa_data on (report_id, turn_index).
            predictions_path = run_dir / "dspy_predictions.csv"
            write_predictions_csv(predictions_path, opt_eval_result.results)
            print(f"\nWrote {predictions_path}")
            analyze_predictions(predictions_path)
            raise SystemExit(0)
        if gepa_name:
            raise RuntimeError(
                f"GEPA_NAME={gepa_name!r} was set, but neither "
                f"{run_dir / 'dspy_optimized_runner.json'} nor "
                f"{run_dir / 'optimized_runner.json'} exists. "
                "GEPA_NAME is only for load-and-evaluate of an existing run. "
                "Unset GEPA_NAME for a new run or use RESUME_GEPA to continue one."
            )

        # Resume vs new-run.
        # ------------------
        # Set RESUME_GEPA=<path> (or "latest") to resume a prior run. GEPA
        # resumes by reading `gepa_state.bin` from `log_dir`, so we point at
        # the same directory and only re-run if the trainset/valset/mode match.
        # Without RESUME_GEPA, we make a fresh timestamped directory tagged
        # by mode so smoke and real runs never collide.
        if resume_target == "latest":
            matches = sorted(Path("runs").glob(f"gepa_{gepa_mode}_*"), key=lambda p: p.name)
            if not matches:
                raise RuntimeError(
                    f"RESUME_GEPA=latest with GEPA_MODE={gepa_mode} but no "
                    f"runs/gepa_{gepa_mode}_* dirs exist"
                )
            run_dir = matches[-1]
            is_resume = True
        elif resume_target:
            run_dir = Path(resume_target)
            if not run_dir.exists() and not run_dir.is_absolute():
                candidate = Path("runs") / resume_target
                if candidate.exists():
                    run_dir = candidate
            if not run_dir.exists():
                raise RuntimeError(
                    f"RESUME_GEPA={resume_target} does not exist "
                    f"(checked {Path(resume_target)} and {Path('runs') / resume_target})"
                )
            is_resume = True
        else:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path("runs") / f"gepa_{gepa_mode}_{run_ts}"
            run_dir.mkdir(parents=True, exist_ok=True)
            is_resume = False

        print("\n" + "=" * 60)
        print(f"GEPA mode: {gepa_mode.upper()}  ({'resuming' if is_resume else 'new'}: {run_dir})")
        print("=" * 60)

        gepa_trainset = conv_examples_train[n_val:]
        gepa_valset = conv_examples_train[:n_val]
        print(f"trainset: {len(gepa_trainset)} conv | valset: {len(gepa_valset)} conv")
        print(f"GEPA kwargs: {gepa_kwargs}")

        # Config fingerprint. GEPA's saved Pareto state is keyed to specific
        # valset POSITIONS — resuming with a different valset would silently
        # mis-score candidates. Same goes for changing mode/num_preds.
        run_config = {
            "mode": gepa_mode,
            "gepa_kwargs": gepa_kwargs,
            "trainset_report_ids": [ex.report_id for ex in gepa_trainset],
            "valset_report_ids": [ex.report_id for ex in gepa_valset],
            "num_preds": len(ConversationRunner().predictors()),
        }
        config_path = run_dir / "config.json"
        if is_resume:
            if not config_path.exists():
                raise RuntimeError(
                    f"{run_dir} has no config.json — was this dir created by an "
                    "older agent.py? Resume is not safe without a recorded config."
                )
            saved_config = json.loads(config_path.read_text())
            mismatches = [k for k in run_config if saved_config.get(k) != run_config[k]]
            if mismatches:
                raise RuntimeError(
                    f"Cannot resume {run_dir} — config differs from saved state on "
                    f"{mismatches}. Resume requires identical mode/trainset/valset/"
                    "num_preds. Start a new run instead, or fix the diverging field."
                )
            print("Config matches saved state — resuming.")
        else:
            config_path.write_text(json.dumps(run_config, indent=2, default=str))

        # log_dir routes GEPA's own per-iteration logs (proposed prompts,
        # reflection traces, etc.) into the same directory as our artifacts.
        optimizer = dspy.GEPA(
            metric=conv_metric_with_feedback,
            num_threads=8,
            track_stats=True,
            track_best_outputs=True,        # persists best-per-val-task predictions
            log_dir=str(run_dir / "dspy_gepa_logs"),
            reflection_minibatch_size=3,
            reflection_lm=lm_max,
            **gepa_kwargs,
        )

        optimized_runner = optimizer.compile(
            ConversationRunner(),
            trainset=gepa_trainset,
            valset=gepa_valset,
        )

        # Re-evaluate the optimized runner on the held-out test set.
        opt_eval_result = dspy.Evaluate(
            devset=test_set,
            metric=conv_turn_accuracy,
            num_threads=8,
            display_progress=True,
        )(optimized_runner)

        print(f"\nBaseline turn accuracy:  {eval_result.score:.1f}%")
        print(f"Optimized turn accuracy: {opt_eval_result.score:.1f}%")
        print(f"Δ = {opt_eval_result.score - eval_result.score:+.1f} pts")
        baseline_joined = _eval_result_to_joined(eval_result, model_label="baseline")
        optimized_joined = _eval_result_to_joined(
            opt_eval_result,
            model_label="optimized",
        )
        print_model_accuracy_table(
            [baseline_joined, optimized_joined],
            slice_col="turn_type",
            title="Turn Type Accuracy by Model",
        )
        print_model_accuracy_table(
            [baseline_joined, optimized_joined],
            slice_col="conv_type",
            title="Conv Type Accuracy by Model",
        )

        # ---- Persist artifacts (all share the run_ts timestamp) ------------

        # 1. The optimized program (DSPy's native save format — JSON of
        #    instructions/demos per predictor; reload with `runner.load(path)`).
        program_path = run_dir / "dspy_optimized_runner.json"
        optimized_runner.save(str(program_path))
        compare_runner_instructions(optimized_runner, program_path)

        # 2. GEPA stats — Pareto frontier, per-candidate scores, lineage.
        # NOTE: dspy 3.2.0's DspyGEPAResult.to_dict() assumes candidates are
        # {name: text} dicts, but they're actually compiled `Module`s — calling
        # it raises AttributeError. Build the dict ourselves from the same
        # fields, extracting per-predictor instructions from each candidate.
        stats_path = run_dir / "dspy_gepa_stats.json"
        details = optimized_runner.detailed_results
        cand_instructions = [
            {name: pred.signature.instructions for name, pred in cand.named_predictors()}
            for cand in details.candidates
        ]
        stats = dict(
            candidates=cand_instructions,
            parents=details.parents,
            val_aggregate_scores=details.val_aggregate_scores,
            val_subscores=details.val_subscores,
            per_val_instance_best_candidates=[
                list(s) if hasattr(s, "__iter__") else s
                for s in details.per_val_instance_best_candidates
            ],
            discovery_eval_counts=details.discovery_eval_counts,
            total_metric_calls=details.total_metric_calls,
            num_full_val_evals=details.num_full_val_evals,
            log_dir=details.log_dir,
            seed=details.seed,
            best_idx=details.best_idx,
        )
        stats_path.write_text(json.dumps(stats, indent=2, default=str))

        # 3. Human-readable summary: which predictor's instructions changed,
        #    plus baseline-vs-optimized scores. This is what you read first.
        summary_path = run_dir / "dspy_summary.json"
        baseline = ConversationRunner()
        instr_diff = {
            name: {
                "baseline": baseline_pred.signature.instructions,
                "optimized": opt_pred.signature.instructions,
            }
            for (name, opt_pred), (_, baseline_pred) in zip(
                optimized_runner.named_predictors(),
                baseline.named_predictors(),
                strict=True,
            )
        }
        summary = {
            "run_tag": run_dir.name,
            "mode": gepa_mode,
            "gepa_kwargs": gepa_kwargs,
            "resumed": is_resume,
            "trainset_size": len(gepa_trainset),
            "valset_size": len(gepa_valset),
            "testset_size": len(test_set),
            "baseline_test_score": eval_result.score,
            "optimized_test_score": opt_eval_result.score,
            "delta_pts": opt_eval_result.score - eval_result.score,
            "total_metric_calls": stats.get("total_metric_calls"),
            "num_full_val_evals": stats.get("num_full_val_evals"),
            "best_candidate_idx": stats.get("best_idx"),
            "predictor_instructions": instr_diff,
        }
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

        # 4. Per-turn predictions for offline error analysis. One row per turn,
        #    join back to qa_data on (report_id, turn_index) where turn_index is
        #    the 0-based position within q_order-sorted turns for that report.
        predictions_path = run_dir / "dspy_predictions.csv"
        write_predictions_csv(predictions_path, opt_eval_result.results)

        print(f"\nArtifacts saved under {run_dir}/")
        print(f"  - dspy_optimized_runner.json ({program_path.stat().st_size:,} bytes)")
        print(f"  - dspy_gepa_stats.json      ({stats_path.stat().st_size:,} bytes)")
        print("  - dspy_summary.json         (human-readable diff + scores)")
        print("  - dspy_predictions.csv    (per-turn dump for analysis)")
        print("  - dspy_gepa_logs/         (GEPA's per-iteration logs)")

        analyze_predictions(predictions_path)
    else:
        print("\n(Skipping GEPA. Set RUN_GEPA=1 to compile an optimized runner.)")
