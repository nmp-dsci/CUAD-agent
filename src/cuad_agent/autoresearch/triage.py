"""Triage step for autoresearch: diagnose wrong answers using an LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from cuad_agent.autoresearch.llm import make_llm
from cuad_agent.autoresearch.prompts.triage_system_prompt import TRIAGE_SYSTEM_PROMPT
from cuad_agent.autoresearch.results import TriageDiagnosis
from cuad_agent.data.dataset import load_datasets
from cuad_agent.prompts.loader import load_prompt_overrides

__all__ = ["triage"]


def triage(
    *,
    eval_results_path: Path,
    question_index: int,
    category: str,
    prompts_file: Path,
    model_id: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 64000,
    output_path: Path,
    dry_run: bool = False,
) -> list[TriageDiagnosis]:
    """Diagnose wrong answers for a single question_index / category.

    Parameters
    ----------
    eval_results_path:
        Path to the CSV produced by the langchain runner, e.g.
        ``outputs/{model_id}/cuad_langchain_eval_results.csv``.
    question_index:
        The question index (0-based) to triage.
    category:
        The CUAD category name (used to look up the system prompt).
    prompts_file:
        Python file that defines ``CATEGORY_SYSTEM_PROMPTS``.
    model_id:
        Identifier for the evaluation being triaged; used to locate results
        and as a cache key.
    model:
        LangChain model string, e.g. ``"deepseek/deepseek-v4-flash"``.
    temperature:
        Sampling temperature for the triage LLM.
    max_tokens:
        Maximum tokens for the triage LLM response.
    output_path:
        Path to write JSONL output, e.g. ``iter_{N}/triage_outputs.jsonl``.
    dry_run:
        If True, return stub diagnoses without calling the LLM.

    Returns
    -------
    list[TriageDiagnosis]
        One diagnosis per wrong-answer row filtered from ``eval_results_path``.
    """
    # ------------------------------------------------------------------
    # 1. Cache check
    # ------------------------------------------------------------------
    if output_path.exists() and output_path.stat().st_size > 0:
        diagnoses: list[TriageDiagnosis] = []
        for line in output_path.read_text().splitlines():
            line = line.strip()
            if line:
                diagnoses.append(TriageDiagnosis.model_validate(json.loads(line)))
        return diagnoses

    # ------------------------------------------------------------------
    # 2. Load and filter eval results
    # ------------------------------------------------------------------
    eval_df = pd.read_csv(eval_results_path)
    wrong_rows = eval_df[
        (eval_df["correct_at_0_5"] == 0) & (eval_df["question_index"] == question_index)
    ].copy()

    # ------------------------------------------------------------------
    # 3. dry_run — return stubs (one per wrong row)
    # ------------------------------------------------------------------
    if dry_run:
        stubs: list[TriageDiagnosis] = []
        for _, row in wrong_rows.iterrows():
            stubs.append(
                TriageDiagnosis(
                    contract_id=int(row["document_row_id"]),
                    golden_answer_location="dry_run",
                    failure_reason="dry_run",
                    proposed_rule="dry_run",
                    confidence="low",
                )
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fh:
            for stub in stubs:
                fh.write(json.dumps(stub.model_dump()) + "\n")
        return stubs

    # ------------------------------------------------------------------
    # 4. Normal operation: load supporting data
    # ------------------------------------------------------------------
    datasets = load_datasets()
    contracts_df: pd.DataFrame = datasets["contracts"]
    questions_df: pd.DataFrame = datasets["questions"]

    prompt_overrides = load_prompt_overrides(prompts_file)
    system_prompt_text = prompt_overrides.get(category, "")

    llm = make_llm(model=model, temperature=temperature, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # 5. Triage each wrong row
    # ------------------------------------------------------------------
    diagnoses_list: list[TriageDiagnosis] = []

    for _, row in wrong_rows.iterrows():
        document_row_id = int(row["document_row_id"])

        # Fetch contract text
        contract_rows = contracts_df[contracts_df["document_row_id"] == document_row_id]
        if contract_rows.empty:
            continue
        contract_row = contract_rows.iloc[0]
        contract_title = str(contract_row.get("title", ""))
        contract_text = str(contract_row.get("context", ""))

        # Fetch question text for this document + question_index
        q_rows = questions_df[
            (questions_df["document_row_id"] == document_row_id)
            & (questions_df["question_index"] == question_index)
        ]
        question_text = (
            str(q_rows.iloc[0]["question"]) if not q_rows.empty else category
        )

        # Parse provided and golden answers
        provided_answer = str(row.get("predicted_answer", ""))
        raw_golden = row.get("golden_answer", "[]")
        try:
            golden_list: list[str] = json.loads(raw_golden)
        except (json.JSONDecodeError, TypeError):
            golden_list = [str(raw_golden)]
        golden_answer = "; ".join(golden_list)

        # Build messages — never use ChatPromptTemplate (contract text may contain {})
        human_content = (
            f"system_prompt:\n{system_prompt_text}\n\n"
            f"question:\n{question_text}\n\n"
            f"contract_title:\n{contract_title}\n\n"
            f"contract_text:\n{contract_text}\n\n"
            f"provided_answer:\n{provided_answer}\n\n"
            f"golden_answer:\n{golden_answer}"
        )
        messages = [
            SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        response = llm.invoke(messages)
        raw_text: str = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        raw_text = raw_text.strip()

        # Strip markdown fences if present, then find the first {...} block.
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.MULTILINE)
        raw_text = re.sub(r"\s*```\s*$", "", raw_text, flags=re.MULTILINE)
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            print(
                f"[triage] WARNING: no JSON object in response for contract "
                f"{document_row_id} — skipping",
                flush=True,
            )
            continue
        try:
            diagnosis_data = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            print(
                f"[triage] WARNING: JSON parse error for contract "
                f"{document_row_id}: {exc} — skipping",
                flush=True,
            )
            continue
        diagnosis_data["contract_id"] = document_row_id
        diagnosis = TriageDiagnosis.model_validate(diagnosis_data)
        diagnoses_list.append(diagnosis)

    # ------------------------------------------------------------------
    # 6. Write output JSONL
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for diag in diagnoses_list:
            fh.write(json.dumps(diag.model_dump()) + "\n")

    return diagnoses_list
