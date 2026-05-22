from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dspy_eval_v1 import (
    EVAL_QUESTION_COUNT,
    EVAL_QUESTION_INDICES,
    build_agents,
    build_agent_system_prompt,
    build_devset,
    build_eval_sample,
    build_evaluation_page_data,
    cuad_overlap_metric,
    eval_results_to_dataframe,
    filter_eval_rows_by_split,
    load_prompt_overrides,
    output_paths,
    parse_bool,
    prompt_name_part,
    slugify_model_id,
    token_overlap_f1,
    write_system_prompts,
    write_evaluation_html,
)
import dspy
import pandas as pd
from cuad_agent.evaluators.langchain_runner import (
    append_result_jsonl,
    build_baseline_comparison,
    empty_results_dataframe,
    load_cached_results,
    load_jsonl_results,
    merge_result_frames,
    parse_cuad_answer,
    resolve_prompt_harness_paths,
)
from cuad_agent.prompts.loader import resolve_prompts_file
import prompt_improve_v2
from prompt_improve_v2 import is_complete_sentence_match, run_harness
import argparse
import threading


def test_token_overlap_f1_exact_and_partial() -> None:
    assert token_overlap_f1("Distributor Agreement", ["DISTRIBUTOR AGREEMENT"]) == 1.0
    partial = token_overlap_f1("Distributor", ["Distributor Agreement"])
    assert 0 < partial < 1


def test_token_overlap_f1_no_answer() -> None:
    assert token_overlap_f1("NO_ANSWER", []) == 1.0
    assert token_overlap_f1("", []) == 1.0
    assert token_overlap_f1("some clause text", []) == 0.0


def test_parse_bool() -> None:
    assert parse_bool(True)
    assert parse_bool("true")
    assert parse_bool("marked impossible")
    assert not parse_bool(False)
    assert not parse_bool("false")


def test_complete_sentence_match_requires_exact_sentence_boundaries() -> None:
    contract = (
        "The buyer may inspect the records. "
        "Neither party may assign this Agreement without consent. "
        "Notices must be in writing."
    )
    assert is_complete_sentence_match(
        "Neither party may assign this Agreement without consent.",
        contract,
    )
    assert not is_complete_sentence_match(
        "Neither party may assign this Agreement without consent",
        contract,
    )
    assert not is_complete_sentence_match("party may assign this Agreement", contract)


def test_model_id_slug_and_output_paths() -> None:
    assert slugify_model_id("deepseek/deepseek-v4-flash") == "deepseek-deepseek-v4-flash"
    assert slugify_model_id("OpenAI/GPT-5.4 Mini:baseline") == "openai-gpt-5.4-mini-baseline"
    paths = output_paths(Path("outputs"), "model-a", None)
    assert paths["model_dir"] == Path("outputs/model-a")
    assert paths["results"] == Path("outputs/model-a/cuad_dspy_eval_results.csv")
    assert paths["summary"] == Path("outputs/model-a/cuad_dspy_eval_summary.json")
    assert paths["html"] == Path("frontend/evaluation_model-a.html")
    assert paths["system_prompts"] == Path("outputs/model-a/system_prompts.py")
    explicit_paths = output_paths(Path("outputs"), "model-a", Path("custom.html"))
    assert explicit_paths["html"] == Path("frontend/custom.html")
    assert prompt_name_part("Anti-Assignment") == "ANTI_ASSIGNMENT"


def test_build_eval_sample_is_deterministic() -> None:
    ids_a, _, rows_a = build_eval_sample(sample_size=50, seed=42)
    ids_b, _, rows_b = build_eval_sample(sample_size=50, seed=42)
    assert ids_a == ids_b
    assert len(ids_a) == 50
    assert len(set(ids_a)) == 50
    assert len(rows_a) == 50 * EVAL_QUESTION_COUNT
    assert len(rows_b) == 50 * EVAL_QUESTION_COUNT
    assert set(rows_a["question_index"]) == set(EVAL_QUESTION_INDICES)


def test_build_agents_preserves_category_metadata() -> None:
    _, _, rows = build_eval_sample(sample_size=50, seed=42)
    agents = build_agents(rows, dry_run=True)
    assert len(agents) == EVAL_QUESTION_COUNT
    assert set(agents) == set(EVAL_QUESTION_INDICES)
    assert agents[0].category == "Document Name"
    assert agents[1].category == "Parties"
    assert agents[2].category == "Agreement Date"
    assert agents[40].category == "Third Party Beneficiary"
    assert len({type(agent) for agent in agents.values()}) == EVAL_QUESTION_COUNT
    assert (
        len({agent.signature_class for agent in agents.values()})
        == EVAL_QUESTION_COUNT
    )
    for question_index, agent in agents.items():
        question = rows.loc[
            rows["question_index"] == question_index, "question"
        ].iloc[0]
        assert agent.signature_class.__doc__ == agent.question
        assert agent.signature_class.__doc__ == question
        assert agent.__class__.__doc__ == agent.question
        assert "answer" in agent.signature_class.output_fields
        assert "marked_impossible" in agent.signature_class.output_fields


def test_write_system_prompts_uses_category_names(tmp_path: Path) -> None:
    _, _, rows = build_eval_sample(sample_size=1, seed=42)
    agents = build_agents(rows, dry_run=True)
    prompt = build_agent_system_prompt(agents[0])
    assert "Category:" in prompt
    assert "Document Name" in prompt
    output_path = tmp_path / "system_prompts.py"
    write_system_prompts({0: agents[0], 18: agents[18]}, output_path)
    text = output_path.read_text(encoding="utf-8")
    assert "DOCUMENT_NAME_SYSTEM_PROMPT" in text
    assert "ANTI_ASSIGNMENT_SYSTEM_PROMPT" in text
    assert "'Document Name': DOCUMENT_NAME_SYSTEM_PROMPT" in text


def test_build_agents_can_use_prompt_overrides(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.py"
    prompt_file.write_text(
        "DOCUMENT_NAME_SYSTEM_PROMPT = 'Custom document-name extraction prompt.'\n"
        "CATEGORY_SYSTEM_PROMPTS = {'Document Name': DOCUMENT_NAME_SYSTEM_PROMPT}\n",
        encoding="utf-8",
    )
    overrides = load_prompt_overrides(prompt_file)
    _, _, rows = build_eval_sample(sample_size=1, seed=42)
    agents = build_agents(rows, dry_run=True, prompt_overrides=overrides)
    assert agents[0].signature_class.__doc__ == "Custom document-name extraction prompt."
    assert build_agent_system_prompt(agents[0]) == "Custom document-name extraction prompt."
    assert agents[1].signature_class.__doc__ == agents[1].question


def test_resolve_prompts_file_uses_model_id_convention(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    v1_prompts = prompts_dir / "system_prompts_v1.py"
    v1_prompts.write_text("CATEGORY_SYSTEM_PROMPTS = {}\n", encoding="utf-8")
    explicit = tmp_path / "custom_prompts.py"
    explicit.write_text("CATEGORY_SYSTEM_PROMPTS = {}\n", encoding="utf-8")

    assert resolve_prompts_file(None, "v1", prompts_dir=prompts_dir) == v1_prompts
    assert resolve_prompts_file(None, "missing", prompts_dir=prompts_dir) is None
    assert resolve_prompts_file(explicit, "v1", prompts_dir=prompts_dir) == explicit


def test_filter_eval_rows_by_split_uses_row_ids() -> None:
    _, _, rows = build_eval_sample(sample_size=1, seed=42)
    first = rows.iloc[0]
    second = rows.iloc[1]
    split_ids = {
        f"{int(first.document_row_id)}:{int(first.question_index)}",
        f"{int(second.document_row_id)}:{int(second.question_index)}",
    }
    filtered = filter_eval_rows_by_split(rows, split_ids)
    assert len(filtered) == 2
    assert set(filtered["row_id"]) == split_ids


def test_dry_run_evaluation_shape_and_order() -> None:
    _, contract_lookup, rows = build_eval_sample(sample_size=2, seed=42)
    agents = build_agents(rows, dry_run=True)
    devset = build_devset(contract_lookup, rows, dry_run=True)
    eval_results = []
    for question_index in EVAL_QUESTION_INDICES:
        question_devset = [
            example for example in devset if example.question_index == question_index
        ]
        evaluator = dspy.Evaluate(
            devset=question_devset,
            metric=cuad_overlap_metric,
            num_threads=2,
            display_progress=False,
            display_table=False,
        )
        eval_results.append(evaluator(agents[question_index]))
    results = eval_results_to_dataframe(eval_results, model_id="dry-run-test")
    assert len(results) == 2 * EVAL_QUESTION_COUNT
    assert set(results["model_id"]) == {"dry-run-test"}
    assert set(results["question_index"]) == set(EVAL_QUESTION_INDICES)
    assert "predicted_marked_impossible" in results.columns
    assert "gold_marked_impossible" in results.columns
    assert results["predicted_marked_impossible"].equals(
        results["gold_marked_impossible"]
    )
    assert results[["document_row_id", "question_index"]].equals(
        results[["document_row_id", "question_index"]].sort_values(
            ["document_row_id", "question_index"]
        ).reset_index(drop=True)
    )


def test_build_evaluation_page_data_groups_by_question() -> None:
    results = pd.DataFrame(
        [
            {
                "document_row_id": 7,
                "title": "Contract A",
                "question_index": 0,
                "category": "Document Name",
                "category_description": "The name of the contract",
                "answer_format": "Contract Name",
                "question": "Question text",
                "gold_answers": '["Agreement"]',
                "predicted_answer": "Agreement",
                "predicted_marked_impossible": False,
                "gold_marked_impossible": False,
                "token_f1": 1.0,
                "correct_at_0_5": True,
            }
        ]
    )
    summary = {
        "sample_size": 1,
        "seed": 42,
        "model_id": "test-model-id",
        "total_examples": 1,
        "questions_per_contract": 41,
        "agent_count": 41,
        "model": "test-model",
        "temperature": 0,
        "max_tokens": 100,
        "num_threads": 1,
        "dry_run": True,
        "overlap_accuracy_mean_f1": 100,
        "correct_at_0_5": 100,
        "per_category": [
            {
                "question_index": 0,
                "category": "Document Name",
                "mean_token_f1": 100,
                "correct_at_0_5": 100,
                "count": 1,
            }
        ],
    }
    page_data = build_evaluation_page_data(results, summary)
    assert len(page_data["questions"]) == 1
    question = page_data["questions"][0]
    assert question["category"] == "Document Name"
    assert question["mean_token_f1"] == 100
    assert question["results"][0]["gold_answers"] == ["Agreement"]
    assert question["results"][0]["predicted_answer"] == "Agreement"
    assert page_data["summary"]["model_id"] == "test-model-id"
    summary_row = page_data["per_category"][0]
    assert summary_row["category_description"] == "The name of the contract"
    assert summary_row["answer_format"] == "Contract Name"


def test_write_evaluation_html_creates_static_page(tmp_path: Path) -> None:
    results = pd.DataFrame(
        [
            {
                "document_row_id": 7,
                "title": "Contract A",
                "question_index": 0,
                "category": "Document Name",
                "category_description": "The name of the contract",
                "answer_format": "Contract Name",
                "question": "Question text",
                "gold_answers": '["Agreement"]',
                "predicted_answer": "Agreement",
                "predicted_marked_impossible": False,
                "gold_marked_impossible": False,
                "token_f1": 1.0,
                "correct_at_0_5": True,
            }
        ]
    )
    summary = {
        "sample_size": 1,
        "seed": 42,
        "model_id": "test-model-id",
        "total_examples": 1,
        "questions_per_contract": 41,
        "agent_count": 41,
        "model": "test-model",
        "temperature": 0,
        "max_tokens": 100,
        "num_threads": 1,
        "dry_run": True,
        "overlap_accuracy_mean_f1": 100,
        "correct_at_0_5": 100,
        "per_category": [
            {
                "question_index": 0,
                "category": "Document Name",
                "mean_token_f1": 100,
                "correct_at_0_5": 100,
                "count": 1,
            }
        ],
    }
    output_path = tmp_path / "evaluation.html"
    write_evaluation_html(results, summary, output_path)
    html = output_path.read_text(encoding="utf-8")
    assert "<title>CUAD Evaluation</title>" in html
    assert "CUAD Evaluation" in html
    assert '<a class="tab" href="explore.html">Explorer</a>' in html
    assert (
        '<a class="tab active" href="evaluation_test-model-id.html" '
        'aria-current="page">'
    ) in html
    assert "Model ID:" in html
    assert "test-model-id" in html
    assert "category_descriptions.csv context" in html
    assert "Question text" in html


def test_prompt_improve_v2_writes_harness_artifacts(tmp_path: Path, monkeypatch) -> None:
    source_results = tmp_path / "cuad_dspy_eval_results.csv"
    pd.DataFrame(
        [
            {
                "model_id": "v1",
                "row_id": "1:18",
                "document_row_id": 1,
                "title": "Contract A",
                "question_index": 18,
                "category": "Anti-Assignment",
                "category_description": "Is consent or notice required for assignment?",
                "answer_format": "Yes/No",
                "question": "Highlight Anti-Assignment clauses.",
                "gold_answers": '["Neither party may assign this Agreement without consent."]',
                "predicted_answer": "Yes",
                "predicted_marked_impossible": False,
                "gold_marked_impossible": False,
                "token_f1": 0.0,
                "correct_at_0_5": False,
            },
            {
                "model_id": "v1",
                "row_id": "2:18",
                "document_row_id": 2,
                "title": "Contract B",
                "question_index": 18,
                "category": "Anti-Assignment",
                "category_description": "Is consent or notice required for assignment?",
                "answer_format": "Yes/No",
                "question": "Highlight Anti-Assignment clauses.",
                "gold_answers": '["Assignment requires prior written notice."]',
                "predicted_answer": "NO_ANSWER",
                "predicted_marked_impossible": True,
                "gold_marked_impossible": False,
                "token_f1": 0.0,
                "correct_at_0_5": False,
            },
            {
                "model_id": "v1",
                "row_id": "3:18",
                "document_row_id": 3,
                "title": "Contract C",
                "question_index": 18,
                "category": "Anti-Assignment",
                "category_description": "Is consent or notice required for assignment?",
                "answer_format": "Yes/No",
                "question": "Highlight Anti-Assignment clauses.",
                "gold_answers": '["This Agreement may not be transferred except by merger."]',
                "predicted_answer": "This Agreement may not be transferred",
                "predicted_marked_impossible": False,
                "gold_marked_impossible": False,
                "token_f1": 0.25,
                "correct_at_0_5": False,
            },
            {
                "model_id": "v1",
                "row_id": "4:0",
                "document_row_id": 4,
                "title": "Contract D",
                "question_index": 0,
                "category": "Document Name",
                "category_description": "The name of the contract",
                "answer_format": "Contract Name",
                "question": "Highlight Document Name clauses.",
                "gold_answers": '["MASTER SERVICES AGREEMENT"]',
                "predicted_answer": "MASTER SERVICES AGREEMENT",
                "predicted_marked_impossible": False,
                "gold_marked_impossible": False,
                "token_f1": 1.0,
                "correct_at_0_5": True,
            },
        ]
    ).to_csv(source_results, index=False)
    prompts_file = tmp_path / "system_prompts_v1.py"
    prompts_file.write_text(
        "ANTI_ASSIGNMENT_SYSTEM_PROMPT = 'Base Anti-Assignment prompt.'\n"
        "CATEGORY_SYSTEM_PROMPTS = {'Anti-Assignment': ANTI_ASSIGNMENT_SYSTEM_PROMPT}\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        source_results=source_results,
        prompts_file=prompts_file,
        model_id="v2",
        output_dir=tmp_path / "outputs",
        generator_examples=1,
        evaluator_examples=1,
        max_loops=3,
        max_workers=1,
        dry_run=True,
    )
    monkeypatch.setattr(
        prompt_improve_v2,
        "evaluate_patch_with_agent",
        lambda agent, request: prompt_improve_v2.PromptReview(
            decision="reject",
            generalization_score=0.2,
            rationale=["forced rejection for candidate-prompt persistence test"],
            likely_fixes=[],
            likely_regressions=["test regression risk"],
            requested_changes=["test requested change"],
        ),
    )
    run_harness(args)
    harness_dir = tmp_path / "outputs" / "v2" / "prompt_harness"
    assert (harness_dir / "splits.json").exists()
    assert (harness_dir / "answer_format_profiles.json").exists()
    assert (harness_dir / "category_runs.jsonl").exists()
    assert (harness_dir / "evaluator_reviews.jsonl").exists()
    assert (harness_dir / "accepted_patches.jsonl").exists()
    assert (harness_dir / "prompt_diffs.jsonl").exists()
    category_runs = (harness_dir / "category_runs.jsonl").read_text(encoding="utf-8")
    assert '"category": "Anti-Assignment"' in category_runs
    assert '"category": "Document Name"' not in category_runs
    candidate = (harness_dir / "prompts_candidate_v2.py").read_text(encoding="utf-8")
    assert "CATEGORY_SYSTEM_PROMPTS" in candidate
    assert "Anti-Assignment" in candidate
    assert "Document Name" in candidate
    assert "V2 extraction guidance" in candidate
    rejected = (harness_dir / "rejected_patches.jsonl").read_text(encoding="utf-8")
    assert '"category": "Anti-Assignment"' in rejected
    dashboard = (tmp_path / "frontend" / "prompt_review_v2.html").read_text(
        encoding="utf-8"
    )
    assert "CUAD V2 Prompt Review" in dashboard
    assert "Anti-Assignment" in dashboard
    assert "Document Name" in dashboard
    assert "no_failures" in dashboard
    assert "Change Insight" in dashboard
    assert "classification_instead_of_span" in dashboard
    assert "verbatim_contract_span" in dashboard


def test_prompt_improve_v2_reruns_prior_revise_status(tmp_path: Path) -> None:
    source_results = tmp_path / "cuad_dspy_eval_results.csv"
    pd.DataFrame(
        [
            {
                "model_id": "v1",
                "row_id": "1:18",
                "document_row_id": 1,
                "title": "Contract A",
                "question_index": 18,
                "category": "Anti-Assignment",
                "category_description": "Is consent or notice required for assignment?",
                "answer_format": "Yes/No",
                "question": "Highlight Anti-Assignment clauses.",
                "gold_answers": '["Neither party may assign this Agreement without consent."]',
                "predicted_answer": "Yes",
                "predicted_marked_impossible": False,
                "gold_marked_impossible": False,
                "token_f1": 0.0,
                "correct_at_0_5": False,
            },
        ]
    ).to_csv(source_results, index=False)
    prompts_file = tmp_path / "system_prompts_v1.py"
    prompts_file.write_text(
        "ANTI_ASSIGNMENT_SYSTEM_PROMPT = 'Base Anti-Assignment prompt.'\n"
        "CATEGORY_SYSTEM_PROMPTS = {'Anti-Assignment': ANTI_ASSIGNMENT_SYSTEM_PROMPT}\n",
        encoding="utf-8",
    )
    harness_dir = tmp_path / "outputs" / "v2" / "prompt_harness"
    harness_dir.mkdir(parents=True)
    (harness_dir / "category_status.jsonl").write_text(
        (
            '{"category": "Anti-Assignment", "decision": "revise", '
            '"loop_count": 1, "candidate_prompt": "stale revise prompt", '
            '"v1_prompt": "Base Anti-Assignment prompt.", '
            '"main_failure_mode": "classification_instead_of_span"}\n'
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        source_results=source_results,
        prompts_file=prompts_file,
        model_id="v2",
        output_dir=tmp_path / "outputs",
        generator_examples=1,
        evaluator_examples=0,
        max_loops=1,
        max_workers=1,
        dry_run=True,
    )
    run_harness(args)
    category_runs = (harness_dir / "category_runs.jsonl").read_text(encoding="utf-8")
    assert '"category": "Anti-Assignment"' in category_runs
    assert "stale revise prompt" not in category_runs


def test_langchain_eval_resolves_prompt_improve_harness_paths(tmp_path: Path) -> None:
    harness_dir = tmp_path / "outputs" / "v2" / "prompt_harness"
    harness_dir.mkdir(parents=True)
    prompts_file = harness_dir / "prompts_candidate_v2.py"
    prompts_file.write_text(
        "CATEGORY_SYSTEM_PROMPTS = {'Document Name': 'candidate prompt'}\n",
        encoding="utf-8",
    )
    (harness_dir / "splits.json").write_text(
        '{"splits": {"holdout_eval": ["1:0"]}}\n',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        prompt_harness_dir=harness_dir,
        prompts_file=None,
        eval_split=None,
        use_harness_holdout=True,
    )
    resolve_prompt_harness_paths(args)
    assert args.prompts_file == prompts_file
    assert args.eval_split == f"{harness_dir / 'splits.json'}:holdout_eval"


def test_langchain_eval_compares_against_v1_results(tmp_path: Path) -> None:
    baseline_path = tmp_path / "v1_results.csv"
    pd.DataFrame(
        [
            {
                "row_id": "1:0",
                "document_row_id": 1,
                "question_index": 0,
                "category": "Document Name",
                "token_f1": 0.25,
                "correct_at_0_5": False,
                "predicted_answer": "old",
                "predicted_marked_impossible": False,
            }
        ]
    ).to_csv(baseline_path, index=False)
    current = pd.DataFrame(
        [
            {
                "row_id": "1:0",
                "document_row_id": 1,
                "question_index": 0,
                "category": "Document Name",
                "token_f1": 0.75,
                "correct_at_0_5": True,
                "predicted_answer": "new",
                "predicted_marked_impossible": False,
            }
        ]
    )
    comparison = build_baseline_comparison(current, baseline_path)
    assert comparison is not None
    assert comparison["matched_examples"] == 1
    assert comparison["mean_token_f1_delta"] == 50.0
    assert comparison["correct_at_0_5_delta"] == 100.0
    assert comparison["per_category"][0]["category"] == "Document Name"
    assert comparison["examples"][0]["baseline_predicted_answer"] == "old"
    assert comparison["examples"][0]["token_f1_delta"] == 0.5


def test_langchain_eval_loads_cached_completed_rows(tmp_path: Path) -> None:
    cache_path = tmp_path / "outputs" / "v2" / "cuad_dspy_eval_results.csv"
    cached_rows = pd.DataFrame(
        [
            {
                "model_id": "v2",
                "row_id": "1:0",
                "document_row_id": 1,
                "title": "Contract A",
                "question_index": 0,
                "category": "Document Name",
                "category_description": "The name of the contract",
                "answer_format": "Contract Name",
                "question": "Highlight Document Name clauses.",
                "gold_answers": '["MASTER SERVICES AGREEMENT"]',
                "predicted_answer": "MASTER SERVICES AGREEMENT",
                "predicted_marked_impossible": False,
                "gold_marked_impossible": False,
                "token_f1": 1.0,
                "correct_at_0_5": True,
                "is_impossible": False,
                "answers_len": 1,
            },
            {
                "model_id": "other",
                "row_id": "1:1",
                "document_row_id": 1,
                "title": "Contract A",
                "question_index": 1,
                "category": "Parties",
                "category_description": "The parties to the contract",
                "answer_format": "Names",
                "question": "Highlight party clauses.",
                "gold_answers": '["Party A"]',
                "predicted_answer": "Party A",
                "predicted_marked_impossible": False,
                "gold_marked_impossible": False,
                "token_f1": 1.0,
                "correct_at_0_5": True,
                "is_impossible": False,
                "answers_len": 1,
            },
        ]
    )
    cache_path.parent.mkdir(parents=True)
    cached_rows.to_csv(cache_path, index=False)
    eval_rows = pd.DataFrame(
        [
            {"document_row_id": 1, "question_index": 0},
            {"document_row_id": 1, "question_index": 1},
        ]
    )
    loaded = load_cached_results(cache_path, eval_rows=eval_rows, model_id="v2")
    assert loaded["row_id"].tolist() == ["1:0"]


def test_langchain_eval_merges_new_rows_over_cache() -> None:
    cached = pd.DataFrame(
        [
            {
                "model_id": "v2",
                "row_id": "1:0",
                "document_row_id": 1,
                "question_index": 0,
                "token_f1": 0.25,
            }
        ]
    )
    new = pd.DataFrame(
        [
            {
                "model_id": "v2",
                "row_id": "1:0",
                "document_row_id": 1,
                "question_index": 0,
                "token_f1": 0.75,
            },
            {
                "model_id": "v2",
                "row_id": "1:1",
                "document_row_id": 1,
                "question_index": 1,
                "token_f1": 1.0,
            },
        ]
    )
    merged = merge_result_frames(cached, new)
    assert merged["row_id"].tolist() == ["1:0", "1:1"]
    assert merged.loc[merged["row_id"] == "1:0", "token_f1"].iloc[0] == 0.75
    assert empty_results_dataframe().empty


def test_langchain_eval_appends_and_loads_jsonl_results(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "outputs" / "v2" / "cuad_dspy_eval_results.jsonl"
    base_record = {
        "model_id": "v2",
        "row_id": "1:0",
        "document_row_id": 1,
        "title": "Contract A",
        "question_index": 0,
        "category": "Document Name",
        "category_description": "The name of the contract",
        "answer_format": "Contract Name",
        "question": "Highlight Document Name clauses.",
        "gold_answers": '["MASTER SERVICES AGREEMENT"]',
        "predicted_answer": "old",
        "predicted_marked_impossible": False,
        "gold_marked_impossible": False,
        "token_f1": 0.25,
        "correct_at_0_5": False,
        "is_impossible": False,
        "answers_len": 1,
    }
    append_result_jsonl(jsonl_path, base_record, threading.Lock())
    append_result_jsonl(
        jsonl_path,
        {**base_record, "predicted_answer": "new", "token_f1": 1.0, "correct_at_0_5": True},
        threading.Lock(),
    )
    eval_rows = pd.DataFrame([{"document_row_id": 1, "question_index": 0}])
    loaded = load_jsonl_results(jsonl_path, eval_rows=eval_rows, model_id="v2")
    assert loaded["row_id"].tolist() == ["1:0"]
    assert loaded["predicted_answer"].iloc[0] == "new"
    assert loaded["token_f1"].iloc[0] == 1.0


def test_langchain_eval_accepts_unstructured_model_output() -> None:
    parsed = parse_cuad_answer("MacroGenics, Inc.\nGreen Cross Corp.")
    assert parsed.answer == "MacroGenics, Inc.\nGreen Cross Corp."
    assert not parsed.marked_impossible
    no_answer = parse_cuad_answer("NO_ANSWER")
    assert no_answer.answer == "NO_ANSWER"
    assert no_answer.marked_impossible
