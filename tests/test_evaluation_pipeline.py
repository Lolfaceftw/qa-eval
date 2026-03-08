"""Regression tests for evaluator parsing and normalized scoring."""

from __future__ import annotations

from src.services.evaluation_pipeline import EvaluationPipeline


def test_process_aligns_answers_and_normalizes_case() -> None:
    """Keep the first valid answer for each known question number."""
    pipeline = EvaluationPipeline()
    processed_batch = pipeline.process(
        questions_by_category={
            "factualness": [
                {"question_number": 1, "question": "Does the summary preserve fact A?"},
                {"question_number": 2, "question": "Does the summary preserve fact B?"},
            ]
        },
        responses_by_category={
            "factualness": """
            [
              {"question_number": 999, "answer": "yes"},
              {"question_number": 1, "answer": "YES"},
              {"question_number": 1, "answer": "no"},
              {"question_number": 2, "answer": "No"}
            ]
            """
        },
    )

    assert processed_batch.answers_by_category["factualness"] == [
        {
            "question_number": 1,
            "question": "Does the summary preserve fact A?",
            "answer": "yes",
        },
        {
            "question_number": 2,
            "question": "Does the summary preserve fact B?",
            "answer": "no",
        },
    ]
    assert processed_batch.report["categories"]["factualness"] == {
        "total_questions": 2,
        "yes_count": 1,
        "no_count": 1,
        "invalid_or_missing_count": 0,
        "score": 0.5,
    }


def test_process_counts_invalid_and_missing_answers_as_no() -> None:
    """Coerce invalid or absent answers to no while tracking the shortfall."""
    pipeline = EvaluationPipeline()
    processed_batch = pipeline.process(
        questions_by_category={
            "naturalness": [
                {"question_number": 1, "question": "Does the summary preserve tone?"},
                {"question_number": 2, "question": "Does the summary preserve flow?"},
            ]
        },
        responses_by_category={
            "naturalness": """
            [
              {"question_number": 1, "answer": "maybe"},
              {"question_number": 3, "answer": "yes"}
            ]
            """
        },
    )

    assert processed_batch.answers_by_category["naturalness"] == [
        {
            "question_number": 1,
            "question": "Does the summary preserve tone?",
            "answer": "no",
        },
        {
            "question_number": 2,
            "question": "Does the summary preserve flow?",
            "answer": "no",
        },
    ]
    assert processed_batch.report["categories"]["naturalness"] == {
        "total_questions": 2,
        "yes_count": 0,
        "no_count": 2,
        "invalid_or_missing_count": 2,
        "score": 0.0,
    }


def test_process_uses_first_valid_duplicate_answer() -> None:
    """Ignore duplicate answers after the first valid one for a question."""
    pipeline = EvaluationPipeline()
    processed_batch = pipeline.process(
        questions_by_category={
            "factualness": [
                {"question_number": 1, "question": "Does the summary preserve fact A?"},
                {"question_number": 2, "question": "Does the summary preserve fact B?"},
            ]
        },
        responses_by_category={
            "factualness": """
            [
              {"question_number": 1, "answer": "maybe"},
              {"question_number": 1, "answer": "yes"},
              {"question_number": 1, "answer": "no"},
              {"question_number": 2, "answer": "no"}
            ]
            """
        },
    )

    assert processed_batch.answers_by_category["factualness"][0]["answer"] == "yes"
    assert processed_batch.report["categories"]["factualness"]["invalid_or_missing_count"] == 0


def test_process_reports_null_scores_for_empty_categories() -> None:
    """Report null scores when no final questions are available to score."""
    pipeline = EvaluationPipeline()
    processed_batch = pipeline.process(
        questions_by_category={
            "factualness": [],
            "naturalness": [],
        },
        responses_by_category={},
    )

    assert processed_batch.answers_by_category["factualness"] == []
    assert processed_batch.answers_by_category["naturalness"] == []
    assert processed_batch.report["categories"]["factualness"]["score"] is None
    assert processed_batch.report["categories"]["naturalness"]["score"] is None
    assert processed_batch.report["global"]["score"] is None
