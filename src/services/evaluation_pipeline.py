"""Validate evaluator responses and compute normalized category scores.

References:
    [1] L. S. Feldt, "Confidence Intervals for the Proportion of Mastery in
    Criterion-Referenced Measurement," Journal of Educational Measurement,
    vol. 33, no. 1, pp. 106-114, Mar. 1996, doi: 10.1111/j.1745-3984.1996.tb00482.x.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src.models.evaluation_models import (
    EvaluationAnswerPayload,
    EvaluableQuestionPayload,
)
from src.prompts.templates import QUESTION_REQUEST_MINIMUMS


@dataclass(slots=True)
class CategoryEvaluationResult:
    """Store the per-question answers and category score details."""

    answers: list[dict[str, Any]]
    total_questions: int
    yes_count: int
    no_count: int
    invalid_or_missing_count: int
    score: float | None


@dataclass(slots=True)
class ProcessedEvaluationBatch:
    """Store the final evaluation answers artifact and summary report."""

    answers_by_category: dict[str, list[dict[str, Any]]]
    report: dict[str, Any]


class EvaluationPipeline:
    """Align evaluator outputs with processed questions and score them."""

    def __init__(
        self,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Store an optional UI log callback."""
        self.log_callback = log_callback

    def _log(self, message: str) -> None:
        """Emit a UI-friendly log line."""
        if self.log_callback:
            self.log_callback(f"- {message}\n")

    def process(
        self,
        questions_by_category: Mapping[str, Sequence[Mapping[str, object]]],
        responses_by_category: Mapping[str, str],
    ) -> ProcessedEvaluationBatch:
        """Evaluate all categories and return the answers artifact plus report."""
        ordered_categories = list(QUESTION_REQUEST_MINIMUMS)
        ordered_categories.extend(
            category
            for category in questions_by_category
            if category not in QUESTION_REQUEST_MINIMUMS
        )

        answers_by_category: dict[str, list[dict[str, Any]]] = {}
        category_reports: dict[str, dict[str, Any]] = {}

        for category in ordered_categories:
            questions = questions_by_category.get(category, [])
            response_text = responses_by_category.get(category, "")
            category_result = self._evaluate_category(
                category=category,
                questions=questions,
                response_text=response_text,
            )
            answers_by_category[category] = category_result.answers
            category_reports[category] = {
                "total_questions": category_result.total_questions,
                "yes_count": category_result.yes_count,
                "no_count": category_result.no_count,
                "invalid_or_missing_count": category_result.invalid_or_missing_count,
                "score": category_result.score,
            }

        report = {
            "global": self._build_global_report(category_reports.values()),
            "categories": category_reports,
        }
        return ProcessedEvaluationBatch(
            answers_by_category=answers_by_category,
            report=report,
        )

    def _evaluate_category(
        self,
        category: str,
        questions: Sequence[Mapping[str, object]],
        response_text: str,
    ) -> CategoryEvaluationResult:
        """Evaluate one category and compute its normalized score."""
        validated_questions = [
            EvaluableQuestionPayload.model_validate(question)
            for question in questions
        ]
        raw_answers = self._extract_json_array(response_text)
        expected_numbers = {question.question_number for question in validated_questions}
        selected_answers: dict[int, str] = {}

        for payload in raw_answers:
            if not isinstance(payload, dict):
                continue
            try:
                parsed_payload = EvaluationAnswerPayload.model_validate(payload)
            except ValidationError:
                continue
            if parsed_payload.question_number not in expected_numbers:
                continue
            selected_answers.setdefault(
                parsed_payload.question_number,
                parsed_payload.answer,
            )

        invalid_or_missing_count = 0
        serialized_answers: list[dict[str, Any]] = []
        for question in validated_questions:
            answer = selected_answers.get(question.question_number)
            if answer is None:
                invalid_or_missing_count += 1
                answer = "no"
            serialized_answers.append(
                {
                    "question_number": question.question_number,
                    "question": question.question,
                    "answer": answer,
                }
            )

        yes_count = sum(answer["answer"] == "yes" for answer in serialized_answers)
        total_questions = len(serialized_answers)
        no_count = total_questions - yes_count
        score = self._score_category(yes_count=yes_count, total_questions=total_questions)
        self._log(
            f"Scored {category}: {yes_count} yes / {total_questions} total "
            f"(invalid or missing: {invalid_or_missing_count})."
        )
        return CategoryEvaluationResult(
            answers=serialized_answers,
            total_questions=total_questions,
            yes_count=yes_count,
            no_count=no_count,
            invalid_or_missing_count=invalid_or_missing_count,
            score=score,
        )

    @staticmethod
    def _extract_json_array(response_text: str) -> list[Any]:
        """Decode the first top-level JSON array found in the response."""
        decoder = json.JSONDecoder()
        for index, character in enumerate(response_text):
            if character != "[":
                continue
            try:
                payload, _ = decoder.raw_decode(response_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, list):
                return payload
        return []

    @staticmethod
    def _score_category(yes_count: int, total_questions: int) -> float | None:
        """Compute the normalized score for one category."""
        if total_questions == 0:
            return None
        # IEEE citation: Adapt simple proportion-of-mastery normalization from [1].
        return yes_count / total_questions

    def _build_global_report(
        self,
        category_reports: Sequence[Mapping[str, object]],
    ) -> dict[str, Any]:
        """Aggregate the category metrics into one global evaluation report."""
        total_questions = sum(
            int(category_report["total_questions"])
            for category_report in category_reports
        )
        yes_count = sum(int(category_report["yes_count"]) for category_report in category_reports)
        invalid_or_missing_count = sum(
            int(category_report["invalid_or_missing_count"])
            for category_report in category_reports
        )
        no_count = total_questions - yes_count
        return {
            "total_questions": total_questions,
            "yes_count": yes_count,
            "no_count": no_count,
            "invalid_or_missing_count": invalid_or_missing_count,
            "score": self._score_category(
                yes_count=yes_count,
                total_questions=total_questions,
            ),
        }
