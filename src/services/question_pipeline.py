"""Parse, validate, and deduplicate generated questions."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from src.models.question_models import GeneratedQuestionPayload
from src.prompts.templates import QUESTION_REQUEST_MINIMUMS
from src.services.question_processor import QuestionProcessor

FACTUALNESS_DIMENSIONS = {
    "attribution",
    "numerical_detail",
    "chronology",
    "causality",
    "qualification",
    "action_outcome",
    "entity_relation",
}
NATURALNESS_DIMENSIONS = {
    "tone",
    "flow",
    "transition",
    "personality",
    "hedging",
    "emphasis",
    "pacing",
    "dialogue_dynamics",
}
CATEGORY_PRIORITY = {
    "factualness": 0,
    "naturalness": 1,
}
NATURALNESS_MARKERS = {
    "conversational",
    "dialogue",
    "emphasis",
    "flow",
    "hedg",
    "pace",
    "pacing",
    "personality",
    "robotic",
    "tone",
    "transition",
    "voice",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "when",
    "with",
}
SUMMARY_WRAPPER_PATTERNS = (
    r"\bthe summary\b",
    r"\bsummary\b",
    r"\bprovided summary\b",
    r"\bin the summary\b",
    r"\bfrom the summary\b",
    r"\bcompared to the transcript\b",
    r"\bcompared with the transcript\b",
    r"\baccording to the transcript\b",
    r"\baccording to the summary\b",
)
QUESTION_VERB_PATTERNS = (
    r"\bmention(?:ed|s|ing)?\b",
    r"\binclude(?:d|s|ing)?\b",
    r"\bcover(?:ed|s|ing)?\b",
    r"\breference(?:d|s|ing)?\b",
    r"\breflect(?:ed|s|ing)?\b",
    r"\bcaptur(?:e|ed|es|ing)\b",
    r"\baddress(?:ed|es|ing)?\b",
    r"\bdiscuss(?:ed|es|ing)?\b",
    r"\bstate(?:d|s|ing)?\b",
    r"\bnote(?:d|s|ing)?\b",
    r"\bfind(?:s|ing)?\b",
    r"\bfound\b",
    r"\blist(?:ed|s|ing)?\b",
    r"\bshow(?:ed|s|ing)?\b",
    r"\bpreserve(?:d|s|ing)?\b",
)
LEADING_AUXILIARY_PATTERN = re.compile(
    r"^(is|does|do|did|are|was|were|can|could|should|would|will)\b"
)


@dataclass(slots=True)
class QuestionCandidate:
    """Carry a validated question through filtering and deduplication."""

    question_number: int | None
    question: str
    dimension: str | None
    source_category: str
    original_order: int
    canonical_question: str
    specificity_score: int


@dataclass(slots=True)
class ParsedQuestionBatch:
    """Store parsed questions and parse-time metrics for one category."""

    questions: list[QuestionCandidate]
    report: dict[str, Any]


@dataclass(slots=True)
class ProcessedQuestionBatch:
    """Store the final question output and filtering metrics."""

    questions_by_category: dict[str, list[dict[str, Any]]]
    report: dict[str, Any]


@dataclass(slots=True)
class RemovedQuestion:
    """Track which representative removed a duplicate question."""

    question: QuestionCandidate
    winner: QuestionCandidate


def canonicalize_question(question: str) -> str:
    """Normalize boilerplate-heavy yes/no questions into comparable text."""
    normalized = question.strip().lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = LEADING_AUXILIARY_PATTERN.sub("", normalized).strip()

    for pattern in SUMMARY_WRAPPER_PATTERNS:
        normalized = re.sub(pattern, " ", normalized)

    for pattern in QUESTION_VERB_PATTERNS:
        normalized = re.sub(pattern, " ", normalized)

    normalized = re.sub(r"\b(answerable|evaluator|reviewer|reader|listener)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class QuestionPipeline:
    """Run post-generation validation, deduplication, and reporting."""

    def __init__(
        self,
        question_processor: QuestionProcessor,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Initialize the pipeline with an embedding-backed processor."""
        self.question_processor = question_processor
        self.log_callback = log_callback

    def _log(self, message: str) -> None:
        """Emit a UI-friendly log line."""
        if self.log_callback:
            self.log_callback(f"- {message}\n")

    def parse_generation_response(
        self,
        category: str,
        response_text: str,
    ) -> ParsedQuestionBatch:
        """Extract and validate the first JSON array in a model response."""
        payload_items = self._extract_json_array(response_text)

        report: dict[str, Any] = {
            "requested": QUESTION_REQUEST_MINIMUMS.get(category, 200),
            "raw_items": len(payload_items),
            "validated": 0,
            "invalid": 0,
            "invalid_examples": [],
        }
        questions: list[QuestionCandidate] = []

        for index, payload in enumerate(payload_items, start=1):
            if not isinstance(payload, dict):
                report["invalid"] += 1
                self._record_example(
                    report["invalid_examples"],
                    f"Item {index} is not an object.",
                )
                continue

            try:
                parsed_payload = GeneratedQuestionPayload.model_validate(payload)
            except ValidationError as exc:
                report["invalid"] += 1
                self._record_example(
                    report["invalid_examples"],
                    f"Item {index} failed validation: {exc.errors()[0]['msg']}",
                )
                continue

            canonical_question = canonicalize_question(parsed_payload.question)
            questions.append(
                QuestionCandidate(
                    question_number=parsed_payload.question_number,
                    question=parsed_payload.question,
                    dimension=parsed_payload.dimension,
                    source_category=category,
                    original_order=index,
                    canonical_question=canonical_question,
                    specificity_score=self._specificity_score(canonical_question),
                )
            )

        report["validated"] = len(questions)
        self._log(
            f"Parsed {len(questions)} valid {category} questions "
            f"(discarded {report['invalid']} invalid items)."
        )
        return ParsedQuestionBatch(questions=questions, report=report)

    def process(
        self,
        questions_by_category: dict[str, Sequence[QuestionCandidate]],
    ) -> ProcessedQuestionBatch:
        """Filter off-rubric items and produce globally unique question sets."""
        report = self._build_initial_report(questions_by_category)
        valid_candidates: list[QuestionCandidate] = []

        for category, questions in questions_by_category.items():
            for question in questions:
                invalid_reason = self._category_validation_error(question)
                if invalid_reason is None:
                    valid_candidates.append(question)
                    continue

                category_report = report["categories"][category]
                category_report["off_rubric"] += 1
                self._record_example(category_report["off_rubric_examples"], invalid_reason)

        report["global"]["off_rubric"] = sum(
            category_report["off_rubric"]
            for category_report in report["categories"].values()
        )
        self._log(
            f"Kept {len(valid_candidates)} questions after category validation "
            f"(dropped {report['global']['off_rubric']} off-rubric items)."
        )

        exact_components = self._group_exact_duplicates(valid_candidates)
        exact_winners, exact_removed = self._select_cluster_winners(exact_components)
        self._apply_duplicate_metrics(report, exact_removed, bucket="exact_duplicates")

        semantic_components = self.question_processor.find_similar_components(
            [question.canonical_question for question in exact_winners]
        )
        semantic_groups = [
            [exact_winners[index] for index in component]
            for component in semantic_components
        ]
        final_candidates, semantic_removed = self._select_cluster_winners(semantic_groups)
        self._apply_duplicate_metrics(
            report,
            semantic_removed,
            bucket="semantic_duplicates",
        )

        final_candidates.sort(
            key=lambda question: (
                CATEGORY_PRIORITY.get(question.source_category, 99),
                question.original_order,
            )
        )
        final_questions = self._serialize_questions(final_candidates)
        self._finalize_report(report, final_questions)
        self._log(
            "Finished filtering with "
            f"{report['global']['final']} unique questions remaining."
        )
        return ProcessedQuestionBatch(
            questions_by_category=final_questions,
            report=report,
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
        raise ValueError("No JSON array found in the model response.")

    @staticmethod
    def _record_example(examples: list[str], message: str, limit: int = 5) -> None:
        """Append a bounded number of human-readable examples."""
        if len(examples) < limit:
            examples.append(message)

    @staticmethod
    def _specificity_score(question: str) -> int:
        """Score questions by their count of non-trivial tokens."""
        tokens = re.findall(r"[a-z0-9]+", question)
        return sum(token not in STOPWORDS for token in tokens)

    def _category_validation_error(self, question: QuestionCandidate) -> str | None:
        """Return a validation error message for off-rubric questions."""
        if not question.canonical_question:
            return "Question collapsed to empty content after canonicalization."

        if question.source_category == "naturalness":
            if not self._is_valid_naturalness_question(question):
                return (
                    "Naturalness question does not reference tone, flow, pacing, "
                    "voice, or another allowed naturalness signal."
                )
        elif (
            question.dimension is not None
            and question.dimension not in FACTUALNESS_DIMENSIONS
        ):
            return f"Unsupported factualness dimension: {question.dimension}."

        return None

    def _is_valid_naturalness_question(self, question: QuestionCandidate) -> bool:
        """Check that a naturalness question targets style rather than entity recall."""
        if (
            question.dimension is not None
            and question.dimension not in NATURALNESS_DIMENSIONS
        ):
            return False

        tokens = re.findall(r"[a-z0-9]+", question.canonical_question)
        if not tokens:
            return False

        token_markers = {
            marker for marker in NATURALNESS_MARKERS if any(marker in token for token in tokens)
        }
        if question.dimension in NATURALNESS_DIMENSIONS:
            return True
        return bool(token_markers)

    @staticmethod
    def _group_exact_duplicates(
        questions: Sequence[QuestionCandidate],
    ) -> list[list[QuestionCandidate]]:
        """Cluster questions that share the same canonical form."""
        grouped_questions: dict[str, list[QuestionCandidate]] = defaultdict(list)
        for question in questions:
            grouped_questions[question.canonical_question].append(question)
        return list(grouped_questions.values())

    def _select_cluster_winners(
        self,
        clusters: Iterable[Sequence[QuestionCandidate]],
    ) -> tuple[list[QuestionCandidate], list[RemovedQuestion]]:
        """Choose one representative per cluster and return removed questions."""
        winners: list[QuestionCandidate] = []
        removed: list[RemovedQuestion] = []

        for cluster in clusters:
            sorted_cluster = sorted(cluster, key=self._representative_sort_key)
            winner = sorted_cluster[0]
            winners.append(winner)
            removed.extend(
                RemovedQuestion(question=question, winner=winner)
                for question in sorted_cluster[1:]
            )

        return winners, removed

    @staticmethod
    def _representative_sort_key(question: QuestionCandidate) -> tuple[int, int, int]:
        """Rank candidates so deterministic winners survive each cluster."""
        return (
            CATEGORY_PRIORITY.get(question.source_category, 99),
            -question.specificity_score,
            question.original_order,
        )

    def _build_initial_report(
        self,
        questions_by_category: dict[str, Sequence[QuestionCandidate]],
    ) -> dict[str, Any]:
        """Initialize a report structure with per-category counters."""
        categories: dict[str, dict[str, Any]] = {}
        total_parsed = 0
        for category, questions in questions_by_category.items():
            requested = QUESTION_REQUEST_MINIMUMS.get(category, 200)
            parsed = len(questions)
            total_parsed += parsed
            categories[category] = {
                "requested": requested,
                "parsed": parsed,
                "off_rubric": 0,
                "off_rubric_examples": [],
                "exact_duplicates": 0,
                "semantic_duplicates": 0,
                "cross_category_drops": 0,
                "final": 0,
                "shortfall": max(requested - parsed, 0),
            }

        return {
            "global": {
                "requested_per_category": dict(QUESTION_REQUEST_MINIMUMS),
                "parsed": total_parsed,
                "off_rubric": 0,
                "exact_duplicates": 0,
                "semantic_duplicates": 0,
                "cross_category_drops": 0,
                "final": 0,
            },
            "categories": categories,
        }

    def _apply_duplicate_metrics(
        self,
        report: dict[str, Any],
        removed_questions: Sequence[RemovedQuestion],
        bucket: str,
    ) -> None:
        """Update report counters for removed duplicate questions."""
        for removed_question in removed_questions:
            report["categories"][removed_question.question.source_category][bucket] += 1
            report["global"][bucket] += 1
            if (
                removed_question.question.source_category
                != removed_question.winner.source_category
            ):
                report["global"]["cross_category_drops"] += 1
                report["categories"][removed_question.question.source_category][
                    "cross_category_drops"
                ] += 1

    def _finalize_report(
        self,
        report: dict[str, Any],
        final_questions: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Compute final per-category counts and shortfalls."""
        total_final = 0
        total_off_rubric = 0
        for category, category_report in report["categories"].items():
            total_off_rubric += category_report["off_rubric"]
            final_count = len(final_questions.get(category, []))
            category_report["final"] = final_count
            category_report["shortfall"] = max(
                category_report["requested"] - final_count,
                0,
            )
            total_final += final_count

        report["global"]["off_rubric"] = total_off_rubric
        report["global"]["final"] = total_final

    @staticmethod
    def _serialize_questions(
        questions: Sequence[QuestionCandidate],
    ) -> dict[str, list[dict[str, Any]]]:
        """Convert internal candidates back to the public output schema."""
        questions_by_category: dict[str, list[QuestionCandidate]] = defaultdict(list)
        for question in questions:
            questions_by_category[question.source_category].append(question)

        serialized: dict[str, list[dict[str, Any]]] = {}
        for category in QUESTION_REQUEST_MINIMUMS:
            category_questions = sorted(
                questions_by_category.get(category, []),
                key=lambda question: question.original_order,
            )
            serialized[category] = [
                {
                    "question_number": index,
                    "question": question.question,
                }
                for index, question in enumerate(category_questions, start=1)
            ]

        return serialized
