"""Regression tests for the post-generation question pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from src.services.question_pipeline import QuestionPipeline, canonicalize_question


class StubQuestionProcessor:
    """Return deterministic duplicate clusters without loading embeddings."""

    def __init__(self, duplicate_groups: Sequence[set[str]] | None = None) -> None:
        """Store canonical-text groups that should be treated as duplicates."""
        self.duplicate_groups = [set(group) for group in duplicate_groups or []]

    def find_similar_components(
        self,
        texts: Sequence[str],
        threshold: float | None = None,
    ) -> list[set[int]]:
        """Group texts whose canonical forms were declared equivalent."""
        del threshold
        parent = list(range(len(texts)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for duplicate_group in self.duplicate_groups:
            matching_indexes = [
                index for index, text in enumerate(texts) if text in duplicate_group
            ]
            for index in range(1, len(matching_indexes)):
                union(matching_indexes[0], matching_indexes[index])

        components: dict[int, set[int]] = {}
        for index in range(len(texts)):
            root = find(index)
            components.setdefault(root, set()).add(index)
        return list(components.values())


def make_pipeline(
    duplicate_groups: Sequence[set[str]] | None = None,
) -> QuestionPipeline:
    """Create a pipeline backed by the deterministic stub processor."""
    return QuestionPipeline(StubQuestionProcessor(duplicate_groups))


def test_canonicalize_question_removes_boilerplate() -> None:
    """Normalize helper verbs and summary wrappers away from core content."""
    canonical = canonicalize_question(
        "Does the summary reference the Money Scope podcast?"
    )

    assert canonical == "the money scope podcast"


def test_parse_generation_response_validates_first_json_array() -> None:
    """Parse only the first JSON array and reject malformed items."""
    pipeline = make_pipeline()
    response = """
    Here is the result.
    [
      {"question_number": 1, "dimension": "tone", "question": "Would a listener hear the cautious tone?"},
      {"question_number": 2, "dimension": "tone", "question": "   "},
      "not an object"
    ]
    trailing text
    """

    parsed_batch = pipeline.parse_generation_response("naturalness", response)

    assert len(parsed_batch.questions) == 1
    assert parsed_batch.report["raw_items"] == 3
    assert parsed_batch.report["invalid"] == 2


def test_process_rejects_off_rubric_naturalness_questions() -> None:
    """Drop naturalness questions that are only factual or entity recall."""
    pipeline = make_pipeline()
    naturalness_batch = pipeline.parse_generation_response(
        "naturalness",
        """
        [
          {"question_number": 1, "dimension": "tone", "question": "Would a listener still hear the speaker's cautious tone when the summary describes market forecasts?"},
          {"question_number": 2, "question": "Is the Meta Quest headset mentioned in the summary?"}
        ]
        """,
    )

    processed_batch = pipeline.process({"naturalness": naturalness_batch.questions})

    assert processed_batch.questions_by_category["naturalness"] == [
        {
            "question_number": 1,
            "question": "Would a listener still hear the speaker's cautious tone when the summary describes market forecasts?",
        }
    ]
    assert processed_batch.report["categories"]["naturalness"]["off_rubric"] == 1


def test_process_prefers_factualness_for_cross_category_duplicates() -> None:
    """Keep the factualness version when categories collide semantically."""
    factual_question = "Is the podcast name Rational Reminder mentioned in the summary?"
    naturalness_question = "Is Rational Reminder the podcast name included?"
    duplicate_groups = [
        {
            canonicalize_question(factual_question),
            canonicalize_question(naturalness_question),
        }
    ]
    pipeline = make_pipeline(duplicate_groups)

    factualness_batch = pipeline.parse_generation_response(
        "factualness",
        f"""
        [
          {{"question_number": 1, "dimension": "entity_relation", "question": "{factual_question}"}}
        ]
        """,
    )
    naturalness_batch = pipeline.parse_generation_response(
        "naturalness",
        f"""
        [
          {{"question_number": 1, "dimension": "tone", "question": "{naturalness_question}"}}
        ]
        """,
    )

    processed_batch = pipeline.process(
        {
            "factualness": factualness_batch.questions,
            "naturalness": naturalness_batch.questions,
        }
    )

    assert processed_batch.questions_by_category["factualness"] == [
        {
            "question_number": 1,
            "question": factual_question,
        }
    ]
    assert processed_batch.questions_by_category["naturalness"] == []
    assert (
        processed_batch.report["categories"]["naturalness"]["cross_category_drops"] == 1
    )
    assert processed_batch.report["categories"]["naturalness"]["semantic_duplicates"] == 1


def test_process_renumbers_questions_sequentially_after_filtering() -> None:
    """Renumber surviving questions so the public artifact stays contiguous."""
    duplicate_groups = [
        {
            canonicalize_question("Does the summary cover the Money Scope podcast?"),
            canonicalize_question("Is the Money Scope podcast included?"),
        }
    ]
    pipeline = make_pipeline(duplicate_groups)
    factualness_batch = pipeline.parse_generation_response(
        "factualness",
        """
        [
          {"question_number": 5, "dimension": "entity_relation", "question": "Does the summary cover the Money Scope podcast?"},
          {"question_number": 8, "dimension": "entity_relation", "question": "Is the Money Scope podcast included?"},
          {"question_number": 13, "dimension": "numerical_detail", "question": "Would a reviewer know the exact episode number from the summary?"}
        ]
        """,
    )

    processed_batch = pipeline.process({"factualness": factualness_batch.questions})

    assert processed_batch.questions_by_category["factualness"] == [
        {
            "question_number": 1,
            "question": "Does the summary cover the Money Scope podcast?",
        },
        {
            "question_number": 2,
            "question": "Would a reviewer know the exact episode number from the summary?",
        },
    ]
