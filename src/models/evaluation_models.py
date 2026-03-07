"""Define validation models for evaluator inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvaluableQuestionPayload(BaseModel):
    """Validate a processed question before sending it to the evaluator."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    question_number: int = Field(ge=1)
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject blank question text."""
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("Question text must not be empty.")
        return cleaned_value


class EvaluationAnswerPayload(BaseModel):
    """Validate one evaluator answer payload item."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    question_number: int = Field(ge=1)
    answer: str

    @field_validator("answer")
    @classmethod
    def normalize_answer(cls, value: str) -> str:
        """Normalize answers and require a strict yes/no contract."""
        normalized = value.strip().lower()
        if normalized not in {"yes", "no"}:
            raise ValueError("Answer must be exactly 'yes' or 'no'.")
        return normalized
