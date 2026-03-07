"""Define validation models for agent-generated question payloads."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GeneratedQuestionPayload(BaseModel):
    """Validate a question object returned by the language model."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    question_number: int | None = Field(default=None)
    question: str
    dimension: str | None = Field(default=None)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject blank question text."""
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("Question text must not be empty.")
        return cleaned_value

    @field_validator("dimension")
    @classmethod
    def normalize_dimension(cls, value: str | None) -> str | None:
        """Normalize optional dimension labels for downstream matching."""
        if value is None:
            return None
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return normalized or None
