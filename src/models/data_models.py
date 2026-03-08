"""Define transcript and summary validation models."""

import re

from pydantic import BaseModel, field_validator

SPEAKER_OPEN_TAG_PATTERN = re.compile(r"<SPEAKER_\d+(?:\s+[^>]*)?>")
SPEAKER_BLOCK_PATTERN = re.compile(
    r"<(SPEAKER_\d+)(?:\s+[^>]*)?>(.*?)</\1>",
    re.DOTALL,
)


class TranscriptSegment(BaseModel):
    """Represent one speaker turn from the source transcript."""

    speaker: str
    start_time: int | float
    end_time: int | float
    text: str


class Transcript(BaseModel):
    """Represent a transcript payload loaded from JSON."""

    segments: list[TranscriptSegment]


class Summary(BaseModel):
    """Represent a tagged summary input."""

    content: str

    @field_validator("content")
    @classmethod
    def validate_summary_format(cls, value: str) -> str:
        """Validate that the summary contains at least one speaker opening tag."""
        if not SPEAKER_OPEN_TAG_PATTERN.search(value):
            raise ValueError(
                "Summary must contain valid <SPEAKER_XX> tags, with optional "
                "opening-tag attributes."
            )
        return value
