import re
from typing import List
from pydantic import BaseModel, field_validator, ValidationInfo


class TranscriptSegment(BaseModel):
    speaker: str
    start_time: int | float
    end_time: int | float
    text: str


class Transcript(BaseModel):
    segments: List[TranscriptSegment]


class Summary(BaseModel):
    content: str

    @field_validator("content")
    def validate_summary_format(cls, v: str, info: ValidationInfo) -> str:
        """Validates that the summary contains expected speaker tags like <SPEAKER_00>."""
        if not re.search(r"<SPEAKER_\d+>", v):
            raise ValueError("Summary must contain valid <SPEAKER_XX> tags.")
        return v
