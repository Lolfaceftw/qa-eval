"""Regression tests for summary speaker-tag handling."""

import pytest
from pydantic import ValidationError

from src.models.data_models import Summary
from src.services.data_processor import DataProcessor


def test_summary_accepts_speaker_tags_with_attributes() -> None:
    """Allow optional XML-style attributes on speaker opening tags."""
    summary = Summary(
        content=(
            '<SPEAKER_00 emo_preset="upbeat">'
            "Example summary text."
            "</SPEAKER_00>"
        )
    )

    assert summary.content.startswith("<SPEAKER_00 ")


def test_summary_rejects_content_without_speaker_tags() -> None:
    """Reject summaries that omit the required speaker tags."""
    with pytest.raises(ValidationError, match="Summary must contain valid"):
        Summary(content="Example summary text without speaker tags.")


def test_extract_summary_speakers_supports_attributes_and_repeated_blocks() -> None:
    """Preserve attributed speaker blocks and concatenate repeated turns."""
    summary = Summary(
        content="""
<SPEAKER_00 emo_preset="upbeat">First turn.</SPEAKER_00>
<SPEAKER_01 emo_preset="engaged">Second speaker.</SPEAKER_01>
<SPEAKER_00 emo_preset="neutral">Third turn.</SPEAKER_00>
""".strip()
    )

    extracted = DataProcessor.extract_summary_speakers(summary)

    assert extracted == {
        "SPEAKER_00": "First turn.\n\nThird turn.",
        "SPEAKER_01": "Second speaker.",
    }
