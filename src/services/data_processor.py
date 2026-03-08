"""Process transcript and summary inputs into prompt-ready structures."""

from src.models.data_models import SPEAKER_BLOCK_PATTERN, Summary, Transcript


class DataProcessor:
    """Process transcript and summary inputs into required formats."""

    @staticmethod
    def process_transcript(transcript: Transcript) -> dict[str, list[str]]:
        """Group transcript segments into a speaker-to-utterances mapping."""
        organized_transcript: dict[str, list[str]] = {}
        for segment in transcript.segments:
            speaker = segment.speaker
            if speaker not in organized_transcript:
                organized_transcript[speaker] = []
            organized_transcript[speaker].append(segment.text.strip())
        return organized_transcript

    @staticmethod
    def extract_summary_speakers(summary: Summary) -> dict[str, str]:
        """Extract summary text per speaker, concatenating repeated speaker blocks."""
        extracted: dict[str, str] = {}
        for match in SPEAKER_BLOCK_PATTERN.finditer(summary.content):
            speaker = match.group(1)
            text = match.group(2).strip()
            existing_text = extracted.get(speaker)
            if existing_text and text:
                extracted[speaker] = f"{existing_text}\n\n{text}"
            else:
                extracted[speaker] = existing_text or text

        return extracted
