import re
from typing import Dict, List
from src.models.data_models import Transcript, Summary


class DataProcessor:
    """Service to process Transcript and Summary into required formats."""

    @staticmethod
    def process_transcript(transcript: Transcript) -> Dict[str, List[str]]:
        """
        Organizes the transcript segments into a dictionary mapping speakers to their texts.
        Returns format: {"SPEAKER_00": ["text1", "text2", ...]}
        """
        organized_transcript: Dict[str, List[str]] = {}
        for segment in transcript.segments:
            speaker = segment.speaker
            if speaker not in organized_transcript:
                organized_transcript[speaker] = []
            organized_transcript[speaker].append(segment.text.strip())
        return organized_transcript

    @staticmethod
    def extract_summary_speakers(summary: Summary) -> Dict[str, str]:
        """
        Parses the summary to extract text per speaker.
        Assuming format like: <SPEAKER_00>text</SPEAKER_00>
        Returns format: {"SPEAKER_00": "text", ...}
        """
        pattern = r"<(SPEAKER_\d+)>(.*?)</\1>"
        matches = re.finditer(pattern, summary.content, re.DOTALL)

        extracted: Dict[str, str] = {}
        for match in matches:
            speaker = match.group(1)
            text = match.group(2).strip()
            extracted[speaker] = text

        return extracted
