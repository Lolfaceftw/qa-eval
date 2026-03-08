"""Regression tests for batch summary evaluation mode."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.services.batch_summary_evaluator import (
    SummaryBatchEvaluator,
    SummaryFilenameParts,
    infer_summary_file_parts,
)
from src.services.evaluation_pipeline import ProcessedEvaluationBatch
from src.services.evaluation_workflow import (
    LoadedSummary,
    LoadedTranscript,
    QuestionBenchmark,
)


class FakeConfig:
    """Return deterministic config values for batch-evaluation tests."""

    def __init__(self, values: dict[str, object]) -> None:
        """Store flattened config values."""
        self._values = values

    def get(self, key: str, default: object = None) -> object:
        """Return the configured value or the provided default."""
        return self._values.get(key, default)


class FakeProvider:
    """Track provider cleanup in batch-evaluation tests."""

    def __init__(self) -> None:
        """Initialize the close counter."""
        self.close_calls = 0

    async def close(self) -> None:
        """Record provider cleanup."""
        self.close_calls += 1


class FakeWorkflow:
    """Return deterministic batch-evaluation results."""

    def __init__(self, provider: object, **_: object) -> None:
        """Store the provided provider and initialize call counters."""
        self.provider = provider
        self.prepare_calls = 0
        self.benchmark_calls = 0
        self.load_summary_calls: list[str] = []
        self.evaluate_calls: list[str] = []

    def load_transcript(self) -> LoadedTranscript:
        """Return a stable prompt-ready transcript."""
        return LoadedTranscript(path="data/transcript.json", text='{"segments": 1}')

    async def prepare_provider(self) -> None:
        """Track provider preparation."""
        self.prepare_calls += 1

    async def build_question_benchmark(self, transcript_text: str) -> QuestionBenchmark:
        """Return one shared benchmark for the whole batch."""
        self.benchmark_calls += 1
        return QuestionBenchmark(
            transcript_text=transcript_text,
            questions_by_category={
                "factualness": [{"question_number": 1, "question": "fact?"}],
                "naturalness": [{"question_number": 1, "question": "tone?"}],
            },
            report={},
        )

    def load_summary(self, summary_path: str | Path) -> LoadedSummary:
        """Return the filename as deterministic summary text."""
        summary_name = Path(summary_path).name
        self.load_summary_calls.append(summary_name)
        return LoadedSummary(path=str(summary_path), text=summary_name)

    async def evaluate_summary(
        self,
        transcript_text: str,
        summary_text: str,
        questions_by_category: dict[str, list[dict[str, object]]],
    ) -> ProcessedEvaluationBatch:
        """Return a deterministic score or fail for one summary."""
        del transcript_text, questions_by_category
        self.evaluate_calls.append(summary_text)
        if "deepseek_reasoner" in summary_text:
            raise RuntimeError("evaluation failed")

        return ProcessedEvaluationBatch(
            answers_by_category={},
            report={
                "global": {
                    "total_questions": 2,
                    "yes_count": 1,
                    "no_count": 1,
                    "invalid_or_missing_count": 0,
                    "score": 0.5,
                },
                "categories": {
                    "factualness": {
                        "total_questions": 1,
                        "yes_count": 1,
                        "no_count": 0,
                        "invalid_or_missing_count": 0,
                        "score": 1.0,
                    },
                    "naturalness": {
                        "total_questions": 1,
                        "yes_count": 0,
                        "no_count": 1,
                        "invalid_or_missing_count": 0,
                        "score": 0.0,
                    },
                },
            },
        )


def test_infer_summary_file_parts_supports_self_pairs_and_new_critics() -> None:
    """Infer folder-local model IDs from self-pairs before parsing the rest."""
    summary_files = [
        Path("qwen3_5_9b_qwen3_5_9b-summary.xml"),
        Path("qwen3_5_4b_qwen3_5_4b-summary.xml"),
        Path("qwen3_5_4b_deepseek_reasoner-summary.xml"),
    ]

    parsed_parts, errors = infer_summary_file_parts(summary_files)

    assert errors == {}
    assert parsed_parts["qwen3_5_9b_qwen3_5_9b-summary.xml"] == SummaryFilenameParts(
        summarizer="qwen3_5_9b",
        critic="qwen3_5_9b",
    )
    assert (
        parsed_parts["qwen3_5_4b_deepseek_reasoner-summary.xml"]
        == SummaryFilenameParts(
            summarizer="qwen3_5_4b",
            critic="deepseek_reasoner",
        )
    )


def test_infer_summary_file_parts_rejects_unknown_filenames() -> None:
    """Record an error instead of guessing when a filename cannot be split safely."""
    summary_files = [
        Path("qwen3_5_4b_qwen3_5_4b-summary.xml"),
        Path("mystery-summary.xml"),
    ]

    parsed_parts, errors = infer_summary_file_parts(summary_files)

    assert "mystery-summary.xml" in errors
    assert parsed_parts["qwen3_5_4b_qwen3_5_4b-summary.xml"] == SummaryFilenameParts(
        summarizer="qwen3_5_4b",
        critic="qwen3_5_4b",
    )


@pytest.mark.anyio
async def test_batch_evaluator_builds_benchmark_once_and_writes_csv(
    tmp_path: Path,
) -> None:
    """Reuse one benchmark across the batch and preserve per-file errors."""
    summaries_dir = tmp_path / "summaries"
    summaries_dir.mkdir()
    for filename in (
        "qwen3_5_4b_qwen3_5_4b-summary.xml",
        "qwen3_5_4b_deepseek_reasoner-summary.xml",
        "unknown-summary.xml",
    ):
        (summaries_dir / filename).write_text(
            "<SPEAKER_00>Example summary text.</SPEAKER_00>",
            encoding="utf-8",
        )

    provider = FakeProvider()
    workflow = FakeWorkflow(provider)
    evaluator = SummaryBatchEvaluator(
        config=FakeConfig({"app.provider": "vllm"}),
        provider_factory=lambda provider_type: provider,
        workflow_factory=lambda **kwargs: workflow,
    )

    result = await evaluator.evaluate_folder(summaries_dir)

    assert result.row_count == 3
    assert result.error_count == 2
    assert workflow.prepare_calls == 1
    assert workflow.benchmark_calls == 1
    assert workflow.load_summary_calls == [
        "qwen3_5_4b_deepseek_reasoner-summary.xml",
        "qwen3_5_4b_qwen3_5_4b-summary.xml",
    ]
    assert workflow.evaluate_calls == workflow.load_summary_calls
    assert provider.close_calls == 1

    with open(result.csv_path, "r", encoding="utf-8", newline="") as file_handle:
        rows = list(csv.DictReader(file_handle))

    assert [row["summary_file"] for row in rows] == [
        "qwen3_5_4b_deepseek_reasoner-summary.xml",
        "qwen3_5_4b_qwen3_5_4b-summary.xml",
        "unknown-summary.xml",
    ]
    assert rows[0]["status"] == "error"
    assert rows[0]["summarizer"] == "qwen3_5_4b"
    assert rows[0]["critic"] == "deepseek_reasoner"
    assert rows[0]["error_message"] == "evaluation failed"
    assert rows[1]["status"] == "ok"
    assert rows[1]["global_score"] == "0.5"
    assert rows[1]["factualness_score"] == "1.0"
    assert rows[1]["naturalness_score"] == "0.0"
    assert rows[2]["status"] == "error"
    assert rows[2]["summarizer"] == ""
    assert rows[2]["critic"] == ""


@pytest.mark.anyio
async def test_batch_stream_response_emits_tokens_to_log_callback() -> None:
    """Stream prompt metadata, reasoning, and answer chunks to the logger."""
    emitted: list[str] = []
    evaluator = SummaryBatchEvaluator(
        log_callback=emitted.append,
    )

    async def fake_stream():
        yield "PROMPT_TOKENS:321\n"
        yield "STREAM_STATUS:REQUEST_SUBMITTED\n"
        yield "FIRST_VISIBLE_CHUNK_LATENCY:1.234\n"
        yield 'REASONING_CHUNK:"Thinking about the answer..."\n'
        yield '[{"question_number": 1}]'
        yield "\nTOTAL_STATS:654/4096\n"

    response_text = await evaluator._stream_response(
        title="Factualness Agent",
        response_stream=fake_stream(),
    )

    assert response_text == '[{"question_number": 1}]'
    log_output = "".join(emitted)
    assert "Factualness Agent" in log_output
    assert "Prompt Tokens: 321" in log_output
    assert "Waiting for the first visible response chunk" in log_output
    assert "First visible response chunk received in 1.234s." in log_output
    assert "[Reasoning]" in log_output
    assert "Thinking about the answer..." in log_output
    assert "[Answer]" in log_output
    assert '[{"question_number": 1}]' in log_output
    assert "Prompt + Streamed Output / Max Context: 654/4096" in log_output
