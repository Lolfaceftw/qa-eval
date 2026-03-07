"""Regression tests for the run-screen provider lifecycle."""

from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.ui.screens as screens
from src.providers.vllm_provider import VLLMPreparationError


class FakeConfig:
    """Return deterministic run-screen config values."""

    def __init__(self, values: dict[str, object]) -> None:
        """Store flattened config values."""
        self._values = values

    def get(self, key: str, default: object = None) -> object:
        """Return the configured value or the supplied default."""
        return self._values.get(key, default)


@dataclass(slots=True)
class FakeConnectionStatus:
    """Describe a prepared fake provider connection."""

    base_url: str
    model: str
    latency_seconds: float


class FakePreparedProvider:
    """Track prepare and close calls from the run screen."""

    def __init__(self) -> None:
        """Initialize the provider call counters."""
        self.prepare_calls = 0
        self.close_calls = 0
        self.connection_status = FakeConnectionStatus(
            base_url="http://localhost:8000/v1",
            model="Qwen/Qwen3.5-35B-A3B",
            latency_seconds=0.123,
        )

    async def prepare(self) -> None:
        """Track that preflight was requested."""
        self.prepare_calls += 1

    async def close(self) -> None:
        """Track provider cleanup."""
        self.close_calls += 1

    async def generate_stream(self, prompt: str):  # pragma: no cover - not used
        """Satisfy the provider protocol for tests that patch `_run_agent`."""
        del prompt
        if False:
            yield ""

    def count_tokens(self, text: str) -> int:
        """Return a deterministic token count."""
        return len(text.split())

    @property
    def max_context(self) -> int:
        """Expose a deterministic context limit."""
        return 4096


class FakeFailingProvider(FakePreparedProvider):
    """Fail during preflight while still allowing cleanup."""

    async def prepare(self) -> None:
        """Raise a preflight error after tracking the call."""
        self.prepare_calls += 1
        raise VLLMPreparationError("preflight failed")


class FakePipeline:
    """Return deterministic parsing and processing results."""

    def __init__(self, *args, **kwargs) -> None:
        """Accept the production constructor signature."""
        del args, kwargs

    def parse_generation_response(
        self,
        category: str,
        response_text: str,
    ) -> SimpleNamespace:
        """Return a one-question parsed batch for each category."""
        del response_text
        return SimpleNamespace(
            questions=[
                {
                    "question_number": 1,
                    "question": f"{category} question?",
                }
            ],
            report={
                "requested": 200,
                "raw_items": 1,
                "validated": 1,
                "invalid": 0,
                "invalid_examples": [],
            },
        )

    def process(
        self,
        questions_by_category: dict[str, list[dict[str, object]]],
    ) -> SimpleNamespace:
        """Return the parsed questions unchanged with an empty report shell."""
        return SimpleNamespace(
            questions_by_category={
                category: [
                    {
                        "question_number": 1,
                        "question": questions[0]["question"],
                    }
                ]
                for category, questions in questions_by_category.items()
            },
            report={"global": {}, "categories": {}},
        )


def write_input_files(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal valid transcript and summary inputs."""
    transcript_path = tmp_path / "transcript.json"
    summary_path = tmp_path / "summary.txt"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "text": "Example transcript text.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        "<SPEAKER_00>Example summary text.</SPEAKER_00>",
        encoding="utf-8",
    )
    return transcript_path, summary_path


def patch_file_io(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect generated artifact writes to the test temp directory."""
    processed_output = tmp_path / "processed_questions.json"
    report_output = tmp_path / "question_filter_report.json"

    def fake_open(path: str | Path, *args, **kwargs):
        if path == "data/processed_questions.json":
            return builtins.open(processed_output, *args, **kwargs)
        if path == "data/question_filter_report.json":
            return builtins.open(report_output, *args, **kwargs)
        return builtins.open(path, *args, **kwargs)

    monkeypatch.setattr(screens, "open", fake_open, raising=False)


@pytest.mark.anyio
async def test_run_pipeline_prepares_provider_once_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preflight once per run, then clean up the provider afterwards."""
    transcript_path, summary_path = write_input_files(tmp_path)
    provider = FakePreparedProvider()
    screen = screens.RunScreen()
    screen.accumulated_output = ""

    monkeypatch.setattr(
        screens,
        "ConfigManager",
        lambda: FakeConfig(
            {
                "data.transcript_path": str(transcript_path),
                "data.summary_path": str(summary_path),
                "app.provider": "vllm",
            }
        ),
    )
    monkeypatch.setattr(screens, "VLLMProvider", FakePreparedProvider)
    monkeypatch.setattr(
        screens.ProviderFactory,
        "create_provider",
        lambda provider_type="vllm": provider,
    )
    monkeypatch.setattr(
        screens.DataProcessor,
        "process_transcript",
        lambda transcript: {"segments": len(transcript.segments)},
    )
    monkeypatch.setattr(
        screens,
        "QuestionProcessor",
        lambda log_callback=None: object(),
    )
    monkeypatch.setattr(screens, "QuestionPipeline", FakePipeline)

    async def fake_run_agent(
        self,
        agent_type: str,
        provider_arg: object,
        transcript_text: str,
        summary_text: str,
    ) -> str:
        del self, provider_arg, transcript_text, summary_text
        return json.dumps(
            [
                {
                    "question_number": 1,
                    "dimension": "tone",
                    "question": f"{agent_type} question?",
                }
            ]
        )

    monkeypatch.setattr(screens.RunScreen, "_run_agent", fake_run_agent)
    patch_file_io(monkeypatch, tmp_path)

    await screen.run_pipeline()

    assert provider.prepare_calls == 1
    assert provider.close_calls == 1
    assert "Connected to" in screen.accumulated_output
    assert "Run completed successfully" in screen.accumulated_output


@pytest.mark.anyio
async def test_run_pipeline_aborts_before_agents_when_preflight_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stop the run early when preflight fails, but still close the provider."""
    transcript_path, summary_path = write_input_files(tmp_path)
    provider = FakeFailingProvider()
    screen = screens.RunScreen()
    screen.accumulated_output = ""
    run_agent_called = False

    monkeypatch.setattr(
        screens,
        "ConfigManager",
        lambda: FakeConfig(
            {
                "data.transcript_path": str(transcript_path),
                "data.summary_path": str(summary_path),
                "app.provider": "vllm",
            }
        ),
    )
    monkeypatch.setattr(screens, "VLLMProvider", FakeFailingProvider)
    monkeypatch.setattr(
        screens.ProviderFactory,
        "create_provider",
        lambda provider_type="vllm": provider,
    )
    monkeypatch.setattr(
        screens.DataProcessor,
        "process_transcript",
        lambda transcript: {"segments": len(transcript.segments)},
    )

    async def fake_run_agent(*args, **kwargs) -> str:
        nonlocal run_agent_called
        del args, kwargs
        run_agent_called = True
        return "[]"

    monkeypatch.setattr(screens.RunScreen, "_run_agent", fake_run_agent)
    patch_file_io(monkeypatch, tmp_path)

    await screen.run_pipeline()

    assert provider.prepare_calls == 1
    assert provider.close_calls == 1
    assert not run_agent_called
    assert "preflight failed" in screen.accumulated_output
    assert "Error during run" in screen.accumulated_output
