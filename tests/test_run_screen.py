"""Regression tests for the run-screen provider lifecycle."""

from __future__ import annotations

import builtins
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.ui.screens as screens
import src.services.evaluation_workflow as workflow_module
from src.agents.agent_factory import REASONING_CHUNK_PREFIX
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

    last_init_kwargs: dict[str, object] | None = None

    def __init__(self, *args, **kwargs) -> None:
        """Accept the production constructor signature."""
        del args
        type(self).last_init_kwargs = kwargs

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


class FakeEvaluationPipeline:
    """Return deterministic evaluator outputs for run-screen tests."""

    def __init__(self, *args, **kwargs) -> None:
        """Accept the production constructor signature."""
        del args, kwargs

    def process(
        self,
        questions_by_category: dict[str, list[dict[str, object]]],
        responses_by_category: dict[str, str],
    ) -> SimpleNamespace:
        """Return one yes answer per question with a deterministic report."""
        del responses_by_category
        answers_by_category = {
            category: [
                {
                    "question_number": question["question_number"],
                    "question": question["question"],
                    "answer": "yes",
                }
                for question in questions
            ]
            for category, questions in questions_by_category.items()
        }
        category_report = {
            category: {
                "total_questions": len(questions),
                "yes_count": len(questions),
                "no_count": 0,
                "invalid_or_missing_count": 0,
                "score": (1.0 if questions else None),
            }
            for category, questions in questions_by_category.items()
        }
        total_questions = sum(
            report["total_questions"] for report in category_report.values()
        )
        return SimpleNamespace(
            answers_by_category=answers_by_category,
            report={
                "global": {
                    "total_questions": total_questions,
                    "yes_count": total_questions,
                    "no_count": 0,
                    "invalid_or_missing_count": 0,
                    "score": (1.0 if total_questions else None),
                },
                "categories": category_report,
            },
        )


class FakeMarkdownView:
    """Capture incremental markdown appends from the run screen."""

    def __init__(self) -> None:
        """Initialize the collected append calls."""
        self.append_calls: list[str] = []

    async def append(self, markdown: str) -> None:
        """Record the appended markdown fragment."""
        self.append_calls.append(markdown)


class FakeScrollContainer:
    """Track scroll-to-end requests made after markdown flushes."""

    def __init__(self) -> None:
        """Initialize the scroll counter."""
        self.scroll_end_calls = 0

    def scroll_end(self, animate: bool = False) -> None:
        """Record the scroll request."""
        del animate
        self.scroll_end_calls += 1


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
    evaluations_output = tmp_path / "question_evaluations.json"
    evaluation_report_output = tmp_path / "evaluation_report.json"

    def fake_open(path: str | Path, *args, **kwargs):
        if path == "data/processed_questions.json":
            return builtins.open(processed_output, *args, **kwargs)
        if path == "data/question_filter_report.json":
            return builtins.open(report_output, *args, **kwargs)
        if path == "data/question_evaluations.json":
            return builtins.open(evaluations_output, *args, **kwargs)
        if path == "data/evaluation_report.json":
            return builtins.open(evaluation_report_output, *args, **kwargs)
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
    FakePipeline.last_init_kwargs = None

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
    monkeypatch.setattr(
        screens.ProviderFactory,
        "create_provider",
        lambda provider_type="vllm": provider,
    )
    monkeypatch.setattr(
        workflow_module.DataProcessor,
        "process_transcript",
        lambda transcript: {"segments": len(transcript.segments)},
    )
    monkeypatch.setattr(
        workflow_module,
        "VLLMProvider",
        FakePreparedProvider,
    )
    monkeypatch.setattr(
        workflow_module,
        "QuestionProcessor",
        lambda log_callback=None: object(),
    )
    monkeypatch.setattr(workflow_module, "QuestionPipeline", FakePipeline)
    monkeypatch.setattr(
        workflow_module,
        "EvaluationPipeline",
        FakeEvaluationPipeline,
    )

    async def fake_run_agent(
        self,
        agent_type: str,
        provider_arg: object,
        transcript_text: str,
    ) -> str:
        del self, provider_arg, transcript_text
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
    assert FakePipeline.last_init_kwargs is not None
    assert FakePipeline.last_init_kwargs["semantic_dedup_enabled"] is True
    assert FakePipeline.last_init_kwargs["requested_per_category"] == {
        "factualness": 200,
        "naturalness": 200,
    }
    assert "Connected to" in screen.accumulated_output
    assert "Evaluation Report" in screen.accumulated_output
    assert "Global: score 1.000" in screen.accumulated_output
    assert "Run completed successfully" in screen.accumulated_output


@pytest.mark.anyio
async def test_run_pipeline_skips_embedding_model_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Skip QuestionProcessor construction when semantic deduplication is disabled."""
    transcript_path, summary_path = write_input_files(tmp_path)
    provider = FakePreparedProvider()
    screen = screens.RunScreen()
    screen.accumulated_output = ""
    FakePipeline.last_init_kwargs = None

    monkeypatch.setattr(
        screens,
        "ConfigManager",
        lambda: FakeConfig(
            {
                "data.transcript_path": str(transcript_path),
                "data.summary_path": str(summary_path),
                "app.provider": "vllm",
                "embedding.enabled": "false",
                "questions.minimum": "125",
            }
        ),
    )
    monkeypatch.setattr(
        screens.ProviderFactory,
        "create_provider",
        lambda provider_type="vllm": provider,
    )
    monkeypatch.setattr(
        workflow_module.DataProcessor,
        "process_transcript",
        lambda transcript: {"segments": len(transcript.segments)},
    )
    monkeypatch.setattr(
        workflow_module,
        "VLLMProvider",
        FakePreparedProvider,
    )

    def fail_question_processor(*args, **kwargs) -> object:
        del args, kwargs
        raise AssertionError("QuestionProcessor should not be constructed")

    monkeypatch.setattr(workflow_module, "QuestionProcessor", fail_question_processor)
    monkeypatch.setattr(workflow_module, "QuestionPipeline", FakePipeline)
    monkeypatch.setattr(
        workflow_module,
        "EvaluationPipeline",
        FakeEvaluationPipeline,
    )

    async def fake_run_agent(
        self,
        agent_type: str,
        provider_arg: object,
        transcript_text: str,
    ) -> str:
        del self, provider_arg, transcript_text
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
    assert FakePipeline.last_init_kwargs is not None
    assert FakePipeline.last_init_kwargs["question_processor"] is None
    assert FakePipeline.last_init_kwargs["semantic_dedup_enabled"] is False
    assert FakePipeline.last_init_kwargs["requested_per_category"] == {
        "factualness": 125,
        "naturalness": 125,
    }
    assert "Semantic deduplication is disabled by `embedding.enabled`" in (
        screen.accumulated_output
    )


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
    monkeypatch.setattr(
        screens.ProviderFactory,
        "create_provider",
        lambda provider_type="vllm": provider,
    )
    monkeypatch.setattr(
        workflow_module.DataProcessor,
        "process_transcript",
        lambda transcript: {"segments": len(transcript.segments)},
    )
    monkeypatch.setattr(
        workflow_module,
        "VLLMProvider",
        FakeFailingProvider,
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


@pytest.mark.anyio
async def test_stream_response_surfaces_reasoning_and_answer_separately() -> None:
    """Render streamed reasoning separately from the answer JSON body."""

    async def fake_stream():
        yield "PROMPT_TOKENS:321\n"
        yield "STREAM_STATUS:REQUEST_SUBMITTED\n"
        yield "FIRST_VISIBLE_CHUNK_LATENCY:1.234\n"
        yield f'{REASONING_CHUNK_PREFIX}"Thinking about the answer..."\n'
        yield '[{"question_number": 1}]'
        yield "\nTOTAL_STATS:654/4096\n"

    screen = screens.RunScreen()
    screen.accumulated_output = ""

    response_text = await screen._stream_response(
        title="Factualness Agent",
        response_stream=fake_stream(),
    )

    assert response_text == '[{"question_number": 1}]'
    assert "**Prompt Tokens:** `321`" in screen.accumulated_output
    assert "Waiting for the first visible response chunk" in screen.accumulated_output
    assert "First visible response chunk received in `1.234s`" in (
        screen.accumulated_output
    )
    assert "**Reasoning**" in screen.accumulated_output
    assert "Thinking about the answer..." in screen.accumulated_output
    assert "**Answer**" in screen.accumulated_output
    assert "```json" in screen.accumulated_output
    assert (
        "**Prompt + Streamed Output / Max Context:** `654/4096`"
        in screen.accumulated_output
    )


@pytest.mark.anyio
async def test_flush_pending_markdown_batches_incremental_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Append only new markdown fragments instead of rerendering the full buffer."""
    screen = screens.RunScreen()
    screen.results_view = FakeMarkdownView()
    scroll_container = FakeScrollContainer()
    screen._pending_markdown_fragments = ["## Title\n", "- line one\n", "- line two\n"]
    screen._markdown_flush_event.set()

    monkeypatch.setattr(screen, "query_one", lambda selector: scroll_container)

    await screen._flush_pending_markdown()

    assert screen.results_view.append_calls == ["## Title\n- line one\n- line two\n"]
    assert screen._pending_markdown_fragments == []
    assert not screen._markdown_flush_event.is_set()
    assert scroll_container.scroll_end_calls == 1
