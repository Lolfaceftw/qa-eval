"""Coordinate transcript loading, benchmark generation, and evaluation runs."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agents.agent_factory import (
    AgentFactory,
    FIRST_VISIBLE_CHUNK_LATENCY_PREFIX,
    PROMPT_TOKENS_PREFIX,
    REASONING_CHUNK_PREFIX,
    STREAM_STATUS_PREFIX,
    TOTAL_STATS_PREFIX,
)
from src.config.config_manager import ConfigManager
from src.config.runtime_settings import (
    QUESTION_CATEGORIES,
    get_question_request_minimum,
    get_question_request_minimums,
    is_embedding_enabled,
)
from src.models.data_models import Summary, Transcript
from src.providers.llm_provider import LLMProvider
from src.providers.vllm_provider import VLLMProvider
from src.services.data_processor import DataProcessor
from src.services.evaluation_pipeline import EvaluationPipeline, ProcessedEvaluationBatch
from src.services.question_pipeline import QuestionPipeline
from src.services.question_processor import QuestionProcessor


type QuestionResponseRunner = Callable[[str, LLMProvider, str], Awaitable[str]]
type EvaluationResponseRunner = Callable[
    [str, LLMProvider, str, str, list[dict[str, Any]]],
    Awaitable[str],
]


@dataclass(slots=True)
class LoadedTranscript:
    """Store the validated transcript input and prompt-ready text."""

    path: str
    text: str


@dataclass(slots=True)
class LoadedSummary:
    """Store the validated summary input and normalized text."""

    path: str
    text: str


@dataclass(slots=True)
class QuestionBenchmark:
    """Store the shared transcript-grounded question benchmark."""

    transcript_text: str
    questions_by_category: dict[str, list[dict[str, Any]]]
    report: dict[str, Any]


async def _collect_response_content(response_stream: AsyncIterable[str]) -> str:
    """Collect only answer-content chunks from a streamed agent response."""
    response_chunks: list[str] = []
    async for chunk in response_stream:
        stripped_chunk = chunk.strip()
        if stripped_chunk.startswith(PROMPT_TOKENS_PREFIX):
            continue
        if stripped_chunk.startswith(STREAM_STATUS_PREFIX):
            continue
        if stripped_chunk.startswith(FIRST_VISIBLE_CHUNK_LATENCY_PREFIX):
            continue
        if stripped_chunk.startswith(REASONING_CHUNK_PREFIX):
            continue
        if stripped_chunk.startswith(TOTAL_STATS_PREFIX):
            continue
        response_chunks.append(chunk)
    return "".join(response_chunks)


async def run_question_agent(
    agent_type: str,
    provider: LLMProvider,
    transcript_text: str,
) -> str:
    """Collect one question-generation response without UI rendering."""
    agent = AgentFactory.create_agent(agent_type, provider)
    return await _collect_response_content(
        agent.generate_questions_stream(transcript_text)
    )


async def run_evaluator_agent(
    category: str,
    provider: LLMProvider,
    transcript_text: str,
    summary_text: str,
    questions: list[dict[str, Any]],
) -> str:
    """Collect one evaluator response without UI rendering."""
    evaluator = AgentFactory.create_evaluator(provider)
    return await _collect_response_content(
        evaluator.generate_evaluation_stream(
            category=category,
            transcript_text=transcript_text,
            summary_text=summary_text,
            questions=questions,
        )
    )


class EvaluationWorkflow:
    """Coordinate the reusable evaluation workflow outside the UI layer."""

    def __init__(
        self,
        provider: LLMProvider,
        config: ConfigManager | None = None,
        log_callback: Callable[[str], None] | None = None,
        question_response_runner: QuestionResponseRunner = run_question_agent,
        evaluation_response_runner: EvaluationResponseRunner = run_evaluator_agent,
    ) -> None:
        """Store the provider, config, and pluggable response runners."""
        self.provider = provider
        self.config = config or ConfigManager()
        self.log_callback = log_callback
        self.question_response_runner = question_response_runner
        self.evaluation_response_runner = evaluation_response_runner

    def _emit(self, message: str) -> None:
        """Emit an optional raw log message."""
        if self.log_callback:
            self.log_callback(message)

    def load_transcript(self) -> LoadedTranscript:
        """Load, validate, and normalize the configured transcript file."""
        transcript_path = str(
            self.config.get("data.transcript_path", "data/transcript.json")
        )
        with open(transcript_path, "r", encoding="utf-8-sig") as file_handle:
            transcript_data = json.load(file_handle)

        transcript_obj = Transcript(**transcript_data)
        processed_transcript = DataProcessor.process_transcript(transcript_obj)
        return LoadedTranscript(
            path=transcript_path,
            text=json.dumps(processed_transcript, indent=2),
        )

    def load_summary(self, summary_path: str | Path | None = None) -> LoadedSummary:
        """Load, validate, and normalize one summary file."""
        resolved_path = summary_path
        if resolved_path is None:
            resolved_path = self.config.get("data.summary_path", "data/summary.txt")
        summary_path_str = str(resolved_path)
        with open(summary_path_str, "r", encoding="utf-8-sig") as file_handle:
            summary_content = file_handle.read()

        summary_obj = Summary(content=summary_content)
        return LoadedSummary(path=summary_path_str, text=summary_obj.content)

    async def prepare_provider(self) -> None:
        """Validate connectivity and log the warmed provider details."""
        self._emit("- Validating endpoint and warming the connection...\n")
        await self.provider.prepare()
        if isinstance(self.provider, VLLMProvider) and self.provider.connection_status:
            status = self.provider.connection_status
            self._emit(
                "- Connected to "
                f"`{status.base_url}` with model `{status.model}` in "
                f"`{status.latency_seconds:.3f}s`.\n"
            )
            return
        self._emit("- Provider connection prepared successfully.\n")

    async def build_question_benchmark(self, transcript_text: str) -> QuestionBenchmark:
        """Generate, parse, filter, and deduplicate the shared benchmark."""
        requested_per_category = get_question_request_minimums(self.config)
        semantic_dedup_enabled = is_embedding_enabled(self.config)
        processor = None
        if semantic_dedup_enabled:
            processor = QuestionProcessor(log_callback=self.log_callback)

        pipeline = QuestionPipeline(
            question_processor=processor,
            log_callback=self.log_callback,
            semantic_dedup_enabled=semantic_dedup_enabled,
            requested_per_category=requested_per_category,
        )

        questions_by_category: dict[str, object] = {}
        parse_reports: dict[str, dict[str, Any]] = {}
        for agent_type in QUESTION_CATEGORIES:
            response_text = await self.question_response_runner(
                agent_type,
                self.provider,
                transcript_text,
            )

            try:
                parsed_batch = pipeline.parse_generation_response(
                    agent_type,
                    response_text,
                )
                questions_by_category[agent_type] = parsed_batch.questions
                parse_reports[agent_type] = parsed_batch.report
                self._emit(
                    "  - "
                    f"Validated {parsed_batch.report['validated']} questions "
                    f"(invalid: {parsed_batch.report['invalid']}).\n"
                )
            except Exception as exc:
                self._emit(
                    f"  - [!WARNING] Error parsing questions for {agent_type}: {exc}\n"
                )
                questions_by_category[agent_type] = []
                parse_reports[agent_type] = self._failed_parse_report(
                    agent_type,
                    exc,
                )

        self._emit("\n### 4. Deduplicating and Filtering Questions...\n")
        if semantic_dedup_enabled:
            embedding_model = self.config.get(
                "embedding.model",
                "Qwen/Qwen3-Embedding-4B",
            )
            self._emit(
                "> [!NOTE]\n"
                f"> Loading `{embedding_model}` for semantic deduplication. "
                "This might take a moment...\n"
            )
        else:
            self._emit(
                "- Semantic deduplication is disabled by `embedding.enabled`; "
                "skipping embedding model loading.\n"
            )

        processed_batch = pipeline.process(questions_by_category)
        return QuestionBenchmark(
            transcript_text=transcript_text,
            questions_by_category=processed_batch.questions_by_category,
            report=self._merge_parse_reports(
                processed_batch.report,
                parse_reports,
            ),
        )

    async def evaluate_summary(
        self,
        transcript_text: str,
        summary_text: str,
        questions_by_category: Mapping[str, list[dict[str, Any]]],
    ) -> ProcessedEvaluationBatch:
        """Evaluate one summary against a prebuilt benchmark."""
        evaluation_responses: dict[str, str] = {}
        for category, questions in questions_by_category.items():
            if not questions:
                self._emit(
                    "  - "
                    f"Skipping {category} evaluation because no questions survived.\n"
                )
                continue
            evaluation_responses[category] = await self.evaluation_response_runner(
                category,
                self.provider,
                transcript_text,
                summary_text,
                questions,
            )

        evaluation_pipeline = EvaluationPipeline(log_callback=self.log_callback)
        return evaluation_pipeline.process(
            questions_by_category=questions_by_category,
            responses_by_category=evaluation_responses,
        )

    def _failed_parse_report(
        self,
        category: str,
        error: Exception,
    ) -> dict[str, Any]:
        """Create a fallback parse report when a category cannot be parsed."""
        return {
            "requested": get_question_request_minimum(category, self.config),
            "raw_items": 0,
            "validated": 0,
            "invalid": 1,
            "invalid_examples": [str(error)],
        }

    @staticmethod
    def _merge_parse_reports(
        report: dict[str, Any],
        parse_reports: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Merge parse-time metrics into the final filtering report."""
        report["global"]["raw_items"] = 0
        report["global"]["invalid"] = 0

        for category, parse_report in parse_reports.items():
            category_report = report["categories"].setdefault(
                category,
                {
                    "requested": parse_report.get("requested", 200),
                    "parsed": 0,
                    "off_rubric": 0,
                    "off_rubric_examples": [],
                    "exact_duplicates": 0,
                    "semantic_duplicates": 0,
                    "cross_category_drops": 0,
                    "final": 0,
                    "shortfall": parse_report.get("requested", 200),
                },
            )
            category_report["raw_items"] = parse_report["raw_items"]
            category_report["invalid"] = parse_report["invalid"]
            category_report["invalid_examples"] = parse_report["invalid_examples"]
            category_report["parsed"] = parse_report["validated"]
            report["global"]["raw_items"] += parse_report["raw_items"]
            report["global"]["invalid"] += parse_report["invalid"]

        return report
