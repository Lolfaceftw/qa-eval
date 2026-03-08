"""Evaluate a folder of summaries and write an aggregate CSV."""

from __future__ import annotations

import csv
import json
from collections.abc import AsyncIterable, Callable, Sequence
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
from src.config.runtime_settings import QUESTION_CATEGORIES
from src.providers.llm_provider import LLMProvider
from src.providers.factory import ProviderFactory
from src.services.evaluation_workflow import (
    EvaluationWorkflow,
    LoadedTranscript,
    QuestionBenchmark,
)

SUMMARY_FILE_SUFFIX = "-summary.xml"
CSV_OUTPUT_NAME = "summary_evaluation_results.csv"
CSV_FIELDNAMES = [
    "summary_file",
    "summarizer",
    "critic",
    "status",
    "error_message",
    "global_total_questions",
    "global_yes_count",
    "global_no_count",
    "global_invalid_or_missing_count",
    "global_score",
    "factualness_total_questions",
    "factualness_yes_count",
    "factualness_no_count",
    "factualness_invalid_or_missing_count",
    "factualness_score",
    "naturalness_total_questions",
    "naturalness_yes_count",
    "naturalness_no_count",
    "naturalness_invalid_or_missing_count",
    "naturalness_score",
]
REPORT_METRICS = (
    "total_questions",
    "yes_count",
    "no_count",
    "invalid_or_missing_count",
    "score",
)


@dataclass(frozen=True, slots=True)
class SummaryFilenameParts:
    """Store the parsed summarizer and critic from one filename."""

    summarizer: str
    critic: str


@dataclass(slots=True)
class BatchSummaryResult:
    """Store the batch CSV outcome and row data."""

    csv_path: Path
    rows: list[dict[str, Any]]

    @property
    def row_count(self) -> int:
        """Return the number of CSV rows."""
        return len(self.rows)

    @property
    def error_count(self) -> int:
        """Return the number of error rows."""
        return sum(row["status"] == "error" for row in self.rows)

    @property
    def had_errors(self) -> bool:
        """Return whether any row recorded an error."""
        return self.error_count > 0


def infer_summary_file_parts(
    summary_files: Sequence[Path],
) -> tuple[dict[str, SummaryFilenameParts], dict[str, str]]:
    """Infer summarizer and critic names from a folder of summary filenames."""
    known_model_ids = _seed_known_model_ids(summary_files)
    parsed_parts: dict[str, SummaryFilenameParts] = {}
    errors: dict[str, str] = {}

    for summary_file in sorted(summary_files, key=lambda path: path.name):
        stem = _summary_stem(summary_file.name)
        matches = [
            candidate
            for candidate in known_model_ids
            if stem.startswith(f"{candidate}_") and stem != candidate
        ]
        if not matches:
            errors[summary_file.name] = (
                "Could not infer `<summarizer>_<critic>` from the filename. "
                "Add at least one self-pair filename such as "
                "`model_model-summary.xml` to seed the folder model IDs."
            )
            continue

        longest_length = max(len(candidate) for candidate in matches)
        longest_matches = sorted(
            candidate for candidate in matches if len(candidate) == longest_length
        )
        if len(longest_matches) != 1:
            errors[summary_file.name] = (
                "Filename split is ambiguous after applying the known model IDs."
            )
            continue

        summarizer = longest_matches[0]
        critic = stem.removeprefix(f"{summarizer}_")
        if not critic:
            errors[summary_file.name] = "Filename critic segment is empty."
            continue

        parsed_parts[summary_file.name] = SummaryFilenameParts(
            summarizer=summarizer,
            critic=critic,
        )
        known_model_ids.add(critic)

    return parsed_parts, errors


class SummaryBatchEvaluator:
    """Evaluate all summary files in a folder with one shared benchmark."""

    def __init__(
        self,
        config: ConfigManager | None = None,
        provider_factory: Callable[[str], object] = ProviderFactory.create_provider,
        workflow_factory: Callable[..., EvaluationWorkflow] = EvaluationWorkflow,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Store the batch dependencies and optional logger."""
        self.config = config or ConfigManager()
        self.provider_factory = provider_factory
        self.workflow_factory = workflow_factory
        self.log_callback = log_callback

    def _emit(self, message: str) -> None:
        """Emit an optional raw log message."""
        if self.log_callback:
            self.log_callback(message)

    async def evaluate_folder(self, summaries_folder: str | Path) -> BatchSummaryResult:
        """Evaluate the folder summaries and write the aggregate CSV."""
        folder_path = Path(summaries_folder)
        summary_files = self._list_summary_files(folder_path)
        self._emit(
            "### Batch Evaluation\n"
            f"- **Summaries Folder:** `{folder_path}`\n"
            f"- **Matched Summary Files:** `{len(summary_files)}`\n\n"
        )
        parsed_parts, filename_errors = infer_summary_file_parts(summary_files)
        rows_by_name: dict[str, dict[str, Any]] = {
            name: self._error_row(
                summary_file=name,
                parts=None,
                error_message=error_message,
            )
            for name, error_message in filename_errors.items()
        }
        for summary_name, error_message in filename_errors.items():
            self._emit(
                f"- [!WARNING] Skipping `{summary_name}` before evaluation: "
                f"{error_message}\n"
            )
        valid_files = [
            summary_file
            for summary_file in summary_files
            if summary_file.name in parsed_parts
        ]

        if valid_files:
            provider_type = str(self.config.get("app.provider", "vllm"))
            provider = self.provider_factory(provider_type)
            workflow = self.workflow_factory(
                provider=provider,
                config=self.config,
                log_callback=self.log_callback,
                question_response_runner=self._run_question_agent,
                evaluation_response_runner=self._run_evaluator_agent,
            )
            try:
                transcript = workflow.load_transcript()
                self._emit(
                    "### 1. Building Shared Benchmark\n"
                    f"- **Transcript:** [{transcript.path}]({transcript.path})\n"
                )
                await workflow.prepare_provider()
                benchmark = await workflow.build_question_benchmark(transcript.text)
                for index, summary_file in enumerate(valid_files, start=1):
                    parts = parsed_parts[summary_file.name]
                    self._emit(
                        "\n"
                        f"### 2. Evaluating Summary {index}/{len(valid_files)}\n"
                        f"- **Summary:** [{summary_file}]({summary_file})\n"
                        f"- **Summarizer:** `{parts.summarizer}`\n"
                        f"- **Critic:** `{parts.critic}`\n"
                    )
                    rows_by_name[summary_file.name] = await self._evaluate_summary_file(
                        workflow=workflow,
                        transcript=transcript,
                        benchmark=benchmark,
                        summary_file=summary_file,
                        parts=parts,
                    )
            finally:
                await provider.close()

        ordered_rows = [rows_by_name[summary_file.name] for summary_file in summary_files]
        csv_path = folder_path / CSV_OUTPUT_NAME
        self._write_csv(csv_path, ordered_rows)
        return BatchSummaryResult(csv_path=csv_path, rows=ordered_rows)

    async def _run_question_agent(
        self,
        agent_type: str,
        provider: LLMProvider,
        transcript_text: str,
    ) -> str:
        """Stream one question-generation response to the console."""
        agent = AgentFactory.create_agent(agent_type, provider)
        return await self._stream_response(
            title=f"{agent_type.capitalize()} Agent",
            response_stream=agent.generate_questions_stream(transcript_text),
        )

    async def _run_evaluator_agent(
        self,
        category: str,
        provider: LLMProvider,
        transcript_text: str,
        summary_text: str,
        questions: list[dict[str, Any]],
    ) -> str:
        """Stream one evaluator response to the console."""
        evaluator = AgentFactory.create_evaluator(provider)
        return await self._stream_response(
            title=f"{category.capitalize()} Evaluator",
            response_stream=evaluator.generate_evaluation_stream(
                category=category,
                transcript_text=transcript_text,
                summary_text=summary_text,
                questions=questions,
            ),
        )

    async def _stream_response(
        self,
        title: str,
        response_stream: AsyncIterable[str],
    ) -> str:
        """Emit streamed agent output to the console and collect the answer body."""
        self._emit(f"\n## {title}\n")
        response_content = ""
        is_reasoning_open = False
        is_answer_open = False

        async for chunk in response_stream:
            stripped_chunk = chunk.strip()
            if stripped_chunk.startswith(PROMPT_TOKENS_PREFIX):
                tokens = stripped_chunk.removeprefix(PROMPT_TOKENS_PREFIX)
                self._emit(f"Prompt Tokens: {tokens}\n")
                continue
            if stripped_chunk.startswith(STREAM_STATUS_PREFIX):
                status = stripped_chunk.removeprefix(STREAM_STATUS_PREFIX)
                if status == "REQUEST_SUBMITTED":
                    self._emit(
                        "Request submitted. Waiting for the first visible response chunk...\n"
                    )
                continue
            if stripped_chunk.startswith(FIRST_VISIBLE_CHUNK_LATENCY_PREFIX):
                latency = stripped_chunk.removeprefix(
                    FIRST_VISIBLE_CHUNK_LATENCY_PREFIX
                )
                self._emit(f"First visible response chunk received in {latency}s.\n")
                continue
            if stripped_chunk.startswith(REASONING_CHUNK_PREFIX):
                if not is_reasoning_open:
                    self._emit("[Reasoning]\n")
                    is_reasoning_open = True
                reasoning_payload = stripped_chunk.removeprefix(REASONING_CHUNK_PREFIX)
                self._emit(json.loads(reasoning_payload))
                continue
            if stripped_chunk.startswith(TOTAL_STATS_PREFIX):
                total_stats = stripped_chunk.removeprefix(TOTAL_STATS_PREFIX)
                self._emit(f"\nPrompt + Streamed Output / Max Context: {total_stats}\n")
                continue

            if is_reasoning_open and not is_answer_open:
                self._emit("\n")
            if not is_answer_open:
                self._emit("[Answer]\n")
                is_answer_open = True
            response_content += chunk
            self._emit(chunk)

        if response_content and not response_content.endswith("\n"):
            self._emit("\n")
        return response_content

    @staticmethod
    def _list_summary_files(folder_path: Path) -> list[Path]:
        """Return direct child summary files in lexicographic order."""
        if not folder_path.exists():
            raise ValueError(f"Summaries folder does not exist: {folder_path}")
        if not folder_path.is_dir():
            raise ValueError(f"Summaries path is not a directory: {folder_path}")

        summary_files = sorted(
            (
                child
                for child in folder_path.iterdir()
                if child.is_file() and child.name.endswith(SUMMARY_FILE_SUFFIX)
            ),
            key=lambda path: path.name,
        )
        if not summary_files:
            raise ValueError(
                f"No `*{SUMMARY_FILE_SUFFIX}` files were found in {folder_path}."
            )
        return summary_files

    async def _evaluate_summary_file(
        self,
        workflow: EvaluationWorkflow,
        transcript: LoadedTranscript,
        benchmark: QuestionBenchmark,
        summary_file: Path,
        parts: SummaryFilenameParts,
    ) -> dict[str, Any]:
        """Evaluate one summary file and return its CSV row."""
        try:
            loaded_summary = workflow.load_summary(summary_file)
            evaluation_batch = await workflow.evaluate_summary(
                transcript_text=transcript.text,
                summary_text=loaded_summary.text,
                questions_by_category=benchmark.questions_by_category,
            )
        except Exception as exc:
            return self._error_row(
                summary_file=summary_file.name,
                parts=parts,
                error_message=str(exc),
            )

        return self._success_row(
            summary_file=summary_file.name,
            parts=parts,
            report=evaluation_batch.report,
        )

    @staticmethod
    def _base_row(
        summary_file: str,
        parts: SummaryFilenameParts | None,
    ) -> dict[str, Any]:
        """Build an empty CSV row with metadata prefilled."""
        row = {fieldname: "" for fieldname in CSV_FIELDNAMES}
        row["summary_file"] = summary_file
        row["summarizer"] = "" if parts is None else parts.summarizer
        row["critic"] = "" if parts is None else parts.critic
        return row

    def _success_row(
        self,
        summary_file: str,
        parts: SummaryFilenameParts,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """Build one successful CSV row from an evaluation report."""
        row = self._base_row(summary_file, parts)
        row["status"] = "ok"
        row["error_message"] = ""
        self._populate_report_metrics(row, report)
        return row

    def _error_row(
        self,
        summary_file: str,
        parts: SummaryFilenameParts | None,
        error_message: str,
    ) -> dict[str, Any]:
        """Build one error CSV row."""
        row = self._base_row(summary_file, parts)
        row["status"] = "error"
        row["error_message"] = error_message
        return row

    @staticmethod
    def _populate_report_metrics(row: dict[str, Any], report: dict[str, Any]) -> None:
        """Flatten the evaluation report into CSV fields."""
        SummaryBatchEvaluator._copy_report_section(
            row=row,
            prefix="global",
            section=report.get("global", {}),
        )
        category_reports = report.get("categories", {})
        for category in QUESTION_CATEGORIES:
            SummaryBatchEvaluator._copy_report_section(
                row=row,
                prefix=category,
                section=category_reports.get(category, {}),
            )

    @staticmethod
    def _copy_report_section(
        row: dict[str, Any],
        prefix: str,
        section: dict[str, Any],
    ) -> None:
        """Copy one report section into the flat CSV row."""
        for metric in REPORT_METRICS:
            value = section.get(metric)
            row[f"{prefix}_{metric}"] = "" if value is None else value

    @staticmethod
    def _write_csv(csv_path: Path, rows: Sequence[dict[str, Any]]) -> None:
        """Write the aggregate batch CSV file."""
        with open(csv_path, "w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)


def _seed_known_model_ids(summary_files: Sequence[Path]) -> set[str]:
    """Seed model IDs from any self-pair filename in the folder."""
    known_model_ids: set[str] = set()
    for summary_file in summary_files:
        repeated_id = _repeated_model_id(_summary_stem(summary_file.name))
        if repeated_id is not None:
            known_model_ids.add(repeated_id)
    return known_model_ids


def _repeated_model_id(stem: str) -> str | None:
    """Return the repeated model ID from a `<id>_<id>` stem when present."""
    repeated_ids = {
        stem[:index]
        for index, character in enumerate(stem)
        if character == "_" and stem[:index] == stem[index + 1 :]
    }
    if len(repeated_ids) != 1:
        return None
    return repeated_ids.pop()


def _summary_stem(filename: str) -> str:
    """Strip the summary filename suffix and return the remaining stem."""
    if not filename.endswith(SUMMARY_FILE_SUFFIX):
        raise ValueError(f"Unsupported summary filename: {filename}")
    return filename.removesuffix(SUMMARY_FILE_SUFFIX)
