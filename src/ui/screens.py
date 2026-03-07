"""Define the Textual screens used by the qa-eval TUI."""

import asyncio
import json
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown as MarkdownWidget,
    TextArea,
)

from src.agents.agent_factory import (
    AgentFactory,
    FIRST_TOKEN_LATENCY_PREFIX,
    PROMPT_TOKENS_PREFIX,
    STREAM_STATUS_PREFIX,
    TOTAL_STATS_PREFIX,
)
from src.config.config_manager import ConfigManager
from src.models.data_models import Summary, Transcript
from src.prompts.templates import QUESTION_REQUEST_MINIMUMS
from src.providers.llm_provider import LLMProvider
from src.providers.vllm_provider import VLLMProvider
from src.services.data_processor import DataProcessor
from src.services.evaluation_pipeline import EvaluationPipeline
from src.services.question_pipeline import QuestionPipeline
from src.services.question_processor import QuestionProcessor


class ProviderFactory:
    """Create language-model providers from config values."""

    @staticmethod
    def create_provider(provider_type: str = "vllm") -> LLMProvider:
        """Return the configured provider implementation."""
        if provider_type.lower() == "vllm":
            return VLLMProvider()
        raise ValueError(f"Unknown provider type: {provider_type}")


def flatten_dict(
    data: dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> dict[str, Any]:
    """Flatten nested config values for the config editor."""
    items: list[tuple[str, Any]] = []
    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


class MainMenuScreen(Screen):
    """Render the main menu with navigation actions."""

    def compose(self) -> ComposeResult:
        """Build the main menu widgets."""
        yield Header()
        with Container(id="menu-container"):
            yield Button("Run", id="btn_run", variant="primary")
            yield Button("Config", id="btn_config")
            yield Button("Exit", id="btn_exit", variant="error")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button presses to the appropriate screen."""
        if event.button.id == "btn_run":
            self.app.push_screen(RunScreen())
        elif event.button.id == "btn_config":
            self.app.push_screen(ConfigScreen())
        elif event.button.id == "btn_exit":
            self.app.exit()


class ConfigScreen(Screen):
    """Edit config values from within the TUI."""

    BINDINGS = [
        Binding("ctrl+s", "save_config", "Save"),
        Binding("left", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        """Build the config editor layout."""
        yield Header()
        self.config_mgr = ConfigManager()
        self.flat_config = flatten_dict(self.config_mgr.get_all())

        with Horizontal():
            with VerticalScroll(id="config-keys-pane"):
                self.list_view = ListView(
                    *[
                        ListItem(Label(key), id=f"item_{key.replace('.', '_')}", name=key)
                        for key in self.flat_config
                    ],
                    id="keys_list",
                )
                yield self.list_view
            with VerticalScroll(id="config-values-pane"):
                self.editor_container = Container(id="editor_container")
                yield self.editor_container
        yield Footer()

    def on_mount(self) -> None:
        """Initialize screen-local state."""
        self.current_key = None

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Load the selected config value into the editor."""
        key = event.item.name
        self.current_key = key
        value = self.flat_config.get(key, "")
        for widget in self.editor_container.query("*"):
            await widget.remove()

        if (isinstance(value, str) and "\n" in value) or "prompt" in key:
            language = "markdown" if "prompt" in key else None
            widget = TextArea(value, id="val_editor", language=language)
        else:
            widget = Input(str(value), id="val_editor")

        await self.editor_container.mount(widget)

    def action_save_config(self) -> None:
        """Persist the currently edited config value."""
        if self.current_key:
            editor = self.query_one("#val_editor")
            new_value = (
                editor.text if hasattr(editor, "text") else getattr(editor, "value", "")
            )
            self.flat_config[self.current_key] = new_value
            self.config_mgr.set(self.current_key, new_value)
            self.app.notify(f"Saved {self.current_key}")

    def action_go_back(self) -> None:
        """Return to the previous screen."""
        self.app.pop_screen()


class FileViewScreen(Screen):
    """Display file contents with syntax highlighting."""

    BINDINGS = [Binding("escape", "go_back", "Back")]

    def __init__(self, path: str, language: str | None) -> None:
        """Store the file path and syntax language for the viewer."""
        super().__init__()
        self.path = path
        self.language = language

    def compose(self) -> ComposeResult:
        """Build the read-only file viewer."""
        yield Header()
        with Container(id="file-viewer-container"):
            yield Label(f"Viewing: {self.path}", id="file-path-label")
            yield TextArea(
                id="file-content-area",
                read_only=True,
                language=self.language,
            )
        yield Footer()

    def on_mount(self) -> None:
        """Load the selected file into the viewer."""
        try:
            with open(self.path, "r", encoding="utf-8-sig") as file_handle:
                content = file_handle.read()
            editor = self.query_one("#file-content-area", TextArea)
            editor.text = content
        except Exception as exc:
            self.app.notify(f"Error reading file: {exc}", variant="error")

    def action_go_back(self) -> None:
        """Return to the previous screen."""
        self.app.pop_screen()


class RunScreen(Screen):
    """Run the question-generation pipeline and stream status updates."""

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        """Build the run-results screen."""
        yield Header()
        with VerticalScroll(id="results-container"):
            self.results_view = MarkdownWidget(id="results-view")
            yield self.results_view
        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the UI sync loop and start the pipeline."""
        self.accumulated_output = ""
        self._last_rendered_text = ""
        self._ui_sync_task = asyncio.create_task(self._ui_sync_loop())
        asyncio.create_task(self.run_pipeline())

    async def on_unmount(self) -> None:
        """Cancel background tasks when the screen unmounts."""
        if hasattr(self, "_ui_sync_task"):
            self._ui_sync_task.cancel()

    async def _ui_sync_loop(self) -> None:
        """Sync the markdown output to the UI at a fixed rate."""
        while True:
            try:
                if self.accumulated_output != self._last_rendered_text:
                    self.results_view.update(self.accumulated_output)
                    self._last_rendered_text = self.accumulated_output
                    self.query_one("#results-container").scroll_end(animate=False)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1)

    def append_text(self, text: str) -> None:
        """Append text to the streamed markdown output."""
        self.accumulated_output += text

    @staticmethod
    def _failed_parse_report(category: str, error: Exception) -> dict[str, Any]:
        """Create a fallback parse report when a category response cannot be parsed."""
        requested = QUESTION_REQUEST_MINIMUMS.get(category, 200)
        return {
            "requested": requested,
            "raw_items": 0,
            "validated": 0,
            "invalid": 1,
            "invalid_examples": [str(error)],
        }

    @staticmethod
    def _merge_parse_reports(
        report: dict[str, Any],
        parse_reports: dict[str, dict[str, Any]],
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

    def on_markdown_link_clicked(self, event: MarkdownWidget.LinkClicked) -> None:
        """Open linked artifacts in the file viewer screen."""
        event.prevent_default()
        path = str(event.href)
        if "transcript" in path.lower() or path.endswith(".json"):
            language = "json"
        elif (
            "summary" in path.lower() or path.endswith(".xml") or path.endswith(".txt")
        ):
            language = "xml"
        else:
            language = None

        self.app.push_screen(FileViewScreen(path, language))

    def action_go_back(self) -> None:
        """Return to the previous screen."""
        self.app.pop_screen()

    async def _run_agent(
        self,
        agent_type: str,
        provider: LLMProvider,
        transcript_text: str,
    ) -> str:
        """Stream one agent response into the UI and return the raw text."""
        agent = AgentFactory.create_agent(agent_type, provider)
        response_stream = agent.generate_questions_stream(transcript_text)
        return await self._stream_response(
            title=f"{agent_type.capitalize()} Agent",
            response_stream=response_stream,
        )

    async def _run_evaluator(
        self,
        category: str,
        provider: LLMProvider,
        transcript_text: str,
        summary_text: str,
        questions: list[dict[str, Any]],
    ) -> str:
        """Stream one evaluator response into the UI and return the raw text."""
        evaluator = AgentFactory.create_evaluator(provider)
        response_stream = evaluator.generate_evaluation_stream(
            category=category,
            transcript_text=transcript_text,
            summary_text=summary_text,
            questions=questions,
        )
        return await self._stream_response(
            title=f"{category.capitalize()} Evaluator",
            response_stream=response_stream,
        )

    async def _stream_response(
        self,
        title: str,
        response_stream: Any,
    ) -> str:
        """Render a streamed JSON-style model response in the results pane."""
        self.append_text(f"\n\n## {title}\n---\n")
        response_content = ""
        is_json_block_open = False

        try:
            async for chunk in response_stream:
                stripped_chunk = chunk.strip()
                if stripped_chunk.startswith(PROMPT_TOKENS_PREFIX):
                    tokens = stripped_chunk.split(":")[1].strip()
                    self.append_text(f"**Prompt Tokens:** `{tokens}`\n\n")
                elif stripped_chunk.startswith(STREAM_STATUS_PREFIX):
                    status = stripped_chunk.removeprefix(STREAM_STATUS_PREFIX)
                    if status == "REQUEST_SUBMITTED":
                        self.append_text(
                            "- Request submitted to the model. Waiting for the first response chunk...\n\n"
                        )
                elif stripped_chunk.startswith(FIRST_TOKEN_LATENCY_PREFIX):
                    latency = stripped_chunk.removeprefix(FIRST_TOKEN_LATENCY_PREFIX)
                    self.append_text(
                        f"- First response chunk received in `{latency}s`.\n\n"
                    )
                elif stripped_chunk.startswith(TOTAL_STATS_PREFIX):
                    if is_json_block_open:
                        self.append_text("\n```\n")
                        is_json_block_open = False
                    stats = stripped_chunk.split(":")[1].strip()
                    self.append_text(
                        f"\n\n**Prompt + Answer / Max Context:** `{stats}`\n"
                    )
                else:
                    if not is_json_block_open:
                        self.append_text("```json\n")
                        is_json_block_open = True
                    response_content += chunk
                    self.append_text(chunk)
        finally:
            if is_json_block_open:
                self.append_text("\n```\n")

        return response_content

    async def run_pipeline(self) -> None:
        """Execute the full question-generation workflow."""
        provider: LLMProvider | None = None
        try:
            config = ConfigManager()
            transcript_path = config.get("data.transcript_path", "data/transcript.json")
            summary_path = config.get("data.summary_path", "data/summary.txt")

            with open(transcript_path, "r", encoding="utf-8-sig") as file_handle:
                transcript_data = json.load(file_handle)

            with open(summary_path, "r", encoding="utf-8-sig") as file_handle:
                summary_content = file_handle.read()

            self.append_text(
                f"### 1. Loading data\n- **Transcript:** [{transcript_path}]({transcript_path})\n- **Summary:** [{summary_path}]({summary_path})\n\n"
            )

            transcript_obj = Transcript(**transcript_data)
            summary_obj = Summary(content=summary_content)

            self.append_text("### 2. Processing data...\n")
            processed_transcript = DataProcessor.process_transcript(transcript_obj)
            transcript_text = json.dumps(processed_transcript, indent=2)
            summary_text = summary_obj.content

            self.append_text("### 3. Initializing LLM Provider...\n")
            provider_type = config.get("app.provider", "vllm")
            provider = ProviderFactory.create_provider(provider_type)
            self.append_text("- Validating endpoint and warming the connection...\n")
            await provider.prepare()
            if isinstance(provider, VLLMProvider) and provider.connection_status:
                status = provider.connection_status
                self.append_text(
                    "- Connected to "
                    f"`{status.base_url}` with model `{status.model}` in "
                    f"`{status.latency_seconds:.3f}s`.\n"
                )
            else:
                self.append_text("- Provider connection prepared successfully.\n")
            processor = QuestionProcessor(log_callback=self.append_text)
            pipeline = QuestionPipeline(
                question_processor=processor,
                log_callback=self.append_text,
            )

            questions_by_category = {}
            parse_reports = {}
            for agent_type in QUESTION_REQUEST_MINIMUMS:
                response_text = await self._run_agent(
                    agent_type,
                    provider,
                    transcript_text,
                )

                try:
                    parsed_batch = pipeline.parse_generation_response(
                        agent_type,
                        response_text,
                    )
                    questions_by_category[agent_type] = parsed_batch.questions
                    parse_reports[agent_type] = parsed_batch.report
                    self.append_text(
                        "  - "
                        f"Validated {parsed_batch.report['validated']} questions "
                        f"(invalid: {parsed_batch.report['invalid']}).\n"
                    )
                except Exception as exc:
                    self.append_text(
                        f"  - [!WARNING] Error parsing questions for {agent_type}: {exc}\n"
                    )
                    questions_by_category[agent_type] = []
                    parse_reports[agent_type] = self._failed_parse_report(
                        agent_type,
                        exc,
                    )

            self.append_text("\n### 4. Deduplicating and Filtering Questions...\n")
            self.append_text(
                "> [!NOTE]\n> Loading Qwen3-Embedding-4B model. This might take a moment...\n"
            )

            try:
                processed_batch = pipeline.process(questions_by_category)
                processed_results = processed_batch.questions_by_category
                report = self._merge_parse_reports(
                    processed_batch.report,
                    parse_reports,
                )

                output_path = "data/processed_questions.json"
                with open(output_path, "w", encoding="utf-8") as file_handle:
                    json.dump(processed_results, file_handle, indent=2)

                report_path = "data/question_filter_report.json"
                with open(report_path, "w", encoding="utf-8") as file_handle:
                    json.dump(report, file_handle, indent=2)

                self.append_text(
                    f"\n- **Final Questions:** [processed_questions.json]({output_path})\n"
                )
                self.append_text(
                    f"- **Filter Report:** [question_filter_report.json]({report_path})\n"
                )
                for category, questions in processed_results.items():
                    category_report = report["categories"][category]
                    self.append_text(
                        "  - "
                        f"{category.capitalize()}: {len(questions)} questions remaining "
                        f"(invalid: {category_report['invalid']}, "
                        f"off-rubric: {category_report['off_rubric']}, "
                        f"exact duplicates: {category_report['exact_duplicates']}, "
                        f"semantic duplicates: {category_report['semantic_duplicates']}, "
                        f"shortfall: {category_report['shortfall']}).\n"
                    )

            except Exception as exc:
                self.append_text(
                    f"\n> [!ERROR]\n> **Error during deduplication:** {str(exc)}"
                )
                return

            self.append_text("\n### 5. Evaluating Questions...\n")
            evaluation_responses: dict[str, str] = {}
            for category, questions in processed_results.items():
                if not questions:
                    self.append_text(
                        "  - "
                        f"Skipping {category} evaluation because no questions survived.\n"
                    )
                    continue
                evaluation_responses[category] = await self._run_evaluator(
                    category=category,
                    provider=provider,
                    transcript_text=transcript_text,
                    summary_text=summary_text,
                    questions=questions,
                )

            evaluation_pipeline = EvaluationPipeline(log_callback=self.append_text)
            evaluation_batch = evaluation_pipeline.process(
                questions_by_category=processed_results,
                responses_by_category=evaluation_responses,
            )

            evaluations_path = "data/question_evaluations.json"
            with open(evaluations_path, "w", encoding="utf-8") as file_handle:
                json.dump(evaluation_batch.answers_by_category, file_handle, indent=2)

            evaluation_report_path = "data/evaluation_report.json"
            with open(evaluation_report_path, "w", encoding="utf-8") as file_handle:
                json.dump(evaluation_batch.report, file_handle, indent=2)

            self.append_text(
                f"\n- **Question Evaluations:** [question_evaluations.json]({evaluations_path})\n"
            )
            self.append_text(
                f"- **Evaluation Report:** [evaluation_report.json]({evaluation_report_path})\n"
            )
            for category, category_report in evaluation_batch.report["categories"].items():
                score = category_report["score"]
                formatted_score = "null" if score is None else f"{score:.3f}"
                self.append_text(
                    "  - "
                    f"{category.capitalize()}: score {formatted_score} "
                    f"({category_report['yes_count']} yes / "
                    f"{category_report['total_questions']} total, "
                    f"invalid or missing: {category_report['invalid_or_missing_count']}).\n"
                )

            global_score = evaluation_batch.report["global"]["score"]
            formatted_global_score = (
                "null" if global_score is None else f"{global_score:.3f}"
            )
            self.append_text(
                "  - "
                f"Global: score {formatted_global_score} "
                f"({evaluation_batch.report['global']['yes_count']} yes / "
                f"{evaluation_batch.report['global']['total_questions']} total, "
                f"invalid or missing: "
                f"{evaluation_batch.report['global']['invalid_or_missing_count']}).\n"
            )

            self.append_text("\n\n---\n**Run completed successfully.**")

        except Exception as exc:
            self.append_text(f"\n\n> [!ERROR]\n> **Error during run:** {str(exc)}")
        finally:
            if provider is not None:
                try:
                    await provider.close()
                except Exception as exc:
                    self.append_text(
                        "\n> [!WARNING]\n"
                        f"> **Error while closing provider:** {exc}"
                    )
