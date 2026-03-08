"""Define the Textual screens used by the qa-eval TUI."""

import asyncio
import contextlib
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
    FIRST_VISIBLE_CHUNK_LATENCY_PREFIX,
    PROMPT_TOKENS_PREFIX,
    REASONING_CHUNK_PREFIX,
    STREAM_STATUS_PREFIX,
    TOTAL_STATS_PREFIX,
)
from src.config.config_manager import ConfigManager
from src.providers.llm_provider import LLMProvider
from src.providers.factory import ProviderFactory
from src.services.evaluation_workflow import EvaluationWorkflow


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

    MARKDOWN_FLUSH_INTERVAL_SECONDS = 0.05

    BINDINGS = [
        Binding("escape", "go_back", "Back"),
    ]

    def __init__(self) -> None:
        """Initialize the buffered markdown state for streaming output."""
        super().__init__()
        self.accumulated_output = ""
        self._pending_markdown_fragments: list[str] = []
        self._markdown_flush_event = asyncio.Event()

    def compose(self) -> ComposeResult:
        """Build the run-results screen."""
        yield Header()
        with VerticalScroll(id="results-container"):
            self.results_view = MarkdownWidget(id="results-view")
            yield self.results_view
        yield Footer()

    async def on_mount(self) -> None:
        """Start the buffered markdown flush loop and pipeline task."""
        self._markdown_flush_task = asyncio.create_task(self._markdown_flush_loop())
        asyncio.create_task(self.run_pipeline())

    async def on_unmount(self) -> None:
        """Cancel background tasks when the screen unmounts."""
        if hasattr(self, "_markdown_flush_task"):
            self._markdown_flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._markdown_flush_task

    async def _markdown_flush_loop(self) -> None:
        """Append buffered markdown fragments to the UI in small batches."""
        while True:
            try:
                await self._markdown_flush_event.wait()
                await asyncio.sleep(self.MARKDOWN_FLUSH_INTERVAL_SECONDS)
                await self._flush_pending_markdown()
            except asyncio.CancelledError:
                await self._flush_pending_markdown()
                break
            except Exception:
                await asyncio.sleep(1)

    async def _flush_pending_markdown(self) -> None:
        """Flush buffered markdown to the results widget incrementally."""
        if not hasattr(self, "results_view"):
            return

        while self._pending_markdown_fragments:
            pending_markdown = "".join(self._pending_markdown_fragments)
            self._pending_markdown_fragments.clear()
            self._markdown_flush_event.clear()
            await self.results_view.append(pending_markdown)
            self.query_one("#results-container").scroll_end(animate=False)

    def append_text(self, text: str) -> None:
        """Append text to the streamed markdown output."""
        self.accumulated_output += text
        if hasattr(self, "results_view"):
            self._pending_markdown_fragments.append(text)
            self._markdown_flush_event.set()

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
        is_reasoning_block_open = False
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
                            "- Request submitted to the model. Waiting for the first visible response chunk...\n\n"
                        )
                elif stripped_chunk.startswith(FIRST_VISIBLE_CHUNK_LATENCY_PREFIX):
                    latency = stripped_chunk.removeprefix(
                        FIRST_VISIBLE_CHUNK_LATENCY_PREFIX
                    )
                    self.append_text(
                        f"- First visible response chunk received in `{latency}s`.\n\n"
                    )
                elif stripped_chunk.startswith(REASONING_CHUNK_PREFIX):
                    if is_json_block_open:
                        self.append_text("\n```\n")
                        is_json_block_open = False
                    if not is_reasoning_block_open:
                        self.append_text("**Reasoning**\n\n```text\n")
                        is_reasoning_block_open = True
                    reasoning_payload = stripped_chunk.removeprefix(
                        REASONING_CHUNK_PREFIX
                    )
                    self.append_text(json.loads(reasoning_payload))
                elif stripped_chunk.startswith(TOTAL_STATS_PREFIX):
                    if is_reasoning_block_open:
                        self.append_text("\n```\n")
                        is_reasoning_block_open = False
                    if is_json_block_open:
                        self.append_text("\n```\n")
                        is_json_block_open = False
                    stats = stripped_chunk.split(":")[1].strip()
                    self.append_text(
                        f"\n\n**Prompt + Streamed Output / Max Context:** `{stats}`\n"
                    )
                else:
                    if is_reasoning_block_open:
                        self.append_text("\n```\n\n**Answer**\n\n")
                        is_reasoning_block_open = False
                    if not is_json_block_open:
                        self.append_text("```json\n")
                        is_json_block_open = True
                    response_content += chunk
                    self.append_text(chunk)
        finally:
            if is_reasoning_block_open:
                self.append_text("\n```\n")
            if is_json_block_open:
                self.append_text("\n```\n")

        return response_content

    async def run_pipeline(self) -> None:
        """Execute the full question-generation workflow."""
        provider: LLMProvider | None = None
        try:
            config = ConfigManager()
            provider_type = config.get("app.provider", "vllm")
            provider = ProviderFactory.create_provider(provider_type)
            workflow = EvaluationWorkflow(
                provider=provider,
                config=config,
                log_callback=self.append_text,
                question_response_runner=self._run_agent,
                evaluation_response_runner=self._run_evaluator,
            )

            loaded_transcript = workflow.load_transcript()
            loaded_summary = workflow.load_summary()
            self.append_text(
                "### 1. Loading data\n"
                f"- **Transcript:** [{loaded_transcript.path}]({loaded_transcript.path})\n"
                f"- **Summary:** [{loaded_summary.path}]({loaded_summary.path})\n\n"
            )

            self.append_text("### 2. Processing data...\n")
            self.append_text("### 3. Initializing LLM Provider...\n")
            await workflow.prepare_provider()
            benchmark = await workflow.build_question_benchmark(loaded_transcript.text)

            output_path = "data/processed_questions.json"
            with open(output_path, "w", encoding="utf-8") as file_handle:
                json.dump(benchmark.questions_by_category, file_handle, indent=2)

            report_path = "data/question_filter_report.json"
            with open(report_path, "w", encoding="utf-8") as file_handle:
                json.dump(benchmark.report, file_handle, indent=2)

            self.append_text(
                f"\n- **Final Questions:** [processed_questions.json]({output_path})\n"
            )
            self.append_text(
                f"- **Filter Report:** [question_filter_report.json]({report_path})\n"
            )
            for category, questions in benchmark.questions_by_category.items():
                category_report = benchmark.report["categories"][category]
                self.append_text(
                    "  - "
                    f"{category.capitalize()}: {len(questions)} questions remaining "
                    f"(invalid: {category_report['invalid']}, "
                    f"off-rubric: {category_report['off_rubric']}, "
                    f"exact duplicates: {category_report['exact_duplicates']}, "
                    f"semantic duplicates: {category_report['semantic_duplicates']}, "
                    f"shortfall: {category_report['shortfall']}).\n"
                )

            self.append_text("\n### 5. Evaluating Questions...\n")
            evaluation_batch = await workflow.evaluate_summary(
                transcript_text=loaded_transcript.text,
                summary_text=loaded_summary.text,
                questions_by_category=benchmark.questions_by_category,
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
