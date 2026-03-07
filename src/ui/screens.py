import asyncio
import json
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Header,
    Footer,
    Button,
    ListView,
    ListItem,
    Label,
    Input,
    TextArea,
    Markdown as MarkdownWidget,
)
from textual.containers import Container, Horizontal, VerticalScroll
from textual.binding import Binding

from src.config.config_manager import ConfigManager
from src.services.data_processor import DataProcessor
from src.models.data_models import Transcript, Summary
from src.providers.vllm_provider import VLLMProvider
from src.providers.llm_provider import LLMProvider
from src.agents.agent_factory import AgentFactory
from src.services.question_processor import QuestionProcessor


class ProviderFactory:
    """Factory Pattern to create LLM providers."""

    @staticmethod
    def create_provider(provider_type: str = "vllm") -> LLMProvider:
        if provider_type.lower() == "vllm":
            return VLLMProvider()
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")


# Helper to flatten dict
def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class MainMenuScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="menu-container"):
            yield Button("Run", id="btn_run", variant="primary")
            yield Button("Config", id="btn_config")
            yield Button("Exit", id="btn_exit", variant="error")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_run":
            self.app.push_screen(RunScreen())
        elif event.button.id == "btn_config":
            self.app.push_screen(ConfigScreen())
        elif event.button.id == "btn_exit":
            self.app.exit()


class ConfigScreen(Screen):
    BINDINGS = [
        Binding("ctrl+s", "save_config", "Save"),
        Binding("left", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        self.config_mgr = ConfigManager()
        self.flat_config = flatten_dict(self.config_mgr.get_all())

        with Horizontal():
            # Left pane: keys
            with VerticalScroll(id="config-keys-pane"):
                self.list_view = ListView(
                    *[
                        ListItem(Label(k), id=f"item_{k.replace('.', '_')}", name=k)
                        for k in self.flat_config.keys()
                    ],
                    id="keys_list",
                )
                yield self.list_view
            # Right pane: values
            with VerticalScroll(id="config-values-pane"):
                self.editor_container = Container(id="editor_container")
                yield self.editor_container
        yield Footer()

    def on_mount(self):
        self.current_key = None

    async def on_list_view_selected(self, event: ListView.Selected):
        key = event.item.name
        self.current_key = key
        val = self.flat_config.get(key, "")
        for widget in self.editor_container.query("*"):
            await widget.remove()

        if (isinstance(val, str) and "\n" in val) or "prompt" in key:
            language = "markdown" if "prompt" in key else None
            widget = TextArea(val, id="val_editor", language=language)
        else:
            widget = Input(str(val), id="val_editor")

        await self.editor_container.mount(widget)

    def action_save_config(self):
        if self.current_key:
            editor = self.query_one("#val_editor")
            new_val = (
                editor.text if hasattr(editor, "text") else getattr(editor, "value", "")
            )
            self.flat_config[self.current_key] = new_val
            self.config_mgr.set(self.current_key, new_val)
            self.app.notify(f"Saved {self.current_key}")

    def action_go_back(self):
        self.app.pop_screen()


class FileViewScreen(Screen):
    """Screen to view file contents with syntax highlighting."""

    BINDINGS = [Binding("escape", "go_back", "Back")]

    def __init__(self, path: str, language: str):
        super().__init__()
        self.path = path
        self.language = language

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="file-viewer-container"):
            yield Label(f"Viewing: {self.path}", id="file-path-label")
            yield TextArea(
                id="file-content-area", read_only=True, language=self.language
            )
        yield Footer()

    def on_mount(self):
        try:
            with open(self.path, "r", encoding="utf-8-sig") as f:
                content = f.read()
            editor = self.query_one("#file-content-area", TextArea)
            editor.text = content
        except Exception as e:
            self.app.notify(f"Error reading file: {e}", variant="error")

    def action_go_back(self):
        self.app.pop_screen()


class RunScreen(Screen):
    BINDINGS = [
        Binding("escape", "go_back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="results-container"):
            self.results_view = MarkdownWidget(id="results-view")
            yield self.results_view
        yield Footer()

    async def on_mount(self):
        self.accumulated_output = ""
        self._last_rendered_text = ""
        self._ui_sync_task = asyncio.create_task(self._ui_sync_loop())
        asyncio.create_task(self.run_pipeline())

    async def on_unmount(self):
        if hasattr(self, "_ui_sync_task"):
            self._ui_sync_task.cancel()

    async def _ui_sync_loop(self):
        """Syncs the UI at a fixed rate (10Hz) to prevent lag during streaming."""
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

    def append_text(self, text: str):
        self.accumulated_output += text

    def on_markdown_link_clicked(self, event: MarkdownWidget.LinkClicked) -> None:
        event.prevent_default()
        path = str(event.href)
        # Determine language based on path or context
        if "transcript" in path.lower() or path.endswith(".json"):
            lang = "json"
        elif (
            "summary" in path.lower() or path.endswith(".xml") or path.endswith(".txt")
        ):
            lang = "xml"
        else:
            lang = None

        self.app.push_screen(FileViewScreen(path, lang))

    def action_go_back(self):
        self.app.pop_screen()

    async def _run_agent(
        self,
        agent_type: str,
        provider: LLMProvider,
        transcript_text: str,
        summary_text: str,
    ):
        title = agent_type.capitalize()
        self.append_text(f"\n\n## {title} Agent\n---\n")
        agent = AgentFactory.create_agent(agent_type, provider)
        response_content = ""
        is_json_block_open = False

        async for chunk in agent.generate_questions_stream(
            transcript_text, summary_text
        ):
            stripped_chunk = chunk.strip()
            if stripped_chunk.startswith("PROMPT_TOKENS:"):
                tokens = stripped_chunk.split(":")[1].strip()
                self.append_text(f"**Prompt Tokens:** `{tokens}`\n\n")
            elif stripped_chunk.startswith("TOTAL_STATS:"):
                if is_json_block_open:
                    self.append_text("\n```\n")
                    is_json_block_open = False
                stats = stripped_chunk.split(":")[1].strip()
                self.append_text(f"\n\n**Prompt + Answer / Max Context:** `{stats}`\n")
            else:
                if not is_json_block_open:
                    self.append_text("```json\n")
                    is_json_block_open = True
                response_content += chunk
                self.append_text(chunk)

        # Fallback close in case TOTAL_STATS was missing
        if is_json_block_open:
            self.append_text("\n```\n")

        return response_content

    async def run_pipeline(self):
        try:
            # Load from config paths
            config = ConfigManager()
            transcript_path = config.get("data.transcript_path", "data/transcript.json")
            summary_path = config.get("data.summary_path", "data/summary.txt")

            with open(transcript_path, "r", encoding="utf-8-sig") as f:
                transcript_data = json.load(f)

            with open(summary_path, "r", encoding="utf-8-sig") as f:
                summary_content = f.read()

            self.append_text(
                f"### 1. Loading data\n- **Transcript:** [{transcript_path}]({transcript_path})\n- **Summary:** [{summary_path}]({summary_path})\n\n"
            )

            transcript_obj = Transcript(**transcript_data)
            summary_obj = Summary(content=summary_content)

            self.append_text("### 2. Processing data...\n")
            proc_transcript = DataProcessor.process_transcript(transcript_obj)
            transcript_text = json.dumps(proc_transcript, indent=2)
            summary_text = summary_obj.content

            self.append_text("### 3. Initializing LLM Provider...\n")
            provider_type = config.get("app.provider", "vllm")
            provider = ProviderFactory.create_provider(provider_type)

            # Run Agents
            questions_by_category = {}
            for agent_type in ["factualness", "naturalness"]:
                response_text = await self._run_agent(
                    agent_type, provider, transcript_text, summary_text
                )

                try:
                    import re

                    json_match = re.search(r"(\[.*\])", response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response_text.strip()

                    questions = json.loads(json_str)
                    questions_by_category[agent_type] = questions
                    self.append_text(f"  - Extracted {len(questions)} questions.\n")
                except Exception as e:
                    self.append_text(
                        f"  - [!WARNING] Error parsing questions for {agent_type}: {e}\n"
                    )
                    questions_by_category[agent_type] = []

            # Deduplication Phase
            self.append_text("\n### 4. Deduplicating and Filtering Questions...\n")
            self.append_text(
                "> [!NOTE]\n> Loading Qwen3-Embedding-4B model. This might take a moment...\n"
            )

            try:
                processor = QuestionProcessor(log_callback=self.append_text)
                processed_results = processor.process_and_limit(questions_by_category)

                # Save results
                output_path = "data/processed_questions.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_results, f, indent=2)

                self.append_text(
                    f"\n- **Final Questions:** [processed_questions.json]({output_path})\n"
                )
                for cat, qs in processed_results.items():
                    self.append_text(
                        f"  - {cat.capitalize()}: {len(qs)} questions remaining.\n"
                    )

            except Exception as e:
                self.append_text(
                    f"\n> [!ERROR]\n> **Error during deduplication:** {str(e)}"
                )

            self.append_text("\n\n---\n**✅ Run completed successfully.**")

        except Exception as e:
            self.append_text(f"\n\n> [!ERROR]\n> **Error during run:** {str(e)}")
