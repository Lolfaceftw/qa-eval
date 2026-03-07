from textual.app import App
from src.ui.screens import MainMenuScreen
from src.config.config_manager import ConfigManager
# TODO: in the model, double check by using rag, after formulating a question, query rag


class QAEvalApp(App):
    """A Textual app to manage QA Eval config and runs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pathlib import Path

        # Get path from config, defaulting to standard location
        raw_path = ConfigManager().get("ui.css_path", "src/ui/css/style.tcss")
        # Resolve path relative to the project root (where this file is located)
        root = Path(__file__).parent.parent
        resolved_path = (root / raw_path).resolve()
        # Set as list of absolute path strings to satisfy Textual
        self.css_path = [str(resolved_path)]

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())


if __name__ == "__main__":
    app = QAEvalApp()
    app.run()
