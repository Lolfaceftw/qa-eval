"""Run the qa-eval TUI or batch summary evaluator."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from src.services.batch_summary_evaluator import SummaryBatchEvaluator

LOGGER = logging.getLogger(__name__)


def _write_console(message: str) -> None:
    """Write batch progress to stdout immediately."""
    sys.stdout.write(message)
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    """Build the qa-eval command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate transcript summaries with the qa-eval workflow."
    )
    parser.add_argument(
        "--summaries",
        type=Path,
        help=(
            "Evaluate all `*-summary.xml` files in the folder and write "
            "`summary_evaluation_results.csv` there."
        ),
    )
    return parser


def run_tui() -> int:
    """Launch the Textual TUI."""
    from textual.app import App

    from src.config.config_manager import ConfigManager
    from src.ui.screens import MainMenuScreen

    class QAEvalApp(App):
        """Run the Textual application shell for qa-eval."""

        def __init__(self, **kwargs: object) -> None:
            """Resolve the configured CSS path before launching Textual."""
            super().__init__(**kwargs)
            raw_path = ConfigManager().get("ui.css_path", "src/ui/css/style.tcss")
            root = Path(__file__).parent.parent
            resolved_path = (root / str(raw_path)).resolve()
            self.css_path = [str(resolved_path)]

        def on_mount(self) -> None:
            """Push the main menu screen when the app starts."""
            self.push_screen(MainMenuScreen())

    app = QAEvalApp()
    app.run()
    return 0


def run_batch_evaluation(summaries_path: Path) -> int:
    """Run folder evaluation mode and return a process exit code."""
    evaluator = SummaryBatchEvaluator(log_callback=_write_console)
    try:
        result = asyncio.run(evaluator.evaluate_folder(summaries_path))
    except Exception as exc:
        LOGGER.error("Batch evaluation failed: %s", exc)
        return 1

    LOGGER.info("Wrote batch results to %s", result.csv_path)
    LOGGER.info(
        "Processed %s summary files with %s error row(s).",
        result.row_count,
        result.error_count,
    )
    return 1 if result.had_errors else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured qa-eval entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.summaries is not None:
        return run_batch_evaluation(args.summaries)
    return run_tui()


if __name__ == "__main__":
    raise SystemExit(main())
