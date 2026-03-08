"""Regression tests for the main entrypoint dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

import src.main as main


def test_main_launches_tui_when_no_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Launch the TUI path when no batch folder is provided."""
    called = {"tui": 0}

    def fake_run_tui() -> int:
        called["tui"] += 1
        return 0

    monkeypatch.setattr(main, "run_tui", fake_run_tui)
    monkeypatch.setattr(
        main,
        "run_batch_evaluation",
        lambda path: pytest.fail(f"unexpected batch call for {path}"),
    )

    assert main.main([]) == 0
    assert called["tui"] == 1


def test_main_runs_batch_mode_when_summaries_flag_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dispatch to batch evaluation when `--summaries` is supplied."""
    called: dict[str, Path] = {}
    summaries_dir = tmp_path / "summaries"
    summaries_dir.mkdir()

    def fake_run_batch_evaluation(path: Path) -> int:
        called["path"] = path
        return 1

    monkeypatch.setattr(
        main,
        "run_tui",
        lambda: pytest.fail("unexpected TUI launch in batch mode"),
    )
    monkeypatch.setattr(main, "run_batch_evaluation", fake_run_batch_evaluation)

    assert main.main(["--summaries", str(summaries_dir)]) == 1
    assert called["path"] == summaries_dir
