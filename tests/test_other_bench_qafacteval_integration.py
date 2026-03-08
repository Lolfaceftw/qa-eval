"""Real integration coverage for the WSL-backed QAFactEval benchmark."""

from __future__ import annotations

import csv
import math
import subprocess
import sys
from pathlib import Path

import pytest

EXPECTED_HEADERS = [
    "summarizer",
    "critic",
    "qafacteval_is_answered",
    "qafacteval_em",
    "qafacteval_f1",
    "qafacteval_lerc_quip",
]


def wsl_is_available() -> bool:
    """Return whether WSL and `uv` are available for the benchmark test."""
    try:
        result = subprocess.run(
            ["wsl.exe", "bash", "-lc", "command -v uv"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


@pytest.mark.integration
def test_qafacteval_benchmark_runs_real_pipeline() -> None:
    """Run the real QAFactEval benchmark for one summary and validate the CSV."""
    if sys.platform != "win32":
        pytest.skip("The QAFactEval benchmark integration test requires Windows + WSL.")
    if not wsl_is_available():
        pytest.skip("WSL with `uv` is unavailable on this machine.")

    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "other_bench" / "quest_eval_qa_fact_eval.py"
    output_path = repo_root / "other_bench" / "test_outputs" / "qafacteval_integration_results.csv"
    if output_path.exists():
        output_path.unlink()

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--summary-pattern",
            "qwen3_5_4b_qwen3_5_4b-summary.xml",
            "--output-path",
            str(output_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=7200,
    )

    if completed.returncode != 0:
        pytest.fail(
            "The QAFactEval benchmark script failed.\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    assert output_path.exists()
    with open(output_path, "r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = list(reader)

    assert reader.fieldnames == EXPECTED_HEADERS
    assert len(rows) == 1

    row = rows[0]
    assert row["summarizer"] == "qwen3_5_4b"
    assert row["critic"] == "qwen3_5_4b"

    is_answered = float(row["qafacteval_is_answered"])
    em_score = float(row["qafacteval_em"])
    f1_score = float(row["qafacteval_f1"])
    lerc_quip_score = float(row["qafacteval_lerc_quip"])

    assert 0.0 <= is_answered <= 1.0
    assert 0.0 <= em_score <= 1.0
    assert 0.0 <= f1_score <= 1.0
    assert math.isfinite(lerc_quip_score)
