"""Run a WSL-backed QAFactEval benchmark over repo-local candidate summaries."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Optional

LOGGER = logging.getLogger(__name__)

SCRIPT_PATH = Path(__file__).resolve()
OTHER_BENCH_DIR = SCRIPT_PATH.parent
REPO_ROOT = OTHER_BENCH_DIR.parent
DEFAULT_SUMMARIES_DIR = OTHER_BENCH_DIR / "candidate_summaries"
DEFAULT_TRANSCRIPT_PATH = OTHER_BENCH_DIR / "ground_truth_transcript" / "transcript.json"
DEFAULT_OUTPUT_PATH = OTHER_BENCH_DIR / "qafacteval_results.csv"
DEFAULT_MODEL_DIR = OTHER_BENCH_DIR / "models" / "qafacteval"
DOWNLOAD_CACHE_DIR = OTHER_BENCH_DIR / ".cache" / "qafacteval_downloads"
BENCHMARK_PYPROJECT_DIR = OTHER_BENCH_DIR
SUMMARY_SUFFIX = "-summary.xml"
CSV_FIELDNAMES = [
    "summarizer",
    "critic",
    "qafacteval_is_answered",
    "qafacteval_em",
    "qafacteval_f1",
    "qafacteval_lerc_quip",
]
QAFactEvalMetricKeys = ("is_answered", "em", "f1", "lerc_quip")
GENERATION_MODEL_ID = "1vVhRgLtsQDAOmxYhY5PMPnxxHUyCOdQU"
ANSWERING_MODEL_ID = "1q2Z3FPP9AYNz0RJKHMlaweNhmLQoyPA8"
LERC_MODEL_ID = "193K7v6pjOtuXdlMenQW-RzF6ft-xY2qd"
LERC_PRETRAINING_MODEL_ID = "1fWBahDT-O1mpsbND300cuZuF73mfObzH"
QUIP_ARCHIVE_URL = "https://storage.googleapis.com/sfr-qafacteval-research/quip-512-mocha.tar.gz"
GENERATION_ARCHIVE_CACHE_NAME = "generation_model.tar.gz"
ANSWERING_ARCHIVE_CACHE_NAME = "answering_model.zip"
LERC_ARCHIVE_CACHE_NAME = "lerc_model.tar.gz"
LERC_PRETRAINING_CACHE_NAME = "lerc_pretraining.tar.gz"
QUIP_ARCHIVE_CACHE_NAME = "quip-512-mocha.tar.gz"
GENERATION_ARCHIVE_MIN_SIZE_BYTES = 1_000_000_000
GENERATION_SERIALIZATION_DIR_NAME = "serialization"
GENERATION_SERIALIZATION_MARKERS = ("config.json", "weights.th", "vocabulary")
ANSWERING_DIR_MARKERS = ("config.json", "pytorch_model.bin", "vocab.txt")
QUIP_DIR_MARKERS = ("config.json", "pytorch_model.bin", "vocab.json", "merges.txt")
BART_LARGE_CACHE_URLS = {
    "config.json": (
        "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/config.json",
        "https://cdn.huggingface.co/facebook/bart-large/config.json",
    ),
    "pytorch_model.bin": (
        "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/pytorch_model.bin",
        "https://cdn.huggingface.co/facebook/bart-large/pytorch_model.bin",
    ),
    "vocab.json": (
        "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/vocab.json",
        "https://cdn.huggingface.co/facebook/bart-large/vocab.json",
    ),
    "merges.txt": (
        "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/merges.txt",
        "https://cdn.huggingface.co/facebook/bart-large/merges.txt",
    ),
}
UV_LOOKUP_COMMAND = "command -v uv"
NETWORK_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_SIZE_BYTES = 1024 * 1024
DOWNLOAD_PROGRESS_UPDATES = 20
QAFACTEVAL_READY_MARKER_NAME = ".ready.json"
QAFACTEVAL_READY_MARKER_VERSION = 1
CPU_LERC_BATCH_SIZE = 8


@dataclass(frozen=True)
class BenchmarkInputRow:
    """Store one normalized transcript-summary pair for WSL-side scoring."""

    summary_file: str
    summarizer: str
    critic: str
    transcript_text: str
    summary_text: str


@dataclass(frozen=True)
class BenchmarkOutputRow:
    """Store one flattened CSV row for the benchmark output."""

    summarizer: str
    critic: str
    qafacteval_is_answered: float
    qafacteval_em: float
    qafacteval_f1: float
    qafacteval_lerc_quip: float


@dataclass
class ProgressTracker:
    """Render a simple ASCII progress bar for major benchmark stages."""

    total_steps: int
    label: str
    completed_steps: int = 0

    def advance(self, message: str) -> None:
        """Advance the bar by one step and log the new status line."""
        self.completed_steps += 1
        LOGGER.info(
            "%s %s",
            render_progress_bar(self.completed_steps, self.total_steps),
            f"{self.label}: {message}",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for host and hidden WSL worker modes."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate repo-local candidate summaries against the ground-truth "
            "transcript with official QAFactEval running inside WSL."
        )
    )
    parser.add_argument(
        "--summaries-dir",
        type=Path,
        default=DEFAULT_SUMMARIES_DIR,
        help="Directory containing `*-summary.xml` files.",
    )
    parser.add_argument(
        "--transcript-path",
        type=Path,
        default=DEFAULT_TRANSCRIPT_PATH,
        help="Path to the ground-truth transcript JSON file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--summary-pattern",
        default="*",
        help="Shell-style filename pattern used to filter summary files.",
    )
    parser.add_argument(
        "--max-summaries",
        type=int,
        default=None,
        help="Optional cap on the number of matched summaries to evaluate.",
    )
    parser.add_argument("--wsl-download-models", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wsl-score-input", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--wsl-score-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Dispatch the host runner or one of the hidden WSL worker modes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for logger_name in ("allennlp", "filelock", "transformers"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    parser = build_parser()
    args = parser.parse_args()

    if args.wsl_download_models:
        ensure_qafacteval_models(args.model_dir.resolve())
        return 0

    if args.wsl_score_input is not None or args.wsl_score_output is not None:
        if args.wsl_score_input is None or args.wsl_score_output is None:
            parser.error("Both `--wsl-score-input` and `--wsl-score-output` are required.")
        score_rows_with_qafacteval(
            input_path=args.wsl_score_input.resolve(),
            output_path=args.wsl_score_output.resolve(),
            model_dir=args.model_dir.resolve(),
        )
        return 0

    run_host_benchmark(
        summaries_dir=args.summaries_dir.resolve(),
        transcript_path=args.transcript_path.resolve(),
        output_path=args.output_path.resolve(),
        summary_pattern=args.summary_pattern,
        max_summaries=args.max_summaries,
        model_dir=args.model_dir.resolve(),
    )
    return 0


def run_host_benchmark(
    summaries_dir: Path,
    transcript_path: Path,
    output_path: Path,
    summary_pattern: str,
    max_summaries: Optional[int],
    model_dir: Path,
) -> None:
    """Run the full benchmark from Windows and write the final CSV output."""
    if os.name != "nt":
        raise RuntimeError("The host benchmark runner must be launched from Windows.")

    progress = ProgressTracker(total_steps=5, label="Host")
    LOGGER.info("Discovering candidate summaries in %s", summaries_dir)
    all_summary_files = list_summary_files(summaries_dir)
    parsed_parts, filename_errors = infer_summary_file_parts_from_repo(all_summary_files)
    selected_summary_files = filter_summary_files(
        summary_files=all_summary_files,
        filename_errors=filename_errors,
        summary_pattern=summary_pattern,
        max_summaries=max_summaries,
    )

    LOGGER.info("Loading and normalizing the ground-truth transcript from %s", transcript_path)
    transcript_text = load_transcript_text(transcript_path)
    LOGGER.info("Preparing %s selected summaries for scoring.", len(selected_summary_files))
    input_rows = [
        BenchmarkInputRow(
            summary_file=summary_file.name,
            summarizer=parsed_parts[summary_file.name]["summarizer"],
            critic=parsed_parts[summary_file.name]["critic"],
            transcript_text=transcript_text,
            summary_text=load_summary_text(summary_file),
        )
        for summary_file in selected_summary_files
    ]
    if not input_rows:
        raise ValueError("No candidate summaries matched the requested selection.")
    progress.advance(f"Prepared {len(input_rows)} normalized transcript-summary pairs.")

    wsl_uv_path = discover_wsl_uv_path()
    ensure_benchmark_environment(wsl_uv_path)
    progress.advance("WSL Python 3.8 benchmark environment is ready.")
    ensure_wsl_models_downloaded(wsl_uv_path, model_dir)
    progress.advance("Official QAFactEval model assets are available.")

    with tempfile.TemporaryDirectory(prefix="qafacteval-benchmark-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        input_path = temp_dir / "input.json"
        output_json_path = temp_dir / "output.json"
        input_path.write_text(
            json.dumps([asdict(row) for row in input_rows], indent=2),
            encoding="utf-8",
        )
        run_wsl_scoring(
            wsl_uv_path=wsl_uv_path,
            input_path=input_path,
            output_path=output_json_path,
            model_dir=model_dir,
        )
        metric_rows = json.loads(output_json_path.read_text(encoding="utf-8"))
    progress.advance("WSL-side QAFactEval scoring completed.")

    output_rows = [
        BenchmarkOutputRow(
            summarizer=input_row.summarizer,
            critic=input_row.critic,
            qafacteval_is_answered=float(metric_row["is_answered"]),
            qafacteval_em=float(metric_row["em"]),
            qafacteval_f1=float(metric_row["f1"]),
            qafacteval_lerc_quip=float(metric_row["lerc_quip"]),
        )
        for input_row, metric_row in zip(input_rows, metric_rows, strict=True)
    ]
    write_output_csv(output_path, output_rows)
    progress.advance(f"Wrote {len(output_rows)} CSV rows to {output_path}.")
    LOGGER.info("Wrote %s benchmark rows to %s", len(output_rows), output_path)


def list_summary_files(summaries_dir: Path) -> list[Path]:
    """Return all supported summary files in deterministic order."""
    if not summaries_dir.exists():
        raise ValueError(f"Summaries directory does not exist: {summaries_dir}")
    if not summaries_dir.is_dir():
        raise ValueError(f"Summaries path is not a directory: {summaries_dir}")

    summary_files = sorted(
        path
        for path in summaries_dir.iterdir()
        if path.is_file() and path.name.endswith(SUMMARY_SUFFIX)
    )
    if not summary_files:
        raise ValueError(f"No `{SUMMARY_SUFFIX}` files were found in {summaries_dir}.")
    return summary_files


def filter_summary_files(
    summary_files: list[Path],
    filename_errors: dict[str, str],
    summary_pattern: str,
    max_summaries: Optional[int],
) -> list[Path]:
    """Filter summary files after directory-wide filename inference."""
    filtered_files = [
        summary_file
        for summary_file in summary_files
        if fnmatch.fnmatch(summary_file.name, summary_pattern)
    ]
    if max_summaries is not None:
        filtered_files = filtered_files[:max_summaries]

    for summary_file in filtered_files:
        error_message = filename_errors.get(summary_file.name)
        if error_message is not None:
            raise ValueError(
                f"Could not infer summarizer/critic for {summary_file.name}: "
                f"{error_message}"
            )

    return filtered_files


def infer_summary_file_parts_from_repo(
    summary_files: list[Path],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Infer summary filename metadata using the repo's existing parsing logic."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.services.batch_summary_evaluator import infer_summary_file_parts

    parsed_parts, errors = infer_summary_file_parts(summary_files)
    normalized_parts = {
        filename: {"summarizer": parts.summarizer, "critic": parts.critic}
        for filename, parts in parsed_parts.items()
    }
    return normalized_parts, errors


def load_transcript_text(transcript_path: Path) -> str:
    """Load and flatten the transcript JSON into speaker-labeled text."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.models.data_models import Transcript

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript = Transcript.model_validate(payload)
    lines = [
        f"{segment.speaker}: {segment.text.strip()}"
        for segment in transcript.segments
        if segment.text.strip()
    ]
    if not lines:
        raise ValueError(f"Transcript did not contain any non-empty segments: {transcript_path}")
    return "\n".join(lines)


def load_summary_text(summary_path: Path) -> str:
    """Load and flatten one tagged summary into speaker-labeled text."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src.models.data_models import SPEAKER_BLOCK_PATTERN, Summary

    summary = Summary(content=summary_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    for match in SPEAKER_BLOCK_PATTERN.finditer(summary.content):
        text = match.group(2).strip()
        if text:
            lines.append(f"{match.group(1)}: {text}")
    if not lines:
        raise ValueError(f"Summary did not contain any non-empty speaker blocks: {summary_path}")
    return "\n".join(lines)


def discover_wsl_uv_path() -> str:
    """Return the absolute path to `uv` inside WSL."""
    command = ["wsl.exe", "bash", "-lc", UV_LOOKUP_COMMAND]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    uv_path = completed.stdout.strip()
    if not uv_path:
        raise RuntimeError("Could not locate `uv` inside WSL.")
    return uv_path


def ensure_benchmark_environment(wsl_uv_path: str) -> None:
    """Install Python 3.8 in WSL and sync the benchmark-local environment."""
    LOGGER.info("Ensuring the WSL Python 3.8 benchmark environment exists.")
    run_wsl_command(
        [wsl_uv_path, "python", "install", "3.8"],
        cwd=REPO_ROOT,
    )
    run_wsl_command(
        [
            wsl_uv_path,
            "sync",
            "--project",
            windows_path_to_wsl(BENCHMARK_PYPROJECT_DIR),
            "--locked",
            "--prerelease",
            "allow",
            "--python",
            "3.8",
        ],
        cwd=REPO_ROOT,
    )


def ensure_wsl_models_downloaded(wsl_uv_path: str, model_dir: Path) -> None:
    """Download official QAFactEval model assets inside the WSL environment."""
    if qafacteval_assets_are_ready(model_dir):
        LOGGER.info("Skipping WSL asset preparation because the benchmark cache is already ready.")
        return

    LOGGER.info("Ensuring official QAFactEval model assets are available.")
    run_wsl_python_script(
        wsl_uv_path=wsl_uv_path,
        extra_args=[
            "--wsl-download-models",
            "--model-dir",
            windows_path_to_wsl(model_dir),
        ],
    )


def run_wsl_scoring(
    wsl_uv_path: str,
    input_path: Path,
    output_path: Path,
    model_dir: Path,
) -> None:
    """Run the hidden WSL-side scorer and write its JSON output file."""
    LOGGER.info("Running QAFactEval inside WSL for %s", input_path)
    run_wsl_python_script(
        wsl_uv_path=wsl_uv_path,
        extra_args=[
            "--wsl-score-input",
            windows_path_to_wsl(input_path),
            "--wsl-score-output",
            windows_path_to_wsl(output_path),
            "--model-dir",
            windows_path_to_wsl(model_dir),
        ],
    )


def run_wsl_python_script(wsl_uv_path: str, extra_args: list[str]) -> None:
    """Run this script inside the benchmark-local WSL environment."""
    command = [
        wsl_uv_path,
        "run",
        "--project",
        windows_path_to_wsl(BENCHMARK_PYPROJECT_DIR),
        "--locked",
        "--prerelease",
        "allow",
        "--python",
        "3.8",
        "python",
        windows_path_to_wsl(SCRIPT_PATH),
        *extra_args,
    ]
    run_wsl_command(command, cwd=REPO_ROOT)


def run_wsl_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one command inside WSL with deterministic Linux working directory."""
    wsl_command = [
        "wsl.exe",
        "--cd",
        windows_path_to_wsl(cwd),
        *command,
    ]
    return subprocess.run(
        wsl_command,
        check=True,
        text=True,
        capture_output=False,
    )


def windows_path_to_wsl(path: Path) -> str:
    """Convert one absolute Windows path into its WSL mount path."""
    resolved = path.resolve()
    windows_path = PureWindowsPath(str(resolved))
    drive = windows_path.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Expected a Windows path with a drive letter, got: {resolved}")
    return str(PurePosixPath("/mnt", drive, *windows_path.parts[1:]))


def get_qafacteval_ready_marker_path(model_dir: Path) -> Path:
    """Return the marker file used to skip repeated WSL model-prep passes."""
    return model_dir / QAFACTEVAL_READY_MARKER_NAME


def qafacteval_assets_are_ready(model_dir: Path) -> bool:
    """Return whether all reusable benchmark assets already exist locally."""
    marker_path = get_qafacteval_ready_marker_path(model_dir)
    if not marker_path.exists():
        return False

    try:
        marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if marker_payload.get("version") != QAFACTEVAL_READY_MARKER_VERSION:
        return False

    generation_dir = model_dir / "generation"
    lerc_dir = model_dir / "lerc"
    return all(
        (
            (generation_dir / "model.tar.gz").exists(),
            (lerc_dir / "model.tar.gz").exists(),
            (lerc_dir / "pretraining.tar.gz").exists(),
        )
    ) and directory_has_markers(
        model_dir / "answering",
        ANSWERING_DIR_MARKERS,
    ) and directory_has_markers(
        model_dir / "quip-512-mocha",
        QUIP_DIR_MARKERS,
    )


def write_qafacteval_ready_marker(model_dir: Path) -> None:
    """Persist the local-ready marker after model prep succeeds once."""
    marker_payload = {"version": QAFACTEVAL_READY_MARKER_VERSION}
    marker_path = get_qafacteval_ready_marker_path(model_dir)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps(marker_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def ensure_qafacteval_models(model_dir: Path) -> None:
    """Download and extract the official QAFactEval model assets."""
    if qafacteval_assets_are_ready(model_dir):
        LOGGER.info("Official QAFactEval assets are already ready at %s. Skipping preparation.", model_dir)
        return

    LOGGER.info("Preparing official QAFactEval assets under %s", model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    generation_dir = model_dir / "generation"
    answering_dir = model_dir / "answering"
    lerc_dir = model_dir / "lerc"
    generation_dir.mkdir(parents=True, exist_ok=True)
    lerc_dir.mkdir(parents=True, exist_ok=True)

    generation_archive = generation_dir / "model.tar.gz"
    if not generation_archive.exists():
        generation_archive.parent.mkdir(parents=True, exist_ok=True)
        if not move_cached_file_into_place(
            candidates=[
                DOWNLOAD_CACHE_DIR / GENERATION_ARCHIVE_CACHE_NAME,
                OTHER_BENCH_DIR / "model.tar.gz",
            ],
            output_path=generation_archive,
            description="generation model archive",
            validator=is_generation_archive,
        ):
            LOGGER.info("Downloading generation model archive.")
            cache_path = DOWNLOAD_CACHE_DIR / GENERATION_ARCHIVE_CACHE_NAME
            download_google_drive_file(GENERATION_MODEL_ID, cache_path)
            shutil.move(str(cache_path), str(generation_archive))
    else:
        LOGGER.info("Generation model archive already exists. Skipping download.")

    if not directory_has_markers(answering_dir, ANSWERING_DIR_MARKERS):
        if not move_cached_directory_into_place(
            candidates=[
                DOWNLOAD_CACHE_DIR / "answering",
                OTHER_BENCH_DIR / "answering",
            ],
            output_path=answering_dir,
            description="answering model directory",
            validator=lambda path: directory_has_markers(path, ANSWERING_DIR_MARKERS),
        ):
            LOGGER.info("Downloading and extracting answering model.")
            zip_path = DOWNLOAD_CACHE_DIR / ANSWERING_ARCHIVE_CACHE_NAME
            move_cached_file_into_place(
                candidates=[OTHER_BENCH_DIR / "model.zip"],
                output_path=zip_path,
                description="cached answering model archive",
            )
            if not zip_path.exists():
                download_google_drive_file(ANSWERING_MODEL_ID, zip_path)
            extract_archive_once(
                archive_path=zip_path,
                output_dir=answering_dir,
                expected_markers=ANSWERING_DIR_MARKERS,
                extractor=extract_zip_archive,
            )
    else:
        LOGGER.info("Answering model directory already exists. Skipping download.")

    lerc_model_archive = lerc_dir / "model.tar.gz"
    if not lerc_model_archive.exists():
        lerc_model_archive.parent.mkdir(parents=True, exist_ok=True)
        if not move_cached_file_into_place(
            candidates=[
                DOWNLOAD_CACHE_DIR / LERC_ARCHIVE_CACHE_NAME,
                OTHER_BENCH_DIR / "model.tar.gz",
            ],
            output_path=lerc_model_archive,
            description="LERC model archive",
            validator=is_lerc_archive,
        ):
            LOGGER.info("Downloading LERC model archive.")
            cache_path = DOWNLOAD_CACHE_DIR / LERC_ARCHIVE_CACHE_NAME
            download_google_drive_file(LERC_MODEL_ID, cache_path)
            shutil.move(str(cache_path), str(lerc_model_archive))
    else:
        LOGGER.info("LERC model archive already exists. Skipping download.")

    lerc_pretraining_archive = lerc_dir / "pretraining.tar.gz"
    if not lerc_pretraining_archive.exists():
        lerc_pretraining_archive.parent.mkdir(parents=True, exist_ok=True)
        if not move_cached_file_into_place(
            candidates=[
                DOWNLOAD_CACHE_DIR / LERC_PRETRAINING_CACHE_NAME,
                OTHER_BENCH_DIR / "pretraining.tar.gz",
            ],
            output_path=lerc_pretraining_archive,
            description="LERC pretraining archive",
        ):
            LOGGER.info("Downloading LERC pretraining archive.")
            cache_path = DOWNLOAD_CACHE_DIR / LERC_PRETRAINING_CACHE_NAME
            download_google_drive_file(LERC_PRETRAINING_MODEL_ID, cache_path)
            shutil.move(str(cache_path), str(lerc_pretraining_archive))
    else:
        LOGGER.info("LERC pretraining archive already exists. Skipping download.")

    quip_dir = model_dir / "quip-512-mocha"
    if not directory_has_markers(quip_dir, QUIP_DIR_MARKERS):
        if not move_cached_directory_into_place(
            candidates=[
                DOWNLOAD_CACHE_DIR / "quip-512-mocha",
                OTHER_BENCH_DIR / "quip-512-mocha",
            ],
            output_path=quip_dir,
            description="Quip model directory",
            validator=lambda path: directory_has_markers(path, QUIP_DIR_MARKERS),
        ):
            LOGGER.info("Downloading and extracting Quip model archive.")
            archive_path = DOWNLOAD_CACHE_DIR / QUIP_ARCHIVE_CACHE_NAME
            move_cached_file_into_place(
                candidates=[OTHER_BENCH_DIR / QUIP_ARCHIVE_CACHE_NAME],
                output_path=archive_path,
                description="cached Quip model archive",
            )
            if not archive_path.exists():
                download_url_to_path(QUIP_ARCHIVE_URL, archive_path)
            extract_archive_once(
                archive_path=archive_path,
                output_dir=quip_dir,
                expected_markers=QUIP_DIR_MARKERS,
                extractor=lambda source, destination: extract_tar_archive(
                    source,
                    destination.parent,
                ),
            )
    else:
        LOGGER.info("Quip model directory already exists. Skipping download.")

    ensure_qaeval_transformers_cache()
    write_qafacteval_ready_marker(model_dir)


def move_cached_file_into_place(
    candidates: list[Path],
    output_path: Path,
    description: str,
    validator: Optional[Callable[[Path], bool]] = None,
) -> bool:
    """Move an already-downloaded file into its final location when possible."""
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        if validator is not None and not validator(candidate):
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Moving cached %s from %s to %s", description, candidate, output_path)
        shutil.move(str(candidate), str(output_path))
        return True
    return False


def move_cached_directory_into_place(
    candidates: list[Path],
    output_path: Path,
    description: str,
    validator: Optional[Callable[[Path], bool]] = None,
) -> bool:
    """Move an already-extracted directory into its final location when possible."""
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if validator is not None and not validator(candidate):
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Moving cached %s from %s to %s", description, candidate, output_path)
        shutil.move(str(candidate), str(output_path))
        return True
    return False


def directory_has_markers(directory: Path, markers: tuple[str, ...]) -> bool:
    """Return whether a directory contains all required files or subdirectories."""
    if not directory.exists() or not directory.is_dir():
        return False
    return all((directory / marker).exists() for marker in markers)


def extract_archive_once(
    archive_path: Path,
    output_dir: Path,
    expected_markers: tuple[str, ...],
    extractor: Callable[[Path, Path], None],
) -> None:
    """Extract one archive only when its reusable output directory is missing or incomplete."""
    if directory_has_markers(output_dir, expected_markers):
        LOGGER.info("Using cached extracted archive at %s", output_dir)
        return

    LOGGER.info("Extracting %s into %s", archive_path, output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{output_dir.name}-",
        dir=output_dir.parent,
    ) as temp_dir_name:
        temp_output_dir = Path(temp_dir_name) / output_dir.name
        extractor(archive_path, temp_output_dir)
        if not directory_has_markers(temp_output_dir, expected_markers):
            raise RuntimeError(
                f"Extracted archive {archive_path} did not produce the expected files in {temp_output_dir}."
            )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(temp_output_dir), str(output_dir))


def extract_zip_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract one ZIP archive into a target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)


def extract_tar_archive(archive_path: Path, output_dir: Path) -> None:
    """Extract one tar.gz archive into a target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(output_dir)


def is_generation_archive(path: Path) -> bool:
    """Return whether a cached archive looks like the large generation model."""
    return path.stat().st_size >= GENERATION_ARCHIVE_MIN_SIZE_BYTES


def is_lerc_archive(path: Path) -> bool:
    """Return whether a cached archive looks like the smaller LERC model."""
    return path.stat().st_size < GENERATION_ARCHIVE_MIN_SIZE_BYTES


def ensure_qaeval_transformers_cache() -> None:
    """Prefetch the legacy BART-large files required by upstream QAEval."""
    from transformers.file_utils import TRANSFORMERS_CACHE, url_to_filename

    cache_dir = Path(TRANSFORMERS_CACHE)
    cache_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = DOWNLOAD_CACHE_DIR / "transformers"
    staging_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(total_steps=len(BART_LARGE_CACHE_URLS), label="Transformers cache")

    for filename, urls in BART_LARGE_CACHE_URLS.items():
        ensure_transformers_cache_asset(
            filename=filename,
            urls=urls,
            cache_dir=cache_dir,
            staging_dir=staging_dir,
            url_to_filename=url_to_filename,
        )
        progress.advance(f"Ready: {filename} ({len(urls)} cache keys)")


def ensure_transformers_cache_asset(
    filename: str,
    urls: tuple[str, ...],
    cache_dir: Path,
    staging_dir: Path,
    url_to_filename: Callable[[str, Optional[str]], str],
) -> None:
    """Ensure all legacy cache aliases exist for one transformers asset."""
    alias_records: list[tuple[str, Optional[str], Path, Path]] = []
    seed_path: Optional[Path] = None

    for url in urls:
        etag = resolve_remote_etag(url)
        cache_name = url_to_filename(url, etag)
        final_path = cache_dir / cache_name
        metadata_path = cache_dir / f"{cache_name}.json"
        alias_records.append((url, etag, final_path, metadata_path))
        if final_path.exists():
            LOGGER.info("Transformers cache hit for %s at %s", filename, final_path)
            seed_path = seed_path or final_path

    if seed_path is None:
        staging_path = staging_dir / filename
        legacy_candidates = [OTHER_BENCH_DIR / filename]
        for url in urls:
            legacy_cache_path = cache_dir / url_to_filename(url)
            if legacy_cache_path not in legacy_candidates:
                legacy_candidates.append(legacy_cache_path)

        if not move_cached_file_into_place(
            candidates=legacy_candidates,
            output_path=staging_path,
            description=f"legacy transformers cache file {filename}",
        ):
            if staging_path.exists():
                LOGGER.info("Using cached staged transformers asset %s", staging_path)
            else:
                LOGGER.info("Prefetching legacy transformers asset %s", filename)
                download_url_to_path(urls[0], staging_path)

        first_path = alias_records[0][2]
        first_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staging_path), str(first_path))
        seed_path = first_path

    for url, etag, final_path, metadata_path in alias_records:
        if not final_path.exists():
            if seed_path is None:
                raise RuntimeError(f"No cache seed path was prepared for {filename}.")
            LOGGER.info("Linking cached transformers asset %s to %s", filename, final_path)
            link_or_copy_file(seed_path, final_path)
        ensure_transformers_cache_metadata(metadata_path, url, etag)


def link_or_copy_file(source_path: Path, output_path: Path) -> None:
    """Create one filesystem-level copy while preferring hard links for cache aliases."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return
    try:
        os.link(source_path, output_path)
    except OSError:
        shutil.copy2(source_path, output_path)


def resolve_remote_etag(url: str) -> Optional[str]:
    """Resolve one remote ETag for legacy transformers cache naming."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
            etag = response.headers.get("ETag")
    except urllib.error.HTTPError as error:
        if error.code in {405, 501}:
            LOGGER.info("HEAD is unavailable for %s. Falling back to an ETag-free cache key.", url)
            return None
        raise
    except (TimeoutError, OSError, urllib.error.URLError) as error:
        LOGGER.info(
            "Could not resolve an ETag for %s (%s). Falling back to an ETag-free cache key.",
            url,
            error,
        )
        return None
    LOGGER.info("Resolved transformers cache ETag for %s: %s", url, etag or "<none>")
    return etag


def ensure_transformers_cache_metadata(metadata_path: Path, url: str, etag: Optional[str]) -> None:
    """Write the metadata sidecar expected by legacy transformers cache lookups."""
    payload = {"url": url, "etag": etag}
    serialized_payload = json.dumps(payload, sort_keys=True)
    if metadata_path.exists():
        existing_payload = metadata_path.read_text(encoding="utf-8").strip()
        if existing_payload == serialized_payload:
            return
    metadata_path.write_text(serialized_payload, encoding="utf-8")


def download_google_drive_file(file_id: str, output_path: Path) -> None:
    """Download one Google Drive file with `gdown` from inside the benchmark env."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qafacteval-download-") as temp_dir_name:
        temp_path = Path(temp_dir_name) / output_path.name
        LOGGER.info("Downloading Google Drive file %s to %s", file_id, output_path)
        command = [
            sys.executable,
            "-m",
            "gdown",
            "--id",
            file_id,
            "-O",
            str(temp_path),
        ]
        subprocess.run(command, check=True)
        shutil.move(str(temp_path), str(output_path))


def download_url_to_path(url: str, output_path: Path) -> None:
    """Download one direct URL to a local filesystem path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s to %s", url, output_path)
    with tempfile.NamedTemporaryFile(
        prefix=f"{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as temp_file_handle:
        temp_path = Path(temp_file_handle.name)

    try:
        with urllib.request.urlopen(url, timeout=NETWORK_TIMEOUT_SECONDS) as response, open(
            temp_path,
            "wb",
        ) as file_handle:
            total_bytes = get_content_length(response.headers)
            downloaded_bytes = 0
            log_interval_bytes = get_download_log_interval(total_bytes)
            next_log_threshold = log_interval_bytes

            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                file_handle.write(chunk)
                downloaded_bytes += len(chunk)
                if downloaded_bytes >= next_log_threshold:
                    if total_bytes > 0:
                        LOGGER.info(
                            "%s Downloading %s (%s/%s)",
                            render_progress_bar(downloaded_bytes, total_bytes),
                            output_path.name,
                            format_bytes(downloaded_bytes),
                            format_bytes(total_bytes),
                        )
                    else:
                        LOGGER.info(
                            "Downloading %s (%s downloaded)",
                            output_path.name,
                            format_bytes(downloaded_bytes),
                        )
                    next_log_threshold += log_interval_bytes
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    shutil.move(str(temp_path), str(output_path))
    LOGGER.info("Finished downloading %s (%s)", output_path.name, format_bytes(output_path.stat().st_size))


def get_content_length(headers: object) -> int:
    """Return the parsed content length or zero when it is unavailable."""
    content_length = headers.get("Content-Length")
    if content_length is None:
        return 0
    try:
        return int(content_length)
    except ValueError:
        return 0


def get_download_log_interval(total_bytes: int) -> int:
    """Return how often download progress should be logged."""
    if total_bytes <= 0:
        return 64 * 1024 * 1024
    return max(total_bytes // DOWNLOAD_PROGRESS_UPDATES, DOWNLOAD_CHUNK_SIZE_BYTES)


def format_bytes(size_in_bytes: int) -> str:
    """Format a byte count for human-readable download logging."""
    if size_in_bytes <= 0:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(size_in_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_in_bytes} B"


def score_rows_with_qafacteval(input_path: Path, output_path: Path, model_dir: Path) -> None:
    """Score the normalized input rows with the official QAFactEval package."""
    import torch
    from qafacteval import QAFactEval

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    cuda_device = select_qafacteval_cuda_device(torch)
    device_label = "cuda:0" if cuda_device >= 0 else "cpu"
    lerc_batch_size = select_qafacteval_lerc_batch_size(torch, cuda_device)
    generation_model_path = ensure_generation_serialization_dir(model_dir / "generation")
    LOGGER.info("Loading QAFactEval models on %s.", device_label)
    LOGGER.info("Using Quip batch size %s.", lerc_batch_size)
    metric = QAFactEval(
        lerc_quip_path=str(model_dir / "quip-512-mocha"),
        generation_model_path=str(generation_model_path),
        answering_model_dir=str(model_dir / "answering"),
        lerc_model_path=str(model_dir / "lerc" / "model.tar.gz"),
        lerc_pretrained_model_path=str(model_dir / "lerc" / "pretraining.tar.gz"),
        cuda_device=cuda_device,
        use_lerc_quip=True,
        verbose=False,
        generation_batch_size=32,
        answering_batch_size=32,
        lerc_batch_size=lerc_batch_size,
    )
    LOGGER.info("Loaded QAFactEval models. Scoring %s summaries.", len(rows))
    LOGGER.info(
        "Running one batched QAFactEval pass for %s summaries to avoid per-summary model reloads.",
        len(rows),
    )
    results = metric.score_batch_qafacteval(
        [row["transcript_text"] for row in rows],
        [[row["summary_text"]] for row in rows],
        return_qa_pairs=False,
    )
    metric_rows = []
    total_rows = len(rows)
    for index, (row, result) in enumerate(zip(rows, results), start=1):
        metric_rows.append(
            {
                key: float(result["qa-eval"][key])
                for key in QAFactEvalMetricKeys
            }
        )
        LOGGER.info(
            "%s Finished summary %s/%s: %s",
            render_progress_bar(index, total_rows),
            index,
            total_rows,
            row["summary_file"],
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metric_rows, indent=2), encoding="utf-8")
    LOGGER.info("WSL scorer wrote %s metric rows to %s", len(metric_rows), output_path)


def ensure_generation_serialization_dir(generation_dir: Path) -> Path:
    """Return the reusable extracted AllenNLP serialization directory for question generation."""
    archive_path = generation_dir / "model.tar.gz"
    serialization_dir = generation_dir / GENERATION_SERIALIZATION_DIR_NAME
    extract_archive_once(
        archive_path=archive_path,
        output_dir=serialization_dir,
        expected_markers=GENERATION_SERIALIZATION_MARKERS,
        extractor=extract_tar_archive,
    )
    return serialization_dir


def select_qafacteval_cuda_device(torch_module: object) -> int:
    """Return the CUDA device index only when the installed torch build can use it."""
    cuda_namespace = torch_module.cuda
    if not cuda_namespace.is_available():
        LOGGER.info("CUDA is unavailable in the benchmark environment. Using CPU scoring.")
        return -1

    try:
        major, minor = cuda_namespace.get_device_capability(0)
    except Exception as error:
        LOGGER.info("Could not inspect CUDA device capability (%s). Falling back to CPU.", error)
        return -1

    compiled_arches = get_compiled_cuda_arches(cuda_namespace)
    device_arch = f"sm_{major}{minor}"
    if compiled_arches and device_arch not in compiled_arches:
        LOGGER.info(
            "Torch was built for %s, but the available GPU requires %s. Falling back to CPU.",
            ", ".join(compiled_arches),
            device_arch,
        )
        return -1

    try:
        probe_tensor = torch_module.zeros(1, device="cuda:0")
        _ = probe_tensor + 1
    except Exception as error:
        LOGGER.info("CUDA probe failed (%s). Falling back to CPU.", error)
        return -1
    LOGGER.info("CUDA probe succeeded for architecture %s.", device_arch)
    return 0


def select_qafacteval_lerc_batch_size(torch_module: object, cuda_device: int) -> int:
    """Choose a larger Quip batch size when the available GPU memory allows it."""
    if cuda_device < 0:
        return CPU_LERC_BATCH_SIZE

    cuda_namespace = torch_module.cuda
    try:
        total_memory_bytes = cuda_namespace.get_device_properties(cuda_device).total_memory
    except Exception as error:
        LOGGER.info(
            "Could not inspect GPU memory for Quip batching (%s). Using %s.",
            error,
            CPU_LERC_BATCH_SIZE,
        )
        return CPU_LERC_BATCH_SIZE

    total_memory_gib = total_memory_bytes / (1024**3)
    if total_memory_gib >= 10:
        return 32
    if total_memory_gib >= 8:
        return 24
    if total_memory_gib >= 6:
        return 16
    return CPU_LERC_BATCH_SIZE


def get_compiled_cuda_arches(cuda_namespace: object) -> list[str]:
    """Return the CUDA architectures compiled into the local torch build."""
    get_arch_list = getattr(cuda_namespace, "get_arch_list", None)
    if get_arch_list is None:
        return []
    try:
        compiled_arches = get_arch_list()
    except Exception:
        return []
    return [str(arch) for arch in compiled_arches]


def write_output_csv(output_path: Path, rows: list[BenchmarkOutputRow]) -> None:
    """Write the final benchmark CSV output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def render_progress_bar(current: int, total: int, width: int = 28) -> str:
    """Render a deterministic ASCII progress bar."""
    if total <= 0:
        return "[----------------------------] 0/0"
    safe_current = max(0, min(current, total))
    filled = int(width * safe_current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {safe_current}/{total}"


if __name__ == "__main__":
    raise SystemExit(main())
