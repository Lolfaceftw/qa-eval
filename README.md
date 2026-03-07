# qa-eval

`qa-eval` is a Textual-based terminal UI for evaluating how much information a summary loses relative to a transcript. It generates large sets of yes/no review questions from a vLLM-compatible model, validates and filters them, deduplicates them with embeddings, evaluates the final question set against the summary, and writes JSON artifacts for downstream analysis.

## What It Does

The current pipeline:

1. Loads a transcript JSON file and a summary text file from configuration.
2. Validates both inputs with Pydantic models.
3. Renders transcript-only prompt templates for `factualness` and `naturalness`.
4. Streams question generation from a vLLM-compatible endpoint using transcript context only.
5. Parses and validates each model response as a raw JSON array of question objects.
6. Filters off-rubric questions, canonicalizes boilerplate-heavy wording, and deduplicates similar questions globally with embeddings.
7. Writes the filtered question set to [`data/processed_questions.json`](data/processed_questions.json) and a filtering report to [`data/question_filter_report.json`](data/question_filter_report.json).
8. Renders an evaluator prompt for each final category question set and streams strict yes/no answers from the same provider.
9. Writes per-question answers to [`data/question_evaluations.json`](data/question_evaluations.json) and normalized category scores to [`data/evaluation_report.json`](data/evaluation_report.json).

The project is aimed at information-loss evaluation. The transcript is treated as ground truth, and the generated questions are built from the transcript alone so the final evaluation remains independent of the candidate summary.

## Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management
- Access to a vLLM-compatible OpenAI Chat Completions endpoint
- Sufficient RAM or VRAM for the configured embedding model if you keep the default embedding stage

## Quickstart

Install dependencies:

```bash
uv sync --dev
```

Review and update [`cfg/config.yaml`](cfg/config.yaml) for your environment, especially:

- input file paths
- prompt template paths
- vLLM base URL, model, and connection settings
- embedding model, threshold, and device

Run the TUI from the repository root:

```bash
uv run python src/main.py
```

The main menu lets you:

- open `Run` to execute the full evaluation pipeline
- open `Config` to edit configuration values inside the TUI
- press `Ctrl+S` in the config screen to persist changes

This repository does not provision the model server for you. Point `vllm.base_url` at an existing compatible deployment.

## Configuration

The app reads configuration from [`cfg/config.yaml`](cfg/config.yaml). A typical setup looks like this:

```yaml
api_keys:
  vllm: EMPTY

data:
  transcript_path: data/transcript.json
  summary_path: data/summary.txt

prompts:
  evaluator: docs/prompts/evaluator.j2
  factualness: docs/prompts/factualness.j2
  naturalness: docs/prompts/naturalness.j2

ui:
  css_path: src/ui/css/style.tcss

vllm:
  base_url: http://localhost:8000/v1
  model: your-model-id
  max_context: 32768
  connection:
    preflight_timeout_seconds: 5.0
    connect_timeout_seconds: 5.0
    read_timeout_seconds: 120.0
    pool_timeout_seconds: 5.0
    keepalive_expiry_seconds: 30.0
    max_retries: 1

embedding:
  model: Qwen/Qwen3-Embedding-4B
  threshold: 0.85
  device: null
```

Notes:

- `app.provider` currently defaults to `vllm` when omitted.
- The run screen now validates `vllm.base_url` once with `models.list()` before starting generation and reuses that warmed connection for the agent requests.
- During each streamed model call, the run screen now shows prompt token count, an explicit "request submitted / waiting for first chunk" status, and first-token latency before the JSON body starts rendering.
- `vllm.connection.preflight_timeout_seconds` limits the initial endpoint check, while `connect_timeout_seconds` and `read_timeout_seconds` control generation requests.
- `vllm.connection.max_retries` is intentionally low by default so unhealthy endpoints fail fast instead of silently stalling.
- Generated outputs under `data/` are treated as local artifacts; the checked-in transcript and summary inputs remain the only repo-tracked files in that directory.
- `embedding.device: null` enables automatic device selection (`cuda` when available, otherwise `cpu`).
- Config edits made in the TUI are written back to disk immediately.

## Input Contracts

### Transcript

The transcript file must be JSON with a top-level `segments` list:

```json
{
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start_time": 0.0,
      "end_time": 3.2,
      "text": "Example transcript text."
    }
  ]
}
```

### Summary

The summary file is plain text, but it must include speaker tags such as `<SPEAKER_00>`:

```xml
<SPEAKER_00>Example summary text.</SPEAKER_00>
<SPEAKER_01>Another speaker block.</SPEAKER_01>
```

The current validator rejects summaries that do not contain speaker tags in the `<SPEAKER_XX>` format.

## Prompt And Output Expectations

Prompt templates live in [`docs/prompts/`](docs/prompts/). The current templates:

- request a minimum of 200 questions for each category
- require yes/no questions
- keep question generation summary-independent by giving the generator transcript context only
- require generated questions to be positively keyed so `yes` means the summary preserved the targeted signal
- require a category `dimension` for each generated question
- focus on information loss relative to the transcript
- expect the model to return a raw JSON array
- keep `factualness` and `naturalness` disjoint so naturalness questions stay about tone, flow, pacing, voice, or personality instead of factual entity recall
- use `docs/prompts/evaluator.j2` to answer the final questions with strict `yes` or `no`

The filtered question artifact is written to [`data/processed_questions.json`](data/processed_questions.json) and has this shape:

```json
{
  "factualness": [
    {
      "question_number": 1,
      "question": "Would a reviewer still know a key fact from the transcript after reading the summary?"
    }
  ],
  "naturalness": [
    {
      "question_number": 1,
      "question": "Does the summary preserve the speaker's natural flow?"
    }
  ]
}
```

The pipeline also writes [`data/question_filter_report.json`](data/question_filter_report.json) with per-category counts for parsed, invalid, off-rubric, exact-duplicate, semantic-duplicate, cross-category-drop, and final-kept questions.

The per-question evaluation artifact is written to [`data/question_evaluations.json`](data/question_evaluations.json):

```json
{
  "factualness": [
    {
      "question_number": 1,
      "question": "Would a reviewer still know a key fact from the transcript after reading the summary?",
      "answer": "yes"
    }
  ]
}
```

The scoring artifact is written to [`data/evaluation_report.json`](data/evaluation_report.json) and stores normalized 0-1 scores per category:

```json
{
  "global": {
    "total_questions": 2,
    "yes_count": 1,
    "no_count": 1,
    "invalid_or_missing_count": 0,
    "score": 0.5
  },
  "categories": {
    "factualness": {
      "total_questions": 1,
      "yes_count": 1,
      "no_count": 0,
      "invalid_or_missing_count": 0,
      "score": 1.0
    }
  }
}
```

Scoring semantics:

- `yes` means the summary preserved the asked factual or naturalness signal.
- `no` means the summary did not preserve it.
- invalid or missing evaluator answers are counted as `no` and are also tracked in `invalid_or_missing_count`.
- a category score is `yes_count / total_questions`.
- if a category has zero final questions, its score is `null`.

## Repository Layout

```text
qa-eval/
|-- cfg/config.yaml
|-- docs/prompts/
|-- data/
|-- src/
|   |-- agents/
|   |-- config/
|   |-- models/
|   |-- prompts/
|   |-- providers/
|   |-- services/
|   `-- ui/
`-- coder_docs/
```

Key files:

- [`src/main.py`](src/main.py): Textual app bootstrap
- [`src/ui/screens.py`](src/ui/screens.py): UI screens and pipeline orchestration
- [`src/config/config_manager.py`](src/config/config_manager.py): YAML config loading and persistence
- [`src/models/question_models.py`](src/models/question_models.py): validation models for raw generated question payloads
- [`src/models/evaluation_models.py`](src/models/evaluation_models.py): validation models for evaluator inputs and strict yes/no outputs
- [`src/services/question_pipeline.py`](src/services/question_pipeline.py): post-generation parsing, filtering, renumbering, and reporting
- [`src/services/evaluation_pipeline.py`](src/services/evaluation_pipeline.py): evaluator-response alignment and normalized scoring
- [`src/services/question_processor.py`](src/services/question_processor.py): embedding-backed similarity processing
- [`src/providers/vllm_provider.py`](src/providers/vllm_provider.py): vLLM-compatible LLM client

## Development

Common commands:

```bash
uv sync --dev
uv run python src/main.py
uv run ruff check .
uv run pytest
```

Current development notes:

- The repository uses `uv` as the package manager and `ruff` as the linter.
- The repository now includes a pytest regression suite for question parsing and deduplication behavior.
- The repository now includes evaluator parsing and scoring regression coverage.
- The provider abstraction exists, but the implemented runtime path is currently the vLLM provider with a one-time preflight and warmed HTTP client per run.
- If you change code in a way that affects behavior, interfaces, setup, outputs, workflows, or operator expectations, update the relevant Markdown documentation in the same change.
- If you materially change runtime flow, tooling, prompts, configuration semantics, or file ownership, also update [`coder_docs/codebase_guide.md`](coder_docs/codebase_guide.md).

## Further Reading

- [`coder_docs/codebase_guide.md`](coder_docs/codebase_guide.md): architecture, runtime flow, and file ownership
- [`coder_docs/academic_standards.md`](coder_docs/academic_standards.md): methodology and citation expectations for robustness-sensitive logic
- [`coder_docs/ruff.md`](coder_docs/ruff.md): lint workflow
- [`coder_docs/uv_package_manager.md`](coder_docs/uv_package_manager.md): dependency and environment workflow
- [`coder_docs/scrapling.md`](coder_docs/scrapling.md): web research and page-retrieval workflow
