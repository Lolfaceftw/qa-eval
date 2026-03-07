# qa-eval

`qa-eval` is a Textual-based terminal UI for evaluating how much information a summary loses relative to a transcript. It generates large sets of yes/no review questions from a vLLM-compatible model, deduplicates them with embeddings, and writes the final question set to a JSON artifact for downstream evaluation.

## What It Does

The current pipeline:

1. Loads a transcript JSON file and a summary text file from configuration.
2. Validates both inputs with Pydantic models.
3. Renders prompt templates for `factualness` and `naturalness`.
4. Streams question generation from a vLLM-compatible endpoint.
5. Parses each model response as a raw JSON array of question objects.
6. Deduplicates similar questions with embeddings.
7. Writes the final result to [`data/processed_questions.json`](data/processed_questions.json).

The project is aimed at information-loss evaluation. The transcript is treated as ground truth, and the generated questions focus on what the summary omitted or failed to preserve.

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
- vLLM base URL and model
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
  factualness: docs/prompts/factualness.j2
  naturalness: docs/prompts/naturalness.j2

ui:
  css_path: src/ui/css/style.tcss

vllm:
  base_url: http://localhost:8000/v1
  model: your-model-id
  max_context: 32768

embedding:
  model: Qwen/Qwen3-Embedding-4B
  threshold: 0.85
  device: null
```

Notes:

- `app.provider` currently defaults to `vllm` when omitted.
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
- focus on information loss relative to the transcript
- expect the model to return a raw JSON array

The final output artifact is written to [`data/processed_questions.json`](data/processed_questions.json) and has this shape:

```json
{
  "factualness": [
    {
      "question_number": 1,
      "question": "Is a key fact from the transcript missing from the summary?"
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
- [`src/services/question_processor.py`](src/services/question_processor.py): embedding-based question deduplication
- [`src/providers/vllm_provider.py`](src/providers/vllm_provider.py): vLLM-compatible LLM client

## Development

Common commands:

```bash
uv sync --dev
uv run python src/main.py
uv run ruff check .
```

Current development notes:

- The repository uses `uv` as the package manager and `ruff` as the linter.
- There is no automated test suite in the repository yet.
- The provider abstraction exists, but the implemented runtime path is currently the vLLM provider.
- If you change code in a way that affects behavior, interfaces, setup, outputs, workflows, or operator expectations, update the relevant Markdown documentation in the same change.
- If you materially change runtime flow, tooling, prompts, configuration semantics, or file ownership, also update [`coder_docs/codebase_guide.md`](coder_docs/codebase_guide.md).

## Further Reading

- [`coder_docs/codebase_guide.md`](coder_docs/codebase_guide.md): architecture, runtime flow, and file ownership
- [`coder_docs/academic_standards.md`](coder_docs/academic_standards.md): methodology and citation expectations for robustness-sensitive logic
- [`coder_docs/ruff.md`](coder_docs/ruff.md): lint workflow
- [`coder_docs/uv_package_manager.md`](coder_docs/uv_package_manager.md): dependency and environment workflow
- [`coder_docs/scrapling.md`](coder_docs/scrapling.md): web research and page-retrieval workflow
