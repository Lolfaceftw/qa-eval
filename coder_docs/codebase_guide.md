# Codebase Guide

Read this document first in every new coding-agent session for this repository.

This file is the project-local map of the current codebase. Update it whenever the repo's runtime flow, configuration keys, architecture, developer workflow, prompts, provider behavior, or output artifacts change in a meaningful way.

## Purpose

`qa-eval` is a Textual-based TUI for evaluating how well a summary preserves information from a transcript. The current workflow:

1. Loads a transcript JSON file and a summary text file from config.
2. Validates them with Pydantic models.
3. Renders prompt templates for two agent categories: factualness and naturalness.
4. Streams question-generation output from a vLLM-compatible endpoint.
5. Parses the model output into JSON question lists.
6. Deduplicates similar questions with embeddings.
7. Writes the final result to `data/processed_questions.json`.

The app is oriented around information-loss evaluation. The prompts ask for yes/no questions that compare transcript and summary, with the transcript as ground truth.

## Read-First Operational Facts

- Current Python target is 3.12+.
- `uv` is the package manager. Use it for sync, run, add, remove, and lock operations.
- `ruff` is the linter. Current baseline is clean with `uv run ruff check .`.
- The current runtime entrypoint is `uv run python src/main.py`.
- The repository uses `src/...` imports without `__init__.py` files. Run commands from the repo root so imports resolve consistently.
- `README.md` now provides the primary onboarding, setup, and usage overview. Use this guide for architecture, runtime flow, and file ownership details.
- For robustness-sensitive formulas, algorithms, heuristics, or scoring logic, use `coder_docs/academic_standards.md` as the authoritative methodology and citation policy. Avoid uncited "magic formulas" and prefer journal-backed methods first.
- For external web research and page retrieval, use `coder_docs/scrapling.md` as the workflow source of truth and prefer Scrapling MCP as the default retrieval layer.
- Treat Markdown documentation review as part of normal coding passes. When code changes affect behavior, interfaces, setup, outputs, workflows, or operator expectations, update the relevant Markdown files in the same change, including `README.md`, `docs/*.md`, and applicable `coder_docs/*.md`.
- There is no automated test suite in the repo at the moment.

## Runtime Flow

### 1. App bootstrap

- `src/main.py` defines `QAEvalApp`, a `textual.app.App`.
- On startup it reads `ui.css_path` from `cfg/config.yaml`, resolves it relative to the project root, and mounts `MainMenuScreen`.

### 2. UI layer

- `src/ui/screens.py` contains the working TUI.
- `MainMenuScreen` routes to run, config, or exit.
- `ConfigScreen` flattens nested config keys and edits values through the singleton config manager.
- `RunScreen` owns the evaluation pipeline and streams results into a Markdown widget.
- `FileViewScreen` opens linked files from the run output for inspection.

### 3. Config and data loading

- `src/config/config_manager.py` is a singleton wrapper around `cfg/config.yaml`.
- `ConfigManager.get()` reads dot-delimited keys.
- `ConfigManager.set()` writes changes back to disk immediately.
- Current important config areas:
  - `data.*` for transcript and summary paths.
  - `prompts.*` for Jinja template paths.
  - `vllm.*` for endpoint, model, and max context.
  - `embedding.*` for deduplication model, threshold, and device.
- The code also supports `app.provider` for provider selection, but the current `cfg/config.yaml` relies on the default provider fallback of `vllm`.

### 4. Validation and preprocessing

- `src/models/data_models.py` defines:
  - `TranscriptSegment`
  - `Transcript`
  - `Summary`
- `Summary` validates that the content contains speaker tags such as `<SPEAKER_00>`.
- `src/services/data_processor.py` converts the transcript into a speaker-to-utterances mapping and can extract speaker blocks from tagged summaries.

### 5. Prompt rendering and generation

- `src/prompts/templates.py` resolves prompt template paths from config and renders them with Jinja2.
- Prompt templates live under `docs/prompts/`.
- Current agent categories:
  - `factualness`
  - `naturalness`
- `src/providers/llm_provider.py` defines the provider interface.
- `src/providers/vllm_provider.py` is the only implemented provider today. It uses `openai.AsyncOpenAI` against a vLLM-compatible base URL and counts tokens with `tiktoken`.
- `src/agents/agent_factory.py` maps the agent type to the corresponding prompt category and streams provider output back to the UI.

### 6. Parsing, deduplication, and output

- `RunScreen._run_agent()` streams output and expects the model to return a raw JSON array.
- `RunScreen.run_pipeline()` extracts the first JSON array it can parse from each agent response.
- `src/services/question_processor.py` loads a Hugging Face embedding model, generates embeddings, computes cosine similarity, and removes later duplicate questions above the configured threshold.
- Final processed questions are written to `data/processed_questions.json`.

## Important Files And Responsibilities

- `src/main.py`: application bootstrap and CSS path resolution.
- `src/ui/screens.py`: TUI screens, config editing, run orchestration, result rendering.
- `src/config/config_manager.py`: singleton config load/save interface.
- `src/models/data_models.py`: transcript and summary validation models.
- `src/services/data_processor.py`: transcript and summary preprocessing helpers.
- `src/services/question_processor.py`: embedding-based deduplication and logging hook.
- `src/prompts/templates.py`: Jinja prompt loading and rendering.
- `src/providers/llm_provider.py`: abstract provider contract.
- `src/providers/vllm_provider.py`: current concrete LLM provider.
- `src/agents/agent_factory.py`: agent selection and prompt-category mapping.
- `docs/prompts/*.j2`: prompt specs that instruct the model how to produce question arrays.
- `cfg/config.yaml`: runtime configuration source.
- `data/transcript.json`: transcript input artifact.
- `data/summary.txt`: summary input artifact.
- `data/processed_questions.json`: pipeline output artifact.
- `coder_docs/academic_standards.md`: standards for academically backed formulas, algorithms, heuristics, and inline IEEE-style citation requirements.

## Data And Prompt Expectations

- Transcript input is JSON with a top-level `segments` list.
- Each segment currently contains `speaker`, `start_time`, `end_time`, and `text`.
- Summary input is plain text with speaker-tagged blocks.
- Prompt templates currently ask for a minimum of 200 questions per category and require the model output to be a raw JSON array of objects with `question_number` and `question`.
- There is no stronger schema validation on the returned question objects beyond JSON parsing and downstream use of the `question` field during deduplication.

## Current Constraints And Watchouts

- Provider abstraction exists, but only the vLLM path is implemented in practice.
- `ConfigManager` writes config changes immediately, so UI config edits are persistent side effects.
- `QuestionProcessor` loads a large embedding model and can be slow or memory-intensive depending on device availability.
- The question processor disables several Hugging Face and tokenizer progress or warning outputs to avoid corrupting the TUI display.
- The pipeline currently mixes UI orchestration, file IO, model calls, parsing, and result writing inside `RunScreen.run_pipeline()`. Changes there can affect multiple stages at once.
- There are existing TODO comments in the codebase. Treat them as signals of incomplete areas, not as authoritative design decisions.

## Standard Developer Commands

- `uv sync --dev`
- `uv run python src/main.py`
- `uv run ruff check .`

## When This Document Must Be Updated

Update this file in the same change whenever any of the following changes:

- Entry points or command workflow.
- Module ownership or major file moves.
- Config keys, config semantics, or config file locations.
- Prompt categories, prompt locations, or output expectations.
- Provider implementations or provider selection behavior.
- Data file formats, validation rules, or output artifact structure.
- Linting, packaging, or testing workflow that affects daily development.
- Documentation maintenance expectations for code changes or review workflow.
- Academic-backing or citation workflow for methodology-sensitive implementations.
