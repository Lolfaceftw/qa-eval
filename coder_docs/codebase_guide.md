# Codebase Guide

Read this document first in every new coding-agent session for this repository.

This file is the project-local map of the current codebase. Update it whenever the repo's runtime flow, configuration keys, architecture, developer workflow, prompts, provider behavior, output artifacts, or testing expectations change in a meaningful way.

## Purpose

`qa-eval` is a Textual-based TUI for evaluating how well a summary preserves information from a transcript. The current workflow:

1. Loads a transcript JSON file and a summary text file from config.
2. Validates them with Pydantic models.
3. Renders prompt templates for two agent categories: factualness and naturalness.
4. Streams question-generation output from a vLLM-compatible endpoint.
5. Parses and validates the model output into question lists with category metadata.
6. Filters off-rubric questions, canonicalizes boilerplate-heavy phrasing, and deduplicates globally with embeddings.
7. Writes the filtered question set to `data/processed_questions.json` and the filtering report to `data/question_filter_report.json`.
8. Renders evaluator prompts for the surviving per-category question sets and streams strict yes/no answers from the same provider.
9. Writes the per-question answers to `data/question_evaluations.json` and the normalized scoring report to `data/evaluation_report.json`.

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
- The repository now includes a pytest suite for question parsing, deduplication, evaluator alignment, and scoring regressions. Run `uv run pytest` from the repo root.
- When implementation changes affect tested behavior, interfaces, contracts, or output artifacts, update or add the relevant tests in the same change.

## Runtime Flow

### 1. App bootstrap

- `src/main.py` defines `QAEvalApp`, a `textual.app.App`.
- On startup it reads `ui.css_path` from `cfg/config.yaml`, resolves it relative to the project root, and mounts `MainMenuScreen`.

### 2. UI layer

- `src/ui/screens.py` contains the working TUI.
- `MainMenuScreen` routes to run, config, or exit.
- `ConfigScreen` flattens nested config keys and edits values through the singleton config manager.
- `RunScreen` owns streaming and file IO, then delegates question parsing/filtering to `QuestionPipeline` and answer alignment/scoring to `EvaluationPipeline`.
- `FileViewScreen` opens linked files from the run output for inspection.

### 3. Config and data loading

- `src/config/config_manager.py` is a singleton wrapper around `cfg/config.yaml`.
- `ConfigManager.get()` reads dot-delimited keys.
- `ConfigManager.set()` writes changes back to disk immediately.
- Current important config areas:
  - `data.*` for transcript and summary paths.
  - `prompts.*` for Jinja template paths.
  - `vllm.*` for endpoint, model, max context, and connection tuning.
  - `embedding.*` for deduplication model, threshold, and device.
- `prompts.evaluator` points at the strict yes/no evaluator template used after question filtering.
- The code also supports `app.provider` for provider selection, but the current `cfg/config.yaml` relies on the default provider fallback of `vllm`.
- The repo tracks `data/transcript.json` and `data/summary.txt` as inputs, while other `data/` artifacts are local outputs and should stay ignored.

### 4. Validation and preprocessing

- `src/models/data_models.py` defines:
  - `TranscriptSegment`
  - `Transcript`
  - `Summary`
- `Summary` validates that the content contains speaker tags such as `<SPEAKER_00>`.
- `src/services/data_processor.py` converts the transcript into a speaker-to-utterances mapping and can extract speaker blocks from tagged summaries.
- `src/models/question_models.py` validates raw question payloads returned by the LLM before they reach deduplication.
- `src/models/evaluation_models.py` validates processed questions before evaluation and strict yes/no evaluator payloads after generation.

### 5. Prompt rendering and generation

- `src/prompts/templates.py` resolves prompt template paths from config and renders them with Jinja2.
- Prompt templates live under `docs/prompts/`.
- Current agent categories:
  - `factualness`
  - `naturalness`
- `docs/prompts/evaluator.j2` evaluates the final question sets and expects `question_number` plus a strict `yes` or `no` answer for each item.
- Prompt rendering now injects a shared minimum-question target through `QUESTION_REQUEST_MINIMUMS`.
- The question-generation contract requires the LLM to return `question_number`, `dimension`, and `question` for each item.
- Question-generation prompts are now positively keyed so `yes` means the summary preserved the targeted factual or naturalness signal.
- `src/providers/llm_provider.py` defines the provider lifecycle and generation interface.
- `src/providers/vllm_provider.py` is the only implemented provider today. It uses `openai.AsyncOpenAI` against a vLLM-compatible base URL, performs a one-time `models.list()` preflight, reuses a tuned `httpx.AsyncClient`, and counts tokens with `tiktoken`.
- `src/agents/agent_factory.py` maps question-generation categories plus the evaluator stage to prompt rendering and shared streamed provider output.

### 6. Parsing, filtering, deduplication, evaluation, and output

- `RunScreen._run_agent()` streams output and expects the model to return a raw JSON array.
- `RunScreen._run_evaluator()` streams evaluator output and expects a raw JSON array of `question_number` and `answer`.
- `RunScreen.run_pipeline()` prepares the provider once before question generation, reuses it for evaluation, reports the validated endpoint/model/latency in the UI, and closes the shared client in a `finally` block.
- `src/services/question_pipeline.py` extracts the first JSON array it can decode, validates each item with Pydantic, canonicalizes boilerplate-heavy phrasing, filters off-rubric naturalness questions, deduplicates exact matches globally, then performs global semantic deduplication.
- Semantic deduplication uses `src/services/question_processor.py`, which loads a Hugging Face embedding model, computes attention-mask-aware pooled sentence embeddings, normalizes them, and clusters near-duplicates with cosine similarity.
- Representative selection is deterministic: `factualness` wins over `naturalness`, then richer canonical questions win, then earlier order wins.
- Final processed questions are written to `data/processed_questions.json`.
- Filtering metrics and shortfalls are written to `data/question_filter_report.json`.
- `src/services/evaluation_pipeline.py` aligns evaluator answers by `question_number`, keeps the first valid answer per known question, coerces invalid or missing answers to `no`, and computes normalized per-category scores.
- Question evaluations are written to `data/question_evaluations.json`.
- Scoring metrics are written to `data/evaluation_report.json`.

## Important Files And Responsibilities

- `src/main.py`: application bootstrap and CSS path resolution.
- `src/ui/screens.py`: TUI screens, config editing, run orchestration, result rendering, and artifact writes.
- `src/config/config_manager.py`: singleton config load/save interface.
- `src/models/data_models.py`: transcript and summary validation models.
- `src/models/question_models.py`: validation models for generated question payloads.
- `src/models/evaluation_models.py`: validation models for evaluator inputs and strict yes/no outputs.
- `src/services/data_processor.py`: transcript and summary preprocessing helpers.
- `src/services/question_pipeline.py`: parsing, category-fit filtering, exact deduplication, semantic deduplication, renumbering, and report generation.
- `src/services/evaluation_pipeline.py`: evaluator-response parsing, answer alignment, invalid-answer coercion, and score reporting.
- `src/services/question_processor.py`: embedding loading, pooling, cosine similarity, and connected-component clustering.
- `src/prompts/templates.py`: Jinja prompt loading, rendering, and shared minimum-question targets for question generation plus evaluator context rendering.
- `src/providers/llm_provider.py`: abstract provider contract.
- `src/providers/vllm_provider.py`: current concrete LLM provider.
- `src/agents/agent_factory.py`: question-generation and evaluator agent selection plus streamed prompt execution.
- `docs/prompts/*.j2`: prompt specs for question generation and strict evaluator yes/no output.
- `cfg/config.yaml`: runtime configuration source.
- `data/transcript.json`: transcript input artifact.
- `data/summary.txt`: summary input artifact.
- `data/processed_questions.json`: filtered question output artifact.
- `data/question_filter_report.json`: filtering metrics and shortfall artifact.
- `data/question_evaluations.json`: per-question yes/no evaluation artifact.
- `data/evaluation_report.json`: normalized scoring artifact.
- `coder_docs/academic_standards.md`: standards for academically backed formulas, algorithms, heuristics, and inline IEEE-style citation requirements.

## Data And Prompt Expectations

- Transcript input is JSON with a top-level `segments` list.
- Each segment currently contains `speaker`, `start_time`, `end_time`, and `text`.
- Summary input is plain text with speaker-tagged blocks.
- Prompt templates currently ask for a minimum of 200 questions per category.
- Question-generation output must be a raw JSON array of objects with `question_number`, `dimension`, and `question`.
- Evaluator output must be a raw JSON array of objects with `question_number` and `answer`, where `answer` is `yes` or `no`.
- `factualness` questions must stay on omitted or altered content.
- `naturalness` questions must stay on tone, flow, pacing, voice, transitions, hedging, emphasis, or speaker personality.
- Generated questions should be phrased so a `yes` answer means the summary preserved the targeted signal.
- The public output artifact strips internal metadata and keeps only `question_number` and `question`.
- The evaluation artifact preserves `question_number`, `question`, and final `answer` per category.
- Category scores are normalized as `yes_count / total_questions`; invalid or missing evaluator answers count as `no`, and zero-question categories report `null`.

## Current Constraints And Watchouts

- Provider abstraction exists, but only the vLLM path is implemented in practice.
- `ConfigManager` writes config changes immediately, so UI config edits are persistent side effects.
- The TUI config editor persists values as strings; `VLLMProvider` now coerces its numeric connection settings locally so timeout and retry values remain usable after in-app edits.
- `QuestionProcessor` loads a large embedding model and can be slow or memory-intensive depending on device availability.
- The question processor disables several Hugging Face and tokenizer progress or warning outputs to avoid corrupting the TUI display.
- `QuestionPipeline` uses a small canonicalization layer for boilerplate yes/no wrappers plus a narrow marker fallback when the model omits `dimension`. Keep that heuristic surface small and prefer prompt/schema improvements over expanding token lists.
- `EvaluationPipeline` treats malformed or missing evaluator answers as `no`, so evaluation remains total-order deterministic even when the model under-produces.
- The filtered question artifact is intentionally stable for downstream consumers, so evaluator metadata stays in the evaluation artifacts instead of changing `data/processed_questions.json`.

## Standard Developer Commands

- `uv sync --dev`
- `uv run python src/main.py`
- `uv run ruff check .`
- `uv run pytest`

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
