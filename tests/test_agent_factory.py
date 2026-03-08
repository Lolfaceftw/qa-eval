"""Regression tests for agent stream lifecycle events."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from src.agents.agent_factory import (
    AgentFactory,
    FIRST_VISIBLE_CHUNK_LATENCY_PREFIX,
    PROMPT_TOKENS_PREFIX,
    REASONING_CHUNK_PREFIX,
    STREAM_STATUS_PREFIX,
    TOTAL_STATS_PREFIX,
    FactualnessAgent,
)
from src.providers.llm_provider import LLMStreamEvent


class FakeProvider:
    """Return deterministic token counts and streamed chunks."""

    def __init__(self) -> None:
        """Store the last prompt passed to the fake provider."""
        self.last_prompt: str | None = None

    async def prepare(self) -> None:
        """Satisfy the provider interface."""

    async def generate_stream(
        self,
        prompt: str,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Yield reasoning text before the JSON payload."""
        self.last_prompt = prompt
        yield LLMStreamEvent(kind="reasoning", text="Thinking...")
        yield LLMStreamEvent(kind="content", text="[")
        yield LLMStreamEvent(kind="content", text="]")

    async def close(self) -> None:
        """Satisfy the provider interface."""

    def count_tokens(self, text: str) -> int:
        """Return a deterministic token count."""
        return len(text)

    @property
    def max_context(self) -> int:
        """Return a fixed context window."""
        return 4096


@pytest.mark.anyio
async def test_question_agent_emits_request_status_before_first_visible_chunk() -> None:
    """Emit lifecycle markers and reasoning chunks before the answer body."""
    provider = FakeProvider()
    agent = FactualnessAgent(provider)

    chunks = []
    async for chunk in agent.generate_questions_stream(
        transcript_text='{"segments": 1}',
    ):
        chunks.append(chunk)

    assert chunks[0].startswith(PROMPT_TOKENS_PREFIX)
    assert chunks[1] == f"{STREAM_STATUS_PREFIX}REQUEST_SUBMITTED\n"
    assert chunks[2].startswith(FIRST_VISIBLE_CHUNK_LATENCY_PREFIX)
    assert chunks[3].startswith(REASONING_CHUNK_PREFIX)
    assert chunks[4:6] == ["[", "]"]
    assert chunks[-1].startswith(f"\n{TOTAL_STATS_PREFIX}")
    assert provider.last_prompt is not None
    assert 'Transcript:\n{"segments": 1}' in provider.last_prompt
    assert "Summary:" not in provider.last_prompt
    assert "Vary openings across the full set." not in provider.last_prompt


@pytest.mark.anyio
async def test_naturalness_prompt_does_not_force_opening_variation() -> None:
    """Avoid prompt wording that can trigger repetitive reasoning loops."""
    provider = FakeProvider()
    agent = AgentFactory.create_agent("naturalness", provider)

    chunks = []
    async for chunk in agent.generate_questions_stream(
        transcript_text='{"segments": 1}',
    ):
        chunks.append(chunk)

    assert chunks
    assert provider.last_prompt is not None
    assert "Vary openings across the full set." not in provider.last_prompt
