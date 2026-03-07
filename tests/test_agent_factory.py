"""Regression tests for agent stream lifecycle events."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from src.agents.agent_factory import (
    FIRST_TOKEN_LATENCY_PREFIX,
    PROMPT_TOKENS_PREFIX,
    STREAM_STATUS_PREFIX,
    TOTAL_STATS_PREFIX,
    FactualnessAgent,
)


class FakeProvider:
    """Return deterministic token counts and streamed chunks."""

    async def prepare(self) -> None:
        """Satisfy the provider interface."""

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Yield a fixed two-chunk response."""
        del prompt
        yield "["
        yield "]"

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
async def test_question_agent_emits_request_status_before_first_chunk() -> None:
    """Emit lifecycle markers so the UI can explain the pre-stream delay."""
    agent = FactualnessAgent(FakeProvider())

    chunks = []
    async for chunk in agent.generate_questions_stream(
        transcript_text='{"segments": 1}',
        summary_text="<SPEAKER_00>summary</SPEAKER_00>",
    ):
        chunks.append(chunk)

    assert chunks[0].startswith(PROMPT_TOKENS_PREFIX)
    assert chunks[1] == f"{STREAM_STATUS_PREFIX}REQUEST_SUBMITTED\n"
    assert chunks[2].startswith(FIRST_TOKEN_LATENCY_PREFIX)
    assert chunks[3:5] == ["[", "]"]
    assert chunks[-1].startswith(f"\n{TOTAL_STATS_PREFIX}")
