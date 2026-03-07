"""Define the agent types used in the evaluation workflow."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import AsyncGenerator

from src.providers.llm_provider import LLMProvider
from src.prompts.templates import PromptRenderer

PROMPT_TOKENS_PREFIX = "PROMPT_TOKENS:"
STREAM_STATUS_PREFIX = "STREAM_STATUS:"
FIRST_TOKEN_LATENCY_PREFIX = "FIRST_TOKEN_LATENCY:"
TOTAL_STATS_PREFIX = "TOTAL_STATS:"


class BaseAgent(ABC):
    """Provide common prompt streaming behavior for agent types."""

    def __init__(self, provider: LLMProvider):
        """Store the shared LLM provider."""
        self.provider = provider

    async def _stream_prompt(self, prompt: str) -> AsyncGenerator[str, None]:
        """Yield the prompt statistics and streamed model response."""
        prompt_tokens = self.provider.count_tokens(prompt)
        yield f"{PROMPT_TOKENS_PREFIX}{prompt_tokens}\n"

        request_started_at = time.perf_counter()
        yield f"{STREAM_STATUS_PREFIX}REQUEST_SUBMITTED\n"

        response_content = ""
        received_first_chunk = False
        async for chunk in self.provider.generate_stream(prompt):
            if not received_first_chunk:
                first_token_latency = time.perf_counter() - request_started_at
                yield f"{FIRST_TOKEN_LATENCY_PREFIX}{first_token_latency:.3f}\n"
                received_first_chunk = True
            response_content += chunk
            yield chunk

        response_tokens = self.provider.count_tokens(response_content)
        total_tokens = prompt_tokens + response_tokens
        max_ctx = self.provider.max_context

        yield f"\n{TOTAL_STATS_PREFIX}{total_tokens}/{max_ctx}\n"


class QuestionGenerationAgent(BaseAgent, ABC):
    """Render a generation prompt for one question category."""

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the configured prompt key for this generation agent."""

    async def generate_questions_stream(
        self,
        transcript_text: str,
    ) -> AsyncGenerator[str, None]:
        """Render and stream the question-generation prompt."""
        prompt = PromptRenderer.render(
            self.agent_type,
            transcript=transcript_text,
        )
        async for chunk in self._stream_prompt(prompt):
            yield chunk


class FactualnessAgent(QuestionGenerationAgent):
    """Generate transcript-grounded factualness questions."""

    @property
    def agent_type(self) -> str:
        """Return the factualness prompt key."""
        return "factualness"


class NaturalnessAgent(QuestionGenerationAgent):
    """Generate transcript-grounded naturalness questions."""

    @property
    def agent_type(self) -> str:
        """Return the naturalness prompt key."""
        return "naturalness"


class EvaluatorAgent(BaseAgent):
    """Evaluate processed questions against the transcript and summary."""

    async def generate_evaluation_stream(
        self,
        category: str,
        transcript_text: str,
        summary_text: str,
        questions: list[Mapping[str, object]],
    ) -> AsyncGenerator[str, None]:
        """Render and stream the evaluator prompt for one category."""
        prompt = PromptRenderer.render(
            "evaluator",
            category=category,
            transcript=transcript_text,
            summary=summary_text,
            questions_json=json.dumps(questions, indent=2),
        )
        async for chunk in self._stream_prompt(prompt):
            yield chunk


class AgentFactory:
    """Create application agents from runtime configuration."""

    @staticmethod
    def create_agent(
        agent_type: str,
        provider: LLMProvider,
    ) -> QuestionGenerationAgent:
        """Return the requested question-generation agent."""
        if agent_type.lower() == "factualness":
            return FactualnessAgent(provider)
        if agent_type.lower() == "naturalness":
            return NaturalnessAgent(provider)
        raise ValueError(f"Unknown agent type: {agent_type}")

    @staticmethod
    def create_evaluator(provider: LLMProvider) -> EvaluatorAgent:
        """Return the evaluator agent."""
        return EvaluatorAgent(provider)
