from abc import ABC, abstractmethod
from typing import AsyncGenerator

from src.providers.llm_provider import LLMProvider
from src.prompts.templates import PromptRenderer


class BaseAgent(ABC):
    """Abstract Base Agent for question generation."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Type of the agent (e.g., 'factualness', 'naturalness')."""
        pass

    async def generate_questions_stream(
        self, transcript_text: str, summary_text: str
    ) -> AsyncGenerator[str, None]:
        """Generates the prompt and yields the streamed response from the provider."""
        prompt = PromptRenderer.render(self.agent_type, transcript_text, summary_text)

        prompt_tokens = self.provider.count_tokens(prompt)
        # Special signal for UI to print prompt tokens
        yield f"PROMPT_TOKENS:{prompt_tokens}\n"

        response_content = ""
        async for chunk in self.provider.generate_stream(prompt):
            response_content += chunk
            yield chunk

        response_tokens = self.provider.count_tokens(response_content)
        total_tokens = prompt_tokens + response_tokens
        max_ctx = self.provider.max_context

        # Special signal for UI to print total stats
        yield f"\nTOTAL_STATS:{total_tokens}/{max_ctx}\n"


class FactualnessAgent(BaseAgent):
    """Agent responsible for factualness questions."""

    @property
    def agent_type(self) -> str:
        return "factualness"


class NaturalnessAgent(BaseAgent):
    """Agent responsible for naturalness questions."""

    @property
    def agent_type(self) -> str:
        return "naturalness"


class AgentFactory:
    """Factory Pattern to create agents."""

    @staticmethod
    def create_agent(agent_type: str, provider: LLMProvider) -> BaseAgent:
        if agent_type.lower() == "factualness":
            return FactualnessAgent(provider)
        elif agent_type.lower() == "naturalness":
            return NaturalnessAgent(provider)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
