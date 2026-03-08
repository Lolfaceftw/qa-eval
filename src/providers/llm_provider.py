"""Provider interfaces for LLM-backed question generation."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class LLMStreamEvent:
    """Describe one streamed output fragment from an LLM provider."""

    kind: Literal["content", "reasoning"]
    text: str


class LLMProvider(ABC):
    """Define the contract for question-generation providers."""

    @abstractmethod
    async def prepare(self) -> None:
        """Validate and warm the provider before generation begins."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Generate text from the LLM, yielding typed stream events."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Release any open resources held by the provider."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        pass

    @property
    @abstractmethod
    def max_context(self) -> int:
        """Return the maximum context length for the model."""
        pass
