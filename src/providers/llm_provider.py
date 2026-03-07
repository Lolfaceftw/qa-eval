"""Provider interfaces for LLM-backed question generation."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMProvider(ABC):
    """Define the contract for question-generation providers."""

    @abstractmethod
    async def prepare(self) -> None:
        """Validate and warm the provider before generation begins."""
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generate text from the LLM, yielding tokens as they arrive."""
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
