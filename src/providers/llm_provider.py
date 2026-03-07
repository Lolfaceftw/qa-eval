from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMProvider(ABC):
    """Abstract Strategy for LLM Generation."""

    @abstractmethod
    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Generates text from the LLM, yielding tokens as they arrive."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in the given text."""
        pass

    @property
    @abstractmethod
    def max_context(self) -> int:
        """Returns the maximum context length for the model."""
        pass
