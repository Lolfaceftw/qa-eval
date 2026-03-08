"""Create configured LLM providers."""

from src.providers.llm_provider import LLMProvider
from src.providers.vllm_provider import VLLMProvider


class ProviderFactory:
    """Create language-model providers from config values."""

    @staticmethod
    def create_provider(provider_type: str = "vllm") -> LLMProvider:
        """Return the configured provider implementation."""
        if provider_type.lower() == "vllm":
            return VLLMProvider()
        raise ValueError(f"Unknown provider type: {provider_type}")
