from typing import AsyncGenerator
from openai import AsyncOpenAI
from src.providers.llm_provider import LLMProvider
from src.config.config_manager import ConfigManager
import tiktoken


class VLLMProvider(LLMProvider):
    """Concrete Strategy for VLLM using the AsyncOpenAI compatible client."""

    def __init__(self):
        config = ConfigManager()
        api_key = config.get("api_keys.vllm", "EMPTY")
        base_url = config.get("vllm.base_url", "http://localhost:8000/v1")
        self.model = config.get("vllm.model", "qwen2.5")

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Streams the response from VLLM asynchronously."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.7,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"\n[Error connecting to VLLM: {str(e)}]\n"

    def count_tokens(self, text: str) -> int:
        """Counts the number of tokens in the given text using tiktoken."""
        return len(self.encoding.encode(text))

    @property
    def max_context(self) -> int:
        """Returns the maximum context length for the model.
        Note: Qwen3.5-35B-A3B usually has a large context,
        defaulting to 32768 if not specified in config.
        """
        config = ConfigManager()
        return config.get("vllm.max_context", 32768)
