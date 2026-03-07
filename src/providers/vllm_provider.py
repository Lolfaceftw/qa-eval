"""vLLM-compatible provider implementation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Protocol

import httpx
import tiktoken
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
)

from src.config.config_manager import ConfigManager
from src.providers.llm_provider import LLMProvider


class _ConfigSource(Protocol):
    """Describe the config access required by the provider."""

    def get(self, key: str, default: object = None) -> object:
        """Return a configuration value by dot-delimited key."""


@dataclass(frozen=True, slots=True)
class VLLMConnectionStatus:
    """Describe a successful provider preflight."""

    base_url: str
    model: str
    latency_seconds: float


class VLLMProviderError(RuntimeError):
    """Base error raised for vLLM provider failures."""


class VLLMConfigurationError(VLLMProviderError):
    """Raise when vLLM configuration cannot be parsed safely."""


class VLLMPreparationError(VLLMProviderError):
    """Raise when vLLM preflight or warmup fails."""


class VLLMGenerationError(VLLMProviderError):
    """Raise when a streaming generation request fails."""


class VLLMProvider(LLMProvider):
    """Connect to a vLLM-compatible endpoint through the OpenAI async client."""

    def __init__(
        self,
        config: _ConfigSource | None = None,
        client_factory: Callable[..., AsyncOpenAI] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        """Load config, build the HTTP client, and prepare token counting."""
        self._config = config or ConfigManager()
        self._client_factory = client_factory or AsyncOpenAI
        self._time_fn = time_fn or time.perf_counter

        api_key = self._read_string("api_keys.vllm", "EMPTY")
        self.base_url = self._read_string(
            "vllm.base_url",
            "http://localhost:8000/v1",
        )
        self.model = self._read_string("vllm.model", "qwen2.5")
        self._max_context_value = self._read_positive_int(
            "vllm.max_context",
            32768,
        )
        self._preflight_timeout_seconds = self._read_positive_float(
            "vllm.connection.preflight_timeout_seconds",
            5.0,
        )
        self._connect_timeout_seconds = self._read_positive_float(
            "vllm.connection.connect_timeout_seconds",
            5.0,
        )
        self._read_timeout_seconds = self._read_positive_float(
            "vllm.connection.read_timeout_seconds",
            120.0,
        )
        self._pool_timeout_seconds = self._read_positive_float(
            "vllm.connection.pool_timeout_seconds",
            5.0,
        )
        self._keepalive_expiry_seconds = self._read_positive_float(
            "vllm.connection.keepalive_expiry_seconds",
            30.0,
        )
        self._max_retries = self._read_non_negative_int(
            "vllm.connection.max_retries",
            1,
        )

        self._request_timeout = httpx.Timeout(
            connect=self._connect_timeout_seconds,
            read=self._read_timeout_seconds,
            write=self._read_timeout_seconds,
            pool=self._pool_timeout_seconds,
        )
        self._preflight_timeout = httpx.Timeout(
            connect=self._connect_timeout_seconds,
            read=self._preflight_timeout_seconds,
            write=self._preflight_timeout_seconds,
            pool=self._pool_timeout_seconds,
        )
        self._http_client = httpx.AsyncClient(
            timeout=self._request_timeout,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=self._keepalive_expiry_seconds,
            ),
        )
        self.client = self._client_factory(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self._request_timeout,
            max_retries=self._max_retries,
            http_client=self._http_client,
        )
        self._prepare_lock = asyncio.Lock()
        self._connection_status: VLLMConnectionStatus | None = None
        self._closed = False

        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def connection_status(self) -> VLLMConnectionStatus | None:
        """Return the last successful connection preflight status."""
        return self._connection_status

    async def prepare(self) -> None:
        """Validate connectivity and warm the shared HTTP connection pool."""
        if self._closed:
            raise VLLMPreparationError("Cannot prepare a closed VLLM provider.")
        if self._connection_status is not None:
            return

        async with self._prepare_lock:
            if self._connection_status is not None:
                return

            started_at = self._time_fn()
            try:
                available_models = await self._list_models()
            except APITimeoutError as exc:
                raise VLLMPreparationError(
                    "Timed out while validating the vLLM endpoint at "
                    f"{self.base_url}. Increase "
                    "`vllm.connection.preflight_timeout_seconds` if the "
                    "endpoint is healthy but slow."
                ) from exc
            except APIConnectionError as exc:
                raise VLLMPreparationError(
                    "Could not connect to the vLLM endpoint at "
                    f"{self.base_url}. Check `vllm.base_url` and network "
                    "reachability."
                ) from exc
            except APIStatusError as exc:
                raise VLLMPreparationError(
                    "The vLLM endpoint at "
                    f"{self.base_url} returned HTTP {exc.status_code} during "
                    "preflight."
                ) from exc
            except OpenAIError as exc:
                raise VLLMPreparationError(
                    f"vLLM preflight failed for {self.base_url}: {exc}"
                ) from exc

            if not available_models:
                raise VLLMPreparationError(
                    f"The vLLM endpoint at {self.base_url} returned no models."
                )
            if self.model not in available_models:
                available = ", ".join(sorted(available_models))
                raise VLLMPreparationError(
                    "Configured model "
                    f"`{self.model}` was not returned by {self.base_url}. "
                    f"Available models: {available}."
                )

            self._connection_status = VLLMConnectionStatus(
                base_url=self.base_url,
                model=self.model,
                latency_seconds=round(self._time_fn() - started_at, 3),
            )

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream tokens from the configured vLLM model."""
        if self._closed:
            raise VLLMGenerationError("Cannot generate with a closed VLLM provider.")

        await self.prepare()
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                temperature=0.7,
                timeout=self._request_timeout,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except APITimeoutError as exc:
            raise VLLMGenerationError(
                "Timed out while waiting for streamed output from "
                f"{self.base_url} using model `{self.model}`. Increase "
                "`vllm.connection.read_timeout_seconds` if this endpoint is "
                "expected to respond slowly."
            ) from exc
        except APIConnectionError as exc:
            raise VLLMGenerationError(
                "Lost connection to the vLLM endpoint at "
                f"{self.base_url} while generating with model `{self.model}`."
            ) from exc
        except APIStatusError as exc:
            raise VLLMGenerationError(
                "The vLLM endpoint at "
                f"{self.base_url} returned HTTP {exc.status_code} while "
                f"generating with model `{self.model}`."
            ) from exc
        except OpenAIError as exc:
            raise VLLMGenerationError(
                f"vLLM generation failed for model `{self.model}`: {exc}"
            ) from exc

    async def close(self) -> None:
        """Close the shared OpenAI client and its HTTP transport."""
        if self._closed:
            return
        try:
            await self.client.close()
        finally:
            await self._http_client.aclose()
            self._closed = True

    def count_tokens(self, text: str) -> int:
        """Count tokens using the configured model tokenizer when available."""
        return len(self.encoding.encode(text))

    @property
    def max_context(self) -> int:
        """Return the configured maximum context length for the model."""
        return self._max_context_value

    async def _list_models(self) -> set[str]:
        """Fetch the set of models exposed by the configured endpoint."""
        model_ids: set[str] = set()
        async for model in self.client.models.list(timeout=self._preflight_timeout):
            model_id = getattr(model, "id", None)
            if model_id:
                model_ids.add(str(model_id))
        return model_ids

    def _read_string(self, key: str, default: str) -> str:
        """Return a non-empty string configuration value."""
        raw_value = self._config.get(key, default)
        if raw_value is None:
            return default
        value = str(raw_value).strip()
        if not value:
            raise VLLMConfigurationError(
                f"Configuration value `{key}` must not be empty."
            )
        return value

    def _read_positive_float(self, key: str, default: float) -> float:
        """Parse a positive float from config."""
        raw_value = self._config.get(key, default)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise VLLMConfigurationError(
                f"Configuration value `{key}` must be a positive number."
            ) from exc
        if value <= 0:
            raise VLLMConfigurationError(
                f"Configuration value `{key}` must be greater than zero."
            )
        return value

    def _read_positive_int(self, key: str, default: int) -> int:
        """Parse a positive integer from config."""
        raw_value = self._config.get(key, default)
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise VLLMConfigurationError(
                f"Configuration value `{key}` must be a positive integer."
            ) from exc
        if value <= 0:
            raise VLLMConfigurationError(
                f"Configuration value `{key}` must be greater than zero."
            )
        return value

    def _read_non_negative_int(self, key: str, default: int) -> int:
        """Parse a non-negative integer from config."""
        raw_value = self._config.get(key, default)
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise VLLMConfigurationError(
                f"Configuration value `{key}` must be a non-negative integer."
            ) from exc
        if value < 0:
            raise VLLMConfigurationError(
                f"Configuration value `{key}` must not be negative."
            )
        return value
