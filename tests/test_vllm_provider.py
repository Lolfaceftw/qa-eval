"""Regression tests for the vLLM provider connection lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError

from src.providers.llm_provider import LLMStreamEvent
from src.providers.vllm_provider import (
    VLLMConfigurationError,
    VLLMGenerationError,
    VLLMPreparationError,
    VLLMProvider,
)


class StaticConfig:
    """Return deterministic config values for provider tests."""

    def __init__(self, values: dict[str, object]) -> None:
        """Store flattened dot-delimited config values."""
        self._values = values

    def get(self, key: str, default: object = None) -> object:
        """Return the configured value or the provided default."""
        return self._values.get(key, default)


class FakeAsyncSequence:
    """Iterate over a fixed sequence of async items."""

    def __init__(self, items: list[object]) -> None:
        """Store the items to yield asynchronously."""
        self._items = items
        self._index = 0

    def __aiter__(self) -> "FakeAsyncSequence":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Yield the next async item or stop iteration."""
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class FakeModelsAPI:
    """Expose a fake `models.list()` call for preflight tests."""

    def __init__(
        self,
        model_ids: list[str] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store the fake model listing behavior."""
        self._model_ids = model_ids or []
        self._error = error
        self.calls = 0
        self.timeouts: list[object] = []

    def list(self, *, timeout: object) -> FakeAsyncSequence:
        """Return the configured model list or raise the configured error."""
        self.calls += 1
        self.timeouts.append(timeout)
        if self._error is not None:
            raise self._error
        items = [SimpleNamespace(id=model_id) for model_id in self._model_ids]
        return FakeAsyncSequence(items)


class FakeStream:
    """Yield prebuilt streaming chunks asynchronously."""

    def __init__(self, chunks: list[object]) -> None:
        """Store the chunks that should be yielded."""
        self._chunks = chunks
        self._index = 0

    def __aiter__(self) -> "FakeStream":
        """Return the async iterator."""
        return self

    async def __anext__(self) -> object:
        """Yield the next chunk or stop iteration."""
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class FakeCompletionsAPI:
    """Expose a fake `chat.completions.create()` call."""

    def __init__(
        self,
        chunks: list[object] | None = None,
        error: Exception | None = None,
    ) -> None:
        """Store the fake streaming behavior."""
        self._chunks = chunks or []
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> FakeStream:
        """Return a fake stream or raise the configured error."""
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return FakeStream(self._chunks)


class FakeClient:
    """Provide the minimal OpenAI-compatible surface used by the provider."""

    def __init__(
        self,
        model_ids: list[str] | None = None,
        model_error: Exception | None = None,
        stream_chunks: list[object] | None = None,
        generation_error: Exception | None = None,
    ) -> None:
        """Store fake model-list and generation behavior."""
        self.models = FakeModelsAPI(model_ids=model_ids, error=model_error)
        self.chat = SimpleNamespace(
            completions=FakeCompletionsAPI(
                chunks=stream_chunks,
                error=generation_error,
            )
        )
        self.close_calls = 0

    async def close(self) -> None:
        """Track provider cleanup."""
        self.close_calls += 1


def make_config(**overrides: object) -> StaticConfig:
    """Build a provider config with stable defaults."""
    values: dict[str, object] = {
        "api_keys.vllm": "EMPTY",
        "vllm.base_url": "http://localhost:8000/v1",
        "vllm.model": "Qwen/Qwen3.5-35B-A3B",
        "vllm.max_context": 32768,
        "vllm.connection.preflight_timeout_seconds": 5.0,
        "vllm.connection.connect_timeout_seconds": 5.0,
        "vllm.connection.read_timeout_seconds": 120.0,
        "vllm.connection.pool_timeout_seconds": 5.0,
        "vllm.connection.keepalive_expiry_seconds": 30.0,
        "vllm.connection.max_retries": 1,
    }
    values.update(overrides)
    return StaticConfig(values)


def make_provider(
    config: StaticConfig,
    client: FakeClient,
    *,
    time_values: list[float] | None = None,
) -> tuple[VLLMProvider, dict[str, object]]:
    """Create a provider with a fake client factory and captured kwargs."""
    factory_kwargs: dict[str, object] = {}
    time_points = iter(time_values or [10.0, 10.25])

    def client_factory(**kwargs: object) -> FakeClient:
        factory_kwargs.update(kwargs)
        return client

    def time_fn() -> float:
        return next(time_points)

    provider = VLLMProvider(
        config=config,
        client_factory=client_factory,
        time_fn=time_fn,
    )
    return provider, factory_kwargs


def make_timeout_error() -> APITimeoutError:
    """Create an OpenAI timeout error instance for test use."""
    request = httpx.Request("GET", "http://localhost:8000/v1/models")
    return APITimeoutError(request=request)


def make_connection_error() -> APIConnectionError:
    """Create an OpenAI connection error instance for test use."""
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    return APIConnectionError(request=request)


@pytest.mark.anyio
async def test_provider_coerces_configured_connection_values() -> None:
    """Parse string-backed config values into the typed client settings."""
    config = make_config(
        **{
            "vllm.max_context": "4096",
            "vllm.connection.preflight_timeout_seconds": "7.5",
            "vllm.connection.connect_timeout_seconds": "2.0",
            "vllm.connection.read_timeout_seconds": "45.0",
            "vllm.connection.pool_timeout_seconds": "3.5",
            "vllm.connection.keepalive_expiry_seconds": "18.0",
            "vllm.connection.max_retries": "2",
        }
    )
    provider, factory_kwargs = make_provider(
        config,
        FakeClient(model_ids=["Qwen/Qwen3.5-35B-A3B"]),
    )

    await provider.prepare()

    request_timeout = factory_kwargs["timeout"]
    assert isinstance(request_timeout, httpx.Timeout)
    assert provider.max_context == 4096
    assert request_timeout.connect == 2.0
    assert request_timeout.read == 45.0
    assert request_timeout.pool == 3.5
    assert factory_kwargs["max_retries"] == 2

    await provider.close()


def test_provider_rejects_invalid_timeout_values() -> None:
    """Fail fast when connection settings cannot be parsed safely."""
    config = make_config(
        **{"vllm.connection.preflight_timeout_seconds": "not-a-number"}
    )

    with pytest.raises(VLLMConfigurationError):
        make_provider(config, FakeClient(model_ids=["Qwen/Qwen3.5-35B-A3B"]))


@pytest.mark.anyio
async def test_prepare_is_idempotent_and_records_connection_status() -> None:
    """Warm the connection only once and expose the measured latency."""
    client = FakeClient(model_ids=["Qwen/Qwen3.5-35B-A3B"])
    provider, _ = make_provider(
        make_config(),
        client,
        time_values=[20.0, 20.4],
    )

    await provider.prepare()
    await provider.prepare()

    assert client.models.calls == 1
    assert provider.connection_status is not None
    assert provider.connection_status.model == "Qwen/Qwen3.5-35B-A3B"
    assert provider.connection_status.latency_seconds == 0.4

    await provider.close()
    assert client.close_calls == 1


@pytest.mark.anyio
async def test_prepare_raises_when_configured_model_is_missing() -> None:
    """Reject endpoints that do not expose the configured model id."""
    provider, _ = make_provider(
        make_config(),
        FakeClient(model_ids=["other-model"]),
    )

    with pytest.raises(VLLMPreparationError, match="Available models: other-model"):
        await provider.prepare()

    await provider.close()


@pytest.mark.anyio
async def test_prepare_raises_timeout_with_actionable_guidance() -> None:
    """Surface timeout errors with the matching config knob in the message."""
    provider, _ = make_provider(
        make_config(),
        FakeClient(model_error=make_timeout_error()),
    )

    with pytest.raises(
        VLLMPreparationError,
        match="preflight_timeout_seconds",
    ):
        await provider.prepare()

    await provider.close()


@pytest.mark.anyio
async def test_generate_stream_yields_reasoning_and_content_events() -> None:
    """Expose reasoning deltas separately from answer content deltas."""
    provider, _ = make_provider(
        make_config(),
        FakeClient(
            model_ids=["Qwen/Qwen3.5-35B-A3B"],
            stream_chunks=[
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="", reasoning=None)
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content=None,
                                reasoning="Thinking...",
                            )
                        )
                    ]
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(
                                content="{",
                                reasoning_content=None,
                            )
                        )
                    ]
                ),
            ],
        ),
    )

    events = [event async for event in provider.generate_stream("Reply with JSON.")]

    assert events == [
        LLMStreamEvent(kind="reasoning", text="Thinking..."),
        LLMStreamEvent(kind="content", text="{"),
    ]

    await provider.close()


@pytest.mark.anyio
async def test_generate_stream_raises_instead_of_yielding_error_text() -> None:
    """Raise a typed generation error when streaming fails mid-request."""
    provider, _ = make_provider(
        make_config(),
        FakeClient(
            model_ids=["Qwen/Qwen3.5-35B-A3B"],
            generation_error=make_connection_error(),
        ),
    )

    with pytest.raises(VLLMGenerationError, match="Lost connection"):
        async for _ in provider.generate_stream("Reply with OK."):
            pass

    await provider.close()
