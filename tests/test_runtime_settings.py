"""Regression tests for runtime configuration coercion."""

from __future__ import annotations

from src.config.runtime_settings import (
    get_question_request_minimums,
    is_embedding_enabled,
)


class FakeConfig:
    """Provide dictionary-backed config values for runtime-setting tests."""

    def __init__(self, values: dict[str, object]) -> None:
        """Store flattened config values."""
        self._values = values

    def get(self, key: str, default: object = None) -> object:
        """Return the configured value or the supplied default."""
        return self._values.get(key, default)


def test_get_question_request_minimums_uses_global_and_category_values() -> None:
    """Read configurable minimums and coerce string values from the TUI."""
    config = FakeConfig(
        {
            "questions.minimum": "150",
            "questions.minimums.naturalness": "90",
        }
    )

    minimums = get_question_request_minimums(config)

    assert minimums == {
        "factualness": 150,
        "naturalness": 90,
    }


def test_is_embedding_enabled_coerces_boolean_like_strings() -> None:
    """Interpret string-backed config values from the editor as booleans."""
    assert is_embedding_enabled(FakeConfig({"embedding.enabled": "false"})) is False
    assert is_embedding_enabled(FakeConfig({"embedding.enabled": "true"})) is True
