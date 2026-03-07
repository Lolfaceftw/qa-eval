"""Read runtime configuration values with local type coercion."""

from __future__ import annotations

from typing import Any

from src.config.config_manager import ConfigManager

QUESTION_CATEGORIES: tuple[str, ...] = (
    "factualness",
    "naturalness",
)
DEFAULT_QUESTION_REQUEST_MINIMUM = 200
DEFAULT_EMBEDDING_ENABLED = True
_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Convert config values into booleans when possible."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return default


def _coerce_non_negative_int(value: Any, *, default: int) -> int:
    """Convert config values into non-negative integers when possible."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value if value >= 0 else default
    if isinstance(value, float):
        if value.is_integer() and value >= 0:
            return int(value)
        return default
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return default
        try:
            parsed_value = int(normalized)
        except ValueError:
            return default
        return parsed_value if parsed_value >= 0 else default
    return default


def get_question_request_minimum(
    category: str,
    config: ConfigManager | None = None,
) -> int:
    """Return the configured minimum number of questions for a category."""
    resolved_config = config or ConfigManager()
    default_minimum = _coerce_non_negative_int(
        resolved_config.get("questions.minimum", DEFAULT_QUESTION_REQUEST_MINIMUM),
        default=DEFAULT_QUESTION_REQUEST_MINIMUM,
    )
    configured_minimum = resolved_config.get(f"questions.minimums.{category}")
    return _coerce_non_negative_int(configured_minimum, default=default_minimum)


def get_question_request_minimums(
    config: ConfigManager | None = None,
) -> dict[str, int]:
    """Return configured question minimums for each generation category."""
    resolved_config = config or ConfigManager()
    return {
        category: get_question_request_minimum(category, resolved_config)
        for category in QUESTION_CATEGORIES
    }


def is_embedding_enabled(config: ConfigManager | None = None) -> bool:
    """Return whether embedding-backed semantic deduplication should run."""
    resolved_config = config or ConfigManager()
    return _coerce_bool(
        resolved_config.get("embedding.enabled", DEFAULT_EMBEDDING_ENABLED),
        default=DEFAULT_EMBEDDING_ENABLED,
    )
