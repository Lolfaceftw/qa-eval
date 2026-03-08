"""Render prompt templates for the application agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from src.config.config_manager import ConfigManager
from src.config.runtime_settings import (
    QUESTION_CATEGORIES,
    get_question_request_minimum,
)


class PromptRenderer:
    """Render Jinja2 prompts loaded from the filesystem."""

    @staticmethod
    def render(template_name: str, **context: Any) -> str:
        """Render the configured template with the supplied context values."""
        config = ConfigManager()
        prompt_path_str = config.get(f"prompts.{template_name}")

        if not prompt_path_str:
            raise ValueError(f"Prompt path for {template_name} not found in config.")

        root = Path(__file__).parent.parent.parent
        prompt_path = (root / prompt_path_str).resolve()

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template file not found: {prompt_path}")

        env = Environment(loader=FileSystemLoader(str(prompt_path.parent)))
        template = env.get_template(prompt_path.name)
        render_context = dict(context)
        if template_name in QUESTION_CATEGORIES:
            render_context.setdefault(
                "minimum_questions",
                get_question_request_minimum(template_name, config),
            )
        return template.render(**render_context)
