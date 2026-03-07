from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from src.config.config_manager import ConfigManager


class PromptRenderer:
    """Uses Jinja2 to render prompts loaded from the filesystem."""

    @staticmethod
    def render(agent_type: str, transcript: str, summary: str) -> str:
        config = ConfigManager()
        prompt_path_str = config.get(f"prompts.{agent_type}")

        if not prompt_path_str:
            raise ValueError(f"Prompt path for {agent_type} not found in config.")

        root = Path(__file__).parent.parent.parent
        prompt_path = (root / prompt_path_str).resolve()

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template file not found: {prompt_path}")

        env = Environment(loader=FileSystemLoader(str(prompt_path.parent)))
        template = env.get_template(prompt_path.name)

        return template.render(transcript=transcript, summary=summary)
