import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigManager:
    """Singleton ConfigManager to handle yaml configuration."""

    _instance = None
    _config: Dict[str, Any] = {}
    _config_path: Path

    def __new__(cls, config_path: str = "cfg/config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config_path = Path(config_path)
            cls._instance.load_config()
        return cls._instance

    def load_config(self) -> None:
        """Loads configuration from the yaml file."""
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path}"
            )

        with open(self._config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def save_config(self) -> None:
        """Saves current configuration to the yaml file."""
        with open(self._config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value by dot-separated key (e.g., 'vllm.model')."""
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        """Sets a configuration value by dot-separated key and saves."""
        keys = key.split(".")
        d = self._config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        self.save_config()

    def get_all(self) -> Dict[str, Any]:
        """Returns the entire configuration dictionary."""
        return self._config

    @classmethod
    def reset(cls):
        """Reset the singleton instance (mostly for testing)."""
        cls._instance = None
