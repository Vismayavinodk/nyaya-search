"""
Robust YAML configuration loader with recursive dictionary/attribute access and validation.
"""

import os
from typing import Any, Dict
import yaml


class Config(dict):
    """
    Hierarchical dictionary wrapper allowing attribute-style dot access (config.a.b)
    as well as standard dictionary key access (config['a']['b']).
    """

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        if isinstance(value, dict):
            value = Config(value)
        self[key] = value

    def __delattr__(self, key: str):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert recursively back to standard Python dict."""
        out = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """
    Loads and validates the project configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config: Config object with dot-accessible attributes.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If essential configuration sections are missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Malformed configuration file: {config_path}")

    # Validate essential high-level sections
    required_sections = ["project", "paths", "data", "sparse_retrieval", "dense_retrieval"]
    for section in required_sections:
        if section not in data:
            raise ValueError(f"Configuration missing required section '{section}'")

    return Config(data)
