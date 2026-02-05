"""
Configuration loader for the RAG pipeline.

Loads settings from YAML config file to separate configuration from code.
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """
    Load pipeline settings from a YAML file (easy to change without touching code).

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dictionary containing all configuration settings
    """
    # Check if file exists first
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read and parse YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)  # safe_load prevents code execution in YAML

    return config


def get_config_value(config: Dict, *keys, default=None) -> Any:
    """
    Safely get a nested config value, or return a default.

    Args:
        config: Configuration dictionary
        *keys: Sequence of keys to traverse
        default: Default value if key not found

    Returns:
        The config value or default
    """
    # Start with the full config dict
    value = config
    # Walk through each key in the path
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]  # Go one level deeper
        else:
            return default  # Key not found, return default
    return value
