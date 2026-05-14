"""
Configuration loading utilities.

Provides functions to load and validate configuration from YAML files,
with support for environment variable overrides and CLI argument merging.
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from .schema import MatPhaseConfig


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with configuration data

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if data is not None else {}


def apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables follow the pattern: MATPHASE_<SECTION>_<KEY>
    Example: MATPHASE_PATHS_DATA_DIR=/path/to/data

    Args:
        config_dict: Base configuration dictionary

    Returns:
        Configuration with environment overrides applied
    """
    prefix = "MATPHASE_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Parse environment variable name
        parts = key[len(prefix) :].lower().split("_")
        if len(parts) < 2:
            continue

        section = parts[0]
        field = "_".join(parts[1:])

        # Apply override if section exists
        if section in config_dict:
            config_dict[section][field] = value

    return config_dict


def load_config(
    config_path: Optional[Path] = None,
    apply_env: bool = True,
    **overrides: Any,
) -> MatPhaseConfig:
    """
    Load and validate MatPhase configuration.

    Args:
        config_path: Path to YAML config file (None = use defaults)
        apply_env: Whether to apply environment variable overrides
        **overrides: Additional keyword arguments to override config values

    Returns:
        Validated MatPhaseConfig instance

    Example:
        >>> config = load_config(Path("configs/defaults.yaml"))
        >>> config = load_config(
        ...     Path("configs/defaults.yaml"),
        ...     preprocessing={"filter_low_freq": 0.02}
        ... )
    """
    # Start with empty dict or load from file
    if config_path is None:
        config_dict = {}
    else:
        config_dict = load_yaml(config_path)

    # Apply environment variable overrides
    if apply_env:
        config_dict = apply_env_overrides(config_dict)

    # Apply programmatic overrides
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config_dict:
            # Merge nested dictionaries
            config_dict[key].update(value)
        else:
            # Direct assignment
            config_dict[key] = value

    # Validate and return
    return MatPhaseConfig(**config_dict)


def save_config(config: MatPhaseConfig, path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: MatPhaseConfig instance to save
        path: Output path for YAML file
    """
    # Convert to dict using Pydantic's model_dump
    config_dict = config.model_dump(mode="python")

    # Convert Path objects to strings for YAML serialization
    def path_to_str(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: path_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [path_to_str(item) for item in obj]
        return obj

    config_dict = path_to_str(config_dict)

    # Write to file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
