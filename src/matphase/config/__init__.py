"""
Configuration module for MatPhase.

Provides schema validation and configuration loading utilities.
"""

from .loader import load_config, save_config
from .schema import MatPhaseConfig

__all__ = ["MatPhaseConfig", "load_config", "save_config"]
