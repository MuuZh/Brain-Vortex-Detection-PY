"""
Logging utilities for MatPhase.

Provides structured logging setup with console and file output support.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure global logging for MatPhase.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (None = console only)
        format_string: Custom format string (None = use default)

    Example:
        >>> setup_logging(level="DEBUG", log_file=Path("matphase.log"))
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default format with timestamp, level, module, and message
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Get root logger
    root_logger = logging.getLogger("matphase")
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()  # Remove any existing handlers

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    # Ensure the logger is under the matphase namespace
    if not name.startswith("matphase"):
        name = f"matphase.{name}"

    return logging.getLogger(name)


class LoggingContext:
    """
    Context manager for temporary logging level changes.

    Example:
        >>> logger = get_logger(__name__)
        >>> with LoggingContext(logger, "DEBUG"):
        ...     logger.debug("This will be shown")
        >>> # Returns to previous level
    """

    def __init__(self, logger: logging.Logger, level: str):
        """
        Initialize logging context.

        Args:
            logger: Logger instance to modify
            level: Temporary logging level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper(), logging.INFO)
        self.old_level: Optional[int] = None

    def __enter__(self) -> logging.Logger:
        """Enter context and set new logging level."""
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore original logging level."""
        if self.old_level is not None:
            self.logger.setLevel(self.old_level)
