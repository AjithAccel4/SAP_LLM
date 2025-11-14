"""
Logging utilities for SAP_LLM.

Provides structured logging with support for JSON formatting, file output,
and integration with monitoring systems.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    format: str = "rich",  # Options: rich, json, simple
    output: str = "stdout",  # Options: stdout, file, both
    file_path: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format (rich, json, simple)
        output: Output destination (stdout, file, both)
        file_path: Path to log file (required if output is 'file' or 'both')
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handlers based on output
    handlers = []

    if output in ("stdout", "both"):
        if format == "rich":
            # Rich handler with nice formatting
            console = Console()
            handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                show_time=True,
                show_level=True,
                show_path=True,
            )
        elif format == "json":
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
        else:  # simple
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        handlers.append(handler)

    if output in ("file", "both"):
        if file_path is None:
            raise ValueError("file_path is required when output is 'file' or 'both'")

        # Create log directory if it doesn't exist
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler with JSON formatting
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(JSONFormatter())
        handlers.append(file_handler)

    # Add handlers to root logger
    for handler in handlers:
        handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing document", extra={"doc_id": "12345"})
    """
    return logging.getLogger(name)
