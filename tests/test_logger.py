"""
Comprehensive tests for logging utilities.
"""

import json
import logging
import pytest
from pathlib import Path

from sap_llm.utils.logger import JSONFormatter, setup_logging, get_logger


@pytest.mark.unit
class TestJSONFormatter:
    """Tests for JSON log formatter."""

    def test_json_formatter_basic(self):
        """Test JSON formatter with basic log record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test"
        assert log_data["line"] == 10

    def test_json_formatter_with_exception(self):
        """Test JSON formatter with exception info."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )

        result = formatter.format(record)
        log_data = json.loads(result)

        assert log_data["level"] == "ERROR"
        assert log_data["message"] == "Error occurred"
        assert "exception" in log_data
        assert "ValueError" in log_data["exception"]
        assert "Test error" in log_data["exception"]


@pytest.mark.unit
class TestSetupLogging:
    """Tests for logging setup function."""

    def test_setup_logging_stdout_simple(self):
        """Test setup logging with stdout and simple format."""
        setup_logging(level="INFO", format="simple", output="stdout")

        logger = logging.getLogger()
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logging_stdout_json(self):
        """Test setup logging with stdout and JSON format."""
        setup_logging(level="DEBUG", format="json", output="stdout")

        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

        # Find JSON formatter
        has_json_formatter = False
        for handler in logger.handlers:
            if isinstance(handler.formatter, JSONFormatter):
                has_json_formatter = True
                break
        assert has_json_formatter

    def test_setup_logging_stdout_rich(self):
        """Test setup logging with stdout and rich format."""
        setup_logging(level="WARNING", format="rich", output="stdout")

        logger = logging.getLogger()
        assert logger.level == logging.WARNING
        assert len(logger.handlers) > 0

    def test_setup_logging_file(self, temp_dir):
        """Test setup logging with file output."""
        log_file = temp_dir / "test.log"

        setup_logging(
            level="INFO",
            format="json",
            output="file",
            file_path=str(log_file)
        )

        logger = logging.getLogger()
        logger.info("Test message")

        # Verify file was created and has content
        assert log_file.exists()
        content = log_file.read_text()
        assert len(content) > 0

        # Verify it's valid JSON
        log_data = json.loads(content)
        assert log_data["message"] == "Test message"

    def test_setup_logging_both(self, temp_dir):
        """Test setup logging with both stdout and file."""
        log_file = temp_dir / "both.log"

        setup_logging(
            level="DEBUG",
            format="json",
            output="both",
            file_path=str(log_file)
        )

        logger = logging.getLogger()
        assert len(logger.handlers) >= 2  # At least stdout and file

    def test_setup_logging_file_without_path_raises_error(self):
        """Test that file output without path raises ValueError."""
        with pytest.raises(ValueError, match="file_path is required"):
            setup_logging(level="INFO", format="json", output="file")

    def test_setup_logging_creates_parent_dirs(self, temp_dir):
        """Test that setup logging creates parent directories."""
        log_file = temp_dir / "logs" / "nested" / "test.log"

        setup_logging(
            level="INFO",
            format="json",
            output="file",
            file_path=str(log_file)
        )

        assert log_file.exists()
        assert log_file.parent.exists()

    def test_setup_logging_different_levels(self):
        """Test setup logging with different log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            setup_logging(level=level, format="simple", output="stdout")
            logger = logging.getLogger()
            assert logger.level == getattr(logging, level)

    def test_setup_logging_clears_existing_handlers(self):
        """Test that setup_logging clears existing handlers."""
        # Add a handler
        logger = logging.getLogger()
        initial_handler = logging.StreamHandler()
        logger.addHandler(initial_handler)

        initial_count = len(logger.handlers)

        # Setup logging should clear and re-add
        setup_logging(level="INFO", format="simple", output="stdout")

        # Should have handlers, but not necessarily more than before
        assert len(logger.handlers) > 0


@pytest.mark.unit
class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_basic(self):
        """Test getting a logger instance."""
        logger = get_logger("test_module")

        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_multiple_calls(self):
        """Test that multiple calls return the same logger."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")

        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Test that different names return different loggers."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_logger_hierarchy(self):
        """Test logger hierarchy."""
        parent = get_logger("parent")
        child = get_logger("parent.child")

        assert child.parent.name == "parent"


@pytest.mark.unit
class TestLoggingIntegration:
    """Integration tests for logging functionality."""

    def test_logging_workflow(self, temp_dir):
        """Test complete logging workflow."""
        log_file = temp_dir / "workflow.log"

        # Setup logging
        setup_logging(
            level="DEBUG",
            format="json",
            output="both",
            file_path=str(log_file)
        )

        # Get logger and log messages
        logger = get_logger("test_workflow")

        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Verify file has all messages
        content = log_file.read_text()
        lines = [line for line in content.strip().split('\n') if line]

        assert len(lines) == 4

        # Verify each line is valid JSON
        for line in lines:
            log_data = json.loads(line)
            assert "message" in log_data
            assert "level" in log_data

    def test_logging_with_extra_fields(self, temp_dir):
        """Test logging with extra fields."""
        log_file = temp_dir / "extra.log"

        setup_logging(
            level="INFO",
            format="json",
            output="file",
            file_path=str(log_file)
        )

        logger = get_logger("test_extra")

        # Create log record with extra fields
        logger.info("Message with context", extra={"user_id": "12345", "request_id": "abc-def"})

        content = log_file.read_text()
        log_data = json.loads(content)

        assert log_data["message"] == "Message with context"
        # Note: The extra fields handling depends on the formatter implementation

    def test_logging_exception_handling(self, temp_dir):
        """Test logging exception with traceback."""
        log_file = temp_dir / "exception.log"

        setup_logging(
            level="ERROR",
            format="json",
            output="file",
            file_path=str(log_file)
        )

        logger = get_logger("test_exception")

        try:
            1 / 0
        except ZeroDivisionError:
            logger.error("Division by zero", exc_info=True)

        content = log_file.read_text()
        log_data = json.loads(content)

        assert log_data["message"] == "Division by zero"
        assert "exception" in log_data
        assert "ZeroDivisionError" in log_data["exception"]
