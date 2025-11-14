"""
Unit tests for utility modules.
"""

import pytest
import time

from sap_llm.utils.hash import hash_file, hash_string
from sap_llm.utils.timer import Timer, timed
from sap_llm.utils.logger import get_logger


@pytest.mark.unit
class TestHash:
    """Tests for hashing utilities."""

    def test_hash_string_sha256(self):
        """Test SHA256 string hashing."""
        text = "Hello, World!"
        hash_result = hash_string(text, algorithm="sha256")

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 produces 64 hex characters

        # Test consistency
        hash_result2 = hash_string(text, algorithm="sha256")
        assert hash_result == hash_result2

    def test_hash_string_md5(self):
        """Test MD5 string hashing."""
        text = "Hello, World!"
        hash_result = hash_string(text, algorithm="md5")

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32  # MD5 produces 32 hex characters

    def test_hash_different_strings(self):
        """Test that different strings produce different hashes."""
        hash1 = hash_string("string1")
        hash2 = hash_string("string2")

        assert hash1 != hash2

    def test_hash_file(self, temp_dir):
        """Test file hashing."""
        # Create temporary file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        # Hash file
        hash_result = hash_file(str(test_file))

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

        # Test consistency
        hash_result2 = hash_file(str(test_file))
        assert hash_result == hash_result2

    def test_hash_nonexistent_file(self):
        """Test hashing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            hash_file("/nonexistent/file.txt")


@pytest.mark.unit
class TestTimer:
    """Tests for timer utilities."""

    def test_timer_context_manager(self):
        """Test Timer as context manager."""
        with Timer() as timer:
            time.sleep(0.1)

        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.2  # Allow some tolerance

    def test_timer_decorator(self):
        """Test timed decorator."""
        @timed
        def slow_function():
            time.sleep(0.1)
            return "result"

        result = slow_function()

        assert result == "result"

    def test_timer_start_stop(self):
        """Test manual timer start/stop."""
        timer = Timer()
        timer.start()
        time.sleep(0.1)
        elapsed = timer.stop()

        assert elapsed >= 0.1
        assert timer.elapsed >= 0.1

    def test_timer_without_start(self):
        """Test timer.stop() without start raises error."""
        timer = Timer()
        with pytest.raises(RuntimeError):
            timer.stop()


@pytest.mark.unit
class TestLogger:
    """Tests for logger utilities."""

    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_logger")

        assert logger is not None
        assert logger.name == "test_logger"

    def test_logger_levels(self):
        """Test logger with different levels."""
        logger = get_logger("test_logger")

        # Should not raise errors
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_logger_with_exception(self):
        """Test logger with exception."""
        logger = get_logger("test_logger")

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.error("Error occurred", exc_info=True)
