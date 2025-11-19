"""
Unit tests for utility modules.
"""

import pytest
import time

from sap_llm.utils.hash import compute_file_hash, compute_hash
from sap_llm.utils.timer import Timer, timer
from sap_llm.utils.logger import get_logger


@pytest.mark.unit
class TestHash:
    """Tests for hashing utilities."""

    def test_hash_string_sha256(self):
        """Test SHA256 string hashing."""
        text = "Hello, World!"
        hash_result = compute_hash(text, algorithm="sha256")

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64  # SHA256 produces 64 hex characters

        # Test consistency
        hash_result2 = compute_hash(text, algorithm="sha256")
        assert hash_result == hash_result2

    def test_hash_string_md5(self):
        """Test MD5 string hashing (should reject insecure algorithm)."""
        text = "Hello, World!"
        # MD5 is insecure and should raise ValueError
        with pytest.raises(ValueError, match="Insecure hash algorithm"):
            compute_hash(text, algorithm="md5")

    def test_hash_different_strings(self):
        """Test that different strings produce different hashes."""
        hash1 = compute_hash("string1")
        hash2 = compute_hash("string2")

        assert hash1 != hash2

    def test_hash_file(self, temp_dir):
        """Test file hashing."""
        # Create temporary file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        # Hash file
        hash_result = compute_file_hash(str(test_file))

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

        # Test consistency
        hash_result2 = compute_file_hash(str(test_file))
        assert hash_result == hash_result2

    def test_hash_nonexistent_file(self):
        """Test hashing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            compute_file_hash("/nonexistent/file.txt")


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
        """Test timer decorator."""
        @timer
        def slow_function():
            time.sleep(0.1)
            return "result"

        result = slow_function()

        assert result == "result"

    def test_timer_start_stop(self):
        """Test manual timer start/stop."""
        t = Timer()
        t.start()
        time.sleep(0.1)
        elapsed = t.stop()

        assert elapsed >= 0.1
        assert t.elapsed >= 0.1

    def test_timer_without_start(self):
        """Test timer.stop() without start raises error."""
        t = Timer()
        with pytest.raises(RuntimeError):
            t.stop()


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


@pytest.mark.unit
class TestHashAdditional:
    """Additional hash tests for edge cases."""

    def test_hash_with_different_algorithms(self):
        """Test hashing with various secure algorithms."""
        text = "test data"

        # Test SHA-256
        hash_sha256 = compute_hash(text, algorithm="sha256")
        assert len(hash_sha256) == 64

        # Test SHA-384
        hash_sha384 = compute_hash(text, algorithm="sha384")
        assert len(hash_sha384) == 96

        # Test SHA-512
        hash_sha512 = compute_hash(text, algorithm="sha512")
        assert len(hash_sha512) == 128

    def test_hash_insecure_algorithm_rejection(self):
        """Test that insecure algorithms are rejected."""
        # MD5 should be rejected
        with pytest.raises(ValueError, match="Insecure hash algorithm"):
            compute_hash("data", algorithm="md5")

        # SHA1 should be rejected
        with pytest.raises(ValueError, match="Insecure hash algorithm"):
            compute_hash("data", algorithm="sha1")

    def test_hash_unknown_algorithm(self):
        """Test that unknown algorithms are rejected."""
        with pytest.raises(ValueError, match="Unknown or unsupported hash algorithm"):
            compute_hash("data", algorithm="invalid_algo")

    def test_hash_bytes_input(self):
        """Test hashing with bytes input."""
        data = b"binary data"
        hash_result = compute_hash(data)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

    def test_file_hash_with_different_algorithms(self, temp_dir):
        """Test file hashing with different algorithms."""
        test_file = temp_dir / "test.bin"
        test_file.write_bytes(b"test content")

        # SHA-256
        hash_sha256 = compute_file_hash(str(test_file), algorithm="sha256")
        assert len(hash_sha256) == 64

        # SHA-512
        hash_sha512 = compute_file_hash(str(test_file), algorithm="sha512")
        assert len(hash_sha512) == 128

    def test_file_hash_insecure_algorithm(self, temp_dir):
        """Test that file hashing rejects insecure algorithms."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        with pytest.raises(ValueError, match="Insecure hash algorithm"):
            compute_file_hash(str(test_file), algorithm="md5")


@pytest.mark.unit
class TestTimerAdditional:
    """Additional timer tests for coverage."""

    def test_timer_string_representation(self):
        """Test timer string representation."""
        t = Timer("Test Operation")

        # Before completion
        assert "not completed" in str(t)

        # After completion
        t.start()
        time.sleep(0.1)
        t.stop()
        assert "Test Operation" in str(t)
        assert "s" in str(t)

    def test_timer_decorator_with_name(self):
        """Test timer decorator with custom name."""
        from sap_llm.utils.timer import timer

        @timer(name="Custom Operation")
        def test_func():
            time.sleep(0.05)
            return "done"

        result = test_func()
        assert result == "done"

    def test_timer_context_manager_name(self):
        """Test timer context manager with custom name."""
        with Timer("Processing Test") as t:
            time.sleep(0.05)

        assert t.elapsed >= 0.05
        assert t.start_time is not None
        assert t.end_time is not None
