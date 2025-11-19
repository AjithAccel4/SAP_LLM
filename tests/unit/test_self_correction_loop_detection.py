"""
Unit tests for Self-Correction Loop Detection.

Tests the enhanced self_corrector.py with loop detection and retry limits.
"""

import pytest
from unittest.mock import Mock, patch

from sap_llm.models.self_corrector import SelfCorrector


class TestLoopDetection:
    """Test suite for loop detection mechanisms."""

    @pytest.fixture
    def corrector(self):
        """Create corrector with loop detection enabled."""
        return SelfCorrector(
            confidence_threshold=0.70,
            max_attempts_per_field=3,
            max_total_iterations=5,
            enable_loop_detection=True
        )

    def test_initialization_with_loop_detection(self, corrector):
        """Test that corrector initializes with loop detection enabled."""
        assert corrector.enable_loop_detection is True
        assert corrector.max_attempts_per_field == 3
        assert corrector.max_total_iterations == 5
        assert corrector.field_attempts == {}
        assert corrector.state_history == set()
        assert corrector.total_iterations == 0

    def test_max_total_iterations_exceeded(self, corrector):
        """Test that correction stops after max total iterations."""
        # Manually set iterations to max
        corrector.total_iterations = 5

        data = {"amount": None}
        quality = {"overall_score": 0.5, "issues": []}

        corrected, metadata = corrector.correct(
            extracted_data=data,
            quality_assessment=quality,
            ocr_text="Test",
            schema={}
        )

        # Should terminate immediately
        assert metadata["termination_reason"] == "max_iterations_exceeded"
        assert metadata["loop_detected"] is True

    def test_state_hash_loop_detection(self, corrector):
        """Test that same state is detected and prevents loop."""
        data = {"amount": "100"}
        quality = {"overall_score": 0.5, "issues": []}

        # First call
        corrector.correct(
            extracted_data=data,
            quality_assessment=quality,
            ocr_text="Test",
            schema={}
        )

        # Second call with same data should detect loop
        corrected, metadata = corrector.correct(
            extracted_data=data,
            quality_assessment=quality,
            ocr_text="Test",
            schema={}
        )

        assert metadata.get("termination_reason") == "loop_detected"
        assert metadata.get("loop_detected") is True

    def test_state_reset_clears_tracking(self, corrector):
        """Test that reset_state clears all tracking data."""
        # Add some tracking data
        corrector.field_attempts["amount"] = 2
        corrector.state_history.add("abc123")
        corrector.total_iterations = 3

        # Reset
        corrector.reset_state()

        # Should be cleared
        assert corrector.field_attempts == {}
        assert corrector.state_history == set()
        assert corrector.total_iterations == 0

    def test_per_field_attempt_limit(self, corrector):
        """Test that per-field attempt limits are enforced."""
        # Manually set field attempts to max
        corrector.field_attempts["vendor"] = 3

        # Try to correct the field
        can_attempt = corrector._can_attempt_field("vendor")

        assert can_attempt is False

    def test_field_attempt_increment(self, corrector):
        """Test that field attempts are incremented correctly."""
        corrector._increment_field_attempts("amount")
        assert corrector.field_attempts["amount"] == 1

        corrector._increment_field_attempts("amount")
        assert corrector.field_attempts["amount"] == 2

    def test_compute_state_hash_deterministic(self, corrector):
        """Test that state hash is deterministic."""
        data1 = {"amount": "100", "vendor": "ABC"}
        data2 = {"vendor": "ABC", "amount": "100"}  # Different order

        hash1 = corrector._compute_state_hash(data1)
        hash2 = corrector._compute_state_hash(data2)

        # Should be same (sorted keys)
        assert hash1 == hash2

    def test_compute_state_hash_different_data(self, corrector):
        """Test that different data produces different hash."""
        data1 = {"amount": "100"}
        data2 = {"amount": "200"}

        hash1 = corrector._compute_state_hash(data1)
        hash2 = corrector._compute_state_hash(data2)

        assert hash1 != hash2


class TestFieldAttemptTracking:
    """Test per-field attempt tracking in correction methods."""

    @pytest.fixture
    def corrector(self):
        """Create corrector for testing."""
        return SelfCorrector(
            max_attempts_per_field=2,
            enable_loop_detection=True
        )

    def test_fix_missing_fields_respects_attempts(self, corrector):
        """Test that _fix_missing_fields respects attempt limits."""
        # Set field to max attempts
        corrector.field_attempts["vendor"] = 2

        with patch.object(corrector, '_lookup_field_in_pmg', return_value=None):
            with patch.object(corrector, '_extract_field_from_ocr', return_value=None):
                corrections = corrector._fix_missing_fields(
                    data={},
                    missing_fields=["vendor"],
                    ocr_text="Test",
                    schema={},
                    pmg_context=None
                )

                # Should be skipped
                assert len(corrections) == 1
                assert corrections[0]["strategy"] == "skipped"
                assert corrections[0]["success"] is False

    def test_fix_missing_fields_tracks_attempts(self, corrector):
        """Test that attempts are tracked during correction."""
        with patch.object(corrector, '_lookup_field_in_pmg', return_value=None):
            with patch.object(corrector, '_extract_field_from_ocr', return_value="Value"):
                corrector._fix_missing_fields(
                    data={},
                    missing_fields=["vendor"],
                    ocr_text="Test",
                    schema={},
                    pmg_context=None
                )

                # Attempt should be tracked
                assert corrector.field_attempts["vendor"] == 1

    def test_fix_low_confidence_fields_respects_attempts(self, corrector):
        """Test that _fix_low_confidence_fields respects attempt limits."""
        corrector.field_attempts["amount"] = 2

        with patch.object(corrector, '_extract_field_from_ocr', return_value=None):
            corrections = corrector._fix_low_confidence_fields(
                data={"amount": "old"},
                low_conf_fields=["amount"],
                ocr_text="Test",
                schema={}
            )

            # Should be skipped
            assert len(corrections) == 1
            assert corrections[0]["strategy"] == "skipped"

    def test_multiple_fields_tracked_independently(self, corrector):
        """Test that different fields are tracked independently."""
        corrector._increment_field_attempts("vendor")
        corrector._increment_field_attempts("amount")
        corrector._increment_field_attempts("vendor")

        assert corrector.field_attempts["vendor"] == 2
        assert corrector.field_attempts["amount"] == 1

    def test_corrections_include_attempt_count(self, corrector):
        """Test that correction metadata includes attempt counts."""
        with patch.object(corrector, '_lookup_field_in_pmg', return_value=None):
            with patch.object(corrector, '_extract_field_from_ocr', return_value="Found"):
                corrections = corrector._fix_missing_fields(
                    data={},
                    missing_fields=["vendor"],
                    ocr_text="Test",
                    schema={},
                    pmg_context=None
                )

                assert "attempts" in corrections[0]
                assert corrections[0]["attempts"] == 1


class TestLoopDetectionDisabled:
    """Test behavior when loop detection is disabled."""

    @pytest.fixture
    def corrector(self):
        """Create corrector with loop detection disabled."""
        return SelfCorrector(
            enable_loop_detection=False
        )

    def test_no_attempt_limits_when_disabled(self, corrector):
        """Test that attempt limits are not enforced when disabled."""
        # Can attempt any field any number of times
        for _ in range(10):
            can_attempt = corrector._can_attempt_field("test_field")
            assert can_attempt is True

    def test_state_not_tracked_when_disabled(self, corrector):
        """Test that state history is not used when disabled."""
        data = {"amount": "100"}
        quality = {"overall_score": 0.5, "issues": []}

        # Call multiple times with same data
        for _ in range(3):
            corrected, metadata = corrector.correct(
                extracted_data=data,
                quality_assessment=quality,
                ocr_text="Test",
                schema={}
            )

            # Should not detect loop
            assert metadata.get("loop_detected") is not True


class TestCorrectionMetadata:
    """Test correction metadata enhancements."""

    @pytest.fixture
    def corrector(self):
        """Create corrector for testing."""
        return SelfCorrector(enable_loop_detection=True)

    def test_metadata_includes_loop_info(self, corrector):
        """Test that metadata includes loop detection info when triggered."""
        corrector.total_iterations = 5  # Set to max

        data = {"amount": "100"}
        quality = {"overall_score": 0.5, "issues": []}

        corrected, metadata = corrector.correct(
            extracted_data=data,
            quality_assessment=quality,
            ocr_text="Test",
            schema={}
        )

        assert "termination_reason" in metadata
        assert "loop_detected" in metadata

    def test_successful_corrections_include_attempts(self, corrector):
        """Test that successful corrections include attempt counts."""
        data = {}
        quality = {
            "overall_score": 0.5,
            "issues": [{"type": "MISSING_FIELD", "field": "vendor"}]
        }

        with patch.object(corrector, '_lookup_field_in_pmg', return_value=None):
            with patch.object(corrector, '_extract_field_from_ocr', return_value="VendorName"):
                corrected, metadata = corrector.correct(
                    extracted_data=data,
                    quality_assessment=quality,
                    ocr_text="Test",
                    schema={},
                    pmg_context=None
                )

                # Check corrections have attempt info
                if metadata["corrections"]:
                    assert "attempts" in metadata["corrections"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
