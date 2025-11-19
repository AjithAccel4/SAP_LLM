"""
Unit tests for ErrorDetector.
"""

import pytest
from sap_llm.correction.error_detector import (
    ErrorDetector,
    AnomalyDetector,
    Error,
    ErrorReport
)


class TestAnomalyDetector:
    """Test cases for AnomalyDetector."""

    def test_detect_negative_amount(self):
        """Test detection of negative amounts."""
        detector = AnomalyDetector()

        prediction = {
            "total_amount": {"value": -100.0, "confidence": 0.95}
        }

        anomalies = detector.detect(prediction)

        assert len(anomalies) == 1
        assert anomalies[0]["type"] == "negative_amount"
        assert anomalies[0]["field"] == "total_amount"

    def test_detect_extreme_amount(self):
        """Test detection of extremely high amounts."""
        detector = AnomalyDetector()

        prediction = {
            "total_amount": {"value": 2000000.0, "confidence": 0.95}
        }

        anomalies = detector.detect(prediction)

        assert len(anomalies) == 1
        assert anomalies[0]["type"] == "extreme_amount"

    def test_no_anomalies(self):
        """Test with normal data."""
        detector = AnomalyDetector()

        prediction = {
            "total_amount": {"value": 1500.50, "confidence": 0.95}
        }

        anomalies = detector.detect(prediction)

        assert len(anomalies) == 0


class TestErrorDetector:
    """Test cases for ErrorDetector."""

    def test_detect_low_confidence(self):
        """Test low confidence detection."""
        detector = ErrorDetector()

        prediction = {
            "total_amount": {"value": 100.0, "confidence": 0.60},
            "vendor_name": {"value": "ACME Corp", "confidence": 0.95}
        }

        context = {"document_type": "INVOICE"}

        report = detector.detect_errors(prediction, context)

        assert report.has_errors
        assert len(report.errors) >= 1
        assert any(e.type == "low_confidence" for e in report.errors)

    def test_detect_rule_violations(self):
        """Test business rule violation detection."""
        detector = ErrorDetector()

        prediction = {
            "total_amount": {"value": 100.0, "confidence": 0.95},
            "line_items": {
                "value": [
                    {"amount": 50.0},
                    {"amount": 60.0}  # Total should be 110, not 100
                ],
                "confidence": 0.95
            }
        }

        context = {"document_type": "INVOICE"}

        report = detector.detect_errors(prediction, context)

        assert report.has_errors
        assert any(e.type == "rule_violation" for e in report.errors)

    def test_no_errors(self):
        """Test with clean data."""
        detector = ErrorDetector()

        prediction = {
            "total_amount": {"value": 100.0, "confidence": 0.95},
            "vendor_name": {"value": "ACME Corp", "confidence": 0.95},
            "invoice_number": {"value": "INV-001", "confidence": 0.95}
        }

        context = {"document_type": "INVOICE"}

        report = detector.detect_errors(prediction, context)

        # May not need correction despite no explicit errors
        assert report.overall_confidence > 0.8

    def test_calculate_overall_confidence(self):
        """Test overall confidence calculation."""
        detector = ErrorDetector()

        prediction = {
            "total_amount": {"value": 100.0, "confidence": 0.90},
            "vendor_name": {"value": "ACME", "confidence": 0.80}
        }

        confidence = detector._calculate_overall_confidence(prediction)

        # Should weight critical fields more
        assert 0.80 <= confidence <= 0.90

    def test_severity_determination(self):
        """Test severity determination."""
        detector = ErrorDetector()

        # High severity: critical field with low confidence
        high_severity_fields = ["total_amount", "vendor_id"]
        severity = detector._determine_severity(high_severity_fields)
        assert severity == "high"

        # Medium severity: many fields
        medium_fields = ["field1", "field2", "field3", "field4"]
        severity = detector._determine_severity(medium_fields)
        assert severity == "medium"

        # Low severity: few non-critical fields
        low_fields = ["optional_field"]
        severity = detector._determine_severity(low_fields)
        assert severity == "low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
