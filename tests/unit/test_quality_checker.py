"""
Comprehensive unit tests for QualityChecker module.
Tests all 6 dimensions of quality assessment.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

from sap_llm.models.quality_checker import QualityChecker


class TestQualityChecker:
    """Comprehensive tests for QualityChecker (6 dimensions)."""

    @pytest.fixture
    def quality_checker(self):
        """Create QualityChecker instance."""
        return QualityChecker()

    @pytest.fixture
    def valid_invoice_data(self):
        """Valid supplier invoice data for testing."""
        return {
            "invoice_number": "INV-2025-001",
            "invoice_date": "2025-01-15",
            "posting_date": "2025-01-16",
            "vendor_id": "VENDOR-123",
            "vendor_name": "Acme Corp",
            "total_amount": 1000.00,
            "currency": "USD",
            "tax_amount": 100.00,
            "net_amount": 900.00,
            "payment_terms": "NET30",
            "purchase_order": "PO-2025-001",
            "line_items": [
                {
                    "item_number": "1",
                    "description": "Product A",
                    "quantity": 10,
                    "unit_price": 90.00,
                    "total": 900.00
                }
            ]
        }

    @pytest.fixture
    def invoice_schema(self):
        """Invoice schema for validation."""
        return {
            "required_fields": [
                "invoice_number",
                "invoice_date",
                "vendor_id",
                "total_amount",
                "currency"
            ],
            "field_types": {
                "invoice_number": "string",
                "invoice_date": "date",
                "total_amount": "float",
                "tax_amount": "float",
                "line_items": "array"
            },
            "field_formats": {
                "invoice_date": r"^\d{4}-\d{2}-\d{2}$",
                "invoice_number": r"^INV-",
                "currency": r"^[A-Z]{3}$"
            }
        }

    # =========================================================================
    # Dimension 1: Completeness Tests
    # =========================================================================

    def test_completeness_all_required_fields_present(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test completeness when all required fields are present."""
        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, {}
        )

        assert result["overall_score"] >= 0.90
        assert "completeness" in result["dimensions"]
        assert result["dimensions"]["completeness"]["score"] == 1.0

    def test_completeness_missing_required_fields(
        self, quality_checker, invoice_schema
    ):
        """Test completeness when required fields are missing."""
        incomplete_data = {
            "invoice_number": "INV-001"
            # Missing: invoice_date, vendor_id, total_amount, currency
        }

        result = quality_checker.check_quality(
            incomplete_data, invoice_schema, {}
        )

        assert result["overall_score"] < 0.5
        assert result["dimensions"]["completeness"]["score"] < 0.5
        assert len(result["dimensions"]["completeness"]["issues"]) >= 4

    def test_completeness_partial_fields(
        self, quality_checker, invoice_schema
    ):
        """Test completeness with partial required fields."""
        partial_data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-01-15",
            "vendor_id": "VENDOR-123"
            # Missing: total_amount, currency
        }

        result = quality_checker.check_quality(
            partial_data, invoice_schema, {}
        )

        assert 0.5 <= result["dimensions"]["completeness"]["score"] < 1.0

    # =========================================================================
    # Dimension 2: Type Validity Tests
    # =========================================================================

    def test_type_validity_correct_types(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test type validity when all types are correct."""
        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, {}
        )

        assert result["dimensions"]["type_validity"]["score"] == 1.0
        assert len(result["dimensions"]["type_validity"]["issues"]) == 0

    def test_type_validity_incorrect_types(
        self, quality_checker, invoice_schema
    ):
        """Test type validity when types are incorrect."""
        invalid_data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-01-15",
            "vendor_id": "VENDOR-123",
            "total_amount": "not a number",  # Should be float
            "currency": "USD",
            "line_items": "not an array"  # Should be array
        }

        result = quality_checker.check_quality(
            invalid_data, invoice_schema, {}
        )

        assert result["dimensions"]["type_validity"]["score"] < 0.8
        assert len(result["dimensions"]["type_validity"]["issues"]) >= 2

    def test_type_validity_mixed_types(
        self, quality_checker, invoice_schema
    ):
        """Test type validity with mix of correct and incorrect types."""
        mixed_data = {
            "invoice_number": "INV-001",
            "invoice_date": "invalid-date",  # Invalid format
            "vendor_id": "VENDOR-123",
            "total_amount": 1000.00,  # Correct
            "currency": "USD"
        }

        result = quality_checker.check_quality(
            mixed_data, invoice_schema, {}
        )

        assert 0.6 <= result["dimensions"]["type_validity"]["score"] < 1.0

    # =========================================================================
    # Dimension 3: Format Validation Tests
    # =========================================================================

    def test_format_validation_all_correct(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test format validation when all formats are correct."""
        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, {}
        )

        assert result["dimensions"]["format_validation"]["score"] == 1.0

    def test_format_validation_incorrect_formats(
        self, quality_checker, invoice_schema
    ):
        """Test format validation with incorrect formats."""
        invalid_format_data = {
            "invoice_number": "WRONG-001",  # Should start with INV-
            "invoice_date": "15/01/2025",  # Wrong date format
            "vendor_id": "VENDOR-123",
            "total_amount": 1000.00,
            "currency": "US"  # Should be 3 letters
        }

        result = quality_checker.check_quality(
            invalid_format_data, invoice_schema, {}
        )

        assert result["dimensions"]["format_validation"]["score"] < 0.7
        assert len(result["dimensions"]["format_validation"]["issues"]) >= 3

    # =========================================================================
    # Dimension 4: Confidence Scoring Tests
    # =========================================================================

    def test_confidence_scoring_high_confidence(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test confidence scoring with high confidence values."""
        field_confidences = {
            "invoice_number": 0.99,
            "invoice_date": 0.98,
            "vendor_id": 0.97,
            "total_amount": 0.96,
            "currency": 0.99
        }

        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, field_confidences
        )

        assert result["dimensions"]["confidence"]["score"] >= 0.95

    def test_confidence_scoring_low_confidence(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test confidence scoring with low confidence values."""
        field_confidences = {
            "invoice_number": 0.60,
            "invoice_date": 0.55,
            "vendor_id": 0.50,
            "total_amount": 0.45,
            "currency": 0.40
        }

        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, field_confidences
        )

        assert result["dimensions"]["confidence"]["score"] < 0.70
        assert len(result["dimensions"]["confidence"]["issues"]) >= 2

    def test_confidence_scoring_mixed_confidence(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test confidence scoring with mixed confidence values."""
        field_confidences = {
            "invoice_number": 0.95,  # High
            "invoice_date": 0.60,    # Low
            "vendor_id": 0.85,       # Medium
            "total_amount": 0.50,    # Low
            "currency": 0.99         # High
        }

        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, field_confidences
        )

        assert 0.70 <= result["dimensions"]["confidence"]["score"] < 0.90

    # =========================================================================
    # Dimension 5: Cross-Field Consistency Tests
    # =========================================================================

    def test_cross_field_consistency_totals_match(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test cross-field consistency when totals match."""
        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, {}
        )

        assert result["dimensions"]["consistency"]["score"] == 1.0

    def test_cross_field_consistency_totals_mismatch(
        self, quality_checker, invoice_schema
    ):
        """Test cross-field consistency when totals don't match."""
        inconsistent_data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-01-15",
            "vendor_id": "VENDOR-123",
            "total_amount": 1000.00,
            "currency": "USD",
            "tax_amount": 100.00,
            "net_amount": 800.00  # Should be 900.00
        }

        result = quality_checker.check_quality(
            inconsistent_data, invoice_schema, {}
        )

        assert result["dimensions"]["consistency"]["score"] < 0.9
        assert any(
            "total" in issue["message"].lower()
            for issue in result["dimensions"]["consistency"]["issues"]
        )

    def test_cross_field_consistency_date_logic(
        self, quality_checker, invoice_schema
    ):
        """Test cross-field consistency for date logic."""
        invalid_dates_data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-01-20",
            "posting_date": "2025-01-15",  # Before invoice date (invalid)
            "vendor_id": "VENDOR-123",
            "total_amount": 1000.00,
            "currency": "USD"
        }

        result = quality_checker.check_quality(
            invalid_dates_data, invoice_schema, {}
        )

        assert result["dimensions"]["consistency"]["score"] < 1.0

    # =========================================================================
    # Dimension 6: Anomaly Detection Tests
    # =========================================================================

    def test_anomaly_detection_normal_values(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test anomaly detection with normal values."""
        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, {}
        )

        assert result["dimensions"]["anomaly_detection"]["score"] >= 0.95

    def test_anomaly_detection_unusual_amount(
        self, quality_checker, invoice_schema
    ):
        """Test anomaly detection with unusual amount."""
        anomaly_data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-01-15",
            "vendor_id": "VENDOR-123",
            "total_amount": 999999999.00,  # Unusual amount
            "currency": "USD"
        }

        result = quality_checker.check_quality(
            anomaly_data, invoice_schema, {}
        )

        # Anomaly detection might flag this
        assert "anomaly_detection" in result["dimensions"]

    def test_anomaly_detection_negative_amount(
        self, quality_checker, invoice_schema
    ):
        """Test anomaly detection with negative amount."""
        anomaly_data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-01-15",
            "vendor_id": "VENDOR-123",
            "total_amount": -1000.00,  # Negative (should be credit note)
            "currency": "USD"
        }

        result = quality_checker.check_quality(
            anomaly_data, invoice_schema, {}
        )

        # Should detect negative amount as potential anomaly
        assert "anomaly_detection" in result["dimensions"]

    # =========================================================================
    # Overall Quality Score Tests
    # =========================================================================

    def test_overall_quality_score_perfect_data(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test overall quality score with perfect data."""
        field_confidences = {
            "invoice_number": 0.99,
            "invoice_date": 0.99,
            "vendor_id": 0.99,
            "total_amount": 0.99,
            "currency": 0.99
        }

        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, field_confidences
        )

        assert result["overall_score"] >= 0.95
        assert result["quality_level"] in ["excellent", "high"]

    def test_overall_quality_score_poor_data(
        self, quality_checker, invoice_schema
    ):
        """Test overall quality score with poor data."""
        poor_data = {
            "invoice_number": "WRONG",  # Wrong format
            "invoice_date": "invalid",   # Invalid date
            # Missing required fields
        }

        field_confidences = {
            "invoice_number": 0.40,
            "invoice_date": 0.35
        }

        result = quality_checker.check_quality(
            poor_data, invoice_schema, field_confidences
        )

        assert result["overall_score"] < 0.50
        assert result["quality_level"] in ["poor", "low"]

    def test_overall_quality_score_medium_data(
        self, quality_checker, invoice_schema
    ):
        """Test overall quality score with medium quality data."""
        medium_data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-01-15",
            "vendor_id": "VENDOR-123",
            "total_amount": 1000.00,
            # Missing some optional fields
        }

        field_confidences = {
            "invoice_number": 0.85,
            "invoice_date": 0.80,
            "vendor_id": 0.75,
            "total_amount": 0.82
        }

        result = quality_checker.check_quality(
            medium_data, invoice_schema, field_confidences
        )

        assert 0.70 <= result["overall_score"] < 0.90
        assert result["quality_level"] in ["medium", "good"]

    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================

    def test_empty_data(self, quality_checker, invoice_schema):
        """Test quality check with empty data."""
        result = quality_checker.check_quality({}, invoice_schema, {})

        assert result["overall_score"] == 0.0
        assert len(result["issues"]) > 0

    def test_none_values(self, quality_checker, invoice_schema):
        """Test quality check with None values."""
        data_with_nones = {
            "invoice_number": None,
            "invoice_date": None,
            "vendor_id": "VENDOR-123",
            "total_amount": 1000.00,
            "currency": "USD"
        }

        result = quality_checker.check_quality(
            data_with_nones, invoice_schema, {}
        )

        assert result["overall_score"] < 0.80

    def test_missing_schema(self, quality_checker, valid_invoice_data):
        """Test quality check with missing schema."""
        result = quality_checker.check_quality(valid_invoice_data, {}, {})

        # Should handle gracefully
        assert "overall_score" in result

    def test_extra_fields(self, quality_checker, valid_invoice_data, invoice_schema):
        """Test quality check with extra fields not in schema."""
        data_with_extras = {
            **valid_invoice_data,
            "extra_field_1": "value1",
            "extra_field_2": "value2"
        }

        result = quality_checker.check_quality(
            data_with_extras, invoice_schema, {}
        )

        # Extra fields should not negatively impact score
        assert result["overall_score"] >= 0.80

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_quality_check_performance(
        self, quality_checker, valid_invoice_data, invoice_schema
    ):
        """Test that quality check completes within reasonable time."""
        import time

        start = time.time()
        result = quality_checker.check_quality(
            valid_invoice_data, invoice_schema, {}
        )
        duration = time.time() - start

        assert duration < 0.1  # Should complete in < 100ms
        assert "overall_score" in result

    def test_quality_check_large_dataset(self, quality_checker, invoice_schema):
        """Test quality check with large line items array."""
        large_data = {
            "invoice_number": "INV-001",
            "invoice_date": "2025-01-15",
            "vendor_id": "VENDOR-123",
            "total_amount": 100000.00,
            "currency": "USD",
            "line_items": [
                {
                    "item_number": str(i),
                    "description": f"Product {i}",
                    "quantity": 10,
                    "unit_price": 100.00,
                    "total": 1000.00
                }
                for i in range(100)  # 100 line items
            ]
        }

        result = quality_checker.check_quality(large_data, invoice_schema, {})

        assert "overall_score" in result
        assert result["overall_score"] > 0


@pytest.mark.unit
class TestQualityCheckerIntegration:
    """Integration tests for QualityChecker with other components."""

    def test_quality_checker_with_unified_model_output(self):
        """Test quality checker with realistic model output."""
        checker = QualityChecker()

        # Simulate extraction output from UnifiedModel
        extraction_result = {
            "invoice_number": "INV-2025-001",
            "invoice_date": "2025-01-15",
            "vendor_id": "VENDOR-123",
            "vendor_name": "Acme Corp",
            "total_amount": 1250.50,
            "currency": "USD",
            "confidence_scores": {
                "invoice_number": 0.95,
                "invoice_date": 0.92,
                "vendor_id": 0.88,
                "total_amount": 0.90
            }
        }

        schema = {
            "required_fields": ["invoice_number", "invoice_date", "vendor_id", "total_amount"],
            "field_types": {"total_amount": "float"},
            "field_formats": {"invoice_date": r"^\d{4}-\d{2}-\d{2}$"}
        }

        result = checker.check_quality(
            extraction_result,
            schema,
            extraction_result.get("confidence_scores", {})
        )

        assert result["overall_score"] >= 0.85
        assert "dimensions" in result
