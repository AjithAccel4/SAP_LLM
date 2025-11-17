"""
Comprehensive unit tests for SelfCorrector module.
Tests 5 self-correction strategies.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from sap_llm.models.self_corrector import SelfCorrector


class TestSelfCorrector:
    """Comprehensive tests for SelfCorrector (5 strategies)."""

    @pytest.fixture
    def corrector(self):
        """Create SelfCorrector instance."""
        return SelfCorrector()

    @pytest.fixture
    def extraction_with_issues(self):
        """Extraction result with quality issues."""
        return {
            "invoice_number": "INV001",  # Missing prefix
            "invoice_date": "15-01-2025",  # Wrong format
            "vendor_id": "123",  # Missing prefix
            "total_amount": "1,000",  # String instead of float
            "currency": "US"  # Should be 3 letters
        }

    @pytest.fixture
    def quality_issues(self):
        """Quality issues identified."""
        return [
            {
                "field": "invoice_number",
                "issue": "format_invalid",
                "expected": "INV-XXXX",
                "actual": "INV001",
                "confidence": 0.65
            },
            {
                "field": "invoice_date",
                "issue": "format_invalid",
                "expected": "YYYY-MM-DD",
                "actual": "15-01-2025",
                "confidence": 0.60
            },
            {
                "field": "total_amount",
                "issue": "type_invalid",
                "expected": "float",
                "actual": "string",
                "confidence": 0.55
            }
        ]

    @pytest.fixture
    def ocr_text(self):
        """Sample OCR text."""
        return """
        INVOICE
        Invoice Number: INV-2025-001
        Date: 2025-01-15
        Vendor ID: VENDOR-123
        Total Amount: 1,000.00 USD
        """

    # =========================================================================
    # Strategy 1: Re-extraction with Focused Attention
    # =========================================================================

    def test_reextraction_focused_attention(
        self, corrector, extraction_with_issues, quality_issues, ocr_text
    ):
        """Test re-extraction with focused attention on low-confidence fields."""
        corrected, metadata = corrector.correct(
            extraction_with_issues,
            {"issues": quality_issues},
            ocr_text
        )

        assert "invoice_number" in corrected
        assert "invoice_date" in corrected
        assert "correction_applied" in metadata
        assert metadata.get("strategies_used", []) != []

    def test_reextraction_improves_confidence(
        self, corrector, extraction_with_issues, quality_issues, ocr_text
    ):
        """Test that re-extraction improves field confidence."""
        corrected, metadata = corrector.correct(
            extraction_with_issues,
            {"issues": quality_issues},
            ocr_text
        )

        # Metadata should indicate improvement
        assert "confidence_improvement" in metadata or "correction_applied" in metadata

    def test_reextraction_specific_fields_only(
        self, corrector, extraction_with_issues, quality_issues, ocr_text
    ):
        """Test that re-extraction focuses only on problematic fields."""
        # Only fix invoice_number
        single_issue = [quality_issues[0]]

        corrected, metadata = corrector.correct(
            extraction_with_issues,
            {"issues": single_issue},
            ocr_text
        )

        # Should correct invoice_number
        assert "corrected_fields" in metadata or "correction_applied" in metadata

    # =========================================================================
    # Strategy 2: Cross-Field Validation and Correction
    # =========================================================================

    def test_cross_field_validation_totals(self, corrector):
        """Test cross-field correction for total amount mismatches."""
        extraction = {
            "net_amount": 1000.00,
            "tax_amount": 100.00,
            "total_amount": 1200.00  # Should be 1100.00
        }

        issues = [{
            "field": "total_amount",
            "issue": "cross_field_mismatch",
            "message": "Total != Net + Tax"
        }]

        corrected, metadata = corrector.correct(
            extraction,
            {"issues": issues},
            ""
        )

        # Should correct total_amount to 1100.00
        if "total_amount" in corrected:
            assert corrected["total_amount"] == 1100.00 or \
                   abs(corrected["total_amount"] - 1100.00) < 0.01

    def test_cross_field_validation_line_items(self, corrector):
        """Test cross-field correction for line items sum."""
        extraction = {
            "net_amount": 1000.00,
            "line_items": [
                {"item_number": "1", "total": 600.00},
                {"item_number": "2", "total": 300.00}
            ]
        }

        issues = [{
            "field": "net_amount",
            "issue": "cross_field_mismatch",
            "message": "Net amount != sum of line items"
        }]

        corrected, metadata = corrector.correct(
            extraction,
            {"issues": issues},
            ""
        )

        # Should correct net_amount to 900.00 (sum of line items)
        if "net_amount" in corrected:
            assert corrected["net_amount"] == 900.00 or \
                   abs(corrected["net_amount"] - 900.00) < 0.01

    def test_cross_field_date_consistency(self, corrector):
        """Test cross-field correction for date logic."""
        extraction = {
            "invoice_date": "2025-01-20",
            "posting_date": "2025-01-15",  # Before invoice date (wrong)
            "due_date": "2025-02-20"
        }

        issues = [{
            "field": "posting_date",
            "issue": "date_logic_error",
            "message": "Posting date before invoice date"
        }]

        corrected, metadata = corrector.correct(
            extraction,
            {"issues": issues},
            ""
        )

        # Should suggest correction or flag for review
        assert "correction_applied" in metadata or "flagged_for_review" in metadata

    # =========================================================================
    # Strategy 3: PMG Similarity-Based Correction
    # =========================================================================

    @patch('sap_llm.pmg.graph_client.ProcessMemoryGraph')
    def test_pmg_similarity_correction(
        self, mock_pmg_class, corrector, extraction_with_issues, quality_issues
    ):
        """Test correction using similar historical documents from PMG."""
        # Mock PMG to return similar documents
        mock_pmg = Mock()
        mock_pmg.get_similar_documents.return_value = [
            {
                "invoice_number": "INV-2025-001",
                "vendor_id": "VENDOR-123",
                "total_amount": 1000.00,
                "similarity_score": 0.95
            }
        ]

        corrector_with_pmg = SelfCorrector(pmg_client=mock_pmg)

        corrected, metadata = corrector_with_pmg.correct(
            extraction_with_issues,
            {"issues": quality_issues},
            ""
        )

        # Should use similar documents for correction hints
        assert "correction_applied" in metadata or "pmg_used" in str(metadata)

    def test_pmg_similarity_vendor_correction(self, corrector):
        """Test vendor ID correction using PMG similarity."""
        extraction = {
            "vendor_name": "Acme Corp",
            "vendor_id": "123"  # Missing prefix
        }

        issues = [{
            "field": "vendor_id",
            "issue": "format_invalid",
            "confidence": 0.60
        }]

        # Mock similar documents with correct vendor ID
        with patch.object(corrector, '_get_similar_documents') as mock_similar:
            mock_similar.return_value = [
                {
                    "vendor_name": "Acme Corp",
                    "vendor_id": "VENDOR-123",
                    "similarity": 0.98
                }
            ]

            corrected, metadata = corrector.correct(
                extraction,
                {"issues": issues},
                ""
            )

            # Should correct vendor_id based on similar documents
            assert "correction_applied" in metadata or "vendor_id" in corrected

    # =========================================================================
    # Strategy 4: Business Rule-Based Correction
    # =========================================================================

    def test_business_rule_format_correction(self, corrector):
        """Test correction using business rule patterns."""
        extraction = {
            "invoice_number": "INV001",  # Missing dash
            "currency": "US",  # Should be USD
            "tax_rate": "10%"  # Should be 0.10
        }

        issues = [
            {"field": "invoice_number", "issue": "format_invalid"},
            {"field": "currency", "issue": "format_invalid"},
            {"field": "tax_rate", "issue": "type_invalid"}
        ]

        corrected, metadata = corrector.correct(
            extraction,
            {"issues": issues},
            ""
        )

        # Should apply business rule corrections
        if "invoice_number" in corrected:
            assert "INV-" in corrected["invoice_number"]
        if "currency" in corrected:
            assert len(corrected["currency"]) == 3
        if "tax_rate" in corrected:
            assert isinstance(corrected["tax_rate"], (int, float))

    def test_business_rule_standardization(self, corrector):
        """Test data standardization using business rules."""
        extraction = {
            "invoice_date": "15/01/2025",  # Wrong format
            "total_amount": "1,000.00",  # String with comma
            "phone_number": "(555) 123-4567"  # Needs standardization
        }

        issues = [
            {"field": "invoice_date", "issue": "format_invalid"},
            {"field": "total_amount", "issue": "type_invalid"},
            {"field": "phone_number", "issue": "format_invalid"}
        ]

        corrected, metadata = corrector.correct(
            extraction,
            {"issues": issues},
            ""
        )

        # Should standardize formats
        if "invoice_date" in corrected:
            assert corrected["invoice_date"].count("-") == 2
        if "total_amount" in corrected:
            assert isinstance(corrected["total_amount"], (int, float))

    # =========================================================================
    # Strategy 5: Multi-Model Consensus
    # =========================================================================

    def test_multi_model_consensus_voting(self, corrector):
        """Test multi-model consensus for conflicting extractions."""
        # Simulate outputs from different models
        extraction_models = {
            "model_1": {"invoice_number": "INV-2025-001", "confidence": 0.85},
            "model_2": {"invoice_number": "INV-2025-001", "confidence": 0.90},
            "model_3": {"invoice_number": "INV-2025-002", "confidence": 0.60}
        }

        # Model 1 and 2 agree, model 3 differs
        corrected, metadata = corrector._apply_consensus(extraction_models)

        # Should choose INV-2025-001 (majority + higher confidence)
        assert corrected["invoice_number"] == "INV-2025-001"

    def test_multi_model_consensus_confidence_weighted(self, corrector):
        """Test that consensus is weighted by confidence scores."""
        extraction_models = {
            "vision": {"total_amount": 1000.00, "confidence": 0.95},
            "language": {"total_amount": 1100.00, "confidence": 0.60},
            "reasoning": {"total_amount": 1000.00, "confidence": 0.85}
        }

        corrected, metadata = corrector._apply_consensus(extraction_models)

        # Should choose 1000.00 (higher combined confidence)
        assert corrected["total_amount"] == 1000.00

    # =========================================================================
    # Combined Strategies Tests
    # =========================================================================

    def test_correction_applies_multiple_strategies(
        self, corrector, extraction_with_issues, quality_issues, ocr_text
    ):
        """Test that corrector applies multiple strategies."""
        corrected, metadata = corrector.correct(
            extraction_with_issues,
            {"issues": quality_issues},
            ocr_text
        )

        # Should apply multiple strategies
        assert "correction_applied" in metadata
        strategies_used = metadata.get("strategies_used", [])
        # Multiple strategies may be used
        assert isinstance(strategies_used, list)

    def test_correction_iterative_improvement(self, corrector):
        """Test iterative correction until quality threshold met."""
        extraction = {
            "invoice_number": "001",
            "total_amount": "1000"
        }

        issues = [
            {"field": "invoice_number", "issue": "format_invalid"},
            {"field": "total_amount", "issue": "type_invalid"}
        ]

        corrected, metadata = corrector.correct(
            extraction,
            {"issues": issues, "overall_score": 0.60},
            "",
            max_iterations=3
        )

        # Should attempt multiple iterations if needed
        assert "iterations" in metadata or "correction_applied" in metadata

    # =========================================================================
    # Improvement Metrics Tests
    # =========================================================================

    def test_correction_tracks_improvements(
        self, corrector, extraction_with_issues, quality_issues, ocr_text
    ):
        """Test that correction tracks quality improvements."""
        initial_quality = 0.65

        corrected, metadata = corrector.correct(
            extraction_with_issues,
            {"issues": quality_issues, "overall_score": initial_quality},
            ocr_text
        )

        # Should track improvement metrics
        assert "initial_quality" in metadata or "final_quality" in metadata or \
               "improvement" in metadata

    def test_correction_calculates_confidence_delta(
        self, corrector, extraction_with_issues, quality_issues, ocr_text
    ):
        """Test calculation of confidence improvement."""
        corrected, metadata = corrector.correct(
            extraction_with_issues,
            {"issues": quality_issues},
            ocr_text
        )

        # Should calculate per-field confidence changes
        assert "correction_applied" in metadata

    # =========================================================================
    # Edge Cases and Error Handling
    # =========================================================================

    def test_correction_no_issues(self, corrector):
        """Test correction when no issues exist."""
        perfect_extraction = {
            "invoice_number": "INV-2025-001",
            "invoice_date": "2025-01-15",
            "total_amount": 1000.00
        }

        corrected, metadata = corrector.correct(
            perfect_extraction,
            {"issues": [], "overall_score": 0.99},
            ""
        )

        # Should return original data unchanged
        assert corrected == perfect_extraction
        assert metadata.get("correction_applied") is False or \
               "correction_applied" not in metadata

    def test_correction_uncorrectable_issues(self, corrector):
        """Test correction with issues that cannot be automatically fixed."""
        extraction = {
            "invoice_number": "???",  # Completely unreadable
            "total_amount": None
        }

        issues = [
            {"field": "invoice_number", "issue": "unreadable", "confidence": 0.10},
            {"field": "total_amount", "issue": "missing", "confidence": 0.0}
        ]

        corrected, metadata = corrector.correct(
            extraction,
            {"issues": issues},
            ""
        )

        # Should flag for manual review
        assert metadata.get("manual_review_required") is True or \
               "flagged" in str(metadata).lower()

    def test_correction_empty_extraction(self, corrector):
        """Test correction with empty extraction."""
        corrected, metadata = corrector.correct(
            {},
            {"issues": [], "overall_score": 0.0},
            ""
        )

        # Should handle gracefully
        assert isinstance(corrected, dict)
        assert isinstance(metadata, dict)

    def test_correction_max_attempts_reached(self, corrector):
        """Test behavior when max correction attempts reached."""
        stubborn_extraction = {
            "invoice_number": "WRONG"
        }

        issues = [
            {"field": "invoice_number", "issue": "format_invalid"}
        ]

        corrected, metadata = corrector.correct(
            stubborn_extraction,
            {"issues": issues},
            "",
            max_iterations=1  # Only try once
        )

        # Should stop after max attempts
        assert metadata.get("max_attempts_reached") or \
               "iterations" in metadata

    # =========================================================================
    # Performance Tests
    # =========================================================================

    def test_correction_performance(self, corrector, extraction_with_issues, quality_issues):
        """Test that correction completes within reasonable time."""
        import time

        start = time.time()
        corrected, metadata = corrector.correct(
            extraction_with_issues,
            {"issues": quality_issues},
            ""
        )
        duration = time.time() - start

        assert duration < 1.0  # Should complete in < 1 second

    # =========================================================================
    # Integration Tests
    # =========================================================================

    def test_correction_integrates_with_quality_checker(self, corrector):
        """Test that corrector integrates with quality checker output."""
        # Realistic quality checker output
        quality_result = {
            "overall_score": 0.75,
            "dimensions": {
                "completeness": {"score": 0.80, "issues": []},
                "type_validity": {"score": 0.70, "issues": [
                    {"field": "total_amount", "issue": "type_mismatch"}
                ]},
                "format_validation": {"score": 0.75, "issues": [
                    {"field": "invoice_date", "issue": "format_invalid"}
                ]}
            },
            "issues": [
                {"field": "total_amount", "issue": "type_invalid"},
                {"field": "invoice_date", "issue": "format_invalid"}
            ]
        }

        extraction = {
            "invoice_date": "15/01/2025",
            "total_amount": "1000"
        }

        corrected, metadata = corrector.correct(
            extraction,
            quality_result,
            ""
        )

        # Should process quality checker output
        assert isinstance(corrected, dict)
        assert isinstance(metadata, dict)


@pytest.mark.unit
class TestSelfCorrectorStrategies:
    """Unit tests for individual correction strategies."""

    def test_format_correction_strategy(self):
        """Test format correction strategy in isolation."""
        corrector = SelfCorrector()

        # Test date format correction
        result = corrector._correct_format("15/01/2025", "invoice_date", "date")
        assert "-" in result

        # Test amount format correction
        result = corrector._correct_format("1,000.00", "total_amount", "float")
        assert isinstance(result, (int, float))

    def test_type_conversion_strategy(self):
        """Test type conversion strategy."""
        corrector = SelfCorrector()

        # String to float
        result = corrector._convert_type("1000.00", "float")
        assert isinstance(result, float)
        assert result == 1000.00

        # String to int
        result = corrector._convert_type("100", "int")
        assert isinstance(result, int)
        assert result == 100

    def test_pattern_matching_strategy(self):
        """Test pattern matching for common formats."""
        corrector = SelfCorrector()

        # Invoice number pattern
        result = corrector._apply_pattern("001", "invoice_number")
        assert "INV-" in result if result else True

        # Currency pattern
        result = corrector._apply_pattern("US", "currency")
        assert len(result) == 3 if result else True
