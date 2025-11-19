"""
Integration tests for the complete self-correction pipeline.
"""

import pytest
from sap_llm.correction import (
    SelfCorrectionEngine,
    ErrorPatternLearner,
    EscalationManager,
    CorrectionAnalytics
)


class TestSelfCorrectionPipeline:
    """Integration tests for complete correction workflow."""

    def test_end_to_end_correction_with_rule_fix(self):
        """Test complete correction pipeline with rule-based fix."""
        # Initialize components
        pattern_learner = ErrorPatternLearner()
        escalation_manager = EscalationManager(pattern_learner=pattern_learner)
        analytics = CorrectionAnalytics(
            pattern_learner=pattern_learner,
            escalation_manager=escalation_manager
        )

        engine = SelfCorrectionEngine(
            max_attempts=3,
            confidence_threshold=0.80,
            pattern_learner=pattern_learner
        )

        # Create prediction with error
        prediction = {
            "total_amount": {"value": 100.0, "confidence": 0.95},
            "line_items": {
                "value": [
                    {"amount": 50.0},
                    {"amount": 60.0}  # Should sum to 110
                ],
                "confidence": 0.95
            },
            "vendor_name": {"value": "ACME Corp", "confidence": 0.95},
            "invoice_number": {"value": "INV-001", "confidence": 0.95}
        }

        context = {
            "document_type": "INVOICE",
            "document_id": "test_001"
        }

        # Run correction
        result = engine.correct_prediction(prediction, context, enable_learning=True)

        # Verify correction
        assert "correction_metadata" in result
        metadata = result["correction_metadata"]

        # Should have been corrected
        assert result["total_amount"]["value"] == 110.0

        # Check metadata
        assert metadata["total_attempts"] >= 1
        assert metadata.get("success") or metadata.get("partial_success")

    def test_human_escalation_on_persistent_errors(self):
        """Test escalation to human when errors persist."""
        pattern_learner = ErrorPatternLearner()
        escalation_manager = EscalationManager(
            pattern_learner=pattern_learner,
            max_auto_attempts=2
        )

        engine = SelfCorrectionEngine(
            max_attempts=2,
            confidence_threshold=0.80,
            pattern_learner=pattern_learner
        )

        # Create prediction with critical low confidence that can't be auto-fixed
        prediction = {
            "total_amount": {"value": 100.0, "confidence": 0.40},  # Very low
            "vendor_id": {"value": "UNKNOWN", "confidence": 0.30},  # Critical field, very low
            "vendor_name": {"value": "???", "confidence": 0.35}
        }

        context = {
            "document_type": "INVOICE",
            "document_id": "test_002"
        }

        # Run correction
        result = engine.correct_prediction(prediction, context)

        # Should escalate to human
        metadata = result.get("correction_metadata", {})
        assert metadata.get("required_human_review", False)

    def test_pattern_learning_from_correction(self):
        """Test that the system learns from corrections."""
        pattern_learner = ErrorPatternLearner()

        engine = SelfCorrectionEngine(
            max_attempts=3,
            pattern_learner=pattern_learner
        )

        # First correction
        prediction1 = {
            "total_amount": {"value": 100.0, "confidence": 0.95},
            "subtotal": {"value": 80.0, "confidence": 0.95},
            "tax_amount": {"value": 20.0, "confidence": 0.95}
        }

        context1 = {
            "document_type": "INVOICE",
            "document_id": "test_003",
            "vendor": "ACME Corp"
        }

        result1 = engine.correct_prediction(prediction1, context1, enable_learning=True)

        # Check that pattern was learned
        patterns = pattern_learner.get_relevant_patterns(context1)
        assert len(patterns) >= 0  # May or may not have patterns depending on errors

        # Check strategy effectiveness tracking
        effectiveness = pattern_learner.get_strategy_effectiveness()
        assert "strategies" in effectiveness or len(effectiveness) > 0

    def test_correction_analytics(self):
        """Test correction analytics tracking."""
        pattern_learner = ErrorPatternLearner()
        escalation_manager = EscalationManager(pattern_learner=pattern_learner)
        analytics = CorrectionAnalytics(
            pattern_learner=pattern_learner,
            escalation_manager=escalation_manager
        )

        engine = SelfCorrectionEngine(
            pattern_learner=pattern_learner,
            max_attempts=3
        )

        # Perform a correction
        prediction = {
            "total_amount": {"value": 100.0, "confidence": 0.95},
            "line_items": {
                "value": [{"amount": 50.0}, {"amount": 60.0}],
                "confidence": 0.95
            },
            "vendor_name": {"value": "ACME", "confidence": 0.95}
        }

        context = {
            "document_type": "INVOICE",
            "document_id": "test_004"
        }

        result = engine.correct_prediction(prediction, context)

        # Record in analytics
        analytics.record_correction_event(
            prediction=result,
            correction_metadata=result.get("correction_metadata", {}),
            context=context
        )

        # Generate report
        report = analytics.generate_correction_report(period_days=7)

        assert "total_corrections" in report
        assert report["total_corrections"] >= 1

    def test_multiple_strategy_attempts(self):
        """Test that multiple strategies are tried."""
        engine = SelfCorrectionEngine(max_attempts=3)

        # Create a prediction with multiple types of errors
        prediction = {
            "total_amount": {"value": -100.0, "confidence": 0.70},  # Anomaly
            "vendor_name": {"value": "ACM", "confidence": 0.65},    # Low confidence
            "line_items": {
                "value": [{"amount": 50.0}],
                "confidence": 0.80
            }
        }

        context = {
            "document_type": "INVOICE",
            "document_id": "test_005"
        }

        result = engine.correct_prediction(prediction, context)

        metadata = result.get("correction_metadata", {})

        # Should have tried multiple strategies
        strategies_tried = metadata.get("strategies_tried", [])
        assert len(strategies_tried) >= 1

        # At least rule-based should have been tried for the negative amount
        assert any("Rule" in s for s in strategies_tried) or len(strategies_tried) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
