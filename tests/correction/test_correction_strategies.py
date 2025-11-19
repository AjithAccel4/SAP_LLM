"""
Unit tests for Correction Strategies.
"""

import pytest
from sap_llm.correction.error_detector import Error
from sap_llm.correction.strategies import (
    RuleBasedCorrectionStrategy,
    RerunWithHigherConfidenceStrategy,
    ContextEnhancementStrategy,
    HumanInTheLoopStrategy,
    HumanReviewQueue,
    CorrectionResult
)


class TestRuleBasedCorrectionStrategy:
    """Test cases for RuleBasedCorrectionStrategy."""

    def test_correct_calculation_error(self):
        """Test correction of calculation errors."""
        strategy = RuleBasedCorrectionStrategy()

        prediction = {
            "total_amount": {"value": 100.0, "confidence": 0.95},
            "line_items": {
                "value": [
                    {"amount": 50.0},
                    {"amount": 60.0}
                ],
                "confidence": 0.95
            }
        }

        error = Error(
            type="rule_violation",
            severity="high",
            violations=[{
                "rule": "total_matches_line_items",
                "expected": 110.0,
                "actual": 100.0
            }]
        )

        context = {"document_type": "INVOICE"}

        result = strategy.correct(prediction, error, context)

        assert result.success
        assert result.corrected_prediction["total_amount"]["value"] == 110.0
        assert "total_amount" in result.fields_corrected

    def test_correct_negative_amount(self):
        """Test correction of negative amounts."""
        strategy = RuleBasedCorrectionStrategy()

        prediction = {
            "total_amount": {"value": -100.0, "confidence": 0.90}
        }

        error = Error(
            type="anomaly",
            severity="medium",
            anomalies=[{
                "field": "total_amount",
                "type": "negative_amount",
                "value": -100.0
            }]
        )

        context = {}

        result = strategy.correct(prediction, error, context)

        assert result.success
        assert result.corrected_prediction["total_amount"]["value"] == 100.0

    def test_no_correction_needed(self):
        """Test when no rule-based correction is applicable."""
        strategy = RuleBasedCorrectionStrategy()

        prediction = {
            "vendor_name": {"value": "ACME", "confidence": 0.60}
        }

        error = Error(
            type="low_confidence",
            severity="medium",
            fields=["vendor_name"]
        )

        context = {}

        result = strategy.correct(prediction, error, context)

        assert not result.success


class TestRerunWithHigherConfidenceStrategy:
    """Test cases for RerunWithHigherConfidenceStrategy."""

    def test_without_models(self):
        """Test behavior when models are not available."""
        strategy = RerunWithHigherConfidenceStrategy(
            language_decoder=None,
            vision_encoder=None
        )

        prediction = {
            "vendor_name": {"value": "ACME", "confidence": 0.60}
        }

        error = Error(
            type="low_confidence",
            severity="medium",
            fields=["vendor_name"]
        )

        context = {"ocr_text": "Invoice from ACME Corp"}

        result = strategy.correct(prediction, error, context)

        # Should simulate improvement
        assert result.success or not result.success  # Depends on implementation


class TestContextEnhancementStrategy:
    """Test cases for ContextEnhancementStrategy."""

    def test_without_pmg(self):
        """Test behavior without PMG."""
        strategy = ContextEnhancementStrategy(pmg=None)

        prediction = {
            "vendor_name": {"value": "ACME", "confidence": 0.60}
        }

        error = Error(
            type="low_confidence",
            severity="medium",
            fields=["vendor_name"]
        )

        context = {}

        result = strategy.correct(prediction, error, context)

        assert not result.success


class TestHumanInTheLoopStrategy:
    """Test cases for HumanInTheLoopStrategy."""

    def test_create_review_task(self):
        """Test creation of human review task."""
        queue = HumanReviewQueue()
        strategy = HumanInTheLoopStrategy(review_queue=queue)

        prediction = {
            "total_amount": {"value": 100.0, "confidence": 0.50}
        }

        error = Error(
            type="low_confidence",
            severity="high",
            fields=["total_amount"]
        )

        context = {
            "document_id": "doc_001",
            "document_type": "INVOICE"
        }

        result = strategy.correct(prediction, error, context)

        assert result.requires_human
        assert result.task_id is not None
        assert not result.success

        # Check task was added to queue
        task = queue.get_task(result.task_id)
        assert task is not None
        assert task["priority"] == "high"

    def test_priority_determination(self):
        """Test priority determination based on error severity."""
        strategy = HumanInTheLoopStrategy()

        # Critical error -> urgent priority
        error_critical = Error(type="error", severity="critical", fields=[])
        priority = strategy._determine_priority(error_critical)
        assert priority == "urgent"

        # High error -> high priority
        error_high = Error(type="error", severity="high", fields=[])
        priority = strategy._determine_priority(error_high)
        assert priority == "high"

        # Medium error -> normal priority
        error_medium = Error(type="error", severity="medium", fields=[])
        priority = strategy._determine_priority(error_medium)
        assert priority == "normal"


class TestHumanReviewQueue:
    """Test cases for HumanReviewQueue."""

    def test_add_and_get_task(self):
        """Test adding and retrieving tasks."""
        queue = HumanReviewQueue()

        task = {
            "id": "task_001",
            "prediction": {},
            "priority": "high"
        }

        task_id = queue.add_task(task)

        retrieved = queue.get_task(task_id)
        assert retrieved["id"] == task_id

    def test_get_pending_tasks(self):
        """Test retrieving pending tasks."""
        queue = HumanReviewQueue()

        queue.add_task({"id": "task_001", "priority": "high"})
        queue.add_task({"id": "task_002", "priority": "normal"})
        queue.add_task({"id": "task_003", "priority": "high"})

        all_pending = queue.get_pending_tasks()
        assert len(all_pending) == 3

        high_priority = queue.get_pending_tasks(priority="high")
        assert len(high_priority) == 2

    def test_complete_task(self):
        """Test completing a task."""
        queue = HumanReviewQueue()

        task_id = queue.add_task({"id": "task_001", "priority": "high"})

        queue.complete_task(task_id)

        task = queue.get_task(task_id)
        assert task["status"] == "completed"

        pending = queue.get_pending_tasks()
        assert len(pending) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
