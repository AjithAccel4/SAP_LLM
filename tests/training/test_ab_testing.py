"""
Unit tests for A/B Testing Framework.

Tests cover:
- Test creation
- Traffic routing
- Prediction recording
- Statistical evaluation
- Winner determination
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json

from sap_llm.models.registry import ModelRegistry
from sap_llm.training.ab_testing import ABTestingManager


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def registry(temp_dir):
    """Create model registry for tests."""
    return ModelRegistry(db_path=str(Path(temp_dir) / "registry.db"))


@pytest.fixture
def ab_testing(temp_dir, registry):
    """Create A/B testing manager for tests."""
    return ABTestingManager(
        model_registry=registry,
        db_path=str(Path(temp_dir) / "ab_tests.db"),
        default_traffic_split=0.1
    )


class TestABTestingManager:
    """Test ABTestingManager class."""

    def test_initialization(self, ab_testing):
        """Test manager initialization."""
        assert ab_testing is not None
        assert ab_testing.default_traffic_split == 0.1

    def test_create_ab_test(self, ab_testing):
        """Test A/B test creation."""
        test_id = ab_testing.create_ab_test(
            champion_id="model_v1",
            challenger_id="model_v2",
            traffic_split=0.1
        )

        assert test_id is not None

        # Verify test was created
        test = ab_testing._get_test(test_id)
        assert test is not None
        assert test["champion_id"] == "model_v1"
        assert test["challenger_id"] == "model_v2"
        assert test["traffic_split"] == 0.1
        assert test["status"] == "active"

    def test_traffic_routing(self, ab_testing):
        """Test traffic routing between models."""
        test_id = ab_testing.create_ab_test(
            champion_id="model_v1",
            challenger_id="model_v2",
            traffic_split=0.1
        )

        # Route many predictions
        routes = []
        for _ in range(1000):
            model_id = ab_testing.route_prediction(test_id)
            routes.append(model_id)

        # Check distribution (should be ~10% to challenger)
        challenger_count = sum(1 for r in routes if r == "model_v2")
        challenger_pct = challenger_count / len(routes)

        assert 0.05 <= challenger_pct <= 0.15  # Allow 5% variance

    def test_record_prediction(self, ab_testing):
        """Test recording predictions."""
        test_id = ab_testing.create_ab_test(
            champion_id="model_v1",
            challenger_id="model_v2"
        )

        # Record prediction
        ab_testing.record_prediction(
            test_id=test_id,
            model_id="model_v1",
            document_id="doc_001",
            prediction={"doc_type": "invoice"},
            ground_truth={"doc_type": "invoice"},
            latency_ms=150.0,
            confidence=0.95
        )

        # Verify prediction was recorded
        predictions = ab_testing._get_predictions(test_id, "model_v1")
        assert len(predictions) == 1
        assert predictions[0]["document_id"] == "doc_001"
        assert predictions[0]["correct"] == True

    def test_metrics_calculation(self, ab_testing):
        """Test metrics calculation."""
        test_id = ab_testing.create_ab_test(
            champion_id="model_v1",
            challenger_id="model_v2"
        )

        # Record predictions with known accuracy
        for i in range(100):
            ab_testing.record_prediction(
                test_id=test_id,
                model_id="model_v1",
                document_id=f"doc_{i}",
                prediction={"doc_type": "invoice"},
                ground_truth={"doc_type": "invoice" if i < 90 else "purchase_order"},
                latency_ms=100.0 + i
            )

        metrics = ab_testing._calculate_metrics(
            ab_testing._get_predictions(test_id, "model_v1")
        )

        assert 0.89 <= metrics["accuracy"] <= 0.91  # 90% accuracy
        assert metrics["sample_count"] == 100
        assert metrics["avg_latency_ms"] > 100

    def test_statistical_significance(self, ab_testing):
        """Test statistical significance calculation."""
        test_id = ab_testing.create_ab_test(
            champion_id="model_v1",
            challenger_id="model_v2"
        )

        # Champion: 90% accuracy
        for i in range(1000):
            ab_testing.record_prediction(
                test_id=test_id,
                model_id="model_v1",
                document_id=f"champ_doc_{i}",
                prediction={"doc_type": "invoice"},
                ground_truth={"doc_type": "invoice" if i < 900 else "purchase_order"}
            )

        # Challenger: 92% accuracy (significantly better)
        for i in range(1000):
            ab_testing.record_prediction(
                test_id=test_id,
                model_id="model_v2",
                document_id=f"chal_doc_{i}",
                prediction={"doc_type": "invoice"},
                ground_truth={"doc_type": "invoice" if i < 920 else "purchase_order"}
            )

        # Evaluate
        result = ab_testing.evaluate_ab_test(test_id, min_samples=100)

        assert result.sample_count_champion >= 1000
        assert result.sample_count_challenger >= 1000
        assert result.winner == "challenger"
        assert result.p_value < 0.05  # Statistically significant
        assert result.recommendation == "promote"

    def test_insufficient_samples(self, ab_testing):
        """Test handling of insufficient samples."""
        test_id = ab_testing.create_ab_test(
            champion_id="model_v1",
            challenger_id="model_v2"
        )

        # Record only a few predictions
        for i in range(50):
            ab_testing.record_prediction(
                test_id=test_id,
                model_id="model_v1",
                document_id=f"doc_{i}",
                prediction={"doc_type": "invoice"},
                ground_truth={"doc_type": "invoice"}
            )

        # Evaluate with min_samples=1000
        result = ab_testing.evaluate_ab_test(test_id, min_samples=1000)

        assert result.winner == "inconclusive"
        assert result.recommendation == "continue_testing"

    def test_no_significant_difference(self, ab_testing):
        """Test when there's no significant difference."""
        test_id = ab_testing.create_ab_test(
            champion_id="model_v1",
            challenger_id="model_v2"
        )

        # Both models: 95% accuracy (same performance)
        for model_id in ["model_v1", "model_v2"]:
            for i in range(1000):
                ab_testing.record_prediction(
                    test_id=test_id,
                    model_id=model_id,
                    document_id=f"{model_id}_doc_{i}",
                    prediction={"doc_type": "invoice"},
                    ground_truth={"doc_type": "invoice" if i < 950 else "purchase_order"}
                )

        result = ab_testing.evaluate_ab_test(test_id, min_samples=100)

        # Should not be statistically significant
        assert result.p_value > 0.05
        assert result.winner in ["inconclusive", "champion"]

    def test_complete_ab_test(self, ab_testing):
        """Test completing an A/B test."""
        test_id = ab_testing.create_ab_test(
            champion_id="model_v1",
            challenger_id="model_v2"
        )

        ab_testing.complete_ab_test(
            test_id=test_id,
            winner="challenger",
            reason="Better performance"
        )

        test = ab_testing._get_test(test_id)
        assert test["status"] == "completed"
        assert test["winner"] == "challenger"

    def test_get_active_tests(self, ab_testing):
        """Test getting active tests."""
        # Create multiple tests
        id1 = ab_testing.create_ab_test("m1", "m2")
        id2 = ab_testing.create_ab_test("m3", "m4")

        active_tests = ab_testing.get_active_tests()
        assert len(active_tests) == 2

        # Complete one test
        ab_testing.complete_ab_test(id1, winner="champion")

        active_tests = ab_testing.get_active_tests()
        assert len(active_tests) == 1

    def test_test_summary(self, ab_testing):
        """Test getting test summary."""
        test_id = ab_testing.create_ab_test(
            champion_id="model_v1",
            challenger_id="model_v2"
        )

        # Record some predictions
        for i in range(10):
            ab_testing.record_prediction(
                test_id=test_id,
                model_id="model_v1",
                document_id=f"doc_{i}",
                prediction={"doc_type": "invoice"},
                ground_truth={"doc_type": "invoice"}
            )

        summary = ab_testing.get_test_summary(test_id)

        assert summary["test"]["id"] == test_id
        assert summary["champion"]["predictions"] == 10
        assert "metrics" in summary["champion"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
