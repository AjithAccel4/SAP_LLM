"""
Integration tests for Continuous Learning Pipeline.

Tests full end-to-end workflows:
- Complete learning cycle
- Drift detection → Retraining → A/B Testing → Promotion
- Rollback scenarios
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn

from sap_llm.training.continuous_learner import ContinuousLearner, LearningConfig
from sap_llm.models.registry import ModelRegistry


class DummyModel(nn.Module):
    """Dummy model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def config():
    """Create test configuration."""
    return LearningConfig(
        drift_threshold_psi=0.25,
        min_improvement_threshold=0.02,
        ab_test_min_samples=100,  # Lower for testing
        enable_auto_retraining=True,
        enable_auto_promotion=True,
        enable_auto_rollback=True
    )


@pytest.fixture
def learner(temp_dir, config):
    """Create continuous learner for tests."""
    registry = ModelRegistry(db_path=str(Path(temp_dir) / "registry.db"))
    return ContinuousLearner(
        config=config,
        model_registry=registry
    )


class TestContinuousLearningIntegration:
    """Integration tests for continuous learning pipeline."""

    def test_learner_initialization(self, learner):
        """Test learner initialization."""
        assert learner is not None
        assert learner.model_registry is not None
        assert learner.drift_detector is not None
        assert learner.ab_testing is not None
        assert learner.promoter is not None
        assert learner.scheduler is not None

    def test_statistics_collection(self, learner):
        """Test statistics collection."""
        stats = learner.get_statistics()

        assert "learning_cycles_run" in stats
        assert "drift_detected_count" in stats
        assert "retraining_triggered" in stats
        assert "ab_tests_created" in stats
        assert "promotions" in stats
        assert "rollbacks" in stats
        assert "model_registry" in stats

    def test_champion_model_management(self, learner):
        """Test getting champion model."""
        # Initially no champion
        champion = learner.get_champion_model("vision_encoder")
        assert champion is None

        # Register and promote a model
        model = DummyModel()
        model_id = learner.model_registry.register_model(
            model=model,
            name="vision_encoder",
            model_type="vision_encoder",
            metrics={"accuracy": 0.95}
        )
        learner.model_registry.promote_to_champion(model_id)

        # Now should have champion
        champion = learner.get_champion_model("vision_encoder")
        assert champion is not None
        assert champion["id"] == model_id

    def test_ab_test_creation(self, learner):
        """Test creating A/B test."""
        model = DummyModel()

        # Create two models
        id1 = learner.model_registry.register_model(
            model=model,
            name="test_model",
            model_type="test",
            metrics={"accuracy": 0.95}
        )

        id2 = learner.model_registry.register_model(
            model=model,
            name="test_model",
            model_type="test",
            metrics={"accuracy": 0.97}
        )

        # Create A/B test
        test_id = learner.create_ab_test(
            champion_id=id1,
            challenger_id=id2,
            traffic_split=0.1
        )

        assert test_id is not None
        assert learner.stats["ab_tests_created"] == 1

    def test_prediction_routing_and_recording(self, learner):
        """Test prediction routing and recording."""
        model = DummyModel()

        # Create models and A/B test
        id1 = learner.model_registry.register_model(
            model=model, name="m1", model_type="test"
        )
        id2 = learner.model_registry.register_model(
            model=model, name="m1", model_type="test"
        )

        test_id = learner.create_ab_test(id1, id2)

        # Route prediction
        model_id = learner.route_prediction(test_id)
        assert model_id in [id1, id2]

        # Record prediction
        learner.record_prediction(
            test_id=test_id,
            model_id=model_id,
            document_id="doc_001",
            prediction={"doc_type": "invoice"},
            ground_truth={"doc_type": "invoice"},
            latency_ms=150.0
        )

        # Verify recorded
        summary = learner.ab_testing.get_test_summary(test_id)
        total_predictions = (
            summary["champion"]["predictions"] +
            summary["challenger"]["predictions"]
        )
        assert total_predictions == 1

    def test_rollback_functionality(self, learner):
        """Test manual rollback."""
        model = DummyModel()

        # Create two champions
        id1 = learner.model_registry.register_model(
            model=model,
            name="test_model",
            model_type="test"
        )
        learner.model_registry.promote_to_champion(id1)

        id2 = learner.model_registry.register_model(
            model=model,
            name="test_model",
            model_type="test"
        )
        learner.model_registry.promote_to_champion(id2)

        # Rollback
        result = learner.rollback(
            model_type="test",
            reason="Testing rollback"
        )

        assert result["success"]
        assert result["restored_champion_id"] == id1
        assert learner.stats["rollbacks"] == 1

    def test_full_learning_cycle_simulation(self, learner):
        """Test simulated full learning cycle."""
        # This would normally trigger drift detection, retraining, etc.
        # For unit test, we just verify the method runs
        result = learner.run_learning_cycle(model_type="vision_encoder")

        assert "timestamp" in result
        assert "model_type" in result
        assert "steps" in result

        assert learner.stats["learning_cycles_run"] == 1

    def test_concurrent_ab_tests(self, learner):
        """Test multiple concurrent A/B tests."""
        model = DummyModel()

        # Create multiple tests
        test_ids = []
        for i in range(3):
            id1 = learner.model_registry.register_model(
                model=model,
                name=f"model_{i}",
                model_type=f"type_{i}"
            )
            id2 = learner.model_registry.register_model(
                model=model,
                name=f"model_{i}",
                model_type=f"type_{i}"
            )

            test_id = learner.create_ab_test(id1, id2)
            test_ids.append(test_id)

        # Verify all active
        active_tests = learner.ab_testing.get_active_tests()
        assert len(active_tests) == 3

    def test_statistics_persistence(self, learner):
        """Test that statistics are tracked correctly."""
        initial_stats = learner.get_statistics()

        # Perform some operations
        model = DummyModel()
        id1 = learner.model_registry.register_model(
            model=model, name="m1", model_type="test"
        )
        id2 = learner.model_registry.register_model(
            model=model, name="m1", model_type="test"
        )

        learner.create_ab_test(id1, id2)

        # Check stats updated
        updated_stats = learner.get_statistics()
        assert updated_stats["ab_tests_created"] > initial_stats["ab_tests_created"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_rollback(self, learner):
        """Test rollback with no previous champion."""
        with pytest.raises(ValueError):
            learner.rollback(
                model_type="nonexistent",
                reason="Test"
            )

    def test_invalid_ab_test(self, learner):
        """Test invalid A/B test creation."""
        # This should work but models don't exist in registry
        # In production, might want to validate model IDs
        test_id = learner.create_ab_test(
            champion_id="invalid_id_1",
            challenger_id="invalid_id_2"
        )
        # Should still create test (validation happens later)
        assert test_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
