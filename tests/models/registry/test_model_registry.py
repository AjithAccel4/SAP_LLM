"""
Unit tests for Model Registry.

Tests cover:
- Model registration
- Versioning
- Champion/Challenger promotion
- Rollback
- Statistics
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch
import torch.nn as nn

from sap_llm.models.registry import ModelRegistry, ModelVersion
from sap_llm.models.registry.storage_backend import LocalStorageBackend


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
def registry(temp_dir):
    """Create model registry for tests."""
    return ModelRegistry(
        storage_backend=LocalStorageBackend(temp_dir),
        db_path=str(Path(temp_dir) / "registry.db")
    )


class TestModelVersion:
    """Test ModelVersion class."""

    def test_version_creation(self):
        """Test version creation."""
        v = ModelVersion(1, 2, 3)
        assert str(v) == "1.2.3"
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_version_from_string(self):
        """Test parsing version from string."""
        v = ModelVersion.from_string("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_version_comparison(self):
        """Test version comparison."""
        v1 = ModelVersion(1, 0, 0)
        v2 = ModelVersion(1, 0, 1)
        v3 = ModelVersion(1, 1, 0)
        v4 = ModelVersion(2, 0, 0)

        assert v2 > v1
        assert v3 > v2
        assert v4 > v3
        assert v1 < v2 < v3 < v4

    def test_version_increment(self):
        """Test version incrementing."""
        v = ModelVersion(1, 2, 3)

        v_patch = v.increment_patch()
        assert str(v_patch) == "1.2.4"

        v_minor = v.increment_minor()
        assert str(v_minor) == "1.3.0"

        v_major = v.increment_major()
        assert str(v_major) == "2.0.0"

    def test_version_compatibility(self):
        """Test version compatibility."""
        v1 = ModelVersion(1, 0, 0)
        v2 = ModelVersion(1, 5, 3)
        v3 = ModelVersion(2, 0, 0)

        assert v1.is_compatible_with(v2)
        assert not v1.is_compatible_with(v3)


class TestModelRegistry:
    """Test ModelRegistry class."""

    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert registry is not None
        stats = registry.get_statistics()
        assert stats["total_models"] == 0

    def test_model_registration(self, registry):
        """Test model registration."""
        model = DummyModel()

        model_id = registry.register_model(
            model=model,
            name="test_model",
            model_type="test",
            metrics={"accuracy": 0.95}
        )

        assert model_id is not None
        assert "test_model" in model_id

        # Verify model exists
        model_info = registry.get_model(model_id)
        assert model_info["name"] == "test_model"
        assert model_info["model_type"] == "test"

    def test_auto_versioning(self, registry):
        """Test automatic version incrementing."""
        model = DummyModel()

        # Register first version
        id1 = registry.register_model(
            model=model,
            name="test_model",
            model_type="test"
        )

        model1 = registry.get_model(id1)
        assert model1["version"] == "1.0.0"

        # Register second version (auto-increment)
        id2 = registry.register_model(
            model=model,
            name="test_model",
            model_type="test"
        )

        model2 = registry.get_model(id2)
        assert model2["version"] == "1.0.1"

    def test_champion_promotion(self, registry):
        """Test champion promotion."""
        model = DummyModel()

        # Register model
        model_id = registry.register_model(
            model=model,
            name="test_model",
            model_type="test",
            metrics={"accuracy": 0.95}
        )

        # Promote to champion
        registry.promote_to_champion(
            model_id=model_id,
            reason="Initial champion"
        )

        # Verify champion
        champion = registry.get_champion("test")
        assert champion is not None
        assert champion["id"] == model_id
        assert champion["status"] == "champion"

    def test_champion_replacement(self, registry):
        """Test replacing champion."""
        model = DummyModel()

        # Register and promote first champion
        id1 = registry.register_model(
            model=model,
            name="test_model",
            model_type="test",
            metrics={"accuracy": 0.95}
        )
        registry.promote_to_champion(id1, reason="First champion")

        # Register and promote second champion
        id2 = registry.register_model(
            model=model,
            name="test_model",
            model_type="test",
            metrics={"accuracy": 0.97}
        )
        registry.promote_to_champion(id2, reason="Better model")

        # Verify new champion
        champion = registry.get_champion("test")
        assert champion["id"] == id2

        # Verify old champion is archived
        old_model = registry.get_model(id1)
        assert old_model["status"] == "archived"

    def test_rollback(self, registry):
        """Test rollback to previous champion."""
        model = DummyModel()

        # Create two champions
        id1 = registry.register_model(
            model=model,
            name="test_model",
            model_type="test"
        )
        registry.promote_to_champion(id1)

        id2 = registry.register_model(
            model=model,
            name="test_model",
            model_type="test"
        )
        registry.promote_to_champion(id2)

        # Rollback
        restored_id = registry.rollback_to_previous_champion(
            model_type="test",
            reason="Performance issue"
        )

        assert restored_id == id1

        # Verify restoration
        champion = registry.get_champion("test")
        assert champion["id"] == id1

    def test_challenger_promotion(self, registry):
        """Test promoting to challenger status."""
        model = DummyModel()

        model_id = registry.register_model(
            model=model,
            name="test_model",
            model_type="test"
        )

        registry.promote_to_challenger(model_id)

        model_info = registry.get_model(model_id)
        assert model_info["status"] == "challenger"

    def test_model_listing(self, registry):
        """Test listing models."""
        model = DummyModel()

        # Register multiple models
        for i in range(5):
            registry.register_model(
                model=model,
                name=f"model_{i}",
                model_type="test"
            )

        # List all models
        models = registry.list_models()
        assert len(models) == 5

        # List with filter
        models_test = registry.list_models(model_type="test")
        assert len(models_test) == 5

    def test_promotion_history(self, registry):
        """Test promotion history tracking."""
        model = DummyModel()

        id1 = registry.register_model(
            model=model,
            name="test_model",
            model_type="test"
        )
        registry.promote_to_champion(id1, reason="First")

        id2 = registry.register_model(
            model=model,
            name="test_model",
            model_type="test"
        )
        registry.promote_to_champion(id2, reason="Second")

        # Check history
        history = registry.get_promotion_history()
        assert len(history) >= 2

        # Check model-specific history
        model_history = registry.get_promotion_history(model_id=id1)
        assert len(model_history) >= 1

    def test_statistics(self, registry):
        """Test registry statistics."""
        model = DummyModel()

        # Register models
        id1 = registry.register_model(
            model=model,
            name="model_1",
            model_type="test"
        )
        registry.promote_to_champion(id1)

        id2 = registry.register_model(
            model=model,
            name="model_2",
            model_type="test"
        )
        registry.promote_to_challenger(id2)

        # Get statistics
        stats = registry.get_statistics()

        assert stats["total_models"] == 2
        assert stats["by_status"]["champion"] == 1
        assert stats["by_status"]["challenger"] == 1
        assert "storage_size_mb" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
