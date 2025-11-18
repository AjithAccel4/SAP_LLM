"""
Tests for Tenant Manager.

Tests cover:
- Tenant registration and isolation
- Multi-tenant coordination
- Personalization
- Resource management
- A/B testing
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tempfile

from sap_llm.federated.tenant_manager import (
    TenantManager,
    TenantConfig,
    TenantModel,
    ABTestingManager
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.adapter = nn.Linear(2, 2)  # Adapter layer

    def forward(self, input_features=None, labels=None, **kwargs):
        x = self.linear(input_features)
        logits = self.adapter(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        class Output:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits

        return Output(loss=loss, logits=logits)


class TestTenantModel(unittest.TestCase):
    """Test TenantModel class."""

    def test_model_creation(self):
        """Test tenant model creation."""
        base_model = SimpleModel()
        config = TenantConfig(
            tenant_id="test_tenant",
            tenant_name="Test Tenant",
            data_path="./data/test",
            enable_personalization=True
        )

        tenant_model = TenantModel("test_tenant", base_model, config)

        self.assertIsNotNone(tenant_model.model)
        self.assertEqual(tenant_model.tenant_id, "test_tenant")

    def test_adapter_freezing(self):
        """Test that base model is frozen but adapters are trainable."""
        base_model = SimpleModel()
        config = TenantConfig(
            tenant_id="test_tenant",
            tenant_name="Test",
            data_path="./data/test",
            enable_personalization=True,
            freeze_base_model=True,
            adapter_layers=["adapter"]
        )

        tenant_model = TenantModel("test_tenant", base_model, config)

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in tenant_model.model.parameters()
            if p.requires_grad
        )

        # Should have some trainable parameters (adapters)
        self.assertGreater(trainable_params, 0)

        # Base linear layer should be frozen
        self.assertFalse(
            tenant_model.model.linear.weight.requires_grad
        )

        # Adapter should be trainable
        self.assertTrue(
            tenant_model.model.adapter.weight.requires_grad
        )

    def test_update_from_global(self):
        """Test updating from global model."""
        base_model = SimpleModel()
        config = TenantConfig(
            tenant_id="test_tenant",
            tenant_name="Test",
            data_path="./data/test",
            enable_personalization=True,
            adapter_layers=["adapter"]
        )

        tenant_model = TenantModel("test_tenant", base_model, config)

        # Create a modified global model state
        global_state = base_model.state_dict()
        global_state["linear.weight"] = torch.ones_like(global_state["linear.weight"])

        # Update from global
        tenant_model.update_from_global(global_state)

        # Linear layer should be updated
        self.assertTrue(torch.allclose(
            tenant_model.model.linear.weight,
            torch.ones_like(tenant_model.model.linear.weight)
        ))

    def test_checkpoint_saving(self):
        """Test checkpoint saving on improvement."""
        base_model = SimpleModel()
        config = TenantConfig(
            tenant_id="test_tenant",
            tenant_name="Test",
            data_path="./data/test"
        )

        tenant_model = TenantModel("test_tenant", base_model, config)

        # Save checkpoint with accuracy 0.8
        tenant_model.save_checkpoint(0.8)
        self.assertEqual(tenant_model.best_accuracy, 0.8)
        self.assertIsNotNone(tenant_model.best_model_state)

        # Save with lower accuracy (should not update)
        old_state = tenant_model.best_model_state
        tenant_model.save_checkpoint(0.7)
        self.assertEqual(tenant_model.best_accuracy, 0.8)
        self.assertEqual(tenant_model.best_model_state, old_state)

        # Save with higher accuracy (should update)
        tenant_model.save_checkpoint(0.9)
        self.assertEqual(tenant_model.best_accuracy, 0.9)

    def test_rollback(self):
        """Test rollback to best model."""
        base_model = SimpleModel()
        config = TenantConfig(
            tenant_id="test_tenant",
            tenant_name="Test",
            data_path="./data/test"
        )

        tenant_model = TenantModel("test_tenant", base_model, config)

        # Save best checkpoint
        best_state = tenant_model.model.state_dict().copy()
        tenant_model.save_checkpoint(0.9)

        # Modify model
        with torch.no_grad():
            for param in tenant_model.model.parameters():
                param.add_(1.0)

        # Rollback
        success = tenant_model.rollback_to_best()
        self.assertTrue(success)


class TestTenantManager(unittest.TestCase):
    """Test TenantManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TenantManager(output_dir=self.temp_dir)
        self.base_model = SimpleModel()

    def test_tenant_registration(self):
        """Test registering tenants."""
        config = TenantConfig(
            tenant_id="tenant_001",
            tenant_name="Acme Corp",
            data_path="./data/tenant_001"
        )

        tenant_id = self.manager.register_tenant(config, self.base_model)

        self.assertEqual(tenant_id, "tenant_001")
        self.assertIn("tenant_001", self.manager.tenants)
        self.assertIn("tenant_001", self.manager.tenant_configs)

    def test_duplicate_registration(self):
        """Test registering same tenant twice."""
        config = TenantConfig(
            tenant_id="tenant_001",
            tenant_name="Acme Corp",
            data_path="./data/tenant_001"
        )

        tenant_id1 = self.manager.register_tenant(config, self.base_model)
        tenant_id2 = self.manager.register_tenant(config, self.base_model)

        # Should return same ID
        self.assertEqual(tenant_id1, tenant_id2)

    def test_tenant_unregistration(self):
        """Test unregistering tenants."""
        config = TenantConfig(
            tenant_id="tenant_001",
            tenant_name="Acme Corp",
            data_path="./data/tenant_001"
        )

        self.manager.register_tenant(config, self.base_model)
        self.manager.unregister_tenant("tenant_001")

        self.assertNotIn("tenant_001", self.manager.tenants)

    def test_tenant_isolation(self):
        """Test that tenants have isolated models."""
        config1 = TenantConfig(
            tenant_id="tenant_001",
            tenant_name="Tenant 1",
            data_path="./data/tenant_001"
        )
        config2 = TenantConfig(
            tenant_id="tenant_002",
            tenant_name="Tenant 2",
            data_path="./data/tenant_002"
        )

        self.manager.register_tenant(config1, self.base_model)
        self.manager.register_tenant(config2, self.base_model)

        # Verify isolation
        isolation_status = self.manager.verify_tenant_isolation()

        self.assertTrue(isolation_status["tenant_001"])
        self.assertTrue(isolation_status["tenant_002"])

    def test_get_tenant_model(self):
        """Test retrieving tenant model."""
        config = TenantConfig(
            tenant_id="tenant_001",
            tenant_name="Tenant 1",
            data_path="./data/tenant_001"
        )

        self.manager.register_tenant(config, self.base_model)
        model = self.manager.get_tenant_model("tenant_001")

        self.assertIsNotNone(model)
        self.assertIsInstance(model, nn.Module)

    def test_update_tenant_from_global(self):
        """Test updating tenant from global model."""
        config = TenantConfig(
            tenant_id="tenant_001",
            tenant_name="Tenant 1",
            data_path="./data/tenant_001"
        )

        self.manager.register_tenant(config, self.base_model)

        # Create global state
        global_state = self.base_model.state_dict()
        global_state["linear.weight"] = torch.ones_like(global_state["linear.weight"])

        # Update tenant
        self.manager.update_tenant_from_global("tenant_001", global_state)

        # Verify update
        model = self.manager.get_tenant_model("tenant_001")
        # Note: adapter layers are not updated from global

    def test_evaluate_tenant(self):
        """Test tenant evaluation."""
        config = TenantConfig(
            tenant_id="tenant_001",
            tenant_name="Tenant 1",
            data_path="./data/tenant_001"
        )

        self.manager.register_tenant(config, self.base_model)

        # Create dummy eval data
        features = torch.randn(20, 10)
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(features, labels)

        def collate_fn(batch):
            features = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
            return {"input_features": features, "labels": labels}

        eval_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        # Evaluate
        metrics = self.manager.evaluate_tenant("tenant_001", eval_loader)

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)

    def test_resource_usage(self):
        """Test resource usage tracking."""
        config = TenantConfig(
            tenant_id="tenant_001",
            tenant_name="Tenant 1",
            data_path="./data/tenant_001"
        )

        self.manager.register_tenant(config, self.base_model)

        usage = self.manager.get_resource_usage("tenant_001")

        self.assertIn("memory_gb", usage)
        self.assertIn("num_parameters", usage)
        self.assertIn("trainable_parameters", usage)

        self.assertGreater(usage["num_parameters"], 0)

    def test_audit_trail(self):
        """Test audit trail logging."""
        config = TenantConfig(
            tenant_id="tenant_001",
            tenant_name="Tenant 1",
            data_path="./data/tenant_001"
        )

        self.manager.register_tenant(config, self.base_model)

        # Get audit trail
        audit_trail = self.manager.get_audit_trail("tenant_001")

        # Should have registration event
        self.assertGreater(len(audit_trail), 0)

        # Check event structure
        event = audit_trail[0]
        self.assertIn("timestamp", event)
        self.assertIn("action", event)
        self.assertIn("tenant_id", event)


class TestABTestingManager(unittest.TestCase):
    """Test A/B testing functionality."""

    def test_group_assignment(self):
        """Test consistent group assignment."""
        ab_manager = ABTestingManager("tenant_001", control_ratio=0.2)

        # Same sample should always get same assignment
        sample_id = "sample_123"
        group1 = ab_manager.assign_to_group(sample_id)
        group2 = ab_manager.assign_to_group(sample_id)

        self.assertEqual(group1, group2)

    def test_control_ratio(self):
        """Test control group ratio approximation."""
        ab_manager = ABTestingManager("tenant_001", control_ratio=0.2)

        # Test many samples
        num_samples = 1000
        control_count = 0

        for i in range(num_samples):
            group = ab_manager.assign_to_group(f"sample_{i}")
            if group == "control":
                control_count += 1

        # Should be approximately 20% (with some tolerance)
        ratio = control_count / num_samples
        self.assertAlmostEqual(ratio, 0.2, delta=0.05)

    def test_metrics_recording(self):
        """Test metrics recording for both groups."""
        ab_manager = ABTestingManager("tenant_001")

        # Record metrics
        ab_manager.record_metric("control", {"accuracy": 0.85, "loss": 0.5})
        ab_manager.record_metric("treatment", {"accuracy": 0.90, "loss": 0.4})

        self.assertEqual(len(ab_manager.control_metrics), 1)
        self.assertEqual(len(ab_manager.treatment_metrics), 1)

    def test_results_calculation(self):
        """Test A/B test results calculation."""
        ab_manager = ABTestingManager("tenant_001")

        # Add multiple metrics
        for _ in range(5):
            ab_manager.record_metric("control", {"accuracy": 0.85})
            ab_manager.record_metric("treatment", {"accuracy": 0.90})

        results = ab_manager.get_test_results()

        self.assertEqual(results["status"], "completed")
        self.assertIn("control_avg", results)
        self.assertIn("treatment_avg", results)
        self.assertIn("improvement", results)

        # Check improvement calculation
        self.assertGreater(results["improvement"]["accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
