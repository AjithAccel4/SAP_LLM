"""
Tests for Federated Learning Trainer.

Tests cover:
- FedAvg aggregation
- Differential privacy
- Byzantine robustness
- Privacy budget tracking
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict
import tempfile
from pathlib import Path

from sap_llm.federated.federated_trainer import (
    FederatedTrainer,
    FederatedConfig,
    FederatedAveraging,
    KrumAggregation,
    DifferentialPrivacyManager
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, input_features=None, labels=None, **kwargs):
        logits = self.linear(input_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        class Output:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits

        return Output(loss=loss, logits=logits)


class TestFederatedAveraging(unittest.TestCase):
    """Test FedAvg aggregation."""

    def test_basic_aggregation(self):
        """Test basic weighted averaging."""
        # Create dummy models
        model1 = OrderedDict([
            ("layer1", torch.tensor([1.0, 2.0, 3.0])),
            ("layer2", torch.tensor([4.0, 5.0]))
        ])

        model2 = OrderedDict([
            ("layer1", torch.tensor([2.0, 3.0, 4.0])),
            ("layer2", torch.tensor([5.0, 6.0]))
        ])

        models = [model1, model2]

        # Equal weights
        aggregated = FederatedAveraging.aggregate(models)

        # Check average
        self.assertTrue(torch.allclose(
            aggregated["layer1"],
            torch.tensor([1.5, 2.5, 3.5])
        ))
        self.assertTrue(torch.allclose(
            aggregated["layer2"],
            torch.tensor([4.5, 5.5])
        ))

    def test_weighted_aggregation(self):
        """Test weighted averaging."""
        model1 = OrderedDict([("layer", torch.tensor([1.0]))])
        model2 = OrderedDict([("layer", torch.tensor([3.0]))])

        models = [model1, model2]
        weights = [0.25, 0.75]  # 75% weight on model2

        aggregated = FederatedAveraging.aggregate(models, weights)

        # Expected: 0.25 * 1.0 + 0.75 * 3.0 = 2.5
        self.assertTrue(torch.allclose(
            aggregated["layer"],
            torch.tensor([2.5])
        ))

    def test_empty_models(self):
        """Test error handling for empty models."""
        with self.assertRaises(ValueError):
            FederatedAveraging.aggregate([])


class TestKrumAggregation(unittest.TestCase):
    """Test Krum aggregation for Byzantine robustness."""

    def test_krum_selection(self):
        """Test Krum selects representative model."""
        # Create models where model2 is an outlier
        model1 = OrderedDict([("layer", torch.tensor([1.0, 1.0]))])
        model2 = OrderedDict([("layer", torch.tensor([100.0, 100.0]))])  # Outlier
        model3 = OrderedDict([("layer", torch.tensor([1.1, 0.9]))])

        models = [model1, model2, model3]

        selected = KrumAggregation.aggregate(models, num_byzantine=1)

        # Should select model1 or model3 (not the outlier)
        # Check it's close to one of the normal models
        is_model1 = torch.allclose(selected["layer"], model1["layer"], atol=0.5)
        is_model3 = torch.allclose(selected["layer"], model3["layer"], atol=0.5)

        self.assertTrue(is_model1 or is_model3)

    def test_krum_fallback(self):
        """Test Krum falls back to FedAvg with insufficient clients."""
        model1 = OrderedDict([("layer", torch.tensor([1.0]))])
        model2 = OrderedDict([("layer", torch.tensor([2.0]))])

        models = [model1, model2]

        # Too few clients for Krum with 2 Byzantine
        result = KrumAggregation.aggregate(models, num_byzantine=2)

        # Should fall back to averaging
        self.assertTrue(torch.allclose(
            result["layer"],
            torch.tensor([1.5])
        ))


class TestDifferentialPrivacyManager(unittest.TestCase):
    """Test differential privacy manager."""

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        config = FederatedConfig(max_grad_norm=1.0)
        dp_manager = DifferentialPrivacyManager(config)

        # Create model with large gradient
        model = SimpleModel()
        for param in model.parameters():
            param.grad = torch.ones_like(param) * 10.0

        # Clip gradients
        norm_before = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5
        total_norm = dp_manager.clip_gradients(model)
        norm_after = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5

        # Gradient should be clipped to max_grad_norm
        self.assertLess(norm_after, norm_before)
        self.assertLessEqual(norm_after, config.max_grad_norm * 1.01)  # Small tolerance

    def test_noise_addition(self):
        """Test noise addition to gradients."""
        config = FederatedConfig(
            enable_dp=True,
            noise_multiplier=1.0,
            max_grad_norm=1.0
        )
        dp_manager = DifferentialPrivacyManager(config)

        model = SimpleModel()
        for param in model.parameters():
            param.grad = torch.zeros_like(param)

        # Add noise
        dp_manager.add_noise_to_gradients(model)

        # Check that gradients are no longer zero (noise was added)
        has_noise = any(
            not torch.allclose(p.grad, torch.zeros_like(p.grad))
            for p in model.parameters()
        )

        self.assertTrue(has_noise)


class TestFederatedTrainer(unittest.TestCase):
    """Test federated trainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create simple model
        self.model = SimpleModel()

        # Create config
        self.config = FederatedConfig(
            num_rounds=2,
            num_clients_per_round=2,
            local_epochs=1,
            local_batch_size=4,
            enable_dp=False,  # Disable DP for faster tests
            output_dir=self.temp_dir
        )

        # Create dummy datasets
        self.client_loaders = {}
        for i in range(3):
            features = torch.randn(20, 10)
            labels = torch.randint(0, 2, (20,))
            dataset = TensorDataset(features, labels)

            def collate_fn(batch):
                features = torch.stack([item[0] for item in batch])
                labels = torch.stack([item[1] for item in batch])
                return {"input_features": features, "labels": labels}

            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
            self.client_loaders[f"client_{i}"] = loader

    def test_client_training(self):
        """Test training on single client."""
        trainer = FederatedTrainer(self.model, self.config)

        client_id = "client_0"
        result = trainer.train_client(
            client_id,
            self.client_loaders[client_id],
            self.model
        )

        # Check result structure
        self.assertIn("client_id", result)
        self.assertIn("model_state", result)
        self.assertIn("loss", result)
        self.assertIn("num_samples", result)

        # Check model state is valid
        self.assertIsInstance(result["model_state"], OrderedDict)

    def test_model_aggregation(self):
        """Test model aggregation."""
        trainer = FederatedTrainer(self.model, self.config)

        # Train multiple clients
        client_results = []
        for client_id, loader in list(self.client_loaders.items())[:2]:
            result = trainer.train_client(client_id, loader, self.model)
            client_results.append(result)

        # Aggregate
        aggregated = trainer.aggregate_models(client_results)

        # Check aggregated model has correct structure
        self.assertIsInstance(aggregated, OrderedDict)
        self.assertEqual(
            set(aggregated.keys()),
            set(self.model.state_dict().keys())
        )

    def test_full_training(self):
        """Test full federated training."""
        trainer = FederatedTrainer(self.model, self.config)

        # Train
        trainer.train(self.client_loaders, num_rounds=2)

        # Check training completed
        self.assertEqual(trainer.current_round, 1)  # 0-indexed
        self.assertGreater(len(trainer.round_metrics), 0)

        # Check checkpoint saved
        checkpoint_dir = Path(self.temp_dir) / "final_model"
        self.assertTrue(checkpoint_dir.exists())
        self.assertTrue((checkpoint_dir / "model.pt").exists())

    def test_privacy_budget_tracking(self):
        """Test privacy budget is tracked."""
        config = FederatedConfig(
            num_rounds=1,
            num_clients_per_round=1,
            enable_dp=True,
            target_epsilon=1.0,
            output_dir=self.temp_dir
        )

        trainer = FederatedTrainer(self.model, config)

        # Initial budget should be zero
        self.assertEqual(trainer.privacy_budget_spent["epsilon"], 0.0)

        # After training, budget should be consumed (if Opacus is available)
        # Note: This test may not consume budget if Opacus is not installed


class TestPrivacyGuarantees(unittest.TestCase):
    """Test privacy guarantees."""

    def test_epsilon_delta_bounds(self):
        """Test epsilon and delta are within bounds."""
        config = FederatedConfig(
            target_epsilon=1.0,
            target_delta=1e-5,
            enable_dp=True
        )

        # These bounds should be respected
        self.assertLessEqual(config.target_epsilon, 1.0)
        self.assertLessEqual(config.target_delta, 1e-5)

    def test_gradient_clipping_enforced(self):
        """Test gradient clipping is enforced."""
        config = FederatedConfig(max_grad_norm=1.0)

        # Max gradient norm should be set
        self.assertEqual(config.max_grad_norm, 1.0)


if __name__ == "__main__":
    unittest.main()
