"""
Multi-Tenant Manager for Federated Learning.

Manages multiple tenants training models simultaneously with:
- Tenant isolation and data privacy
- Per-tenant model personalization
- Resource allocation and scheduling
- Performance tracking and A/B testing
- Automatic rollback on degradation
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import copy
import threading
from queue import Queue

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Configuration for individual tenant."""

    tenant_id: str
    tenant_name: str

    # Data settings
    data_path: str
    validation_split: float = 0.2

    # Training settings
    local_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4

    # Personalization settings
    enable_personalization: bool = True
    adapter_layers: List[str] = field(default_factory=lambda: ["adapter"])
    freeze_base_model: bool = True

    # Performance thresholds
    min_accuracy: float = 0.85
    max_degradation: float = 0.05  # Max acceptable accuracy drop

    # Resource limits
    max_memory_gb: float = 16.0
    max_training_time_hours: float = 24.0

    # A/B testing
    enable_ab_testing: bool = True
    control_group_ratio: float = 0.2


class TenantModel:
    """Wrapper for tenant-specific model with personalization."""

    def __init__(
        self,
        tenant_id: str,
        base_model: nn.Module,
        config: TenantConfig
    ):
        """Initialize tenant model."""
        self.tenant_id = tenant_id
        self.config = config

        # Clone base model
        self.model = copy.deepcopy(base_model)

        # Add personalization layers if enabled
        if config.enable_personalization:
            self._add_adapter_layers()

        # Freeze base model if configured
        if config.freeze_base_model:
            self._freeze_base_model()

        # Performance tracking
        self.performance_history = []
        self.best_accuracy = 0.0
        self.best_model_state = None

        logger.info(f"Initialized model for tenant: {tenant_id}")

    def _add_adapter_layers(self):
        """Add adapter layers for personalization."""
        # Add adapter modules to transformer layers
        # This is a simplified implementation
        # In production, use proper adapter architecture (e.g., LoRA, Adapter modules)

        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in self.config.adapter_layers):
                # Mark as trainable adapter
                for param in module.parameters():
                    param.requires_grad = True

        logger.info(f"Added adapter layers for {self.tenant_id}")

    def _freeze_base_model(self):
        """Freeze base model parameters."""
        for name, param in self.model.named_parameters():
            if not any(layer in name for layer in self.config.adapter_layers):
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info(
            f"Frozen base model for {self.tenant_id}: "
            f"{trainable_params}/{total_params} trainable parameters"
        )

    def update_from_global(self, global_model_state: Dict):
        """Update base model from global aggregation."""
        # Update only non-adapter parameters
        current_state = self.model.state_dict()

        for key, value in global_model_state.items():
            # Skip adapter layers
            if not any(layer in key for layer in self.config.adapter_layers):
                current_state[key] = value

        self.model.load_state_dict(current_state)
        logger.info(f"Updated {self.tenant_id} from global model")

    def save_checkpoint(self, accuracy: float):
        """Save checkpoint if performance improved."""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            logger.info(
                f"Saved best model for {self.tenant_id}: "
                f"accuracy={accuracy:.4f}"
            )

    def rollback_to_best(self):
        """Rollback to best performing model."""
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info(
                f"Rolled back {self.tenant_id} to best model: "
                f"accuracy={self.best_accuracy:.4f}"
            )
            return True
        return False

    def track_performance(self, metrics: Dict[str, float]):
        """Track performance metrics."""
        metrics["timestamp"] = datetime.now().isoformat()
        self.performance_history.append(metrics)

        # Check for degradation
        if len(self.performance_history) >= 2:
            current_acc = metrics.get("accuracy", 0.0)
            previous_acc = self.performance_history[-2].get("accuracy", 0.0)
            degradation = previous_acc - current_acc

            if degradation > self.config.max_degradation:
                logger.warning(
                    f"Performance degradation detected for {self.tenant_id}: "
                    f"{degradation:.4f}"
                )
                return False

        return True


class ABTestingManager:
    """Manages A/B testing for tenant models."""

    def __init__(self, tenant_id: str, control_ratio: float = 0.2):
        """Initialize A/B testing manager."""
        self.tenant_id = tenant_id
        self.control_ratio = control_ratio

        # Track control and treatment groups
        self.control_model = None
        self.treatment_model = None

        # Performance metrics
        self.control_metrics = []
        self.treatment_metrics = []

    def setup_test(self, base_model: nn.Module, new_model: nn.Module):
        """Setup A/B test with control and treatment models."""
        self.control_model = base_model
        self.treatment_model = new_model
        logger.info(f"A/B test setup for {self.tenant_id}")

    def assign_to_group(self, sample_id: str) -> str:
        """Assign sample to control or treatment group."""
        # Use hash-based assignment for consistency
        import hashlib
        hash_val = int(hashlib.md5(sample_id.encode()).hexdigest(), 16)
        return "control" if (hash_val % 100) < (self.control_ratio * 100) else "treatment"

    def record_metric(self, group: str, metrics: Dict[str, float]):
        """Record metrics for group."""
        if group == "control":
            self.control_metrics.append(metrics)
        else:
            self.treatment_metrics.append(metrics)

    def get_test_results(self) -> Dict[str, Any]:
        """Get A/B test results."""
        if not self.control_metrics or not self.treatment_metrics:
            return {"status": "insufficient_data"}

        # Calculate average metrics
        control_avg = {
            key: sum(m[key] for m in self.control_metrics) / len(self.control_metrics)
            for key in self.control_metrics[0].keys()
        }

        treatment_avg = {
            key: sum(m[key] for m in self.treatment_metrics) / len(self.treatment_metrics)
            for key in self.treatment_metrics[0].keys()
        }

        # Calculate improvement
        improvement = {
            key: ((treatment_avg[key] - control_avg.get(key, 0)) / (control_avg.get(key, 1e-6)) * 100)
            for key in treatment_avg.keys()
        }

        return {
            "status": "completed",
            "control_avg": control_avg,
            "treatment_avg": treatment_avg,
            "improvement": improvement,
            "num_control_samples": len(self.control_metrics),
            "num_treatment_samples": len(self.treatment_metrics)
        }


class TenantManager:
    """
    Manages multiple tenants in federated learning.

    Features:
    - Tenant isolation and privacy
    - Per-tenant personalization
    - Resource management
    - Performance tracking
    - A/B testing
    - Automatic rollback
    """

    def __init__(self, output_dir: str = "./models/tenants"):
        """Initialize tenant manager."""
        self.tenants: Dict[str, TenantModel] = {}
        self.tenant_configs: Dict[str, TenantConfig] = {}
        self.ab_tests: Dict[str, ABTestingManager] = {}

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resource tracking
        self.resource_usage: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Access control
        self.access_locks: Dict[str, threading.Lock] = {}

        # Audit trail
        self.audit_log = []

        logger.info("TenantManager initialized")

    def register_tenant(
        self,
        config: TenantConfig,
        base_model: nn.Module
    ) -> str:
        """
        Register new tenant.

        Args:
            config: Tenant configuration
            base_model: Base model to personalize

        Returns:
            Tenant ID
        """
        tenant_id = config.tenant_id

        if tenant_id in self.tenants:
            logger.warning(f"Tenant {tenant_id} already registered")
            return tenant_id

        # Create tenant model
        tenant_model = TenantModel(tenant_id, base_model, config)

        # Register
        self.tenants[tenant_id] = tenant_model
        self.tenant_configs[tenant_id] = config
        self.access_locks[tenant_id] = threading.Lock()

        # Setup A/B testing if enabled
        if config.enable_ab_testing:
            self.ab_tests[tenant_id] = ABTestingManager(
                tenant_id,
                config.control_group_ratio
            )

        # Audit log
        self._log_audit("tenant_registered", tenant_id, {
            "tenant_name": config.tenant_name,
            "personalization_enabled": config.enable_personalization
        })

        logger.info(f"Registered tenant: {tenant_id} ({config.tenant_name})")

        return tenant_id

    def unregister_tenant(self, tenant_id: str):
        """Unregister tenant and cleanup resources."""
        if tenant_id not in self.tenants:
            logger.warning(f"Tenant {tenant_id} not found")
            return

        # Remove tenant
        del self.tenants[tenant_id]
        del self.tenant_configs[tenant_id]
        del self.access_locks[tenant_id]

        if tenant_id in self.ab_tests:
            del self.ab_tests[tenant_id]

        # Audit log
        self._log_audit("tenant_unregistered", tenant_id, {})

        logger.info(f"Unregistered tenant: {tenant_id}")

    def get_tenant_model(self, tenant_id: str) -> Optional[nn.Module]:
        """Get model for specific tenant."""
        if tenant_id not in self.tenants:
            logger.warning(f"Tenant {tenant_id} not found")
            return None

        with self.access_locks[tenant_id]:
            return self.tenants[tenant_id].model

    def get_tenant_data_loader(
        self,
        tenant_id: str,
        dataset: Dataset
    ) -> DataLoader:
        """Create data loader for tenant."""
        if tenant_id not in self.tenant_configs:
            raise ValueError(f"Tenant {tenant_id} not found")

        config = self.tenant_configs[tenant_id]

        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def update_tenant_from_global(
        self,
        tenant_id: str,
        global_model_state: Dict
    ):
        """Update tenant model from global aggregation."""
        if tenant_id not in self.tenants:
            logger.warning(f"Tenant {tenant_id} not found")
            return

        with self.access_locks[tenant_id]:
            self.tenants[tenant_id].update_from_global(global_model_state)

        self._log_audit("model_updated", tenant_id, {
            "source": "global_aggregation"
        })

    def evaluate_tenant(
        self,
        tenant_id: str,
        eval_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate tenant model."""
        if tenant_id not in self.tenants:
            logger.warning(f"Tenant {tenant_id} not found")
            return {}

        model = self.tenants[tenant_id].model
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in eval_loader:
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()

                if hasattr(outputs, 'logits') and 'labels' in batch:
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)

        metrics = {
            "loss": total_loss / len(eval_loader),
            "accuracy": correct / total if total > 0 else 0.0
        }

        # Track performance
        self.tenants[tenant_id].track_performance(metrics)

        # Check for degradation and rollback if needed
        if not self.tenants[tenant_id].track_performance(metrics):
            logger.warning(f"Degradation detected for {tenant_id}, rolling back")
            self.tenants[tenant_id].rollback_to_best()

        return metrics

    def get_all_tenant_models(self) -> Dict[str, nn.Module]:
        """Get all tenant models for aggregation."""
        return {
            tenant_id: tenant.model
            for tenant_id, tenant in self.tenants.items()
        }

    def save_tenant_checkpoint(self, tenant_id: str, metrics: Dict[str, float]):
        """Save checkpoint for tenant."""
        if tenant_id not in self.tenants:
            return

        # Save model checkpoint
        checkpoint_dir = self.output_dir / tenant_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        tenant = self.tenants[tenant_id]
        tenant.save_checkpoint(metrics.get("accuracy", 0.0))

        # Save to disk
        torch.save(
            tenant.model.state_dict(),
            checkpoint_dir / "model.pt"
        )

        # Save metadata
        metadata = {
            "tenant_id": tenant_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "best_accuracy": tenant.best_accuracy
        }

        with open(checkpoint_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved checkpoint for {tenant_id}")

    def get_resource_usage(self, tenant_id: str) -> Dict[str, float]:
        """Get resource usage for tenant."""
        if tenant_id not in self.tenants:
            return {}

        model = self.tenants[tenant_id].model

        # Calculate memory usage
        memory_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        memory_gb = memory_bytes / (1024 ** 3)

        usage = {
            "memory_gb": memory_gb,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        }

        self.resource_usage[tenant_id] = usage

        return usage

    def verify_tenant_isolation(self) -> Dict[str, bool]:
        """Verify that tenants are properly isolated."""
        isolation_status = {}

        for tenant_id in self.tenants.keys():
            # Check that tenant has separate model instance
            is_isolated = True

            # Verify no shared parameters with other tenants
            tenant_params = set(id(p) for p in self.tenants[tenant_id].model.parameters())

            for other_id in self.tenants.keys():
                if other_id != tenant_id:
                    other_params = set(id(p) for p in self.tenants[other_id].model.parameters())
                    if tenant_params & other_params:  # Intersection
                        is_isolated = False
                        logger.error(
                            f"Isolation violation: {tenant_id} shares parameters with {other_id}"
                        )

            isolation_status[tenant_id] = is_isolated

        return isolation_status

    def _log_audit(self, action: str, tenant_id: str, details: Dict[str, Any]):
        """Log audit event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "tenant_id": tenant_id,
            "details": details
        }

        self.audit_log.append(event)

        # Save to file
        audit_file = self.output_dir / "audit_log.jsonl"
        with open(audit_file, 'a') as f:
            f.write(json.dumps(event) + "\n")

    def get_audit_trail(
        self,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get audit trail for tenant or all tenants."""
        if tenant_id:
            return [
                event for event in self.audit_log
                if event["tenant_id"] == tenant_id
            ]
        return self.audit_log

    def generate_tenant_report(self, tenant_id: str) -> Dict[str, Any]:
        """Generate comprehensive report for tenant."""
        if tenant_id not in self.tenants:
            return {}

        tenant = self.tenants[tenant_id]
        config = self.tenant_configs[tenant_id]

        report = {
            "tenant_id": tenant_id,
            "tenant_name": config.tenant_name,
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "personalization_enabled": config.enable_personalization,
                "adapter_layers": config.adapter_layers,
                "ab_testing_enabled": config.enable_ab_testing
            },
            "performance": {
                "best_accuracy": tenant.best_accuracy,
                "num_evaluations": len(tenant.performance_history),
                "recent_metrics": tenant.performance_history[-5:] if tenant.performance_history else []
            },
            "resources": self.get_resource_usage(tenant_id),
            "isolation_verified": self.verify_tenant_isolation().get(tenant_id, False)
        }

        # Add A/B test results if available
        if tenant_id in self.ab_tests:
            report["ab_test_results"] = self.ab_tests[tenant_id].get_test_results()

        return report


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example tenant configuration
    tenant_config = TenantConfig(
        tenant_id="tenant_001",
        tenant_name="Acme Corporation",
        data_path="./data/tenant_001",
        enable_personalization=True,
        enable_ab_testing=True
    )

    print("Tenant manager module loaded successfully")
