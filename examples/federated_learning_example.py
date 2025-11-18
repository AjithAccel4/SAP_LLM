"""
Federated Learning Example for SAP_LLM.

Demonstrates multi-tenant federated learning with:
- Differential privacy (DP-SGD)
- Secure aggregation
- Tenant-specific personalization
- Privacy auditing
- Compliance reporting

This example shows how to train a model across multiple tenants
without sharing raw data, while maintaining privacy guarantees.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path

# Import federated learning components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sap_llm.federated import (
    FederatedTrainer,
    FederatedConfig,
    SecureAggregator,
    EncryptionConfig,
    TenantManager,
    TenantConfig,
    PrivacyAuditor,
    ComplianceFramework
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Dummy model for demonstration
class SimpleClassifier(nn.Module):
    """Simple classification model for demonstration."""

    def __init__(self, input_dim: int = 768, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_features=None, labels=None, **kwargs):
        """Forward pass with loss computation."""
        logits = self.classifier(input_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # Return output similar to HuggingFace models
        class Output:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits

        return Output(loss=loss, logits=logits)


# Dummy dataset generator
def create_dummy_dataset(
    num_samples: int = 1000,
    input_dim: int = 768,
    num_classes: int = 10,
    seed: Optional[int] = None
) -> TensorDataset:
    """Create dummy dataset for demonstration."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate random features
    features = torch.randn(num_samples, input_dim)

    # Generate random labels
    labels = torch.randint(0, num_classes, (num_samples,))

    return TensorDataset(features, labels)


def prepare_batch(batch):
    """Prepare batch for model input."""
    features, labels = batch
    return {
        "input_features": features,
        "labels": labels
    }


def create_tenant_datasets(
    num_tenants: int = 5,
    samples_per_tenant: int = 1000
) -> Dict[str, Dataset]:
    """Create datasets for multiple tenants."""
    tenant_datasets = {}

    for i in range(num_tenants):
        tenant_id = f"tenant_{i:03d}"

        # Create tenant-specific dataset (with different seed for diversity)
        dataset = create_dummy_dataset(
            num_samples=samples_per_tenant,
            seed=i * 42
        )

        tenant_datasets[tenant_id] = dataset

    return tenant_datasets


def example_basic_federated_training():
    """Example 1: Basic federated training with differential privacy."""
    logger.info("=" * 80)
    logger.info("Example 1: Basic Federated Training")
    logger.info("=" * 80)

    # Create global model
    model = SimpleClassifier(input_dim=768, num_classes=10)

    # Configure federated learning
    config = FederatedConfig(
        num_rounds=10,
        num_clients_per_round=3,
        local_epochs=2,
        local_batch_size=32,
        local_learning_rate=1e-3,
        enable_dp=True,
        target_epsilon=1.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        output_dir="./models/federated/basic_example"
    )

    # Create test dataset
    test_dataset = create_dummy_dataset(num_samples=500, seed=999)

    # Collate function to prepare batches
    def collate_fn(batch):
        features = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return {"input_features": features, "labels": labels}

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=collate_fn
    )

    # Create federated trainer
    trainer = FederatedTrainer(
        global_model=model,
        config=config,
        test_loader=test_loader
    )

    # Create client datasets
    num_clients = 5
    client_datasets = create_tenant_datasets(
        num_tenants=num_clients,
        samples_per_tenant=1000
    )

    # Create client data loaders
    client_loaders = {}
    for client_id, dataset in client_datasets.items():
        client_loaders[client_id] = DataLoader(
            dataset,
            batch_size=config.local_batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    # Train
    logger.info(f"Starting training with {num_clients} clients...")
    trainer.train(client_data_loaders=client_loaders, num_rounds=config.num_rounds)

    logger.info("Training completed!")
    logger.info(f"Final epsilon: {trainer.privacy_budget_spent['epsilon']:.4f}")
    logger.info(f"Best accuracy: {trainer.best_accuracy:.4f}")


def example_secure_aggregation():
    """Example 2: Federated learning with secure aggregation."""
    logger.info("=" * 80)
    logger.info("Example 2: Secure Aggregation")
    logger.info("=" * 80)

    # Create encryption config
    encryption_config = EncryptionConfig(
        key_size=2048,
        enable_encryption=True,
        enable_smpc=True,
        enable_zkp=True
    )

    # Create secure aggregator
    aggregator = SecureAggregator(encryption_config)

    # Register clients
    client_ids = [f"client_{i}" for i in range(3)]
    for client_id in client_ids:
        aggregator.register_client(client_id)

    # Simulate client model updates
    logger.info("Simulating client updates...")

    model = SimpleClassifier()
    client_updates = []

    for i, client_id in enumerate(client_ids):
        # Clone model and perturb weights slightly
        client_model = SimpleClassifier()
        client_model.load_state_dict(model.state_dict())

        # Add some noise to simulate training
        with torch.no_grad():
            for param in client_model.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        client_updates.append(client_model.state_dict())

        # Verify contribution
        is_valid = aggregator.verify_client_contribution(
            client_id,
            client_model.state_dict(),
            expected_norm_range=(0.0, 100.0)
        )
        logger.info(f"Client {client_id} contribution valid: {is_valid}")

    # Detect Byzantine clients
    byzantine_clients = aggregator.detect_byzantine_clients(
        client_updates,
        client_ids,
        threshold=3.0
    )

    if byzantine_clients:
        logger.warning(f"Byzantine clients detected: {byzantine_clients}")
    else:
        logger.info("No Byzantine clients detected")

    # Secure aggregation
    logger.info("Performing secure aggregation...")
    aggregated_model = aggregator.secure_aggregate_smpc(client_updates, client_ids)

    logger.info("Secure aggregation completed!")


def example_multi_tenant_management():
    """Example 3: Multi-tenant management with personalization."""
    logger.info("=" * 80)
    logger.info("Example 3: Multi-Tenant Management")
    logger.info("=" * 80)

    # Create tenant manager
    tenant_manager = TenantManager(output_dir="./models/tenants/example")

    # Create base model
    base_model = SimpleClassifier()

    # Register multiple tenants
    num_tenants = 3
    tenant_configs = []

    for i in range(num_tenants):
        config = TenantConfig(
            tenant_id=f"tenant_{i:03d}",
            tenant_name=f"Acme Corp {i}",
            data_path=f"./data/tenant_{i:03d}",
            enable_personalization=True,
            enable_ab_testing=True,
            min_accuracy=0.85
        )

        tenant_id = tenant_manager.register_tenant(config, base_model)
        tenant_configs.append(config)
        logger.info(f"Registered: {tenant_id}")

    # Verify tenant isolation
    isolation_status = tenant_manager.verify_tenant_isolation()
    logger.info(f"Tenant isolation verified: {all(isolation_status.values())}")

    # Create dummy datasets for each tenant
    tenant_datasets = create_tenant_datasets(
        num_tenants=num_tenants,
        samples_per_tenant=500
    )

    # Collate function
    def collate_fn(batch):
        features = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return {"input_features": features, "labels": labels}

    # Evaluate each tenant
    for config in tenant_configs:
        tenant_id = config.tenant_id

        # Create evaluation loader
        eval_loader = DataLoader(
            tenant_datasets[tenant_id],
            batch_size=32,
            collate_fn=collate_fn
        )

        # Evaluate
        metrics = tenant_manager.evaluate_tenant(tenant_id, eval_loader)
        logger.info(f"{tenant_id} metrics: {metrics}")

        # Save checkpoint
        tenant_manager.save_tenant_checkpoint(tenant_id, metrics)

        # Get resource usage
        resources = tenant_manager.get_resource_usage(tenant_id)
        logger.info(f"{tenant_id} resources: {resources}")

    # Generate reports
    for config in tenant_configs:
        report = tenant_manager.generate_tenant_report(config.tenant_id)
        logger.info(f"Report for {config.tenant_id}:")
        logger.info(f"  - Best accuracy: {report['performance']['best_accuracy']:.4f}")
        logger.info(f"  - Isolation verified: {report['isolation_verified']}")


def example_privacy_auditing():
    """Example 4: Privacy auditing and compliance."""
    logger.info("=" * 80)
    logger.info("Example 4: Privacy Auditing")
    logger.info("=" * 80)

    # Create privacy auditor
    auditor = PrivacyAuditor(
        epsilon_limit=1.0,
        delta_limit=1e-5,
        output_dir="./audit_reports/example"
    )

    # Enable compliance frameworks
    auditor.enable_framework(ComplianceFramework.GDPR)
    auditor.enable_framework(ComplianceFramework.HIPAA)

    # Simulate tenant activities
    tenant_ids = [f"tenant_{i:03d}" for i in range(3)]

    for tenant_id in tenant_ids:
        # Allocate privacy budget
        auditor.budget_tracker.allocate_budget(
            tenant_id,
            epsilon=1.0,
            delta=1e-5
        )

        # Simulate training rounds
        for round_idx in range(5):
            # Consume budget
            auditor.track_privacy_budget(
                tenant_id,
                epsilon_spent=0.15,
                delta_spent=1e-6
            )

            # Log training event
            auditor.log_event("training_round", tenant_id, {
                "round": round_idx,
                "samples_processed": 1000
            })

        # Verify data minimization
        score = auditor.verify_data_minimization(
            tenant_id,
            data_fields_used=["field1", "field2", "field3"],
            necessary_fields=["field1", "field2"]
        )
        logger.info(f"{tenant_id} data minimization score: {score:.2f}")

        # Verify compliance
        audit_data = {
            "consent_obtained": True,
            "purpose_specified": True,
            "purpose_changed": False,
            "data_fields_used": ["field1", "field2"],
            "necessary_fields": ["field1", "field2"],
            "data_validation_performed": True,
            "retention_period_days": 180,
            "max_retention_days": 365,
            "encryption_enabled": True,
            "access_control_enabled": True,
            "audit_trail_enabled": True,
            "phi_identified": False,
            "security_management": True,
            "workforce_training": True,
            "access_authorization": True,
            "facility_access_controls": True,
            "device_security": True,
            "access_controls": True,
            "audit_controls": True,
            "transmission_security": True,
            "breach_notification_enabled": True
        }

        compliance_results = auditor.verify_compliance(tenant_id, audit_data)

        for framework, results in compliance_results.items():
            logger.info(f"{tenant_id} {framework.value} compliance:")
            for requirement, compliant in results.items():
                status = "✓" if compliant else "✗"
                logger.info(f"  {status} {requirement}")

    # Generate privacy report
    report = auditor.generate_privacy_report()

    logger.info("\nPrivacy Audit Report:")
    logger.info(f"  - Report ID: {report.report_id}")
    logger.info(f"  - Privacy budget used: ε={report.privacy_budget_used['epsilon']:.4f}")
    logger.info(f"  - Budget compliance: {report.budget_compliance}")
    logger.info(f"  - Number of violations: {len(report.violations)}")
    logger.info(f"  - Data minimization score: {report.data_minimization_score:.2f}")
    logger.info(f"  - Recommendations: {len(report.recommendations)}")

    if report.violations:
        logger.warning("\nViolations detected:")
        for violation in report.violations[:3]:  # Show first 3
            logger.warning(f"  - {violation.severity.upper()}: {violation.description}")


def example_end_to_end_workflow():
    """Example 5: End-to-end federated learning workflow."""
    logger.info("=" * 80)
    logger.info("Example 5: End-to-End Federated Learning Workflow")
    logger.info("=" * 80)

    # Step 1: Initialize components
    logger.info("\n[1/5] Initializing components...")

    # Privacy auditor
    auditor = PrivacyAuditor(epsilon_limit=1.0, delta_limit=1e-5)
    auditor.enable_framework(ComplianceFramework.GDPR)

    # Tenant manager
    tenant_manager = TenantManager()

    # Base model
    base_model = SimpleClassifier()

    # Step 2: Register tenants
    logger.info("\n[2/5] Registering tenants...")

    num_tenants = 3
    for i in range(num_tenants):
        config = TenantConfig(
            tenant_id=f"tenant_{i:03d}",
            tenant_name=f"Company {chr(65 + i)}",
            data_path=f"./data/tenant_{i:03d}",
            enable_personalization=True
        )

        tenant_manager.register_tenant(config, base_model)

        # Allocate privacy budget
        auditor.budget_tracker.allocate_budget(
            config.tenant_id,
            epsilon=1.0,
            delta=1e-5
        )

    # Step 3: Prepare federated training
    logger.info("\n[3/5] Preparing federated training...")

    config = FederatedConfig(
        num_rounds=5,
        num_clients_per_round=2,
        local_epochs=2,
        enable_dp=True,
        target_epsilon=1.0,
        target_delta=1e-5,
        output_dir="./models/federated/end_to_end"
    )

    # Create datasets
    tenant_datasets = create_tenant_datasets(num_tenants=num_tenants)

    # Collate function
    def collate_fn(batch):
        features = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return {"input_features": features, "labels": labels}

    # Create data loaders
    client_loaders = {}
    for tenant_id, dataset in tenant_datasets.items():
        client_loaders[tenant_id] = DataLoader(
            dataset,
            batch_size=config.local_batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    # Step 4: Run federated training
    logger.info("\n[4/5] Running federated training...")

    test_dataset = create_dummy_dataset(num_samples=500, seed=999)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    trainer = FederatedTrainer(base_model, config, test_loader)

    # Train for fewer rounds for demo
    trainer.train(client_loaders, num_rounds=3)

    # Step 5: Generate reports
    logger.info("\n[5/5] Generating reports...")

    # Privacy report
    privacy_report = auditor.generate_privacy_report()

    logger.info("\nFinal Results:")
    logger.info(f"  - Training rounds completed: {trainer.current_round + 1}")
    logger.info(f"  - Best global accuracy: {trainer.best_accuracy:.4f}")
    logger.info(f"  - Privacy epsilon spent: {trainer.privacy_budget_spent['epsilon']:.4f}")
    logger.info(f"  - Privacy compliance: {privacy_report.budget_compliance}")
    logger.info(f"  - Data minimization score: {privacy_report.data_minimization_score:.2f}")

    logger.info("\nWorkflow completed successfully!")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 80)
    logger.info("Federated Learning Examples for SAP_LLM")
    logger.info("=" * 80 + "\n")

    try:
        # Run examples
        example_basic_federated_training()
        print("\n")

        example_secure_aggregation()
        print("\n")

        example_multi_tenant_management()
        print("\n")

        example_privacy_auditing()
        print("\n")

        example_end_to_end_workflow()

        logger.info("\n" + "=" * 80)
        logger.info("All examples completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
