"""
Federated Learning Module for Multi-Tenant Privacy.

Implements federated learning with differential privacy for SAP_LLM,
enabling multi-tenant model training without sharing raw data.

Components:
- FederatedTrainer: Main federated learning orchestrator with FedAvg
- SecureAggregation: Encrypted aggregation protocols
- TenantManager: Multi-tenant coordination and management
- PrivacyAuditor: Privacy compliance and audit trail
"""

from .federated_trainer import FederatedTrainer, FederatedConfig
from .secure_aggregation import SecureAggregator, EncryptionConfig
from .tenant_manager import TenantManager, TenantConfig
from .privacy_auditor import PrivacyAuditor, PrivacyReport

__all__ = [
    "FederatedTrainer",
    "FederatedConfig",
    "SecureAggregator",
    "EncryptionConfig",
    "TenantManager",
    "TenantConfig",
    "PrivacyAuditor",
    "PrivacyReport",
]

__version__ = "1.0.0"
