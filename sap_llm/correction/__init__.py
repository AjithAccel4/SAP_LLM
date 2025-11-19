"""
Advanced Self-Correction System with Multi-Strategy Retry.

This module provides comprehensive error detection, correction strategies,
pattern learning, and human escalation for document processing.

Enterprise Features:
- Configuration management
- Input validation and sanitization
- Audit logging
- Retry logic with exponential backoff
- Circuit breaker pattern
- Data masking for PII
- PMG interface and mock
"""

from sap_llm.correction.error_detector import ErrorDetector, Error, ErrorReport
from sap_llm.correction.strategies import (
    CorrectionStrategy,
    RuleBasedCorrectionStrategy,
    RerunWithHigherConfidenceStrategy,
    ContextEnhancementStrategy,
    HumanInTheLoopStrategy,
    CorrectionResult,
)
from sap_llm.correction.correction_engine import SelfCorrectionEngine
from sap_llm.correction.pattern_learner import ErrorPatternLearner
from sap_llm.correction.escalation import EscalationManager
from sap_llm.correction.analytics import CorrectionAnalytics
from sap_llm.correction.config import (
    CorrectionConfig,
    ConfigurationManager,
    get_config_manager,
)
from sap_llm.correction.pmg_interface import (
    PMGInterface,
    MockPMG,
    create_pmg_instance,
)
from sap_llm.correction.utils import (
    InputValidator,
    AuditLogger,
    get_audit_logger,
    AuditEventType,
    DataMasker,
    CircuitBreaker,
    retry_with_backoff,
)

__all__ = [
    # Core components
    "ErrorDetector",
    "Error",
    "ErrorReport",
    "CorrectionStrategy",
    "RuleBasedCorrectionStrategy",
    "RerunWithHigherConfidenceStrategy",
    "ContextEnhancementStrategy",
    "HumanInTheLoopStrategy",
    "CorrectionResult",
    "SelfCorrectionEngine",
    "ErrorPatternLearner",
    "EscalationManager",
    "CorrectionAnalytics",
    # Configuration
    "CorrectionConfig",
    "ConfigurationManager",
    "get_config_manager",
    # PMG
    "PMGInterface",
    "MockPMG",
    "create_pmg_instance",
    # Utilities
    "InputValidator",
    "AuditLogger",
    "get_audit_logger",
    "AuditEventType",
    "DataMasker",
    "CircuitBreaker",
    "retry_with_backoff",
]

__version__ = "1.0.0"
