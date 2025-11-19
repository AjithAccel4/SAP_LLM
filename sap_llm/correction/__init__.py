"""
Advanced Self-Correction System with Multi-Strategy Retry.

This module provides comprehensive error detection, correction strategies,
pattern learning, and human escalation for document processing.
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

__all__ = [
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
]
