"""
SAP_LLM Continuous Learning and Self-Improvement System

A comprehensive auto-learning framework that enables SAP_LLM to continuously
improve through:
- Online learning from user feedback
- Active learning for uncertain predictions
- Transfer learning across document types
- Automatic model updates and versioning
- Knowledge augmentation from production data
- Self-improvement without human intervention
"""

from sap_llm.learning.adaptive_learning import AdaptiveLearningEngine
from sap_llm.learning.feedback_loop import FeedbackLoopSystem
from sap_llm.learning.knowledge_augmentation import KnowledgeAugmentationEngine
from sap_llm.learning.online_learning import OnlineLearningEngine
from sap_llm.learning.self_improvement import SelfImprovementPipeline

__all__ = [
    "AdaptiveLearningEngine",
    "FeedbackLoopSystem",
    "KnowledgeAugmentationEngine",
    "OnlineLearningEngine",
    "SelfImprovementPipeline",
]

__version__ = "1.0.0"
