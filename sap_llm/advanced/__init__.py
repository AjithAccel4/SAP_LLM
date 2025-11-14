"""Advanced features module for SAP_LLM"""

from sap_llm.advanced.multilingual import (
    MultilingualProcessor,
    LanguageDetector,
    SUPPORTED_LANGUAGES,
    multilingual_processor,
    process_multilingual_document,
    detect_document_language,
    get_supported_languages,
)

from sap_llm.advanced.explainability import (
    ExplainabilityEngine,
    AttentionVisualizer,
    FeatureImportanceAnalyzer,
    ConfidenceExplainer,
    CounterfactualGenerator,
    ExplanationType,
    Explanation,
    explainability_engine,
    explain_extraction,
)

from sap_llm.advanced.federated_learning import (
    FederatedLearningOrchestrator,
    FederatedServer,
    FederatedClient,
    ClientConfig,
    ModelUpdate,
    FederatedRound,
    AggregationStrategy,
    DifferentialPrivacy,
    SecureAggregation,
    federated_orchestrator,
)

from sap_llm.advanced.online_learning import (
    OnlineLearningSystem,
    FeedbackBuffer,
    ActiveLearner,
    IncrementalLearner,
    PerformanceMonitor,
    Feedback,
    FeedbackType,
    UncertaintySampling,
    online_learning_system,
    process_with_online_learning,
    add_user_feedback,
)

__all__ = [
    # Multilingual
    "MultilingualProcessor",
    "LanguageDetector",
    "SUPPORTED_LANGUAGES",
    "multilingual_processor",
    "process_multilingual_document",
    "detect_document_language",
    "get_supported_languages",
    # Explainability
    "ExplainabilityEngine",
    "AttentionVisualizer",
    "FeatureImportanceAnalyzer",
    "ConfidenceExplainer",
    "CounterfactualGenerator",
    "ExplanationType",
    "Explanation",
    "explainability_engine",
    "explain_extraction",
    # Federated Learning
    "FederatedLearningOrchestrator",
    "FederatedServer",
    "FederatedClient",
    "ClientConfig",
    "ModelUpdate",
    "FederatedRound",
    "AggregationStrategy",
    "DifferentialPrivacy",
    "SecureAggregation",
    "federated_orchestrator",
    # Online Learning
    "OnlineLearningSystem",
    "FeedbackBuffer",
    "ActiveLearner",
    "IncrementalLearner",
    "PerformanceMonitor",
    "Feedback",
    "FeedbackType",
    "UncertaintySampling",
    "online_learning_system",
    "process_with_online_learning",
    "add_user_feedback",
]
