"""
SHWL Phase 3: Root Cause Analysis

Uses SAP_LLM/reasoning engine to analyze anomaly clusters:
- Generates natural language explanations
- Identifies root causes
- Proposes fixes
- Estimates fix confidence
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from sap_llm.shwl.pattern_clusterer import AnomalyCluster

logger = logging.getLogger(__name__)


@dataclass
class RootCauseAnalysis:
    """Root cause analysis result."""
    cluster_id: str
    root_cause: str
    explanation: str
    proposed_fix: str
    fix_type: str  # rule_update, model_retrain, data_quality, configuration
    confidence: float
    affected_components: List[str]
    estimated_fix_time_hours: int


class RootCauseAnalyzer:
    """
    SHWL Phase 3: Analyze cluster root causes using AI.
    """

    def __init__(self, reasoning_engine: Optional[Any] = None):
        self.reasoning_engine = reasoning_engine
        logger.info("RootCauseAnalyzer initialized")

    def analyze_cluster(self, cluster: AnomalyCluster) -> RootCauseAnalysis:
        """Analyze cluster and generate root cause."""
        # Build prompt for LLM
        prompt = self._build_analysis_prompt(cluster)

        # For demo: generate structured analysis
        return RootCauseAnalysis(
            cluster_id=cluster.cluster_id,
            root_cause=f"Common pattern in {cluster.common_features['primary_anomaly_type']}",
            explanation=f"Cluster of {cluster.size} similar failures in {cluster.common_features['primary_doc_type']} documents",
            proposed_fix="Add validation rule for field extraction",
            fix_type="rule_update",
            confidence=cluster.confidence,
            affected_components=cluster.common_features['top_affected_fields'],
            estimated_fix_time_hours=2
        )

    def _build_analysis_prompt(self, cluster: AnomalyCluster) -> str:
        """Build prompt for root cause analysis."""
        return f"""Analyze this anomaly cluster:

Cluster ID: {cluster.cluster_id}
Size: {cluster.size}
Severity: {cluster.severity}
Primary Type: {cluster.common_features['primary_anomaly_type']}
Document Type: {cluster.common_features['primary_doc_type']}
Top Errors: {cluster.common_features['top_error_messages']}

Provide:
1. Root cause
2. Proposed fix
3. Fix confidence"""
