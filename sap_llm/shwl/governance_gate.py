"""
SHWL Phase 4: Governance Gate

Implements approval workflow:
- Auto-approve low-risk fixes (confidence > 0.95)
- Human review for high-risk changes
- Risk assessment
- Approval tracking
"""

import logging
from typing import Any, Dict, List
from dataclasses import dataclass
from sap_llm.shwl.root_cause_analyzer import RootCauseAnalysis

logger = logging.getLogger(__name__)


@dataclass
class GovernanceDecision:
    """Governance approval decision."""
    analysis_id: str
    approved: bool
    auto_approved: bool
    risk_level: str  # LOW, MEDIUM, HIGH
    reviewer: Optional[str]
    review_notes: Optional[str]
    timestamp: str


class GovernanceGate:
    """
    SHWL Phase 4: Governance and approval workflow.
    """

    def __init__(self, auto_approve_threshold: float = 0.95):
        self.auto_approve_threshold = auto_approve_threshold
        self.pending_reviews: List[RootCauseAnalysis] = []
        logger.info(f"GovernanceGate initialized (auto_approve_threshold={auto_approve_threshold})")

    def evaluate_fix(self, analysis: RootCauseAnalysis) -> GovernanceDecision:
        """Evaluate if fix can be auto-approved."""
        from datetime import datetime

        risk_level = self._assess_risk(analysis)

        # Auto-approve low-risk, high-confidence fixes
        if risk_level == "LOW" and analysis.confidence >= self.auto_approve_threshold:
            return GovernanceDecision(
                analysis_id=analysis.cluster_id,
                approved=True,
                auto_approved=True,
                risk_level=risk_level,
                reviewer="system",
                review_notes=f"Auto-approved: low risk, {analysis.confidence:.2%} confidence",
                timestamp=datetime.now().isoformat()
            )
        else:
            # Queue for human review
            self.pending_reviews.append(analysis)
            return GovernanceDecision(
                analysis_id=analysis.cluster_id,
                approved=False,
                auto_approved=False,
                risk_level=risk_level,
                reviewer=None,
                review_notes="Queued for human review",
                timestamp=datetime.now().isoformat()
            )

    def _assess_risk(self, analysis: RootCauseAnalysis) -> str:
        """Assess risk level of proposed fix."""
        if analysis.fix_type in ["rule_update", "data_quality"]:
            return "LOW"
        elif analysis.fix_type == "configuration":
            return "MEDIUM"
        else:  # model_retrain
            return "HIGH"
