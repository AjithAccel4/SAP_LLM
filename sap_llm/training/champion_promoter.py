"""
Champion Promotion System with Automated Decision Making and Rollback.

Handles:
- Automated promotion based on A/B test results
- Safe promotion with validation
- Rollback capability
- Notification system
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from sap_llm.models.registry import ModelRegistry
from sap_llm.training.ab_testing import ABTestingManager, ABTestResult

logger = logging.getLogger(__name__)


class PromotionDecision:
    """Promotion decision constants."""
    PROMOTE = "promote"
    KEEP_CHAMPION = "keep_champion"
    CONTINUE_TESTING = "continue_testing"
    ROLLBACK = "rollback"


class ChampionPromoter:
    """
    Manages champion model promotion and rollback.

    Features:
    - Automated A/B test evaluation
    - Safe promotion with backup
    - Rollback capability
    - Promotion history tracking
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        ab_testing: Optional[ABTestingManager] = None,
        min_improvement: float = 0.02,  # 2% improvement required
        max_degradation: float = 0.01   # 1% degradation allowed
    ):
        """
        Initialize champion promoter.

        Args:
            model_registry: Model registry
            ab_testing: A/B testing manager
            min_improvement: Minimum improvement for promotion
            max_degradation: Maximum acceptable degradation
        """
        self.model_registry = model_registry or ModelRegistry()
        self.ab_testing = ab_testing or ABTestingManager(model_registry=self.model_registry)
        self.min_improvement = min_improvement
        self.max_degradation = max_degradation

        logger.info(
            f"ChampionPromoter initialized "
            f"(min_improvement={min_improvement:.1%}, "
            f"max_degradation={max_degradation:.1%})"
        )

    def evaluate_and_promote(
        self,
        test_id: str,
        min_samples: int = 1000,
        auto_promote: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate A/B test and promote if challenger wins.

        Args:
            test_id: A/B test ID
            min_samples: Minimum samples for evaluation
            auto_promote: Automatically promote if criteria met

        Returns:
            Promotion result
        """
        logger.info(f"Evaluating A/B test for promotion: {test_id}")

        # Evaluate A/B test
        result = self.ab_testing.evaluate_ab_test(
            test_id=test_id,
            min_samples=min_samples
        )

        # Make decision
        decision = self._make_promotion_decision(result)

        logger.info(
            f"Promotion decision: {decision} "
            f"(winner={result.winner}, p_value={result.p_value:.4f})"
        )

        promotion_result = {
            "test_id": test_id,
            "decision": decision,
            "ab_test_result": result,
            "timestamp": datetime.now().isoformat()
        }

        # Execute decision
        if auto_promote and decision == PromotionDecision.PROMOTE:
            self._execute_promotion(test_id, result)
            promotion_result["promoted"] = True
            promotion_result["promoted_at"] = datetime.now().isoformat()
        elif decision == PromotionDecision.KEEP_CHAMPION:
            self._reject_challenger(test_id, result)
            promotion_result["promoted"] = False
            promotion_result["reason"] = "champion_better"
        elif decision == PromotionDecision.CONTINUE_TESTING:
            promotion_result["promoted"] = False
            promotion_result["reason"] = "insufficient_evidence"
        else:
            promotion_result["promoted"] = False
            promotion_result["reason"] = "unknown"

        return promotion_result

    def _make_promotion_decision(self, result: ABTestResult) -> str:
        """
        Make promotion decision based on A/B test result.

        Args:
            result: A/B test result

        Returns:
            Decision (promote, keep_champion, continue_testing)
        """
        # Check if enough samples
        if result.recommendation == "continue_testing":
            return PromotionDecision.CONTINUE_TESTING

        # Calculate improvement
        improvement = (
            result.challenger_metrics["accuracy"] -
            result.champion_metrics["accuracy"]
        )

        # Decision logic
        if result.winner == "challenger":
            # Challenger won with statistical significance
            if improvement >= self.min_improvement:
                logger.info(
                    f"Challenger wins with {improvement:.1%} improvement "
                    f"(p={result.p_value:.4f})"
                )
                return PromotionDecision.PROMOTE
            else:
                logger.info(
                    f"Challenger wins but improvement ({improvement:.1%}) "
                    f"below threshold ({self.min_improvement:.1%})"
                )
                return PromotionDecision.KEEP_CHAMPION

        elif result.winner == "champion":
            # Champion remains better
            if abs(improvement) <= self.max_degradation:
                logger.info("Champion remains better, keeping current champion")
                return PromotionDecision.KEEP_CHAMPION
            else:
                logger.warning(
                    f"Significant degradation detected: {improvement:.1%}"
                )
                return PromotionDecision.KEEP_CHAMPION

        else:
            # Inconclusive
            logger.info("Test inconclusive, continuing testing")
            return PromotionDecision.CONTINUE_TESTING

    def _execute_promotion(self, test_id: str, result: ABTestResult):
        """
        Execute promotion of challenger to champion.

        Args:
            test_id: A/B test ID
            result: A/B test result
        """
        logger.info(f"Promoting challenger to champion: {result.challenger_id}")

        try:
            # Backup current champion (already done by model_registry.promote_to_champion)

            # Promote challenger
            self.model_registry.promote_to_champion(
                model_id=result.challenger_id,
                reason=f"A/B test winner (improvement: {result.details.get('improvement', 0):.2%})",
                metrics=result.challenger_metrics
            )

            # Complete A/B test
            self.ab_testing.complete_ab_test(
                test_id=test_id,
                winner="challenger",
                reason=f"Promoted: {result.details.get('improvement', 0):.2%} improvement"
            )

            # Send notification
            self._notify_promotion(result)

            logger.info(f"✅ Successfully promoted {result.challenger_id} to champion")

        except Exception as e:
            logger.error(f"Promotion failed: {e}")
            # In production: Alert team, consider rollback
            raise

    def _reject_challenger(self, test_id: str, result: ABTestResult):
        """
        Reject challenger and keep current champion.

        Args:
            test_id: A/B test ID
            result: A/B test result
        """
        logger.info(f"Rejecting challenger: {result.challenger_id}")

        # Complete A/B test
        self.ab_testing.complete_ab_test(
            test_id=test_id,
            winner="champion",
            reason=f"Champion better or insufficient improvement"
        )

        logger.info(f"Kept current champion: {result.champion_id}")

    def rollback_to_previous_champion(
        self,
        model_type: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Rollback to previous champion.

        Use cases:
        - New champion causing issues in production
        - Performance degradation detected
        - Manual intervention required

        Args:
            model_type: Model type to rollback
            reason: Reason for rollback

        Returns:
            Rollback result
        """
        logger.warning(f"⚠️  Initiating rollback for {model_type}. Reason: {reason}")

        try:
            # Execute rollback
            previous_champion_id = self.model_registry.rollback_to_previous_champion(
                model_type=model_type,
                reason=reason
            )

            result = {
                "success": True,
                "model_type": model_type,
                "restored_champion_id": previous_champion_id,
                "reason": reason,
                "rolled_back_at": datetime.now().isoformat()
            }

            # Send notification
            self._notify_rollback(result)

            logger.info(f"✅ Rollback successful. Restored champion: {previous_champion_id}")

            return result

        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return {
                "success": False,
                "model_type": model_type,
                "reason": reason,
                "error": str(e),
                "rolled_back_at": datetime.now().isoformat()
            }

    def _notify_promotion(self, result: ABTestResult):
        """
        Send notification about promotion.

        Args:
            result: A/B test result
        """
        message = (
            f"🎉 Model Promotion\n"
            f"Challenger {result.challenger_id} promoted to champion\n"
            f"Improvement: {result.details.get('improvement', 0):.2%}\n"
            f"Champion accuracy: {result.champion_metrics['accuracy']:.2%}\n"
            f"Challenger accuracy: {result.challenger_metrics['accuracy']:.2%}\n"
            f"P-value: {result.p_value:.4f}\n"
            f"Confidence: {result.confidence:.1%}"
        )

        logger.info(message)

        # In production: Send to Slack, email, etc.
        # self.notification_service.send(message)

    def _notify_rollback(self, result: Dict[str, Any]):
        """
        Send notification about rollback.

        Args:
            result: Rollback result
        """
        message = (
            f"⚠️  Model Rollback\n"
            f"Model type: {result['model_type']}\n"
            f"Restored champion: {result['restored_champion_id']}\n"
            f"Reason: {result['reason']}"
        )

        logger.warning(message)

        # In production: Send alert
        # self.notification_service.send_alert(message)

    def monitor_champion_health(
        self,
        model_type: str,
        recent_predictions: list,
        auto_rollback: bool = True
    ) -> Dict[str, Any]:
        """
        Monitor champion model health and auto-rollback if issues detected.

        Args:
            model_type: Model type to monitor
            recent_predictions: Recent predictions with ground truth
            auto_rollback: Automatically rollback if critical issues

        Returns:
            Health report
        """
        champion = self.model_registry.get_champion(model_type)
        if not champion:
            return {"status": "no_champion", "healthy": False}

        # Calculate current metrics
        if not recent_predictions:
            return {"status": "no_data", "healthy": True}

        # Simple accuracy check
        correct = sum(
            1 for p in recent_predictions
            if p.get("prediction") == p.get("ground_truth")
        )
        current_accuracy = correct / len(recent_predictions)

        # Get baseline accuracy
        import json
        baseline_metrics = json.loads(champion.get("metrics", "{}"))
        baseline_accuracy = baseline_metrics.get("accuracy", 0.0)

        # Check degradation
        degradation = baseline_accuracy - current_accuracy

        health_report = {
            "model_id": champion["id"],
            "model_type": model_type,
            "baseline_accuracy": baseline_accuracy,
            "current_accuracy": current_accuracy,
            "degradation": degradation,
            "sample_count": len(recent_predictions),
            "timestamp": datetime.now().isoformat()
        }

        # Determine health status
        if degradation > self.max_degradation * 3:  # 3x threshold = critical
            health_report["healthy"] = False
            health_report["severity"] = "critical"

            if auto_rollback:
                logger.warning(
                    f"Critical degradation detected: {degradation:.1%}. "
                    f"Initiating auto-rollback..."
                )

                rollback_result = self.rollback_to_previous_champion(
                    model_type=model_type,
                    reason=f"Auto-rollback: {degradation:.1%} accuracy degradation"
                )

                health_report["rollback"] = rollback_result

        elif degradation > self.max_degradation:
            health_report["healthy"] = False
            health_report["severity"] = "warning"
            logger.warning(
                f"Performance degradation detected: {degradation:.1%}"
            )
        else:
            health_report["healthy"] = True
            health_report["severity"] = "normal"

        return health_report
