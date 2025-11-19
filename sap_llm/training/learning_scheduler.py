"""
Continuous Learning Scheduler.

Orchestrates the full automated learning loop:
1. Periodic drift checks
2. Automated retraining triggers
3. A/B test monitoring
4. Champion promotion
5. Health monitoring
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    logging.warning("schedule library not available. Install with: pip install schedule")

from sap_llm.models.registry import ModelRegistry
from sap_llm.training.retraining_orchestrator import RetrainingOrchestrator
from sap_llm.training.ab_testing import ABTestingManager
from sap_llm.training.champion_promoter import ChampionPromoter

logger = logging.getLogger(__name__)


class LearningScheduler:
    """
    Continuous learning scheduler for automated model lifecycle management.

    Schedule:
    - Hourly: Drift detection and performance monitoring
    - Every 6 hours: A/B test evaluation and promotion
    - Daily: Champion health monitoring
    - Weekly: Performance reports
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        orchestrator: Optional[RetrainingOrchestrator] = None,
        ab_testing: Optional[ABTestingManager] = None,
        promoter: Optional[ChampionPromoter] = None,
        enable_auto_retraining: bool = True,
        enable_auto_promotion: bool = True,
        enable_auto_rollback: bool = True
    ):
        """
        Initialize learning scheduler.

        Args:
            model_registry: Model registry
            orchestrator: Retraining orchestrator
            ab_testing: A/B testing manager
            promoter: Champion promoter
            enable_auto_retraining: Enable automatic retraining
            enable_auto_promotion: Enable automatic promotion
            enable_auto_rollback: Enable automatic rollback
        """
        if not SCHEDULE_AVAILABLE:
            raise ImportError(
                "schedule library required. Install with: pip install schedule"
            )

        self.model_registry = model_registry or ModelRegistry()
        self.orchestrator = orchestrator or RetrainingOrchestrator(
            model_registry=self.model_registry
        )
        self.ab_testing = ab_testing or ABTestingManager(
            model_registry=self.model_registry
        )
        self.promoter = promoter or ChampionPromoter(
            model_registry=self.model_registry,
            ab_testing=self.ab_testing
        )

        self.enable_auto_retraining = enable_auto_retraining
        self.enable_auto_promotion = enable_auto_promotion
        self.enable_auto_rollback = enable_auto_rollback

        # Statistics
        self.stats = {
            "cycles_run": 0,
            "retraining_triggered": 0,
            "promotions": 0,
            "rollbacks": 0,
            "errors": 0,
            "started_at": datetime.now().isoformat()
        }

        logger.info(
            f"LearningScheduler initialized "
            f"(auto_retraining={enable_auto_retraining}, "
            f"auto_promotion={enable_auto_promotion}, "
            f"auto_rollback={enable_auto_rollback})"
        )

    def start(self):
        """Start the continuous learning loop."""
        logger.info("🚀 Starting continuous learning scheduler...")

        # Schedule jobs
        schedule.every(1).hours.do(self._run_drift_check)
        schedule.every(6).hours.do(self._run_ab_test_evaluation)
        schedule.every(1).days.at("09:00").do(self._run_health_monitoring)
        schedule.every().monday.at("09:00").do(self._generate_weekly_report)

        # Run initial checks
        self._run_drift_check()

        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
            self._generate_shutdown_report()

        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            self.stats["errors"] += 1
            raise

    def run_single_cycle(self) -> Dict[str, Any]:
        """
        Run a single learning cycle (for testing/manual execution).

        Returns:
            Cycle results
        """
        logger.info("Running single learning cycle...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "drift_check": None,
            "ab_test_evaluation": None,
            "health_monitoring": None
        }

        try:
            # Drift check and retraining
            results["drift_check"] = self._run_drift_check()

            # A/B test evaluation
            results["ab_test_evaluation"] = self._run_ab_test_evaluation()

            # Health monitoring
            results["health_monitoring"] = self._run_health_monitoring()

            self.stats["cycles_run"] += 1

        except Exception as e:
            logger.error(f"Cycle error: {e}")
            results["error"] = str(e)
            self.stats["errors"] += 1

        return results

    def _run_drift_check(self) -> Dict[str, Any]:
        """Run drift detection and trigger retraining if needed."""
        logger.info("📊 Running drift check and performance monitoring...")

        try:
            # Check all model types
            # In production: Get model types from configuration
            model_types = ["vision_encoder", "language_decoder"]

            results = {}
            for model_type in model_types:
                logger.info(f"Checking {model_type}...")

                if self.enable_auto_retraining:
                    job_id = self.orchestrator.check_and_trigger_retraining(
                        model_type=model_type
                    )

                    if job_id:
                        logger.info(f"✅ Retraining triggered: {job_id}")
                        self.stats["retraining_triggered"] += 1

                        results[model_type] = {
                            "retraining_triggered": True,
                            "job_id": job_id
                        }
                    else:
                        results[model_type] = {
                            "retraining_triggered": False
                        }
                else:
                    logger.info("Auto-retraining disabled")
                    results[model_type] = {
                        "retraining_triggered": False,
                        "reason": "auto_retraining_disabled"
                    }

            return results

        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return {"error": str(e)}

    def _run_ab_test_evaluation(self) -> Dict[str, Any]:
        """Evaluate active A/B tests and promote if appropriate."""
        logger.info("🧪 Evaluating A/B tests...")

        try:
            active_tests = self.ab_testing.get_active_tests()

            if not active_tests:
                logger.info("No active A/B tests")
                return {"active_tests": 0}

            logger.info(f"Found {len(active_tests)} active A/B test(s)")

            results = []
            for test in active_tests:
                test_id = test["id"]
                logger.info(f"Evaluating test: {test_id}")

                # Check if test has run long enough
                if not self._test_ready_for_evaluation(test):
                    logger.info(f"Test {test_id} not ready for evaluation yet")
                    continue

                # Evaluate and promote
                if self.enable_auto_promotion:
                    promotion_result = self.promoter.evaluate_and_promote(
                        test_id=test_id,
                        auto_promote=True
                    )

                    if promotion_result.get("promoted"):
                        self.stats["promotions"] += 1
                        logger.info(f"✅ Model promoted from test {test_id}")

                    results.append(promotion_result)
                else:
                    logger.info("Auto-promotion disabled")

            return {
                "active_tests": len(active_tests),
                "evaluated_tests": len(results),
                "promotions": sum(1 for r in results if r.get("promoted")),
                "results": results
            }

        except Exception as e:
            logger.error(f"A/B test evaluation failed: {e}")
            return {"error": str(e)}

    def _run_health_monitoring(self) -> Dict[str, Any]:
        """Monitor champion model health."""
        logger.info("🏥 Running champion health monitoring...")

        try:
            # Get all champion models
            model_types = ["vision_encoder", "language_decoder"]

            results = {}
            for model_type in model_types:
                champion = self.model_registry.get_champion(model_type)

                if not champion:
                    results[model_type] = {
                        "status": "no_champion"
                    }
                    continue

                # Get recent predictions (in production: from PMG/database)
                recent_predictions = self._get_recent_predictions(
                    champion["id"],
                    hours=24
                )

                # Monitor health
                health_report = self.promoter.monitor_champion_health(
                    model_type=model_type,
                    recent_predictions=recent_predictions,
                    auto_rollback=self.enable_auto_rollback
                )

                results[model_type] = health_report

                if not health_report.get("healthy"):
                    logger.warning(
                        f"⚠️  {model_type} champion unhealthy: "
                        f"{health_report.get('severity')}"
                    )

                    if health_report.get("rollback"):
                        self.stats["rollbacks"] += 1

            return results

        except Exception as e:
            logger.error(f"Health monitoring failed: {e}")
            return {"error": str(e)}

    def _generate_weekly_report(self):
        """Generate weekly performance report."""
        logger.info("📈 Generating weekly report...")

        try:
            report = {
                "period": "weekly",
                "timestamp": datetime.now().isoformat(),
                "scheduler_stats": self.stats.copy(),
                "model_registry_stats": self.model_registry.get_statistics(),
                "active_tests": len(self.ab_testing.get_active_tests())
            }

            logger.info(f"Weekly Report:\n{self._format_report(report)}")

            # In production: Send report via email/Slack
            # self.notification_service.send_report(report)

        except Exception as e:
            logger.error(f"Report generation failed: {e}")

    def _generate_shutdown_report(self):
        """Generate report on shutdown."""
        logger.info("Generating shutdown report...")

        uptime = datetime.now() - datetime.fromisoformat(self.stats["started_at"])

        report = f"""
        Continuous Learning Scheduler Shutdown Report
        ============================================
        Uptime: {uptime}
        Cycles run: {self.stats['cycles_run']}
        Retraining triggered: {self.stats['retraining_triggered']}
        Promotions: {self.stats['promotions']}
        Rollbacks: {self.stats['rollbacks']}
        Errors: {self.stats['errors']}
        """

        logger.info(report)

    def _test_ready_for_evaluation(self, test: Dict[str, Any]) -> bool:
        """
        Check if A/B test has run long enough for evaluation.

        Args:
            test: A/B test record

        Returns:
            True if ready
        """
        # Minimum runtime: 24 hours
        started_at = datetime.fromisoformat(test["started_at"])
        runtime = datetime.now() - started_at

        if runtime < timedelta(hours=24):
            return False

        # Check sample count
        summary = self.ab_testing.get_test_summary(test["id"])

        champion_samples = summary["champion"]["predictions"]
        challenger_samples = summary["challenger"]["predictions"]

        # Minimum 1000 samples each
        return champion_samples >= 1000 and challenger_samples >= 1000

    def _get_recent_predictions(
        self,
        model_id: str,
        hours: int = 24
    ) -> list:
        """
        Get recent predictions for a model.

        Args:
            model_id: Model ID
            hours: Lookback period in hours

        Returns:
            List of predictions
        """
        # In production: Query from database/PMG
        # For now: Return empty list
        return []

    def _format_report(self, report: Dict[str, Any]) -> str:
        """Format report for display."""
        import json
        return json.dumps(report, indent=2)

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        stats = self.stats.copy()

        # Add uptime
        started_at = datetime.fromisoformat(stats["started_at"])
        stats["uptime_seconds"] = (datetime.now() - started_at).total_seconds()

        # Add model registry stats
        stats["model_registry"] = self.model_registry.get_statistics()

        return stats
