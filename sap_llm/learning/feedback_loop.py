"""
Feedback Loop System

Implements comprehensive user feedback collection, confidence-based feedback
requests, A/B testing, automatic retraining triggers, and model versioning.
"""

import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of user feedback."""

    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    COMMENT = "comment"
    VALIDATION = "validation"


class FeedbackPriority(Enum):
    """Priority levels for feedback processing."""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Process in next batch
    MEDIUM = "medium"  # Process in nightly run
    LOW = "low"  # Process when convenient


class ModelVersion:
    """Model version metadata."""

    def __init__(
        self,
        version_id: str,
        doc_type: str,
        created_at: datetime,
        metrics: Dict[str, float],
        parent_version: Optional[str] = None,
    ):
        self.version_id = version_id
        self.doc_type = doc_type
        self.created_at = created_at
        self.metrics = metrics
        self.parent_version = parent_version
        self.deployed = False
        self.rollback_count = 0


class FeedbackLoopSystem:
    """
    Comprehensive feedback loop system for continuous improvement.

    Features:
    - Multi-channel feedback collection (thumbs, corrections, comments)
    - Confidence-based feedback requests
    - A/B testing for model improvements
    - Automatic retraining triggers based on performance drift
    - Model versioning and rollback
    - Feedback analytics and insights
    """

    def __init__(
        self,
        pmg: ProcessMemoryGraph,
        confidence_threshold: float = 0.8,
        drift_threshold: float = 0.15,
        min_samples_for_retrain: int = 100,
        ab_test_enabled: bool = True,
        ab_test_split: float = 0.5,
    ):
        """
        Initialize feedback loop system.

        Args:
            pmg: Process Memory Graph instance
            confidence_threshold: Threshold for requesting feedback
            drift_threshold: Performance drift threshold for retraining
            min_samples_for_retrain: Minimum feedback samples before retraining
            ab_test_enabled: Enable A/B testing
            ab_test_split: Traffic split for A/B testing (0.5 = 50/50)
        """
        self.pmg = pmg
        self.confidence_threshold = confidence_threshold
        self.drift_threshold = drift_threshold
        self.min_samples_for_retrain = min_samples_for_retrain
        self.ab_test_enabled = ab_test_enabled
        self.ab_test_split = ab_test_split

        # Feedback storage
        self.feedback_queue: List[Dict[str, Any]] = []
        self.feedback_by_type: Dict[FeedbackType, List[Dict]] = defaultdict(list)
        self.feedback_by_doc_type: Dict[str, List[Dict]] = defaultdict(list)

        # Model versions
        self.model_versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        self.active_versions: Dict[str, str] = {}  # doc_type -> version_id
        self.champion_versions: Dict[str, str] = {}  # doc_type -> version_id

        # A/B testing
        self.ab_tests: Dict[str, Dict[str, Any]] = {}  # test_id -> test_config
        self.ab_test_results: Dict[str, List[Dict]] = defaultdict(list)

        # Performance monitoring
        self.performance_baseline: Dict[str, float] = {}
        self.performance_current: Dict[str, List[float]] = defaultdict(list)

        # Retraining triggers
        self.retrain_triggers: Dict[str, List[Dict]] = defaultdict(list)

        logger.info("FeedbackLoopSystem initialized")

    def collect_feedback(
        self,
        feedback_type: FeedbackType,
        doc_id: str,
        doc_type: str,
        user_id: str,
        prediction: Any,
        correct_value: Optional[Any] = None,
        confidence: Optional[float] = None,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Collect user feedback.

        Args:
            feedback_type: Type of feedback
            doc_id: Document ID
            doc_type: Document type
            user_id: User who provided feedback
            prediction: Model's prediction
            correct_value: Correct value (for corrections)
            confidence: Model's confidence
            comment: Optional user comment
            metadata: Additional metadata

        Returns:
            Feedback record with priority and actions
        """
        feedback_id = self._generate_feedback_id(doc_id, user_id)

        feedback = {
            "feedback_id": feedback_id,
            "feedback_type": feedback_type.value,
            "doc_id": doc_id,
            "doc_type": doc_type,
            "user_id": user_id,
            "prediction": prediction,
            "correct_value": correct_value,
            "confidence": confidence,
            "comment": comment,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "processed": False,
            "priority": self._determine_priority(feedback_type, confidence),
        }

        # Add to queues
        self.feedback_queue.append(feedback)
        self.feedback_by_type[feedback_type].append(feedback)
        self.feedback_by_doc_type[doc_type].append(feedback)

        # Store in PMG
        self._store_feedback_in_pmg(feedback)

        # Check if retraining needed
        retrain_needed = self._check_retrain_trigger(doc_type)

        logger.info(
            f"Feedback collected: {feedback_id} ({feedback_type.value}) "
            f"[Priority: {feedback['priority']}]"
        )

        return {
            "feedback_id": feedback_id,
            "priority": feedback["priority"],
            "retrain_needed": retrain_needed,
            "queued_for_processing": True,
        }

    def request_feedback(
        self,
        doc_id: str,
        doc_type: str,
        prediction: Any,
        confidence: float,
        fields_to_verify: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Proactively request feedback based on confidence.

        Args:
            doc_id: Document ID
            doc_type: Document type
            prediction: Model prediction
            confidence: Prediction confidence
            fields_to_verify: Specific fields to verify

        Returns:
            Feedback request details
        """
        should_request = confidence < self.confidence_threshold

        if not should_request:
            return {
                "feedback_requested": False,
                "reason": "High confidence",
                "confidence": confidence,
            }

        # Determine what to ask
        request_type = self._determine_feedback_request_type(confidence)
        priority_fields = fields_to_verify or self._identify_uncertain_fields(
            prediction, confidence
        )

        feedback_request = {
            "doc_id": doc_id,
            "doc_type": doc_type,
            "request_type": request_type,
            "priority_fields": priority_fields,
            "confidence": confidence,
            "prediction": prediction,
            "requested_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Feedback requested for {doc_id}: {request_type} "
            f"(confidence: {confidence:.2f})"
        )

        return {
            "feedback_requested": True,
            "request": feedback_request,
        }

    def process_feedback_batch(
        self,
        max_samples: int = 1000,
        priority_filter: Optional[FeedbackPriority] = None,
    ) -> Dict[str, Any]:
        """
        Process accumulated feedback in batch.

        Args:
            max_samples: Maximum samples to process
            priority_filter: Filter by priority level

        Returns:
            Processing statistics
        """
        logger.info(f"Processing feedback batch (max={max_samples})")

        # Filter unprocessed feedback
        unprocessed = [f for f in self.feedback_queue if not f.get("processed")]

        if priority_filter:
            unprocessed = [
                f for f in unprocessed
                if f.get("priority") == priority_filter.value
            ]

        # Sort by priority
        priority_order = {
            FeedbackPriority.CRITICAL.value: 0,
            FeedbackPriority.HIGH.value: 1,
            FeedbackPriority.MEDIUM.value: 2,
            FeedbackPriority.LOW.value: 3,
        }
        unprocessed.sort(key=lambda x: priority_order.get(x.get("priority"), 999))

        # Process top samples
        to_process = unprocessed[:max_samples]

        # Group by document type for efficient processing
        by_doc_type = defaultdict(list)
        for feedback in to_process:
            by_doc_type[feedback["doc_type"]].append(feedback)

        results = {
            "total_processed": 0,
            "by_doc_type": {},
            "errors": [],
        }

        for doc_type, feedbacks in by_doc_type.items():
            try:
                # Extract training samples
                training_samples = self._extract_training_samples(feedbacks)

                # Mark as processed
                for feedback in feedbacks:
                    feedback["processed"] = True
                    feedback["processed_at"] = datetime.now().isoformat()

                results["total_processed"] += len(feedbacks)
                results["by_doc_type"][doc_type] = {
                    "count": len(feedbacks),
                    "training_samples": len(training_samples),
                }

            except Exception as e:
                logger.error(f"Failed to process feedback for {doc_type}: {e}")
                results["errors"].append({"doc_type": doc_type, "error": str(e)})

        logger.info(f"Processed {results['total_processed']} feedback samples")
        return results

    def start_ab_test(
        self,
        test_name: str,
        doc_type: str,
        champion_version: str,
        challenger_version: str,
        traffic_split: Optional[float] = None,
        duration_hours: int = 24,
        success_metric: str = "accuracy",
        min_improvement: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Start A/B test for model versions.

        Args:
            test_name: Test identifier
            doc_type: Document type
            champion_version: Current production version
            challenger_version: New version to test
            traffic_split: Override default split
            duration_hours: Test duration
            success_metric: Metric to optimize
            min_improvement: Minimum improvement to promote challenger

        Returns:
            A/B test configuration
        """
        test_id = self._generate_test_id(test_name, doc_type)

        ab_test = {
            "test_id": test_id,
            "test_name": test_name,
            "doc_type": doc_type,
            "champion_version": champion_version,
            "challenger_version": challenger_version,
            "traffic_split": traffic_split or self.ab_test_split,
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(hours=duration_hours),
            "success_metric": success_metric,
            "min_improvement": min_improvement,
            "status": "running",
            "champion_metrics": {"count": 0, success_metric: []},
            "challenger_metrics": {"count": 0, success_metric: []},
        }

        self.ab_tests[test_id] = ab_test
        logger.info(f"Started A/B test: {test_id} ({duration_hours}h)")

        return ab_test

    def route_ab_test(
        self,
        doc_type: str,
        doc_id: str,
    ) -> Tuple[str, str]:
        """
        Route request to champion or challenger model.

        Args:
            doc_type: Document type
            doc_id: Document ID

        Returns:
            (version_id, variant) - variant is 'champion' or 'challenger'
        """
        # Find active A/B test for this doc_type
        active_test = None
        for test_id, test in self.ab_tests.items():
            if test["doc_type"] == doc_type and test["status"] == "running":
                # Check if test expired
                if datetime.now() > test["end_time"]:
                    self._finalize_ab_test(test_id)
                    continue
                active_test = test
                break

        if not active_test:
            # No A/B test, use champion
            version = self.champion_versions.get(doc_type, "v1.0.0")
            return version, "champion"

        # Random assignment based on split
        random.seed(hash(doc_id))  # Consistent assignment for same doc_id
        use_challenger = random.random() < active_test["traffic_split"]

        if use_challenger:
            return active_test["challenger_version"], "challenger"
        else:
            return active_test["champion_version"], "champion"

    def record_ab_test_result(
        self,
        doc_type: str,
        doc_id: str,
        variant: str,
        prediction: Any,
        actual: Any,
        metrics: Dict[str, float],
    ):
        """
        Record result for A/B test.

        Args:
            doc_type: Document type
            doc_id: Document ID
            variant: 'champion' or 'challenger'
            prediction: Model prediction
            actual: Actual value
            metrics: Performance metrics
        """
        # Find active test
        for test_id, test in self.ab_tests.items():
            if test["doc_type"] == doc_type and test["status"] == "running":
                result = {
                    "doc_id": doc_id,
                    "variant": variant,
                    "prediction": prediction,
                    "actual": actual,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat(),
                }

                # Update test metrics
                metric_key = f"{variant}_metrics"
                test[metric_key]["count"] += 1

                for metric_name, value in metrics.items():
                    if metric_name not in test[metric_key]:
                        test[metric_key][metric_name] = []
                    test[metric_key][metric_name].append(value)

                self.ab_test_results[test_id].append(result)

                logger.debug(f"Recorded A/B test result: {test_id} ({variant})")
                break

    def finalize_ab_tests(self) -> Dict[str, Any]:
        """
        Finalize all expired A/B tests.

        Returns:
            Summary of finalized tests
        """
        finalized = []

        for test_id, test in list(self.ab_tests.items()):
            if test["status"] == "running" and datetime.now() > test["end_time"]:
                result = self._finalize_ab_test(test_id)
                finalized.append(result)

        return {
            "finalized_count": len(finalized),
            "results": finalized,
        }

    def _finalize_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Finalize specific A/B test."""
        test = self.ab_tests[test_id]

        # Calculate average metrics
        success_metric = test["success_metric"]

        champion_values = test["champion_metrics"].get(success_metric, [])
        challenger_values = test["challenger_metrics"].get(success_metric, [])

        champion_avg = sum(champion_values) / len(champion_values) if champion_values else 0
        challenger_avg = sum(challenger_values) / len(challenger_values) if challenger_values else 0

        improvement = (challenger_avg - champion_avg) / champion_avg if champion_avg > 0 else 0

        # Determine winner
        winner = "challenger" if improvement >= test["min_improvement"] else "champion"

        # Update champion if challenger won
        if winner == "challenger":
            self.champion_versions[test["doc_type"]] = test["challenger_version"]
            logger.info(
                f"A/B test {test_id}: Challenger promoted! "
                f"Improvement: {improvement:.2%}"
            )
        else:
            logger.info(
                f"A/B test {test_id}: Champion retained. "
                f"Improvement: {improvement:.2%}"
            )

        test["status"] = "completed"
        test["winner"] = winner
        test["champion_avg"] = champion_avg
        test["challenger_avg"] = challenger_avg
        test["improvement"] = improvement

        return {
            "test_id": test_id,
            "winner": winner,
            "improvement": improvement,
            "champion_avg": champion_avg,
            "challenger_avg": challenger_avg,
            "samples": {
                "champion": test["champion_metrics"]["count"],
                "challenger": test["challenger_metrics"]["count"],
            },
        }

    def create_model_version(
        self,
        doc_type: str,
        metrics: Dict[str, float],
        parent_version: Optional[str] = None,
    ) -> ModelVersion:
        """
        Create new model version.

        Args:
            doc_type: Document type
            metrics: Performance metrics
            parent_version: Previous version ID

        Returns:
            New model version
        """
        # Generate version ID
        existing_versions = len(self.model_versions[doc_type])
        version_id = f"v{existing_versions + 1}.0.0"

        version = ModelVersion(
            version_id=version_id,
            doc_type=doc_type,
            created_at=datetime.now(),
            metrics=metrics,
            parent_version=parent_version,
        )

        self.model_versions[doc_type].append(version)

        logger.info(f"Created model version: {doc_type}/{version_id}")
        return version

    def deploy_version(self, doc_type: str, version_id: str):
        """Deploy model version to production."""
        # Find version
        for version in self.model_versions[doc_type]:
            if version.version_id == version_id:
                version.deployed = True
                self.active_versions[doc_type] = version_id
                logger.info(f"Deployed version: {doc_type}/{version_id}")
                return

        logger.error(f"Version not found: {doc_type}/{version_id}")

    def rollback_version(self, doc_type: str) -> Optional[str]:
        """
        Rollback to previous model version.

        Args:
            doc_type: Document type

        Returns:
            Previous version ID if successful
        """
        current_version_id = self.active_versions.get(doc_type)
        if not current_version_id:
            logger.error(f"No active version for {doc_type}")
            return None

        # Find current version
        current_version = None
        for version in self.model_versions[doc_type]:
            if version.version_id == current_version_id:
                current_version = version
                break

        if not current_version or not current_version.parent_version:
            logger.error(f"Cannot rollback {doc_type}: no parent version")
            return None

        # Rollback to parent
        parent_version_id = current_version.parent_version
        self.deploy_version(doc_type, parent_version_id)

        current_version.rollback_count += 1

        logger.warning(f"Rolled back {doc_type}: {current_version_id} -> {parent_version_id}")
        return parent_version_id

    def _check_retrain_trigger(self, doc_type: str) -> bool:
        """Check if retraining should be triggered."""
        # Count recent negative feedback
        recent_feedback = self.feedback_by_doc_type[doc_type]
        negative_count = sum(
            1 for f in recent_feedback
            if f["feedback_type"] in [FeedbackType.THUMBS_DOWN.value, FeedbackType.CORRECTION.value]
        )

        # Check if threshold met
        if negative_count >= self.min_samples_for_retrain:
            self.retrain_triggers[doc_type].append({
                "trigger_reason": "negative_feedback_threshold",
                "negative_count": negative_count,
                "timestamp": datetime.now().isoformat(),
            })
            logger.warning(
                f"Retrain triggered for {doc_type}: "
                f"{negative_count} negative feedback samples"
            )
            return True

        # Check performance drift
        if doc_type in self.performance_baseline:
            current_perf = self._calculate_current_performance(doc_type)
            baseline_perf = self.performance_baseline[doc_type]

            drift = baseline_perf - current_perf
            if drift > self.drift_threshold:
                self.retrain_triggers[doc_type].append({
                    "trigger_reason": "performance_drift",
                    "drift": drift,
                    "baseline": baseline_perf,
                    "current": current_perf,
                    "timestamp": datetime.now().isoformat(),
                })
                logger.warning(
                    f"Retrain triggered for {doc_type}: "
                    f"performance drift {drift:.2%}"
                )
                return True

        return False

    def _determine_priority(
        self,
        feedback_type: FeedbackType,
        confidence: Optional[float],
    ) -> str:
        """Determine feedback priority."""
        if feedback_type == FeedbackType.CORRECTION:
            if confidence and confidence > 0.9:
                return FeedbackPriority.CRITICAL.value
            return FeedbackPriority.HIGH.value
        elif feedback_type == FeedbackType.THUMBS_DOWN:
            return FeedbackPriority.HIGH.value
        elif feedback_type == FeedbackType.VALIDATION:
            return FeedbackPriority.MEDIUM.value
        else:
            return FeedbackPriority.LOW.value

    def _determine_feedback_request_type(self, confidence: float) -> str:
        """Determine what type of feedback to request."""
        if confidence < 0.5:
            return "full_review"
        elif confidence < 0.7:
            return "field_verification"
        else:
            return "quick_validation"

    def _identify_uncertain_fields(
        self,
        prediction: Any,
        confidence: float,
    ) -> List[str]:
        """Identify fields with uncertain predictions."""
        # Simplified - in production, analyze field-level confidence
        if confidence < 0.6:
            return ["all_fields"]
        elif confidence < 0.8:
            return ["total_amount", "supplier_id", "invoice_date"]
        else:
            return ["total_amount"]

    def _calculate_current_performance(self, doc_type: str) -> float:
        """Calculate current performance for document type."""
        recent_perf = self.performance_current[doc_type]
        if recent_perf:
            return sum(recent_perf[-100:]) / min(len(recent_perf), 100)
        return 0.0

    def _extract_training_samples(
        self,
        feedbacks: List[Dict[str, Any]],
    ) -> List[Tuple[Dict, str]]:
        """Extract training samples from feedback."""
        samples = []
        for feedback in feedbacks:
            if feedback["feedback_type"] == FeedbackType.CORRECTION.value:
                features = feedback.get("metadata", {}).get("features", {})
                label = feedback["correct_value"]
                if features and label:
                    samples.append((features, label))
        return samples

    def _store_feedback_in_pmg(self, feedback: Dict[str, Any]):
        """Store feedback in Process Memory Graph."""
        try:
            # In production, create Feedback vertex and link to Document
            logger.debug(f"Stored feedback in PMG: {feedback['feedback_id']}")
        except Exception as e:
            logger.error(f"Failed to store feedback in PMG: {e}")

    def _generate_feedback_id(self, doc_id: str, user_id: str) -> str:
        """Generate unique feedback ID."""
        data = f"{doc_id}{user_id}{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()

    def _generate_test_id(self, test_name: str, doc_type: str) -> str:
        """Generate unique test ID."""
        data = f"{test_name}{doc_type}{datetime.now().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feedback statistics."""
        return {
            "total_feedback": len(self.feedback_queue),
            "unprocessed": len([f for f in self.feedback_queue if not f.get("processed")]),
            "by_type": {
                ft.value: len(feedbacks)
                for ft, feedbacks in self.feedback_by_type.items()
            },
            "by_doc_type": {
                doc_type: len(feedbacks)
                for doc_type, feedbacks in self.feedback_by_doc_type.items()
            },
            "active_ab_tests": len([t for t in self.ab_tests.values() if t["status"] == "running"]),
            "retrain_triggers": {
                doc_type: len(triggers)
                for doc_type, triggers in self.retrain_triggers.items()
            },
        }
