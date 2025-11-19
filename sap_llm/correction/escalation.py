"""
Escalation Manager for Human-in-the-Loop Review.

Manages escalation of difficult cases to human reviewers:
1. Determines when to escalate
2. Creates human review tasks
3. Manages review queue with priorities
4. Tracks SLA compliance
5. Processes human feedback
6. Learns from human corrections
"""

import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sap_llm.correction.error_detector import ErrorReport
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SLATracker:
    """Tracks SLA (Service Level Agreement) for human review tasks."""

    def __init__(self):
        """Initialize SLA tracker."""
        # SLA times in hours
        self.sla_times = {
            "urgent": 2,    # 2 hours
            "high": 8,      # 8 hours (same business day)
            "normal": 24,   # 1 business day
            "low": 72       # 3 business days
        }

    def get_sla(self, priority: str) -> Dict[str, Any]:
        """
        Get SLA for a given priority.

        Args:
            priority: Task priority

        Returns:
            SLA information
        """
        hours = self.sla_times.get(priority, 24)
        deadline = datetime.now() + timedelta(hours=hours)

        return {
            "priority": priority,
            "sla_hours": hours,
            "deadline": deadline.isoformat(),
            "deadline_timestamp": deadline
        }

    def is_sla_breached(self, task: Dict[str, Any]) -> bool:
        """Check if task SLA is breached."""
        sla = task.get("sla", {})
        deadline = sla.get("deadline_timestamp")

        if not deadline:
            return False

        if isinstance(deadline, str):
            from dateutil import parser
            deadline = parser.parse(deadline)

        return datetime.now() > deadline

    def get_remaining_time(self, task: Dict[str, Any]) -> timedelta:
        """Get remaining time until SLA deadline."""
        sla = task.get("sla", {})
        deadline = sla.get("deadline_timestamp")

        if not deadline:
            return timedelta(hours=24)

        if isinstance(deadline, str):
            from dateutil import parser
            deadline = parser.parse(deadline)

        return deadline - datetime.now()


class EscalationManager:
    """
    Manages escalation to human review and processes feedback.

    Responsibilities:
    1. Decide when to escalate
    2. Create and queue review tasks
    3. Track SLA compliance
    4. Process human feedback
    5. Learn from corrections
    """

    def __init__(
        self,
        review_queue=None,
        pattern_learner=None,
        pmg=None,
        confidence_threshold: float = 0.70,
        max_auto_attempts: int = 3
    ):
        """
        Initialize escalation manager.

        Args:
            review_queue: Human review queue
            pattern_learner: Error pattern learner
            pmg: ProcessMemoryGraph for storing corrections
            confidence_threshold: Minimum confidence to avoid escalation
            max_auto_attempts: Maximum automatic correction attempts
        """
        from sap_llm.correction.strategies import HumanReviewQueue

        self.review_queue = review_queue or HumanReviewQueue()
        self.pattern_learner = pattern_learner
        self.pmg = pmg
        self.confidence_threshold = confidence_threshold
        self.max_auto_attempts = max_auto_attempts

        # SLA tracker
        self.sla_tracker = SLATracker()

        # Escalation statistics
        self.escalation_stats = defaultdict(int)

        logger.info("EscalationManager initialized")

    def should_escalate(
        self,
        prediction: Dict[str, Any],
        error_report: ErrorReport,
        attempts: int,
        context: Dict[str, Any]
    ) -> bool:
        """
        Determine if prediction should be escalated to human review.

        Args:
            prediction: Current prediction
            error_report: Error detection report
            attempts: Number of correction attempts made
            context: Context information

        Returns:
            True if should escalate
        """
        logger.info("Evaluating escalation criteria")

        # Criterion 1: Low confidence after multiple attempts
        if attempts >= self.max_auto_attempts:
            if error_report.overall_confidence < self.confidence_threshold:
                logger.info(
                    f"Escalating: low confidence ({error_report.overall_confidence:.3f}) "
                    f"after {attempts} attempts"
                )
                self.escalation_stats["low_confidence_after_attempts"] += 1
                return True

        # Criterion 2: High or critical severity errors
        high_severity_errors = [
            e for e in error_report.errors
            if e.severity in ["high", "critical"]
        ]

        if high_severity_errors:
            logger.info(
                f"Escalating: {len(high_severity_errors)} high/critical severity error(s)"
            )
            self.escalation_stats["high_severity_errors"] += 1
            return True

        # Criterion 3: Business-critical fields with low confidence
        critical_fields = ["total_amount", "vendor_id", "vendor_name", "invoice_number"]

        for field in critical_fields:
            if field in prediction:
                field_data = prediction[field]

                if isinstance(field_data, dict):
                    confidence = field_data.get("confidence", 1.0)

                    if confidence < 0.80:
                        logger.info(
                            f"Escalating: critical field '{field}' has low confidence ({confidence:.3f})"
                        )
                        self.escalation_stats["critical_field_low_confidence"] += 1
                        return True

        # Criterion 4: Multiple failed correction attempts
        if attempts >= self.max_auto_attempts and error_report.has_errors:
            logger.info(
                f"Escalating: max attempts ({attempts}) reached with unresolved errors"
            )
            self.escalation_stats["max_attempts_reached"] += 1
            return True

        # Criterion 5: Business rule violations that couldn't be auto-corrected
        rule_violations = [
            e for e in error_report.errors
            if e.type == "rule_violation"
        ]

        if rule_violations and attempts >= 2:
            logger.info(
                f"Escalating: {len(rule_violations)} rule violation(s) after {attempts} attempts"
            )
            self.escalation_stats["uncorrected_rule_violations"] += 1
            return True

        logger.info("Escalation not required")
        return False

    def escalate_to_human(
        self,
        prediction: Dict[str, Any],
        error_report: ErrorReport,
        context: Dict[str, Any],
        priority: Optional[str] = None
    ) -> str:
        """
        Create human review task and add to queue.

        Args:
            prediction: Current prediction
            error_report: Error detection report
            context: Context information
            priority: Optional priority override

        Returns:
            Task ID
        """
        logger.info("Creating human review task")

        # Determine priority if not provided
        if priority is None:
            priority = self._determine_priority(error_report)

        # Generate task ID
        task_id = str(uuid.uuid4())

        # Get SLA
        sla = self.sla_tracker.get_sla(priority)

        # Create task
        task = {
            "id": task_id,
            "document_id": context.get("document_id", "unknown"),
            "prediction": prediction,
            "errors": [e.to_dict() for e in error_report.errors],
            "error_report": error_report.to_dict(),
            "context": {
                "document_type": context.get("document_type"),
                "vendor": prediction.get("vendor_name", {}).get("value"),
                "invoice_number": prediction.get("invoice_number", {}).get("value"),
                "total_amount": prediction.get("total_amount", {}).get("value"),
            },
            "priority": priority,
            "sla": sla,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "assigned_to": None,
            "completed_at": None
        }

        # Add to review queue
        try:
            self.review_queue.add_task(task)
            logger.info(
                f"Human review task created: task_id={task_id}, "
                f"priority={priority}, sla_hours={sla['sla_hours']}"
            )

            # Send notification (would integrate with actual notification system)
            self._notify_reviewers(task)

            self.escalation_stats["total_escalations"] += 1

        except Exception as e:
            logger.error(f"Failed to create review task: {e}")
            raise

        return task_id

    def process_human_feedback(
        self,
        task_id: str,
        corrected_prediction: Dict[str, Any],
        reviewer_id: str,
        reviewer_notes: Optional[str] = None
    ):
        """
        Process human review feedback and learn from it.

        Args:
            task_id: Review task ID
            corrected_prediction: Human-corrected prediction
            reviewer_id: ID of reviewer
            reviewer_notes: Optional notes from reviewer
        """
        logger.info(f"Processing human feedback for task {task_id}")

        # Get original task
        task = self.review_queue.get_task(task_id)

        if not task:
            logger.error(f"Task {task_id} not found")
            raise ValueError(f"Task {task_id} not found")

        original_prediction = task["prediction"]

        # Learn from human correction
        if self.pattern_learner:
            try:
                self.pattern_learner.learn_from_correction(
                    original_prediction=original_prediction,
                    corrected_prediction=corrected_prediction,
                    correction_strategy="human_review",
                    context=task["context"]
                )
                logger.info("Learned from human correction")
            except Exception as e:
                logger.error(f"Failed to learn from human correction: {e}")

        # Update PMG with ground truth
        if self.pmg:
            try:
                self.pmg.update_with_ground_truth(
                    document_id=task["document_id"],
                    corrected_data=corrected_prediction,
                    reviewer_id=reviewer_id
                )
                logger.info("Updated PMG with ground truth")
            except Exception as e:
                logger.error(f"Failed to update PMG: {e}")

        # Complete task
        task["status"] = "completed"
        task["completed_at"] = datetime.now().isoformat()
        task["assigned_to"] = reviewer_id
        task["reviewer_notes"] = reviewer_notes
        task["corrected_prediction"] = corrected_prediction

        # Check SLA compliance
        sla_breached = self.sla_tracker.is_sla_breached(task)
        task["sla_breached"] = sla_breached

        if sla_breached:
            logger.warning(f"Task {task_id} completed after SLA breach")
            self.escalation_stats["sla_breaches"] += 1

        # Mark complete in queue
        self.review_queue.complete_task(task_id)

        self.escalation_stats["completed_reviews"] += 1

        logger.info(
            f"Human feedback processed: task_id={task_id}, "
            f"reviewer={reviewer_id}, sla_breached={sla_breached}"
        )

    def get_pending_tasks(
        self,
        priority: Optional[str] = None,
        include_sla_info: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get pending human review tasks.

        Args:
            priority: Optional priority filter
            include_sla_info: Include SLA remaining time info

        Returns:
            List of pending tasks
        """
        tasks = self.review_queue.get_pending_tasks(priority=priority)

        if include_sla_info:
            for task in tasks:
                remaining = self.sla_tracker.get_remaining_time(task)
                task["sla_remaining_hours"] = remaining.total_seconds() / 3600
                task["sla_breached"] = self.sla_tracker.is_sla_breached(task)

        return tasks

    def get_escalation_stats(self) -> Dict[str, Any]:
        """
        Get escalation statistics.

        Returns:
            Escalation statistics
        """
        total = self.escalation_stats.get("total_escalations", 0)
        completed = self.escalation_stats.get("completed_reviews", 0)

        return {
            "total_escalations": total,
            "completed_reviews": completed,
            "pending_reviews": total - completed,
            "sla_breaches": self.escalation_stats.get("sla_breaches", 0),
            "escalation_reasons": {
                "low_confidence_after_attempts": self.escalation_stats.get("low_confidence_after_attempts", 0),
                "high_severity_errors": self.escalation_stats.get("high_severity_errors", 0),
                "critical_field_low_confidence": self.escalation_stats.get("critical_field_low_confidence", 0),
                "max_attempts_reached": self.escalation_stats.get("max_attempts_reached", 0),
                "uncorrected_rule_violations": self.escalation_stats.get("uncorrected_rule_violations", 0),
            }
        }

    def _determine_priority(self, error_report: ErrorReport) -> str:
        """
        Determine task priority based on error severity.

        Args:
            error_report: Error detection report

        Returns:
            Priority level
        """
        # Check for critical errors
        if any(e.severity == "critical" for e in error_report.errors):
            return "urgent"

        # Check for high severity errors
        if any(e.severity == "high" for e in error_report.errors):
            return "high"

        # Check overall confidence
        if error_report.overall_confidence < 0.50:
            return "high"
        elif error_report.overall_confidence < 0.70:
            return "normal"

        return "low"

    def _notify_reviewers(self, task: Dict[str, Any]):
        """
        Send notification to reviewers about new task.

        Args:
            task: Review task

        Note:
            This is a placeholder. In production, would integrate with
            actual notification system (email, Slack, etc.)
        """
        logger.info(
            f"[NOTIFICATION] New review task: "
            f"priority={task['priority']}, "
            f"doc_type={task['context']['document_type']}, "
            f"sla_hours={task['sla']['sla_hours']}"
        )

        # In production, would send actual notifications:
        # - Email to reviewer team
        # - Slack message
        # - Dashboard notification
        # - SMS for urgent tasks
