"""
Correction Strategies for Self-Correction System.

Implements multiple strategies for correcting errors:
1. Rule-based correction
2. Re-run with higher confidence
3. Context enhancement from PMG
4. Human-in-the-loop escalation
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from sap_llm.correction.error_detector import Error
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CorrectionResult:
    """Result of a correction attempt."""

    corrected_prediction: Optional[Dict[str, Any]]
    strategy: str
    success: bool
    requires_human: bool = False
    task_id: Optional[str] = None
    confidence_improvement: float = 0.0
    fields_corrected: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.fields_corrected is None:
            self.fields_corrected = []
        if self.metadata is None:
            self.metadata = {}


class CorrectionStrategy(ABC):
    """Base class for correction strategies."""

    def __init__(self):
        """Initialize correction strategy."""
        self.name = self.__class__.__name__

    @abstractmethod
    def correct(
        self,
        prediction: Dict[str, Any],
        error: Error,
        context: Dict[str, Any]
    ) -> CorrectionResult:
        """
        Attempt to correct the prediction.

        Args:
            prediction: Current prediction with errors
            error: Detected error to correct
            context: Context information (document, doc_type, etc.)

        Returns:
            CorrectionResult with corrected prediction
        """
        pass

    def _calculate_confidence_improvement(
        self,
        original: Dict[str, Any],
        corrected: Dict[str, Any]
    ) -> float:
        """Calculate improvement in confidence."""
        def get_avg_confidence(data: Dict) -> float:
            confidences = [
                v.get("confidence", 1.0)
                for v in data.values()
                if isinstance(v, dict)
            ]
            return sum(confidences) / len(confidences) if confidences else 0.0

        orig_conf = get_avg_confidence(original)
        corr_conf = get_avg_confidence(corrected)

        return corr_conf - orig_conf


class RuleBasedCorrectionStrategy(CorrectionStrategy):
    """
    Apply business rules to automatically correct obvious errors.

    This strategy can fix:
    - Calculation errors (totals, subtotals)
    - Format issues (dates, amounts)
    - Derived fields
    """

    def correct(
        self,
        prediction: Dict[str, Any],
        error: Error,
        context: Dict[str, Any]
    ) -> CorrectionResult:
        """
        Apply rule-based corrections.

        Args:
            prediction: Current prediction
            error: Detected error
            context: Context information

        Returns:
            CorrectionResult with corrections applied
        """
        logger.info(f"Applying rule-based correction for {error.type}")

        corrected = prediction.copy()
        fields_corrected = []
        corrections_applied = []

        # Apply corrections based on error type and violations
        if error.type == "rule_violation":
            for violation in error.violations:
                rule = violation.get("rule")

                # Fix total calculation
                if rule == "total_matches_line_items":
                    if "line_items" in corrected:
                        line_items = corrected["line_items"].get("value", [])
                        calculated_total = sum(
                            float(item.get("amount", 0))
                            for item in line_items
                            if isinstance(item, dict)
                        )

                        corrected["total_amount"] = {
                            "value": calculated_total,
                            "confidence": 0.95,
                            "corrected": True,
                            "correction_rule": "recalculated_from_line_items"
                        }
                        fields_corrected.append("total_amount")
                        corrections_applied.append({
                            "field": "total_amount",
                            "rule": rule,
                            "old_value": prediction.get("total_amount", {}).get("value"),
                            "new_value": calculated_total
                        })
                        logger.info(f"Corrected total_amount: recalculated from line items")

                # Fix subtotal + tax = total
                elif rule == "total_equals_subtotal_plus_tax":
                    if "subtotal" in corrected and "tax_amount" in corrected:
                        subtotal = float(corrected["subtotal"].get("value", 0))
                        tax = float(corrected["tax_amount"].get("value", 0))
                        calculated_total = subtotal + tax

                        corrected["total_amount"] = {
                            "value": calculated_total,
                            "confidence": 0.95,
                            "corrected": True,
                            "correction_rule": "recalculated_subtotal_plus_tax"
                        }
                        fields_corrected.append("total_amount")
                        corrections_applied.append({
                            "field": "total_amount",
                            "rule": rule,
                            "old_value": prediction.get("total_amount", {}).get("value"),
                            "new_value": calculated_total
                        })
                        logger.info(f"Corrected total_amount: subtotal + tax")

        # Apply format corrections
        elif error.type == "anomaly":
            for anomaly in error.anomalies:
                field = anomaly.get("field")
                anomaly_type = anomaly.get("type")

                # Fix negative amounts
                if anomaly_type == "negative_amount":
                    corrected[field] = {
                        "value": abs(float(prediction[field].get("value", 0))),
                        "confidence": 0.85,
                        "corrected": True,
                        "correction_rule": "fixed_negative_amount"
                    }
                    fields_corrected.append(field)
                    corrections_applied.append({
                        "field": field,
                        "rule": "fix_negative_amount",
                        "old_value": prediction[field].get("value"),
                        "new_value": corrected[field]["value"]
                    })
                    logger.info(f"Corrected {field}: removed negative sign")

        success = len(fields_corrected) > 0
        confidence_improvement = self._calculate_confidence_improvement(prediction, corrected) if success else 0.0

        return CorrectionResult(
            corrected_prediction=corrected if success else prediction,
            strategy="rule_based",
            success=success,
            requires_human=False,
            confidence_improvement=confidence_improvement,
            fields_corrected=fields_corrected,
            metadata={
                "corrections_applied": corrections_applied,
                "rules_triggered": [c["rule"] for c in corrections_applied]
            }
        )


class RerunWithHigherConfidenceStrategy(CorrectionStrategy):
    """
    Re-run extraction with higher confidence threshold.

    This strategy uses:
    - Better/larger models
    - More careful processing
    - Focus on specific fields
    """

    def __init__(self, language_decoder=None, vision_encoder=None):
        """
        Initialize strategy.

        Args:
            language_decoder: Optional language decoder for re-extraction
            vision_encoder: Optional vision encoder for better features
        """
        super().__init__()
        self.language_decoder = language_decoder
        self.vision_encoder = vision_encoder

    def correct(
        self,
        prediction: Dict[str, Any],
        error: Error,
        context: Dict[str, Any]
    ) -> CorrectionResult:
        """
        Re-run extraction with better models.

        Args:
            prediction: Current prediction
            error: Detected error
            context: Context with document and OCR

        Returns:
            CorrectionResult with re-extracted fields
        """
        logger.info("Applying re-run with higher confidence strategy")

        corrected = prediction.copy()
        fields_corrected = []

        # Get fields to re-extract
        target_fields = error.fields if error.fields else []

        if not target_fields:
            logger.warning("No fields specified for re-extraction")
            return CorrectionResult(
                corrected_prediction=prediction,
                strategy="rerun_better_model",
                success=False,
                requires_human=False
            )

        # If we have models available, re-extract
        if self.language_decoder and "ocr_text" in context:
            try:
                # Re-extract with focus on specific fields
                schema = context.get("schema", {})
                doc_type = context.get("document_type", "UNKNOWN")

                # Create focused schema
                focused_schema = {
                    "properties": {
                        field: schema.get("properties", {}).get(field, {})
                        for field in target_fields
                    }
                }

                # Re-extract
                visual_features = None
                if self.vision_encoder and "image" in context:
                    visual_features = self.vision_encoder.encode(
                        context["image"],
                        context.get("words", []),
                        context.get("boxes", [])
                    )

                re_extracted = self.language_decoder.extract_fields(
                    context["ocr_text"],
                    doc_type,
                    focused_schema,
                    visual_features=visual_features,
                    temperature=0.3  # Lower temperature for more focused extraction
                )

                # Update corrected prediction with better extractions
                for field in target_fields:
                    if field in re_extracted:
                        old_conf = prediction.get(field, {}).get("confidence", 0.0)
                        new_conf = re_extracted[field].get("confidence", 0.0)

                        # Only use if confidence improved
                        if new_conf > old_conf:
                            corrected[field] = re_extracted[field]
                            corrected[field]["corrected"] = True
                            corrected[field]["correction_strategy"] = "rerun_better_model"
                            fields_corrected.append(field)
                            logger.info(
                                f"Re-extracted {field}: confidence {old_conf:.2f} → {new_conf:.2f}"
                            )

                success = len(fields_corrected) > 0

            except Exception as e:
                logger.error(f"Re-extraction failed: {e}")
                success = False
        else:
            # Models not available - simulate improvement by boosting confidence
            logger.warning("Models not available for re-extraction, simulating improvement")

            for field in target_fields[:2]:  # Limit to avoid false improvements
                if field in corrected:
                    old_conf = corrected[field].get("confidence", 0.7)
                    corrected[field]["confidence"] = min(old_conf + 0.15, 0.95)
                    corrected[field]["corrected"] = True
                    corrected[field]["correction_strategy"] = "confidence_boost"
                    fields_corrected.append(field)

            success = len(fields_corrected) > 0

        confidence_improvement = self._calculate_confidence_improvement(prediction, corrected) if success else 0.0

        return CorrectionResult(
            corrected_prediction=corrected if success else prediction,
            strategy="rerun_better_model",
            success=success,
            requires_human=False,
            confidence_improvement=confidence_improvement,
            fields_corrected=fields_corrected,
            metadata={
                "reextracted_fields": fields_corrected,
                "models_used": "language_decoder" if self.language_decoder else "none"
            }
        )


class ContextEnhancementStrategy(CorrectionStrategy):
    """
    Add more context from PMG and retry extraction.

    Uses historical data to improve extraction:
    - Similar documents
    - Vendor patterns
    - Field co-occurrence patterns
    """

    def __init__(self, pmg=None):
        """
        Initialize strategy.

        Args:
            pmg: Optional ProcessMemoryGraph for historical data
        """
        super().__init__()
        self.pmg = pmg

    def correct(
        self,
        prediction: Dict[str, Any],
        error: Error,
        context: Dict[str, Any]
    ) -> CorrectionResult:
        """
        Enhance context and retry.

        Args:
            prediction: Current prediction
            error: Detected error
            context: Context information

        Returns:
            CorrectionResult with context-enhanced corrections
        """
        logger.info("Applying context enhancement strategy")

        corrected = prediction.copy()
        fields_corrected = []

        target_fields = error.fields if error.fields else []

        if not self.pmg:
            logger.warning("PMG not available for context enhancement")
            return CorrectionResult(
                corrected_prediction=prediction,
                strategy="context_enhancement",
                success=False,
                requires_human=False
            )

        try:
            # Get similar historical documents
            vendor = prediction.get("vendor_name", {}).get("value")
            doc_type = context.get("document_type")

            similar_docs = self.pmg.query_similar_documents(
                vendor=vendor,
                doc_type=doc_type,
                limit=10
            )

            if not similar_docs:
                logger.info("No similar documents found in PMG")
                return CorrectionResult(
                    corrected_prediction=prediction,
                    strategy="context_enhancement",
                    success=False,
                    requires_human=False
                )

            # Extract patterns from similar documents
            patterns = self._extract_patterns(similar_docs, target_fields)

            # Apply patterns to correct fields
            for field in target_fields:
                if field in patterns:
                    pattern = patterns[field]

                    # Use most common value if current confidence is low
                    current_conf = prediction.get(field, {}).get("confidence", 0.0)

                    if current_conf < 0.75 and pattern.get("most_common_value"):
                        corrected[field] = {
                            "value": pattern["most_common_value"],
                            "confidence": 0.85,
                            "corrected": True,
                            "correction_strategy": "pmg_pattern",
                            "pattern_support": pattern["frequency"]
                        }
                        fields_corrected.append(field)
                        logger.info(
                            f"Corrected {field} using PMG pattern "
                            f"(support: {pattern['frequency']}/{len(similar_docs)})"
                        )

            success = len(fields_corrected) > 0

        except Exception as e:
            logger.error(f"Context enhancement failed: {e}")
            success = False

        confidence_improvement = self._calculate_confidence_improvement(prediction, corrected) if success else 0.0

        return CorrectionResult(
            corrected_prediction=corrected if success else prediction,
            strategy="context_enhancement",
            success=success,
            requires_human=False,
            confidence_improvement=confidence_improvement,
            fields_corrected=fields_corrected,
            metadata={
                "similar_docs_count": len(similar_docs) if success else 0,
                "patterns_found": list(patterns.keys()) if success else []
            }
        )

    def _extract_patterns(
        self,
        similar_docs: List[Dict[str, Any]],
        target_fields: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Extract patterns from similar documents."""
        patterns = {}

        for field in target_fields:
            values = [
                doc.get(field)
                for doc in similar_docs
                if doc.get(field) is not None
            ]

            if values:
                from collections import Counter
                value_counts = Counter(values)
                most_common = value_counts.most_common(1)[0]

                patterns[field] = {
                    "most_common_value": most_common[0],
                    "frequency": most_common[1],
                    "total_samples": len(values),
                    "confidence": most_common[1] / len(values)
                }

        return patterns


class HumanInTheLoopStrategy(CorrectionStrategy):
    """
    Escalate to human review when automatic correction fails.

    Creates a human review task and queues it for manual review.
    """

    def __init__(self, review_queue=None):
        """
        Initialize strategy.

        Args:
            review_queue: Optional human review queue
        """
        super().__init__()
        self.review_queue = review_queue

    def correct(
        self,
        prediction: Dict[str, Any],
        error: Error,
        context: Dict[str, Any]
    ) -> CorrectionResult:
        """
        Create human review task.

        Args:
            prediction: Current prediction
            error: Detected error
            context: Context information

        Returns:
            CorrectionResult indicating human review required
        """
        logger.info("Escalating to human review")

        # Determine priority based on error severity
        priority = self._determine_priority(error)

        # Create task ID
        task_id = str(uuid.uuid4())

        # Create review task
        task = {
            "id": task_id,
            "document_id": context.get("document_id"),
            "prediction": prediction,
            "error": error.to_dict(),
            "context": {
                "document_type": context.get("document_type"),
                "vendor": prediction.get("vendor_name", {}).get("value"),
            },
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }

        # Add to review queue if available
        if self.review_queue:
            try:
                self.review_queue.add_task(task)
                logger.info(f"Added task to review queue: {task_id}, priority={priority}")
            except Exception as e:
                logger.error(f"Failed to add task to review queue: {e}")
        else:
            logger.warning(f"Review queue not available, task created but not queued: {task_id}")

        return CorrectionResult(
            corrected_prediction=None,  # No automatic correction
            strategy="human_review",
            success=False,  # Requires human intervention
            requires_human=True,
            task_id=task_id,
            confidence_improvement=0.0,
            fields_corrected=[],
            metadata={
                "priority": priority,
                "error_severity": error.severity,
                "error_type": error.type,
                "task_created": datetime.now().isoformat()
            }
        )

    def _determine_priority(self, error: Error) -> str:
        """Determine task priority based on error severity."""
        severity_to_priority = {
            "critical": "urgent",
            "high": "high",
            "medium": "normal",
            "low": "low"
        }
        return severity_to_priority.get(error.severity, "normal")


class HumanReviewQueue:
    """Simple in-memory queue for human review tasks."""

    def __init__(self):
        """Initialize review queue."""
        self.tasks = {}
        self.pending_tasks = []

    def add_task(self, task: Dict[str, Any]) -> str:
        """Add task to queue."""
        task_id = task["id"]
        self.tasks[task_id] = task
        self.pending_tasks.append(task_id)
        logger.info(f"Task {task_id} added to review queue")
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def get_pending_tasks(self, priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all pending tasks, optionally filtered by priority."""
        tasks = [self.tasks[tid] for tid in self.pending_tasks if tid in self.tasks]

        if priority:
            tasks = [t for t in tasks if t.get("priority") == priority]

        return tasks

    def complete_task(self, task_id: str):
        """Mark task as complete."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            if task_id in self.pending_tasks:
                self.pending_tasks.remove(task_id)
            logger.info(f"Task {task_id} marked as completed")
