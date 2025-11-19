"""
Self-Correction Engine - Orchestrates Error Detection and Correction.

Manages the complete self-correction workflow:
1. Detect errors in prediction
2. Try multiple correction strategies in sequence
3. Re-validate after each correction
4. Escalate to human if all strategies fail
5. Learn from corrections
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from sap_llm.correction.error_detector import ErrorDetector, ErrorReport
from sap_llm.correction.strategies import (
    CorrectionStrategy,
    CorrectionResult,
    RuleBasedCorrectionStrategy,
    RerunWithHigherConfidenceStrategy,
    ContextEnhancementStrategy,
    HumanInTheLoopStrategy,
    HumanReviewQueue,
)
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SelfCorrectionEngine:
    """
    Orchestrates the self-correction process.

    Workflow:
    1. Detect errors using ErrorDetector
    2. Try correction strategies in order of effectiveness
    3. Re-validate after each correction
    4. Continue until no errors or max attempts reached
    5. Escalate to human if unable to correct
    """

    def __init__(
        self,
        pmg=None,
        language_decoder=None,
        vision_encoder=None,
        max_attempts: int = 3,
        confidence_threshold: float = 0.80,
        pattern_learner=None
    ):
        """
        Initialize self-correction engine.

        Args:
            pmg: Optional ProcessMemoryGraph for historical data
            language_decoder: Optional language decoder for re-extraction
            vision_encoder: Optional vision encoder for better features
            max_attempts: Maximum correction attempts
            confidence_threshold: Minimum acceptable confidence
            pattern_learner: Optional error pattern learner
        """
        self.pmg = pmg
        self.language_decoder = language_decoder
        self.vision_encoder = vision_encoder
        self.max_attempts = max_attempts
        self.confidence_threshold = confidence_threshold
        self.pattern_learner = pattern_learner

        # Initialize error detector
        self.error_detector = ErrorDetector(pmg=pmg)

        # Initialize human review queue
        self.review_queue = HumanReviewQueue()

        # Initialize correction strategies in order of preference
        self.strategies = self._initialize_strategies()

        # Correction history
        self.correction_history = []

        logger.info(
            f"SelfCorrectionEngine initialized with {len(self.strategies)} strategies, "
            f"max_attempts={max_attempts}, confidence_threshold={confidence_threshold}"
        )

    def _initialize_strategies(self) -> List[CorrectionStrategy]:
        """Initialize correction strategies in order of application."""
        strategies = []

        # 1. Rule-based (fast, high confidence)
        strategies.append(RuleBasedCorrectionStrategy())

        # 2. Context enhancement (uses historical data)
        if self.pmg:
            strategies.append(ContextEnhancementStrategy(pmg=self.pmg))

        # 3. Re-run with better models (slower, more compute)
        if self.language_decoder:
            strategies.append(RerunWithHigherConfidenceStrategy(
                language_decoder=self.language_decoder,
                vision_encoder=self.vision_encoder
            ))

        # 4. Human-in-the-loop (last resort)
        strategies.append(HumanInTheLoopStrategy(review_queue=self.review_queue))

        logger.info(f"Initialized {len(strategies)} correction strategies")
        return strategies

    def correct_prediction(
        self,
        prediction: Dict[str, Any],
        context: Dict[str, Any],
        enable_learning: bool = True
    ) -> Dict[str, Any]:
        """
        Attempt to correct errors in prediction.

        Args:
            prediction: Original prediction with potential errors
            context: Context information (document, doc_type, etc.)
            enable_learning: Whether to learn from corrections

        Returns:
            Corrected prediction with metadata
        """
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("Starting self-correction workflow")
        logger.info("=" * 80)

        # Initialize correction tracking
        correction_metadata = {
            "original_prediction": prediction.copy(),
            "attempts": [],
            "total_attempts": 0,
            "strategies_tried": [],
            "success": False,
            "required_human_review": False,
            "start_time": start_time.isoformat(),
        }

        # Initial error detection
        error_report = self.error_detector.detect_errors(prediction, context)

        logger.info(
            f"Initial error detection: {len(error_report.errors)} error type(s), "
            f"confidence={error_report.overall_confidence:.3f}, "
            f"needs_correction={error_report.needs_correction}"
        )

        # If no errors detected, return original prediction
        if not error_report.needs_correction:
            logger.info("No correction needed - prediction quality acceptable")

            correction_metadata.update({
                "success": True,
                "end_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
                "initial_quality": "acceptable"
            })

            return {
                **prediction,
                "correction_metadata": correction_metadata
            }

        # Attempt corrections
        current_prediction = prediction.copy()
        current_error_report = error_report

        for attempt in range(self.max_attempts):
            logger.info(f"\n--- Correction Attempt {attempt + 1}/{self.max_attempts} ---")

            correction_metadata["total_attempts"] = attempt + 1

            # Get the highest priority error to address
            target_error = current_error_report.get_highest_severity_error()

            if not target_error:
                logger.info("No more errors to correct")
                break

            logger.info(
                f"Targeting error: type={target_error.type}, "
                f"severity={target_error.severity}, "
                f"fields={target_error.fields}"
            )

            # Try strategies for this error
            correction_successful = False

            for strategy in self.strategies:
                # Check if this strategy is appropriate for this error/attempt
                if not self._should_try_strategy(strategy, target_error, attempt):
                    continue

                logger.info(f"Trying strategy: {strategy.name}")
                correction_metadata["strategies_tried"].append(strategy.name)

                try:
                    # Attempt correction
                    result = strategy.correct(current_prediction, target_error, context)

                    # Log attempt
                    attempt_log = {
                        "attempt_number": attempt + 1,
                        "strategy": strategy.name,
                        "error_type": target_error.type,
                        "error_severity": target_error.severity,
                        "success": result.success,
                        "requires_human": result.requires_human,
                        "fields_corrected": result.fields_corrected,
                        "confidence_improvement": result.confidence_improvement,
                        "metadata": result.metadata
                    }
                    correction_metadata["attempts"].append(attempt_log)

                    # Handle human escalation
                    if result.requires_human:
                        logger.warning("Correction requires human review")
                        correction_metadata["required_human_review"] = True
                        correction_metadata["human_review_task_id"] = result.task_id
                        correction_metadata["end_time"] = datetime.now().isoformat()
                        correction_metadata["duration_seconds"] = (
                            datetime.now() - start_time
                        ).total_seconds()

                        return self._prepare_for_human_review(
                            current_prediction,
                            result.task_id,
                            correction_metadata
                        )

                    # If correction was successful, update prediction
                    if result.success:
                        original_prediction = current_prediction.copy()
                        current_prediction = result.corrected_prediction

                        logger.info(
                            f"✓ Strategy successful: {strategy.name}, "
                            f"corrected {len(result.fields_corrected)} field(s), "
                            f"confidence improvement: {result.confidence_improvement:+.3f}"
                        )

                        # Re-detect errors in corrected prediction
                        current_error_report = self.error_detector.detect_errors(
                            current_prediction,
                            context
                        )

                        logger.info(
                            f"Post-correction: {len(current_error_report.errors)} error type(s), "
                            f"confidence={current_error_report.overall_confidence:.3f}"
                        )

                        # Learn from successful correction
                        if enable_learning and self.pattern_learner:
                            try:
                                self.pattern_learner.learn_from_correction(
                                    original_prediction=original_prediction,
                                    corrected_prediction=current_prediction,
                                    correction_strategy=strategy.name,
                                    context=context
                                )
                            except Exception as e:
                                logger.error(f"Failed to learn from correction: {e}")

                        correction_successful = True

                        # Check if all errors are now resolved
                        if not current_error_report.needs_correction:
                            logger.info("✓ All errors corrected successfully!")
                            correction_metadata["success"] = True
                            correction_metadata["final_confidence"] = current_error_report.overall_confidence
                            correction_metadata["end_time"] = datetime.now().isoformat()
                            correction_metadata["duration_seconds"] = (
                                datetime.now() - start_time
                            ).total_seconds()

                            return {
                                **current_prediction,
                                "correction_metadata": correction_metadata
                            }

                        # Break to re-evaluate with new error state
                        break

                    else:
                        logger.info(f"✗ Strategy unsuccessful: {strategy.name}")

                except Exception as e:
                    logger.error(f"Strategy {strategy.name} failed with error: {e}", exc_info=True)
                    attempt_log = {
                        "attempt_number": attempt + 1,
                        "strategy": strategy.name,
                        "error_type": target_error.type,
                        "success": False,
                        "error": str(e)
                    }
                    correction_metadata["attempts"].append(attempt_log)

            # If no strategy succeeded this attempt, continue to next attempt
            if not correction_successful:
                logger.warning(f"Attempt {attempt + 1} did not improve prediction")

        # Max attempts reached without full correction
        logger.warning(
            f"Max attempts ({self.max_attempts}) reached. "
            f"Errors remaining: {len(current_error_report.errors)}"
        )

        # Decide whether to escalate to human
        if self._should_escalate(current_error_report, self.max_attempts):
            logger.warning("Escalating to human review")
            return self._escalate_to_human(
                current_prediction,
                current_error_report,
                context,
                correction_metadata
            )

        # Return best effort correction
        correction_metadata.update({
            "success": False,
            "partial_success": current_prediction != prediction,
            "final_confidence": current_error_report.overall_confidence,
            "remaining_errors": [e.to_dict() for e in current_error_report.errors],
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
        })

        logger.info(
            f"Self-correction complete: partial success, "
            f"final confidence={current_error_report.overall_confidence:.3f}"
        )

        return {
            **current_prediction,
            "correction_metadata": correction_metadata
        }

    def _should_try_strategy(
        self,
        strategy: CorrectionStrategy,
        error: ErrorReport,
        attempt: int
    ) -> bool:
        """
        Determine if a strategy should be tried.

        Args:
            strategy: Correction strategy
            error: Current error
            attempt: Current attempt number

        Returns:
            True if strategy should be tried
        """
        # Try rule-based first (fast, deterministic)
        if isinstance(strategy, RuleBasedCorrectionStrategy):
            return error.type == "rule_violation" or error.type == "anomaly"

        # Try context enhancement on first/second attempt if PMG available
        if isinstance(strategy, ContextEnhancementStrategy):
            return attempt <= 1 and self.pmg is not None

        # Try re-extraction on second attempt
        if isinstance(strategy, RerunWithHigherConfidenceStrategy):
            return attempt == 1 and self.language_decoder is not None

        # Human escalation on final attempt or critical errors
        if isinstance(strategy, HumanInTheLoopStrategy):
            return attempt >= self.max_attempts - 1 or error.severity == "critical"

        return True

    def _should_escalate(
        self,
        error_report: ErrorReport,
        attempts: int
    ) -> bool:
        """
        Determine if prediction should be escalated to human review.

        Args:
            error_report: Current error report
            attempts: Number of attempts made

        Returns:
            True if should escalate to human
        """
        # Escalate if:
        # 1. Low confidence after multiple attempts
        if attempts >= self.max_attempts and error_report.overall_confidence < self.confidence_threshold:
            logger.info(
                f"Escalating: confidence {error_report.overall_confidence:.3f} < "
                f"threshold {self.confidence_threshold}"
            )
            return True

        # 2. High/critical severity errors remain
        if any(e.severity in ["high", "critical"] for e in error_report.errors):
            logger.info("Escalating: high/critical severity errors present")
            return True

        # 3. Business-critical fields have low confidence
        critical_fields = ["total_amount", "vendor_id", "invoice_number"]
        for field in critical_fields:
            # This would need access to the prediction, which we don't have here
            # In practice, this check would be done by caller or with prediction passed in
            pass

        return False

    def _prepare_for_human_review(
        self,
        prediction: Dict[str, Any],
        task_id: str,
        correction_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare prediction for human review."""
        return {
            **prediction,
            "requires_human_review": True,
            "human_review_task_id": task_id,
            "correction_metadata": correction_metadata,
            "status": "pending_human_review"
        }

    def _escalate_to_human(
        self,
        prediction: Dict[str, Any],
        error_report: ErrorReport,
        context: Dict[str, Any],
        correction_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Escalate prediction to human review."""
        # Use human-in-the-loop strategy to create review task
        human_strategy = HumanInTheLoopStrategy(review_queue=self.review_queue)

        # Get highest severity error
        target_error = error_report.get_highest_severity_error()

        result = human_strategy.correct(prediction, target_error, context)

        correction_metadata.update({
            "required_human_review": True,
            "human_review_task_id": result.task_id,
            "escalation_reason": "max_attempts_reached",
            "final_confidence": error_report.overall_confidence,
            "remaining_errors": [e.to_dict() for e in error_report.errors],
            "end_time": datetime.now().isoformat(),
        })

        return {
            **prediction,
            "requires_human_review": True,
            "human_review_task_id": result.task_id,
            "correction_metadata": correction_metadata,
            "status": "pending_human_review"
        }

    def get_correction_stats(self) -> Dict[str, Any]:
        """Get statistics about correction attempts."""
        total_corrections = len(self.correction_history)

        if total_corrections == 0:
            return {"total_corrections": 0}

        successful = sum(1 for c in self.correction_history if c.get("success"))
        human_escalations = sum(
            1 for c in self.correction_history
            if c.get("required_human_review")
        )

        return {
            "total_corrections": total_corrections,
            "successful": successful,
            "success_rate": successful / total_corrections,
            "human_escalations": human_escalations,
            "escalation_rate": human_escalations / total_corrections,
        }
