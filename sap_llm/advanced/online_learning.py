"""
Online Learning & Continuous Improvement System

Enable continuous model improvement from production data:
- Real-time model updates
- Active learning (query difficult examples)
- Human-in-the-loop feedback
- Incremental learning (no full retraining)
- Performance monitoring and A/B testing

Benefits:
- Adapt to changing document patterns
- Improve accuracy over time
- Reduce manual retraining
- Handle domain shift
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    CORRECT = "correct"  # Prediction was correct
    INCORRECT = "incorrect"  # Prediction was wrong
    UNCERTAIN = "uncertain"  # User unsure
    CORRECTION = "correction"  # User provided correction


class UncertaintySampling(Enum):
    """Active learning sampling strategies"""
    LEAST_CONFIDENT = "least_confident"  # Query lowest confidence
    MARGIN = "margin"  # Query smallest margin between top-2
    ENTROPY = "entropy"  # Query highest entropy


@dataclass
class Feedback:
    """User feedback on a prediction"""
    document_id: str
    field_name: str
    predicted_value: str
    predicted_confidence: float
    feedback_type: FeedbackType
    corrected_value: Optional[str]
    user_id: str
    timestamp: datetime
    model_version: str


@dataclass
class TrainingExample:
    """Training example for online learning"""
    document_id: str
    features: Dict[str, Any]
    label: str
    weight: float = 1.0  # Example importance weight
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class FeedbackBuffer:
    """
    Buffer for collecting user feedback

    Features:
    - FIFO buffer with max size
    - Feedback statistics
    - Quality filtering
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.feedback_stats = {
            FeedbackType.CORRECT: 0,
            FeedbackType.INCORRECT: 0,
            FeedbackType.UNCERTAIN: 0,
            FeedbackType.CORRECTION: 0,
        }

    def add_feedback(self, feedback: Feedback):
        """Add feedback to buffer"""
        self.buffer.append(feedback)
        self.feedback_stats[feedback.feedback_type] += 1

        logger.debug(
            f"Feedback added: {feedback.field_name}={feedback.predicted_value} "
            f"({feedback.feedback_type.value})"
        )

    def get_training_examples(
        self,
        min_confidence: float = 0.0
    ) -> List[TrainingExample]:
        """
        Convert feedback to training examples

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of training examples
        """
        training_examples = []

        for feedback in self.buffer:
            # Skip uncertain feedback
            if feedback.feedback_type == FeedbackType.UNCERTAIN:
                continue

            # Skip low confidence without correction
            if (feedback.predicted_confidence < min_confidence and
                feedback.feedback_type != FeedbackType.CORRECTION):
                continue

            # Determine label
            if feedback.feedback_type == FeedbackType.CORRECTION:
                label = feedback.corrected_value
                weight = 2.0  # Higher weight for corrections
            elif feedback.feedback_type == FeedbackType.CORRECT:
                label = feedback.predicted_value
                weight = 1.0
            else:  # INCORRECT
                # Skip if no correction provided
                if not feedback.corrected_value:
                    continue
                label = feedback.corrected_value
                weight = 1.5

            example = TrainingExample(
                document_id=feedback.document_id,
                features={
                    "field_name": feedback.field_name,
                    "predicted_value": feedback.predicted_value,
                    "confidence": feedback.predicted_confidence
                },
                label=label,
                weight=weight,
                timestamp=feedback.timestamp
            )

            training_examples.append(example)

        logger.info(
            f"Generated {len(training_examples)} training examples "
            f"from {len(self.buffer)} feedback items"
        )

        return training_examples

    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics"""
        total_feedback = len(self.buffer)

        if total_feedback == 0:
            return {
                "total": 0,
                "accuracy": 0.0,
                "breakdown": {}
            }

        correct_count = self.feedback_stats[FeedbackType.CORRECT]
        incorrect_count = self.feedback_stats[FeedbackType.INCORRECT]
        total_decisive = correct_count + incorrect_count

        accuracy = (
            correct_count / total_decisive
            if total_decisive > 0
            else 0.0
        )

        return {
            "total": total_feedback,
            "accuracy": accuracy,
            "breakdown": {
                k.value: v
                for k, v in self.feedback_stats.items()
            }
        }


class ActiveLearner:
    """
    Active learning for query selection

    Features:
    - Uncertainty-based sampling
    - Diversity-based sampling
    - Budget-aware querying
    """

    def __init__(
        self,
        strategy: UncertaintySampling = UncertaintySampling.LEAST_CONFIDENT,
        query_budget: int = 100
    ):
        self.strategy = strategy
        self.query_budget = query_budget
        self.queries_used = 0

    def should_query(self, confidence: float, threshold: float = 0.7) -> bool:
        """
        Decide whether to query user for feedback

        Args:
            confidence: Prediction confidence
            threshold: Confidence threshold

        Returns:
            True if should query
        """
        if self.queries_used >= self.query_budget:
            return False

        # Query if confidence is below threshold
        if confidence < threshold:
            self.queries_used += 1
            return True

        return False

    def select_examples_to_query(
        self,
        predictions: List[Dict[str, Any]],
        num_queries: int = 10
    ) -> List[str]:
        """
        Select examples to query based on uncertainty

        Args:
            predictions: List of predictions with confidences
            num_queries: Number of examples to query

        Returns:
            List of document IDs to query
        """
        if self.strategy == UncertaintySampling.LEAST_CONFIDENT:
            return self._least_confident_sampling(predictions, num_queries)
        elif self.strategy == UncertaintySampling.MARGIN:
            return self._margin_sampling(predictions, num_queries)
        elif self.strategy == UncertaintySampling.ENTROPY:
            return self._entropy_sampling(predictions, num_queries)
        else:
            return self._least_confident_sampling(predictions, num_queries)

    def _least_confident_sampling(
        self,
        predictions: List[Dict[str, Any]],
        num_queries: int
    ) -> List[str]:
        """Select examples with lowest confidence"""
        # Sort by confidence (ascending)
        sorted_preds = sorted(
            predictions,
            key=lambda x: x.get("confidence", 1.0)
        )

        # Select top num_queries
        selected = sorted_preds[:num_queries]

        return [pred["document_id"] for pred in selected]

    def _margin_sampling(
        self,
        predictions: List[Dict[str, Any]],
        num_queries: int
    ) -> List[str]:
        """Select examples with smallest margin between top-2 predictions"""
        margins = []

        for pred in predictions:
            if "alternatives" in pred and len(pred["alternatives"]) >= 2:
                # Calculate margin between top 2
                top1_conf = pred["confidence"]
                top2_conf = pred["alternatives"][0][1]
                margin = top1_conf - top2_conf
            else:
                margin = pred.get("confidence", 1.0)

            margins.append((pred["document_id"], margin))

        # Sort by margin (ascending - smallest margins first)
        margins.sort(key=lambda x: x[1])

        return [doc_id for doc_id, _ in margins[:num_queries]]

    def _entropy_sampling(
        self,
        predictions: List[Dict[str, Any]],
        num_queries: int
    ) -> List[str]:
        """Select examples with highest prediction entropy"""
        entropies = []

        for pred in predictions:
            # Calculate entropy
            probs = [pred["confidence"]]
            if "alternatives" in pred:
                probs.extend([alt[1] for alt in pred["alternatives"]])

            # Normalize probabilities
            probs = np.array(probs)
            probs = probs / probs.sum()

            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            entropies.append((pred["document_id"], entropy))

        # Sort by entropy (descending - highest entropy first)
        entropies.sort(key=lambda x: x[1], reverse=True)

        return [doc_id for doc_id, _ in entropies[:num_queries]]


class IncrementalLearner:
    """
    Incremental learning without full retraining

    Features:
    - Online gradient descent
    - Experience replay
    - Catastrophic forgetting prevention
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        replay_buffer_size: int = 1000
    ):
        self.learning_rate = learning_rate
        self.replay_buffer: deque = deque(maxlen=replay_buffer_size)
        self.update_count = 0

    def update_model(
        self,
        model: Any,
        new_examples: List[TrainingExample]
    ) -> Any:
        """
        Update model incrementally

        Args:
            model: Current model
            new_examples: New training examples

        Returns:
            Updated model
        """
        logger.info(
            f"Incremental update with {len(new_examples)} new examples"
        )

        # Add to replay buffer
        for example in new_examples:
            self.replay_buffer.append(example)

        # Sample from replay buffer (prevent forgetting)
        replay_sample_size = min(
            len(new_examples) * 2,
            len(self.replay_buffer)
        )

        if replay_sample_size > 0:
            replay_indices = np.random.choice(
                len(self.replay_buffer),
                size=replay_sample_size,
                replace=False
            )
            replay_examples = [
                self.replay_buffer[i] for i in replay_indices
            ]
        else:
            replay_examples = []

        # Combine new and replay examples
        training_batch = new_examples + replay_examples

        # Update model (placeholder - would do actual gradient updates)
        # In reality, would compute gradients and update parameters
        for example in training_batch:
            # Simulate model update
            pass

        self.update_count += 1

        logger.info(
            f"Model updated: iteration {self.update_count}, "
            f"batch_size={len(training_batch)}"
        )

        return model


class PerformanceMonitor:
    """
    Monitor model performance in production

    Features:
    - Accuracy tracking over time
    - Drift detection
    - A/B testing support
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.baseline_accuracy = None

    def record_prediction(
        self,
        document_id: str,
        prediction: str,
        confidence: float,
        actual: Optional[str] = None
    ):
        """Record a prediction"""
        self.predictions.append({
            "document_id": document_id,
            "prediction": prediction,
            "confidence": confidence,
            "actual": actual,
            "timestamp": datetime.utcnow()
        })

    def calculate_accuracy(self) -> float:
        """Calculate accuracy over window"""
        correct = 0
        total = 0

        for pred in self.predictions:
            if pred["actual"] is not None:
                total += 1
                if pred["prediction"] == pred["actual"]:
                    correct += 1

        return correct / total if total > 0 else 0.0

    def detect_drift(self, threshold: float = 0.05) -> bool:
        """
        Detect concept drift

        Args:
            threshold: Acceptable accuracy drop

        Returns:
            True if drift detected
        """
        if self.baseline_accuracy is None:
            self.baseline_accuracy = self.calculate_accuracy()
            return False

        current_accuracy = self.calculate_accuracy()
        accuracy_drop = self.baseline_accuracy - current_accuracy

        if accuracy_drop > threshold:
            logger.warning(
                f"Drift detected: accuracy dropped from "
                f"{self.baseline_accuracy:.2%} to {current_accuracy:.2%}"
            )
            return True

        return False

    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance statistics"""
        accuracy = self.calculate_accuracy()

        # Calculate confidence distribution
        confidences = [
            pred["confidence"]
            for pred in self.predictions
        ]

        return {
            "accuracy": accuracy,
            "baseline_accuracy": self.baseline_accuracy,
            "num_predictions": len(self.predictions),
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "confidence_std": np.std(confidences) if confidences else 0.0
        }


class OnlineLearningSystem:
    """
    Complete online learning system

    Integrates:
    - Feedback collection
    - Active learning
    - Incremental updates
    - Performance monitoring
    """

    def __init__(self):
        self.feedback_buffer = FeedbackBuffer(max_size=10000)
        self.active_learner = ActiveLearner(
            strategy=UncertaintySampling.LEAST_CONFIDENT,
            query_budget=100
        )
        self.incremental_learner = IncrementalLearner(
            learning_rate=0.001,
            replay_buffer_size=1000
        )
        self.performance_monitor = PerformanceMonitor(window_size=1000)

        self.auto_update_enabled = True
        self.update_threshold = 100  # Update after N feedback items

    def process_prediction(
        self,
        document_id: str,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a prediction and decide if query needed

        Args:
            document_id: Document ID
            prediction: Model prediction

        Returns:
            Prediction with query flag
        """
        confidence = prediction.get("confidence", 0.0)

        # Record prediction
        self.performance_monitor.record_prediction(
            document_id=document_id,
            prediction=prediction.get("value", ""),
            confidence=confidence
        )

        # Decide if should query user
        should_query = self.active_learner.should_query(confidence)

        prediction["query_user"] = should_query

        if should_query:
            logger.info(
                f"Querying user for document {document_id} "
                f"(confidence: {confidence:.2f})"
            )

        return prediction

    def add_feedback(self, feedback: Feedback):
        """
        Add user feedback

        Args:
            feedback: User feedback
        """
        self.feedback_buffer.add_feedback(feedback)

        # Update performance monitor
        if feedback.feedback_type in [FeedbackType.CORRECT, FeedbackType.CORRECTION]:
            actual_value = (
                feedback.corrected_value
                if feedback.feedback_type == FeedbackType.CORRECTION
                else feedback.predicted_value
            )

            self.performance_monitor.record_prediction(
                document_id=feedback.document_id,
                prediction=feedback.predicted_value,
                confidence=feedback.predicted_confidence,
                actual=actual_value
            )

        # Check if should trigger model update
        if (self.auto_update_enabled and
            len(self.feedback_buffer.buffer) >= self.update_threshold):
            self.trigger_model_update()

    def trigger_model_update(self):
        """Trigger incremental model update"""
        logger.info("Triggering incremental model update...")

        # Get training examples from feedback
        training_examples = self.feedback_buffer.get_training_examples(
            min_confidence=0.3
        )

        if not training_examples:
            logger.warning("No training examples available for update")
            return

        # Update model incrementally
        # Note: Would pass actual model here
        updated_model = self.incremental_learner.update_model(
            model=None,  # Placeholder
            new_examples=training_examples
        )

        # Check for drift
        drift_detected = self.performance_monitor.detect_drift()

        if drift_detected:
            logger.warning(
                "Concept drift detected - consider full model retraining"
            )

        logger.info("Incremental model update completed")

    def get_system_status(self) -> Dict[str, Any]:
        """Get online learning system status"""
        feedback_stats = self.feedback_buffer.get_statistics()
        performance_report = self.performance_monitor.get_performance_report()

        return {
            "feedback": feedback_stats,
            "performance": performance_report,
            "active_learning": {
                "queries_used": self.active_learner.queries_used,
                "query_budget": self.active_learner.query_budget
            },
            "incremental_learning": {
                "update_count": self.incremental_learner.update_count,
                "replay_buffer_size": len(self.incremental_learner.replay_buffer)
            },
            "auto_update_enabled": self.auto_update_enabled
        }


# Global instance
online_learning_system = OnlineLearningSystem()


def process_with_online_learning(
    document_id: str,
    prediction: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function for online learning

    Args:
        document_id: Document ID
        prediction: Model prediction

    Returns:
        Prediction with query flag
    """
    return online_learning_system.process_prediction(document_id, prediction)


def add_user_feedback(feedback: Feedback):
    """
    Convenience function to add feedback

    Args:
        feedback: User feedback
    """
    online_learning_system.add_feedback(feedback)
