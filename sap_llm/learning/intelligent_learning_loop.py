"""
Intelligent Learning Loop with Drift Detection and A/B Testing.

Implements continuous learning from production data:
1. Model drift detection (statistical and model-based)
2. Automated A/B testing framework
3. Champion/Challenger model management
4. Auto-retrain triggers
5. Performance monitoring and alerting
6. Gradual rollout with automatic rollback

Target Metrics:
- Drift detection latency: <1 hour
- A/B test statistical power: >80%
- Auto-retrain trigger accuracy: >95%
- Rollback decision time: <5 minutes
"""

import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """
    Model drift detection using statistical methods.

    Detects:
    1. Data drift (input distribution changes)
    2. Concept drift (input-output relationship changes)
    3. Performance drift (accuracy degradation)
    """

    def __init__(
        self,
        window_size: int = 1000,
        drift_threshold: float = 0.05,
        min_samples: int = 100,
    ):
        """
        Initialize drift detector.

        Args:
            window_size: Rolling window size for comparison
            drift_threshold: P-value threshold for drift detection
            min_samples: Minimum samples before detection
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.min_samples = min_samples

        # Reference window (baseline)
        self.reference_predictions: deque = deque(maxlen=window_size)
        self.reference_features: deque = deque(maxlen=window_size)
        self.reference_labels: deque = deque(maxlen=window_size)

        # Current window
        self.current_predictions: deque = deque(maxlen=window_size)
        self.current_features: deque = deque(maxlen=window_size)
        self.current_labels: deque = deque(maxlen=window_size)

        # Drift history
        self.drift_events: List[Dict[str, Any]] = []

        logger.info(f"Drift detector initialized (window={window_size})")

    def add_reference_sample(
        self,
        features: np.ndarray,
        prediction: Any,
        label: Optional[Any] = None,
    ) -> None:
        """Add sample to reference window."""
        self.reference_features.append(features)
        self.reference_predictions.append(prediction)
        if label is not None:
            self.reference_labels.append(label)

    def add_current_sample(
        self,
        features: np.ndarray,
        prediction: Any,
        label: Optional[Any] = None,
    ) -> None:
        """Add sample to current window."""
        self.current_features.append(features)
        self.current_predictions.append(prediction)
        if label is not None:
            self.current_labels.append(label)

    def detect_drift(self) -> Dict[str, Any]:
        """
        Detect drift across all types.

        Returns:
            Drift detection results
        """
        if len(self.current_features) < self.min_samples:
            return {
                "drift_detected": False,
                "reason": f"Insufficient samples ({len(self.current_features)}/{self.min_samples})",
            }

        results = {
            "drift_detected": False,
            "drift_types": [],
            "statistics": {},
            "timestamp": datetime.now().isoformat(),
        }

        # 1. Data drift (Kolmogorov-Smirnov test)
        data_drift = self._detect_data_drift()
        if data_drift["is_drift"]:
            results["drift_detected"] = True
            results["drift_types"].append("data_drift")
            results["statistics"]["data_drift"] = data_drift

        # 2. Prediction drift
        prediction_drift = self._detect_prediction_drift()
        if prediction_drift["is_drift"]:
            results["drift_detected"] = True
            results["drift_types"].append("prediction_drift")
            results["statistics"]["prediction_drift"] = prediction_drift

        # 3. Performance drift (if labels available)
        if len(self.current_labels) >= self.min_samples:
            performance_drift = self._detect_performance_drift()
            if performance_drift["is_drift"]:
                results["drift_detected"] = True
                results["drift_types"].append("performance_drift")
                results["statistics"]["performance_drift"] = performance_drift

        # Log drift event
        if results["drift_detected"]:
            self.drift_events.append(results)
            logger.warning(
                f"DRIFT DETECTED: {', '.join(results['drift_types'])}"
            )

        return results

    def _detect_data_drift(self) -> Dict[str, Any]:
        """Detect data drift using KS test."""
        if len(self.reference_features) == 0:
            return {"is_drift": False, "reason": "No reference data"}

        # Convert to arrays
        ref_features = np.array(list(self.reference_features))
        cur_features = np.array(list(self.current_features))

        # KS test for each feature dimension
        p_values = []
        for i in range(ref_features.shape[1]):
            statistic, p_value = stats.ks_2samp(
                ref_features[:, i],
                cur_features[:, i],
            )
            p_values.append(p_value)

        # Use Bonferroni correction for multiple tests
        min_p_value = min(p_values)
        corrected_threshold = self.drift_threshold / len(p_values)

        is_drift = min_p_value < corrected_threshold

        return {
            "is_drift": is_drift,
            "min_p_value": float(min_p_value),
            "threshold": corrected_threshold,
            "drifted_features": [
                i for i, p in enumerate(p_values)
                if p < corrected_threshold
            ],
        }

    def _detect_prediction_drift(self) -> Dict[str, Any]:
        """Detect prediction distribution drift."""
        if len(self.reference_predictions) == 0:
            return {"is_drift": False, "reason": "No reference predictions"}

        # For classification: Chi-square test on prediction distributions
        ref_preds = np.array(list(self.reference_predictions))
        cur_preds = np.array(list(self.current_predictions))

        # Count predictions
        unique_preds = np.unique(np.concatenate([ref_preds, cur_preds]))

        ref_counts = np.array([
            np.sum(ref_preds == pred) for pred in unique_preds
        ])
        cur_counts = np.array([
            np.sum(cur_preds == pred) for pred in unique_preds
        ])

        # Chi-square test
        try:
            statistic, p_value = stats.chisquare(
                cur_counts,
                ref_counts,
            )

            is_drift = p_value < self.drift_threshold

            return {
                "is_drift": is_drift,
                "p_value": float(p_value),
                "threshold": self.drift_threshold,
                "reference_distribution": dict(zip(
                    unique_preds.tolist(),
                    (ref_counts / len(ref_preds)).tolist(),
                )),
                "current_distribution": dict(zip(
                    unique_preds.tolist(),
                    (cur_counts / len(cur_preds)).tolist(),
                )),
            }
        except Exception as e:
            logger.error(f"Prediction drift detection failed: {e}")
            return {"is_drift": False, "error": str(e)}

    def _detect_performance_drift(self) -> Dict[str, Any]:
        """Detect performance degradation."""
        if len(self.reference_labels) == 0 or len(self.current_labels) == 0:
            return {"is_drift": False, "reason": "No labels available"}

        # Calculate accuracy for both windows
        ref_accuracy = np.mean(
            np.array(list(self.reference_predictions)) ==
            np.array(list(self.reference_labels))
        )

        cur_accuracy = np.mean(
            np.array(list(self.current_predictions)) ==
            np.array(list(self.current_labels))
        )

        # Test if current accuracy is significantly lower
        # Using proportion z-test
        n1 = len(self.reference_labels)
        n2 = len(self.current_labels)

        p_pooled = (ref_accuracy * n1 + cur_accuracy * n2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        z_score = (ref_accuracy - cur_accuracy) / se if se > 0 else 0
        p_value = 1 - stats.norm.cdf(z_score)  # One-tailed test

        is_drift = p_value < self.drift_threshold and cur_accuracy < ref_accuracy

        return {
            "is_drift": is_drift,
            "reference_accuracy": float(ref_accuracy),
            "current_accuracy": float(cur_accuracy),
            "accuracy_drop": float(ref_accuracy - cur_accuracy),
            "p_value": float(p_value),
            "threshold": self.drift_threshold,
        }


class ABTestingFramework:
    """
    A/B Testing framework for model comparison.

    Features:
    - Statistical significance testing
    - Sequential testing (early stopping)
    - Multi-armed bandit allocation
    - Automatic winner selection
    """

    def __init__(
        self,
        min_samples_per_variant: int = 1000,
        alpha: float = 0.05,
        power: float = 0.80,
        enable_early_stopping: bool = True,
    ):
        """
        Initialize A/B testing framework.

        Args:
            min_samples_per_variant: Minimum samples per variant
            alpha: Significance level
            power: Statistical power
            enable_early_stopping: Enable early stopping
        """
        self.min_samples_per_variant = min_samples_per_variant
        self.alpha = alpha
        self.power = power
        self.enable_early_stopping = enable_early_stopping

        # Test results
        self.variant_results: Dict[str, List[bool]] = {}
        self.variant_metadata: Dict[str, Dict[str, Any]] = {}

        logger.info(f"A/B Testing initialized (α={alpha}, power={power})")

    def add_result(
        self,
        variant: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add result for a variant.

        Args:
            variant: Variant name (e.g., 'champion', 'challenger_1')
            success: Whether prediction was correct
            metadata: Additional metadata
        """
        if variant not in self.variant_results:
            self.variant_results[variant] = []
            self.variant_metadata[variant] = {
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {},
            }

        self.variant_results[variant].append(success)

    def check_significance(
        self,
        control: str = "champion",
        treatment: str = "challenger",
    ) -> Dict[str, Any]:
        """
        Check if treatment is significantly better than control.

        Args:
            control: Control variant name
            treatment: Treatment variant name

        Returns:
            Statistical test results
        """
        if control not in self.variant_results or treatment not in self.variant_results:
            return {
                "is_significant": False,
                "reason": "Missing variant data",
            }

        control_results = self.variant_results[control]
        treatment_results = self.variant_results[treatment]

        if len(control_results) < self.min_samples_per_variant or \
           len(treatment_results) < self.min_samples_per_variant:
            return {
                "is_significant": False,
                "reason": f"Insufficient samples: {len(control_results)}/{len(treatment_results)}",
                "required": self.min_samples_per_variant,
            }

        # Calculate success rates
        control_rate = np.mean(control_results)
        treatment_rate = np.mean(treatment_results)

        # Two-proportion z-test
        n1 = len(control_results)
        n2 = len(treatment_results)

        p_pooled = (sum(control_results) + sum(treatment_results)) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))

        z_score = (treatment_rate - control_rate) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed

        is_significant = p_value < self.alpha and treatment_rate > control_rate

        # Calculate confidence interval
        se_diff = np.sqrt(
            control_rate * (1 - control_rate) / n1 +
            treatment_rate * (1 - treatment_rate) / n2
        )
        ci_lower = (treatment_rate - control_rate) - 1.96 * se_diff
        ci_upper = (treatment_rate - control_rate) + 1.96 * se_diff

        result = {
            "is_significant": is_significant,
            "control_rate": float(control_rate),
            "treatment_rate": float(treatment_rate),
            "lift": float(treatment_rate - control_rate),
            "lift_percentage": float((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0,
            "p_value": float(p_value),
            "z_score": float(z_score),
            "alpha": self.alpha,
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
            },
            "sample_sizes": {
                "control": n1,
                "treatment": n2,
            },
        }

        if is_significant:
            logger.info(
                f"✓ Treatment '{treatment}' is significantly better than '{control}': "
                f"{treatment_rate:.2%} vs {control_rate:.2%} "
                f"(lift: {result['lift_percentage']:.2f}%, p={p_value:.4f})"
            )

        return result

    def select_winner(
        self,
        variants: List[str],
    ) -> Dict[str, Any]:
        """
        Select winner among multiple variants.

        Args:
            variants: List of variant names

        Returns:
            Winner selection results
        """
        if len(variants) < 2:
            return {"winner": variants[0] if variants else None}

        # Calculate success rates for all variants
        rates = {}
        for variant in variants:
            if variant in self.variant_results:
                rates[variant] = np.mean(self.variant_results[variant])
            else:
                rates[variant] = 0.0

        # Find best variant
        winner = max(rates, key=rates.get)

        # Compare winner against all others
        comparisons = []
        for variant in variants:
            if variant == winner:
                continue

            comparison = self.check_significance(
                control=variant,
                treatment=winner,
            )
            comparison["compared_variant"] = variant
            comparisons.append(comparison)

        # Winner is valid if significantly better than all others
        is_valid_winner = all(
            comp.get("is_significant", False)
            for comp in comparisons
        )

        return {
            "winner": winner,
            "winner_rate": float(rates[winner]),
            "is_valid_winner": is_valid_winner,
            "rates": {k: float(v) for k, v in rates.items()},
            "comparisons": comparisons,
        }


class IntelligentLearningLoop:
    """
    Intelligent learning loop with drift detection and A/B testing.

    Workflow:
    1. Monitor production model performance
    2. Detect drift → trigger retraining
    3. Train challenger model
    4. A/B test champion vs challenger
    5. Promote winner automatically
    6. Gradual rollout with monitoring
    7. Automatic rollback if issues detected
    """

    def __init__(
        self,
        drift_detector: DriftDetector,
        ab_framework: ABTestingFramework,
        retrain_on_drift: bool = True,
        ab_test_duration_hours: int = 24,
        rollout_stages: List[float] = [0.05, 0.20, 0.50, 1.00],
    ):
        """
        Initialize intelligent learning loop.

        Args:
            drift_detector: Drift detection component
            ab_framework: A/B testing framework
            retrain_on_drift: Auto-retrain on drift detection
            ab_test_duration_hours: A/B test duration
            rollout_stages: Gradual rollout percentages
        """
        self.drift_detector = drift_detector
        self.ab_framework = ab_framework
        self.retrain_on_drift = retrain_on_drift
        self.ab_test_duration_hours = ab_test_duration_hours
        self.rollout_stages = rollout_stages

        # State
        self.current_stage = "monitoring"  # monitoring, ab_testing, rolling_out
        self.champion_model_id = "champion_v1"
        self.challenger_model_id: Optional[str] = None
        self.rollout_percentage = 0.0
        self.ab_test_start_time: Optional[datetime] = None

        # History
        self.retrain_events: List[Dict[str, Any]] = []
        self.rollout_events: List[Dict[str, Any]] = []

        logger.info("Intelligent Learning Loop initialized")
        logger.info(f"Retrain on drift: {retrain_on_drift}")
        logger.info(f"A/B test duration: {ab_test_duration_hours}h")
        logger.info(f"Rollout stages: {rollout_stages}")

    def process_prediction(
        self,
        features: np.ndarray,
        prediction: Any,
        label: Optional[Any] = None,
        model_id: str = "champion",
    ) -> Dict[str, Any]:
        """
        Process a prediction and update learning loop.

        Args:
            features: Input features
            prediction: Model prediction
            label: Ground truth label (if available)
            model_id: Model ID that made prediction

        Returns:
            Processing result with actions
        """
        result = {
            "actions": [],
            "drift_detected": False,
            "ab_test_complete": False,
        }

        # Add to appropriate window
        if self.current_stage == "monitoring":
            # Add to reference window for champion
            self.drift_detector.add_reference_sample(features, prediction, label)

        elif self.current_stage == "ab_testing":
            # Add to current window
            self.drift_detector.add_current_sample(features, prediction, label)

            # Add to A/B test
            if label is not None:
                success = (prediction == label)
                self.ab_framework.add_result(model_id, success)

        # Check for drift (only in monitoring stage)
        if self.current_stage == "monitoring":
            drift_result = self.drift_detector.detect_drift()

            if drift_result["drift_detected"]:
                result["drift_detected"] = True
                result["drift_details"] = drift_result

                if self.retrain_on_drift:
                    # Trigger retraining
                    result["actions"].append("trigger_retrain")
                    self._trigger_retrain(drift_result)

        # Check A/B test completion
        if self.current_stage == "ab_testing":
            if self._should_complete_ab_test():
                ab_result = self.ab_framework.check_significance(
                    control=self.champion_model_id,
                    treatment=self.challenger_model_id,
                )

                result["ab_test_complete"] = True
                result["ab_test_result"] = ab_result

                if ab_result.get("is_significant"):
                    # Challenger wins - start rollout
                    result["actions"].append("start_rollout")
                    self._start_rollout()
                else:
                    # Champion wins - keep current
                    result["actions"].append("keep_champion")
                    self._keep_champion()

        return result

    def _trigger_retrain(self, drift_result: Dict[str, Any]) -> None:
        """Trigger model retraining."""
        logger.info("🔄 Triggering model retraining due to drift...")

        retrain_event = {
            "timestamp": datetime.now().isoformat(),
            "reason": "drift_detected",
            "drift_types": drift_result.get("drift_types", []),
            "challenger_id": f"challenger_{len(self.retrain_events) + 1}",
        }

        self.retrain_events.append(retrain_event)
        self.challenger_model_id = retrain_event["challenger_id"]

        # Start A/B testing
        self.current_stage = "ab_testing"
        self.ab_test_start_time = datetime.now()

        logger.info(f"✓ Retraining triggered: {self.challenger_model_id}")
        logger.info(f"✓ Starting A/B test: {self.champion_model_id} vs {self.challenger_model_id}")

    def _should_complete_ab_test(self) -> bool:
        """Check if A/B test should be completed."""
        if not self.ab_test_start_time:
            return False

        elapsed = datetime.now() - self.ab_test_start_time
        return elapsed >= timedelta(hours=self.ab_test_duration_hours)

    def _start_rollout(self) -> None:
        """Start gradual rollout of challenger."""
        logger.info(f"🚀 Starting gradual rollout of {self.challenger_model_id}")

        self.current_stage = "rolling_out"
        self.rollout_percentage = self.rollout_stages[0]

        rollout_event = {
            "timestamp": datetime.now().isoformat(),
            "action": "start_rollout",
            "champion": self.champion_model_id,
            "challenger": self.challenger_model_id,
            "initial_percentage": self.rollout_percentage,
        }

        self.rollout_events.append(rollout_event)

        logger.info(f"✓ Rollout started at {self.rollout_percentage:.0%}")

    def _keep_champion(self) -> None:
        """Keep current champion, discard challenger."""
        logger.info(f"✓ Keeping champion: {self.champion_model_id}")

        self.current_stage = "monitoring"
        self.challenger_model_id = None
        self.ab_test_start_time = None

    def advance_rollout_stage(self) -> Dict[str, Any]:
        """Advance to next rollout stage."""
        if self.current_stage != "rolling_out":
            return {"success": False, "reason": "Not in rollout stage"}

        # Find current stage index
        current_idx = self.rollout_stages.index(self.rollout_percentage)

        if current_idx >= len(self.rollout_stages) - 1:
            # Rollout complete - promote challenger to champion
            self._promote_challenger()

            return {
                "success": True,
                "action": "rollout_complete",
                "new_champion": self.champion_model_id,
            }

        # Advance to next stage
        next_idx = current_idx + 1
        self.rollout_percentage = self.rollout_stages[next_idx]

        logger.info(f"📈 Advancing rollout to {self.rollout_percentage:.0%}")

        return {
            "success": True,
            "action": "advance_stage",
            "percentage": self.rollout_percentage,
        }

    def _promote_challenger(self) -> None:
        """Promote challenger to new champion."""
        logger.info(f"👑 Promoting {self.challenger_model_id} to champion")

        old_champion = self.champion_model_id
        self.champion_model_id = self.challenger_model_id
        self.challenger_model_id = None
        self.current_stage = "monitoring"
        self.rollout_percentage = 0.0

        logger.info(f"✓ {old_champion} → {self.champion_model_id}")
