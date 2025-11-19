"""
Drift Detection for Continuous Learning.

Implements multiple drift detection methods:
- Population Stability Index (PSI) for data drift
- Feature drift detection
- Concept drift detection (accuracy degradation)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Drift detection report."""
    psi_score: float
    feature_drift: Dict[str, float]
    concept_drift: float
    drift_detected: bool
    severity: str  # "low", "medium", "high"
    drift_types: List[str]
    timestamp: str
    baseline_window: int
    current_window: int
    details: Dict[str, Any]


class DriftDetector:
    """
    Comprehensive drift detection for model monitoring.

    Detects:
    1. Data Drift: Distribution changes in features (PSI)
    2. Feature Drift: Individual feature distribution shifts
    3. Concept Drift: Accuracy degradation over time
    """

    def __init__(
        self,
        psi_threshold: float = 0.25,
        feature_drift_threshold: float = 0.1,
        concept_drift_threshold: float = 0.05
    ):
        """
        Initialize drift detector.

        Args:
            psi_threshold: PSI threshold for data drift (typical: 0.1-0.25)
            feature_drift_threshold: Threshold for feature drift (KS statistic)
            concept_drift_threshold: Threshold for concept drift (accuracy drop)
        """
        self.psi_threshold = psi_threshold
        self.feature_drift_threshold = feature_drift_threshold
        self.concept_drift_threshold = concept_drift_threshold

        logger.info(
            f"DriftDetector initialized "
            f"(psi_threshold={psi_threshold}, "
            f"feature_threshold={feature_drift_threshold}, "
            f"concept_threshold={concept_drift_threshold})"
        )

    def detect_data_drift(
        self,
        baseline_data: List[Dict[str, Any]],
        current_data: List[Dict[str, Any]]
    ) -> DriftReport:
        """
        Detect data drift using PSI and feature drift analysis.

        Args:
            baseline_data: Baseline/training distribution
            current_data: Current production distribution

        Returns:
            DriftReport with drift analysis
        """
        logger.info(
            f"Detecting drift (baseline={len(baseline_data)}, current={len(current_data)})"
        )

        # Calculate PSI
        psi_score = self._calculate_psi(baseline_data, current_data)

        # Calculate feature drift
        feature_drift = self._calculate_feature_drift(baseline_data, current_data)

        # Calculate concept drift if labels available
        concept_drift = self._calculate_concept_drift(baseline_data, current_data)

        # Determine drift types
        drift_types = []
        if psi_score > self.psi_threshold:
            drift_types.append("data_drift")

        if any(score > self.feature_drift_threshold for score in feature_drift.values()):
            drift_types.append("feature_drift")

        if concept_drift > self.concept_drift_threshold:
            drift_types.append("concept_drift")

        drift_detected = len(drift_types) > 0

        # Determine severity
        if psi_score > 0.4 or concept_drift > 0.10:
            severity = "high"
        elif psi_score > 0.25 or concept_drift > 0.05:
            severity = "medium"
        else:
            severity = "low"

        report = DriftReport(
            psi_score=psi_score,
            feature_drift=feature_drift,
            concept_drift=concept_drift,
            drift_detected=drift_detected,
            severity=severity,
            drift_types=drift_types,
            timestamp=datetime.now().isoformat(),
            baseline_window=len(baseline_data),
            current_window=len(current_data),
            details={
                "psi_threshold": self.psi_threshold,
                "feature_drift_threshold": self.feature_drift_threshold,
                "concept_drift_threshold": self.concept_drift_threshold
            }
        )

        if drift_detected:
            logger.warning(
                f"Drift detected! PSI={psi_score:.4f}, "
                f"Concept={concept_drift:.4f}, "
                f"Types={drift_types}, "
                f"Severity={severity}"
            )
        else:
            logger.info(f"No drift detected (PSI={psi_score:.4f})")

        return report

    def _calculate_psi(
        self,
        baseline_data: List[Dict[str, Any]],
        current_data: List[Dict[str, Any]],
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures distribution shift:
        PSI = Σ (actual% - expected%) × ln(actual% / expected%)

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.25: Moderate change, investigate
        - PSI ≥ 0.25: Significant change, retrain needed

        Args:
            baseline_data: Baseline distribution
            current_data: Current distribution
            bins: Number of bins for discretization

        Returns:
            PSI score
        """
        # Extract predictions for PSI calculation
        baseline_values = self._extract_prediction_values(baseline_data)
        current_values = self._extract_prediction_values(current_data)

        if not baseline_values or not current_values:
            logger.warning("Insufficient data for PSI calculation")
            return 0.0

        # For categorical predictions
        if isinstance(baseline_values[0], str):
            return self._calculate_categorical_psi(baseline_values, current_values)

        # For numerical predictions
        return self._calculate_numerical_psi(baseline_values, current_values, bins)

    def _calculate_categorical_psi(
        self,
        baseline_values: List[str],
        current_values: List[str]
    ) -> float:
        """Calculate PSI for categorical distributions."""
        # Get all categories
        all_categories = set(baseline_values + current_values)

        # Calculate proportions
        baseline_total = len(baseline_values)
        current_total = len(current_values)

        psi = 0.0
        for category in all_categories:
            # Expected (baseline) proportion
            baseline_count = baseline_values.count(category)
            expected_pct = (baseline_count + 1) / (baseline_total + len(all_categories))  # Laplace smoothing

            # Actual (current) proportion
            current_count = current_values.count(category)
            actual_pct = (current_count + 1) / (current_total + len(all_categories))  # Laplace smoothing

            # PSI contribution
            psi += (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

        return psi

    def _calculate_numerical_psi(
        self,
        baseline_values: List[float],
        current_values: List[float],
        bins: int = 10
    ) -> float:
        """Calculate PSI for numerical distributions."""
        baseline_array = np.array(baseline_values)
        current_array = np.array(current_values)

        # Create bins based on baseline distribution
        _, bin_edges = np.histogram(baseline_array, bins=bins)

        # Calculate baseline distribution
        baseline_hist, _ = np.histogram(baseline_array, bins=bin_edges)
        baseline_pct = baseline_hist / len(baseline_array)

        # Calculate current distribution
        current_hist, _ = np.histogram(current_array, bins=bin_edges)
        current_pct = current_hist / len(current_array)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-5
        baseline_pct = baseline_pct + epsilon
        current_pct = current_pct + epsilon

        # Calculate PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return psi

    def _calculate_feature_drift(
        self,
        baseline_data: List[Dict[str, Any]],
        current_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate feature drift using Kolmogorov-Smirnov test.

        Args:
            baseline_data: Baseline data
            current_data: Current data

        Returns:
            Dictionary of feature names to KS statistics
        """
        feature_drift = {}

        # Extract features
        baseline_features = self._extract_features(baseline_data)
        current_features = self._extract_features(current_data)

        # Calculate KS statistic for each feature
        for feature_name in baseline_features.keys():
            if feature_name not in current_features:
                continue

            baseline_values = baseline_features[feature_name]
            current_values = current_features[feature_name]

            # Skip non-numerical features
            if not all(isinstance(v, (int, float)) for v in baseline_values[:10]):
                continue

            try:
                # Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.ks_2samp(baseline_values, current_values)
                feature_drift[feature_name] = ks_statistic
            except Exception as e:
                logger.warning(f"Failed to calculate drift for feature {feature_name}: {e}")
                continue

        return feature_drift

    def _calculate_concept_drift(
        self,
        baseline_data: List[Dict[str, Any]],
        current_data: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate concept drift (accuracy degradation).

        Args:
            baseline_data: Baseline data with ground truth
            current_data: Current data with ground truth

        Returns:
            Accuracy drop (0.0 to 1.0)
        """
        # Calculate accuracy for baseline
        baseline_accuracy = self._calculate_accuracy(baseline_data)

        # Calculate accuracy for current
        current_accuracy = self._calculate_accuracy(current_data)

        # Concept drift is the drop in accuracy
        concept_drift = max(0.0, baseline_accuracy - current_accuracy)

        return concept_drift

    def _calculate_accuracy(self, data: List[Dict[str, Any]]) -> float:
        """Calculate accuracy from prediction data."""
        if not data:
            return 0.0

        correct = 0
        total = 0

        for item in data:
            prediction = item.get("prediction", {})
            ground_truth = item.get("human_correction") or item.get("ground_truth")

            if not ground_truth:
                continue

            # Compare predictions
            pred_value = self._extract_single_prediction(prediction)
            truth_value = self._extract_single_prediction(ground_truth)

            if pred_value == truth_value:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def _extract_prediction_values(self, data: List[Dict[str, Any]]) -> List[Any]:
        """Extract prediction values from data."""
        values = []
        for item in data:
            prediction = item.get("prediction", {})
            value = self._extract_single_prediction(prediction)
            if value is not None:
                values.append(value)
        return values

    def _extract_single_prediction(self, prediction: Dict[str, Any]) -> Any:
        """Extract single prediction value."""
        # Try common field names
        for field in ["doc_type", "class", "label", "category", "prediction", "value"]:
            if field in prediction:
                return prediction[field]

        # If prediction has a single key, return its value
        if len(prediction) == 1:
            return list(prediction.values())[0]

        return None

    def _extract_features(self, data: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Extract features from data."""
        features = {}

        for item in data:
            # Extract from document or features field
            doc = item.get("document", {})

            for key, value in doc.items():
                if key not in features:
                    features[key] = []
                features[key].append(value)

        return features


class PerformanceMonitor:
    """
    Monitor model performance in production.

    Tracks:
    - Accuracy
    - F1 scores per field
    - Latency (p50, p95, p99)
    - Error rates
    """

    def __init__(self):
        """Initialize performance monitor."""
        self.baseline_metrics = {}
        logger.info("PerformanceMonitor initialized")

    def monitor_model_performance(
        self,
        predictions: List[Dict[str, Any]],
        model_id: str,
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Monitor model performance.

        Args:
            predictions: List of predictions with ground truth
            model_id: Model identifier
            baseline_metrics: Baseline metrics for comparison

        Returns:
            Performance report
        """
        logger.info(f"Monitoring performance for {model_id} ({len(predictions)} samples)")

        # Calculate metrics
        accuracy = self._calculate_accuracy(predictions)
        f1_scores = self._calculate_f1_per_field(predictions)
        latency_p95 = self._calculate_latency_percentile(predictions, 95)
        error_rate = self._calculate_error_rate(predictions)

        current_metrics = {
            "accuracy": accuracy,
            "f1_scores": f1_scores,
            "latency_p95": latency_p95,
            "error_rate": error_rate
        }

        # Calculate degradation if baseline provided
        degradation = 0.0
        if baseline_metrics:
            baseline_acc = baseline_metrics.get("accuracy", accuracy)
            degradation = max(0.0, baseline_acc - accuracy)

        needs_retraining = degradation > 0.05  # >5% degradation

        report = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(predictions),
            "metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
            "degradation": degradation,
            "needs_retraining": needs_retraining
        }

        if needs_retraining:
            logger.warning(
                f"Performance degradation detected: {degradation:.2%}. "
                f"Retraining recommended."
            )

        return report

    def _calculate_accuracy(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate overall accuracy."""
        if not predictions:
            return 0.0

        correct = 0
        total = 0

        for pred in predictions:
            prediction = pred.get("prediction", {})
            ground_truth = pred.get("ground_truth") or pred.get("human_correction")

            if not ground_truth:
                continue

            if prediction == ground_truth:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def _calculate_f1_per_field(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate F1 score per field."""
        # Placeholder implementation
        # In production: Calculate true F1 for each extraction field
        return {
            "doc_type": 0.95,
            "total_amount": 0.92,
            "supplier": 0.88
        }

    def _calculate_latency_percentile(
        self,
        predictions: List[Dict[str, Any]],
        percentile: int
    ) -> float:
        """Calculate latency percentile."""
        latencies = [p.get("latency_ms", 100) for p in predictions]
        if not latencies:
            return 0.0

        return float(np.percentile(latencies, percentile))

    def _calculate_error_rate(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate error rate."""
        if not predictions:
            return 0.0

        errors = sum(1 for p in predictions if p.get("error") or p.get("exception"))
        return errors / len(predictions)
