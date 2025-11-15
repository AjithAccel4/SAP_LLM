"""
Adaptive Learning Engine

Real-time performance monitoring, concept drift detection, data distribution
shift monitoring, automatic model refresh, and per-customer learning.
"""

import hashlib
import json
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """Real-time performance metrics."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def add(
        self,
        prediction: Any,
        actual: Any,
        confidence: float,
        timestamp: Optional[datetime] = None,
    ):
        """Add prediction result."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.confidences.append(confidence)
        self.timestamps.append(timestamp or datetime.now())

    def get_accuracy(self) -> float:
        """Calculate accuracy over window."""
        if not self.predictions or not self.actuals:
            return 0.0
        try:
            return accuracy_score(list(self.actuals), list(self.predictions))
        except:
            return 0.0

    def get_f1_score(self, average: str = 'weighted') -> float:
        """Calculate F1 score."""
        if not self.predictions or not self.actuals:
            return 0.0
        try:
            return f1_score(
                list(self.actuals),
                list(self.predictions),
                average=average,
                zero_division=0,
            )
        except:
            return 0.0

    def get_precision(self, average: str = 'weighted') -> float:
        """Calculate precision."""
        if not self.predictions or not self.actuals:
            return 0.0
        try:
            return precision_score(
                list(self.actuals),
                list(self.predictions),
                average=average,
                zero_division=0,
            )
        except:
            return 0.0

    def get_recall(self, average: str = 'weighted') -> float:
        """Calculate recall."""
        if not self.predictions or not self.actuals:
            return 0.0
        try:
            return recall_score(
                list(self.actuals),
                list(self.predictions),
                average=average,
                zero_division=0,
            )
        except:
            return 0.0

    def get_avg_confidence(self) -> float:
        """Calculate average confidence."""
        if not self.confidences:
            return 0.0
        return float(np.mean(list(self.confidences)))

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics."""
        return {
            "accuracy": self.get_accuracy(),
            "f1_score": self.get_f1_score(),
            "precision": self.get_precision(),
            "recall": self.get_recall(),
            "avg_confidence": self.get_avg_confidence(),
            "sample_count": len(self.predictions),
        }


class DriftDetector:
    """Concept drift and data distribution shift detector."""

    def __init__(
        self,
        window_size: int = 500,
        reference_window_size: int = 2000,
        drift_threshold: float = 0.25,
    ):
        self.window_size = window_size
        self.reference_window_size = reference_window_size
        self.drift_threshold = drift_threshold

        # Reference distribution (baseline)
        self.reference_distributions: Dict[str, np.ndarray] = {}

        # Current window
        self.current_windows: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def add_sample(self, feature_name: str, value: float):
        """Add sample to current window."""
        self.current_windows[feature_name].append(value)

    def set_reference(self, feature_name: str, values: List[float]):
        """Set reference distribution."""
        self.reference_distributions[feature_name] = np.array(values)
        logger.info(f"Set reference distribution for {feature_name}: {len(values)} samples")

    def detect_drift(
        self,
        feature_name: str,
        method: str = 'ks',
    ) -> Tuple[bool, float]:
        """
        Detect drift in feature distribution.

        Args:
            feature_name: Feature to check
            method: Detection method ('ks', 'psi', 'chi2')

        Returns:
            (drift_detected, drift_score)
        """
        if feature_name not in self.reference_distributions:
            logger.warning(f"No reference distribution for {feature_name}")
            return False, 0.0

        if feature_name not in self.current_windows:
            return False, 0.0

        current = np.array(list(self.current_windows[feature_name]))
        reference = self.reference_distributions[feature_name]

        if len(current) < self.window_size // 2:
            # Not enough samples
            return False, 0.0

        # Detect drift using specified method
        if method == 'ks':
            drift_score = self._kolmogorov_smirnov_test(reference, current)
        elif method == 'psi':
            drift_score = self._population_stability_index(reference, current)
        elif method == 'chi2':
            drift_score = self._chi_square_test(reference, current)
        else:
            raise ValueError(f"Unknown drift detection method: {method}")

        drift_detected = drift_score > self.drift_threshold

        if drift_detected:
            logger.warning(
                f"Drift detected for {feature_name}: "
                f"{method}={drift_score:.4f} (threshold={self.drift_threshold})"
            )

        return drift_detected, drift_score

    def detect_all_drifts(
        self,
        method: str = 'ks',
    ) -> Dict[str, Tuple[bool, float]]:
        """Detect drift for all features."""
        results = {}
        for feature_name in self.reference_distributions.keys():
            drift_detected, score = self.detect_drift(feature_name, method)
            results[feature_name] = (drift_detected, score)
        return results

    def _kolmogorov_smirnov_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Kolmogorov-Smirnov test for distribution difference."""
        try:
            statistic, p_value = stats.ks_2samp(reference, current)
            # Return statistic (higher = more different)
            return float(statistic)
        except:
            return 0.0

    def _population_stability_index(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        num_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change (drift)
        """
        try:
            # Create bins based on reference distribution
            _, bins = np.histogram(reference, bins=num_bins)

            # Calculate distributions
            ref_hist, _ = np.histogram(reference, bins=bins)
            curr_hist, _ = np.histogram(current, bins=bins)

            # Convert to proportions
            ref_prop = ref_hist / len(reference)
            curr_prop = curr_hist / len(current)

            # Avoid log(0)
            ref_prop = np.where(ref_prop == 0, 0.0001, ref_prop)
            curr_prop = np.where(curr_prop == 0, 0.0001, curr_prop)

            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))

            return float(psi)

        except Exception as e:
            logger.error(f"PSI calculation failed: {e}")
            return 0.0

    def _chi_square_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        num_bins: int = 10,
    ) -> float:
        """Chi-square test for distribution difference."""
        try:
            # Create bins
            _, bins = np.histogram(reference, bins=num_bins)

            # Calculate histograms
            ref_hist, _ = np.histogram(reference, bins=bins)
            curr_hist, _ = np.histogram(current, bins=bins)

            # Avoid zeros
            ref_hist = np.where(ref_hist == 0, 1, ref_hist)

            # Chi-square statistic
            chi2 = np.sum((curr_hist - ref_hist) ** 2 / ref_hist)

            # Normalize by sample size
            chi2_norm = chi2 / len(current)

            return float(chi2_norm)

        except Exception as e:
            logger.error(f"Chi-square test failed: {e}")
            return 0.0


class TenantSpecificLearning:
    """Per-customer (tenant) learning and adaptation."""

    def __init__(self, global_model_path: str = "/tmp/sap_llm/global_model"):
        self.global_model_path = global_model_path
        self.tenant_models: Dict[str, Any] = {}
        self.tenant_data: Dict[str, List[Dict]] = defaultdict(list)
        self.tenant_performance: Dict[str, PerformanceMetrics] = {}

    def create_tenant_model(
        self,
        tenant_id: str,
        doc_type: str,
        use_transfer_learning: bool = True,
    ):
        """Create tenant-specific model."""
        model_key = f"{tenant_id}:{doc_type}"

        if use_transfer_learning:
            # Start from global model
            logger.info(f"Creating tenant model for {model_key} (transfer learning)")
            # In production, load global model and fine-tune
        else:
            logger.info(f"Creating tenant model for {model_key} (from scratch)")

        self.tenant_models[model_key] = {
            "tenant_id": tenant_id,
            "doc_type": doc_type,
            "created_at": datetime.now().isoformat(),
            "transfer_learning": use_transfer_learning,
        }

        self.tenant_performance[model_key] = PerformanceMetrics()

    def add_tenant_data(
        self,
        tenant_id: str,
        doc_type: str,
        features: Dict[str, Any],
        label: str,
    ):
        """Add tenant-specific training data."""
        model_key = f"{tenant_id}:{doc_type}"
        self.tenant_data[model_key].append({
            "features": features,
            "label": label,
            "timestamp": datetime.now().isoformat(),
        })

    def get_tenant_model(
        self,
        tenant_id: str,
        doc_type: str,
    ) -> Optional[Any]:
        """Get tenant-specific model if available."""
        model_key = f"{tenant_id}:{doc_type}"
        return self.tenant_models.get(model_key)

    def should_use_tenant_model(
        self,
        tenant_id: str,
        doc_type: str,
        min_samples: int = 100,
        min_accuracy: float = 0.8,
    ) -> bool:
        """Determine if tenant-specific model should be used."""
        model_key = f"{tenant_id}:{doc_type}"

        # Check if tenant model exists
        if model_key not in self.tenant_models:
            return False

        # Check if enough data
        if len(self.tenant_data[model_key]) < min_samples:
            return False

        # Check if performance is good
        if model_key in self.tenant_performance:
            accuracy = self.tenant_performance[model_key].get_accuracy()
            if accuracy < min_accuracy:
                return False

        return True


class AdaptiveLearningEngine:
    """
    Comprehensive adaptive learning engine with real-time monitoring.

    Features:
    - Real-time accuracy tracking per document type
    - Concept drift detection
    - Data distribution shift monitoring
    - Automatic model refresh when performance degrades
    - Per-customer (tenant) learning
    - Performance analytics and alerts
    """

    def __init__(
        self,
        pmg: ProcessMemoryGraph,
        performance_window: int = 1000,
        drift_window: int = 500,
        drift_threshold: float = 0.25,
        refresh_threshold: float = 0.15,
        min_accuracy: float = 0.85,
    ):
        """
        Initialize adaptive learning engine.

        Args:
            pmg: Process Memory Graph instance
            performance_window: Window size for performance metrics
            drift_window: Window size for drift detection
            drift_threshold: Drift detection threshold
            refresh_threshold: Performance drop threshold for refresh
            min_accuracy: Minimum acceptable accuracy
        """
        self.pmg = pmg
        self.performance_window = performance_window
        self.drift_window = drift_window
        self.drift_threshold = drift_threshold
        self.refresh_threshold = refresh_threshold
        self.min_accuracy = min_accuracy

        # Performance tracking per document type
        self.performance_trackers: Dict[str, PerformanceMetrics] = {}

        # Drift detection
        self.drift_detector = DriftDetector(
            window_size=drift_window,
            drift_threshold=drift_threshold,
        )

        # Tenant-specific learning
        self.tenant_learning = TenantSpecificLearning()

        # Baseline performance
        self.baseline_performance: Dict[str, float] = {}

        # Refresh triggers
        self.refresh_triggers: Dict[str, List[Dict]] = defaultdict(list)

        # Alerts
        self.alerts: List[Dict[str, Any]] = []

        logger.info("AdaptiveLearningEngine initialized")

    def track_prediction(
        self,
        doc_type: str,
        prediction: Any,
        actual: Any,
        confidence: float,
        features: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Track prediction for performance monitoring and drift detection.

        Args:
            doc_type: Document type
            prediction: Model prediction
            actual: Actual/ground truth value
            confidence: Prediction confidence
            features: Optional features for drift detection
            tenant_id: Optional tenant ID for per-customer tracking
        """
        # Get or create performance tracker
        if doc_type not in self.performance_trackers:
            self.performance_trackers[doc_type] = PerformanceMetrics(
                window_size=self.performance_window
            )

        # Add to performance tracker
        self.performance_trackers[doc_type].add(
            prediction=prediction,
            actual=actual,
            confidence=confidence,
        )

        # Track for drift detection
        if features:
            for feature_name, value in features.items():
                if isinstance(value, (int, float)):
                    self.drift_detector.add_sample(f"{doc_type}:{feature_name}", float(value))

        # Track tenant-specific if provided
        if tenant_id:
            model_key = f"{tenant_id}:{doc_type}"
            if model_key not in self.tenant_learning.tenant_performance:
                self.tenant_learning.tenant_performance[model_key] = PerformanceMetrics()

            self.tenant_learning.tenant_performance[model_key].add(
                prediction=prediction,
                actual=actual,
                confidence=confidence,
            )

    def check_performance_degradation(
        self,
        doc_type: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if performance has degraded significantly.

        Args:
            doc_type: Document type

        Returns:
            (degraded, metrics)
        """
        if doc_type not in self.performance_trackers:
            return False, {}

        metrics = self.performance_trackers[doc_type].get_all_metrics()
        current_accuracy = metrics["accuracy"]

        # Check against baseline
        if doc_type in self.baseline_performance:
            baseline = self.baseline_performance[doc_type]
            degradation = baseline - current_accuracy

            if degradation > self.refresh_threshold:
                # Performance degraded significantly
                self._trigger_refresh(doc_type, "performance_degradation", {
                    "baseline": baseline,
                    "current": current_accuracy,
                    "degradation": degradation,
                })

                return True, {
                    "degraded": True,
                    "baseline": baseline,
                    "current": current_accuracy,
                    "degradation": degradation,
                    "metrics": metrics,
                }

        # Check against minimum threshold
        if current_accuracy < self.min_accuracy:
            self._trigger_refresh(doc_type, "below_minimum", {
                "min_accuracy": self.min_accuracy,
                "current": current_accuracy,
            })

            return True, {
                "degraded": True,
                "reason": "below_minimum",
                "min_accuracy": self.min_accuracy,
                "current": current_accuracy,
                "metrics": metrics,
            }

        return False, metrics

    def check_drift(
        self,
        doc_type: str,
        features: Optional[List[str]] = None,
        method: str = 'psi',
    ) -> Dict[str, Any]:
        """
        Check for concept drift or distribution shift.

        Args:
            doc_type: Document type
            features: Specific features to check (None = all)
            method: Drift detection method

        Returns:
            Drift detection results
        """
        results = {
            "doc_type": doc_type,
            "drift_detected": False,
            "features_with_drift": [],
            "drift_scores": {},
        }

        # Check drift for all features
        all_drifts = self.drift_detector.detect_all_drifts(method=method)

        for feature_key, (drift_detected, score) in all_drifts.items():
            # Filter by doc_type and feature list
            if not feature_key.startswith(f"{doc_type}:"):
                continue

            feature_name = feature_key.split(":", 1)[1]

            if features and feature_name not in features:
                continue

            results["drift_scores"][feature_name] = score

            if drift_detected:
                results["drift_detected"] = True
                results["features_with_drift"].append(feature_name)

        # Trigger refresh if drift detected
        if results["drift_detected"]:
            self._trigger_refresh(doc_type, "drift_detected", results)

        return results

    def set_baseline_performance(
        self,
        doc_type: str,
        accuracy: Optional[float] = None,
    ):
        """
        Set baseline performance for document type.

        Args:
            doc_type: Document type
            accuracy: Baseline accuracy (or calculate from current)
        """
        if accuracy is not None:
            self.baseline_performance[doc_type] = accuracy
        else:
            # Calculate from current tracker
            if doc_type in self.performance_trackers:
                metrics = self.performance_trackers[doc_type].get_all_metrics()
                self.baseline_performance[doc_type] = metrics["accuracy"]

        logger.info(
            f"Set baseline performance for {doc_type}: "
            f"{self.baseline_performance[doc_type]:.3f}"
        )

    def set_baseline_distribution(
        self,
        doc_type: str,
        feature_name: str,
        values: List[float],
    ):
        """Set baseline distribution for drift detection."""
        self.drift_detector.set_reference(f"{doc_type}:{feature_name}", values)

    def get_performance_summary(
        self,
        doc_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get performance summary for all or specific document type.

        Args:
            doc_type: Optional document type filter

        Returns:
            Performance summary
        """
        summary = {}

        doc_types = [doc_type] if doc_type else self.performance_trackers.keys()

        for dt in doc_types:
            if dt not in self.performance_trackers:
                continue

            metrics = self.performance_trackers[dt].get_all_metrics()

            summary[dt] = {
                "metrics": metrics,
                "baseline": self.baseline_performance.get(dt),
                "degradation": None,
                "needs_refresh": False,
            }

            # Calculate degradation
            if dt in self.baseline_performance:
                baseline = self.baseline_performance[dt]
                current = metrics["accuracy"]
                degradation = baseline - current
                summary[dt]["degradation"] = degradation

                if degradation > self.refresh_threshold:
                    summary[dt]["needs_refresh"] = True

        return summary

    def get_tenant_performance(
        self,
        tenant_id: str,
        doc_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get performance for specific tenant."""
        results = {}

        for model_key, perf_tracker in self.tenant_learning.tenant_performance.items():
            tenant, dt = model_key.split(":", 1)

            if tenant != tenant_id:
                continue

            if doc_type and dt != doc_type:
                continue

            metrics = perf_tracker.get_all_metrics()
            results[dt] = metrics

        return results

    def should_refresh_model(self, doc_type: str) -> bool:
        """Check if model should be refreshed."""
        return len(self.refresh_triggers.get(doc_type, [])) > 0

    def get_refresh_triggers(self, doc_type: str) -> List[Dict[str, Any]]:
        """Get refresh triggers for document type."""
        return self.refresh_triggers.get(doc_type, [])

    def clear_refresh_triggers(self, doc_type: str):
        """Clear refresh triggers after model update."""
        if doc_type in self.refresh_triggers:
            self.refresh_triggers[doc_type].clear()
            logger.info(f"Cleared refresh triggers for {doc_type}")

    def _trigger_refresh(
        self,
        doc_type: str,
        reason: str,
        details: Dict[str, Any],
    ):
        """Trigger model refresh."""
        trigger = {
            "reason": reason,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }

        self.refresh_triggers[doc_type].append(trigger)

        # Create alert
        alert = {
            "alert_type": "model_refresh_needed",
            "doc_type": doc_type,
            "reason": reason,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self.alerts.append(alert)

        logger.warning(
            f"Model refresh triggered for {doc_type}: {reason}"
        )

    def get_alerts(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        alerts = self.alerts

        if since:
            alerts = [
                a for a in alerts
                if datetime.fromisoformat(a["timestamp"]) > since
            ]

        return alerts[-limit:]
