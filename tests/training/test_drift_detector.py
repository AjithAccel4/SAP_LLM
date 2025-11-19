"""
Unit tests for Drift Detector.

Tests cover:
- PSI calculation
- Feature drift detection
- Concept drift detection
- Performance monitoring
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from sap_llm.training.drift_detector import DriftDetector, PerformanceMonitor


@pytest.fixture
def drift_detector():
    """Create drift detector for tests."""
    return DriftDetector(
        psi_threshold=0.25,
        feature_drift_threshold=0.1,
        concept_drift_threshold=0.05
    )


@pytest.fixture
def performance_monitor():
    """Create performance monitor for tests."""
    return PerformanceMonitor()


class TestDriftDetector:
    """Test DriftDetector class."""

    def test_initialization(self, drift_detector):
        """Test detector initialization."""
        assert drift_detector.psi_threshold == 0.25
        assert drift_detector.feature_drift_threshold == 0.1
        assert drift_detector.concept_drift_threshold == 0.05

    def test_categorical_psi_no_drift(self, drift_detector):
        """Test PSI calculation with no drift (categorical)."""
        # Same distribution
        baseline = [
            {"prediction": {"doc_type": "invoice"}} for _ in range(100)
        ]
        current = [
            {"prediction": {"doc_type": "invoice"}} for _ in range(100)
        ]

        report = drift_detector.detect_data_drift(baseline, current)

        assert report.psi_score < 0.1  # Low PSI
        assert not report.drift_detected or "data_drift" not in report.drift_types

    def test_categorical_psi_with_drift(self, drift_detector):
        """Test PSI calculation with drift (categorical)."""
        # Different distributions
        baseline = [
            {"prediction": {"doc_type": "invoice"}} for _ in range(100)
        ]
        current = [
            {"prediction": {"doc_type": "purchase_order"}} for _ in range(100)
        ]

        report = drift_detector.detect_data_drift(baseline, current)

        assert report.psi_score > 0.25  # High PSI
        assert report.drift_detected
        assert "data_drift" in report.drift_types

    def test_numerical_psi(self, drift_detector):
        """Test PSI calculation with numerical values."""
        # Create baseline with normal distribution
        np.random.seed(42)
        baseline = [
            {"prediction": {"confidence": float(x)}}
            for x in np.random.normal(0.8, 0.1, 200)
        ]

        # Current with shifted distribution
        current = [
            {"prediction": {"confidence": float(x)}}
            for x in np.random.normal(0.6, 0.1, 200)
        ]

        report = drift_detector.detect_data_drift(baseline, current)

        # Should detect drift due to shift
        assert report.psi_score > 0

    def test_concept_drift_detection(self, drift_detector):
        """Test concept drift (accuracy degradation)."""
        # Baseline with high accuracy
        baseline = [
            {
                "prediction": {"doc_type": "invoice"},
                "human_correction": {"doc_type": "invoice"}
            }
            for _ in range(100)
        ]

        # Current with low accuracy
        current = [
            {
                "prediction": {"doc_type": "invoice"},
                "human_correction": {"doc_type": "purchase_order"}
            }
            for _ in range(100)
        ]

        report = drift_detector.detect_data_drift(baseline, current)

        assert report.concept_drift > 0.05  # Significant degradation
        assert "concept_drift" in report.drift_types

    def test_mixed_drift_detection(self, drift_detector):
        """Test detection of multiple drift types."""
        # Create data with both data and concept drift
        baseline = [
            {
                "prediction": {"doc_type": "invoice"},
                "human_correction": {"doc_type": "invoice"},
                "document": {"amount": 100.0}
            }
            for _ in range(100)
        ]

        current = [
            {
                "prediction": {"doc_type": "purchase_order"},
                "human_correction": {"doc_type": "invoice"},  # Wrong prediction
                "document": {"amount": 1000.0}  # Different distribution
            }
            for _ in range(100)
        ]

        report = drift_detector.detect_data_drift(baseline, current)

        # Should detect multiple drift types
        assert report.drift_detected
        assert len(report.drift_types) > 0

    def test_severity_levels(self, drift_detector):
        """Test drift severity classification."""
        # Test high severity
        baseline_high = [
            {"prediction": {"doc_type": "invoice"}} for _ in range(100)
        ]
        current_high = [
            {"prediction": {"doc_type": "purchase_order"}} for _ in range(100)
        ]

        report_high = drift_detector.detect_data_drift(baseline_high, current_high)
        assert report_high.severity in ["high", "medium"]

        # Test low severity (minimal drift)
        baseline_low = [
            {"prediction": {"doc_type": "invoice"}} for _ in range(100)
        ]
        current_low = [
            {"prediction": {"doc_type": "invoice"}} for _ in range(98)
        ] + [
            {"prediction": {"doc_type": "purchase_order"}} for _ in range(2)
        ]

        report_low = drift_detector.detect_data_drift(baseline_low, current_low)
        if report_low.drift_detected:
            assert report_low.severity in ["low", "medium"]


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""

    def test_initialization(self, performance_monitor):
        """Test monitor initialization."""
        assert performance_monitor is not None

    def test_accuracy_calculation(self, performance_monitor):
        """Test accuracy calculation."""
        predictions = [
            {
                "prediction": {"doc_type": "invoice"},
                "ground_truth": {"doc_type": "invoice"}
            }
            for _ in range(90)
        ] + [
            {
                "prediction": {"doc_type": "invoice"},
                "ground_truth": {"doc_type": "purchase_order"}
            }
            for _ in range(10)
        ]

        report = performance_monitor.monitor_model_performance(
            predictions=predictions,
            model_id="test_model"
        )

        # Should have 90% accuracy
        assert 0.89 <= report["metrics"]["accuracy"] <= 0.91

    def test_degradation_detection(self, performance_monitor):
        """Test performance degradation detection."""
        predictions = [
            {
                "prediction": {"doc_type": "invoice"},
                "ground_truth": {"doc_type": "invoice"}
            }
            for _ in range(50)
        ] + [
            {
                "prediction": {"doc_type": "invoice"},
                "ground_truth": {"doc_type": "purchase_order"}
            }
            for _ in range(50)
        ]

        baseline_metrics = {"accuracy": 0.95}

        report = performance_monitor.monitor_model_performance(
            predictions=predictions,
            model_id="test_model",
            baseline_metrics=baseline_metrics
        )

        # Should detect degradation
        assert report["degradation"] > 0.05
        assert report["needs_retraining"]

    def test_latency_monitoring(self, performance_monitor):
        """Test latency calculation."""
        predictions = [
            {
                "prediction": {"doc_type": "invoice"},
                "ground_truth": {"doc_type": "invoice"},
                "latency_ms": 100 + i
            }
            for i in range(100)
        ]

        report = performance_monitor.monitor_model_performance(
            predictions=predictions,
            model_id="test_model"
        )

        # Should calculate latency percentile
        assert report["metrics"]["latency_p95"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
