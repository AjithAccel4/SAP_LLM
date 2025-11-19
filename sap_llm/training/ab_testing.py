"""
A/B Testing Framework for Champion/Challenger Model Evaluation.

Provides:
- Traffic splitting between champion and challenger
- Statistical significance testing
- Automated decision making
- Comprehensive metrics collection
"""

import json
import logging
import random
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from sap_llm.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """A/B test evaluation result."""
    test_id: str
    champion_id: str
    challenger_id: str
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    winner: str  # "champion", "challenger", "inconclusive"
    p_value: float
    confidence: float  # 0.0 to 1.0
    sample_count_champion: int
    sample_count_challenger: int
    recommendation: str  # "promote", "keep_champion", "continue_testing"
    details: Dict[str, Any]


class ABTestingManager:
    """
    Manages A/B testing between champion and challenger models.

    Features:
    - Traffic splitting (e.g., 90/10 split)
    - Metrics collection per model
    - Statistical significance testing
    - Automated promotion decisions
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        db_path: str = "./model_registry/ab_tests.db",
        default_traffic_split: float = 0.1
    ):
        """
        Initialize A/B testing manager.

        Args:
            model_registry: Model registry
            db_path: Path to SQLite database for A/B test data
            default_traffic_split: Default traffic % to challenger (0.0-1.0)
        """
        self.model_registry = model_registry or ModelRegistry()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_traffic_split = default_traffic_split

        self._init_database()

        logger.info(
            f"ABTestingManager initialized "
            f"(default_traffic_split={default_traffic_split})"
        )

    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # A/B tests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_tests (
                id TEXT PRIMARY KEY,
                champion_id TEXT NOT NULL,
                challenger_id TEXT NOT NULL,
                traffic_split REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                winner TEXT,
                reason TEXT
            )
        """)

        # Predictions table (for metrics collection)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                prediction TEXT NOT NULL,
                ground_truth TEXT,
                correct BOOLEAN,
                latency_ms REAL,
                confidence REAL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (test_id) REFERENCES ab_tests(id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_status ON ab_tests(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_test_id ON ab_predictions(test_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_model_id ON ab_predictions(model_id)")

        conn.commit()
        conn.close()

    def create_ab_test(
        self,
        champion_id: str,
        challenger_id: str,
        traffic_split: Optional[float] = None
    ) -> str:
        """
        Create new A/B test.

        Args:
            champion_id: Champion model ID
            challenger_id: Challenger model ID
            traffic_split: Traffic % to challenger (0.0-1.0)

        Returns:
            Test ID
        """
        test_id = str(uuid.uuid4())
        traffic_split = traffic_split or self.default_traffic_split

        logger.info(
            f"Creating A/B test: "
            f"test_id={test_id}, "
            f"champion={champion_id}, "
            f"challenger={challenger_id}, "
            f"traffic_split={traffic_split}"
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO ab_tests (
                id, champion_id, challenger_id, traffic_split,
                status, created_at, started_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id,
            champion_id,
            challenger_id,
            traffic_split,
            "active",
            datetime.now(),
            datetime.now()
        ))

        conn.commit()
        conn.close()

        logger.info(f"A/B test created: {test_id}")

        return test_id

    def route_prediction(self, test_id: str) -> str:
        """
        Route prediction to champion or challenger based on traffic split.

        Args:
            test_id: A/B test ID

        Returns:
            Model ID to use for prediction
        """
        test = self._get_test(test_id)

        if not test:
            raise ValueError(f"A/B test not found: {test_id}")

        # Random selection based on traffic split
        if random.random() < test["traffic_split"]:
            return test["challenger_id"]
        else:
            return test["champion_id"]

    def record_prediction(
        self,
        test_id: str,
        model_id: str,
        document_id: str,
        prediction: Dict[str, Any],
        ground_truth: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        confidence: Optional[float] = None
    ):
        """
        Record prediction result for A/B test.

        Args:
            test_id: A/B test ID
            model_id: Model that made prediction
            document_id: Document ID
            prediction: Prediction result
            ground_truth: Ground truth (if available)
            latency_ms: Prediction latency
            confidence: Prediction confidence
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Determine correctness
        correct = None
        if ground_truth:
            correct = prediction == ground_truth

        cursor.execute("""
            INSERT INTO ab_predictions (
                test_id, model_id, document_id, prediction,
                ground_truth, correct, latency_ms, confidence, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id,
            model_id,
            document_id,
            json.dumps(prediction),
            json.dumps(ground_truth) if ground_truth else None,
            correct,
            latency_ms,
            confidence,
            datetime.now()
        ))

        conn.commit()
        conn.close()

    def evaluate_ab_test(
        self,
        test_id: str,
        min_samples: int = 1000,
        significance_level: float = 0.05
    ) -> ABTestResult:
        """
        Evaluate A/B test and determine winner.

        Args:
            test_id: A/B test ID
            min_samples: Minimum samples per model
            significance_level: Statistical significance level (typical: 0.05)

        Returns:
            ABTestResult with evaluation
        """
        logger.info(f"Evaluating A/B test: {test_id}")

        test = self._get_test(test_id)
        if not test:
            raise ValueError(f"A/B test not found: {test_id}")

        # Get predictions for both models
        champion_preds = self._get_predictions(test_id, test["champion_id"])
        challenger_preds = self._get_predictions(test_id, test["challenger_id"])

        # Check if enough samples
        if len(champion_preds) < min_samples or len(challenger_preds) < min_samples:
            logger.info(
                f"Insufficient samples: "
                f"champion={len(champion_preds)}, "
                f"challenger={len(challenger_preds)} "
                f"(need {min_samples})"
            )

            return ABTestResult(
                test_id=test_id,
                champion_id=test["champion_id"],
                challenger_id=test["challenger_id"],
                champion_metrics={},
                challenger_metrics={},
                winner="inconclusive",
                p_value=1.0,
                confidence=0.0,
                sample_count_champion=len(champion_preds),
                sample_count_challenger=len(challenger_preds),
                recommendation="continue_testing",
                details={"reason": "insufficient_samples"}
            )

        # Calculate metrics
        champion_metrics = self._calculate_metrics(champion_preds)
        challenger_metrics = self._calculate_metrics(challenger_preds)

        logger.info(f"Champion metrics: {champion_metrics}")
        logger.info(f"Challenger metrics: {challenger_metrics}")

        # Statistical significance test
        p_value = self._test_significance(
            champion_preds,
            challenger_preds
        )

        # Determine winner
        is_significant = p_value < significance_level
        challenger_better = (
            challenger_metrics["accuracy"] > champion_metrics["accuracy"]
        )

        if is_significant and challenger_better:
            winner = "challenger"
            recommendation = "promote"
            confidence = 1.0 - p_value
        elif is_significant and not challenger_better:
            winner = "champion"
            recommendation = "keep_champion"
            confidence = 1.0 - p_value
        else:
            winner = "inconclusive"
            recommendation = "continue_testing"
            confidence = 0.5

        result = ABTestResult(
            test_id=test_id,
            champion_id=test["champion_id"],
            challenger_id=test["challenger_id"],
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            winner=winner,
            p_value=p_value,
            confidence=confidence,
            sample_count_champion=len(champion_preds),
            sample_count_challenger=len(challenger_preds),
            recommendation=recommendation,
            details={
                "significance_level": significance_level,
                "is_significant": is_significant,
                "improvement": challenger_metrics["accuracy"] - champion_metrics["accuracy"]
            }
        )

        logger.info(
            f"A/B test evaluation: "
            f"winner={winner}, "
            f"p_value={p_value:.4f}, "
            f"recommendation={recommendation}"
        )

        return result

    def complete_ab_test(
        self,
        test_id: str,
        winner: str,
        reason: Optional[str] = None
    ):
        """
        Mark A/B test as completed.

        Args:
            test_id: A/B test ID
            winner: Winner ("champion" or "challenger")
            reason: Reason for decision
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE ab_tests
            SET status = 'completed',
                completed_at = ?,
                winner = ?,
                reason = ?
            WHERE id = ?
        """, (datetime.now(), winner, reason, test_id))

        conn.commit()
        conn.close()

        logger.info(f"A/B test completed: {test_id}, winner={winner}")

    def _get_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test record."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM ab_tests WHERE id = ?", (test_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def _get_predictions(
        self,
        test_id: str,
        model_id: str
    ) -> List[Dict[str, Any]]:
        """Get predictions for a model in A/B test."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM ab_predictions
            WHERE test_id = ? AND model_id = ?
            ORDER BY timestamp DESC
        """, (test_id, model_id))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def _calculate_metrics(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate metrics from predictions."""
        if not predictions:
            return {
                "accuracy": 0.0,
                "avg_latency_ms": 0.0,
                "avg_confidence": 0.0,
                "sample_count": 0
            }

        # Accuracy (only for predictions with ground truth)
        predictions_with_gt = [p for p in predictions if p["ground_truth"] is not None]
        if predictions_with_gt:
            correct = sum(1 for p in predictions_with_gt if p["correct"])
            accuracy = correct / len(predictions_with_gt)
        else:
            accuracy = 0.0

        # Latency
        latencies = [p["latency_ms"] for p in predictions if p["latency_ms"] is not None]
        avg_latency = np.mean(latencies) if latencies else 0.0

        # Confidence
        confidences = [p["confidence"] for p in predictions if p["confidence"] is not None]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        return {
            "accuracy": accuracy,
            "avg_latency_ms": float(avg_latency),
            "avg_confidence": float(avg_confidence),
            "sample_count": len(predictions),
            "samples_with_ground_truth": len(predictions_with_gt)
        }

    def _test_significance(
        self,
        champion_preds: List[Dict[str, Any]],
        challenger_preds: List[Dict[str, Any]]
    ) -> float:
        """
        Test statistical significance using proportion test.

        Args:
            champion_preds: Champion predictions
            challenger_preds: Challenger predictions

        Returns:
            P-value
        """
        # Filter predictions with ground truth
        champion_with_gt = [p for p in champion_preds if p["ground_truth"] is not None]
        challenger_with_gt = [p for p in challenger_preds if p["ground_truth"] is not None]

        if not champion_with_gt or not challenger_with_gt:
            return 1.0  # Cannot test significance

        # Count correct predictions
        champion_correct = sum(1 for p in champion_with_gt if p["correct"])
        challenger_correct = sum(1 for p in challenger_with_gt if p["correct"])

        # Sample sizes
        n_champion = len(champion_with_gt)
        n_challenger = len(challenger_with_gt)

        # Two-proportion z-test
        p1 = champion_correct / n_champion
        p2 = challenger_correct / n_challenger

        # Pooled proportion
        p_pooled = (champion_correct + challenger_correct) / (n_champion + n_challenger)

        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_champion + 1/n_challenger))

        # Z-statistic
        if se == 0:
            return 1.0

        z = (p2 - p1) / se

        # P-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return p_value

    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get all active A/B tests."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM ab_tests
            WHERE status = 'active'
            ORDER BY created_at DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of A/B test."""
        test = self._get_test(test_id)
        if not test:
            raise ValueError(f"A/B test not found: {test_id}")

        champion_preds = self._get_predictions(test_id, test["champion_id"])
        challenger_preds = self._get_predictions(test_id, test["challenger_id"])

        champion_metrics = self._calculate_metrics(champion_preds)
        challenger_metrics = self._calculate_metrics(challenger_preds)

        return {
            "test": test,
            "champion": {
                "model_id": test["champion_id"],
                "predictions": len(champion_preds),
                "metrics": champion_metrics
            },
            "challenger": {
                "model_id": test["challenger_id"],
                "predictions": len(challenger_preds),
                "metrics": challenger_metrics
            }
        }
