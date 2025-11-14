"""
Continuous Learning from Process Memory Graph

Monitors PMG for new transactions and uses successful cases to improve models.
Implements drift detection and automatic retraining triggers.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ContinuousLearner:
    """
    Continuous learning engine for SAP_LLM.

    Monitors PMG for:
    - High-confidence successful transactions
    - Model drift indicators
    - New patterns and edge cases

    Triggers:
    - Incremental model updates
    - Full retraining when drift detected
    - Rule adjustments
    """

    def __init__(
        self,
        pmg: ProcessMemoryGraph,
        config: Optional[Any] = None,
    ):
        """
        Initialize continuous learner.

        Args:
            pmg: Process Memory Graph instance
            config: Learning configuration
        """
        self.pmg = pmg
        self.config = config

        # Learning parameters
        self.min_confidence = 0.95
        self.batch_size = 100
        self.drift_threshold = 0.25
        self.retraining_interval = "daily"

        if config and hasattr(config, "continuous_learning"):
            cl_config = config.continuous_learning
            self.batch_size = getattr(cl_config, "batch_size", 100)
            self.drift_threshold = getattr(cl_config, "drift_threshold", 0.25)
            self.retraining_interval = getattr(cl_config, "retraining_interval", "daily")

        logger.info("ContinuousLearner initialized")

    def learn_from_feedback(self, days: int = 7) -> Dict[str, Any]:
        """
        Learn from recent transactions with outcomes.

        Args:
            days: Number of days to look back

        Returns:
            Learning statistics
        """
        logger.info(f"Learning from feedback (last {days} days)...")

        # Query PMG for recent successful transactions
        successful_docs = self._get_successful_transactions(days)

        if not successful_docs:
            logger.warning("No successful transactions found for learning")
            return {
                "num_samples": 0,
                "drift_detected": False,
                "retraining_triggered": False,
            }

        # Filter high-confidence cases
        high_confidence_docs = [
            doc for doc in successful_docs
            if doc.get("confidence", 0) >= self.min_confidence
        ]

        logger.info(
            f"Found {len(successful_docs)} successful transactions, "
            f"{len(high_confidence_docs)} high-confidence"
        )

        # Detect drift
        drift_score = self.detect_drift(high_confidence_docs)
        drift_detected = drift_score > self.drift_threshold

        if drift_detected:
            logger.warning(f"Model drift detected! PSI score: {drift_score:.4f}")

        # Trigger retraining if needed
        retraining_triggered = False
        if drift_detected or len(high_confidence_docs) >= self.batch_size:
            logger.info("Triggering model retraining...")
            retraining_triggered = self._trigger_retraining(high_confidence_docs)

        return {
            "num_samples": len(successful_docs),
            "high_confidence_samples": len(high_confidence_docs),
            "drift_score": drift_score,
            "drift_detected": drift_detected,
            "retraining_triggered": retraining_triggered,
        }

    def _get_successful_transactions(self, days: int) -> List[Dict[str, Any]]:
        """Get successful transactions from PMG."""
        # Calculate lookback date
        lookback_date = datetime.now() - timedelta(days=days)

        # This is a simplified query - in production would use Gremlin
        # to find documents with successful SAP responses
        try:
            # Get documents
            similar_docs = self.pmg.find_similar_documents(
                doc_type="SUPPLIER_INVOICE",  # Example
                limit=1000,
            )

            # Filter by date and success
            successful = []
            for doc in similar_docs:
                # Check if document has successful outcome
                # In real implementation, would check SAP response
                successful.append(doc)

            return successful

        except Exception as e:
            logger.error(f"Failed to get successful transactions: {e}")
            return []

    def detect_drift(
        self,
        recent_samples: List[Dict[str, Any]],
        lookback_days: int = 90,
    ) -> float:
        """
        Detect model drift using Population Stability Index (PSI).

        Args:
            recent_samples: Recent transaction samples
            lookback_days: Days to look back for baseline

        Returns:
            PSI score (0 = no drift, >0.25 = significant drift)
        """
        if not recent_samples:
            return 0.0

        try:
            # Get historical baseline
            baseline_docs = self._get_baseline_distribution(lookback_days)

            if not baseline_docs:
                logger.warning("No baseline data for drift detection")
                return 0.0

            # Compute distributions
            recent_dist = self._compute_distribution(
                recent_samples,
                field="total_amount",
            )
            baseline_dist = self._compute_distribution(
                baseline_docs,
                field="total_amount",
            )

            # Calculate PSI
            psi = self._calculate_psi(baseline_dist, recent_dist)

            logger.debug(f"PSI score: {psi:.4f}")
            return psi

        except Exception as e:
            logger.error(f"Failed to detect drift: {e}")
            return 0.0

    def _get_baseline_distribution(
        self,
        days: int,
    ) -> List[Dict[str, Any]]:
        """Get historical baseline for drift detection."""
        # Get historical documents
        return self.pmg.find_similar_documents(
            doc_type="SUPPLIER_INVOICE",
            limit=10000,
        )

    def _compute_distribution(
        self,
        samples: List[Dict[str, Any]],
        field: str,
        num_bins: int = 10,
    ) -> Dict[int, float]:
        """
        Compute distribution of field values.

        Args:
            samples: Document samples
            field: Field to compute distribution for
            num_bins: Number of bins

        Returns:
            Dictionary mapping bin index to proportion
        """
        # Extract values
        values = []
        for sample in samples:
            if field in sample:
                try:
                    values.append(float(sample[field]))
                except (ValueError, TypeError):
                    continue

        if not values:
            return {}

        # Create bins
        values_array = np.array(values)
        min_val = values_array.min()
        max_val = values_array.max()

        bins = np.linspace(min_val, max_val, num_bins + 1)
        hist, _ = np.histogram(values_array, bins=bins)

        # Convert to proportions
        total = hist.sum()
        distribution = {}
        for i, count in enumerate(hist):
            distribution[i] = count / total if total > 0 else 0

        return distribution

    def _calculate_psi(
        self,
        expected: Dict[int, float],
        actual: Dict[int, float],
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI = sum((actual% - expected%) * ln(actual% / expected%))

        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change
        - PSI >= 0.25: Significant change (drift)
        """
        psi = 0.0

        # Get all bins
        all_bins = set(expected.keys()) | set(actual.keys())

        for bin_idx in all_bins:
            exp_prop = expected.get(bin_idx, 0.001)  # Small value to avoid log(0)
            act_prop = actual.get(bin_idx, 0.001)

            if exp_prop > 0 and act_prop > 0:
                psi += (act_prop - exp_prop) * np.log(act_prop / exp_prop)

        return psi

    def _trigger_retraining(self, training_samples: List[Dict[str, Any]]) -> bool:
        """
        Trigger model retraining with new samples.

        Args:
            training_samples: Samples to use for retraining

        Returns:
            True if retraining was triggered successfully
        """
        logger.info(f"Triggering retraining with {len(training_samples)} samples")

        # In production, this would:
        # 1. Prepare training data
        # 2. Submit training job to training cluster
        # 3. Monitor training progress
        # 4. Validate new model
        # 5. Deploy if validation passes

        # For now, just log
        logger.info("[MOCK] Retraining job submitted")

        return True

    def collect_training_data(
        self,
        min_samples: int = 1000,
        max_samples: int = 100000,
    ) -> List[Dict[str, Any]]:
        """
        Collect high-quality training data from PMG.

        Args:
            min_samples: Minimum samples required
            max_samples: Maximum samples to collect

        Returns:
            List of training samples
        """
        logger.info(f"Collecting training data (min={min_samples}, max={max_samples})")

        training_data = []

        # Query PMG for successful, high-confidence transactions
        # across all document types
        doc_types = [
            "PURCHASE_ORDER",
            "SUPPLIER_INVOICE",
            "SALES_ORDER",
        ]

        for doc_type in doc_types:
            docs = self.pmg.find_similar_documents(
                doc_type=doc_type,
                limit=max_samples // len(doc_types),
            )

            # Filter for high confidence
            high_conf = [
                doc for doc in docs
                if doc.get("confidence", 0) >= self.min_confidence
            ]

            training_data.extend(high_conf)

            logger.info(f"Collected {len(high_conf)} samples for {doc_type}")

        logger.info(f"Total training samples collected: {len(training_data)}")

        return training_data
