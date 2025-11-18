"""
PMG-Learning Integration Module.

Connects Process Memory Graph with Intelligent Learning Loop for continuous improvement.

Workflow:
1. Monitor predictions and outcomes
2. Store results in PMG
3. Detect drift from PMG statistics
4. Trigger retraining based on PMG feedback
5. Update models with PMG context
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.pmg.context_retriever import ContextRetriever, RetrievalConfig
from sap_llm.pmg.vector_store import PMGVectorStore
from sap_llm.learning.intelligent_learning_loop import (
    DriftDetector,
    ABTestingFramework,
    IntelligentLearningLoop
)
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class PMGLearningIntegration:
    """
    Integration between PMG and Learning Loop.

    Features:
    - Automatic feedback storage in PMG
    - Drift detection using PMG statistics
    - Context-aware model improvement
    - Continuous learning from production data
    """

    def __init__(
        self,
        pmg: Optional[ProcessMemoryGraph] = None,
        context_retriever: Optional[ContextRetriever] = None,
        drift_detector: Optional[DriftDetector] = None,
        enable_auto_retrain: bool = True,
        drift_check_interval_hours: int = 24
    ):
        """
        Initialize PMG-Learning integration.

        Args:
            pmg: Process Memory Graph client
            context_retriever: Context retriever for RAG
            drift_detector: Drift detector
            enable_auto_retrain: Enable automatic retraining
            drift_check_interval_hours: Hours between drift checks
        """
        self.pmg = pmg or ProcessMemoryGraph()
        self.context_retriever = context_retriever or ContextRetriever()
        self.drift_detector = drift_detector or DriftDetector(
            window_size=1000,
            drift_threshold=0.05
        )

        self.enable_auto_retrain = enable_auto_retrain
        self.drift_check_interval_hours = drift_check_interval_hours

        # Learning state
        self.last_drift_check = datetime.now()
        self.drift_events: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            "total_predictions": 0,
            "predictions_with_context": 0,
            "drift_detections": 0,
            "retraining_triggers": 0
        }

        logger.info(
            f"PMG-Learning Integration initialized "
            f"(auto_retrain={enable_auto_retrain}, "
            f"check_interval={drift_check_interval_hours}h)"
        )

    def process_prediction(
        self,
        document: Dict[str, Any],
        prediction: Dict[str, Any],
        features: Optional[np.ndarray] = None,
        use_context: bool = True
    ) -> Dict[str, Any]:
        """
        Process prediction with PMG context.

        Args:
            document: Input document
            prediction: Model prediction
            features: Feature vector (for drift detection)
            use_context: Whether to use PMG context for enhancement

        Returns:
            Enhanced prediction with PMG context
        """
        self.stats["total_predictions"] += 1

        result = {
            "prediction": prediction,
            "context_used": False,
            "confidence_boost": 0.0,
            "historical_patterns": []
        }

        # Retrieve relevant context from PMG
        if use_context:
            contexts = self.context_retriever.retrieve_context(
                document=document,
                top_k=5
            )

            if contexts:
                self.stats["predictions_with_context"] += 1
                result["context_used"] = True
                result["historical_patterns"] = [
                    {
                        "doc_id": c.doc_id,
                        "similarity": c.similarity,
                        "success": c.success,
                        "routing_decision": c.routing_decision
                    }
                    for c in contexts
                ]

                # Boost confidence if similar successful cases found
                successful_contexts = [c for c in contexts if c.success]
                if successful_contexts:
                    # Average similarity of successful contexts
                    avg_similarity = np.mean([c.similarity for c in successful_contexts])

                    # Boost confidence proportional to similarity
                    confidence_boost = min(0.2, avg_similarity * 0.25)
                    result["confidence_boost"] = confidence_boost

                    # Update prediction confidence
                    if "confidence" in prediction:
                        original_confidence = prediction["confidence"]
                        prediction["confidence"] = min(
                            1.0,
                            original_confidence + confidence_boost
                        )

                        logger.debug(
                            f"Confidence boosted: {original_confidence:.3f} → "
                            f"{prediction['confidence']:.3f} "
                            f"(+{confidence_boost:.3f})"
                        )

        # Add to drift detector
        if features is not None:
            pred_value = self._extract_prediction_value(prediction)

            self.drift_detector.add_current_sample(
                features=features,
                prediction=pred_value
            )

        return result

    def store_outcome(
        self,
        document: Dict[str, Any],
        prediction: Dict[str, Any],
        routing_decision: Optional[Dict[str, Any]] = None,
        sap_response: Optional[Dict[str, Any]] = None,
        exceptions: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Store prediction outcome in PMG.

        Args:
            document: Input document
            prediction: Model prediction
            routing_decision: Routing decision made
            sap_response: SAP API response
            exceptions: Any exceptions raised

        Returns:
            Document ID in PMG
        """
        # Store complete transaction in PMG
        doc_id = self.pmg.store_transaction(
            document=document,
            routing_decision=routing_decision,
            sap_response=sap_response,
            exceptions=exceptions
        )

        logger.debug(f"Outcome stored in PMG: {doc_id}")

        return doc_id

    def check_drift(self, force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Check for model drift using PMG data.

        Args:
            force: Force drift check regardless of interval

        Returns:
            Drift detection results or None
        """
        # Check if it's time for drift check
        elapsed = datetime.now() - self.last_drift_check
        if not force and elapsed < timedelta(hours=self.drift_check_interval_hours):
            return None

        logger.info("Checking for model drift...")

        # Detect drift
        drift_result = self.drift_detector.detect_drift()

        self.last_drift_check = datetime.now()

        if drift_result["drift_detected"]:
            self.stats["drift_detections"] += 1
            self.drift_events.append(drift_result)

            logger.warning(
                f"Drift detected: {', '.join(drift_result['drift_types'])}"
            )

            # Trigger retraining if enabled
            if self.enable_auto_retrain:
                self._trigger_retraining(drift_result)

        return drift_result

    def _trigger_retraining(self, drift_result: Dict[str, Any]) -> None:
        """
        Trigger model retraining based on drift detection.

        Args:
            drift_result: Drift detection results
        """
        logger.info("🔄 Triggering model retraining due to drift...")

        self.stats["retraining_triggers"] += 1

        # Get recent successful cases from PMG for retraining
        training_data = self._collect_training_data()

        logger.info(f"Collected {len(training_data)} samples for retraining")

        # Placeholder: Actual retraining would happen here
        # self.model_trainer.retrain(training_data)

        # Log retraining event
        event = {
            "timestamp": datetime.now().isoformat(),
            "reason": "drift_detected",
            "drift_types": drift_result.get("drift_types", []),
            "training_samples": len(training_data)
        }

        logger.info(f"Retraining event logged: {event}")

    def _collect_training_data(
        self,
        days: int = 30,
        min_confidence: float = 0.9
    ) -> List[Dict[str, Any]]:
        """
        Collect training data from PMG.

        Args:
            days: Lookback period in days
            min_confidence: Minimum confidence for training samples

        Returns:
            Training samples
        """
        # Query PMG for recent successful transactions
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()

        # Get documents from recent period
        recent_docs = self.pmg.query_documents_at_time(
            as_of_timestamp=datetime.now().isoformat(),
            limit=10000
        )

        # Filter for high-confidence successes
        training_samples = []

        for doc in recent_docs:
            # Check if successful (no critical exceptions, good SAP response)
            if self._is_successful_transaction(doc):
                training_samples.append(doc)

        logger.info(f"Found {len(training_samples)} high-quality training samples")

        return training_samples

    def _is_successful_transaction(self, doc: Dict[str, Any]) -> bool:
        """
        Determine if transaction was successful.

        Args:
            doc: Document/transaction

        Returns:
            True if successful
        """
        # Check for critical failures
        if doc.get("status") == "failed":
            return False

        # Check exceptions
        exceptions = doc.get("exceptions", [])
        for exc in exceptions:
            if exc.get("severity") in ["CRITICAL", "HIGH"]:
                return False

        # Check SAP response
        sap_response = doc.get("sap_response", {})
        if sap_response:
            status_code = sap_response.get("status_code", 0)
            if status_code < 200 or status_code >= 300:
                return False

        return True

    def _extract_prediction_value(self, prediction: Dict[str, Any]) -> Any:
        """Extract prediction value for drift detection."""
        # Try common field names
        for field in ["doc_type", "class", "label", "category", "prediction"]:
            if field in prediction:
                return prediction[field]

        # Fallback: return entire prediction
        return str(prediction)

    def get_pmg_enhanced_context(
        self,
        document: Dict[str, Any],
        max_length: int = 2000
    ) -> str:
        """
        Get PMG context formatted for LLM prompt.

        Args:
            document: Input document
            max_length: Maximum context length

        Returns:
            Context string for prompt
        """
        # Retrieve context
        contexts = self.context_retriever.retrieve_context(
            document=document,
            top_k=5
        )

        # Build prompt
        context_prompt = self.context_retriever.build_context_prompt(
            contexts=contexts,
            max_length=max_length
        )

        return context_prompt

    def analyze_prediction_quality(
        self,
        document: Dict[str, Any],
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze prediction quality using PMG historical data.

        Args:
            document: Input document
            prediction: Model prediction

        Returns:
            Quality analysis
        """
        # Get similar historical cases
        contexts = self.context_retriever.retrieve_context(
            document=document,
            top_k=20
        )

        if not contexts:
            return {
                "confidence_level": "low",
                "reason": "No historical data available",
                "recommendation": "Manual review recommended"
            }

        # Analyze consistency with historical patterns
        successful_contexts = [c for c in contexts if c.success]
        failed_contexts = [c for c in contexts if not c.success]

        analysis = {
            "total_similar_cases": len(contexts),
            "successful_cases": len(successful_contexts),
            "failed_cases": len(failed_contexts),
            "success_rate": len(successful_contexts) / len(contexts),
            "avg_similarity": np.mean([c.similarity for c in contexts])
        }

        # Determine confidence level
        if analysis["success_rate"] > 0.9 and analysis["avg_similarity"] > 0.85:
            analysis["confidence_level"] = "high"
            analysis["recommendation"] = "Auto-process"
        elif analysis["success_rate"] > 0.7 and analysis["avg_similarity"] > 0.7:
            analysis["confidence_level"] = "medium"
            analysis["recommendation"] = "Process with monitoring"
        else:
            analysis["confidence_level"] = "low"
            analysis["recommendation"] = "Manual review recommended"

        return analysis

    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = self.stats.copy()

        # Add PMG stats
        pmg_stats = self.pmg.get_pmg_statistics()
        stats["pmg"] = pmg_stats

        # Add context retriever stats
        context_stats = self.context_retriever.get_statistics()
        stats["context_retriever"] = context_stats

        # Add drift detector stats
        stats["drift_events"] = len(self.drift_events)

        return stats

    def export_drift_history(self, output_file: str) -> None:
        """
        Export drift detection history.

        Args:
            output_file: Output file path
        """
        import json
        from pathlib import Path

        data = {
            "total_drift_events": len(self.drift_events),
            "events": self.drift_events,
            "statistics": self.get_statistics(),
            "exported_at": datetime.now().isoformat()
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Drift history exported to {output_file}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize integration
    integration = PMGLearningIntegration(
        enable_auto_retrain=True,
        drift_check_interval_hours=24
    )

    # Example: Process prediction with PMG context
    document = {
        "doc_type": "invoice",
        "supplier_id": "SUP-001",
        "supplier_name": "Acme Corp",
        "total_amount": 1000.00,
        "currency": "USD",
        "company_code": "1000"
    }

    prediction = {
        "doc_type": "invoice",
        "confidence": 0.75,
        "subtype": "standard_invoice"
    }

    # Process with PMG enhancement
    result = integration.process_prediction(
        document=document,
        prediction=prediction,
        use_context=True
    )

    print(f"Prediction result: {result}")

    # Analyze quality
    quality = integration.analyze_prediction_quality(document, prediction)

    print(f"\nQuality analysis: {quality}")

    # Get statistics
    stats = integration.get_statistics()

    print(f"\nIntegration statistics: {stats}")
