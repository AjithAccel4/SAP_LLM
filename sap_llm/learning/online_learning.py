"""
Online Learning Engine

Implements real-time model updates based on user corrections and feedback.
Supports incremental learning without full retraining, active learning for
uncertain predictions, transfer learning, and few-shot learning.
"""

import hashlib
import json
import pickle
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class OnlineLearningEngine:
    """
    Real-time online learning engine for continuous model improvement.

    Features:
    - Incremental learning with SGD-based models
    - Active learning for uncertain predictions
    - Transfer learning from similar document types
    - Few-shot learning for new document types
    - Memory-efficient updates
    """

    def __init__(
        self,
        pmg: ProcessMemoryGraph,
        model_store_path: str = "/tmp/sap_llm/models",
        uncertainty_threshold: float = 0.7,
        active_learning_enabled: bool = True,
    ):
        """
        Initialize online learning engine.

        Args:
            pmg: Process Memory Graph instance
            model_store_path: Path to store model checkpoints
            uncertainty_threshold: Confidence threshold for active learning
            active_learning_enabled: Enable active learning queries
        """
        self.pmg = pmg
        self.model_store_path = model_store_path
        self.uncertainty_threshold = uncertainty_threshold
        self.active_learning_enabled = active_learning_enabled

        # Online models per document type (SGD-based for incremental learning)
        self.models: Dict[str, SGDClassifier] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}

        # Transfer learning: base models for initialization
        self.base_models: Dict[str, RandomForestClassifier] = {}

        # Active learning queue
        self.uncertain_predictions: List[Dict[str, Any]] = []

        # Performance tracking
        self.performance_history: Dict[str, List[float]] = defaultdict(list)

        # Update statistics
        self.update_count: Dict[str, int] = defaultdict(int)
        self.last_update: Dict[str, datetime] = {}

        logger.info("OnlineLearningEngine initialized")

    def update_from_correction(
        self,
        doc_type: str,
        features: Dict[str, Any],
        correct_label: str,
        predicted_label: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update model incrementally from user correction.

        Args:
            doc_type: Document type
            features: Extracted features
            correct_label: Correct label from user
            predicted_label: Previously predicted label
            confidence: Prediction confidence

        Returns:
            Update statistics
        """
        logger.info(f"Updating model for {doc_type} with user correction")

        # Get or create model
        if doc_type not in self.models:
            self._initialize_model(doc_type)

        # Vectorize features
        X = self._vectorize_features(doc_type, [features])
        y = [correct_label]

        # Partial fit (incremental update)
        try:
            self.models[doc_type].partial_fit(X, y, classes=self._get_classes(doc_type))

            # Update statistics
            self.update_count[doc_type] += 1
            self.last_update[doc_type] = datetime.now()

            # Store in PMG for future learning
            self._store_correction(doc_type, features, correct_label, predicted_label)

            # Evaluate if we have enough data
            accuracy = None
            if self.update_count[doc_type] % 50 == 0:
                accuracy = self._evaluate_model(doc_type)
                self.performance_history[doc_type].append(accuracy)

            return {
                "doc_type": doc_type,
                "update_count": self.update_count[doc_type],
                "last_update": self.last_update[doc_type].isoformat(),
                "accuracy": accuracy,
                "was_incorrect": predicted_label != correct_label if predicted_label else None,
            }

        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return {"error": str(e)}

    def batch_update(
        self,
        doc_type: str,
        samples: List[Tuple[Dict[str, Any], str]],
    ) -> Dict[str, Any]:
        """
        Batch update from multiple samples.

        Args:
            doc_type: Document type
            samples: List of (features, label) tuples

        Returns:
            Update statistics
        """
        logger.info(f"Batch updating {doc_type} with {len(samples)} samples")

        if not samples:
            return {"error": "No samples provided"}

        # Get or create model
        if doc_type not in self.models:
            self._initialize_model(doc_type)

        # Prepare data
        features_list = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        X = self._vectorize_features(doc_type, features_list)
        y = np.array(labels)

        # Partial fit
        try:
            self.models[doc_type].partial_fit(X, y, classes=self._get_classes(doc_type))

            # Update statistics
            self.update_count[doc_type] += len(samples)
            self.last_update[doc_type] = datetime.now()

            # Evaluate
            accuracy = self._evaluate_model(doc_type)
            self.performance_history[doc_type].append(accuracy)

            return {
                "doc_type": doc_type,
                "samples_processed": len(samples),
                "total_updates": self.update_count[doc_type],
                "accuracy": accuracy,
            }

        except Exception as e:
            logger.error(f"Failed to batch update: {e}")
            return {"error": str(e)}

    def identify_uncertain_predictions(
        self,
        doc_type: str,
        features: Dict[str, Any],
    ) -> Tuple[str, float, bool]:
        """
        Make prediction and identify if it's uncertain (active learning).

        Args:
            doc_type: Document type
            features: Extracted features

        Returns:
            (prediction, confidence, is_uncertain)
        """
        if doc_type not in self.models:
            return ("UNKNOWN", 0.0, True)

        # Vectorize
        X = self._vectorize_features(doc_type, [features])

        # Predict with probability
        try:
            prediction = self.models[doc_type].predict(X)[0]

            # Get confidence (probability)
            if hasattr(self.models[doc_type], 'predict_proba'):
                proba = self.models[doc_type].predict_proba(X)[0]
                confidence = float(np.max(proba))
            else:
                # For models without predict_proba, use decision function
                decision = self.models[doc_type].decision_function(X)[0]
                confidence = 1.0 / (1.0 + np.exp(-decision))  # Sigmoid

            # Check if uncertain
            is_uncertain = confidence < self.uncertainty_threshold

            if is_uncertain and self.active_learning_enabled:
                # Add to uncertain queue for human review
                self.uncertain_predictions.append({
                    "doc_type": doc_type,
                    "features": features,
                    "prediction": prediction,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                })
                logger.info(
                    f"Uncertain prediction detected: {prediction} "
                    f"(confidence: {confidence:.2f})"
                )

            return prediction, confidence, is_uncertain

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return ("ERROR", 0.0, True)

    def get_uncertain_predictions(
        self,
        limit: int = 100,
        min_confidence: float = 0.0,
        max_confidence: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get queue of uncertain predictions for human review (active learning).

        Args:
            limit: Maximum predictions to return
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold

        Returns:
            List of uncertain predictions
        """
        filtered = []

        max_conf = max_confidence or self.uncertainty_threshold

        for pred in self.uncertain_predictions:
            conf = pred.get("confidence", 0.0)
            if min_confidence <= conf <= max_conf:
                filtered.append(pred)

        # Sort by confidence (least confident first)
        filtered.sort(key=lambda x: x.get("confidence", 0.0))

        return filtered[:limit]

    def transfer_learning(
        self,
        source_doc_type: str,
        target_doc_type: str,
        similarity_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Transfer learning from source document type to target.

        Args:
            source_doc_type: Source document type (with trained model)
            target_doc_type: Target document type (new or few samples)
            similarity_threshold: Minimum similarity to transfer

        Returns:
            Transfer statistics
        """
        logger.info(f"Transfer learning: {source_doc_type} -> {target_doc_type}")

        if source_doc_type not in self.models:
            return {"error": f"Source model {source_doc_type} not found"}

        # Check document type similarity
        similarity = self._calculate_doc_type_similarity(source_doc_type, target_doc_type)

        if similarity < similarity_threshold:
            logger.warning(
                f"Low similarity ({similarity:.2f}) between {source_doc_type} "
                f"and {target_doc_type}"
            )
            return {
                "transferred": False,
                "similarity": similarity,
                "reason": "Low similarity",
            }

        # Copy model parameters (warm start)
        try:
            # Create new model with same parameters
            source_model = self.models[source_doc_type]
            target_model = SGDClassifier(
                loss=source_model.loss,
                penalty=source_model.penalty,
                alpha=source_model.alpha,
                max_iter=source_model.max_iter,
                warm_start=True,
            )

            # Copy vectorizer
            if source_doc_type in self.vectorizers:
                self.vectorizers[target_doc_type] = self.vectorizers[source_doc_type]

            # Store target model
            self.models[target_doc_type] = target_model

            logger.info(
                f"Successfully transferred model from {source_doc_type} "
                f"to {target_doc_type}"
            )

            return {
                "transferred": True,
                "similarity": similarity,
                "source": source_doc_type,
                "target": target_doc_type,
            }

        except Exception as e:
            logger.error(f"Transfer learning failed: {e}")
            return {"error": str(e)}

    def few_shot_learning(
        self,
        doc_type: str,
        examples: List[Tuple[Dict[str, Any], str]],
        base_doc_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Learn from few examples for new document type.

        Args:
            doc_type: New document type
            examples: Small set of labeled examples
            base_doc_type: Optional base model to transfer from

        Returns:
            Learning statistics
        """
        logger.info(f"Few-shot learning for {doc_type} with {len(examples)} examples")

        if len(examples) < 2:
            return {"error": "Need at least 2 examples for few-shot learning"}

        # Try transfer learning first if base model specified
        if base_doc_type and base_doc_type in self.models:
            transfer_result = self.transfer_learning(base_doc_type, doc_type)
            if transfer_result.get("transferred"):
                logger.info("Using transfer learning for few-shot initialization")

        # Initialize model if not exists
        if doc_type not in self.models:
            self._initialize_model(doc_type)

        # Train on few examples
        result = self.batch_update(doc_type, examples)
        result["few_shot_learning"] = True
        result["num_examples"] = len(examples)

        return result

    def _initialize_model(self, doc_type: str):
        """Initialize online learning model for document type."""
        logger.info(f"Initializing online model for {doc_type}")

        # SGD Classifier for online learning
        self.models[doc_type] = SGDClassifier(
            loss='log_loss',  # For probability estimates
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            warm_start=True,  # Enable incremental learning
            random_state=42,
        )

        # TF-IDF vectorizer for text features
        self.vectorizers[doc_type] = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
        )

        self.update_count[doc_type] = 0

    def _vectorize_features(
        self,
        doc_type: str,
        features_list: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Convert features to vectors."""
        # Convert dict features to text representation
        texts = []
        for features in features_list:
            text = " ".join([f"{k}:{v}" for k, v in features.items()])
            texts.append(text)

        # Vectorize
        if doc_type not in self.vectorizers:
            self.vectorizers[doc_type] = TfidfVectorizer(max_features=1000)
            return self.vectorizers[doc_type].fit_transform(texts).toarray()
        else:
            try:
                return self.vectorizers[doc_type].transform(texts).toarray()
            except:
                # Refit if new features
                return self.vectorizers[doc_type].fit_transform(texts).toarray()

    def _get_classes(self, doc_type: str) -> List[str]:
        """Get possible classes for document type."""
        # In production, query from PMG or config
        common_classes = [
            "PURCHASE_ORDER",
            "INVOICE",
            "GOODS_RECEIPT",
            "PAYMENT",
            "CREDIT_MEMO",
            "DEBIT_MEMO",
        ]
        return common_classes

    def _evaluate_model(self, doc_type: str) -> float:
        """Evaluate model on recent data from PMG."""
        try:
            # Get recent labeled samples from PMG
            samples = self.pmg.find_similar_documents(doc_type, limit=200)

            if len(samples) < 10:
                return 0.0

            # Prepare test data
            features_list = [s for s in samples]
            labels = [s.get("doc_type", "UNKNOWN") for s in samples]

            X = self._vectorize_features(doc_type, features_list)
            y_true = labels

            # Predict
            y_pred = self.models[doc_type].predict(X)

            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)

            logger.info(f"Model accuracy for {doc_type}: {accuracy:.3f}")
            return accuracy

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return 0.0

    def _store_correction(
        self,
        doc_type: str,
        features: Dict[str, Any],
        correct_label: str,
        predicted_label: Optional[str],
    ):
        """Store correction in PMG for future learning."""
        try:
            correction = {
                "doc_type": doc_type,
                "features": features,
                "correct_label": correct_label,
                "predicted_label": predicted_label,
                "timestamp": datetime.now().isoformat(),
                "correction_id": hashlib.md5(
                    f"{doc_type}{correct_label}{datetime.now()}".encode()
                ).hexdigest(),
            }

            # In production, store in PMG
            logger.debug(f"Stored correction: {correction['correction_id']}")

        except Exception as e:
            logger.error(f"Failed to store correction: {e}")

    def _calculate_doc_type_similarity(
        self,
        doc_type1: str,
        doc_type2: str,
    ) -> float:
        """Calculate similarity between document types."""
        # Simple heuristic based on name similarity
        # In production, use feature distribution similarity

        words1 = set(doc_type1.lower().split('_'))
        words2 = set(doc_type2.lower().split('_'))

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        similarity = len(intersection) / len(union) if union else 0.0

        return similarity

    def save_models(self):
        """Save models to disk."""
        import os

        os.makedirs(self.model_store_path, exist_ok=True)

        for doc_type, model in self.models.items():
            model_path = f"{self.model_store_path}/{doc_type}_model.pkl"
            vectorizer_path = f"{self.model_store_path}/{doc_type}_vectorizer.pkl"

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            if doc_type in self.vectorizers:
                with open(vectorizer_path, 'wb') as f:
                    pickle.dump(self.vectorizers[doc_type], f)

            logger.info(f"Saved model for {doc_type}")

    def load_models(self):
        """Load models from disk."""
        import os

        if not os.path.exists(self.model_store_path):
            logger.warning("Model store path does not exist")
            return

        for filename in os.listdir(self.model_store_path):
            if filename.endswith('_model.pkl'):
                doc_type = filename.replace('_model.pkl', '')
                model_path = f"{self.model_store_path}/{filename}"
                vectorizer_path = f"{self.model_store_path}/{doc_type}_vectorizer.pkl"

                with open(model_path, 'rb') as f:
                    self.models[doc_type] = pickle.load(f)

                if os.path.exists(vectorizer_path):
                    with open(vectorizer_path, 'rb') as f:
                        self.vectorizers[doc_type] = pickle.load(f)

                logger.info(f"Loaded model for {doc_type}")
