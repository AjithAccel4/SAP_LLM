"""
Exception Clusterer - Groups similar exceptions using HDBSCAN

Uses semantic embeddings and density-based clustering to identify
patterns in exceptions that can be resolved with rule changes.
"""

import uuid
from typing import Any, Dict, List

import hdbscan
import numpy as np
from sentence_transformers import SentenceTransformer

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ExceptionClusterer:
    """
    Cluster similar exceptions for pattern detection.

    Uses:
    - SentenceTransformers for semantic embeddings
    - HDBSCAN for density-based clustering
    - Minimum cluster size filtering
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_cluster_size: int = 15,
        metric: str = "cosine",
        cluster_selection_epsilon: float = 0.3,
    ):
        """
        Initialize exception clusterer.

        Args:
            embedding_model: Sentence embedding model
            min_cluster_size: Minimum exceptions per cluster
            metric: Distance metric (cosine, euclidean)
            cluster_selection_epsilon: HDBSCAN epsilon parameter
        """
        self.embedding_model_name = embedding_model
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        logger.info(
            f"ExceptionClusterer initialized "
            f"(min_cluster_size={min_cluster_size}, metric={metric})"
        )

    def cluster(
        self,
        exceptions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Cluster exceptions by similarity.

        Args:
            exceptions: List of exception dictionaries

        Returns:
            List of clusters, each containing:
            - id: Cluster ID
            - exceptions: List of exceptions in cluster
            - category: Most common category
            - severity: Most common severity
            - centroid: Cluster centroid embedding
        """
        if not exceptions:
            logger.warning("No exceptions to cluster")
            return []

        logger.info(f"Clustering {len(exceptions)} exceptions...")

        # Generate embeddings
        embeddings = self._embed_exceptions(exceptions)

        # Cluster using HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )

        labels = clusterer.fit_predict(embeddings)

        # Group by cluster
        clusters_dict: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise
                continue

            if label not in clusters_dict:
                clusters_dict[label] = []

            clusters_dict[label].append(idx)

        logger.info(f"Found {len(clusters_dict)} clusters")

        # Create cluster objects
        clusters = []
        for label, indices in clusters_dict.items():
            cluster_exceptions = [exceptions[i] for i in indices]
            cluster_embeddings = embeddings[indices]

            # Compute centroid
            centroid = cluster_embeddings.mean(axis=0)

            # Determine cluster properties
            categories = [e.get("category", "UNKNOWN") for e in cluster_exceptions]
            severities = [e.get("severity", "MEDIUM") for e in cluster_exceptions]

            most_common_category = max(set(categories), key=categories.count)
            most_common_severity = max(set(severities), key=severities.count)

            cluster = {
                "id": str(uuid.uuid4()),
                "label": int(label),
                "exceptions": cluster_exceptions,
                "size": len(cluster_exceptions),
                "category": most_common_category,
                "severity": most_common_severity,
                "centroid": centroid.tolist(),
            }

            clusters.append(cluster)

        # Sort by size (descending)
        clusters.sort(key=lambda c: c["size"], reverse=True)

        logger.info(
            f"Clustering complete: {len(clusters)} clusters, "
            f"largest={clusters[0]['size'] if clusters else 0}"
        )

        return clusters

    def _embed_exceptions(self, exceptions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Generate embeddings for exceptions.

        Args:
            exceptions: List of exceptions

        Returns:
            Numpy array of embeddings [N, embedding_dim]
        """
        # Convert exceptions to text
        texts = [self._exception_to_text(exc) for exc in exceptions]

        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(exceptions) > 100,
        )

        # Store embeddings in exceptions for future use
        for exc, embedding in zip(exceptions, embeddings):
            exc["embedding"] = embedding.tolist()

        return embeddings

    def _exception_to_text(self, exception: Dict[str, Any]) -> str:
        """
        Convert exception to text for embedding.

        Args:
            exception: Exception dictionary

        Returns:
            Text representation
        """
        parts = []

        # Category
        if "category" in exception:
            parts.append(f"Category: {exception['category']}")

        # Severity
        if "severity" in exception:
            parts.append(f"Severity: {exception['severity']}")

        # Field
        if "field" in exception:
            parts.append(f"Field: {exception['field']}")

        # Expected/Actual
        if "expected" in exception and "actual" in exception:
            parts.append(
                f"Expected: {exception['expected']}, "
                f"Actual: {exception.get('value', exception.get('actual'))}"
            )

        # Message
        if "message" in exception:
            parts.append(f"Message: {exception['message']}")

        return " | ".join(parts)

    def find_cluster_for_exception(
        self,
        exception: Dict[str, Any],
        clusters: List[Dict[str, Any]],
        similarity_threshold: float = 0.8,
    ) -> Dict[str, Any] | None:
        """
        Find which cluster an exception belongs to.

        Args:
            exception: Exception to classify
            clusters: List of clusters
            similarity_threshold: Minimum similarity

        Returns:
            Matching cluster or None
        """
        # Generate embedding for exception
        text = self._exception_to_text(exception)
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)

        # Compute similarity to each cluster centroid
        best_cluster = None
        best_similarity = 0.0

        for cluster in clusters:
            centroid = np.array(cluster["centroid"])

            # Cosine similarity
            similarity = np.dot(embedding, centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(centroid)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster

        # Return if above threshold
        if best_similarity >= similarity_threshold:
            return best_cluster

        return None
