"""
SHWL Phase 2: Pattern Clustering

Clusters anomalies using DBSCAN/HDBSCAN on embeddings:
- Embeds anomalies using semantic similarity
- Groups similar anomalies into clusters
- Identifies common failure patterns
- Ranks clusters by size and severity

Uses density-based clustering to find natural groupings.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import Counter, defaultdict

# Import clustering libraries
try:
    from sklearn.cluster import DBSCAN, HDBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Install with: pip install scikit-learn hdbscan")

# Import PMG embedding generator
from sap_llm.pmg.embedding_generator import EnhancedEmbeddingGenerator
from sap_llm.shwl.anomaly_detector import Anomaly

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for pattern clustering."""
    algorithm: str = "hdbscan"  # hdbscan or dbscan
    min_cluster_size: int = 15
    min_samples: int = 5
    eps: float = 0.3  # For DBSCAN
    metric: str = "euclidean"
    min_severity: str = "MEDIUM"


@dataclass
class AnomalyCluster:
    """Represents a cluster of similar anomalies."""
    cluster_id: str
    size: int
    anomalies: List[Anomaly]
    centroid_embedding: np.ndarray
    representative_anomaly: Anomaly
    common_features: Dict[str, Any]
    severity: str
    confidence: float
    timestamp: str


class PatternClusterer:
    """
    SHWL Phase 2: Cluster anomalies using density-based clustering.

    Workflow:
    1. Generate embeddings for anomalies
    2. Apply DBSCAN/HDBSCAN clustering
    3. Identify cluster patterns
    4. Rank by size and severity
    """

    def __init__(
        self,
        embedding_generator: Optional[EnhancedEmbeddingGenerator] = None,
        config: Optional[ClusterConfig] = None
    ):
        """
        Initialize pattern clusterer.

        Args:
            embedding_generator: Embedding generator
            config: Clustering configuration
        """
        self.embedding_gen = embedding_generator or EnhancedEmbeddingGenerator()
        self.config = config or ClusterConfig()

        # Check clustering library availability
        if not SKLEARN_AVAILABLE:
            logger.error("sklearn not available. Clustering will not work.")

        # Statistics
        self.stats = {
            "total_anomalies": 0,
            "total_clusters": 0,
            "noise_points": 0,
            "largest_cluster_size": 0,
            "avg_cluster_size": 0
        }

        logger.info(
            f"PatternClusterer initialized "
            f"(algorithm={self.config.algorithm}, "
            f"min_cluster_size={self.config.min_cluster_size})"
        )

    def cluster_anomalies(self, anomalies: List[Anomaly]) -> List[AnomalyCluster]:
        """
        Cluster anomalies using configured algorithm.

        Args:
            anomalies: List of anomalies to cluster

        Returns:
            List of anomaly clusters
        """
        if not anomalies:
            logger.warning("No anomalies to cluster")
            return []

        if not SKLEARN_AVAILABLE:
            logger.error("sklearn not available. Cannot cluster.")
            return []

        logger.info(f"Clustering {len(anomalies)} anomalies...")

        # Step 1: Generate embeddings
        embeddings = self._generate_embeddings(anomalies)

        # Step 2: Perform clustering
        labels = self._perform_clustering(embeddings)

        # Step 3: Build clusters
        clusters = self._build_clusters(anomalies, embeddings, labels)

        # Step 4: Filter and rank
        significant_clusters = self._filter_clusters(clusters)
        ranked_clusters = self._rank_clusters(significant_clusters)

        # Update statistics
        self._update_stats(anomalies, ranked_clusters, labels)

        logger.info(
            f"Clustering complete: {len(ranked_clusters)} clusters "
            f"(noise: {self.stats['noise_points']})"
        )

        return ranked_clusters

    def _generate_embeddings(self, anomalies: List[Anomaly]) -> np.ndarray:
        """
        Generate embeddings for anomalies.

        Converts anomalies to text and embeds them.
        """
        logger.debug(f"Generating embeddings for {len(anomalies)} anomalies...")

        # Convert anomalies to text
        texts = [self._anomaly_to_text(anomaly) for anomaly in anomalies]

        # Generate embeddings
        embeddings = self.embedding_gen.generate_batch_embeddings(
            texts,
            show_progress=False
        )

        logger.debug(f"Generated embeddings shape: {embeddings.shape}")

        return embeddings

    def _anomaly_to_text(self, anomaly: Anomaly) -> str:
        """
        Convert anomaly to text for embedding.
        """
        parts = []

        # Anomaly type
        parts.append(f"Anomaly type: {anomaly.anomaly_type}")

        # Document type
        doc_type = anomaly.document.get('doc_type', 'unknown')
        parts.append(f"Document type: {doc_type}")

        # Error message
        if anomaly.error_message:
            parts.append(f"Error: {anomaly.error_message}")

        # Field name
        if anomaly.field_name:
            parts.append(f"Field: {anomaly.field_name}")

        # Confidence
        if anomaly.confidence_score is not None:
            parts.append(f"Confidence: {anomaly.confidence_score:.3f}")

        # SAP response
        if anomaly.sap_response:
            status_code = anomaly.sap_response.get('status_code')
            if status_code:
                parts.append(f"SAP status: {status_code}")

        # Severity
        parts.append(f"Severity: {anomaly.severity}")

        return ". ".join(parts)

    def _perform_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Perform density-based clustering on embeddings.

        Returns:
            Cluster labels (-1 for noise)
        """
        if self.config.algorithm == "hdbscan":
            try:
                # HDBSCAN (hierarchical density-based clustering)
                clusterer = HDBSCAN(
                    min_cluster_size=self.config.min_cluster_size,
                    min_samples=self.config.min_samples,
                    metric=self.config.metric,
                    cluster_selection_method='eom'  # Excess of Mass
                )
                labels = clusterer.fit_predict(embeddings)

            except NameError:
                # Fallback to DBSCAN if HDBSCAN not available
                logger.warning("HDBSCAN not available, using DBSCAN")
                labels = self._dbscan_clustering(embeddings)

        elif self.config.algorithm == "dbscan":
            labels = self._dbscan_clustering(embeddings)

        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(f"Clustering found {n_clusters} clusters, {n_noise} noise points")

        return labels

    def _dbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering."""
        clusterer = DBSCAN(
            eps=self.config.eps,
            min_samples=self.config.min_samples,
            metric=self.config.metric
        )
        labels = clusterer.fit_predict(embeddings)
        return labels

    def _build_clusters(
        self,
        anomalies: List[Anomaly],
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> List[AnomalyCluster]:
        """
        Build cluster objects from labels.
        """
        from datetime import datetime

        clusters = []

        # Group by label (skip noise = -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        for label in unique_labels:
            # Get anomalies in this cluster
            indices = np.where(labels == label)[0]
            cluster_anomalies = [anomalies[i] for i in indices]
            cluster_embeddings = embeddings[indices]

            # Compute centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Find representative anomaly (closest to centroid)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            representative_idx = indices[np.argmin(distances)]
            representative = anomalies[representative_idx]

            # Extract common features
            common_features = self._extract_common_features(cluster_anomalies)

            # Determine cluster severity (highest in cluster)
            severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            cluster_severity = max(
                (a.severity for a in cluster_anomalies),
                key=lambda s: severity_order.get(s, 0)
            )

            # Compute confidence (cohesiveness)
            confidence = self._compute_cluster_confidence(cluster_embeddings, centroid)

            cluster = AnomalyCluster(
                cluster_id=f"CLUSTER-{int(label):03d}",
                size=len(cluster_anomalies),
                anomalies=cluster_anomalies,
                centroid_embedding=centroid,
                representative_anomaly=representative,
                common_features=common_features,
                severity=cluster_severity,
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            )

            clusters.append(cluster)

        return clusters

    def _extract_common_features(self, anomalies: List[Anomaly]) -> Dict[str, Any]:
        """
        Extract common features from cluster anomalies.
        """
        # Count anomaly types
        anomaly_types = Counter(a.anomaly_type for a in anomalies)
        most_common_type = anomaly_types.most_common(1)[0]

        # Count document types
        doc_types = Counter(a.document.get('doc_type', 'unknown') for a in anomalies)
        most_common_doc_type = doc_types.most_common(1)[0] if doc_types else ('unknown', 0)

        # Count error messages (top 3)
        error_messages = Counter(
            a.error_message for a in anomalies if a.error_message
        )
        top_errors = error_messages.most_common(3)

        # Count affected fields
        fields = Counter(
            a.field_name for a in anomalies if a.field_name
        )
        top_fields = fields.most_common(3)

        return {
            "primary_anomaly_type": most_common_type[0],
            "anomaly_type_coverage": most_common_type[1] / len(anomalies),
            "primary_doc_type": most_common_doc_type[0],
            "doc_type_coverage": most_common_doc_type[1] / len(anomalies),
            "top_error_messages": [msg for msg, count in top_errors],
            "top_affected_fields": [field for field, count in top_fields]
        }

    def _compute_cluster_confidence(
        self,
        embeddings: np.ndarray,
        centroid: np.ndarray
    ) -> float:
        """
        Compute cluster confidence based on cohesiveness.

        Uses average distance to centroid (lower = higher confidence).
        """
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        avg_distance = np.mean(distances)

        # Convert to confidence score (0-1)
        # Assuming max meaningful distance is ~2.0 for normalized embeddings
        confidence = max(0.0, 1.0 - (avg_distance / 2.0))

        return float(confidence)

    def _filter_clusters(self, clusters: List[AnomalyCluster]) -> List[AnomalyCluster]:
        """
        Filter clusters by size and severity.
        """
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        min_severity_value = severity_order.get(self.config.min_severity, 2)

        filtered = [
            c for c in clusters
            if c.size >= self.config.min_cluster_size
            and severity_order.get(c.severity, 0) >= min_severity_value
        ]

        logger.debug(
            f"Filtered {len(clusters)} -> {len(filtered)} clusters "
            f"(min_size={self.config.min_cluster_size}, min_severity={self.config.min_severity})"
        )

        return filtered

    def _rank_clusters(self, clusters: List[AnomalyCluster]) -> List[AnomalyCluster]:
        """
        Rank clusters by severity and size.
        """
        severity_weights = {"CRITICAL": 100, "HIGH": 50, "MEDIUM": 25, "LOW": 10}

        def cluster_score(cluster: AnomalyCluster) -> float:
            severity_weight = severity_weights.get(cluster.severity, 1)
            return cluster.size * severity_weight

        ranked = sorted(clusters, key=cluster_score, reverse=True)

        return ranked

    def _update_stats(
        self,
        anomalies: List[Anomaly],
        clusters: List[AnomalyCluster],
        labels: np.ndarray
    ):
        """Update clustering statistics."""
        self.stats["total_anomalies"] = len(anomalies)
        self.stats["total_clusters"] = len(clusters)
        self.stats["noise_points"] = list(labels).count(-1)

        if clusters:
            cluster_sizes = [c.size for c in clusters]
            self.stats["largest_cluster_size"] = max(cluster_sizes)
            self.stats["avg_cluster_size"] = sum(cluster_sizes) / len(cluster_sizes)

    def get_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        return self.stats.copy()

    def visualize_clusters(
        self,
        anomalies: List[Anomaly],
        embeddings: np.ndarray,
        labels: np.ndarray,
        output_file: str
    ):
        """
        Visualize clusters using t-SNE or UMAP.

        Args:
            anomalies: Anomalies
            embeddings: Embeddings
            labels: Cluster labels
            output_file: Output image file
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt

            logger.info("Generating cluster visualization...")

            # Reduce to 2D using t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)

            # Plot
            plt.figure(figsize=(12, 8))

            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                if label == -1:
                    # Noise points
                    color = 'gray'

                indices = labels == label
                plt.scatter(
                    embeddings_2d[indices, 0],
                    embeddings_2d[indices, 1],
                    c=[color],
                    label=f"Cluster {label}" if label != -1 else "Noise",
                    alpha=0.6,
                    s=50
                )

            plt.title("Anomaly Clusters (t-SNE)")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_file, dpi=300)

            logger.info(f"Visualization saved to {output_file}")

        except ImportError as e:
            logger.warning(f"Cannot visualize: {e}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Mock anomalies for testing
    from sap_llm.shwl.anomaly_detector import AnomalyDetector
    from sap_llm.pmg.graph_client import ProcessMemoryGraph

    pmg = ProcessMemoryGraph()
    detector = AnomalyDetector(pmg)
    anomalies = detector.detect_anomalies()

    # Cluster anomalies
    clusterer = PatternClusterer()
    clusters = clusterer.cluster_anomalies(anomalies)

    print(f"Found {len(clusters)} clusters:")
    for cluster in clusters:
        print(f"\nCluster {cluster.cluster_id}:")
        print(f"  Size: {cluster.size}")
        print(f"  Severity: {cluster.severity}")
        print(f"  Confidence: {cluster.confidence:.3f}")
        print(f"  Primary type: {cluster.common_features['primary_anomaly_type']}")
        print(f"  Representative: {cluster.representative_anomaly.error_message}")

    stats = clusterer.get_statistics()
    print(f"\nClustering stats: {stats}")
