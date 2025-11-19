"""
Semantic Ranking Engine for Search Results.

Uses sentence embeddings to rank search results by semantic similarity
to the query and context. Provides more accurate ranking than keyword matching.
"""

import hashlib
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SemanticRanker:
    """
    Semantic ranking engine using sentence embeddings.

    Features:
    - Uses SentenceTransformers for embedding generation
    - Cosine similarity for ranking
    - Context-aware ranking (query + context)
    - Near-duplicate detection using semantic similarity
    - Batch processing for efficiency
    - Embedding caching for performance

    Example:
        >>> ranker = SemanticRanker()
        >>> ranked = ranker.rank_results(query, results, context="SAP vendor management")
        >>> deduped = ranker.remove_semantic_duplicates(results, threshold=0.9)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = False,
        batch_size: int = 32,
        cache_size: int = 1000
    ):
        """
        Initialize semantic ranker.

        Args:
            model_name: Name of sentence transformer model
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for embedding computation
            cache_size: Size of embedding cache (LRU)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.model = None

        # Statistics
        self.stats = {
            "total_rankings": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_ranking_time_ms": 0.0
        }

        # Initialize model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize sentence transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers"
            )
            logger.warning("Semantic ranking will be disabled")
            return

        try:
            import torch
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"

            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(
                f"Semantic ranker initialized with model: {self.model_name} "
                f"on device: {device}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize semantic ranker: {e}")
            self.model = None

    def is_available(self) -> bool:
        """Check if semantic ranking is available."""
        return self.model is not None

    @lru_cache(maxsize=1000)
    def _get_embedding_cached(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector or None
        """
        if not self.model:
            return None

        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            self.stats["cache_misses"] += 1
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text (with caching).

        Args:
            text: Input text

        Returns:
            Embedding vector or None
        """
        # Try cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()

        # Use LRU cache
        return self._get_embedding_cached(text)

    def _get_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Get embeddings for multiple texts (batch processing).

        Args:
            texts: List of texts

        Returns:
            Matrix of embeddings or None
        """
        if not self.model:
            return None

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            return None

    def rank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank search results by semantic similarity to query.

        Args:
            query: Search query
            results: List of search results
            context: Optional context for contextual ranking
            top_k: Return only top K results (None = all)

        Returns:
            Ranked list of results with semantic_score added
        """
        if not self.is_available():
            logger.warning("Semantic ranking unavailable, returning original order")
            return results

        if not results:
            return []

        import time
        start_time = time.time()
        self.stats["total_rankings"] += 1

        try:
            # Construct query embedding (with context if provided)
            if context:
                query_text = f"{query} Context: {context}"
            else:
                query_text = query

            query_embedding = self._get_embedding(query_text)
            if query_embedding is None:
                return results

            # Generate embeddings for all results
            result_texts = []
            for result in results:
                # Combine title and snippet for better semantic understanding
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                result_text = f"{title}. {snippet}"
                result_texts.append(result_text)

            result_embeddings = self._get_embeddings_batch(result_texts)
            if result_embeddings is None:
                return results

            # Calculate cosine similarities
            similarities = self._cosine_similarity_batch(
                query_embedding,
                result_embeddings
            )

            # Add semantic scores to results
            for i, result in enumerate(results):
                result["semantic_score"] = float(similarities[i])

            # Sort by semantic score (descending)
            ranked_results = sorted(
                results,
                key=lambda x: x.get("semantic_score", 0.0),
                reverse=True
            )

            # Limit to top K if specified
            if top_k:
                ranked_results = ranked_results[:top_k]

            # Update statistics
            elapsed_ms = (time.time() - start_time) * 1000
            total = self.stats["total_rankings"]
            avg = self.stats["avg_ranking_time_ms"]
            self.stats["avg_ranking_time_ms"] = (avg * (total - 1) + elapsed_ms) / total

            logger.info(
                f"Ranked {len(results)} results in {elapsed_ms:.2f}ms. "
                f"Top score: {ranked_results[0].get('semantic_score', 0.0):.3f}"
            )

            return ranked_results

        except Exception as e:
            logger.error(f"Semantic ranking error: {e}")
            return results

    def remove_semantic_duplicates(
        self,
        results: List[Dict[str, Any]],
        threshold: float = 0.90,
        key_field: str = "snippet"
    ) -> List[Dict[str, Any]]:
        """
        Remove near-duplicate results using semantic similarity.

        Args:
            results: List of search results
            threshold: Similarity threshold (0-1) above which results are duplicates
            key_field: Field to use for comparison (snippet, title, etc.)

        Returns:
            Deduplicated results
        """
        if not self.is_available() or not results:
            return results

        try:
            # Extract texts for comparison
            texts = [result.get(key_field, "") for result in results]

            # Generate embeddings
            embeddings = self._get_embeddings_batch(texts)
            if embeddings is None:
                return results

            # Track unique results
            unique_results = []
            unique_embeddings = []

            for i, result in enumerate(results):
                is_duplicate = False
                current_embedding = embeddings[i]

                # Check against all unique results so far
                for unique_embedding in unique_embeddings:
                    similarity = self._cosine_similarity(
                        current_embedding,
                        unique_embedding
                    )

                    if similarity >= threshold:
                        is_duplicate = True
                        logger.debug(
                            f"Semantic duplicate found (similarity: {similarity:.3f}): "
                            f"{result.get('title', '')[:50]}"
                        )
                        break

                if not is_duplicate:
                    unique_results.append(result)
                    unique_embeddings.append(current_embedding)

            logger.info(
                f"Semantic deduplication: {len(results)} -> {len(unique_results)} "
                f"(removed {len(results) - len(unique_results)} duplicates)"
            )

            return unique_results

        except Exception as e:
            logger.error(f"Semantic deduplication error: {e}")
            return results

    def compute_result_diversity(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """
        Compute diversity score for search results (1.0 = very diverse, 0.0 = similar).

        Args:
            results: List of search results

        Returns:
            Diversity score between 0.0 and 1.0
        """
        if not self.is_available() or len(results) < 2:
            return 1.0

        try:
            # Generate embeddings
            texts = [
                f"{r.get('title', '')}. {r.get('snippet', '')}"
                for r in results
            ]
            embeddings = self._get_embeddings_batch(texts)
            if embeddings is None:
                return 1.0

            # Calculate pairwise similarities
            similarities = []
            n = len(embeddings)

            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)

            # Diversity = 1 - average similarity
            avg_similarity = np.mean(similarities)
            diversity = 1.0 - avg_similarity

            return float(diversity)

        except Exception as e:
            logger.error(f"Diversity computation error: {e}")
            return 1.0

    def get_most_relevant_snippets(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[str]:
        """
        Extract the most relevant snippets from results.

        Args:
            query: Search query
            results: List of search results
            top_k: Number of top snippets to return

        Returns:
            List of most relevant snippets
        """
        if not self.is_available() or not results:
            return [r.get("snippet", "") for r in results[:top_k]]

        try:
            # Rank by semantic similarity
            ranked = self.rank_results(query, results, top_k=top_k)

            # Extract snippets
            snippets = [r.get("snippet", "") for r in ranked[:top_k]]

            return snippets

        except Exception as e:
            logger.error(f"Snippet extraction error: {e}")
            return [r.get("snippet", "") for r in results[:top_k]]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (-1 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def _cosine_similarity_batch(
        query_vec: np.ndarray,
        result_vecs: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and multiple result vectors.

        Args:
            query_vec: Query vector (1D)
            result_vecs: Result vectors (2D matrix)

        Returns:
            Array of similarities
        """
        # Normalize query vector
        query_norm = query_vec / np.linalg.norm(query_vec)

        # Normalize result vectors
        result_norms = result_vecs / np.linalg.norm(
            result_vecs,
            axis=1,
            keepdims=True
        )

        # Compute dot products (cosine similarities)
        similarities = np.dot(result_norms, query_norm)

        return similarities

    def get_stats(self) -> Dict[str, Any]:
        """
        Get semantic ranker statistics.

        Returns:
            Statistics dictionary
        """
        cache_info = self._get_embedding_cached.cache_info()

        return {
            **self.stats,
            "model_name": self.model_name,
            "available": self.is_available(),
            "cache_size": cache_info.currsize,
            "cache_maxsize": cache_info.maxsize,
            "cache_hit_rate": (
                cache_info.hits / (cache_info.hits + cache_info.misses)
                if (cache_info.hits + cache_info.misses) > 0 else 0.0
            )
        }

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._get_embedding_cached.cache_clear()
        logger.info("Embedding cache cleared")
