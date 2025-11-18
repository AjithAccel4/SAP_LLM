"""
Context Retrieval System for Process Memory Graph.

Implements RAG (Retrieval-Augmented Generation) for SAP_LLM:
- Retrieves relevant historical context from PMG
- Finds similar documents and their processing outcomes
- Provides context-aware routing suggestions
- Learns from past successes and failures

Used to enhance SAP_LLM predictions with historical patterns.
"""

import logging
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np

# Import PMG components
from .graph_client import ProcessMemoryGraph
from .vector_store import PMGVectorStore
from .embedding_generator import EnhancedEmbeddingGenerator, EmbeddingConfig

logger = logging.getLogger(__name__)

# Try importing Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install with: pip install redis")


@dataclass
class RetrievalConfig:
    """Configuration for context retrieval."""
    top_k: int = 5
    min_similarity: float = 0.7
    similarity_metric: str = "cosine"
    time_window_days: Optional[int] = None  # None = no time restriction
    include_failures: bool = True
    include_successes: bool = True
    weight_by_recency: bool = True
    recency_decay_days: int = 30
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


@dataclass
class ContextResult:
    """Single context retrieval result."""
    doc_id: str
    document: Dict[str, Any]
    similarity: float
    routing_decision: Optional[Dict[str, Any]]
    sap_response: Optional[Dict[str, Any]]
    exceptions: List[Dict[str, Any]]
    success: bool
    timestamp: str
    recency_weight: float


class ContextRetriever:
    """
    Retrieves relevant processing context from PMG for RAG.

    Workflow:
    1. Embed new document
    2. Find similar documents via vector search
    3. Retrieve full context from graph (routing, outcomes, exceptions)
    4. Rank by relevance and success
    5. Return top-K contexts
    """

    def __init__(
        self,
        graph_client: Optional[ProcessMemoryGraph] = None,
        vector_store: Optional[PMGVectorStore] = None,
        embedding_generator: Optional[EnhancedEmbeddingGenerator] = None,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize context retriever.

        Args:
            graph_client: PMG graph client
            vector_store: Vector store for similarity search
            embedding_generator: Embedding generator
            config: Retrieval configuration
        """
        self.graph = graph_client or ProcessMemoryGraph()
        self.vector_store = vector_store or PMGVectorStore()
        self.embedding_gen = embedding_generator or EnhancedEmbeddingGenerator()
        self.config = config or RetrievalConfig()

        # Initialize Redis cache if enabled and available
        self.redis_client = None
        if self.config.enable_cache and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=False  # We'll handle encoding
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"Redis cache connected: {self.config.redis_host}:{self.config.redis_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Running without cache.")
                self.redis_client = None

        # Statistics
        self.stats = {
            "total_retrievals": 0,
            "avg_similarity": 0.0,
            "avg_results_per_query": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        logger.info("ContextRetriever initialized")

    def retrieve_context(
        self,
        document: Dict[str, Any],
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None
    ) -> List[ContextResult]:
        """
        Retrieve relevant processing context for document.

        Args:
            document: Input document to find context for
            top_k: Number of results (overrides config)
            min_similarity: Minimum similarity (overrides config)

        Returns:
            List of context results, ranked by relevance
        """
        top_k = top_k or self.config.top_k
        min_similarity = min_similarity or self.config.min_similarity

        logger.debug(f"Retrieving context (top_k={top_k}, min_sim={min_similarity})")

        # Check cache first
        cache_key = self._generate_cache_key(document, top_k, min_similarity)
        cached_result = self._get_from_cache(cache_key)

        if cached_result is not None:
            self.stats["cache_hits"] += 1
            logger.debug("Cache hit for context retrieval")
            return cached_result

        self.stats["cache_misses"] += 1

        # Step 1: Generate embedding for input document
        query_embedding = self.embedding_gen.generate_document_embedding(document)

        # Step 2: Vector similarity search
        similar_docs = self.vector_store.search_by_document(
            document=document,
            k=top_k * 2,  # Get extra results for filtering
            min_similarity=min_similarity
        )

        if not similar_docs:
            logger.debug("No similar documents found")
            return []

        # Step 3: Retrieve full context from graph
        contexts = []

        for similar_doc in similar_docs[:top_k]:
            # Get full context from graph
            context = self._retrieve_full_context(
                doc_id=similar_doc["doc_id"],
                document=similar_doc["document"],
                similarity=similar_doc["similarity"]
            )

            if context:
                contexts.append(context)

        # Step 4: Apply filters
        contexts = self._apply_filters(contexts)

        # Step 5: Rank and weight
        contexts = self._rank_contexts(contexts)

        # Update statistics
        self._update_stats(contexts)

        # Cache result
        self._save_to_cache(cache_key, contexts[:top_k])

        logger.info(f"Retrieved {len(contexts)} context results")

        return contexts[:top_k]

    def _retrieve_full_context(
        self,
        doc_id: str,
        document: Dict[str, Any],
        similarity: float
    ) -> Optional[ContextResult]:
        """
        Retrieve full processing context for document from graph.
        """
        try:
            # For now, extract from document metadata if available
            # In production, would query graph for:
            # - Routing decision
            # - SAP response
            # - Exceptions
            # - Processing outcome

            routing_decision = document.get("routing_decision")
            sap_response = document.get("sap_response")
            exceptions = document.get("exceptions", [])

            # Determine success
            success = self._determine_success(sap_response, exceptions)

            # Get timestamp
            timestamp = document.get("processing_timestamp") or \
                       document.get("ingestion_timestamp") or \
                       datetime.now().isoformat()

            # Calculate recency weight
            recency_weight = self._calculate_recency_weight(timestamp)

            return ContextResult(
                doc_id=doc_id,
                document=document,
                similarity=similarity,
                routing_decision=routing_decision,
                sap_response=sap_response,
                exceptions=exceptions,
                success=success,
                timestamp=timestamp,
                recency_weight=recency_weight
            )

        except Exception as e:
            logger.error(f"Error retrieving context for {doc_id}: {e}")
            return None

    def _determine_success(
        self,
        sap_response: Optional[Dict[str, Any]],
        exceptions: List[Dict[str, Any]]
    ) -> bool:
        """
        Determine if processing was successful.
        """
        # Check SAP response
        if sap_response:
            status_code = sap_response.get("status_code", 0)
            if status_code >= 200 and status_code < 300:
                # HTTP success
                return True

        # Check for critical exceptions
        if exceptions:
            for exc in exceptions:
                if exc.get("severity") in ["CRITICAL", "HIGH"]:
                    return False

        # Default: success if no response (might be in processing)
        return sap_response is None or len(exceptions) == 0

    def _calculate_recency_weight(self, timestamp: str) -> float:
        """
        Calculate weight based on document recency.

        Recent documents get higher weight if configured.
        """
        if not self.config.weight_by_recency:
            return 1.0

        try:
            doc_time = datetime.fromisoformat(timestamp)
            now = datetime.now()

            days_ago = (now - doc_time).days

            # Exponential decay
            decay_factor = np.exp(-days_ago / self.config.recency_decay_days)

            return float(decay_factor)

        except Exception as e:
            logger.warning(f"Error calculating recency weight: {e}")
            return 1.0

    def _apply_filters(self, contexts: List[ContextResult]) -> List[ContextResult]:
        """
        Apply configured filters to contexts.
        """
        filtered = []

        for context in contexts:
            # Filter by success/failure
            if context.success and not self.config.include_successes:
                continue
            if not context.success and not self.config.include_failures:
                continue

            # Filter by time window
            if self.config.time_window_days is not None:
                try:
                    doc_time = datetime.fromisoformat(context.timestamp)
                    cutoff = datetime.now() - timedelta(days=self.config.time_window_days)

                    if doc_time < cutoff:
                        continue
                except Exception:
                    pass

            filtered.append(context)

        return filtered

    def _rank_contexts(self, contexts: List[ContextResult]) -> List[ContextResult]:
        """
        Rank contexts by combined score.

        Score = similarity * recency_weight * success_boost
        """
        for context in contexts:
            # Base score from similarity
            score = context.similarity

            # Apply recency weight
            score *= context.recency_weight

            # Boost successful outcomes
            if context.success:
                score *= 1.2

            # Store score (add as attribute)
            context.metadata = {"score": score}

        # Sort by score (descending)
        contexts.sort(key=lambda c: c.metadata.get("score", 0), reverse=True)

        return contexts

    def retrieve_for_low_confidence(
        self,
        document: Dict[str, Any],
        confidence_score: float,
        field_name: Optional[str] = None
    ) -> List[ContextResult]:
        """
        Retrieve context specifically for low-confidence predictions.

        Focuses on similar documents where the same field was problematic.

        Args:
            document: Input document
            confidence_score: Current confidence score
            field_name: Specific field with low confidence

        Returns:
            Relevant contexts for improving prediction
        """
        logger.info(
            f"Retrieving context for low confidence: {confidence_score:.3f} "
            f"(field: {field_name})"
        )

        # Get general context
        contexts = self.retrieve_context(
            document=document,
            top_k=10,  # Get more results for low confidence
            min_similarity=0.6  # Lower threshold for more examples
        )

        # If specific field, filter for contexts with that field
        if field_name and contexts:
            field_contexts = []

            for context in contexts:
                # Check if this context has information about the field
                if field_name in context.document:
                    field_contexts.append(context)

                # Check if field appeared in exceptions
                for exc in context.exceptions:
                    if exc.get("field") == field_name:
                        field_contexts.append(context)
                        break

            if field_contexts:
                logger.debug(f"Found {len(field_contexts)} contexts for field {field_name}")
                return field_contexts

        return contexts

    def retrieve_vendor_patterns(
        self,
        vendor_id: str,
        doc_type: str,
        limit: int = 10
    ) -> List[ContextResult]:
        """
        Retrieve historical patterns for specific vendor.

        Useful for vendor-specific processing rules.

        Args:
            vendor_id: Vendor/supplier ID
            doc_type: Document type
            limit: Maximum results

        Returns:
            Vendor-specific processing patterns
        """
        logger.info(f"Retrieving patterns for vendor {vendor_id}, type {doc_type}")

        # Query graph for vendor documents
        if self.graph.mock_mode:
            logger.debug("Graph in mock mode, returning empty results")
            return []

        # In production, would query:
        # g.V().has('Document', 'supplier_id', vendor_id).has('doc_type', doc_type)

        vendor_docs = self.graph.find_similar_documents(
            doc_type=doc_type,
            supplier_id=vendor_id,
            limit=limit
        )

        # Convert to ContextResults
        contexts = []

        for doc in vendor_docs:
            context = self._retrieve_full_context(
                doc_id=doc.get("id", ""),
                document=doc,
                similarity=1.0  # Exact match
            )

            if context:
                contexts.append(context)

        logger.info(f"Found {len(contexts)} vendor patterns")

        return contexts

    def build_context_prompt(
        self,
        contexts: List[ContextResult],
        max_length: int = 2000
    ) -> str:
        """
        Build natural language context prompt for LLM.

        Summarizes retrieved contexts in format suitable for RAG.

        Args:
            contexts: Retrieved contexts
            max_length: Maximum prompt length

        Returns:
            Context prompt string
        """
        if not contexts:
            return "No similar historical documents found."

        prompt_parts = [
            "Based on similar historical documents:\n"
        ]

        for i, context in enumerate(contexts[:5], 1):  # Top 5 only
            doc = context.document

            # Document summary
            summary = f"{i}. Similar {doc.get('doc_type', 'document')}"

            if "supplier_name" in doc:
                summary += f" from {doc['supplier_name']}"

            if "total_amount" in doc:
                summary += f" for {doc['total_amount']} {doc.get('currency', 'USD')}"

            summary += f" (similarity: {context.similarity:.2f})"

            prompt_parts.append(summary)

            # Outcome
            if context.success:
                prompt_parts.append(f"   ✓ Successfully processed")

                if context.routing_decision:
                    endpoint = context.routing_decision.get("endpoint", "")
                    if endpoint:
                        prompt_parts.append(f"   → Routed to: {endpoint}")
            else:
                prompt_parts.append(f"   ✗ Processing failed")

                # Include exception details
                if context.exceptions:
                    for exc in context.exceptions[:2]:  # First 2 exceptions
                        msg = exc.get("message", "")
                        if msg:
                            prompt_parts.append(f"   ⚠ {msg}")

            prompt_parts.append("")  # Blank line

        prompt = "\n".join(prompt_parts)

        # Truncate if too long
        if len(prompt) > max_length:
            prompt = prompt[:max_length] + "..."

        return prompt

    def _update_stats(self, contexts: List[ContextResult]):
        """Update retrieval statistics."""
        self.stats["total_retrievals"] += 1

        if contexts:
            avg_sim = sum(c.similarity for c in contexts) / len(contexts)
            self.stats["avg_similarity"] = (
                (self.stats["avg_similarity"] * (self.stats["total_retrievals"] - 1) + avg_sim)
                / self.stats["total_retrievals"]
            )
            self.stats["avg_results_per_query"] = (
                (self.stats["avg_results_per_query"] * (self.stats["total_retrievals"] - 1) + len(contexts))
                / self.stats["total_retrievals"]
            )

    def _generate_cache_key(
        self,
        document: Dict[str, Any],
        top_k: int,
        min_similarity: float
    ) -> str:
        """
        Generate cache key for document retrieval.

        Args:
            document: Document dictionary
            top_k: Number of results
            min_similarity: Minimum similarity

        Returns:
            Cache key string
        """
        # Create a normalized representation of the document
        key_parts = [
            document.get('doc_type', ''),
            document.get('supplier_id', ''),
            document.get('company_code', ''),
            str(document.get('total_amount', 0)),
            str(top_k),
            str(min_similarity)
        ]

        # Hash to create compact key
        key_str = '|'.join(key_parts)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        return f"pmg:context:{key_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[List[ContextResult]]:
        """
        Get results from Redis cache.

        Args:
            cache_key: Cache key

        Returns:
            Cached results or None
        """
        if not self.redis_client:
            return None

        try:
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                # Deserialize
                data = json.loads(cached_data.decode('utf-8'))

                # Convert back to ContextResult objects
                results = []
                for item in data:
                    result = ContextResult(
                        doc_id=item['doc_id'],
                        document=item['document'],
                        similarity=item['similarity'],
                        routing_decision=item.get('routing_decision'),
                        sap_response=item.get('sap_response'),
                        exceptions=item.get('exceptions', []),
                        success=item['success'],
                        timestamp=item['timestamp'],
                        recency_weight=item['recency_weight']
                    )
                    results.append(result)

                return results

        except Exception as e:
            logger.warning(f"Cache get failed: {e}")

        return None

    def _save_to_cache(self, cache_key: str, results: List[ContextResult]) -> None:
        """
        Save results to Redis cache.

        Args:
            cache_key: Cache key
            results: Results to cache
        """
        if not self.redis_client:
            return

        try:
            # Serialize ContextResults
            data = [asdict(r) for r in results]

            # Save to Redis with TTL
            self.redis_client.setex(
                cache_key,
                self.config.cache_ttl_seconds,
                json.dumps(data).encode('utf-8')
            )

        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    def clear_cache(self) -> None:
        """Clear all cached results."""
        if not self.redis_client:
            logger.warning("No Redis client available")
            return

        try:
            # Delete all keys matching pattern
            for key in self.redis_client.scan_iter("pmg:context:*"):
                self.redis_client.delete(key)

            logger.info("Cache cleared")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = self.stats.copy()

        # Add cache hit rate
        total_requests = stats['cache_hits'] + stats['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_requests
        else:
            stats['cache_hit_rate'] = 0.0

        return stats

    def export_contexts(
        self,
        contexts: List[ContextResult],
        output_file: str
    ):
        """
        Export contexts to JSON file.

        Args:
            contexts: Contexts to export
            output_file: Output file path
        """
        import json
        from pathlib import Path

        data = {
            "total_contexts": len(contexts),
            "exported_at": datetime.now().isoformat(),
            "contexts": [asdict(c) for c in contexts]
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(contexts)} contexts to {output_file}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize retriever
    retriever = ContextRetriever()

    # Example document
    document = {
        "doc_type": "invoice",
        "supplier_name": "Acme Corp",
        "supplier_id": "SUP-001",
        "total_amount": 1000.00,
        "currency": "USD",
        "company_code": "1000"
    }

    # Retrieve context
    contexts = retriever.retrieve_context(document, top_k=5)

    print(f"Retrieved {len(contexts)} contexts")

    for context in contexts:
        print(f"\nDoc ID: {context.doc_id}")
        print(f"Similarity: {context.similarity:.3f}")
        print(f"Success: {context.success}")
        print(f"Recency weight: {context.recency_weight:.3f}")

    # Build prompt
    if contexts:
        prompt = retriever.build_context_prompt(contexts)
        print("\nContext Prompt:")
        print(prompt)

    # Get statistics
    stats = retriever.get_statistics()
    print(f"\nRetrieval stats: {stats}")
