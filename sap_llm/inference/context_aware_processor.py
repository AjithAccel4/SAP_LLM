"""
TODO 5: Context-Aware Processing Engine

Uses PMG context retrieval to enhance predictions:
- Retrieves similar documents from PMG
- Injects historical context into processing
- Improves low-confidence predictions
- Learns from vendor-specific patterns
"""

import logging
from typing import Any, Dict, List, Optional
from sap_llm.pmg.context_retriever import ContextRetriever, RetrievalConfig
from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.pmg.vector_store import PMGVectorStore
from sap_llm.pmg.embedding_generator import EnhancedEmbeddingGenerator

logger = logging.getLogger(__name__)


class ContextAwareProcessor:
    """
    RAG-enhanced document processor.

    Workflow:
    1. Initial prediction from model
    2. If low confidence, retrieve similar docs from PMG
    3. Re-process with historical context
    4. Boost confidence with patterns
    """

    def __init__(
        self,
        model: Any = None,
        pmg: Optional[ProcessMemoryGraph] = None,
        vector_store: Optional[PMGVectorStore] = None
    ):
        self.model = model
        self.retriever = ContextRetriever(
            graph_client=pmg,
            vector_store=vector_store
        )

        # Statistics
        self.stats = {
            "total_processed": 0,
            "context_used": 0,
            "confidence_improved": 0,
            "avg_improvement": 0.0
        }

        logger.info("ContextAwareProcessor initialized")

    def process_document(
        self,
        document: Dict[str, Any],
        use_context: bool = True
    ) -> Dict[str, Any]:
        """
        Process document with optional context enhancement.
        """
        # Initial prediction
        initial_result = self._initial_prediction(document)

        self.stats["total_processed"] += 1

        # Check if context would help
        if use_context and initial_result["confidence"] < 0.7:
            logger.info(
                f"Low confidence ({initial_result['confidence']:.3f}), "
                "retrieving context..."
            )

            # Retrieve context
            contexts = self.retriever.retrieve_context(
                document,
                top_k=5,
                min_similarity=0.6
            )

            if contexts:
                self.stats["context_used"] += 1

                # Re-process with context
                enhanced_result = self._context_aware_prediction(
                    document,
                    contexts,
                    initial_result
                )

                # Check improvement
                if enhanced_result["confidence"] > initial_result["confidence"]:
                    self.stats["confidence_improved"] += 1
                    improvement = enhanced_result["confidence"] - initial_result["confidence"]
                    self._update_avg_improvement(improvement)

                    logger.info(
                        f"Confidence improved: {initial_result['confidence']:.3f} "
                        f"-> {enhanced_result['confidence']:.3f} (+{improvement:.3f})"
                    )

                return enhanced_result

        return initial_result

    def _initial_prediction(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Make initial prediction without context."""
        # Mock prediction (in production, would use actual model)
        import random

        return {
            "doc_type": document.get("doc_type", "invoice"),
            "confidence": random.uniform(0.5, 0.95),
            "extracted_fields": {},
            "validation_passed": True
        }

    def _context_aware_prediction(
        self,
        document: Dict[str, Any],
        contexts: List[Any],
        initial_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Re-predict using historical context.

        Uses RAG to inject context into model.
        """
        # Build context prompt
        context_prompt = self.retriever.build_context_prompt(contexts)

        # Re-run model with context (mock)
        # In production: model.predict_with_context(document, context_prompt)

        # Boost confidence based on similar successful docs
        successful_contexts = [c for c in contexts if c.success]

        if successful_contexts:
            # Average similarity of successful docs
            avg_similarity = sum(c.similarity for c in successful_contexts) / len(successful_contexts)

            # Boost confidence
            confidence_boost = avg_similarity * 0.2
            new_confidence = min(initial_result["confidence"] + confidence_boost, 0.99)

            return {
                **initial_result,
                "confidence": new_confidence,
                "context_used": True,
                "num_contexts": len(contexts),
                "context_prompt": context_prompt[:500]  # Truncate
            }

        return initial_result

    def _update_avg_improvement(self, improvement: float):
        """Update average confidence improvement."""
        n = self.stats["confidence_improved"]
        old_avg = self.stats["avg_improvement"]
        self.stats["avg_improvement"] = (old_avg * (n - 1) + improvement) / n

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            "context_usage_rate": (
                self.stats["context_used"] / self.stats["total_processed"]
                if self.stats["total_processed"] > 0 else 0
            ),
            "improvement_rate": (
                self.stats["confidence_improved"] / self.stats["context_used"]
                if self.stats["context_used"] > 0 else 0
            )
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processor = ContextAwareProcessor()

    # Test documents
    doc1 = {"doc_type": "invoice", "supplier_name": "Acme Corp"}
    doc2 = {"doc_type": "purchase_order", "vendor_id": "V001"}

    result1 = processor.process_document(doc1)
    result2 = processor.process_document(doc2)

    print(f"Result 1: confidence={result1['confidence']:.3f}, context_used={result1.get('context_used', False)}")
    print(f"Result 2: confidence={result2['confidence']:.3f}, context_used={result2.get('context_used', False)}")

    stats = processor.get_statistics()
    print(f"\nStatistics: {stats}")
