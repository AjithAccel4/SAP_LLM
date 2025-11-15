"""
Enhanced Embedding Generation for Process Memory Graph.

Generates high-quality 768-dimensional embeddings for:
- Document semantic search
- Anomaly clustering
- Similar document retrieval
- Context-aware processing

Uses state-of-the-art embedding models optimized for business documents.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try importing embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"  # 768 dims
    dimension: int = 768
    batch_size: int = 32
    normalize: bool = True
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    max_seq_length: int = 512


class EnhancedEmbeddingGenerator:
    """
    Production-grade embedding generator for PMG.

    Features:
    - 768-dim embeddings for high quality
    - Batch processing for efficiency
    - GPU acceleration when available
    - Document-specific text extraction
    - Caching for performance
    - Multiple model support
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding generator.

        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()

        # Initialize model
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._load_model()
        else:
            logger.error("sentence-transformers not available. Embeddings will be random.")

        # Embedding cache
        self.cache: Dict[str, np.ndarray] = {}

        logger.info(
            f"EmbeddingGenerator initialized: "
            f"model={self.config.model_name}, "
            f"dim={self.config.dimension}, "
            f"device={self.config.device}"
        )

    def _load_model(self):
        """Load embedding model."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")

            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )

            # Set max sequence length
            self.model.max_seq_length = self.config.max_seq_length

            logger.info(f"Model loaded successfully on {self.config.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text
            use_cache: Whether to use cache

        Returns:
            768-dim embedding vector
        """
        # Check cache
        if use_cache and text in self.cache:
            return self.cache[text]

        # Generate embedding
        if self.model is not None:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False
            )
        else:
            # Fallback: random embedding
            embedding = np.random.randn(self.config.dimension).astype(np.float32)
            if self.config.normalize:
                embedding = embedding / np.linalg.norm(embedding)

        # Cache
        if use_cache:
            self.cache[text] = embedding

        return embedding

    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batched for efficiency).

        Args:
            texts: List of input texts
            batch_size: Batch size (default from config)
            show_progress: Show progress bar

        Returns:
            Array of embeddings (N x 768)
        """
        if not texts:
            return np.array([])

        batch_size = batch_size or self.config.batch_size

        if self.model is not None:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=show_progress
            )
        else:
            # Fallback: random embeddings
            embeddings = np.random.randn(len(texts), self.config.dimension).astype(np.float32)
            if self.config.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms

        return embeddings

    def generate_document_embedding(
        self,
        document: Dict[str, Any],
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for structured document.

        Extracts relevant fields and creates semantic representation.

        Args:
            document: Document dictionary
            use_cache: Whether to use cache

        Returns:
            768-dim embedding vector
        """
        # Extract text representation
        text = self._document_to_text(document)

        # Generate embedding
        return self.generate_embedding(text, use_cache=use_cache)

    def generate_document_batch_embeddings(
        self,
        documents: List[Dict[str, Any]],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple documents (batched).

        Args:
            documents: List of documents
            show_progress: Show progress bar

        Returns:
            Array of embeddings (N x 768)
        """
        # Extract text for all documents
        texts = [self._document_to_text(doc) for doc in documents]

        # Generate embeddings
        return self.generate_batch_embeddings(texts, show_progress=show_progress)

    def _document_to_text(self, document: Dict[str, Any]) -> str:
        """
        Convert structured document to text for embedding.

        Extracts key business fields in natural language format.

        Args:
            document: Document dictionary

        Returns:
            Text representation
        """
        parts = []

        # Document type and subtype
        if "doc_type" in document:
            parts.append(f"This is a {document['doc_type']}")

            if "doc_subtype" in document:
                parts.append(f"of type {document['doc_subtype']}")

        # Supplier/Vendor information
        if "supplier_name" in document:
            parts.append(f"from supplier {document['supplier_name']}")
        elif "vendor_name" in document:
            parts.append(f"from vendor {document['vendor_name']}")

        if "supplier_id" in document:
            parts.append(f"with ID {document['supplier_id']}")

        # Company information
        if "company_code" in document:
            parts.append(f"for company {document['company_code']}")

        # Monetary information
        if "total_amount" in document:
            currency = document.get("currency", "USD")
            parts.append(f"with total amount {document['total_amount']} {currency}")

        if "subtotal" in document:
            parts.append(f"subtotal {document['subtotal']}")

        if "tax_amount" in document:
            parts.append(f"tax {document['tax_amount']}")

        # Key identifiers
        for id_field in ["po_number", "invoice_number", "sales_order_number", "delivery_number"]:
            if id_field in document:
                field_name = id_field.replace("_", " ")
                parts.append(f"{field_name} {document[id_field]}")

        # Dates
        for date_field in ["invoice_date", "po_date", "delivery_date", "posting_date"]:
            if date_field in document:
                field_name = date_field.replace("_", " ")
                parts.append(f"{field_name} {document[date_field]}")

        # Line items (summarize)
        if "line_items" in document and isinstance(document["line_items"], list):
            num_items = len(document["line_items"])
            parts.append(f"containing {num_items} line items")

            # Include first few item descriptions
            for item in document["line_items"][:3]:
                if "description" in item:
                    parts.append(f"item: {item['description']}")

        # Processing context
        if "routing_decision" in document:
            endpoint = document["routing_decision"].get("endpoint", "")
            if endpoint:
                parts.append(f"routed to {endpoint}")

        # Exceptions
        if "exceptions" in document and isinstance(document["exceptions"], list):
            if document["exceptions"]:
                parts.append(f"with {len(document['exceptions'])} exceptions")

                for exc in document["exceptions"][:2]:
                    if "message" in exc:
                        parts.append(f"exception: {exc['message']}")

        # Join all parts
        text = ". ".join(parts) + "."

        # Truncate if too long
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        return text

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine', 'euclidean', 'dot')

        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1 / (1 + distance)
            return float(similarity)

        elif metric == "dot":
            # Dot product (for normalized embeddings)
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def find_similar_embeddings(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.0,
        metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding (768-dim)
            embeddings: Array of embeddings to search (N x 768)
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            metric: Similarity metric

        Returns:
            List of results with indices and similarities
        """
        if len(embeddings) == 0:
            return []

        # Compute similarities
        similarities = np.array([
            self.compute_similarity(query_embedding, emb, metric)
            for emb in embeddings
        ])

        # Filter by minimum similarity
        valid_indices = np.where(similarities >= min_similarity)[0]

        if len(valid_indices) == 0:
            return []

        # Sort by similarity (descending)
        sorted_indices = valid_indices[np.argsort(-similarities[valid_indices])]

        # Take top K
        top_indices = sorted_indices[:top_k]

        # Build results
        results = [
            {
                "index": int(idx),
                "similarity": float(similarities[idx])
            }
            for idx in top_indices
        ]

        return results

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self.cache),
            "cache_memory_mb": sum(
                emb.nbytes for emb in self.cache.values()
            ) / (1024 * 1024)
        }

    def export_embeddings(
        self,
        embeddings: np.ndarray,
        output_file: str,
        format: str = "npy"
    ):
        """
        Export embeddings to file.

        Args:
            embeddings: Embeddings array
            output_file: Output file path
            format: Format ('npy', 'npz', 'txt')
        """
        if format == "npy":
            np.save(output_file, embeddings)
        elif format == "npz":
            np.savez_compressed(output_file, embeddings=embeddings)
        elif format == "txt":
            np.savetxt(output_file, embeddings)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Exported {len(embeddings)} embeddings to {output_file}")

    def load_embeddings(
        self,
        input_file: str,
        format: str = "npy"
    ) -> np.ndarray:
        """
        Load embeddings from file.

        Args:
            input_file: Input file path
            format: Format ('npy', 'npz', 'txt')

        Returns:
            Embeddings array
        """
        if format == "npy":
            embeddings = np.load(input_file)
        elif format == "npz":
            data = np.load(input_file)
            embeddings = data["embeddings"]
        elif format == "txt":
            embeddings = np.loadtxt(input_file)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Loaded {len(embeddings)} embeddings from {input_file}")

        return embeddings


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize generator
    generator = EnhancedEmbeddingGenerator()

    # Generate embedding for single text
    text = "This is an invoice from Acme Corp for $1000"
    embedding = generator.generate_embedding(text)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 5): {embedding[:5]}")

    # Generate embeddings for documents
    documents = [
        {
            "doc_type": "invoice",
            "supplier_name": "Acme Corp",
            "total_amount": 1000.00,
            "currency": "USD"
        },
        {
            "doc_type": "purchase_order",
            "supplier_name": "Tech Supplies Inc",
            "total_amount": 5000.00,
            "currency": "EUR"
        }
    ]

    doc_embeddings = generator.generate_document_batch_embeddings(documents, show_progress=True)
    print(f"Generated {len(doc_embeddings)} document embeddings")

    # Compute similarity
    similarity = generator.compute_similarity(doc_embeddings[0], doc_embeddings[1])
    print(f"Similarity: {similarity:.4f}")

    # Cache stats
    stats = generator.get_cache_stats()
    print(f"Cache stats: {stats}")
