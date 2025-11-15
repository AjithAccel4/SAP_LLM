"""
PMG Vector Store - Semantic Search over Documents

Uses sentence embeddings and FAISS for fast similarity search
over historical documents, exceptions, and routing decisions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class PMGVectorStore:
    """
    Vector store for semantic search over PMG data.

    Uses:
    - SentenceTransformers for embedding generation
    - FAISS for efficient similarity search
    - Local file storage for persistence
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        index_type: str = "IndexFlatL2",
        storage_path: Optional[str] = None,
    ):
        """
        Initialize vector store.

        Args:
            embedding_model: SentenceTransformer model name
            dimension: Embedding dimension
            index_type: FAISS index type (IndexFlatL2, IndexIVFFlat, etc.)
            storage_path: Path to store index and metadata
        """
        self.embedding_model_name = embedding_model
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path) if storage_path else None

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize FAISS index
        self.index = self._create_index()

        # Metadata storage (maps index position to document data)
        self.metadata: List[Dict[str, Any]] = []

        # Load existing index if available
        if self.storage_path and self.storage_path.exists():
            self.load()

        logger.info(f"Vector store initialized (dimension={dimension})")

    def _create_index(self) -> faiss.Index:
        """Create FAISS index."""
        if self.index_type == "IndexFlatL2":
            # Exact search with L2 distance
            return faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            # Exact search with inner product
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            # Inverted file index (faster but approximate)
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            logger.warning(f"Unknown index type: {self.index_type}, using IndexFlatL2")
            return faiss.IndexFlatL2(self.dimension)

    def add_document(
        self,
        doc_id: str,
        document: Dict[str, Any],
    ) -> None:
        """
        Add document to vector store.

        Args:
            doc_id: Document ID
            document: Document data
        """
        # Create text representation
        text = self._document_to_text(document)

        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)

        # Add to index
        self.index.add(np.array([embedding]).astype("float32"))

        # Store metadata
        self.metadata.append({
            "doc_id": doc_id,
            "document": document,
            "text": text,
        })

        logger.debug(f"Added document to vector store: {doc_id}")

    def add_batch(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """
        Add multiple documents in batch.

        Args:
            documents: List of (doc_id, document) tuples
        """
        texts = []
        for doc_id, doc in documents:
            text = self._document_to_text(doc)
            texts.append(text)

            self.metadata.append({
                "doc_id": doc_id,
                "document": doc,
                "text": text,
            })

        # Generate embeddings in batch
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        # Add to index
        self.index.add(embeddings.astype("float32"))

        logger.info(f"Added {len(documents)} documents to vector store")

    def search(
        self,
        query: str,
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            k: Number of results
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of results with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
        )

        # Search
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"),
            k,
        )

        # Convert distances to similarity scores (for L2 distance)
        # similarity = 1 / (1 + distance)
        similarities = 1 / (1 + distances[0])

        # Filter by minimum similarity
        results = []
        for idx, similarity in zip(indices[0], similarities):
            if idx < len(self.metadata) and similarity >= min_similarity:
                result = self.metadata[idx].copy()
                result["similarity"] = float(similarity)
                results.append(result)

        return results

    def search_by_document(
        self,
        document: Dict[str, Any],
        k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to a given document.

        Args:
            document: Document to find similar documents for
            k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar documents
        """
        text = self._document_to_text(document)
        return self.search(text, k, min_similarity)

    def _document_to_text(self, document: Dict[str, Any]) -> str:
        """
        Convert document to searchable text.

        Args:
            document: Document data

        Returns:
            Text representation
        """
        parts = []

        # Document type
        if "doc_type" in document:
            parts.append(f"Type: {document['doc_type']}")

        if "doc_subtype" in document:
            parts.append(f"Subtype: {document['doc_subtype']}")

        # Supplier/Vendor
        if "supplier_name" in document:
            parts.append(f"Supplier: {document['supplier_name']}")
        elif "vendor_name" in document:
            parts.append(f"Vendor: {document['vendor_name']}")

        if "supplier_id" in document:
            parts.append(f"Supplier ID: {document['supplier_id']}")

        # Company code
        if "company_code" in document:
            parts.append(f"Company: {document['company_code']}")

        # Monetary values
        if "total_amount" in document:
            currency = document.get("currency", "USD")
            parts.append(f"Amount: {document['total_amount']} {currency}")

        # Key identifiers
        for field in ["po_number", "invoice_number", "sales_order_number"]:
            if field in document:
                parts.append(f"{field}: {document[field]}")

        # Description if available
        if "description" in document:
            parts.append(f"Description: {document['description']}")

        return " | ".join(parts)

    def save(self, path: Optional[str] = None) -> None:
        """
        Save index and metadata to disk.

        Args:
            path: Optional path (uses self.storage_path if not provided)
        """
        save_path = Path(path) if path else self.storage_path

        if save_path is None:
            raise ValueError("No storage path specified")

        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata - SECURITY: Use JSON instead of pickle to prevent RCE vulnerabilities
        metadata_path = save_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        # Save config
        config_path = save_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({
                "embedding_model": self.embedding_model_name,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "num_documents": len(self.metadata),
            }, f, indent=2)

        logger.info(f"Vector store saved to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """
        Load index and metadata from disk.

        Args:
            path: Optional path (uses self.storage_path if not provided)
        """
        load_path = Path(path) if path else self.storage_path

        if load_path is None or not load_path.exists():
            logger.warning("No saved vector store found")
            return

        try:
            # Load FAISS index
            index_path = load_path / "index.faiss"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))

            # Load metadata - SECURITY: Use JSON instead of pickle
            # Try new JSON format first, fall back to old pickle format for backwards compatibility
            metadata_json_path = load_path / "metadata.json"
            metadata_pkl_path = load_path / "metadata.pkl"

            if metadata_json_path.exists():
                with open(metadata_json_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            elif metadata_pkl_path.exists():
                # Legacy support - warn about pickle usage
                logger.warning(
                    "Loading metadata from pickle file (insecure). "
                    "Please re-save to convert to JSON format."
                )
                import pickle
                with open(metadata_pkl_path, "rb") as f:
                    self.metadata = pickle.load(f)

            logger.info(f"Vector store loaded from {load_path} ({len(self.metadata)} documents)")

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")

    def size(self) -> int:
        """Get number of documents in store."""
        return len(self.metadata)

    def clear(self) -> None:
        """Clear all data from store."""
        self.index = self._create_index()
        self.metadata = []
        logger.info("Vector store cleared")
