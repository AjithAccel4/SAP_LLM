"""
Knowledge Base Storage

Stores and manages SAP API knowledge including:
- API schemas
- Field mappings
- Business rules
- Transformation functions
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from pymongo import MongoClient, ASCENDING
from sentence_transformers import SentenceTransformer

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBaseStorage:
    """
    Stores and retrieves SAP API knowledge.

    Uses:
    - MongoDB for structured storage
    - FAISS for semantic vector search
    - File system for large data
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        database_name: str = "sap_llm_kb",
        embedding_model: str = "all-MiniLM-L6-v2",
        storage_path: Optional[str] = None,
    ):
        """
        Initialize knowledge base storage.

        Args:
            mongo_uri: MongoDB connection URI
            database_name: MongoDB database name
            embedding_model: Model for embeddings
            storage_path: Path for file storage
        """
        # MongoDB connection
        if mongo_uri:
            self.mongo_client = MongoClient(mongo_uri)
            self.db = self.mongo_client[database_name]
            self.mock_mode = False
            logger.info(f"Connected to MongoDB: {database_name}")
        else:
            self.mongo_client = None
            self.db = None
            self.mock_mode = True
            logger.warning("MongoDB not configured, using mock mode")

        # Collections
        if not self.mock_mode:
            self.api_schemas_col = self.db["api_schemas"]
            self.field_mappings_col = self.db["field_mappings"]
            self.business_rules_col = self.db["business_rules"]
            self.examples_col = self.db["examples"]

            # Create indexes
            self._create_indexes()

        # File storage
        self.storage_path = Path(storage_path or "data/knowledge_base")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # FAISS indexes
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.api_index: Optional[faiss.IndexFlatIP] = None
        self.field_index: Optional[faiss.IndexFlatIP] = None
        self.rule_index: Optional[faiss.IndexFlatIP] = None

        # Metadata for FAISS
        self.api_metadata: List[Dict[str, Any]] = []
        self.field_metadata: List[Dict[str, Any]] = []
        self.rule_metadata: List[Dict[str, Any]] = []

        # Initialize indexes
        self._initialize_indexes()

        logger.info("Knowledge Base Storage initialized")

    def _create_indexes(self):
        """Create MongoDB indexes for performance."""
        # API schemas
        self.api_schemas_col.create_index([("title", ASCENDING)])
        self.api_schemas_col.create_index([("type", ASCENDING)])

        # Field mappings
        self.field_mappings_col.create_index([("field_name", ASCENDING)])
        self.field_mappings_col.create_index([("sap_field", ASCENDING)])

        # Business rules
        self.business_rules_col.create_index([("rule_id", ASCENDING)])
        self.business_rules_col.create_index([("type", ASCENDING)])

        logger.debug("MongoDB indexes created")

    def _initialize_indexes(self):
        """Initialize FAISS indexes."""
        self.api_index = faiss.IndexFlatIP(self.embedding_dim)
        self.field_index = faiss.IndexFlatIP(self.embedding_dim)
        self.rule_index = faiss.IndexFlatIP(self.embedding_dim)

        logger.debug("FAISS indexes initialized")

    def store_crawled_data(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
        """
        Store crawled data.

        Args:
            data: Dictionary with api_schemas, field_mappings, business_rules

        Returns:
            Count of stored items
        """
        counts = {
            "api_schemas": 0,
            "field_mappings": 0,
            "business_rules": 0,
            "examples": 0,
        }

        # Store API schemas
        for schema in data.get("api_schemas", []):
            self.store_api_schema(schema)
            counts["api_schemas"] += 1

        # Store field mappings
        for mapping in data.get("field_mappings", []):
            self.store_field_mapping(mapping)
            counts["field_mappings"] += 1

        # Store business rules
        for rule in data.get("business_rules", []):
            self.store_business_rule(rule)
            counts["business_rules"] += 1

        # Store examples
        for example in data.get("examples", []):
            self.store_example(example)
            counts["examples"] += 1

        logger.info(
            f"Stored data: {counts['api_schemas']} APIs, "
            f"{counts['field_mappings']} mappings, "
            f"{counts['business_rules']} rules"
        )

        # Save indexes
        self.save_indexes()

        return counts

    def store_api_schema(self, schema: Dict[str, Any]) -> str:
        """
        Store API schema.

        Args:
            schema: API schema dictionary

        Returns:
            Schema ID
        """
        if not self.mock_mode:
            result = self.api_schemas_col.insert_one(schema)
            schema_id = str(result.inserted_id)
        else:
            schema_id = f"mock_{len(self.api_metadata)}"

        # Add to FAISS index
        if "embedding" in schema:
            embedding = np.array(schema["embedding"], dtype="float32")
            self.api_index.add(np.array([embedding]))

            # Store metadata
            metadata = {k: v for k, v in schema.items() if k != "embedding"}
            metadata["id"] = schema_id
            self.api_metadata.append(metadata)

        return schema_id

    def store_field_mapping(self, mapping: Dict[str, Any]) -> str:
        """
        Store field mapping.

        Args:
            mapping: Field mapping dictionary

        Returns:
            Mapping ID
        """
        if not self.mock_mode:
            result = self.field_mappings_col.insert_one(mapping)
            mapping_id = str(result.inserted_id)
        else:
            mapping_id = f"mock_{len(self.field_metadata)}"

        # Add to FAISS index
        if "embedding" in mapping:
            embedding = np.array(mapping["embedding"], dtype="float32")
            self.field_index.add(np.array([embedding]))

            # Store metadata
            metadata = {k: v for k, v in mapping.items() if k != "embedding"}
            metadata["id"] = mapping_id
            self.field_metadata.append(metadata)

        return mapping_id

    def store_business_rule(self, rule: Dict[str, Any]) -> str:
        """
        Store business rule.

        Args:
            rule: Business rule dictionary

        Returns:
            Rule ID
        """
        if not self.mock_mode:
            result = self.business_rules_col.insert_one(rule)
            rule_id = str(result.inserted_id)
        else:
            rule_id = f"mock_{len(self.rule_metadata)}"

        # Add to FAISS index
        if "embedding" in rule:
            embedding = np.array(rule["embedding"], dtype="float32")
            self.rule_index.add(np.array([embedding]))

            # Store metadata
            metadata = {k: v for k, v in rule.items() if k != "embedding"}
            metadata["id"] = rule_id
            self.rule_metadata.append(metadata)

        return rule_id

    def store_example(self, example: Dict[str, Any]) -> str:
        """
        Store example.

        Args:
            example: Example dictionary

        Returns:
            Example ID
        """
        if not self.mock_mode:
            result = self.examples_col.insert_one(example)
            example_id = str(result.inserted_id)
        else:
            example_id = f"mock_example_{len(self.api_metadata)}"

        return example_id

    def search_apis(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant API schemas.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of API schemas with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")

        # Search FAISS index
        if self.api_index.ntotal == 0:
            return []

        distances, indices = self.api_index.search(
            np.array([query_embedding]), min(k, self.api_index.ntotal)
        )

        # Get results with metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.api_metadata):
                result = self.api_metadata[idx].copy()
                result["similarity"] = float(distance)
                results.append(result)

        return results

    def search_field_mappings(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant field mappings.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of field mappings with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")

        # Search FAISS index
        if self.field_index.ntotal == 0:
            return []

        distances, indices = self.field_index.search(
            np.array([query_embedding]), min(k, self.field_index.ntotal)
        )

        # Get results with metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.field_metadata):
                result = self.field_metadata[idx].copy()
                result["similarity"] = float(distance)
                results.append(result)

        return results

    def search_business_rules(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant business rules.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of business rules with similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")

        # Search FAISS index
        if self.rule_index.ntotal == 0:
            return []

        distances, indices = self.rule_index.search(
            np.array([query_embedding]), min(k, self.rule_index.ntotal)
        )

        # Get results with metadata
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.rule_metadata):
                result = self.rule_metadata[idx].copy()
                result["similarity"] = float(distance)
                results.append(result)

        return results

    def get_api_by_type(self, doc_type: str) -> List[Dict[str, Any]]:
        """
        Get APIs for specific document type.

        Args:
            doc_type: Document type (e.g., "purchase_order")

        Returns:
            List of relevant APIs
        """
        query = f"SAP API for {doc_type.replace('_', ' ')}"
        return self.search_apis(query, k=5)

    def get_field_mapping(self, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Get SAP field mapping for a field name.

        Args:
            field_name: Field name

        Returns:
            Field mapping or None
        """
        results = self.search_field_mappings(field_name, k=1)
        return results[0] if results else None

    def save_indexes(self) -> None:
        """Save FAISS indexes to disk."""
        # Save indexes
        faiss.write_index(self.api_index, str(self.storage_path / "api_index.faiss"))
        faiss.write_index(self.field_index, str(self.storage_path / "field_index.faiss"))
        faiss.write_index(self.rule_index, str(self.storage_path / "rule_index.faiss"))

        # Save metadata
        with open(self.storage_path / "api_metadata.json", "w") as f:
            json.dump(self.api_metadata, f, indent=2)

        with open(self.storage_path / "field_metadata.json", "w") as f:
            json.dump(self.field_metadata, f, indent=2)

        with open(self.storage_path / "rule_metadata.json", "w") as f:
            json.dump(self.rule_metadata, f, indent=2)

        logger.info("Indexes saved to disk")

    def load_indexes(self) -> bool:
        """
        Load FAISS indexes from disk.

        Returns:
            True if successful
        """
        try:
            # Load indexes
            self.api_index = faiss.read_index(str(self.storage_path / "api_index.faiss"))
            self.field_index = faiss.read_index(str(self.storage_path / "field_index.faiss"))
            self.rule_index = faiss.read_index(str(self.storage_path / "rule_index.faiss"))

            # Load metadata
            with open(self.storage_path / "api_metadata.json", "r") as f:
                self.api_metadata = json.load(f)

            with open(self.storage_path / "field_metadata.json", "r") as f:
                self.field_metadata = json.load(f)

            with open(self.storage_path / "rule_metadata.json", "r") as f:
                self.rule_metadata = json.load(f)

            logger.info(
                f"Indexes loaded: {len(self.api_metadata)} APIs, "
                f"{len(self.field_metadata)} mappings, "
                f"{len(self.rule_metadata)} rules"
            )

            return True

        except FileNotFoundError:
            logger.warning("Index files not found")
            return False

        except Exception as e:
            logger.error(f"Failed to load indexes: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """
        Get knowledge base statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "api_schemas": self.api_index.ntotal if self.api_index else 0,
            "field_mappings": self.field_index.ntotal if self.field_index else 0,
            "business_rules": self.rule_index.ntotal if self.rule_index else 0,
        }
