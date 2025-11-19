"""
Process Memory Graph (PMG) Interface and Mock Implementation.

Provides abstract interface for PMG operations and a mock implementation
for testing and development without a real PMG system.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class PMGInterface(ABC):
    """Abstract interface for Process Memory Graph operations."""

    @abstractmethod
    def query_similar_documents(
        self,
        vendor: Optional[str] = None,
        doc_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query for similar historical documents.

        Args:
            vendor: Vendor name to filter by
            doc_type: Document type to filter by
            limit: Maximum number of results

        Returns:
            List of similar documents
        """
        pass

    @abstractmethod
    def add_error_pattern(self, pattern: Dict[str, Any]):
        """
        Add an error pattern to PMG.

        Args:
            pattern: Error pattern dictionary
        """
        pass

    @abstractmethod
    def query_error_patterns(
        self,
        document_type: Optional[str] = None,
        vendor: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query for error patterns.

        Args:
            document_type: Document type to filter by
            vendor: Vendor to filter by
            limit: Maximum number of results

        Returns:
            List of error patterns
        """
        pass

    @abstractmethod
    def update_with_ground_truth(
        self,
        document_id: str,
        corrected_data: Dict[str, Any],
        reviewer_id: str
    ):
        """
        Update PMG with ground truth from human review.

        Args:
            document_id: Document identifier
            corrected_data: Human-corrected data
            reviewer_id: ID of reviewer
        """
        pass


class MockPMG(PMGInterface):
    """
    Mock PMG implementation for testing and development.

    Provides in-memory storage with persistence option.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize mock PMG.

        Args:
            storage_path: Optional path to persist data
        """
        self.storage_path = storage_path
        self.documents: List[Dict[str, Any]] = []
        self.error_patterns: List[Dict[str, Any]] = []
        self.ground_truth: Dict[str, Dict[str, Any]] = {}

        # Load existing data if storage path provided
        if storage_path:
            self._load_data()

        logger.info(f"MockPMG initialized with {len(self.documents)} documents")

    def query_similar_documents(
        self,
        vendor: Optional[str] = None,
        doc_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query for similar historical documents."""
        try:
            results = []

            for doc in self.documents:
                # Filter by vendor
                if vendor and doc.get('vendor') != vendor:
                    continue

                # Filter by document type
                if doc_type and doc.get('doc_type') != doc_type:
                    continue

                results.append(doc)

                if len(results) >= limit:
                    break

            logger.debug(
                f"PMG query: vendor={vendor}, doc_type={doc_type}, "
                f"results={len(results)}"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to query similar documents: {e}", exc_info=True)
            return []

    def add_document(self, document: Dict[str, Any]):
        """
        Add a document to the mock PMG.

        Args:
            document: Document data
        """
        try:
            # Add timestamp if not present
            if 'timestamp' not in document:
                document['timestamp'] = datetime.now().isoformat()

            self.documents.append(document)

            # Persist if storage path configured
            if self.storage_path:
                self._save_data()

            logger.debug(f"Added document to PMG: {document.get('id', 'unknown')}")

        except Exception as e:
            logger.error(f"Failed to add document: {e}", exc_info=True)

    def add_error_pattern(self, pattern: Dict[str, Any]):
        """Add an error pattern to PMG."""
        try:
            if 'timestamp' not in pattern:
                pattern['timestamp'] = datetime.now().isoformat()

            self.error_patterns.append(pattern)

            # Persist if storage path configured
            if self.storage_path:
                self._save_data()

            logger.debug(f"Added error pattern to PMG: {pattern.get('id', 'unknown')}")

        except Exception as e:
            logger.error(f"Failed to add error pattern: {e}", exc_info=True)

    def query_error_patterns(
        self,
        document_type: Optional[str] = None,
        vendor: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Query for error patterns."""
        try:
            results = []

            for pattern in self.error_patterns:
                # Filter by document type
                if document_type and pattern.get('document_type') != document_type:
                    continue

                # Filter by vendor
                if vendor and pattern.get('vendor') != vendor:
                    continue

                results.append(pattern)

                if len(results) >= limit:
                    break

            logger.debug(
                f"PMG error pattern query: doc_type={document_type}, "
                f"vendor={vendor}, results={len(results)}"
            )

            return results

        except Exception as e:
            logger.error(f"Failed to query error patterns: {e}", exc_info=True)
            return []

    def update_with_ground_truth(
        self,
        document_id: str,
        corrected_data: Dict[str, Any],
        reviewer_id: str
    ):
        """Update PMG with ground truth from human review."""
        try:
            self.ground_truth[document_id] = {
                'corrected_data': corrected_data,
                'reviewer_id': reviewer_id,
                'timestamp': datetime.now().isoformat()
            }

            # Persist if storage path configured
            if self.storage_path:
                self._save_data()

            logger.info(
                f"Updated PMG with ground truth: doc_id={document_id}, "
                f"reviewer={reviewer_id}"
            )

        except Exception as e:
            logger.error(f"Failed to update ground truth: {e}", exc_info=True)

    def get_ground_truth(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get ground truth for a document."""
        return self.ground_truth.get(document_id)

    def populate_sample_data(self, num_documents: int = 100):
        """
        Populate with sample data for testing.

        Args:
            num_documents: Number of sample documents to generate
        """
        logger.info(f"Populating PMG with {num_documents} sample documents")

        vendors = ["ACME Corp", "TechSupply Inc", "Global Parts Ltd", "MegaVendor Co"]
        doc_types = ["INVOICE", "PURCHASE_ORDER", "DELIVERY_NOTE"]

        for i in range(num_documents):
            vendor = vendors[i % len(vendors)]
            doc_type = doc_types[i % len(doc_types)]

            document = {
                'id': f'DOC-{i:05d}',
                'vendor': vendor,
                'doc_type': doc_type,
                'total_amount': 1000.0 + (i * 10),
                'currency': 'USD',
                'payment_terms': 'NET30',
                'timestamp': datetime.now().isoformat()
            }

            self.add_document(document)

        logger.info("Sample data populated successfully")

    def _save_data(self):
        """Save data to storage."""
        if not self.storage_path:
            return

        try:
            storage_file = Path(self.storage_path) / 'mock_pmg_data.json'
            storage_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'documents': self.documents,
                'error_patterns': self.error_patterns,
                'ground_truth': self.ground_truth,
                'last_updated': datetime.now().isoformat()
            }

            with open(storage_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"PMG data saved to {storage_file}")

        except Exception as e:
            logger.error(f"Failed to save PMG data: {e}", exc_info=True)

    def _load_data(self):
        """Load data from storage."""
        if not self.storage_path:
            return

        try:
            storage_file = Path(self.storage_path) / 'mock_pmg_data.json'

            if not storage_file.exists():
                logger.info("No existing PMG data file found")
                return

            with open(storage_file, 'r') as f:
                data = json.load(f)

            self.documents = data.get('documents', [])
            self.error_patterns = data.get('error_patterns', [])
            self.ground_truth = data.get('ground_truth', {})

            logger.info(
                f"PMG data loaded: {len(self.documents)} documents, "
                f"{len(self.error_patterns)} patterns"
            )

        except Exception as e:
            logger.error(f"Failed to load PMG data: {e}", exc_info=True)

    def clear(self):
        """Clear all data (mainly for testing)."""
        self.documents = []
        self.error_patterns = []
        self.ground_truth = {}

        if self.storage_path:
            self._save_data()

        logger.info("PMG data cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get PMG statistics."""
        return {
            'total_documents': len(self.documents),
            'total_error_patterns': len(self.error_patterns),
            'total_ground_truth': len(self.ground_truth),
            'unique_vendors': len(set(d.get('vendor') for d in self.documents if d.get('vendor'))),
            'unique_doc_types': len(set(d.get('doc_type') for d in self.documents if d.get('doc_type')))
        }


def create_pmg_instance(
    pmg_type: str = "mock",
    storage_path: Optional[str] = None
) -> PMGInterface:
    """
    Factory function to create PMG instance.

    Args:
        pmg_type: Type of PMG ("mock" or "real")
        storage_path: Optional storage path for mock PMG

    Returns:
        PMG instance
    """
    if pmg_type == "mock":
        pmg = MockPMG(storage_path=storage_path)
        logger.info("Created MockPMG instance")
        return pmg
    elif pmg_type == "real":
        # In production, this would create actual PMG connection
        raise NotImplementedError("Real PMG implementation not yet available")
    else:
        raise ValueError(f"Unknown PMG type: {pmg_type}")
