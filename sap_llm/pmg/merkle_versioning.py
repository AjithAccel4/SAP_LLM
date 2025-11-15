"""
Merkle Tree Document Versioning for Process Memory Graph.

Implements content-addressable storage using Merkle trees for:
- Efficient document version tracking
- Tamper detection
- Deduplication
- Temporal queries (as-of time travel)
- Audit trail with cryptographic verification

Each document version gets a unique hash based on its content,
enabling efficient diff computation and version history.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DocumentVersion:
    """Represents a single version of a document."""
    version_hash: str
    parent_hash: Optional[str]
    document_data: Dict[str, Any]
    timestamp: str
    version_number: int
    change_summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MerkleNode:
    """Node in the Merkle tree."""
    hash: str
    left_hash: Optional[str]
    right_hash: Optional[str]
    data: Optional[Dict[str, Any]]
    is_leaf: bool


class MerkleVersioning:
    """
    Merkle tree-based document versioning system.

    Features:
    - Content-based addressing (hash = f(content))
    - Efficient deduplication
    - Tamper-proof history
    - Fast version comparison
    - As-of temporal queries
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        """
        Initialize Merkle versioning system.

        Args:
            storage_backend: Optional backend for persistent storage (e.g., Redis, Postgres)
        """
        self.storage = storage_backend

        # In-memory version storage
        self.versions: Dict[str, DocumentVersion] = {}

        # Document ID to version hashes mapping
        self.doc_versions: Dict[str, List[str]] = defaultdict(list)

        # Hash to version mapping (for fast lookup)
        self.hash_to_version: Dict[str, DocumentVersion] = {}

        logger.info("MerkleVersioning initialized")

    def compute_document_hash(self, document: Dict[str, Any]) -> str:
        """
        Compute content hash for document.

        Uses SHA-256 for cryptographic security.

        Args:
            document: Document data

        Returns:
            Hex-encoded SHA-256 hash
        """
        # Normalize document for consistent hashing
        normalized = self._normalize_document(document)

        # Convert to JSON with sorted keys
        json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=False)

        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))

        return hash_obj.hexdigest()

    def _normalize_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize document for consistent hashing.

        Removes volatile fields like timestamps, UUIDs, etc.
        """
        normalized = {}

        # Fields to exclude from hashing (volatile/metadata fields)
        exclude_fields = {
            'processing_timestamp',
            'ingestion_timestamp',
            'uuid',
            'version_hash',
            'parent_hash',
            'created_at',
            'updated_at'
        }

        for key, value in document.items():
            if key not in exclude_fields:
                # Recursively normalize nested dicts
                if isinstance(value, dict):
                    normalized[key] = self._normalize_document(value)
                elif isinstance(value, list):
                    # Normalize lists
                    normalized[key] = [
                        self._normalize_document(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    normalized[key] = value

        return normalized

    def create_version(
        self,
        doc_id: str,
        document: Dict[str, Any],
        change_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentVersion:
        """
        Create new version of document.

        Args:
            doc_id: Document identifier
            document: Document data
            change_summary: Optional description of changes
            metadata: Optional metadata

        Returns:
            Created document version
        """
        # Compute content hash
        content_hash = self.compute_document_hash(document)

        # Check if this exact version already exists (deduplication)
        if content_hash in self.hash_to_version:
            logger.info(f"Document version already exists (deduplicated): {content_hash[:8]}")
            return self.hash_to_version[content_hash]

        # Get parent hash (previous version)
        parent_hash = None
        version_number = 1

        if doc_id in self.doc_versions and self.doc_versions[doc_id]:
            parent_hash = self.doc_versions[doc_id][-1]
            parent_version = self.hash_to_version[parent_hash]
            version_number = parent_version.version_number + 1

        # Create version
        version = DocumentVersion(
            version_hash=content_hash,
            parent_hash=parent_hash,
            document_data=document,
            timestamp=datetime.now().isoformat(),
            version_number=version_number,
            change_summary=change_summary,
            metadata=metadata or {}
        )

        # Store version
        self.versions[content_hash] = version
        self.hash_to_version[content_hash] = version
        self.doc_versions[doc_id].append(content_hash)

        # Persist if backend available
        if self.storage:
            self._persist_version(doc_id, version)

        logger.info(
            f"Created version {version_number} for {doc_id}: {content_hash[:8]}"
        )

        return version

    def get_version(self, version_hash: str) -> Optional[DocumentVersion]:
        """
        Get specific version by hash.

        Args:
            version_hash: Version hash

        Returns:
            Document version or None
        """
        return self.hash_to_version.get(version_hash)

    def get_latest_version(self, doc_id: str) -> Optional[DocumentVersion]:
        """
        Get latest version of document.

        Args:
            doc_id: Document ID

        Returns:
            Latest document version or None
        """
        if doc_id not in self.doc_versions or not self.doc_versions[doc_id]:
            return None

        latest_hash = self.doc_versions[doc_id][-1]
        return self.hash_to_version.get(latest_hash)

    def get_version_history(self, doc_id: str) -> List[DocumentVersion]:
        """
        Get complete version history for document.

        Args:
            doc_id: Document ID

        Returns:
            List of versions (oldest to newest)
        """
        if doc_id not in self.doc_versions:
            return []

        return [
            self.hash_to_version[vh]
            for vh in self.doc_versions[doc_id]
            if vh in self.hash_to_version
        ]

    def get_version_at_time(
        self,
        doc_id: str,
        timestamp: datetime
    ) -> Optional[DocumentVersion]:
        """
        Get document version as of specific time (time travel query).

        Args:
            doc_id: Document ID
            timestamp: Target timestamp

        Returns:
            Document version at that time or None
        """
        versions = self.get_version_history(doc_id)

        if not versions:
            return None

        # Find latest version before timestamp
        target_version = None

        for version in versions:
            version_time = datetime.fromisoformat(version.timestamp)

            if version_time <= timestamp:
                target_version = version
            else:
                break

        return target_version

    def compute_diff(
        self,
        version_hash_1: str,
        version_hash_2: str
    ) -> Dict[str, Any]:
        """
        Compute diff between two versions.

        Args:
            version_hash_1: First version hash
            version_hash_2: Second version hash

        Returns:
            Diff object showing changes
        """
        v1 = self.get_version(version_hash_1)
        v2 = self.get_version(version_hash_2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        diff = {
            "from_version": version_hash_1[:8],
            "to_version": version_hash_2[:8],
            "added": {},
            "removed": {},
            "modified": {}
        }

        # Compare documents
        doc1 = v1.document_data
        doc2 = v2.document_data

        # Find added fields
        for key in doc2:
            if key not in doc1:
                diff["added"][key] = doc2[key]

        # Find removed fields
        for key in doc1:
            if key not in doc2:
                diff["removed"][key] = doc1[key]

        # Find modified fields
        for key in doc1:
            if key in doc2 and doc1[key] != doc2[key]:
                diff["modified"][key] = {
                    "old": doc1[key],
                    "new": doc2[key]
                }

        return diff

    def verify_version_integrity(self, version_hash: str) -> bool:
        """
        Verify version hasn't been tampered with.

        Recomputes hash from data and compares.

        Args:
            version_hash: Version hash to verify

        Returns:
            True if integrity verified, False otherwise
        """
        version = self.get_version(version_hash)

        if not version:
            logger.error(f"Version not found: {version_hash}")
            return False

        # Recompute hash from document data
        computed_hash = self.compute_document_hash(version.document_data)

        # Compare with stored hash
        is_valid = computed_hash == version_hash

        if not is_valid:
            logger.error(
                f"Integrity check FAILED for {version_hash[:8]}: "
                f"expected {version_hash[:8]}, got {computed_hash[:8]}"
            )

        return is_valid

    def verify_chain_integrity(self, doc_id: str) -> bool:
        """
        Verify entire version chain for document.

        Args:
            doc_id: Document ID

        Returns:
            True if entire chain is valid
        """
        versions = self.get_version_history(doc_id)

        if not versions:
            return True

        # Check each version
        for version in versions:
            if not self.verify_version_integrity(version.version_hash):
                return False

        # Verify parent links
        for i, version in enumerate(versions):
            if i == 0:
                # First version should have no parent
                if version.parent_hash is not None:
                    logger.error(f"First version has parent: {version.version_hash[:8]}")
                    return False
            else:
                # Subsequent versions should link to previous
                expected_parent = versions[i - 1].version_hash
                if version.parent_hash != expected_parent:
                    logger.error(
                        f"Version {version.version_hash[:8]} has invalid parent: "
                        f"expected {expected_parent[:8]}, got {version.parent_hash[:8] if version.parent_hash else 'None'}"
                    )
                    return False

        logger.info(f"Chain integrity verified for {doc_id} ({len(versions)} versions)")
        return True

    def build_merkle_tree(self, version_hashes: List[str]) -> MerkleNode:
        """
        Build Merkle tree from version hashes.

        Useful for batch verification and efficient sync.

        Args:
            version_hashes: List of version hashes

        Returns:
            Root node of Merkle tree
        """
        if not version_hashes:
            raise ValueError("Cannot build tree from empty list")

        # Create leaf nodes
        nodes = [
            MerkleNode(
                hash=vh,
                left_hash=None,
                right_hash=None,
                data=None,
                is_leaf=True
            )
            for vh in version_hashes
        ]

        # Build tree bottom-up
        while len(nodes) > 1:
            parent_nodes = []

            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left

                # Compute parent hash
                combined = f"{left.hash}{right.hash}"
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()

                parent = MerkleNode(
                    hash=parent_hash,
                    left_hash=left.hash,
                    right_hash=right.hash,
                    data=None,
                    is_leaf=False
                )

                parent_nodes.append(parent)

            nodes = parent_nodes

        return nodes[0]

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get versioning storage statistics.

        Returns:
            Statistics dictionary
        """
        total_versions = len(self.versions)
        total_documents = len(self.doc_versions)

        # Compute deduplication ratio
        total_entries = sum(len(vlist) for vlist in self.doc_versions.values())
        unique_versions = len(self.hash_to_version)
        dedup_ratio = (1 - unique_versions / total_entries) if total_entries > 0 else 0

        # Version distribution
        versions_per_doc = [len(vlist) for vlist in self.doc_versions.values()]
        avg_versions = sum(versions_per_doc) / len(versions_per_doc) if versions_per_doc else 0
        max_versions = max(versions_per_doc) if versions_per_doc else 0

        return {
            "total_documents": total_documents,
            "total_versions": total_versions,
            "unique_versions": unique_versions,
            "deduplication_ratio": round(dedup_ratio * 100, 2),
            "avg_versions_per_document": round(avg_versions, 2),
            "max_versions_per_document": max_versions
        }

    def _persist_version(self, doc_id: str, version: DocumentVersion):
        """
        Persist version to storage backend.

        Placeholder for actual storage implementation.
        """
        if not self.storage:
            return

        # Implementation would depend on storage backend
        # E.g., for Redis:
        # self.storage.hset(f"version:{version.version_hash}", mapping=asdict(version))
        # self.storage.rpush(f"doc_versions:{doc_id}", version.version_hash)

        pass

    def export_version_history(self, doc_id: str) -> Dict[str, Any]:
        """
        Export complete version history in JSON format.

        Useful for auditing and compliance.

        Args:
            doc_id: Document ID

        Returns:
            Version history export
        """
        versions = self.get_version_history(doc_id)

        export = {
            "document_id": doc_id,
            "total_versions": len(versions),
            "created_at": versions[0].timestamp if versions else None,
            "last_updated": versions[-1].timestamp if versions else None,
            "versions": [
                {
                    "version_number": v.version_number,
                    "version_hash": v.version_hash,
                    "parent_hash": v.parent_hash,
                    "timestamp": v.timestamp,
                    "change_summary": v.change_summary,
                    "document_data": v.document_data
                }
                for v in versions
            ]
        }

        return export


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize versioning system
    versioning = MerkleVersioning()

    # Create first version
    doc_v1 = {
        "doc_type": "invoice",
        "invoice_number": "INV-001",
        "total_amount": 1000.00,
        "vendor_name": "Acme Corp"
    }

    v1 = versioning.create_version(
        doc_id="doc_123",
        document=doc_v1,
        change_summary="Initial version"
    )

    print(f"Version 1: {v1.version_hash[:8]}")

    # Create second version (modified)
    doc_v2 = doc_v1.copy()
    doc_v2["total_amount"] = 1200.00

    v2 = versioning.create_version(
        doc_id="doc_123",
        document=doc_v2,
        change_summary="Updated total amount"
    )

    print(f"Version 2: {v2.version_hash[:8]}")

    # Compute diff
    diff = versioning.compute_diff(v1.version_hash, v2.version_hash)
    print(f"Diff: {json.dumps(diff, indent=2)}")

    # Verify integrity
    is_valid = versioning.verify_chain_integrity("doc_123")
    print(f"Chain integrity: {is_valid}")

    # Get stats
    stats = versioning.get_storage_stats()
    print(f"Stats: {json.dumps(stats, indent=2)}")
