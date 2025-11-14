"""
Document Collector for training data acquisition.

Supports multiple sources:
- Local filesystem
- S3/MinIO object storage
- SAP systems via APIs
- Public datasets (RVL-CDIP, FUNSD, CORD, etc.)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class DocumentCollector:
    """
    Collects documents from various sources for model training.

    Target: 2.1M documents across 8 SAP document types.
    """

    def __init__(self, output_dir: str, max_workers: int = 10):
        """
        Initialize document collector.

        Args:
            output_dir: Directory to store collected documents
            max_workers: Number of parallel workers for collection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers

        self.stats = {
            "total_collected": 0,
            "by_source": {},
            "by_type": {},
            "errors": 0
        }

        logger.info(f"DocumentCollector initialized: output_dir={output_dir}")

    def collect_from_local(self,
                           source_dir: str,
                           file_patterns: List[str] = ["*.pdf", "*.png", "*.jpg"],
                           metadata_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Collect documents from local filesystem.

        Args:
            source_dir: Source directory containing documents
            file_patterns: File patterns to match (glob patterns)
            metadata_file: Optional JSON file with document metadata

        Returns:
            List of collected document metadata
        """
        source_path = Path(source_dir)
        documents = []

        # Load metadata if provided
        metadata = {}
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        # Collect documents
        for pattern in file_patterns:
            for file_path in source_path.rglob(pattern):
                try:
                    # Copy to output directory
                    relative_path = file_path.relative_to(source_path)
                    output_path = self.output_dir / relative_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    import shutil
                    shutil.copy2(file_path, output_path)

                    # Create document metadata
                    doc_id = file_path.stem
                    doc_meta = {
                        "id": doc_id,
                        "source": "local",
                        "original_path": str(file_path),
                        "stored_path": str(output_path),
                        "filename": file_path.name,
                        "size_bytes": file_path.stat().st_size,
                    }

                    # Add metadata from file if available
                    if doc_id in metadata:
                        doc_meta.update(metadata[doc_id])

                    documents.append(doc_meta)
                    self.stats["total_collected"] += 1

                except Exception as e:
                    logger.error(f"Error collecting {file_path}: {e}")
                    self.stats["errors"] += 1

        self._update_stats("local", documents)
        logger.info(f"Collected {len(documents)} documents from local filesystem")

        return documents

    def collect_from_s3(self,
                        bucket: str,
                        prefix: str = "",
                        endpoint_url: Optional[str] = None,
                        access_key: Optional[str] = None,
                        secret_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Collect documents from S3/MinIO.

        Args:
            bucket: S3 bucket name
            prefix: Prefix to filter objects
            endpoint_url: Custom endpoint for MinIO
            access_key: AWS access key or MinIO access key
            secret_key: AWS secret key or MinIO secret key

        Returns:
            List of collected document metadata
        """
        # Initialize S3 client
        s3_config = {}
        if endpoint_url:
            s3_config["endpoint_url"] = endpoint_url
        if access_key and secret_key:
            s3_config["aws_access_key_id"] = access_key
            s3_config["aws_secret_access_key"] = secret_key

        s3_client = boto3.client('s3', **s3_config)
        documents = []

        try:
            # List objects in bucket
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            # Collect objects
            for page in pages:
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']

                    # Skip if not a document file
                    if not any(key.lower().endswith(ext) for ext in ['.pdf', '.png', '.jpg', '.jpeg']):
                        continue

                    try:
                        # Download file
                        filename = Path(key).name
                        output_path = self.output_dir / filename

                        s3_client.download_file(bucket, key, str(output_path))

                        # Create document metadata
                        doc_meta = {
                            "id": Path(key).stem,
                            "source": "s3",
                            "bucket": bucket,
                            "key": key,
                            "stored_path": str(output_path),
                            "filename": filename,
                            "size_bytes": obj['Size'],
                            "last_modified": obj['LastModified'].isoformat(),
                        }

                        documents.append(doc_meta)
                        self.stats["total_collected"] += 1

                    except ClientError as e:
                        logger.error(f"Error downloading {key}: {e}")
                        self.stats["errors"] += 1

        except ClientError as e:
            logger.error(f"Error listing S3 bucket: {e}")
            raise

        self._update_stats("s3", documents)
        logger.info(f"Collected {len(documents)} documents from S3/MinIO")

        return documents

    def collect_public_datasets(self,
                                 datasets: List[str] = ["rvl-cdip", "funsd", "cord"]) -> List[Dict[str, Any]]:
        """
        Download and collect public document AI datasets.

        Supported datasets:
        - RVL-CDIP: 400K document images (16 categories)
        - FUNSD: Form understanding dataset
        - CORD: Consolidated receipt dataset
        - SROIE: Scanned receipts OCR and IE

        Args:
            datasets: List of dataset names to download

        Returns:
            List of collected document metadata
        """
        all_documents = []

        for dataset_name in datasets:
            logger.info(f"Downloading dataset: {dataset_name}")

            if dataset_name == "rvl-cdip":
                docs = self._collect_rvl_cdip()
            elif dataset_name == "funsd":
                docs = self._collect_funsd()
            elif dataset_name == "cord":
                docs = self._collect_cord()
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue

            all_documents.extend(docs)
            self._update_stats(f"public/{dataset_name}", docs)

        return all_documents

    def _collect_rvl_cdip(self) -> List[Dict[str, Any]]:
        """Download RVL-CDIP dataset (400K document images)."""
        # Implementation would download from:
        # https://www.cs.cmu.edu/~aharley/rvl-cdip/
        logger.info("RVL-CDIP collection not yet implemented - placeholder")
        return []

    def _collect_funsd(self) -> List[Dict[str, Any]]:
        """Download FUNSD dataset (form understanding)."""
        # Implementation would download from:
        # https://guillaumejaume.github.io/FUNSD/
        logger.info("FUNSD collection not yet implemented - placeholder")
        return []

    def _collect_cord(self) -> List[Dict[str, Any]]:
        """Download CORD dataset (receipts)."""
        # Implementation would download from:
        # https://github.com/clovaai/cord
        logger.info("CORD collection not yet implemented - placeholder")
        return []

    def collect_synthetic_documents(self,
                                      template_dir: str,
                                      num_documents: int = 10000,
                                      document_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Generate synthetic documents from templates.

        Uses templates and data generation to create realistic SAP documents:
        - Invoices
        - Purchase Orders
        - Delivery Notes
        - Material Documents
        - Sales Orders
        - Goods Receipts
        - Packing Lists
        - Shipping Notices

        Args:
            template_dir: Directory containing document templates
            num_documents: Number of synthetic documents to generate
            document_types: Specific document types to generate

        Returns:
            List of generated document metadata
        """
        if document_types is None:
            document_types = [
                "invoice", "purchase_order", "delivery_note", "material_document",
                "sales_order", "goods_receipt", "packing_list", "shipping_notice"
            ]

        documents = []
        docs_per_type = num_documents // len(document_types)

        for doc_type in document_types:
            logger.info(f"Generating {docs_per_type} synthetic {doc_type} documents")

            for i in range(docs_per_type):
                # Generate synthetic document (placeholder implementation)
                doc_id = f"synthetic_{doc_type}_{i:06d}"
                output_path = self.output_dir / f"{doc_id}.pdf"

                # In real implementation, would use template engine + fake data
                # For now, just create metadata
                doc_meta = {
                    "id": doc_id,
                    "source": "synthetic",
                    "document_type": doc_type,
                    "stored_path": str(output_path),
                    "generated": True,
                }

                documents.append(doc_meta)
                self.stats["total_collected"] += 1

        self._update_stats("synthetic", documents)
        logger.info(f"Generated {len(documents)} synthetic documents")

        return documents

    def _update_stats(self, source: str, documents: List[Dict]):
        """Update collection statistics."""
        self.stats["by_source"][source] = self.stats["by_source"].get(source, 0) + len(documents)

        for doc in documents:
            doc_type = doc.get("document_type", "unknown")
            self.stats["by_type"][doc_type] = self.stats["by_type"].get(doc_type, 0) + 1

    def save_metadata(self, metadata_file: str, documents: List[Dict[str, Any]]):
        """
        Save collected documents metadata to JSON file.

        Args:
            metadata_file: Output JSON file path
            documents: List of document metadata
        """
        output_path = Path(metadata_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "total_documents": len(documents),
                "statistics": self.stats,
                "documents": documents
            }, f, indent=2)

        logger.info(f"Saved metadata for {len(documents)} documents to {metadata_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.stats.copy()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    collector = DocumentCollector(output_dir="./data/raw")

    # Collect from local directory
    # docs = collector.collect_from_local("./sample_documents")

    # Collect synthetic documents
    synthetic_docs = collector.collect_synthetic_documents(
        template_dir="./templates",
        num_documents=1000
    )

    # Save metadata
    collector.save_metadata("./data/metadata.json", synthetic_docs)

    print(f"Collection statistics: {collector.get_statistics()}")
