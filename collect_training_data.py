#!/usr/bin/env python3
"""
Data Collection Pipeline Orchestrator for SAP_LLM.

Orchestrates collection of 1M+ training documents from multiple sources:
1. QorSync PostgreSQL database (300K target)
2. SAP Business Accelerator Hub APIs (200K target)
3. Public datasets (200K target)
4. Synthetic document generation (500K target)

Based on the exploration report findings and training data requirements.

Usage:
    python collect_training_data.py --all
    python collect_training_data.py --source qorsync
    python collect_training_data.py --source sap_hub
    python collect_training_data.py --source synthetic --count 500000
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sap_llm.data_pipeline.corpus_builder import CorpusBuilder
from sap_llm.data_pipeline.validate_corpus import validate_corpus
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class DataCollectionOrchestrator:
    """
    Orchestrates data collection from multiple sources.

    Pipeline:
    1. Source-specific extraction
    2. Quality validation
    3. Deduplication
    4. Format standardization
    5. Train/val/test splitting
    6. Final validation
    """

    def __init__(self, output_dir: str = "data/training_corpus"):
        """Initialize data collection orchestrator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Source-specific output dirs
        self.qorsync_dir = self.output_dir / "qorsync"
        self.sap_hub_dir = self.output_dir / "sap_hub"
        self.public_dir = self.output_dir / "public_datasets"
        self.synthetic_dir = self.output_dir / "synthetic"

        for dir_path in [self.qorsync_dir, self.sap_hub_dir, self.public_dir, self.synthetic_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "qorsync_collected": 0,
            "sap_hub_collected": 0,
            "public_collected": 0,
            "synthetic_generated": 0,
            "total_documents": 0,
            "duplicates_removed": 0,
            "validation_failures": 0,
        }

        logger.info(f"Data Collection Orchestrator initialized")
        logger.info(f"Output directory: {self.output_dir}")

    def collect_qorsync_data(self, target: int = 300000) -> Dict[str, Any]:
        """
        Collect documents from QorSync PostgreSQL database.

        Target: 300K documents (invoices, POs, receipts, etc.)

        Args:
            target: Target number of documents to collect

        Returns:
            Collection statistics
        """
        logger.info("=" * 80)
        logger.info(f"COLLECTING FROM QORSYNC DATABASE (Target: {target:,})")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Check if QorSync connection is configured
        db_url = os.getenv("QORSYNC_DB_URL")
        if not db_url:
            logger.warning("QORSYNC_DB_URL not set. Skipping QorSync collection.")
            logger.info("Set environment variable: export QORSYNC_DB_URL=postgresql://user:pass@host/db")
            return {"status": "skipped", "reason": "database_not_configured"}

        try:
            # Import QorSync extractor
            from sap_llm.data_pipeline.qorsync_extractor import QorSyncExtractor

            extractor = QorSyncExtractor(db_url=db_url)

            # Extract documents
            logger.info("Extracting documents from QorSync...")
            documents = extractor.extract_documents(limit=target)

            logger.info(f"Extracted {len(documents):,} documents")

            # Save to disk
            output_file = self.qorsync_dir / "documents.jsonl"
            with open(output_file, 'w') as f:
                for doc in tqdm(documents, desc="Saving QorSync documents"):
                    f.write(json.dumps(doc) + '\n')

            self.stats["qorsync_collected"] = len(documents)
            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"QorSync collection completed in {duration:.2f}s")
            logger.info(f"Saved to {output_file}")

            return {
                "status": "completed",
                "count": len(documents),
                "duration": duration,
                "output_file": str(output_file)
            }

        except ImportError:
            logger.error("QorSyncExtractor not found. Implement in sap_llm/data_pipeline/qorsync_extractor.py")
            return {"status": "error", "reason": "extractor_not_implemented"}
        except Exception as e:
            logger.error(f"QorSync collection failed: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    def collect_sap_hub_data(self, target: int = 200000) -> Dict[str, Any]:
        """
        Collect documents from SAP Business Accelerator Hub.

        Target: 200K documents from SAP APIs and documentation

        Args:
            target: Target number of documents to collect

        Returns:
            Collection statistics
        """
        logger.info("=" * 80)
        logger.info(f"COLLECTING FROM SAP BUSINESS HUB (Target: {target:,})")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Check if SAP API key is configured
        api_key = os.getenv("SAP_API_KEY")
        if not api_key:
            logger.warning("SAP_API_KEY not set. Skipping SAP Hub collection.")
            logger.info("Set environment variable: export SAP_API_KEY=your_api_key")
            return {"status": "skipped", "reason": "api_key_not_configured"}

        try:
            # Import SAP Hub scraper
            from sap_llm.data_pipeline.sap_hub_scraper import SAPHubScraper

            scraper = SAPHubScraper(api_key=api_key)

            # Scrape documents
            logger.info("Scraping documents from SAP Business Hub...")
            documents = scraper.scrape_documents(limit=target)

            logger.info(f"Scraped {len(documents):,} documents")

            # Save to disk
            output_file = self.sap_hub_dir / "documents.jsonl"
            with open(output_file, 'w') as f:
                for doc in tqdm(documents, desc="Saving SAP Hub documents"):
                    f.write(json.dumps(doc) + '\n')

            self.stats["sap_hub_collected"] = len(documents)
            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"SAP Hub collection completed in {duration:.2f}s")
            logger.info(f"Saved to {output_file}")

            return {
                "status": "completed",
                "count": len(documents),
                "duration": duration,
                "output_file": str(output_file)
            }

        except ImportError:
            logger.error("SAPHubScraper not found. Implement in sap_llm/data_pipeline/sap_hub_scraper.py")
            return {"status": "error", "reason": "scraper_not_implemented"}
        except Exception as e:
            logger.error(f"SAP Hub collection failed: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    def collect_public_datasets(self, target: int = 200000) -> Dict[str, Any]:
        """
        Collect documents from public datasets.

        Sources:
        - FUNSD (Form Understanding in Noisy Scanned Documents)
        - CORD (Consolidated Receipt Dataset)
        - SROIE (Scanned Receipts OCR and Information Extraction)
        - RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)

        Args:
            target: Target number of documents to collect

        Returns:
            Collection statistics
        """
        logger.info("=" * 80)
        logger.info(f"COLLECTING FROM PUBLIC DATASETS (Target: {target:,})")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # Import public dataset loaders
            from sap_llm.data_pipeline.public_datasets import PublicDatasetLoader

            loader = PublicDatasetLoader(cache_dir=str(self.public_dir))

            # Load datasets
            logger.info("Downloading and loading public datasets...")
            documents = loader.load_all_datasets(limit=target)

            logger.info(f"Loaded {len(documents):,} documents from public datasets")

            # Save to disk
            output_file = self.public_dir / "documents.jsonl"
            with open(output_file, 'w') as f:
                for doc in tqdm(documents, desc="Saving public dataset documents"):
                    f.write(json.dumps(doc) + '\n')

            self.stats["public_collected"] = len(documents)
            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"Public dataset collection completed in {duration:.2f}s")
            logger.info(f"Saved to {output_file}")

            return {
                "status": "completed",
                "count": len(documents),
                "duration": duration,
                "output_file": str(output_file)
            }

        except ImportError:
            logger.error("PublicDatasetLoader not found. Implement in sap_llm/data_pipeline/public_datasets.py")
            return {"status": "error", "reason": "loader_not_implemented"}
        except Exception as e:
            logger.error(f"Public dataset collection failed: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    def generate_synthetic_data(self, count: int = 500000) -> Dict[str, Any]:
        """
        Generate synthetic training documents.

        Uses existing generate_500k_synthetic.py script.

        Target: 500K synthetic documents with:
        - 5 languages (EN, DE, FR, ES, IT)
        - 10 template variations per document type
        - Image augmentations (rotation, noise, blur)

        Args:
            count: Number of synthetic documents to generate

        Returns:
            Generation statistics
        """
        logger.info("=" * 80)
        logger.info(f"GENERATING SYNTHETIC DOCUMENTS (Count: {count:,})")
        logger.info("=" * 80)

        start_time = datetime.now()

        try:
            # Check if synthetic generation script exists
            script_path = Path("generate_500k_synthetic.py")
            if not script_path.exists():
                logger.warning("generate_500k_synthetic.py not found. Skipping synthetic generation.")
                return {"status": "skipped", "reason": "script_not_found"}

            # Import synthetic generator
            from generate_500k_synthetic import SyntheticDocumentGenerator

            generator = SyntheticDocumentGenerator(
                output_dir=str(self.synthetic_dir),
                num_documents=count
            )

            # Generate documents
            logger.info("Generating synthetic documents...")
            documents = generator.generate_all()

            logger.info(f"Generated {len(documents):,} synthetic documents")

            # Save to disk
            output_file = self.synthetic_dir / "documents.jsonl"
            with open(output_file, 'w') as f:
                for doc in tqdm(documents, desc="Saving synthetic documents"):
                    f.write(json.dumps(doc) + '\n')

            self.stats["synthetic_generated"] = len(documents)
            duration = (datetime.now() - start_time).total_seconds()

            logger.info(f"Synthetic generation completed in {duration:.2f}s")
            logger.info(f"Saved to {output_file}")

            return {
                "status": "completed",
                "count": len(documents),
                "duration": duration,
                "output_file": str(output_file)
            }

        except ImportError as e:
            logger.error(f"Synthetic generator import failed: {e}")
            return {"status": "error", "reason": "generator_not_found"}
        except Exception as e:
            logger.error(f"Synthetic generation failed: {e}", exc_info=True)
            return {"status": "error", "reason": str(e)}

    def merge_and_validate(self) -> Dict[str, Any]:
        """
        Merge all collected data and perform validation.

        Steps:
        1. Load all collected documents
        2. Deduplicate
        3. Quality validation
        4. Train/val/test split (80/10/10)
        5. Save final corpus
        """
        logger.info("=" * 80)
        logger.info("MERGING AND VALIDATING COLLECTED DATA")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Load all documents
        all_documents = []

        for source_dir, source_name in [
            (self.qorsync_dir, "QorSync"),
            (self.sap_hub_dir, "SAP Hub"),
            (self.public_dir, "Public"),
            (self.synthetic_dir, "Synthetic")
        ]:
            doc_file = source_dir / "documents.jsonl"
            if doc_file.exists():
                logger.info(f"Loading {source_name} documents...")
                with open(doc_file, 'r') as f:
                    docs = [json.loads(line) for line in f]
                    all_documents.extend(docs)
                    logger.info(f"  Loaded {len(docs):,} documents")

        logger.info(f"Total documents before deduplication: {len(all_documents):,}")

        # Deduplicate
        logger.info("Deduplicating documents...")
        seen_hashes = set()
        unique_documents = []

        for doc in tqdm(all_documents, desc="Deduplicating"):
            # Create hash from document content
            import hashlib
            doc_str = json.dumps(doc, sort_keys=True)
            doc_hash = hashlib.md5(doc_str.encode()).hexdigest()

            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_documents.append(doc)
            else:
                self.stats["duplicates_removed"] += 1

        logger.info(f"Documents after deduplication: {len(unique_documents):,}")
        logger.info(f"Duplicates removed: {self.stats['duplicates_removed']:,}")

        # Validate
        logger.info("Validating documents...")
        valid_documents = []

        for doc in tqdm(unique_documents, desc="Validating"):
            if self._validate_document(doc):
                valid_documents.append(doc)
            else:
                self.stats["validation_failures"] += 1

        logger.info(f"Valid documents: {len(valid_documents):,}")
        logger.info(f"Validation failures: {self.stats['validation_failures']:,}")

        # Split into train/val/test
        logger.info("Splitting into train/val/test sets...")
        import random
        random.shuffle(valid_documents)

        n_train = int(len(valid_documents) * 0.8)
        n_val = int(len(valid_documents) * 0.1)

        train_docs = valid_documents[:n_train]
        val_docs = valid_documents[n_train:n_train + n_val]
        test_docs = valid_documents[n_train + n_val:]

        logger.info(f"Train: {len(train_docs):,}")
        logger.info(f"Val: {len(val_docs):,}")
        logger.info(f"Test: {len(test_docs):,}")

        # Save splits
        for split, docs in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
            output_file = self.output_dir / f"{split}.jsonl"
            with open(output_file, 'w') as f:
                for doc in tqdm(docs, desc=f"Saving {split} split"):
                    f.write(json.dumps(doc) + '\n')
            logger.info(f"Saved {split} split to {output_file}")

        self.stats["total_documents"] = len(valid_documents)
        duration = (datetime.now() - start_time).total_seconds()

        # Save statistics
        stats_file = self.output_dir / "collection_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_file}")

        logger.info(f"Merge and validation completed in {duration:.2f}s")

        return {
            "status": "completed",
            "total_documents": len(valid_documents),
            "train_size": len(train_docs),
            "val_size": len(val_docs),
            "test_size": len(test_docs),
            "duration": duration
        }

    def _validate_document(self, doc: Dict[str, Any]) -> bool:
        """
        Validate a single document.

        Checks:
        - Has required fields
        - Image exists and is readable
        - OCR text is not empty
        - Annotations are present
        """
        required_fields = ["doc_id", "doc_type", "image_path", "ocr_text", "annotations"]

        for field in required_fields:
            if field not in doc:
                return False

        # Check OCR text is not empty
        if not doc["ocr_text"] or len(doc["ocr_text"].strip()) == 0:
            return False

        # Check image exists
        if doc.get("image_path") and not Path(doc["image_path"]).exists():
            return False

        return True

    def collect_all(self) -> Dict[str, Any]:
        """Collect from all sources."""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE DATA COLLECTION PIPELINE")
        logger.info("=" * 80)

        overall_start = datetime.now()

        # Collect from all sources
        qorsync_results = self.collect_qorsync_data()
        sap_hub_results = self.collect_sap_hub_data()
        public_results = self.collect_public_datasets()
        synthetic_results = self.generate_synthetic_data()

        # Merge and validate
        merge_results = self.merge_and_validate()

        overall_duration = (datetime.now() - overall_start).total_seconds()

        # Summary
        logger.info("=" * 80)
        logger.info("DATA COLLECTION PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total duration: {overall_duration:.2f}s ({overall_duration/3600:.2f}h)")
        logger.info(f"Total documents: {self.stats['total_documents']:,}")
        logger.info(f"  QorSync: {self.stats['qorsync_collected']:,}")
        logger.info(f"  SAP Hub: {self.stats['sap_hub_collected']:,}")
        logger.info(f"  Public: {self.stats['public_collected']:,}")
        logger.info(f"  Synthetic: {self.stats['synthetic_generated']:,}")
        logger.info(f"Duplicates removed: {self.stats['duplicates_removed']:,}")
        logger.info(f"Validation failures: {self.stats['validation_failures']:,}")

        return {
            "overall_duration": overall_duration,
            "qorsync": qorsync_results,
            "sap_hub": sap_hub_results,
            "public": public_results,
            "synthetic": synthetic_results,
            "merge": merge_results,
            "statistics": self.stats
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect training data for SAP_LLM")
    parser.add_argument(
        "--source",
        type=str,
        choices=["qorsync", "sap_hub", "public", "synthetic", "all"],
        default="all",
        help="Data source to collect from"
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Number of documents to collect (for synthetic)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/training_corpus",
        help="Output directory for collected data"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    # Initialize orchestrator
    orchestrator = DataCollectionOrchestrator(output_dir=args.output_dir)

    # Run collection
    try:
        if args.source == "qorsync":
            results = orchestrator.collect_qorsync_data()
        elif args.source == "sap_hub":
            results = orchestrator.collect_sap_hub_data()
        elif args.source == "public":
            results = orchestrator.collect_public_datasets()
        elif args.source == "synthetic":
            count = args.count or 500000
            results = orchestrator.generate_synthetic_data(count=count)
        else:  # all
            results = orchestrator.collect_all()

        logger.info("Data collection completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Data collection failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
