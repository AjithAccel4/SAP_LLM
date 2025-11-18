#!/usr/bin/env python3
"""
Master Data Collection and Generation Script for SAP_LLM.

This script orchestrates the complete data collection pipeline:
1. Generate 500K synthetic documents (immediate start)
2. Extract from QorSync PostgreSQL (if configured)
3. Scrape SAP Business Accelerator Hub
4. Download public datasets
5. Organize into train/val/test splits
6. Generate quality statistics

Usage:
    # Full pipeline
    python scripts/run_data_collection.py --all

    # Synthetic data only (quick start)
    python scripts/run_data_collection.py --synthetic-only --count 500000

    # Real data only
    python scripts/run_data_collection.py --real-only
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sap_llm.data_pipeline.corpus_builder import CorpusBuilder, CorpusConfig
from sap_llm.data_pipeline.synthetic_generator import SyntheticDocumentGenerator
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"data_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def generate_synthetic_data(output_dir: Path, count: int = 500_000):
    """
    Generate synthetic training data.

    Args:
        output_dir: Directory to save generated documents
        count: Number of documents to generate
    """
    logger.info("="*80)
    logger.info("STEP 1: SYNTHETIC DATA GENERATION")
    logger.info("="*80)
    logger.info(f"Target: {count:,} documents")

    # Initialize generator
    generator = SyntheticDocumentGenerator(
        template_dir=str(output_dir / "templates"),
        output_dir=str(output_dir / "raw" / "synthetic"),
        languages=["en", "de", "es", "fr"]
    )

    # Document type distribution
    distribution = {
        "invoice": int(count * 0.30),  # 150K
        "purchase_order": int(count * 0.30),  # 150K
        "goods_receipt": int(count * 0.20),  # 100K
        "sales_order": int(count * 0.10),  # 50K
        "delivery_note": int(count * 0.05),  # 25K
        "material_document": int(count * 0.05),  # 25K
    }

    logger.info(f"Distribution: {distribution}")

    # Generate documents by type
    all_documents = []
    for doc_type, num_docs in distribution.items():
        logger.info(f"\nGenerating {num_docs:,} {doc_type} documents...")
        try:
            docs = generator.generate_documents(
                document_type=doc_type,
                count=num_docs,
                include_variations=True
            )
            all_documents.extend(docs)
            logger.info(f"✅ Generated {len(docs):,} {doc_type} documents")
        except Exception as e:
            logger.error(f"❌ Error generating {doc_type}: {e}")

    # Save metadata
    metadata_file = output_dir / "raw" / "synthetic" / "metadata.json"
    import json
    metadata = {
        "total_generated": len(all_documents),
        "distribution": distribution,
        "generation_time": datetime.now().isoformat(),
        "documents": [vars(doc) for doc in all_documents]
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ Synthetic data generation complete!")
    logger.info(f"Total documents: {len(all_documents):,}")
    logger.info(f"Metadata saved to: {metadata_file}")

    return all_documents


def download_public_datasets(output_dir: Path):
    """
    Download and process public datasets.

    Datasets:
    - RVL-CDIP: 100K documents (sampled from 400K)
    - CORD: 11K receipts
    - FUNSD: 200 forms
    - SROIE: 1K receipts
    """
    logger.info("="*80)
    logger.info("STEP 2: PUBLIC DATASET DOWNLOAD")
    logger.info("="*80)

    public_dir = output_dir / "raw" / "public"
    public_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sap_llm.data_pipeline.public_datasets_downloader import PublicDatasetDownloader

        downloader = PublicDatasetDownloader(str(public_dir))

        # Download datasets
        logger.info("Downloading RVL-CDIP (sampled)...")
        downloader.download_rvl_cdip(sample_size=100_000)

        logger.info("Downloading CORD...")
        downloader.download_cord()

        logger.info("Downloading FUNSD...")
        downloader.download_funsd()

        logger.info("Downloading SROIE...")
        downloader.download_sroie()

        logger.info("✅ Public dataset download complete!")

    except ImportError as e:
        logger.warning(f"⚠️  Public dataset downloader not available: {e}")
        logger.info("Skipping public datasets for now...")
    except Exception as e:
        logger.error(f"❌ Error downloading public datasets: {e}")


def extract_qorsync_data(output_dir: Path):
    """
    Extract documents from QorSync PostgreSQL database.

    Requires:
    - QORSYNC_DB_URI environment variable
    - Database access
    """
    logger.info("="*80)
    logger.info("STEP 3: QORSYNC POSTGRESQL EXTRACTION")
    logger.info("="*80)

    import os
    db_uri = os.getenv("QORSYNC_DB_URI")

    if not db_uri:
        logger.warning("⚠️  QORSYNC_DB_URI not set. Skipping PostgreSQL extraction.")
        logger.info("To enable: export QORSYNC_DB_URI='postgresql://user:pass@host:port/db'")
        return []

    try:
        from sap_llm.data_pipeline.collector import DocumentCollector

        collector = DocumentCollector(
            output_dir=str(output_dir / "raw" / "qorsync"),
            max_workers=8
        )

        logger.info("Extracting from PostgreSQL...")
        documents = collector.collect_from_postgres(
            db_uri=db_uri,
            tables=["documents", "extractions", "validations"],
            date_range_years=3,
            target_count=300_000
        )

        logger.info(f"✅ Extracted {len(documents):,} documents from QorSync")
        return documents

    except ImportError as e:
        logger.warning(f"⚠️  DocumentCollector not available: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Error extracting from PostgreSQL: {e}")
        return []


def scrape_sap_hub(output_dir: Path):
    """
    Scrape SAP Business Accelerator Hub for API schemas and examples.

    Target: 200K documents
    """
    logger.info("="*80)
    logger.info("STEP 4: SAP BUSINESS ACCELERATOR HUB SCRAPING")
    logger.info("="*80)

    try:
        from sap_llm.knowledge_base.crawler import SAPAcceleratorHubCrawler

        crawler = SAPAcceleratorHubCrawler(
            output_dir=str(output_dir / "raw" / "sap_hub"),
            rate_limit=10  # requests per second
        )

        logger.info("Scraping SAP API schemas...")
        documents = crawler.scrape_api_schemas(target_count=200_000)

        logger.info(f"✅ Scraped {len(documents):,} documents from SAP Hub")
        return documents

    except ImportError as e:
        logger.warning(f"⚠️  SAP Hub crawler not available: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Error scraping SAP Hub: {e}")
        return []


def create_data_splits(output_dir: Path):
    """
    Organize all collected data into train/val/test splits.

    Split: 70% train / 15% val / 15% test
    """
    logger.info("="*80)
    logger.info("STEP 5: DATA SPLITTING")
    logger.info("="*80)

    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create splits directories
    (processed_dir / "train").mkdir(exist_ok=True)
    (processed_dir / "val").mkdir(exist_ok=True)
    (processed_dir / "test").mkdir(exist_ok=True)

    import json
    import random
    from pathlib import Path

    # Collect all documents
    all_docs = []
    for source_dir in raw_dir.iterdir():
        if source_dir.is_dir():
            logger.info(f"Processing {source_dir.name}...")
            # Load metadata if available
            metadata_file = source_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    docs = metadata.get("documents", [])
                    all_docs.extend([(doc, source_dir.name) for doc in docs])

    logger.info(f"Total documents collected: {len(all_docs):,}")

    # Shuffle and split
    random.shuffle(all_docs)

    total = len(all_docs)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_docs = all_docs[:train_end]
    val_docs = all_docs[train_end:val_end]
    test_docs = all_docs[val_end:]

    # Save splits
    for split_name, split_docs in [("train", train_docs), ("val", val_docs), ("test", test_docs)]:
        split_file = processed_dir / split_name / "metadata.json"
        with open(split_file, 'w') as f:
            json.dump({
                "count": len(split_docs),
                "documents": [doc[0] for doc in split_docs]
            }, f, indent=2)
        logger.info(f"  {split_name}: {len(split_docs):,} documents")

    # Save split info
    split_info_file = output_dir / "metadata" / "splits.json"
    split_info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_info_file, 'w') as f:
        json.dump({
            "total": total,
            "train": len(train_docs),
            "val": len(val_docs),
            "test": len(test_docs),
            "split_time": datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"✅ Data splits created successfully!")
    logger.info(f"Split info saved to: {split_info_file}")


def generate_statistics(output_dir: Path):
    """Generate comprehensive statistics about the collected data."""
    logger.info("="*80)
    logger.info("STEP 6: STATISTICS GENERATION")
    logger.info("="*80)

    import json

    stats = {
        "collection_date": datetime.now().isoformat(),
        "sources": {},
        "document_types": {},
        "languages": {},
        "total_documents": 0,
        "splits": {}
    }

    # Collect from raw directories
    raw_dir = output_dir / "raw"
    for source_dir in raw_dir.iterdir():
        if source_dir.is_dir():
            metadata_file = source_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    count = metadata.get("total_generated", len(metadata.get("documents", [])))
                    stats["sources"][source_dir.name] = count
                    stats["total_documents"] += count

    # Load split info
    split_file = output_dir / "metadata" / "splits.json"
    if split_file.exists():
        with open(split_file) as f:
            stats["splits"] = json.load(f)

    # Save statistics
    stats_file = output_dir / "metadata" / "statistics.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("DATA COLLECTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total Documents: {stats['total_documents']:,}")
    logger.info("\nBy Source:")
    for source, count in stats["sources"].items():
        logger.info(f"  {source}: {count:,}")

    if stats["splits"]:
        logger.info("\nData Splits:")
        logger.info(f"  Train: {stats['splits'].get('train', 0):,}")
        logger.info(f"  Val: {stats['splits'].get('val', 0):,}")
        logger.info(f"  Test: {stats['splits'].get('test', 0):,}")

    logger.info(f"\n✅ Statistics saved to: {stats_file}")
    logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SAP_LLM Data Collection Pipeline")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--synthetic-only", action="store_true", help="Generate synthetic data only")
    parser.add_argument("--real-only", action="store_true", help="Collect real data only")
    parser.add_argument("--count", type=int, default=500_000, help="Number of synthetic docs to generate")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--skip-splits", action="store_true", help="Skip train/val/test splitting")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = setup_logging(output_dir / "logs")
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Output directory: {output_dir}")

    # Run pipeline
    start_time = datetime.now()

    if args.synthetic_only:
        # Quick start: synthetic data only
        generate_synthetic_data(output_dir, args.count)

    elif args.real_only:
        # Real data collection only
        extract_qorsync_data(output_dir)
        scrape_sap_hub(output_dir)
        download_public_datasets(output_dir)

    elif args.all:
        # Full pipeline
        generate_synthetic_data(output_dir, args.count)
        extract_qorsync_data(output_dir)
        scrape_sap_hub(output_dir)
        download_public_datasets(output_dir)

    else:
        logger.error("Please specify --all, --synthetic-only, or --real-only")
        sys.exit(1)

    # Always create splits and statistics (unless skipped)
    if not args.skip_splits:
        create_data_splits(output_dir)
        generate_statistics(output_dir)

    # Done
    duration = datetime.now() - start_time
    logger.info("\n" + "="*80)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Duration: {duration}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
