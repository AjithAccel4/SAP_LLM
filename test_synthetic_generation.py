#!/usr/bin/env python3
"""
Quick test of synthetic document generation.
Generate a small batch to verify everything works before scaling to 500K.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sap_llm.data_pipeline.synthetic_generator import SyntheticDocumentGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Testing Synthetic Document Generator...")

    # Initialize generator
    generator = SyntheticDocumentGenerator(
        template_dir="./data/templates",
        output_dir="./data/test_synthetic",
        languages=["en"]
    )

    # Generate a small test batch
    logger.info("Generating 10 test invoices...")
    try:
        docs = generator.generate_documents(
            document_type="invoice",
            count=10,
            include_variations=True
        )
        logger.info(f"✅ Successfully generated {len(docs)} documents!")

        # Print first document details
        if docs:
            logger.info(f"\nSample document:")
            logger.info(f"  ID: {docs[0].id}")
            logger.info(f"  Type: {docs[0].document_type}")
            logger.info(f"  File: {docs[0].file_path}")
            logger.info(f"  Fields: {list(docs[0].fields.keys())[:5]}...")

        return True

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
