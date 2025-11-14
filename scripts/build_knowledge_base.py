#!/usr/bin/env python3
"""
Build SAP Knowledge Base

Crawls SAP API documentation and builds the knowledge base for
intelligent document routing and field mapping.

Usage:
    python scripts/build_knowledge_base.py [--mock] [--urls URL1 URL2 ...]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sap_llm.knowledge_base.crawler import SAPAPICrawler
from sap_llm.knowledge_base.storage import KnowledgeBaseStorage
from sap_llm.knowledge_base.query import KnowledgeBaseQuery
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


async def build_knowledge_base(
    mock: bool = False,
    urls: list = None,
    mongo_uri: str = None,
) -> None:
    """
    Build knowledge base by crawling SAP API documentation.

    Args:
        mock: Use mock data instead of real crawling
        urls: List of URLs to crawl
        mongo_uri: MongoDB connection URI
    """
    logger.info("=" * 80)
    logger.info("Building SAP Knowledge Base")
    logger.info("=" * 80)

    # Initialize crawler
    crawler = SAPAPICrawler(base_urls=urls)

    # Crawl data
    if mock:
        logger.info("Using mock data...")
        data = await crawler.crawl_mock_data()
    else:
        logger.info("Crawling SAP API documentation...")
        data = await crawler.crawl_all()

    # Initialize storage
    storage = KnowledgeBaseStorage(mongo_uri=mongo_uri)

    # Store data
    logger.info("Storing knowledge base data...")
    counts = storage.store_crawled_data(data)

    logger.info(f"\nStored:")
    logger.info(f"  - {counts['api_schemas']} API schemas")
    logger.info(f"  - {counts['field_mappings']} field mappings")
    logger.info(f"  - {counts['business_rules']} business rules")
    logger.info(f"  - {counts['examples']} examples")

    # Test queries
    logger.info("\n" + "=" * 80)
    logger.info("Testing Knowledge Base Queries")
    logger.info("=" * 80)

    query_engine = KnowledgeBaseQuery(storage)

    # Test 1: Find API for Purchase Order
    logger.info("\nTest 1: Find API for Purchase Order")
    apis = query_engine.find_api_for_document("purchase_order")
    if apis:
        logger.info(f"  Found: {apis[0].get('title')} (similarity: {apis[0].get('similarity', 0):.3f})")
    else:
        logger.info("  No APIs found")

    # Test 2: Map fields
    logger.info("\nTest 2: Map ADC fields to SAP")
    test_adc = {
        "po_number": "4500123456",
        "vendor_id": "100001",
        "total_amount": 1234.56,
    }
    mapping_result = query_engine.map_fields_to_sap(test_adc, "purchase_order")
    logger.info(f"  Mapped {len(mapping_result['mappings'])} fields")
    for adc_field, sap_field in mapping_result['mappings'].items():
        logger.info(f"    {adc_field} -> {sap_field}")

    # Test 3: Find validation rules
    logger.info("\nTest 3: Find validation rules")
    rules = query_engine.find_validation_rules("purchase_order")
    logger.info(f"  Found {len(rules)} validation rules")
    for rule in rules[:3]:
        logger.info(f"    - {rule.get('description')}")

    # Test 4: Get endpoint for action
    logger.info("\nTest 4: Get endpoint for creating Purchase Order")
    endpoint = query_engine.get_endpoint_for_action("purchase_order", "create")
    if endpoint:
        logger.info(f"  Endpoint: {endpoint.get('method')} {endpoint.get('endpoint')}")
    else:
        logger.info("  No endpoint found")

    # Get stats
    logger.info("\n" + "=" * 80)
    logger.info("Knowledge Base Statistics")
    logger.info("=" * 80)
    stats = query_engine.get_stats()
    logger.info(f"  Total items: {stats['total_items']}")
    logger.info(f"  API schemas: {stats['storage']['api_schemas']}")
    logger.info(f"  Field mappings: {stats['storage']['field_mappings']}")
    logger.info(f"  Business rules: {stats['storage']['business_rules']}")

    logger.info("\n" + "=" * 80)
    logger.info("Knowledge Base build complete!")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build SAP Knowledge Base from API documentation"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of real crawling",
    )
    parser.add_argument(
        "--urls",
        nargs="+",
        help="List of URLs to crawl",
    )
    parser.add_argument(
        "--mongo-uri",
        help="MongoDB connection URI",
    )

    args = parser.parse_args()

    # Run build
    asyncio.run(
        build_knowledge_base(
            mock=args.mock,
            urls=args.urls,
            mongo_uri=args.mongo_uri,
        )
    )


if __name__ == "__main__":
    main()
