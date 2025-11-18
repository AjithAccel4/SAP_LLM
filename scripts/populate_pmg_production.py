#!/usr/bin/env python3
"""
PMG Production Data Population Script.

This script populates the Process Memory Graph with production-ready data
to enable continuous learning following PLAN_02.md Phase 5.

Fulfills TODO 5 requirements:
- Populates PMG with 100K+ documents
- Generates 768-dim embeddings (<50ms per document)
- Builds HNSW index for <100ms similarity search
- Enables temporal queries and Merkle versioning
- Integrates with intelligent learning loop

Usage:
    python scripts/populate_pmg_production.py --mode mock --count 100000
    python scripts/populate_pmg_production.py --mode postgres --uri postgresql://...
    python scripts/populate_pmg_production.py --verify-only
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sap_llm.pmg.data_ingestion import PMGDataIngestion
from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.pmg.vector_store import PMGVectorStore
from sap_llm.pmg.context_retriever import ContextRetriever
from sap_llm.pmg.pmg_learning_integration import PMGLearningIntegration
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class PMGProductionPopulator:
    """
    Production PMG population orchestrator.

    Coordinates:
    1. Data ingestion from multiple sources
    2. Embedding generation
    3. Vector index building
    4. Verification of success criteria
    5. Learning integration setup
    """

    def __init__(
        self,
        output_dir: str = "./pmg_production_data",
        postgres_uri: str = None,
        neo4j_uri: str = None
    ):
        """
        Initialize PMG populator.

        Args:
            output_dir: Output directory for PMG data
            postgres_uri: PostgreSQL connection string (optional)
            neo4j_uri: Neo4j connection URI (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.postgres_uri = postgres_uri
        self.neo4j_uri = neo4j_uri

        # Initialize components
        self.ingestion = None
        self.pmg = None
        self.vector_store = None
        self.context_retriever = None
        self.learning_integration = None

        # Results
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "statistics": {},
            "verification": {},
            "errors": []
        }

        logger.info(f"PMG Production Populator initialized: {output_dir}")

    def setup_components(self):
        """Initialize all PMG components."""
        logger.info("Setting up PMG components...")

        # PMG Graph Client
        self.pmg = ProcessMemoryGraph()

        # Vector Store with HNSW index
        self.vector_store = PMGVectorStore(
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            dimension=768,
            index_type="IndexHNSWFlat",
            storage_path=str(self.output_dir / "vector_store")
        )

        # Context Retriever with Redis caching
        self.context_retriever = ContextRetriever(
            graph_client=self.pmg,
            vector_store=self.vector_store
        )

        # Learning Integration
        self.learning_integration = PMGLearningIntegration(
            pmg=self.pmg,
            context_retriever=self.context_retriever,
            enable_auto_retrain=True
        )

        # Data Ingestion
        self.ingestion = PMGDataIngestion(
            postgres_uri=self.postgres_uri,
            neo4j_uri=self.neo4j_uri,
            batch_size=100,
            output_dir=str(self.output_dir)
        )

        logger.info("✓ All components initialized")

    def populate_from_mock(self, count: int = 100000):
        """
        Populate PMG with mock data for testing.

        Args:
            count: Number of documents to generate
        """
        logger.info(f"Populating PMG with {count:,} mock documents...")

        try:
            # Generate mock documents
            documents = self.ingestion._generate_mock_documents(count)

            # Ingest
            stats = self.ingestion.ingest_documents(documents, show_progress=True)

            self.results["statistics"]["ingestion"] = stats

            logger.info(f"✓ Populated {stats['documents_ingested']:,} documents")

        except Exception as e:
            logger.error(f"Failed to populate from mock data: {e}")
            self.results["errors"].append(str(e))
            raise

    def populate_from_postgres(self, limit: int = 100000):
        """
        Populate PMG from PostgreSQL database.

        Args:
            limit: Maximum documents to fetch
        """
        logger.info(f"Populating PMG from PostgreSQL (limit={limit:,})...")

        try:
            # Connect to PostgreSQL
            if not self.ingestion.connect_postgres():
                raise RuntimeError("Failed to connect to PostgreSQL")

            # Fetch documents
            documents = self.ingestion.fetch_documents_from_postgres(limit=limit)

            if not documents:
                raise RuntimeError("No documents fetched from PostgreSQL")

            # Ingest
            stats = self.ingestion.ingest_documents(documents, show_progress=True)

            self.results["statistics"]["ingestion"] = stats

            logger.info(f"✓ Populated {stats['documents_ingested']:,} documents")

        except Exception as e:
            logger.error(f"Failed to populate from PostgreSQL: {e}")
            self.results["errors"].append(str(e))
            raise

    def populate_from_neo4j(self):
        """Populate PMG with classification patterns from Neo4j."""
        logger.info("Fetching classification patterns from Neo4j...")

        try:
            # Connect to Neo4j
            if not self.ingestion.connect_neo4j():
                logger.warning("Failed to connect to Neo4j, skipping patterns")
                return

            # Fetch patterns
            patterns = self.ingestion.fetch_classification_patterns_from_neo4j()

            logger.info(f"✓ Fetched {len(patterns)} classification patterns")

            self.results["statistics"]["neo4j_patterns"] = len(patterns)

        except Exception as e:
            logger.warning(f"Failed to fetch Neo4j patterns: {e}")
            self.results["errors"].append(str(e))

    def verify_success_criteria(self) -> Dict[str, Any]:
        """
        Verify PMG population meets success criteria.

        Success Criteria (from TODO 5):
        - Documents in PMG: ≥100,000
        - Embedding generation: 768-dim, <50ms per document
        - Similarity search: <100ms for top-5 results
        - Context retrieval accuracy: ≥90% relevant
        - Storage efficiency: <10GB for 100K documents

        Returns:
            Verification results
        """
        logger.info("="*80)
        logger.info("VERIFYING SUCCESS CRITERIA")
        logger.info("="*80)

        verification = self.ingestion.verify_ingestion()

        # Print results
        logger.info(f"\nOverall Status: {'✓ PASS' if verification['success'] else '✗ FAIL'}\n")

        for criterion, result in verification["criteria"].items():
            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            logger.info(f"{criterion}: {status}")
            logger.info(f"  Required: {result['required']}")
            logger.info(f"  Actual:   {result['actual']}")
            logger.info("")

        logger.info("="*80)

        self.results["verification"] = verification

        return verification

    def test_continuous_learning(self):
        """Test PMG-Learning integration."""
        logger.info("Testing continuous learning integration...")

        try:
            # Test prediction with context
            test_document = {
                "doc_type": "invoice",
                "supplier_id": "SUP-0001",
                "supplier_name": "Acme Corp",
                "total_amount": 1000.00,
                "currency": "USD",
                "company_code": "1000"
            }

            test_prediction = {
                "doc_type": "invoice",
                "confidence": 0.75,
                "subtype": "standard_invoice"
            }

            # Process with PMG context
            result = self.learning_integration.process_prediction(
                document=test_document,
                prediction=test_prediction,
                use_context=True
            )

            # Analyze quality
            quality = self.learning_integration.analyze_prediction_quality(
                document=test_document,
                prediction=test_prediction
            )

            logger.info(f"✓ Context used: {result['context_used']}")
            logger.info(f"✓ Confidence boost: +{result['confidence_boost']:.3f}")
            logger.info(f"✓ Quality level: {quality['confidence_level']}")
            logger.info(f"✓ Recommendation: {quality['recommendation']}")

            self.results["statistics"]["learning_test"] = {
                "context_used": result["context_used"],
                "confidence_boost": result["confidence_boost"],
                "quality_level": quality["confidence_level"]
            }

        except Exception as e:
            logger.error(f"Learning integration test failed: {e}")
            self.results["errors"].append(str(e))

    def export_results(self):
        """Export results to JSON file."""
        output_file = self.output_dir / "population_results.json"

        # Add component statistics
        self.results["statistics"]["pmg"] = self.pmg.get_pmg_statistics()
        self.results["statistics"]["vector_store_size"] = self.vector_store.size()
        self.results["statistics"]["learning"] = self.learning_integration.get_statistics()

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"\n✓ Results exported to: {output_file}")

    def generate_summary_report(self):
        """Generate human-readable summary report."""
        logger.info("\n" + "="*80)
        logger.info("PMG PRODUCTION POPULATION SUMMARY")
        logger.info("="*80)

        # Overview
        logger.info("\n📊 OVERVIEW")
        logger.info(f"  Timestamp: {self.results['timestamp']}")
        logger.info(f"  Success: {'✓ YES' if self.results.get('verification', {}).get('success', False) else '✗ NO'}")
        logger.info(f"  Errors: {len(self.results['errors'])}")

        # Statistics
        if "ingestion" in self.results["statistics"]:
            stats = self.results["statistics"]["ingestion"]
            logger.info("\n📈 INGESTION STATISTICS")
            logger.info(f"  Documents ingested: {stats.get('documents_ingested', 0):,}")
            logger.info(f"  Embeddings generated: {stats.get('embeddings_generated', 0):,}")
            logger.info(f"  Avg embedding time: {stats.get('avg_embedding_time_ms', 0):.2f}ms")
            logger.info(f"  Total time: {stats.get('total_time_seconds', 0):.2f}s")
            logger.info(f"  Throughput: {stats.get('documents_per_second', 0):.2f} docs/sec")

        # Verification
        if "verification" in self.results:
            verify = self.results["verification"]
            logger.info("\n✅ VERIFICATION")

            for criterion, result in verify.get("criteria", {}).items():
                status = "✓" if result["passed"] else "✗"
                logger.info(f"  {status} {criterion}: {result['actual']} (req: {result['required']})")

        # Learning
        if "learning_test" in self.results["statistics"]:
            learn = self.results["statistics"]["learning_test"]
            logger.info("\n🎓 LEARNING INTEGRATION")
            logger.info(f"  Context used: {'✓ YES' if learn.get('context_used') else '✗ NO'}")
            logger.info(f"  Confidence boost: +{learn.get('confidence_boost', 0):.3f}")
            logger.info(f"  Quality level: {learn.get('quality_level', 'unknown')}")

        # Errors
        if self.results["errors"]:
            logger.info("\n⚠️  ERRORS")
            for error in self.results["errors"]:
                logger.info(f"  - {error}")

        logger.info("\n" + "="*80)

    def run(self, mode: str, count: int = 100000):
        """
        Run PMG population.

        Args:
            mode: Population mode ('mock', 'postgres', 'neo4j', 'all')
            count: Document count (for mock mode)
        """
        start_time = time.time()

        try:
            # Setup components
            self.setup_components()

            # Populate based on mode
            if mode == "mock":
                self.populate_from_mock(count)

            elif mode == "postgres":
                self.populate_from_postgres(count)
                self.populate_from_neo4j()  # Also fetch patterns

            elif mode == "all":
                # Try PostgreSQL first, fallback to mock
                try:
                    self.populate_from_postgres(count)
                    self.populate_from_neo4j()
                except Exception as e:
                    logger.warning(f"PostgreSQL failed, using mock data: {e}")
                    self.populate_from_mock(count)

            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Verify success criteria
            verification = self.verify_success_criteria()
            self.results["success"] = verification["success"]

            # Test learning integration
            self.test_continuous_learning()

            # Export results
            self.export_results()

            # Generate summary
            self.generate_summary_report()

            # Calculate total time
            total_time = time.time() - start_time
            logger.info(f"\n✓ Population completed in {total_time:.2f}s")

            return self.results["success"]

        except Exception as e:
            logger.error(f"Population failed: {e}", exc_info=True)
            self.results["errors"].append(str(e))
            self.export_results()
            return False

        finally:
            # Cleanup
            if self.ingestion:
                self.ingestion.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Populate PMG with production data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Populate with 100K mock documents (fast, for testing)
  python scripts/populate_pmg_production.py --mode mock --count 100000

  # Populate from PostgreSQL database
  python scripts/populate_pmg_production.py --mode postgres --postgres-uri postgresql://...

  # Populate from all sources (PostgreSQL + Neo4j, fallback to mock)
  python scripts/populate_pmg_production.py --mode all --count 150000

  # Verify existing population
  python scripts/populate_pmg_production.py --verify-only --output-dir ./pmg_production_data
        """
    )

    parser.add_argument(
        "--mode",
        choices=["mock", "postgres", "neo4j", "all"],
        default="mock",
        help="Population mode"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=100000,
        help="Document count (default: 100000)"
    )

    parser.add_argument(
        "--output-dir",
        default="./pmg_production_data",
        help="Output directory"
    )

    parser.add_argument(
        "--postgres-uri",
        help="PostgreSQL connection URI"
    )

    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="Neo4j connection URI"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing population"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create populator
    populator = PMGProductionPopulator(
        output_dir=args.output_dir,
        postgres_uri=args.postgres_uri,
        neo4j_uri=args.neo4j_uri
    )

    # Run
    if args.verify_only:
        logger.info("Running verification only...")
        populator.setup_components()
        populator.verify_success_criteria()
        populator.export_results()
        success = True
    else:
        success = populator.run(mode=args.mode, count=args.count)

    # Exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
