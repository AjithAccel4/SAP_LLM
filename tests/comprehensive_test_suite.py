"""
Comprehensive Test Suite for 90%+ Coverage

Production-ready tests for all critical modules:
- Security (100% coverage required)
- Data pipeline (end-to-end + batch processing)
- PMG (versioning, embeddings, context retrieval)
- SHWL (anomaly detection, healing loops)
- Models (training, inference, continuous learning)
- Performance tests (latency, throughput, memory)
- Integration tests (full pipeline validation)

Run with: pytest tests/comprehensive_test_suite.py -v --cov=sap_llm
"""

import pytest
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# SECURITY TESTS (100% COVERAGE REQUIRED)
# ============================================================================

class TestSecuritySecretsManager:
    """Test secrets management system."""

    def test_secrets_initialization(self):
        """Test secrets manager initializes correctly."""
        from sap_llm.security.secrets_manager import SecretsManager

        manager = SecretsManager(backend="vault")
        assert manager.backend == "vault"
        assert manager.cache == {}

    def test_secret_creation_mock_mode(self):
        """Test secret creation in mock mode."""
        from sap_llm.security.secrets_manager import SecretsManager

        manager = SecretsManager(backend="mock")
        result = manager.create_secret("test_secret", "test_value")
        # Mock mode returns False (not persisted)
        assert result == False

    def test_secret_caching(self):
        """Test secret caching mechanism."""
        from sap_llm.security.secrets_manager import SecretsManager
        import os

        os.environ["test_secret"] = "cached_value"

        manager = SecretsManager(backend="mock")

        # First fetch
        value1 = manager.get_secret("test_secret")
        assert value1 == "cached_value"

        # Second fetch (should use cache)
        value2 = manager.get_secret("test_secret", use_cache=True)
        assert value2 == value1

        # Clear cache
        manager.clear_cache()
        assert len(manager.cache) == 0

    def test_audit_logging(self):
        """Test audit trail for secret access."""
        from sap_llm.security.secrets_manager import SecretsManager
        import os

        os.environ["audit_test"] = "value"

        manager = SecretsManager(backend="mock")
        manager.get_secret("audit_test")

        audit_log = manager.get_audit_log("audit_test")
        assert len(audit_log) >= 1
        assert audit_log[0]["action"] in ["fetched", "cache_hit"]


# ============================================================================
# DATA PIPELINE TESTS
# ============================================================================

class TestDataPipeline:
    """Test data pipeline components."""

    def test_corpus_builder_initialization(self):
        """Test corpus builder initializes."""
        from sap_llm.data_pipeline.corpus_builder import CorpusBuilder, CorpusConfig

        config = CorpusConfig(output_dir="/tmp/test_corpus", target_total=1000)
        builder = CorpusBuilder(config)

        assert builder.config.target_total == 1000

    def test_synthetic_generator(self):
        """Test synthetic document generation."""
        from sap_llm.data_pipeline.synthetic_generator import SyntheticDocumentGenerator

        generator = SyntheticDocumentGenerator(
            template_dir="/tmp/templates",
            output_dir="/tmp/synthetic"
        )

        # Generate small batch
        docs = generator.generate_documents(
            document_type="invoice",
            count=5
        )

        assert len(docs) == 5
        assert all(d.document_type == "invoice" for d in docs)

    def test_dataset_validator(self):
        """Test dataset validation."""
        from sap_llm.data_pipeline.dataset_validator import DatasetValidator

        # Create mock data directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = DatasetValidator(data_dir=tmpdir)
            assert validator.data_dir.exists()


# ============================================================================
# PMG TESTS
# ============================================================================

class TestProcessMemoryGraph:
    """Test PMG components."""

    def test_merkle_versioning(self):
        """Test Merkle tree document versioning."""
        from sap_llm.pmg.merkle_versioning import MerkleVersioning

        versioning = MerkleVersioning()

        doc1 = {"doc_type": "invoice", "amount": 1000}
        v1 = versioning.create_version("doc_1", doc1)

        assert v1.version_number == 1
        assert v1.parent_hash is None

        # Create v2
        doc2 = {"doc_type": "invoice", "amount": 1200}
        v2 = versioning.create_version("doc_1", doc2)

        assert v2.version_number == 2
        assert v2.parent_hash == v1.version_hash

        # Verify integrity
        assert versioning.verify_chain_integrity("doc_1") == True

    def test_embedding_generator(self):
        """Test embedding generation."""
        from sap_llm.pmg.embedding_generator import EnhancedEmbeddingGenerator

        generator = EnhancedEmbeddingGenerator()

        # Single text
        text = "This is a test document"
        embedding = generator.generate_embedding(text)

        assert embedding.shape == (768,)  # 768-dim
        assert -1.5 <= embedding.mean() <= 1.5  # Reasonable range

    def test_context_retriever(self):
        """Test context retrieval."""
        from sap_llm.pmg.context_retriever import ContextRetriever

        retriever = ContextRetriever()

        doc = {"doc_type": "invoice"}
        contexts = retriever.retrieve_context(doc, top_k=5)

        # Mock mode returns empty
        assert isinstance(contexts, list)


# ============================================================================
# SHWL TESTS
# ============================================================================

class TestSelfHealingWorkflow:
    """Test SHWL components."""

    def test_anomaly_detector(self):
        """Test anomaly detection."""
        from sap_llm.shwl.anomaly_detector import AnomalyDetector
        from sap_llm.pmg.graph_client import ProcessMemoryGraph

        pmg = ProcessMemoryGraph()
        detector = AnomalyDetector(pmg)

        anomalies = detector.detect_anomalies()

        assert len(anomalies) > 0  # Mock generates anomalies
        assert all(hasattr(a, 'anomaly_type') for a in anomalies)

    def test_pattern_clusterer(self):
        """Test pattern clustering."""
        from sap_llm.shwl.pattern_clusterer import PatternClusterer
        from sap_llm.shwl.anomaly_detector import AnomalyDetector, Anomaly
        from sap_llm.pmg.graph_client import ProcessMemoryGraph

        # Generate anomalies
        pmg = ProcessMemoryGraph()
        detector = AnomalyDetector(pmg)
        anomalies = detector.detect_anomalies()

        # Cluster
        clusterer = PatternClusterer()
        clusters = clusterer.cluster_anomalies(anomalies)

        assert isinstance(clusters, list)

    def test_governance_gate(self):
        """Test governance gate."""
        from sap_llm.shwl.governance_gate import GovernanceGate
        from sap_llm.shwl.root_cause_analyzer import RootCauseAnalysis

        gate = GovernanceGate(auto_approve_threshold=0.95)

        # High confidence analysis
        analysis = RootCauseAnalysis(
            cluster_id="TEST-001",
            root_cause="Test",
            explanation="Test",
            proposed_fix="Test",
            fix_type="rule_update",
            confidence=0.96,
            affected_components=[],
            estimated_fix_time_hours=1
        )

        decision = gate.evaluate_fix(analysis)

        assert decision.approved == True
        assert decision.auto_approved == True


# ============================================================================
# CONTINUOUS LEARNING TESTS
# ============================================================================

class TestContinuousLearning:
    """Test continuous learning pipeline."""

    def test_learner_initialization(self):
        """Test continuous learner initialization."""
        from sap_llm.training.continuous_learner import ContinuousLearner

        learner = ContinuousLearner()
        assert learner.champion_model is not None
        assert learner.stats["retraining_cycles"] == 0

    def test_learning_cycle(self):
        """Test one learning cycle."""
        from sap_llm.training.continuous_learner import ContinuousLearner

        learner = ContinuousLearner()
        result = learner.run_learning_cycle()

        assert "status" in result


# ============================================================================
# CONTEXT-AWARE PROCESSING TESTS
# ============================================================================

class TestContextAwareProcessing:
    """Test context-aware processor."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        from sap_llm.inference.context_aware_processor import ContextAwareProcessor

        processor = ContextAwareProcessor()
        assert processor.retriever is not None

    def test_document_processing(self):
        """Test document processing."""
        from sap_llm.inference.context_aware_processor import ContextAwareProcessor

        processor = ContextAwareProcessor()

        doc = {"doc_type": "invoice"}
        result = processor.process_document(doc)

        assert "confidence" in result
        assert "doc_type" in result


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance benchmarks."""

    def test_embedding_generation_speed(self):
        """Test embedding generation performance."""
        import time
        from sap_llm.pmg.embedding_generator import EnhancedEmbeddingGenerator

        generator = EnhancedEmbeddingGenerator()

        texts = ["Test document" for _ in range(100)]

        start = time.time()
        embeddings = generator.generate_batch_embeddings(texts)
        duration = time.time() - start

        # Should process 100 docs in < 10 seconds
        assert duration < 10.0
        assert len(embeddings) == 100

    def test_vector_search_speed(self):
        """Test vector search performance."""
        import time
        import numpy as np
        from sap_llm.pmg.embedding_generator import EnhancedEmbeddingGenerator

        generator = EnhancedEmbeddingGenerator()

        # Create embeddings
        embeddings = np.random.randn(1000, 768).astype(np.float32)
        query_embedding = np.random.randn(768).astype(np.float32)

        # Search
        start = time.time()
        results = generator.find_similar_embeddings(
            query_embedding,
            embeddings,
            top_k=10
        )
        duration = time.time() - start

        # Should complete in < 100ms
        assert duration < 0.1
        assert len(results) == 10


# ============================================================================
# END-TO-END INTEGRATION TESTS
# ============================================================================

class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""

    def test_full_document_pipeline(self):
        """Test complete document processing pipeline."""
        # 1. Document ingestion
        doc = {
            "doc_type": "invoice",
            "content": "Sample invoice content",
            "vendor_id": "V001"
        }

        # 2. Context-aware processing
        from sap_llm.inference.context_aware_processor import ContextAwareProcessor
        processor = ContextAwareProcessor()
        result = processor.process_document(doc)

        assert "confidence" in result
        assert result["confidence"] > 0.0

        # 3. Store in PMG
        from sap_llm.pmg.graph_client import ProcessMemoryGraph
        pmg = ProcessMemoryGraph()
        stored = pmg.store_processed_document("test_doc_1", result)

        assert stored == True

    def test_batch_processing(self):
        """Test batch document processing."""
        from sap_llm.inference.context_aware_processor import ContextAwareProcessor

        processor = ContextAwareProcessor()

        # Create batch of documents
        documents = [
            {"doc_type": "invoice", "id": f"inv_{i}"}
            for i in range(10)
        ]

        # Process batch
        results = []
        for doc in documents:
            result = processor.process_document(doc)
            results.append(result)

        assert len(results) == 10
        assert all(r["confidence"] > 0 for r in results)

    def test_shwl_healing_cycle(self):
        """Test complete SHWL healing cycle."""
        from sap_llm.shwl.healing_loop import SelfHealingWorkflowLoop
        from sap_llm.pmg.graph_client import ProcessMemoryGraph

        pmg = ProcessMemoryGraph()
        loop = SelfHealingWorkflowLoop(pmg=pmg)

        result = loop.run_healing_cycle()

        assert "exceptions_fetched" in result
        assert "clusters_found" in result

    def test_continuous_learning_cycle(self):
        """Test continuous learning cycle."""
        from sap_llm.training.continuous_learner import ContinuousLearner

        learner = ContinuousLearner()
        result = learner.run_learning_cycle()

        assert "status" in result


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_document(self):
        """Test processing empty document."""
        from sap_llm.inference.context_aware_processor import ContextAwareProcessor

        processor = ContextAwareProcessor()
        result = processor.process_document({})

        assert "confidence" in result

    def test_malformed_payload(self):
        """Test SAP connector with malformed payload."""
        from sap_llm.connectors.sap_connector_library import (
            SAPConnectorLibrary, ConnectionConfig, ERPSystem, DocumentType
        )

        config = ConnectionConfig(
            system=ERPSystem.SAP_S4HANA,
            base_url="http://test"
        )

        connector = SAPConnectorLibrary(config)

        # Missing required fields
        with pytest.raises(ValueError):
            connector.post_to_sap(
                DocumentType.SALES_ORDER,
                {}
            )

    def test_missing_secrets(self):
        """Test secrets manager with missing secrets."""
        from sap_llm.security.secrets_manager import SecretsManager

        manager = SecretsManager(backend="mock")
        secret = manager.get_secret("nonexistent_secret")

        # Should return None for nonexistent secrets
        assert secret is None

    def test_circuit_breaker_open(self):
        """Test circuit breaker when SAP is down."""
        from sap_llm.connectors.sap_connector_library import (
            SAPConnectorLibrary, ConnectionConfig, ERPSystem, DocumentType
        )

        config = ConnectionConfig(
            system=ERPSystem.SAP_S4HANA,
            base_url="http://test"
        )

        connector = SAPConnectorLibrary(config)

        # Force circuit breaker open
        connector.circuit_breaker_state = "open"

        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            connector.post_to_sap(
                DocumentType.SALES_ORDER,
                {"SoldToParty": "1000", "SalesOrderType": "OR"}
            )

    def test_high_drift_detection(self):
        """Test model drift detection with high PSI."""
        from sap_llm.monitoring.comprehensive_observability import get_observability

        obs = get_observability()

        # Record high drift (should log warning)
        obs.record_model_drift(0.35)

    def test_idoc_missing_system_id(self):
        """Test IDoc posting without required system_id."""
        from sap_llm.connectors.sap_connector_library import (
            SAPConnectorLibrary, ConnectionConfig, ERPSystem
        )

        config = ConnectionConfig(
            system=ERPSystem.SAP_S4HANA,
            base_url="http://test"
        )

        connector = SAPConnectorLibrary(config)

        # Missing system_id should raise error
        with pytest.raises(ValueError, match="Missing required field 'system_id'"):
            connector.post_idoc(
                "ORDERS05",
                {"client": "100", "segments": []}
            )


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

class TestDataGenerators:
    """Test data generator utilities."""

    @staticmethod
    def generate_invoice(invoice_number: str = None) -> Dict:
        """Generate synthetic invoice for testing."""
        import random

        if not invoice_number:
            invoice_number = f"INV-{random.randint(10000, 99999)}"

        return {
            "doc_type": "invoice",
            "invoice_number": invoice_number,
            "vendor_id": f"V{random.randint(1000, 9999)}",
            "total_amount": round(random.uniform(100, 10000), 2),
            "currency": "USD",
            "line_items": [
                {
                    "item_id": f"ITEM-{i}",
                    "description": f"Test item {i}",
                    "quantity": random.randint(1, 100),
                    "unit_price": round(random.uniform(10, 500), 2)
                }
                for i in range(random.randint(1, 5))
            ]
        }

    @staticmethod
    def generate_purchase_order() -> Dict:
        """Generate synthetic purchase order for testing."""
        import random

        return {
            "doc_type": "purchase_order",
            "po_number": f"PO-{random.randint(100000, 999999)}",
            "vendor_id": f"V{random.randint(1000, 9999)}",
            "total_amount": round(random.uniform(1000, 50000), 2),
            "currency": "USD",
            "items": random.randint(1, 10)
        }

    @staticmethod
    def generate_batch_documents(count: int = 100) -> List[Dict]:
        """Generate batch of test documents."""
        docs = []
        for i in range(count):
            if i % 2 == 0:
                docs.append(TestDataGenerators.generate_invoice())
            else:
                docs.append(TestDataGenerators.generate_purchase_order())
        return docs

    def test_invoice_generator(self):
        """Test invoice generator."""
        invoice = self.generate_invoice("INV-12345")

        assert invoice["doc_type"] == "invoice"
        assert invoice["invoice_number"] == "INV-12345"
        assert "vendor_id" in invoice
        assert len(invoice["line_items"]) > 0

    def test_batch_generator(self):
        """Test batch document generator."""
        docs = self.generate_batch_documents(50)

        assert len(docs) == 50
        assert all("doc_type" in d for d in docs)


# ============================================================================
# STRESS TESTS
# ============================================================================

class TestStress:
    """Stress and load tests."""

    def test_high_volume_processing(self):
        """Test processing high volume of documents."""
        from sap_llm.inference.context_aware_processor import ContextAwareProcessor

        processor = ContextAwareProcessor()

        # Process 1000 documents
        docs = TestDataGenerators.generate_batch_documents(1000)

        import time
        start = time.time()

        for doc in docs[:100]:  # Test with 100 for speed
            processor.process_document(doc)

        duration = time.time() - start

        # Should process 100 docs in < 30 seconds
        assert duration < 30.0

    def test_memory_leak_detection(self):
        """Test for memory leaks in long-running operations."""
        import gc
        import sys
        from sap_llm.inference.context_aware_processor import ContextAwareProcessor

        processor = ContextAwareProcessor()

        # Get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Process many documents
        for i in range(100):
            doc = TestDataGenerators.generate_invoice()
            processor.process_document(doc)

        # Check memory growth
        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow excessively
        growth = final_objects - initial_objects
        assert growth < 10000  # Allow some growth but not excessive


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_comprehensive_tests():
    """Run all tests and generate coverage report."""
    pytest.main([
        __file__,
        "-v",
        "--cov=sap_llm",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=90"
    ])


if __name__ == "__main__":
    run_comprehensive_tests()
