"""
Enhanced comprehensive unit tests for PMG (Process Memory Graph) components.

Tests cover:
- 768-dim embedding generation (<50ms)
- HNSW vector search (<100ms)
- Merkle versioning and temporal queries
- Redis caching
- PMG-Learning integration
- Data ingestion pipeline
"""

import pytest
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.pmg.vector_store import PMGVectorStore
from sap_llm.pmg.embedding_generator import EnhancedEmbeddingGenerator, EmbeddingConfig
from sap_llm.pmg.merkle_versioning import MerkleVersioning
from sap_llm.pmg.context_retriever import ContextRetriever, RetrievalConfig
from sap_llm.pmg.pmg_learning_integration import PMGLearningIntegration
from sap_llm.pmg.data_ingestion import PMGDataIngestion


@pytest.mark.unit
@pytest.mark.pmg
class TestEnhancedEmbeddingGenerator:
    """Tests for 768-dim embedding generation."""

    @pytest.fixture
    def embedding_gen(self):
        """Create embedding generator."""
        config = EmbeddingConfig(
            model_name="sentence-transformers/all-mpnet-base-v2",
            dimension=768
        )
        return EnhancedEmbeddingGenerator(config=config)

    def test_embedding_dimension(self, embedding_gen):
        """Test embedding dimension is 768."""
        text = "This is a test invoice from Acme Corp for $1000"
        embedding = embedding_gen.generate_embedding(text)

        assert embedding.shape == (768,), f"Expected 768-dim, got {embedding.shape}"
        assert isinstance(embedding, np.ndarray)

    def test_embedding_generation_speed(self, embedding_gen):
        """Test embedding generation is <50ms per document."""
        document = {
            "doc_type": "invoice",
            "supplier_name": "Acme Corp",
            "total_amount": 1000.00,
            "currency": "USD"
        }

        # Warmup
        embedding_gen.generate_document_embedding(document)

        # Time 10 generations
        times = []
        for _ in range(10):
            start = time.time()
            embedding_gen.generate_document_embedding(document)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

        avg_time_ms = np.mean(times)

        assert avg_time_ms < 50, f"Embedding took {avg_time_ms:.2f}ms, expected <50ms"

    def test_batch_embedding_generation(self, embedding_gen):
        """Test batch embedding generation for efficiency."""
        documents = [
            {
                "doc_type": "invoice",
                "supplier_name": f"Supplier {i}",
                "total_amount": 1000.00 * i,
                "currency": "USD"
            }
            for i in range(100)
        ]

        start = time.time()
        embeddings = embedding_gen.generate_document_batch_embeddings(documents)
        elapsed = time.time() - start

        assert embeddings.shape == (100, 768)
        assert elapsed < 5.0, f"Batch generation took {elapsed:.2f}s for 100 docs"

    def test_embedding_normalization(self, embedding_gen):
        """Test embeddings are normalized."""
        text = "Test document"
        embedding = embedding_gen.generate_embedding(text)

        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5), f"Embedding not normalized: norm={norm}"

    def test_embedding_consistency(self, embedding_gen):
        """Test same input produces same embedding."""
        text = "Test document"

        emb1 = embedding_gen.generate_embedding(text)
        emb2 = embedding_gen.generate_embedding(text)

        assert np.allclose(emb1, emb2), "Embeddings not consistent"

    def test_document_to_text_conversion(self, embedding_gen):
        """Test document to text conversion."""
        document = {
            "doc_type": "invoice",
            "doc_subtype": "standard",
            "supplier_name": "Acme Corp",
            "total_amount": 1000.00,
            "currency": "USD",
            "po_number": "PO-123456"
        }

        text = embedding_gen._document_to_text(document)

        assert "invoice" in text.lower()
        assert "Acme Corp" in text
        assert "1000" in text
        assert "USD" in text


@pytest.mark.unit
@pytest.mark.pmg
class TestPMGVectorStore:
    """Tests for HNSW vector store."""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create vector store with HNSW index."""
        return PMGVectorStore(
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            dimension=768,
            index_type="IndexHNSWFlat",
            storage_path=str(tmp_path / "vector_store")
        )

    def test_hnsw_index_creation(self, vector_store):
        """Test HNSW index is created."""
        assert vector_store.index is not None
        assert vector_store.dimension == 768

    def test_vector_search_speed(self, vector_store):
        """Test vector search is <100ms for top-5."""
        # Add 1000 documents
        documents = [
            (f"doc_{i}", {
                "doc_type": "invoice",
                "supplier_id": f"SUP-{i:04d}",
                "total_amount": 1000.00 * i,
                "currency": "USD"
            })
            for i in range(1000)
        ]

        vector_store.add_batch(documents)

        # Search
        query = {
            "doc_type": "invoice",
            "supplier_id": "SUP-0500",
            "total_amount": 5000.00,
            "currency": "USD"
        }

        # Warmup
        vector_store.search_by_document(query, k=5)

        # Time 10 searches
        times = []
        for _ in range(10):
            start = time.time()
            results = vector_store.search_by_document(query, k=5)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

        avg_time_ms = np.mean(times)

        assert avg_time_ms < 100, f"Search took {avg_time_ms:.2f}ms, expected <100ms"
        assert len(results) <= 5

    def test_vector_store_save_load(self, vector_store):
        """Test vector store persistence."""
        # Add documents
        documents = [
            (f"doc_{i}", {"doc_type": "invoice", "amount": 1000 * i})
            for i in range(100)
        ]

        vector_store.add_batch(documents)

        # Save
        vector_store.save()

        # Load in new instance
        new_store = PMGVectorStore(
            dimension=768,
            storage_path=vector_store.storage_path
        )

        assert new_store.size() == 100


@pytest.mark.unit
@pytest.mark.pmg
class TestMerkleVersioning:
    """Tests for Merkle tree versioning."""

    @pytest.fixture
    def versioning(self):
        """Create Merkle versioning system."""
        return MerkleVersioning()

    def test_version_creation(self, versioning):
        """Test creating document versions."""
        doc = {
            "doc_type": "invoice",
            "invoice_number": "INV-001",
            "total_amount": 1000.00
        }

        version = versioning.create_version(
            doc_id="doc_123",
            document=doc,
            change_summary="Initial version"
        )

        assert version.version_number == 1
        assert version.parent_hash is None
        assert len(version.version_hash) == 64  # SHA-256 hex

    def test_version_chain(self, versioning):
        """Test version chain integrity."""
        doc_v1 = {"amount": 1000}
        doc_v2 = {"amount": 1200}
        doc_v3 = {"amount": 1500}

        v1 = versioning.create_version("doc_123", doc_v1)
        v2 = versioning.create_version("doc_123", doc_v2)
        v3 = versioning.create_version("doc_123", doc_v3)

        assert v2.parent_hash == v1.version_hash
        assert v3.parent_hash == v2.version_hash
        assert versioning.verify_chain_integrity("doc_123")

    def test_temporal_queries(self, versioning):
        """Test as-of temporal queries."""
        # Create versions at different times
        doc_v1 = {"amount": 1000}

        v1 = versioning.create_version("doc_123", doc_v1)
        time_v1 = datetime.fromisoformat(v1.timestamp)

        # Wait and create v2
        time.sleep(0.1)
        doc_v2 = {"amount": 1200}
        v2 = versioning.create_version("doc_123", doc_v2)

        # Query as of v1 time
        version_at_v1 = versioning.get_version_at_time("doc_123", time_v1 + timedelta(seconds=0.05))

        assert version_at_v1.version_number == 1
        assert version_at_v1.document_data["amount"] == 1000

    def test_deduplication(self, versioning):
        """Test content-based deduplication."""
        doc = {"amount": 1000}

        v1 = versioning.create_version("doc_123", doc)
        v2 = versioning.create_version("doc_456", doc)  # Same content, different doc

        # Same content hash
        assert v1.version_hash == v2.version_hash

        stats = versioning.get_storage_stats()
        assert stats["unique_versions"] == 1
        assert stats["total_versions"] == 1


@pytest.mark.unit
@pytest.mark.pmg
class TestContextRetriever:
    """Tests for context retrieval with caching."""

    @pytest.fixture
    def retriever(self):
        """Create context retriever with mock Redis."""
        config = RetrievalConfig(
            enable_cache=False,  # Disable actual Redis for tests
            top_k=5,
            min_similarity=0.7
        )

        return ContextRetriever(config=config)

    def test_context_retrieval(self, retriever):
        """Test basic context retrieval."""
        document = {
            "doc_type": "invoice",
            "supplier_id": "SUP-001",
            "total_amount": 1000.00
        }

        contexts = retriever.retrieve_context(document, top_k=5)

        assert isinstance(contexts, list)

    def test_cache_key_generation(self, retriever):
        """Test cache key generation is consistent."""
        doc = {
            "doc_type": "invoice",
            "supplier_id": "SUP-001",
            "total_amount": 1000.00
        }

        key1 = retriever._generate_cache_key(doc, 5, 0.7)
        key2 = retriever._generate_cache_key(doc, 5, 0.7)

        assert key1 == key2

    def test_retrieval_stats(self, retriever):
        """Test retrieval statistics tracking."""
        document = {
            "doc_type": "invoice",
            "supplier_id": "SUP-001",
            "total_amount": 1000.00
        }

        # Retrieve context multiple times
        for _ in range(5):
            retriever.retrieve_context(document)

        stats = retriever.get_statistics()

        assert stats["total_retrievals"] == 5


@pytest.mark.unit
@pytest.mark.pmg
class TestProcessMemoryGraph:
    """Tests for PMG graph client with enhancements."""

    @pytest.fixture
    def pmg(self):
        """Create PMG client (mock mode)."""
        return ProcessMemoryGraph()

    def test_pmg_initialization(self, pmg):
        """Test PMG initializes in mock mode."""
        assert pmg.mock_mode is True
        assert pmg.versioning is not None
        assert pmg.embedding_gen is not None

    def test_store_transaction_with_embedding(self, pmg):
        """Test storing transaction generates embedding."""
        document = {
            "doc_type": "invoice",
            "supplier_id": "SUP-001",
            "total_amount": 1000.00
        }

        doc_id = pmg.store_transaction(document)

        assert doc_id is not None

        # Check version was created
        version = pmg.versioning.get_latest_version(doc_id)
        assert version is not None

    def test_temporal_query(self, pmg):
        """Test as-of temporal queries."""
        past_time = (datetime.now() - timedelta(days=7)).isoformat()

        results = pmg.query_documents_at_time(
            as_of_timestamp=past_time,
            doc_type="invoice",
            limit=10
        )

        assert isinstance(results, list)

    def test_pmg_statistics(self, pmg):
        """Test PMG statistics."""
        stats = pmg.get_pmg_statistics()

        assert "mode" in stats or "total_documents" in stats


@pytest.mark.unit
@pytest.mark.pmg
class TestPMGLearningIntegration:
    """Tests for PMG-Learning integration."""

    @pytest.fixture
    def integration(self):
        """Create PMG-Learning integration."""
        return PMGLearningIntegration(
            enable_auto_retrain=False  # Disable for tests
        )

    def test_process_prediction_with_context(self, integration):
        """Test processing prediction with PMG context."""
        document = {
            "doc_type": "invoice",
            "supplier_id": "SUP-001",
            "total_amount": 1000.00
        }

        prediction = {
            "doc_type": "invoice",
            "confidence": 0.75
        }

        result = integration.process_prediction(
            document=document,
            prediction=prediction,
            use_context=True
        )

        assert "prediction" in result
        assert "context_used" in result

    def test_outcome_storage(self, integration):
        """Test storing prediction outcomes."""
        document = {
            "doc_type": "invoice",
            "supplier_id": "SUP-001"
        }

        prediction = {
            "doc_type": "invoice",
            "confidence": 0.9
        }

        routing_decision = {
            "endpoint": "API_INVOICE_PROCESS",
            "method": "POST",
            "confidence": 0.9
        }

        sap_response = {
            "status_code": 200,
            "success": True
        }

        doc_id = integration.store_outcome(
            document=document,
            prediction=prediction,
            routing_decision=routing_decision,
            sap_response=sap_response
        )

        assert doc_id is not None

    def test_prediction_quality_analysis(self, integration):
        """Test prediction quality analysis."""
        document = {
            "doc_type": "invoice",
            "supplier_id": "SUP-001"
        }

        prediction = {
            "doc_type": "invoice",
            "confidence": 0.85
        }

        analysis = integration.analyze_prediction_quality(document, prediction)

        assert "confidence_level" in analysis
        assert "recommendation" in analysis

    def test_statistics_tracking(self, integration):
        """Test integration statistics."""
        stats = integration.get_statistics()

        assert "total_predictions" in stats
        assert "pmg" in stats


@pytest.mark.unit
@pytest.mark.pmg
class TestDataIngestion:
    """Tests for data ingestion pipeline."""

    @pytest.fixture
    def ingestion(self, tmp_path):
        """Create data ingestion pipeline."""
        return PMGDataIngestion(
            postgres_uri=None,  # Use mock data
            batch_size=50,
            output_dir=str(tmp_path / "pmg_data")
        )

    def test_mock_data_generation(self, ingestion):
        """Test mock data generation."""
        documents = ingestion._generate_mock_documents(count=1000)

        assert len(documents) == 1000
        assert all("doc_type" in doc for doc in documents)
        assert all("total_amount" in doc for doc in documents)

    def test_batch_ingestion(self, ingestion):
        """Test batch ingestion performance."""
        documents = ingestion._generate_mock_documents(count=100)

        start = time.time()
        stats = ingestion.ingest_documents(documents, show_progress=False)
        elapsed = time.time() - start

        assert stats["documents_ingested"] == 100
        assert stats["embeddings_generated"] == 100
        assert stats["avg_embedding_time_ms"] < 50
        assert elapsed < 10.0  # Should process 100 docs in <10s

    def test_verification_criteria(self, ingestion):
        """Test ingestion verification."""
        # Ingest small sample
        documents = ingestion._generate_mock_documents(count=100)
        ingestion.ingest_documents(documents, show_progress=False)

        # Note: Full verification requires 100K docs
        # This just tests the verification logic
        verification = ingestion.verify_ingestion()

        assert "success" in verification
        assert "criteria" in verification
        assert "document_count" in verification["criteria"]
        assert "embedding_time_ms" in verification["criteria"]


@pytest.mark.unit
@pytest.mark.pmg
@pytest.mark.performance
class TestPerformanceTargets:
    """Tests for performance targets from success criteria."""

    def test_embedding_generation_target_50ms(self):
        """Test: Embedding generation <50ms per document."""
        generator = EnhancedEmbeddingGenerator()

        document = {
            "doc_type": "invoice",
            "supplier_name": "Acme Corp",
            "total_amount": 1000.00,
            "currency": "USD"
        }

        # Warmup
        generator.generate_document_embedding(document)

        # Measure
        times = []
        for _ in range(20):
            start = time.time()
            generator.generate_document_embedding(document)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(f"\nEmbedding generation: avg={avg_time:.2f}ms, p95={p95_time:.2f}ms")

        assert avg_time < 50, f"Average time {avg_time:.2f}ms exceeds 50ms target"
        assert p95_time < 100, f"P95 time {p95_time:.2f}ms too high"

    def test_similarity_search_target_100ms(self, tmp_path):
        """Test: Similarity search <100ms for top-5."""
        vector_store = PMGVectorStore(
            dimension=768,
            index_type="IndexHNSWFlat",
            storage_path=str(tmp_path / "vs")
        )

        # Add 10K documents
        documents = [
            (f"doc_{i}", {
                "doc_type": "invoice",
                "supplier_id": f"SUP-{i:05d}",
                "total_amount": 1000.00 * i
            })
            for i in range(10000)
        ]

        print("\nAdding 10K documents to vector store...")
        vector_store.add_batch(documents)

        # Search
        query = {
            "doc_type": "invoice",
            "supplier_id": "SUP-05000",
            "total_amount": 5000000.00
        }

        # Warmup
        vector_store.search_by_document(query, k=5)

        # Measure
        times = []
        for _ in range(20):
            start = time.time()
            results = vector_store.search_by_document(query, k=5)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        print(f"Similarity search (10K docs): avg={avg_time:.2f}ms, p95={p95_time:.2f}ms")

        assert avg_time < 100, f"Average time {avg_time:.2f}ms exceeds 100ms target"
        assert len(results) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
