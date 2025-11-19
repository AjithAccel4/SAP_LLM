"""
Comprehensive unit tests for PMG (Process Memory Graph) modules.

Tests cover:
- Context retrieval
- Vector similarity search
- Cache management
- Statistics and export
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from sap_llm.pmg.context_retriever import (
    RetrievalConfig,
    ContextResult,
    ContextRetriever,
)


@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return {
        "doc_type": "invoice",
        "supplier_name": "Acme Corp",
        "supplier_id": "SUP-001",
        "total_amount": 1000.00,
        "currency": "USD",
        "company_code": "1000"
    }


@pytest.fixture
def sample_context_result():
    """Sample context result for testing."""
    return ContextResult(
        doc_id="doc-123",
        document={"doc_type": "invoice", "total_amount": 1000.00},
        similarity=0.95,
        routing_decision={"endpoint": "/api/invoices"},
        sap_response={"status_code": 200},
        exceptions=[],
        success=True,
        timestamp=datetime.now().isoformat(),
        recency_weight=0.9
    )


@pytest.fixture
def mock_graph_client():
    """Mock PMG graph client."""
    client = Mock()
    client.mock_mode = False
    client.find_similar_documents.return_value = []
    return client


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = Mock()
    store.search_by_document.return_value = []
    return store


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator."""
    gen = Mock()
    gen.generate_document_embedding.return_value = [0.1] * 768
    return gen


@pytest.mark.unit
class TestRetrievalConfig:
    """Tests for RetrievalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetrievalConfig()

        assert config.top_k == 5
        assert config.min_similarity == 0.7
        assert config.similarity_metric == "cosine"
        assert config.include_failures is True
        assert config.include_successes is True
        assert config.enable_cache is True
        assert config.cache_ttl_seconds == 3600

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetrievalConfig(
            top_k=10,
            min_similarity=0.8,
            time_window_days=30,
            weight_by_recency=False
        )

        assert config.top_k == 10
        assert config.min_similarity == 0.8
        assert config.time_window_days == 30
        assert config.weight_by_recency is False


@pytest.mark.unit
class TestContextResult:
    """Tests for ContextResult dataclass."""

    def test_creation(self, sample_context_result):
        """Test context result creation."""
        assert sample_context_result.doc_id == "doc-123"
        assert sample_context_result.similarity == 0.95
        assert sample_context_result.success is True

    def test_to_dict(self, sample_context_result):
        """Test conversion to dictionary."""
        d = asdict(sample_context_result)
        assert isinstance(d, dict)
        assert d["doc_id"] == "doc-123"
        assert d["similarity"] == 0.95


@pytest.mark.unit
class TestContextRetriever:
    """Tests for ContextRetriever class."""

    def test_initialization_default(self):
        """Test default initialization."""
        with patch('sap_llm.pmg.context_retriever.ProcessMemoryGraph'):
            with patch('sap_llm.pmg.context_retriever.PMGVectorStore'):
                with patch('sap_llm.pmg.context_retriever.EnhancedEmbeddingGenerator'):
                    retriever = ContextRetriever()

                    assert retriever.config is not None
                    assert retriever.stats["total_retrievals"] == 0

    def test_initialization_with_config(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test initialization with custom config."""
        config = RetrievalConfig(top_k=10, enable_cache=False)

        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=config
        )

        assert retriever.config.top_k == 10
        assert retriever.redis_client is None

    @patch('sap_llm.pmg.context_retriever.redis.Redis')
    def test_initialization_with_redis(
        self, mock_redis_class, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test initialization with Redis cache."""
        mock_redis = Mock()
        mock_redis_class.return_value = mock_redis

        config = RetrievalConfig(enable_cache=True)

        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=config
        )

        assert retriever.redis_client is not None

    def test_retrieve_context_no_results(
        self, sample_document, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test retrieval when no similar documents found."""
        mock_vector_store.search_by_document.return_value = []

        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=RetrievalConfig(enable_cache=False)
        )

        results = retriever.retrieve_context(sample_document)

        assert results == []
        assert retriever.stats["total_retrievals"] == 1

    def test_retrieve_context_with_results(
        self, sample_document, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test retrieval with similar documents."""
        mock_vector_store.search_by_document.return_value = [
            {
                "doc_id": "similar-1",
                "document": {
                    "doc_type": "invoice",
                    "sap_response": {"status_code": 200},
                    "processing_timestamp": datetime.now().isoformat()
                },
                "similarity": 0.95
            },
            {
                "doc_id": "similar-2",
                "document": {
                    "doc_type": "invoice",
                    "processing_timestamp": datetime.now().isoformat()
                },
                "similarity": 0.85
            }
        ]

        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=RetrievalConfig(enable_cache=False)
        )

        results = retriever.retrieve_context(sample_document, top_k=2)

        assert len(results) <= 2
        assert retriever.stats["total_retrievals"] == 1

    def test_retrieve_context_override_params(
        self, sample_document, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test retrieval with overridden parameters."""
        mock_vector_store.search_by_document.return_value = []

        config = RetrievalConfig(top_k=5, min_similarity=0.7)
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=config
        )

        retriever.retrieve_context(sample_document, top_k=10, min_similarity=0.5)

        # Should use overridden values
        mock_vector_store.search_by_document.assert_called_once()

    def test_determine_success_http_success(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test success determination with HTTP success."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        sap_response = {"status_code": 200}
        result = retriever._determine_success(sap_response, [])
        assert result is True

        sap_response = {"status_code": 201}
        result = retriever._determine_success(sap_response, [])
        assert result is True

    def test_determine_success_http_failure(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test success determination with HTTP failure."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        sap_response = {"status_code": 500}
        result = retriever._determine_success(sap_response, [])
        assert result is False

    def test_determine_success_critical_exception(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test success determination with critical exception."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        exceptions = [{"severity": "CRITICAL", "message": "Error"}]
        result = retriever._determine_success(None, exceptions)
        assert result is False

    def test_determine_success_no_response(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test success determination with no response."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        result = retriever._determine_success(None, [])
        assert result is True  # Default success if no response

    def test_calculate_recency_weight_recent(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test recency weight for recent document."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        recent_timestamp = datetime.now().isoformat()
        weight = retriever._calculate_recency_weight(recent_timestamp)

        assert weight > 0.9  # Recent should have high weight

    def test_calculate_recency_weight_old(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test recency weight for old document."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        old_timestamp = (datetime.now() - timedelta(days=60)).isoformat()
        weight = retriever._calculate_recency_weight(old_timestamp)

        assert weight < 0.5  # Old should have lower weight

    def test_calculate_recency_weight_disabled(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test recency weight when disabled."""
        config = RetrievalConfig(weight_by_recency=False)
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=config
        )

        weight = retriever._calculate_recency_weight(datetime.now().isoformat())
        assert weight == 1.0

    def test_apply_filters_success_only(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator, sample_context_result
    ):
        """Test filtering to include only successes."""
        config = RetrievalConfig(include_failures=False, include_successes=True)
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=config
        )

        # Create success and failure contexts
        success_context = sample_context_result
        failure_context = ContextResult(
            doc_id="doc-456",
            document={},
            similarity=0.9,
            routing_decision=None,
            sap_response=None,
            exceptions=[{"severity": "CRITICAL"}],
            success=False,
            timestamp=datetime.now().isoformat(),
            recency_weight=1.0
        )

        results = retriever._apply_filters([success_context, failure_context])
        assert len(results) == 1
        assert results[0].success is True

    def test_apply_filters_time_window(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test filtering by time window."""
        config = RetrievalConfig(time_window_days=7)
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=config
        )

        recent = ContextResult(
            doc_id="recent",
            document={},
            similarity=0.9,
            routing_decision=None,
            sap_response=None,
            exceptions=[],
            success=True,
            timestamp=datetime.now().isoformat(),
            recency_weight=1.0
        )

        old = ContextResult(
            doc_id="old",
            document={},
            similarity=0.9,
            routing_decision=None,
            sap_response=None,
            exceptions=[],
            success=True,
            timestamp=(datetime.now() - timedelta(days=30)).isoformat(),
            recency_weight=0.5
        )

        results = retriever._apply_filters([recent, old])
        assert len(results) == 1
        assert results[0].doc_id == "recent"

    def test_rank_contexts(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test context ranking."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        low_score = ContextResult(
            doc_id="low",
            document={},
            similarity=0.7,
            routing_decision=None,
            sap_response=None,
            exceptions=[],
            success=False,
            timestamp=datetime.now().isoformat(),
            recency_weight=0.5
        )

        high_score = ContextResult(
            doc_id="high",
            document={},
            similarity=0.95,
            routing_decision=None,
            sap_response=None,
            exceptions=[],
            success=True,
            timestamp=datetime.now().isoformat(),
            recency_weight=1.0
        )

        results = retriever._rank_contexts([low_score, high_score])

        # High score should be first
        assert results[0].doc_id == "high"

    def test_build_context_prompt_empty(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test prompt building with no contexts."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        prompt = retriever.build_context_prompt([])
        assert "No similar historical documents" in prompt

    def test_build_context_prompt_with_contexts(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator, sample_context_result
    ):
        """Test prompt building with contexts."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        prompt = retriever.build_context_prompt([sample_context_result])

        assert "Based on similar historical documents" in prompt
        assert "Successfully processed" in prompt

    def test_build_context_prompt_max_length(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test prompt truncation at max length."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        # Create context with long data
        context = ContextResult(
            doc_id="doc",
            document={
                "doc_type": "invoice",
                "supplier_name": "A" * 1000,
                "total_amount": 1000.00,
                "currency": "USD"
            },
            similarity=0.95,
            routing_decision={"endpoint": "/api/" + "x" * 1000},
            sap_response=None,
            exceptions=[],
            success=True,
            timestamp=datetime.now().isoformat(),
            recency_weight=1.0
        )

        prompt = retriever.build_context_prompt([context] * 10, max_length=500)
        assert len(prompt) <= 503  # 500 + "..."

    def test_generate_cache_key(
        self, sample_document, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test cache key generation."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        key1 = retriever._generate_cache_key(sample_document, 5, 0.7)
        key2 = retriever._generate_cache_key(sample_document, 5, 0.7)

        assert key1 == key2  # Same inputs = same key
        assert key1.startswith("pmg:context:")

    def test_generate_cache_key_different_params(
        self, sample_document, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test cache key differs for different parameters."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        key1 = retriever._generate_cache_key(sample_document, 5, 0.7)
        key2 = retriever._generate_cache_key(sample_document, 10, 0.7)

        assert key1 != key2

    def test_get_from_cache_no_redis(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test cache get when Redis not available."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=RetrievalConfig(enable_cache=False)
        )

        result = retriever._get_from_cache("test-key")
        assert result is None

    def test_save_to_cache_no_redis(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator, sample_context_result
    ):
        """Test cache save when Redis not available."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=RetrievalConfig(enable_cache=False)
        )

        # Should not raise
        retriever._save_to_cache("test-key", [sample_context_result])

    def test_get_statistics(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test getting statistics."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        stats = retriever.get_statistics()

        assert "total_retrievals" in stats
        assert "cache_hit_rate" in stats

    def test_retrieve_vendor_patterns_mock_mode(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test vendor pattern retrieval in mock mode."""
        mock_graph_client.mock_mode = True

        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        results = retriever.retrieve_vendor_patterns("SUP-001", "invoice")
        assert results == []

    def test_retrieve_for_low_confidence(
        self, sample_document, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test retrieval for low confidence predictions."""
        mock_vector_store.search_by_document.return_value = [
            {
                "doc_id": "doc-1",
                "document": {
                    "doc_type": "invoice",
                    "processing_timestamp": datetime.now().isoformat()
                },
                "similarity": 0.75
            }
        ]

        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=RetrievalConfig(enable_cache=False)
        )

        results = retriever.retrieve_for_low_confidence(
            sample_document,
            confidence_score=0.5
        )

        # Should use lower similarity threshold
        assert mock_vector_store.search_by_document.called

    def test_export_contexts(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator,
        sample_context_result, temp_dir
    ):
        """Test exporting contexts to file."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        output_file = temp_dir / "exported_contexts.json"
        retriever.export_contexts([sample_context_result], str(output_file))

        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert data["total_contexts"] == 1
        assert "exported_at" in data

    def test_clear_cache_no_redis(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test clearing cache when Redis not available."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=RetrievalConfig(enable_cache=False)
        )

        # Should not raise
        retriever.clear_cache()


@pytest.mark.unit
class TestContextRetrieverCaching:
    """Tests for caching functionality."""

    @patch('sap_llm.pmg.context_retriever.redis.Redis')
    def test_cache_hit(
        self, mock_redis_class, sample_document, sample_context_result,
        mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test cache hit scenario."""
        mock_redis = Mock()

        # Serialize context for cache
        cached_data = json.dumps([asdict(sample_context_result)]).encode('utf-8')
        mock_redis.get.return_value = cached_data
        mock_redis_class.return_value = mock_redis

        config = RetrievalConfig(enable_cache=True)
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=config
        )

        results = retriever.retrieve_context(sample_document)

        assert retriever.stats["cache_hits"] == 1
        assert len(results) == 1

    @patch('sap_llm.pmg.context_retriever.redis.Redis')
    def test_cache_miss(
        self, mock_redis_class, sample_document,
        mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test cache miss scenario."""
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis_class.return_value = mock_redis

        mock_vector_store.search_by_document.return_value = []

        config = RetrievalConfig(enable_cache=True)
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=config
        )

        retriever.retrieve_context(sample_document)

        assert retriever.stats["cache_misses"] == 1


@pytest.mark.unit
class TestContextRetrieverEdgeCases:
    """Edge case tests for ContextRetriever."""

    def test_empty_document(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test retrieval with empty document."""
        mock_vector_store.search_by_document.return_value = []

        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator,
            config=RetrievalConfig(enable_cache=False)
        )

        results = retriever.retrieve_context({})
        assert results == []

    def test_invalid_timestamp(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test recency calculation with invalid timestamp."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        weight = retriever._calculate_recency_weight("invalid-timestamp")
        assert weight == 1.0  # Default on error

    def test_update_stats_empty_results(
        self, mock_graph_client, mock_vector_store, mock_embedding_generator
    ):
        """Test stats update with empty results."""
        retriever = ContextRetriever(
            graph_client=mock_graph_client,
            vector_store=mock_vector_store,
            embedding_generator=mock_embedding_generator
        )

        retriever._update_stats([])

        assert retriever.stats["total_retrievals"] == 1
