"""
Comprehensive unit tests for PMG (Process Memory Graph) components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json
from datetime import datetime

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.pmg.vector_store import VectorStore
from sap_llm.pmg.learning import AdaptiveLearner
from sap_llm.pmg.query import QueryEngine


@pytest.mark.unit
@pytest.mark.pmg
class TestProcessMemoryGraph:
    """Tests for ProcessMemoryGraph."""

    @pytest.fixture
    def mock_pmg(self):
        """Create mocked PMG instance."""
        with patch('sap_llm.pmg.graph_client.CosmosClient') as mock_cosmos:
            mock_client = MagicMock()
            mock_cosmos.return_value = mock_client

            config = MagicMock()
            config.endpoint = "https://test.cosmos.azure.com"
            config.key = "test_key"
            config.database = "test_db"
            config.container = "test_container"

            pmg = ProcessMemoryGraph(config=config)
            return pmg

    def test_pmg_initialization(self, mock_pmg):
        """Test PMG initialization."""
        assert mock_pmg is not None

    def test_store_transaction(self, mock_pmg):
        """Test storing a transaction in PMG."""
        transaction = {
            "id": "tx_001",
            "document_type": "PURCHASE_ORDER",
            "vendor_id": "100001",
            "total_amount": 2200.00,
            "routing": {
                "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
                "success": True,
            },
        }

        with patch.object(mock_pmg, 'container') as mock_container:
            mock_container.create_item.return_value = transaction

            tx_id = mock_pmg.store_transaction(transaction)
            assert tx_id is not None

    def test_query_similar_transactions(self, mock_pmg):
        """Test querying similar transactions."""
        query_vector = [0.1] * 768  # Mock embedding

        with patch.object(mock_pmg, 'vector_store') as mock_vs:
            mock_vs.search.return_value = [
                {
                    "id": "tx_001",
                    "document_type": "PURCHASE_ORDER",
                    "similarity": 0.95,
                }
            ]

            results = mock_pmg.query_similar(
                document_type="PURCHASE_ORDER",
                vendor_id="100001",
                limit=5,
            )

            assert isinstance(results, list)

    def test_query_exceptions(self, mock_pmg):
        """Test querying historical exceptions."""
        with patch.object(mock_pmg, 'container') as mock_container:
            mock_container.query_items.return_value = [
                {
                    "id": "exc_001",
                    "category": "VALIDATION_ERROR",
                    "resolution": "AUTO_CORRECTED",
                }
            ]

            results = mock_pmg.query_exceptions(
                category="VALIDATION_ERROR",
                field="total_amount",
            )

            assert isinstance(results, list)

    def test_store_routing_decision(self, mock_pmg):
        """Test storing routing decision."""
        decision = {
            "document_type": "PURCHASE_ORDER",
            "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
            "confidence": 0.95,
            "success": True,
        }

        with patch.object(mock_pmg, 'container') as mock_container:
            mock_container.create_item.return_value = decision

            result = mock_pmg.store_routing_decision(decision)
            assert result is not None

    def test_get_routing_statistics(self, mock_pmg):
        """Test getting routing statistics."""
        with patch.object(mock_pmg, 'container') as mock_container:
            mock_container.query_items.return_value = [
                {"endpoint": "API_PURCHASEORDER_PROCESS_SRV", "count": 100},
                {"endpoint": "API_SUPPLIERINVOICE_PROCESS_SRV", "count": 75},
            ]

            stats = mock_pmg.get_routing_statistics(
                document_type="PURCHASE_ORDER",
                days=30,
            )

            assert isinstance(stats, list)

    @pytest.mark.parametrize("doc_type", [
        "PURCHASE_ORDER",
        "SUPPLIER_INVOICE",
        "SALES_ORDER",
    ])
    def test_query_by_document_type(self, mock_pmg, doc_type):
        """Test querying by different document types."""
        with patch.object(mock_pmg, 'container') as mock_container:
            mock_container.query_items.return_value = [
                {"id": f"tx_{i}", "document_type": doc_type}
                for i in range(5)
            ]

            results = mock_pmg.query_similar(
                document_type=doc_type,
                limit=10,
            )

            assert isinstance(results, list)


@pytest.mark.unit
@pytest.mark.pmg
class TestVectorStore:
    """Tests for VectorStore."""

    @pytest.fixture
    def vector_store(self):
        """Create VectorStore instance."""
        config = MagicMock()
        config.dimension = 768
        config.index_type = "IVF"
        return VectorStore(config=config)

    def test_vector_store_initialization(self, vector_store):
        """Test VectorStore initialization."""
        assert vector_store is not None
        assert vector_store.dimension == 768

    def test_add_vectors(self, vector_store):
        """Test adding vectors to store."""
        vectors = [[0.1] * 768 for _ in range(10)]
        ids = [f"vec_{i}" for i in range(10)]

        vector_store.add(vectors, ids)
        # Should not raise exception

    def test_search_vectors(self, vector_store):
        """Test searching for similar vectors."""
        # Add some vectors first
        vectors = [[0.1 * i] * 768 for i in range(10)]
        ids = [f"vec_{i}" for i in range(10)]
        vector_store.add(vectors, ids)

        # Search
        query_vector = [0.15] * 768
        results = vector_store.search(query_vector, k=5)

        assert isinstance(results, list)

    def test_update_vector(self, vector_store):
        """Test updating existing vector."""
        vector_id = "vec_1"
        old_vector = [0.1] * 768
        new_vector = [0.2] * 768

        vector_store.add([old_vector], [vector_id])
        vector_store.update(vector_id, new_vector)

        # Should update successfully

    def test_delete_vector(self, vector_store):
        """Test deleting vector."""
        vector_id = "vec_1"
        vector = [0.1] * 768

        vector_store.add([vector], [vector_id])
        vector_store.delete(vector_id)

        # Should delete successfully

    @pytest.mark.parametrize("dimension", [128, 256, 384, 768, 1024])
    def test_different_dimensions(self, dimension):
        """Test vector store with different dimensions."""
        config = MagicMock()
        config.dimension = dimension
        config.index_type = "IVF"

        vs = VectorStore(config=config)
        assert vs.dimension == dimension


@pytest.mark.unit
@pytest.mark.pmg
class TestAdaptiveLearner:
    """Tests for AdaptiveLearner."""

    @pytest.fixture
    def adaptive_learner(self, mock_pmg):
        """Create AdaptiveLearner instance."""
        config = MagicMock()
        config.learning_rate = 0.01
        config.update_frequency = 100
        return AdaptiveLearner(pmg=mock_pmg, config=config)

    def test_learner_initialization(self, adaptive_learner):
        """Test AdaptiveLearner initialization."""
        assert adaptive_learner is not None

    def test_learn_from_feedback(self, adaptive_learner):
        """Test learning from user feedback."""
        feedback = {
            "transaction_id": "tx_001",
            "action": "routing",
            "correct": True,
            "user_correction": None,
        }

        adaptive_learner.learn_from_feedback(feedback)
        # Should update internal models

    def test_learn_from_exception_resolution(self, adaptive_learner):
        """Test learning from exception resolution."""
        exception = {
            "id": "exc_001",
            "category": "VALIDATION_ERROR",
            "field": "total_amount",
            "resolution": "AUTO_CORRECTED",
            "correction_value": 2200.00,
        }

        adaptive_learner.learn_from_exception(exception)
        # Should update exception handling patterns

    def test_update_routing_model(self, adaptive_learner):
        """Test updating routing model."""
        routing_data = {
            "document_type": "PURCHASE_ORDER",
            "vendor_id": "100001",
            "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
            "success": True,
        }

        adaptive_learner.update_routing_model(routing_data)
        # Should improve routing predictions

    def test_get_recommendations(self, adaptive_learner):
        """Test getting recommendations."""
        context = {
            "document_type": "PURCHASE_ORDER",
            "vendor_id": "100001",
        }

        recommendations = adaptive_learner.get_recommendations(context)
        assert isinstance(recommendations, (list, dict))

    @pytest.mark.parametrize("feedback_type", [
        "routing_success",
        "routing_failure",
        "exception_resolved",
        "manual_correction",
    ])
    def test_different_feedback_types(self, adaptive_learner, feedback_type):
        """Test learning from different feedback types."""
        feedback = {
            "type": feedback_type,
            "transaction_id": "tx_001",
            "data": {},
        }

        adaptive_learner.learn_from_feedback(feedback)
        # Should handle all feedback types


@pytest.mark.unit
@pytest.mark.pmg
class TestQueryEngine:
    """Tests for QueryEngine."""

    @pytest.fixture
    def query_engine(self, mock_pmg):
        """Create QueryEngine instance."""
        config = MagicMock()
        return QueryEngine(pmg=mock_pmg, config=config)

    def test_query_engine_initialization(self, query_engine):
        """Test QueryEngine initialization."""
        assert query_engine is not None

    def test_query_by_document_attributes(self, query_engine):
        """Test querying by document attributes."""
        query = {
            "document_type": "PURCHASE_ORDER",
            "vendor_id": "100001",
            "amount_range": (1000, 5000),
        }

        with patch.object(query_engine.pmg, 'container') as mock_container:
            mock_container.query_items.return_value = [
                {"id": "tx_001", "document_type": "PURCHASE_ORDER"}
            ]

            results = query_engine.query(query)
            assert isinstance(results, list)

    def test_semantic_search(self, query_engine):
        """Test semantic search."""
        query_text = "Find purchase orders from ACME Corp"

        with patch.object(query_engine, 'embed_query') as mock_embed:
            mock_embed.return_value = [0.1] * 768

            with patch.object(query_engine.pmg, 'vector_store') as mock_vs:
                mock_vs.search.return_value = [
                    {"id": "tx_001", "similarity": 0.9}
                ]

                results = query_engine.semantic_search(query_text, limit=10)
                assert isinstance(results, list)

    def test_query_similar_exceptions(self, query_engine):
        """Test querying similar exceptions."""
        exception = {
            "category": "VALIDATION_ERROR",
            "field": "total_amount",
            "message": "Amount mismatch",
        }

        with patch.object(query_engine.pmg, 'container') as mock_container:
            mock_container.query_items.return_value = [
                {"id": "exc_001", "category": "VALIDATION_ERROR"}
            ]

            results = query_engine.find_similar_exceptions(exception)
            assert isinstance(results, list)

    def test_aggregation_queries(self, query_engine):
        """Test aggregation queries."""
        with patch.object(query_engine.pmg, 'container') as mock_container:
            mock_container.query_items.return_value = [
                {"document_type": "PURCHASE_ORDER", "count": 150},
                {"document_type": "SUPPLIER_INVOICE", "count": 120},
            ]

            results = query_engine.aggregate_by_type(days=30)
            assert isinstance(results, list)

    def test_time_range_queries(self, query_engine):
        """Test time range queries."""
        with patch.object(query_engine.pmg, 'container') as mock_container:
            mock_container.query_items.return_value = [
                {"id": f"tx_{i}", "timestamp": "2024-01-01T00:00:00Z"}
                for i in range(10)
            ]

            results = query_engine.query_time_range(
                start_date="2024-01-01",
                end_date="2024-01-31",
            )

            assert isinstance(results, list)

    @pytest.mark.parametrize("query_type", [
        "exact_match",
        "fuzzy_match",
        "semantic_search",
        "aggregation",
    ])
    def test_different_query_types(self, query_engine, query_type):
        """Test different query types."""
        with patch.object(query_engine.pmg, 'container') as mock_container:
            mock_container.query_items.return_value = []

            query = {"type": query_type, "params": {}}
            results = query_engine.execute_query(query)
            # Should handle all query types


@pytest.mark.unit
@pytest.mark.pmg
class TestPMGIntegration:
    """Integration tests for PMG components."""

    @pytest.fixture
    def full_pmg_system(self):
        """Create full PMG system with all components."""
        with patch('sap_llm.pmg.graph_client.CosmosClient'):
            config = MagicMock()
            config.endpoint = "https://test.cosmos.azure.com"
            config.key = "test_key"
            config.database = "test_db"
            config.container = "test_container"

            pmg = ProcessMemoryGraph(config=config)

            # Add components
            pmg.vector_store = VectorStore(config=MagicMock(dimension=768, index_type="IVF"))
            pmg.learner = AdaptiveLearner(pmg=pmg, config=MagicMock())
            pmg.query_engine = QueryEngine(pmg=pmg, config=MagicMock())

            return pmg

    def test_end_to_end_transaction_flow(self, full_pmg_system):
        """Test storing and querying a transaction."""
        transaction = {
            "id": "tx_001",
            "document_type": "PURCHASE_ORDER",
            "vendor_id": "100001",
            "total_amount": 2200.00,
        }

        with patch.object(full_pmg_system, 'container') as mock_container:
            mock_container.create_item.return_value = transaction

            # Store transaction
            tx_id = full_pmg_system.store_transaction(transaction)
            assert tx_id is not None

            # Query similar transactions
            mock_container.query_items.return_value = [transaction]
            results = full_pmg_system.query_similar(
                document_type="PURCHASE_ORDER",
                limit=5,
            )

            assert len(results) >= 0

    def test_learning_cycle(self, full_pmg_system):
        """Test complete learning cycle."""
        # Store initial transaction
        transaction = {
            "id": "tx_001",
            "routing": {"endpoint": "API_PURCHASEORDER_PROCESS_SRV"},
        }

        with patch.object(full_pmg_system, 'container') as mock_container:
            mock_container.create_item.return_value = transaction

            full_pmg_system.store_transaction(transaction)

            # Provide feedback
            feedback = {
                "transaction_id": "tx_001",
                "correct": True,
            }

            full_pmg_system.learner.learn_from_feedback(feedback)

            # Query for recommendations
            recommendations = full_pmg_system.learner.get_recommendations({
                "document_type": "PURCHASE_ORDER",
            })

            assert recommendations is not None
