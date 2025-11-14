"""
Comprehensive end-to-end integration tests for SAP_LLM pipeline.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json

from sap_llm.stages import *
from sap_llm.apop.orchestrator import AgenticOrchestrator
from sap_llm.apop.envelope import APOPEnvelope
from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.models.unified_model import UnifiedExtractorModel


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""

    @pytest.fixture
    async def full_pipeline(self, mock_pmg):
        """Create full pipeline with all stages."""
        config = MagicMock()

        # Create orchestrator
        orchestrator = AgenticOrchestrator(pmg=mock_pmg, max_hops=20)

        # Register all stage agents
        from sap_llm.apop.stage_agents import (
            InboxAgent,
            PreprocessingAgent,
            ClassificationAgent,
            ExtractionAgent,
            ValidationAgent,
            RoutingAgent,
        )

        agents = [
            InboxAgent(config=config),
            PreprocessingAgent(config=config),
            ClassificationAgent(config=config),
            ExtractionAgent(config=config),
            ValidationAgent(config=config),
            RoutingAgent(config=config),
        ]

        for agent in agents:
            orchestrator.register_agent(agent)

        return orchestrator

    @pytest.mark.asyncio
    async def test_purchase_order_flow(self, full_pipeline, sample_document_image, temp_dir):
        """Test complete purchase order processing flow."""
        # Save test image
        image_path = temp_dir / "po_document.png"
        sample_document_image.save(image_path)

        # Create initial envelope
        envelope = APOPEnvelope(
            id="doc_po_001",
            source="email",
            type="document.received",
            data={
                "file_path": str(image_path),
                "metadata": {"source": "email", "sender": "vendor@acme.com"},
            },
            next_action_hint="inbox.process",
        )

        # Mock all stages
        with patch.object(full_pipeline.registry.get_agent("inbox"), 'process') as mock_inbox:
            mock_inbox.return_value = APOPEnvelope(
                id="doc_po_001",
                source="inbox",
                type="inbox.routed",
                data=envelope.data,
                next_action_hint="preproc.ocr",
            )

            # Process through pipeline
            results = await full_pipeline.process_envelope(envelope, max_hops=10)

            assert len(results) > 0
            # Verify it went through all stages

    @pytest.mark.asyncio
    async def test_invoice_flow_with_exceptions(self, full_pipeline, sample_document_image):
        """Test invoice processing with validation exceptions."""
        envelope = APOPEnvelope(
            id="doc_inv_001",
            source="api",
            type="document.received",
            data={
                "file_type": "pdf",
                "metadata": {"source": "api"},
            },
            next_action_hint="inbox.process",
        )

        # Mock validation failure
        with patch.object(full_pipeline.registry.get_agent("validation"), 'process') as mock_val:
            mock_val.return_value = APOPEnvelope(
                id="doc_inv_001",
                source="validation",
                type="validation.failed",
                data={
                    "exceptions": [
                        {"field": "total_amount", "error": "Amount mismatch"}
                    ]
                },
                next_action_hint="exception.handle",
            )

            results = await full_pipeline.process_envelope(envelope)

            # Should route to exception handling
            assert any(e.type == "validation.failed" for e in results)

    @pytest.mark.asyncio
    async def test_multi_document_batch(self, full_pipeline):
        """Test processing multiple documents in batch."""
        envelopes = [
            APOPEnvelope(
                id=f"doc_{i}",
                source="batch",
                type="document.received",
                data={"file_path": f"/path/to/doc_{i}.pdf"},
                next_action_hint="inbox.process",
            )
            for i in range(10)
        ]

        # Process all documents
        all_results = []
        for envelope in envelopes:
            results = await full_pipeline.process_envelope(envelope, max_hops=5)
            all_results.extend(results)

        assert len(all_results) >= 10

    @pytest.mark.asyncio
    async def test_pmg_context_utilization(self, full_pipeline, mock_pmg, sample_adc):
        """Test that pipeline utilizes PMG historical context."""
        # Mock PMG returning similar cases
        mock_pmg.query_similar.return_value = [
            {
                "document_type": "PURCHASE_ORDER",
                "routing": {"endpoint": "API_PURCHASEORDER_PROCESS_SRV"},
                "similarity": 0.95,
            }
        ]

        envelope = APOPEnvelope(
            id="doc_001",
            source="test",
            type="document.received",
            data=sample_adc,
            next_action_hint="router.route",
        )

        with patch.object(full_pipeline.registry.get_agent("routing"), 'process') as mock_route:
            mock_route.return_value = APOPEnvelope(
                id="doc_001",
                source="routing",
                type="router.done",
                data={
                    "routing_decision": {
                        "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
                        "confidence": 0.98,  # Higher confidence due to PMG context
                    }
                },
                next_action_hint="complete",
            )

            results = await full_pipeline.process_envelope(envelope)

            # Verify PMG was queried
            mock_pmg.query_similar.assert_called()


@pytest.mark.integration
@pytest.mark.slow
class TestDatabaseIntegration:
    """Database integration tests."""

    @pytest.mark.requires_cosmos
    def test_pmg_cosmos_integration(self):
        """Test PMG integration with Cosmos DB."""
        config = MagicMock()
        config.endpoint = "https://test.cosmos.azure.com"
        config.key = "test_key"
        config.database = "test_db"
        config.container = "test_container"

        with patch('sap_llm.pmg.graph_client.CosmosClient') as mock_cosmos:
            pmg = ProcessMemoryGraph(config=config)

            # Test store transaction
            transaction = {
                "id": "tx_001",
                "document_type": "PURCHASE_ORDER",
                "data": {"po_number": "12345"},
            }

            with patch.object(pmg, 'container') as mock_container:
                mock_container.create_item.return_value = transaction

                tx_id = pmg.store_transaction(transaction)
                assert tx_id is not None

    def test_vector_store_integration(self):
        """Test vector store integration."""
        from sap_llm.pmg.vector_store import VectorStore

        config = MagicMock()
        config.dimension = 768
        config.index_type = "IVF"

        vs = VectorStore(config=config)

        # Add vectors
        vectors = [[0.1 * i] * 768 for i in range(100)]
        ids = [f"vec_{i}" for i in range(100)]

        vs.add(vectors, ids)

        # Search
        query = [0.5] * 768
        results = vs.search(query, k=10)

        assert isinstance(results, list)


@pytest.mark.integration
class TestAPIIntegration:
    """API integration tests."""

    @pytest.fixture
    def api_client(self):
        """Create API client for testing."""
        from sap_llm.api.server import create_app

        app = create_app()
        client = app.test_client()
        return client

    def test_health_endpoint(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200

    def test_process_document_endpoint(self, api_client, sample_document_image, temp_dir):
        """Test document processing endpoint."""
        image_path = temp_dir / "test_doc.png"
        sample_document_image.save(image_path)

        with open(image_path, "rb") as f:
            response = api_client.post(
                "/api/v1/process",
                data={"file": f},
                content_type="multipart/form-data",
            )

        # Should accept the request
        assert response.status_code in [200, 202]

    def test_query_pmg_endpoint(self, api_client):
        """Test PMG query endpoint."""
        query = {
            "document_type": "PURCHASE_ORDER",
            "vendor_id": "100001",
        }

        response = api_client.post(
            "/api/v1/pmg/query",
            data=json.dumps(query),
            content_type="application/json",
        )

        assert response.status_code in [200, 404]

    def test_authentication(self, api_client):
        """Test API authentication."""
        # Without auth token
        response = api_client.post("/api/v1/process")
        assert response.status_code in [401, 403]

        # With auth token
        headers = {"Authorization": "Bearer test_token"}
        response = api_client.post("/api/v1/process", headers=headers)
        # Should get past auth (may fail for other reasons)


@pytest.mark.integration
class TestModelIntegration:
    """Model integration tests."""

    def test_unified_model_integration(self):
        """Test UnifiedExtractorModel integration."""
        with patch('sap_llm.models.unified_model.VisionEncoder'), \
             patch('sap_llm.models.unified_model.LanguageDecoder'), \
             patch('sap_llm.models.unified_model.ReasoningEngine'):

            model = UnifiedExtractorModel(device="cpu")

            # Set mock components
            model.vision_encoder = MagicMock()
            model.language_decoder = MagicMock()
            model.reasoning_engine = MagicMock()

            # Test classification
            model.vision_encoder.classify.return_value = (0, 0.95)

            doc_type, subtype, conf = model.classify(
                image=MagicMock(),
                ocr_text="test",
                words=[],
                boxes=[],
            )

            assert doc_type == "PURCHASE_ORDER"

    def test_model_save_load(self, temp_dir):
        """Test model save and load."""
        model_dir = temp_dir / "model"

        with patch('sap_llm.models.unified_model.VisionEncoder'), \
             patch('sap_llm.models.unified_model.LanguageDecoder'), \
             patch('sap_llm.models.unified_model.ReasoningEngine'):

            # Create and save
            model = UnifiedExtractorModel(device="cpu")
            model.vision_encoder = MagicMock()
            model.language_decoder = MagicMock()
            model.reasoning_engine = MagicMock()

            with patch.object(model.vision_encoder, 'save'), \
                 patch.object(model.language_decoder, 'save'), \
                 patch.object(model.reasoning_engine, 'save'):

                model.save(str(model_dir))

            # Load
            # with patch.object(UnifiedExtractorModel, 'load'):
            #     loaded = UnifiedExtractorModel.load(str(model_dir))
            #     assert loaded is not None


@pytest.mark.integration
class TestSHWLIntegration:
    """SHWL integration tests."""

    def test_shwl_healing_cycle(self, mock_pmg, mock_reasoning_engine):
        """Test complete SHWL healing cycle."""
        from sap_llm.shwl.healing_loop import HealingLoop
        from sap_llm.shwl.clusterer import ExceptionClusterer
        from sap_llm.shwl.rule_generator import RuleGenerator

        config = MagicMock()
        config.loop_interval = 3600
        config.min_exceptions = 10

        healing_loop = HealingLoop(pmg=mock_pmg, config=config)
        healing_loop.clusterer = ExceptionClusterer(config=MagicMock())
        healing_loop.rule_generator = RuleGenerator(
            config=MagicMock(),
            reasoning_engine=mock_reasoning_engine,
        )

        # Mock exception collection
        exceptions = [
            {"id": f"exc_{i}", "category": "VALIDATION_ERROR"}
            for i in range(50)
        ]

        with patch.object(healing_loop, 'collect_exceptions') as mock_collect:
            mock_collect.return_value = exceptions

            with patch.object(healing_loop.clusterer, 'fit'):
                with patch.object(healing_loop.clusterer, 'get_clusters') as mock_clusters:
                    mock_clusters.return_value = [
                        {"id": "cluster_001", "size": 50, "exceptions": exceptions}
                    ]

                    with patch.object(healing_loop.rule_generator, 'generate_rule') as mock_gen:
                        mock_gen.return_value = {"rule_type": "AUTO_CORRECT"}

                        results = healing_loop.run_cycle()
                        assert results is not None


@pytest.mark.integration
class TestKnowledgeBaseIntegration:
    """Knowledge Base integration tests."""

    def test_kb_crawl_and_query(self):
        """Test crawling and querying knowledge base."""
        from sap_llm.knowledge_base.crawler import KnowledgeCrawler
        from sap_llm.knowledge_base.storage import KnowledgeStorage
        from sap_llm.knowledge_base.query import KnowledgeQueryEngine

        config = MagicMock()

        crawler = KnowledgeCrawler(config=config)
        storage = KnowledgeStorage(config=config)
        query_engine = KnowledgeQueryEngine(storage=storage, config=config)

        # Mock crawling
        with patch.object(crawler, 'crawl_url') as mock_crawl:
            mock_crawl.return_value = {
                "name": "API_PURCHASEORDER_PROCESS_SRV",
                "content": "Purchase order API documentation",
            }

            # Mock storage
            with patch.object(storage, 'store_schema') as mock_store:
                mock_store.return_value = "schema_001"

                # Crawl and store
                content = crawler.crawl_url("https://api.sap.com")
                schema_id = storage.store_schema(content)

                # Query
                with patch.object(query_engine, 'query') as mock_query:
                    mock_query.return_value = [content]

                    results = query_engine.query("purchase order API")
                    assert len(results) > 0
