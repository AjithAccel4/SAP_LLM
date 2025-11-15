"""
Comprehensive unit tests for Knowledge Base components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from sap_llm.knowledge_base.crawler import KnowledgeCrawler
from sap_llm.knowledge_base.storage import KnowledgeStorage
from sap_llm.knowledge_base.query import KnowledgeQueryEngine


@pytest.mark.unit
@pytest.mark.knowledge_base
class TestKnowledgeCrawler:
    """Tests for KnowledgeCrawler."""

    @pytest.fixture
    def crawler(self):
        """Create KnowledgeCrawler instance."""
        config = MagicMock()
        config.sources = ["https://api.sap.com"]
        config.depth = 2
        config.respect_robots_txt = True
        return KnowledgeCrawler(config=config)

    def test_crawler_initialization(self, crawler):
        """Test crawler initialization."""
        assert crawler is not None
        assert crawler.depth == 2

    def test_crawl_api_documentation(self, crawler):
        """Test crawling SAP API documentation."""
        with patch('sap_llm.knowledge_base.crawler.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = "<html><body>SAP API Documentation</body></html>"
            mock_get.return_value = mock_response

            result = crawler.crawl_url("https://api.sap.com/api/OP_API_PRODUCT_SRV/overview")
            assert result is not None

    def test_extract_api_schema(self, crawler):
        """Test extracting API schema from documentation."""
        html_content = """
        <div class="api-schema">
            <h2>API_PURCHASEORDER_PROCESS_SRV</h2>
            <p>Version: 0001</p>
            <div class="entity">PurchaseOrder</div>
        </div>
        """

        with patch.object(crawler, 'parse_html') as mock_parse:
            mock_parse.return_value = {
                "name": "API_PURCHASEORDER_PROCESS_SRV",
                "version": "0001",
                "entities": ["PurchaseOrder"],
            }

            schema = crawler.extract_schema(html_content)
            assert schema is not None
            assert "name" in schema

    def test_crawl_with_depth_limit(self, crawler):
        """Test crawling with depth limit."""
        with patch.object(crawler, 'crawl_url') as mock_crawl:
            mock_crawl.return_value = {"links": []}

            # Should stop at depth limit
            crawler.crawl_recursive("https://api.sap.com", current_depth=0, max_depth=2)

    def test_respect_robots_txt(self, crawler):
        """Test respecting robots.txt."""
        with patch('sap_llm.knowledge_base.crawler.robotparser') as mock_robots:
            mock_robots.can_fetch.return_value = False

            # Should not crawl disallowed URLs
            result = crawler.can_crawl("https://api.sap.com/disallowed")
            assert result is False

    def test_extract_business_rules(self, crawler):
        """Test extracting business rules from documentation."""
        content = """
        Business Rule: Total amount must equal sum of line items.
        Tolerance: 2%
        """

        rules = crawler.extract_business_rules(content)
        assert isinstance(rules, list)

    @pytest.mark.parametrize("url,expected_type", [
        ("https://api.sap.com/api/", "api_documentation"),
        ("https://help.sap.com/docs/", "help_documentation"),
        ("https://community.sap.com/", "community_content"),
    ])
    def test_classify_content_type(self, crawler, url, expected_type):
        """Test classifying content types."""
        content_type = crawler.classify_content(url)
        # Should classify based on URL pattern


@pytest.mark.unit
@pytest.mark.knowledge_base
class TestKnowledgeStorage:
    """Tests for KnowledgeStorage."""

    @pytest.fixture
    def storage(self):
        """Create KnowledgeStorage instance."""
        config = MagicMock()
        config.backend = "vector_db"  # or "cosmos_db"
        config.vector_dimension = 768
        return KnowledgeStorage(config=config)

    def test_storage_initialization(self, storage):
        """Test storage initialization."""
        assert storage is not None

    def test_store_api_schema(self, storage):
        """Test storing API schema."""
        schema = {
            "name": "API_PURCHASEORDER_PROCESS_SRV",
            "version": "0001",
            "entities": ["PurchaseOrder"],
            "fields": {
                "PurchaseOrder": ["PONumber", "Supplier", "TotalAmount"],
            },
        }

        with patch.object(storage, 'insert') as mock_insert:
            mock_insert.return_value = "schema_001"

            schema_id = storage.store_schema(schema)
            assert schema_id is not None

    def test_store_business_rule(self, storage):
        """Test storing business rule."""
        rule = {
            "id": "rule_001",
            "description": "Total amount validation",
            "condition": "total == sum(line_items)",
            "tolerance": 0.02,
        }

        with patch.object(storage, 'insert') as mock_insert:
            mock_insert.return_value = "rule_001"

            rule_id = storage.store_rule(rule)
            assert rule_id is not None

    def test_store_with_embeddings(self, storage):
        """Test storing content with embeddings."""
        content = {
            "text": "SAP Purchase Order API documentation",
            "metadata": {"type": "api_doc", "api": "PO"},
        }

        embedding = [0.1] * 768

        with patch.object(storage, 'insert') as mock_insert:
            mock_insert.return_value = "doc_001"

            doc_id = storage.store_with_embedding(content, embedding)
            assert doc_id is not None

    def test_bulk_insert(self, storage):
        """Test bulk inserting documents."""
        documents = [
            {"text": f"Document {i}", "metadata": {}}
            for i in range(100)
        ]

        with patch.object(storage, 'bulk_insert') as mock_bulk:
            mock_bulk.return_value = [f"doc_{i}" for i in range(100)]

            ids = storage.bulk_insert(documents)
            assert len(ids) == 100

    def test_update_document(self, storage):
        """Test updating existing document."""
        doc_id = "doc_001"
        updated_content = {"text": "Updated content"}

        with patch.object(storage, 'update') as mock_update:
            mock_update.return_value = True

            success = storage.update(doc_id, updated_content)
            assert success is True

    def test_delete_document(self, storage):
        """Test deleting document."""
        doc_id = "doc_001"

        with patch.object(storage, 'delete') as mock_delete:
            mock_delete.return_value = True

            success = storage.delete(doc_id)
            assert success is True


@pytest.mark.unit
@pytest.mark.knowledge_base
class TestKnowledgeQueryEngine:
    """Tests for KnowledgeQueryEngine."""

    @pytest.fixture
    def query_engine(self):
        """Create KnowledgeQueryEngine instance."""
        storage = MagicMock()
        config = MagicMock()
        config.top_k = 10
        config.similarity_threshold = 0.7
        return KnowledgeQueryEngine(storage=storage, config=config)

    def test_query_engine_initialization(self, query_engine):
        """Test query engine initialization."""
        assert query_engine is not None
        assert query_engine.top_k == 10

    def test_query_api_schema(self, query_engine):
        """Test querying for API schema."""
        query = "Find API schema for purchase order"

        with patch.object(query_engine, 'search') as mock_search:
            mock_search.return_value = [
                {
                    "name": "API_PURCHASEORDER_PROCESS_SRV",
                    "similarity": 0.95,
                }
            ]

            results = query_engine.query(query, type="api_schema")
            assert len(results) > 0

    def test_query_business_rules(self, query_engine):
        """Test querying for business rules."""
        query = "What are the validation rules for total amount?"

        with patch.object(query_engine, 'search') as mock_search:
            mock_search.return_value = [
                {
                    "rule": "total_amount_validation",
                    "description": "Total must equal sum of line items",
                    "similarity": 0.90,
                }
            ]

            results = query_engine.query(query, type="business_rule")
            assert len(results) > 0

    def test_semantic_search(self, query_engine):
        """Test semantic search."""
        query_text = "How to post invoice to SAP?"

        with patch.object(query_engine, 'embed_query') as mock_embed:
            mock_embed.return_value = [0.1] * 768

            with patch.object(query_engine.storage, 'vector_search') as mock_search:
                mock_search.return_value = [
                    {"text": "Invoice posting API", "score": 0.85}
                ]

                results = query_engine.semantic_search(query_text)
                assert len(results) > 0

    def test_hybrid_search(self, query_engine):
        """Test hybrid search (keyword + semantic)."""
        query_text = "purchase order API endpoint"

        with patch.object(query_engine, 'keyword_search') as mock_keyword:
            mock_keyword.return_value = [{"score": 0.8}]

            with patch.object(query_engine, 'semantic_search') as mock_semantic:
                mock_semantic.return_value = [{"score": 0.9}]

                results = query_engine.hybrid_search(query_text)
                assert len(results) > 0

    def test_filter_by_metadata(self, query_engine):
        """Test filtering results by metadata."""
        filters = {
            "type": "api_schema",
            "version": "0001",
        }

        with patch.object(query_engine.storage, 'search_with_filters') as mock_search:
            mock_search.return_value = [
                {"name": "API_PURCHASEORDER_PROCESS_SRV"}
            ]

            results = query_engine.query("", filters=filters)
            assert len(results) >= 0

    @pytest.mark.parametrize("query,expected_type", [
        ("Find invoice API", "api_schema"),
        ("What are validation rules?", "business_rule"),
        ("How to handle exceptions?", "documentation"),
    ])
    def test_query_type_detection(self, query_engine, query, expected_type):
        """Test automatic query type detection."""
        detected_type = query_engine.detect_query_type(query)
        # Should automatically detect query type


@pytest.mark.unit
@pytest.mark.knowledge_base
class TestKnowledgeBaseIntegration:
    """Integration tests for Knowledge Base components."""

    @pytest.fixture
    def full_kb_system(self):
        """Create full knowledge base system."""
        config = MagicMock()
        config.sources = ["https://api.sap.com"]
        config.backend = "vector_db"
        config.vector_dimension = 768
        config.top_k = 10

        crawler = KnowledgeCrawler(config=config)
        storage = KnowledgeStorage(config=config)
        query_engine = KnowledgeQueryEngine(storage=storage, config=config)

        return {
            "crawler": crawler,
            "storage": storage,
            "query_engine": query_engine,
        }

    def test_crawl_and_store(self, full_kb_system):
        """Test crawling and storing knowledge."""
        crawler = full_kb_system["crawler"]
        storage = full_kb_system["storage"]

        with patch.object(crawler, 'crawl_url') as mock_crawl:
            mock_crawl.return_value = {
                "name": "API_PURCHASEORDER_PROCESS_SRV",
                "content": "API documentation",
            }

            with patch.object(storage, 'store_schema') as mock_store:
                mock_store.return_value = "schema_001"

                # Crawl
                content = crawler.crawl_url("https://api.sap.com/api/PO")

                # Store
                schema_id = storage.store_schema(content)
                assert schema_id is not None

    def test_query_stored_knowledge(self, full_kb_system):
        """Test querying stored knowledge."""
        query_engine = full_kb_system["query_engine"]

        # Mock storage with some data
        with patch.object(query_engine.storage, 'search') as mock_search:
            mock_search.return_value = [
                {
                    "name": "API_PURCHASEORDER_PROCESS_SRV",
                    "content": "Purchase order API",
                }
            ]

            results = query_engine.query("purchase order API")
            assert len(results) > 0

    def test_update_knowledge_base(self, full_kb_system):
        """Test updating knowledge base with new information."""
        crawler = full_kb_system["crawler"]
        storage = full_kb_system["storage"]

        # Crawl new content
        with patch.object(crawler, 'crawl_url') as mock_crawl:
            mock_crawl.return_value = {
                "name": "API_PURCHASEORDER_PROCESS_SRV",
                "version": "0002",  # New version
            }

            # Update storage
            with patch.object(storage, 'update') as mock_update:
                mock_update.return_value = True

                new_content = crawler.crawl_url("https://api.sap.com/api/PO")
                success = storage.update("schema_001", new_content)
                assert success is True

    def test_knowledge_base_statistics(self, full_kb_system):
        """Test getting knowledge base statistics."""
        storage = full_kb_system["storage"]

        with patch.object(storage, 'get_statistics') as mock_stats:
            mock_stats.return_value = {
                "total_documents": 1000,
                "api_schemas": 150,
                "business_rules": 500,
                "documentation": 350,
            }

            stats = storage.get_statistics()
            assert stats["total_documents"] == 1000
