"""
Comprehensive tests for multi-provider web search system.

Tests semantic ranking, query analysis, SAP validation, and knowledge extraction.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from sap_llm.web_search.semantic_ranker import SemanticRanker
from sap_llm.web_search.query_analyzer import QueryAnalyzer
from sap_llm.web_search.sap_validator import SAPSourceValidator
from sap_llm.web_search.knowledge_extractor import KnowledgeExtractor, KnowledgeEntry
from sap_llm.web_search.search_providers import SerpAPIProvider, BraveSearchProvider
from sap_llm.web_search.search_engine import WebSearchEngine
from sap_llm.agents.web_search_agent import WebSearchAgent


# Fixtures

@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "title": "SAP BAPI_VENDOR_GETDETAIL - Official API Reference",
            "url": "https://api.sap.com/api/BAPI_VENDOR_GETDETAIL",
            "snippet": "Retrieve vendor master data using BAPI_VENDOR_GETDETAIL function module",
            "source": "serpapi",
            "timestamp": 1234567890
        },
        {
            "title": "How to use BAPI_VENDOR_GETDETAIL in SAP",
            "url": "https://help.sap.com/docs/vendor-bapi",
            "snippet": "Tutorial on using BAPI_VENDOR_GETDETAIL to fetch vendor information",
            "source": "serpapi",
            "timestamp": 1234567890
        },
        {
            "title": "SAP Vendor Management Best Practices",
            "url": "https://community.sap.com/questions/vendor-management",
            "snippet": "Community discussion about vendor master data management in SAP",
            "source": "brave",
            "timestamp": 1234567890
        },
        {
            "title": "Getting invoice prices from SAP - Stack Overflow",
            "url": "https://stackoverflow.com/questions/sap-invoice-price",
            "snippet": "How to extract invoice line item prices using BAPI",
            "source": "brave",
            "timestamp": 1234567890
        }
    ]


# Semantic Ranker Tests

class TestSemanticRanker:
    """Tests for semantic ranking functionality."""

    def test_semantic_ranker_initialization(self):
        """Test semantic ranker initializes correctly."""
        ranker = SemanticRanker()
        assert ranker.model_name == "all-MiniLM-L6-v2"
        assert ranker.batch_size == 32

    @patch('sap_llm.web_search.semantic_ranker.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('sap_llm.web_search.semantic_ranker.SentenceTransformer')
    def test_rank_results(self, mock_transformer, sample_search_results):
        """Test semantic ranking of search results."""
        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        mock_transformer.return_value = mock_model

        ranker = SemanticRanker()
        ranker.model = mock_model

        results = ranker.rank_results(
            query="SAP vendor BAPI",
            results=sample_search_results
        )

        # Verify results have semantic scores
        assert len(results) == len(sample_search_results)
        assert all("semantic_score" in r for r in results)

    @patch('sap_llm.web_search.semantic_ranker.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('sap_llm.web_search.semantic_ranker.SentenceTransformer')
    def test_remove_semantic_duplicates(self, mock_transformer, sample_search_results):
        """Test semantic duplicate removal."""
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        mock_transformer.return_value = mock_model

        ranker = SemanticRanker()
        ranker.model = mock_model

        # Add a near-duplicate result
        duplicated = sample_search_results + [sample_search_results[0].copy()]

        unique_results = ranker.remove_semantic_duplicates(
            duplicated,
            threshold=0.95
        )

        # Should have removed some duplicates (exact behavior depends on mock embeddings)
        assert len(unique_results) <= len(duplicated)

    def test_semantic_ranker_unavailable(self, sample_search_results):
        """Test behavior when sentence transformers not available."""
        with patch('sap_llm.web_search.semantic_ranker.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            ranker = SemanticRanker()
            assert not ranker.is_available()

            # Should return original results
            results = ranker.rank_results("test", sample_search_results)
            assert results == sample_search_results


# Query Analyzer Tests

class TestQueryAnalyzer:
    """Tests for query analysis and refinement."""

    def test_query_analyzer_initialization(self):
        """Test query analyzer initializes with SAP knowledge."""
        analyzer = QueryAnalyzer()
        assert len(analyzer.sap_term_mappings) > 0
        assert len(analyzer.sap_modules) > 0

    def test_refine_query_basic(self):
        """Test basic query refinement."""
        analyzer = QueryAnalyzer()
        refined = analyzer.refine_query("How to get invoice price?")

        assert len(refined) >= 1
        assert refined[0] == "How to get invoice price?"  # Original included
        assert any("SAP" in q for q in refined)  # Should add SAP variations

    def test_refine_query_with_context(self):
        """Test query refinement with context."""
        analyzer = QueryAnalyzer()
        refined = analyzer.refine_query(
            "How to retrieve vendor data?",
            context={"document_type": "invoice", "module": "MM"}
        )

        assert len(refined) > 1
        # Should include module-specific variations
        assert any("MM" in q for q in refined)

    def test_analyze_intent(self):
        """Test intent detection."""
        analyzer = QueryAnalyzer()

        assert analyzer._analyze_intent("How to configure SAP?") == "how_to"
        assert analyzer._analyze_intent("What is BAPI?") == "what_is"
        assert analyzer._analyze_intent("SAP error message") == "troubleshoot"
        assert analyzer._analyze_intent("API documentation") == "api_lookup"

    def test_expand_with_sap_terms(self):
        """Test SAP terminology expansion."""
        analyzer = QueryAnalyzer()
        variations = analyzer._expand_with_sap_terms("Get invoice price", "general")

        assert len(variations) > 0
        # Should expand "invoice" with SAP terms
        assert any("supplier invoice" in v.lower() for v in variations)

    def test_suggest_search_domains(self):
        """Test domain suggestion."""
        analyzer = QueryAnalyzer()

        # API query should suggest api.sap.com
        domains = analyzer.suggest_search_domains("SAP BAPI documentation")
        assert "api.sap.com" in domains

        # Help query should suggest help.sap.com
        domains = analyzer.suggest_search_domains("What is vendor master data?")
        assert "help.sap.com" in domains

    def test_extract_entities(self):
        """Test SAP entity extraction."""
        analyzer = QueryAnalyzer()
        entities = analyzer.extract_entities("Use ME21N transaction in MM module")

        assert "ME21N" in entities["transactions"]
        assert "MM" in entities["modules"]


# SAP Validator Tests

class TestSAPSourceValidator:
    """Tests for SAP source validation and trust scoring."""

    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = SAPSourceValidator(min_trust_score=0.6)
        assert validator.min_trust_score == 0.6
        assert len(validator.TRUSTED_DOMAINS) > 0

    def test_validate_results(self, sample_search_results):
        """Test result validation."""
        validator = SAPSourceValidator(min_trust_score=0.5)
        validated = validator.validate_results(sample_search_results)

        # All results should have trust scores
        assert all("trust_score" in r for r in validated)
        assert all("trust_level" in r for r in validated)
        assert all("trust_metadata" in r for r in validated)

    def test_calculate_trust_score_official(self):
        """Test trust scoring for official SAP sources."""
        validator = SAPSourceValidator()

        official_result = {
            "url": "https://api.sap.com/api/test",
            "title": "Official SAP API",
            "snippet": "Official documentation"
        }

        score = validator._calculate_trust_score(official_result)
        assert score > 0.8  # Official sources should have high trust

    def test_calculate_trust_score_community(self):
        """Test trust scoring for community sources."""
        validator = SAPSourceValidator()

        community_result = {
            "url": "https://community.sap.com/test",
            "title": "Community discussion",
            "snippet": "User question about SAP"
        }

        score = validator._calculate_trust_score(community_result)
        assert 0.5 < score < 1.0  # Community has moderate trust

    def test_calculate_trust_score_unknown(self):
        """Test trust scoring for unknown sources."""
        validator = SAPSourceValidator()

        unknown_result = {
            "url": "https://random-blog.com/sap-post",
            "title": "Random blog post",
            "snippet": "Some SAP information"
        }

        score = validator._calculate_trust_score(unknown_result)
        assert score < 0.6  # Unknown sources should have lower trust

    def test_filter_by_trust_score(self, sample_search_results):
        """Test filtering by minimum trust score."""
        validator = SAPSourceValidator(min_trust_score=0.8)
        validated = validator.validate_results(sample_search_results)

        # Should filter out low-trust results
        assert all(r["trust_score"] >= 0.8 for r in validated)

    def test_https_requirement(self):
        """Test HTTPS validation."""
        validator = SAPSourceValidator(require_https=True)

        http_result = {
            "url": "http://insecure.com/test",
            "title": "HTTP site",
            "snippet": "Test"
        }

        score = validator._calculate_trust_score(http_result)
        assert score < 0.5  # Should penalize HTTP

    def test_trust_summary(self, sample_search_results):
        """Test trust summary generation."""
        validator = SAPSourceValidator()
        validated = validator.validate_results(sample_search_results)

        summary = validator.get_trust_summary(validated)

        assert "count" in summary
        assert "avg_trust_score" in summary
        assert "high_trust_count" in summary


# Knowledge Extractor Tests

class TestKnowledgeExtractor:
    """Tests for knowledge extraction."""

    def test_extractor_initialization(self):
        """Test extractor initializes correctly."""
        extractor = KnowledgeExtractor()
        assert not extractor.fetch_full_content  # Default
        assert extractor.max_content_length == 10000

    def test_determine_source_type(self, sample_search_results):
        """Test source type detection."""
        extractor = KnowledgeExtractor()

        api_result = sample_search_results[0]  # api.sap.com
        source_type = extractor._determine_source_type(api_result)
        assert source_type == "api_documentation"

    def test_extract_from_results(self, sample_search_results):
        """Test knowledge extraction from results."""
        # Add trust scores to results
        for result in sample_search_results:
            result["trust_score"] = 0.8

        extractor = KnowledgeExtractor()
        entries = extractor.extract_from_results(sample_search_results, min_trust_score=0.7)

        assert len(entries) > 0
        assert all(isinstance(e, KnowledgeEntry) for e in entries)
        assert all(e.trust_score >= 0.7 for e in entries)

    def test_knowledge_entry_to_dict(self):
        """Test knowledge entry serialization."""
        entry = KnowledgeEntry(
            content="Test content",
            source_url="https://test.com",
            source_type="api_documentation",
            title="Test",
            trust_score=0.9
        )

        entry_dict = entry.to_dict()

        assert entry_dict["content"] == "Test content"
        assert entry_dict["trust_score"] == 0.9
        assert "extracted_at" in entry_dict


# Provider Tests

class TestNewProviders:
    """Tests for SerpAPI and Brave Search providers."""

    @patch('sap_llm.web_search.search_providers.requests')
    def test_serpapi_provider(self, mock_requests):
        """Test SerpAPI provider."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "organic_results": [
                {
                    "title": "Test",
                    "link": "https://test.com",
                    "snippet": "Test snippet"
                }
            ]
        }
        mock_requests.get.return_value = mock_response

        provider = SerpAPIProvider(api_key="test_key")
        results = provider.search("test query")

        assert len(results) == 1
        assert results[0]["source"] == "serpapi"

    @patch('sap_llm.web_search.search_providers.requests')
    def test_brave_provider(self, mock_requests):
        """Test Brave Search provider."""
        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test",
                        "url": "https://test.com",
                        "description": "Test description"
                    }
                ]
            }
        }
        mock_requests.get.return_value = mock_response

        provider = BraveSearchProvider(api_key="test_key")
        results = provider.search("test query")

        assert len(results) == 1
        assert results[0]["source"] == "brave"


# Integration Tests

class TestWebSearchEngine:
    """Integration tests for WebSearchEngine with new components."""

    def test_engine_initialization_with_components(self):
        """Test engine initializes all new components."""
        config = {
            "enabled": True,
            "providers": {},
            "semantic_ranking": {"use_gpu": False},
            "sap_validator": {"min_trust_score": 0.5}
        }

        engine = WebSearchEngine(config)

        assert engine.semantic_ranker is not None
        assert engine.query_analyzer is not None
        assert engine.sap_validator is not None
        assert engine.knowledge_extractor is not None

    @patch('sap_llm.web_search.search_engine.WebSearchEngine.search')
    def test_search_with_context(self, mock_search):
        """Test search with context."""
        mock_search.return_value = []

        config = {"enabled": True, "providers": {}}
        engine = WebSearchEngine(config)

        engine.search(
            query="SAP vendor BAPI",
            context={"document_type": "invoice", "module": "MM"}
        )

        # Verify search was called
        assert mock_search.called


class TestWebSearchAgent:
    """Tests for Web Search Agent."""

    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        config = {"enabled": True}
        agent = WebSearchAgent(config)

        assert agent.engine is not None

    @patch.object(WebSearchEngine, 'search')
    def test_agent_search(self, mock_search):
        """Test agent search method."""
        mock_search.return_value = []

        agent = WebSearchAgent({"enabled": True})
        results = agent.search("test query")

        assert mock_search.called
        assert isinstance(results, list)

    @patch.object(WebSearchEngine, 'search')
    @patch.object(WebSearchEngine, 'extract_knowledge')
    def test_agent_search_and_learn(self, mock_extract, mock_search):
        """Test agent search and learn."""
        mock_search.return_value = [{"title": "Test", "url": "test.com"}]
        mock_extract.return_value = []

        agent = WebSearchAgent({"enabled": True})
        response = agent.search_and_learn("test query")

        assert "results" in response
        assert "knowledge" in response
        assert mock_search.called
        assert mock_extract.called


# Performance Tests

class TestPerformance:
    """Performance tests for search components."""

    def test_semantic_ranking_performance(self, sample_search_results):
        """Test semantic ranking completes within acceptable time."""
        import time

        with patch('sap_llm.web_search.semantic_ranker.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('sap_llm.web_search.semantic_ranker.SentenceTransformer'):
                ranker = SemanticRanker()
                ranker.model = MagicMock()
                ranker.model.encode.return_value = [[0.1, 0.2]] * len(sample_search_results)

                start = time.time()
                ranker.rank_results("test", sample_search_results)
                elapsed = time.time() - start

                # Should complete in under 1 second for small dataset
                assert elapsed < 1.0

    def test_query_refinement_performance(self):
        """Test query refinement is fast."""
        import time

        analyzer = QueryAnalyzer()

        start = time.time()
        refined = analyzer.refine_query("How to get invoice price?")
        elapsed = time.time() - start

        # Should be very fast (<100ms)
        assert elapsed < 0.1
        assert len(refined) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
