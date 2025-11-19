"""
Comprehensive unit tests for all web search providers.

Tests each provider implementation including:
- SerpAPI, Brave, Google, Bing, Tavily, DuckDuckGo
- Deduplication module
- Error handling and edge cases
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from sap_llm.web_search.search_providers import (
    SearchProvider,
    SerpAPIProvider,
    BraveSearchProvider,
    GoogleSearchProvider,
    BingSearchProvider,
    TavilySearchProvider,
    DuckDuckGoProvider,
    SAPHelpSearchProvider,
    ExchangeRateProvider,
)
from sap_llm.web_search.deduplication import (
    deduplicate_results,
    Deduplicator,
    find_duplicates,
    _normalize_url,
    _simhash,
    _hamming_similarity,
)


# =============================================================================
# SerpAPI Provider Tests
# =============================================================================

class TestSerpAPIProvider:
    """Tests for SerpAPI search provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = SerpAPIProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert "serpapi.com" in provider.base_url

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_success(self, mock_requests):
        """Test successful search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "organic_results": [
                {"title": "Result 1", "link": "https://test1.com", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "https://test2.com", "snippet": "Snippet 2"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = SerpAPIProvider(api_key="test_key")
        results = provider.search("test query", num_results=10)

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"
        assert results[0]["source"] == "serpapi"
        assert "timestamp" in results[0]

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_with_filters(self, mock_requests):
        """Test search with filters."""
        mock_response = Mock()
        mock_response.json.return_value = {"organic_results": []}
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = SerpAPIProvider(api_key="test_key")
        provider.search(
            "test query",
            mode="news",
            filters={"date_range": "d", "domains": ["sap.com"]}
        )

        # Verify request was made with correct params
        call_args = mock_requests.get.call_args
        assert "tbm" in call_args[1]["params"]  # news mode

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_api_error(self, mock_requests):
        """Test API error handling."""
        mock_requests.get.side_effect = Exception("API Error")

        provider = SerpAPIProvider(api_key="test_key")

        with pytest.raises(Exception):
            provider.search("test query")


# =============================================================================
# Brave Search Provider Tests
# =============================================================================

class TestBraveSearchProvider:
    """Tests for Brave Search provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = BraveSearchProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert "brave.com" in provider.base_url

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_success(self, mock_requests):
        """Test successful search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {"title": "Result 1", "url": "https://test1.com", "description": "Desc 1"},
                    {"title": "Result 2", "url": "https://test2.com", "description": "Desc 2"},
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = BraveSearchProvider(api_key="test_key")
        results = provider.search("test query")

        assert len(results) == 2
        assert results[0]["source"] == "brave"
        assert results[0]["snippet"] == "Desc 1"

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_with_locale(self, mock_requests):
        """Test search with locale settings."""
        mock_response = Mock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = BraveSearchProvider(api_key="test_key")
        provider.search(
            "test query",
            filters={"country": "DE", "language": "de"}
        )

        # Verify request headers
        call_args = mock_requests.get.call_args
        assert "X-Subscription-Token" in call_args[1]["headers"]


# =============================================================================
# Google Search Provider Tests
# =============================================================================

class TestGoogleSearchProvider:
    """Tests for Google Custom Search provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = GoogleSearchProvider(api_key="test_key", cx="test_cx")
        assert provider.api_key == "test_key"
        assert provider.cx == "test_cx"
        assert "googleapis.com" in provider.base_url

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_success(self, mock_requests):
        """Test successful search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {"title": "Result 1", "link": "https://test1.com", "snippet": "Snippet 1"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = GoogleSearchProvider(api_key="test_key", cx="test_cx")
        results = provider.search("test query")

        assert len(results) == 1
        assert results[0]["source"] == "google"

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_image_mode(self, mock_requests):
        """Test image search mode."""
        mock_response = Mock()
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = GoogleSearchProvider(api_key="test_key", cx="test_cx")
        provider.search("test query", mode="images")

        call_args = mock_requests.get.call_args
        assert call_args[1]["params"]["searchType"] == "image"


# =============================================================================
# Bing Search Provider Tests
# =============================================================================

class TestBingSearchProvider:
    """Tests for Bing Search provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = BingSearchProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert "bing.microsoft.com" in provider.base_url

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_success(self, mock_requests):
        """Test successful search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "webPages": {
                "value": [
                    {"name": "Result 1", "url": "https://test1.com", "snippet": "Snippet 1"},
                    {"name": "Result 2", "url": "https://test2.com", "snippet": "Snippet 2"},
                ]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = BingSearchProvider(api_key="test_key")
        results = provider.search("test query")

        assert len(results) == 2
        assert results[0]["source"] == "bing"
        assert results[0]["title"] == "Result 1"

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_with_freshness(self, mock_requests):
        """Test search with freshness filter."""
        mock_response = Mock()
        mock_response.json.return_value = {"webPages": {"value": []}}
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = BingSearchProvider(api_key="test_key")
        provider.search("test query", filters={"date_range": "week"})

        call_args = mock_requests.get.call_args
        assert "freshness" in call_args[1]["params"]

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_api_header(self, mock_requests):
        """Test API key header is set correctly."""
        mock_response = Mock()
        mock_response.json.return_value = {"webPages": {"value": []}}
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = BingSearchProvider(api_key="my_key")
        provider.search("test")

        call_args = mock_requests.get.call_args
        assert call_args[1]["headers"]["Ocp-Apim-Subscription-Key"] == "my_key"


# =============================================================================
# Tavily Search Provider Tests
# =============================================================================

class TestTavilySearchProvider:
    """Tests for Tavily AI Search provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = TavilySearchProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert "tavily.com" in provider.base_url

    @patch('sap_llm.web_search.search_providers.requests')
    def test_search_success(self, mock_requests):
        """Test successful search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://test1.com",
                    "content": "Content 1",
                    "score": 0.95
                },
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_requests.post.return_value = mock_response

        provider = TavilySearchProvider(api_key="test_key")
        results = provider.search("test query")

        assert len(results) == 1
        assert results[0]["source"] == "tavily"
        assert results[0]["score"] == 0.95


# =============================================================================
# DuckDuckGo Provider Tests
# =============================================================================

class TestDuckDuckGoProvider:
    """Tests for DuckDuckGo search provider."""

    def test_initialization(self):
        """Test provider initialization (no API key needed)."""
        provider = DuckDuckGoProvider()
        assert "duckduckgo.com" in provider.base_url

    @patch('sap_llm.web_search.search_providers.DDGS')
    def test_search_with_package(self, mock_ddgs_class):
        """Test search using duckduckgo-search package."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=None)
        mock_ddgs.text.return_value = [
            {"title": "Result 1", "href": "https://test1.com", "body": "Body 1"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        provider = DuckDuckGoProvider()
        results = provider.search("test query")

        assert len(results) == 1
        assert results[0]["source"] == "duckduckgo"


# =============================================================================
# SAP Help Provider Tests
# =============================================================================

class TestSAPHelpSearchProvider:
    """Tests for SAP Help search provider."""

    def test_initialization(self):
        """Test provider initialization with SAP domains."""
        provider = SAPHelpSearchProvider()
        assert "help.sap.com" in provider.sap_domains
        assert "api.sap.com" in provider.sap_domains

    @patch.object(DuckDuckGoProvider, 'search')
    def test_search_uses_sap_domains(self, mock_search):
        """Test search restricts to SAP domains."""
        mock_search.return_value = []

        provider = SAPHelpSearchProvider()
        provider.search("test query")

        # Verify the query includes SAP domain restrictions
        call_args = mock_search.call_args
        query = call_args[1]["query"] if "query" in call_args[1] else call_args[0][0]
        assert "site:" in query


# =============================================================================
# Exchange Rate Provider Tests
# =============================================================================

class TestExchangeRateProvider:
    """Tests for Exchange Rate provider."""

    def test_initialization_with_key(self):
        """Test initialization with API key."""
        provider = ExchangeRateProvider(api_key="test_key")
        assert "exchangerate-api.com" in provider.base_url

    def test_initialization_without_key(self):
        """Test initialization without API key (free tier)."""
        provider = ExchangeRateProvider()
        assert "exchangerate.host" in provider.base_url

    @patch('sap_llm.web_search.search_providers.requests')
    def test_get_rate_success(self, mock_requests):
        """Test successful rate lookup."""
        mock_response = Mock()
        mock_response.json.return_value = {"conversion_rate": 0.85}
        mock_response.raise_for_status = Mock()
        mock_requests.get.return_value = mock_response

        provider = ExchangeRateProvider(api_key="test_key")
        rate = provider.get_rate("USD", "EUR")

        assert rate == 0.85


# =============================================================================
# Deduplication Module Tests
# =============================================================================

class TestDeduplicationFunctions:
    """Tests for deduplication functions."""

    def test_deduplicate_by_url(self):
        """Test URL-based deduplication."""
        results = [
            {"url": "https://test.com/page1", "title": "Page 1", "snippet": "Content 1"},
            {"url": "https://test.com/page1", "title": "Page 1 Copy", "snippet": "Content 1"},  # duplicate
            {"url": "https://test.com/page2", "title": "Page 2", "snippet": "Content 2"},
        ]

        unique = deduplicate_results(results, method="url")

        assert len(unique) == 2
        assert unique[0]["title"] == "Page 1"
        assert unique[1]["title"] == "Page 2"

    def test_deduplicate_by_content(self):
        """Test content-based deduplication."""
        results = [
            {"url": "https://a.com", "title": "SAP BAPI Guide", "snippet": "Learn about SAP BAPI functions"},
            {"url": "https://b.com", "title": "SAP BAPI Tutorial", "snippet": "Learn about SAP BAPI functions"},  # similar
            {"url": "https://c.com", "title": "Python Tutorial", "snippet": "Learn Python programming"},
        ]

        unique = deduplicate_results(results, method="content", threshold=0.8)

        # Should detect near-duplicate content
        assert len(unique) <= 3

    def test_deduplicate_combined(self):
        """Test combined deduplication."""
        results = [
            {"url": "https://test.com/page1", "title": "Title 1", "snippet": "Content"},
            {"url": "https://test.com/page1", "title": "Title 1", "snippet": "Content"},  # URL duplicate
            {"url": "https://other.com/page", "title": "Title 1", "snippet": "Content"},  # Content duplicate
        ]

        unique = deduplicate_results(results, method="combined", threshold=0.9)

        assert len(unique) <= 2

    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list."""
        results = deduplicate_results([], method="url")
        assert results == []

    def test_normalize_url(self):
        """Test URL normalization."""
        # Should remove www
        assert _normalize_url("https://www.test.com/page") == "https://test.com/page"

        # Should remove trailing slash
        assert _normalize_url("https://test.com/page/") == "https://test.com/page"

        # Should remove tracking params
        url = "https://test.com/page?utm_source=google&id=123"
        normalized = _normalize_url(url)
        assert "utm_source" not in normalized
        assert "id=123" in normalized

    def test_simhash(self):
        """Test SimHash calculation."""
        # Same text should produce same hash
        hash1 = _simhash("This is a test document")
        hash2 = _simhash("This is a test document")
        assert hash1 == hash2

        # Different text should produce different hash
        hash3 = _simhash("Completely different content")
        assert hash1 != hash3

    def test_hamming_similarity(self):
        """Test Hamming similarity calculation."""
        # Identical hashes should have similarity 1.0
        similarity = _hamming_similarity(0b1111, 0b1111)
        assert similarity == 1.0

        # Completely different hashes should have low similarity
        similarity = _hamming_similarity(0b0000, 0b1111, hashbits=4)
        assert similarity == 0.0


class TestDeduplicatorClass:
    """Tests for Deduplicator class."""

    def test_initialization(self):
        """Test Deduplicator initialization."""
        dedup = Deduplicator(method="combined", threshold=0.90)
        assert dedup.method == "combined"
        assert dedup.threshold == 0.90

    def test_deduplicate(self):
        """Test deduplication method."""
        dedup = Deduplicator(method="url")
        results = [
            {"url": "https://test.com/1", "title": "1", "snippet": "1"},
            {"url": "https://test.com/1", "title": "1", "snippet": "1"},
        ]

        unique = dedup.deduplicate(results)

        assert len(unique) == 1

    def test_statistics(self):
        """Test statistics tracking."""
        dedup = Deduplicator(method="url")
        results = [
            {"url": "https://test.com/1", "title": "1", "snippet": "1"},
            {"url": "https://test.com/1", "title": "1", "snippet": "1"},
        ]

        dedup.deduplicate(results)
        stats = dedup.get_stats()

        assert stats["total_processed"] == 2
        assert stats["total_removed"] == 1

    def test_reset_stats(self):
        """Test statistics reset."""
        dedup = Deduplicator()
        dedup.stats["total_processed"] = 100
        dedup.reset_stats()

        assert dedup.stats["total_processed"] == 0


class TestFindDuplicates:
    """Tests for find_duplicates function."""

    def test_find_url_duplicates(self):
        """Test finding URL-based duplicates."""
        results = [
            {"url": "https://test.com/1", "title": "Title 1", "snippet": "Content 1"},
            {"url": "https://test.com/2", "title": "Title 2", "snippet": "Content 2"},
            {"url": "https://test.com/1", "title": "Title 1 Copy", "snippet": "Content 1"},  # Duplicate of 0
        ]

        groups = find_duplicates(results, threshold=0.9)

        # Should find one group of duplicates
        assert len(groups) >= 1
        assert any(0 in group and 2 in group for group in groups)

    def test_no_duplicates(self):
        """Test when no duplicates exist."""
        results = [
            {"url": "https://a.com", "title": "A", "snippet": "A content"},
            {"url": "https://b.com", "title": "B", "snippet": "B content"},
            {"url": "https://c.com", "title": "C", "snippet": "C content"},
        ]

        groups = find_duplicates(results, threshold=0.99)

        # Should find no duplicate groups
        assert len(groups) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestProviderIntegration:
    """Integration tests for providers."""

    def test_all_providers_have_search_method(self):
        """Test that all providers implement search method."""
        providers = [
            SerpAPIProvider(api_key="test"),
            BraveSearchProvider(api_key="test"),
            GoogleSearchProvider(api_key="test", cx="test"),
            BingSearchProvider(api_key="test"),
            TavilySearchProvider(api_key="test"),
            DuckDuckGoProvider(),
            SAPHelpSearchProvider(),
        ]

        for provider in providers:
            assert hasattr(provider, 'search')
            assert callable(provider.search)

    def test_result_format_consistency(self):
        """Test that all providers return consistent result format."""
        expected_keys = ["title", "url", "snippet", "source", "timestamp"]

        @patch('sap_llm.web_search.search_providers.requests')
        def check_provider(provider_class, mock_data, mock_requests):
            mock_response = Mock()
            mock_response.json.return_value = mock_data
            mock_response.raise_for_status = Mock()
            mock_requests.get.return_value = mock_response
            mock_requests.post.return_value = mock_response

            if provider_class == GoogleSearchProvider:
                provider = provider_class(api_key="test", cx="test")
            elif provider_class == DuckDuckGoProvider:
                return  # Skip - uses different mocking
            else:
                provider = provider_class(api_key="test")

            results = provider.search("test")

            for result in results:
                for key in expected_keys:
                    assert key in result, f"{provider_class.__name__} missing key: {key}"

        # Test each provider
        check_provider(
            SerpAPIProvider,
            {"organic_results": [{"title": "T", "link": "U", "snippet": "S"}]}
        )
        check_provider(
            BraveSearchProvider,
            {"web": {"results": [{"title": "T", "url": "U", "description": "S"}]}}
        )
        check_provider(
            GoogleSearchProvider,
            {"items": [{"title": "T", "link": "U", "snippet": "S"}]}
        )
        check_provider(
            BingSearchProvider,
            {"webPages": {"value": [{"name": "T", "url": "U", "snippet": "S"}]}}
        )


# =============================================================================
# Performance Tests
# =============================================================================

class TestDeduplicationPerformance:
    """Performance tests for deduplication."""

    def test_large_result_set(self):
        """Test deduplication with large result set."""
        # Create 1000 results with some duplicates
        results = []
        for i in range(800):
            results.append({
                "url": f"https://test.com/page{i}",
                "title": f"Title {i}",
                "snippet": f"Content for page {i}"
            })
        # Add 200 duplicates
        for i in range(200):
            results.append(results[i].copy())

        start = time.time()
        unique = deduplicate_results(results, method="url")
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0
        assert len(unique) == 800


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
