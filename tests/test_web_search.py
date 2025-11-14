"""
Unit tests for Web Search module.

Tests all components of the web search system including search engine,
caching, rate limiting, entity enrichment, and integrations.
"""

import time
import unittest
from unittest.mock import MagicMock, Mock, patch

from sap_llm.web_search.cache_manager import SearchCacheManager
from sap_llm.web_search.entity_enrichment import EntityEnricher
from sap_llm.web_search.integrations import (
    KnowledgeBaseUpdater,
    QualityCheckEnhancer,
    RoutingEnhancer,
    ValidationEnhancer,
)
from sap_llm.web_search.rate_limiter import RateLimiter
from sap_llm.web_search.result_processor import ResultProcessor
from sap_llm.web_search.search_engine import SearchMode, WebSearchEngine
from sap_llm.web_search.search_providers import DuckDuckGoProvider


class TestRateLimiter(unittest.TestCase):
    """Test rate limiter functionality."""

    def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limits."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_day=100)

        # Should allow first requests
        for _ in range(10):
            self.assertTrue(limiter.can_proceed())
            limiter.record_request()

    def test_rate_limiter_blocks_excess(self):
        """Test that rate limiter blocks excess requests."""
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=100)

        # Record 5 requests
        for _ in range(5):
            self.assertTrue(limiter.can_proceed())
            limiter.record_request()

        # 6th request should be blocked
        self.assertFalse(limiter.can_proceed())

    def test_rate_limiter_statistics(self):
        """Test rate limiter statistics."""
        limiter = RateLimiter(requests_per_minute=10, requests_per_day=100)

        limiter.record_request()
        limiter.record_request()

        stats = limiter.get_stats()
        self.assertEqual(stats['total_requests'], 2)
        self.assertEqual(stats['current_daily_count'], 2)

    def test_rate_limiter_reset(self):
        """Test rate limiter reset."""
        limiter = RateLimiter(requests_per_minute=5, requests_per_day=100)

        for _ in range(5):
            limiter.record_request()

        limiter.reset()
        stats = limiter.get_stats()
        self.assertEqual(stats['current_daily_count'], 0)
        self.assertEqual(stats['current_minute_count'], 0)


class TestCacheManager(unittest.TestCase):
    """Test cache manager functionality."""

    def test_in_memory_cache_set_get(self):
        """Test in-memory cache set and get."""
        cache = SearchCacheManager(enabled=True)

        test_data = [{"title": "Test", "url": "http://test.com"}]
        cache.set("test_key", test_data)

        result = cache.get("test_key")
        self.assertEqual(result, test_data)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = SearchCacheManager(enabled=True)

        result = cache.get("nonexistent_key")
        self.assertIsNone(result)

    def test_cache_disabled(self):
        """Test that disabled cache doesn't store."""
        cache = SearchCacheManager(enabled=False)

        test_data = [{"title": "Test"}]
        cache.set("test_key", test_data)

        result = cache.get("test_key")
        self.assertIsNone(result)

    def test_cache_statistics(self):
        """Test cache statistics."""
        cache = SearchCacheManager(enabled=True)

        # Set and get
        cache.set("key1", [{"test": "data"}])
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['sets'], 1)


class TestResultProcessor(unittest.TestCase):
    """Test result processor functionality."""

    def test_result_validation(self):
        """Test result validation."""
        processor = ResultProcessor()

        # Valid result
        valid_result = {
            "title": "Test",
            "url": "https://test.com",
            "snippet": "Test snippet"
        }
        self.assertTrue(processor._validate_result(valid_result))

        # Invalid result (missing URL)
        invalid_result = {
            "title": "Test",
            "snippet": "Test"
        }
        self.assertFalse(processor._validate_result(invalid_result))

    def test_domain_blocking(self):
        """Test domain blocking."""
        processor = ResultProcessor(blocked_domains=["spam.com"])

        blocked_result = {
            "title": "Spam",
            "url": "https://spam.com/test",
            "snippet": "Spam content"
        }
        self.assertTrue(processor._is_blocked(blocked_result))

        good_result = {
            "title": "Good",
            "url": "https://good.com/test",
            "snippet": "Good content"
        }
        self.assertFalse(processor._is_blocked(good_result))

    def test_relevance_scoring(self):
        """Test relevance scoring."""
        processor = ResultProcessor()

        result = {
            "title": "SAP API Documentation",
            "url": "https://help.sap.com/api",
            "snippet": "SAP API reference for developers"
        }

        score = processor._calculate_relevance(result, "SAP API")
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_deduplication(self):
        """Test result deduplication."""
        processor = ResultProcessor()

        results = [
            {"title": "Test 1", "url": "https://test.com/page", "snippet": "Test 1"},
            {"title": "Test 2", "url": "https://test.com/page", "snippet": "Test 2"},  # Duplicate
            {"title": "Test 3", "url": "https://other.com/page", "snippet": "Test 3"}
        ]

        unique = processor._deduplicate(results)
        self.assertEqual(len(unique), 2)


class TestEntityEnricher(unittest.TestCase):
    """Test entity enrichment functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock search engine
        self.mock_engine = MagicMock()
        self.enricher = EntityEnricher(self.mock_engine)

    def test_iban_validation_valid(self):
        """Test IBAN validation with valid IBAN."""
        valid_iban = "DE89370400440532013000"
        result = self.enricher.validate_iban(valid_iban)

        self.assertTrue(result['format_valid'])
        self.assertTrue(result['length_valid'])
        self.assertTrue(result['checksum_valid'])
        self.assertTrue(result['valid'])

    def test_iban_validation_invalid(self):
        """Test IBAN validation with invalid IBAN."""
        invalid_iban = "DE12345678901234567890"  # Wrong checksum
        result = self.enricher.validate_iban(invalid_iban)

        self.assertFalse(result['checksum_valid'])
        self.assertFalse(result['valid'])

    def test_tax_id_format_validation(self):
        """Test tax ID format validation."""
        # Valid German VAT
        self.assertTrue(
            self.enricher._validate_tax_id_format("DE123456789", "DE")
        )

        # Invalid format
        self.assertFalse(
            self.enricher._validate_tax_id_format("INVALID", "DE")
        )

    def test_price_extraction(self):
        """Test price extraction from search results."""
        results = [
            {
                "title": "Product for $99.99",
                "url": "https://shop.com",
                "snippet": "Buy now for only $99.99"
            }
        ]

        prices = self.enricher._extract_prices(results)
        self.assertGreater(len(prices), 0)
        self.assertEqual(prices[0]['amount'], 99.99)


class TestSearchEngine(unittest.TestCase):
    """Test search engine functionality."""

    def test_search_engine_initialization(self):
        """Test search engine initialization."""
        config = {
            "enabled": True,
            "providers": {
                "duckduckgo": {"enabled": True}
            },
            "provider_priority": ["duckduckgo"],
            "rate_limits": {},
            "cache": {},
            "trusted_domains": [],
            "blocked_domains": []
        }

        engine = WebSearchEngine(config)
        self.assertTrue(engine.enabled)
        self.assertIsNotNone(engine.cache_manager)

    def test_offline_mode(self):
        """Test offline mode."""
        config = {
            "enabled": True,
            "offline_mode": True,
            "providers": {},
            "provider_priority": [],
            "rate_limits": {},
            "cache": {},
            "trusted_domains": [],
            "blocked_domains": []
        }

        engine = WebSearchEngine(config)
        results = engine.search("test query")
        self.assertEqual(results, [])

    def test_cache_key_generation(self):
        """Test cache key generation."""
        config = {
            "enabled": True,
            "providers": {},
            "provider_priority": [],
            "rate_limits": {},
            "cache": {},
            "trusted_domains": [],
            "blocked_domains": []
        }

        engine = WebSearchEngine(config)

        key1 = engine._generate_cache_key("query1", 10, SearchMode.WEB, None)
        key2 = engine._generate_cache_key("query1", 10, SearchMode.WEB, None)
        key3 = engine._generate_cache_key("query2", 10, SearchMode.WEB, None)

        # Same query should generate same key
        self.assertEqual(key1, key2)
        # Different query should generate different key
        self.assertNotEqual(key1, key3)


class TestValidationEnhancer(unittest.TestCase):
    """Test validation enhancer."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = MagicMock()
        self.enhancer = ValidationEnhancer(self.mock_engine)

    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertTrue(self.enhancer.enabled)
        self.assertIsNotNone(self.enhancer.enricher)

    def test_validation_disabled(self):
        """Test validation when disabled."""
        enhancer = ValidationEnhancer(self.mock_engine, enabled=False)
        result = enhancer.validate_vendor_data("Test Vendor")

        self.assertFalse(result['validated'])
        self.assertIn('reason', result)


class TestRoutingEnhancer(unittest.TestCase):
    """Test routing enhancer."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = MagicMock()
        self.enhancer = RoutingEnhancer(self.mock_engine)

    def test_api_endpoint_extraction(self):
        """Test API endpoint extraction."""
        results = [
            {
                "title": "SAP BAPI_PO_CREATE",
                "url": "https://help.sap.com/bapi_po_create",
                "snippet": "Use BAPI_PO_CREATE to create purchase orders"
            }
        ]

        endpoints = self.enhancer._extract_api_endpoints(results)
        self.assertGreater(len(endpoints), 0)
        self.assertEqual(endpoints[0]['type'], 'BAPI')
        self.assertIn('BAPI_PO_CREATE', endpoints[0]['name'])


class TestQualityCheckEnhancer(unittest.TestCase):
    """Test quality check enhancer."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = MagicMock()
        self.enhancer = QualityCheckEnhancer(self.mock_engine)

    def test_date_validation_reasonable(self):
        """Test date validation for reasonable dates."""
        result = self.enhancer._verify_date_reasonable("2024-01-15")
        self.assertTrue(result['reasonable'])

    def test_date_validation_future(self):
        """Test date validation for future dates."""
        result = self.enhancer._verify_date_reasonable("2099-12-31")
        self.assertFalse(result['reasonable'])
        self.assertIn("future", result['reason'].lower())


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimiter))
    suite.addTests(loader.loadTestsFromTestCase(TestCacheManager))
    suite.addTests(loader.loadTestsFromTestCase(TestResultProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestEntityEnricher))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationEnhancer))
    suite.addTests(loader.loadTestsFromTestCase(TestRoutingEnhancer))
    suite.addTests(loader.loadTestsFromTestCase(TestQualityCheckEnhancer))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
