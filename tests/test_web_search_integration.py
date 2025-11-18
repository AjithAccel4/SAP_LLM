"""
Comprehensive Integration Tests for Web Search System.

Tests all enhancements including:
- 3-tier caching (memory, Redis, disk)
- Multi-provider failover
- Extraction enrichment
- Provider health monitoring
"""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sap_llm.web_search import (
    ExtractionEnhancer,
    SearchCacheManager,
    WebSearchEngine,
)


class TestThreeTierCaching(unittest.TestCase):
    """Test 3-tier caching implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_config = {
            "disk_cache_dir": self.temp_dir,
            "max_disk_cache_size_mb": 10,
            "ttl": 60
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_l1_memory_cache(self):
        """Test L1 (memory) cache hit."""
        cache = SearchCacheManager(**self.cache_config)

        test_data = [{"title": "Test", "url": "http://test.com"}]
        cache.set("test_key", test_data)

        # Should hit L1
        result = cache.get("test_key")
        self.assertEqual(result, test_data)
        self.assertEqual(cache.stats["l1_hits"], 1)
        self.assertEqual(cache.stats["l2_hits"], 0)
        self.assertEqual(cache.stats["l3_hits"], 0)

    def test_l3_disk_cache(self):
        """Test L3 (disk) cache hit and promotion."""
        cache = SearchCacheManager(**self.cache_config)

        test_data = [{"title": "Disk Test", "url": "http://disk.com"}]
        cache.set("disk_key", test_data)

        # Clear L1 to force L3 read
        cache.in_memory_cache.clear()

        # Should hit L3 and promote to L1
        result = cache.get("disk_key")
        self.assertEqual(result, test_data)
        self.assertEqual(cache.stats["l3_hits"], 1)

        # Verify promotion to L1
        self.assertIn("disk_key", cache.in_memory_cache)

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache_config = {
            **self.cache_config,
            "ttl": 1  # 1 second TTL
        }
        cache = SearchCacheManager(**cache_config)

        test_data = [{"title": "TTL Test"}]
        cache.set("ttl_key", test_data, ttl=1)

        # Should be cached
        result = cache.get("ttl_key")
        self.assertIsNotNone(result)

        # Wait for expiration
        time.sleep(2)

        # Should be expired
        result = cache.get("ttl_key")
        self.assertIsNone(result)

    def test_disk_cache_size_limit(self):
        """Test disk cache size limit enforcement."""
        cache = SearchCacheManager(**self.cache_config)

        # Create large data to exceed limit
        for i in range(20):
            large_data = [{"data": "x" * 100000}]  # ~100KB each
            cache.set(f"large_key_{i}", large_data)

        # Get disk cache size
        stats = cache.get_stats()
        disk_size_mb = stats.get("l3_size_mb", 0)

        # Should not exceed max_disk_cache_size_mb
        self.assertLessEqual(disk_size_mb, self.cache_config["max_disk_cache_size_mb"])


class TestMultiProviderFailover(unittest.TestCase):
    """Test multi-provider failover functionality."""

    def test_provider_failover(self):
        """Test automatic failover when provider fails."""
        config = {
            "enabled": True,
            "offline_mode": False,
            "providers": {
                "duckduckgo": {"enabled": True}
            },
            "provider_priority": ["google", "bing", "tavily", "duckduckgo"],
            "cache_enabled": False  # Disable cache for this test
        }

        engine = WebSearchEngine(config)

        # Mock provider failures
        with patch.object(engine, 'providers', {
            "duckduckgo": Mock(search=Mock(return_value=[{"title": "DuckDuckGo Result"}]))
        }):
            # Should fall back to DuckDuckGo
            results = engine.search("test query", num_results=1)
            self.assertGreater(len(results), 0)

    def test_provider_health_check(self):
        """Test provider health check functionality."""
        config = {
            "enabled": True,
            "providers": {
                "duckduckgo": {"enabled": True}
            }
        }

        engine = WebSearchEngine(config)
        health = engine.health_check()

        self.assertIn("providers", health)
        self.assertIn("cache", health)
        self.assertIn("rate_limiters", health)

    def test_provider_status(self):
        """Test provider status reporting."""
        config = {
            "enabled": True,
            "providers": {
                "duckduckgo": {"enabled": True}
            },
            "provider_priority": ["tavily", "google", "bing", "duckduckgo"]
        }

        engine = WebSearchEngine(config)
        status = engine.get_provider_status()

        self.assertIn("total_providers", status)
        self.assertIn("available_providers", status)
        self.assertIn("failover_ready", status)
        self.assertIn("providers", status)


class TestExtractionEnrichment(unittest.TestCase):
    """Test extraction enrichment integration."""

    def test_vendor_enrichment(self):
        """Test vendor field enrichment."""
        # Mock search engine
        mock_engine = Mock()
        mock_engine.search = Mock(return_value=[
            {
                "title": "SAP SE - Company Info",
                "url": "https://sap.com/about",
                "snippet": "SAP SE, Dietmar-Hopp-Allee 16, 69190 Walldorf, Germany. VAT: DE123456789"
            }
        ])
        mock_engine.get_exchange_rate = Mock(return_value=None)

        enricher = ExtractionEnhancer(mock_engine)

        extracted_data = {
            "vendor_name": "SAP SE",
            "country": "Germany",
            "total_amount": 1000
        }

        enriched = enricher.enrich_extracted_data(extracted_data)

        self.assertIn("web_enrichments", enriched)
        self.assertEqual(enricher.enrichment_stats["vendor_enrichments"], 1)

    def test_currency_lookup(self):
        """Test currency exchange rate lookup."""
        # Mock search engine
        mock_engine = Mock()
        mock_engine.search = Mock(return_value=[])
        mock_engine.get_exchange_rate = Mock(return_value={
            "rate": 1.2,
            "from_currency": "EUR",
            "to_currency": "USD",
            "date": "2024-01-01"
        })

        enricher = ExtractionEnhancer(mock_engine)

        extracted_data = {
            "currency": "EUR",
            "total_amount": 1000
        }

        enriched = enricher.enrich_extracted_data(extracted_data)

        self.assertIn("amount_usd", enriched)
        self.assertEqual(enriched["amount_usd"], 1200)  # 1000 * 1.2
        self.assertEqual(enricher.enrichment_stats["currency_lookups"], 1)

    def test_enrichment_stats(self):
        """Test enrichment statistics tracking."""
        mock_engine = Mock()
        mock_engine.search = Mock(return_value=[])
        mock_engine.get_exchange_rate = Mock(return_value=None)

        enricher = ExtractionEnhancer(mock_engine)

        # Get initial stats
        stats = enricher.get_stats()
        self.assertEqual(stats["vendor_enrichments"], 0)
        self.assertEqual(stats["product_enrichments"], 0)
        self.assertEqual(stats["currency_lookups"], 0)


class TestCachePerformance(unittest.TestCase):
    """Test cache performance and hit rate."""

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        temp_dir = tempfile.mkdtemp()
        cache = SearchCacheManager(
            disk_cache_dir=temp_dir,
            ttl=60
        )

        # Simulate cache hits and misses
        test_data = [{"title": "Test"}]
        cache.set("key1", test_data)
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit

        stats = cache.get_stats()
        hit_rate = stats["hit_rate"]

        # 3 hits out of 4 requests = 75%
        self.assertAlmostEqual(hit_rate, 0.75, places=2)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestSuccessCriteria(unittest.TestCase):
    """Test success criteria verification."""

    def test_latency_target(self):
        """Test that cache reads meet latency target (<200ms)."""
        temp_dir = tempfile.mkdtemp()
        cache = SearchCacheManager(disk_cache_dir=temp_dir)

        test_data = [{"title": "Latency Test"}]
        cache.set("latency_key", test_data)

        # Measure cache read latency
        start = time.time()
        result = cache.get("latency_key")
        latency_ms = (time.time() - start) * 1000

        # Cache read should be <10ms (well under 200ms target)
        self.assertLess(latency_ms, 10)
        self.assertIsNotNone(result)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_integration_tests():
    """Run all integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestThreeTierCaching))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiProviderFailover))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractionEnrichment))
    suite.addTests(loader.loadTestsFromTestCase(TestCachePerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestSuccessCriteria))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
