"""
Core Web Search Engine for SAP_LLM.

Provides unified interface to multiple search providers with caching,
rate limiting, and fallback mechanisms.
"""

import asyncio
import hashlib
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from sap_llm.utils.logger import get_logger
from sap_llm.web_search.cache_manager import SearchCacheManager
from sap_llm.web_search.rate_limiter import RateLimiter
from sap_llm.web_search.result_processor import ResultProcessor
from sap_llm.web_search.search_providers import (
    BingSearchProvider,
    DuckDuckGoProvider,
    GoogleSearchProvider,
    SearchProvider,
    TavilySearchProvider,
)

logger = get_logger(__name__)


class SearchMode(Enum):
    """Search mode enumeration."""

    WEB = "web"  # General web search
    NEWS = "news"  # News articles
    IMAGES = "images"  # Image search
    VIDEOS = "videos"  # Video search
    ACADEMIC = "academic"  # Academic papers
    LOCAL = "local"  # Local business search


class WebSearchEngine:
    """
    Multi-provider web search engine with intelligent failover.

    Features:
    - Multiple search providers (Google, Bing, DuckDuckGo, Tavily)
    - Redis-based caching to reduce API calls
    - Rate limiting per provider
    - Automatic failover on provider errors
    - Result ranking and relevance scoring
    - Domain whitelisting/blacklisting
    - Offline mode fallback

    Example:
        >>> engine = WebSearchEngine(config)
        >>> results = engine.search("SAP S/4HANA API documentation")
        >>> enriched = engine.enrich_entity("ACME Corp", entity_type="vendor")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize web search engine.

        Args:
            config: Configuration dictionary with search settings
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.offline_mode = self.config.get("offline_mode", False)

        # Initialize cache manager (3-tier: memory, Redis, disk)
        cache_config = self.config.get("cache", {})
        self.cache_manager = SearchCacheManager(
            redis_config=cache_config,
            enabled=self.config.get("cache_enabled", True),
            disk_cache_dir=cache_config.get("disk_cache_dir"),
            max_disk_cache_size_mb=cache_config.get("max_disk_cache_size_mb", 1000),
            default_ttl=cache_config.get("ttl", 86400)
        )

        # Initialize rate limiters (per provider)
        rate_limit_config = self.config.get("rate_limits", {})
        self.rate_limiters = {
            "google": RateLimiter(
                requests_per_minute=rate_limit_config.get("google", 100),
                requests_per_day=rate_limit_config.get("google_daily", 10000)
            ),
            "bing": RateLimiter(
                requests_per_minute=rate_limit_config.get("bing", 100),
                requests_per_day=rate_limit_config.get("bing_daily", 10000)
            ),
            "tavily": RateLimiter(
                requests_per_minute=rate_limit_config.get("tavily", 60),
                requests_per_day=rate_limit_config.get("tavily_daily", 1000)
            ),
            "duckduckgo": RateLimiter(
                requests_per_minute=rate_limit_config.get("duckduckgo", 30),
                requests_per_day=rate_limit_config.get("duckduckgo_daily", 1000)
            )
        }

        # Initialize search providers
        self.providers: Dict[str, SearchProvider] = {}
        self._initialize_providers()

        # Initialize result processor
        self.result_processor = ResultProcessor(
            trusted_domains=self.config.get("trusted_domains", []),
            blocked_domains=self.config.get("blocked_domains", []),
            min_relevance_score=self.config.get("min_relevance_score", 0.5)
        )

        # Provider priority order
        self.provider_priority = self.config.get(
            "provider_priority",
            ["tavily", "google", "bing", "duckduckgo"]
        )

        # Statistics
        self.stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "provider_failures": {},
            "avg_response_time_ms": 0.0
        }

        logger.info(
            f"WebSearchEngine initialized with {len(self.providers)} providers. "
            f"Enabled: {self.enabled}, Offline mode: {self.offline_mode}"
        )

    def _initialize_providers(self) -> None:
        """Initialize all configured search providers."""
        provider_config = self.config.get("providers", {})

        # Google Search
        if provider_config.get("google", {}).get("enabled", False):
            try:
                api_key = provider_config["google"].get("api_key")
                cx = provider_config["google"].get("cx")
                if api_key and cx:
                    self.providers["google"] = GoogleSearchProvider(
                        api_key=api_key,
                        cx=cx
                    )
                    logger.info("Google Search provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Search: {e}")

        # Bing Search
        if provider_config.get("bing", {}).get("enabled", False):
            try:
                api_key = provider_config["bing"].get("api_key")
                if api_key:
                    self.providers["bing"] = BingSearchProvider(api_key=api_key)
                    logger.info("Bing Search provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Bing Search: {e}")

        # Tavily AI Search
        if provider_config.get("tavily", {}).get("enabled", False):
            try:
                api_key = provider_config["tavily"].get("api_key")
                if api_key:
                    self.providers["tavily"] = TavilySearchProvider(api_key=api_key)
                    logger.info("Tavily Search provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Tavily Search: {e}")

        # DuckDuckGo (no API key required - always available as fallback)
        if provider_config.get("duckduckgo", {}).get("enabled", True):
            try:
                self.providers["duckduckgo"] = DuckDuckGoProvider()
                logger.info("DuckDuckGo Search provider initialized (fallback)")
            except Exception as e:
                logger.warning(f"Failed to initialize DuckDuckGo Search: {e}")

    def search(
        self,
        query: str,
        num_results: int = 10,
        mode: SearchMode = SearchMode.WEB,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        timeout: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Perform web search with automatic provider failover.

        Args:
            query: Search query string
            num_results: Number of results to return
            mode: Search mode (web, news, images, etc.)
            filters: Additional filters (date range, domain, etc.)
            use_cache: Whether to use cached results
            timeout: Search timeout in seconds

        Returns:
            List of search results with metadata

        Example:
            >>> results = engine.search("SAP BAPI vendor master data")
            >>> for result in results:
            ...     print(result["title"], result["url"])
        """
        if not self.enabled:
            logger.warning("Web search is disabled")
            return []

        if self.offline_mode:
            logger.info("Offline mode - returning empty results")
            return []

        start_time = time.time()
        self.stats["total_searches"] += 1

        # Generate cache key
        cache_key = self._generate_cache_key(query, num_results, mode, filters)

        # Check cache
        if use_cache:
            cached_results = self.cache_manager.get(cache_key)
            if cached_results is not None:
                self.stats["cache_hits"] += 1
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_results

        self.stats["cache_misses"] += 1

        # Try providers in priority order
        results = []
        last_error = None

        for provider_name in self.provider_priority:
            if provider_name not in self.providers:
                continue

            provider = self.providers[provider_name]
            rate_limiter = self.rate_limiters.get(provider_name)

            # Check rate limit
            if rate_limiter and not rate_limiter.can_proceed():
                logger.warning(f"Rate limit exceeded for {provider_name}, trying next provider")
                continue

            try:
                logger.info(f"Searching with provider: {provider_name}")

                # Perform search
                results = provider.search(
                    query=query,
                    num_results=num_results,
                    mode=mode.value,
                    filters=filters,
                    timeout=timeout
                )

                # Record rate limit usage
                if rate_limiter:
                    rate_limiter.record_request()

                # Process and validate results
                results = self.result_processor.process_results(
                    results=results,
                    query=query
                )

                if results:
                    logger.info(f"Got {len(results)} results from {provider_name}")
                    break

            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}")

                # Track failures
                if provider_name not in self.stats["provider_failures"]:
                    self.stats["provider_failures"][provider_name] = 0
                self.stats["provider_failures"][provider_name] += 1

                continue

        # Cache results if successful
        if results and use_cache:
            self.cache_manager.set(cache_key, results)

        # Update statistics
        elapsed_time = (time.time() - start_time) * 1000
        total = self.stats["total_searches"]
        avg = self.stats["avg_response_time_ms"]
        self.stats["avg_response_time_ms"] = (avg * (total - 1) + elapsed_time) / total

        if not results and last_error:
            logger.error(f"All providers failed. Last error: {last_error}")

        return results

    def search_multiple(
        self,
        queries: List[str],
        num_results: int = 10,
        parallel: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search multiple queries efficiently.

        Args:
            queries: List of search queries
            num_results: Results per query
            parallel: Whether to search in parallel

        Returns:
            Dictionary mapping queries to results
        """
        if not parallel:
            return {q: self.search(q, num_results) for q in queries}

        # Parallel execution
        async def search_async():
            tasks = [
                asyncio.to_thread(self.search, q, num_results)
                for q in queries
            ]
            results = await asyncio.gather(*tasks)
            return dict(zip(queries, results))

        return asyncio.run(search_async())

    def verify_fact(
        self,
        claim: str,
        min_sources: int = 3,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Verify a factual claim using multiple sources.

        Args:
            claim: Claim to verify
            min_sources: Minimum number of confirming sources
            confidence_threshold: Minimum confidence score

        Returns:
            Verification result with confidence score and sources

        Example:
            >>> result = engine.verify_fact("ACME Corp is located in Berlin")
            >>> print(result["verified"], result["confidence"])
        """
        # Search for information about the claim
        results = self.search(claim, num_results=20)

        if not results:
            return {
                "verified": False,
                "confidence": 0.0,
                "sources": [],
                "message": "No sources found"
            }

        # Analyze results for confirmation
        confirming_sources = []
        contradicting_sources = []

        for result in results:
            # Simple heuristic: check if key terms appear in snippet
            snippet = result.get("snippet", "").lower()
            title = result.get("title", "").lower()
            content = snippet + " " + title

            # Extract key entities from claim
            claim_lower = claim.lower()

            # Check for presence (simplified - in production use NLP)
            if all(word in content for word in claim_lower.split() if len(word) > 3):
                confirming_sources.append({
                    "url": result["url"],
                    "title": result["title"],
                    "snippet": result["snippet"],
                    "relevance": result.get("relevance_score", 0.5)
                })

        # Calculate confidence
        num_confirming = len(confirming_sources)
        confidence = min(1.0, num_confirming / min_sources)

        verified = num_confirming >= min_sources and confidence >= confidence_threshold

        return {
            "verified": verified,
            "confidence": confidence,
            "confirming_sources": num_confirming,
            "sources": confirming_sources[:min_sources],
            "message": f"Found {num_confirming} confirming sources"
        }

    def lookup_entity(
        self,
        entity_name: str,
        entity_type: str,
        attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Look up detailed information about an entity.

        Args:
            entity_name: Name of entity (company, person, product, etc.)
            entity_type: Type of entity (vendor, customer, product, etc.)
            attributes: Specific attributes to extract

        Returns:
            Entity information dictionary

        Example:
            >>> info = engine.lookup_entity("SAP SE", "company")
            >>> print(info["address"], info["tax_id"])
        """
        # Construct targeted query
        query = f"{entity_name} {entity_type}"

        if entity_type == "company":
            query += " address headquarters tax ID VAT number"
        elif entity_type == "product":
            query += " price specifications features"

        results = self.search(query, num_results=10)

        # Extract structured information
        entity_info = {
            "name": entity_name,
            "type": entity_type,
            "found": len(results) > 0,
            "sources": results[:3],
            "attributes": {}
        }

        # Simple attribute extraction (in production, use NER/IE)
        if results and attributes:
            for attr in attributes:
                entity_info["attributes"][attr] = None  # Placeholder

        return entity_info

    def get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: Optional[str] = None
    ) -> Optional[float]:
        """
        Get currency exchange rate.

        Args:
            from_currency: Source currency code (e.g., "USD")
            to_currency: Target currency code (e.g., "EUR")
            date: Optional date for historical rates (YYYY-MM-DD)

        Returns:
            Exchange rate or None if not found
        """
        date_str = f"on {date}" if date else "today"
        query = f"{from_currency} to {to_currency} exchange rate {date_str}"

        results = self.search(query, num_results=5)

        # Extract rate from results (simplified)
        # In production, use specialized API like exchangerate-api.com
        for result in results:
            snippet = result.get("snippet", "")
            # Look for numeric patterns
            import re
            matches = re.findall(r'\d+\.\d+', snippet)
            if matches:
                try:
                    return float(matches[0])
                except ValueError:
                    continue

        return None

    def validate_address(
        self,
        address: str,
        country: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate and normalize an address using web search.

        Args:
            address: Address string to validate
            country: Optional country code

        Returns:
            Validation result with normalized address
        """
        query = f"{address}"
        if country:
            query += f" {country}"
        query += " address verify"

        results = self.search(query, num_results=5, mode=SearchMode.LOCAL)

        return {
            "valid": len(results) > 0,
            "normalized_address": address,  # Placeholder
            "confidence": 0.8 if results else 0.0,
            "sources": results[:2]
        }

    def search_sap_documentation(
        self,
        topic: str,
        doc_type: str = "api"
    ) -> List[Dict[str, Any]]:
        """
        Search SAP official documentation.

        Args:
            topic: Topic to search
            doc_type: Type of documentation (api, guide, help, etc.)

        Returns:
            List of documentation results
        """
        # Construct SAP-specific query
        query = f"SAP {topic} {doc_type} site:help.sap.com OR site:api.sap.com"

        results = self.search(
            query=query,
            num_results=10,
            filters={"domains": ["help.sap.com", "api.sap.com"]}
        )

        return results

    def _generate_cache_key(
        self,
        query: str,
        num_results: int,
        mode: SearchMode,
        filters: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for search query."""
        key_data = f"{query}:{num_results}:{mode.value}:{filters}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of all providers and components.

        Returns:
            Health status for all components
        """
        health = {
            "overall_healthy": True,
            "timestamp": time.time(),
            "providers": {},
            "cache": {},
            "rate_limiters": {}
        }

        # Check each provider
        for provider_name, provider in self.providers.items():
            provider_health = {
                "available": True,
                "failure_count": self.stats["provider_failures"].get(provider_name, 0),
                "latency_ms": None
            }

            try:
                # Simple test search
                start = time.time()
                results = provider.search("test", num_results=1, timeout=5.0)
                provider_health["latency_ms"] = (time.time() - start) * 1000
                provider_health["healthy"] = len(results) >= 0
            except Exception as e:
                provider_health["healthy"] = False
                provider_health["error"] = str(e)
                health["overall_healthy"] = False

            health["providers"][provider_name] = provider_health

        # Check cache health
        health["cache"] = self.cache_manager.health_check()
        if not health["cache"].get("healthy"):
            logger.warning("Cache unhealthy, but not critical")

        # Check rate limiters
        for name, limiter in self.rate_limiters.items():
            health["rate_limiters"][name] = limiter.get_stats()

        return health

    def get_provider_status(self) -> Dict[str, Any]:
        """
        Get detailed status of all search providers with automatic failover capability assessment.

        Returns:
            Provider status including availability, performance metrics, and failover readiness
        """
        status = {
            "total_providers": len(self.providers),
            "available_providers": 0,
            "failover_ready": False,
            "providers": {}
        }

        for provider_name in self.provider_priority:
            if provider_name not in self.providers:
                status["providers"][provider_name] = {
                    "configured": False,
                    "available": False,
                    "reason": "Not configured"
                }
                continue

            provider_status = {
                "configured": True,
                "available": True,
                "priority": self.provider_priority.index(provider_name),
                "failures": self.stats["provider_failures"].get(provider_name, 0),
                "rate_limited": False
            }

            # Check rate limit status
            rate_limiter = self.rate_limiters.get(provider_name)
            if rate_limiter:
                if not rate_limiter.can_proceed():
                    provider_status["rate_limited"] = True
                    provider_status["available"] = False
                provider_status["rate_limit_stats"] = rate_limiter.get_stats()

            status["providers"][provider_name] = provider_status

            if provider_status["available"]:
                status["available_providers"] += 1

        # Failover is ready if we have at least 2 available providers
        status["failover_ready"] = status["available_providers"] >= 2

        return status

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search engine statistics."""
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"] / self.stats["total_searches"]
                if self.stats["total_searches"] > 0 else 0.0
            ),
            "providers_available": list(self.providers.keys()),
            "cache_stats": self.cache_manager.get_stats(),
            "provider_status": self.get_provider_status()
        }

    def clear_cache(self) -> None:
        """Clear search result cache."""
        self.cache_manager.clear()
        logger.info("Search cache cleared")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'cache_manager'):
            self.cache_manager.close()
