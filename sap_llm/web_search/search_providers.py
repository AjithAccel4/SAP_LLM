"""
Search Provider Implementations.

Concrete implementations for Google, Bing, DuckDuckGo, and Tavily search APIs.
"""

import abc
import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SearchProvider(abc.ABC):
    """
    Abstract base class for search providers.

    All search providers must implement the search() method.
    """

    @abc.abstractmethod
    def search(
        self,
        query: str,
        num_results: int = 10,
        mode: str = "web",
        filters: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Perform search and return results.

        Args:
            query: Search query string
            num_results: Number of results to return
            mode: Search mode (web, news, images, etc.)
            filters: Additional filters
            timeout: Request timeout in seconds

        Returns:
            List of search result dictionaries with keys:
                - title: Result title
                - url: Result URL
                - snippet: Result description/snippet
                - source: Provider name
                - timestamp: When result was fetched
        """
        pass


class GoogleSearchProvider(SearchProvider):
    """
    Google Custom Search API provider.

    Requires:
    - API Key from Google Cloud Console
    - Custom Search Engine ID (cx)

    Documentation: https://developers.google.com/custom-search/v1/overview
    """

    def __init__(self, api_key: str, cx: str):
        """
        Initialize Google Search provider.

        Args:
            api_key: Google API key
            cx: Custom Search Engine ID
        """
        self.api_key = api_key
        self.cx = cx
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(
        self,
        query: str,
        num_results: int = 10,
        mode: str = "web",
        filters: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Perform Google search."""
        results = []
        filters = filters or {}

        try:
            # Build parameters
            params = {
                "key": self.api_key,
                "cx": self.cx,
                "q": query,
                "num": min(num_results, 10)  # Max 10 per request
            }

            # Add search type
            if mode == "images":
                params["searchType"] = "image"

            # Add date filter if specified
            if "date_range" in filters:
                params["dateRestrict"] = filters["date_range"]

            # Add site restriction if specified
            if "domains" in filters:
                sites = " OR ".join([f"site:{d}" for d in filters["domains"]])
                params["q"] = f"{query} ({sites})"

            # Make request
            response = requests.get(
                self.base_url,
                params=params,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()

            # Parse results
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google",
                    "timestamp": time.time()
                })

            logger.info(f"Google search returned {len(results)} results for: {query[:50]}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Google search API error: {e}")
            raise

        return results


class BingSearchProvider(SearchProvider):
    """
    Bing Search API provider.

    Requires API key from Azure Cognitive Services.

    Documentation: https://docs.microsoft.com/en-us/bing/search-apis/
    """

    def __init__(self, api_key: str):
        """
        Initialize Bing Search provider.

        Args:
            api_key: Bing Search API key
        """
        self.api_key = api_key
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"

    def search(
        self,
        query: str,
        num_results: int = 10,
        mode: str = "web",
        filters: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Perform Bing search."""
        results = []
        filters = filters or {}

        try:
            # Build parameters
            params = {
                "q": query,
                "count": min(num_results, 50),  # Max 50 per request
                "responseFilter": "Webpages"
            }

            # Add freshness filter
            if "date_range" in filters:
                params["freshness"] = filters["date_range"]

            # Add market/region
            params["mkt"] = filters.get("market", "en-US")

            # Headers
            headers = {
                "Ocp-Apim-Subscription-Key": self.api_key
            }

            # Make request
            response = requests.get(
                self.base_url,
                params=params,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()

            # Parse results
            for item in data.get("webPages", {}).get("value", []):
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "bing",
                    "timestamp": time.time()
                })

            logger.info(f"Bing search returned {len(results)} results for: {query[:50]}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Bing search API error: {e}")
            raise

        return results


class TavilySearchProvider(SearchProvider):
    """
    Tavily AI Search provider.

    AI-optimized search API designed for LLM applications.
    Returns highly relevant, fact-checked results.

    Documentation: https://tavily.com
    """

    def __init__(self, api_key: str):
        """
        Initialize Tavily Search provider.

        Args:
            api_key: Tavily API key
        """
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"

    def search(
        self,
        query: str,
        num_results: int = 10,
        mode: str = "web",
        filters: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Perform Tavily AI search."""
        results = []
        filters = filters or {}

        try:
            # Build request payload
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": min(num_results, 20),
                "search_depth": filters.get("depth", "basic"),  # basic or advanced
                "include_answer": False,
                "include_raw_content": False
            }

            # Add domain filtering
            if "domains" in filters:
                payload["include_domains"] = filters["domains"]

            # Make request
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            data = response.json()

            # Parse results
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "source": "tavily",
                    "timestamp": time.time(),
                    "score": item.get("score", 0.0)  # Tavily provides relevance scores
                })

            logger.info(f"Tavily search returned {len(results)} results for: {query[:50]}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily search API error: {e}")
            raise

        return results


class DuckDuckGoProvider(SearchProvider):
    """
    DuckDuckGo search provider (no API key required).

    Uses DuckDuckGo's HTML search interface. Slower and less reliable
    than API-based providers, but useful as a fallback.

    Note: This is a basic implementation. For production use, consider
    using the duckduckgo-search Python package.
    """

    def __init__(self):
        """Initialize DuckDuckGo provider."""
        self.base_url = "https://html.duckduckgo.com/html/"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def search(
        self,
        query: str,
        num_results: int = 10,
        mode: str = "web",
        filters: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> List[Dict[str, Any]]:
        """Perform DuckDuckGo search."""
        results = []

        try:
            # Try to import duckduckgo_search if available
            try:
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    search_results = ddgs.text(
                        query,
                        max_results=num_results
                    )

                    for item in search_results:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("href", ""),
                            "snippet": item.get("body", ""),
                            "source": "duckduckgo",
                            "timestamp": time.time()
                        })

                logger.info(f"DuckDuckGo search returned {len(results)} results")

            except ImportError:
                # Fallback to basic HTML parsing
                logger.warning(
                    "duckduckgo_search package not available. "
                    "Install with: pip install duckduckgo-search"
                )

                # Make request to HTML interface
                params = {"q": query}
                response = self.session.post(
                    self.base_url,
                    data=params,
                    timeout=timeout
                )
                response.raise_for_status()

                # Parse HTML (basic implementation)
                # In production, use BeautifulSoup or similar
                html = response.text

                # This is a simplified parser - in production use proper HTML parsing
                import re

                # Extract result links
                link_pattern = r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
                snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>'

                links = re.findall(link_pattern, html)
                snippets = re.findall(snippet_pattern, html)

                for i, (url, title) in enumerate(links[:num_results]):
                    snippet = snippets[i] if i < len(snippets) else ""

                    results.append({
                        "title": title.strip(),
                        "url": url.strip(),
                        "snippet": snippet.strip(),
                        "source": "duckduckgo",
                        "timestamp": time.time()
                    })

                logger.info(f"DuckDuckGo (HTML) returned {len(results)} results")

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            # Don't raise - DuckDuckGo is a fallback, failures are expected

        return results


class SAPHelpSearchProvider(SearchProvider):
    """
    Specialized provider for SAP official documentation.

    Searches SAP Help Portal and API documentation sites.
    """

    def __init__(self):
        """Initialize SAP Help search provider."""
        self.sap_domains = [
            "help.sap.com",
            "api.sap.com",
            "developers.sap.com",
            "community.sap.com",
            "support.sap.com"
        ]

    def search(
        self,
        query: str,
        num_results: int = 10,
        mode: str = "web",
        filters: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Search SAP documentation.

        This is a wrapper that uses DuckDuckGo with SAP domain filtering.
        """
        # Use DuckDuckGo as underlying provider
        ddg = DuckDuckGoProvider()

        # Add site restrictions to query
        site_filter = " OR ".join([f"site:{domain}" for domain in self.sap_domains])
        enhanced_query = f"{query} ({site_filter})"

        results = ddg.search(
            query=enhanced_query,
            num_results=num_results,
            mode=mode,
            filters=filters,
            timeout=timeout
        )

        # Update source
        for result in results:
            result["source"] = "sap_help"

        return results


class ExchangeRateProvider:
    """
    Specialized provider for currency exchange rates.

    Uses free exchangerate-api.com service.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize exchange rate provider.

        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
        if api_key:
            self.base_url = f"https://v6.exchangerate-api.com/v6/{api_key}"
        else:
            # Use free tier
            self.base_url = "https://api.exchangerate.host"

    def get_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: Optional[str] = None
    ) -> Optional[float]:
        """
        Get exchange rate between currencies.

        Args:
            from_currency: Source currency code (e.g., "USD")
            to_currency: Target currency code (e.g., "EUR")
            date: Optional date (YYYY-MM-DD) for historical rates

        Returns:
            Exchange rate or None if error
        """
        try:
            if self.api_key:
                # Use exchangerate-api.com
                url = f"{self.base_url}/pair/{from_currency}/{to_currency}"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                return data.get("conversion_rate")
            else:
                # Use exchangerate.host (free)
                if date:
                    url = f"{self.base_url}/{date}"
                else:
                    url = f"{self.base_url}/latest"

                params = {
                    "base": from_currency,
                    "symbols": to_currency
                }

                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()

                return data.get("rates", {}).get(to_currency)

        except Exception as e:
            logger.error(f"Exchange rate lookup error: {e}")
            return None
