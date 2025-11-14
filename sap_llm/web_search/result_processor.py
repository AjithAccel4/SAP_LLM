"""
Search Result Processing and Ranking.

Validates, filters, ranks, and sanitizes search results.
"""

import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ResultProcessor:
    """
    Process and rank search results.

    Features:
    - Domain whitelisting/blacklisting
    - Result deduplication
    - Relevance scoring and ranking
    - Content sanitization
    - URL validation
    - Spam filtering

    Example:
        >>> processor = ResultProcessor(
        ...     trusted_domains=["sap.com", "microsoft.com"],
        ...     blocked_domains=["spam.com"]
        ... )
        >>> results = processor.process_results(raw_results, query)
    """

    def __init__(
        self,
        trusted_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        min_relevance_score: float = 0.0
    ):
        """
        Initialize result processor.

        Args:
            trusted_domains: List of trusted domains (boosts ranking)
            blocked_domains: List of blocked domains (filtered out)
            min_relevance_score: Minimum relevance score to include
        """
        self.trusted_domains = set(trusted_domains or [])
        self.blocked_domains = set(blocked_domains or [])
        self.min_relevance_score = min_relevance_score

        # Default trusted domains for SAP-related searches
        self.trusted_domains.update([
            "sap.com",
            "help.sap.com",
            "api.sap.com",
            "developers.sap.com",
            "community.sap.com",
            "support.sap.com",
            "wikipedia.org",
            "github.com"
        ])

        # Default blocked domains
        self.blocked_domains.update([
            "example.com",
            "localhost"
        ])

        logger.info(
            f"ResultProcessor initialized: "
            f"{len(self.trusted_domains)} trusted domains, "
            f"{len(self.blocked_domains)} blocked domains"
        )

    def process_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Process and rank search results.

        Args:
            results: Raw search results from provider
            query: Original search query

        Returns:
            Processed and ranked results
        """
        if not results:
            return []

        processed = []

        for result in results:
            # Validate result structure
            if not self._validate_result(result):
                continue

            # Filter blocked domains
            if self._is_blocked(result):
                logger.debug(f"Blocked result from: {result.get('url')}")
                continue

            # Sanitize content
            result = self._sanitize_result(result)

            # Calculate relevance score
            relevance = self._calculate_relevance(result, query)
            result["relevance_score"] = relevance

            # Filter by minimum relevance
            if relevance < self.min_relevance_score:
                continue

            processed.append(result)

        # Deduplicate results
        processed = self._deduplicate(processed)

        # Rank results
        processed = self._rank_results(processed, query)

        logger.info(
            f"Processed {len(results)} results -> {len(processed)} after filtering"
        )

        return processed

    def _validate_result(self, result: Dict[str, Any]) -> bool:
        """Validate result has required fields."""
        required_fields = ["title", "url", "snippet"]

        for field in required_fields:
            if field not in result or not result[field]:
                logger.warning(f"Result missing required field: {field}")
                return False

        # Validate URL format
        try:
            parsed = urlparse(result["url"])
            if not parsed.scheme or not parsed.netloc:
                logger.warning(f"Invalid URL: {result['url']}")
                return False
        except Exception as e:
            logger.warning(f"URL parsing error: {e}")
            return False

        return True

    def _is_blocked(self, result: Dict[str, Any]) -> bool:
        """Check if result is from blocked domain."""
        try:
            url = result.get("url", "")
            domain = urlparse(url).netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check exact match
            if domain in self.blocked_domains:
                return True

            # Check if any blocked domain is a suffix
            for blocked in self.blocked_domains:
                if domain.endswith(f".{blocked}") or domain == blocked:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Error checking blocked domain: {e}")
            return False

    def _is_trusted(self, result: Dict[str, Any]) -> bool:
        """Check if result is from trusted domain."""
        try:
            url = result.get("url", "")
            domain = urlparse(url).netloc.lower()

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check exact match
            if domain in self.trusted_domains:
                return True

            # Check if any trusted domain is a suffix
            for trusted in self.trusted_domains:
                if domain.endswith(f".{trusted}") or domain == trusted:
                    return True

            return False

        except Exception:
            return False

    def _sanitize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize result content."""
        # Clean title
        result["title"] = self._clean_text(result.get("title", ""))

        # Clean snippet
        result["snippet"] = self._clean_text(result.get("snippet", ""))

        # Ensure URL is properly formatted
        url = result.get("url", "")
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        result["url"] = url

        return result

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:()\-\'/"]', '', text)

        # Trim
        text = text.strip()

        return text

    def _calculate_relevance(
        self,
        result: Dict[str, Any],
        query: str
    ) -> float:
        """
        Calculate relevance score for result.

        Args:
            result: Search result
            query: Original query

        Returns:
            Relevance score between 0.0 and 1.0
        """
        score = 0.0
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Get result text
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        url = result.get("url", "").lower()

        # Base score from provider (if available)
        if "score" in result:
            score += result["score"] * 0.3

        # Title match (weighted higher)
        title_terms = set(title.split())
        title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
        score += title_overlap * 0.4

        # Snippet match
        snippet_terms = set(snippet.split())
        snippet_overlap = len(query_terms & snippet_terms) / max(len(query_terms), 1)
        score += snippet_overlap * 0.2

        # Exact phrase match bonus
        if query_lower in title:
            score += 0.2
        elif query_lower in snippet:
            score += 0.1

        # URL quality indicators
        if any(term in url for term in ["docs", "documentation", "help", "api"]):
            score += 0.1

        # Trusted domain bonus
        if self._is_trusted(result):
            score += 0.15

        # Normalize to 0-1 range
        score = min(1.0, score)

        return score

    def _rank_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rank results by relevance and other factors.

        Args:
            results: List of results to rank
            query: Original query

        Returns:
            Sorted list of results
        """
        # Sort by relevance score (descending)
        ranked = sorted(
            results,
            key=lambda x: (
                x.get("relevance_score", 0.0),
                self._is_trusted(x),  # Trusted domains higher
                -len(x.get("url", ""))  # Shorter URLs slightly preferred
            ),
            reverse=True
        )

        # Add rank position
        for i, result in enumerate(ranked):
            result["rank"] = i + 1

        return ranked

    def _deduplicate(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on URL.

        Args:
            results: List of results

        Returns:
            Deduplicated list
        """
        seen_urls: Set[str] = set()
        seen_normalized: Set[str] = set()
        unique = []

        for result in results:
            url = result.get("url", "")

            # Normalize URL for comparison
            normalized = self._normalize_url(url)

            # Skip if seen
            if url in seen_urls or normalized in seen_normalized:
                logger.debug(f"Duplicate result: {url}")
                continue

            seen_urls.add(url)
            seen_normalized.add(normalized)
            unique.append(result)

        return unique

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        try:
            parsed = urlparse(url.lower())

            # Remove www. prefix
            netloc = parsed.netloc
            if netloc.startswith("www."):
                netloc = netloc[4:]

            # Remove trailing slash from path
            path = parsed.path.rstrip("/")

            # Ignore fragments and common tracking parameters
            query_params = parsed.query
            # Remove common tracking params
            tracking_params = ["utm_source", "utm_medium", "utm_campaign", "ref"]
            if query_params:
                params = [
                    p for p in query_params.split("&")
                    if not any(p.startswith(f"{tp}=") for tp in tracking_params)
                ]
                query_params = "&".join(params)

            # Reconstruct normalized URL
            normalized = f"{parsed.scheme}://{netloc}{path}"
            if query_params:
                normalized += f"?{query_params}"

            return normalized

        except Exception:
            return url.lower()

    def filter_by_domain(
        self,
        results: List[Dict[str, Any]],
        domains: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter results to only include specific domains.

        Args:
            results: List of results
            domains: List of domains to include

        Returns:
            Filtered results
        """
        filtered = []
        domains_set = set(d.lower() for d in domains)

        for result in results:
            try:
                url = result.get("url", "")
                domain = urlparse(url).netloc.lower()

                if domain.startswith("www."):
                    domain = domain[4:]

                # Check if domain matches
                if domain in domains_set:
                    filtered.append(result)
                else:
                    # Check for subdomain match
                    for allowed_domain in domains_set:
                        if domain.endswith(f".{allowed_domain}") or domain == allowed_domain:
                            filtered.append(result)
                            break

            except Exception:
                continue

        return filtered

    def extract_entities(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Set[str]]:
        """
        Extract common entities from search results.

        Args:
            results: List of search results

        Returns:
            Dictionary of entity types to sets of entities
        """
        entities = {
            "urls": set(),
            "domains": set(),
            "emails": set(),
            "phones": set()
        }

        for result in results:
            # Extract URL and domain
            url = result.get("url", "")
            if url:
                entities["urls"].add(url)
                try:
                    domain = urlparse(url).netloc
                    entities["domains"].add(domain)
                except Exception:
                    pass

            # Extract from snippet
            snippet = result.get("snippet", "")

            # Email pattern
            emails = re.findall(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                snippet
            )
            entities["emails"].update(emails)

            # Phone pattern (simple)
            phones = re.findall(
                r'\b\+?[\d\s\-\(\)]{10,}\b',
                snippet
            )
            entities["phones"].update(phones)

        # Convert sets to sorted lists for consistency
        return {k: sorted(list(v)) for k, v in entities.items()}
