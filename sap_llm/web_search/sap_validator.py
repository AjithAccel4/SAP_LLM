"""
SAP Source Validator with Trust Scoring.

Validates search results against SAP official sources and assigns detailed
trust scores based on domain authority, content type, and freshness.
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SAPSourceValidator:
    """
    Validator for SAP-related search results with trust scoring.

    Features:
    - Multi-tier domain trust classification
    - Content type detection (API docs, tutorials, forums, etc.)
    - Freshness scoring
    - Official documentation verification
    - Community content credibility assessment

    Example:
        >>> validator = SAPSourceValidator()
        >>> validated = validator.validate_results(results)
        >>> high_trust = [r for r in validated if r["trust_score"] >= 0.8]
    """

    # Trusted SAP domains with authority levels (0.0-1.0)
    TRUSTED_DOMAINS = {
        # Tier 1: Official SAP Documentation (Highest Trust)
        "help.sap.com": 1.0,
        "api.sap.com": 1.0,
        "support.sap.com": 1.0,
        "launchpad.support.sap.com": 1.0,

        # Tier 2: Official SAP Platforms (High Trust)
        "developers.sap.com": 0.9,
        "community.sap.com": 0.85,
        "answers.sap.com": 0.85,
        "blogs.sap.com": 0.8,
        "learning.sap.com": 0.85,

        # Tier 3: SAP Corporate & Product Pages (Good Trust)
        "sap.com": 0.75,
        "news.sap.com": 0.7,
        "events.sap.com": 0.7,

        # Tier 4: SAP Partner/Ecosystem (Moderate Trust)
        "sapinsider.org": 0.65,
        "sapcommunity.com": 0.6,
    }

    # Content type indicators and their weights
    CONTENT_TYPE_INDICATORS = {
        "api_documentation": {
            "patterns": [
                r"/api/",
                r"/odata/",
                r"/reference/",
                r"/apidoc/",
                r"swagger",
                r"openapi"
            ],
            "boost": 0.3
        },
        "official_help": {
            "patterns": [
                r"help\.sap\.com",
                r"/documentation/",
                r"/guides/",
                r"/help/"
            ],
            "boost": 0.25
        },
        "tutorial": {
            "patterns": [
                r"/tutorial/",
                r"/getting-started/",
                r"/quickstart/",
                r"step-by-step"
            ],
            "boost": 0.2
        },
        "blog": {
            "patterns": [
                r"/blog/",
                r"/posts/",
                r"/article/"
            ],
            "boost": 0.1
        },
        "forum": {
            "patterns": [
                r"/questions/",
                r"/answers/",
                r"/community/",
                r"/forums/"
            ],
            "boost": 0.05
        }
    }

    # URL patterns that indicate official SAP content
    OFFICIAL_PATTERNS = [
        r"help\.sap\.com/.*/(en|EN)/",
        r"api\.sap\.com/api/",
        r"support\.sap\.com/.*SAP_Notes",
        r"developers\.sap\.com/tutorials/",
    ]

    def __init__(
        self,
        min_trust_score: float = 0.5,
        freshness_weight: float = 0.1,
        require_https: bool = True
    ):
        """
        Initialize SAP source validator.

        Args:
            min_trust_score: Minimum trust score threshold (0.0-1.0)
            freshness_weight: Weight for content freshness (0.0-1.0)
            require_https: Whether to require HTTPS URLs
        """
        self.min_trust_score = min_trust_score
        self.freshness_weight = freshness_weight
        self.require_https = require_https

        # Statistics
        self.stats = {
            "total_validated": 0,
            "high_trust_count": 0,
            "medium_trust_count": 0,
            "low_trust_count": 0,
            "rejected_count": 0
        }

        logger.info(
            f"SAPSourceValidator initialized with min_trust_score={min_trust_score}"
        )

    def validate_results(
        self,
        results: List[Dict[str, Any]],
        require_sap_domain: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Validate and score search results.

        Args:
            results: List of search results
            require_sap_domain: Whether to filter only SAP-related domains

        Returns:
            Validated results with trust scores and metadata
        """
        if not results:
            return []

        validated_results = []

        for result in results:
            self.stats["total_validated"] += 1

            # Calculate trust score
            trust_score = self._calculate_trust_score(result)
            result["trust_score"] = trust_score

            # Add trust metadata
            result["trust_metadata"] = self._generate_trust_metadata(result)

            # Filter by minimum trust score
            if trust_score < self.min_trust_score:
                self.stats["rejected_count"] += 1
                logger.debug(
                    f"Result rejected (low trust {trust_score:.2f}): "
                    f"{result.get('url', '')}"
                )
                continue

            # Filter by SAP domain if required
            if require_sap_domain:
                if not self._is_sap_domain(result.get("url", "")):
                    self.stats["rejected_count"] += 1
                    continue

            # Categorize by trust level
            if trust_score >= 0.8:
                self.stats["high_trust_count"] += 1
                result["trust_level"] = "high"
            elif trust_score >= 0.6:
                self.stats["medium_trust_count"] += 1
                result["trust_level"] = "medium"
            else:
                self.stats["low_trust_count"] += 1
                result["trust_level"] = "low"

            validated_results.append(result)

        logger.info(
            f"Validated {len(results)} results -> {len(validated_results)} passed "
            f"(high: {self.stats['high_trust_count']}, "
            f"medium: {self.stats['medium_trust_count']}, "
            f"low: {self.stats['low_trust_count']})"
        )

        # Sort by trust score (descending)
        validated_results.sort(
            key=lambda x: x.get("trust_score", 0.0),
            reverse=True
        )

        return validated_results

    def _calculate_trust_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate comprehensive trust score for a result.

        Args:
            result: Search result

        Returns:
            Trust score between 0.0 and 1.0
        """
        score = 0.0
        url = result.get("url", "")

        # 1. Domain authority score (40% weight)
        domain_score = self._calculate_domain_score(url)
        score += domain_score * 0.4

        # 2. Content type score (30% weight)
        content_score = self._calculate_content_type_score(url)
        score += content_score * 0.3

        # 3. Official content verification (20% weight)
        official_score = self._calculate_official_score(url, result)
        score += official_score * 0.2

        # 4. Freshness score (10% weight, configurable)
        freshness_score = self._calculate_freshness_score(result)
        score += freshness_score * self.freshness_weight

        # 5. Security bonus (HTTPS)
        if url.startswith("https://"):
            score += 0.05
        elif self.require_https and url.startswith("http://"):
            score *= 0.5  # Penalty for non-HTTPS

        # Normalize to 0-1 range
        score = min(1.0, max(0.0, score))

        return score

    def _calculate_domain_score(self, url: str) -> float:
        """
        Calculate domain authority score.

        Args:
            url: Result URL

        Returns:
            Domain score (0.0-1.0)
        """
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc

            # Remove www. prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Check exact match
            if domain in self.TRUSTED_DOMAINS:
                return self.TRUSTED_DOMAINS[domain]

            # Check subdomain matches
            for trusted_domain, score in self.TRUSTED_DOMAINS.items():
                if domain.endswith(f".{trusted_domain}") or domain == trusted_domain:
                    return score * 0.9  # Slight penalty for subdomains

            # Unknown domain
            return 0.3  # Base score for unknown domains

        except Exception as e:
            logger.warning(f"Domain parsing error: {e}")
            return 0.0

    def _calculate_content_type_score(self, url: str) -> float:
        """
        Calculate score based on content type.

        Args:
            url: Result URL

        Returns:
            Content type score (0.0-1.0)
        """
        score = 0.5  # Base score
        url_lower = url.lower()

        # Check for content type indicators
        for content_type, config in self.CONTENT_TYPE_INDICATORS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, url_lower):
                    score += config["boost"]
                    logger.debug(f"Content type detected: {content_type}")
                    break

        return min(1.0, score)

    def _calculate_official_score(
        self,
        url: str,
        result: Dict[str, Any]
    ) -> float:
        """
        Calculate official content score.

        Args:
            url: Result URL
            result: Full result dictionary

        Returns:
            Official score (0.0-1.0)
        """
        score = 0.0

        # Check URL patterns for official content
        for pattern in self.OFFICIAL_PATTERNS:
            if re.search(pattern, url):
                score = 1.0
                break

        # Check title for official indicators
        title = result.get("title", "").lower()
        if any(indicator in title for indicator in [
            "official",
            "documentation",
            "api reference",
            "sap help portal"
        ]):
            score = max(score, 0.8)

        # Check for SAP note numbers
        if re.search(r"SAP\s*Note\s*\d+", result.get("snippet", ""), re.IGNORECASE):
            score = max(score, 0.9)

        return score

    def _calculate_freshness_score(self, result: Dict[str, Any]) -> float:
        """
        Calculate freshness score based on result age.

        Args:
            result: Search result

        Returns:
            Freshness score (0.0-1.0)
        """
        # Check if result has timestamp or age metadata
        timestamp = result.get("timestamp")
        age_str = result.get("age", "")

        if timestamp:
            try:
                # Assume timestamp is Unix time
                result_time = datetime.fromtimestamp(timestamp)
                age_days = (datetime.now() - result_time).days

                # Score decreases with age
                if age_days < 30:
                    return 1.0
                elif age_days < 180:
                    return 0.8
                elif age_days < 365:
                    return 0.6
                elif age_days < 730:
                    return 0.4
                else:
                    return 0.2

            except Exception:
                pass

        # Parse age string (e.g., "2 days ago", "1 month ago")
        if age_str:
            if "hour" in age_str or "minute" in age_str:
                return 1.0
            elif "day" in age_str:
                days = self._extract_number(age_str)
                return max(0.0, 1.0 - (days / 30.0))
            elif "week" in age_str:
                weeks = self._extract_number(age_str)
                return max(0.0, 1.0 - (weeks / 12.0))
            elif "month" in age_str:
                months = self._extract_number(age_str)
                return max(0.0, 1.0 - (months / 24.0))
            elif "year" in age_str:
                return 0.2

        # No freshness info - return neutral score
        return 0.5

    def _is_sap_domain(self, url: str) -> bool:
        """
        Check if URL is from SAP domain.

        Args:
            url: Result URL

        Returns:
            True if SAP domain
        """
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc

            if domain.startswith("www."):
                domain = domain[4:]

            return domain in self.TRUSTED_DOMAINS or domain.endswith(".sap.com")

        except Exception:
            return False

    def _generate_trust_metadata(
        self,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate detailed trust metadata for result.

        Args:
            result: Search result

        Returns:
            Trust metadata dictionary
        """
        url = result.get("url", "")

        metadata = {
            "is_sap_domain": self._is_sap_domain(url),
            "is_official": self._is_official_content(url),
            "is_secure": url.startswith("https://"),
            "domain_authority": self._calculate_domain_score(url),
            "content_type": self._detect_content_type(url),
            "trust_factors": []
        }

        # List trust factors
        if metadata["is_official"]:
            metadata["trust_factors"].append("official_documentation")
        if metadata["is_sap_domain"]:
            metadata["trust_factors"].append("sap_domain")
        if metadata["is_secure"]:
            metadata["trust_factors"].append("https_secured")
        if metadata["content_type"] == "api_documentation":
            metadata["trust_factors"].append("api_reference")

        return metadata

    def _is_official_content(self, url: str) -> bool:
        """Check if URL points to official SAP content."""
        for pattern in self.OFFICIAL_PATTERNS:
            if re.search(pattern, url):
                return True
        return False

    def _detect_content_type(self, url: str) -> str:
        """Detect content type from URL."""
        url_lower = url.lower()

        for content_type, config in self.CONTENT_TYPE_INDICATORS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, url_lower):
                    return content_type

        return "general"

    @staticmethod
    def _extract_number(text: str) -> int:
        """Extract first number from text."""
        match = re.search(r'\d+', text)
        return int(match.group()) if match else 1

    def get_trust_summary(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get trust score summary for results.

        Args:
            results: List of validated results

        Returns:
            Summary statistics
        """
        if not results:
            return {
                "count": 0,
                "avg_trust_score": 0.0,
                "high_trust_percentage": 0.0,
                "official_content_count": 0
            }

        trust_scores = [r.get("trust_score", 0.0) for r in results]
        high_trust_count = sum(1 for r in results if r.get("trust_level") == "high")
        official_count = sum(
            1 for r in results
            if r.get("trust_metadata", {}).get("is_official", False)
        )

        return {
            "count": len(results),
            "avg_trust_score": sum(trust_scores) / len(trust_scores),
            "min_trust_score": min(trust_scores),
            "max_trust_score": max(trust_scores),
            "high_trust_count": high_trust_count,
            "high_trust_percentage": (high_trust_count / len(results)) * 100,
            "official_content_count": official_count,
            "sap_domain_count": sum(
                1 for r in results
                if r.get("trust_metadata", {}).get("is_sap_domain", False)
            )
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get validator statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "trusted_domains_count": len(self.TRUSTED_DOMAINS),
            "min_trust_score": self.min_trust_score
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            "total_validated": 0,
            "high_trust_count": 0,
            "medium_trust_count": 0,
            "low_trust_count": 0,
            "rejected_count": 0
        }
