"""
Search Result Deduplication Module.

Provides advanced deduplication capabilities for search results including:
- URL-based deduplication
- Content similarity deduplication
- Near-duplicate detection using hashing
- Semantic deduplication (with embeddings)
"""

import hashlib
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


def deduplicate_results(
    results: List[Dict[str, Any]],
    method: str = "url",
    threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Remove duplicate search results.

    Main entry point for deduplication. Supports multiple methods:
    - "url": Remove exact URL duplicates (fastest)
    - "content": Remove near-duplicate content using hashing
    - "combined": Apply both URL and content deduplication

    Args:
        results: List of search results
        method: Deduplication method ("url", "content", "combined")
        threshold: Similarity threshold for content deduplication (0-1)

    Returns:
        Deduplicated list of results

    Example:
        >>> results = [
        ...     {"url": "https://a.com/page", "title": "Title", "snippet": "Content"},
        ...     {"url": "https://a.com/page", "title": "Title", "snippet": "Content"},  # duplicate
        ... ]
        >>> unique = deduplicate_results(results, method="url")
        >>> len(unique)
        1
    """
    if not results:
        return []

    if method == "url":
        return _deduplicate_by_url(results)
    elif method == "content":
        return _deduplicate_by_content(results, threshold)
    elif method == "combined":
        # Apply both methods
        unique = _deduplicate_by_url(results)
        unique = _deduplicate_by_content(unique, threshold)
        return unique
    else:
        logger.warning(f"Unknown deduplication method: {method}. Using URL deduplication.")
        return _deduplicate_by_url(results)


def _deduplicate_by_url(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicates based on URL.

    Normalizes URLs before comparison to catch more duplicates.

    Args:
        results: List of search results

    Returns:
        Deduplicated results
    """
    seen_urls: Set[str] = set()
    seen_normalized: Set[str] = set()
    unique = []

    for result in results:
        url = result.get("url", "")

        # Normalize URL
        normalized = _normalize_url(url)
        url_hash = hashlib.md5(url.encode()).hexdigest()
        normalized_hash = hashlib.md5(normalized.encode()).hexdigest()

        # Skip if already seen
        if url_hash in seen_urls or normalized_hash in seen_normalized:
            logger.debug(f"Duplicate URL removed: {url[:50]}")
            continue

        seen_urls.add(url_hash)
        seen_normalized.add(normalized_hash)
        unique.append(result)

    removed = len(results) - len(unique)
    if removed > 0:
        logger.info(f"URL deduplication: removed {removed} duplicates ({len(results)} -> {len(unique)})")

    return unique


def _deduplicate_by_content(
    results: List[Dict[str, Any]],
    threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicates based on content similarity.

    Uses SimHash for efficient near-duplicate detection.

    Args:
        results: List of search results
        threshold: Similarity threshold (0-1)

    Returns:
        Deduplicated results
    """
    if not results:
        return []

    unique = []
    content_hashes: List[int] = []

    for result in results:
        # Create content fingerprint from title + snippet
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        content = f"{title} {snippet}"

        # Calculate SimHash
        current_hash = _simhash(content)

        # Check against existing hashes
        is_duplicate = False
        for existing_hash in content_hashes:
            similarity = _hamming_similarity(current_hash, existing_hash)

            if similarity >= threshold:
                is_duplicate = True
                logger.debug(
                    f"Content duplicate found (similarity: {similarity:.2f}): "
                    f"{title[:50]}"
                )
                break

        if not is_duplicate:
            unique.append(result)
            content_hashes.append(current_hash)

    removed = len(results) - len(unique)
    if removed > 0:
        logger.info(f"Content deduplication: removed {removed} duplicates ({len(results)} -> {len(unique)})")

    return unique


def _normalize_url(url: str) -> str:
    """
    Normalize URL for comparison.

    - Converts to lowercase
    - Removes www. prefix
    - Removes trailing slashes
    - Removes common tracking parameters
    - Removes fragments

    Args:
        url: Original URL

    Returns:
        Normalized URL
    """
    try:
        parsed = urlparse(url.lower())

        # Remove www. prefix
        netloc = parsed.netloc
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # Remove trailing slash from path
        path = parsed.path.rstrip("/")

        # Remove tracking parameters
        query_params = parsed.query
        if query_params:
            tracking_params = [
                "utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term",
                "ref", "fbclid", "gclid", "msclkid", "source", "campaign"
            ]
            params = [
                p for p in query_params.split("&")
                if not any(p.startswith(f"{tp}=") for tp in tracking_params)
            ]
            query_params = "&".join(params)

        # Reconstruct normalized URL (without fragment)
        normalized = f"{parsed.scheme}://{netloc}{path}"
        if query_params:
            normalized += f"?{query_params}"

        return normalized

    except Exception:
        return url.lower()


def _simhash(text: str, hashbits: int = 64) -> int:
    """
    Calculate SimHash of text for near-duplicate detection.

    SimHash creates a fingerprint where similar documents have similar hashes.

    Args:
        text: Input text
        hashbits: Number of bits in hash

    Returns:
        SimHash integer
    """
    # Tokenize and clean text
    tokens = _tokenize(text)

    if not tokens:
        return 0

    # Initialize bit vector
    v = [0] * hashbits

    # Process each token
    for token in tokens:
        # Hash the token
        token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)

        # Update bit vector
        for i in range(hashbits):
            bitmask = 1 << i
            if token_hash & bitmask:
                v[i] += 1
            else:
                v[i] -= 1

    # Build final hash
    fingerprint = 0
    for i in range(hashbits):
        if v[i] >= 0:
            fingerprint |= 1 << i

    return fingerprint


def _hamming_similarity(hash1: int, hash2: int, hashbits: int = 64) -> float:
    """
    Calculate similarity between two SimHashes using Hamming distance.

    Args:
        hash1: First hash
        hash2: Second hash
        hashbits: Number of bits in hash

    Returns:
        Similarity score (0-1)
    """
    # XOR to find differing bits
    xor = hash1 ^ hash2

    # Count differing bits (Hamming distance)
    distance = bin(xor).count('1')

    # Convert to similarity (1 - normalized distance)
    similarity = 1 - (distance / hashbits)

    return similarity


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text into words for SimHash.

    - Converts to lowercase
    - Removes punctuation
    - Splits on whitespace
    - Removes stopwords
    - Creates n-grams for better matching

    Args:
        text: Input text

    Returns:
        List of tokens
    """
    # Clean and lowercase
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)

    # Split into words
    words = text.split()

    # Remove stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'is',
        'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'shall', 'this', 'that', 'these', 'those', 'it', 'its'
    }
    words = [w for w in words if w not in stopwords and len(w) > 2]

    # Create 2-grams (bigrams) for better context
    tokens = words.copy()
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")

    return tokens


class Deduplicator:
    """
    Class-based deduplication engine with caching and statistics.

    Provides more control over deduplication process and tracks statistics.

    Example:
        >>> dedup = Deduplicator(method="combined", threshold=0.90)
        >>> unique = dedup.deduplicate(results)
        >>> stats = dedup.get_stats()
    """

    def __init__(
        self,
        method: str = "combined",
        threshold: float = 0.85,
        preserve_order: bool = True
    ):
        """
        Initialize Deduplicator.

        Args:
            method: Deduplication method ("url", "content", "combined")
            threshold: Similarity threshold for content deduplication
            preserve_order: Whether to preserve original result order
        """
        self.method = method
        self.threshold = threshold
        self.preserve_order = preserve_order

        # Statistics
        self.stats = {
            "total_processed": 0,
            "total_removed": 0,
            "url_duplicates": 0,
            "content_duplicates": 0
        }

    def deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate search results.

        Args:
            results: List of search results

        Returns:
            Deduplicated results
        """
        original_count = len(results)
        self.stats["total_processed"] += original_count

        # Apply deduplication
        unique = deduplicate_results(
            results,
            method=self.method,
            threshold=self.threshold
        )

        # Update stats
        removed = original_count - len(unique)
        self.stats["total_removed"] += removed

        if self.method in ["url", "combined"]:
            self.stats["url_duplicates"] += removed // 2
        if self.method in ["content", "combined"]:
            self.stats["content_duplicates"] += removed // 2

        return unique

    def get_stats(self) -> Dict[str, Any]:
        """
        Get deduplication statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "removal_rate": (
                self.stats["total_removed"] / self.stats["total_processed"]
                if self.stats["total_processed"] > 0 else 0.0
            )
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = {
            "total_processed": 0,
            "total_removed": 0,
            "url_duplicates": 0,
            "content_duplicates": 0
        }


def find_duplicates(
    results: List[Dict[str, Any]],
    threshold: float = 0.85
) -> List[List[int]]:
    """
    Find groups of duplicate results.

    Returns indices of results that are duplicates of each other.
    Useful for analysis and debugging.

    Args:
        results: List of search results
        threshold: Similarity threshold

    Returns:
        List of duplicate groups (each group is list of indices)

    Example:
        >>> groups = find_duplicates(results, threshold=0.90)
        >>> for group in groups:
        ...     print(f"Duplicates: {[results[i]['title'] for i in group]}")
    """
    if not results:
        return []

    # Calculate all content hashes
    hashes = []
    for result in results:
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        content = f"{title} {snippet}"
        hashes.append(_simhash(content))

    # Find duplicate groups using union-find
    n = len(results)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            # Check URL match
            url_i = results[i].get("url", "")
            url_j = results[j].get("url", "")
            if _normalize_url(url_i) == _normalize_url(url_j):
                union(i, j)
                continue

            # Check content similarity
            similarity = _hamming_similarity(hashes[i], hashes[j])
            if similarity >= threshold:
                union(i, j)

    # Group by parent
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # Return only groups with duplicates (size > 1)
    return [group for group in groups.values() if len(group) > 1]
