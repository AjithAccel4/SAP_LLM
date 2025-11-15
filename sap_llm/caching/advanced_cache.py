"""
ENHANCEMENT 4: Advanced Caching Layer (Redis Cluster + CDN)

Multi-tier caching architecture:
- L1: In-memory LRU cache (milliseconds)
- L2: Redis Cluster (< 10ms)
- L3: CDN cache for static assets (< 50ms)
- Smart invalidation and warming
- Cache-aside pattern with TTL
"""

import logging
import hashlib
import json
import pickle
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

# Try Redis imports
try:
    from redis.cluster import RedisCluster
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis-py-cluster not available")


@dataclass
class CacheConfig:
    """Cache configuration."""
    l1_enabled: bool = True
    l1_max_size: int = 1000
    l1_ttl_seconds: int = 300  # 5 minutes

    l2_enabled: bool = True
    l2_cluster_nodes: List[Dict[str, Any]] = None
    l2_ttl_seconds: int = 3600  # 1 hour

    l3_enabled: bool = True
    l3_cdn_url: str = ""
    l3_ttl_seconds: int = 86400  # 24 hours

    default_ttl: int = 3600


class LRUCache:
    """
    L1: In-memory LRU cache with TTL.

    Ultra-fast access (< 1ms) for hot data.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []

        logger.info(f"LRU Cache initialized: max_size={max_size}, ttl={default_ttl}s")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Check expiration
        if datetime.now() > entry["expires_at"]:
            self.delete(key)
            return None

        # Update access order (LRU)
        self.access_order.remove(key)
        self.access_order.append(key)

        return entry["value"]

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        ttl = ttl or self.default_ttl

        # Evict if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        # Add to cache
        self.cache[key] = {
            "value": value,
            "expires_at": datetime.now() + timedelta(seconds=ttl),
            "created_at": datetime.now()
        }

        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def delete(self, key: str):
        """Delete key from cache."""
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)

    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size
        }


class RedisClusterCache:
    """
    L2: Redis Cluster cache.

    Distributed cache with < 10ms latency.
    Supports millions of keys across cluster.
    """

    def __init__(self, cluster_nodes: List[Dict[str, Any]], default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.cluster_nodes = cluster_nodes or [
            {"host": "localhost", "port": 6379}
        ]

        if REDIS_AVAILABLE:
            try:
                self.client = RedisCluster(
                    startup_nodes=self.cluster_nodes,
                    decode_responses=False,
                    skip_full_coverage_check=True
                )
                logger.info(f"Redis Cluster connected: {len(self.cluster_nodes)} nodes")
            except Exception as e:
                logger.error(f"Redis Cluster connection failed: {e}")
                self.client = None
        else:
            self.client = None
            logger.warning("Redis Cluster not available - using mock mode")

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        if not self.client:
            return None

        try:
            data = self.client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except RedisError as e:
            logger.error(f"Redis GET error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis."""
        if not self.client:
            return

        ttl = ttl or self.default_ttl

        try:
            data = pickle.dumps(value)
            self.client.setex(key, ttl, data)
        except RedisError as e:
            logger.error(f"Redis SET error: {e}")

    def delete(self, key: str):
        """Delete key from Redis."""
        if not self.client:
            return

        try:
            self.client.delete(key)
        except RedisError as e:
            logger.error(f"Redis DELETE error: {e}")

    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern."""
        if not self.client:
            return

        try:
            # Scan for keys (cluster-safe)
            keys = []
            for key in self.client.scan_iter(match=pattern, count=100):
                keys.append(key)

            if keys:
                self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys matching {pattern}")
        except RedisError as e:
            logger.error(f"Redis CLEAR error: {e}")


class CDNCache:
    """
    L3: CDN cache for static assets.

    Global edge cache with < 50ms latency.
    Ideal for model artifacts, embeddings, static responses.
    """

    def __init__(self, cdn_url: str, default_ttl: int = 86400):
        self.cdn_url = cdn_url
        self.default_ttl = default_ttl

        logger.info(f"CDN Cache initialized: {cdn_url}")

    def get_url(self, key: str) -> str:
        """Get CDN URL for key."""
        # Hash key for CDN path
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return f"{self.cdn_url}/{key_hash[:2]}/{key_hash[2:4]}/{key_hash}"

    def upload(self, key: str, value: Any, ttl: Optional[int] = None):
        """Upload to CDN (would integrate with Azure CDN, CloudFront, etc.)"""
        # Mock implementation - in production would upload to blob storage + CDN
        logger.info(f"CDN upload: {key} -> {self.get_url(key)}")

    def invalidate(self, key: str):
        """Invalidate CDN cache entry."""
        logger.info(f"CDN invalidate: {key}")


class AdvancedCacheLayer:
    """
    Multi-tier caching system.

    Architecture:
    - L1 (In-memory): < 1ms, 1K items, 5min TTL
    - L2 (Redis Cluster): < 10ms, millions of items, 1hr TTL
    - L3 (CDN): < 50ms, unlimited, 24hr TTL

    Cache-aside pattern with automatic population.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()

        # Initialize tiers
        self.l1 = LRUCache(
            max_size=self.config.l1_max_size,
            default_ttl=self.config.l1_ttl_seconds
        ) if self.config.l1_enabled else None

        self.l2 = RedisClusterCache(
            cluster_nodes=self.config.l2_cluster_nodes,
            default_ttl=self.config.l2_ttl_seconds
        ) if self.config.l2_enabled else None

        self.l3 = CDNCache(
            cdn_url=self.config.l3_cdn_url,
            default_ttl=self.config.l3_ttl_seconds
        ) if self.config.l3_enabled else None

        # Statistics
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "misses": 0,
            "sets": 0
        }

        logger.info("AdvancedCacheLayer initialized")

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_parts = [prefix]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))

        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get from cache (try all tiers).

        Returns value from fastest available tier.
        Promotes value to faster tiers on hit.
        """
        # Try L1
        if self.l1:
            value = self.l1.get(key)
            if value is not None:
                self.stats["l1_hits"] += 1
                return value

        # Try L2
        if self.l2:
            value = self.l2.get(key)
            if value is not None:
                self.stats["l2_hits"] += 1
                # Promote to L1
                if self.l1:
                    self.l1.set(key, value)
                return value

        # Try L3 (would fetch from CDN)
        # Skipped in this implementation

        self.stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set in cache (write to all tiers).

        Populates all cache tiers for maximum hit rate.
        """
        ttl = ttl or self.config.default_ttl

        # Set in L1
        if self.l1:
            self.l1.set(key, value, min(ttl, self.config.l1_ttl_seconds))

        # Set in L2
        if self.l2:
            self.l2.set(key, value, min(ttl, self.config.l2_ttl_seconds))

        # Set in L3 for static data
        if self.l3 and ttl >= 3600:  # Only cache long-lived data in CDN
            self.l3.upload(key, value, ttl)

        self.stats["sets"] += 1

    def delete(self, key: str):
        """Delete from all cache tiers."""
        if self.l1:
            self.l1.delete(key)

        if self.l2:
            self.l2.delete(key)

        if self.l3:
            self.l3.invalidate(key)

    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern."""
        # L1 doesn't support pattern matching - clear all
        if self.l1:
            self.l1.clear()

        # L2 supports pattern matching
        if self.l2:
            self.l2.clear_pattern(pattern)

    def warm_cache(self, data_loader: Callable[[], List[tuple]]):
        """
        Warm cache with frequently accessed data.

        Args:
            data_loader: Function returning list of (key, value) tuples
        """
        logger.info("Warming cache...")

        data = data_loader()
        for key, value in data:
            self.set(key, value)

        logger.info(f"Cache warmed with {len(data)} entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = sum([
            self.stats["l1_hits"],
            self.stats["l2_hits"],
            self.stats["l3_hits"],
            self.stats["misses"]
        ])

        hit_rate = (
            (self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]) /
            total_requests * 100
        ) if total_requests > 0 else 0

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "l1_stats": self.l1.get_stats() if self.l1 else None
        }


# Decorator for automatic caching
def cached(
    cache: AdvancedCacheLayer,
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None
):
    """
    Decorator for automatic function result caching.

    Usage:
        @cached(cache_instance, ttl=3600, key_prefix="classify")
        def classify_document(doc_id):
            ...
    """
    def decorator(func):
        prefix = key_prefix or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache._generate_key(prefix, *args, **kwargs)

            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result

            # Call function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result, ttl)

            return result

        return wrapper
    return decorator


# Singleton instance
_cache: Optional[AdvancedCacheLayer] = None


def get_cache() -> AdvancedCacheLayer:
    """Get singleton cache instance."""
    global _cache

    if _cache is None:
        _cache = AdvancedCacheLayer()

    return _cache


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize cache
    cache = get_cache()

    # Simulate requests
    for i in range(100):
        key = f"doc_{i % 10}"  # 10 unique docs, lots of repeats

        # Try get
        value = cache.get(key)

        if value is None:
            # Cache miss - compute and store
            value = {"doc_type": "invoice", "amount": i * 100}
            cache.set(key, value)

    # Print statistics
    stats = cache.get_stats()
    print("\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
