"""
Cache Manager for Web Search Results.

Provides Redis-based caching with TTL, compression, and statistics.
"""

import gzip
import json
import time
from typing import Any, Dict, List, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SearchCacheManager:
    """
    Redis-based cache manager for search results.

    Features:
    - Automatic TTL (time-to-live) expiration
    - Optional compression for large results
    - LRU eviction when memory limit reached
    - Statistics tracking (hit rate, memory usage)
    - Graceful fallback if Redis unavailable

    Example:
        >>> cache = SearchCacheManager(redis_config={"host": "localhost"})
        >>> cache.set("query_hash", results, ttl=3600)
        >>> cached = cache.get("query_hash")
    """

    def __init__(
        self,
        redis_config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        compress_threshold: int = 1024,
        default_ttl: int = 3600
    ):
        """
        Initialize cache manager.

        Args:
            redis_config: Redis connection configuration
            enabled: Whether caching is enabled
            compress_threshold: Compress results larger than this (bytes)
            default_ttl: Default cache TTL in seconds
        """
        self.enabled = enabled
        self.compress_threshold = compress_threshold
        self.default_ttl = default_ttl
        self.redis_client: Optional[Any] = None
        self.in_memory_cache: Dict[str, tuple] = {}  # Fallback cache

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
            "compression_ratio": 0.0
        }

        if not self.enabled:
            logger.info("Cache manager disabled")
            return

        # Initialize Redis connection
        if REDIS_AVAILABLE and redis_config:
            self._connect_redis(redis_config)
        else:
            if not REDIS_AVAILABLE:
                logger.warning("Redis not available, using in-memory cache fallback")
            self.redis_client = None

    def _connect_redis(self, config: Dict[str, Any]) -> None:
        """Connect to Redis server."""
        try:
            self.redis_client = redis.Redis(
                host=config.get("host", "localhost"),
                port=config.get("port", 6379),
                db=config.get("db", 0),
                password=config.get("password"),
                decode_responses=False,  # We handle encoding/decoding
                socket_connect_timeout=config.get("timeout", 5),
                socket_timeout=config.get("timeout", 5),
                max_connections=config.get("max_connections", 50)
            )

            # Test connection
            self.redis_client.ping()
            logger.info(
                f"Connected to Redis at {config.get('host')}:{config.get('port')}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}. Using in-memory cache.")
            self.redis_client = None

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached search results.

        Args:
            key: Cache key (usually query hash)

        Returns:
            Cached results or None if not found
        """
        if not self.enabled:
            return None

        try:
            if self.redis_client:
                # Try Redis first
                data = self.redis_client.get(f"search:{key}")
                if data:
                    results = self._deserialize(data)
                    self.stats["hits"] += 1
                    logger.debug(f"Cache hit for key: {key[:16]}...")
                    return results
            else:
                # Fallback to in-memory cache
                if key in self.in_memory_cache:
                    results, expiry = self.in_memory_cache[key]
                    if time.time() < expiry:
                        self.stats["hits"] += 1
                        return results
                    else:
                        # Expired
                        del self.in_memory_cache[key]

            self.stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats["errors"] += 1
            return None

    def set(
        self,
        key: str,
        value: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache search results.

        Args:
            key: Cache key
            value: Search results to cache
            ttl: Time-to-live in seconds (None = use default)

        Returns:
            True if successful
        """
        if not self.enabled:
            return False

        ttl = ttl or self.default_ttl

        try:
            data = self._serialize(value)

            if self.redis_client:
                # Store in Redis
                self.redis_client.setex(
                    f"search:{key}",
                    ttl,
                    data
                )
            else:
                # Store in-memory
                expiry = time.time() + ttl
                self.in_memory_cache[key] = (value, expiry)

                # Evict expired entries (simple cleanup)
                self._cleanup_memory_cache()

            self.stats["sets"] += 1
            logger.debug(f"Cached results for key: {key[:16]}... (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.stats["errors"] += 1
            return False

    def _serialize(self, data: List[Dict[str, Any]]) -> bytes:
        """Serialize and optionally compress data."""
        json_data = json.dumps(data).encode('utf-8')

        # Compress if large enough
        if len(json_data) > self.compress_threshold:
            compressed = gzip.compress(json_data)

            # Track compression ratio
            ratio = len(compressed) / len(json_data)
            stats_ratio = self.stats["compression_ratio"]
            count = self.stats["sets"]
            self.stats["compression_ratio"] = (stats_ratio * count + ratio) / (count + 1)

            # Prepend marker for compressed data
            return b"GZIP:" + compressed
        else:
            return json_data

    def _deserialize(self, data: bytes) -> List[Dict[str, Any]]:
        """Deserialize and decompress data."""
        # Check for compression marker
        if data.startswith(b"GZIP:"):
            data = gzip.decompress(data[5:])

        return json.loads(data.decode('utf-8'))

    def _cleanup_memory_cache(self) -> None:
        """Remove expired entries from in-memory cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry) in self.in_memory_cache.items()
            if current_time >= expiry
        ]

        for key in expired_keys:
            del self.in_memory_cache[key]

        # Also enforce max size (keep 1000 most recent)
        if len(self.in_memory_cache) > 1000:
            # Sort by expiry time, keep newest
            sorted_items = sorted(
                self.in_memory_cache.items(),
                key=lambda x: x[1][1],
                reverse=True
            )
            self.in_memory_cache = dict(sorted_items[:1000])

    def delete(self, key: str) -> bool:
        """
        Delete cached entry.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted
        """
        try:
            if self.redis_client:
                self.redis_client.delete(f"search:{key}")
            else:
                self.in_memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all cached search results.

        Returns:
            True if successful
        """
        try:
            if self.redis_client:
                # Delete all keys matching pattern
                keys = self.redis_client.keys("search:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries from Redis")
            else:
                count = len(self.in_memory_cache)
                self.in_memory_cache.clear()
                logger.info(f"Cleared {count} cache entries from memory")

            return True

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0 else 0.0
        )

        stats = {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "backend": "redis" if self.redis_client else "memory"
        }

        # Add Redis-specific stats if available
        if self.redis_client:
            try:
                info = self.redis_client.info("memory")
                stats["memory_used_mb"] = info.get("used_memory", 0) / 1024 / 1024
                stats["memory_peak_mb"] = info.get("used_memory_peak", 0) / 1024 / 1024
            except Exception:
                pass
        else:
            # In-memory cache size
            stats["memory_entries"] = len(self.in_memory_cache)

        return stats

    def health_check(self) -> Dict[str, Any]:
        """
        Check cache health.

        Returns:
            Health check results
        """
        health = {
            "healthy": False,
            "backend": "none",
            "latency_ms": None,
            "error": None
        }

        if not self.enabled:
            health["error"] = "Cache disabled"
            return health

        try:
            start = time.time()

            if self.redis_client:
                # Test Redis connection
                self.redis_client.ping()
                health["healthy"] = True
                health["backend"] = "redis"
            else:
                # In-memory cache is always available
                health["healthy"] = True
                health["backend"] = "memory"

            health["latency_ms"] = (time.time() - start) * 1000

        except Exception as e:
            health["error"] = str(e)

        return health

    def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
