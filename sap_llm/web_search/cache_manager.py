"""
Cache Manager for Web Search Results.

Provides 3-tier caching (memory, Redis, disk) with TTL, compression, and statistics.
"""

import gzip
import json
import os
import time
from pathlib import Path
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
    3-tier cache manager for search results: L1 (memory), L2 (Redis), L3 (disk).

    Features:
    - L1: In-memory cache (fastest, volatile)
    - L2: Redis cache (fast, shared, persistent)
    - L3: Disk cache (slowest, most persistent)
    - Automatic TTL (time-to-live) expiration
    - Optional compression for large results
    - LRU eviction when memory limit reached
    - Statistics tracking (hit rate, memory usage)
    - Cache promotion (L3 -> L2 -> L1 when found)

    Example:
        >>> cache = SearchCacheManager(redis_config={"host": "localhost"}, disk_cache_dir="/tmp/cache")
        >>> cache.set("query_hash", results, ttl=3600)
        >>> cached = cache.get("query_hash")
    """

    def __init__(
        self,
        redis_config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        compress_threshold: int = 1024,
        default_ttl: int = 3600,
        disk_cache_dir: Optional[str] = None,
        max_disk_cache_size_mb: int = 1000
    ):
        """
        Initialize 3-tier cache manager.

        Args:
            redis_config: Redis connection configuration
            enabled: Whether caching is enabled
            compress_threshold: Compress results larger than this (bytes)
            default_ttl: Default cache TTL in seconds
            disk_cache_dir: Directory for disk cache (None = disabled)
            max_disk_cache_size_mb: Maximum disk cache size in MB
        """
        self.enabled = enabled
        self.compress_threshold = compress_threshold
        self.default_ttl = default_ttl
        self.redis_client: Optional[Any] = None
        self.in_memory_cache: Dict[str, tuple] = {}  # L1: Memory cache

        # L3: Disk cache configuration
        self.disk_cache_enabled = disk_cache_dir is not None
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else None
        self.max_disk_cache_size_mb = max_disk_cache_size_mb

        if self.disk_cache_enabled:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Disk cache enabled at: {self.disk_cache_dir}")

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "errors": 0,
            "compression_ratio": 0.0,
            "l1_hits": 0,  # Memory cache hits
            "l2_hits": 0,  # Redis cache hits
            "l3_hits": 0,  # Disk cache hits
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
        Get cached search results from 3-tier cache (L1 -> L2 -> L3).

        Args:
            key: Cache key (usually query hash)

        Returns:
            Cached results or None if not found
        """
        if not self.enabled:
            return None

        try:
            # L1: Try in-memory cache first (fastest)
            if key in self.in_memory_cache:
                results, expiry = self.in_memory_cache[key]
                if time.time() < expiry:
                    self.stats["hits"] += 1
                    self.stats["l1_hits"] += 1
                    logger.debug(f"L1 (memory) cache hit for key: {key[:16]}...")
                    return results
                else:
                    # Expired
                    del self.in_memory_cache[key]

            # L2: Try Redis cache (fast, shared)
            if self.redis_client:
                data = self.redis_client.get(f"search:{key}")
                if data:
                    results = self._deserialize(data)
                    self.stats["hits"] += 1
                    self.stats["l2_hits"] += 1
                    logger.debug(f"L2 (Redis) cache hit for key: {key[:16]}...")

                    # Promote to L1
                    expiry = time.time() + self.default_ttl
                    self.in_memory_cache[key] = (results, expiry)
                    return results

            # L3: Try disk cache (slowest, most persistent)
            if self.disk_cache_enabled:
                results = self._get_from_disk(key)
                if results is not None:
                    self.stats["hits"] += 1
                    self.stats["l3_hits"] += 1
                    logger.debug(f"L3 (disk) cache hit for key: {key[:16]}...")

                    # Promote to L2 and L1
                    expiry = time.time() + self.default_ttl
                    self.in_memory_cache[key] = (results, expiry)

                    if self.redis_client:
                        data = self._serialize(results)
                        self.redis_client.setex(f"search:{key}", self.default_ttl, data)

                    return results

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
        Cache search results across all tiers (L1, L2, L3).

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
            # L1: Store in-memory cache
            expiry = time.time() + ttl
            self.in_memory_cache[key] = (value, expiry)
            self._cleanup_memory_cache()

            # L2: Store in Redis
            if self.redis_client:
                data = self._serialize(value)
                self.redis_client.setex(f"search:{key}", ttl, data)

            # L3: Store on disk
            if self.disk_cache_enabled:
                self._set_to_disk(key, value, ttl)
                self._cleanup_disk_cache()

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

    def _get_from_disk(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached results from disk."""
        if not self.disk_cache_enabled:
            return None

        try:
            cache_file = self.disk_cache_dir / f"{key}.cache.gz"
            meta_file = self.disk_cache_dir / f"{key}.meta.json"

            if not cache_file.exists() or not meta_file.exists():
                return None

            # Check TTL from metadata
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            expiry = meta.get("expiry", 0)
            if time.time() >= expiry:
                # Expired - delete files
                cache_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)
                return None

            # Read and decompress data
            with gzip.open(cache_file, 'rb') as f:
                data = f.read()

            return json.loads(data.decode('utf-8'))

        except Exception as e:
            logger.debug(f"Disk cache read error for key {key[:16]}: {e}")
            return None

    def _set_to_disk(self, key: str, value: List[Dict[str, Any]], ttl: int) -> bool:
        """Store results to disk cache."""
        if not self.disk_cache_enabled:
            return False

        try:
            cache_file = self.disk_cache_dir / f"{key}.cache.gz"
            meta_file = self.disk_cache_dir / f"{key}.meta.json"

            # Write compressed data
            data = json.dumps(value).encode('utf-8')
            with gzip.open(cache_file, 'wb') as f:
                f.write(data)

            # Write metadata
            meta = {
                "key": key,
                "expiry": time.time() + ttl,
                "created_at": time.time(),
                "size_bytes": len(data)
            }
            with open(meta_file, 'w') as f:
                json.dump(meta, f)

            return True

        except Exception as e:
            logger.error(f"Disk cache write error: {e}")
            return False

    def _cleanup_disk_cache(self) -> None:
        """Clean up expired and oversized disk cache."""
        if not self.disk_cache_enabled:
            return

        try:
            current_time = time.time()
            total_size = 0
            files_info = []

            # Collect all cache files with metadata
            for meta_file in self.disk_cache_dir.glob("*.meta.json"):
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)

                    key = meta.get("key")
                    cache_file = self.disk_cache_dir / f"{key}.cache.gz"

                    # Check if expired
                    if current_time >= meta.get("expiry", 0):
                        # Delete expired files
                        cache_file.unlink(missing_ok=True)
                        meta_file.unlink(missing_ok=True)
                        continue

                    # Track size
                    if cache_file.exists():
                        size = cache_file.stat().st_size
                        total_size += size
                        files_info.append({
                            "meta_file": meta_file,
                            "cache_file": cache_file,
                            "size": size,
                            "created_at": meta.get("created_at", 0)
                        })
                except Exception:
                    continue

            # Check if total size exceeds limit
            max_size_bytes = self.max_disk_cache_size_mb * 1024 * 1024
            if total_size > max_size_bytes:
                # Sort by age (oldest first) and delete until under limit
                files_info.sort(key=lambda x: x["created_at"])

                for file_info in files_info:
                    if total_size <= max_size_bytes:
                        break

                    file_info["cache_file"].unlink(missing_ok=True)
                    file_info["meta_file"].unlink(missing_ok=True)
                    total_size -= file_info["size"]

                logger.info(f"Disk cache cleanup complete. Size: {total_size / 1024 / 1024:.2f} MB")

        except Exception as e:
            logger.error(f"Disk cache cleanup error: {e}")

    def delete(self, key: str) -> bool:
        """
        Delete cached entry from all tiers.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted
        """
        try:
            # L1: Delete from memory
            self.in_memory_cache.pop(key, None)

            # L2: Delete from Redis
            if self.redis_client:
                self.redis_client.delete(f"search:{key}")

            # L3: Delete from disk
            if self.disk_cache_enabled:
                cache_file = self.disk_cache_dir / f"{key}.cache.gz"
                meta_file = self.disk_cache_dir / f"{key}.meta.json"
                cache_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)

            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all cached search results from all tiers.

        Returns:
            True if successful
        """
        try:
            # L1: Clear memory cache
            mem_count = len(self.in_memory_cache)
            self.in_memory_cache.clear()
            logger.info(f"Cleared {mem_count} entries from memory cache")

            # L2: Clear Redis cache
            if self.redis_client:
                keys = self.redis_client.keys("search:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} entries from Redis cache")

            # L3: Clear disk cache
            if self.disk_cache_enabled:
                disk_count = 0
                for cache_file in self.disk_cache_dir.glob("*.cache.gz"):
                    cache_file.unlink(missing_ok=True)
                    disk_count += 1
                for meta_file in self.disk_cache_dir.glob("*.meta.json"):
                    meta_file.unlink(missing_ok=True)
                logger.info(f"Cleared {disk_count} entries from disk cache")

            return True

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for all tiers.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (
            self.stats["hits"] / total_requests
            if total_requests > 0 else 0.0
        )

        # Calculate tier-specific hit rates
        l1_hit_rate = self.stats["l1_hits"] / total_requests if total_requests > 0 else 0.0
        l2_hit_rate = self.stats["l2_hits"] / total_requests if total_requests > 0 else 0.0
        l3_hit_rate = self.stats["l3_hits"] / total_requests if total_requests > 0 else 0.0

        stats = {
            **self.stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "l1_hit_rate": l1_hit_rate,
            "l2_hit_rate": l2_hit_rate,
            "l3_hit_rate": l3_hit_rate,
            "backends": []
        }

        # L1: Memory cache stats
        stats["l1_entries"] = len(self.in_memory_cache)
        stats["backends"].append("memory")

        # L2: Redis stats
        if self.redis_client:
            try:
                info = self.redis_client.info("memory")
                stats["l2_memory_used_mb"] = info.get("used_memory", 0) / 1024 / 1024
                stats["l2_memory_peak_mb"] = info.get("used_memory_peak", 0) / 1024 / 1024
                stats["backends"].append("redis")
            except Exception:
                pass

        # L3: Disk cache stats
        if self.disk_cache_enabled:
            try:
                disk_count = len(list(self.disk_cache_dir.glob("*.cache.gz")))
                disk_size = sum(
                    f.stat().st_size
                    for f in self.disk_cache_dir.glob("*.cache.gz")
                )
                stats["l3_entries"] = disk_count
                stats["l3_size_mb"] = disk_size / 1024 / 1024
                stats["l3_max_size_mb"] = self.max_disk_cache_size_mb
                stats["backends"].append("disk")
            except Exception:
                pass

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
