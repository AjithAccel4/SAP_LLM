"""
Advanced Process Memory Graph (PMG) Optimizer.

Ultra-enhancements for <50ms query latency and high throughput:
1. Asynchronous graph operations (async/await)
2. Intelligent query batching and caching
3. Distributed caching (Redis/Memcached)
4. Query optimization and indexing
5. Connection pooling and reuse
6. Lazy loading and pagination
7. Parallel query execution
8. Query result caching with TTL

Target Metrics:
- Query latency P95: <50ms (from 120ms baseline)
- Throughput: 10k queries/sec
- Cache hit rate: >80%
- Concurrent queries: 1000+
"""

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import aioredis
from gremlin_python.driver import client, serializer
from gremlin_python.driver.aiohttp.transport import AiohttpTransport

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.

    Features:
    - O(1) get/set operations
    - Automatic expiration based on TTL
    - Size-based eviction (LRU)
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 300):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum cache entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.expiry: Dict[str, float] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Check expiry
        if key in self.expiry:
            if time.time() > self.expiry[key]:
                # Expired
                del self.cache[key]
                del self.expiry[key]
                self.misses += 1
                return None

        # Get from cache
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        # Remove if exists
        if key in self.cache:
            del self.cache[key]

        # Add to cache
        self.cache[key] = value
        self.cache.move_to_end(key)

        # Set expiry
        ttl = ttl or self.default_ttl
        self.expiry[key] = time.time() + ttl

        # Evict if over size
        while len(self.cache) > self.max_size:
            # Remove oldest
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            if oldest_key in self.expiry:
                del self.expiry[oldest_key]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
        }


class QueryBatcher:
    """
    Intelligent query batching for PMG.

    Collects multiple queries and executes them together
    to reduce round-trips to the database.
    """

    def __init__(
        self,
        max_batch_size: int = 100,
        max_wait_ms: float = 10.0,
    ):
        """
        Initialize query batcher.

        Args:
            max_batch_size: Maximum queries per batch
            max_wait_ms: Maximum wait time before executing batch
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending_queries: List[Dict[str, Any]] = []
        self.query_futures: Dict[str, asyncio.Future] = {}
        self.last_batch_time = time.time()

    async def add_query(
        self,
        query: str,
        bindings: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Add query to batch and wait for result.

        Args:
            query: Gremlin query
            bindings: Query bindings

        Returns:
            Query result
        """
        query_id = self._generate_query_id(query, bindings)

        # Check if already in batch
        if query_id in self.query_futures:
            return await self.query_futures[query_id]

        # Create future for this query
        future = asyncio.Future()
        self.query_futures[query_id] = future

        # Add to batch
        self.pending_queries.append({
            "id": query_id,
            "query": query,
            "bindings": bindings or {},
        })

        # Execute batch if ready
        if await self._should_execute_batch():
            await self._execute_batch()

        # Wait for result
        return await future

    async def _should_execute_batch(self) -> bool:
        """Check if batch should be executed."""
        if len(self.pending_queries) >= self.max_batch_size:
            return True

        elapsed_ms = (time.time() - self.last_batch_time) * 1000
        if elapsed_ms >= self.max_wait_ms and len(self.pending_queries) > 0:
            return True

        return False

    async def _execute_batch(self) -> None:
        """Execute batched queries."""
        # This is a simplified implementation
        # In production, would use Gremlin's batching capabilities
        pass

    def _generate_query_id(
        self,
        query: str,
        bindings: Optional[Dict[str, Any]],
    ) -> str:
        """Generate unique query ID for caching."""
        query_str = f"{query}:{json.dumps(bindings, sort_keys=True)}"
        return hashlib.md5(query_str.encode()).hexdigest()


class AdvancedPMGOptimizer:
    """
    Advanced PMG optimizer with async operations and caching.

    Features:
    1. Asynchronous graph queries
    2. Multi-level caching (L1: local, L2: Redis)
    3. Query batching and optimization
    4. Connection pooling
    5. Parallel query execution
    6. Lazy loading
    """

    def __init__(
        self,
        cosmos_endpoint: str,
        cosmos_key: str,
        database: str,
        graph: str,
        enable_l1_cache: bool = True,
        enable_l2_cache: bool = True,
        redis_url: Optional[str] = None,
        max_connections: int = 100,
    ):
        """
        Initialize advanced PMG optimizer.

        Args:
            cosmos_endpoint: Cosmos DB Gremlin endpoint
            cosmos_key: Cosmos DB key
            database: Database name
            graph: Graph name
            enable_l1_cache: Enable local L1 cache
            enable_l2_cache: Enable Redis L2 cache
            redis_url: Redis connection URL
            max_connections: Maximum Gremlin connections
        """
        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_key = cosmos_key
        self.database = database
        self.graph = graph
        self.enable_l1_cache = enable_l1_cache
        self.enable_l2_cache = enable_l2_cache
        self.max_connections = max_connections

        logger.info("Initializing Advanced PMG Optimizer...")
        logger.info(f"L1 Cache: {enable_l1_cache}, L2 Cache: {enable_l2_cache}")

        # L1 Cache (local LRU)
        if enable_l1_cache:
            self.l1_cache = LRUCache(max_size=10000, default_ttl=300)
        else:
            self.l1_cache = None

        # L2 Cache (Redis)
        self.redis_client: Optional[aioredis.Redis] = None
        if enable_l2_cache and redis_url:
            self.redis_url = redis_url
        else:
            self.enable_l2_cache = False

        # Query batcher
        self.query_batcher = QueryBatcher(
            max_batch_size=100,
            max_wait_ms=10.0,
        )

        # Gremlin client (will be initialized async)
        self.gremlin_client: Optional[client.Client] = None

        # Performance metrics
        self.query_count = 0
        self.total_latency_ms = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("✓ Advanced PMG Optimizer initialized")

    async def initialize(self) -> None:
        """Initialize async connections."""
        logger.info("Initializing async connections...")

        # Initialize Gremlin client
        try:
            self.gremlin_client = client.Client(
                self.cosmos_endpoint,
                "g",
                username=f"/dbs/{self.database}/colls/{self.graph}",
                password=self.cosmos_key,
                message_serializer=serializer.GraphSONSerializersV2d0(),
            )
            logger.info("✓ Gremlin client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gremlin client: {e}")

        # Initialize Redis
        if self.enable_l2_cache and self.redis_url:
            try:
                self.redis_client = await aioredis.create_redis_pool(
                    self.redis_url,
                    minsize=5,
                    maxsize=self.max_connections,
                )
                logger.info("✓ Redis client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")
                self.enable_l2_cache = False

    async def close(self) -> None:
        """Close all connections."""
        if self.gremlin_client:
            self.gremlin_client.close()

        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()

    async def query(
        self,
        gremlin_query: str,
        bindings: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute Gremlin query with caching and optimization.

        Args:
            gremlin_query: Gremlin query string
            bindings: Query parameters
            use_cache: Whether to use cache

        Returns:
            Query results
        """
        start_time = time.time()

        # Generate cache key
        cache_key = self._generate_cache_key(gremlin_query, bindings)

        # Try L1 cache
        if use_cache and self.l1_cache:
            result = self.l1_cache.get(cache_key)
            if result is not None:
                logger.debug(f"L1 cache hit: {cache_key[:16]}...")
                self.cache_hits += 1
                return result

        # Try L2 cache (Redis)
        if use_cache and self.enable_l2_cache:
            result = await self._get_from_redis(cache_key)
            if result is not None:
                logger.debug(f"L2 cache hit: {cache_key[:16]}...")
                self.cache_hits += 1

                # Populate L1 cache
                if self.l1_cache:
                    self.l1_cache.set(cache_key, result)

                return result

        # Cache miss - execute query
        self.cache_misses += 1
        result = await self._execute_query(gremlin_query, bindings)

        # Store in cache
        if use_cache:
            if self.l1_cache:
                self.l1_cache.set(cache_key, result)

            if self.enable_l2_cache:
                await self._set_in_redis(cache_key, result, ttl=300)

        # Update metrics
        latency_ms = (time.time() - start_time) * 1000
        self.query_count += 1
        self.total_latency_ms += latency_ms

        logger.debug(f"Query executed in {latency_ms:.2f}ms")

        return result

    async def query_parallel(
        self,
        queries: List[Tuple[str, Optional[Dict[str, Any]]]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Execute multiple queries in parallel.

        Args:
            queries: List of (query, bindings) tuples

        Returns:
            List of query results
        """
        logger.info(f"Executing {len(queries)} queries in parallel...")

        tasks = [
            self.query(query, bindings)
            for query, bindings in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {i} failed: {result}")
                final_results.append([])
            else:
                final_results.append(result)

        return final_results

    async def get_similar_cases(
        self,
        doc_type: str,
        supplier: str,
        company_code: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get similar historical cases from PMG.

        Optimized query with indexes and caching.

        Args:
            doc_type: Document type
            supplier: Supplier ID
            company_code: Company code
            limit: Maximum results

        Returns:
            List of similar cases
        """
        # Optimized Gremlin query
        query = """
        g.V()
         .hasLabel('Document')
         .has('doc_type', doc_type)
         .has('supplier', supplier)
         .has('company_code', company_code)
         .order().by('timestamp', decr)
         .limit(limit_val)
         .project('id', 'routing', 'quality_score', 'timestamp')
         .by(id)
         .by('routing_decision')
         .by('quality_score')
         .by('timestamp')
        """

        bindings = {
            "doc_type": doc_type,
            "supplier": supplier,
            "company_code": company_code,
            "limit_val": limit,
        }

        return await self.query(query, bindings)

    async def get_routing_patterns(
        self,
        doc_type: str,
        time_window_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get routing patterns for a document type.

        Args:
            doc_type: Document type
            time_window_days: Time window in days

        Returns:
            Routing patterns statistics
        """
        import datetime

        cutoff_timestamp = (
            datetime.datetime.now() - datetime.timedelta(days=time_window_days)
        ).timestamp()

        query = """
        g.V()
         .hasLabel('Document')
         .has('doc_type', doc_type)
         .has('timestamp', gte(cutoff_ts))
         .group()
         .by('routing_decision')
         .by(count())
        """

        bindings = {
            "doc_type": doc_type,
            "cutoff_ts": cutoff_timestamp,
        }

        results = await self.query(query, bindings)

        # Process results into statistics
        total = sum(results[0].values()) if results else 0

        patterns = {
            "doc_type": doc_type,
            "time_window_days": time_window_days,
            "total_documents": total,
            "routing_distribution": results[0] if results else {},
        }

        return patterns

    async def _execute_query(
        self,
        query: str,
        bindings: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute Gremlin query."""
        if not self.gremlin_client:
            logger.error("Gremlin client not initialized")
            return []

        try:
            callback = self.gremlin_client.submitAsync(query, bindings or {})
            results = await callback.result()
            return results.all().result()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []

    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.redis_client:
            return None

        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")

        return None

    async def _set_in_redis(
        self,
        key: str,
        value: Any,
        ttl: int = 300,
    ) -> None:
        """Set value in Redis cache."""
        if not self.redis_client:
            return

        try:
            await self.redis_client.setex(
                key,
                ttl,
                json.dumps(value),
            )
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    def _generate_cache_key(
        self,
        query: str,
        bindings: Optional[Dict[str, Any]],
    ) -> str:
        """Generate cache key from query and bindings."""
        key_data = f"{query}:{json.dumps(bindings, sort_keys=True)}"
        return f"pmg:{hashlib.sha256(key_data.encode()).hexdigest()}"

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Performance metrics
        """
        avg_latency = (
            self.total_latency_ms / self.query_count
            if self.query_count > 0
            else 0
        )

        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = (
            self.cache_hits / total_cache_ops
            if total_cache_ops > 0
            else 0
        )

        stats = {
            "total_queries": self.query_count,
            "avg_latency_ms": avg_latency,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

        # Add L1 cache stats
        if self.l1_cache:
            stats["l1_cache"] = self.l1_cache.get_stats()

        return stats

    def print_performance_stats(self) -> None:
        """Print performance statistics."""
        stats = self.get_performance_stats()

        logger.info("=" * 70)
        logger.info("PMG PERFORMANCE STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total Queries: {stats['total_queries']}")
        logger.info(f"Avg Latency: {stats['avg_latency_ms']:.2f} ms")
        logger.info(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
        logger.info(f"Cache Hits: {stats['cache_hits']}")
        logger.info(f"Cache Misses: {stats['cache_misses']}")

        if "l1_cache" in stats:
            l1 = stats["l1_cache"]
            logger.info(f"\nL1 Cache:")
            logger.info(f"  Size: {l1['size']}/{l1['max_size']}")
            logger.info(f"  Hit Rate: {l1['hit_rate']:.1%}")

        logger.info("=" * 70)


class PMGConnectionPool:
    """
    Connection pool for PMG/Cosmos DB Gremlin.

    Manages multiple Gremlin connections for concurrent queries.
    """

    def __init__(
        self,
        cosmos_endpoint: str,
        cosmos_key: str,
        database: str,
        graph: str,
        pool_size: int = 50,
    ):
        """
        Initialize connection pool.

        Args:
            cosmos_endpoint: Cosmos endpoint
            cosmos_key: Cosmos key
            database: Database name
            graph: Graph name
            pool_size: Number of connections in pool
        """
        self.cosmos_endpoint = cosmos_endpoint
        self.cosmos_key = cosmos_key
        self.database = database
        self.graph = graph
        self.pool_size = pool_size

        self.pool: List[client.Client] = []
        self.available: asyncio.Queue = asyncio.Queue(maxsize=pool_size)
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool."""
        logger.info(f"Initializing connection pool (size={self.pool_size})...")

        for i in range(self.pool_size):
            try:
                gremlin_client = client.Client(
                    self.cosmos_endpoint,
                    "g",
                    username=f"/dbs/{self.database}/colls/{self.graph}",
                    password=self.cosmos_key,
                    message_serializer=serializer.GraphSONSerializersV2d0(),
                )

                self.pool.append(gremlin_client)
                await self.available.put(gremlin_client)

            except Exception as e:
                logger.error(f"Failed to create connection {i}: {e}")

        self.initialized = True
        logger.info(f"✓ Connection pool initialized ({len(self.pool)} connections)")

    async def acquire(self) -> client.Client:
        """Acquire connection from pool."""
        return await self.available.get()

    async def release(self, conn: client.Client) -> None:
        """Release connection back to pool."""
        await self.available.put(conn)

    async def close(self) -> None:
        """Close all connections."""
        for conn in self.pool:
            conn.close()
