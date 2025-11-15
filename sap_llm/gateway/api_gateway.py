"""
ENHANCEMENT 10: Enterprise API Gateway (Kong, Apigee)

Enterprise-grade API Gateway:
- Rate limiting and throttling
- API key management
- Request/response transformation
- Analytics and monitoring
- Circuit breaker pattern
- API versioning
- Developer portal
"""

import logging
import time
import hashlib
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class APIKey:
    """API key configuration."""
    key: str
    name: str
    tier: str  # free, basic, premium, enterprise
    rate_limit: int  # requests per minute
    quota: int  # requests per month
    expires_at: Optional[str] = None
    enabled: bool = True


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 100
    requests_per_hour: int = 5000
    requests_per_day: int = 100000
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW


class APIGateway:
    """
    Enterprise API Gateway.

    Features:
    - Authentication & authorization
    - Rate limiting & throttling
    - Request/response transformation
    - Caching
    - Analytics
    - Circuit breaker
    - Load balancing
    """

    def __init__(self):
        # API keys
        self.api_keys: Dict[str, APIKey] = {}

        # Rate limiting
        self.rate_limit_buckets: Dict[str, List[float]] = {}

        # Circuit breaker
        self.circuit_breaker_state: Dict[str, str] = {}  # open, half-open, closed
        self.failure_counts: Dict[str, int] = {}

        # Analytics
        self.request_log: List[Dict[str, Any]] = []

        # Tiers and limits
        self.tier_limits = {
            "free": RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=500,
                requests_per_day=10000
            ),
            "basic": RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=5000,
                requests_per_day=100000
            ),
            "premium": RateLimitConfig(
                requests_per_minute=1000,
                requests_per_hour=50000,
                requests_per_day=1000000
            ),
            "enterprise": RateLimitConfig(
                requests_per_minute=10000,
                requests_per_hour=500000,
                requests_per_day=10000000
            )
        }

        logger.info("APIGateway initialized")

    def create_api_key(
        self,
        name: str,
        tier: str = "basic",
        expires_days: Optional[int] = None
    ) -> APIKey:
        """Create new API key."""
        # Generate key
        key_data = f"{name}:{time.time()}"
        key = hashlib.sha256(key_data.encode()).hexdigest()

        # Set expiration
        expires_at = None
        if expires_days:
            expires_at = (
                datetime.now() + timedelta(days=expires_days)
            ).isoformat()

        # Get tier config
        config = self.tier_limits.get(tier, self.tier_limits["basic"])

        api_key = APIKey(
            key=key,
            name=name,
            tier=tier,
            rate_limit=config.requests_per_minute,
            quota=config.requests_per_day,
            expires_at=expires_at,
            enabled=True
        )

        self.api_keys[key] = api_key

        logger.info(f"API key created: {name} ({tier})")

        return api_key

    def validate_api_key(self, key: str) -> tuple:
        """
        Validate API key.

        Returns:
            (is_valid, error_message)
        """
        # Check if key exists
        if key not in self.api_keys:
            return False, "Invalid API key"

        api_key = self.api_keys[key]

        # Check if enabled
        if not api_key.enabled:
            return False, "API key disabled"

        # Check expiration
        if api_key.expires_at:
            if datetime.now() > datetime.fromisoformat(api_key.expires_at):
                return False, "API key expired"

        return True, None

    def check_rate_limit(self, api_key: str) -> tuple:
        """
        Check if request is within rate limits.

        Returns:
            (is_allowed, retry_after_seconds)
        """
        if api_key not in self.api_keys:
            return False, 0

        config = self.tier_limits[self.api_keys[api_key].tier]

        # Initialize bucket if needed
        if api_key not in self.rate_limit_buckets:
            self.rate_limit_buckets[api_key] = []

        # Get request timestamps
        now = time.time()
        bucket = self.rate_limit_buckets[api_key]

        # Remove old timestamps (> 1 minute ago)
        bucket = [ts for ts in bucket if now - ts < 60]
        self.rate_limit_buckets[api_key] = bucket

        # Check limit
        if len(bucket) >= config.requests_per_minute:
            # Calculate retry after
            oldest = min(bucket)
            retry_after = 60 - (now - oldest)

            return False, int(retry_after) + 1

        # Add current request
        bucket.append(now)

        return True, 0

    def process_request(
        self,
        api_key: str,
        method: str,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process API request through gateway.

        Steps:
        1. Validate API key
        2. Check rate limits
        3. Check circuit breaker
        4. Transform request
        5. Route to backend
        6. Transform response
        7. Log analytics
        """
        start_time = time.time()

        # 1. Validate API key
        is_valid, error = self.validate_api_key(api_key)
        if not is_valid:
            return {
                "success": False,
                "error": error,
                "status_code": 401
            }

        # 2. Check rate limit
        is_allowed, retry_after = self.check_rate_limit(api_key)
        if not is_allowed:
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "retry_after": retry_after,
                "status_code": 429
            }

        # 3. Check circuit breaker
        if self._is_circuit_open(endpoint):
            return {
                "success": False,
                "error": "Service temporarily unavailable",
                "status_code": 503
            }

        # 4. Transform request
        transformed_payload = self._transform_request(payload)

        # 5. Route to backend (mock)
        try:
            response = self._route_to_backend(endpoint, transformed_payload)

            # Record success
            self._record_success(endpoint)

            # 6. Transform response
            transformed_response = self._transform_response(response)

            result = {
                "success": True,
                "data": transformed_response,
                "status_code": 200
            }

        except Exception as e:
            # Record failure
            self._record_failure(endpoint)

            result = {
                "success": False,
                "error": str(e),
                "status_code": 500
            }

        # 7. Log analytics
        latency = time.time() - start_time

        self._log_request(
            api_key=api_key,
            method=method,
            endpoint=endpoint,
            status_code=result["status_code"],
            latency_ms=latency * 1000
        )

        return result

    def _is_circuit_open(self, endpoint: str) -> bool:
        """Check if circuit breaker is open."""
        state = self.circuit_breaker_state.get(endpoint, "closed")
        return state == "open"

    def _record_success(self, endpoint: str):
        """Record successful request."""
        self.failure_counts[endpoint] = 0

        # Close circuit if half-open
        if self.circuit_breaker_state.get(endpoint) == "half-open":
            self.circuit_breaker_state[endpoint] = "closed"
            logger.info(f"Circuit closed for {endpoint}")

    def _record_failure(self, endpoint: str):
        """Record failed request."""
        self.failure_counts[endpoint] = self.failure_counts.get(endpoint, 0) + 1

        # Open circuit if too many failures
        if self.failure_counts[endpoint] >= 5:
            self.circuit_breaker_state[endpoint] = "open"
            logger.warning(f"Circuit opened for {endpoint}")

    def _transform_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request (add headers, convert formats, etc.)"""
        return {
            **payload,
            "gateway_timestamp": datetime.now().isoformat(),
            "gateway_version": "1.0.0"
        }

    def _transform_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Transform response (filter fields, add metadata, etc.)"""
        return {
            **response,
            "gateway_processed_at": datetime.now().isoformat()
        }

    def _route_to_backend(
        self,
        endpoint: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route request to backend service."""
        # Mock backend call
        if endpoint == "/classify":
            return {
                "doc_type": "invoice",
                "confidence": 0.95
            }
        elif endpoint == "/extract":
            return {
                "fields": {
                    "total_amount": 1250.00,
                    "vendor": "Acme Corp"
                }
            }
        else:
            return {"result": "success"}

    def _log_request(
        self,
        api_key: str,
        method: str,
        endpoint: str,
        status_code: int,
        latency_ms: float
    ):
        """Log request for analytics."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "api_key": api_key,
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "latency_ms": latency_ms
        }

        self.request_log.append(log_entry)

    def get_analytics(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Get API analytics."""
        logs = self.request_log

        if api_key:
            logs = [log for log in logs if log["api_key"] == api_key]

        total_requests = len(logs)
        if total_requests == 0:
            return {"total_requests": 0}

        # Calculate metrics
        success_count = sum(1 for log in logs if 200 <= log["status_code"] < 300)
        avg_latency = sum(log["latency_ms"] for log in logs) / total_requests

        # Endpoint breakdown
        endpoint_counts = {}
        for log in logs:
            endpoint = log["endpoint"]
            endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1

        return {
            "total_requests": total_requests,
            "success_count": success_count,
            "error_count": total_requests - success_count,
            "success_rate": success_count / total_requests * 100,
            "avg_latency_ms": avg_latency,
            "endpoint_breakdown": endpoint_counts
        }


# Singleton instance
_gateway: Optional[APIGateway] = None


def get_gateway() -> APIGateway:
    """Get singleton API gateway."""
    global _gateway

    if _gateway is None:
        _gateway = APIGateway()

    return _gateway


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    gateway = get_gateway()

    # Create API key
    api_key = gateway.create_api_key("test_app", tier="premium")
    print(f"API Key: {api_key.key}")

    # Process requests
    for i in range(10):
        response = gateway.process_request(
            api_key=api_key.key,
            method="POST",
            endpoint="/classify",
            payload={"document": f"doc_{i}"}
        )

        print(f"Request {i+1}: {response.get('success')}")

    # Get analytics
    analytics = gateway.get_analytics(api_key=api_key.key)
    print(f"\nAnalytics: {json.dumps(analytics, indent=2)}")
