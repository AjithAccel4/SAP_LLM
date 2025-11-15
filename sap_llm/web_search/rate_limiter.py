"""
Rate Limiter for Web Search API Calls.

Provides token bucket and sliding window rate limiting algorithms.
"""

import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Thread-safe rate limiter with per-minute and per-day limits.

    Uses sliding window algorithm for accurate rate limiting and
    token bucket for burst handling.

    Features:
    - Per-minute and per-day limits
    - Thread-safe for concurrent requests
    - Automatic reset at midnight
    - Burst handling with token bucket
    - Waiting time estimation

    Example:
        >>> limiter = RateLimiter(requests_per_minute=100, requests_per_day=10000)
        >>> if limiter.can_proceed():
        ...     limiter.record_request()
        ...     # Make API call
    """

    def __init__(
        self,
        requests_per_minute: int = 100,
        requests_per_day: int = 10000,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            requests_per_day: Maximum requests per day
            burst_size: Maximum burst size (default: 2x per_minute)
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.burst_size = burst_size or (requests_per_minute * 2)

        # Sliding window for minute-level limiting
        self.minute_window: deque = deque()
        self.minute_window_seconds = 60

        # Daily counter
        self.daily_count = 0
        self.daily_reset_time = self._get_next_midnight()

        # Token bucket for burst handling
        self.tokens = float(self.burst_size)
        self.last_refill = time.time()
        self.refill_rate = requests_per_minute / 60.0  # Tokens per second

        # Thread safety
        self.lock = threading.RLock()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "minute_limit_hits": 0,
            "daily_limit_hits": 0
        }

        logger.debug(
            f"RateLimiter initialized: {requests_per_minute}/min, "
            f"{requests_per_day}/day, burst={self.burst_size}"
        )

    def can_proceed(self) -> bool:
        """
        Check if a request can proceed without blocking.

        Returns:
            True if request is allowed
        """
        with self.lock:
            self._refill_tokens()
            self._cleanup_minute_window()
            self._check_daily_reset()

            # Check daily limit
            if self.daily_count >= self.requests_per_day:
                self.stats["blocked_requests"] += 1
                self.stats["daily_limit_hits"] += 1
                logger.warning(
                    f"Daily limit reached ({self.requests_per_day}). "
                    f"Resets at {self.daily_reset_time}"
                )
                return False

            # Check minute limit
            if len(self.minute_window) >= self.requests_per_minute:
                self.stats["blocked_requests"] += 1
                self.stats["minute_limit_hits"] += 1
                return False

            # Check token bucket (for burst)
            if self.tokens < 1.0:
                self.stats["blocked_requests"] += 1
                return False

            return True

    def record_request(self) -> None:
        """Record that a request was made."""
        with self.lock:
            current_time = time.time()

            # Add to minute window
            self.minute_window.append(current_time)

            # Increment daily counter
            self.daily_count += 1

            # Consume token
            self.tokens = max(0.0, self.tokens - 1.0)

            # Update statistics
            self.stats["total_requests"] += 1

            logger.debug(
                f"Request recorded. Daily: {self.daily_count}/{self.requests_per_day}, "
                f"Minute window: {len(self.minute_window)}/{self.requests_per_minute}"
            )

    def wait_if_needed(self, timeout: Optional[float] = None) -> bool:
        """
        Wait if necessary until request can proceed.

        Args:
            timeout: Maximum time to wait in seconds (None = infinite)

        Returns:
            True if can proceed, False if timeout reached
        """
        start_time = time.time()

        while True:
            if self.can_proceed():
                return True

            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                logger.warning(f"Rate limit wait timeout after {timeout}s")
                return False

            # Calculate wait time
            wait_time = self._calculate_wait_time()

            if wait_time > 0:
                # Sleep for a fraction of wait time to check frequently
                sleep_time = min(wait_time, 0.1)
                time.sleep(sleep_time)
            else:
                time.sleep(0.01)  # Small sleep to prevent busy loop

    def _calculate_wait_time(self) -> float:
        """
        Calculate how long to wait before next request.

        Returns:
            Wait time in seconds
        """
        with self.lock:
            self._cleanup_minute_window()
            self._refill_tokens()

            # If daily limit hit, wait until midnight
            if self.daily_count >= self.requests_per_day:
                now = datetime.now()
                wait_seconds = (self.daily_reset_time - now).total_seconds()
                return max(0.0, wait_seconds)

            # If minute window full, wait until oldest expires
            if len(self.minute_window) >= self.requests_per_minute:
                oldest = self.minute_window[0]
                wait_seconds = self.minute_window_seconds - (time.time() - oldest)
                return max(0.0, wait_seconds)

            # If no tokens, wait for refill
            if self.tokens < 1.0:
                tokens_needed = 1.0 - self.tokens
                wait_seconds = tokens_needed / self.refill_rate
                return max(0.0, wait_seconds)

            return 0.0

    def _refill_tokens(self) -> None:
        """Refill token bucket based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_refill

        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)

        self.last_refill = current_time

    def _cleanup_minute_window(self) -> None:
        """Remove expired entries from minute window."""
        current_time = time.time()
        cutoff_time = current_time - self.minute_window_seconds

        # Remove entries older than window
        while self.minute_window and self.minute_window[0] < cutoff_time:
            self.minute_window.popleft()

    def _check_daily_reset(self) -> None:
        """Check if daily counter should reset."""
        now = datetime.now()

        if now >= self.daily_reset_time:
            logger.info(f"Daily rate limit reset. Previous count: {self.daily_count}")
            self.daily_count = 0
            self.daily_reset_time = self._get_next_midnight()

    def _get_next_midnight(self) -> datetime:
        """Get datetime for next midnight."""
        now = datetime.now()
        next_midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return next_midnight

    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            self._cleanup_minute_window()

            return {
                **self.stats,
                "current_minute_count": len(self.minute_window),
                "current_daily_count": self.daily_count,
                "available_tokens": self.tokens,
                "daily_limit": self.requests_per_day,
                "minute_limit": self.requests_per_minute,
                "next_daily_reset": self.daily_reset_time.isoformat(),
                "block_rate": (
                    self.stats["blocked_requests"] / self.stats["total_requests"]
                    if self.stats["total_requests"] > 0 else 0.0
                )
            }

    def reset(self) -> None:
        """Reset all counters and windows."""
        with self.lock:
            self.minute_window.clear()
            self.daily_count = 0
            self.tokens = float(self.burst_size)
            self.daily_reset_time = self._get_next_midnight()
            logger.info("Rate limiter reset")

    def get_remaining_quota(self) -> dict:
        """
        Get remaining request quota.

        Returns:
            Dictionary with remaining quotas
        """
        with self.lock:
            self._cleanup_minute_window()

            return {
                "minute_remaining": max(
                    0,
                    self.requests_per_minute - len(self.minute_window)
                ),
                "daily_remaining": max(
                    0,
                    self.requests_per_day - self.daily_count
                ),
                "can_burst": self.tokens >= 1.0
            }


class MultiProviderRateLimiter:
    """
    Manages rate limiters for multiple providers.

    Useful when coordinating rate limits across different APIs.

    Example:
        >>> limiter = MultiProviderRateLimiter({
        ...     "google": {"rpm": 100, "rpd": 10000},
        ...     "bing": {"rpm": 50, "rpd": 5000}
        ... })
        >>> limiter.can_proceed("google")
    """

    def __init__(self, provider_configs: dict):
        """
        Initialize multi-provider rate limiter.

        Args:
            provider_configs: Dict mapping provider names to config dicts
                             Config format: {"rpm": int, "rpd": int}
        """
        self.limiters = {}

        for provider, config in provider_configs.items():
            self.limiters[provider] = RateLimiter(
                requests_per_minute=config.get("rpm", 100),
                requests_per_day=config.get("rpd", 10000),
                burst_size=config.get("burst", None)
            )

        logger.info(f"Initialized rate limiters for {len(self.limiters)} providers")

    def can_proceed(self, provider: str) -> bool:
        """Check if provider can handle request."""
        if provider not in self.limiters:
            logger.warning(f"No rate limiter for provider: {provider}")
            return True

        return self.limiters[provider].can_proceed()

    def record_request(self, provider: str) -> None:
        """Record request for provider."""
        if provider in self.limiters:
            self.limiters[provider].record_request()

    def get_stats(self, provider: Optional[str] = None) -> dict:
        """Get statistics for provider(s)."""
        if provider:
            return self.limiters.get(provider, RateLimiter()).get_stats()

        return {
            name: limiter.get_stats()
            for name, limiter in self.limiters.items()
        }

    def reset_all(self) -> None:
        """Reset all provider rate limiters."""
        for limiter in self.limiters.values():
            limiter.reset()
