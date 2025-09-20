# src/rate_limiter.py
import time
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque

from src.logging_config import get_logger
from src.exceptions import RateLimitError

logger = get_logger("rate_limiter")

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, provider: str, retry_after: float):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {provider}")

class TokenBucketLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        initial_tokens: Optional[float] = None
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self):
        """Refill tokens based on time elapsed."""
        now = time.time()
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> Tuple[bool, float]:
        """
        Try to consume tokens.

        Returns:
            Tuple of (success, retry_after_seconds)
        """
        with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True, 0.0
            else:
                # Calculate time needed to get enough tokens
                needed_tokens = tokens - self.tokens
                retry_after = needed_tokens / self.refill_rate
                return False, retry_after

class FixedWindowLimiter:
    """Fixed window rate limiter implementation."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: Dict[int, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _get_window(self, timestamp: Optional[float] = None) -> int:
        """Get current window number."""
        ts = timestamp or time.time()
        return int(ts / self.window_seconds)

    def _cleanup_old_windows(self, current_window: int):
        """Remove old windows to prevent memory leaks."""
        windows_to_remove = [
            window for window in self.requests.keys()
            if window < current_window - 10  # Keep last 10 windows
        ]
        for window in windows_to_remove:
            del self.requests[window]

    def consume(self, count: int = 1) -> Tuple[bool, float]:
        """
        Try to consume from the current window.

        Returns:
            Tuple of (success, retry_after_seconds)
        """
        with self._lock:
            current_window = self._get_window()
            self._cleanup_old_windows(current_window)

            current_count = self.requests[current_window]

            if current_count + count <= self.limit:
                self.requests[current_window] += count
                return True, 0.0
            else:
                # Calculate time until next window
                next_window_start = (current_window + 1) * self.window_seconds
                retry_after = next_window_start - time.time()
                return False, retry_after

class SlidingWindowLimiter:
    """Sliding window rate limiter implementation."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = threading.Lock()

    def consume(self, count: int = 1) -> Tuple[bool, float]:
        """
        Try to consume using sliding window.

        Returns:
            Tuple of (success, retry_after_seconds)
        """
        with self._lock:
            now = time.time()

            # Remove requests outside the window
            while self.requests and self.requests[0] <= now - self.window_seconds:
                self.requests.popleft()

            current_count = sum(self.requests)

            if current_count + count <= self.limit:
                # Add new requests
                for _ in range(count):
                    self.requests.append(now)
                return True, 0.0
            else:
                # Calculate time until oldest request expires
                if self.requests:
                    retry_after = self.requests[0] + self.window_seconds - now
                else:
                    retry_after = 0.0
                return False, retry_after

class ProviderRateLimiter:
    """Rate limiter for API providers."""

    def __init__(self):
        self.limiters: Dict[str, Dict[str, TokenBucketLimiter]] = defaultdict(dict)
        self._setup_default_limits()

    def _setup_default_limits(self):
        """Setup default rate limits for providers."""

        # Gemini rate limits (requests per minute)
        self.limiters["gemini"]["requests"] = TokenBucketLimiter(
            capacity=60,  # 60 requests
            refill_rate=1.0  # 1 request per second
        )

        # OpenRouter rate limits (more conservative)
        self.limiters["openrouter"]["requests"] = TokenBucketLimiter(
            capacity=30,  # 30 requests
            refill_rate=0.5  # 2 requests per second
        )

    def set_limit(
        self,
        provider: str,
        limit_type: str,
        capacity: float,
        refill_rate: float
    ):
        """Set custom rate limit for a provider."""
        self.limiters[provider][limit_type] = TokenBucketLimiter(
            capacity=capacity,
            refill_rate=refill_rate
        )

    def check_limit(
        self,
        provider: str,
        limit_type: str = "requests",
        tokens: float = 1.0
    ) -> Tuple[bool, float]:
        """
        Check if request can proceed under rate limits.

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        if provider not in self.limiters or limit_type not in self.limiters[provider]:
            # No limit configured, allow request
            return True, 0.0

        limiter = self.limiters[provider][limit_type]
        return limiter.consume(tokens)

    def wait_if_needed(
        self,
        provider: str,
        limit_type: str = "requests",
        tokens: float = 1.0
    ):
        """Wait if rate limit would be exceeded."""
        allowed, retry_after = self.check_limit(provider, limit_type, tokens)

        if not allowed:
            if retry_after > 0:
                logger.info(
                    "Rate limit exceeded, waiting",
                    provider=provider,
                    limit_type=limit_type,
                    retry_after=f"{retry_after:.2f}s"
                )
                time.sleep(retry_after)

class GlobalRateLimiter:
    """Global rate limiter with provider-specific limits."""

    def __init__(self):
        self.provider_limiter = ProviderRateLimiter()
        self.global_limiter = TokenBucketLimiter(
            capacity=100,  # 100 requests total
            refill_rate=2.0  # 2 requests per second
        )

    def check_request(
        self,
        provider: str,
        limit_type: str = "requests",
        tokens: float = 1.0
    ) -> Tuple[bool, float]:
        """
        Check both global and provider-specific limits.

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        # Check global limit
        global_allowed, global_retry = self.global_limiter.consume(tokens)
        if not global_allowed:
            return False, global_retry

        # Check provider-specific limit
        provider_allowed, provider_retry = self.provider_limiter.check_limit(
            provider, limit_type, tokens
        )
        if not provider_allowed:
            # Rollback global limit consumption
            self.global_limiter.tokens += tokens
            return False, provider_retry

        return True, 0.0

    def wait_if_needed(
        self,
        provider: str,
        limit_type: str = "requests",
        tokens: float = 1.0
    ):
        """Wait if any rate limit would be exceeded."""
        allowed, retry_after = self.check_request(provider, limit_type, tokens)

        if not allowed:
            if retry_after > 0:
                logger.info(
                    "Rate limit exceeded, waiting",
                    provider=provider,
                    limit_type=limit_type,
                    retry_after=f"{retry_after:.2f}s"
                )
                time.sleep(retry_after)

def rate_limited(provider: str, limit_type: str = "requests", tokens: float = 1.0):
    """Decorator to apply rate limiting to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            from src.config import get_rate_limiter
            limiter = get_rate_limiter()

            try:
                limiter.wait_if_needed(provider, limit_type, tokens)
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "Rate limited function failed",
                    function=func.__name__,
                    provider=provider,
                    error=str(e)
                )
                raise RateLimitError(provider=provider, original_error=e)
        return wrapper
    return decorator

# Global rate limiter instance
_rate_limiter_instance = None

def get_rate_limiter() -> GlobalRateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter_instance

    if _rate_limiter_instance is None:
        _rate_limiter_instance = GlobalRateLimiter()
        logger.info("Initialized global rate limiter")

    return _rate_limiter_instance