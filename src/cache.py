# src/cache.py
import json
import hashlib
import time
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.logging_config import get_logger

logger = get_logger("cache")

class CacheError(Exception):
    """Exception for cache-related errors."""
    pass

class BaseCache:
    """Base cache interface."""

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError

    def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError

class MemoryCache(BaseCache):
    """In-memory cache implementation."""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[str]:
        """Get value from memory cache."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if entry.get('expires_at') and datetime.now() > entry['expires_at']:
            del self._cache[key]
            return None

        return entry.get('value')

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        try:
            # Remove expired entries if cache is full
            if len(self._cache) >= self._max_size:
                self._cleanup_expired()

            if len(self._cache) >= self._max_size:
                # Remove oldest entries
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].get('created_at', datetime.min)
                )
                for old_key, _ in sorted_entries[:len(self._cache) // 4]:
                    del self._cache[old_key]

            expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None

            self._cache[key] = {
                'value': value,
                'created_at': datetime.now(),
                'expires_at': expires_at
            }
            return True
        except Exception as e:
            logger.error("Memory cache set failed", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        return self._cache.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        value = self.get(key)
        return value is not None

    def clear(self) -> bool:
        """Clear all cache entries."""
        self._cache.clear()
        return True

    def _cleanup_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.get('expires_at') and now > entry['expires_at']
        ]
        for key in expired_keys:
            del self._cache[key]

class RedisCache(BaseCache):
    """Redis cache implementation."""

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, **kwargs):
        if not REDIS_AVAILABLE:
            raise CacheError("Redis is not available. Install redis package.")

        try:
            self._client = redis.Redis(host=host, port=port, db=db, **kwargs)
            self._client.ping()  # Test connection
        except redis.RedisError as e:
            raise CacheError(f"Redis connection failed: {e}")

    def get(self, key: str) -> Optional[str]:
        """Get value from Redis cache."""
        try:
            value = self._client.get(key)
            return value.decode('utf-8') if value else None
        except Exception as e:
            logger.error("Redis cache get failed", key=key, error=str(e))
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            return self._client.set(key, value, ex=ttl)
        except Exception as e:
            logger.error("Redis cache set failed", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            return bool(self._client.delete(key))
        except Exception as e:
            logger.error("Redis cache delete failed", key=key, error=str(e))
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error("Redis cache exists failed", key=key, error=str(e))
            return False

    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            return bool(self._client.flushdb())
        except Exception as e:
            logger.error("Redis cache clear failed", error=str(e))
            return False

class CacheManager:
    """High-level cache manager with fallback support."""

    def __init__(self, primary: BaseCache, secondary: Optional[BaseCache] = None):
        self._primary = primary
        self._secondary = secondary

    def get(self, key: str) -> Optional[str]:
        """Get value with fallback to secondary cache."""
        value = self._primary.get(key)
        if value is not None:
            return value

        if self._secondary:
            value = self._secondary.get(key)
            if value is not None:
                # Populate primary cache
                self._primary.set(key, value)
                return value

        return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in both primary and secondary caches."""
        primary_success = self._primary.set(key, value, ttl)

        if self._secondary:
            secondary_success = self._secondary.set(key, value, ttl)
            return primary_success and secondary_success

        return primary_success

    def delete(self, key: str) -> bool:
        """Delete value from both caches."""
        primary_success = self._primary.delete(key)

        if self._secondary:
            secondary_success = self._secondary.delete(key)
            return primary_success and secondary_success

        return primary_success

    def exists(self, key: str) -> bool:
        """Check if key exists in either cache."""
        return self._primary.exists(key) or (
            self._secondary.exists(key) if self._secondary else False
        )

    def clear(self) -> bool:
        """Clear all cache entries."""
        primary_success = self._primary.clear()

        if self._secondary:
            secondary_success = self._secondary.clear()
            return primary_success and secondary_success

        return primary_success

def create_cache_key(*args, **kwargs) -> str:
    """Create a deterministic cache key from arguments."""
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_json = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_json.encode('utf-8')).hexdigest()

def cached(ttl: Optional[int] = 3600, key_prefix: str = "paraphrase"):
    """Decorator to cache function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Skip caching for certain conditions
            if kwargs.get('skip_cache', False):
                return func(*args, **kwargs)

            # Create cache key
            cache_key = f"{key_prefix}:{create_cache_key(func.__name__, *args, **kwargs)}"

            # Try to get from cache
            from src.config import get_cache
            cache = get_cache()
            cached_result = cache.get(cache_key)

            if cached_result is not None:
                logger.debug("Cache hit", function=func.__name__, key=cache_key)
                return json.loads(cached_result)

            # Execute function
            result = func(*args, **kwargs)

            # Cache result
            try:
                cache.set(cache_key, json.dumps(result), ttl)
                logger.debug("Cache set", function=func.__name__, key=cache_key)
            except Exception as e:
                logger.warning("Cache set failed", error=str(e))

            return result
        return wrapper
    return decorator

# Global cache instance
_cache_instance = None

def get_cache() -> CacheManager:
    """Get global cache instance."""
    global _cache_instance

    if _cache_instance is None:
        # Try Redis first, fallback to memory cache
        if REDIS_AVAILABLE:
            try:
                primary_cache = RedisCache()
                logger.info("Using Redis cache")
            except CacheError:
                primary_cache = MemoryCache()
                logger.info("Redis unavailable, using memory cache")
        else:
            primary_cache = MemoryCache()
            logger.info("Redis not available, using memory cache")

        _cache_instance = CacheManager(primary_cache)

    return _cache_instance