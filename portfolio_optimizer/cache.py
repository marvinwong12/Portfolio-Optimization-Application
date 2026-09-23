"""Small in-memory TTL cache used to avoid re-hitting the Yahoo Finance API
for the same symbols/date-range on every page view."""
import time
import threading


class TTLCache:
    """In-memory cache with per-entry expiry, thread-safe for Flask's
    threaded request handling. Tracks hit/miss counts so the actual API-call
    reduction can be measured and reported rather than assumed."""

    def __init__(self, ttl_seconds):
        self.ttl_seconds = ttl_seconds
        self._store = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                self.misses += 1
                return None
            self.hits += 1
            return value

    def set(self, key, value):
        with self._lock:
            self._store[key] = (value, time.time() + self.ttl_seconds)

    def clear(self):
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    @property
    def hit_rate(self):
        """Fraction of get() calls served from cache (0.0-1.0), or None if
        get() has never been called."""
        total = self.hits + self.misses
        return None if total == 0 else self.hits / total


# Historical price data rarely needs refreshing more than a few times an hour.
price_cache = TTLCache(ttl_seconds=900)
# The T-bill risk-free rate changes at most once a day.
risk_free_rate_cache = TTLCache(ttl_seconds=3600)
