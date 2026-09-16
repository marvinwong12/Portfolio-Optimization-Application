"""Small in-memory TTL cache used to avoid re-hitting the Yahoo Finance API
for the same symbols/date-range on every page view."""
import time
import threading


class TTLCache:
    """In-memory cache with per-entry expiry, thread-safe for Flask's
    threaded request handling."""

    def __init__(self, ttl_seconds):
        self.ttl_seconds = ttl_seconds
        self._store = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key, value):
        with self._lock:
            self._store[key] = (value, time.time() + self.ttl_seconds)

    def clear(self):
        with self._lock:
            self._store.clear()


# Historical price data rarely needs refreshing more than a few times an hour.
price_cache = TTLCache(ttl_seconds=900)
# The T-bill risk-free rate changes at most once a day.
risk_free_rate_cache = TTLCache(ttl_seconds=3600)
