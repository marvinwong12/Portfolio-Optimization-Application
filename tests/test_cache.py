import pytest

from portfolio_optimizer.cache import TTLCache


def test_get_returns_none_for_missing_key():
    cache = TTLCache(ttl_seconds=60)
    assert cache.get('missing') is None


def test_set_then_get_returns_value():
    cache = TTLCache(ttl_seconds=60)
    cache.set('key', {'a': 1})
    assert cache.get('key') == {'a': 1}


def test_entry_expires_after_ttl(monkeypatch):
    cache = TTLCache(ttl_seconds=10)
    current_time = [1000.0]
    monkeypatch.setattr('portfolio_optimizer.cache.time.time', lambda: current_time[0])

    cache.set('key', 'value')
    assert cache.get('key') == 'value'

    current_time[0] += 11  # past the 10s TTL
    assert cache.get('key') is None


def test_entry_still_valid_just_before_ttl(monkeypatch):
    cache = TTLCache(ttl_seconds=10)
    current_time = [1000.0]
    monkeypatch.setattr('portfolio_optimizer.cache.time.time', lambda: current_time[0])

    cache.set('key', 'value')
    current_time[0] += 9
    assert cache.get('key') == 'value'


def test_clear_removes_all_entries():
    cache = TTLCache(ttl_seconds=60)
    cache.set('a', 1)
    cache.set('b', 2)
    cache.clear()
    assert cache.get('a') is None
    assert cache.get('b') is None


def test_hit_rate_is_none_before_any_get():
    cache = TTLCache(ttl_seconds=60)
    assert cache.hit_rate is None


def test_hit_rate_reflects_hits_and_misses():
    cache = TTLCache(ttl_seconds=60)
    cache.set('key', 'value')
    cache.get('key')       # hit
    cache.get('key')       # hit
    cache.get('missing')   # miss
    assert cache.hits == 2
    assert cache.misses == 1
    assert cache.hit_rate == pytest.approx(2 / 3)


def test_expired_entry_counts_as_a_miss(monkeypatch):
    cache = TTLCache(ttl_seconds=10)
    current_time = [1000.0]
    monkeypatch.setattr('portfolio_optimizer.cache.time.time', lambda: current_time[0])

    cache.set('key', 'value')
    current_time[0] += 11
    cache.get('key')
    assert cache.misses == 1
    assert cache.hits == 0


def test_clear_resets_hit_and_miss_counters():
    cache = TTLCache(ttl_seconds=60)
    cache.set('key', 'value')
    cache.get('key')
    cache.get('missing')
    cache.clear()
    assert cache.hits == 0
    assert cache.misses == 0
    assert cache.hit_rate is None

