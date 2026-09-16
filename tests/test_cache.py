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
