from portfolio_optimizer.routes import MAX_PEERS, _parse_peers


def test_blank_input_means_use_defaults():
    assert _parse_peers('', 'AAPL') is None
    assert _parse_peers('  ,  ', 'AAPL') is None


def test_cleans_uppercases_and_dedupes():
    assert _parse_peers('msft, googl msft', 'AAPL') == ['MSFT', 'GOOGL']


def test_removes_own_symbol_and_invalid_tokens():
    assert _parse_peers('AAPL, MSFT, <script>, BRK-B', 'AAPL') == ['MSFT', 'BRK-B']


def test_caps_number_of_peers():
    raw = ','.join(f'T{i}' for i in range(20))
    assert len(_parse_peers(raw, 'AAPL')) == MAX_PEERS
