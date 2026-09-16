from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.data_fetcher import StockDataFetcher
from portfolio_optimizer import data_fetcher as data_fetcher_module


def _make_history_df(n=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-02', periods=n)
    return pd.DataFrame({'Close': rng.random(n) * 50 + 100}, index=dates)


class _FakeTicker:
    call_count = 0

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None, interval=None, period=None):
        _FakeTicker.call_count += 1
        if self.symbol == 'MISSING':
            return pd.DataFrame()
        return _make_history_df()


@pytest.fixture(autouse=True)
def _reset_fake_ticker_calls():
    _FakeTicker.call_count = 0
    yield


class TestGetHistoricalData:
    def test_returns_close_series(self, monkeypatch):
        monkeypatch.setattr(data_fetcher_module.yf, 'Ticker', _FakeTicker)
        fetcher = StockDataFetcher()
        end = datetime.now()
        start = end - timedelta(days=60)

        result = fetcher.get_historical_data('AAPL', start, end)
        assert isinstance(result, pd.Series)
        assert len(result) == 30

    def test_raises_value_error_when_empty(self, monkeypatch):
        monkeypatch.setattr(data_fetcher_module.yf, 'Ticker', _FakeTicker)
        fetcher = StockDataFetcher()
        end = datetime.now()
        start = end - timedelta(days=60)

        with pytest.raises(ValueError):
            fetcher.get_historical_data('MISSING', start, end)

    def test_repeated_calls_are_cached(self, monkeypatch):
        monkeypatch.setattr(data_fetcher_module.yf, 'Ticker', _FakeTicker)
        fetcher = StockDataFetcher()
        end = datetime.now()
        start = end - timedelta(days=60)

        fetcher.get_historical_data('AAPL', start, end)
        fetcher.get_historical_data('AAPL', start, end)
        assert _FakeTicker.call_count == 1

    def test_cached_result_is_a_copy_not_a_shared_reference(self, monkeypatch):
        monkeypatch.setattr(data_fetcher_module.yf, 'Ticker', _FakeTicker)
        fetcher = StockDataFetcher()
        end = datetime.now()
        start = end - timedelta(days=60)

        first = fetcher.get_historical_data('AAPL', start, end)
        first.iloc[0] = -999999
        second = fetcher.get_historical_data('AAPL', start, end)
        assert second.iloc[0] != -999999


class TestGetMultipleStocks:
    def test_batches_symbols_into_one_download_call(self, monkeypatch):
        calls = []

        def fake_download(symbols, **kwargs):
            calls.append(symbols)
            dates = pd.bdate_range('2024-01-02', periods=20)
            columns = pd.MultiIndex.from_product([symbols, ['Close', 'Open']])
            data = np.random.default_rng(1).random((20, len(columns))) * 100
            return pd.DataFrame(data, index=dates, columns=columns)

        monkeypatch.setattr(data_fetcher_module.yf, 'download', fake_download)
        fetcher = StockDataFetcher()
        end = datetime.now()
        start = end - timedelta(days=60)

        df = fetcher.get_multiple_stocks(['AAA', 'BBB'], start, end)
        assert len(calls) == 1  # one batched call, not one per symbol
        assert list(df.columns) == ['AAA', 'BBB']
        assert len(df) == 20

    def test_raises_when_download_is_empty(self, monkeypatch):
        monkeypatch.setattr(data_fetcher_module.yf, 'download', lambda *a, **k: pd.DataFrame())
        fetcher = StockDataFetcher()
        end = datetime.now()
        start = end - timedelta(days=60)

        with pytest.raises(ValueError):
            fetcher.get_multiple_stocks(['AAA'], start, end)


class TestGetRiskFreeRate:
    def test_converts_percent_to_annual_decimal(self, monkeypatch):
        # ^IRX quoted as 5.25 (%) should become 0.0525, not deannualized.
        # yfinance returns single-ticker downloads with a MultiIndex
        # (field, ticker) column, which is why the production code does a
        # double .iloc[-1] - reproduce that shape here rather than a plain
        # single-level 'Close' column.
        dates = pd.bdate_range('2024-01-02', periods=5)
        columns = pd.MultiIndex.from_tuples([('Close', '^IRX')])
        values = np.array([5.0, 5.1, 5.2, 5.25, 5.25]).reshape(-1, 1)
        fake_df = pd.DataFrame(values, index=dates, columns=columns)

        def fake_download(*a, **k):
            return fake_df

        monkeypatch.setattr(data_fetcher_module.yf, 'download', fake_download)
        fetcher = StockDataFetcher()

        rate = fetcher.get_risk_free_rate()
        assert rate == pytest.approx(0.0525)

    def test_returns_none_on_failure(self, monkeypatch):
        def fake_download(*a, **k):
            raise RuntimeError('network down')

        monkeypatch.setattr(data_fetcher_module.yf, 'download', fake_download)
        fetcher = StockDataFetcher()
        assert fetcher.get_risk_free_rate() is None
