import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.stock_analysis import StockAnalysis
from portfolio_optimizer import stock_analysis as stock_analysis_module


def _make_history_df(n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        'Close': close,
        'High': close + 1,
        'Low': close - 1,
    }, index=dates)


class _FakeTicker:
    def __init__(self, symbol, info=None, dividends=None):
        self.symbol = symbol
        self.info = info if info is not None else {}
        self.dividends = dividends if dividends is not None else pd.Series(dtype=float)

    def history(self, period=None):
        return _make_history_df()


def _make_stock_analysis(monkeypatch, info=None, dividends=None):
    def fake_ticker_factory(symbol):
        return _FakeTicker(symbol, info=info, dividends=dividends)

    monkeypatch.setattr(stock_analysis_module.yf, 'Ticker', fake_ticker_factory)
    return StockAnalysis('TEST')


class TestDividendAnalysis:
    def test_non_dividend_stock_does_not_crash(self, monkeypatch):
        """Regression test: dividendYield of None used to crash with
        TypeError (None / 100) for any non-dividend-paying stock."""
        stock = _make_stock_analysis(monkeypatch, info={'dividendYield': None})
        result = stock.dividend_analysis()
        assert result['dividend_yield'] is None
        assert result['has_dividends'] is False

    def test_dividend_yield_is_converted_to_decimal(self, monkeypatch):
        dates = pd.bdate_range('2023-01-02', periods=4)
        dividends = pd.Series([0.5, 0.5, 0.5, 0.5], index=dates)
        stock = _make_stock_analysis(monkeypatch, info={'dividendYield': 2.5}, dividends=dividends)
        result = stock.dividend_analysis()
        assert result['dividend_yield'] == pytest.approx(0.025)
        assert result['has_dividends'] is True

    def test_dividend_frequency_estimation(self, monkeypatch):
        dates = pd.bdate_range('2023-01-02', periods=4, freq='90D')
        dividends = pd.Series([0.5] * 4, index=dates)
        stock = _make_stock_analysis(monkeypatch, info={'dividendYield': None}, dividends=dividends)
        result = stock.dividend_analysis()
        assert result['dividend_frequency'] in ('Quarterly', 'Semi-Annual', 'Annual')


class TestPerformanceMetrics:
    def test_metrics_are_finite_for_normal_price_series(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch, info={})
        # Avoid a real network call for the SPY beta lookup.
        monkeypatch.setattr(
            stock_analysis_module.yf, 'Ticker',
            lambda symbol: _FakeTicker(symbol, info={})
        )
        metrics = stock.calculate_performance_metrics()
        assert np.isfinite(metrics['annualized_return'])
        assert metrics['annualized_volatility'] > 0
