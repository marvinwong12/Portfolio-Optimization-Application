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


class TestEstimateDividendFrequency:
    def test_empty_dividends_is_unknown(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch)
        assert stock._estimate_dividend_frequency(pd.Series(dtype=float)) == 'Unknown'

    def test_single_dividend_is_unknown(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch)
        series = pd.Series([0.5], index=pd.bdate_range('2023-01-02', periods=1))
        assert stock._estimate_dividend_frequency(series) == 'Unknown'

    def test_monthly_spacing_is_quarterly_bucket(self, monkeypatch):
        # avg_days < 40 -> "Quarterly" in this simplified 3-bucket model.
        stock = _make_stock_analysis(monkeypatch)
        dates = pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
        assert stock._estimate_dividend_frequency(pd.Series([0.1] * 3, index=dates)) == 'Quarterly'

    def test_annual_spacing_is_annual(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch)
        dates = pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01'])
        assert stock._estimate_dividend_frequency(pd.Series([0.1] * 3, index=dates)) == 'Annual'


class TestTechnicalIndicators:
    def test_keys_present_and_finite_with_enough_history(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch)
        stock.historical_data = _make_history_df(n=250)
        indicators = stock.calculate_technical_indicators()
        for key in ('sma_50', 'sma_200', 'ema_20', 'rsi', 'macd', 'macd_signal',
                    'bollinger_upper', 'bollinger_lower', 'bollinger_percent'):
            assert key in indicators
            assert np.isfinite(indicators[key])

    def test_rsi_is_bounded_between_0_and_100(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch)
        stock.historical_data = _make_history_df(n=250)
        rsi = stock.calculate_technical_indicators()['rsi']
        assert 0 <= rsi <= 100

    def test_bollinger_bands_bracket_the_moving_average(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch)
        stock.historical_data = _make_history_df(n=250)
        indicators = stock.calculate_technical_indicators()
        assert indicators['bollinger_lower'] < indicators['bollinger_upper']

    def test_monotonically_rising_prices_give_rsi_of_100(self, monkeypatch):
        """A pure uptrend has zero losses, so RSI saturates at 100 rather
        than raising a division-by-zero error."""
        stock = _make_stock_analysis(monkeypatch)
        dates = pd.bdate_range('2023-01-02', periods=60)
        close = 100 + np.arange(60, dtype=float)
        stock.historical_data = pd.DataFrame({'Close': close, 'High': close + 1, 'Low': close - 1}, index=dates)
        assert stock.calculate_technical_indicators()['rsi'] == pytest.approx(100.0)

    def test_fetches_data_when_not_already_loaded(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch)
        assert stock.historical_data is None
        indicators = stock.calculate_technical_indicators()
        assert stock.historical_data is not None
        assert 'rsi' in indicators


class TestValuationRatios:
    def test_passes_through_known_info_fields(self, monkeypatch):
        info = {
            'trailingPE': 25.4, 'forwardPE': 22.1, 'pegRatio': 1.8,
            'priceToSalesTrailing12Months': 6.2, 'priceToBook': 12.5,
            'enterpriseToEbitda': 18.0, 'enterpriseToRevenue': 5.9, 'dividendYield': 0.5,
        }
        stock = _make_stock_analysis(monkeypatch, info=info)
        ratios = stock.calculate_valuation_ratios()
        assert ratios == {
            'pe_ratio': 25.4, 'forward_pe': 22.1, 'peg_ratio': 1.8,
            'price_to_sales': 6.2, 'price_to_book': 12.5,
            'ev_to_ebitda': 18.0, 'ev_to_revenue': 5.9, 'dividend_yield': 0.5,
        }

    def test_missing_fields_are_none(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch, info={})
        ratios = stock.calculate_valuation_ratios()
        assert all(value is None for value in ratios.values())


class _FakeTickerWithFinancials(_FakeTicker):
    def __init__(self, symbol, info=None, dividends=None, financials=None,
                 income_stmt=None, balance_sheet=None, cash_flow=None):
        super().__init__(symbol, info=info, dividends=dividends)
        self.financials = financials if financials is not None else pd.DataFrame()
        self.income_stmt = income_stmt if income_stmt is not None else pd.DataFrame()
        self.balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame()
        self.cash_flow = cash_flow if cash_flow is not None else pd.DataFrame()


def _make_stock_with_financials(monkeypatch, **kwargs):
    def factory(symbol):
        return _FakeTickerWithFinancials(symbol, **kwargs)
    monkeypatch.setattr(stock_analysis_module.yf, 'Ticker', factory)
    return StockAnalysis('TEST')


class TestProfitabilityMetrics:
    def test_computes_margins_and_returns_from_financials(self, monkeypatch):
        year = pd.Timestamp('2023-12-31')
        financials = pd.DataFrame(
            {year: [500.0, 300.0, 200.0, 1000.0]},
            index=['Gross Profit', 'Operating Income', 'Net Income', 'Total Revenue'],
        )
        income_stmt = pd.DataFrame({year: [200.0]}, index=['Net Income'])
        balance_sheet = pd.DataFrame(
            {year: [800.0, 2000.0]},
            index=['Total Stockholder Equity', 'Total Assets'],
        )
        stock = _make_stock_with_financials(
            monkeypatch, financials=financials, income_stmt=income_stmt, balance_sheet=balance_sheet,
        )
        metrics = stock.calculate_profitability_metrics()
        assert metrics['gross_margin'] == pytest.approx(0.5)
        assert metrics['operating_margin'] == pytest.approx(0.3)
        assert metrics['net_margin'] == pytest.approx(0.2)
        assert metrics['return_on_equity'] == pytest.approx(0.25)
        assert metrics['return_on_assets'] == pytest.approx(0.1)

    def test_missing_line_items_yield_none_without_crashing(self, monkeypatch):
        year = pd.Timestamp('2023-12-31')
        financials = pd.DataFrame({year: [1000.0]}, index=['Total Revenue'])  # margins missing
        stock = _make_stock_with_financials(monkeypatch, financials=financials)
        metrics = stock.calculate_profitability_metrics()
        assert metrics['gross_margin'] is None
        assert metrics['operating_margin'] is None
        assert metrics['net_margin'] is None

    def test_empty_financials_returns_all_none(self, monkeypatch):
        stock = _make_stock_with_financials(monkeypatch)  # empty DataFrames -> IndexError internally
        metrics = stock.calculate_profitability_metrics()
        assert metrics == {
            'gross_margin': None, 'operating_margin': None, 'net_margin': None,
            'return_on_equity': None, 'return_on_assets': None,
        }


class TestDCFValuation:
    def test_hand_computed_fair_value(self, monkeypatch):
        year = pd.Timestamp('2023-12-31')
        cash_flow = pd.DataFrame({year: [100.0]}, index=['Free Cash Flow'])
        balance_sheet = pd.DataFrame({year: [50.0, 20.0]}, index=['Cash', 'Total Debt'])
        stock = _make_stock_with_financials(
            monkeypatch, info={'sharesOutstanding': 10.0}, cash_flow=cash_flow, balance_sheet=balance_sheet,
        )

        discount_rate, growth = 0.08, 0.02
        fcf = 100.0
        future = [fcf * (1 + growth) ** y / (1 + discount_rate) ** y for y in range(1, 6)]
        terminal = (future[-1] * (1 + growth)) / (discount_rate - growth) / (1 + discount_rate) ** 5
        expected_equity = sum(future) + terminal - 20.0 + 50.0
        expected_fair_value = expected_equity / 10.0

        assert stock.dcf_valuation(discount_rate, growth) == pytest.approx(expected_fair_value)

    def test_estimates_fcf_when_not_directly_available(self, monkeypatch):
        year = pd.Timestamp('2023-12-31')
        cash_flow = pd.DataFrame(
            {year: [120.0, -20.0]}, index=['Operating Cash Flow', 'Capital Expenditure'],
        )
        balance_sheet = pd.DataFrame({year: [0.0, 0.0]}, index=['Cash', 'Total Debt'])
        stock = _make_stock_with_financials(
            monkeypatch, info={'sharesOutstanding': 100.0}, cash_flow=cash_flow, balance_sheet=balance_sheet,
        )
        assert stock.dcf_valuation() is not None

    def test_returns_none_without_shares_outstanding(self, monkeypatch):
        year = pd.Timestamp('2023-12-31')
        cash_flow = pd.DataFrame({year: [100.0]}, index=['Free Cash Flow'])
        stock = _make_stock_with_financials(monkeypatch, info={}, cash_flow=cash_flow)
        assert stock.dcf_valuation() is None

    def test_returns_none_when_cash_flow_is_empty(self, monkeypatch):
        stock = _make_stock_with_financials(monkeypatch, info={'sharesOutstanding': 10.0})
        assert stock.dcf_valuation() is None

    def test_returns_none_when_discount_rate_equals_growth_rate(self, monkeypatch):
        """Terminal value divides by (discount_rate - growth): equal rates
        must not raise ZeroDivisionError, just fail to produce a valuation."""
        year = pd.Timestamp('2023-12-31')
        cash_flow = pd.DataFrame({year: [100.0]}, index=['Free Cash Flow'])
        stock = _make_stock_with_financials(
            monkeypatch, info={'sharesOutstanding': 10.0}, cash_flow=cash_flow,
        )
        assert stock.dcf_valuation(discount_rate=0.05, perpetual_growth=0.05) is None


class TestRelativeValuation:
    def test_combines_base_and_peer_metrics(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch, info={'trailingPE': 20.0})
        result = stock.relative_valuation(['PEER1', 'PEER2'])
        assert result['base_company']['pe_ratio'] == 20.0
        assert set(result['peers']) == {'PEER1', 'PEER2'}
        assert result['peers']['PEER1']['pe_ratio'] == 20.0  # fake Ticker returns the same info for every symbol

    def test_a_failing_peer_does_not_break_the_others(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch, info={'trailingPE': 20.0})

        calls = {'n': 0}
        real_init = StockAnalysis.__init__

        def flaky_init(self, ticker_symbol):
            calls['n'] += 1
            if ticker_symbol == 'BAD':
                raise ValueError('simulated fetch failure')
            real_init(self, ticker_symbol)

        monkeypatch.setattr(StockAnalysis, '__init__', flaky_init)
        result = stock.relative_valuation(['GOOD', 'BAD'])
        assert result['peers']['BAD'] == 'Error fetching data'
        assert result['peers']['GOOD']['pe_ratio'] == 20.0  # fake Ticker returns the same info for every symbol


class TestPeerComparison:
    def test_peer_median_uses_only_numeric_loaded_peers(self, monkeypatch):
        infos = {'A': {'trailingPE': 10.0}, 'B': {'trailingPE': 30.0}, 'C': {'trailingPE': 50.0}}

        def factory(symbol):
            return _FakeTicker(symbol, info=infos.get(symbol, {'trailingPE': 20.0}))

        monkeypatch.setattr(stock_analysis_module.yf, 'Ticker', factory)
        stock = StockAnalysis('TEST')
        result = stock.relative_valuation(['A', 'B', 'C'])
        assert result['peer_median']['pe_ratio'] == 30.0
        assert result['peer_median']['peg_ratio'] is None  # nobody reports it

    def test_median_ignores_failed_peers(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch, info={'trailingPE': 20.0})
        real_init = StockAnalysis.__init__

        def flaky_init(self, ticker_symbol):
            if ticker_symbol == 'BAD':
                raise ValueError('boom')
            real_init(self, ticker_symbol)

        monkeypatch.setattr(StockAnalysis, '__init__', flaky_init)
        result = stock.relative_valuation(['GOOD', 'BAD'])
        assert list(result['peers']) == ['GOOD', 'BAD']  # order preserved
        assert result['peer_median']['pe_ratio'] == 20.0

    def test_default_peers_are_same_sector_and_exclude_self(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch, info={'sector': 'Technology'})
        stock.symbol = 'AAPL'
        peers = stock.default_peers()
        assert 'AAPL' not in peers
        assert len(peers) == 4

    def test_default_peers_empty_for_unknown_sector(self, monkeypatch):
        stock = _make_stock_analysis(monkeypatch, info={'sector': 'Made Up'})
        assert stock.default_peers() == []

    def test_comprehensive_analysis_includes_peer_comparison(self, monkeypatch):
        stock = _make_stock_with_financials(monkeypatch, info={'sector': 'Technology', 'dividendYield': None})
        results = stock.comprehensive_analysis()
        assert 'relative_valuation' in results
        assert results['relative_valuation']['peer_median']

    def test_explicit_empty_peer_list_skips_comparison(self, monkeypatch):
        stock = _make_stock_with_financials(monkeypatch, info={'sector': 'Technology', 'dividendYield': None})
        assert 'relative_valuation' not in stock.comprehensive_analysis(peers=[])


class TestComprehensiveAnalysis:
    def test_assembles_every_section(self, monkeypatch):
        info = {
            'longName': 'Test Corp', 'sector': 'Technology', 'industry': 'Software',
            'marketCap': 1_000_000, 'regularMarketPrice': 42.0,
            'fiftyTwoWeekHigh': 50.0, 'fiftyTwoWeekLow': 30.0,
            'recommendationKey': 'buy', 'targetMeanPrice': 55.0, 'numberOfAnalystOpinions': 12,
            'dividendYield': None,
        }
        stock = _make_stock_with_financials(monkeypatch, info=info)
        results = stock.comprehensive_analysis()

        assert results['basic_info']['name'] == 'Test Corp'
        assert results['basic_info']['sector'] == 'Technology'
        assert results['analyst_data']['recommendation'] == 'buy'
        for section in ('performance_metrics', 'technical_indicators', 'valuation_ratios',
                         'profitability_metrics', 'dividend_analysis'):
            assert section in results
        assert results['dcf_valuation'] is None  # no financials provided
        assert stock.analysis_results is results

    def test_falls_back_to_symbol_when_long_name_is_missing(self, monkeypatch):
        stock = _make_stock_with_financials(monkeypatch, info={'dividendYield': None})
        results = stock.comprehensive_analysis()
        assert results['basic_info']['name'] == 'TEST'
