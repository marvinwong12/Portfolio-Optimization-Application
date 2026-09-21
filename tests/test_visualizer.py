import base64
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from portfolio_optimizer import visualizer
from portfolio_optimizer.visualizer import PortfolioVisualizer, _date_axis, _percent_axis, _spread_labels

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def assert_valid_png(encoded):
    assert isinstance(encoded, str)
    data = base64.b64decode(encoded)
    assert data.startswith(PNG_SIGNATURE)
    assert len(data) > 5_000  # a real rendered chart, not an empty canvas


def _returns(n_assets=4, n=300, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range('2023-01-02', periods=n)
    columns = [f'A{i}' for i in range(n_assets)]
    return pd.DataFrame(rng.normal(0.0005, 0.01, (n, n_assets)), index=index, columns=columns)


def _cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.15, 0.30, n)
    rets = rng.uniform(0.05, 0.25, n)
    return rets, vols, (rets - 0.04) / vols


class TestEfficientFrontierChart:
    def test_renders_with_only_the_cloud(self):
        rets, vols, sharpe = _cloud()
        assert_valid_png(PortfolioVisualizer.plot_efficient_frontier(rets, vols, sharpe))

    def test_renders_with_every_optional_layer(self):
        rets, vols, sharpe = _cloud()
        frontier = {'returns': np.linspace(0.10, 0.24, 20), 'volatilities': np.linspace(0.14, 0.29, 20)}
        encoded = PortfolioVisualizer.plot_efficient_frontier(
            rets, vols, sharpe,
            optimal_portfolio={'volatility': 0.18, 'return': 0.16, 'sharpe_ratio': 0.67},
            frontier=frontier,
            min_variance_portfolio={'volatility': 0.14, 'return': 0.10},
            assets={'AAA': {'volatility': 0.25, 'return': 0.14}, 'BBB': {'volatility': 0.30, 'return': 0.09}},
            risk_free_rate=0.04,
        )
        assert_valid_png(encoded)

    def test_capital_market_line_is_skipped_when_tangency_does_not_beat_the_risk_free_rate(self):
        rets, vols, sharpe = _cloud()
        encoded = PortfolioVisualizer.plot_efficient_frontier(
            rets, vols, sharpe,
            optimal_portfolio={'volatility': 0.18, 'return': 0.02}, risk_free_rate=0.04,
        )
        assert_valid_png(encoded)  # no crash on a non-positive slope

    def test_accepts_an_empty_frontier(self):
        rets, vols, sharpe = _cloud()
        empty = {'returns': np.array([]), 'volatilities': np.array([])}
        assert_valid_png(PortfolioVisualizer.plot_efficient_frontier(rets, vols, sharpe, frontier=empty))


class TestWeightsChart:
    def test_long_only_donut(self):
        assert_valid_png(PortfolioVisualizer.plot_weights(
            np.array([0.5, 0.3, 0.2, 0.0]), ['A', 'B', 'C', 'D'], 'Weights'))

    def test_donut_with_a_single_holding(self):
        assert_valid_png(PortfolioVisualizer.plot_weights(np.array([1.0, 0.0]), ['A', 'B'], 'Weights'))

    def test_donut_with_more_assets_than_palette_colors(self):
        n = len(visualizer.ASSET_COLORS) + 5
        weights = np.ones(n) / n
        assert_valid_png(PortfolioVisualizer.plot_weights(weights, [f'S{i}' for i in range(n)], 'Weights'))

    def test_long_short_bars_with_negative_weights(self):
        assert_valid_png(PortfolioVisualizer.plot_weights(
            np.array([0.6, -0.2, 0.4, 0.2]), ['A', 'B', 'C', 'D'], 'Weights', long_only=False))

    def test_accepts_a_pandas_index_of_symbols(self):
        returns = _returns(3)
        assert_valid_png(PortfolioVisualizer.plot_weights(np.ones(3) / 3, returns.columns, 'Weights'))


class TestCorrelationChart:
    @pytest.mark.parametrize('n_assets', [1, 2, 5, 12])
    def test_renders_for_any_number_of_assets(self, n_assets):
        assert_valid_png(PortfolioVisualizer.plot_correlation_matrix(_returns(n_assets).corr()))


class TestReturnsTimeSeriesChart:
    @pytest.mark.parametrize('n_assets', [1, 4, 13])
    def test_renders_for_any_number_of_assets(self, n_assets):
        assert_valid_png(PortfolioVisualizer.plot_returns_time_series(_returns(n_assets)))


class TestBacktestChart:
    def _curves(self, names):
        index = pd.bdate_range('2020-01-02', periods=400)
        rng = np.random.default_rng(1)
        return {name: (1 + pd.Series(rng.normal(0.0004, 0.01, 400), index=index)).cumprod()
                for name in names}

    def test_strategies_with_a_benchmark(self):
        curves = self._curves(['Tangency', 'Minimum Variance', 'Equal Weight', 'SPY (Buy & Hold)'])
        assert_valid_png(PortfolioVisualizer.plot_backtest_comparison(curves))

    def test_a_single_curve(self):
        assert_valid_png(PortfolioVisualizer.plot_backtest_comparison(self._curves(['Equal Weight'])))

    def test_unknown_strategy_names_still_get_a_color(self):
        assert_valid_png(PortfolioVisualizer.plot_backtest_comparison(self._curves(['Custom A', 'Custom B'])))


class TestWeightHistoryChart:
    def test_two_snapshots(self):
        dates = list(pd.to_datetime(['2024-01-01', '2024-04-01']))
        matrix = np.array([[0.6, 0.4], [0.5, 0.5]])
        assert_valid_png(PortfolioVisualizer.plot_weight_history(dates, matrix, ['A', 'B'], 'History'))

    def test_snapshots_only_seconds_apart(self):
        dates = list(pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 10:00:07']))
        matrix = np.array([[0.6, 0.4], [0.6, 0.4]])
        assert_valid_png(PortfolioVisualizer.plot_weight_history(dates, matrix, ['A', 'B'], 'History'))


class TestRenderingHygiene:
    def test_no_pyplot_figures_are_left_open(self):
        """Charts are built on standalone Figure objects, not pyplot's global
        state, so rendering must not accumulate open figures (a memory leak in
        a long-running server)."""
        before = set(plt.get_fignums())
        rets, vols, sharpe = _cloud()
        PortfolioVisualizer.plot_efficient_frontier(rets, vols, sharpe)
        PortfolioVisualizer.plot_weights(np.array([0.5, 0.5]), ['A', 'B'], 'W')
        PortfolioVisualizer.plot_correlation_matrix(_returns(3).corr())
        PortfolioVisualizer.plot_returns_time_series(_returns(3))
        assert set(plt.get_fignums()) == before

    def test_concurrent_rendering_is_safe(self):
        """The deployed app runs gunicorn with threads; pyplot's global
        state is not thread-safe, standalone Figures are."""
        def render(seed):
            rets, vols, sharpe = _cloud(seed=seed)
            PortfolioVisualizer.plot_efficient_frontier(rets, vols, sharpe)
            PortfolioVisualizer.plot_weights(np.array([0.5, 0.3, 0.2]), ['A', 'B', 'C'], 'W')
            return PortfolioVisualizer.plot_correlation_matrix(_returns(4, seed=seed).corr())

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(render, range(16)))
        for encoded in results:
            assert_valid_png(encoded)


class TestHelpers:
    def test_spread_labels_enforces_the_minimum_gap_and_keeps_order(self):
        positions = np.array([1.00, 1.01, 1.02, 3.0])
        spread = _spread_labels(positions, min_gap=0.5)
        order = np.argsort(positions)
        gaps = np.diff(spread[order])
        assert np.all(gaps >= 0.5 - 1e-9)
        assert list(np.argsort(spread)) == list(order)

    def test_spread_labels_leaves_well_separated_labels_alone(self):
        positions = np.array([1.0, 2.0, 3.0])
        assert _spread_labels(positions, min_gap=0.1) == pytest.approx(positions)

    def test_percent_axis_does_not_round_half_percent_ticks_to_whole_numbers(self):
        """Regression: with whole-percent formatting, a tick at 32.5% was
        labelled "33%", which misstates the value."""
        fig = Figure()
        ax = fig.subplots()
        ax.plot([0, 1], [0.30, 0.35])
        ax.set_yticks([0.30, 0.325, 0.35])
        _percent_axis(ax.yaxis)
        fig.canvas.draw()
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        assert '32.5%' in labels

    def test_date_axis_ticks_stay_inside_the_data_range(self):
        """The right-hand margin reserved for end-of-line labels must not
        get its own ticks (e.g. a "2027" tick after the last data point)."""
        import matplotlib.dates as mdates
        index = pd.bdate_range('2021-09-20', '2026-09-18')
        fig = Figure()
        ax = fig.subplots()
        ax.plot(index, np.arange(len(index)))
        _date_axis(ax, index[0], index[-1])
        ax.set_xlim(index[0], index[-1] + (index[-1] - index[0]) * 0.2)
        fig.canvas.draw()
        low, high = mdates.date2num(index[0]), mdates.date2num(index[-1])
        ticks = ax.get_xticks()
        assert len(ticks) > 0
        assert all(low <= tick <= high for tick in ticks)
