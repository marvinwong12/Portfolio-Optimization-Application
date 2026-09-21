import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.backtest import (
    run_backtest,
    compare_strategies,
    compute_performance_metrics,
    BacktestResult,
    STRATEGIES,
)

RISK_FREE_RATE = 0.04


def _synthetic_returns(n=800, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-02', periods=n)
    data = rng.normal(loc=0.0005, scale=0.012, size=(n, 3))
    return pd.DataFrame(data, index=dates, columns=['AAA', 'BBB', 'CCC'])


class TestRunBacktestValidation:
    def test_raises_for_unknown_strategy(self):
        returns = _synthetic_returns(n=400)
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_backtest(returns, RISK_FREE_RATE, strategy='not_a_strategy')

    def test_raises_when_not_enough_data(self):
        returns = _synthetic_returns(n=100)
        with pytest.raises(ValueError, match="Not enough data"):
            run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight', lookback_days=252)


class TestEqualWeightBacktest:
    def test_matches_manual_calculation(self):
        """Equal-weight backtest output should exactly match manually
        computing (returns @ [1/n, ...]) over the out-of-sample period,
        since equal weight never depends on the estimation window."""
        returns = _synthetic_returns(n=400, seed=2)
        lookback = 100

        # drift=False is the constant-mix model: the same target weights are
        # applied to every day's returns.
        result = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight',
                               lookback_days=lookback, rebalance_days=50, drift=False)

        n_assets = returns.shape[1]
        expected = returns.iloc[lookback:].values @ (np.ones(n_assets) / n_assets)
        np.testing.assert_allclose(result.daily_returns.values, expected, atol=1e-12)

    def test_output_index_starts_after_lookback_window(self):
        returns = _synthetic_returns(n=400, seed=2)
        lookback = 100
        result = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight',
                               lookback_days=lookback, rebalance_days=50)
        assert result.daily_returns.index[0] == returns.index[lookback]
        assert result.daily_returns.index[-1] == returns.index[-1]
        assert len(result.daily_returns) == len(returns) - lookback


class TestNoLookaheadBias:
    def test_weights_at_a_rebalance_date_are_unaffected_by_later_data(self):
        """Regression-style test for the core correctness property of a
        walk-forward backtest: weights computed at a given rebalance date
        must depend only on returns strictly before it. We run the backtest
        twice with identical history up to a cutoff and then diverging
        future returns, and check the first rebalance's weights are
        identical in both runs."""
        lookback = 150
        rebalance = 50
        base = _synthetic_returns(n=lookback + rebalance + 10, seed=3)

        # Two datasets that agree exactly up to `lookback`, then diverge.
        rng = np.random.default_rng(999)
        future_a = base.copy()
        future_b = base.copy()
        future_b.iloc[lookback:] = rng.normal(0.05, 0.05, size=future_b.iloc[lookback:].shape)

        result_a = run_backtest(future_a, RISK_FREE_RATE, strategy='tangency',
                                 lookback_days=lookback, rebalance_days=rebalance)
        result_b = run_backtest(future_b, RISK_FREE_RATE, strategy='tangency',
                                 lookback_days=lookback, rebalance_days=rebalance)

        first_weights_a = result_a.weights_history[0]['weights']
        first_weights_b = result_b.weights_history[0]['weights']
        assert first_weights_a == pytest.approx(first_weights_b)


class TestLongOnlyWeightsAreValid:
    @pytest.mark.parametrize('strategy', STRATEGIES)
    def test_weights_sum_to_one_at_every_rebalance(self, strategy):
        returns = _synthetic_returns(n=600, seed=4)
        result = run_backtest(returns, RISK_FREE_RATE, strategy=strategy,
                               lookback_days=200, rebalance_days=63, long_only=True)

        assert len(result.weights_history) > 1
        for entry in result.weights_history:
            weights = np.array(list(entry['weights'].values()))
            assert weights.sum() == pytest.approx(1.0)
            assert np.all(weights >= -1e-9)


class TestPerformanceMetrics:
    def test_metrics_of_flat_returns(self):
        dates = pd.bdate_range('2023-01-02', periods=100)
        flat = pd.Series(np.zeros(100), index=dates)
        metrics = compute_performance_metrics(flat, RISK_FREE_RATE)
        assert metrics['total_return'] == pytest.approx(0.0)
        assert metrics['annualized_volatility'] == pytest.approx(0.0)
        assert metrics['sharpe_ratio'] == 0  # guarded zero-volatility case

    def test_metrics_of_empty_series(self):
        empty = pd.Series([], dtype=float)
        metrics = compute_performance_metrics(empty, RISK_FREE_RATE)
        assert np.isnan(metrics['sharpe_ratio'])

    def test_max_drawdown_is_negative_for_a_decline(self):
        dates = pd.bdate_range('2023-01-02', periods=5)
        # +10%, then -50%, then flat: max drawdown should be roughly -45%.
        returns = pd.Series([0.10, -0.50, 0.0, 0.0, 0.0], index=dates)
        metrics = compute_performance_metrics(returns, RISK_FREE_RATE)
        assert metrics['max_drawdown'] < -0.4


class TestCompareStrategies:
    def test_returns_a_result_for_each_requested_strategy(self):
        returns = _synthetic_returns(n=500, seed=5)
        results = compare_strategies(returns, RISK_FREE_RATE,
                                      lookback_days=150, rebalance_days=50)
        assert set(results.keys()) == set(STRATEGIES)
        for result in results.values():
            assert isinstance(result, BacktestResult)
            assert 'sharpe_ratio' in result.metrics


class TestOptimizationAddsValueOnFavorableData:
    def test_tangency_outperforms_equal_weight_when_one_asset_is_clearly_better(self):
        """The whole point of backtesting: on data where one asset has a
        persistently much higher risk-adjusted return, a walk-forward
        tangency strategy that re-estimates from trailing history should
        tilt toward it and realize a better Sharpe ratio out-of-sample than
        naively equal-weighting all three assets. The parameters are chosen
        with a stark, persistent separation and a long enough history that
        this holds robustly rather than by chance of a single seed."""
        n = 1500
        rng = np.random.default_rng(2024)
        dates = pd.bdate_range('2015-01-02', periods=n)
        good = rng.normal(0.0015, 0.006, n)
        bad1 = rng.normal(-0.0006, 0.022, n)
        bad2 = rng.normal(0.0001, 0.018, n)
        returns = pd.DataFrame({'GOOD': good, 'BAD1': bad1, 'BAD2': bad2}, index=dates)

        results = compare_strategies(
            returns, RISK_FREE_RATE, strategies=('tangency', 'equal_weight'),
            lookback_days=252, rebalance_days=63,
        )

        tangency_sharpe = results['tangency'].metrics['sharpe_ratio']
        equal_weight_sharpe = results['equal_weight'].metrics['sharpe_ratio']

        assert tangency_sharpe > equal_weight_sharpe

        # It should also have learned to overweight the good asset on
        # average across rebalances.
        avg_good_weight = np.mean([
            entry['weights']['GOOD'] for entry in results['tangency'].weights_history
        ])
        assert avg_good_weight > 1 / 3
