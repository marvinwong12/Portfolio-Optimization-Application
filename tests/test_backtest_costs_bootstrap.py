"""Tests for the backtest's buy-and-hold drift, transaction costs, benchmark
wrapper, and block-bootstrap confidence intervals."""
import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.backtest import (
    run_backtest,
    compare_strategies,
    compute_performance_metrics,
    benchmark_result,
    bootstrap_sharpe_ci,
    bootstrap_sharpe_difference,
    bootstrap_summary,
    _bootstrap_indices,
    STRATEGIES,
)

RISK_FREE_RATE = 0.04


def _synthetic_returns(n=800, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-02', periods=n)
    data = rng.normal(loc=0.0005, scale=0.012, size=(n, 3))
    return pd.DataFrame(data, index=dates, columns=['AAA', 'BBB', 'CCC'])


class TestBuyAndHoldDrift:
    def test_period_growth_matches_buy_and_hold_by_hand(self):
        """With no rebalance inside the period and zero cost, an
        equal-weight buy-and-hold portfolio's value is just the average of
        each asset's own cumulative growth."""
        returns = _synthetic_returns(n=400, seed=3)
        lookback, rebalance = 100, 50
        result = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight',
                               lookback_days=lookback, rebalance_days=rebalance)

        first_period = returns.iloc[lookback:lookback + rebalance]
        expected_growth = (1 + first_period).prod().mean()
        assert result.equity_curve.iloc[rebalance - 1] == pytest.approx(expected_growth)

    def test_drift_differs_from_constant_mix(self):
        returns = _synthetic_returns(n=400, seed=3)
        drifting = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight',
                                 lookback_days=100, rebalance_days=50, drift=True)
        constant = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight',
                                 lookback_days=100, rebalance_days=50, drift=False)
        assert not np.allclose(drifting.daily_returns.values, constant.daily_returns.values)


class TestTransactionCosts:
    def test_default_is_free(self):
        returns = _synthetic_returns(n=500, seed=4)
        default = run_backtest(returns, RISK_FREE_RATE, strategy='tangency')
        explicit = run_backtest(returns, RISK_FREE_RATE, strategy='tangency', transaction_cost_bps=0.0)
        np.testing.assert_allclose(default.daily_returns.values, explicit.daily_returns.values)

    def test_negative_cost_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            run_backtest(_synthetic_returns(n=400), RISK_FREE_RATE, transaction_cost_bps=-1)

    def test_higher_costs_mean_lower_final_equity(self):
        returns = _synthetic_returns(n=800, seed=4)
        finals = [
            run_backtest(returns, RISK_FREE_RATE, strategy='tangency',
                          transaction_cost_bps=bps).equity_curve.iloc[-1]
            for bps in (0, 10, 50, 200)
        ]
        assert finals == sorted(finals, reverse=True)
        assert finals[0] > finals[-1]

    def test_only_the_initial_purchase_is_charged_when_nothing_is_rebalanced(self):
        """Equal weight held as a constant mix: every rebalance targets the
        weights already held, so the only trade is buying in from cash."""
        returns = _synthetic_returns(n=500, seed=5)
        free = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight',
                             drift=False, transaction_cost_bps=0)
        paid = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight',
                             drift=False, transaction_cost_bps=100)  # 1%

        assert paid.rebalance_turnovers[0] == pytest.approx(1.0)
        assert all(t == pytest.approx(0.0) for t in paid.rebalance_turnovers[1:])
        assert paid.daily_returns.iloc[0] == pytest.approx((1 + free.daily_returns.iloc[0]) * 0.99 - 1)
        np.testing.assert_allclose(paid.daily_returns.iloc[1:].values, free.daily_returns.iloc[1:].values)
        assert paid.equity_curve.iloc[-1] == pytest.approx(free.equity_curve.iloc[-1] * 0.99)

    def test_final_equity_is_reduced_by_exactly_the_traded_amounts(self):
        """Cost is proportional across assets, so it scales portfolio value
        without changing weights: final equity must equal the free run's
        times the product of (1 - traded * cost) over every rebalance."""
        returns = _synthetic_returns(n=800, seed=6)
        bps = 25
        free = run_backtest(returns, RISK_FREE_RATE, strategy='tangency', transaction_cost_bps=0)
        paid = run_backtest(returns, RISK_FREE_RATE, strategy='tangency', transaction_cost_bps=bps)

        np.testing.assert_allclose(paid.rebalance_turnovers, free.rebalance_turnovers)
        drag = np.prod([1 - t * bps / 10_000 for t in paid.rebalance_turnovers])
        assert paid.equity_curve.iloc[-1] == pytest.approx(free.equity_curve.iloc[-1] * drag)

    def test_turnover_is_reported(self):
        returns = _synthetic_returns(n=800, seed=6)
        constant_mix = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight', drift=False)
        buy_and_hold = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight', drift=True)
        assert constant_mix.metrics['annual_turnover'] == pytest.approx(0.0)
        assert buy_and_hold.metrics['annual_turnover'] > 0  # drift has to be traded back
        assert run_backtest(returns, RISK_FREE_RATE, strategy='tangency').metrics['annual_turnover'] > 0


class TestBenchmarkResult:
    def test_aligns_to_the_strategy_dates_and_has_no_turnover(self):
        returns = _synthetic_returns(n=400, seed=7)
        strategy = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight', lookback_days=100)
        benchmark_returns = pd.Series(
            np.random.default_rng(1).normal(0.0004, 0.01, len(returns)), index=returns.index
        )

        bench = benchmark_result('SPY (Buy & Hold)', benchmark_returns,
                                  strategy.daily_returns.index, RISK_FREE_RATE)
        assert list(bench.daily_returns.index) == list(strategy.daily_returns.index)
        assert bench.metrics['annual_turnover'] == 0.0
        assert bench.strategy == 'SPY (Buy & Hold)'

    def test_raises_when_the_benchmark_is_missing_dates(self):
        returns = _synthetic_returns(n=400, seed=7)
        strategy = run_backtest(returns, RISK_FREE_RATE, strategy='equal_weight', lookback_days=100)
        short = pd.Series(0.0, index=returns.index[:-10])
        with pytest.raises(ValueError, match="missing returns"):
            benchmark_result('SPY', short, strategy.daily_returns.index, RISK_FREE_RATE)


class TestBootstrapIndices:
    def test_shape_and_range(self):
        indices = _bootstrap_indices(n_days=100, n_boot=50, block_size=21, seed=1)
        assert indices.shape == (50, 100)
        assert indices.min() >= 0 and indices.max() < 100

    def test_same_seed_is_reproducible_and_different_seeds_differ(self):
        first = _bootstrap_indices(100, 20, 10, seed=1)
        assert np.array_equal(first, _bootstrap_indices(100, 20, 10, seed=1))
        assert not np.array_equal(first, _bootstrap_indices(100, 20, 10, seed=2))

    def test_resamples_contiguous_blocks(self):
        indices = _bootstrap_indices(n_days=200, n_boot=5, block_size=10, seed=3)
        steps = np.diff(indices[:, :10], axis=1) % 200
        assert np.all(steps == 1)  # consecutive dates inside a block


class TestBootstrapSharpeCI:
    def _series(self, n=800, mean=0.0006, vol=0.01, seed=1):
        return pd.Series(np.random.default_rng(seed).normal(mean, vol, n))

    def test_point_estimate_matches_performance_metrics(self):
        series = self._series()
        ci = bootstrap_sharpe_ci(series, RISK_FREE_RATE)
        expected = compute_performance_metrics(series, RISK_FREE_RATE)['sharpe_ratio']
        assert ci['sharpe'] == pytest.approx(expected)

    def test_interval_brackets_the_estimate(self):
        ci = bootstrap_sharpe_ci(self._series(), RISK_FREE_RATE)
        assert ci['ci_low'] < ci['sharpe'] < ci['ci_high']

    def test_is_reproducible_for_a_given_seed(self):
        series = self._series()
        assert (bootstrap_sharpe_ci(series, RISK_FREE_RATE, seed=5)
                == bootstrap_sharpe_ci(series, RISK_FREE_RATE, seed=5))

    def test_more_data_gives_a_tighter_interval(self):
        short = bootstrap_sharpe_ci(self._series(n=300), RISK_FREE_RATE)
        long = bootstrap_sharpe_ci(self._series(n=3000), RISK_FREE_RATE)
        assert (long['ci_high'] - long['ci_low']) < (short['ci_high'] - short['ci_low'])

    def test_higher_confidence_gives_a_wider_interval(self):
        series = self._series()
        narrow = bootstrap_sharpe_ci(series, RISK_FREE_RATE, confidence=0.80)
        wide = bootstrap_sharpe_ci(series, RISK_FREE_RATE, confidence=0.99)
        assert (wide['ci_high'] - wide['ci_low']) > (narrow['ci_high'] - narrow['ci_low'])

    def test_blocks_widen_the_interval_for_autocorrelated_returns(self):
        """The reason to resample blocks: with autocorrelated returns an
        i.i.d. (block size 1) bootstrap understates uncertainty."""
        rng = np.random.default_rng(11)
        shocks = rng.normal(0.0004, 0.01, 2000)
        values = np.zeros(2000)
        for t in range(1, 2000):
            values[t] = 0.5 * values[t - 1] + shocks[t]
        series = pd.Series(values)

        iid = bootstrap_sharpe_ci(series, RISK_FREE_RATE, block_size=1)
        blocked = bootstrap_sharpe_ci(series, RISK_FREE_RATE, block_size=21)
        assert (blocked['ci_high'] - blocked['ci_low']) > (iid['ci_high'] - iid['ci_low'])

    def test_interval_covers_the_true_sharpe_about_as_often_as_claimed(self):
        mean, vol, n_days = 0.0006, 0.01, 756
        big = np.random.default_rng(999).normal(mean, vol, 2_000_000)
        true_annual_return = np.exp(np.log1p(big).mean() * 252) - 1
        true_sharpe = (true_annual_return - RISK_FREE_RATE) / (vol * np.sqrt(252))

        covered = 0
        trials = 150
        for seed in range(trials):
            series = pd.Series(np.random.default_rng(seed).normal(mean, vol, n_days))
            ci = bootstrap_sharpe_ci(series, RISK_FREE_RATE, n_boot=400, seed=seed)
            covered += ci['ci_low'] <= true_sharpe <= ci['ci_high']
        assert covered / trials >= 0.85  # nominal 95%; bootstrap CIs run slightly short


class TestBootstrapSharpeDifference:
    def _pair(self, n=800, seed=1):
        rng = np.random.default_rng(seed)
        common = rng.normal(0.0005, 0.01, n)
        return (pd.Series(common + rng.normal(0, 0.003, n)),
                pd.Series(common + rng.normal(0, 0.003, n)))

    def test_identical_series_have_no_difference(self):
        series, _ = self._pair()
        result = bootstrap_sharpe_difference(series, series, RISK_FREE_RATE)
        assert result['difference'] == pytest.approx(0.0)
        assert result['ci_low'] == pytest.approx(0.0)
        assert result['ci_high'] == pytest.approx(0.0)
        assert result['significant'] is False

    def test_swapping_the_series_flips_the_sign_exactly(self):
        a, b = self._pair()
        ab = bootstrap_sharpe_difference(a, b, RISK_FREE_RATE)
        ba = bootstrap_sharpe_difference(b, a, RISK_FREE_RATE)
        assert ab['difference'] == pytest.approx(-ba['difference'])
        assert ab['ci_low'] == pytest.approx(-ba['ci_high'])
        assert ab['ci_high'] == pytest.approx(-ba['ci_low'])

    def test_a_clearly_better_series_is_significant(self):
        a, b = self._pair()
        result = bootstrap_sharpe_difference(a + 0.001, b, RISK_FREE_RATE)  # +0.1% per day
        assert result['significant'] is True
        assert result['ci_low'] > 0
        assert result['prob_a_better'] > 0.99

    def test_noise_level_differences_are_not_significant(self):
        a, b = self._pair()
        result = bootstrap_sharpe_difference(a, b, RISK_FREE_RATE)
        assert result['significant'] is False
        assert result['ci_low'] < 0 < result['ci_high']

    def test_paired_interval_is_tighter_than_an_individual_one(self):
        a, b = self._pair()
        diff = bootstrap_sharpe_difference(a, b, RISK_FREE_RATE)
        ci_a = bootstrap_sharpe_ci(a, RISK_FREE_RATE)
        assert (diff['ci_high'] - diff['ci_low']) < (ci_a['ci_high'] - ci_a['ci_low'])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            bootstrap_sharpe_difference(pd.Series([0.01] * 10), pd.Series([0.01] * 11), RISK_FREE_RATE)


class TestBootstrapSummary:
    def test_structure_and_baseline_exclusion(self):
        returns = _synthetic_returns(n=700, seed=8)
        results = compare_strategies(returns, RISK_FREE_RATE, lookback_days=200, rebalance_days=50)
        summary = bootstrap_summary(results, ['equal_weight'], RISK_FREE_RATE, n_boot=300)

        assert set(summary['sharpe_ci']) == set(STRATEGIES)
        assert set(summary['versus']['equal_weight']) == {'tangency', 'minimum_variance'}
        for entry in summary['versus']['equal_weight'].values():
            assert {'difference', 'ci_low', 'ci_high', 'prob_a_better', 'significant'} <= set(entry)
