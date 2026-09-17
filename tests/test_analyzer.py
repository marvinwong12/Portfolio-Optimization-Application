import numpy as np
import pandas as pd
import pytest

from portfolio_optimizer.analyzer import PortfolioAnalyzer


RISK_FREE_RATE = 0.04


@pytest.fixture
def analyzer_long_only(synthetic_returns):
    return PortfolioAnalyzer(synthetic_returns, RISK_FREE_RATE, long_only=True)


@pytest.fixture
def analyzer_long_short(synthetic_returns):
    return PortfolioAnalyzer(synthetic_returns, RISK_FREE_RATE, long_only=False)


class TestMinimumVariancePortfolio:
    def test_long_only_weights_sum_to_one(self, analyzer_long_only):
        weights = analyzer_long_only.minimum_variance_portfolio()
        assert weights.sum() == pytest.approx(1.0)

    def test_long_only_weights_are_non_negative(self, analyzer_long_only):
        weights = analyzer_long_only.minimum_variance_portfolio()
        assert np.all(weights >= -1e-9)

    def test_long_short_weights_sum_to_one(self, analyzer_long_short):
        weights = analyzer_long_short.minimum_variance_portfolio()
        assert weights.sum() == pytest.approx(1.0)

    def test_falls_back_to_equal_weights_on_singular_matrix(self, synthetic_returns):
        analyzer = PortfolioAnalyzer(synthetic_returns, RISK_FREE_RATE, long_only=True)
        # A zero covariance matrix is singular - linalg.solve should raise
        # LinAlgError and the method should fall back to equal weights.
        analyzer.cov_matrix = np.zeros((4, 4))
        weights = analyzer.minimum_variance_portfolio()
        assert weights == pytest.approx(np.ones(4) / 4)


class TestTangencyPortfolio:
    def test_long_only_weights_sum_to_one(self, analyzer_long_only):
        weights = analyzer_long_only.tangency_portfolio()
        assert weights.sum() == pytest.approx(1.0)

    def test_long_only_weights_are_non_negative(self, analyzer_long_only):
        weights = analyzer_long_only.tangency_portfolio()
        assert np.all(weights >= -1e-9)

    def test_long_short_weights_abs_sum_to_one(self, analyzer_long_short):
        weights = analyzer_long_short.tangency_portfolio()
        assert np.abs(weights).sum() == pytest.approx(1.0)

    def test_falls_back_to_equal_weights_when_all_excess_returns_negative(self):
        """Regression test: if every asset's expected return is below the
        risk-free rate, all long-only weights clip to zero and dividing by
        that zero sum used to produce NaN weights instead of a sane fallback."""
        dates = pd.bdate_range('2023-01-02', periods=250)
        rng = np.random.default_rng(99)
        # Both assets have a slightly negative mean return - with a high
        # risk-free rate, their excess returns will be negative too.
        returns = pd.DataFrame({
            'A': rng.normal(-0.0005, 0.01, 250),
            'B': rng.normal(-0.0003, 0.01, 250),
        }, index=dates)

        analyzer = PortfolioAnalyzer(returns, risk_free_rate=0.5, long_only=True)
        weights = analyzer.tangency_portfolio()

        assert not np.any(np.isnan(weights))
        assert weights == pytest.approx(np.ones(2) / 2)

    def test_negative_raw_weight_sum_does_not_flip_asset_selection(self):
        """Regression test for a bug where normalizing by the raw (possibly
        negative) weight sum before clipping negatives could flip every
        sign and select the wrong assets. Two assets, engineered so the
        unconstrained tangency solution's raw weights sum to a negative
        number, used to make this fail before the fix (clip-then-normalize)."""
        dates = pd.bdate_range('2023-01-02', periods=250)
        rng = np.random.default_rng(123)
        # Asset A: clearly better risk-adjusted return than asset B.
        returns_a = rng.normal(0.0015, 0.01, 250)
        returns_b = rng.normal(-0.0005, 0.01, 250)
        returns = pd.DataFrame({'A': returns_a, 'B': returns_b}, index=dates)

        analyzer = PortfolioAnalyzer(returns, RISK_FREE_RATE, long_only=True)
        weights = analyzer.tangency_portfolio()

        assert np.all(weights >= -1e-9)
        assert weights.sum() == pytest.approx(1.0)
        # The higher-Sharpe asset should not be excluded in favor of the
        # clearly worse one.
        symbol_weight = dict(zip(returns.columns, weights))
        assert symbol_weight['A'] > symbol_weight['B']


class TestPortfolioMetrics:
    def test_equal_weight_metrics_are_finite(self, analyzer_long_only):
        n = len(analyzer_long_only.mean_returns)
        weights = np.ones(n) / n
        metrics = analyzer_long_only.calculate_portfolio_metrics(weights)
        assert np.isfinite(metrics['return'])
        assert metrics['volatility'] > 0
        assert np.isfinite(metrics['sharpe_ratio'])

    def test_sharpe_uses_risk_free_rate_directly(self, analyzer_long_only):
        """Regression test: risk_free_rate must be used as-is (annual),
        not deannualized, since portfolio_return here is already annualized."""
        n = len(analyzer_long_only.mean_returns)
        weights = np.ones(n) / n
        metrics = analyzer_long_only.calculate_portfolio_metrics(weights)
        expected_sharpe = (metrics['return'] - RISK_FREE_RATE) / metrics['volatility']
        assert metrics['sharpe_ratio'] == pytest.approx(expected_sharpe)


class TestBetaAndRelatedMetrics:
    def test_beta_of_market_matched_returns_is_approximately_one(self):
        dates = pd.bdate_range('2023-01-02', periods=200)
        rng = np.random.default_rng(7)
        market = pd.Series(rng.normal(0.0005, 0.01, 200), index=dates)
        # Asset that exactly tracks the market has beta ~= 1. Not exactly 1:
        # calculate_beta mixes np.cov (ddof=1) with np.var (ddof=0), so an
        # identical series comes out as n/(n-1) rather than 1.0 - a pre-existing
        # minor inconsistency, not something this test suite fixes.
        returns = pd.DataFrame({'ONLY': market.values}, index=dates)

        analyzer = PortfolioAnalyzer(returns, RISK_FREE_RATE, long_only=True)
        beta = analyzer.calculate_beta(np.array([1.0]), market)
        assert beta == pytest.approx(1.0, abs=0.01)

    def test_beta_is_nan_with_insufficient_overlap(self, synthetic_returns):
        analyzer = PortfolioAnalyzer(synthetic_returns, RISK_FREE_RATE, long_only=True)
        # Market series that shares only one overlapping date with returns.
        market = pd.Series([0.01], index=[synthetic_returns.index[0]])
        beta = analyzer.calculate_beta(np.ones(4) / 4, market)
        assert np.isnan(beta)

    def test_treynor_ratio_is_nan_when_beta_is_zero_or_nan(self, synthetic_returns):
        analyzer = PortfolioAnalyzer(synthetic_returns, RISK_FREE_RATE, long_only=True)
        market = pd.Series([0.01], index=[synthetic_returns.index[0]])
        treynor = analyzer.calculate_treynor_ratio(np.ones(4) / 4, market)
        assert np.isnan(treynor)


class TestMonteCarloSimulation:
    def test_output_shapes(self, analyzer_long_only):
        n_assets = len(analyzer_long_only.mean_returns)
        results, weights_record = analyzer_long_only.monte_carlo_simulation(500)
        assert results.shape == (3, 500)
        assert len(weights_record) == 500
        assert weights_record[0].shape == (n_assets,)

    def test_long_only_simulated_weights_are_valid(self, analyzer_long_only):
        _, weights_record = analyzer_long_only.monte_carlo_simulation(200)
        for weights in weights_record:
            assert np.all(weights >= 0)
            assert weights.sum() == pytest.approx(1.0)

    def test_long_short_simulated_weights_are_valid(self, analyzer_long_short):
        _, weights_record = analyzer_long_short.monte_carlo_simulation(200)
        for weights in weights_record:
            assert np.abs(weights).sum() == pytest.approx(1.0)

    def test_sharpe_ratio_matches_return_and_volatility(self, analyzer_long_only):
        results, _ = analyzer_long_only.monte_carlo_simulation(200)
        returns, volatilities, sharpe_ratios = results
        expected = (returns - RISK_FREE_RATE) / volatilities
        assert sharpe_ratios == pytest.approx(expected)


class TestEfficientFrontier:
    def test_weights_sum_to_one_and_are_non_negative_for_long_only(self, analyzer_long_only):
        frontier = analyzer_long_only.efficient_frontier(num_points=15)
        assert len(frontier['weights']) > 0
        for weights in frontier['weights']:
            assert weights.sum() == pytest.approx(1.0, abs=1e-6)
            assert np.all(weights >= -1e-8)

    def test_volatility_is_non_decreasing_with_target_return(self, analyzer_long_only):
        """The efficient frontier is, by definition, the minimum-variance
        boundary: moving to a higher target return should never require
        less risk than a lower one (that region would be dominated)."""
        frontier = analyzer_long_only.efficient_frontier(num_points=20)
        volatilities = frontier['volatilities']
        # Allow tiny numerical slack rather than requiring a strictly
        # monotonic sequence from a numerical optimizer.
        assert np.all(np.diff(volatilities) >= -1e-6)

    def test_frontier_volatility_is_at_least_the_global_minimum_variance(self, analyzer_long_only):
        """Every point on the frontier is variance-minimized for its target
        return, so none of them can beat the unconstrained global minimum
        variance portfolio."""
        min_var_weights = analyzer_long_only.minimum_variance_portfolio()
        min_var_metrics = analyzer_long_only.calculate_portfolio_metrics(min_var_weights)

        frontier = analyzer_long_only.efficient_frontier(num_points=20)
        assert np.all(frontier['volatilities'] >= min_var_metrics['volatility'] - 1e-6)

    def test_returns_are_within_asset_return_range(self, analyzer_long_only):
        frontier = analyzer_long_only.efficient_frontier(num_points=10)
        assert frontier['returns'].min() >= analyzer_long_only.mean_returns.min() - 1e-8
        assert frontier['returns'].max() <= analyzer_long_only.mean_returns.max() + 1e-8

    def test_max_weight_constraint_is_respected(self, analyzer_long_only):
        frontier = analyzer_long_only.efficient_frontier(num_points=15, max_weight=0.4)
        for weights in frontier['weights']:
            assert np.all(weights <= 0.4 + 1e-6)
            assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_infeasible_max_weight_raises(self, analyzer_long_only):
        # 4 assets, so a 20% cap per asset can sum to at most 80% - infeasible.
        with pytest.raises(ValueError, match='infeasible'):
            analyzer_long_only.efficient_frontier(max_weight=0.2)

    def test_long_short_weights_sum_to_one(self, analyzer_long_short):
        frontier = analyzer_long_short.efficient_frontier(num_points=10)
        assert len(frontier['weights']) > 0
        for weights in frontier['weights']:
            assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_tangency_portfolio_lies_on_or_above_the_frontier(self, analyzer_long_only):
        """Sanity check tying the closed-form tangency solution to the
        numerically-optimized frontier: for the tangency portfolio's own
        return level, the frontier's minimized volatility should not be
        higher than the tangency portfolio's actual volatility (the
        tangency portfolio must itself be efficient)."""
        tangency_weights = analyzer_long_only.tangency_portfolio()
        tangency_metrics = analyzer_long_only.calculate_portfolio_metrics(tangency_weights)

        frontier = analyzer_long_only.efficient_frontier(num_points=40)
        closest_idx = np.argmin(np.abs(frontier['returns'] - tangency_metrics['return']))
        assert frontier['volatilities'][closest_idx] <= tangency_metrics['volatility'] + 1e-3
