"""Portfolio analysis and optimization math. No Flask/DB/network dependencies,
so this module can be unit-tested in isolation with synthetic returns data."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


class PortfolioAnalyzer:
    """
    A class to perform portfolio analysis and optimization.

    Methods:
        calculate_portfolio_metrics: Calculate return, volatility, and Sharpe ratio
        calculate_treynor_ratio: Calculate Treynor ratio (risk-adjusted return)
        calculate_beta: Calculate portfolio beta relative to market
        calculate_jensen_alpha: Calculate Jensen's alpha (excess return)
        minimum_variance_portfolio: Calculate minimum variance portfolio weights
        tangency_portfolio: Calculate tangency portfolio (max Sharpe ratio) weights
        monte_carlo_simulation: Run Monte Carlo simulation for portfolio optimization
        efficient_frontier: Solve the exact efficient frontier via constrained optimization
        calculate_tail_risk: Historical VaR, CVaR and max drawdown for a weight vector
        risk_contributions: Each asset's share of total portfolio variance
    """

    def __init__(self, returns_data, risk_free_rate, long_only=True):
        """
        Initialize the PortfolioAnalyzer.

        Args:
            returns_data (pd.DataFrame): Historical returns data
            risk_free_rate (float): Risk-free rate for calculations
            long_only (bool): Whether to enforce long-only constraints
        """
        self.returns = returns_data
        self.risk_free_rate = risk_free_rate
        self.cov_matrix = LedoitWolf().fit(self.returns).covariance_ * 252  # Annualized
        self.mean_returns = self.returns.mean() * 252  # Annualized
        self.long_only = long_only

    def calculate_portfolio_metrics(self, weights):
        """
        Calculate portfolio performance metrics.

        Args:
            weights (np.array): Portfolio weights

        Returns:
            dict: Dictionary containing return, volatility, and Sharpe ratio
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std != 0 else 0

        return {
            'return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio,
            'weights': weights
        }

    def calculate_treynor_ratio(self, weights, market_returns):
        """
        Calculate Treynor ratio (return per unit of systematic risk).

        Args:
            weights (np.array): Portfolio weights
            market_returns (pd.Series): Market returns data

        Returns:
            float: Treynor ratio or NaN if calculation fails
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        beta = self.calculate_beta(weights, market_returns)
        return (portfolio_return - self.risk_free_rate) / beta if beta != 0 and not np.isnan(beta) else np.nan

    def calculate_beta(self, weights, market_returns):
        """
        Calculate portfolio beta relative to market.

        Args:
            weights (np.array): Portfolio weights
            market_returns (pd.Series): Market returns data

        Returns:
            float: Portfolio beta or NaN if calculation fails
        """
        # Ensure market_returns is aligned with portfolio returns
        portfolio_returns = np.dot(self.returns, weights)

        # Align market returns with portfolio returns dates
        aligned_market_returns = market_returns.reindex(self.returns.index).dropna()
        aligned_portfolio_returns = portfolio_returns[self.returns.index.isin(aligned_market_returns.index)]

        if len(aligned_market_returns) < 2 or len(aligned_portfolio_returns) < 2:
            return np.nan

        # Calculate covariance and variance
        covariance = np.cov(aligned_portfolio_returns, aligned_market_returns)[0, 1]
        market_variance = np.var(aligned_market_returns)
        return covariance / market_variance if market_variance != 0 else np.nan

    def calculate_jensen_alpha(self, weights, market_returns):
        """
        Calculate Jensen's alpha (excess return over expected return).

        Args:
            weights (np.array): Portfolio weights
            market_returns (pd.Series): Market returns data

        Returns:
            float: Jensen's alpha or NaN if calculation fails
        """
        portfolio_return = np.sum(self.mean_returns * weights)
        beta = self.calculate_beta(weights, market_returns)

        if np.isnan(beta):
            return np.nan

        # Use annualized market return
        aligned_market_returns = market_returns.reindex(self.returns.index).dropna()
        if len(aligned_market_returns) == 0:
            return np.nan

        market_return = aligned_market_returns.mean() * 252
        expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
        return portfolio_return - expected_return

    def _solve_long_only(self, objective):
        """
        Minimize `objective(weights)` subject to weights summing to 1 and
        0 <= w <= 1, via SLSQP. Returns None if the solver fails to
        converge so callers can fall back explicitly.

        This is the true long-only optimum. The closed-form (unconstrained)
        solution followed by clipping negatives and renormalizing is only
        an approximation: once any weight is clipped, the remaining
        weights are no longer optimal for the reduced problem.
        """
        n = len(self.mean_returns)
        result = minimize(
            objective, np.ones(n) / n, method='SLSQP',
            bounds=tuple((0.0, 1.0) for _ in range(n)),
            constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},),
            options={'maxiter': 500, 'ftol': 1e-12},
        )
        if not result.success:
            return None
        weights = np.clip(result.x, 0.0, None)
        return weights / weights.sum()

    def portfolio_daily_returns(self, weights):
        """
        Daily returns of a constant-mix portfolio over the analyzer's window
        (the same weights applied to every day, i.e. implicitly rebalanced
        daily).

        Returns:
            pd.Series: Indexed like self.returns
        """
        return pd.Series(self.returns.values @ np.asarray(weights), index=self.returns.index)

    def calculate_tail_risk(self, weights, confidence=0.95):
        """
        Historical tail-risk metrics for a portfolio over the analyzer's
        window. Sign convention matches the single-stock page: losses are
        negative numbers (e.g. a VaR of -0.021 is a 2.1% one-day loss).

        Args:
            weights (np.array): Portfolio weights
            confidence (float): VaR/CVaR confidence level

        Returns:
            dict: {
                'var': the (1-confidence) quantile of daily returns - the
                    daily loss exceeded on only (1-confidence) of days,
                'cvar': expected shortfall - the average daily return on
                    those worst days (always <= var),
                'max_drawdown': worst peak-to-trough decline of the
                    cumulative-return curve,
                'confidence': confidence,
            }
        """
        daily = self.portfolio_daily_returns(weights)
        cutoff = float(np.percentile(daily, (1 - confidence) * 100))
        tail = daily[daily <= cutoff]

        equity_curve = (1 + daily).cumprod()
        drawdown = equity_curve / equity_curve.cummax() - 1

        return {
            'var': cutoff,
            'cvar': float(tail.mean()),
            'max_drawdown': float(drawdown.min()),
            'confidence': confidence,
        }

    def risk_contributions(self, weights):
        """
        Each asset's fractional contribution to total portfolio variance:
        RC_i = w_i * (Sigma w)_i / (w' Sigma w). Contributions sum to 1, and
        differ from weights whenever assets have different volatilities or
        correlations - a 25% weight in a volatile, highly correlated stock
        can be far more than 25% of the risk. (In a long-short portfolio an
        asset that hedges the rest can have a negative contribution.)

        Args:
            weights (np.array): Portfolio weights

        Returns:
            np.array: Fractional risk contributions, same order as weights
                (all zeros if the portfolio has zero variance)
        """
        weights = np.asarray(weights, dtype=float)
        portfolio_variance = float(weights @ self.cov_matrix @ weights)
        if portfolio_variance <= 0:
            return np.zeros(len(weights))
        return weights * (self.cov_matrix @ weights) / portfolio_variance

    def minimum_variance_portfolio(self):
        """
        Calculate weights for the minimum variance portfolio.

        Long-only: solved as a bounded optimization (see _solve_long_only).
        Long-short: closed-form solution, w = S^-1 1 / (1' S^-1 1).

        Returns:
            np.array: Portfolio weights
        """
        n = len(self.mean_returns)

        if self.long_only:
            cov_matrix = self.cov_matrix
            weights = self._solve_long_only(lambda w: float(w @ cov_matrix @ w))
            # Fallback to equal weights if the solver fails to converge
            return weights if weights is not None else np.ones(n) / n

        ones = np.ones(n)
        try:
            # Solve cov_matrix @ x = ones directly instead of computing the
            # full inverse - faster and more numerically stable.
            raw_weights = np.linalg.solve(self.cov_matrix, ones)
            return raw_weights / np.dot(ones, raw_weights)
        except np.linalg.LinAlgError:
            # Fallback to equal weights if matrix is singular
            return np.ones(n) / n

    def tangency_portfolio(self):
        """
        Calculate weights for the tangency portfolio (maximum Sharpe ratio).

        Long-only: solved as a bounded optimization that maximizes the
        Sharpe ratio directly (see _solve_long_only). If no asset's
        expected return exceeds the risk-free rate there is no meaningful
        max-Sharpe portfolio, so this falls back to equal weights.
        Long-short: closed-form solution, normalized by the absolute
        weight sum.

        Returns:
            np.array: Portfolio weights
        """
        n = len(self.mean_returns)
        excess_returns = (self.mean_returns - self.risk_free_rate).values

        if self.long_only:
            if np.all(excess_returns <= 0):
                return np.ones(n) / n

            cov_matrix = self.cov_matrix

            def negative_sharpe(w):
                volatility = np.sqrt(max(float(w @ cov_matrix @ w), 1e-16))
                return -float(w @ excess_returns) / volatility

            weights = self._solve_long_only(negative_sharpe)
            return weights if weights is not None else np.ones(n) / n

        try:
            # Solve cov_matrix @ x = excess_returns directly instead of
            # computing the full inverse - faster and more numerically stable.
            weights = np.linalg.solve(self.cov_matrix, excess_returns)
            return weights / np.sum(np.abs(weights))
        except np.linalg.LinAlgError:
            return np.ones(n) / n

    def monte_carlo_simulation(self, num_portfolios=10000):
        """
        Run Monte Carlo simulation to generate random portfolios.

        Args:
            num_portfolios (int): Number of portfolios to simulate

        Returns:
            tuple: Results array and weights record
        """
        n = len(self.mean_returns)

        # Generate all random portfolios at once and normalize row-wise,
        # instead of looping in Python - this is the hot path for large
        # num_portfolios and vectorizing it is an order of magnitude faster.
        if self.long_only:
            # Long-only: weights between 0 and 1
            weights_matrix = np.random.random((num_portfolios, n))
            weights_matrix /= weights_matrix.sum(axis=1, keepdims=True)
        else:
            # Long-short: weights between -1 and 1
            weights_matrix = np.random.uniform(-1, 1, (num_portfolios, n))
            weights_matrix /= np.abs(weights_matrix).sum(axis=1, keepdims=True)

        # Portfolio return per simulation: (num_portfolios, n) @ (n,) -> (num_portfolios,)
        portfolio_returns = weights_matrix @ self.mean_returns.values

        # Portfolio volatility per simulation via batched quadratic form
        portfolio_stds = np.sqrt(
            np.einsum('ij,jk,ik->i', weights_matrix, self.cov_matrix, weights_matrix)
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            sharpe_ratios = np.where(
                portfolio_stds != 0,
                (portfolio_returns - self.risk_free_rate) / portfolio_stds,
                0
            )

        results = np.vstack([portfolio_returns, portfolio_stds, sharpe_ratios])
        weights_record = list(weights_matrix)

        return results, weights_record

    def efficient_frontier(self, num_points=50, max_weight=None):
        """
        Solve the exact efficient frontier: for a range of target returns
        from the minimum-variance point up to the highest-returning asset,
        minimize portfolio variance subject to that target - via
        constrained optimization (SLSQP), not the Monte Carlo cloud used
        elsewhere for visualization. Monte Carlo sampling only ever
        approximates the frontier from below/inside; this actually solves
        for it, and additionally supports a per-asset weight cap that the
        closed-form tangency/min-variance solutions can't express at all.

        Returns are only swept from the minimum-variance point upward,
        since below it a higher return is achievable at the same risk -
        that lower region is dominated and excluded from the efficient
        frontier by definition.

        Args:
            num_points (int): Number of points along the frontier
            max_weight (float): Optional per-asset weight cap (e.g. 0.4 for
                a 40% max allocation to any single asset). Only meaningful
                when long_only is True - ignored otherwise, since a
                long-short portfolio's per-asset exposure isn't bounded by
                the same [0, 1] logic.

        Returns:
            dict: {
                'returns': np.array of target returns actually achieved,
                'volatilities': np.array of minimized volatilities (same
                    order as 'returns'),
                'weights': list of np.array weight vectors, one per point,
            }
            Points where the optimizer failed to converge (e.g. an
            infeasible max_weight) are silently skipped rather than
            included with garbage values.

        Raises:
            ValueError: If max_weight is too restrictive to let weights
                sum to 1 at all (max_weight * n_assets < 1).
        """
        n = len(self.mean_returns)
        mean_returns = self.mean_returns.values
        cov_matrix = self.cov_matrix

        if self.long_only:
            upper = max_weight if max_weight is not None else 1.0
            if upper * n < 1.0:
                raise ValueError(
                    f"max_weight={max_weight} is infeasible for {n} assets: "
                    f"weights can't sum to 1 if every asset is capped below 1/{n}."
                )
            bounds = tuple((0.0, upper) for _ in range(n))
        else:
            bounds = tuple((-1.0, 1.0) for _ in range(n))

        def portfolio_variance(weights):
            return float(weights.T @ cov_matrix @ weights)

        x0 = np.ones(n) / n

        # The efficient frontier proper only covers the non-dominated upper
        # half of the return range: below the minimum-variance point, a
        # higher return is achievable at the SAME risk, so that region is
        # dominated and excluded by definition. Anchor the sweep's lower
        # bound at the minimum-variance return under these exact bounds
        # (not the closed-form unconstrained solution, which ignores
        # max_weight) rather than the worst asset's return.
        min_var_result = minimize(
            portfolio_variance, x0, method='SLSQP',
            bounds=bounds, constraints=({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},),
            options={'maxiter': 500, 'ftol': 1e-10},
        )
        lower_bound_return = (
            float(min_var_result.x @ mean_returns) if min_var_result.success
            else mean_returns.min()
        )

        target_returns = np.linspace(lower_bound_return, mean_returns.max(), num_points)

        frontier_returns = []
        frontier_volatilities = []
        frontier_weights = []

        # Warm-start the sweep from the minimum-variance solution (or equal
        # weights if that failed to converge), then from each iteration's
        # own result - consecutive target returns have similar optimal
        # weights, so this converges faster than restarting from scratch.
        x0 = min_var_result.x if min_var_result.success else np.ones(n) / n
        for target in target_returns:
            constraints = (
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w, target=target: w @ mean_returns - target},
            )
            result = minimize(
                portfolio_variance, x0, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10},
            )
            if result.success:
                frontier_returns.append(target)
                frontier_volatilities.append(np.sqrt(portfolio_variance(result.x)))
                frontier_weights.append(result.x)
                x0 = result.x

        return {
            'returns': np.array(frontier_returns),
            'volatilities': np.array(frontier_volatilities),
            'weights': frontier_weights,
        }
