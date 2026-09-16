"""Portfolio analysis and optimization math. No Flask/DB/network dependencies,
so this module can be unit-tested in isolation with synthetic returns data."""
import numpy as np
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

    def minimum_variance_portfolio(self):
        """
        Calculate weights for minimum variance portfolio.

        Returns:
            np.array: Portfolio weights
        """
        n = len(self.mean_returns)

        ones = np.ones(n)
        try:
            # Solve cov_matrix @ x = ones directly instead of computing the
            # full inverse - faster and more numerically stable.
            raw_weights = np.linalg.solve(self.cov_matrix, ones)

            if self.long_only:
                weights = raw_weights / np.dot(ones, raw_weights)

                # Ensure no negative weights (long-only constraint)
                weights = np.maximum(weights, 0)
                weight_sum = weights.sum()
                if weight_sum == 0:
                    # Every weight clipped to zero - fall back to equal
                    # weights rather than dividing by zero into NaNs.
                    return np.ones(n) / n
                weights /= weight_sum

                return weights
            else:
                # Long-short portfolio (no constraints)
                weights = raw_weights / np.dot(ones, raw_weights)
                return weights
        except np.linalg.LinAlgError:
            # Fallback to equal weights if matrix is singular
            return np.ones(n) / n

    def tangency_portfolio(self):
        """
        Calculate weights for tangency portfolio (maximum Sharpe ratio).

        Returns:
            np.array: Portfolio weights
        """
        n = len(self.mean_returns)
        excess_returns = self.mean_returns - self.risk_free_rate

        try:
            # Solve cov_matrix @ x = excess_returns directly instead of
            # computing the full inverse - faster and more numerically stable.
            weights = np.linalg.solve(self.cov_matrix, excess_returns)

            if self.long_only:
                # Clip negative weights before normalizing, since normalizing
                # by a possibly-negative raw sum first can flip every sign
                # and select the wrong assets.
                weights = np.maximum(weights, 0)
                weight_sum = weights.sum()
                if weight_sum == 0:
                    # No asset has a positive tangency weight under the
                    # long-only constraint (e.g. every asset's expected
                    # excess return over the risk-free rate is negative) -
                    # fall back to equal weights rather than dividing by zero.
                    return np.ones(n) / n
                weights /= weight_sum

                return weights
            else:
                # Long-short portfolio (no constraints)
                # Normalize weights but allow negative values
                weights /= np.sum(np.abs(weights))  # Use absolute sum for normalization
                return weights
        except np.linalg.LinAlgError:
            # Fallback to equal weights
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
