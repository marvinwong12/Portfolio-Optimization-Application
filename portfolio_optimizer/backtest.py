"""Walk-forward backtesting for the portfolio optimization strategies.

The rest of the app computes optimized weights from a single static lookback
window and presents them as if they were "the" answer, with no check on
whether they'd have actually performed well going forward. This module
answers that question: at each rebalance date, weights are computed using
only returns strictly *before* that date (no lookahead), held fixed through
the next holding period, and the realized (out-of-sample) return over that
period is recorded. Repeating this across the whole history gives a
realistic equity curve that can be compared against a naive benchmark
(equal weight) to see whether the optimization is actually adding value.

No Flask/DB/network dependencies, so this is unit-testable with synthetic
returns data.
"""
import numpy as np
import pandas as pd

from .analyzer import PortfolioAnalyzer

STRATEGIES = ('tangency', 'minimum_variance', 'equal_weight')


class BacktestResult:
    """Container for one strategy's walk-forward backtest output."""

    def __init__(self, strategy, daily_returns, weights_history, risk_free_rate):
        self.strategy = strategy
        self.daily_returns = daily_returns  # pd.Series, out-of-sample daily returns
        self.equity_curve = (1 + daily_returns).cumprod()
        self.weights_history = weights_history  # list of {'date', 'weights': {symbol: weight}}
        self.risk_free_rate = risk_free_rate
        self.metrics = compute_performance_metrics(daily_returns, risk_free_rate)

    def __repr__(self):
        return f'<BacktestResult {self.strategy} sharpe={self.metrics["sharpe_ratio"]:.2f}>'


def compute_performance_metrics(daily_returns, risk_free_rate):
    """
    Compute realized performance metrics from a series of daily returns.

    Args:
        daily_returns (pd.Series): Realized daily returns
        risk_free_rate (float): Annual risk-free rate as a decimal

    Returns:
        dict: total_return, annualized_return, annualized_volatility,
            sharpe_ratio, max_drawdown
    """
    if len(daily_returns) == 0:
        return {
            'total_return': np.nan,
            'annualized_return': np.nan,
            'annualized_volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
        }

    equity_curve = (1 + daily_returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1

    n_days = len(daily_returns)
    # Geometric annualization from realized total return, rather than
    # arithmetic mean * 252, since this is meant to reflect what actually
    # would have happened to invested capital.
    annualized_return = equity_curve.iloc[-1] ** (252 / n_days) - 1
    annualized_volatility = daily_returns.std() * np.sqrt(252)

    sharpe_ratio = (
        (annualized_return - risk_free_rate) / annualized_volatility
        if annualized_volatility != 0 else 0
    )

    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
    }


def _weights_for_strategy(analyzer, strategy):
    if strategy == 'tangency':
        return analyzer.tangency_portfolio()
    elif strategy == 'minimum_variance':
        return analyzer.minimum_variance_portfolio()
    elif strategy == 'equal_weight':
        n = len(analyzer.mean_returns)
        return np.ones(n) / n
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {STRATEGIES}.")


def run_backtest(returns, risk_free_rate, strategy='tangency', long_only=True,
                  lookback_days=252, rebalance_days=63):
    """
    Run a walk-forward backtest for a single strategy.

    At each rebalance date, weights are estimated from the `lookback_days`
    of returns immediately preceding it (never including the holding period
    itself), then applied as constant target weights (implicitly rebalanced
    daily, not allowed to drift) while realized returns accumulate over the
    next `rebalance_days`. This repeats until the data is exhausted.

    Args:
        returns (pd.DataFrame): Daily returns, one column per asset, sorted
            ascending by date.
        risk_free_rate (float): Annual risk-free rate as a decimal.
        strategy (str): One of 'tangency', 'minimum_variance', 'equal_weight'.
        long_only (bool): Whether to enforce long-only constraints.
        lookback_days (int): Trading days of history used to estimate weights.
        rebalance_days (int): Trading days each set of weights is held before
            being recomputed.

    Returns:
        BacktestResult

    Raises:
        ValueError: If there isn't enough data for at least one full
            lookback window plus one holding period.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {STRATEGIES}.")

    n_total = len(returns)
    if n_total <= lookback_days:
        raise ValueError(
            f"Not enough data to backtest: {n_total} rows available, "
            f"need more than lookback_days={lookback_days}."
        )

    out_of_sample_index = returns.index[lookback_days:]
    daily_returns = pd.Series(index=out_of_sample_index, dtype=float)
    weights_history = []

    idx = lookback_days
    while idx < n_total:
        # Estimation window strictly precedes the holding period - no
        # lookahead into the returns being evaluated.
        window = returns.iloc[idx - lookback_days:idx]
        analyzer = PortfolioAnalyzer(window, risk_free_rate, long_only)
        weights = _weights_for_strategy(analyzer, strategy)

        period_end = min(idx + rebalance_days, n_total)
        period_returns = returns.iloc[idx:period_end]

        # The same target weights are applied to every day's returns in the
        # holding period, i.e. a constant-mix portfolio that is implicitly
        # rebalanced back to target daily - NOT buy-and-hold, where weights
        # would drift with relative performance between rebalance dates.
        period_portfolio_returns = period_returns.values @ weights
        daily_returns.iloc[idx - lookback_days:period_end - lookback_days] = period_portfolio_returns

        weights_history.append({
            'date': returns.index[idx],
            'weights': dict(zip(returns.columns, weights)),
        })

        idx = period_end

    return BacktestResult(strategy, daily_returns, weights_history, risk_free_rate)


def compare_strategies(returns, risk_free_rate, strategies=STRATEGIES, long_only=True,
                        lookback_days=252, rebalance_days=63):
    """
    Run a walk-forward backtest for each of several strategies over the same
    period, so their realized performance can be compared directly.

    Returns:
        dict: {strategy_name: BacktestResult}
    """
    return {
        strategy: run_backtest(
            returns, risk_free_rate, strategy=strategy, long_only=long_only,
            lookback_days=lookback_days, rebalance_days=rebalance_days,
        )
        for strategy in strategies
    }
