"""Walk-forward backtesting for the portfolio optimization strategies.

The rest of the app computes optimized weights from a single static lookback
window and presents them as if they were "the" answer, with no check on
whether they'd have actually performed well going forward. This module
answers that question: at each rebalance date, weights are computed using
only returns strictly *before* that date (no lookahead), held through the
next holding period, and the realized (out-of-sample) return over that
period is recorded. Repeating this across the whole history gives a
realistic equity curve that can be compared against benchmarks.

Two things keep the comparison honest:

* Transaction costs. Each rebalance trades from the weights the portfolio
  has actually drifted to (buy-and-hold between rebalances), and pays a
  cost proportional to the traded amount - so a strategy that churns pays
  more than one that barely moves.
* Uncertainty. A backtest is a single historical path, so a difference in
  Sharpe ratios can easily be noise. A block bootstrap puts a confidence
  interval on each strategy's Sharpe ratio and on its difference from a
  baseline, so "roughly matched" becomes a statistical statement.

No Flask/DB/network dependencies, so this is unit-testable with synthetic
returns data.
"""
import numpy as np
import pandas as pd

from .analyzer import PortfolioAnalyzer

STRATEGIES = ('tangency', 'minimum_variance', 'equal_weight')
TRADING_DAYS = 252


class BacktestResult:
    """Container for one strategy's (or benchmark's) walk-forward backtest output."""

    def __init__(self, strategy, daily_returns, weights_history, risk_free_rate,
                 rebalance_turnovers=None, rebalance_days=63, transaction_cost_bps=0.0):
        self.strategy = strategy
        self.daily_returns = daily_returns  # pd.Series, out-of-sample daily returns (after costs)
        self.equity_curve = (1 + daily_returns).cumprod()
        self.weights_history = weights_history  # list of {'date', 'weights': {symbol: weight}}
        self.risk_free_rate = risk_free_rate
        self.transaction_cost_bps = transaction_cost_bps
        # Fraction of portfolio value traded (buys + sells) at each rebalance;
        # the first entry is the initial purchase from cash.
        self.rebalance_turnovers = list(rebalance_turnovers or [])
        self.metrics = compute_performance_metrics(daily_returns, risk_free_rate)
        self.metrics['annual_turnover'] = _annual_turnover(self.rebalance_turnovers, rebalance_days)

    def __repr__(self):
        return f'<BacktestResult {self.strategy} sharpe={self.metrics["sharpe_ratio"]:.2f}>'


def _annual_turnover(rebalance_turnovers, rebalance_days):
    """One-way turnover per year, ignoring the initial purchase from cash:
    the average amount traded per rebalance, halved (a dollar sold and a
    dollar bought is one dollar of turnover), times rebalances per year."""
    later = rebalance_turnovers[1:]
    if not later:
        return 0.0
    return float(np.mean(later)) / 2 * (TRADING_DAYS / rebalance_days)


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
    annualized_return = equity_curve.iloc[-1] ** (TRADING_DAYS / n_days) - 1
    annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)

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
                  lookback_days=252, rebalance_days=63, transaction_cost_bps=0.0,
                  drift=True):
    """
    Run a walk-forward backtest for a single strategy.

    At each rebalance date, weights are estimated from the `lookback_days`
    of returns immediately preceding it (never including the holding period
    itself) and the portfolio is traded to those weights, paying
    `transaction_cost_bps` on everything bought or sold. The portfolio is
    then held for `rebalance_days`. This repeats until the data is
    exhausted.

    Args:
        returns (pd.DataFrame): Daily returns, one column per asset, sorted
            ascending by date.
        risk_free_rate (float): Annual risk-free rate as a decimal.
        strategy (str): One of 'tangency', 'minimum_variance', 'equal_weight'.
        long_only (bool): Whether to enforce long-only constraints.
        lookback_days (int): Trading days of history used to estimate weights.
        rebalance_days (int): Trading days between rebalances.
        transaction_cost_bps (float): Cost per unit traded, in basis points
            (10 = 0.10% of every dollar bought or sold). The cost is paid at
            each rebalance, including the initial purchase from cash, and
            comes out of that day's return.
        drift (bool): If True (default), weights drift with relative asset
            performance between rebalances (true buy-and-hold), so each
            rebalance trades only the difference between the drifted weights
            and the new targets. If False, the same target weights are
            applied to every day's returns - a constant-mix portfolio that
            is implicitly rebalanced daily, whose intra-period trading is
            NOT charged any cost.

    Returns:
        BacktestResult

    Raises:
        ValueError: If there isn't enough data for at least one full
            lookback window plus one holding period.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of {STRATEGIES}.")
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative.")

    n_total = len(returns)
    if n_total <= lookback_days:
        raise ValueError(
            f"Not enough data to backtest: {n_total} rows available, "
            f"need more than lookback_days={lookback_days}."
        )

    cost_rate = transaction_cost_bps / 10_000
    out_of_sample_index = returns.index[lookback_days:]
    daily_returns = pd.Series(index=out_of_sample_index, dtype=float)
    weights_history = []
    turnovers = []

    held = np.zeros(returns.shape[1])  # start in cash
    idx = lookback_days
    while idx < n_total:
        # Estimation window strictly precedes the holding period - no
        # lookahead into the returns being evaluated.
        window = returns.iloc[idx - lookback_days:idx]
        analyzer = PortfolioAnalyzer(window, risk_free_rate, long_only)
        target = _weights_for_strategy(analyzer, strategy)

        traded = float(np.abs(target - held).sum())
        turnovers.append(traded)

        period_end = min(idx + rebalance_days, n_total)
        period_returns = returns.iloc[idx:period_end].values

        if drift:
            current = target.copy()
            period_portfolio_returns = np.empty(len(period_returns))
            for day, asset_returns in enumerate(period_returns):
                portfolio_return = float(current @ asset_returns)
                period_portfolio_returns[day] = portfolio_return
                current = current * (1 + asset_returns) / (1 + portfolio_return)
            held = current
        else:
            period_portfolio_returns = period_returns @ target
            held = target

        # The rebalancing cost is paid on the first day of the period.
        period_portfolio_returns[0] = (1 + period_portfolio_returns[0]) * (1 - traded * cost_rate) - 1

        daily_returns.iloc[idx - lookback_days:period_end - lookback_days] = period_portfolio_returns
        weights_history.append({
            'date': returns.index[idx],
            'weights': dict(zip(returns.columns, target)),
        })

        idx = period_end

    return BacktestResult(
        strategy, daily_returns, weights_history, risk_free_rate,
        rebalance_turnovers=turnovers, rebalance_days=rebalance_days,
        transaction_cost_bps=transaction_cost_bps,
    )


def compare_strategies(returns, risk_free_rate, strategies=STRATEGIES, long_only=True,
                        lookback_days=252, rebalance_days=63, transaction_cost_bps=0.0,
                        drift=True):
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
            transaction_cost_bps=transaction_cost_bps, drift=drift,
        )
        for strategy in strategies
    }


def benchmark_result(name, benchmark_returns, index, risk_free_rate):
    """
    Wrap a benchmark's daily returns (e.g. SPY buy-and-hold) as a
    BacktestResult over the same out-of-sample dates as the strategies, so
    it can be compared and bootstrapped in exactly the same way. Buy-and-hold
    never rebalances, so it carries no turnover or costs.

    Args:
        name (str): Label, e.g. "SPY (Buy & Hold)"
        benchmark_returns (pd.Series): Benchmark daily returns
        index (pd.DatetimeIndex): The strategies' out-of-sample dates
        risk_free_rate (float): Annual risk-free rate as a decimal

    Raises:
        ValueError: If the benchmark is missing data for any of `index`.
    """
    aligned = benchmark_returns.reindex(index)
    if aligned.isna().any():
        raise ValueError(f"Benchmark '{name}' is missing returns for {int(aligned.isna().sum())} backtest dates.")
    return BacktestResult(name, aligned, [], risk_free_rate)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _bootstrap_indices(n_days, n_boot, block_size, seed):
    """
    Circular moving-block bootstrap indices, shape (n_boot, n_days). Resampling
    contiguous blocks (rather than single days) preserves the short-range
    autocorrelation and volatility clustering in daily returns, which an
    i.i.d. day-by-day bootstrap would destroy and thereby understate
    uncertainty. The same seed always gives the same indices, so two
    strategies resampled with the same seed are resampled on the same dates
    (a paired bootstrap).
    """
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_days / block_size))
    starts = rng.integers(0, n_days, size=(n_boot, n_blocks))
    indices = (starts[:, :, None] + np.arange(block_size)[None, None, :]) % n_days
    return indices.reshape(n_boot, -1)[:, :n_days]


def _sharpe_of_samples(samples, risk_free_rate):
    """Sharpe ratio of each row of `samples`, computed exactly as in
    compute_performance_metrics (geometric annualized return)."""
    n_days = samples.shape[1]
    annualized_return = np.prod(1 + samples, axis=1) ** (TRADING_DAYS / n_days) - 1
    annualized_volatility = samples.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(
            annualized_volatility > 0,
            (annualized_return - risk_free_rate) / annualized_volatility,
            0.0,
        )


def bootstrap_sharpe_ci(daily_returns, risk_free_rate, n_boot=2000, block_size=21,
                         confidence=0.95, seed=42):
    """
    Block-bootstrap confidence interval for a strategy's Sharpe ratio.

    Returns:
        dict: {'sharpe': point estimate, 'ci_low', 'ci_high'}
    """
    values = np.asarray(daily_returns, dtype=float)
    indices = _bootstrap_indices(len(values), n_boot, block_size, seed)
    boot = _sharpe_of_samples(values[indices], risk_free_rate)
    alpha = (1 - confidence) / 2
    return {
        'sharpe': float(_sharpe_of_samples(values[None, :], risk_free_rate)[0]),
        'ci_low': float(np.quantile(boot, alpha)),
        'ci_high': float(np.quantile(boot, 1 - alpha)),
    }


def bootstrap_sharpe_difference(returns_a, returns_b, risk_free_rate, n_boot=2000,
                                 block_size=21, confidence=0.95, seed=42):
    """
    Paired block-bootstrap for the difference in Sharpe ratios, A minus B.
    Both series are resampled on the same dates, so the strong correlation
    between two strategies holding overlapping assets is respected - which
    is what makes the interval on the *difference* much tighter than the
    two individual intervals would suggest.

    Returns:
        dict: {
            'difference': point estimate of Sharpe(A) - Sharpe(B),
            'ci_low', 'ci_high': confidence interval for the difference,
            'prob_a_better': share of resamples in which A's Sharpe beats B's,
            'significant': True if the interval excludes zero,
        }
    """
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Return series must have the same length to be compared.")

    indices = _bootstrap_indices(len(a), n_boot, block_size, seed)
    boot = (_sharpe_of_samples(a[indices], risk_free_rate)
            - _sharpe_of_samples(b[indices], risk_free_rate))
    alpha = (1 - confidence) / 2
    ci_low, ci_high = float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))
    point = float(_sharpe_of_samples(a[None, :], risk_free_rate)[0]
                  - _sharpe_of_samples(b[None, :], risk_free_rate)[0])
    return {
        'difference': point,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'prob_a_better': float((boot > 0).mean()),
        'significant': bool(ci_low > 0 or ci_high < 0),
    }


def bootstrap_summary(results, baselines, risk_free_rate, n_boot=2000, block_size=21,
                       confidence=0.95, seed=42):
    """
    Confidence intervals for every result's Sharpe ratio, and a paired
    comparison of every result against each named baseline.

    Args:
        results (dict): {name: BacktestResult}, including the baselines
        baselines (list): names in `results` to compare everything else against

    Returns:
        dict: {
            'sharpe_ci': {name: {'sharpe', 'ci_low', 'ci_high'}},
            'versus': {baseline: {name: bootstrap_sharpe_difference(...)}},
        }
    """
    sharpe_ci = {
        name: bootstrap_sharpe_ci(r.daily_returns, risk_free_rate, n_boot, block_size, confidence, seed)
        for name, r in results.items()
    }
    versus = {
        baseline: {
            name: bootstrap_sharpe_difference(
                r.daily_returns, results[baseline].daily_returns, risk_free_rate,
                n_boot, block_size, confidence, seed,
            )
            for name, r in results.items() if name != baseline
        }
        for baseline in baselines
    }
    return {'sharpe_ci': sharpe_ci, 'versus': versus}
