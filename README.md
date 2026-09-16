# Portfolio Optimization Application

[![Tests](https://github.com/marvinwong12/Portfolio-Optimization-Application/actions/workflows/tests.yml/badge.svg)](https://github.com/marvinwong12/Portfolio-Optimization-Application/actions/workflows/tests.yml)

A Flask web app for mean-variance portfolio optimization, individual stock
analysis, and **walk-forward backtesting** — built to check whether the
optimization is actually doing something useful out-of-sample, not just to
draw a nice efficient-frontier scatter plot.

Each user registers an account and manages their own set of portfolios.
Given a list of tickers, the app fetches historical price data, computes
minimum-variance and tangency (max Sharpe) portfolios analytically, runs a
Monte Carlo simulation to visualize the efficient frontier, and backtests
each strategy against a naive equal-weight benchmark.

## Screenshots

**Efficient frontier** (Monte Carlo simulation, optimal portfolio marked):

![Efficient frontier](docs/screenshots/efficient_frontier.png)

**Minimum-variance and tangency portfolio weights:**

<p>
  <img src="docs/screenshots/weights_min_var.png" width="49%" alt="Minimum variance weights">
  <img src="docs/screenshots/weights_tangency.png" width="49%" alt="Tangency portfolio weights">
</p>

**Asset correlation matrix and cumulative returns:**

<p>
  <img src="docs/screenshots/correlation_matrix.png" width="49%" alt="Correlation matrix">
  <img src="docs/screenshots/cumulative_returns.png" width="49%" alt="Cumulative returns">
</p>

**Walk-forward backtest**, out-of-sample, rebalanced quarterly:

![Backtest comparison](docs/screenshots/backtest_comparison.png)

## Why the backtest matters

It's easy to build a portfolio optimizer that looks good: fit mean-variance
weights on a lookback window and show the resulting efficient frontier. The
harder — and more honest — question is whether those weights would have
actually performed well *afterward*.

The backtest module answers that by walking forward through history: at each
rebalance date, it estimates weights using only the trailing 252 trading
days (never touching the period it's about to be evaluated on), holds those
weights fixed through the next 63-day period, and records the realized
return. Repeating this across years of data produces a genuine
out-of-sample equity curve per strategy.

Running it on five mega-cap stocks (AAPL, MSFT, GOOGL, AMZN, JPM) over the
last six years gives:

| Strategy | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown |
|---|---|---|---|---|
| Tangency (max Sharpe) | 15.4% | 24.5% | 0.47 | -33.3% |
| Minimum Variance | 16.5% | 20.7% | 0.61 | -34.5% |
| Equal Weight | 17.5% | 22.9% | 0.59 | -33.9% |

For this particular basket, the naive equal-weight benchmark actually held
its own against — and briefly beat — the "optimized" portfolios. That's not
a bug; it's the textbook estimation-error problem with mean-variance
optimization (Michaud's "error maximization"): with only a handful of assets
and a noisy covariance/return estimate, the optimizer chases sampling noise
in the lookback window that doesn't persist out-of-sample. Minimum variance
comes out ahead on a risk-adjusted basis here because it only depends on the
covariance matrix (better estimated than expected returns), not on the
noisier return forecast that tangency weighting requires. Surfacing this
kind of result — rather than only ever showing a favorable backtest — is the
whole point of validating with real out-of-sample testing.

## Features

- **Portfolio optimization**: minimum-variance and tangency (max Sharpe)
  portfolios, solved analytically via Ledoit-Wolf shrinkage covariance
  estimation; long-only or long-short constraints
- **Efficient frontier visualization** via Monte Carlo simulation (vectorized
  NumPy, not a Python loop)
- **Walk-forward backtesting** comparing each strategy's realized,
  out-of-sample performance against an equal-weight benchmark
- **Risk-adjusted performance metrics**: Sharpe ratio, Treynor ratio, beta,
  and Jensen's alpha against a market benchmark (SPY)
- **Individual stock analysis**: valuation ratios, profitability metrics,
  technical indicators (RSI, MACD, Bollinger Bands), dividend analysis, and
  a simplified DCF valuation
- **Per-user accounts**: each user registers, logs in, and only ever sees
  and manages their own portfolios
- **Caching**: a TTL cache in front of the Yahoo Finance API so repeated
  page views don't re-fetch and re-render on every request

## Architecture

The Flask app is a thin entry point over the `portfolio_optimizer` package,
split by responsibility so the quantitative core has zero web/DB
dependencies and can be unit-tested in isolation:

```
app.py                    # Entry point: app = create_app()
portfolio_optimizer/
├── __init__.py           # App factory (create_app), extension wiring
├── auth.py               # Register / login / logout (Flask-Login)
├── models.py             # User, Portfolios (SQLAlchemy)
├── routes.py             # Flask routes / ownership enforcement
├── data_fetcher.py       # Yahoo Finance fetching + caching
├── cache.py              # TTL cache
├── analyzer.py           # Mean-variance math (pure NumPy/pandas)
├── backtest.py           # Walk-forward backtest engine
├── visualizer.py         # Matplotlib chart rendering
├── portfolio_service.py  # Orchestrates fetch -> analyze -> store
└── stock_analysis.py     # Single-stock fundamental/technical analysis
migrations/               # Alembic schema migrations (Flask-Migrate)
tests/                    # pytest suite (99 tests)
```

## Tech stack

Flask, Flask-SQLAlchemy, Flask-Login, Flask-Migrate (Alembic), SQLite,
[yfinance](https://github.com/ranaroussi/yfinance), NumPy, pandas,
scikit-learn (Ledoit-Wolf covariance shrinkage), matplotlib/seaborn, pytest.

## Getting started

```bash
git clone https://github.com/marvinwong12/Portfolio-Optimization-Application.git
cd Portfolio-Optimization-Application

python3 -m venv .venv
source .venv/bin/activate       # .venv\Scripts\activate on Windows
pip install -r requirements.txt

export FLASK_APP=app.py
flask db upgrade                # create/update the database schema

python app.py                   # http://127.0.0.1:5000
```

Register an account, add a portfolio (a comma-separated list of tickers),
then use **Access** to run the optimization or **Backtest** to walk it
forward through history.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

99 tests cover the optimization math (including regression tests for a
handful of real bugs found along the way — a tangency-weight sign flip, a
risk-free-rate unit mismatch, a `None` dividend yield crash), the backtest
engine's no-lookahead guarantee, caching, and per-user access control, all
with `yfinance` mocked so the suite runs fully offline.
