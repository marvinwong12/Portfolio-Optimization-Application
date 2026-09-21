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
- **Efficient frontier visualization**: a Monte Carlo cloud (vectorized
  NumPy, not a Python loop) overlaid with the *exact* frontier curve, solved
  via constrained optimization (`scipy.optimize`, SLSQP) rather than sampled
  - which also supports an optional per-asset weight cap that the
  closed-form tangency/min-variance solutions can't express
- **Walk-forward backtesting** comparing each strategy's realized,
  out-of-sample performance against an equal-weight benchmark
- **Weight history**: every `/access` run records a snapshot of each
  strategy's weights, so `/history/<id>` can chart how the "optimal"
  allocation actually drifted across runs instead of only showing the
  latest one
- **Risk-adjusted performance metrics**: Sharpe ratio, Treynor ratio, beta,
  and Jensen's alpha against a market benchmark (SPY)
- **Individual stock analysis**: valuation ratios, profitability metrics,
  technical indicators (RSI, MACD, Bollinger Bands), dividend analysis, and
  a simplified DCF valuation
- **Per-user accounts**: each user registers, logs in, and only ever sees
  and manages their own portfolios - plus a one-click **Try Demo** login
  (`flask seed-demo-user`) for exploring without registering
- **Caching**: a TTL cache in front of the Yahoo Finance API so repeated
  page views don't re-fetch and re-render on every request
- **CSRF protection** on every state-changing request (Flask-WTF), and a
  hardened `SECRET_KEY` requirement outside debug/testing mode

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
tests/                    # pytest suite (152 tests)
```

## Tech stack

Flask, Flask-SQLAlchemy, Flask-Login, Flask-Migrate (Alembic), Flask-WTF
(CSRF protection), SQLite, [yfinance](https://github.com/ranaroussi/yfinance),
NumPy, pandas, scikit-learn (Ledoit-Wolf covariance shrinkage),
matplotlib/seaborn, pytest.

## Getting started

```bash
git clone https://github.com/marvinwong12/Portfolio-Optimization-Application.git
cd Portfolio-Optimization-Application

python3 -m venv .venv
source .venv/bin/activate       # .venv\Scripts\activate on Windows
pip install -r requirements.txt

export FLASK_APP=app.py
export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
flask db upgrade                # create/update the database schema
flask seed-demo-user            # optional: seeds a one-click "Try Demo" account

python app.py                   # http://127.0.0.1:5000
```

`SECRET_KEY` signs session cookies and CSRF tokens, so the app refuses to
start without one outside of debug/testing mode - generate it once and keep
it somewhere durable (not just your shell history) for a real deployment.
Alternatively, set `FLASK_DEBUG=1` for pure local development to skip this
(uses a fixed, clearly-insecure dev key instead).

Register an account, add a portfolio (a comma-separated list of tickers),
then use **Access** to run the optimization or **Backtest** to walk it
forward through history. Or click **Try Demo** on the login page to skip
registration and explore two pre-loaded sample portfolios immediately.

## Running tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

152 tests cover the optimization math (including regression tests for a
handful of real bugs found along the way — a tangency-weight sign flip, a
risk-free-rate unit mismatch, a `None` dividend yield crash), the exact
efficient frontier (bounds, monotonicity, dominated-region exclusion), the
backtest engine's no-lookahead guarantee, weight-history snapshot
persistence, caching, per-user access control, and CSRF protection
(verified end-to-end with it explicitly turned back on), all with
`yfinance` mocked so the suite runs fully offline.

## Deploying to Render (free tier)

`render.yaml` describes the service. If you set it up by hand in the dashboard instead, use:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `flask db upgrade && flask seed-demo-user && gunicorn app:app --workers 1 --threads 4 --timeout 120`
- **Environment variables:** `SECRET_KEY` (required - the app refuses to boot without it), `FLASK_APP=app.py`, `PYTHON_VERSION=3.13.5`

The free tier's filesystem is ephemeral, so the SQLite database (registered users, saved
portfolios) resets on every deploy and after each idle spin-down. The start command re-creates
the schema and the **Try Demo** account on every boot, so the demo always works; for durable
user data, attach a persistent disk or switch to Render Postgres.
