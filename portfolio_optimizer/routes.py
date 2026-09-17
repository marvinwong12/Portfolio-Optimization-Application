"""Flask routes for the portfolio optimization app."""
import json
from datetime import datetime, timedelta

import numpy as np
from flask import Blueprint, render_template, request, redirect, abort, flash
from flask_login import login_required, current_user

from .models import db, Portfolios, PortfolioSnapshot
from .portfolio_service import PortfolioApp
from .stock_analysis import StockAnalysis
from .data_fetcher import StockDataFetcher
from .backtest import compare_strategies
from .visualizer import PortfolioVisualizer

bp = Blueprint('main', __name__)


def _get_owned_portfolio_or_404(id):
    """Fetch a portfolio by id, 404 if it doesn't exist, 403 if the current
    user doesn't own it. Centralized here so every portfolio route enforces
    ownership the same way."""
    portfolio = Portfolios.query.get_or_404(id)
    if portfolio.user_id != current_user.id:
        abort(403)
    return portfolio


def _get_user_portfolios():
    return (Portfolios.query
            .filter_by(user_id=current_user.id)
            .order_by(Portfolios.date_created)
            .all())


def _safe_float(value):
    """Convert a metric (possibly a numpy scalar, possibly None) to a
    plain Python float for storage, without raising on None."""
    return None if value is None else float(value)

# Walk-forward backtest parameters: weights are re-estimated from the
# trailing BACKTEST_LOOKBACK_DAYS of returns every BACKTEST_REBALANCE_DAYS,
# and held fixed over each holding period.
BACKTEST_LOOKBACK_DAYS = 252
BACKTEST_REBALANCE_DAYS = 63
# Fetch more history than /access does (3y): a backtest needs both the
# lookback window AND a meaningful out-of-sample period on top of it.
BACKTEST_HISTORY_YEARS = 6


@bp.route('/', methods=['POST', 'GET'])
@login_required
def index():
    """
    Main page route - display the current user's portfolios and handle
    form submission.

    Returns:
        Rendered template or redirect
    """
    if request.method == 'POST':
        # Process form submission
        portfolio_name = request.form.get('name', 'Unnamed Portfolio')
        stocks_chosen = request.form.get('stocks', '')
        long_only = request.form.get('long_only', 'true').lower() == 'true'  # Get the long_only option

        # Validate stocks input
        symbols = [stock.strip().upper() for stock in stocks_chosen.split(",") if stock.strip()]
        if not symbols:
            flash('Please enter at least one stock symbol.')
            return render_template('index.html', portfolios=_get_user_portfolios())

        # Check the tickers actually exist before saving the portfolio,
        # rather than only discovering a typo later when the user clicks
        # Access and gets a dead end.
        try:
            _, invalid_symbols = StockDataFetcher().validate_symbols(symbols)
        except Exception as e:
            flash(f"Couldn't verify stock symbols right now ({str(e)}). Please try again.")
            return render_template('index.html', portfolios=_get_user_portfolios())

        if invalid_symbols:
            flash(
                f"Could not find data for: {', '.join(invalid_symbols)}. "
                f"Double-check the ticker symbols and try again."
            )
            return render_template('index.html', portfolios=_get_user_portfolios())

        # Create new portfolio, owned by the logged-in user
        new_portfolio = Portfolios(
            user_id=current_user.id,
            name=portfolio_name,
            stocks=stocks_chosen,
            description=request.form.get('description', ''),
            long_only=long_only  # Store the long_only setting
        )

        try:
            # Save to database
            db.session.add(new_portfolio)
            db.session.commit()
            return redirect('/')
        except Exception as e:
            flash(f'There was an issue adding your portfolio: {str(e)}')
            return render_template('index.html', portfolios=_get_user_portfolios())
    else:
        # Display only the current user's portfolios
        return render_template('index.html', portfolios=_get_user_portfolios())


@bp.route('/delete/<int:id>', methods=['POST'])
@login_required
def delete(id):
    """
    Delete a portfolio by ID. Only the owning user may delete it.

    POST-only (not GET): deleting is a state-changing action, and CSRF
    protection only covers state-changing HTTP methods - a plain GET link
    would stay forgeable (e.g. via an <img src="...">) even with
    CSRFProtect installed.

    Args:
        id (int): Portfolio ID

    Returns:
        Redirect or error message
    """
    portfolio_to_delete = _get_owned_portfolio_or_404(id)

    try:
        # Delete portfolio from database
        db.session.delete(portfolio_to_delete)
        db.session.commit()
        return redirect('/')
    except Exception as e:
        print(f"Error deleting portfolio {id}: {e}")
        flash('There was a problem deleting that portfolio.')
        return redirect('/')


@bp.route('/access/<int:id>', methods=['GET', 'POST'])
@login_required
def access(id):
    portfolio = _get_owned_portfolio_or_404(id)

    try:
        # Parse stock symbols
        symbols = [stock.strip().upper() for stock in portfolio.stocks.split(",") if stock.strip()]

        if not symbols:
            flash('Portfolio contains no valid stock symbols.')
            return redirect('/')

        # Set date range (3 years of historical data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3 * 365)

        # Use a request-local instance instead of shared global state, so
        # concurrent requests don't overwrite each other's in-progress analysis.
        portfolio_app = PortfolioApp()

        # Fetch and analyze data with the portfolio's long_only setting
        portfolio_app.fetch_data(symbols, start_date, end_date, interval='1d', long_only=portfolio.long_only)

        # Check if all symbols were found
        if len(portfolio_app.returns.columns) != len(symbols):
            flash('One or more stock symbols in this portfolio could not be found.')
            return redirect('/')

        # Run portfolio analysis
        analysis_results = portfolio_app.run_analysis(market_symbol='SPY')

        # Rest of the function remains the same...
        # Get portfolio performance data
        portfolio_names = portfolio_app.get_portfolio_names()
        market_data = portfolio_app.market_data
        comparison = portfolio_app.compare_portfolios(portfolio_names, market_data)

        # Get individual asset performance
        asset_performance = {}
        for symbol in symbols:
            if symbol in portfolio_app.returns.columns:
                asset_return = portfolio_app.returns[symbol].mean() * 252
                asset_volatility = portfolio_app.returns[symbol].std() * (252 ** 0.5)
                asset_sharpe = (asset_return - portfolio_app.analyzer.risk_free_rate) / asset_volatility
                asset_performance[symbol] = {
                    'return': asset_return,
                    'volatility': asset_volatility,
                    'sharpe_ratio': asset_sharpe
                }

        # Get weights for each portfolio strategy
        portfolio_weights = {}
        for portfolio_name in portfolio_names:
            portfolio_data = portfolio_app.get_portfolio(portfolio_name)
            if portfolio_data:
                weights_dict = {}
                for i, symbol in enumerate(portfolio_app.returns.columns):
                    if i < len(portfolio_data['weights']):
                        weights_dict[symbol] = portfolio_data['weights'][i]
                    else:
                        weights_dict[symbol] = 0.0
                portfolio_weights[portfolio_name] = weights_dict

        # Persist the tangency (max Sharpe) portfolio's weights so the
        # `weights` column reflects the latest analysis instead of staying
        # permanently empty, and record a PortfolioSnapshot for every
        # computed strategy so weight drift over time can be shown on
        # /history/<id> - not just the single latest run.
        tangency_weights = portfolio_weights.get('Tangency')
        try:
            if tangency_weights:
                portfolio.weights = json.dumps({
                    symbol: float(weight) for symbol, weight in tangency_weights.items()
                })

            for strategy_name, weights_dict in portfolio_weights.items():
                metrics = comparison.get(strategy_name, {})
                db.session.add(PortfolioSnapshot(
                    portfolio_id=portfolio.id,
                    strategy=strategy_name,
                    weights=json.dumps({symbol: float(weight) for symbol, weight in weights_dict.items()}),
                    portfolio_return=_safe_float(metrics.get('return')),
                    volatility=_safe_float(metrics.get('volatility')),
                    sharpe_ratio=_safe_float(metrics.get('sharpe_ratio')),
                ))

            db.session.commit()
        except Exception as e:
            print(f"Error saving portfolio weights/snapshot: {e}")
            db.session.rollback()

        # Get analysis details
        analysis_details = {
            'risk_free_rate': portfolio_app.risk_free_rate,
            'start_date': portfolio_app.returns.index[0].date() if portfolio_app.returns is not None else None,
            'end_date': portfolio_app.returns.index[-1].date() if portfolio_app.returns is not None else None,
            'trading_days': len(portfolio_app.returns) if portfolio_app.returns is not None else 0,
            'portfolio_type': 'Long-only' if portfolio.long_only else 'Long-short'
        }

        # Render analysis results
        return render_template('access.html',
                             portfolio=portfolio,
                             symbols=symbols,
                             comparison=comparison,
                             portfolio_names=portfolio_names,
                             asset_performance=asset_performance,
                             image_data=analysis_results['image_data'],
                             market_available=analysis_results['market_returns_available'],
                             analysis_details=analysis_details,
                             portfolio_weights=portfolio_weights)

    except Exception as e:
        print(f"Error analyzing portfolio: {str(e)}")
        flash(f'There was a problem accessing your portfolio: {str(e)}')
        return redirect('/')


@bp.route('/backtest/<int:id>')
@login_required
def backtest(id):
    """
    Walk-forward backtest a portfolio's strategies against each other, to
    check whether the optimization actually would have outperformed a naive
    equal-weight benchmark out-of-sample - as opposed to /access, which only
    shows weights computed once from a single static lookback window.
    """
    portfolio = _get_owned_portfolio_or_404(id)

    try:
        symbols = [stock.strip().upper() for stock in portfolio.stocks.split(",") if stock.strip()]
        if not symbols:
            flash('Portfolio contains no valid stock symbols.')
            return redirect('/')

        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_HISTORY_YEARS * 365)

        fetcher = StockDataFetcher()
        price_data = fetcher.get_multiple_stocks(symbols, start_date, end_date, interval='1d')

        if len(price_data.columns) != len(symbols):
            flash('One or more stock symbols in this portfolio could not be found.')
            return redirect('/')

        returns = price_data.pct_change().dropna()
        risk_free_rate = fetcher.get_risk_free_rate()

        min_required_days = BACKTEST_LOOKBACK_DAYS + BACKTEST_REBALANCE_DAYS
        if len(returns) <= min_required_days:
            flash(
                f'Not enough historical data to backtest this portfolio: need more than '
                f'{min_required_days} trading days of history, found {len(returns)}.'
            )
            return redirect('/')

        results = compare_strategies(
            returns, risk_free_rate,
            strategies=('tangency', 'minimum_variance', 'equal_weight'),
            long_only=portfolio.long_only,
            lookback_days=BACKTEST_LOOKBACK_DAYS,
            rebalance_days=BACKTEST_REBALANCE_DAYS,
        )

        equity_curves = {name: result.equity_curve for name, result in results.items()}
        chart = PortfolioVisualizer.plot_backtest_comparison(equity_curves)
        metrics_by_strategy = {name: result.metrics for name, result in results.items()}

        return render_template('backtest.html',
                             portfolio=portfolio,
                             symbols=symbols,
                             chart=chart,
                             metrics_by_strategy=metrics_by_strategy,
                             lookback_days=BACKTEST_LOOKBACK_DAYS,
                             rebalance_days=BACKTEST_REBALANCE_DAYS,
                             risk_free_rate=risk_free_rate)

    except Exception as e:
        print(f"Error backtesting portfolio: {str(e)}")
        flash(f'There was a problem backtesting your portfolio: {str(e)}')
        return redirect('/')


@bp.route('/history/<int:id>')
@login_required
def history(id):
    """
    Show how each strategy's weight allocation has drifted across
    successive /access runs, using the PortfolioSnapshot rows recorded
    each time - unlike /access, which only ever shows the latest snapshot.
    """
    portfolio = _get_owned_portfolio_or_404(id)

    snapshots = (PortfolioSnapshot.query
                 .filter_by(portfolio_id=portfolio.id)
                 .order_by(PortfolioSnapshot.created_at)
                 .all())

    if not snapshots:
        flash('No analysis history yet for this portfolio - run Access at least once to start recording snapshots.')
        return redirect(f'/access/{portfolio.id}')

    # Group by strategy, preserving the order strategies were first seen,
    # so the strategy picker below lists them consistently run to run.
    by_strategy = {}
    for snapshot in snapshots:
        by_strategy.setdefault(snapshot.strategy, []).append(snapshot)

    strategy = request.args.get('strategy')
    if strategy not in by_strategy:
        strategy = next(iter(by_strategy))

    strategy_snapshots = by_strategy[strategy]
    parsed_weights = [json.loads(s.weights) for s in strategy_snapshots]
    symbols = sorted({symbol for weights in parsed_weights for symbol in weights})

    dates = [s.created_at for s in strategy_snapshots]
    weights_matrix = np.array([
        [weights.get(symbol, 0.0) for symbol in symbols]
        for weights in parsed_weights
    ])

    chart = None
    if len(dates) >= 2:
        # A single snapshot has nothing to show drift against - the table
        # below still shows it, just without a (degenerate) one-point chart.
        chart = PortfolioVisualizer.plot_weight_history(
            dates, weights_matrix, symbols, f'{strategy} Weight History'
        )

    return render_template('history.html',
                         portfolio=portfolio,
                         strategies=list(by_strategy.keys()),
                         selected_strategy=strategy,
                         strategy_snapshots=strategy_snapshots,
                         parsed_weights=parsed_weights,
                         symbols=symbols,
                         chart=chart)


@bp.route('/update/<int:id>', methods=['GET', 'POST'])
@login_required
def update(id):
    portfolio = _get_owned_portfolio_or_404(id)
    if request.method == 'POST':
        stocks_chosen = request.form.get('stocks', '')
        symbols = [stock.strip().upper() for stock in stocks_chosen.split(",") if stock.strip()]

        if not symbols:
            flash('Please enter at least one stock symbol.')
            return render_template('update.html', portfolio=portfolio)

        # Same reasoning as portfolio creation: catch a bad ticker here,
        # not later when the user clicks Access.
        try:
            _, invalid_symbols = StockDataFetcher().validate_symbols(symbols)
        except Exception as e:
            flash(f"Couldn't verify stock symbols right now ({str(e)}). Please try again.")
            return render_template('update.html', portfolio=portfolio)

        if invalid_symbols:
            flash(
                f"Could not find data for: {', '.join(invalid_symbols)}. "
                f"Double-check the ticker symbols and try again."
            )
            return render_template('update.html', portfolio=portfolio)

        # Update portfolio stocks and long_only setting
        portfolio.stocks = stocks_chosen
        portfolio.long_only = request.form.get('long_only', 'true').lower() == 'true'

        try:
            # Save changes to database
            db.session.commit()
            return redirect('/')

        except Exception as e:
            flash(f'There was a problem updating your portfolio: {str(e)}')
            return render_template('update.html', portfolio=portfolio)

    else:
        # Display update form
        return render_template('update.html', portfolio=portfolio)


@bp.route('/analyze_stock', methods=['POST'])
@login_required
def analyze_stock():
    """
    Analyze an individual stock and display results.
    """
    ticker_symbol = request.form['ticker_symbol'].strip().upper()

    if not ticker_symbol:
        return redirect('/')

    try:
        # Perform comprehensive analysis
        analyzer = StockAnalysis(ticker_symbol)
        analysis_results = analyzer.comprehensive_analysis()

        return render_template('stock_analysis.html',
                             analysis=analysis_results,
                             symbol=ticker_symbol)

    except Exception as e:
        print(f"Error analyzing stock {ticker_symbol}: {str(e)}")
        return render_template('stock_analysis.html',
                             error=str(e),
                             symbol=ticker_symbol)
