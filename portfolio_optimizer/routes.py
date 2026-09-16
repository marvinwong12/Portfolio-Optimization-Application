"""Flask routes for the portfolio optimization app."""
import json
from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, redirect, abort
from flask_login import login_required, current_user

from .models import db, Portfolios
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
        stocks_chosen = request.form['stocks']
        long_only = request.form.get('long_only', 'true').lower() == 'true'  # Get the long_only option

        # Validate stocks input
        symbols = [stock.strip().upper() for stock in stocks_chosen.split(",") if stock.strip()]
        if not symbols:
            return 'Please enter valid stock symbols separated by commas'

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
            return f'There was an issue adding your portfolio: {str(e)}'
    else:
        # Display only the current user's portfolios
        portfolios = (Portfolios.query
                      .filter_by(user_id=current_user.id)
                      .order_by(Portfolios.date_created)
                      .all())
        return render_template('index.html', portfolios=portfolios)


@bp.route('/delete/<int:id>')
@login_required
def delete(id):
    """
    Delete a portfolio by ID. Only the owning user may delete it.

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
        return 'There was a problem deleting that portfolio'


@bp.route('/access/<int:id>', methods=['GET', 'POST'])
@login_required
def access(id):
    portfolio = _get_owned_portfolio_or_404(id)

    try:
        # Parse stock symbols
        symbols = [stock.strip().upper() for stock in portfolio.stocks.split(",") if stock.strip()]

        if not symbols:
            return 'Portfolio contains no valid stock symbols'

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
            return 'One or more stock symbols can not be found.'

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
        # permanently empty.
        tangency_weights = portfolio_weights.get('Tangency')
        if tangency_weights:
            try:
                portfolio.weights = json.dumps({
                    symbol: float(weight) for symbol, weight in tangency_weights.items()
                })
                db.session.commit()
            except Exception as e:
                print(f"Error saving portfolio weights: {e}")
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
        return f'There was a problem accessing your portfolio: {str(e)}'


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
            return 'Portfolio contains no valid stock symbols'

        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_HISTORY_YEARS * 365)

        fetcher = StockDataFetcher()
        price_data = fetcher.get_multiple_stocks(symbols, start_date, end_date, interval='1d')

        if len(price_data.columns) != len(symbols):
            return 'One or more stock symbols can not be found.'

        returns = price_data.pct_change().dropna()
        risk_free_rate = fetcher.get_risk_free_rate()

        min_required_days = BACKTEST_LOOKBACK_DAYS + BACKTEST_REBALANCE_DAYS
        if len(returns) <= min_required_days:
            return (
                f'Not enough historical data to backtest this portfolio: need more than '
                f'{min_required_days} trading days of history, found {len(returns)}.'
            )

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
        return f'There was a problem backtesting your portfolio: {str(e)}'


@bp.route('/update/<int:id>', methods=['GET', 'POST'])
@login_required
def update(id):
    portfolio = _get_owned_portfolio_or_404(id)
    if request.method == 'POST':
        # Update portfolio stocks and long_only setting
        portfolio.stocks = request.form['stocks']
        portfolio.long_only = request.form.get('long_only', 'true').lower() == 'true'

        try:
            # Save changes to database
            db.session.commit()
            return redirect('/')

        except Exception as e:
            return f'There was a problem updating your portfolio: {str(e)}'

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
