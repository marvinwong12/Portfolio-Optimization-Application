"""Orchestrates fetching, analyzing, and storing portfolios for a single
analysis session. A fresh instance should be created per request rather than
shared globally, since it holds request-scoped mutable state."""
from datetime import datetime
import numpy as np
import pandas as pd

from .data_fetcher import StockDataFetcher
from .analyzer import PortfolioAnalyzer
from .visualizer import PortfolioVisualizer


class PortfolioApp:
    """
    Main application class for portfolio management and analysis.

    Methods:
        fetch_data: Fetch stock data for analysis
        get_portfolio: Retrieve a specific portfolio
        get_all_portfolios: Retrieve all stored portfolios
        get_portfolio_names: Get list of portfolio names
        get_portfolio_metrics: Get metrics for a specific portfolio
        get_portfolio_performance: Get comprehensive performance metrics
        compare_portfolios: Compare multiple portfolios
        store_portfolio: Store a portfolio with metadata
        remove_portfolio: Remove a portfolio from storage
        export_portfolios: Export portfolios to CSV
        import_portfolios: Import portfolios from CSV
        run_analysis: Run complete portfolio analysis
    """

    def __init__(self):
        """Initialize the PortfolioApp."""
        self.api = StockDataFetcher()
        self.portfolio_data = None
        self.returns = None
        self.analyzer = None
        self.risk_free_rate = None
        self.long_only = True
        self.portfolios = {}  # Dictionary to store all portfolios
        self.market_data = None
        self.analysis_results = {}

    def fetch_data(self, symbols, start_date, end_date, interval='1d', long_only=True):
        """
        Fetch and prepare stock data for analysis.

        Args:
            symbols (list): List of stock symbols
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            interval (str): Data interval
            long_only (bool): Whether to use long-only constraints

        Raises:
            ValueError: If no data is available after processing
        """
        print("=" * 60)
        print("PORTFOLIO ANALYSIS APPLICATION")
        print("=" * 60)

        # Set the long-only constraint
        self.long_only = long_only

        # Fetch data for all symbols
        self.portfolio_data = self.api.get_multiple_stocks(symbols, start_date, end_date, interval)

        if self.portfolio_data.empty:
            raise ValueError("No data available after fetching and cleaning")

        # Calculate daily returns
        self.returns = self.portfolio_data.pct_change().dropna()
        self.risk_free_rate = self.api.get_risk_free_rate()

        # Initialize analyzer with long_only setting
        self.analyzer = PortfolioAnalyzer(self.returns, self.risk_free_rate, self.long_only)

        # Print summary information
        print(f"\nData Summary:")
        print(f"Period: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
        print(f"Number of trading days: {len(self.returns)}")
        print(f"Assets: {list(self.returns.columns)}")
        print(f"Risk-free rate: {self.risk_free_rate:.2%}")
        print(f"Portfolio Type: {'Long-only' if self.long_only else 'Long-short'}")

    def get_portfolio(self, portfolio_name):
        """
        Retrieve a specific portfolio by name.

        Args:
            portfolio_name (str): Name of the portfolio

        Returns:
            dict: Portfolio data or None if not found
        """
        return self.portfolios.get(portfolio_name)

    def get_all_portfolios(self):
        """
        Retrieve all stored portfolios.

        Returns:
            dict: Copy of all portfolios
        """
        return self.portfolios.copy()

    def get_portfolio_names(self):
        """
        Get list of all portfolio names.

        Returns:
            list: List of portfolio names
        """
        return list(self.portfolios.keys())

    def get_portfolio_metrics(self, portfolio_name):
        """
        Get metrics for a specific portfolio.

        Args:
            portfolio_name (str): Name of the portfolio

        Returns:
            dict: Portfolio metrics or None if not found
        """
        portfolio = self.get_portfolio(portfolio_name)
        if portfolio:
            return self.analyzer.calculate_portfolio_metrics(portfolio['weights'])
        return None

    def get_portfolio_performance(self, portfolio_name, market_returns=None):
        """
        Get comprehensive performance metrics for a portfolio.

        Args:
            portfolio_name (str): Name of the portfolio
            market_returns (pd.Series): Market returns data

        Returns:
            dict: Portfolio performance metrics or None if not found
        """
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            return None

        # Calculate basic metrics
        metrics = self.analyzer.calculate_portfolio_metrics(portfolio['weights'])
        performance = {
            'name': portfolio_name,
            'weights': portfolio['weights'],
            'return': metrics['return'],
            'volatility': metrics['volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'description': portfolio.get('description', '')
        }

        # Add risk-adjusted metrics if market data is available
        if market_returns is not None:
            performance['beta'] = self.analyzer.calculate_beta(portfolio['weights'], market_returns)
            performance['treynor_ratio'] = self.analyzer.calculate_treynor_ratio(portfolio['weights'], market_returns)
            performance['jensen_alpha'] = self.analyzer.calculate_jensen_alpha(portfolio['weights'], market_returns)

        return performance

    def compare_portfolios(self, portfolio_names, market_returns=None):
        """
        Compare multiple portfolios side by side.

        Args:
            portfolio_names (list): List of portfolio names to compare
            market_returns (pd.Series): Market returns data

        Returns:
            dict: Dictionary with comparison results
        """
        comparison = {}
        for name in portfolio_names:
            performance = self.get_portfolio_performance(name, market_returns)
            if performance:
                comparison[name] = performance
        return comparison

    def store_portfolio(self, name, weights, description=""):
        """
        Store a portfolio with metadata.

        Args:
            name (str): Portfolio name
            weights (np.array): Portfolio weights
            description (str): Portfolio description

        Returns:
            dict: Stored portfolio data
        """
        self.portfolios[name] = {
            'weights': weights,
            'description': description,
            'assets': list(self.returns.columns) if self.returns is not None else [],
            'stored_date': datetime.now()
        }
        return self.portfolios[name]

    def remove_portfolio(self, portfolio_name):
        """
        Remove a portfolio from storage.

        Args:
            portfolio_name (str): Name of the portfolio to remove

        Returns:
            bool: True if removed, False if not found
        """
        if portfolio_name in self.portfolios:
            del self.portfolios[portfolio_name]
            return True
        return False

    def export_portfolios(self, filename='portfolios_export.csv'):
        """
        Export all portfolios to CSV.

        Args:
            filename (str): Output filename

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.portfolios:
            print("No portfolios to export")
            return False

        export_data = []
        for name, portfolio in self.portfolios.items():
            row = {'portfolio_name': name, 'description': portfolio['description']}
            for i, asset in enumerate(portfolio['assets']):
                row[asset] = portfolio['weights'][i] if i < len(portfolio['weights']) else 0
            export_data.append(row)

        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"Portfolios exported to {filename}")
        return True

    def import_portfolios(self, filename='portfolios_export.csv'):
        """
        Import portfolios from CSV.

        Args:
            filename (str): Input filename

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                weights = []
                assets = []
                for col in df.columns:
                    if col not in ['portfolio_name', 'description']:
                        weights.append(row[col])
                        assets.append(col)

                self.store_portfolio(
                    row['portfolio_name'],
                    np.array(weights),
                    row.get('description', '')
                )
            print(f"Successfully imported {len(df)} portfolios from {filename}")
            return True
        except Exception as e:
            print(f"Error importing portfolios: {e}")
            return False

    def run_analysis(self, market_symbol='SPY', save_images=True):
        """
        Run complete portfolio analysis with optimizations and visualizations.

        Args:
            market_symbol (str): Market benchmark symbol
            save_images (bool): Whether to save visualization images

        Returns:
            dict: Analysis results including image paths
        """
        if self.analyzer is None:
            print("Please fetch data first.")
            return

        print("\n" + "=" * 60)
        print("RUNNING PORTFOLIO ANALYSIS")
        print("=" * 60)

        # Clear previous portfolios
        self.portfolios.clear()

        # Get market data for benchmark comparison
        try:
            market_data = self.api.get_historical_data(
                market_symbol,
                self.portfolio_data.index[0],
                self.portfolio_data.index[-1]
            )
            market_returns = market_data.pct_change().dropna()
            aligned_market_returns = market_returns.reindex(self.returns.index).dropna()
            self.market_data = aligned_market_returns if len(aligned_market_returns) > 0 else None
        except Exception as e:
            print(f"Warning: Could not fetch market data: {e}")
            self.market_data = None

        # Portfolio optimizations
        print("\n1. PORTFOLIO OPTIMIZATIONS")
        print("-" * 40)

        # Create portfolio strategies
        min_var_weights = self.analyzer.minimum_variance_portfolio()
        tangency_weights = self.analyzer.tangency_portfolio()
        equal_weights = np.ones(len(self.returns.columns)) / len(self.returns.columns)

        # Store portfolio strategies
        self.store_portfolio("Minimum Variance", min_var_weights, "Minimum variance optimized portfolio")
        self.store_portfolio("Tangency", tangency_weights, "Maximum Sharpe ratio portfolio")
        self.store_portfolio("Equal Weight", equal_weights, "Equal weight benchmark portfolio")

        # Monte Carlo Simulation
        print("Running Monte Carlo simulation...")
        mc_results, mc_weights = self.analyzer.monte_carlo_simulation(10000)

        # Find optimal portfolio from simulation
        max_sharpe_idx = np.argmax(mc_results[2])
        optimal_weights = mc_weights[max_sharpe_idx]
        self.store_portfolio("Monte Carlo Optimal", optimal_weights, "Optimal portfolio from Monte Carlo simulation")

        # Exact efficient frontier via constrained optimization, to overlay
        # on the (necessarily noisy) Monte Carlo cloud below. Best-effort:
        # if the solver fails on a pathological covariance matrix, fall
        # back to showing the Monte Carlo cloud alone rather than failing
        # the whole analysis over a chart enhancement.
        try:
            frontier = self.analyzer.efficient_frontier(num_points=50)
        except Exception as e:
            print(f"Warning: Could not compute exact efficient frontier: {e}")
            frontier = None

        # Generate plots and save as images
        image_data = {}

        # Efficient Frontier - mark the exact (closed-form) tangency
        # portfolio rather than the Monte Carlo cloud's best sample, since
        # that's the actual max-Sharpe point, not an approximation of it.
        tangency_metrics = self.analyzer.calculate_portfolio_metrics(tangency_weights)
        image_data['efficient_frontier'] = PortfolioVisualizer.plot_efficient_frontier(
            mc_results[0], mc_results[1], mc_results[2], tangency_metrics, frontier=frontier
        )

        # Portfolio weights
        image_data['weights_min_var'] = PortfolioVisualizer.plot_weights(
            min_var_weights, self.returns.columns, "Minimum Variance Portfolio Weights", self.long_only
        )

        image_data['weights_tangency'] = PortfolioVisualizer.plot_weights(
            tangency_weights, self.returns.columns, "Tangency Portfolio Weights", self.long_only
        )

        # Correlation matrix
        image_data['correlation_matrix'] = PortfolioVisualizer.plot_correlation_matrix(
            self.returns.corr()
        )

        # Cumulative returns
        image_data['cumulative_returns'] = PortfolioVisualizer.plot_returns_time_series(
            self.returns
        )

        # Store analysis results
        self.analysis_results = {
            'mc_results': mc_results,
            'mc_weights': mc_weights,
            'market_returns_available': self.market_data is not None,
            'image_data': image_data  # Now contains base64 strings instead of file paths
        }

        return self.analysis_results
