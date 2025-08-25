# Set matplotlib backend to Agg for non-interactive plotting (server environments)
import matplotlib
matplotlib.use('Agg')

# Import required libraries
from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.covariance import OAS, LedoitWolf

# Initialize Flask application
app = Flask(__name__)

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///portfolios.db'
db = SQLAlchemy(app)


class StockDataFetcher:
    """
    A class to fetch and process stock data from Yahoo Finance API.
    
    Methods:
        get_historical_data: Fetch historical price data for a single symbol
        get_multiple_stocks: Fetch historical price data for multiple symbols
        deannualize: Convert annual rate to periodic rate
        get_risk_free_rate: Fetch current risk-free rate from 3-month T-bills
    """
    
    def __init__(self):
        """Initialize the StockDataFetcher."""
        pass
    
    def get_historical_data(self, symbol, start_date, end_date, interval='1d'):
        """
        Fetch historical closing prices for a given stock symbol.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for historical data
            end_date (datetime): End date for historical data
            interval (str): Data interval ('1d' for daily)
            
        Returns:
            pd.Series: Historical closing prices
            
        Raises:
            ValueError: If no data is found for the symbol
        """
        try:
            # Convert dates to string format for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_str, end=end_str, interval=interval)
            
            # Check if data is empty
            if df.empty:
                raise ValueError(f"No data found for {symbol}")
                
            return df['Close']
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def get_multiple_stocks(self, symbols, start_date, end_date, interval='1d'):
        """
        Fetch historical data for multiple stock symbols.
        
        Args:
            symbols (list): List of stock ticker symbols
            start_date (datetime): Start date for historical data
            end_date (datetime): End date for historical data
            interval (str): Data interval ('1d' for daily)
            
        Returns:
            pd.DataFrame: DataFrame with closing prices for all symbols
            
        Raises:
            ValueError: If no data is successfully fetched for any symbol
        """
        data = {}
        successful_symbols = []
        
        # Fetch data for each symbol
        for symbol in symbols:
            try:
                print(f"Fetching data for {symbol}...")
                data[symbol] = self.get_historical_data(symbol, start_date, end_date, interval)
                successful_symbols.append(symbol)
                print(f"✓ Successfully fetched data for {symbol}")
            except Exception as e:
                print(f"✗ Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        # Check if any data was fetched
        if not data:
            raise ValueError("No data was successfully fetched for any symbol")
        
        # Create DataFrame and align dates
        df = pd.DataFrame(data)
        df = df.dropna()  # Remove rows with missing values
        
        print(f"\nSuccessfully retrieved data for {len(successful_symbols)} symbols: {successful_symbols}")
        return df
    
    def deannualize(self, annual_rate, periods=365):
        """
        Convert an annual rate to a periodic rate.
        
        Args:
            annual_rate (float): Annual interest rate in percentage
            periods (int): Number of periods in a year
            
        Returns:
            float: Periodic interest rate
        """
        return (1 + annual_rate/100) ** (1/periods) - 1

    def get_risk_free_rate(self):
        """
        Get the most recent daily risk-free rate from 3-month T-bills.
        
        Returns:
            float: Daily risk-free rate or None if unavailable
        """
        try:
            # Download 3-month US Treasury bill rates
            annualized = yf.download("^IRX", period="1mo", auto_adjust=True)['Close']

            if annualized.empty:
                raise ValueError("No data returned from Yahoo Finance")

            # Convert to daily rate
            daily_rate = self.deannualize(annualized.iloc[-1].iloc[-1])
            
        except Exception as e:
            print(f"Error fetching risk-free rate: {e}")
            return None
        
        return daily_rate


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
            # Use matrix inversion to calculate minimum variance weights
            inv_cov = np.linalg.inv(self.cov_matrix)
            denominator = np.dot(ones.T, np.dot(inv_cov, ones))
            weights = np.dot(inv_cov, ones) / denominator
            
            if self.long_only:
                 # Ensure no negative weights (long-only constraint)
                weights = np.maximum(weights, 0)
                weights /= weights.sum()
                
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
            # Use matrix inversion to calculate tangency portfolio weights
            inv_cov = np.linalg.inv(self.cov_matrix)
            weights = np.dot(inv_cov, excess_returns)
            weights /= weights.sum()
            
            if self.long_only:
                 # Ensure no negative weights (long-only constraint)
                weights = np.maximum(weights, 0)
                weights /= weights.sum()
                
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
        results = np.zeros((3, num_portfolios))
        weights_record = []
        n = len(self.mean_returns)
        
        # Generate random portfolios
        for i in range(num_portfolios):
            weights = np.random.random(n)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            # Store results
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std
            results[2, i] = (portfolio_return - self.risk_free_rate) / portfolio_std if portfolio_std != 0 else 0
        
        return results, weights_record


class PortfolioVisualizer:
    """
    A class to create visualizations for portfolio analysis.
    
    Static Methods:
        plot_efficient_frontier: Plot efficient frontier from Monte Carlo simulation
        plot_weights: Plot portfolio weights as pie chart
        plot_correlation_matrix: Plot correlation matrix heatmap
        plot_returns_time_series: Plot cumulative returns over time
    """
    
    @staticmethod
    def plot_efficient_frontier(returns, volatilities, sharpe_ratios, optimal_portfolio=None, save_path=None):
        """
        Plot efficient frontier with Monte Carlo simulation results.
        
        Args:
            returns (np.array): Portfolio returns
            volatilities (np.array): Portfolio volatilities
            sharpe_ratios (np.array): Portfolio Sharpe ratios
            optimal_portfolio (dict): Optimal portfolio metrics
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.title('Efficient Frontier - Monte Carlo Simulation')
        
        if optimal_portfolio:
            plt.scatter(optimal_portfolio['volatility'], optimal_portfolio['return'], 
                       color='red', s=200, marker='*', label='Optimal Portfolio')
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_weights(weights, symbols, title, save_path=None):
        """
        Plot portfolio weights as a pie chart.
        
        Args:
            weights (np.array): Portfolio weights
            symbols (list): Asset symbols
            title (str): Chart title
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        wedges, texts, autotexts = plt.pie(weights, labels=symbols, autopct='%1.1f%%', colors=colors)
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_correlation_matrix(correlation_matrix, save_path=None):
        """
        Plot correlation matrix as a heatmap.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, mask=mask, fmt='.2f')
        plt.title('Asset Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_returns_time_series(returns_data, save_path=None):
        """
        Plot cumulative returns over time.
        
        Args:
            returns_data (pd.DataFrame): Returns data
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(12, 6))
        cumulative_returns = (1 + returns_data).cumprod() - 1
        cumulative_returns.plot()
        plt.title('Cumulative Returns Over Time')
        plt.ylabel('Cumulative Return')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=100)
            plt.close()
        else:
            plt.show()


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
    
    def fetch_data(self, symbols, start_date, end_date, interval='1d'):
        """
        Fetch and prepare stock data for analysis.
        
        Args:
            symbols (list): List of stock symbols
            start_date (datetime): Start date for data
            end_date (datetime): End date for data
            interval (str): Data interval
            
        Raises:
            ValueError: If no data is available after processing
        """
        print("=" * 60)
        print("PORTFOLIO ANALYSIS APPLICATION")
        print("=" * 60)
        
        # Fetch data for all symbols
        self.portfolio_data = self.api.get_multiple_stocks(symbols, start_date, end_date, interval)
        
        if self.portfolio_data.empty:
            raise ValueError("No data available after fetching and cleaning")
            
        # Calculate daily returns
        self.returns = self.portfolio_data.pct_change().dropna()
        self.risk_free_rate = self.api.get_risk_free_rate()
        
        # Initialize analyzer
        self.analyzer = PortfolioAnalyzer(self.returns, self.risk_free_rate, self.long_only)
        
        # Print summary information
        print(f"\nData Summary:")
        print(f"Period: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
        print(f"Number of trading days: {len(self.returns)}")
        print(f"Assets: {list(self.returns.columns)}")
        print(f"Risk-free rate: {self.risk_free_rate:.2%}")
        print(f"Long-only Portfolio: {self.long_only}")
    
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
        
        # Generate plots and save as images
        image_paths = {}
        if save_images:
            import os
            os.makedirs('static/images', exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Efficient Frontier
            optimal_metrics = self.analyzer.calculate_portfolio_metrics(optimal_weights)
            eff_frontier_path = f'static/images/efficient_frontier_{timestamp}.png'
            PortfolioVisualizer.plot_efficient_frontier(
                mc_results[0], mc_results[1], mc_results[2], optimal_metrics, eff_frontier_path
            )
            image_paths['efficient_frontier'] = eff_frontier_path
            
            # Portfolio weights
            weights_min_var_path = f'static/images/weights_min_var_{timestamp}.png'
            PortfolioVisualizer.plot_weights(
                min_var_weights, self.returns.columns, "Minimum Variance Portfolio Weights", weights_min_var_path
            )
            image_paths['weights_min_var'] = weights_min_var_path
            
            weights_tangency_path = f'static/images/weights_tangency_{timestamp}.png'
            PortfolioVisualizer.plot_weights(
                tangency_weights, self.returns.columns, "Tangency Portfolio Weights", weights_tangency_path
            )
            image_paths['weights_tangency'] = weights_tangency_path
            
            # Correlation matrix
            corr_matrix_path = f'static/images/correlation_matrix_{timestamp}.png'
            PortfolioVisualizer.plot_correlation_matrix(self.returns.corr(), corr_matrix_path)
            image_paths['correlation_matrix'] = corr_matrix_path
            
            # Cumulative returns
            returns_path = f'static/images/cumulative_returns_{timestamp}.png'
            PortfolioVisualizer.plot_returns_time_series(self.returns, returns_path)
            image_paths['cumulative_returns'] = returns_path
        
        # Store analysis results
        self.analysis_results = {
            'mc_results': mc_results,
            'mc_weights': mc_weights,
            'market_returns_available': self.market_data is not None,
            'image_paths': image_paths
        }
        
        return self.analysis_results


class Portfolios(db.Model):
    """
    Database model for storing portfolio information.
    
    Attributes:
        id (int): Primary key
        name (str): Portfolio name
        stocks (str): Comma-separated stock symbols
        description (str): Portfolio description
        weights (str): Portfolio weights as JSON string
        date_created (datetime): Creation timestamp
    """
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    stocks = db.Column(db.String(500), nullable=False)  # Increased length
    description = db.Column(db.String(200))
    weights = db.Column(db.String(500))  # Store weights as JSON string
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        """String representation of Portfolio object."""
        return f'<Portfolio {self.name}>'


# Initialize the portfolio application
Portfolio_app = PortfolioApp()

# Create database tables
with app.app_context():
    db.create_all()  # This will create tables with the new structure


@app.route('/', methods=['POST','GET'])
def index():
    """
    Main page route - display all portfolios and handle form submission.
    
    Returns:
        Rendered template or redirect
    """
    if request.method == 'POST':
        # Process form submission
        portfolio_name = request.form.get('name', 'Unnamed Portfolio')
        stocks_chosen = request.form['stocks']
        
        # Validate stocks input
        symbols = [stock.strip().upper() for stock in stocks_chosen.split(",") if stock.strip()]
        if not symbols:
            return 'Please enter valid stock symbols separated by commas'
        
        # Create new portfolio
        new_portfolio = Portfolios(
            name=portfolio_name,
            stocks=stocks_chosen,
            description=request.form.get('description', '')
        )

        try: 
            # Save to database
            db.session.add(new_portfolio)
            db.session.commit()
            return redirect('/')
        except Exception as e:
            return f'There was an issue adding your portfolio: {str(e)}'
    else:
        # Display all portfolios
        portfolios = Portfolios.query.order_by(Portfolios.date_created).all()
        return render_template('index.html', portfolios=portfolios)

@app.route('/delete/<int:id>')
def delete(id):
    """
    Delete a portfolio by ID.
    
    Args:
        id (int): Portfolio ID
        
    Returns:
        Redirect or error message
    """
    portfolio_to_delete = Portfolios.query.get_or_404(id)

    try:
        # Delete portfolio from database
        db.session.delete(portfolio_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that portfolio'

@app.route('/access/<int:id>', methods=['GET', 'POST'])
def access(id):
    """
    Access and analyze a specific portfolio.
    
    Args:
        id (int): Portfolio ID
        
    Returns:
        Rendered template or error message
    """
    global Portfolio_app
    portfolio = Portfolios.query.get_or_404(id)
    
    try:
        # Parse stock symbols
        symbols = [stock.strip().upper() for stock in portfolio.stocks.split(",") if stock.strip()]
        
        if not symbols:
            return 'Portfolio contains no valid stock symbols'
        
        # Set date range (3 years of historical data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)

        
        # Fetch and analyze data
        Portfolio_app.fetch_data(symbols, start_date, end_date, interval='1d')

        # Check if all symbols were found
        if len(Portfolio_app.returns.columns) != len(symbols):
            return 'One or more stock symbols can not be found.'

        # Run portfolio analysis
        analysis_results = Portfolio_app.run_analysis(market_symbol='SPY', save_images=True)
        
        # Get portfolio performance data
        portfolio_names = Portfolio_app.get_portfolio_names()
        market_data = Portfolio_app.market_data
        comparison = Portfolio_app.compare_portfolios(portfolio_names, market_data)

        
        # Get individual asset performance
        asset_performance = {}
        for symbol in symbols:
            if symbol in Portfolio_app.returns.columns:
                asset_return = Portfolio_app.returns[symbol].mean() * 252
                asset_volatility = Portfolio_app.returns[symbol].std() * np.sqrt(252)
                asset_sharpe = (asset_return - Portfolio_app.analyzer.risk_free_rate) / asset_volatility
                asset_performance[symbol] = {
                    'return': asset_return,
                    'volatility': asset_volatility,
                    'sharpe_ratio': asset_sharpe
                }
        
        # Get weights for each portfolio strategy
        portfolio_weights = {}
        for portfolio_name in portfolio_names:
            portfolio_data = Portfolio_app.get_portfolio(portfolio_name)
            if portfolio_data:
                weights_dict = {}
                for i, symbol in enumerate(Portfolio_app.returns.columns):
                    if i < len(portfolio_data['weights']):
                        weights_dict[symbol] = portfolio_data['weights'][i]
                    else:
                        weights_dict[symbol] = 0.0
                portfolio_weights[portfolio_name] = weights_dict
        
        # Get analysis details
        analysis_details = {
            'risk_free_rate': Portfolio_app.risk_free_rate,
            'start_date': Portfolio_app.returns.index[0].date() if Portfolio_app.returns is not None else None,
            'end_date': Portfolio_app.returns.index[-1].date() if Portfolio_app.returns is not None else None,
            'trading_days': len(Portfolio_app.returns) if Portfolio_app.returns is not None else 0
        }
        
        # Render analysis results
        return render_template('access.html', 
                             portfolio=portfolio,
                             symbols=symbols,
                             comparison=comparison,
                             portfolio_names=portfolio_names,
                             asset_performance=asset_performance,
                             image_paths=analysis_results['image_paths'],
                             market_available=analysis_results['market_returns_available'],
                             analysis_details=analysis_details,
                             portfolio_weights=portfolio_weights)
                             
    except Exception as e:
        print(f"Error analyzing portfolio: {str(e)}")
        return f'There was a problem accessing your portfolio: {str(e)}'

@app.route('/update/<int:id>', methods = ['GET','POST'])
def update(id):
    """
    Update a portfolio's stock symbols.
    
    Args:
        id (int): Portfolio ID
        
    Returns:
        Rendered template or redirect
    """
    portfolio = Portfolios.query.get_or_404(id)
    if request.method == 'POST':
        # Update portfolio stocks
        portfolio.stocks = request.form['stocks']

        try:
            # Save changes to database
            db.session.commit()
            return redirect('/')

        except Exception as e:
            return f'There was a problem updating your portfolio: {str(e)}'

    else:
        # Display update form
        return render_template('update.html', portfolio=portfolio)

if __name__ == "__main__":
    # Run the Flask application
    app.run(debug=True)