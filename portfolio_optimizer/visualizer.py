"""Matplotlib chart rendering, converted to base64 PNGs for embedding in HTML."""
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
    def plot_to_base64():
        """Convert current matplotlib figure to base64 string"""
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        plot_data = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()  # Make sure to close the figure
        plt.clf()    # Clear the current figure
        return plot_data

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

        return PortfolioVisualizer.plot_to_base64()

    @staticmethod
    def plot_weights(weights, symbols, title, long_only=True):
        """
        Plot portfolio weights as a pie chart (long-only) or bar chart (long-short).

        Args:
            weights (np.array): Portfolio weights
            symbols (list): Asset symbols
            title (str): Chart title
            long_only (bool): Whether the portfolio is long-only or long-short

        Returns:
            str: Base64 encoded image data
        """
        if long_only:
            # Use pie chart for long-only portfolios
            plt.figure(figsize=(10, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
            wedges, texts, autotexts = plt.pie(weights, labels=symbols, autopct='%1.1f%%', colors=colors)
            plt.title(title)
        else:
            # Use bar chart for long-short portfolios (to handle negative values)
            plt.figure(figsize=(12, 6))

            # Create color array: green for positive, red for negative
            colors = ['green' if w >= 0 else 'red' for w in weights]

            # Create bar chart
            bars = plt.bar(symbols, weights, color=colors, alpha=0.7)

            # Add value labels on top of bars
            for i, (symbol, weight) in enumerate(zip(symbols, weights)):
                plt.text(i, weight + (0.01 if weight >= 0 else -0.03),
                        f'{weight:.2%}', ha='center', va='bottom' if weight >= 0 else 'top')

            plt.title(title)
            plt.xlabel('Assets')
            plt.ylabel('Weight')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')

            # Add horizontal line at zero
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

        return PortfolioVisualizer.plot_to_base64()

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

        return PortfolioVisualizer.plot_to_base64()

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

        return PortfolioVisualizer.plot_to_base64()

    @staticmethod
    def plot_backtest_comparison(equity_curves):
        """
        Plot walk-forward backtest equity curves for multiple strategies
        on the same axes, so realized (out-of-sample) performance can be
        compared directly.

        Args:
            equity_curves (dict): {strategy_name: pd.Series} of cumulative
                growth-of-$1 equity curves, as produced by BacktestResult.

        Returns:
            str: Base64 encoded image data
        """
        plt.figure(figsize=(12, 6))
        for name, curve in equity_curves.items():
            curve.plot(label=name)
        plt.title('Walk-Forward Backtest: Growth of $1 (Out-of-Sample)')
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        plt.tight_layout()

        return PortfolioVisualizer.plot_to_base64()
