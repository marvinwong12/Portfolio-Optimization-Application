"""Single-stock fundamental/technical analysis using yFinance data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import yfinance as yf

from .cache import price_cache


# Same-sector large caps used when the user does not name peers (Yahoo sector labels).
SECTOR_PEERS = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL', 'CRM'],
    'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'T', 'VZ'],
    'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW'],
    'Consumer Defensive': ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM'],
    'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V'],
    'Healthcare': ['LLY', 'JNJ', 'UNH', 'MRK', 'ABBV', 'PFE'],
    'Industrials': ['GE', 'CAT', 'RTX', 'HON', 'UPS', 'BA'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE'],
    'Real Estate': ['PLD', 'AMT', 'EQIX', 'SPG', 'O', 'WELL'],
    'Basic Materials': ['LIN', 'SHW', 'FCX', 'NEM', 'APD', 'ECL'],
}


class StockAnalysis:
    """
    A comprehensive class for analyzing individual stocks using yFinance data.
    Provides valuation, profitability, risk, and technical analysis.
    """

    def __init__(self, ticker_symbol):
        """
        Initialize with a ticker symbol.

        Args:
            ticker_symbol (str): Stock ticker symbol
        """
        self.ticker = yf.Ticker(ticker_symbol)
        self.symbol = ticker_symbol
        self.info = self.ticker.info
        self.historical_data = None
        self.analysis_results = {}

    def fetch_data(self, period="3y"):
        """
        Fetch historical data for analysis.

        Args:
            period (str): Time period for historical data
        """
        self.historical_data = self.ticker.history(period=period)
        return self.historical_data

    def calculate_performance_metrics(self):
        """Calculate performance and risk metrics."""
        if self.historical_data is None:
            self.fetch_data()

        closes = self.historical_data['Close']
        daily_returns = closes.pct_change().dropna()

        metrics = {
            'annualized_return': daily_returns.mean() * 252,
            'annualized_volatility': daily_returns.std() * np.sqrt(252),
            'sharpe_ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0,
            'max_drawdown': (closes / closes.cummax() - 1).min(),
            'var_95': np.percentile(daily_returns, 5),
            'cvar_95': daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean()
        }

        # Calculate beta if we have market data (using SPY as proxy)
        try:
            cache_key = ('spy_3y', datetime.now().date())
            market_data = price_cache.get(cache_key)
            if market_data is None:
                market_data = yf.Ticker("SPY").history(period="3y")['Close']
                price_cache.set(cache_key, market_data)
            market_returns = market_data.pct_change().dropna()
            aligned_returns = daily_returns.reindex(market_returns.index).dropna()
            aligned_market = market_returns.reindex(aligned_returns.index)

            covariance = np.cov(aligned_returns, aligned_market)[0, 1]
            market_variance = np.var(aligned_market)
            metrics['beta'] = covariance / market_variance if market_variance > 0 else np.nan
        except Exception as e:
            print(f"Error calculating beta: {e}")
            metrics['beta'] = np.nan

        return metrics

    def calculate_technical_indicators(self):
        """Calculate various technical indicators."""
        if self.historical_data is None:
            self.fetch_data()

        closes = self.historical_data['Close']
        highs = self.historical_data['High']
        lows = self.historical_data['Low']

        # Moving averages
        indicators = {
            'sma_50': closes.rolling(window=50).mean().iloc[-1],
            'sma_200': closes.rolling(window=200).mean().iloc[-1],
            'ema_20': closes.ewm(span=20).mean().iloc[-1]
        }

        # RSI
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]

        # MACD
        ema_12 = closes.ewm(span=12).mean()
        ema_26 = closes.ewm(span=26).mean()
        indicators['macd'] = (ema_12 - ema_26).iloc[-1]
        indicators['macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]

        # Bollinger Bands
        sma_20 = closes.rolling(window=20).mean()
        std_20 = closes.rolling(window=20).std()
        indicators['bollinger_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
        indicators['bollinger_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
        indicators['bollinger_percent'] = ((closes.iloc[-1] - indicators['bollinger_lower']) /
                                         (indicators['bollinger_upper'] - indicators['bollinger_lower'])) * 100

        return indicators

    def calculate_valuation_ratios(self):
        """Calculate valuation ratios."""
        ratios = {
            'pe_ratio': self.info.get('trailingPE'),
            'forward_pe': self.info.get('forwardPE'),
            'peg_ratio': self.info.get('pegRatio'),
            'price_to_sales': self.info.get('priceToSalesTrailing12Months'),
            'price_to_book': self.info.get('priceToBook'),
            'ev_to_ebitda': self.info.get('enterpriseToEbitda'),
            'ev_to_revenue': self.info.get('enterpriseToRevenue'),
            'dividend_yield': self.info.get('dividendYield')
        }
        return ratios

    def calculate_profitability_metrics(self):
        """Calculate profitability metrics."""
        try:
            financials = self.ticker.financials
            income_stmt = self.ticker.income_stmt
            balance_sheet = self.ticker.balance_sheet

            # Get the most recent year's data
            recent_year = financials.columns[0]

            metrics = {
                'gross_margin': financials.loc['Gross Profit', recent_year] / financials.loc['Total Revenue', recent_year] if 'Gross Profit' in financials.index and 'Total Revenue' in financials.index else None,
                'operating_margin': financials.loc['Operating Income', recent_year] / financials.loc['Total Revenue', recent_year] if 'Operating Income' in financials.index and 'Total Revenue' in financials.index else None,
                'net_margin': financials.loc['Net Income', recent_year] / financials.loc['Total Revenue', recent_year] if 'Net Income' in financials.index and 'Total Revenue' in financials.index else None,
                'return_on_equity': income_stmt.loc['Net Income', recent_year] / balance_sheet.loc['Total Stockholder Equity', recent_year] if 'Net Income' in income_stmt.index and 'Total Stockholder Equity' in balance_sheet.index else None,
                'return_on_assets': income_stmt.loc['Net Income', recent_year] / balance_sheet.loc['Total Assets', recent_year] if 'Net Income' in income_stmt.index and 'Total Assets' in balance_sheet.index else None
            }
        except Exception as e:
            print(f"Error calculating profitability metrics: {e}")
            metrics = {
                'gross_margin': None,
                'operating_margin': None,
                'net_margin': None,
                'return_on_equity': None,
                'return_on_assets': None
            }

        return metrics

    def dividend_analysis(self):
        """Analyze dividend information."""
        dividends = self.ticker.dividends

        dividend_yield = self.info.get('dividendYield')
        analysis = {
            'dividend_yield': dividend_yield / 100 if dividend_yield is not None else None,
            'dividend_growth_5y': self.info.get('dividendGrowth5y'),
            'payout_ratio': self.info.get('payoutRatio'),
            'has_dividends': not dividends.empty,
            'last_dividend': dividends.iloc[-1] if not dividends.empty else 0,
            'dividend_frequency': self._estimate_dividend_frequency(dividends)
        }

        return analysis

    def _estimate_dividend_frequency(self, dividends):
        """Estimate dividend payment frequency."""
        if dividends.empty or len(dividends) < 2:
            return "Unknown"

        # Calculate average days between payments
        dates = dividends.index.sort_values()
        if len(dates) > 1:
            avg_days = (dates[-1] - dates[0]).days / (len(dates) - 1)
            if avg_days < 40:
                return "Quarterly"
            elif avg_days < 100:
                return "Semi-Annual"
            else:
                return "Annual"
        return "Unknown"

    def dcf_valuation(self, discount_rate=0.08, perpetual_growth=0.02):
        """Simplified DCF valuation."""
        try:
            cash_flow = self.ticker.cash_flow
            balance_sheet = self.ticker.balance_sheet

            if cash_flow.empty or balance_sheet.empty:
                return None

            # Get most recent free cash flow
            if 'Free Cash Flow' in cash_flow.index:
                fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
            else:
                # Estimate FCF if not directly available
                operating_cash_flow = cash_flow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow.index else 0
                cap_ex = cash_flow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cash_flow.index else 0
                fcf = operating_cash_flow + cap_ex  # CapEx is typically negative

            # Forecast future cash flows
            forecast_years = 5
            future_cash_flows = []

            for year in range(1, forecast_years + 1):
                future_fcf = fcf * (1 + perpetual_growth) ** year
                future_cash_flows.append(future_fcf / (1 + discount_rate) ** year)

            # Terminal value
            terminal_value = (future_cash_flows[-1] * (1 + perpetual_growth)) / (discount_rate - perpetual_growth)
            terminal_value_discounted = terminal_value / (1 + discount_rate) ** forecast_years

            # Total enterprise value
            enterprise_value = sum(future_cash_flows) + terminal_value_discounted

            # Adjust for cash and debt
            cash = balance_sheet.loc['Cash'].iloc[0] if 'Cash' in balance_sheet.index else 0
            debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0

            equity_value = enterprise_value - debt + cash
            shares_outstanding = self.info.get('sharesOutstanding')

            if shares_outstanding:
                fair_value = equity_value / shares_outstanding
                return fair_value
        except Exception as e:
            print(f"Error in DCF valuation: {e}")

        return None

    def default_peers(self, count=4):
        """Large-cap peers from the same sector, excluding this stock. Yahoo
        Finance has no peer-lookup endpoint, so this uses a small curated map."""
        candidates = SECTOR_PEERS.get(self.info.get('sector'), [])
        return [t for t in candidates if t != self.symbol.upper()][:count]

    def relative_valuation(self, comparable_tickers):
        """Compare valuation multiples with peer companies. Peers are fetched
        concurrently; a peer that fails becomes an error marker rather than
        breaking the comparison. Includes the median of the peers that loaded."""
        base_metrics = self.calculate_valuation_ratios()

        def fetch_peer(peer):
            try:
                return StockAnalysis(peer).calculate_valuation_ratios()
            except Exception as e:
                print(f"Error fetching data for peer {peer}: {e}")
                return "Error fetching data"

        peers = list(comparable_tickers)
        with ThreadPoolExecutor(max_workers=max(1, min(len(peers), 5))) as pool:
            peer_metrics = dict(zip(peers, pool.map(fetch_peer, peers)))

        loaded = [m for m in peer_metrics.values() if isinstance(m, dict)]
        peer_median = {}
        for metric in base_metrics:
            values = [m[metric] for m in loaded
                      if isinstance(m.get(metric), (int, float)) and not isinstance(m.get(metric), bool)]
            peer_median[metric] = float(np.median(values)) if values else None

        return {
            'base_company': base_metrics,
            'peers': peer_metrics,
            'peer_median': peer_median,
        }

    def comprehensive_analysis(self, peers=None):
        """Perform comprehensive analysis of the stock. `peers` is a list of
        comparable tickers; None picks same-sector defaults, [] skips the comparison."""
        self.fetch_data()

        self.analysis_results = {
            'basic_info': {
                'name': self.info.get('longName', self.symbol),
                'sector': self.info.get('sector'),
                'industry': self.info.get('industry'),
                'market_cap': self.info.get('marketCap'),
                'current_price': self.info.get('regularMarketPrice'),
                '52_week_high': self.info.get('fiftyTwoWeekHigh'),
                '52_week_low': self.info.get('fiftyTwoWeekLow')
            },
            'performance_metrics': self.calculate_performance_metrics(),
            'technical_indicators': self.calculate_technical_indicators(),
            'valuation_ratios': self.calculate_valuation_ratios(),
            'profitability_metrics': self.calculate_profitability_metrics(),
            'dividend_analysis': self.dividend_analysis(),
            'dcf_valuation': self.dcf_valuation(),
            'analyst_data': {
                'recommendation': self.info.get('recommendationKey'),
                'target_price': self.info.get('targetMeanPrice'),
                'number_of_analysts': self.info.get('numberOfAnalystOpinions')
            }
        }

        if peers is None:
            peers = self.default_peers()
        if peers:
            self.analysis_results['relative_valuation'] = self.relative_valuation(peers)

        return self.analysis_results
