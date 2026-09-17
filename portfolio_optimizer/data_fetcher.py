"""Fetches and caches stock/market data from Yahoo Finance."""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from .cache import price_cache, risk_free_rate_cache


class StockDataFetcher:
    """
    A class to fetch and process stock data from Yahoo Finance API.

    Methods:
        get_historical_data: Fetch historical price data for a single symbol
        get_multiple_stocks: Fetch historical price data for multiple symbols
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
        # Cache key rounded to the day: with a '1d' interval the data for a
        # given day doesn't change once the market has closed, so repeated
        # page views within the TTL window can reuse the same fetch.
        cache_key = ('single', symbol, start_date.date(), end_date.date(), interval)
        cached = price_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

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

            closes = df['Close']
            price_cache.set(cache_key, closes)
            return closes.copy()
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
        # Cache key rounded to the day, same rationale as get_historical_data:
        # avoids re-fetching the same batch of symbols on every page view.
        cache_key = ('multiple', tuple(sorted(symbols)), start_date.date(), end_date.date(), interval)
        cached = price_cache.get(cache_key)
        if cached is not None:
            print(f"Using cached data for {len(symbols)} symbols: {symbols}")
            return cached.copy()

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # Fetch all symbols in a single batched request instead of one
        # request per symbol - much faster and avoids hammering the API.
        print(f"Fetching data for {len(symbols)} symbols: {symbols}")
        raw = yf.download(symbols, start=start_str, end=end_str, interval=interval,
                           group_by='ticker', auto_adjust=True, progress=False)

        data = {}
        successful_symbols = []

        if not raw.empty:
            for symbol in symbols:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        closes = raw[symbol]['Close']
                    else:
                        # Only one symbol was requested, so columns aren't
                        # nested per-ticker.
                        closes = raw['Close']

                    closes = closes.dropna()
                    if closes.empty:
                        raise ValueError(f"No data found for {symbol}")

                    data[symbol] = closes
                    successful_symbols.append(symbol)
                    print(f"✓ Successfully fetched data for {symbol}")
                except (KeyError, ValueError) as e:
                    print(f"✗ Failed to fetch data for {symbol}: {str(e)}")
                    continue

        # Check if any data was fetched
        if not data:
            raise ValueError("No data was successfully fetched for any symbol")

        # Create DataFrame and align dates
        df = pd.DataFrame(data)
        df = df.dropna()  # Remove rows with missing values

        print(f"\nSuccessfully retrieved data for {len(successful_symbols)} symbols: {successful_symbols}")
        price_cache.set(cache_key, df)
        return df.copy()

    def validate_symbols(self, symbols):
        """
        Check which of the given ticker symbols actually return price data,
        without fetching a full history. Meant to catch typos/invalid
        tickers immediately (e.g. when a portfolio is created or edited),
        rather than only failing later when the portfolio is analyzed.

        Args:
            symbols (list): List of stock ticker symbols

        Returns:
            tuple: (valid_symbols, invalid_symbols), each preserving the
                order of the input list
        """
        if not symbols:
            return [], []

        # A short window is enough to confirm a symbol exists and is
        # actively trading - no need for the full history used elsewhere.
        cache_key = ('validate', tuple(sorted(symbols)), datetime.now().date())
        cached = price_cache.get(cache_key)
        if cached is not None:
            return cached

        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        raw = yf.download(symbols, start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'), interval='1d',
                           group_by='ticker', auto_adjust=True, progress=False)

        valid_symbols = []
        invalid_symbols = []
        for symbol in symbols:
            try:
                if raw.empty:
                    raise ValueError("No data returned")

                if isinstance(raw.columns, pd.MultiIndex):
                    closes = raw[symbol]['Close']
                else:
                    # Only one symbol was requested, so columns aren't
                    # nested per-ticker.
                    closes = raw['Close']

                if closes.dropna().empty:
                    raise ValueError("No data found")

                valid_symbols.append(symbol)
            except (KeyError, ValueError):
                invalid_symbols.append(symbol)

        result = (valid_symbols, invalid_symbols)
        price_cache.set(cache_key, result)
        return result

    def get_risk_free_rate(self):
        """
        Get the most recent annualized risk-free rate from 3-month T-bills.

        Returns:
            float: Annual risk-free rate as a decimal, or None if unavailable
        """
        cache_key = ('risk_free_rate', datetime.now().date())
        cached = risk_free_rate_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            # Download 3-month US Treasury bill rates
            annualized = yf.download("^IRX", period="1mo", auto_adjust=True)['Close']

            if annualized.empty:
                raise ValueError("No data returned from Yahoo Finance")

            # ^IRX is quoted as an annualized percentage (e.g. 5.25 for 5.25%).
            # Every consumer of this value (Sharpe/Treynor/Jensen's alpha,
            # tangency weights) compares it against annualized returns, so it
            # must stay annual rather than being deannualized to a daily rate.
            annual_rate = annualized.iloc[-1].iloc[-1] / 100
            risk_free_rate_cache.set(cache_key, annual_rate)

        except Exception as e:
            print(f"Error fetching risk-free rate: {e}")
            return None

        return annual_rate
