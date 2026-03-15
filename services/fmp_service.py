"""
FinancialModelingPrep API Service

Handles all interactions with the FMP API for fetching financial data.
Adapted from the C# StockSimulation project architecture.
"""

import os
import httpx
from datetime import date, datetime
from typing import Optional

from schemas import Company, StockPrice


class FMPService:
    """Service for interacting with FinancialModelingPrep API."""

    BASE_URL = "https://financialmodelingprep.com/stable"

    # Free tier allowed companies (matching C# implementation)
    FREE_TIER_ALLOWED_SYMBOLS = {
        "AAPL", "TSLA", "AMZN", "MSFT", "NVDA", "GOOGL", "GOOG", "META",
        "NFLX", "JPM", "V", "BAC", "AMD", "INTC", "WMT", "DIS", "PYPL",
        "ADBE", "CRM", "CSCO", "PEP", "KO", "NKE", "MCD", "COST"
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP Service.

        Args:
            api_key: FMP API key. If not provided, reads from FMP_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FMP API key is required. Set FMP_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=30.0
        )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def _add_api_key(self, url: str) -> str:
        """Append API key to URL query string."""
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}apikey={self.api_key}"

    def _is_allowed_symbol(self, symbol: str) -> bool:
        """Check if symbol is in free tier allowed list."""
        return symbol.upper() in self.FREE_TIER_ALLOWED_SYMBOLS

    async def search_by_symbol(self, symbol: str) -> list[Company]:
        """
        Search for companies by stock ticker symbol.

        Args:
            symbol: Stock ticker symbol to search for.

        Returns:
            List of matching Company objects.
        """
        try:
            url = self._add_api_key(f"/search-symbol?query={symbol}")
            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()
            companies = []

            for item in data:
                # Filter by free tier allowed symbols
                item_symbol = item.get("symbol", "")
                if not self._is_allowed_symbol(item_symbol):
                    continue

                companies.append(Company(
                    symbol=item_symbol,
                    name=item.get("name", ""),
                    currency=item.get("currency"),
                    exchange=item.get("exchange"),
                    exchange_full_name=item.get("exchangeFullName")
                ))

            return companies

        except httpx.HTTPError as e:
            print(f"Error searching by symbol: {e}")
            return []

    async def search_by_name(self, name: str) -> list[Company]:
        """
        Search for companies by company name.

        Args:
            name: Company name to search for.

        Returns:
            List of matching Company objects.
        """
        try:
            url = self._add_api_key(f"/search-name?query={name}")
            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()
            companies = []

            for item in data:
                # Filter by free tier allowed symbols
                item_symbol = item.get("symbol", "")
                if not self._is_allowed_symbol(item_symbol):
                    continue

                companies.append(Company(
                    symbol=item_symbol,
                    name=item.get("name", ""),
                    currency=item.get("currency"),
                    exchange=item.get("exchange"),
                    exchange_full_name=item.get("exchangeFullName")
                ))

            return companies

        except httpx.HTTPError as e:
            print(f"Error searching by name: {e}")
            return []

    async def get_historical_prices(
        self,
        symbol: str,
        from_date: Optional[date] = None,
        years: int = 2
    ) -> list[StockPrice]:
        """
        Fetch historical end-of-day stock prices.

        Args:
            symbol: Stock ticker symbol.
            from_date: Start date for historical data. Defaults to 'years' ago.
            years: Number of years of data to fetch if from_date not specified.

        Returns:
            List of StockPrice objects sorted by date ascending.
        """
        try:
            # Calculate from_date if not provided
            if from_date is None:
                today = date.today()
                from_date = date(today.year - years, today.month, today.day)

            date_str = from_date.strftime("%Y-%m-%d")
            url = self._add_api_key(
                f"/historical-price-eod/full?symbol={symbol.upper()}&from={date_str}"
            )

            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()

            if not data:
                return []

            prices = []
            for item in data:
                try:
                    # Parse date string to date object
                    price_date = datetime.strptime(
                        item.get("date", ""), "%Y-%m-%d"
                    ).date()

                    prices.append(StockPrice(
                        symbol=item.get("symbol", symbol.upper()),
                        date=price_date,
                        open=float(item.get("open", 0)),
                        high=float(item.get("high", 0)),
                        low=float(item.get("low", 0)),
                        close=float(item.get("close", 0)),
                        volume=int(item.get("volume", 0)),
                        change=item.get("change"),
                        change_percent=item.get("changePercent"),
                        vwap=item.get("vwap")
                    ))
                except (ValueError, KeyError) as e:
                    print(f"Error parsing price data: {e}")
                    continue

            # Sort by date ascending
            prices.sort(key=lambda p: p.date)

            return prices

        except httpx.HTTPError as e:
            print(f"Error fetching historical prices: {e}")
            return []

    async def get_company_info(self, symbol: str) -> Optional[Company]:
        """
        Get company information for a specific symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Company object if found, None otherwise.
        """
        companies = await self.search_by_symbol(symbol)

        # Find exact match
        symbol_upper = symbol.upper()
        for company in companies:
            if company.symbol.upper() == symbol_upper:
                return company

        return companies[0] if companies else None
