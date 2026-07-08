"""Market-data exceptions."""

from __future__ import annotations


class MarketDataError(Exception):
    """Base class for market-data adapter failures."""


class SymbolNotAvailableError(MarketDataError):
    """No data available for the requested symbol."""
