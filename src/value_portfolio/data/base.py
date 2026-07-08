from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from value_portfolio.data.types import Bar, Quote


class MarketDataSource(ABC):
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Most recent quote (real, or synthesized from a bar close)."""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> list[Bar]:
        """OHLCV bars for ``[start, end]`` at the given timeframe."""
