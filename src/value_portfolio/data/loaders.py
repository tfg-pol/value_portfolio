
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from value_portfolio.config import AlpacaSettings
from value_portfolio.data.alpaca import AlpacaMarketData
from value_portfolio.data.in_memory import InMemoryMarketData
from value_portfolio.data.types import Bar


def load_bars_from_alpaca(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    timeframe: str = "1Day",
    settings: AlpacaSettings | None = None,
) -> InMemoryMarketData:

    if not symbols:
        raise ValueError("load_bars_from_alpaca requires at least one symbol")

    source = AlpacaMarketData(settings)
    bars: dict[str, list[Bar]] = {}
    for symbol in symbols:
        symbol_bars = source.get_bars(symbol, start, end, timeframe)
        if not symbol_bars:
            raise ValueError(
                f"Alpaca returned no {timeframe} bars for {symbol!r} "
                f"over [{start.isoformat()}, {end.isoformat()}]"
            )
        bars[symbol] = symbol_bars

    return InMemoryMarketData(bars)
