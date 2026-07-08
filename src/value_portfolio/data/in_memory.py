
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal

from value_portfolio.data.base import MarketDataSource
from value_portfolio.data.exceptions import SymbolNotAvailableError
from value_portfolio.data.types import Bar, Quote

_ZERO = Decimal("0")


class InMemoryMarketData(MarketDataSource):

    def __init__(self, bars: Mapping[str, Sequence[Bar]]) -> None:
        if not bars:
            raise ValueError("InMemoryMarketData requires at least one symbol")
        self._bars: dict[str, list[Bar]] = {
            symbol: sorted(symbol_bars, key=lambda b: b.timestamp)
            for symbol, symbol_bars in bars.items()
        }
        self._timestamps: dict[str, list[datetime]] = {
            symbol: [b.timestamp for b in symbol_bars] for symbol, symbol_bars in self._bars.items()
        }
        self._symbols: frozenset[str] = frozenset(self._bars.keys())
        self._initial_clock: datetime = self._earliest_timestamp()
        self._clock: datetime = self._initial_clock

    # Clock control

    @property
    def now(self) -> datetime:
        return self._clock

    @property
    def symbols(self) -> frozenset[str]:
        return self._symbols

    @property
    def timeline(self) -> tuple[datetime, ...]:
        merged = {ts for symbol_ts in self._timestamps.values() for ts in symbol_ts}
        return tuple(sorted(merged))

    def advance_to(self, timestamp: datetime) -> None:
        if timestamp < self._clock:
            raise ValueError(f"cannot advance backwards: now={self._clock}, target={timestamp}")
        self._clock = timestamp

    def reset(self) -> None:
        self._clock = self._initial_clock

    def get_quote(self, symbol: str) -> Quote:
        if symbol not in self._symbols:
            raise SymbolNotAvailableError(f"no bars loaded for symbol {symbol!r}")
        bar = self.latest_bar(symbol)
        if bar is None:
            raise SymbolNotAvailableError(f"no bar available for {symbol!r} as of {self._clock}")
        # Bars carry no bid/ask; quote the close on both sides.
        return Quote(
            symbol=symbol,
            bid_price=bar.close,
            ask_price=bar.close,
            bid_size=_ZERO,
            ask_size=_ZERO,
            timestamp=bar.timestamp,
        )

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> list[Bar]:
        del timeframe  # bars are stored at one timeframe; trust the caller
        if symbol not in self._symbols:
            raise SymbolNotAvailableError(f"no bars loaded for symbol {symbol!r}")
        cutoff = min(end, self._clock)
        return [b for b in self._bars[symbol] if start <= b.timestamp <= cutoff]


    def latest_bar(self, symbol: str) -> Bar | None:
        symbol_bars = self._bars.get(symbol)
        if not symbol_bars:
            return None
        idx = bisect_right(self._timestamps[symbol], self._clock) - 1
        if idx < 0:
            return None
        return symbol_bars[idx]

    def is_delisted(self, symbol: str) -> bool:
        symbol_ts = self._timestamps.get(symbol)
        if not symbol_ts:
            return False
        return symbol_ts[-1] < self._clock

    def next_bar_after(self, symbol: str, after: datetime) -> Bar | None:
        """Next bar strictly after `after`, but only if already visible (<= now)."""
        symbol_bars = self._bars.get(symbol)
        if not symbol_bars:
            return None
        idx = bisect_right(self._timestamps[symbol], after)
        if idx >= len(symbol_bars):
            return None
        bar = symbol_bars[idx]
        if bar.timestamp > self._clock:
            return None
        return bar

    # Internals

    def _earliest_timestamp(self) -> datetime:
        timestamps = [ts[0] for ts in self._timestamps.values() if ts]
        if not timestamps:
            raise ValueError("at least one symbol must have bars")
        return min(timestamps)
