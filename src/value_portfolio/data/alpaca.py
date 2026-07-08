
from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from typing import Any

from value_portfolio.config import AlpacaSettings
from value_portfolio.data.base import MarketDataSource
from value_portfolio.data.exceptions import MarketDataError, SymbolNotAvailableError
from value_portfolio.data.types import Bar, Quote

_TIMEFRAME_RE = re.compile(r"^(\d+)(Min|Hour|Day|Week|Month)$")


class AlpacaMarketData(MarketDataSource):

    def __init__(self, settings: AlpacaSettings | None = None) -> None:
        from alpaca.data.historical import StockHistoricalDataClient

        self._settings = settings or AlpacaSettings()  # type: ignore[call-arg]
        self._client = StockHistoricalDataClient(
            api_key=self._settings.api_key,
            secret_key=self._settings.api_secret,
        )

    def get_quote(self, symbol: str) -> Quote:
        from alpaca.data.requests import StockLatestQuoteRequest

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        with _translate_errors():
            response = self._client.get_stock_latest_quote(request)
        if symbol not in response:
            raise SymbolNotAvailableError(f"no quote available for {symbol!r}")
        raw = response[symbol]
        return Quote(
            symbol=symbol,
            bid_price=Decimal(str(raw.bid_price)),
            ask_price=Decimal(str(raw.ask_price)),
            bid_size=Decimal(str(raw.bid_size)),
            ask_size=Decimal(str(raw.ask_size)),
            timestamp=raw.timestamp,
        )

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> list[Bar]:
        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest

        # Split- and dividend-adjusted bars (Alpaca defaults to unadjusted, under
        # which a split looks like a price crash). ALL gives a total-return series.
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_parse_timeframe(timeframe),
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
        )
        with _translate_errors():
            # alpaca-py types this as `BarSet | dict`; we never use raw mode.
            response: Any = self._client.get_stock_bars(request)
        rows = response.data.get(symbol, [])
        return [
            Bar(
                symbol=symbol,
                timestamp=r.timestamp,
                open=Decimal(str(r.open)),
                high=Decimal(str(r.high)),
                low=Decimal(str(r.low)),
                close=Decimal(str(r.close)),
                volume=Decimal(str(r.volume)),
                timeframe=timeframe,
            )
            for r in rows
        ]


def _parse_timeframe(timeframe: str) -> Any:
    """Convert ``"1Day"`` / ``"5Min"`` etc. to ``alpaca.data.TimeFrame``."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    match = _TIMEFRAME_RE.match(timeframe)
    if not match:
        raise ValueError(
            f"Unrecognized timeframe {timeframe!r}; expected '<int><Min|Hour|Day|Week|Month>'"
        )
    amount = int(match.group(1))
    unit = TimeFrameUnit(match.group(2))
    return TimeFrame(amount, unit)


class _translate_errors:
    """Re-raise Alpaca SDK errors as `MarketDataError` subclasses."""

    def __enter__(self) -> _translate_errors:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        if exc is None:
            return
        from alpaca.common.exceptions import APIError

        if not isinstance(exc, APIError):
            return
        message = str(exc).lower()
        if "not found" in message or ("asset" in message and "invalid" in message):
            raise SymbolNotAvailableError(str(exc)) from exc
        raise MarketDataError(str(exc)) from exc
