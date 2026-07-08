"""Unit tests for market-data domain types — pure-Python, no network."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from value_portfolio.data.types import Bar, Quote


class TestQuote:
    def test_is_frozen(self) -> None:
        q = Quote(
            symbol="AAPL",
            bid_price=Decimal("100"),
            ask_price=Decimal("100.05"),
            bid_size=Decimal("10"),
            ask_size=Decimal("12"),
            timestamp=datetime.now(UTC),
        )
        with pytest.raises(FrozenInstanceError):
            q.bid_price = Decimal("999")  # type: ignore[misc]

    def test_preserves_decimal_precision(self) -> None:
        bid = Decimal("123.456789")
        q = Quote(
            symbol="AAPL",
            bid_price=bid,
            ask_price=Decimal("123.456790"),
            bid_size=Decimal("1"),
            ask_size=Decimal("1"),
            timestamp=datetime.now(UTC),
        )
        assert q.bid_price == bid
        assert q.ask_price - q.bid_price == Decimal("0.000001")


class TestBar:
    def test_construction(self) -> None:
        b = Bar(
            symbol="AAPL",
            timestamp=datetime.now(UTC),
            open=Decimal("150"),
            high=Decimal("152"),
            low=Decimal("149"),
            close=Decimal("151"),
            volume=Decimal("100000"),
            timeframe="1Day",
        )
        assert b.high >= b.low
        assert b.timeframe == "1Day"
