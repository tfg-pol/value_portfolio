"""Tests for `InMemoryMarketData`: the read surface, the monotonic clock, the
simulator-facing helpers, and look-ahead prevention.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.data import InMemoryMarketData, SymbolNotAvailableError
from value_portfolio.data.types import Bar


def _make_bars(
    symbol: str,
    prices: list[float],
    *,
    start: datetime | None = None,
    timeframe: str = "1Day",
) -> list[Bar]:
    start = start or datetime(2026, 1, 1, tzinfo=UTC)
    bars: list[Bar] = []
    for i, close in enumerate(prices):
        ts = start + timedelta(days=i)
        c = Decimal(str(close))
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts,
                open=c,
                high=c,
                low=c,
                close=c,
                volume=Decimal("1000"),
                timeframe=timeframe,
            )
        )
    return bars


@pytest.fixture
def data() -> InMemoryMarketData:
    return InMemoryMarketData(
        {
            "AAPL": _make_bars("AAPL", [100.0, 101.0, 102.0, 103.0, 104.0]),
            "MSFT": _make_bars("MSFT", [200.0, 200.0, 200.0, 200.0, 200.0]),
        }
    )


class TestConstruction:
    def test_empty_bars_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one symbol"):
            InMemoryMarketData({})

    def test_symbols_exposed_as_frozenset(self, data: InMemoryMarketData) -> None:
        assert data.symbols == frozenset({"AAPL", "MSFT"})

    def test_clock_starts_at_earliest_bar(self, data: InMemoryMarketData) -> None:
        assert data.now == datetime(2026, 1, 1, tzinfo=UTC)


class TestTimeline:
    def test_returns_sorted_distinct_timestamps(self, data: InMemoryMarketData) -> None:
        expected = tuple(datetime(2026, 1, d, tzinfo=UTC) for d in range(1, 6))
        assert data.timeline == expected

    def test_not_truncated_by_clock(self, data: InMemoryMarketData) -> None:
        # The timeline is the trading calendar, not market data — advancing
        # the clock must not shorten it.
        full = data.timeline
        data.advance_to(datetime(2026, 1, 3, tzinfo=UTC))
        assert data.timeline == full

    def test_merges_misaligned_symbols(self) -> None:
        data = InMemoryMarketData(
            {
                "AAPL": _make_bars("AAPL", [100.0, 101.0]),
                "MSFT": _make_bars("MSFT", [200.0], start=datetime(2026, 1, 3, tzinfo=UTC)),
            }
        )
        assert data.timeline == (
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
        )


class TestClock:
    def test_advance_forward(self, data: InMemoryMarketData) -> None:
        target = datetime(2026, 1, 3, tzinfo=UTC)
        data.advance_to(target)
        assert data.now == target

    def test_advance_backwards_rejected(self, data: InMemoryMarketData) -> None:
        data.advance_to(datetime(2026, 1, 3, tzinfo=UTC))
        with pytest.raises(ValueError, match="cannot advance backwards"):
            data.advance_to(datetime(2026, 1, 1, tzinfo=UTC))

    def test_reset_restores_initial_clock(self, data: InMemoryMarketData) -> None:
        data.advance_to(datetime(2026, 1, 4, tzinfo=UTC))
        data.reset()
        assert data.now == datetime(2026, 1, 1, tzinfo=UTC)


class TestGetQuote:
    def test_returns_latest_visible_bar(self, data: InMemoryMarketData) -> None:
        data.advance_to(datetime(2026, 1, 3, tzinfo=UTC))
        quote = data.get_quote("AAPL")
        assert quote.symbol == "AAPL"
        assert quote.bid_price == Decimal("102")  # day-3 close
        assert quote.ask_price == Decimal("102")

    def test_unknown_symbol_raises(self, data: InMemoryMarketData) -> None:
        with pytest.raises(SymbolNotAvailableError):
            data.get_quote("ZZZZ")


class TestGetBars:
    def test_truncates_to_now(self, data: InMemoryMarketData) -> None:
        # Clock at day 1 by default — only day-1 bar should be visible.
        bars = data.get_bars(
            "AAPL",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 10, tzinfo=UTC),
            timeframe="1Day",
        )
        assert len(bars) == 1
        assert bars[0].timestamp == datetime(2026, 1, 1, tzinfo=UTC)

    def test_visible_after_advance(self, data: InMemoryMarketData) -> None:
        data.advance_to(datetime(2026, 1, 3, tzinfo=UTC))
        bars = data.get_bars(
            "AAPL",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 10, tzinfo=UTC),
            timeframe="1Day",
        )
        assert [b.timestamp for b in bars] == [
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 3, tzinfo=UTC),
        ]

    def test_respects_start_filter(self, data: InMemoryMarketData) -> None:
        data.advance_to(datetime(2026, 1, 5, tzinfo=UTC))
        bars = data.get_bars(
            "AAPL",
            start=datetime(2026, 1, 3, tzinfo=UTC),
            end=datetime(2026, 1, 10, tzinfo=UTC),
            timeframe="1Day",
        )
        assert [b.timestamp for b in bars] == [
            datetime(2026, 1, 3, tzinfo=UTC),
            datetime(2026, 1, 4, tzinfo=UTC),
            datetime(2026, 1, 5, tzinfo=UTC),
        ]

    def test_unknown_symbol_raises(self, data: InMemoryMarketData) -> None:
        with pytest.raises(SymbolNotAvailableError):
            data.get_bars(
                "ZZZZ",
                start=datetime(2026, 1, 1, tzinfo=UTC),
                end=datetime(2026, 1, 10, tzinfo=UTC),
                timeframe="1Day",
            )


class TestSimulatorHelpers:
    def test_latest_bar_after_advance(self, data: InMemoryMarketData) -> None:
        data.advance_to(datetime(2026, 1, 3, tzinfo=UTC))
        bar = data.latest_bar("AAPL")
        assert bar is not None
        assert bar.timestamp == datetime(2026, 1, 3, tzinfo=UTC)

    def test_latest_bar_unknown_symbol(self, data: InMemoryMarketData) -> None:
        assert data.latest_bar("ZZZZ") is None

    def test_next_bar_after_returns_visible_bar(self, data: InMemoryMarketData) -> None:
        data.advance_to(datetime(2026, 1, 2, tzinfo=UTC))
        bar = data.next_bar_after("AAPL", datetime(2026, 1, 1, tzinfo=UTC))
        assert bar is not None
        assert bar.timestamp == datetime(2026, 1, 2, tzinfo=UTC)

    def test_next_bar_after_hides_future(self, data: InMemoryMarketData) -> None:
        # Clock still at day-1; next bar after day-1 would be day-2, not yet visible.
        bar = data.next_bar_after("AAPL", datetime(2026, 1, 1, tzinfo=UTC))
        assert bar is None

    def test_next_bar_after_runs_out(self, data: InMemoryMarketData) -> None:
        data.advance_to(datetime(2026, 1, 5, tzinfo=UTC))
        # day-5 is the last bar; nothing strictly after it.
        bar = data.next_bar_after("AAPL", datetime(2026, 1, 5, tzinfo=UTC))
        assert bar is None
