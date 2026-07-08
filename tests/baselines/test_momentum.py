"""Tests for `Momentum.decide`: cadence, top-K ranking, warm-up, and
look-ahead safety against the data clock.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.agent import DecisionContext
from value_portfolio.baselines import Momentum
from value_portfolio.broker import SimulatedBroker
from value_portfolio.data import InMemoryMarketData
from value_portfolio.data.types import Bar

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _context(broker: SimulatedBroker, data: InMemoryMarketData) -> DecisionContext:
    """Build the read-only context the agent sees as of the data clock."""
    return DecisionContext(now=data.now, account=broker.get_account(), data=data)


def _bars(symbol: str, closes: list[float]) -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            timestamp=_START + timedelta(days=i),
            open=Decimal(str(c)),
            high=Decimal(str(c)),
            low=Decimal(str(c)),
            close=Decimal(str(c)),
            volume=Decimal("1000"),
            timeframe="1Day",
        )
        for i, c in enumerate(closes)
    ]


def _ctx(bars_by_symbol: dict[str, list[Bar]]) -> tuple[SimulatedBroker, InMemoryMarketData]:
    data = InMemoryMarketData(bars_by_symbol)
    return SimulatedBroker(market_data=data), data


class TestConstruction:
    def test_rejects_empty_symbols(self) -> None:
        with pytest.raises(ValueError):
            Momentum([])

    def test_rejects_duplicate_symbols(self) -> None:
        with pytest.raises(ValueError):
            Momentum(["AAPL", "AAPL"])

    def test_rejects_non_positive_lookback(self) -> None:
        with pytest.raises(ValueError):
            Momentum(["AAPL"], lookback=0)

    def test_rejects_negative_skip(self) -> None:
        with pytest.raises(ValueError):
            Momentum(["AAPL"], skip=-1)

    def test_rejects_non_positive_rebalance_every(self) -> None:
        with pytest.raises(ValueError):
            Momentum(["AAPL"], rebalance_every=0)

    def test_rejects_top_k_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            Momentum(["AAPL", "MSFT"], top_k=3)
        with pytest.raises(ValueError):
            Momentum(["AAPL"], top_k=0)


class TestDecide:
    def test_returns_none_during_warmup(self) -> None:
        # lookback needs at least 4 bars; only 2 visible at NOW.
        bars = {"AAPL": _bars("AAPL", [100, 100])}
        broker, data = _ctx(bars)
        agent = Momentum(["AAPL"], lookback=3, top_k=1, rebalance_every=1)

        data.advance_to(_START + timedelta(days=1))
        assert agent.decide(_context(broker, data)) is None

    def test_picks_top_k_by_past_return(self) -> None:
        # Over 3 bars: AAPL +20 %, MSFT +5 %, GOOG -10 %. Top-2 -> AAPL, MSFT.
        bars = {
            "AAPL": _bars("AAPL", [100, 110, 120]),
            "MSFT": _bars("MSFT", [100, 102, 105]),
            "GOOG": _bars("GOOG", [100, 95, 90]),
        }
        broker, data = _ctx(bars)
        agent = Momentum(["AAPL", "MSFT", "GOOG"], lookback=2, top_k=2, rebalance_every=1)

        data.advance_to(_START + timedelta(days=2))
        weights = agent.decide(_context(broker, data))

        half = Decimal(1) / Decimal(2)
        assert weights == {"AAPL": half, "MSFT": half}

    def test_skip_drops_most_recent_bar(self) -> None:
        # With skip=1, the most recent bar is ignored. AAPL went 100 -> 50
        # in the last bar, but its 100 -> 110 trajectory over the prior
        # window is what gets ranked.
        bars = {
            "AAPL": _bars("AAPL", [100, 110, 50]),
            "MSFT": _bars("MSFT", [100, 101, 200]),
        }
        broker, data = _ctx(bars)
        agent = Momentum(["AAPL", "MSFT"], lookback=1, skip=1, top_k=1, rebalance_every=1)

        data.advance_to(_START + timedelta(days=2))
        weights = agent.decide(_context(broker, data))

        assert weights == {"AAPL": Decimal(1)}

    def test_rebalance_every_skips_intermediate_steps(self) -> None:
        bars = {"AAPL": _bars("AAPL", [100, 110, 120, 130, 140])}
        broker, data = _ctx(bars)
        agent = Momentum(["AAPL"], lookback=1, top_k=1, rebalance_every=2)

        data.advance_to(_START + timedelta(days=4))
        assert agent.decide(_context(broker, data)) is not None  # step 0
        assert agent.decide(_context(broker, data)) is None  # step 1
        assert agent.decide(_context(broker, data)) is not None  # step 2

    def test_excludes_symbols_without_enough_history(self) -> None:
        # AAPL has 3 bars, MSFT has 1. lookback=2 -> only AAPL ranked.
        bars = {
            "AAPL": _bars("AAPL", [100, 110, 120]),
            "MSFT": _bars("MSFT", [100]),
        }
        broker, data = _ctx(bars)
        agent = Momentum(["AAPL", "MSFT"], lookback=2, top_k=1, rebalance_every=1)

        data.advance_to(_START + timedelta(days=2))
        weights = agent.decide(_context(broker, data))

        assert weights == {"AAPL": Decimal(1)}

    def test_does_not_peek_past_the_clock(self) -> None:
        # All five bars exist, but the clock is at index 1. Only AAPL's
        # first two closes are visible -> lookback=2 cannot be satisfied
        # and the agent must return None, not silently use future bars.
        bars = {"AAPL": _bars("AAPL", [100, 90, 200, 300, 400])}
        broker, data = _ctx(bars)
        agent = Momentum(["AAPL"], lookback=2, top_k=1, rebalance_every=1)

        data.advance_to(_START + timedelta(days=1))
        assert agent.decide(_context(broker, data)) is None
