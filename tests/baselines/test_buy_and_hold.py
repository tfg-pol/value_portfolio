"""Tests for `BuyAndHold.decide`: equal weights once, then ``None``."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from value_portfolio.agent import DecisionContext
from value_portfolio.baselines import BuyAndHold
from value_portfolio.broker import SimulatedBroker
from value_portfolio.data import InMemoryMarketData
from value_portfolio.data.types import Bar

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _ctx() -> DecisionContext:
    """A throwaway context; buy-and-hold ignores all observations."""
    bar = Bar(
        symbol="AAPL",
        timestamp=_NOW,
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        volume=Decimal("1000"),
        timeframe="1Day",
    )
    data = InMemoryMarketData({"AAPL": [bar]})
    broker = SimulatedBroker(market_data=data)
    return DecisionContext(now=_NOW, account=broker.get_account(), data=data)


class TestConstruction:
    def test_rejects_empty_symbols(self) -> None:
        with pytest.raises(ValueError):
            BuyAndHold([])

    def test_rejects_duplicate_symbols(self) -> None:
        with pytest.raises(ValueError):
            BuyAndHold(["AAPL", "AAPL"])


class TestDecide:
    def test_first_call_returns_equal_weights(self) -> None:
        ctx = _ctx()
        agent = BuyAndHold(["AAPL", "MSFT", "GOOG"])

        weights = agent.decide(ctx)

        third = Decimal(1) / Decimal(3)
        assert weights == {"AAPL": third, "MSFT": third, "GOOG": third}

    def test_subsequent_calls_return_none(self) -> None:
        ctx = _ctx()
        agent = BuyAndHold(["AAPL"])

        assert agent.decide(ctx) is not None
        assert agent.decide(ctx) is None
        assert agent.decide(ctx) is None

    def test_weights_do_not_exceed_one(self) -> None:
        ctx = _ctx()
        agent = BuyAndHold(["AAPL", "MSFT", "GOOG"])

        weights = agent.decide(ctx)

        assert weights is not None
        assert sum(weights.values()) <= Decimal(1)
