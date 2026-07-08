"""Tests for `EqualWeight.decide`: an equal-weight target on rebalance steps,
``None`` in between.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from value_portfolio.agent import DecisionContext
from value_portfolio.baselines import EqualWeight
from value_portfolio.broker import SimulatedBroker
from value_portfolio.data import InMemoryMarketData
from value_portfolio.data.types import Bar

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _ctx() -> DecisionContext:
    """A throwaway context; equal-weight ignores all observations."""
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
            EqualWeight([])

    def test_rejects_duplicate_symbols(self) -> None:
        with pytest.raises(ValueError):
            EqualWeight(["AAPL", "AAPL"])

    def test_rejects_non_positive_rebalance_every(self) -> None:
        with pytest.raises(ValueError):
            EqualWeight(["AAPL"], rebalance_every=0)


class TestDecide:
    def test_returns_equal_weights_every_step_by_default(self) -> None:
        ctx = _ctx()
        agent = EqualWeight(["AAPL", "MSFT"])

        half = Decimal(1) / Decimal(2)
        for _ in range(3):
            assert agent.decide(ctx) == {"AAPL": half, "MSFT": half}

    def test_rebalance_every_skips_intermediate_steps(self) -> None:
        ctx = _ctx()
        agent = EqualWeight(["AAPL"], rebalance_every=3)

        assert agent.decide(ctx) is not None  # step 0
        assert agent.decide(ctx) is None  # step 1
        assert agent.decide(ctx) is None  # step 2
        assert agent.decide(ctx) is not None  # step 3
