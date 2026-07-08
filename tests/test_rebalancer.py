"""Tests for `Rebalancer`: weight validation, the cash buffer, minimum trade
size, liquidation of dropped symbols, and SELL-before-BUY ordering.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.broker import OrderSide, OrderStatus, SimulatedBroker
from value_portfolio.data import InMemoryMarketData
from value_portfolio.data.types import Bar
from value_portfolio.rebalancer import Rebalancer

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _day(i: int) -> datetime:
    return _START + timedelta(days=i)


def _make_bars(symbol: str, prices: list[float]) -> list[Bar]:
    """Build a sequence of daily bars with open == close == given price."""
    bars: list[Bar] = []
    for i, price in enumerate(prices):
        c = Decimal(str(price))
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=_day(i),
                open=c,
                high=c,
                low=c,
                close=c,
                volume=Decimal("1000"),
                timeframe="1Day",
            )
        )
    return bars


def _make_sim(
    bars: dict[str, list[Bar]],
    *,
    starting_cash: Decimal = Decimal("10000"),
) -> SimulatedBroker:
    return SimulatedBroker(
        market_data=InMemoryMarketData(bars),
        starting_cash=starting_cash,
        commission_per_share=Decimal("0"),
        slippage_bps=Decimal("0"),
    )


class TestConstruction:
    def test_rejects_cash_buffer_at_or_above_one(self) -> None:
        with pytest.raises(ValueError):
            Rebalancer(cash_buffer=Decimal("1"))

    def test_rejects_negative_cash_buffer(self) -> None:
        with pytest.raises(ValueError):
            Rebalancer(cash_buffer=Decimal("-0.1"))

    def test_rejects_negative_min_trade_notional(self) -> None:
        with pytest.raises(ValueError):
            Rebalancer(min_trade_notional=Decimal("-1"))


class TestWeightValidation:
    def test_rejects_negative_weight(self) -> None:
        sim = _make_sim({"AAPL": _make_bars("AAPL", [100, 100])})
        rb = Rebalancer(cash_buffer=Decimal("0"))
        with pytest.raises(ValueError, match="negative"):
            rb.rebalance({"AAPL": Decimal("-0.1")}, sim, sim.market_data)

    def test_rejects_overallocation(self) -> None:
        bars = {"AAPL": _make_bars("AAPL", [100, 100]), "MSFT": _make_bars("MSFT", [100, 100])}
        sim = _make_sim(bars)
        rb = Rebalancer(cash_buffer=Decimal("0"))
        with pytest.raises(ValueError, match="exceeds 1"):
            rb.rebalance({"AAPL": Decimal("0.6"), "MSFT": Decimal("0.6")}, sim, sim.market_data)


class TestRebalance:
    def test_basic_buy_deploys_full_equity(self) -> None:
        sim = _make_sim({"AAPL": _make_bars("AAPL", [100, 100])}, starting_cash=Decimal("10000"))
        rb = Rebalancer(cash_buffer=Decimal("0"))

        orders = rb.rebalance({"AAPL": Decimal("1")}, sim, sim.market_data)

        assert len(orders) == 1
        assert orders[0].side is OrderSide.BUY
        assert orders[0].symbol == "AAPL"
        assert orders[0].qty == Decimal("100")  # 10000 / 100

    def test_cash_buffer_reserves_cash(self) -> None:
        sim = _make_sim({"AAPL": _make_bars("AAPL", [100, 100])}, starting_cash=Decimal("10000"))
        rb = Rebalancer(cash_buffer=Decimal("0.10"))  # deploy only 90%

        orders = rb.rebalance({"AAPL": Decimal("1")}, sim, sim.market_data)

        assert orders[0].qty == Decimal("90")  # 9000 / 100

    def test_min_trade_notional_skips_small_trades(self) -> None:
        sim = _make_sim({"AAPL": _make_bars("AAPL", [100, 100])}, starting_cash=Decimal("10000"))
        # target value 10000 * 0.0001 = 1, below a threshold of 10
        strict = Rebalancer(cash_buffer=Decimal("0"), min_trade_notional=Decimal("10"))
        loose = Rebalancer(cash_buffer=Decimal("0"), min_trade_notional=Decimal("0"))

        assert strict.rebalance({"AAPL": Decimal("0.0001")}, sim, sim.market_data) == []
        assert len(loose.rebalance({"AAPL": Decimal("0.0001")}, sim, sim.market_data)) == 1

    def test_sells_submitted_before_buys(self) -> None:
        bars = {
            "AAPL": _make_bars("AAPL", [100, 100, 100]),
            "MSFT": _make_bars("MSFT", [50, 50, 50]),
        }
        sim = _make_sim(bars, starting_cash=Decimal("10000"))
        rb = Rebalancer(cash_buffer=Decimal("0"))

        # Day 0: go all-in on AAPL, then advance so the BUY fills.
        rb.rebalance({"AAPL": Decimal("1")}, sim, sim.market_data)
        sim.advance_to(_day(1))
        assert sim.get_position("AAPL") is not None

        # Day 1: switch the whole book to MSFT.
        orders = rb.rebalance({"MSFT": Decimal("1")}, sim, sim.market_data)

        assert [o.side for o in orders] == [OrderSide.SELL, OrderSide.BUY]
        assert orders[0].symbol == "AAPL"
        assert orders[1].symbol == "MSFT"

        # The SELL frees the cash the BUY needs; both fill on the next bar.
        sim.advance_to(_day(2))
        assert sim.get_position("AAPL") is None
        msft = sim.get_position("MSFT")
        assert msft is not None
        assert msft.qty == Decimal("200")  # 10000 / 50

    def test_symbol_absent_from_target_is_liquidated(self) -> None:
        bars = {
            "AAPL": _make_bars("AAPL", [100, 100, 100]),
            "MSFT": _make_bars("MSFT", [50, 50, 50]),
        }
        sim = _make_sim(bars, starting_cash=Decimal("10000"))
        rb = Rebalancer(cash_buffer=Decimal("0"))

        rb.rebalance({"AAPL": Decimal("1")}, sim, sim.market_data)
        sim.advance_to(_day(1))

        # Empty target -> liquidate everything held.
        orders = rb.rebalance({}, sim, sim.market_data)

        assert len(orders) == 1
        assert orders[0].side is OrderSide.SELL
        assert orders[0].symbol == "AAPL"

        sim.advance_to(_day(2))
        assert sim.get_position("AAPL") is None
        assert all(o.status is OrderStatus.FILLED for o in sim.list_orders())
