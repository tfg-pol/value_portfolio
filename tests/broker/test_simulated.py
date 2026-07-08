"""Tests for `SimulatedBroker`: account state, order lifecycle, fills with cost
and slippage, position tracking, and clock-advance semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.broker import (
    AccountSnapshot,
    BrokerClient,
    OrderRejectedError,
    OrderStatus,
    SimulatedBroker,
    SymbolNotFoundError,
)
from value_portfolio.data import InMemoryMarketData
from value_portfolio.data.types import Bar


def _make_bars(
    symbol: str,
    prices: list[float],
    *,
    start: datetime | None = None,
    timeframe: str = "1Day",
) -> list[Bar]:
    """Helper: build a sequence of daily bars with given closes (open=close)."""
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


def _make_sim(
    bars: dict[str, list[Bar]],
    *,
    starting_cash: Decimal = Decimal("10000"),
    commission_per_share: Decimal = Decimal("0"),
    slippage_bps: Decimal = Decimal("0"),
) -> SimulatedBroker:
    return SimulatedBroker(
        market_data=InMemoryMarketData(bars),
        starting_cash=starting_cash,
        commission_per_share=commission_per_share,
        slippage_bps=slippage_bps,
    )


@pytest.fixture
def sim() -> SimulatedBroker:
    bars = {
        "AAPL": _make_bars("AAPL", [100.0, 101.0, 102.0, 103.0, 104.0]),
        "MSFT": _make_bars("MSFT", [200.0, 200.0, 200.0, 200.0, 200.0]),
    }
    return _make_sim(bars)


class TestContract:
    def test_implements_broker_client(self, sim: SimulatedBroker) -> None:
        assert isinstance(sim, BrokerClient)

    def test_exposes_wrapped_market_data(self, sim: SimulatedBroker) -> None:
        # The broker is a joystick, but it does expose the data source it
        # was constructed with so agents can share the same clock-truncated view.
        assert isinstance(sim.market_data, InMemoryMarketData)


class TestInitialState:
    def test_clock_starts_at_earliest_bar(self, sim: SimulatedBroker) -> None:
        assert sim.now == datetime(2026, 1, 1, tzinfo=UTC)

    def test_get_account_initial(self, sim: SimulatedBroker) -> None:
        snap = sim.get_account()
        assert isinstance(snap, AccountSnapshot)
        assert snap.cash == Decimal("10000")
        assert snap.equity == Decimal("10000")
        assert snap.positions == ()

    def test_no_initial_positions(self, sim: SimulatedBroker) -> None:
        assert sim.get_positions() == []
        assert sim.get_position("AAPL") is None


class TestBuyAndSell:
    def test_buy_fills_at_next_bar_open(self, sim: SimulatedBroker) -> None:
        order = sim.buy("AAPL", Decimal("10"))
        assert order.status is OrderStatus.NEW

        terminal = sim.advance_to(datetime(2026, 1, 2, tzinfo=UTC))
        assert len(terminal) == 1
        filled = terminal[0]
        assert filled.id == order.id
        assert filled.status is OrderStatus.FILLED
        assert filled.filled_qty == Decimal("10")
        assert filled.filled_avg_price == Decimal("101")  # day-2 open

    def test_buy_decrements_cash_and_creates_position(self, sim: SimulatedBroker) -> None:
        sim.buy("AAPL", Decimal("10"))
        sim.advance_to(datetime(2026, 1, 2, tzinfo=UTC))

        assert sim.get_account().cash == Decimal("10000") - Decimal("10") * Decimal("101")
        pos = sim.get_position("AAPL")
        assert pos is not None
        assert pos.qty == Decimal("10")
        assert pos.avg_entry_price == Decimal("101")

    def test_sell_returns_cash_and_clears_position(self, sim: SimulatedBroker) -> None:
        sim.buy("AAPL", Decimal("10"))
        sim.advance_to(datetime(2026, 1, 2, tzinfo=UTC))
        sim.sell("AAPL", Decimal("10"))
        sim.advance_to(datetime(2026, 1, 3, tzinfo=UTC))

        # Bought at 101, sold at 102 — net profit 10.
        assert sim.get_account().cash == Decimal("10000") + Decimal("10")
        assert sim.get_position("AAPL") is None

    def test_unrealized_pl_tracks_current_price(self, sim: SimulatedBroker) -> None:
        sim.buy("AAPL", Decimal("10"))
        sim.advance_to(datetime(2026, 1, 5, tzinfo=UTC))  # open=101 (bought), close=104
        pos = sim.get_position("AAPL")
        assert pos is not None
        assert pos.current_price == Decimal("104")
        assert pos.unrealized_pl == (Decimal("104") - Decimal("101")) * Decimal("10")


class TestRejectionsAndErrors:
    def test_zero_qty_rejected(self, sim: SimulatedBroker) -> None:
        with pytest.raises(OrderRejectedError):
            sim.buy("AAPL", Decimal("0"))

    def test_unknown_symbol_rejected(self, sim: SimulatedBroker) -> None:
        with pytest.raises(SymbolNotFoundError):
            sim.buy("ZZZZ", Decimal("1"))

    def test_short_sell_rejected(self, sim: SimulatedBroker) -> None:
        with pytest.raises(OrderRejectedError):
            sim.sell("AAPL", Decimal("1"))

    def test_oversell_rejected(self, sim: SimulatedBroker) -> None:
        sim.buy("AAPL", Decimal("5"))
        sim.advance_to(datetime(2026, 1, 2, tzinfo=UTC))
        with pytest.raises(OrderRejectedError):
            sim.sell("AAPL", Decimal("10"))

    def test_cancel_unknown_order_raises(self, sim: SimulatedBroker) -> None:
        with pytest.raises(Exception):  # noqa: B017 — BrokerError; loose match is fine here
            sim.cancel_order("does-not-exist")

    def test_underfunded_buy_is_rejected_at_fill_not_raised(self) -> None:
        # Sized to fit at 100, but the next open gaps to 120: the order must be
        # rejected (insufficient buying power), leaving cash and positions
        # untouched and the simulation alive.
        broker = _make_sim(
            {"GAPPY": _make_bars("GAPPY", [100, 120, 120])},
            starting_cash=Decimal("1000"),
        )
        order = broker.buy("GAPPY", Decimal("10"))  # 10 * 120 = 1200 > 1000

        terminal = broker.advance_to(datetime(2026, 1, 2, tzinfo=UTC))

        assert broker.get_order(order.id).status is OrderStatus.REJECTED
        assert [o.status for o in terminal] == [OrderStatus.REJECTED]
        assert broker.get_account().cash == Decimal("1000")
        assert broker.get_positions() == []


class TestCommissionAndSlippage:
    def test_commission_charged_on_buy(self) -> None:
        bars = {"AAPL": _make_bars("AAPL", [100.0, 100.0, 100.0])}
        sim = _make_sim(
            bars,
            starting_cash=Decimal("11000"),
            commission_per_share=Decimal("0.01"),
        )
        sim.buy("AAPL", Decimal("100"))
        sim.advance_to(datetime(2026, 1, 2, tzinfo=UTC))
        # cash spent: 100 shares * $100 + 100 * $0.01 commission = $10001
        assert sim.get_account().cash == Decimal("11000") - Decimal("10001")

    def test_slippage_makes_buy_more_expensive(self) -> None:
        bars = {"AAPL": _make_bars("AAPL", [100.0, 100.0, 100.0])}
        sim = _make_sim(
            bars,
            starting_cash=Decimal("100000"),
            slippage_bps=Decimal("100"),  # 1%
        )
        sim.buy("AAPL", Decimal("10"))
        terminal = sim.advance_to(datetime(2026, 1, 2, tzinfo=UTC))
        # Expected fill: 100 * (1 + 100/10000) = 101
        assert terminal[0].filled_avg_price == Decimal("101.00")


class TestClockSemantics:
    def test_advance_backwards_rejected(self, sim: SimulatedBroker) -> None:
        # `sim.advance_to` delegates to the wrapped data source, which is the
        # authoritative monotonic clock.
        sim.advance_to(datetime(2026, 1, 3, tzinfo=UTC))
        with pytest.raises(ValueError, match="cannot advance backwards"):
            sim.advance_to(datetime(2026, 1, 1, tzinfo=UTC))

    def test_sim_and_data_clocks_stay_in_sync(self, sim: SimulatedBroker) -> None:
        target = datetime(2026, 1, 3, tzinfo=UTC)
        sim.advance_to(target)
        assert sim.now == target
        assert sim.market_data.now == target


class TestOrderManagement:
    def test_cancel_pending_order(self, sim: SimulatedBroker) -> None:
        order = sim.buy("AAPL", Decimal("1"))
        sim.cancel_order(order.id)
        assert sim.get_order(order.id).status is OrderStatus.CANCELED

    def test_cancel_filled_order_is_noop(self, sim: SimulatedBroker) -> None:
        order = sim.buy("AAPL", Decimal("1"))
        sim.advance_to(datetime(2026, 1, 2, tzinfo=UTC))
        sim.cancel_order(order.id)  # already filled — should not raise
        assert sim.get_order(order.id).status is OrderStatus.FILLED

    def test_list_orders_filter_by_status(self, sim: SimulatedBroker) -> None:
        o1 = sim.buy("AAPL", Decimal("1"))
        sim.advance_to(datetime(2026, 1, 2, tzinfo=UTC))  # o1 fills
        o2 = sim.buy("AAPL", Decimal("1"))  # still NEW

        assert {o.id for o in sim.list_orders(OrderStatus.FILLED)} == {o1.id}
        assert {o.id for o in sim.list_orders(OrderStatus.NEW)} == {o2.id}
        assert len(sim.list_orders()) == 2


class TestReset:
    def test_reset_clears_state(self, sim: SimulatedBroker) -> None:
        sim.buy("AAPL", Decimal("1"))
        sim.advance_to(datetime(2026, 1, 3, tzinfo=UTC))
        assert sim.get_position("AAPL") is not None

        sim.reset()
        assert sim.get_account().cash == Decimal("10000")
        assert sim.get_positions() == []
        assert sim.list_orders() == []
        assert sim.now == datetime(2026, 1, 1, tzinfo=UTC)
        # Data source clock also reset.
        assert sim.market_data.now == datetime(2026, 1, 1, tzinfo=UTC)


class TestDelisting:
    def _make_held_delisting_sim(
        self, *, delisting_fill_price: Callable[[Decimal], Decimal] | None = None
    ) -> SimulatedBroker:
        # ALIVE trades through day 4; DEAD's price series ends at day 2.
        bars = {
            "ALIVE": _make_bars("ALIVE", [100.0, 100.0, 100.0, 100.0, 100.0]),
            "DEAD": _make_bars("DEAD", [10.0, 10.0, 12.0]),
        }
        sim = SimulatedBroker(
            market_data=InMemoryMarketData(bars),
            starting_cash=Decimal("1000"),
            delisting_fill_price=delisting_fill_price,
        )
        # Buy 5 DEAD, filled at day 1 open (10) -> cash 950, position 5 @ 10.
        sim.buy("DEAD", Decimal("5"))
        sim.advance_to(datetime(2026, 1, 2, tzinfo=UTC))  # day 1
        assert sim.get_position("DEAD") is not None
        assert sim.get_account().cash == Decimal("950")
        return sim

    def test_held_position_liquidated_at_last_close(self) -> None:
        sim = self._make_held_delisting_sim()

        # Day 2 (final DEAD bar): not yet delisted, marked at close 12 -> mv 60.
        sim.advance_to(datetime(2026, 1, 3, tzinfo=UTC))
        pos = sim.get_position("DEAD")
        assert pos is not None
        assert pos.market_value == Decimal("60")

        # Day 3: DEAD's series has ended -> liquidated at last close (12).
        sim.advance_to(datetime(2026, 1, 4, tzinfo=UTC))
        assert sim.get_position("DEAD") is None
        # Cash credited 5 * 12 = 60; not carried at a stale mark.
        assert sim.get_account().cash == Decimal("1010")
        assert sim.get_account().equity == Decimal("1010")

    def test_delisting_fill_price_policy_is_configurable(self) -> None:
        # A 50% haircut on the last close models a distressed delisting.
        sim = self._make_held_delisting_sim(
            delisting_fill_price=lambda close: close * Decimal("0.5")
        )

        sim.advance_to(datetime(2026, 1, 4, tzinfo=UTC))  # day 3 -> delisted
        assert sim.get_position("DEAD") is None
        # 5 * (12 * 0.5) = 30 credited to the 950 cash.
        assert sim.get_account().cash == Decimal("980")

    def test_active_symbol_is_not_liquidated(self) -> None:
        sim = self._make_held_delisting_sim()
        sim.buy("ALIVE", Decimal("1"))
        sim.advance_to(datetime(2026, 1, 3, tzinfo=UTC))  # ALIVE fills, still trading

        sim.advance_to(datetime(2026, 1, 5, tzinfo=UTC))
        assert sim.get_position("ALIVE") is not None  # ALIVE survives
        assert sim.get_position("DEAD") is None  # DEAD delisted and gone
