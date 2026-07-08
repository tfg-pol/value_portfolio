"""Unit tests for broker domain types — pure-Python, no network or credentials."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from value_portfolio.broker.types import (
    AccountSnapshot,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)


class TestEnums:
    def test_order_side_values(self) -> None:
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_status_values(self) -> None:
        assert OrderStatus.NEW.value == "new"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.CANCELED.value == "canceled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"


class TestImmutability:
    """Frozen dataclasses must reject post-construction mutation."""

    def test_position_is_frozen(self) -> None:
        p = Position(
            symbol="AAPL",
            qty=Decimal("10"),
            avg_entry_price=Decimal("150"),
            market_value=Decimal("1600"),
            unrealized_pl=Decimal("100"),
            current_price=Decimal("160"),
        )
        with pytest.raises(FrozenInstanceError):
            p.qty = Decimal("0")  # type: ignore[misc]

    def test_order_is_frozen(self) -> None:
        o = Order(
            id="abc",
            client_order_id=None,
            symbol="AAPL",
            qty=Decimal("1"),
            side=OrderSide.BUY,
            status=OrderStatus.NEW,
            submitted_at=datetime.now(UTC),
        )
        with pytest.raises(FrozenInstanceError):
            o.status = OrderStatus.FILLED  # type: ignore[misc]


class TestDecimalSemantics:
    """Money/quantity must round-trip without float drift."""

    def test_position_short_has_negative_qty(self) -> None:
        p = Position(
            symbol="AAPL",
            qty=Decimal("-5"),
            avg_entry_price=Decimal("150"),
            market_value=Decimal("-750"),
            unrealized_pl=Decimal("0"),
            current_price=Decimal("150"),
        )
        assert p.qty < 0


class TestOrderDefaults:
    def test_market_order_minimal_construction(self) -> None:
        o = Order(
            id="1",
            client_order_id=None,
            symbol="AAPL",
            qty=Decimal("1"),
            side=OrderSide.BUY,
            status=OrderStatus.NEW,
            submitted_at=datetime.now(UTC),
        )
        assert o.filled_qty == Decimal("0")
        assert o.filled_avg_price is None
        assert o.filled_at is None


class TestAccountSnapshot:
    def test_default_positions_empty(self) -> None:
        snap = AccountSnapshot(
            account_id="acct-1",
            cash=Decimal("1000"),
            equity=Decimal("1000"),
            buying_power=Decimal("4000"),
            timestamp=datetime.now(UTC),
        )
        assert snap.positions == ()

    def test_positions_are_immutable_tuple(self) -> None:
        pos = Position(
            symbol="AAPL",
            qty=Decimal("1"),
            avg_entry_price=Decimal("150"),
            market_value=Decimal("160"),
            unrealized_pl=Decimal("10"),
            current_price=Decimal("160"),
        )
        snap = AccountSnapshot(
            account_id="acct-1",
            cash=Decimal("0"),
            equity=Decimal("160"),
            buying_power=Decimal("0"),
            timestamp=datetime.now(UTC),
            positions=(pos,),
        )
        assert isinstance(snap.positions, tuple)
        assert snap.positions[0] is pos
