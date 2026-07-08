"""Smoke tests for `AlpacaBroker` (real paper endpoint). Skipped without
credentials; run with ``uv run pytest -m smoke``.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from value_portfolio import (
    AccountSnapshot,
    AlpacaBroker,
    AlpacaSettings,
    Order,
    OrderSide,
    OrderStatus,
)

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def broker() -> AlpacaBroker:
    # AlpacaSettings reads .env via pydantic-settings, so this is the
    # authoritative check for whether credentials are configured.
    try:
        settings = AlpacaSettings()  # type: ignore[call-arg]
    except ValidationError:
        pytest.skip("Alpaca credentials not configured (env or .env); skipping smoke tests.")
    if not settings.paper:
        pytest.skip("Refusing to run smoke tests against a live (non-paper) account.")
    return AlpacaBroker(settings)


def test_get_account_returns_snapshot(broker: AlpacaBroker) -> None:
    snap = broker.get_account()
    assert isinstance(snap, AccountSnapshot)
    assert snap.account_id
    assert snap.cash >= Decimal("0")
    assert snap.equity >= Decimal("0")


def test_is_market_open_returns_bool(broker: AlpacaBroker) -> None:
    assert isinstance(broker.is_market_open(), bool)


def test_buy_one_share_aapl(broker: AlpacaBroker) -> None:
    """End-to-end write-path test: place a real market BUY for 1 AAPL on the paper account.

    NOTE: every run of this test places a new order. If market is open the order
    fills near-instantly and one share lands in your positions; if market is
    closed, Alpaca queues the order and it fills at next session open. Either
    way the order/position is visible in the Alpaca paper dashboard.
    """
    order = broker.buy("AAPL", Decimal("1"))

    assert isinstance(order, Order)
    assert order.id, "expected non-empty order id from Alpaca"
    assert order.symbol == "AAPL"
    assert order.side is OrderSide.BUY
    assert order.qty == Decimal("1")
    assert order.status in {
        OrderStatus.NEW,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.FILLED,
    }, f"unexpected status: {order.status}"

    fetched = broker.get_order(order.id)
    assert fetched.id == order.id
    assert fetched.symbol == "AAPL"

    print(
        f"\n[buy test] submitted order id={order.id} status={order.status.value} "
        f"filled_qty={fetched.filled_qty}"
    )
