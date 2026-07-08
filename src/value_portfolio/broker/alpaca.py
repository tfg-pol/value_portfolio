from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from value_portfolio.broker.base import BrokerClient
from value_portfolio.broker.exceptions import (
    AuthenticationError,
    BrokerError,
    InsufficientFundsError,
    OrderRejectedError,
    SymbolNotFoundError,
)
from value_portfolio.broker.types import (
    AccountSnapshot,
    Order,
    OrderSide,
    OrderStatus,
    Position,
)
from value_portfolio.config import AlpacaSettings

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient


class AlpacaBroker(BrokerClient):
    """`BrokerClient` backed by Alpaca's REST API via `alpaca-py`."""

    def __init__(self, settings: AlpacaSettings | None = None) -> None:
        from alpaca.trading.client import TradingClient

        self._settings = settings or AlpacaSettings()  # type: ignore[call-arg]
        self._trading: TradingClient = TradingClient(
            api_key=self._settings.api_key,
            secret_key=self._settings.api_secret,
            paper=self._settings.paper,
        )

    # Account / state

    def get_account(self) -> AccountSnapshot:
        with _translate_errors():
            # alpaca-py types this as `TradeAccount | dict`; we never use raw mode.
            account: Any = self._trading.get_account()
            positions = self.get_positions()
        return AccountSnapshot(
            account_id=str(account.id),
            cash=Decimal(str(account.cash)),
            equity=Decimal(str(account.equity)),
            buying_power=Decimal(str(account.buying_power)),
            timestamp=datetime.now().astimezone(),
            positions=tuple(positions),
        )

    def get_positions(self) -> list[Position]:
        with _translate_errors():
            raw = self._trading.get_all_positions()
        return [_to_position(p) for p in raw]

    def get_position(self, symbol: str) -> Position | None:
        from alpaca.common.exceptions import APIError

        try:
            with _translate_errors():
                raw = self._trading.get_open_position(symbol)
        except APIError as exc:
            if "position does not exist" in str(exc).lower():
                return None
            raise
        return _to_position(raw)

    # Execution (market orders only)

    def buy(
        self,
        symbol: str,
        qty: Decimal,
        client_order_id: str | None = None,
    ) -> Order:
        return self._submit_market(symbol, qty, OrderSide.BUY, client_order_id)

    def sell(
        self,
        symbol: str,
        qty: Decimal,
        client_order_id: str | None = None,
    ) -> Order:
        return self._submit_market(symbol, qty, OrderSide.SELL, client_order_id)

    def _submit_market(
        self,
        symbol: str,
        qty: Decimal,
        side: OrderSide,
        client_order_id: str | None,
    ) -> Order:
        from alpaca.trading.enums import OrderSide as AlpacaSide
        from alpaca.trading.enums import TimeInForce as AlpacaTIF
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=symbol,
            qty=float(qty),
            side=AlpacaSide.BUY if side is OrderSide.BUY else AlpacaSide.SELL,
            time_in_force=AlpacaTIF.DAY,
            client_order_id=client_order_id,
        )
        with _translate_errors():
            raw = self._trading.submit_order(order_data=request)
        return _to_order(raw)

    # Order management

    def cancel_order(self, order_id: str) -> None:
        with _translate_errors():
            self._trading.cancel_order_by_id(order_id)

    def get_order(self, order_id: str) -> Order:
        with _translate_errors():
            raw = self._trading.get_order_by_id(order_id)
        return _to_order(raw)

    def list_orders(self, status: OrderStatus | None = None) -> list[Order]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        query_status: QueryOrderStatus | None = None
        if status is not None:
            query_status = (
                QueryOrderStatus.OPEN
                if status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)
                else QueryOrderStatus.CLOSED
            )
        request = GetOrdersRequest(status=query_status) if query_status else GetOrdersRequest()
        with _translate_errors():
            raw = self._trading.get_orders(filter=request)
        return [_to_order(o) for o in raw]

    # Calendar

    def is_market_open(self) -> bool:
        with _translate_errors():
            # alpaca-py types this as `Clock | dict`; we never use raw mode.
            clock: Any = self._trading.get_clock()
        return bool(clock.is_open)


# Translation helpers


def _to_position(raw: Any) -> Position:
    return Position(
        symbol=str(raw.symbol),
        qty=Decimal(str(raw.qty)),
        avg_entry_price=Decimal(str(raw.avg_entry_price)),
        market_value=Decimal(str(raw.market_value)),
        unrealized_pl=Decimal(str(raw.unrealized_pl)),
        current_price=Decimal(str(raw.current_price)),
    )


_ALPACA_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.NEW,
    "accepted": OrderStatus.NEW,
    "pending_new": OrderStatus.NEW,
    "accepted_for_bidding": OrderStatus.NEW,
    "held": OrderStatus.NEW,
    "calculated": OrderStatus.NEW,
    "replaced": OrderStatus.NEW,
    "pending_replace": OrderStatus.NEW,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "stopped": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "pending_cancel": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "done_for_day": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.REJECTED,
}


def _to_order(raw: Any) -> Order:
    raw_status = str(raw.status.value)
    status = _ALPACA_STATUS_MAP.get(raw_status)
    if status is None:
        raise BrokerError(f"Unrecognized Alpaca order status: {raw_status!r}")
    return Order(
        id=str(raw.id),
        client_order_id=str(raw.client_order_id) if raw.client_order_id else None,
        symbol=str(raw.symbol),
        qty=Decimal(str(raw.qty)) if raw.qty is not None else Decimal("0"),
        side=OrderSide(str(raw.side.value)),
        status=status,
        filled_qty=Decimal(str(raw.filled_qty)) if raw.filled_qty is not None else Decimal("0"),
        filled_avg_price=(
            Decimal(str(raw.filled_avg_price)) if raw.filled_avg_price is not None else None
        ),
        submitted_at=raw.submitted_at,
        filled_at=raw.filled_at,
    )


class _translate_errors:
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
        if "unauthorized" in message or "forbidden" in message or "401" in message:
            raise AuthenticationError(str(exc)) from exc
        if "insufficient" in message or "buying power" in message:
            raise InsufficientFundsError(str(exc)) from exc
        if "not found" in message or ("asset" in message and "invalid" in message):
            raise SymbolNotFoundError(str(exc)) from exc
        if "rejected" in message or "qty" in message or "order" in message:
            raise OrderRejectedError(str(exc)) from exc
        raise BrokerError(str(exc)) from exc
