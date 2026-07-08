
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from decimal import Decimal

from value_portfolio.broker.base import BrokerClient
from value_portfolio.broker.exceptions import (
    BrokerError,
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
from value_portfolio.data.in_memory import InMemoryMarketData

_ZERO = Decimal("0")
_BPS_DIVISOR = Decimal("10000")


def _last_available_close(last_close: Decimal) -> Decimal:
    return last_close


class SimulatedBroker(BrokerClient):
    def __init__(
        self,
        market_data: InMemoryMarketData,
        starting_cash: Decimal = Decimal("100000"),
        commission_per_share: Decimal = _ZERO,
        slippage_bps: Decimal = _ZERO,
        account_id: str = "sim-account",
        delisting_fill_price: Callable[[Decimal], Decimal] | None = None,
    ) -> None:
        if starting_cash < 0:
            raise ValueError("starting_cash must be non-negative")
        if commission_per_share < 0:
            raise ValueError("commission_per_share must be non-negative")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")

        self._market_data = market_data
        self._initial_cash = starting_cash
        self._commission_per_share = commission_per_share
        self._slippage_bps = slippage_bps
        self._account_id = account_id
        # Maps a delisted holding's last close to its liquidation price.
        self._delisting_fill_price = delisting_fill_price or _last_available_close

        self._cash: Decimal = starting_cash
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, Order] = {}

    # Clock control

    @property
    def market_data(self) -> InMemoryMarketData:
        return self._market_data

    @property
    def now(self) -> datetime:
        return self._market_data.now

    def advance_to(self, timestamp: datetime) -> list[Order]:
        self._market_data.advance_to(timestamp)

        terminal: list[Order] = []
        for order_id, order in list(self._orders.items()):
            if order.status is not OrderStatus.NEW:
                continue
            updated = self._try_fill(order)
            if updated is not order:
                self._orders[order_id] = updated
                if updated.status is not OrderStatus.NEW:
                    terminal.append(updated)

        self._liquidate_delisted()
        return terminal

    def reset(self, starting_cash: Decimal | None = None) -> None:
        self._cash = starting_cash if starting_cash is not None else self._initial_cash
        self._positions = {}
        self._orders = {}
        self._market_data.reset()

    # Account / state

    def get_account(self) -> AccountSnapshot:
        positions = self.get_positions()
        equity = self._cash + sum((p.market_value for p in positions), start=_ZERO)
        return AccountSnapshot(
            account_id=self._account_id,
            cash=self._cash,
            equity=equity,
            buying_power=self._cash,
            timestamp=self.now,
            positions=tuple(positions),
        )

    def get_positions(self) -> list[Position]:
        return [self._refresh_position(p) for p in self._positions.values()]

    def get_position(self, symbol: str) -> Position | None:
        position = self._positions.get(symbol)
        if position is None:
            return None
        return self._refresh_position(position)

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
        if qty <= 0:
            raise OrderRejectedError(f"qty must be positive, got {qty}")
        if symbol not in self._market_data.symbols:
            raise SymbolNotFoundError(f"no bars loaded for symbol {symbol!r}")

        if side is OrderSide.SELL:
            held = self._positions.get(symbol)
            held_qty = held.qty if held is not None else _ZERO
            if held_qty < qty:
                raise OrderRejectedError(
                    f"insufficient position to SELL {qty} {symbol}: holding {held_qty}. "
                    "Shorting is not supported in SimulatedBroker v1."
                )

        order = Order(
            id=str(uuid.uuid4()),
            client_order_id=client_order_id,
            symbol=symbol,
            qty=qty,
            side=side,
            status=OrderStatus.NEW,
            submitted_at=self.now,
        )
        self._orders[order.id] = order
        return order

    # Order management

    def cancel_order(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order is None:
            raise BrokerError(f"unknown order id: {order_id}")
        if order.status is not OrderStatus.NEW:
            return  # already terminal — idempotent no-op
        self._orders[order_id] = replace(order, status=OrderStatus.CANCELED)

    def get_order(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise BrokerError(f"unknown order id: {order_id}")
        return order

    def list_orders(self, status: OrderStatus | None = None) -> list[Order]:
        orders = list(self._orders.values())
        if status is None:
            return orders
        return [o for o in orders if o.status is status]

    # Calendar

    def is_market_open(self) -> bool:
        # Open iff the sim clock matches a bar timestamp on some symbol. Uses the
        # sim clock, never datetime.now(), so backtests stay reproducible.
        clock = self.now
        for symbol in self._market_data.symbols:
            bar = self._market_data.latest_bar(symbol)
            if bar is not None and bar.timestamp == clock:
                return True
        return False

    # Internals

    def _try_fill(self, order: Order) -> Order:
        """Fill `order` at the next bar's open; return it unchanged if no bar yet."""
        next_bar = self._market_data.next_bar_after(order.symbol, order.submitted_at)
        if next_bar is None:
            return order  # next bar hasn't arrived yet

        fill_price = self._apply_slippage(next_bar.open, order.side)
        commission = self._commission_per_share * order.qty
        notional = fill_price * order.qty

        if order.side is OrderSide.BUY:
            total_cost = notional + commission
            if total_cost > self._cash:
                return replace(order, status=OrderStatus.REJECTED)
            self._cash -= total_cost
            self._add_to_position(order.symbol, order.qty, fill_price)
        else:  # SELL
            self._cash += notional - commission
            self._reduce_position(order.symbol, order.qty)

        return replace(
            order,
            status=OrderStatus.FILLED,
            filled_qty=order.qty,
            filled_avg_price=fill_price,
            filled_at=next_bar.timestamp,
        )

    def _liquidate_delisted(self) -> None:
        for symbol in list(self._positions):
            if not self._market_data.is_delisted(symbol):
                continue
            last_bar = self._market_data.latest_bar(symbol)
            if last_bar is None:
                continue  # defensive: a delisted symbol always has a final bar
            position = self._positions[symbol]
            fill_price = self._delisting_fill_price(last_bar.close)
            self._cash += position.qty * fill_price
            del self._positions[symbol]

    def _apply_slippage(self, price: Decimal, side: OrderSide) -> Decimal:
        """Slippage is unfavorable: higher buy price, lower sell price."""
        adjustment = price * self._slippage_bps / _BPS_DIVISOR
        return price + adjustment if side is OrderSide.BUY else price - adjustment

    def _add_to_position(self, symbol: str, qty: Decimal, price: Decimal) -> None:
        existing = self._positions.get(symbol)
        if existing is None:
            self._positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                avg_entry_price=price,
                market_value=qty * price,
                unrealized_pl=_ZERO,
                current_price=price,
            )
            return
        new_qty = existing.qty + qty
        new_avg = ((existing.avg_entry_price * existing.qty) + (price * qty)) / new_qty
        self._positions[symbol] = replace(
            existing, qty=new_qty, avg_entry_price=new_avg, current_price=price
        )

    def _reduce_position(self, symbol: str, qty: Decimal) -> None:
        existing = self._positions[symbol]  # presence guaranteed by submit_order check
        new_qty = existing.qty - qty
        if new_qty <= 0:
            del self._positions[symbol]
        else:
            self._positions[symbol] = replace(existing, qty=new_qty)

    def _refresh_position(self, position: Position) -> Position:
        """Recompute market_value / unrealized_pl using the latest visible bar."""
        bar = self._market_data.latest_bar(position.symbol)
        current_price = bar.close if bar is not None else position.current_price
        return replace(
            position,
            current_price=current_price,
            market_value=position.qty * current_price,
            unrealized_pl=(current_price - position.avg_entry_price) * position.qty,
        )
