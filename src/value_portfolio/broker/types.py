
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(StrEnum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class Position:
    symbol: str
    qty: Decimal
    avg_entry_price: Decimal
    market_value: Decimal
    unrealized_pl: Decimal
    current_price: Decimal


@dataclass(frozen=True, slots=True)
class Order:
    id: str
    client_order_id: str | None
    symbol: str
    qty: Decimal
    side: OrderSide
    status: OrderStatus
    submitted_at: datetime
    filled_qty: Decimal = Decimal("0")
    filled_avg_price: Decimal | None = None
    filled_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    account_id: str
    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    timestamp: datetime
    positions: tuple[Position, ...] = field(default_factory=tuple)
