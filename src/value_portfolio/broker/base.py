
from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from value_portfolio.broker.types import (
    AccountSnapshot,
    Order,
    OrderStatus,
    Position,
)


class BrokerClient(ABC):

    @abstractmethod
    def get_account(self) -> AccountSnapshot:
        """PIT snapshot of the account"""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return currently held positions"""

    @abstractmethod
    def get_position(self, symbol: str) -> Position | None:
        """Return position for the given symbol, None if not held"""

    @abstractmethod
    def buy(
        self,
        symbol: str,
        qty: Decimal,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit a market BUY for `qty` shares"""

    @abstractmethod
    def sell(
        self,
        symbol: str,
        qty: Decimal,
        client_order_id: str | None = None,
    ) -> Order:
        """Submit a market SELL for `qty` shares."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order"""

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """Fetch a single order by id."""

    @abstractmethod
    def list_orders(self, status: OrderStatus | None = None) -> list[Order]:
        """List orders."""

    # Calendar

    @abstractmethod
    def is_market_open(self) -> bool:
        """Whether the exchange is currently accepting trades."""
