
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from value_portfolio.broker.base import BrokerClient
from value_portfolio.broker.types import Order
from value_portfolio.data.base import MarketDataSource

_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_SUM_TOLERANCE = Decimal("1e-9")


class Rebalancer:
    def __init__(
        self,
        cash_buffer: Decimal = Decimal("0.02"),
        min_trade_notional: Decimal = _ZERO,
    ) -> None:
        if not (_ZERO <= cash_buffer < _ONE):
            raise ValueError(f"cash_buffer must be in [0, 1), got {cash_buffer}")
        if min_trade_notional < _ZERO:
            raise ValueError(f"min_trade_notional must be non-negative, got {min_trade_notional}")
        self._cash_buffer = cash_buffer
        self._min_trade_notional = min_trade_notional

    def rebalance(
        self,
        target_weights: Mapping[str, Decimal],
        broker: BrokerClient,
        data: MarketDataSource,
    ) -> list[Order]:
        weights = self._validate(target_weights)

        equity = broker.get_account().equity
        deployable = equity * (_ONE - self._cash_buffer)

        held: dict[str, Decimal] = {p.symbol: p.qty for p in broker.get_positions()}

        sells: list[tuple[str, Decimal]] = []
        buys: list[tuple[str, Decimal]] = []
        for symbol in sorted(set(weights) | set(held)):
            price = self._mid_price(data, symbol)
            target_qty = (deployable * weights.get(symbol, _ZERO)) / price
            delta = target_qty - held.get(symbol, _ZERO)
            if abs(delta) * price < self._min_trade_notional:
                continue
            if delta < _ZERO:
                sells.append((symbol, -delta))
            elif delta > _ZERO:
                buys.append((symbol, delta))

        orders: list[Order] = []
        orders.extend(broker.sell(symbol, qty) for symbol, qty in sells)
        orders.extend(broker.buy(symbol, qty) for symbol, qty in buys)
        return orders

    def _validate(self, target_weights: Mapping[str, Decimal]) -> dict[str, Decimal]:
        weights = dict(target_weights)
        total = _ZERO
        for symbol, weight in weights.items():
            if weight < _ZERO:
                raise ValueError(
                    f"target weight for {symbol!r} is negative ({weight}); "
                    "shorting is not supported"
                )
            total += weight
        if total > _ONE + _SUM_TOLERANCE:
            raise ValueError(f"target weights sum to {total}, which exceeds 1")
        return weights

    def _mid_price(self, data: MarketDataSource, symbol: str) -> Decimal:
        quote = data.get_quote(symbol)
        mid = (quote.bid_price + quote.ask_price) / _TWO
        if mid <= _ZERO:
            raise ValueError(f"non-positive mid price for {symbol!r}: {mid}")
        return mid
