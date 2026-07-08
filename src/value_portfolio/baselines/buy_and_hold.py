
from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from value_portfolio.agent import Agent, DecisionContext, require_unique_symbols


class BuyAndHold(Agent):
    def __init__(self, symbols: Sequence[str]) -> None:
        self._symbols = require_unique_symbols(symbols, "BuyAndHold")
        self._weight = Decimal(1) / Decimal(len(self._symbols))
        self._allocated = False

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        del context  # buy-and-hold ignores all observations
        if self._allocated:
            return None
        self._allocated = True
        return {symbol: self._weight for symbol in self._symbols}
