
from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from value_portfolio.agent import DecisionContext, RebalancingAgent, require_unique_symbols


class EqualWeight(RebalancingAgent):
    def __init__(self, symbols: Sequence[str], rebalance_every: int = 1) -> None:
        super().__init__(rebalance_every)
        self._symbols = require_unique_symbols(symbols, "EqualWeight")
        self._weight = Decimal(1) / Decimal(len(self._symbols))

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        del context  # equal-weight ignores all observations
        if not self._should_rebalance():
            return None
        return {symbol: self._weight for symbol in self._symbols}
