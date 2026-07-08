
from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from value_portfolio.agent import DecisionContext, RebalancingAgent
from value_portfolio.learning.selection import select_top_scored


class ScoreTopK(RebalancingAgent):

    def __init__(
        self,
        symbols: Sequence[str] | None = None,
        top_k: int = 20,
        rebalance_every: int = 21,
    ) -> None:
        super().__init__(rebalance_every)
        if symbols is not None and len(set(symbols)) != len(symbols):
            raise ValueError(f"ScoreTopK symbols must be unique, got {list(symbols)}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self._symbols = list(symbols) if symbols is not None else None
        self._top_k = top_k

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        if not self._should_rebalance():
            return None
        if context.scores is None:
            return None

        candidates = self._symbols if self._symbols is not None else context.scores.symbols()
        selected = select_top_scored(
            context.scores,
            context.now,
            candidates,
            self._top_k,
            universe=context.universe,
        )
        if not selected:
            return None

        weight = Decimal(1) / Decimal(len(selected))
        return {symbol: weight for symbol, _ in selected}
