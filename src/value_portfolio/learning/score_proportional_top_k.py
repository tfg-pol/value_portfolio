
from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_DOWN, Decimal

from value_portfolio.agent import DecisionContext, RebalancingAgent
from value_portfolio.learning.selection import select_top_scored

_ZERO = Decimal(0)
_WEIGHT_QUANTUM = Decimal("0.000001")


class ScoreProportionalTopK(RebalancingAgent):

    def __init__(
        self,
        symbols: Sequence[str] | None = None,
        top_k: int = 20,
        rebalance_every: int = 21,
    ) -> None:
        super().__init__(rebalance_every)
        if symbols is not None and len(set(symbols)) != len(symbols):
            raise ValueError(f"ScoreProportionalTopK symbols must be unique, got {list(symbols)}")
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

        return _proportional_weights(selected)


def _proportional_weights(selected: list[tuple[str, Decimal]]) -> dict[str, Decimal]:
    total = sum((score for _, score in selected if score > _ZERO), _ZERO)
    if total == _ZERO:
        weight = (Decimal(1) / Decimal(len(selected))).quantize(
            _WEIGHT_QUANTUM, rounding=ROUND_DOWN
        )
        return {symbol: weight for symbol, _ in selected}

    weights: dict[str, Decimal] = {}
    for symbol, score in selected:
        if score <= _ZERO:
            continue
        weight = (score / total).quantize(_WEIGHT_QUANTUM, rounding=ROUND_DOWN)
        if weight > _ZERO:
            weights[symbol] = weight
    return weights
