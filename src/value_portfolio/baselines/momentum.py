
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal

from value_portfolio.agent import DecisionContext, RebalancingAgent, require_unique_symbols

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ZERO = Decimal("0")


class Momentum(RebalancingAgent):
    def __init__(
        self,
        symbols: Sequence[str],
        lookback: int = 126,
        skip: int = 0,
        top_k: int | None = None,
        rebalance_every: int = 21,
        timeframe: str = "1Day",
    ) -> None:
        super().__init__(rebalance_every)
        self._symbols = require_unique_symbols(symbols, "Momentum")
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if skip < 0:
            raise ValueError(f"skip must be >= 0, got {skip}")
        resolved_top_k = top_k if top_k is not None else max(1, len(self._symbols) // 2)
        if resolved_top_k < 1 or resolved_top_k > len(self._symbols):
            raise ValueError(f"top_k must be in [1, {len(self._symbols)}], got {top_k}")

        self._lookback = lookback
        self._skip = skip
        self._top_k = resolved_top_k
        self._timeframe = timeframe

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        # Momentum reads prices, not account state.
        if not self._should_rebalance():
            return None

        needed = self._lookback + self._skip + 1
        scored: dict[str, Decimal] = {}
        for symbol in self._symbols:
            bars = context.data.get_bars(symbol, _EPOCH, context.now, self._timeframe)
            if len(bars) < needed:
                continue
            recent = bars[-1 - self._skip]
            past = bars[-1 - self._skip - self._lookback]
            if past.close <= _ZERO:
                continue
            scored[symbol] = (recent.close - past.close) / past.close

        if not scored:
            return None

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        winners = [symbol for symbol, _ in ranked[: self._top_k]]
        weight = Decimal(1) / Decimal(len(winners))
        return {symbol: weight for symbol in winners}
