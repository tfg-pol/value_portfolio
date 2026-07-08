
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from value_portfolio.agent import DecisionContext, RebalancingAgent
from value_portfolio.data.base import MarketDataSource
from value_portfolio.learning.selection import select_top_scored

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ZERO = Decimal("0")
_WEIGHT_QUANTUM = Decimal("0.000001")
_TRADING_DAYS = 252.0
# Expected annual return per 1 std of score — turns the unitless score into a
# return magnitude so λ, gamma are O(1). A fixed modeling assumption, not tuned.
_SCORE_RETURN_SCALE = 0.05


class CostAwareAllocator(RebalancingAgent):
    def __init__(
        self,
        symbols: Sequence[str] | None = None,
        top_k: int = 20,
        rebalance_every: int = 21,
        lookback: int = 252,
        risk_aversion: float = 1.0,
        turnover_aversion: float = 1.0,
        max_weight: float = 1.0,
        ridge: float = 1e-6,
        timeframe: str = "1Day",
    ) -> None:
        super().__init__(rebalance_every)
        if symbols is not None and len(set(symbols)) != len(symbols):
            raise ValueError(f"CostAwareAllocator symbols must be unique, got {list(symbols)}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if risk_aversion < 0.0:
            raise ValueError(f"risk_aversion must be >= 0, got {risk_aversion}")
        if turnover_aversion < 0.0:
            raise ValueError(f"turnover_aversion must be >= 0, got {turnover_aversion}")
        if not 0.0 < max_weight <= 1.0:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if ridge < 0.0:
            raise ValueError(f"ridge must be >= 0, got {ridge}")

        self._symbols = list(symbols) if symbols is not None else None
        self._top_k = top_k
        self._lookback = lookback
        self._risk_aversion = risk_aversion
        self._turnover_aversion = turnover_aversion
        self._max_weight = max_weight
        self._ridge = ridge
        self._timeframe = timeframe

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

        # Keep only names with enough price history; preserve selection order so
        # scores, returns and current weights stay aligned.
        score_by_symbol = dict(selected)
        returns_by_symbol = self._collect_returns(
            context.data, context.now, [symbol for symbol, _ in selected]
        )
        symbols = list(returns_by_symbol)
        if len(symbols) < 2:
            return None

        returns_matrix = np.array(
            [returns_by_symbol[symbol] for symbol in symbols], dtype=np.float64
        ).T  # shape (lookback, n)
        cov = np.cov(returns_matrix, rowvar=False, ddof=1) * _TRADING_DAYS
        cov = cov + self._ridge * np.eye(len(symbols))

        mu = _expected_returns([score_by_symbol[symbol] for symbol in symbols])
        w_now = self._current_weights(context, symbols)

        weights = self._solve(cov, mu, w_now)
        if weights is None:
            return None
        return self._to_decimal_weights(symbols, weights)

    def _collect_returns(
        self,
        data: MarketDataSource,
        now: datetime,
        candidates: Sequence[str],
    ) -> dict[str, list[float]]:
        needed_bars = self._lookback + 1
        returns_by_symbol: dict[str, list[float]] = {}
        for symbol in candidates:
            bars = data.get_bars(symbol, _EPOCH, now, self._timeframe)
            if len(bars) < needed_bars:
                continue
            closes = [float(bar.close) for bar in bars[-needed_bars:]]
            if any(c <= 0.0 for c in closes[:-1]):
                continue
            returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
            returns_by_symbol[symbol] = returns
        return returns_by_symbol

    @staticmethod
    def _current_weights(context: DecisionContext, symbols: list[str]) -> NDArray[np.float64]:
        """Current portfolio weight of each selected name (0 if not held)."""
        equity = float(context.account.equity)
        if equity <= 0.0:
            return np.zeros(len(symbols), dtype=np.float64)
        held = {p.symbol: float(p.market_value) / equity for p in context.account.positions}
        return np.array([held.get(symbol, 0.0) for symbol in symbols], dtype=np.float64)

    def _solve(
        self,
        cov: NDArray[np.float64],
        mu: NDArray[np.float64],
        w_now: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        # Minimize (λ/2) wᵀΣw - μᵀw + (gamma/2)‖w - w_now‖².
        n = cov.shape[0]
        lam = self._risk_aversion
        gam = self._turnover_aversion

        def objective(w: NDArray[np.float64]) -> float:
            diff = w - w_now
            return float(0.5 * lam * (w @ cov @ w) - (mu @ w) + 0.5 * gam * (diff @ diff))

        def gradient(w: NDArray[np.float64]) -> NDArray[np.float64]:
            return lam * (cov @ w) - mu + gam * (w - w_now)

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: float(w.sum() - 1.0),
                "jac": lambda w: np.ones_like(w),
            }
        ]
        # Cap per name, but never below 1/n or the sum-to-1 constraint is infeasible.
        upper = max(self._max_weight, 1.0 / n)
        bounds = [(0.0, upper)] * n
        x0 = np.full(n, 1.0 / n, dtype=np.float64)

        result = minimize(  # type: ignore[call-overload]
            objective,
            x0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 200},
        )
        if not bool(result.success):
            return None
        return np.asarray(result.x, dtype=np.float64)

    @staticmethod
    def _to_decimal_weights(
        symbols: list[str],
        weights: NDArray[np.float64],
    ) -> dict[str, Decimal]:
        clipped = np.where(weights < 0.0, 0.0, weights)
        total = float(clipped.sum())
        if total <= 0.0:
            # Degenerate solver output; fall back to equal weight.
            equal = Decimal(1) / Decimal(len(symbols))
            return {symbol: equal for symbol in symbols}
        normalised = clipped / total
        # Quantise *down* so rounding can never push the sum above 1.
        return {
            symbol: Decimal(str(float(w))).quantize(_WEIGHT_QUANTUM, rounding=ROUND_DOWN)
            for symbol, w in zip(symbols, normalised, strict=True)
        }


def _expected_returns(scores: Sequence[Decimal]) -> NDArray[np.float64]:
    """Map raw scores to an expected-return vector: z-score within the selection,
    then scale to an annual-return magnitude. A flat cross-section yields zeros.
    """
    raw = np.array([float(s) for s in scores], dtype=np.float64)
    std = float(raw.std())
    if std == 0.0:
        return np.zeros_like(raw)
    return np.asarray(_SCORE_RETURN_SCALE * (raw - raw.mean()) / std, dtype=np.float64)
