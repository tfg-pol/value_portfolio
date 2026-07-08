
from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from value_portfolio.agent import DecisionContext, RebalancingAgent, require_unique_symbols
from value_portfolio.data.base import MarketDataSource

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ZERO = Decimal("0")
_WEIGHT_QUANTUM = Decimal("0.000001")

Mode = Literal["min_var", "mean_var"]
_VALID_MODES: tuple[Mode, ...] = ("min_var", "mean_var")


class MeanVariance(RebalancingAgent):
    def __init__(
        self,
        symbols: Sequence[str],
        lookback: int = 252,
        mode: Mode = "min_var",
        risk_aversion: Decimal = Decimal("1"),
        ridge: Decimal = Decimal("1e-6"),
        rebalance_every: int = 21,
        timeframe: str = "1Day",
    ) -> None:
        super().__init__(rebalance_every)
        self._symbols = require_unique_symbols(symbols, "MeanVariance")
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        if risk_aversion < _ZERO:
            raise ValueError(f"risk_aversion must be >= 0, got {risk_aversion}")
        if ridge < _ZERO:
            raise ValueError(f"ridge must be >= 0, got {ridge}")

        self._lookback = lookback
        self._mode: Mode = mode
        self._risk_aversion = risk_aversion
        self._ridge = ridge
        self._timeframe = timeframe

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        # Mean-variance reads prices, not account state.
        if not self._should_rebalance():
            return None

        returns_by_symbol = self._collect_returns(context.data, context.now)
        if len(returns_by_symbol) < 2:
            return None

        symbols = list(returns_by_symbol)
        returns_matrix = np.array(
            [returns_by_symbol[symbol] for symbol in symbols],
            dtype=np.float64,
        ).T  # shape (lookback, n)

        mu = returns_matrix.mean(axis=0)
        cov = np.cov(returns_matrix, rowvar=False, ddof=1)
        cov = cov + float(self._ridge) * np.eye(len(symbols))

        weights = self._solve(cov, mu)
        if weights is None:
            return None

        return self._to_decimal_weights(symbols, weights)

    def _collect_returns(
        self,
        data: MarketDataSource,
        now: datetime,
    ) -> dict[str, list[float]]:
        needed_bars = self._lookback + 1
        returns_by_symbol: dict[str, list[float]] = {}
        for symbol in self._symbols:
            bars = data.get_bars(symbol, _EPOCH, now, self._timeframe)
            if len(bars) < needed_bars:
                continue
            closes = [float(bar.close) for bar in bars[-needed_bars:]]
            if any(c <= 0.0 for c in closes[:-1]):
                continue
            returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
            returns_by_symbol[symbol] = returns
        return returns_by_symbol

    def _solve(
        self,
        cov: NDArray[np.float64],
        mu: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        n = cov.shape[0]
        if self._mode == "min_var":

            def objective(w: NDArray[np.float64]) -> float:
                return float(0.5 * w @ cov @ w)

            def gradient(w: NDArray[np.float64]) -> NDArray[np.float64]:
                return cov @ w
        else:
            gamma = float(self._risk_aversion)

            def objective(w: NDArray[np.float64]) -> float:
                return float(0.5 * gamma * (w @ cov @ w) - (mu @ w))

            def gradient(w: NDArray[np.float64]) -> NDArray[np.float64]:
                return gamma * (cov @ w) - mu

        constraints = [
            {
                "type": "eq",
                "fun": lambda w: float(w.sum() - 1.0),
                "jac": lambda w: np.ones_like(w),
            }
        ]
        bounds = [(0.0, 1.0)] * n
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
            equal = Decimal(1) / Decimal(len(symbols))
            return {symbol: equal for symbol in symbols}
        normalised = clipped / total
        return {
            symbol: Decimal(str(float(w))).quantize(_WEIGHT_QUANTUM, rounding=ROUND_DOWN)
            for symbol, w in zip(symbols, normalised, strict=True)
        }
