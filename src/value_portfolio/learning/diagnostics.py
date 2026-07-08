
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

_DEFAULT_MIN_NAMES = 30


@dataclass(frozen=True, slots=True)
class SignalMonth:
    date: datetime
    n_names: int
    rank_ic: float
    decile_spread: float | None


@dataclass(frozen=True, slots=True)
class SignalDiagnostics:
    months: tuple[SignalMonth, ...]
    nw_lags: int

    @property
    def ic_series(self) -> tuple[float, ...]:
        return tuple(m.rank_ic for m in self.months)

    @property
    def mean_ic(self) -> float:
        return float(np.mean(self.ic_series)) if self.months else math.nan

    @property
    def ic_tstat(self) -> float:
        return newey_west_tstat(self.ic_series, lags=self.nw_lags)

    @property
    def ic_hit_rate(self) -> float:
        if not self.months:
            return math.nan
        return sum(m.rank_ic > 0 for m in self.months) / len(self.months)

    @property
    def spread_series(self) -> tuple[float, ...]:
        return tuple(m.decile_spread for m in self.months if m.decile_spread is not None)

    @property
    def mean_spread(self) -> float:
        series = self.spread_series
        return float(np.mean(series)) if series else math.nan

    @property
    def spread_tstat(self) -> float:
        return newey_west_tstat(self.spread_series, lags=self.nw_lags)

    def summary(self) -> str:
        return (
            f"months={len(self.months)}  "
            f"rank IC mean={self.mean_ic:+.4f} (NW t={self.ic_tstat:+.2f}, "
            f"hit rate={self.ic_hit_rate:.0%})  "
            f"decile spread (gross)={self.mean_spread:+.4%}/mo (NW t={self.spread_tstat:+.2f})"
        )


def spearman_rank_ic(scores: np.ndarray, forward_returns: np.ndarray) -> float:

    if scores.shape != forward_returns.shape or scores.ndim != 1:
        raise ValueError("scores and forward_returns must be equal-length 1-D arrays")
    if len(scores) < 3:
        return math.nan
    rs, rr = _average_ranks(scores), _average_ranks(forward_returns)
    rs_c, rr_c = rs - rs.mean(), rr - rr.mean()
    denom = math.sqrt(float(rs_c @ rs_c) * float(rr_c @ rr_c))
    if denom == 0:
        return math.nan
    return float(rs_c @ rr_c) / denom


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average ranks within tied groups.
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def decile_spread(
    scores: np.ndarray, forward_returns: np.ndarray, *, n_deciles: int = 10
) -> float | None:
    n = len(scores)
    if n < 2 * n_deciles:
        return None
    order = np.argsort(scores, kind="stable")
    size = n // n_deciles
    bottom = forward_returns[order[:size]]
    top = forward_returns[order[-size:]]
    return float(top.mean() - bottom.mean())


def newey_west_tstat(series: Sequence[float], *, lags: int = 3) -> float:

    x = np.asarray(series, dtype=np.float64)
    t = len(x)
    if t < 2:
        return math.nan
    demeaned = x - x.mean()
    gamma0 = float(demeaned @ demeaned) / t
    lrv = gamma0
    for lag in range(1, min(lags, t - 1) + 1):
        gamma = float(demeaned[lag:] @ demeaned[:-lag]) / t
        lrv += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    if lrv <= 0:
        return math.nan
    return float(x.mean() / math.sqrt(lrv / t))


def evaluate_signal(
    scores_by_date: Mapping[datetime, Mapping[str, float]],
    returns_by_date: Mapping[datetime, Mapping[str, float]],
    *,
    n_deciles: int = 10,
    min_names: int = _DEFAULT_MIN_NAMES,
    nw_lags: int = 3,
) -> SignalDiagnostics:

    months: list[SignalMonth] = []
    for date in sorted(scores_by_date.keys() & returns_by_date.keys()):
        scores_map, returns_map = scores_by_date[date], returns_by_date[date]
        symbols = sorted(scores_map.keys() & returns_map.keys())
        if len(symbols) < min_names:
            continue
        scores = np.asarray([scores_map[s] for s in symbols], dtype=np.float64)
        rets = np.asarray([returns_map[s] for s in symbols], dtype=np.float64)
        months.append(
            SignalMonth(
                date=date,
                n_names=len(symbols),
                rank_ic=spearman_rank_ic(scores, rets),
                decile_spread=decile_spread(scores, rets, n_deciles=n_deciles),
            )
        )
    return SignalDiagnostics(months=tuple(months), nw_lags=nw_lags)
