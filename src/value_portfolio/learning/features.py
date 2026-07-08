
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np

from value_portfolio.data.fundamentals import FundamentalsDataSource
from value_portfolio.data.universe import Universe
from value_portfolio.learning._asof import AsOfSeries

_DAILY_PATH = Path(__file__).resolve().parents[3] / "data" / "sharadar" / "daily" / "daily.parquet"

_MAX_STALENESS_DAYS = 10
_MAX_STALENESS_NS = _MAX_STALENESS_DAYS * 86_400_000_000_000


def forward_return(
    prices: AsOfSeries, symbol: str, now: datetime, after: datetime
) -> float | None:

    entry = prices.lookup(symbol, now)
    if entry is None:
        return None
    now_ns = int(np.datetime64(now.replace(tzinfo=None), "ns").astype(np.int64))
    if now_ns - entry[0] > _MAX_STALENESS_NS:
        return None
    exit_ = prices.lookup(symbol, after)
    if exit_ is None or exit_[0] <= entry[0]:
        return None
    return exit_[1] / entry[1] - 1.0


@dataclass(frozen=True)
class CrossSection:
    date: datetime
    symbols: tuple[str, ...]
    features: np.ndarray
    target: np.ndarray
    industry: np.ndarray | None = None


class DailyMarketCap:

    def __init__(self, series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        # series: symbol -> (dates as int64 nanoseconds ascending, marketcap float64)
        self._asof = AsOfSeries(series)

    @classmethod
    def from_parquet(
        cls, symbols: Sequence[str] | None = None, path: Path | str = _DAILY_PATH
    ) -> DailyMarketCap:
        return cls(AsOfSeries.from_parquet(path, "marketcap", symbols=symbols).series)

    def value_at(self, symbol: str, as_of: datetime) -> float | None:
        hit = self._asof.lookup(symbol, as_of)
        if hit is None:
            return None
        ts_ns, value = hit
        stamp = int(np.datetime64(as_of.replace(tzinfo=None), "ns").astype(np.int64))
        if stamp - ts_ns > _MAX_STALENESS_DAYS * 86_400_000_000_000:
            return None
        return value if value > 0 else None


def month_end_dates(timeline: Sequence[datetime]) -> list[datetime]:
    month_last: dict[tuple[int, int], datetime] = {}
    for stamp in timeline:
        month_last[(stamp.year, stamp.month)] = stamp
    return sorted(month_last.values())


def rank_normalize(features: np.ndarray) -> np.ndarray:
    normalized = np.zeros_like(features, dtype=np.float64)
    for col in range(features.shape[1]):
        column = features[:, col]
        valid = ~np.isnan(column)
        n = int(valid.sum())
        if n < 2:
            continue
        order = np.argsort(column[valid], kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        normalized[valid, col] = ranks / (n - 1) * 2.0 - 1.0
    return normalized


# SF1 fields read directly (levels and vendor-computed ratios).
_LEVEL_FIELDS = (
    "revenue",
    "gp",
    "opinc",
    "ebit",
    "netinc",
    "ebitda",
    "eps",
    "epsdil",
    "assets",
    "liabilities",
    "equity",
    "debt",
    "cashneq",
    "workingcapital",
    "fcf",
    "ncfo",
    "capex",
    "rnd",
)
_RATIO_FIELDS = (
    "roe",
    "roa",
    "roic",
    "ros",
    "grossmargin",
    "netmargin",
    "ebitdamargin",
    "assetturnover",
    "currentratio",
    "de",
    "payoutratio",
    "divyield",
    "bvps",
    "sps",
    "fcfps",
)

# Every SF1 column the loader must provide for `compute_feature_row`.
REQUIRED_FIELDS: tuple[str, ...] = tuple(sorted({*_LEVEL_FIELDS, *_RATIO_FIELDS, "intexp"}))

_DERIVED_NAMES = (
    "gross_profitability",  # gp / assets (Novy-Marx 2013)
    "fcf_to_assets",
    "debt_to_assets",
    "cash_to_assets",
    "interest_coverage",  # ebit / interest expense
    "accruals_to_assets",  # (netinc - ncfo) / assets (Sloan 1996)
    "delta_roa",  # Piotroski-style YoY trajectory, continuous
    "delta_gross_profitability",
    "delta_grossmargin",
    "delta_assetturnover",
    "delta_debt_to_equity",
    "delta_currentratio",
    "revenue_growth",
    "eps_change",
)

FEATURE_NAMES: tuple[str, ...] = (*_LEVEL_FIELDS, *_RATIO_FIELDS, *_DERIVED_NAMES)

_YEAR_LAG = timedelta(days=365)

_ASSET_SCALED_FIELDS: tuple[str, ...] = tuple(
    f for f in _LEVEL_FIELDS if f not in ("eps", "epsdil", "assets")
)


def scale_levels_by_assets(features: np.ndarray) -> np.ndarray:
    out = features.copy()
    assets = features[:, FEATURE_NAMES.index("assets")]
    safe = assets > 0  # NaN or non-positive assets -> undefined ratios
    for field in _ASSET_SCALED_FIELDS:
        col = FEATURE_NAMES.index(field)
        scaled = np.full_like(assets, np.nan)
        np.divide(features[:, col], assets, out=scaled, where=safe)
        out[:, col] = scaled
    log_assets = np.full_like(assets, np.nan)
    np.log(assets, out=log_assets, where=safe)
    out[:, FEATURE_NAMES.index("assets")] = log_assets
    return out


def compute_feature_row(
    fundamentals: FundamentalsDataSource,
    symbol: str,
    as_of: datetime,
    *,
    dimension: str = "ART",
) -> list[float]:

    def read(field: str, when: datetime = as_of) -> float:
        value = fundamentals.value(symbol, field, when, dimension=dimension)
        return float(value) if value is not None else math.nan

    def div(numerator: float, denominator: float) -> float:
        if math.isnan(numerator) or math.isnan(denominator) or denominator <= 0:
            return math.nan
        return numerator / denominator

    now = {field: read(field) for field in {*_LEVEL_FIELDS, *_RATIO_FIELDS, "intexp"}}
    then = as_of - _YEAR_LAG

    def delta(field: str) -> float:
        return now[field] - read(field, then)

    gross_profitability = div(now["gp"], now["assets"])
    derived = (
        gross_profitability,
        div(now["fcf"], now["assets"]),
        div(now["debt"], now["assets"]),
        div(now["cashneq"], now["assets"]),
        div(now["ebit"], now["intexp"]),
        div(now["netinc"] - now["ncfo"], now["assets"]),
        delta("roa"),
        gross_profitability - div(read("gp", then), read("assets", then)),
        delta("grossmargin"),
        delta("assetturnover"),
        delta("de"),
        delta("currentratio"),
        div(now["revenue"], read("revenue", then)) - 1,
        delta("eps"),
    )
    return [*(now[f] for f in _LEVEL_FIELDS), *(now[f] for f in _RATIO_FIELDS), *derived]

_TARGET_DEFLATOR: dict[str, str] = {"mb": "equity", "ma": "assets"}


def build_cross_sections(
    fundamentals: FundamentalsDataSource,
    universe: Universe,
    marketcap: DailyMarketCap,
    dates: Sequence[datetime],
    *,
    target: Literal["cap", "mb", "ma", "ret"] = "cap",
    industry_map: Mapping[str, int] | None = None,
    dimension: str = "ART",
    min_names: int = 50,
    min_features: int = 10,
    prices: AsOfSeries | None = None,
) -> list[CrossSection]:
    if target == "ret" and prices is None:
        raise ValueError("the 'ret' (forward-return) target requires a `prices` source")
    deflator_field = _TARGET_DEFLATOR.get(target)
    deflator_idx = FEATURE_NAMES.index(deflator_field) if deflator_field is not None else None
    # Forward-return label needs the next month's date; the last date has none.
    next_date = dict(pairwise(dates)) if target == "ret" else {}
    dropped_nonpositive_deflator = 0
    sections: list[CrossSection] = []
    for date in dates:
        symbols: list[str] = []
        rows: list[list[float]] = []
        targets: list[float] = []
        for symbol in sorted(universe.members_at(date)):
            cap = marketcap.value_at(symbol, date)
            if cap is None:
                continue
            row = compute_feature_row(fundamentals, symbol, date, dimension=dimension)
            if sum(not math.isnan(v) for v in row) < min_features:
                continue
            if target == "ret":
                assert prices is not None  # guarded above
                # NaN label where no forward return exists (see the docstring).
                after = next_date.get(date)
                ret = forward_return(prices, symbol, date, after) if after is not None else None
                target_value = ret if ret is not None else math.nan
            elif deflator_idx is not None:
                deflator = row[deflator_idx]
                if math.isnan(deflator) or deflator <= 0:
                    dropped_nonpositive_deflator += 1
                    continue
                target_value = math.log(cap) - math.log(deflator)
            else:
                target_value = math.log(cap)
            symbols.append(symbol)
            rows.append(row)
            targets.append(target_value)

        if len(symbols) < min_names:
            continue
        target_arr = np.asarray(targets, dtype=np.float64)
        if target == "ret":
            finite = ~np.isnan(target_arr)
            mean = float(target_arr[finite].mean()) if finite.any() else 0.0
            target_arr = target_arr - mean
        else:
            target_arr = target_arr - target_arr.mean()
        industry = (
            np.asarray([industry_map.get(s, math.nan) for s in symbols], dtype=np.float64)
            if industry_map is not None
            else None
        )
        sections.append(
            CrossSection(
                date=date,
                symbols=tuple(symbols),
                features=np.asarray(rows, dtype=np.float64),
                target=target_arr,
                industry=industry,
            )
        )
    if deflator_field is not None and dropped_nonpositive_deflator:
        print(
            f"{target} target: dropped {dropped_nonpositive_deflator} name-months "
            f"with missing or non-positive {deflator_field}."
        )
    return sections
