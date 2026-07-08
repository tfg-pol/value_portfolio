
from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow.parquet as pq

from value_portfolio.data.fundamentals import FundamentalRecord, InMemoryFundamentals
from value_portfolio.data.in_memory import InMemoryMarketData
from value_portfolio.data.types import Bar
from value_portfolio.data.universe import InMemoryUniverse

_SHARADAR_DIR = Path(__file__).resolve().parents[3] / "data" / "sharadar"
_MEMBERSHIP_PATH = _SHARADAR_DIR / "sp500_membership.csv"
_SEP_PATH = _SHARADAR_DIR / "sep" / "sep.parquet"
_SF1_PATH = _SHARADAR_DIR / "sf1" / "sf1.parquet"

_TIMEFRAME = "1Day"

_DEFAULT_FUNDAMENTAL_FIELDS = (
    "revenue",
    "gp",
    "opinc",
    "netinc",
    "ebitda",
    "eps",
    "epsdil",
    "assets",
    "liabilities",
    "equity",
    "debt",
    "fcf",
    "ncfo",
    "roe",
    "roa",
)
_SF1_IDENTITY_COLUMNS = ("ticker", "dimension", "datekey")


def _to_utc(value: pd.Timestamp | datetime | str) -> datetime:
    ts = pd.Timestamp(value)
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    return cast(datetime, ts.to_pydatetime())


def _dec(value: object) -> Decimal:
    return Decimal(str(value))


def load_universe_from_sharadar(path: Path | str = _MEMBERSHIP_PATH) -> InMemoryUniverse:
    frame = pd.read_csv(path, dtype=str).fillna("")
    membership: dict[str, list[tuple[datetime, datetime | None]]] = {}
    for ticker, start_date, end_date in zip(
        frame["ticker"], frame["start_date"], frame["end_date"], strict=True
    ):
        start = _to_utc(start_date)
        end = _to_utc(end_date) if end_date else None
        membership.setdefault(ticker, []).append((start, end))
    if not membership:
        raise ValueError(f"no membership rows found in {path}")
    return InMemoryUniverse(membership)


def load_bars_from_sharadar(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    price: str = "closeadj",
    path: Path | str = _SEP_PATH,
) -> InMemoryMarketData:
    if not symbols:
        raise ValueError("load_bars_from_sharadar requires at least one symbol")
    if price not in {"closeadj", "closeunadj", "close"}:
        raise ValueError(f"price must be one of closeadj/closeunadj/close, got {price!r}")

    start_naive = start.replace(tzinfo=None)
    end_naive = end.replace(tzinfo=None)
    columns = ["ticker", "date", "open", "high", "low", "close", "closeadj", "closeunadj", "volume"]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("ticker", "in", list(dict.fromkeys(symbols))),
            ("date", ">=", start_naive),
            ("date", "<=", end_naive),
        ],
    )
    if frame.empty:
        raise ValueError(
            f"Sharadar SEP returned no bars for {list(symbols)} over "
            f"[{start.date()}, {end.date()}] (check tickers / window)"
        )

    bars: dict[str, list[Bar]] = {}
    for row in frame.itertuples(index=False):
        close = _dec(row.close)
        target = _dec(getattr(row, price))
        if close > 0:
            factor = target / close
            open_v = _dec(row.open) * factor
            high_v = _dec(row.high) * factor
            low_v = _dec(row.low) * factor
        else:
            open_v = high_v = low_v = target
        bars.setdefault(row.ticker, []).append(
            Bar(
                symbol=row.ticker,
                timestamp=_to_utc(row.date),
                open=open_v,
                high=high_v,
                low=low_v,
                close=target,
                volume=_dec(row.volume),
                timeframe=_TIMEFRAME,
            )
        )
    return InMemoryMarketData(bars)


def load_fundamentals_from_sharadar(
    symbols: Sequence[str] | None = None,
    fields: Sequence[str] | None = None,
    dimensions: Sequence[str] = ("ARQ", "ART", "ARY"),
    path: Path | str = _SF1_PATH,
) -> InMemoryFundamentals:
    requested = list(fields) if fields is not None else list(_DEFAULT_FUNDAMENTAL_FIELDS)
    available = set(pq.read_schema(path).names)  # type: ignore[no-untyped-call]
    value_fields = [f for f in requested if f in available]
    if not value_fields:
        raise ValueError(f"none of the requested fundamental fields exist in {path}: {requested}")

    columns = list(_SF1_IDENTITY_COLUMNS) + value_fields
    filters: list[tuple[str, str, object]] = [("dimension", "in", list(dimensions))]
    if symbols is not None:
        filters.append(("ticker", "in", list(dict.fromkeys(symbols))))
    frame = pd.read_parquet(path, columns=columns, filters=filters)

    records = list(_records_from_frame(frame, value_fields))
    if not records:
        raise ValueError("Sharadar SF1 returned no rows for the requested symbols/dimensions")
    return InMemoryFundamentals(records)


def _records_from_frame(
    frame: pd.DataFrame, value_fields: Sequence[str]
) -> Iterable[FundamentalRecord]:
    for row in frame.itertuples(index=False):
        as_dict = row._asdict()
        values = {
            field: _dec(as_dict[field]) for field in value_fields if not pd.isna(as_dict[field])
        }
        if not values:
            continue
        yield FundamentalRecord(
            symbol=as_dict["ticker"],
            dimension=as_dict["dimension"],
            datekey=_to_utc(as_dict["datekey"]),
            values=values,
        )
