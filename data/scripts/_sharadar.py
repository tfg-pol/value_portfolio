"""Shared helpers for the ``fetch_sharadar_*.py`` ingestors: API auth from
``SHARADAR_US_BUNDLE_API_KEY``, batched/paginated downloads with retry, and
Parquet+CSV output under ``data/sharadar/``. Run ``fetch_sharadar_sp500.py``
first — the other ingestors read its membership table for their ticker list.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import nasdaqdatalink as ndl
import pandas as pd

from value_portfolio.config import SharadarSettings

# data/scripts/_sharadar.py -> data/sharadar/
DATA_DIR = Path(__file__).resolve().parent.parent / "sharadar"

# Project start date. The investable universe is restricted to names that were
# S&P 500 members on or after this date *and* have Sharadar SEP price coverage,
# so every universe ticker is actually testable. ~1998 is also where SEP's broad
# coverage begins, so little is lost. Override per-run with ``--since``.
DEFAULT_SINCE = "1998-01-01"

# Tickers per get_table call. The paginated API has a per-call row ceiling
# (~1M rows); 100 tickers keeps even the daily tables (~100 tickers x 6.5k days =
# 650k rows/batch) under it while limiting the number of HTTP round-trips.
_TICKER_BATCH_SIZE = 100
_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 2.0


def configure_api() -> None:
    """Authenticate ``nasdaqdatalink`` from ``SHARADAR_US_BUNDLE_API_KEY``."""
    settings = SharadarSettings()
    ndl.ApiConfig.api_key = settings.us_bundle_api_key


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _get_table_with_retry(table_code: str, **kwargs: object) -> pd.DataFrame:
    """Single ``get_table`` call with exponential backoff on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return ndl.get_table(table_code, paginate=True, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"  ! {table_code} request failed (attempt {attempt}/{_MAX_RETRIES}): "
                f"{exc}; retrying in {wait:.0f}s"
            )
            time.sleep(wait)
    raise RuntimeError(f"{table_code} request failed after {_MAX_RETRIES} attempts") from last_exc


def fetch_table(
    table_code: str,
    *,
    tickers: Sequence[str] | None = None,
    batch_size: int = _TICKER_BATCH_SIZE,
    **filters: object,
) -> pd.DataFrame:
    """Download a Sharadar table; ``filters`` pass through to ``get_table``.

    With ``tickers``, chunk into ``batch_size``-ticker calls (to stay under the
    per-call row cap) and concatenate; with ``None``, fetch the whole table once.
    """
    if tickers is None:
        print(f"Fetching {table_code} (no ticker filter) ...")
        return _get_table_with_retry(table_code, **filters)

    unique = list(dict.fromkeys(tickers))  # de-dupe, preserve order
    n_batches = (len(unique) + batch_size - 1) // batch_size
    frames: list[pd.DataFrame] = []
    for i, batch in enumerate(_batched(unique, batch_size), start=1):
        print(f"Fetching {table_code} batch {i}/{n_batches} ({len(batch)} tickers) ...")
        frame = _get_table_with_retry(table_code, ticker=list(batch), **filters)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_both(
    df: pd.DataFrame,
    stem: str,
    *,
    sort_by: Sequence[str] | None = None,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Write ``df`` as ``{stem}.parquet`` and ``{stem}.csv`` under ``out_dir``,
    stably sorted by ``sort_by`` (present columns) for diffable output.
    """
    target_dir = out_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if sort_by:
        cols = [c for c in sort_by if c in df.columns]
        if cols:
            df = df.sort_values(cols, kind="stable").reset_index(drop=True)

    parquet_path = target_dir / f"{stem}.parquet"
    csv_path = target_dir / f"{stem}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df):,} rows -> {parquet_path.name} + {csv_path.name} (in {target_dir})")
    return parquet_path, csv_path


def load_universe_tickers(membership_path: Path | None = None) -> list[str]:
    """Sorted, unique tickers from the membership table written by
    ``fetch_sharadar_sp500.py`` — the survivorship-free universe (incl. delisted).
    """
    path = membership_path or (DATA_DIR / "sp500_membership.csv")
    if not path.exists():
        raise SystemExit(
            f"{path} not found; run fetch_sharadar_sp500.py first to build the universe."
        )
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise SystemExit(f"{path} is missing the required 'ticker' column")
    return sorted(df["ticker"].dropna().astype(str).unique())
