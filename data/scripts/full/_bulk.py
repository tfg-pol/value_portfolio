"""Shared helpers for the ``fetch_full_*.py`` bulk ingestors in this folder.

These scripts mirror *entire* Sharadar tables — every ticker, full available
history — for the broad-universe stage-1 experiments, in contrast to the
``data/scripts/fetch_sharadar_*.py`` ingestors, which are restricted to the
ever-S&P-500 universe. At this volume (tens of millions of rows) the paginated
``get_table`` API is impractical, so we use the Nasdaq Data Link bulk-export
endpoint instead: the server bundles the whole (optionally filtered) table into
a zipped CSV, ``nasdaqdatalink.export_table`` polls every 30s until the file is
generated and downloads it, and we convert it to Parquet in bounded-memory
chunks with an explicit per-table schema.

Deliberate deviations from the small ingestors, at this scale:

- Parquet only, no CSV mirror — the mirror would double multi-GB disk usage
  for no auditability gain; ``--keep-zip`` retains the raw vendor CSV instead.
- No global (ticker, date) sort — sorting tens of millions of rows requires
  the full table in memory; every consumer sorts after filtering anyway
  (e.g. ``DailyMarketCap.from_parquet``).

API key from ``SHARADAR_US_BUNDLE_API_KEY`` (via ``SharadarSettings``), never
hardcoded. Output goes under ``data/sharadar_full/``.
"""

from __future__ import annotations

import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import nasdaqdatalink as ndl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from value_portfolio.config import SharadarSettings

# data/scripts/full/_bulk.py -> data/sharadar_full/
FULL_DATA_DIR = Path(__file__).resolve().parents[2] / "sharadar_full"

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 60.0
_CHUNK_ROWS = 1_000_000


def configure_api() -> None:
    """Authenticate ``nasdaqdatalink`` from ``SHARADAR_US_BUNDLE_API_KEY``."""
    settings = SharadarSettings()
    ndl.ApiConfig.api_key = settings.us_bundle_api_key


def download_export(table_code: str, dest_zip: Path, **filters: object) -> Path:
    """Bulk-export ``table_code`` (with optional column filters) to ``dest_zip``.

    ``export_table`` itself polls the server until the file is generated; the
    retry loop here only covers transport failures, with a slow backoff because
    each attempt re-triggers a server-side export of the whole table.
    """
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            print(f"Requesting bulk export of {table_code} {filters or ''} ...", flush=True)
            ndl.export_table(table_code, filename=str(dest_zip), **filters)
            return dest_zip
        except Exception as exc:
            last_exc = exc
            wait = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(
                f"  ! {table_code} export failed (attempt {attempt}/{_MAX_RETRIES}): "
                f"{exc}; retrying in {wait:.0f}s",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(f"{table_code} export failed after {_MAX_RETRIES} attempts") from last_exc


@dataclass(frozen=True)
class ConversionStats:
    rows: int
    tickers: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp


def zip_csv_to_parquet(
    zip_path: Path,
    parquet_path: Path,
    *,
    string_cols: Sequence[str],
    date_cols: Sequence[str],
    report_date_col: str,
) -> ConversionStats:
    """Convert the single CSV inside ``zip_path`` to ``parquet_path`` in
    ``_CHUNK_ROWS``-row chunks (bounded memory regardless of table size).

    Columns are typed explicitly — ``string_cols`` as strings, ``date_cols``
    as timestamps, everything else cast to float64 — so every chunk produces
    the identical Arrow schema (inference per chunk could flip e.g. an
    all-missing column between int and float and break the writer).
    """
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"{zip_path} should contain exactly one CSV, found {members}")

        rows = 0
        tickers: set[str] = set()
        date_min: pd.Timestamp | None = None
        date_max: pd.Timestamp | None = None
        writer: pq.ParquetWriter | None = None

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with archive.open(members[0]) as handle:
                reader = pd.read_csv(
                    handle,
                    chunksize=_CHUNK_ROWS,
                    dtype={col: "string" for col in string_cols},
                )
                for chunk in reader:
                    for col in date_cols:
                        chunk[col] = pd.to_datetime(chunk[col], format="%Y-%m-%d")
                    typed = {*string_cols, *date_cols}
                    for col in chunk.columns:
                        if col not in typed:
                            chunk[col] = chunk[col].astype("float64")

                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(parquet_path, table.schema)
                    writer.write_table(table)

                    rows += len(chunk)
                    tickers.update(chunk["ticker"].dropna())
                    lo, hi = chunk[report_date_col].min(), chunk[report_date_col].max()
                    date_min = lo if date_min is None else min(date_min, lo)
                    date_max = hi if date_max is None else max(date_max, hi)
                    print(f"  ... {rows:,} rows converted", flush=True)
        finally:
            if writer is not None:
                writer.close()

    if rows == 0 or date_min is None or date_max is None:
        raise RuntimeError(f"{zip_path} contained no rows")
    return ConversionStats(rows=rows, tickers=len(tickers), date_min=date_min, date_max=date_max)


def ingest(
    table_code: str,
    *,
    out_dir: Path,
    stem: str,
    string_cols: Sequence[str],
    date_cols: Sequence[str],
    report_date_col: str,
    keep_zip: bool,
    **filters: object,
) -> None:
    """Download ``table_code`` in full and write ``{out_dir}/{stem}.parquet``."""
    configure_api()
    zip_path = out_dir / f"{stem}.zip"
    parquet_path = out_dir / f"{stem}.parquet"

    download_export(table_code, zip_path, **filters)
    size_mb = zip_path.stat().st_size / 1e6
    print(f"Downloaded {zip_path.name} ({size_mb:,.0f} MB); converting to Parquet ...", flush=True)

    stats = zip_csv_to_parquet(
        zip_path,
        parquet_path,
        string_cols=string_cols,
        date_cols=date_cols,
        report_date_col=report_date_col,
    )
    if not keep_zip:
        zip_path.unlink()

    print(
        f"Done: {stats.rows:,} rows, {stats.tickers:,} tickers, "
        f"{report_date_col} {stats.date_min.date()} -> {stats.date_max.date()} "
        f"written to {parquet_path}.",
        flush=True,
    )
