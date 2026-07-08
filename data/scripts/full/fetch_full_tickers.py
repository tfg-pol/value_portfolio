
from __future__ import annotations

import argparse
from pathlib import Path

import _bulk
import nasdaqdatalink as ndl
import pandas as pd

_TABLE_SCOPES = ["SF1", "SEP"]
_DATE_COLS = ("firstpricedate", "lastpricedate", "firstadded", "lastupdated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", default=None, help="default: data/sharadar_full/tickers/")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _bulk.FULL_DATA_DIR / "tickers"
    out_dir.mkdir(parents=True, exist_ok=True)

    _bulk.configure_api()
    print(f"Fetching SHARADAR/TICKERS (table scopes {_TABLE_SCOPES}) ...", flush=True)
    frame = ndl.get_table("SHARADAR/TICKERS", table=_TABLE_SCOPES, paginate=True)
    if frame.empty:
        raise SystemExit("SHARADAR/TICKERS returned no rows")

    for col in _DATE_COLS:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    frame = frame.sort_values(["table", "ticker"], kind="stable").reset_index(drop=True)

    path = out_dir / "tickers.parquet"
    frame.to_parquet(path, index=False)
    print(
        f"Done: {len(frame):,} rows, {frame['ticker'].nunique():,} tickers, "
        f"{frame['category'].nunique()} categories written to {path}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
