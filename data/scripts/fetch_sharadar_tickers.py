
from __future__ import annotations

import argparse
from pathlib import Path

import _sharadar as sh


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default=None, help="default: data/sharadar/")
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else sh.DATA_DIR

    tickers = sh.load_universe_tickers()
    sh.configure_api()
    print(f"Downloading TICKERS metadata for {len(tickers)} universe tickers ...")
    meta = sh.fetch_table("SHARADAR/TICKERS", tickers=tickers, table="SEP")
    if meta.empty:
        raise SystemExit("SHARADAR/TICKERS returned no rows for the universe.")

    sh.write_both(meta, "tickers", sort_by=["ticker"], out_dir=out_dir)
    missing = sorted(set(tickers) - set(meta["ticker"]))
    print(f"Done: {len(meta)} metadata rows. {len(missing)} universe tickers without a SEP entry.")
    if missing:
        print(f"  no SEP metadata for: {missing}")


if __name__ == "__main__":
    main()
