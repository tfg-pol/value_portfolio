
from __future__ import annotations

import argparse
from pathlib import Path

import _sharadar as sh


def _select_tickers(tickers_arg: str | None, limit: int) -> list[str]:
    if tickers_arg:
        tickers = [t.strip().upper() for t in tickers_arg.split(",") if t.strip()]
    else:
        tickers = sh.load_universe_tickers()
    if limit and limit > 0:
        tickers = tickers[:limit]
    return tickers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default=None, help="default: data/sharadar/daily/")
    parser.add_argument(
        "--tickers",
        default=None,
        help="comma-separated override of the universe (e.g. AAPL,LEHMQ); for dev runs",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="cap to the first N tickers (0 = all); for dev runs"
    )
    parser.add_argument(
        "--start",
        default=sh.DEFAULT_SINCE,
        help=f"earliest date YYYY-MM-DD (default {sh.DEFAULT_SINCE}; pass 1900-01-01 for all)",
    )
    parser.add_argument("--end", default=None, help="latest date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else sh.DATA_DIR / "daily"
    tickers = _select_tickers(args.tickers, args.limit)
    if not tickers:
        raise SystemExit("no tickers selected")

    date_filter: dict[str, str] = {}
    if args.start:
        date_filter["gte"] = args.start
    if args.end:
        date_filter["lte"] = args.end

    sh.configure_api()
    print(f"Downloading DAILY valuation ratios for {len(tickers)} tickers ...")
    filters: dict[str, object] = {"date": date_filter} if date_filter else {}
    ratios = sh.fetch_table("SHARADAR/DAILY", tickers=tickers, **filters)
    if ratios.empty:
        raise SystemExit("SHARADAR/DAILY returned no rows for the selected tickers.")

    sh.write_both(ratios, "daily", sort_by=["ticker", "date"], out_dir=out_dir)
    print(f"Done: {ratios['ticker'].nunique()} tickers, {len(ratios):,} daily rows.")


if __name__ == "__main__":
    main()
