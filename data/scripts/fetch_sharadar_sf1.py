from __future__ import annotations

import argparse
from pathlib import Path

import _sharadar as sh

# As-reported dimensions only — see the module docstring on restatement leakage.
_AS_REPORTED_DIMENSIONS = ["ARQ", "ART", "ARY"]


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
    parser.add_argument("--output-dir", default=None, help="default: data/sharadar/sf1/")
    parser.add_argument(
        "--tickers",
        default=None,
        help="comma-separated override of the universe (e.g. AAPL,LEHMQ); for dev runs",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="cap to the first N tickers (0 = all); for dev runs"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else sh.DATA_DIR / "sf1"
    tickers = _select_tickers(args.tickers, args.limit)
    if not tickers:
        raise SystemExit("no tickers selected")

    sh.configure_api()
    print(
        f"Downloading SF1 fundamentals for {len(tickers)} tickers "
        f"(dimensions {_AS_REPORTED_DIMENSIONS}, datekey-aligned) ..."
    )
    fundamentals = sh.fetch_table(
        "SHARADAR/SF1", tickers=tickers, dimension=_AS_REPORTED_DIMENSIONS
    )
    if fundamentals.empty:
        raise SystemExit("SHARADAR/SF1 returned no rows for the selected tickers.")

    sh.write_both(fundamentals, "sf1", sort_by=["ticker", "datekey", "dimension"], out_dir=out_dir)
    print(
        f"Done: {fundamentals['ticker'].nunique()} tickers, {len(fundamentals):,} "
        f"statement rows across dimensions {sorted(fundamentals['dimension'].unique())}."
    )


if __name__ == "__main__":
    main()
