
from __future__ import annotations

import argparse
from pathlib import Path

import yfinance as yf

_DATA_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ticker", default="^SP500TR", help="yfinance ticker (default: ^SP500TR)")
    parser.add_argument("--start", default="2000-01-01", help="start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="end date YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--output",
        default=None,
        help="output CSV path (default: data/<ticker>.csv)",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else _DATA_DIR / f"{args.ticker.lstrip('^')}.csv"

    print(f"Downloading {args.ticker} from {args.start} to {args.end or 'today'} ...")
    frame = yf.download(
        args.ticker,
        start=args.start,
        end=args.end,
        auto_adjust=False,
        progress=False,
    )
    if frame is None or frame.empty:
        raise SystemExit(f"yfinance returned no data for {args.ticker!r}; check the ticker/dates.")

    if hasattr(frame.columns, "nlevels") and frame.columns.nlevels > 1:
        frame.columns = frame.columns.get_level_values(0)

    frame = frame[["Close"]].round(2)

    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output)
    print(f"Wrote {len(frame)} rows to {output}")


if __name__ == "__main__":
    main()
