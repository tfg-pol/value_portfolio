
from __future__ import annotations

import argparse
from pathlib import Path

import _bulk

_AS_REPORTED_DIMENSIONS = ["ARQ", "ART", "ARY"]
_STRING_COLS = ("ticker", "dimension", "fiscalperiod")
_DATE_COLS = ("calendardate", "datekey", "reportperiod", "lastupdated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", default=None, help="default: data/sharadar_full/sf1/")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=_AS_REPORTED_DIMENSIONS,
        help="SF1 dimensions to export (default: as-reported only)",
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="keep the downloaded CSV zip after conversion"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _bulk.FULL_DATA_DIR / "sf1"
    _bulk.ingest(
        "SHARADAR/SF1",
        out_dir=out_dir,
        stem="sf1",
        string_cols=_STRING_COLS,
        date_cols=_DATE_COLS,
        report_date_col="datekey",
        keep_zip=args.keep_zip,
        dimension=list(args.dimensions),
    )


if __name__ == "__main__":
    main()
