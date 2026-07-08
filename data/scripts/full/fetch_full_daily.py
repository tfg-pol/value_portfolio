
from __future__ import annotations

import argparse
from pathlib import Path

import _bulk

_STRING_COLS = ("ticker",)
_DATE_COLS = ("date", "lastupdated")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", default=None, help="default: data/sharadar_full/daily/")
    parser.add_argument(
        "--keep-zip", action="store_true", help="keep the downloaded CSV zip after conversion"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _bulk.FULL_DATA_DIR / "daily"
    _bulk.ingest(
        "SHARADAR/DAILY",
        out_dir=out_dir,
        stem="daily",
        string_cols=_STRING_COLS,
        date_cols=_DATE_COLS,
        report_date_col="date",
        keep_zip=args.keep_zip,
    )


if __name__ == "__main__":
    main()
