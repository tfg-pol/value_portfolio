"""Smoke check for Sharadar access: fetch Lehman (``LEHMQ``) 2008 prices from
``SHARADAR/SEP`` to confirm the key unlocks delisted-ticker history. A diagnostic,
not part of the pipeline.

Run with: ``uv run python data/scripts/ndl_ingestor.py``
"""

from __future__ import annotations

import _sharadar as sh
import nasdaqdatalink as ndl


def main() -> None:
    sh.configure_api()  # reads SHARADAR_US_BUNDLE_API_KEY from .env — never hardcoded
    px = ndl.get_table(
        "SHARADAR/SEP",
        ticker="LEHMQ",
        date={"gte": "2008-01-01", "lte": "2008-10-31"},
        paginate=True,
    )

    print(f"{len(px)} price rows returned for LEHMQ\n")
    if px.empty:
        print(">>> EMPTY: this key does not open SEP (delisted price history).")
        print(">>> A full SEP subscription is required for survivorship-free prices.")
        return

    print(">>> SUCCESS: real access to delisted price series (SEP).\n")
    cols = ["ticker", "date", "close", "closeadj", "volume"]
    print(px.sort_values("date")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
