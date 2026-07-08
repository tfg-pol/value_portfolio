
from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from value_portfolio.data import load_scores_from_parquet
from value_portfolio.data.sharadar import load_universe_from_sharadar
from value_portfolio.learning._asof import AsOfSeries
from value_portfolio.learning.selection import select_top_scored

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TARGET = sys.argv[1] if len(sys.argv) > 1 else "ret"
_SUFFIX = "" if _TARGET == "cap" else f"_{_TARGET}"
_SCORES = _REPO_ROOT / "data" / "scores" / f"valuation_gbt_broad{_SUFFIX}.parquet"
_SEP = _REPO_ROOT / "data" / "sharadar_full" / "sep" / "sep.parquet"

SNAPSHOT_DATES = [
    datetime(2018, 1, 31, tzinfo=UTC),
    datetime(2020, 1, 31, tzinfo=UTC),
    datetime(2020, 4, 30, tzinfo=UTC),  # just after the COVID crash
    datetime(2023, 1, 31, tzinfo=UTC),
]
TRACK = ("AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA")


def _forward_return(prices: AsOfSeries, symbol: str, now: datetime) -> float | None:
    """Realised ~1-month forward total return (adjusted close), or None."""
    r0 = prices.lookup(symbol, now)
    r1 = prices.lookup(symbol, now + timedelta(days=30))
    if r0 is None or r1 is None or r0[1] == 0.0:
        return None
    return r1[1] / r0[1] - 1.0


def main() -> None:
    scores = load_scores_from_parquet(_SCORES)
    universe = load_universe_from_sharadar()
    prices = AsOfSeries.from_parquet(_SEP, "closeadj", symbols=sorted(scores.symbols()))
    candidates = sorted(scores.symbols())

    for date in SNAPSHOT_DATES:
        selected = select_top_scored(scores, date, candidates, 20, universe=universe)
        print(f"\n=== Top-20 held on {date.date()} (S&P 500 members, by ret score) ===")
        for rank, (symbol, score) in enumerate(selected, 1):
            fwd = _forward_return(prices, symbol, date)
            fwd_txt = f"{fwd:+.1%}" if fwd is not None else "  n/a"
            print(f"{rank:2}. {symbol:6} score={float(score):+.4f}  next-month={fwd_txt}")

    print("\n\n=== Notable names: months held and forward return while held ===")
    all_dates = sorted({date for date in _month_ends(2010, 2025)})
    for symbol in TRACK:
        held_months = []
        for date in all_dates:
            selected = select_top_scored(scores, date, candidates, 20, universe=universe)
            if symbol in {s for s, _ in selected}:
                fwd = _forward_return(prices, symbol, date)
                held_months.append((date, fwd))
        if not held_months:
            print(f"\n{symbol}: never in the top-20.")
            continue
        rets = [f for _, f in held_months if f is not None]
        avg = sum(rets) / len(rets) if rets else float("nan")
        first, last = held_months[0][0].date(), held_months[-1][0].date()
        print(
            f"\n{symbol}: held {len(held_months)} months, {first} -> {last}; "
            f"avg next-month return while held {avg:+.2%}"
        )
        for date, fwd in held_months[:6]:
            fwd_txt = f"{fwd:+.1%}" if fwd is not None else "n/a"
            print(f"    {date.date()}  next-month={fwd_txt}")
        if len(held_months) > 6:
            print(f"    ... (+{len(held_months) - 6} more)")


def _month_ends(start_year: int, end_year: int) -> list[datetime]:
    out = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # last calendar day is fine; select_top_scored reads scores as-of.
            day = 28
            out.append(datetime(year, month, day, tzinfo=UTC))
    return out


if __name__ == "__main__":
    main()
