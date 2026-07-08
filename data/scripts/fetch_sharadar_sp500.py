
from __future__ import annotations

import argparse
from pathlib import Path

import _sharadar as sh
import pandas as pd


def reconstruct_intervals(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[events["action"].isin(["added", "removed"])].copy()
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev.sort_values(["ticker", "date", "action"], kind="stable")

    inception = ev["date"].min()

    tickers_with_add = set(ev.loc[ev["action"] == "added", "ticker"])

    one_day = pd.Timedelta(days=1)
    rows: list[tuple[str, pd.Timestamp, pd.Timestamp | None]] = []
    for ticker, grp in ev.groupby("ticker", sort=True):
        open_start: pd.Timestamp | None = None
        for action, date in zip(grp["action"], grp["date"], strict=True):
            if action == "added":
                if open_start is None:
                    open_start = date
            else:  # removed
                if open_start is None:
                    if ticker in tickers_with_add:
                        continue

                    open_start = inception
                    print(
                        f"  anomaly: {ticker} removed on {date.date()} with no "
                        f"'added' event anywhere; anchoring start at inception "
                        f"{inception.date()}."
                    )
                last_day = date - one_day
                if last_day >= open_start:
                    rows.append((ticker, open_start, last_day))
                open_start = None
        if open_start is not None:
            rows.append((ticker, open_start, None))

    out = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])
    out["start_date"] = out["start_date"].dt.strftime("%Y-%m-%d")
    out["end_date"] = out["end_date"].dt.strftime("%Y-%m-%d").fillna("")
    return out


def _sep_available_tickers(tickers: list[str]) -> set[str]:
    meta = sh.fetch_table("SHARADAR/TICKERS", tickers=tickers, table="SEP")
    return set(meta["ticker"]) if not meta.empty else set()


def filter_universe(membership: pd.DataFrame, since: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(since)
    end = pd.to_datetime(membership["end_date"].replace("", None))
    active = end.isna() | (end >= cutoff)
    out = membership.loc[active].copy()

    n_dropped_time = membership["ticker"].nunique() - out["ticker"].nunique()
    print(f"Filter (member on/after {since}): dropped {n_dropped_time} pre-cutoff-only tickers.")

    sep_tickers = _sep_available_tickers(sorted(out["ticker"].unique()))
    untestable = sorted(set(out["ticker"]) - sep_tickers)
    if untestable:
        print(f"Filter (SEP coverage): dropped {len(untestable)} untestable: {untestable}")
        out = out.loc[out["ticker"].isin(sep_tickers)].copy()
    return out.reset_index(drop=True)


def _reconcile(membership: pd.DataFrame, events: pd.DataFrame) -> None:
    open_spells = set(membership.loc[membership["end_date"] == "", "ticker"])
    current = set(events.loc[events["action"] == "current", "ticker"])
    print(
        f"Reconciliation: {len(open_spells)} open spells vs {len(current)} 'current' constituents."
    )
    only_open = open_spells - current
    only_current = current - open_spells
    if only_open:
        print(f"  open but not in current snapshot ({len(only_open)}): {sorted(only_open)}")
    if only_current:
        print(f"  current but no open spell ({len(only_current)}): {sorted(only_current)}")
    if not only_open and not only_current:
        print("  exact match.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory (default: data/sharadar/)",
    )
    parser.add_argument(
        "--since",
        default=sh.DEFAULT_SINCE,
        help=(
            "keep only constituents that were members on/after this date "
            f"(YYYY-MM-DD; default {sh.DEFAULT_SINCE})"
        ),
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else sh.DATA_DIR

    sh.configure_api()
    events = sh.fetch_table("SHARADAR/SP500")
    if events.empty:
        raise SystemExit("SHARADAR/SP500 returned no rows; check API access.")

    sh.write_both(events, "sp500_constituents", sort_by=["date", "ticker"], out_dir=out_dir)

    full = reconstruct_intervals(events)
    _reconcile(full, events)

    membership = filter_universe(full, args.since)
    sh.write_both(membership, "sp500_membership", sort_by=["ticker", "start_date"], out_dir=out_dir)

    n_tickers = membership["ticker"].nunique()
    n_spells = len(membership)
    print(
        f"Testable universe (since {args.since}): {n_tickers} unique tickers "
        f"across {n_spells} membership spells."
    )


if __name__ == "__main__":
    main()
