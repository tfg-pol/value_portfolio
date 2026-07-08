
from __future__ import annotations

import argparse
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
from _cli import parse_date
from _results import record_run

from value_portfolio.data.sharadar import load_universe_from_sharadar
from value_portfolio.learning._asof import AsOfSeries
from value_portfolio.learning.diagnostics import SignalDiagnostics, evaluate_signal

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEP_PATH = _REPO_ROOT / "data" / "sharadar" / "sep" / "sep.parquet"
_SCORES_DIR = _REPO_ROOT / "data" / "scores"

_MAX_PRICE_STALENESS_NS = 10 * 86_400_000_000_000  # at the score date itself


def _load_score_panel(
    path: Path, start: datetime, end: datetime
) -> dict[datetime, dict[str, float]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    panel: dict[datetime, dict[str, float]] = {}
    for date, group in frame.groupby("date"):
        stamp = date.to_pydatetime().replace(tzinfo=UTC)
        if start <= stamp <= end:
            panel[stamp] = dict(zip(group["ticker"], group["score"].astype(float), strict=True))
    return panel


def _load_prices(symbols: set[str]) -> AsOfSeries:
    return AsOfSeries.from_parquet(_SEP_PATH, "closeadj", symbols=sorted(symbols))


def _forward_returns(
    prices: AsOfSeries,
    panel_dates: list[datetime],
    symbols_by_date: dict[datetime, set[str]],
) -> dict[datetime, dict[str, float]]:
    returns: dict[datetime, dict[str, float]] = {}
    for now, after in pairwise(panel_dates):
        row: dict[str, float] = {}
        now_ns = np.datetime64(now.replace(tzinfo=None), "ns").astype(np.int64)
        for symbol in symbols_by_date[now]:
            entry = prices.lookup(symbol, now)
            if entry is None or now_ns - entry[0] > _MAX_PRICE_STALENESS_NS:
                continue  # no fresh price at t: not actually tradeable
            exit_ = prices.lookup(symbol, after)
            if exit_ is None or exit_[0] <= entry[0]:
                continue  # never traded after t: no realised forward return
            row[symbol] = exit_[1] / entry[1] - 1.0
        returns[now] = row
    return returns


def _diagnostics_payload(result: SignalDiagnostics) -> dict[str, object]:
    return {
        "n_months": len(result.months),
        "mean_rank_ic": result.mean_ic,
        "rank_ic_nw_tstat": result.ic_tstat,
        "ic_hit_rate": result.ic_hit_rate,
        "mean_decile_spread_gross": result.mean_spread,
        "decile_spread_nw_tstat": result.spread_tstat,
        "months": [
            {
                "date": m.date.date().isoformat(),
                "n_names": m.n_names,
                "rank_ic": m.rank_ic,
                "decile_spread_gross": m.decile_spread,
            }
            for m in result.months
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("ridge", "gbt"), default=["ridge", "gbt"])
    parser.add_argument(
        "--target",
        choices=("cap", "mb", "ma", "ret"),
        default="cap",
        help="which stage-1 label's scores to load: cap (log market cap), "
        "mb (log market-to-book), ma (log market-to-assets) or ret (forward return)",
    )
    parser.add_argument(
        "--industry", action="store_true", help="load the industry-aware (_ind) scores"
    )
    parser.add_argument("--start", type=parse_date, default=parse_date("2008-01-01"))
    parser.add_argument("--end", type=parse_date, default=parse_date("2025-12-31"))
    parser.add_argument("--nw-lags", type=int, default=3)
    args = parser.parse_args()

    sp500 = load_universe_from_sharadar()  # the traded / deployment cross-section
    target_tag = "" if args.target == "cap" else f"_{args.target}"
    suffix = f"_broad{target_tag}{'_ind' if args.industry else ''}"

    for model in dict.fromkeys(args.models):
        path = _SCORES_DIR / f"valuation_{model}{suffix}.parquet"
        if not path.exists():
            print(f"[{model}] {path.name} not found; skipped.")
            continue

        panel = _load_score_panel(path, args.start, args.end)
        # Deployment cross-section: index members as of each score date.
        filtered = {
            date: {s: v for s, v in row.items() if s in members}
            for date, row in panel.items()
            if (members := sp500.members_at(date))
        }
        dates = sorted(filtered)
        prices = _load_prices({s for row in filtered.values() for s in row})
        returns = _forward_returns(prices, dates, {d: set(filtered[d]) for d in dates})
        result = evaluate_signal(filtered, returns, nw_lags=args.nw_lags)

        print(f"[{model}] within-S&P-500: {result.summary()}")
        record_run(
            kind="signal_diagnostics",
            slug=f"signal_{model}{suffix}",
            params={
                "model": model,
                "score_universe": "broad",
                "target": args.target,
                "industry": args.industry,
                "within": "sp500",
                "start": args.start.date().isoformat(),
                "end": args.end.date().isoformat(),
                "nw_lags": args.nw_lags,
            },
            payload=_diagnostics_payload(result),
            headline={
                "n_months": len(result.months),
                "mean_rank_ic": round(result.mean_ic, 6),
                "rank_ic_nw_tstat": round(result.ic_tstat, 4),
                "ic_hit_rate": round(result.ic_hit_rate, 4),
                "mean_decile_spread_gross": round(result.mean_spread, 6),
                "decile_spread_nw_tstat": round(result.spread_tstat, 4),
            },
        )


if __name__ == "__main__":
    main()
