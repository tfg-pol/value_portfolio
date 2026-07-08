
from __future__ import annotations

import argparse
import math
from datetime import datetime
from statistics import median

from _cli import parse_date
from signal_diagnostics import _SCORES_DIR, _forward_returns, _load_prices, _load_score_panel
from train_valuation import _FULL_DAILY_PATH, _FULL_SF1_PATH, _trading_dates, load_broad_universe

from value_portfolio.data.scores import ScoreRecord
from value_portfolio.data.sharadar import load_fundamentals_from_sharadar, load_universe_from_sharadar
from value_portfolio.learning.diagnostics import SignalDiagnostics, evaluate_signal
from value_portfolio.learning.features import (
    REQUIRED_FIELDS,
    CrossSection,
    DailyMarketCap,
    build_cross_sections,
    month_end_dates,
)
from value_portfolio.learning.valuation import ValuationConfig, fit_predict_expanding


def _records_to_panel(
    records: list[ScoreRecord], start: datetime, end: datetime
) -> dict[datetime, dict[str, float]]:
    panel: dict[datetime, dict[str, float]] = {}
    for rec in records:
        if start <= rec.date <= end:
            panel.setdefault(rec.date, {})[rec.symbol] = float(rec.score)
    return panel


def _median_ape(panel: dict[datetime, dict[str, float]]) -> float:
    scores = [abs(v) for row in panel.values() for v in row.values()]
    return math.expm1(median(scores)) if scores else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_date, default=parse_date("2000-01-01"))
    parser.add_argument("--end", type=parse_date, default=parse_date("2025-12-31"))
    parser.add_argument("--eval-start", type=parse_date, default=parse_date("2008-01-01"))
    parser.add_argument("--burn-in", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nw-lags", type=int, default=3)
    args = parser.parse_args()

    # --- broad estimation panel (same inputs as train_valuation) ---
    dates = month_end_dates(_trading_dates(args.start, args.end, _FULL_DAILY_PATH))
    universe = load_broad_universe()
    tickers = sorted({s for d in dates for s in universe.members_at(d)})
    print(f"{len(dates)} dates, {len(tickers)} broad tickers.", flush=True)

    fundamentals = load_fundamentals_from_sharadar(
        tickers, fields=REQUIRED_FIELDS, dimensions=("ART",), path=_FULL_SF1_PATH
    )
    marketcap = DailyMarketCap.from_parquet(tickers, path=_FULL_DAILY_PATH)

    print("Building cap and mb cross-sections...", flush=True)
    sections_cap = build_cross_sections(fundamentals, universe, marketcap, dates, target="cap")
    sections_mb = build_cross_sections(fundamentals, universe, marketcap, dates, target="mb")

    def fit(sections: list[CrossSection], target: str, scale: bool) -> list[ScoreRecord]:
        cfg = ValuationConfig(
            model="gbt",
            burn_in_sections=args.burn_in,
            target=target,  # type: ignore[arg-type]
            scale_features=scale,
            seed=args.seed,
        )
        return fit_predict_expanding(sections, cfg)

    # --- the four panels (target x scaling); only the off-diagonal pair is re-fit ---
    print("Fitting cap+scaled ...", flush=True)
    cap_scaled = _records_to_panel(fit(sections_cap, "cap", True), args.eval_start, args.end)
    print("Fitting mb+raw ...", flush=True)
    mb_raw = _records_to_panel(fit(sections_mb, "mb", False), args.eval_start, args.end)

    cap_raw = _load_score_panel(
        _SCORES_DIR / "valuation_gbt_broad.parquet", args.eval_start, args.end
    )
    mb_scaled = _load_score_panel(
        _SCORES_DIR / "valuation_gbt_broad_mb.parquet", args.eval_start, args.end
    )

    cells = {
        ("cap", "raw"): cap_raw,
        ("cap", "scaled"): cap_scaled,
        ("mb", "raw"): mb_raw,
        ("mb", "scaled"): mb_scaled,
    }

    # --- measure every cell with the identical within-S&P-500 pipeline ---
    sp500 = load_universe_from_sharadar()
    filtered = {
        key: {
            d: {s: v for s, v in row.items() if s in members}
            for d, row in panel.items()
            if (members := sp500.members_at(d))
        }
        for key, panel in cells.items()
    }
    prices = _load_prices({s for fp in filtered.values() for row in fp.values() for s in row})

    results: dict[tuple[str, str], SignalDiagnostics] = {}
    for key, fp in filtered.items():
        ds = sorted(fp)
        returns = _forward_returns(prices, ds, {d: set(fp[d]) for d in ds})
        results[key] = evaluate_signal(fp, returns, nw_lags=args.nw_lags)

    # --- table ---
    header = (
        f"\n{'cell':<14}{'median APE':>12}{'rank IC':>11}{'IC t':>8}"
        f"{'spread/mo':>12}{'spread t':>10}{'months':>8}"
    )
    lines = [header, "-" * len(header.strip())]
    for target in ("cap", "mb"):
        for scale in ("raw", "scaled"):
            r = results[(target, scale)]
            lines.append(
                f"{f'{target}+{scale}':<14}"
                f"{_median_ape(cells[(target, scale)]):>11.1%}"
                f"{r.mean_ic:>+11.4f}{r.ic_tstat:>+8.2f}"
                f"{r.mean_spread:>+11.4%}{r.spread_tstat:>+10.2f}{len(r.months):>8}"
            )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
