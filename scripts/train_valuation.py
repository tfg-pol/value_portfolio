
from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

from _cli import parse_date

from value_portfolio.data.sharadar import load_fundamentals_from_sharadar
from value_portfolio.data.universe import InMemoryUniverse, Universe
from value_portfolio.learning._asof import AsOfSeries
from value_portfolio.learning.features import (
    FEATURE_NAMES,
    REQUIRED_FIELDS,
    DailyMarketCap,
    build_cross_sections,
    month_end_dates,
)
from value_portfolio.learning.valuation import (
    ValuationConfig,
    fit_predict_expanding,
    write_scores_parquet,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _REPO_ROOT / "data" / "scores"

_FULL_DIR = _REPO_ROOT / "data" / "sharadar_full"
_FULL_DAILY_PATH = _FULL_DIR / "daily" / "daily.parquet"
_FULL_SF1_PATH = _FULL_DIR / "sf1" / "sf1.parquet"
_FULL_SEP_PATH = _FULL_DIR / "sep" / "sep.parquet"
_FULL_TICKERS_PATH = _FULL_DIR / "tickers" / "tickers.parquet"

_ELIGIBLE_CATEGORIES = ("Domestic Common Stock", "Domestic Common Stock Primary Class")


def _trading_dates(start: datetime, end: datetime, daily_path: Path) -> list[datetime]:
    """All DAILY observation dates in [start, end] (the de-facto trading calendar)."""
    import pandas as pd

    dates = pd.read_parquet(daily_path, columns=["date"])["date"].unique()
    stamps = sorted(d.to_pydatetime().replace(tzinfo=UTC) for d in pd.to_datetime(dates))
    return [d for d in stamps if start <= d <= end]


def load_broad_universe(path: Path = _FULL_TICKERS_PATH) -> Universe:
    import pandas as pd

    frame = pd.read_parquet(
        path, columns=["ticker", "table", "category", "firstpricedate", "lastpricedate"]
    )
    frame = frame[(frame["table"] == "SF1") & frame["category"].isin(_ELIGIBLE_CATEGORIES)]
    membership: dict[str, list[tuple[datetime, datetime | None]]] = {}
    for row in frame.itertuples(index=False):
        if pd.isna(row.firstpricedate):
            continue
        start = row.firstpricedate.to_pydatetime().replace(tzinfo=UTC)
        end = (
            None
            if pd.isna(row.lastpricedate)
            else row.lastpricedate.to_pydatetime().replace(tzinfo=UTC)
        )
        membership.setdefault(str(row.ticker), []).append((start, end))
    return InMemoryUniverse(membership)


def load_industry_map(path: Path = _FULL_TICKERS_PATH, *, field: str = "sector") -> dict[str, int]:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["ticker", "table", "category", field])
    frame = frame[(frame["table"] == "SF1") & frame["category"].isin(_ELIGIBLE_CATEGORIES)]
    frame = frame.dropna(subset=[field])
    codes = {value: code for code, value in enumerate(sorted(frame[field].unique()))}
    return {str(row.ticker): codes[getattr(row, field)] for row in frame.itertuples(index=False)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=parse_date, default=parse_date("2000-01-01"))
    parser.add_argument("--end", type=parse_date, default=parse_date("2019-12-31"))
    parser.add_argument(
        "--target",
        choices=("cap", "mb", "ma", "ret"),
        default="cap",
        help="regression label: demeaned log market cap (cap), log market-to-book "
        "(mb; scale-free, Geertsema & Lu 2023), log market-to-assets (ma; "
        "scale-free, positive-by-construction deflator) or t->t+1 forward return "
        "(ret; direct return prediction, Gu-Kelly-Xiu 2020)",
    )
    parser.add_argument("--models", nargs="+", choices=("ridge", "gbt"), default=["ridge", "gbt"])
    parser.add_argument(
        "--industry",
        action="store_true",
        help="add sector as a categorical peer-group feature (GBT only; Ridge "
        "unchanged). Tags output with _ind.",
    )
    parser.add_argument("--burn-in", type=int, default=96, help="monthly sections before scoring")
    parser.add_argument(
        "--scale-features",
        action="store_true",
        help="scale GBT dollar levels by total assets (stationary design matrix) "
        "even with the cap target; tags output with _sc. Ridge is unaffected.",
    )
    parser.add_argument(
        "--shuffle-labels",
        action="store_true",
        help="placebo leak test: permute labels within each date before fitting, "
        "destroying the feature->label link. A clean pipeline then scores ~zero "
        "IC. Tags output with _placebo.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    dates = month_end_dates(_trading_dates(args.start, args.end, _FULL_DAILY_PATH))
    print(f"{len(dates)} monthly cross-section dates ({dates[0].date()} -> {dates[-1].date()}).")

    universe = load_broad_universe()
    tickers = sorted({symbol for date in dates for symbol in universe.members_at(date)})
    print(f"{len(tickers)} tickers ever in the broad universe across the window.")

    fundamentals = load_fundamentals_from_sharadar(
        tickers,
        fields=REQUIRED_FIELDS,
        dimensions=("ART",),
        path=_FULL_SF1_PATH,
    )
    marketcap = DailyMarketCap.from_parquet(tickers, path=_FULL_DAILY_PATH)
    industry_map = load_industry_map() if args.industry else None
    if industry_map is not None:
        print(f"industry: {len(set(industry_map.values()))} sectors as a GBT categorical feature.")
    prices = None
    if args.target == "ret":
        print("loading SEP closeadj prices for forward-return labels...")
        prices = AsOfSeries.from_parquet(_FULL_SEP_PATH, "closeadj", symbols=tickers)
    sections = build_cross_sections(
        fundamentals,
        universe,
        marketcap,
        dates,
        target=args.target,
        industry_map=industry_map,
        prices=prices,
    )
    sizes = [len(s.symbols) for s in sections]
    print(f"{len(FEATURE_NAMES)} features per name (levels, health ratios, YoY trajectory).")
    print(
        f"{len(sections)} usable cross-sections; names/date min={min(sizes)} "
        f"median={sorted(sizes)[len(sizes) // 2]} max={max(sizes)}."
    )

    target_tag = "" if args.target == "cap" else f"_{args.target}"
    ind_tag = "_ind" if args.industry else ""
    sc_tag = "_sc" if args.scale_features else ""
    pl_tag = "_placebo" if args.shuffle_labels else ""
    for model in args.models:
        config = ValuationConfig(
            model=model,
            burn_in_sections=args.burn_in,
            target=args.target,
            scale_features=True if args.scale_features else None,
            industry=args.industry,
            shuffle_labels=args.shuffle_labels,
            seed=args.seed,
        )
        records = fit_predict_expanding(sections, config)
        if not records:
            print(f"[{model}] nothing scored — fewer than burn_in={args.burn_in} sections.")
            continue
        out = args.out / f"valuation_{model}_broad{target_tag}{ind_tag}{sc_tag}{pl_tag}.parquet"
        write_scores_parquet(records, out)
        first, last = records[0].date.date(), records[-1].date.date()
        print(f"[{model}] {len(records)} scores over {first} -> {last} written to {out}.")
        if args.target == "ret":
            mags = sorted(abs(float(rec.score)) for rec in records)
            print(f"[{model}] median |predicted demeaned return|: {mags[len(mags) // 2]:.2%}")
        else:
            median_ape = math.expm1(median(abs(float(rec.score)) for rec in records))
            print(f"[{model}] median absolute valuation error: {median_ape:.1%}")


if __name__ == "__main__":
    main()
