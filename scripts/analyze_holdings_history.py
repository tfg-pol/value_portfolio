
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from value_portfolio.data import load_scores_from_parquet
from value_portfolio.data.sharadar import load_universe_from_sharadar
from value_portfolio.learning.selection import select_top_scored

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCORES = _REPO_ROOT / "data" / "scores" / "valuation_gbt_broad_ret.parquet"
_TICKERS = _REPO_ROOT / "data" / "sharadar_full" / "tickers" / "tickers.parquet"

TOP_K = 20


def load_sector_map() -> dict[str, str]:
    import pandas as pd

    frame = pd.read_parquet(_TICKERS, columns=["ticker", "table", "sector"])
    frame = frame[frame["table"] == "SF1"].dropna(subset=["sector"])
    return {str(r.ticker): str(r.sector) for r in frame.itertuples(index=False)}


def month_ends(scores) -> list[datetime]:  # noqa: ANN001
    import pandas as pd

    frame = pd.read_parquet(_SCORES, columns=["date"])
    dates = sorted(frame["date"].unique())
    return [pd.Timestamp(d).to_pydatetime().replace(tzinfo=UTC) for d in dates]


def main() -> None:
    scores = load_scores_from_parquet(_SCORES)
    universe = load_universe_from_sharadar()
    sectors = load_sector_map()
    candidates = sorted(scores.symbols())
    dates = month_ends(scores)

    monthly: list[tuple[datetime, list[str]]] = []
    for date in dates:
        selected = select_top_scored(scores, date, candidates, TOP_K, universe=universe)
        monthly.append((date, [s for s, _ in selected]))

    # --- Sector composition per year (mean count of the 20 held) ---
    print("### SECTOR SHARE BY YEAR (mean names of 20) ###")
    by_year_sector: dict[int, Counter] = defaultdict(Counter)
    by_year_n: dict[int, int] = defaultdict(int)
    for date, held in monthly:
        by_year_n[date.year] += 1
        for sym in held:
            by_year_sector[date.year][sectors.get(sym, "Unknown")] += 1
    all_sectors = sorted({s for c in by_year_sector.values() for s in c})
    header = "year " + " ".join(f"{s[:10]:>11}" for s in all_sectors)
    print(header)
    for year in sorted(by_year_sector):
        n = by_year_n[year]
        row = " ".join(f"{by_year_sector[year][s] / n:11.1f}" for s in all_sectors)
        print(f"{year} {row}")

    # --- Turnover per year (mean fraction of book replaced vs prev month) ---
    print("\n### TURNOVER BY YEAR (mean fraction of 20 replaced) ###")
    turn_by_year: dict[int, list[float]] = defaultdict(list)
    for (_, prev), (date, cur) in zip(monthly, monthly[1:], strict=False):
        replaced = len(set(cur) - set(prev)) / TOP_K
        turn_by_year[date.year].append(replaced)
    for year in sorted(turn_by_year):
        vals = turn_by_year[year]
        print(f"{year}: {sum(vals) / len(vals):.0%}  (n={len(vals)})")
    overall = [v for vals in turn_by_year.values() for v in vals]
    print(f"OVERALL mean monthly turnover: {sum(overall) / len(overall):.0%}")

    # --- Most persistent names ---
    print("\n### MOST PERSISTENT NAMES (months in top-20) ###")
    tenure: Counter = Counter()
    for _, held in monthly:
        tenure.update(held)
    for sym, months in tenure.most_common(25):
        span_dates = [d for d, held in monthly if sym in held]
        print(
            f"{sym:6} {months:3} months  ({span_dates[0].date()} .. {span_dates[-1].date()})  "
            f"sector={sectors.get(sym, '?')}"
        )

    # --- Concentration: distinct names ever held / total slots ---
    print("\n### BREADTH ###")
    ever = set(tenure)
    print(f"{len(ever)} distinct names ever held across {len(monthly)} months.")
    print(f"Mean tenure {sum(tenure.values()) / len(ever):.1f} months per name held.")

    _plot_sector_area(monthly, sectors, all_sectors)


def _plot_sector_area(monthly, sectors, all_sectors) -> None:  # noqa: ANN001
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = [d for d, _ in monthly]
    shares = {s: [] for s in all_sectors}
    for _, held in monthly:
        counts = Counter(sectors.get(sym, "Unknown") for sym in held)
        total = max(len(held), 1)
        for s in all_sectors:
            shares[s].append(counts[s] / total)
    ys = np.array([shares[s] for s in all_sectors])

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab20")
    ax.stackplot(x, ys, labels=all_sectors, colors=[cmap(i) for i in range(len(all_sectors))])
    ax.set_ylabel("Share of the 20 held")
    ax.set_xlabel("Year")
    ax.set_ylim(0, 1)
    ax.margins(x=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8, frameon=False)
    ax.set_title("Sector composition of the top-20 forward-returns book, 2008--2025")
    fig.tight_layout()
    out = _REPO_ROOT / "paper" / "figures" / "sector_composition_ret_top20.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
