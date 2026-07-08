
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from value_portfolio import Agent, Rebalancer, SimulatedBroker, run_backtest
from value_portfolio.backtest import BacktestReport, BenchmarkSeries
from value_portfolio.data import (
    FundamentalsDataSource,
    InMemoryMarketData,
    ScoreSource,
    Universe,
    load_scores_from_parquet,
)
from value_portfolio.data.sharadar import (
    load_bars_from_sharadar,
    load_fundamentals_from_sharadar,
    load_universe_from_sharadar,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BENCHMARK_CSV = _REPO_ROOT / "data" / "SP500TR.csv"

# Shared demo parameters (one window / basket / cost model so reports compare).
DEFAULT_START = datetime(2010, 1, 1, tzinfo=UTC)
DEFAULT_END = datetime(2019, 12, 31, tzinfo=UTC)
DEFAULT_MAX_SYMBOLS = 40
STARTING_CASH = Decimal("100000")
COMMISSION_PER_SHARE = Decimal("0.005")
SLIPPAGE_BPS = Decimal("5")


@dataclass(frozen=True, slots=True)
class SharadarBacktest:
    """The shared, ready-to-run inputs for a baseline demo."""

    symbols: list[str]
    data: InMemoryMarketData
    broker: SimulatedBroker
    benchmark: BenchmarkSeries
    universe: Universe
    fundamentals: FundamentalsDataSource
    scores: ScoreSource | None = None


def build_sharadar_backtest(
    *,
    start: datetime = DEFAULT_START,
    end: datetime = DEFAULT_END,
    max_symbols: int = DEFAULT_MAX_SYMBOLS,
    scores_path: Path | None = None,
) -> SharadarBacktest:
    """Load universe, bars, benchmark and fundamentals; print what was loaded and
    return a `SharadarBacktest` whose `symbols` is the basket actually backed by bars.

    With `scores_path` (a stage-1 Parquet from ``scripts/train_valuation.py``),
    the basket is every scored ticker instead of the `max_symbols`-capped slice —
    score-driven strategies select from the whole scored universe, point-in-time
    filtered at each step — and `scores` is threaded onto the backtest.
    """
    universe = load_universe_from_sharadar()
    scores: ScoreSource | None = None
    if scores_path is not None:
        scores = load_scores_from_parquet(scores_path)
        basket = sorted(scores.symbols())
        print(f"Scored basket from {scores_path.name}: {len(basket)} names.")
    else:
        basket = sorted(universe.members_at(start))[:max_symbols]
        print(f"Universe at {start.date()}: {len(basket)} names (capped to {max_symbols}).")

    data = load_bars_from_sharadar(basket, start, end)
    symbols = sorted(data.symbols)
    timeline = data.timeline
    print(
        f"Loaded {len(symbols)} symbols with bars, {len(timeline)} trading days "
        f"({timeline[0].date()} -> {timeline[-1].date()})."
    )

    benchmark = BenchmarkSeries.from_csv(_BENCHMARK_CSV)
    fundamentals = load_fundamentals_from_sharadar(symbols)
    _print_sample_fundamental(fundamentals, symbols, end)

    broker = SimulatedBroker(
        market_data=data,
        starting_cash=STARTING_CASH,
        commission_per_share=COMMISSION_PER_SHARE,
        slippage_bps=SLIPPAGE_BPS,
    )
    return SharadarBacktest(
        symbols=symbols,
        data=data,
        broker=broker,
        benchmark=benchmark,
        universe=universe,
        fundamentals=fundamentals,
        scores=scores,
    )


def run_and_report(
    backtest: SharadarBacktest,
    agent: Agent,
    title: str,
    *,
    extra_lines: Sequence[str] = (),
) -> BacktestReport:
    """Run `agent` over the shared inputs, print the benchmark-relative report,
    and return it (so callers can persist the results).
    """
    report = run_backtest(
        agent,
        backtest.broker,
        backtest.data,
        rebalancer=Rebalancer(),
        benchmark=backtest.benchmark,
        universe=backtest.universe,
        fundamentals=backtest.fundamentals,
        scores=backtest.scores,
    )
    print()
    print(title)
    for line in extra_lines:
        print(line)
    print(report.summary())
    return report


def _print_sample_fundamental(
    fundamentals: FundamentalsDataSource, symbols: Sequence[str], as_of: datetime
) -> None:
    """Print one look-ahead-safe fundamentals read (revenue known as of the date)."""
    for symbol in symbols:
        revenue = fundamentals.value(symbol, "revenue", as_of)
        if revenue is not None:
            print(f"Fundamentals seam — {symbol} revenue as of {as_of.date()}: {revenue}")
            return
    print("Fundamentals seam — no revenue available for the basket at the as-of date.")
