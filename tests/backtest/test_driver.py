"""Tests for `run_backtest` against a `SimulatedBroker` on synthetic bars:
equity, buy-and-hold "trade once", equal-weight rebalancing, determinism, and
the universe/fundamentals threading.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.agent import Agent, DecisionContext
from value_portfolio.backtest import BacktestReport, BenchmarkSeries, run_backtest
from value_portfolio.baselines import BuyAndHold, EqualWeight
from value_portfolio.broker import OrderStatus, SimulatedBroker
from value_portfolio.data import InMemoryMarketData, InMemoryUniverse
from value_portfolio.data.types import Bar
from value_portfolio.rebalancer import Rebalancer

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _make_bars(symbol: str, prices: list[float]) -> list[Bar]:
    """Build a sequence of daily bars with open == close == given price."""
    bars: list[Bar] = []
    for i, price in enumerate(prices):
        c = Decimal(str(price))
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=_START + timedelta(days=i),
                open=c,
                high=c,
                low=c,
                close=c,
                volume=Decimal("1000"),
                timeframe="1Day",
            )
        )
    return bars


def _make_sim(bars: dict[str, list[Bar]], starting_cash: Decimal) -> SimulatedBroker:
    return SimulatedBroker(
        market_data=InMemoryMarketData(bars),
        starting_cash=starting_cash,
        commission_per_share=Decimal("0"),
        slippage_bps=Decimal("0"),
    )


class TestRunBacktest:
    def test_buy_and_hold_exact_equity(self) -> None:
        # One symbol: bought at 100 on the first fill, held while it
        # rises to 110. 10 shares -> final equity 1100, +10%.
        broker = _make_sim({"AAPL": _make_bars("AAPL", [100, 100, 110])}, Decimal("1000"))
        report = run_backtest(
            BuyAndHold(["AAPL"]),
            broker,
            broker.market_data,
            rebalancer=Rebalancer(cash_buffer=Decimal("0")),
        )

        assert report.starting_equity == Decimal("1000")
        assert report.final_equity == Decimal("1100")
        assert report.total_return == Decimal("0.1")
        assert report.n_fills == 1

    def test_buy_and_hold_trades_once(self) -> None:
        bars = {
            "AAPL": _make_bars("AAPL", [100, 100, 100, 100, 100]),
            "MSFT": _make_bars("MSFT", [200, 200, 200, 200, 200]),
            "GOOG": _make_bars("GOOG", [50, 50, 50, 50, 50]),
        }
        broker = _make_sim(bars, Decimal("3000"))
        report = run_backtest(
            BuyAndHold(["AAPL", "MSFT", "GOOG"]),
            broker,
            broker.market_data,
            rebalancer=Rebalancer(cash_buffer=Decimal("0")),
        )

        # One BUY per symbol on the first fill, then never again.
        assert report.n_fills == 3
        assert len(broker.list_orders()) == 3
        assert all(o.status is OrderStatus.FILLED for o in broker.list_orders())

    def test_equal_weight_rebalances_on_drift(self) -> None:
        bars = {
            "AAPL": _make_bars("AAPL", [100, 100, 100, 100]),
            "MSFT": _make_bars("MSFT", [100, 100, 120, 120]),
        }
        broker = _make_sim(bars, Decimal("2000"))
        report = run_backtest(
            EqualWeight(["AAPL", "MSFT"]),
            broker,
            broker.market_data,
            rebalancer=Rebalancer(cash_buffer=Decimal("0.2")),
        )

        # More fills than the initial two BUYs => it re-traded as MSFT drifted.
        assert report.n_fills > 2
        assert broker.get_position("AAPL") is not None
        assert broker.get_position("MSFT") is not None

    def test_deterministic_equity_curve(self) -> None:
        def run() -> BacktestReport:
            broker = _make_sim({"AAPL": _make_bars("AAPL", [100, 105, 110])}, Decimal("1000"))
            return run_backtest(
                BuyAndHold(["AAPL"]),
                broker,
                broker.market_data,
                rebalancer=Rebalancer(cash_buffer=Decimal("0.05")),
            )

        # Same bars + same strategy => identical equity curve. Order ids
        # are random uuids, so only the snapshots are compared.
        assert run().snapshots == run().snapshots

    def test_empty_timeline_rejected(self) -> None:
        broker = _make_sim({"AAPL": _make_bars("AAPL", [100, 100])}, Decimal("1000"))
        with pytest.raises(ValueError, match="non-empty timeline"):
            run_backtest(BuyAndHold(["AAPL"]), broker, broker.market_data, timeline=[])


class TestBenchmarkThreading:
    def test_no_benchmark_leaves_report_unchanged(self) -> None:
        # Same run with and without a benchmark -> identical snapshots and a
        # null benchmark series in the no-benchmark case (proof of additivity).
        broker = _make_sim({"AAPL": _make_bars("AAPL", [100, 100, 110])}, Decimal("1000"))
        report = run_backtest(
            BuyAndHold(["AAPL"]),
            broker,
            broker.market_data,
            rebalancer=Rebalancer(cash_buffer=Decimal("0")),
        )

        assert report.benchmark_levels is None
        assert report.beta is None

    def test_benchmark_levels_aligned_to_timeline(self) -> None:
        # The benchmark level recorded at each step is level_at(step timestamp),
        # captured in lockstep with the snapshots (same length, same order).
        bars = _make_bars("AAPL", [100, 100, 100])
        timeline = [b.timestamp for b in bars]
        broker = _make_sim({"AAPL": bars}, Decimal("1000"))
        benchmark = BenchmarkSeries.from_levels(
            {ts: Decimal(str(level)) for ts, level in zip(timeline, [50, 51, 52], strict=True)}
        )

        report = run_backtest(
            BuyAndHold(["AAPL"]),
            broker,
            broker.market_data,
            rebalancer=Rebalancer(cash_buffer=Decimal("0")),
            benchmark=benchmark,
        )

        assert report.benchmark_levels == (Decimal("50"), Decimal("51"), Decimal("52"))
        # Each recorded level matches the snapshot timestamp it is paired with.
        for snap, level in zip(report.snapshots, report.benchmark_levels, strict=True):
            assert benchmark.level_at(snap.timestamp) == level

    def test_sparse_benchmark_carries_last_level_forward(self) -> None:
        # A benchmark defined only on a subset of timeline dates pairs by clock:
        # level_at carries the last known level forward, never peeking ahead.
        bars = _make_bars("AAPL", [100, 100, 100, 100])
        timeline = [b.timestamp for b in bars]
        broker = _make_sim({"AAPL": bars}, Decimal("1000"))
        # Levels only at steps 0 and 2.
        benchmark = BenchmarkSeries.from_levels(
            {timeline[0]: Decimal("100"), timeline[2]: Decimal("120")}
        )

        report = run_backtest(
            BuyAndHold(["AAPL"]),
            broker,
            broker.market_data,
            rebalancer=Rebalancer(cash_buffer=Decimal("0")),
            benchmark=benchmark,
        )

        assert report.benchmark_levels == (
            Decimal("100"),
            Decimal("100"),
            Decimal("120"),
            Decimal("120"),
        )


class TestBacktestReport:
    def test_rejects_empty_snapshots(self) -> None:
        with pytest.raises(ValueError, match="at least one account snapshot"):
            BacktestReport(snapshots=(), fills=())


class _UniverseEqualWeight(Agent):
    """Test agent: equal-weights the members of the point-in-time universe.

    Records the members it observed at each step so the test can assert
    the universe was consumed at the correct clock.
    """

    def __init__(self) -> None:
        self.seen: list[tuple[datetime, tuple[str, ...]]] = []

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        assert context.universe is not None
        members = tuple(sorted(context.universe.members_at(context.now)))
        self.seen.append((context.now, members))
        if not members:
            return None
        weight = Decimal(1) / Decimal(len(members))
        return {symbol: weight for symbol in members}


class _UniverseProbe(Agent):
    """Test agent that records whatever `context.universe` it is handed."""

    def __init__(self) -> None:
        self.universes: list[object] = []

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        self.universes.append(context.universe)
        return None


class TestUniverseThreading:
    def test_context_universe_is_none_when_not_supplied(self) -> None:
        broker = _make_sim({"AAPL": _make_bars("AAPL", [100, 100, 100])}, Decimal("1000"))
        probe = _UniverseProbe()

        run_backtest(probe, broker, broker.market_data)

        assert probe.universes == [None, None, None]

    def test_universe_aware_agent_follows_membership_changes(self) -> None:
        # AAA is a member throughout; BBB only joins at step 2 (day 2).
        bars = {
            "AAA": _make_bars("AAA", [100, 100, 100, 100, 100]),
            "BBB": _make_bars("BBB", [50, 50, 50, 50, 50]),
        }
        broker = _make_sim(bars, Decimal("10000"))
        universe = InMemoryUniverse(
            {
                "AAA": [(_START, None)],
                "BBB": [(_START + timedelta(days=2), None)],
            }
        )
        agent = _UniverseEqualWeight()

        report = run_backtest(
            agent,
            broker,
            broker.market_data,
            rebalancer=Rebalancer(cash_buffer=Decimal("0.02")),
            universe=universe,
        )

        # Membership observed at the clock: AAA only until day 2, then both.
        observed = [members for _, members in agent.seen]
        assert observed[0] == ("AAA",)
        assert observed[1] == ("AAA",)
        assert observed[2] == ("AAA", "BBB")
        assert observed[-1] == ("AAA", "BBB")

        # BBB, a member only from day 2, ends up held — the agent acted on it.
        final_symbols = {p.symbol for p in report.snapshots[-1].positions}
        assert "BBB" in final_symbols

    def test_existing_agent_ignores_universe_and_behaves_identically(self) -> None:
        # An agent that does not read context.universe produces the same run
        # whether or not a universe is threaded through the driver.
        def run(with_universe: bool) -> BacktestReport:
            bars = {"AAPL": _make_bars("AAPL", [100, 100, 110])}
            broker = _make_sim(bars, Decimal("1000"))
            universe = InMemoryUniverse({"AAPL": [(_START, None)]}) if with_universe else None
            return run_backtest(
                BuyAndHold(["AAPL"]),
                broker,
                broker.market_data,
                rebalancer=Rebalancer(cash_buffer=Decimal("0")),
                universe=universe,
            )

        assert run(with_universe=True).snapshots == run(with_universe=False).snapshots


class _FundamentalsProbe(Agent):
    """Test agent that records whatever `context.fundamentals` it is handed."""

    def __init__(self) -> None:
        self.sources: list[object] = []

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        self.sources.append(context.fundamentals)
        return None


class TestFundamentalsThreading:
    def test_context_fundamentals_is_none_when_not_supplied(self) -> None:
        broker = _make_sim({"AAPL": _make_bars("AAPL", [100, 100, 100])}, Decimal("1000"))
        probe = _FundamentalsProbe()

        run_backtest(probe, broker, broker.market_data)

        assert probe.sources == [None, None, None]

    def test_fundamentals_source_is_placed_on_context(self) -> None:
        from value_portfolio.data import FundamentalRecord, InMemoryFundamentals

        broker = _make_sim({"AAPL": _make_bars("AAPL", [100, 100, 100])}, Decimal("1000"))
        probe = _FundamentalsProbe()
        fundamentals = InMemoryFundamentals(
            [
                FundamentalRecord(
                    symbol="AAPL",
                    dimension="ART",
                    datekey=_START,
                    values={"revenue": Decimal("1")},
                )
            ]
        )

        run_backtest(probe, broker, broker.market_data, fundamentals=fundamentals)

        # The same source instance is handed to the agent on every step.
        assert probe.sources == [fundamentals, fundamentals, fundamentals]


class _ScoresProbe(Agent):
    """Test agent that records whatever `context.scores` it is handed."""

    def __init__(self) -> None:
        self.sources: list[object] = []

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        self.sources.append(context.scores)
        return None


class TestScoresThreading:
    def test_context_scores_is_none_when_not_supplied(self) -> None:
        broker = _make_sim({"AAPL": _make_bars("AAPL", [100, 100, 100])}, Decimal("1000"))
        probe = _ScoresProbe()

        run_backtest(probe, broker, broker.market_data)

        assert probe.sources == [None, None, None]

    def test_score_source_is_placed_on_context(self) -> None:
        from value_portfolio.data import InMemoryScores, ScoreRecord

        broker = _make_sim({"AAPL": _make_bars("AAPL", [100, 100, 100])}, Decimal("1000"))
        probe = _ScoresProbe()
        scores = InMemoryScores([ScoreRecord(symbol="AAPL", date=_START, score=Decimal("0.5"))])

        run_backtest(probe, broker, broker.market_data, scores=scores)

        # The same source instance is handed to the agent on every step.
        assert probe.sources == [scores, scores, scores]


class TestDelistingIntegration:
    def test_delisted_holding_is_liquidated_mid_backtest(self) -> None:
        # ALIVE trades through day 4; DEAD's series ends at day 2.
        bars = {
            "ALIVE": _make_bars("ALIVE", [100, 100, 100, 100, 100]),
            "DEAD": _make_bars("DEAD", [10, 10, 12]),
        }
        broker = _make_sim(bars, Decimal("1000"))

        report = run_backtest(
            BuyAndHold(["ALIVE", "DEAD"]),
            broker,
            broker.market_data,
            rebalancer=Rebalancer(cash_buffer=Decimal("0")),
        )

        # DEAD is gone by the end — not carried at a stale mark.
        final = report.snapshots[-1]
        assert "DEAD" not in {p.symbol for p in final.positions}
        assert "ALIVE" in {p.symbol for p in final.positions}

        # Equity = cash (incl. DEAD liquidated at last close 12: 50*12=600)
        # + ALIVE 5 shares @ 100 = 500 -> 1100.
        assert final.equity == Decimal("1100")
        alive_value = next(p.market_value for p in final.positions if p.symbol == "ALIVE")
        assert final.cash + alive_value == final.equity
