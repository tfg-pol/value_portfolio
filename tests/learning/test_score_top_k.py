"""Tests for the rule-based stage-2 allocator (`ScoreTopK`) and the shared
selection helper: the ≤ top-k cardinality constraint, equal weights summing to
at most 1, point-in-time universe filtering, rebalance cadence, and
deterministic tie-breaking.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from value_portfolio.agent import DecisionContext
from value_portfolio.broker.types import AccountSnapshot
from value_portfolio.data import InMemoryMarketData, InMemoryScores, InMemoryUniverse, ScoreRecord
from value_portfolio.data.types import Bar
from value_portfolio.learning import ScoreTopK, select_top_scored

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _bars(symbol: str, prices: list[float]) -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            timestamp=_START + timedelta(days=i),
            open=Decimal(str(p)),
            high=Decimal(str(p)),
            low=Decimal(str(p)),
            close=Decimal(str(p)),
            volume=Decimal("1000"),
            timeframe="1Day",
        )
        for i, p in enumerate(prices)
    ]


def _scores(values: dict[str, str], date: datetime = _START) -> InMemoryScores:
    return InMemoryScores(
        [ScoreRecord(symbol=s, date=date, score=Decimal(v)) for s, v in values.items()]
    )


def _context(scores: InMemoryScores, universe: InMemoryUniverse | None = None) -> DecisionContext:
    data = InMemoryMarketData({"AAA": _bars("AAA", [100])})
    account = AccountSnapshot(
        account_id="sim",
        cash=Decimal("1000"),
        equity=Decimal("1000"),
        buying_power=Decimal("1000"),
        timestamp=_START,
    )
    return DecisionContext(now=_START, account=account, data=data, universe=universe, scores=scores)


class TestSelectTopScored:
    def test_caps_at_top_k_best_first(self) -> None:
        scores = _scores({"AAA": "0.1", "BBB": "0.3", "CCC": "0.2"})
        selected = select_top_scored(scores, _START, ["AAA", "BBB", "CCC"], top_k=2)
        assert selected == [("BBB", Decimal("0.3")), ("CCC", Decimal("0.2"))]

    def test_ties_break_by_symbol_ascending(self) -> None:
        scores = _scores({"BBB": "0.5", "AAA": "0.5", "CCC": "0.5"})
        selected = select_top_scored(scores, _START, ["CCC", "BBB", "AAA"], top_k=2)
        assert [s for s, _ in selected] == ["AAA", "BBB"]

    def test_unscored_candidates_are_dropped(self) -> None:
        scores = _scores({"AAA": "0.1"})
        selected = select_top_scored(scores, _START, ["AAA", "ZZZ"], top_k=5)
        assert [s for s, _ in selected] == ["AAA"]

    def test_universe_filters_non_members_point_in_time(self) -> None:
        scores = _scores({"AAA": "0.9", "BBB": "0.1"})
        universe = InMemoryUniverse({"BBB": [(_START, None)]})
        selected = select_top_scored(scores, _START, ["AAA", "BBB"], top_k=5, universe=universe)
        assert [s for s, _ in selected] == ["BBB"]


class TestScoreTopK:
    def test_holds_at_most_top_k_equal_weighted(self) -> None:
        scores = _scores({"AAA": "0.1", "BBB": "0.4", "CCC": "0.3", "DDD": "0.2"})
        agent = ScoreTopK(symbols=["AAA", "BBB", "CCC", "DDD"], top_k=2, rebalance_every=1)

        target = agent.decide(_context(scores))

        assert target is not None
        assert set(target) == {"BBB", "CCC"}
        assert all(w == Decimal("0.5") for w in target.values())
        assert sum(target.values()) <= Decimal("1")

    def test_returns_none_off_cadence(self) -> None:
        scores = _scores({"AAA": "0.1"})
        agent = ScoreTopK(symbols=["AAA"], top_k=1, rebalance_every=2)

        assert agent.decide(_context(scores)) is not None
        assert agent.decide(_context(scores)) is None
        assert agent.decide(_context(scores)) is not None

    def test_returns_none_without_scores_on_context(self) -> None:
        agent = ScoreTopK(symbols=["AAA"], top_k=1, rebalance_every=1)
        context = _context(_scores({"AAA": "0.1"}))
        bare = DecisionContext(now=context.now, account=context.account, data=context.data)
        assert agent.decide(bare) is None

    def test_returns_none_when_nothing_scored_yet(self) -> None:
        # Scores dated after `now` are invisible — the agent must stay in cash.
        late = _scores({"AAA": "0.1"}, date=_START + timedelta(days=30))
        agent = ScoreTopK(symbols=["AAA"], top_k=1, rebalance_every=1)
        assert agent.decide(_context(late)) is None

    def test_defaults_to_all_scored_symbols(self) -> None:
        scores = _scores({"AAA": "0.2", "BBB": "0.1"})
        agent = ScoreTopK(top_k=1, rebalance_every=1)

        target = agent.decide(_context(scores))

        assert target == {"AAA": Decimal("1")}
