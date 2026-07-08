"""Tests for the score-proportional stage-2 allocator (`ScoreProportionalTopK`):
weights proportional to the clipped mispricing magnitude, non-positive scores
dropped, the equal-weight fallback when nothing is positive, the ≤ 1 weight-sum
constraint, rebalance cadence, point-in-time score visibility, and determinism.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from value_portfolio.agent import DecisionContext
from value_portfolio.broker.types import AccountSnapshot
from value_portfolio.data import InMemoryMarketData, InMemoryScores, InMemoryUniverse, ScoreRecord
from value_portfolio.data.types import Bar
from value_portfolio.learning import ScoreProportionalTopK

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


class TestScoreProportionalTopK:
    def test_weights_proportional_to_score(self) -> None:
        # Top-2 of these are BBB (0.6) and CCC (0.3); weights split 2:1.
        scores = _scores({"AAA": "0.1", "BBB": "0.6", "CCC": "0.3"})
        agent = ScoreProportionalTopK(symbols=["AAA", "BBB", "CCC"], top_k=2, rebalance_every=1)

        target = agent.decide(_context(scores))

        assert target is not None
        assert set(target) == {"BBB", "CCC"}
        assert target["BBB"] == Decimal("0.666666")  # 0.6 / 0.9, ROUND_DOWN
        assert target["CCC"] == Decimal("0.333333")  # 0.3 / 0.9, ROUND_DOWN
        assert target["BBB"] > target["CCC"]
        assert sum(target.values()) <= Decimal("1")

    def test_non_positive_scores_get_zero_weight(self) -> None:
        # Selected set is the top-2 (BBB, AAA); AAA's score is negative, so it is
        # dropped and BBB takes the whole book.
        scores = _scores({"AAA": "-0.2", "BBB": "0.5", "CCC": "-0.4"})
        agent = ScoreProportionalTopK(symbols=["AAA", "BBB", "CCC"], top_k=2, rebalance_every=1)

        target = agent.decide(_context(scores))

        assert target == {"BBB": Decimal("1")}

    def test_all_non_positive_falls_back_to_equal_weight(self) -> None:
        scores = _scores({"AAA": "-0.1", "BBB": "-0.2", "CCC": "0"})
        agent = ScoreProportionalTopK(symbols=["AAA", "BBB", "CCC"], top_k=2, rebalance_every=1)

        target = agent.decide(_context(scores))

        assert target is not None
        assert len(target) == 2
        assert all(w == Decimal("0.5") for w in target.values())
        assert sum(target.values()) <= Decimal("1")

    def test_weights_sum_at_most_one(self) -> None:
        # 1/3 each does not divide evenly; ROUND_DOWN keeps the sum below 1.
        scores = _scores({"AAA": "0.2", "BBB": "0.2", "CCC": "0.2"})
        agent = ScoreProportionalTopK(symbols=["AAA", "BBB", "CCC"], top_k=3, rebalance_every=1)

        target = agent.decide(_context(scores))

        assert target is not None
        assert len(target) == 3
        assert sum(target.values()) <= Decimal("1")

    def test_returns_none_off_cadence(self) -> None:
        scores = _scores({"AAA": "0.1"})
        agent = ScoreProportionalTopK(symbols=["AAA"], top_k=1, rebalance_every=2)

        assert agent.decide(_context(scores)) is not None
        assert agent.decide(_context(scores)) is None
        assert agent.decide(_context(scores)) is not None

    def test_returns_none_without_scores_on_context(self) -> None:
        agent = ScoreProportionalTopK(symbols=["AAA"], top_k=1, rebalance_every=1)
        context = _context(_scores({"AAA": "0.1"}))
        bare = DecisionContext(now=context.now, account=context.account, data=context.data)
        assert agent.decide(bare) is None

    def test_returns_none_when_nothing_scored_yet(self) -> None:
        # Scores dated after `now` are invisible — the agent must stay in cash.
        late = _scores({"AAA": "0.1"}, date=_START + timedelta(days=30))
        agent = ScoreProportionalTopK(symbols=["AAA"], top_k=1, rebalance_every=1)
        assert agent.decide(_context(late)) is None

    def test_universe_filters_non_members_point_in_time(self) -> None:
        scores = _scores({"AAA": "0.9", "BBB": "0.1"})
        universe = InMemoryUniverse({"BBB": [(_START, None)]})
        agent = ScoreProportionalTopK(symbols=["AAA", "BBB"], top_k=5, rebalance_every=1)

        target = agent.decide(_context(scores, universe=universe))

        assert target == {"BBB": Decimal("1")}

    def test_deterministic(self) -> None:
        scores = _scores({"AAA": "0.1", "BBB": "0.6", "CCC": "0.3"})
        first = ScoreProportionalTopK(symbols=["AAA", "BBB", "CCC"], top_k=2, rebalance_every=1)
        second = ScoreProportionalTopK(symbols=["AAA", "BBB", "CCC"], top_k=2, rebalance_every=1)

        assert first.decide(_context(scores)) == second.decide(_context(scores))
