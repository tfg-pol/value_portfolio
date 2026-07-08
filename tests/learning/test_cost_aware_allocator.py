"""Tests for `CostAwareAllocator.decide`: construction guards, cadence/warm-up/
no-scores, the long-only weight-sum constraint, top-K selection, the score tilt
(reward term), and the turnover spring (a large turnover penalty keeps the book
on the current holdings; a zero penalty follows the scores). Synthetic data with
clear structure; determinism checked by repeated runs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.agent import DecisionContext
from value_portfolio.broker.types import AccountSnapshot, Position
from value_portfolio.data import InMemoryMarketData, InMemoryScores, ScoreRecord
from value_portfolio.data.types import Bar
from value_portfolio.learning import CostAwareAllocator

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _closes_from_returns(start: float, returns: list[float]) -> list[float]:
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1.0 + r))
    return closes


def _bars(symbol: str, closes: list[float]) -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            timestamp=_START + timedelta(days=i),
            open=Decimal(str(c)),
            high=Decimal(str(c)),
            low=Decimal(str(c)),
            close=Decimal(str(c)),
            volume=Decimal("1000"),
            timeframe="1Day",
        )
        for i, c in enumerate(closes)
    ]


# Three symbols with distinct, low-correlation return paths and enough history.
_BARS = {
    "AAA": _bars("AAA", _closes_from_returns(100, [0.01, -0.01, 0.02, -0.01, 0.01, 0.0])),
    "BBB": _bars("BBB", _closes_from_returns(100, [-0.01, 0.01, -0.01, 0.02, -0.01, 0.01])),
    "CCC": _bars("CCC", _closes_from_returns(100, [0.0, 0.01, 0.01, -0.02, 0.01, -0.01])),
}
_SYMBOLS = ["AAA", "BBB", "CCC"]


def _scores(values: dict[str, str]) -> InMemoryScores:
    return InMemoryScores(
        [ScoreRecord(symbol=s, date=_START, score=Decimal(v)) for s, v in values.items()]
    )


def _context(
    scores: InMemoryScores,
    *,
    weights_now: dict[str, float] | None = None,
    clock: datetime | None = None,
) -> DecisionContext:
    """Context as of the last bar (or `clock`), with an account whose positions
    encode `weights_now` (current weights, against an equity of 100)."""
    data = InMemoryMarketData(_BARS)
    now = clock if clock is not None else _START + timedelta(days=5)
    data.advance_to(now)

    equity = Decimal("100")
    positions: tuple[Position, ...] = ()
    market_value_total = Decimal("0")
    if weights_now:
        positions = tuple(
            Position(
                symbol=symbol,
                qty=Decimal("1"),
                avg_entry_price=Decimal("1"),
                market_value=Decimal(str(w)) * equity,
                unrealized_pl=Decimal("0"),
                current_price=Decimal("1"),
            )
            for symbol, w in weights_now.items()
        )
        market_value_total = sum((p.market_value for p in positions), start=Decimal("0"))
    account = AccountSnapshot(
        account_id="t",
        cash=equity - market_value_total,
        equity=equity,
        buying_power=equity,
        timestamp=now,
        positions=positions,
    )
    return DecisionContext(now=now, account=account, data=data, scores=scores)


def _agent(**kwargs: object) -> CostAwareAllocator:
    params: dict[str, object] = {"symbols": _SYMBOLS, "lookback": 4, "rebalance_every": 1}
    params.update(kwargs)
    return CostAwareAllocator(**params)  # type: ignore[arg-type]


class TestConstruction:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"symbols": ["AAA", "AAA"]},
            {"top_k": 0},
            {"lookback": 1},
            {"risk_aversion": -1.0},
            {"turnover_aversion": -1.0},
            {"max_weight": 0.0},
            {"max_weight": 1.5},
            {"ridge": -0.1},
        ],
    )
    def test_rejects_bad_params(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            CostAwareAllocator(**kwargs)  # type: ignore[arg-type]


class TestDecide:
    def test_returns_none_off_cadence(self) -> None:
        agent = _agent(rebalance_every=2)
        scores = _scores({"AAA": "0.6", "BBB": "0.3", "CCC": "0.1"})
        assert agent.decide(_context(scores)) is not None
        assert agent.decide(_context(scores)) is None
        assert agent.decide(_context(scores)) is not None

    def test_returns_none_without_scores(self) -> None:
        agent = _agent()
        ctx = _context(_scores({"AAA": "0.6", "BBB": "0.3", "CCC": "0.1"}))
        bare = DecisionContext(now=ctx.now, account=ctx.account, data=ctx.data)
        assert agent.decide(bare) is None

    def test_returns_none_during_warmup(self) -> None:
        # lookback=4 needs 5 bars; only 3 visible two days in.
        agent = _agent()
        scores = _scores({"AAA": "0.6", "BBB": "0.3", "CCC": "0.1"})
        ctx = _context(scores, clock=_START + timedelta(days=2))
        assert agent.decide(ctx) is None

    def test_weights_long_only_and_sum_le_one(self) -> None:
        agent = _agent()
        scores = _scores({"AAA": "0.6", "BBB": "0.3", "CCC": "0.1"})
        target = agent.decide(_context(scores))
        assert target is not None
        assert all(w >= Decimal("0") for w in target.values())
        assert sum(target.values()) <= Decimal("1")

    def test_selection_is_subset_of_top_k(self) -> None:
        agent = _agent(top_k=2)
        scores = _scores({"AAA": "0.6", "BBB": "0.3", "CCC": "0.1"})
        target = agent.decide(_context(scores))
        assert target is not None
        assert set(target) == {"AAA", "BBB"}  # CCC is below the top-2 cut

    def test_zero_turnover_follows_the_scores(self) -> None:
        # No spring: the high-score name outweighs the low-score one, regardless
        # of where the book currently sits.
        agent = _agent(turnover_aversion=0.0, risk_aversion=1.0)
        scores = _scores({"AAA": "0.6", "BBB": "0.3", "CCC": "0.1"})
        target = agent.decide(_context(scores, weights_now={"AAA": 0.2, "BBB": 0.3, "CCC": 0.5}))
        assert target is not None
        assert target["AAA"] >= target["BBB"] >= target["CCC"]
        assert target["AAA"] > target["CCC"]

    def test_large_turnover_aversion_stays_on_current_book(self) -> None:
        # A stiff spring keeps the weights on the current holdings even though the
        # scores favour AAA and the book is concentrated in CCC.
        agent = _agent(turnover_aversion=1000.0, risk_aversion=1.0)
        scores = _scores({"AAA": "0.6", "BBB": "0.3", "CCC": "0.1"})
        weights_now = {"AAA": 0.2, "BBB": 0.3, "CCC": 0.5}
        target = agent.decide(_context(scores, weights_now=weights_now))
        assert target is not None
        for symbol, w_now in weights_now.items():
            assert abs(float(target[symbol]) - w_now) < 0.03

    def test_deterministic(self) -> None:
        scores = _scores({"AAA": "0.6", "BBB": "0.3", "CCC": "0.1"})
        first = _agent().decide(_context(scores, weights_now={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}))
        second = _agent().decide(_context(scores, weights_now={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}))
        assert first == second
