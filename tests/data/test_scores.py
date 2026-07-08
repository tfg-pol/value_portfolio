"""Unit tests for `InMemoryScores`: point-in-time reads anchored on the
computation date (most recent score known at the query date, no future
leakage) and last-wins de-duplication.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from value_portfolio.data import InMemoryScores, ScoreRecord


def _d(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _rec(symbol: str, date: datetime, score: str) -> ScoreRecord:
    return ScoreRecord(symbol=symbol, date=date, score=Decimal(score))


class TestPointInTime:
    def test_returns_most_recent_known_score(self) -> None:
        src = InMemoryScores(
            [
                _rec("AAA", _d(2020, 1, 31), "0.10"),
                _rec("AAA", _d(2020, 2, 28), "0.20"),
                _rec("AAA", _d(2020, 3, 31), "0.30"),
            ]
        )
        assert src.score("AAA", _d(2020, 3, 15)) == Decimal("0.20")

    def test_query_on_exact_date_includes_that_score(self) -> None:
        src = InMemoryScores([_rec("AAA", _d(2020, 1, 31), "0.10")])
        assert src.score("AAA", _d(2020, 1, 31)) == Decimal("0.10")

    def test_does_not_leak_future_scores(self) -> None:
        src = InMemoryScores([_rec("AAA", _d(2020, 1, 31), "0.10")])
        assert src.score("AAA", _d(2020, 1, 31) - timedelta(days=1)) is None

    def test_unknown_symbol_is_none(self) -> None:
        src = InMemoryScores([_rec("AAA", _d(2020, 1, 31), "0.10")])
        assert src.score("ZZZ", _d(2021, 1, 1)) is None


class TestDeduplication:
    def test_duplicate_symbol_date_last_wins(self) -> None:
        src = InMemoryScores(
            [
                _rec("AAA", _d(2020, 1, 31), "0.10"),
                _rec("AAA", _d(2020, 1, 31), "0.99"),
            ]
        )
        assert src.score("AAA", _d(2020, 2, 1)) == Decimal("0.99")


class TestSymbols:
    def test_symbols_carries_every_scored_name(self) -> None:
        src = InMemoryScores(
            [
                _rec("AAA", _d(2020, 1, 31), "0.1"),
                _rec("BBB", _d(2020, 2, 28), "0.2"),
            ]
        )
        assert src.symbols() == frozenset({"AAA", "BBB"})
