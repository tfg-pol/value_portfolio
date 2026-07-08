"""Tests for `BenchmarkSeries`: construction/validation, `from_levels` sorting,
and the look-ahead-safe `level_at` read.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.backtest import BenchmarkSeries

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _levels(values: list[float]) -> dict[datetime, Decimal]:
    return {_START + timedelta(days=i): Decimal(str(v)) for i, v in enumerate(values)}


class TestConstruction:
    def test_from_levels_sorts_by_timestamp(self) -> None:
        unordered = {
            _START + timedelta(days=2): Decimal("102"),
            _START: Decimal("100"),
            _START + timedelta(days=1): Decimal("101"),
        }
        series = BenchmarkSeries.from_levels(unordered)

        assert series.timestamps == (
            _START,
            _START + timedelta(days=1),
            _START + timedelta(days=2),
        )
        assert series.levels == (Decimal("100"), Decimal("101"), Decimal("102"))

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one level"):
            BenchmarkSeries.from_levels({})

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            BenchmarkSeries(timestamps=(_START,), levels=(Decimal("1"), Decimal("2")))

    def test_non_ascending_timestamps_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            BenchmarkSeries(timestamps=(_START, _START), levels=(Decimal("1"), Decimal("2")))

    def test_non_positive_levels_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            BenchmarkSeries.from_levels(_levels([100.0, 0.0]))


class TestLevelAt:
    def test_returns_most_recent_level_at_or_before(self) -> None:
        series = BenchmarkSeries.from_levels(_levels([100.0, 110.0, 120.0]))

        assert series.level_at(_START) == Decimal("100")
        assert series.level_at(_START + timedelta(days=1)) == Decimal("110")
        # Between known points -> carries the last known level forward.
        assert series.level_at(_START + timedelta(days=1, hours=12)) == Decimal("110")
        # After the last point -> still the last known level, never beyond.
        assert series.level_at(_START + timedelta(days=99)) == Decimal("120")

    def test_none_before_first_level(self) -> None:
        series = BenchmarkSeries.from_levels(_levels([100.0, 110.0]))

        assert series.level_at(_START - timedelta(days=1)) is None

    def test_does_not_peek_into_the_future(self) -> None:
        # Querying at day 0 must never surface day 1's level.
        series = BenchmarkSeries.from_levels(_levels([100.0, 110.0]))

        assert series.level_at(_START) == Decimal("100")
