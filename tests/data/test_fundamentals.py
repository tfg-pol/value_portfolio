"""Unit tests for `InMemoryFundamentals`: point-in-time reads anchored on
``datekey`` (most-recent filing known at the query date, no future leakage),
per-dimension independence, sparse fields, and last-wins de-duplication.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from value_portfolio.data import FundamentalRecord, InMemoryFundamentals


def _d(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _rec(
    symbol: str, dimension: str, datekey: datetime, values: dict[str, Decimal]
) -> FundamentalRecord:
    return FundamentalRecord(symbol=symbol, dimension=dimension, datekey=datekey, values=values)


class TestPointInTime:
    def test_returns_most_recent_known_filing(self) -> None:
        src = InMemoryFundamentals(
            [
                _rec("AAA", "ART", _d(2020, 2, 1), {"revenue": Decimal("100")}),
                _rec("AAA", "ART", _d(2020, 5, 1), {"revenue": Decimal("110")}),
                _rec("AAA", "ART", _d(2020, 8, 1), {"revenue": Decimal("120")}),
            ]
        )
        # Between the second and third filings, the second is the latest known.
        assert src.value("AAA", "revenue", _d(2020, 6, 15)) == Decimal("110")

    def test_query_on_exact_datekey_includes_that_filing(self) -> None:
        src = InMemoryFundamentals([_rec("AAA", "ART", _d(2020, 5, 1), {"eps": Decimal("2")})])
        assert src.value("AAA", "eps", _d(2020, 5, 1)) == Decimal("2")

    def test_does_not_leak_future_filings(self) -> None:
        src = InMemoryFundamentals(
            [_rec("AAA", "ART", _d(2020, 5, 1), {"revenue": Decimal("110")})]
        )
        # A query before the filing's availability date sees nothing.
        assert src.value("AAA", "revenue", _d(2020, 4, 30)) is None

    def test_unknown_symbol_or_dimension_is_none(self) -> None:
        src = InMemoryFundamentals([_rec("AAA", "ART", _d(2020, 5, 1), {"revenue": Decimal("1")})])
        assert src.value("ZZZ", "revenue", _d(2021, 1, 1)) is None
        assert src.value("AAA", "revenue", _d(2021, 1, 1), dimension="ARQ") is None


class TestDimensions:
    def test_dimensions_are_independent(self) -> None:
        src = InMemoryFundamentals(
            [
                _rec("AAA", "ARQ", _d(2020, 5, 1), {"revenue": Decimal("30")}),
                _rec("AAA", "ART", _d(2020, 5, 1), {"revenue": Decimal("120")}),
            ]
        )
        assert src.value("AAA", "revenue", _d(2020, 6, 1), dimension="ARQ") == Decimal("30")
        assert src.value("AAA", "revenue", _d(2020, 6, 1), dimension="ART") == Decimal("120")


class TestSparseFields:
    def test_missing_field_returns_none(self) -> None:
        # The filing exists but does not carry the requested field.
        src = InMemoryFundamentals(
            [_rec("AAA", "ART", _d(2020, 5, 1), {"revenue": Decimal("100")})]
        )
        assert src.value("AAA", "inventory", _d(2020, 6, 1)) is None

    def test_fields_and_symbols_reflect_contents(self) -> None:
        src = InMemoryFundamentals(
            [
                _rec("AAA", "ART", _d(2020, 5, 1), {"revenue": Decimal("1"), "eps": Decimal("2")}),
                _rec("BBB", "ART", _d(2020, 5, 1), {"netinc": Decimal("3")}),
            ]
        )
        assert src.symbols() == frozenset({"AAA", "BBB"})
        assert src.fields() == frozenset({"revenue", "eps", "netinc"})


class TestDeduplication:
    def test_duplicate_datekey_last_wins(self) -> None:
        # Two records share (symbol, dimension, datekey); the later one given wins.
        src = InMemoryFundamentals(
            [
                _rec("AAA", "ART", _d(2020, 5, 1), {"revenue": Decimal("100")}),
                _rec("AAA", "ART", _d(2020, 5, 1), {"revenue": Decimal("105")}),
            ]
        )
        assert src.value("AAA", "revenue", _d(2020, 7, 1)) == Decimal("105")
