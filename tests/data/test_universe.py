"""Tests for `InMemoryUniverse`: point-in-time membership, no future leakage,
multiple spells, and construction validation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from value_portfolio.data import InMemoryUniverse

_2010 = datetime(2010, 1, 1, tzinfo=UTC)
_2012 = datetime(2012, 1, 1, tzinfo=UTC)
_2015 = datetime(2015, 1, 1, tzinfo=UTC)
_2020 = datetime(2020, 1, 1, tzinfo=UTC)


class TestConstruction:
    def test_empty_membership_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one symbol"):
            InMemoryUniverse({})

    def test_symbol_without_intervals_rejected(self) -> None:
        with pytest.raises(ValueError, match="no membership intervals"):
            InMemoryUniverse({"AAPL": []})

    def test_interval_ending_before_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="ending before it starts"):
            InMemoryUniverse({"AAPL": [(_2015, _2010)]})


class TestMembersAt:
    def test_membership_across_changes(self) -> None:
        # OLD is a member 2010..2012 (then removed); NEW joins in 2015 and stays.
        universe = InMemoryUniverse(
            {
                "OLD": [(_2010, _2012)],
                "NEW": [(_2015, None)],
            }
        )

        assert universe.members_at(_2010) == {"OLD"}
        assert universe.members_at(_2012) == {"OLD"}  # inclusive end
        assert universe.members_at(datetime(2013, 1, 1, tzinfo=UTC)) == set()
        assert universe.members_at(_2015) == {"NEW"}
        assert universe.members_at(_2020) == {"NEW"}

    def test_does_not_leak_future_membership(self) -> None:
        # A name that joins only in 2015 must be invisible in a 2010 query.
        universe = InMemoryUniverse({"NEW": [(_2015, None)]})

        assert universe.members_at(_2010) == set()
        assert universe.members_at(_2015) == {"NEW"}

    def test_membership_before_start_is_empty(self) -> None:
        universe = InMemoryUniverse({"OLD": [(_2012, None)]})

        assert universe.members_at(_2010) == set()

    def test_multiple_disjoint_spells(self) -> None:
        # Removed in 2012, re-added in 2015 — a member in both spells, not between.
        universe = InMemoryUniverse({"REENTRY": [(_2010, _2012), (_2015, None)]})

        assert universe.members_at(_2010) == {"REENTRY"}
        assert universe.members_at(datetime(2013, 6, 1, tzinfo=UTC)) == set()
        assert universe.members_at(_2020) == {"REENTRY"}
