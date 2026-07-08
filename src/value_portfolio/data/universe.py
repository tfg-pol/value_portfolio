from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime


class Universe(ABC):
    @abstractmethod
    def members_at(self, date: datetime) -> set[str]:
        """Symbols that were index members as of `date` (no future leakage)."""


MembershipIntervals = Mapping[str, Sequence[tuple[datetime, datetime | None]]]


class InMemoryUniverse(Universe):

    def __init__(self, membership: MembershipIntervals) -> None:
        if not membership:
            raise ValueError("InMemoryUniverse requires at least one symbol")
        self._membership: dict[str, tuple[tuple[datetime, datetime | None], ...]] = {}
        for symbol, intervals in membership.items():
            if not intervals:
                raise ValueError(f"symbol {symbol!r} has no membership intervals")
            for start, end in intervals:
                if end is not None and end < start:
                    raise ValueError(
                        f"symbol {symbol!r} has an interval ending before it starts: "
                        f"({start}, {end})"
                    )
            self._membership[symbol] = tuple(intervals)

    def members_at(self, date: datetime) -> set[str]:
        return {
            symbol
            for symbol, intervals in self._membership.items()
            if any(start <= date and (end is None or date <= end) for start, end in intervals)
        }
