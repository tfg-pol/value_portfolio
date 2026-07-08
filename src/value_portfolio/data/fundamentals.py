
from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

# ART = as-reported trailing-twelve-months; callers may ask for ARQ or ARY.
_DEFAULT_DIMENSION = "ART"


@dataclass(frozen=True, slots=True)
class FundamentalRecord:
    symbol: str
    dimension: str
    datekey: datetime
    values: Mapping[str, Decimal]


class FundamentalsDataSource(ABC):
    @abstractmethod
    def value(
        self, symbol: str, field: str, as_of: datetime, *, dimension: str = _DEFAULT_DIMENSION
    ) -> Decimal | None:
        """`field` from the most recent filing known by `as_of`, else ``None``."""

    @abstractmethod
    def fields(self) -> frozenset[str]: ...

    @abstractmethod
    def symbols(self) -> frozenset[str]: ...


class InMemoryFundamentals(FundamentalsDataSource):
    def __init__(self, records: Iterable[FundamentalRecord]) -> None:
        # Collapse duplicate (symbol, dimension, datekey) keys (last wins).
        latest: dict[tuple[str, str, datetime], Mapping[str, Decimal]] = {}
        for rec in records:
            latest[(rec.symbol, rec.dimension, rec.datekey)] = rec.values

        self._series: dict[tuple[str, str], tuple[list[datetime], list[Mapping[str, Decimal]]]] = {}
        fields: set[str] = set()
        symbols: set[str] = set()
        for (symbol, dimension, datekey), values in sorted(latest.items(), key=lambda kv: kv[0][2]):
            datekeys, value_maps = self._series.setdefault((symbol, dimension), ([], []))
            datekeys.append(datekey)
            value_maps.append(values)
            fields.update(values)
            symbols.add(symbol)

        self._fields = frozenset(fields)
        self._symbols = frozenset(symbols)

    def value(
        self, symbol: str, field: str, as_of: datetime, *, dimension: str = _DEFAULT_DIMENSION
    ) -> Decimal | None:
        series = self._series.get((symbol, dimension))
        if series is None:
            return None
        datekeys, value_maps = series
        idx = bisect_right(datekeys, as_of) - 1
        if idx < 0:
            return None
        return value_maps[idx].get(field)

    def fields(self) -> frozenset[str]:
        return self._fields

    def symbols(self) -> frozenset[str]:
        return self._symbols
