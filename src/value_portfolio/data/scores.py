from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    symbol: str
    date: datetime
    score: Decimal


class ScoreSource(ABC):
    @abstractmethod
    def score(self, symbol: str, as_of: datetime) -> Decimal | None: ...

    @abstractmethod
    def symbols(self) -> frozenset[str]: ...


class InMemoryScores(ScoreSource):

    def __init__(self, records: Iterable[ScoreRecord]) -> None:
        latest: dict[tuple[str, datetime], Decimal] = {}
        for rec in records:
            latest[(rec.symbol, rec.date)] = rec.score

        self._series: dict[str, tuple[list[datetime], list[Decimal]]] = {}
        for (symbol, date), value in sorted(latest.items(), key=lambda kv: kv[0][1]):
            dates, values = self._series.setdefault(symbol, ([], []))
            dates.append(date)
            values.append(value)

        self._symbols = frozenset(self._series)

    def score(self, symbol: str, as_of: datetime) -> Decimal | None:
        series = self._series.get(symbol)
        if series is None:
            return None
        dates, values = series
        idx = bisect_right(dates, as_of) - 1
        if idx < 0:
            return None
        return values[idx]

    def symbols(self) -> frozenset[str]:
        return self._symbols


def load_scores_from_parquet(
    path: Path | str, symbols: Sequence[str] | None = None
) -> InMemoryScores:
    import pandas as pd  # offline loader boundary, like data.sharadar

    frame = pd.read_parquet(path, columns=["date", "ticker", "score"])
    if symbols is not None:
        frame = frame[frame["ticker"].isin(set(symbols))]

    records: list[ScoreRecord] = []
    for date, ticker, score in frame.itertuples(index=False):
        stamp = date.to_pydatetime()
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        records.append(ScoreRecord(symbol=str(ticker), date=stamp, score=Decimal(str(score))))
    return InMemoryScores(records)
