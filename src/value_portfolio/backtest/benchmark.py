from __future__ import annotations

import csv
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class BenchmarkSeries:
    timestamps: tuple[datetime, ...]
    levels: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if not self.timestamps:
            raise ValueError("a BenchmarkSeries requires at least one level")
        if len(self.timestamps) != len(self.levels):
            raise ValueError("timestamps and levels must have equal length")
        for earlier, later in zip(self.timestamps, self.timestamps[1:], strict=False):
            if later <= earlier:
                raise ValueError("benchmark timestamps must be strictly ascending")
        if any(level <= _ZERO for level in self.levels):
            raise ValueError("benchmark levels must be strictly positive")

    @classmethod
    def from_levels(cls, levels: Mapping[datetime, Decimal]) -> BenchmarkSeries:
        if not levels:
            raise ValueError("a BenchmarkSeries requires at least one level")
        ordered = sorted(levels.items())
        return cls(
            timestamps=tuple(ts for ts, _ in ordered),
            levels=tuple(level for _, level in ordered),
        )

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        *,
        date_column: str = "Date",
        level_column: str = "Close",
    ) -> BenchmarkSeries:
        levels: dict[datetime, Decimal] = {}
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if fields is None or date_column not in fields or level_column not in fields:
                raise ValueError(
                    f"{path} must have '{date_column}' and '{level_column}' columns; got {fields}"
                )
            for record in reader:
                raw_level = record[level_column]
                if raw_level is None or raw_level == "":
                    continue
                when = datetime.fromisoformat(record[date_column]).replace(tzinfo=UTC)
                levels[when] = Decimal(raw_level)
        if not levels:
            raise ValueError(f"no benchmark levels parsed from {path}")
        return cls.from_levels(levels)

    def level_at(self, when: datetime) -> Decimal | None:
        idx = bisect_right(self.timestamps, when) - 1
        if idx < 0:
            return None
        return self.levels[idx]
