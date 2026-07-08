
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from value_portfolio.backtest.report import BacktestReport

_ZERO = Decimal("0")
_CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class EquitySeries:
    timestamps: tuple[datetime, ...]
    equity: tuple[Decimal, ...]
    benchmark: tuple[Decimal | None, ...] | None = None

    def __post_init__(self) -> None:
        if not self.timestamps:
            raise ValueError("an EquitySeries requires at least one step")
        if len(self.equity) != len(self.timestamps):
            raise ValueError(
                "equity must align one-to-one with timestamps: "
                f"got {len(self.equity)} values for {len(self.timestamps)} timestamps"
            )
        if self.benchmark is not None and len(self.benchmark) != len(self.timestamps):
            raise ValueError(
                "benchmark must align one-to-one with timestamps: "
                f"got {len(self.benchmark)} levels for {len(self.timestamps)} timestamps"
            )
        if any(
            later <= earlier
            for earlier, later in zip(self.timestamps, self.timestamps[1:], strict=False)
        ):
            raise ValueError("timestamps must be strictly ascending")


def series_from_report(report: BacktestReport) -> EquitySeries:
    return EquitySeries(
        timestamps=tuple(s.timestamp for s in report.snapshots),
        equity=report.equity_curve,
        benchmark=report.benchmark_levels,
    )


def series_to_dict(series: EquitySeries) -> dict[str, object]:

    def render(value: Decimal | None) -> str | None:
        return str(value.quantize(_CENT)) if value is not None else None

    return {
        "timestamps": [stamp.isoformat() for stamp in series.timestamps],
        "equity": [render(value) for value in series.equity],
        "benchmark": (
            [render(level) for level in series.benchmark] if series.benchmark is not None else None
        ),
    }


def series_from_dict(data: Mapping[str, object]) -> EquitySeries:
    try:
        timestamps = data["timestamps"]
        equity = data["equity"]
        benchmark = data["benchmark"]
    except KeyError as missing:
        raise ValueError(f"series dict is missing key {missing}") from None
    if not isinstance(timestamps, list) or not isinstance(equity, list):
        raise ValueError("series timestamps and equity must be lists")
    if benchmark is not None and not isinstance(benchmark, list):
        raise ValueError("series benchmark must be a list or null")
    return EquitySeries(
        timestamps=tuple(datetime.fromisoformat(str(stamp)) for stamp in timestamps),
        equity=tuple(Decimal(str(value)) for value in equity),
        benchmark=(
            tuple(Decimal(str(level)) if level is not None else None for level in benchmark)
            if benchmark is not None
            else None
        ),
    )


def normalized_levels(values: Sequence[Decimal | None]) -> tuple[Decimal | None, ...]:
    base = next((value for value in values if value is not None), None)
    if base is None:
        return tuple(values)
    if base == _ZERO:
        raise ValueError("cannot normalize a series whose first known value is zero")
    return tuple(value / base if value is not None else None for value in values)


def drawdown_levels(values: Sequence[Decimal]) -> tuple[Decimal, ...]:
    drawdowns: list[Decimal] = []
    peak: Decimal | None = None
    for value in values:
        if peak is None or value > peak:
            peak = value
        drawdowns.append((value - peak) / peak if peak > _ZERO else _ZERO)
    return tuple(drawdowns)
