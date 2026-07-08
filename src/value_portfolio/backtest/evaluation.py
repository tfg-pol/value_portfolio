from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from value_portfolio.backtest._stats import mean, median, sample_std
from value_portfolio.backtest.report import BacktestReport
from value_portfolio.backtest.series import series_from_report, series_to_dict

_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class Window:

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"window end must be after start, got {self.start} -> {self.end}")

    @property
    def label(self) -> str:
        return f"{self.start.date()} -> {self.end.date()}"


def rolling_windows(
    start: datetime,
    end: datetime,
    *,
    window_years: int = 5,
    step_months: int = 12,
) -> list[Window]:
    if window_years < 1:
        raise ValueError(f"window_years must be >= 1, got {window_years}")
    if step_months < 1:
        raise ValueError(f"step_months must be >= 1, got {step_months}")

    windows: list[Window] = []
    cursor = start
    while True:
        window_end = _add_months(cursor, 12 * window_years)
        if window_end > end:
            break
        windows.append(Window(start=cursor, end=window_end))
        cursor = _add_months(cursor, step_months)
    return windows


def _add_months(stamp: datetime, months: int) -> datetime:
    month_index = stamp.month - 1 + months
    year = stamp.year + month_index // 12
    month = month_index % 12 + 1
    return stamp.replace(year=year, month=month)


@dataclass(frozen=True, slots=True)
class WindowResult:
    window: Window
    report: BacktestReport


_METRICS: dict[str, Callable[[BacktestReport], Decimal | None]] = {
    "total_return": lambda r: r.total_return,
    "cagr": lambda r: r.cagr,
    "volatility": lambda r: r.volatility(_TRADING_DAYS_PER_YEAR),
    "sharpe_ratio": lambda r: r.sharpe_ratio(periods_per_year=_TRADING_DAYS_PER_YEAR),
    "max_drawdown": lambda r: r.max_drawdown,
    "turnover": lambda r: r.turnover,
    "benchmark_return": lambda r: r.benchmark_return,
    "active_return": lambda r: r.active_return,
    "tracking_error": lambda r: r.tracking_error(_TRADING_DAYS_PER_YEAR),
    "information_ratio": lambda r: r.information_ratio(_TRADING_DAYS_PER_YEAR),
    "beta": lambda r: r.beta,
    "jensens_alpha": lambda r: r.jensens_alpha(_TRADING_DAYS_PER_YEAR),
}

_PERCENT_METRICS = frozenset(
    {
        "total_return",
        "cagr",
        "volatility",
        "max_drawdown",
        "benchmark_return",
        "active_return",
        "tracking_error",
        "jensens_alpha",
    }
)


@dataclass(frozen=True, slots=True)
class MultiWindowEvaluation:
    results: tuple[WindowResult, ...]

    def metric_values(self, metric: str) -> tuple[Decimal, ...]:
        extractor = _METRICS[metric]
        return tuple(
            value for result in self.results if (value := extractor(result.report)) is not None
        )

    def summary(self) -> str:
        lines = [
            f"Windows evaluated  : {len(self.results)}",
            f"{'Metric':<18} {'mean':>9} {'std':>9} {'min':>9} {'median':>9} {'max':>9}",
        ]
        for metric in _METRICS:
            values = self.metric_values(metric)
            if not values:
                continue
            scale = Decimal("100") if metric in _PERCENT_METRICS else Decimal("1")
            suffix = " %" if metric in _PERCENT_METRICS else ""
            cells = (
                mean(values) * scale,
                sample_std(values) * scale,
                min(values) * scale,
                median(values) * scale,
                max(values) * scale,
            )
            row = " ".join(f"{cell:>9.3f}" for cell in cells)
            lines.append(f"{metric:<18} {row}{suffix}")
        return "\n".join(lines)

    def per_window_lines(self, metric: str) -> list[str]:
        extractor = _METRICS[metric]
        lines = []
        for result in self.results:
            value = extractor(result.report)
            rendered = f"{value:.4f}" if value is not None else "n/a"
            lines.append(f"  {result.window.label}: {rendered}")
        return lines


def report_metrics(report: BacktestReport) -> dict[str, Decimal | None]:
    return {name: extractor(report) for name, extractor in _METRICS.items()}


def evaluation_to_dict(
    evaluation: MultiWindowEvaluation,
    *,
    include_series: bool = True,
) -> dict[str, object]:
    def render(value: Decimal | None) -> str | None:
        return str(value) if value is not None else None

    windows: list[dict[str, object]] = [
        {
            "start": result.window.start.date().isoformat(),
            "end": result.window.end.date().isoformat(),
            **{name: render(value) for name, value in report_metrics(result.report).items()},
            **(
                {"series": series_to_dict(series_from_report(result.report))}
                if include_series
                else {}
            ),
        }
        for result in evaluation.results
    ]
    summary: dict[str, dict[str, str]] = {}
    for metric in _METRICS:
        values = evaluation.metric_values(metric)
        if not values:
            continue
        summary[metric] = {
            "mean": str(mean(values)),
            "std": str(sample_std(values)),
            "min": str(min(values)),
            "median": str(median(values)),
            "max": str(max(values)),
        }
    return {"n_windows": len(evaluation.results), "windows": windows, "summary": summary}


def evaluate_windows(
    windows: Sequence[Window],
    run: Callable[[Window], BacktestReport | None],
) -> MultiWindowEvaluation:
    results: list[WindowResult] = []
    for window in windows:
        report = run(window)
        if report is not None:
            results.append(WindowResult(window=window, report=report))
    return MultiWindowEvaluation(results=tuple(results))
