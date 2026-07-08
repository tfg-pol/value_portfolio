"""Tests for the plottable `EquitySeries`: report extraction, JSON round-trip,
curve math (normalization, drawdown) against hand-computed values, and
validation of malformed inputs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.backtest import (
    BacktestReport,
    EquitySeries,
    drawdown_levels,
    normalized_levels,
    series_from_dict,
    series_from_report,
    series_to_dict,
)
from value_portfolio.broker.types import AccountSnapshot

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _snap(equity: float, day_offset: int) -> AccountSnapshot:
    eq = Decimal(str(equity))
    return AccountSnapshot(
        account_id="test",
        cash=eq,
        equity=eq,
        buying_power=eq,
        timestamp=_START + timedelta(days=day_offset),
        positions=(),
    )


def _series(
    equities: list[float],
    benchmark: list[float | None] | None = None,
) -> EquitySeries:
    return EquitySeries(
        timestamps=tuple(_START + timedelta(days=i) for i in range(len(equities))),
        equity=tuple(Decimal(str(eq)) for eq in equities),
        benchmark=(
            tuple(None if v is None else Decimal(str(v)) for v in benchmark)
            if benchmark is not None
            else None
        ),
    )


class TestSeriesFromReport:
    def test_fields_match_the_report_one_to_one(self) -> None:
        snapshots = tuple(_snap(eq, i) for i, eq in enumerate([100, 110, 99]))
        bench = (None, Decimal("50"), Decimal("51"))
        report = BacktestReport(snapshots=snapshots, fills=(), benchmark_levels=bench)

        series = series_from_report(report)

        assert series.timestamps == tuple(s.timestamp for s in snapshots)
        assert series.equity == report.equity_curve
        assert series.benchmark == bench

    def test_no_benchmark_stays_none(self) -> None:
        report = BacktestReport(snapshots=(_snap(100, 0), _snap(101, 1)), fills=())

        assert series_from_report(report).benchmark is None


class TestJsonRoundTrip:
    def test_round_trips_with_benchmark_and_gaps(self) -> None:
        series = _series([100, 110.5, 99], benchmark=[None, 50.25, 51])

        restored = series_from_dict(json.loads(json.dumps(series_to_dict(series))))

        assert restored == series

    def test_round_trips_without_benchmark(self) -> None:
        series = _series([100, 110])

        restored = series_from_dict(json.loads(json.dumps(series_to_dict(series))))

        assert restored == series
        assert restored.benchmark is None

    def test_timestamps_keep_their_timezone(self) -> None:
        series = _series([100, 110])

        restored = series_from_dict(series_to_dict(series))

        assert restored.timestamps[0].tzinfo is not None
        assert restored.timestamps[0] == _START

    def test_levels_are_quantized_to_cents(self) -> None:
        series = _series([100.123456])

        rendered = series_to_dict(series)

        assert rendered["equity"] == ["100.12"]

    def test_missing_key_is_a_clear_error(self) -> None:
        with pytest.raises(ValueError, match="missing key"):
            series_from_dict({"timestamps": [], "equity": []})

    def test_mismatched_lengths_are_rejected(self) -> None:
        data = series_to_dict(_series([100, 110]))
        data["equity"] = ["100.00"]

        with pytest.raises(ValueError, match="one-to-one"):
            series_from_dict(data)


class TestValidation:
    def test_empty_series_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            EquitySeries(timestamps=(), equity=())

    def test_benchmark_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="one-to-one"):
            EquitySeries(
                timestamps=(_START,),
                equity=(Decimal("100"),),
                benchmark=(Decimal("50"), Decimal("51")),
            )

    def test_non_ascending_timestamps_rejected(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            EquitySeries(
                timestamps=(_START + timedelta(days=1), _START),
                equity=(Decimal("100"), Decimal("110")),
            )


class TestNormalizedLevels:
    def test_first_known_value_maps_to_one(self) -> None:
        levels = normalized_levels((Decimal("200"), Decimal("220"), Decimal("190")))

        assert levels == (Decimal("1"), Decimal("1.1"), Decimal("0.95"))

    def test_leading_nones_are_preserved(self) -> None:
        levels = normalized_levels((None, None, Decimal("50"), Decimal("75")))

        assert levels == (None, None, Decimal("1"), Decimal("1.5"))

    def test_all_none_stays_all_none(self) -> None:
        assert normalized_levels((None, None)) == (None, None)

    def test_zero_base_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero"):
            normalized_levels((Decimal("0"), Decimal("10")))


class TestDrawdownLevels:
    def test_hand_computed_sequence(self) -> None:
        values = tuple(Decimal(v) for v in ("100", "120", "90", "130", "91"))

        drawdowns = drawdown_levels(values)

        assert drawdowns == (
            Decimal("0"),
            Decimal("0"),
            Decimal("-0.25"),
            Decimal("0"),
            Decimal("-0.3"),
        )

    def test_monotonic_rise_is_all_zeros(self) -> None:
        values = tuple(Decimal(v) for v in ("100", "110", "120"))

        assert drawdown_levels(values) == (Decimal("0"), Decimal("0"), Decimal("0"))

    def test_minimum_matches_report_max_drawdown(self) -> None:
        equities = [100, 120, 90, 130, 91]
        report = BacktestReport(
            snapshots=tuple(_snap(eq, i) for i, eq in enumerate(equities)), fills=()
        )

        assert min(drawdown_levels(report.equity_curve)) == report.max_drawdown


class TestEvaluationSeriesEmbedding:
    def _evaluation_payload(self, *, include_series: bool) -> dict[str, object]:
        from value_portfolio.backtest import Window, evaluate_windows
        from value_portfolio.backtest.evaluation import evaluation_to_dict

        report = BacktestReport(
            snapshots=tuple(_snap(eq, i) for i, eq in enumerate([100, 100, 110])), fills=()
        )
        evaluation = evaluate_windows(
            [Window(_START, _START + timedelta(days=365))], lambda w: report
        )
        rendered: dict[str, object] = json.loads(
            json.dumps(evaluation_to_dict(evaluation, include_series=include_series))
        )
        return rendered

    def test_windows_carry_their_series(self) -> None:
        payload = self._evaluation_payload(include_series=True)

        windows = payload["windows"]
        assert isinstance(windows, list)
        series = windows[0]["series"]
        assert len(series["timestamps"]) == 3
        assert len(series["equity"]) == 3
        assert series["benchmark"] is None

    def test_include_series_false_omits_the_block(self) -> None:
        payload = self._evaluation_payload(include_series=False)

        windows = payload["windows"]
        assert isinstance(windows, list)
        assert "series" not in windows[0]
