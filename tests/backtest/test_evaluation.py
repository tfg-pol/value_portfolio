"""Tests for the multi-window evaluator: rolling-window generation, metric
distribution math, skipped windows, and an end-to-end run over synthetic bars.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.backtest import (
    BacktestReport,
    Window,
    evaluate_windows,
    rolling_windows,
    run_backtest,
)
from value_portfolio.baselines import BuyAndHold
from value_portfolio.broker import SimulatedBroker
from value_portfolio.data import InMemoryMarketData
from value_portfolio.data.types import Bar
from value_portfolio.rebalancer import Rebalancer

_START = datetime(2010, 1, 1, tzinfo=UTC)


def _d(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


class TestRollingWindows:
    def test_yearly_steps_cover_the_range(self) -> None:
        windows = rolling_windows(_d(2008), _d(2015), window_years=5, step_months=12)
        assert [w.start.year for w in windows] == [2008, 2009, 2010]
        assert [w.end.year for w in windows] == [2013, 2014, 2015]

    def test_last_window_must_fit_entirely(self) -> None:
        windows = rolling_windows(_d(2008), _d(2014, 12, 31), window_years=5, step_months=12)
        # 2010 -> 2015 would overrun 2014-12-31, so 2009 -> 2014 is the last.
        assert windows[-1].end == _d(2014)

    def test_sub_year_steps(self) -> None:
        windows = rolling_windows(_d(2010), _d(2012), window_years=1, step_months=6)
        assert [(w.start.year, w.start.month) for w in windows] == [(2010, 1), (2010, 7), (2011, 1)]

    def test_empty_when_window_does_not_fit(self) -> None:
        assert rolling_windows(_d(2010), _d(2012), window_years=5) == []

    def test_invalid_window_rejected(self) -> None:
        with pytest.raises(ValueError):
            Window(start=_d(2010), end=_d(2010))


class TestEvaluateWindows:
    def _report_for(self, prices: list[float]) -> BacktestReport:
        bars = [
            Bar(
                symbol="AAA",
                timestamp=_START + timedelta(days=i),
                open=Decimal(str(p)),
                high=Decimal(str(p)),
                low=Decimal(str(p)),
                close=Decimal(str(p)),
                volume=Decimal("1000"),
                timeframe="1Day",
            )
            for i, p in enumerate(prices)
        ]
        broker = SimulatedBroker(
            market_data=InMemoryMarketData({"AAA": bars}),
            starting_cash=Decimal("1000"),
        )
        return run_backtest(
            BuyAndHold(["AAA"]),
            broker,
            broker.market_data,
            rebalancer=Rebalancer(cash_buffer=Decimal("0")),
        )

    def test_collects_reports_and_summarizes_distributions(self) -> None:
        # Two windows with known total returns: 0% and +10% (10 shares of AAA).
        reports = {2010: self._report_for([100, 100, 100]), 2011: self._report_for([100, 100, 110])}
        windows = [Window(_d(2010), _d(2011)), Window(_d(2011), _d(2012))]

        evaluation = evaluate_windows(windows, lambda w: reports[w.start.year])

        returns = evaluation.metric_values("total_return")
        assert returns == (Decimal("0"), Decimal("0.1"))
        summary = evaluation.summary()
        assert "Windows evaluated  : 2" in summary
        assert "total_return" in summary

    def test_none_windows_are_skipped(self) -> None:
        windows = [Window(_d(2010), _d(2011)), Window(_d(2011), _d(2012))]
        report = self._report_for([100, 100, 105])

        evaluation = evaluate_windows(windows, lambda w: report if w.start.year == 2010 else None)

        assert len(evaluation.results) == 1
        assert evaluation.results[0].window.start.year == 2010

    def test_undefined_metrics_are_dropped_not_zeroed(self) -> None:
        # No benchmark threaded -> benchmark-relative metrics are undefined.
        evaluation = evaluate_windows(
            [Window(_d(2010), _d(2011))], lambda w: self._report_for([100, 100])
        )
        assert evaluation.metric_values("information_ratio") == ()
        assert "information_ratio" not in evaluation.summary()

    def test_per_window_lines_are_chronological(self) -> None:
        windows = [Window(_d(2010), _d(2011)), Window(_d(2011), _d(2012))]
        evaluation = evaluate_windows(windows, lambda w: self._report_for([100, 100, 110]))
        lines = evaluation.per_window_lines("total_return")
        assert len(lines) == 2
        assert "2010-01-01" in lines[0] and "2011-01-01" in lines[1]


class TestSerialization:
    def _report(self) -> BacktestReport:
        return TestEvaluateWindows._report_for(TestEvaluateWindows(), [100, 100, 110])

    def test_report_metrics_covers_the_catalog(self) -> None:
        from value_portfolio.backtest.evaluation import report_metrics

        metrics = report_metrics(self._report())
        assert metrics["total_return"] == Decimal("0.1")
        # No benchmark threaded -> relative metrics are None, not zero.
        assert metrics["information_ratio"] is None

    def test_evaluation_to_dict_round_trips_through_json(self) -> None:
        import json

        from value_portfolio.backtest.evaluation import evaluation_to_dict

        evaluation = evaluate_windows(
            [Window(_d(2010), _d(2011)), Window(_d(2011), _d(2012))],
            lambda w: self._report(),
        )
        payload = json.loads(json.dumps(evaluation_to_dict(evaluation)))

        assert payload["n_windows"] == 2
        assert payload["windows"][0]["start"] == "2010-01-01"
        # Decimals persisted as strings, full precision.
        assert payload["windows"][0]["total_return"] == "0.1"
        assert payload["summary"]["total_return"]["mean"] == "0.1"
        assert "information_ratio" not in payload["summary"]
