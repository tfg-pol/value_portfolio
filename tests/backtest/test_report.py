"""Tests for `BacktestReport` metrics, checked against hand-computed values on
synthetic snapshot/order fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.backtest import BacktestReport
from value_portfolio.broker.types import AccountSnapshot, Order, OrderSide, OrderStatus

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


def _fill(symbol: str, qty: float, price: float, day_offset: int) -> Order:
    return Order(
        id=f"{symbol}-{day_offset}",
        client_order_id=None,
        symbol=symbol,
        qty=Decimal(str(qty)),
        side=OrderSide.BUY,
        status=OrderStatus.FILLED,
        submitted_at=_START + timedelta(days=day_offset),
        filled_qty=Decimal(str(qty)),
        filled_avg_price=Decimal(str(price)),
        filled_at=_START + timedelta(days=day_offset),
    )


def _report(equities: list[float], fills: tuple[Order, ...] = ()) -> BacktestReport:
    snapshots = tuple(_snap(eq, i) for i, eq in enumerate(equities))
    return BacktestReport(snapshots=snapshots, fills=fills)


def _report_with_bench(equities: list[float], levels: list[float | None]) -> BacktestReport:
    snapshots = tuple(_snap(eq, i) for i, eq in enumerate(equities))
    bench = tuple(None if v is None else Decimal(str(v)) for v in levels)
    return BacktestReport(snapshots=snapshots, fills=(), benchmark_levels=bench)


class TestEquityAndReturns:
    def test_starting_and_final_equity(self) -> None:
        report = _report([1000, 1100, 1200])

        assert report.starting_equity == Decimal("1000")
        assert report.final_equity == Decimal("1200")

    def test_total_return(self) -> None:
        report = _report([1000, 1500])

        assert report.total_return == Decimal("0.5")

    def test_periodic_returns(self) -> None:
        report = _report([100, 110, 99])

        rets = report.periodic_returns
        assert rets[0] == Decimal("0.1")
        assert rets[1] == Decimal("-0.1")
        assert len(rets) == 2


class TestMaxDrawdown:
    def test_monotonic_increase_has_zero_drawdown(self) -> None:
        report = _report([100, 110, 120, 130])

        assert report.max_drawdown == Decimal("0")

    def test_peak_to_trough_decline(self) -> None:
        # peak 120, trough 90 -> -25 %.
        report = _report([100, 120, 110, 90, 100])

        assert report.max_drawdown == Decimal("-0.25")

    def test_drawdown_does_not_recover_to_zero(self) -> None:
        # Even if the curve recovers above the prior peak, max drawdown
        # remains the worst observed dip.
        report = _report([100, 80, 200])

        assert report.max_drawdown == Decimal("-0.2")


class TestSharpeAndVolatility:
    def test_constant_equity_has_no_sharpe(self) -> None:
        report = _report([100, 100, 100, 100])

        assert report.sharpe_ratio() is None
        assert report.volatility() == Decimal("0")

    def test_sharpe_with_known_returns(self) -> None:
        # Equity 100 -> 110 -> 121 -> two equal +10 % returns.
        # std == 0 -> Sharpe is undefined.
        report = _report([100, 110, 121])

        assert report.sharpe_ratio() is None

    def test_sharpe_positive_when_mean_return_positive(self) -> None:
        # Up-down-up: nonzero variance, positive mean -> positive Sharpe.
        report = _report([100, 110, 105, 115])

        sharpe = report.sharpe_ratio()
        assert sharpe is not None
        assert sharpe > Decimal("0")

    def test_volatility_scales_with_period(self) -> None:
        report = _report([100, 110, 105, 115])

        daily = report.volatility(periods_per_year=252)
        weekly = report.volatility(periods_per_year=52)
        # Higher annualisation factor -> larger annualised vol.
        assert daily > weekly

    def test_sharpe_rejects_non_positive_periods(self) -> None:
        report = _report([100, 110, 120])
        with pytest.raises(ValueError):
            report.sharpe_ratio(periods_per_year=0)


class TestCagr:
    def test_cagr_matches_total_return_over_one_year(self) -> None:
        # ~365.25 days between first and last snapshot.
        snapshots = (
            _snap(1000, 0),
            _snap(1100, 365),
            _snap(1210, 731),  # ~2 years total
        )
        report = BacktestReport(snapshots=snapshots, fills=())

        cagr = report.cagr
        assert cagr is not None
        # final/start = 1.21 over ~2 years -> CAGR ~10 %.
        assert abs(cagr - Decimal("0.1")) < Decimal("0.01")

    def test_cagr_none_when_no_time_elapsed(self) -> None:
        snapshots = (_snap(1000, 0), _snap(1100, 0))
        report = BacktestReport(snapshots=snapshots, fills=())

        assert report.cagr is None


class TestTurnover:
    def test_turnover_with_no_fills_is_zero(self) -> None:
        report = _report([1000, 1100])

        assert report.turnover == Decimal("0")
        assert report.total_fill_notional == Decimal("0")

    def test_turnover_sums_fill_notional(self) -> None:
        # Two fills: 10 @ 50 + 5 @ 100 = 500 + 500 = 1000. Equity 1000 -> 1.0x.
        fills = (_fill("AAPL", 10, 50, 0), _fill("MSFT", 5, 100, 1))
        report = _report([1000, 1100], fills=fills)

        assert report.total_fill_notional == Decimal("1000")
        assert report.turnover == Decimal("1")


class TestReportValidation:
    def test_rejects_empty_snapshots(self) -> None:
        with pytest.raises(ValueError, match="at least one account snapshot"):
            BacktestReport(snapshots=(), fills=())

    def test_rejects_misaligned_benchmark_levels(self) -> None:
        snapshots = (_snap(100, 0), _snap(110, 1))
        with pytest.raises(ValueError, match="align one-to-one"):
            BacktestReport(snapshots=snapshots, fills=(), benchmark_levels=(Decimal("1"),))


class TestBenchmarkRelativeMetrics:
    def test_all_none_without_benchmark(self) -> None:
        report = _report([100, 110, 105, 115])

        assert report.benchmark_return is None
        assert report.active_return is None
        assert report.tracking_error() is None
        assert report.information_ratio() is None
        assert report.beta is None
        assert report.jensens_alpha() is None

    def test_portfolio_equals_benchmark(self) -> None:
        # Identical curves: zero active return, beta 1, zero alpha, and an
        # undefined information ratio (no active-return variance).
        curve = [100.0, 110.0, 105.0, 115.0]
        report = _report_with_bench(curve, curve)

        assert report.benchmark_return == report.total_return
        assert report.active_return == Decimal("0")
        assert report.tracking_error() == Decimal("0")
        assert report.information_ratio() is None
        assert report.beta == Decimal("1")
        assert report.jensens_alpha() == Decimal("0")

    def test_asymmetric_case_hand_computed(self) -> None:
        # Portfolio returns: +0.10, -0.05, +0.20 ; benchmark: +0.04, -0.02, +0.06.
        equities = [100.0, 110.0, 104.5, 125.4]
        levels = [100.0, 104.0, 101.92, 108.0352]
        report = _report_with_bench(equities, levels)

        # Total returns: port (125.4-100)/100 = 0.254 ; bench 0.080352.
        assert report.benchmark_return == Decimal("0.080352")
        assert report.active_return == Decimal("0.254") - Decimal("0.080352")

        # beta = cov(r_p, r_b) / var(r_b) = 0.093 / 0.0312 = 2.980769...
        beta = report.beta
        assert beta is not None
        assert abs(beta - Decimal("2.9807692307")) < Decimal("1e-9")

        # Active returns vary (0.06, -0.03, 0.14) -> IR and TE are defined.
        assert report.information_ratio() is not None
        te = report.tracking_error()
        assert te is not None
        assert te > Decimal("0")

        # alpha = (mean_p - beta * mean_b) * periods, mean_p = 0.25/3, mean_b = 0.08/3.
        alpha = report.jensens_alpha(periods_per_year=1)
        assert alpha is not None
        mean_p = Decimal("0.25") / Decimal("3")
        mean_b = Decimal("0.08") / Decimal("3")
        assert abs(alpha - (mean_p - beta * mean_b)) < Decimal("1e-9")

    def test_metrics_none_with_fewer_than_two_paired_returns(self) -> None:
        # Only the last step has a known level -> zero usable pairs.
        report = _report_with_bench([100.0, 110.0, 120.0], [None, None, 120.0])

        assert report.beta is None
        assert report.information_ratio() is None
        assert report.tracking_error() is None
        # Endpoints need two known levels.
        assert report.benchmark_return is None

    def test_summary_includes_benchmark_block_only_when_attached(self) -> None:
        plain = _report([100, 110, 105, 115])
        assert "vs benchmark" not in plain.summary()

        curve = [100.0, 110.0, 105.0, 115.0]
        with_bench = _report_with_bench(curve, curve)
        block = with_bench.summary()
        assert "vs benchmark" in block
        assert "Information ratio :" in block
        # Undefined IR renders aligned as n/a.
        assert "n/a" in block
