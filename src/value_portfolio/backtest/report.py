
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from value_portfolio.backtest._stats import sample_cov, sample_std
from value_portfolio.broker.types import AccountSnapshot, Order

_ZERO = Decimal("0")
_ONE = Decimal("1")
_SECONDS_PER_YEAR = Decimal("31557600")
_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class BacktestReport:
    snapshots: tuple[AccountSnapshot, ...]
    fills: tuple[Order, ...]
    benchmark_levels: tuple[Decimal | None, ...] | None = None

    def __post_init__(self) -> None:
        if not self.snapshots:
            raise ValueError("a BacktestReport requires at least one account snapshot")
        if self.benchmark_levels is not None and len(self.benchmark_levels) != len(self.snapshots):
            raise ValueError(
                "benchmark_levels must align one-to-one with snapshots: "
                f"got {len(self.benchmark_levels)} levels for {len(self.snapshots)} snapshots"
            )

    # Equity curve

    @property
    def starting_equity(self) -> Decimal:
        return self.snapshots[0].equity

    @property
    def final_equity(self) -> Decimal:
        return self.snapshots[-1].equity

    @property
    def equity_curve(self) -> tuple[Decimal, ...]:
        return tuple(s.equity for s in self.snapshots)

    @property
    def periodic_returns(self) -> tuple[Decimal, ...]:
        eq = self.equity_curve
        return tuple(
            (eq[i] - eq[i - 1]) / eq[i - 1] for i in range(1, len(eq)) if eq[i - 1] > _ZERO
        )

    # Headline returns

    @property
    def total_return(self) -> Decimal:
        return (self.final_equity - self.starting_equity) / self.starting_equity

    @property
    def cagr(self) -> Decimal | None:
        if self.starting_equity <= _ZERO or self.final_equity <= _ZERO:
            return None
        start = self.snapshots[0].timestamp
        end = self.snapshots[-1].timestamp
        elapsed_seconds = Decimal(str((end - start).total_seconds()))
        if elapsed_seconds <= _ZERO:
            return None
        years = elapsed_seconds / _SECONDS_PER_YEAR
        ratio = self.final_equity / self.starting_equity
        return ((ratio.ln() / years).exp()) - _ONE

    # Risk-adjusted metrics

    @property
    def max_drawdown(self) -> Decimal:
        peak = self.snapshots[0].equity
        worst = _ZERO
        for snap in self.snapshots:
            if snap.equity > peak:
                peak = snap.equity
            if peak > _ZERO:
                drawdown = (snap.equity - peak) / peak
                if drawdown < worst:
                    worst = drawdown
        return worst

    def volatility(self, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> Decimal:
        if periods_per_year < 1:
            raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
        rets = self.periodic_returns
        if len(rets) < 2:
            return _ZERO
        std = sample_std(rets)
        return std * Decimal(periods_per_year).sqrt()

    def sharpe_ratio(
        self,
        periods_per_year: int = _TRADING_DAYS_PER_YEAR,
    ) -> Decimal | None:
        if periods_per_year < 1:
            raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
        rets = self.periodic_returns
        if len(rets) < 2:
            return None
        mean = sum(rets, _ZERO) / Decimal(len(rets))
        std = sample_std(rets)
        if std == _ZERO:
            return None
        return (mean / std) * Decimal(periods_per_year).sqrt()

    def _paired_returns(self) -> tuple[tuple[Decimal, Decimal], ...]:
        levels = self.benchmark_levels
        if levels is None:
            return ()
        eq = self.equity_curve
        pairs: list[tuple[Decimal, Decimal]] = []
        for i in range(1, len(eq)):
            prev_eq, prev_lvl, cur_lvl = eq[i - 1], levels[i - 1], levels[i]
            if prev_eq <= _ZERO or prev_lvl is None or prev_lvl <= _ZERO or cur_lvl is None:
                continue
            port_ret = (eq[i] - prev_eq) / prev_eq
            bench_ret = (cur_lvl - prev_lvl) / prev_lvl
            pairs.append((port_ret, bench_ret))
        return tuple(pairs)

    def _benchmark_endpoints(self) -> tuple[Decimal, Decimal] | None:
        if self.benchmark_levels is None:
            return None
        known = [lvl for lvl in self.benchmark_levels if lvl is not None]
        if len(known) < 2:
            return None
        return known[0], known[-1]

    @property
    def benchmark_return(self) -> Decimal | None:
        endpoints = self._benchmark_endpoints()
        if endpoints is None:
            return None
        first, last = endpoints
        if first <= _ZERO:
            return None
        return (last - first) / first

    @property
    def active_return(self) -> Decimal | None:
        bench = self.benchmark_return
        if bench is None:
            return None
        return self.total_return - bench

    def tracking_error(self, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> Decimal | None:
        if periods_per_year < 1:
            raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
        pairs = self._paired_returns()
        if len(pairs) < 2:
            return None
        active = tuple(port - bench for port, bench in pairs)
        return sample_std(active) * Decimal(periods_per_year).sqrt()

    def information_ratio(self, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> Decimal | None:
        if periods_per_year < 1:
            raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
        pairs = self._paired_returns()
        if len(pairs) < 2:
            return None
        active = tuple(port - bench for port, bench in pairs)
        std = sample_std(active)
        if std == _ZERO:
            return None
        mean = sum(active, _ZERO) / Decimal(len(active))
        return (mean / std) * Decimal(periods_per_year).sqrt()

    @property
    def beta(self) -> Decimal | None:
        pairs = self._paired_returns()
        if len(pairs) < 2:
            return None
        port = tuple(p for p, _ in pairs)
        bench = tuple(b for _, b in pairs)
        var_b = sample_cov(bench, bench)
        if var_b == _ZERO:
            return None
        return sample_cov(port, bench) / var_b

    def jensens_alpha(self, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> Decimal | None:
        if periods_per_year < 1:
            raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
        beta = self.beta
        if beta is None:
            return None
        pairs = self._paired_returns()
        port = tuple(p for p, _ in pairs)
        bench = tuple(b for _, b in pairs)
        mean_p = sum(port, _ZERO) / Decimal(len(port))
        mean_b = sum(bench, _ZERO) / Decimal(len(bench))
        return (mean_p - beta * mean_b) * Decimal(periods_per_year)

    # Activity

    @property
    def n_fills(self) -> int:
        return len(self.fills)

    @property
    def total_fill_notional(self) -> Decimal:
        return sum(
            (
                order.filled_qty * order.filled_avg_price
                for order in self.fills
                if order.filled_avg_price is not None
            ),
            _ZERO,
        )

    @property
    def turnover(self) -> Decimal:
        if self.starting_equity <= _ZERO:
            return _ZERO
        return self.total_fill_notional / self.starting_equity

    # Presentation

    def summary(self, periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> str:
        cagr = self.cagr
        sharpe = self.sharpe_ratio(periods_per_year)
        lines = [
            f"Starting equity   : {self.starting_equity:>15,.2f} USD",
            f"Final equity      : {self.final_equity:>15,.2f} USD",
            f"Total return      : {self.total_return * 100:>14,.2f} %",
            f"CAGR              : {cagr * 100:>14,.2f} %"
            if cagr is not None
            else f"CAGR              : {'n/a':>15}",
            f"Annualised vol    : {self.volatility(periods_per_year) * 100:>14,.2f} %",
            f"Sharpe ratio      : {sharpe:>15,.3f}"
            if sharpe is not None
            else f"Sharpe ratio      : {'n/a':>15}",
            f"Max drawdown      : {self.max_drawdown * 100:>14,.2f} %",
            f"Turnover          : {self.turnover:>14,.2f} x",
            f"Orders filled     : {self.n_fills:>15}",
        ]
        if self.benchmark_levels is not None:
            lines.extend(self._benchmark_summary_lines(periods_per_year))
        return "\n".join(lines)

    def _benchmark_summary_lines(self, periods_per_year: int) -> list[str]:
        def pct(value: Decimal | None) -> str:
            return f"{value * 100:>14,.2f} %" if value is not None else f"{'n/a':>15}"

        def num(value: Decimal | None) -> str:
            return f"{value:>15,.3f}" if value is not None else f"{'n/a':>15}"

        return [
            "-- vs benchmark --",
            f"Benchmark return  : {pct(self.benchmark_return)}",
            f"Active return     : {pct(self.active_return)}",
            f"Tracking error    : {pct(self.tracking_error(periods_per_year))}",
            f"Information ratio : {num(self.information_ratio(periods_per_year))}",
            f"Beta              : {num(self.beta)}",
            f"Jensen's alpha    : {pct(self.jensens_alpha(periods_per_year))}",
        ]
