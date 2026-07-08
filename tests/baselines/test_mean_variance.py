"""Tests for `MeanVariance.decide`: cadence, warm-up, look-ahead safety, and the
two modes (``min_var`` / ``mean_var``) on synthetic returns with clear optima.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from value_portfolio.agent import DecisionContext
from value_portfolio.baselines import MeanVariance
from value_portfolio.broker import SimulatedBroker
from value_portfolio.data import InMemoryMarketData
from value_portfolio.data.types import Bar

_START = datetime(2026, 1, 1, tzinfo=UTC)


def _context(broker: SimulatedBroker, data: InMemoryMarketData) -> DecisionContext:
    """Build the read-only context the agent sees as of the data clock."""
    return DecisionContext(now=data.now, account=broker.get_account(), data=data)


def _bars(symbol: str, closes: list[float]) -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            timestamp=_START + timedelta(days=i),
            open=Decimal(str(c)),
            high=Decimal(str(c)),
            low=Decimal(str(c)),
            close=Decimal(str(c)),
            volume=Decimal("1000"),
            timeframe="1Day",
        )
        for i, c in enumerate(closes)
    ]


def _ctx(bars_by_symbol: dict[str, list[Bar]]) -> tuple[SimulatedBroker, InMemoryMarketData]:
    data = InMemoryMarketData(bars_by_symbol)
    return SimulatedBroker(market_data=data), data


def _closes_from_returns(start: float, returns: list[float]) -> list[float]:
    """Build a close-price series from an initial price and per-step returns."""
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1.0 + r))
    return closes


class TestConstruction:
    def test_rejects_empty_symbols(self) -> None:
        with pytest.raises(ValueError):
            MeanVariance([])

    def test_rejects_duplicate_symbols(self) -> None:
        with pytest.raises(ValueError):
            MeanVariance(["AAPL", "AAPL"])

    def test_rejects_lookback_below_two(self) -> None:
        with pytest.raises(ValueError):
            MeanVariance(["AAPL", "MSFT"], lookback=1)

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError):
            MeanVariance(["AAPL", "MSFT"], mode="max_sharpe")  # type: ignore[arg-type]

    def test_rejects_negative_risk_aversion(self) -> None:
        with pytest.raises(ValueError):
            MeanVariance(["AAPL", "MSFT"], risk_aversion=Decimal("-1"))

    def test_rejects_negative_ridge(self) -> None:
        with pytest.raises(ValueError):
            MeanVariance(["AAPL", "MSFT"], ridge=Decimal("-0.0001"))

    def test_rejects_non_positive_rebalance_every(self) -> None:
        with pytest.raises(ValueError):
            MeanVariance(["AAPL", "MSFT"], rebalance_every=0)


class TestDecide:
    def test_returns_none_during_warmup(self) -> None:
        # lookback=5 needs 6 bars; only 3 visible at NOW.
        bars = {
            "AAPL": _bars("AAPL", [100, 101, 102, 103, 104, 105, 106]),
            "MSFT": _bars("MSFT", [100, 101, 102, 103, 104, 105, 106]),
        }
        broker, data = _ctx(bars)
        agent = MeanVariance(["AAPL", "MSFT"], lookback=5, rebalance_every=1)

        data.advance_to(_START + timedelta(days=2))  # only 3 bars visible
        assert agent.decide(_context(broker, data)) is None

    def test_rebalance_every_skips_intermediate_steps(self) -> None:
        bars = {
            "AAPL": _bars("AAPL", _closes_from_returns(100, [0.01, -0.005] * 10)),
            "MSFT": _bars("MSFT", _closes_from_returns(100, [-0.005, 0.01] * 10)),
        }
        broker, data = _ctx(bars)
        agent = MeanVariance(["AAPL", "MSFT"], lookback=5, rebalance_every=2)

        data.advance_to(_START + timedelta(days=20))
        assert agent.decide(_context(broker, data)) is not None  # step 0
        assert agent.decide(_context(broker, data)) is None  # step 1
        assert agent.decide(_context(broker, data)) is not None  # step 2

    def test_min_var_balances_two_symmetric_assets(self) -> None:
        # Two anti-correlated streams with the *same* per-step variance.
        # By symmetry the minimum-variance long-only portfolio is 50/50.
        rets_a = [0.01, -0.01] * 10
        rets_b = [-0.01, 0.01] * 10
        bars = {
            "A": _bars("A", _closes_from_returns(100, rets_a)),
            "B": _bars("B", _closes_from_returns(100, rets_b)),
        }
        broker, data = _ctx(bars)
        agent = MeanVariance(["A", "B"], lookback=10, mode="min_var", rebalance_every=1)

        data.advance_to(_START + timedelta(days=len(rets_a)))
        weights = agent.decide(_context(broker, data))

        assert weights is not None
        assert abs(weights["A"] - Decimal("0.5")) < Decimal("0.01")
        assert abs(weights["B"] - Decimal("0.5")) < Decimal("0.01")

    def test_min_var_tilts_toward_less_volatile_asset(self) -> None:
        # LOW: tiny ±0.1 % swings. HIGH: large ±5 % swings. Both have
        # zero mean per step; the cycles are offset so they are not
        # perfectly correlated. Min-variance must put most weight on LOW.
        low_rets = [0.001, -0.001] * 15
        high_rets = [0.05, 0.05, -0.05, -0.05] * 8  # 32 returns; trim
        high_rets = high_rets[: len(low_rets)]
        bars = {
            "LOW": _bars("LOW", _closes_from_returns(100, low_rets)),
            "HIGH": _bars("HIGH", _closes_from_returns(100, high_rets)),
        }
        broker, data = _ctx(bars)
        agent = MeanVariance(["LOW", "HIGH"], lookback=20, mode="min_var", rebalance_every=1)

        data.advance_to(_START + timedelta(days=len(low_rets)))
        weights = agent.decide(_context(broker, data))

        assert weights is not None
        assert weights["LOW"] > Decimal("0.9")
        assert weights["HIGH"] < Decimal("0.1")

    def test_mean_var_with_large_gamma_matches_min_var(self) -> None:
        # Same data as the previous test. With gamma very large the
        # risk term dominates the return term and the solution must
        # agree with the min-variance one to within tolerance.
        low_rets = [0.001, -0.001] * 15
        high_rets = [0.05, 0.05, -0.05, -0.05] * 8
        high_rets = high_rets[: len(low_rets)]
        bars = {
            "LOW": _bars("LOW", _closes_from_returns(100, low_rets)),
            "HIGH": _bars("HIGH", _closes_from_returns(100, high_rets)),
        }
        broker, data = _ctx(bars)
        mv_agent = MeanVariance(
            ["LOW", "HIGH"],
            lookback=20,
            mode="mean_var",
            risk_aversion=Decimal("1000000"),
            rebalance_every=1,
        )
        ref_agent = MeanVariance(["LOW", "HIGH"], lookback=20, mode="min_var", rebalance_every=1)

        data.advance_to(_START + timedelta(days=len(low_rets)))
        mv_weights = mv_agent.decide(_context(broker, data))
        ref_weights = ref_agent.decide(_context(broker, data))

        assert mv_weights is not None and ref_weights is not None
        assert abs(mv_weights["LOW"] - ref_weights["LOW"]) < Decimal("0.01")

    def test_mean_var_with_small_gamma_tilts_toward_higher_mean(self) -> None:
        # WINNER drifts up at +1 % per step; LOSER drifts down at -1 %.
        # Variances are identical. With small gamma the return term
        # dominates -> nearly all weight should land on WINNER.
        winner_closes = _closes_from_returns(100, [0.01] * 20)
        loser_closes = _closes_from_returns(100, [-0.01] * 20)
        bars = {
            "WIN": _bars("WIN", winner_closes),
            "LOSE": _bars("LOSE", loser_closes),
        }
        broker, data = _ctx(bars)
        agent = MeanVariance(
            ["WIN", "LOSE"],
            lookback=15,
            mode="mean_var",
            risk_aversion=Decimal("0.01"),
            rebalance_every=1,
        )

        data.advance_to(_START + timedelta(days=20))
        weights = agent.decide(_context(broker, data))

        assert weights is not None
        assert weights["WIN"] > Decimal("0.9")
        assert weights["LOSE"] < Decimal("0.1")

    def test_weights_are_long_only_and_fully_invested(self) -> None:
        # Mixed return streams; any feasible solution must respect the
        # constraints regardless of the specific numerical answer.
        bars = {
            "A": _bars("A", _closes_from_returns(100, [0.02, -0.01] * 15)),
            "B": _bars("B", _closes_from_returns(100, [-0.01, 0.03, -0.02] * 10)),
            "C": _bars("C", _closes_from_returns(100, [0.005, 0.005, -0.01] * 10)),
        }
        broker, data = _ctx(bars)
        agent = MeanVariance(["A", "B", "C"], lookback=15, mode="min_var", rebalance_every=1)

        data.advance_to(_START + timedelta(days=29))
        weights = agent.decide(_context(broker, data))

        assert weights is not None
        assert all(w >= Decimal("0") for w in weights.values())
        total = sum(weights.values(), Decimal("0"))
        # Quantisation can shave up to 1e-6 per asset off the float sum.
        assert Decimal("0.999") <= total <= Decimal("1")

    def test_excludes_symbols_without_enough_history(self) -> None:
        # A and C have plenty of history. B only has 2 bars (one return).
        # With lookback=5 only A and C qualify; the optimiser must run on
        # those two alone, and B must not appear in the returned weights.
        bars = {
            "A": _bars("A", _closes_from_returns(100, [0.01, -0.01] * 10)),
            "B": _bars("B", [100, 101]),
            "C": _bars("C", _closes_from_returns(100, [-0.01, 0.01] * 10)),
        }
        broker, data = _ctx(bars)
        agent = MeanVariance(["A", "B", "C"], lookback=5, mode="min_var", rebalance_every=1)

        data.advance_to(_START + timedelta(days=19))
        weights = agent.decide(_context(broker, data))

        assert weights is not None
        assert set(weights.keys()) == {"A", "C"}

    def test_does_not_peek_past_the_clock(self) -> None:
        # 30 bars exist for both symbols, but the clock is at index 2.
        # Only 3 bars are visible -> lookback=5 cannot be satisfied and
        # the agent must return None, not silently use future bars.
        bars = {
            "A": _bars("A", _closes_from_returns(100, [0.01, -0.01] * 15)),
            "B": _bars("B", _closes_from_returns(100, [-0.01, 0.01] * 15)),
        }
        broker, data = _ctx(bars)
        agent = MeanVariance(["A", "B"], lookback=5, mode="min_var", rebalance_every=1)

        data.advance_to(_START + timedelta(days=2))
        assert agent.decide(_context(broker, data)) is None
