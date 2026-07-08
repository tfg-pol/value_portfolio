"""Signal-diagnostics unit tests: known-answer checks on small panels."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from value_portfolio.learning.diagnostics import (
    decile_spread,
    evaluate_signal,
    newey_west_tstat,
    spearman_rank_ic,
)


def _date(month: int) -> datetime:
    return datetime(2020, month, 28, tzinfo=UTC)


class TestSpearmanRankIC:
    def test_perfect_agreement_is_one(self) -> None:
        scores = np.array([0.1, 0.5, 0.3, 0.9])
        returns = np.array([0.01, 0.05, 0.03, 0.09])
        assert spearman_rank_ic(scores, returns) == pytest.approx(1.0)

    def test_perfect_disagreement_is_minus_one(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        returns = np.array([0.04, 0.03, 0.02, 0.01])
        assert spearman_rank_ic(scores, returns) == pytest.approx(-1.0)

    def test_monotone_transform_invariance(self) -> None:
        rng = np.random.default_rng(7)
        scores = rng.normal(size=50)
        returns = rng.normal(size=50)
        base = spearman_rank_ic(scores, returns)
        assert spearman_rank_ic(np.exp(scores), returns) == pytest.approx(base)

    def test_ties_use_average_ranks(self) -> None:
        # Hand-computed: scores (1, 1, 2) -> ranks (1.5, 1.5, 3);
        # returns (1, 2, 3) -> ranks (1, 2, 3). Pearson on those ranks:
        # cov = 1.5, sd_s = sqrt(1.5), sd_r = sqrt(2) -> rho = sqrt(3)/2.
        scores = np.array([1.0, 1.0, 2.0])
        returns = np.array([0.01, 0.02, 0.03])
        assert spearman_rank_ic(scores, returns) == pytest.approx(math.sqrt(3) / 2)

    def test_constant_scores_undefined(self) -> None:
        scores = np.array([1.0, 1.0, 1.0])
        returns = np.array([0.01, 0.02, 0.03])
        assert math.isnan(spearman_rank_ic(scores, returns))


class TestDecileSpread:
    def test_top_minus_bottom(self) -> None:
        # 20 names, scores 0..19, returns = score / 100: top decile (scores
        # 18, 19) returns mean 0.185, bottom (0, 1) mean 0.005.
        scores = np.arange(20, dtype=np.float64)
        returns = scores / 100.0
        assert decile_spread(scores, returns) == pytest.approx(0.18)

    def test_too_thin_returns_none(self) -> None:
        scores = np.arange(15, dtype=np.float64)
        assert decile_spread(scores, scores) is None


class TestNeweyWest:
    def test_zero_mean_series(self) -> None:
        assert newey_west_tstat([1.0, -1.0, 1.0, -1.0]) == pytest.approx(0.0)

    def test_iid_matches_plain_tstat_at_zero_lags(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(0.5, 1.0, size=200)
        # lags=0 -> long-run variance is the population variance.
        expected = x.mean() / math.sqrt(x.var() / len(x))
        assert newey_west_tstat(x.tolist(), lags=0) == pytest.approx(expected)

    def test_positive_autocorrelation_shrinks_t(self) -> None:
        rng = np.random.default_rng(11)
        noise = rng.normal(size=300)
        ar = np.empty(300)
        ar[0] = noise[0]
        for i in range(1, 300):  # AR(1), rho=0.8: strongly autocorrelated
            ar[i] = 0.8 * ar[i - 1] + noise[i]
        ar += 0.5
        assert abs(newey_west_tstat(ar.tolist(), lags=12)) < abs(
            newey_west_tstat(ar.tolist(), lags=0)
        )


class TestEvaluateSignal:
    def test_aligns_dates_and_symbols(self) -> None:
        scores = {
            _date(1): {f"S{i}": float(i) for i in range(40)},
            _date(2): {f"S{i}": float(-i) for i in range(40)},
            _date(3): {f"S{i}": float(i) for i in range(5)},  # too thin: skipped
        }
        returns = {
            _date(1): {f"S{i}": i / 100.0 for i in range(40)},
            _date(2): {f"S{i}": i / 100.0 for i in range(40)},
            _date(3): {f"S{i}": i / 100.0 for i in range(5)},
            _date(4): {f"S{i}": i / 100.0 for i in range(40)},  # unscored: ignored
        }
        result = evaluate_signal(scores, returns, min_names=30)
        assert [m.date for m in result.months] == [_date(1), _date(2)]
        assert result.months[0].rank_ic == pytest.approx(1.0)
        assert result.months[1].rank_ic == pytest.approx(-1.0)
        assert result.mean_ic == pytest.approx(0.0)
        assert result.ic_hit_rate == pytest.approx(0.5)

    def test_symbols_missing_a_return_are_dropped(self) -> None:
        scores = {_date(1): {f"S{i}": float(i) for i in range(40)}}
        returns = {_date(1): {f"S{i}": i / 100.0 for i in range(35)}}
        result = evaluate_signal(scores, returns, min_names=30)
        assert result.months[0].n_names == 35

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(5)
        scores = {_date(m): {f"S{i}": float(rng.normal()) for i in range(60)} for m in (1, 2, 3)}
        returns = {_date(m): {f"S{i}": float(rng.normal()) for i in range(60)} for m in (1, 2, 3)}
        a = evaluate_signal(scores, returns)
        b = evaluate_signal(scores, returns)
        assert a == b
