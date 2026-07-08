"""Known-answer and property tests for PSR / expected-max-Sharpe / DSR."""

from __future__ import annotations

import math

import pytest

from value_portfolio.backtest.statistics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    sample_kurtosis,
    sample_skewness,
)


class TestProbabilisticSharpeRatio:
    def test_sr_equal_to_benchmark_is_half(self) -> None:
        psr = probabilistic_sharpe_ratio(0.1, sr_benchmark=0.1, n_obs=120)
        assert psr == pytest.approx(0.5)

    def test_gaussian_known_value(self) -> None:
        # Normal returns: var = 1 + (3-1)/4 * 0.1^2 = 1.005;
        # z = 0.1 * sqrt(99) / sqrt(1.005) = 0.99251; Phi(z) = 0.83952.
        psr = probabilistic_sharpe_ratio(0.1, sr_benchmark=0.0, n_obs=100)
        assert psr == pytest.approx(0.83952, abs=1e-4)

    def test_more_observations_raise_confidence(self) -> None:
        short = probabilistic_sharpe_ratio(0.1, sr_benchmark=0.0, n_obs=60)
        long = probabilistic_sharpe_ratio(0.1, sr_benchmark=0.0, n_obs=600)
        assert long > short

    def test_negative_skew_and_fat_tails_reduce_confidence(self) -> None:
        normal = probabilistic_sharpe_ratio(0.2, sr_benchmark=0.0, n_obs=120)
        ugly = probabilistic_sharpe_ratio(
            0.2, sr_benchmark=0.0, n_obs=120, skewness=-1.0, kurtosis=6.0
        )
        assert ugly < normal

    def test_too_few_observations_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_obs"):
            probabilistic_sharpe_ratio(0.1, sr_benchmark=0.0, n_obs=1)


class TestExpectedMaxSharpe:
    def test_grows_with_trials(self) -> None:
        few = expected_max_sharpe(5, sr_variance=0.01)
        many = expected_max_sharpe(100, sr_variance=0.01)
        assert 0 < few < many

    def test_scales_with_dispersion(self) -> None:
        # E[max] is linear in the std-dev across trials.
        narrow = expected_max_sharpe(10, sr_variance=0.01)
        wide = expected_max_sharpe(10, sr_variance=0.04)
        assert wide == pytest.approx(2.0 * narrow)

    def test_zero_variance_trials_have_zero_max(self) -> None:
        assert expected_max_sharpe(10, sr_variance=0.0) == 0.0

    def test_single_trial_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            expected_max_sharpe(1, sr_variance=0.01)


class TestDeflatedSharpeRatio:
    def test_more_trials_deflate_more(self) -> None:
        kwargs = {"sr_variance": 0.02, "n_obs": 216, "skewness": -0.5, "kurtosis": 4.0}
        few = deflated_sharpe_ratio(0.15, n_trials=3, **kwargs)
        many = deflated_sharpe_ratio(0.15, n_trials=50, **kwargs)
        assert many < few

    def test_observed_at_expected_max_is_half(self) -> None:
        benchmark = expected_max_sharpe(10, sr_variance=0.02)
        dsr = deflated_sharpe_ratio(benchmark, n_trials=10, sr_variance=0.02, n_obs=216)
        assert dsr == pytest.approx(0.5)

    def test_strong_sharpe_survives_deflation(self) -> None:
        dsr = deflated_sharpe_ratio(0.5, n_trials=10, sr_variance=0.01, n_obs=216)
        assert dsr > 0.95

    def test_marginal_sharpe_does_not_survive_many_trials(self) -> None:
        best_of_noise = expected_max_sharpe(100, sr_variance=0.02)
        dsr = deflated_sharpe_ratio(best_of_noise * 0.9, n_trials=100, sr_variance=0.02, n_obs=216)
        assert dsr < 0.5

    def test_paper_example_magnitude(self) -> None:
        # Bailey & López de Prado (2014) flavor: monthly SR ~0.32 looks great
        # in isolation (PSR ~ 0.96) but much less so as the best of 45 trials.
        plain = probabilistic_sharpe_ratio(0.32, sr_benchmark=0.0, n_obs=24)
        deflated = deflated_sharpe_ratio(0.32, n_trials=45, sr_variance=0.05, n_obs=24)
        assert plain > 0.9
        assert deflated < plain - 0.2

    def test_deterministic(self) -> None:
        a = deflated_sharpe_ratio(0.2, n_trials=12, sr_variance=0.02, n_obs=216)
        b = deflated_sharpe_ratio(0.2, n_trials=12, sr_variance=0.02, n_obs=216)
        assert a == b
        assert not math.isnan(a)


class TestPerPeriodSharpe:
    def test_known_value(self) -> None:
        # mean = 0.02, sample std of [0.01, 0.03] = 0.0141421..., ratio = 1.41421...
        assert per_period_sharpe([0.01, 0.03]) == pytest.approx(2.0 / math.sqrt(2.0))

    def test_is_unannualised_mean_over_sample_std(self) -> None:
        # The bare mean/std (no sqrt(252) factor): the per-period quantity the
        # DSR deflates, annualised only for display.
        returns = [0.001, -0.002, 0.003, 0.0, 0.004, -0.001]
        n = len(returns)
        mean = sum(returns) / n
        std = math.sqrt(sum((r - mean) ** 2 for r in returns) / (n - 1))
        assert per_period_sharpe(returns) == pytest.approx(mean / std)

    def test_too_few_returns_rejected(self) -> None:
        with pytest.raises(ValueError, match="2 returns"):
            per_period_sharpe([0.01])

    def test_zero_variance_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero-variance"):
            per_period_sharpe([0.01, 0.01, 0.01])


class TestSampleMoments:
    def test_symmetric_series_has_zero_skew(self) -> None:
        assert sample_skewness([-2.0, -1.0, 0.0, 1.0, 2.0]) == pytest.approx(0.0, abs=1e-12)

    def test_skew_sign_follows_the_long_tail(self) -> None:
        right_tailed = sample_skewness([0.0, 0.0, 0.0, 0.0, 10.0])
        left_tailed = sample_skewness([0.0, 0.0, 0.0, 0.0, -10.0])
        assert right_tailed > 0 > left_tailed

    def test_kurtosis_is_non_excess(self) -> None:
        # A symmetric two-point ±1 distribution has kurtosis exactly 1.0
        # (non-excess convention; a Gaussian would be 3.0).
        assert sample_kurtosis([-1.0, 1.0, -1.0, 1.0]) == pytest.approx(1.0)

    def test_fat_tails_raise_kurtosis(self) -> None:
        peaky = sample_kurtosis([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, -5.0])
        flat = sample_kurtosis([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert peaky > flat

    def test_moments_reject_degenerate_input(self) -> None:
        with pytest.raises(ValueError, match="zero-variance"):
            sample_skewness([3.0, 3.0])
        with pytest.raises(ValueError, match="zero-variance"):
            sample_kurtosis([3.0, 3.0])
