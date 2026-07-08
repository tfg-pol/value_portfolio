from __future__ import annotations

import math
from collections.abc import Sequence

_EULER_MASCHERONI = 0.5772156649015329


def per_period_sharpe(returns: Sequence[float]) -> float:
    n = len(returns)
    if n < 2:
        raise ValueError(f"per_period_sharpe needs >= 2 returns, got {n}")
    mean = math.fsum(returns) / n
    variance = math.fsum((r - mean) ** 2 for r in returns) / (n - 1)
    if variance <= 0:
        raise ValueError("per_period_sharpe is undefined for zero-variance returns")
    return mean / math.sqrt(variance)


def sample_skewness(returns: Sequence[float]) -> float:
    n = len(returns)
    if n < 2:
        raise ValueError(f"sample_skewness needs >= 2 returns, got {n}")
    mean = math.fsum(returns) / n
    m2 = math.fsum((r - mean) ** 2 for r in returns) / n
    m3 = math.fsum((r - mean) ** 3 for r in returns) / n
    if m2 <= 0:
        raise ValueError("sample_skewness is undefined for zero-variance returns")
    return m3 / (m2 * math.sqrt(m2))  # m2 ** 1.5, but float**float types as Any


def sample_kurtosis(returns: Sequence[float]) -> float:
    n = len(returns)
    if n < 2:
        raise ValueError(f"sample_kurtosis needs >= 2 returns, got {n}")
    mean = math.fsum(returns) / n
    m2 = math.fsum((r - mean) ** 2 for r in returns) / n
    m4 = math.fsum((r - mean) ** 4 for r in returns) / n
    if m2 <= 0:
        raise ValueError("sample_kurtosis is undefined for zero-variance returns")
    return m4 / m2**2


def probabilistic_sharpe_ratio(
    observed_sr: float,
    *,
    sr_benchmark: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    if n_obs < 2:
        raise ValueError(f"n_obs must be >= 2, got {n_obs}")
    variance = 1.0 - skewness * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2
    if variance <= 0:
        raise ValueError(
            f"non-positive SR variance ({variance:.6f}) for skew={skewness}, "
            f"kurtosis={kurtosis}, sr={observed_sr}"
        )
    z = (observed_sr - sr_benchmark) * math.sqrt(n_obs - 1.0) / math.sqrt(variance)
    return _normal_cdf(z)


def expected_max_sharpe(n_trials: int, *, sr_variance: float) -> float:
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2 for an expected maximum, got {n_trials}")
    if sr_variance < 0:
        raise ValueError(f"sr_variance must be >= 0, got {sr_variance}")
    gamma = _EULER_MASCHERONI
    return math.sqrt(sr_variance) * (
        (1.0 - gamma) * _normal_ppf(1.0 - 1.0 / n_trials)
        + gamma * _normal_ppf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(
    observed_sr: float,
    *,
    n_trials: int,
    sr_variance: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    benchmark = expected_max_sharpe(n_trials, sr_variance=sr_variance)
    return probabilistic_sharpe_ratio(
        observed_sr,
        sr_benchmark=benchmark,
        n_obs=n_obs,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    from scipy.stats import norm

    return float(norm.ppf(p))
