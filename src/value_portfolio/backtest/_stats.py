
from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

_ZERO = Decimal("0")


def mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, _ZERO) / Decimal(len(values))


def median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / Decimal(2)


def sample_std(values: Sequence[Decimal]) -> Decimal:
    if len(values) < 2:
        return _ZERO
    m = mean(values)
    variance = sum(((v - m) ** 2 for v in values), _ZERO) / Decimal(len(values) - 1)
    if variance <= _ZERO:
        return _ZERO
    return variance.sqrt()


def sample_cov(xs: Sequence[Decimal], ys: Sequence[Decimal]) -> Decimal:
    n = Decimal(len(xs))
    mean_x = sum(xs, _ZERO) / n
    mean_y = sum(ys, _ZERO) / n
    return sum(((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)), _ZERO) / (
        n - Decimal(1)
    )
