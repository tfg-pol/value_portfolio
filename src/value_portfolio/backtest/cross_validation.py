from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True, slots=True)
class PurgedSplit:
    train: tuple[int, ...]
    test: tuple[int, ...]


def _validate(n_samples: int, label_span: int, embargo: int) -> None:
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    if label_span < 0:
        raise ValueError(f"label_span must be >= 0, got {label_span}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")


def _contiguous_blocks(indices: Sequence[int]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    for idx in indices:
        if blocks and idx == blocks[-1][1] + 1:
            blocks[-1] = (blocks[-1][0], idx)
        else:
            blocks.append((idx, idx))
    return blocks


def _train_indices(
    n_samples: int, test: Sequence[int], *, label_span: int, embargo: int
) -> tuple[int, ...]:
    test_set = set(test)
    blocks = _contiguous_blocks(sorted(test_set))
    train: list[int] = []
    for j in range(n_samples):
        if j in test_set:
            continue
        if all(j + label_span < a or j > b + label_span + embargo for a, b in blocks):
            train.append(j)
    return tuple(train)


def _contiguous_folds(n_samples: int, n_folds: int) -> list[tuple[int, ...]]:
    base, extra = divmod(n_samples, n_folds)
    folds: list[tuple[int, ...]] = []
    start = 0
    for k in range(n_folds):
        size = base + (1 if k < extra else 0)
        folds.append(tuple(range(start, start + size)))
        start += size
    return folds


def purged_kfold_splits(
    n_samples: int,
    *,
    n_splits: int = 5,
    label_span: int = 0,
    embargo: int = 0,
) -> list[PurgedSplit]:
    _validate(n_samples, label_span, embargo)
    if not 2 <= n_splits <= n_samples:
        raise ValueError(f"n_splits must be in [2, n_samples={n_samples}], got {n_splits}")
    folds = _contiguous_folds(n_samples, n_splits)
    return [
        PurgedSplit(
            train=_train_indices(n_samples, fold, label_span=label_span, embargo=embargo),
            test=fold,
        )
        for fold in folds
    ]


def combinatorial_purged_splits(
    n_samples: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    label_span: int = 0,
    embargo: int = 0,
) -> list[PurgedSplit]:
    """CPCV: test every choice of `n_test_groups` of `n_groups` contiguous groups, purged."""
    _validate(n_samples, label_span, embargo)
    if not 2 <= n_groups <= n_samples:
        raise ValueError(f"n_groups must be in [2, n_samples={n_samples}], got {n_groups}")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError(f"n_test_groups must be in [1, n_groups={n_groups}), got {n_test_groups}")
    groups = _contiguous_folds(n_samples, n_groups)
    splits: list[PurgedSplit] = []
    for chosen in combinations(range(n_groups), n_test_groups):
        test = tuple(sorted(idx for g in chosen for idx in groups[g]))
        splits.append(
            PurgedSplit(
                train=_train_indices(n_samples, test, label_span=label_span, embargo=embargo),
                test=test,
            )
        )
    return splits


def n_combinatorial_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of CPCV splits, ``C(n_groups, n_test_groups)``."""
    return math.comb(n_groups, n_test_groups)


class PurgedKFold:
    """scikit-learn-style splitter over `purged_kfold_splits`; splits by position, no shuffle."""

    def __init__(self, n_splits: int = 5, *, label_span: int = 0, embargo: int = 0) -> None:
        self.n_splits = n_splits
        self.label_span = label_span
        self.embargo = embargo

    def get_n_splits(self, *_args: object, **_kwargs: object) -> int:
        return self.n_splits

    def split(
        self, x: Sequence[object], y: object = None, groups: object = None
    ) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
        for split in purged_kfold_splits(
            len(x),
            n_splits=self.n_splits,
            label_span=self.label_span,
            embargo=self.embargo,
        ):
            yield split.train, split.test
