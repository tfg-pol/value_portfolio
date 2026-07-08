"""Tests for purged / combinatorial cross-validation with embargo: train and
test never overlap, purged/embargoed neighbourhoods are removed, split counts
match the combinatorics, and everything is deterministic.
"""

from __future__ import annotations

import math

import pytest

from value_portfolio.backtest import (
    PurgedKFold,
    combinatorial_purged_splits,
    n_combinatorial_paths,
    purged_kfold_splits,
)
from value_portfolio.backtest.cross_validation import _contiguous_blocks


class TestPurgedKFoldSplits:
    def test_folds_partition_the_test_sets_in_time_order(self) -> None:
        splits = purged_kfold_splits(10, n_splits=5)
        test_sets = [s.test for s in splits]
        # Contiguous, ordered, and together they cover every index exactly once.
        assert test_sets == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    def test_uneven_folds_put_the_remainder_first(self) -> None:
        # 11 into 3 -> sizes 4, 4, 3.
        splits = purged_kfold_splits(11, n_splits=3)
        assert [len(s.test) for s in splits] == [4, 4, 3]

    def test_train_and_test_never_overlap(self) -> None:
        for split in purged_kfold_splits(20, n_splits=4, label_span=2, embargo=1):
            assert set(split.train).isdisjoint(split.test)

    def test_purge_removes_label_overlap_on_the_left(self) -> None:
        # Test fold [4..7] with label_span 2: a left-side train index j is kept
        # only if j + 2 < 4, i.e. j <= 1. So 2 and 3 are purged.
        split = next(s for s in purged_kfold_splits(12, n_splits=3, label_span=2) if 4 in s.test)
        assert split.test == (4, 5, 6, 7)
        assert 1 in split.train and 2 not in split.train and 3 not in split.train

    def test_embargo_extends_the_post_test_gap(self) -> None:
        # Test fold [4..7], label_span 1, embargo 2: right-side kept iff
        # j > 7 + 1 + 2 = 10, so 8, 9, 10 are embargoed, 11 survives.
        split = next(
            s for s in purged_kfold_splits(12, n_splits=3, label_span=1, embargo=2) if 4 in s.test
        )
        assert all(j not in split.train for j in (8, 9, 10))
        assert 11 in split.train

    def test_no_purge_no_embargo_uses_all_other_samples(self) -> None:
        split = purged_kfold_splits(10, n_splits=5)[2]
        assert sorted(split.train + split.test) == list(range(10))

    def test_deterministic(self) -> None:
        a = purged_kfold_splits(15, n_splits=3, label_span=1, embargo=1)
        b = purged_kfold_splits(15, n_splits=3, label_span=1, embargo=1)
        assert a == b

    @pytest.mark.parametrize("n_splits", [1, 0, 11])
    def test_invalid_n_splits_rejected(self, n_splits: int) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            purged_kfold_splits(10, n_splits=n_splits)

    def test_negative_label_span_or_embargo_rejected(self) -> None:
        with pytest.raises(ValueError, match="label_span"):
            purged_kfold_splits(10, n_splits=2, label_span=-1)
        with pytest.raises(ValueError, match="embargo"):
            purged_kfold_splits(10, n_splits=2, embargo=-1)


class TestCombinatorialPurgedSplits:
    def test_split_count_is_n_choose_k(self) -> None:
        splits = combinatorial_purged_splits(12, n_groups=6, n_test_groups=2)
        assert len(splits) == math.comb(6, 2) == 15
        assert n_combinatorial_paths(6, 2) == 15

    def test_test_sets_can_be_non_contiguous(self) -> None:
        # Groups of width 2 over 12 samples; choosing groups 0 and 2 gives the
        # disjoint test set {0,1} + {4,5}.
        splits = combinatorial_purged_splits(12, n_groups=6, n_test_groups=2)
        disjoint = next(s for s in splits if s.test == (0, 1, 4, 5))
        assert _contiguous_blocks(disjoint.test) == [(0, 1), (4, 5)]

    def test_train_disjoint_from_test_with_purge_and_embargo(self) -> None:
        for split in combinatorial_purged_splits(
            24, n_groups=6, n_test_groups=2, label_span=1, embargo=1
        ):
            assert set(split.train).isdisjoint(split.test)

    def test_purge_applies_around_every_test_block(self) -> None:
        # Non-contiguous test {0,1} + {4,5}; with label_span 1 the index 3
        # (3+1 = 4 overlaps the second block's start) must be purged.
        split = next(
            s
            for s in combinatorial_purged_splits(12, n_groups=6, n_test_groups=2, label_span=1)
            if s.test == (0, 1, 4, 5)
        )
        assert 3 not in split.train

    def test_invalid_group_parameters_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_groups"):
            combinatorial_purged_splits(10, n_groups=1, n_test_groups=1)
        with pytest.raises(ValueError, match="n_test_groups"):
            combinatorial_purged_splits(10, n_groups=4, n_test_groups=4)


class TestPurgedKFoldClass:
    def test_split_matches_the_functional_core(self) -> None:
        splitter = PurgedKFold(n_splits=4, label_span=2, embargo=1)
        data = list(range(20))
        from_class = list(splitter.split(data))
        from_func = [
            (s.train, s.test) for s in purged_kfold_splits(20, n_splits=4, label_span=2, embargo=1)
        ]
        assert from_class == from_func
        assert splitter.get_n_splits() == 4
