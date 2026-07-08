"""Tests for the stage-1 valuation model: signal recovery on a planted
relation, no-look-ahead (earlier scores are unchanged by later data), and
determinism (same config, same scores). Also covers the per-date rank
normalisation. Skipped without the ``learning`` extra.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

pytest.importorskip("sklearn")

from value_portfolio.learning.features import CrossSection, rank_normalize
from value_portfolio.learning.valuation import ValuationConfig, fit_predict_expanding


def _section(month: int, mispricing: dict[int, float] | None = None) -> CrossSection:
    """A cross-section whose log market cap is an exact linear function of one
    fundamental, minus a planted per-index mispricing (positive = cheap).
    """
    n = 8
    feature = (np.arange(n, dtype=np.float64) + month) % n  # rotate across months
    fair = 2.0 * (rank_normalize(feature.reshape(-1, 1))[:, 0])
    target = fair.copy()
    for index, amount in (mispricing or {}).items():
        target[index] -= amount
    return CrossSection(
        date=datetime(2020, month, 28, tzinfo=UTC),
        symbols=tuple(f"S{i}" for i in range(n)),
        features=feature.reshape(-1, 1),
        target=target - target.mean(),
    )


class TestRankNormalize:
    def test_maps_to_unit_interval_with_nan_as_median(self) -> None:
        column = np.array([[10.0], [np.nan], [30.0], [20.0]])
        normalized = rank_normalize(column)
        assert normalized[0, 0] == pytest.approx(-1.0)
        assert normalized[1, 0] == pytest.approx(0.0)  # missing -> median rank
        assert normalized[2, 0] == pytest.approx(1.0)
        assert normalized[3, 0] == pytest.approx(0.0)

    def test_uses_only_the_given_cross_section(self) -> None:
        a = rank_normalize(np.array([[1.0], [2.0]]))
        b = rank_normalize(np.array([[100.0], [200.0]]))
        assert np.array_equal(a, b)  # scale-free: only within-date order matters


class TestSignalRecovery:
    def test_ridge_scores_the_planted_cheap_name_highest(self) -> None:
        sections = [_section(m) for m in range(1, 6)]
        sections.append(_section(6, mispricing={3: 1.0}))  # S3 trades 1.0 below fair
        config = ValuationConfig(model="ridge", burn_in_sections=5, ridge_alpha=1e-6)

        records = fit_predict_expanding(sections, config)

        by_symbol = {rec.symbol: rec.score for rec in records}
        assert max(by_symbol, key=lambda s: by_symbol[s]) == "S3"
        assert by_symbol["S3"] > Decimal("0.5")

    def test_gbt_scores_the_planted_cheap_name_highest(self) -> None:
        sections = [_section(m) for m in range(1, 6)]
        sections.append(_section(6, mispricing={3: 1.0}))
        config = ValuationConfig(model="gbt", burn_in_sections=5, seed=0)

        records = fit_predict_expanding(sections, config)

        by_symbol = {rec.symbol: rec.score for rec in records}
        assert max(by_symbol, key=lambda s: by_symbol[s]) == "S3"


class TestNoLookAhead:
    def test_earlier_scores_unchanged_when_later_sections_are_appended(self) -> None:
        sections = [_section(m, mispricing={m % 8: 0.5}) for m in range(1, 9)]
        config = ValuationConfig(model="ridge", burn_in_sections=4)

        short = fit_predict_expanding(sections[:6], config)
        full = fit_predict_expanding(sections, config)

        assert full[: len(short)] == short


class TestDeterminism:
    @pytest.mark.parametrize("model", ["ridge", "gbt"])
    def test_same_config_same_scores(self, model: str) -> None:
        sections = [_section(m, mispricing={m % 8: 0.3}) for m in range(1, 8)]
        config = ValuationConfig(model=model, burn_in_sections=4, seed=7)  # type: ignore[arg-type]

        assert fit_predict_expanding(sections, config) == fit_predict_expanding(sections, config)


class TestScoreCalibration:
    """Scores are demeaned within each scored date: magnitudes are honest
    valuation errors, and the demeaning (a per-date constant) must not move
    any within-date ranking.
    """

    @pytest.mark.parametrize("model", ["ridge", "gbt"])
    def test_scores_have_zero_mean_per_date(self, model: str) -> None:
        sections = [_section(m, mispricing={m % 8: 0.4}) for m in range(1, 8)]
        config = ValuationConfig(model=model, burn_in_sections=4, seed=0)  # type: ignore[arg-type]

        records = fit_predict_expanding(sections, config)

        by_date: dict[object, list[float]] = {}
        for rec in records:
            by_date.setdefault(rec.date, []).append(float(rec.score))
        assert by_date
        for scores in by_date.values():
            assert sum(scores) / len(scores) == pytest.approx(0.0, abs=1e-5)

    def test_planted_cheap_name_still_ranks_first(self) -> None:
        # The calibration is a per-date constant: the mispriced name must
        # still carry the highest score after it.
        sections = [_section(m) for m in range(1, 6)]
        sections.append(_section(6, mispricing={3: 1.0}))
        config = ValuationConfig(model="ridge", burn_in_sections=5, ridge_alpha=1e-6)

        records = fit_predict_expanding(sections, config)

        by_symbol = {rec.symbol: rec.score for rec in records}
        assert max(by_symbol, key=lambda s: by_symbol[s]) == "S3"

    def test_mb_target_config_demeans_and_preserves_ranking(self) -> None:
        # The mb target only changes the label and (for the GBT) the design
        # matrix; the residual demeaning and within-date ranking are invariant.
        # Ridge's design matrix is rank-normalised, so the synthetic one-feature
        # panel exercises the target="mb" config path directly.
        sections = [_section(m) for m in range(1, 6)]
        sections.append(_section(6, mispricing={3: 1.0}))
        config = ValuationConfig(model="ridge", burn_in_sections=5, ridge_alpha=1e-6, target="mb")

        records = fit_predict_expanding(sections, config)

        by_symbol = {rec.symbol: rec.score for rec in records}
        assert sum(float(rec.score) for rec in records) / len(records) == pytest.approx(
            0.0, abs=1e-5
        )
        assert max(by_symbol, key=lambda s: by_symbol[s]) == "S3"


class TestScaleFeaturesOverride:
    """`scale_features` decouples GBT asset-scaling from the target so the
    two changes can be ablated independently (auto = scale iff mb).
    """

    def test_gbt_mb_runs_unscaled_when_override_is_false(self) -> None:
        # With auto-scaling, a GBT mb fit would call scale_levels_by_assets,
        # which needs the full feature layout and would fail on this synthetic
        # one-feature panel. scale_features=False must take the raw-feature path.
        sections = [_section(m, mispricing={m % 8: 0.3}) for m in range(1, 8)]
        config = ValuationConfig(
            model="gbt", burn_in_sections=4, target="mb", scale_features=False, seed=0
        )

        records = fit_predict_expanding(sections, config)

        assert records
        by_date: dict[object, list[float]] = {}
        for rec in records:
            by_date.setdefault(rec.date, []).append(float(rec.score))
        for scores in by_date.values():
            assert sum(scores) / len(scores) == pytest.approx(0.0, abs=1e-5)


class TestForwardReturnTarget:
    """The ``ret`` target regresses the t -> t+1 return directly; the score is the
    prediction, and the realised return of the scored date must never be read.
    """

    def test_score_ignores_test_target_no_lookahead(self) -> None:
        sections = [_section(m) for m in range(1, 7)]
        config = ValuationConfig(model="ridge", burn_in_sections=4, target="ret", ridge_alpha=1e-6)
        base = fit_predict_expanding(sections, config)
        # The realised forward return of the *scored* date is known only at t+1.
        # Corrupt only the last (test-only) section's target; if scoring read it,
        # the last date's scores would move. They must not.
        bumped = [*sections[:-1], replace(sections[-1], target=sections[-1].target + 99.0)]
        assert fit_predict_expanding(bumped, config) == base

    def test_nan_labels_are_masked_from_training_but_still_scored(self) -> None:
        # A name with a NaN forward return (delists next month) must not break the
        # fit — it is dropped from training — yet every name is still scored at t.
        sections = [_section(m) for m in range(1, 7)]
        labels = sections[2].target.copy()
        labels[0] = np.nan  # one un-trainable name in a training section
        sections[2] = replace(sections[2], target=labels)
        config = ValuationConfig(model="ridge", burn_in_sections=4, target="ret", ridge_alpha=1e-6)

        records = fit_predict_expanding(sections, config)

        assert records
        per_date: dict[object, int] = {}
        for rec in records:
            per_date[rec.date] = per_date.get(rec.date, 0) + 1
        assert all(count == 8 for count in per_date.values())  # all names scored

    def test_placebo_shuffles_deterministically_and_changes_scores(self) -> None:
        # shuffle_labels permutes labels within date (leak test); it must be
        # deterministic given the seed and must change the scores vs unshuffled.
        sections = [_section(m, mispricing={m % 8: 0.5}) for m in range(1, 8)]
        base = ValuationConfig(model="ridge", burn_in_sections=4, target="ret", ridge_alpha=1e-6)
        placebo = replace(base, shuffle_labels=True, seed=3)

        real = fit_predict_expanding(sections, base)
        p1 = fit_predict_expanding(sections, placebo)
        p2 = fit_predict_expanding(sections, placebo)
        assert p1 == p2  # deterministic given the seed
        assert p1 != real  # shuffling actually changed the fit

    @pytest.mark.parametrize("model", ["ridge", "gbt"])
    def test_ret_scores_zero_mean_per_date(self, model: str) -> None:
        sections = [_section(m) for m in range(1, 7)]
        config = ValuationConfig(model=model, burn_in_sections=4, target="ret", seed=0)  # type: ignore[arg-type]

        records = fit_predict_expanding(sections, config)

        assert records
        by_date: dict[object, list[float]] = {}
        for rec in records:
            by_date.setdefault(rec.date, []).append(float(rec.score))
        for scores in by_date.values():
            assert sum(scores) / len(scores) == pytest.approx(0.0, abs=1e-5)


class TestIndustryFeature:
    """`industry=True` appends the section's sector code as a categorical column
    for the GBT (the last feature), priced against peers rather than the market.
    """

    def test_gbt_runs_with_industry_categorical(self) -> None:
        # Attach a two-sector code (alternating) to each synthetic section and
        # confirm the GBT trains with the trailing categorical column and still
        # returns per-date-demeaned scores.
        codes = np.array([i % 2 for i in range(8)], dtype=np.float64)
        sections = [
            replace(_section(m, mispricing={m % 8: 0.3}), industry=codes) for m in range(1, 8)
        ]
        config = ValuationConfig(model="gbt", burn_in_sections=4, industry=True, seed=0)

        records = fit_predict_expanding(sections, config)

        assert records
        by_date: dict[object, list[float]] = {}
        for rec in records:
            by_date.setdefault(rec.date, []).append(float(rec.score))
        for scores in by_date.values():
            assert sum(scores) / len(scores) == pytest.approx(0.0, abs=1e-5)
