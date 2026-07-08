"""Tests for the stage-1 feature catalog: derived health ratios, year-over-year
trajectory features, point-in-time lagged reads, division guards, and the
cross-section eligibility rule. Skipped without the ``learning`` extra.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

import pytest

np = pytest.importorskip("numpy")

from value_portfolio.data import FundamentalRecord, InMemoryFundamentals, InMemoryUniverse
from value_portfolio.learning._asof import AsOfSeries
from value_portfolio.learning.features import (
    FEATURE_NAMES,
    REQUIRED_FIELDS,
    DailyMarketCap,
    build_cross_sections,
    compute_feature_row,
    scale_levels_by_assets,
)

_NOW = datetime(2021, 6, 30, tzinfo=UTC)
_LAST_YEAR = datetime(2020, 6, 1, tzinfo=UTC)


def _source(
    current: dict[str, str], previous: dict[str, str] | None = None
) -> InMemoryFundamentals:
    records = [
        FundamentalRecord(
            symbol="AAA",
            dimension="ART",
            datekey=_NOW,
            values={k: Decimal(v) for k, v in current.items()},
        )
    ]
    if previous is not None:
        records.append(
            FundamentalRecord(
                symbol="AAA",
                dimension="ART",
                datekey=_LAST_YEAR,
                values={k: Decimal(v) for k, v in previous.items()},
            )
        )
    return InMemoryFundamentals(records)


def _feature(row: list[float], name: str) -> float:
    return row[FEATURE_NAMES.index(name)]


class TestDerivedFeatures:
    def test_health_ratios(self) -> None:
        src = _source({"gp": "40", "assets": "200", "netinc": "10", "ncfo": "25", "debt": "50"})
        row = compute_feature_row(src, "AAA", _NOW)
        assert _feature(row, "gross_profitability") == pytest.approx(0.2)  # 40/200
        assert _feature(row, "accruals_to_assets") == pytest.approx(-0.075)  # (10-25)/200
        assert _feature(row, "debt_to_assets") == pytest.approx(0.25)

    def test_division_guards_yield_nan(self) -> None:
        # assets missing, intexp zero -> derived ratios undefined, not inf/crash.
        src = _source({"gp": "40", "ebit": "10", "intexp": "0"})
        row = compute_feature_row(src, "AAA", _NOW)
        assert math.isnan(_feature(row, "gross_profitability"))
        assert math.isnan(_feature(row, "interest_coverage"))

    def test_unreported_fields_are_nan(self) -> None:
        src = _source({"revenue": "100"})
        row = compute_feature_row(src, "AAA", _NOW)
        assert _feature(row, "revenue") == pytest.approx(100.0)
        assert math.isnan(_feature(row, "roe"))

    def test_row_matches_catalog_length(self) -> None:
        src = _source({"revenue": "100"})
        assert len(compute_feature_row(src, "AAA", _NOW)) == len(FEATURE_NAMES)


class TestTrajectoryFeatures:
    def test_yoy_deltas_use_the_lagged_filing(self) -> None:
        src = _source(
            {"roa": "0.12", "revenue": "110", "eps": "5"},
            previous={"roa": "0.10", "revenue": "100", "eps": "4"},
        )
        row = compute_feature_row(src, "AAA", _NOW)
        assert _feature(row, "delta_roa") == pytest.approx(0.02)
        assert _feature(row, "revenue_growth") == pytest.approx(0.10)
        assert _feature(row, "eps_change") == pytest.approx(1.0)

    def test_deltas_nan_without_a_year_of_history(self) -> None:
        src = _source({"roa": "0.12", "revenue": "110"})
        row = compute_feature_row(src, "AAA", _NOW)
        assert math.isnan(_feature(row, "delta_roa"))
        assert math.isnan(_feature(row, "revenue_growth"))

    def test_lagged_read_is_point_in_time(self) -> None:
        # The "previous" filing is dated only 3 months back: at the one-year
        # lag date it did not exist yet, so the delta must be undefined.
        late_prev = datetime(2021, 3, 31, tzinfo=UTC)
        src = InMemoryFundamentals(
            [
                FundamentalRecord(
                    symbol="AAA", dimension="ART", datekey=late_prev, values={"roa": Decimal("0.1")}
                ),
                FundamentalRecord(
                    symbol="AAA", dimension="ART", datekey=_NOW, values={"roa": Decimal("0.12")}
                ),
            ]
        )
        row = compute_feature_row(src, "AAA", _NOW)
        assert math.isnan(_feature(row, "delta_roa"))


class TestBuildCrossSections:
    def test_sparse_names_are_excluded_by_min_features(self) -> None:
        rich = {field: "1" for field in REQUIRED_FIELDS}
        fundamentals = InMemoryFundamentals(
            [
                FundamentalRecord(
                    symbol="RICH",
                    dimension="ART",
                    datekey=_LAST_YEAR,
                    values={k: Decimal(v) for k, v in rich.items()},
                ),
                FundamentalRecord(
                    symbol="POOR",
                    dimension="ART",
                    datekey=_LAST_YEAR,
                    values={"revenue": Decimal("5")},
                ),
            ]
        )
        universe = InMemoryUniverse({"RICH": [(_LAST_YEAR, None)], "POOR": [(_LAST_YEAR, None)]})
        day = np.datetime64(_NOW.replace(tzinfo=None), "ns").astype(np.int64)
        caps = DailyMarketCap(
            {
                "RICH": (np.array([day]), np.array([1000.0])),
                "POOR": (np.array([day]), np.array([1000.0])),
            }
        )

        sections = build_cross_sections(
            fundamentals, universe, caps, [_NOW], min_names=1, min_features=10
        )

        assert len(sections) == 1
        assert sections[0].symbols == ("RICH",)
        assert sections[0].features.shape == (1, len(FEATURE_NAMES))


def _rich_panel(
    value_by_symbol: dict[str, str], field: str = "equity"
) -> tuple[InMemoryFundamentals, InMemoryUniverse]:
    """A fundamentals/universe pair where every named symbol reports the full
    `REQUIRED_FIELDS` (so it clears `min_features`), with one `field` overridden
    per symbol — the deflator a scale-free target reads (``equity`` for mb,
    ``assets`` for ma).
    """
    records = []
    membership = {}
    for symbol, value in value_by_symbol.items():
        values = {f: Decimal("1") for f in REQUIRED_FIELDS}
        values[field] = Decimal(value)
        records.append(
            FundamentalRecord(symbol=symbol, dimension="ART", datekey=_LAST_YEAR, values=values)
        )
        membership[symbol] = [(_LAST_YEAR, None)]
    return InMemoryFundamentals(records), InMemoryUniverse(membership)


def _caps(cap_by_symbol: dict[str, float]) -> DailyMarketCap:
    day = np.datetime64(_NOW.replace(tzinfo=None), "ns").astype(np.int64)
    return DailyMarketCap(
        {s: (np.array([day]), np.array([cap])) for s, cap in cap_by_symbol.items()}
    )


class TestMarketToBookTarget:
    def test_target_is_demeaned_log_market_to_book(self) -> None:
        fundamentals, universe = _rich_panel({"AAA": "100", "BBB": "200"})
        caps = _caps({"AAA": 1000.0, "BBB": 4000.0})

        sections = build_cross_sections(
            fundamentals, universe, caps, [_NOW], target="mb", min_names=1, min_features=10
        )

        assert len(sections) == 1
        section = sections[0]
        assert section.symbols == ("AAA", "BBB")  # sorted
        raw = np.array([math.log(1000.0) - math.log(100.0), math.log(4000.0) - math.log(200.0)])
        expected = raw - raw.mean()
        assert section.target == pytest.approx(expected)
        assert float(section.target.mean()) == pytest.approx(0.0, abs=1e-12)

    def test_non_positive_book_is_excluded(self) -> None:
        fundamentals, universe = _rich_panel({"GOOD": "100", "ZERO": "0", "NEG": "-50"})
        caps = _caps({"GOOD": 1000.0, "ZERO": 1000.0, "NEG": 1000.0})

        sections = build_cross_sections(
            fundamentals, universe, caps, [_NOW], target="mb", min_names=1, min_features=10
        )

        assert len(sections) == 1
        assert sections[0].symbols == ("GOOD",)  # zero and negative book dropped

    def test_cap_target_keeps_non_positive_book_names(self) -> None:
        # The book-equity filter is mb-only; the cap target prices every name.
        fundamentals, universe = _rich_panel({"GOOD": "100", "NEG": "-50"})
        caps = _caps({"GOOD": 1000.0, "NEG": 1000.0})

        sections = build_cross_sections(
            fundamentals, universe, caps, [_NOW], target="cap", min_names=1, min_features=10
        )

        assert sections[0].symbols == ("GOOD", "NEG")


class TestMarketToAssetsTarget:
    def test_target_is_demeaned_log_market_to_assets(self) -> None:
        fundamentals, universe = _rich_panel({"AAA": "500", "BBB": "2000"}, field="assets")
        caps = _caps({"AAA": 1000.0, "BBB": 1000.0})

        sections = build_cross_sections(
            fundamentals, universe, caps, [_NOW], target="ma", min_names=1, min_features=10
        )

        assert len(sections) == 1
        section = sections[0]
        assert section.symbols == ("AAA", "BBB")
        raw = np.array([math.log(1000.0) - math.log(500.0), math.log(1000.0) - math.log(2000.0)])
        assert section.target == pytest.approx(raw - raw.mean())
        assert float(section.target.mean()) == pytest.approx(0.0, abs=1e-12)

    def test_non_positive_assets_is_excluded(self) -> None:
        fundamentals, universe = _rich_panel({"GOOD": "500", "ZERO": "0"}, field="assets")
        caps = _caps({"GOOD": 1000.0, "ZERO": 1000.0})

        sections = build_cross_sections(
            fundamentals, universe, caps, [_NOW], target="ma", min_names=1, min_features=10
        )

        assert sections[0].symbols == ("GOOD",)


_NEXT = datetime(2021, 7, 31, tzinfo=UTC)


def _prices(points_by_symbol: dict[str, list[tuple[datetime, float]]]) -> AsOfSeries:
    series = {}
    for symbol, points in points_by_symbol.items():
        dates = np.array(
            [np.datetime64(d.replace(tzinfo=None), "ns").astype(np.int64) for d, _ in points]
        )
        values = np.array([p for _, p in points], dtype=np.float64)
        series[symbol] = (dates, values)
    return AsOfSeries(series)


class TestForwardReturnTarget:
    def test_target_is_demeaned_forward_return(self) -> None:
        fundamentals, universe = _rich_panel({"AAA": "1", "BBB": "1"})
        caps = _caps({"AAA": 1000.0, "BBB": 1000.0})
        prices = _prices(
            {
                "AAA": [(_NOW, 100.0), (_NEXT, 110.0)],  # +10%
                "BBB": [(_NOW, 100.0), (_NEXT, 90.0)],  # -10%
            }
        )
        sections = build_cross_sections(
            fundamentals,
            universe,
            caps,
            [_NOW, _NEXT],
            target="ret",
            prices=prices,
            min_names=1,
            min_features=10,
        )
        # The last date (_NEXT) has no next month, so only _NOW is scored.
        assert len(sections) == 1
        section = sections[0]
        assert section.date == _NOW
        assert section.symbols == ("AAA", "BBB")
        raw = np.array([0.10, -0.10])
        assert section.target == pytest.approx(raw - raw.mean())

    def test_name_without_a_forward_price_is_kept_with_nan_label(self) -> None:
        # No survivorship look-ahead: a name that delists before t+1 is still
        # *scored* at t (kept in the panel with a NaN label, which training masks).
        fundamentals, universe = _rich_panel({"AAA": "1", "GONE": "1"})
        caps = _caps({"AAA": 1000.0, "GONE": 1000.0})
        prices = _prices(
            {
                "AAA": [(_NOW, 100.0), (_NEXT, 110.0)],
                "GONE": [(_NOW, 100.0)],  # delists before _NEXT: no realised return
            }
        )
        sections = build_cross_sections(
            fundamentals,
            universe,
            caps,
            [_NOW, _NEXT],
            target="ret",
            prices=prices,
            min_names=1,
            min_features=10,
        )
        section = sections[0]
        assert section.symbols == ("AAA", "GONE")  # both kept, no survivorship drop
        label = dict(zip(section.symbols, section.target, strict=True))
        assert not np.isnan(label["AAA"])  # has a realised forward return
        assert np.isnan(label["GONE"])  # no return -> NaN (masked from training)

    def test_ret_target_requires_prices(self) -> None:
        fundamentals, universe = _rich_panel({"AAA": "1"})
        caps = _caps({"AAA": 1000.0})
        with pytest.raises(ValueError, match="forward-return"):
            build_cross_sections(
                fundamentals, universe, caps, [_NOW, _NEXT], target="ret", min_names=1
            )


class TestIndustryMap:
    def test_codes_attached_in_symbol_order(self) -> None:
        fundamentals, universe = _rich_panel({"AAA": "1", "BBB": "1"})
        caps = _caps({"AAA": 1000.0, "BBB": 1000.0})

        sections = build_cross_sections(
            fundamentals,
            universe,
            caps,
            [_NOW],
            industry_map={"AAA": 3, "CCC": 7},
            min_names=1,
            min_features=10,
        )

        section = sections[0]
        assert section.symbols == ("AAA", "BBB")
        assert section.industry is not None
        assert section.industry[0] == pytest.approx(3.0)  # AAA mapped
        assert math.isnan(section.industry[1])  # BBB unmapped -> NaN (missing category)

    def test_industry_is_none_without_a_map(self) -> None:
        fundamentals, universe = _rich_panel({"AAA": "1", "BBB": "1"})
        caps = _caps({"AAA": 1000.0, "BBB": 1000.0})

        sections = build_cross_sections(
            fundamentals, universe, caps, [_NOW], min_names=1, min_features=10
        )

        assert sections[0].industry is None


class TestScaleLevelsByAssets:
    def _matrix(self, rows: list[dict[str, float]]) -> np.ndarray:
        matrix = np.full((len(rows), len(FEATURE_NAMES)), np.nan)
        for i, row in enumerate(rows):
            for name, value in row.items():
                matrix[i, FEATURE_NAMES.index(name)] = value
        return matrix

    def test_dollar_levels_become_ratios_to_assets(self) -> None:
        matrix = self._matrix(
            [{"assets": 200.0, "revenue": 40.0, "gp": 80.0, "eps": 5.0, "roe": 0.1}]
        )
        scaled = scale_levels_by_assets(matrix)

        assert scaled[0, FEATURE_NAMES.index("revenue")] == pytest.approx(0.2)  # 40/200
        assert scaled[0, FEATURE_NAMES.index("gp")] == pytest.approx(0.4)  # 80/200
        assert scaled[0, FEATURE_NAMES.index("assets")] == pytest.approx(math.log(200.0))
        # per-share and ratio columns pass through untouched
        assert scaled[0, FEATURE_NAMES.index("eps")] == pytest.approx(5.0)
        assert scaled[0, FEATURE_NAMES.index("roe")] == pytest.approx(0.1)

    def test_non_positive_assets_yield_nan(self) -> None:
        matrix = self._matrix([{"assets": 0.0, "revenue": 10.0, "gp": 20.0}])
        scaled = scale_levels_by_assets(matrix)

        assert math.isnan(scaled[0, FEATURE_NAMES.index("revenue")])
        assert math.isnan(scaled[0, FEATURE_NAMES.index("gp")])
        assert math.isnan(scaled[0, FEATURE_NAMES.index("assets")])

    def test_does_not_mutate_input(self) -> None:
        matrix = self._matrix([{"assets": 200.0, "revenue": 40.0}])
        before = matrix.copy()
        scale_levels_by_assets(matrix)
        assert np.allclose(matrix, before, equal_nan=True)
