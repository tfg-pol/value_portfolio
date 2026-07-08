"""Unit tests for the Sharadar file loaders. Hermetic: each test synthesizes a
tiny Parquet/CSV slice in ``tmp_path``, so the suite never depends on the real
``data/sharadar/`` dataset. Covers total-return price scaling, membership parsing
(blank end + multi-spell), and sparse fundamentals loading.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from value_portfolio.data import InMemoryFundamentals, InMemoryMarketData, InMemoryUniverse
from value_portfolio.data.sharadar import (
    load_bars_from_sharadar,
    load_fundamentals_from_sharadar,
    load_universe_from_sharadar,
)


def _write_sep(path: Path) -> None:
    # AAA: raw split-adjusted close 100 -> 110; closeadj scaled down for dividends
    # (factor 0.5 on day 1, 0.6 on day 2); open offset 1 below close.
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "date": [pd.Timestamp("2010-01-04"), pd.Timestamp("2010-01-05")],
            "open": [99.0, 109.0],
            "high": [101.0, 111.0],
            "low": [98.0, 108.0],
            "close": [100.0, 110.0],
            "closeadj": [50.0, 66.0],
            "closeunadj": [200.0, 220.0],
            "volume": [1000.0, 1200.0],
        }
    )
    frame.to_parquet(path, index=False)


class TestLoadBars:
    def test_total_return_scaling_is_default(self, tmp_path: Path) -> None:
        sep = tmp_path / "sep.parquet"
        _write_sep(sep)

        data = load_bars_from_sharadar(
            ["AAA"],
            datetime(2010, 1, 1, tzinfo=UTC),
            datetime(2010, 1, 31, tzinfo=UTC),
            path=sep,
        )
        assert isinstance(data, InMemoryMarketData)
        # advance the clock so all bars are visible
        data.advance_to(datetime(2010, 1, 31, tzinfo=UTC))
        bars = data.get_bars(
            "AAA",
            datetime(2010, 1, 1, tzinfo=UTC),
            datetime(2010, 1, 31, tzinfo=UTC),
            "1Day",
        )

        # close == closeadj (total-return basis); open scaled by closeadj/close.
        assert [b.close for b in bars] == [_dec("50"), _dec("66")]
        # day 1 factor 0.5 -> open 99*0.5 = 49.5 ; day 2 factor 0.6 -> 109*0.6 = 65.4
        assert [b.open for b in bars] == [_dec("49.5"), _dec("65.4")]
        # timestamps are tz-aware UTC
        assert all(b.timestamp.tzinfo is UTC for b in bars)

    def test_price_close_keeps_raw_split_adjusted(self, tmp_path: Path) -> None:
        sep = tmp_path / "sep.parquet"
        _write_sep(sep)
        data = load_bars_from_sharadar(
            ["AAA"],
            datetime(2010, 1, 1, tzinfo=UTC),
            datetime(2010, 1, 31, tzinfo=UTC),
            price="close",
            path=sep,
        )
        data.advance_to(datetime(2010, 1, 31, tzinfo=UTC))
        bars = data.get_bars(
            "AAA",
            datetime(2010, 1, 1, tzinfo=UTC),
            datetime(2010, 1, 31, tzinfo=UTC),
            "1Day",
        )
        assert [b.close for b in bars] == [_dec("100"), _dec("110")]
        assert [b.open for b in bars] == [_dec("99"), _dec("109")]

    def test_empty_window_raises(self, tmp_path: Path) -> None:
        sep = tmp_path / "sep.parquet"
        _write_sep(sep)
        with pytest.raises(ValueError, match="no bars"):
            load_bars_from_sharadar(
                ["AAA"],
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 12, 31, tzinfo=UTC),
                path=sep,
            )

    def test_no_symbols_raises(self, tmp_path: Path) -> None:
        sep = tmp_path / "sep.parquet"
        _write_sep(sep)
        with pytest.raises(ValueError, match="at least one symbol"):
            load_bars_from_sharadar(
                [], datetime(2010, 1, 1, tzinfo=UTC), datetime(2010, 1, 31, tzinfo=UTC), path=sep
            )


class TestLoadUniverse:
    def test_parses_blank_end_and_multiple_spells(self, tmp_path: Path) -> None:
        membership = tmp_path / "membership.csv"
        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "BBB"],
                "start_date": ["2005-01-01", "2008-01-01", "2015-01-01"],
                "end_date": ["", "2012-01-01", ""],  # AAA still in; BBB left then rejoined
            }
        ).to_csv(membership, index=False)

        universe = load_universe_from_sharadar(membership)
        assert isinstance(universe, InMemoryUniverse)
        assert universe.members_at(datetime(2006, 1, 1, tzinfo=UTC)) == {"AAA"}
        assert universe.members_at(datetime(2009, 1, 1, tzinfo=UTC)) == {"AAA", "BBB"}
        assert universe.members_at(datetime(2013, 1, 1, tzinfo=UTC)) == {"AAA"}  # BBB out
        assert universe.members_at(datetime(2016, 1, 1, tzinfo=UTC)) == {"AAA", "BBB"}  # BBB back


class TestLoadFundamentals:
    def test_loads_sparse_and_point_in_time(self, tmp_path: Path) -> None:
        sf1 = tmp_path / "sf1.parquet"
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "dimension": ["ART", "ART"],
                "datekey": [pd.Timestamp("2020-02-15"), pd.Timestamp("2020-05-15")],
                "calendardate": [pd.Timestamp("2019-12-31"), pd.Timestamp("2020-03-31")],
                "lastupdated": [pd.Timestamp("2020-02-16"), pd.Timestamp("2020-05-16")],
                "revenue": [100.0, 110.0],
                "netinc": [float("nan"), 20.0],  # first filing missing netinc
            }
        ).to_parquet(sf1, index=False)

        src = load_fundamentals_from_sharadar(["AAA"], fields=["revenue", "netinc"], path=sf1)
        assert isinstance(src, InMemoryFundamentals)
        # point-in-time: between the two filings the first is the latest known.
        assert src.value("AAA", "revenue", datetime(2020, 3, 1, tzinfo=UTC)) == _dec("100")
        # sparse: the first filing dropped the NaN netinc entirely.
        assert src.value("AAA", "netinc", datetime(2020, 3, 1, tzinfo=UTC)) is None
        # after the second filing both fields are present.
        assert src.value("AAA", "netinc", datetime(2020, 6, 1, tzinfo=UTC)) == _dec("20")

    def test_unknown_fields_rejected(self, tmp_path: Path) -> None:
        sf1 = tmp_path / "sf1.parquet"
        pd.DataFrame(
            {
                "ticker": ["AAA"],
                "dimension": ["ART"],
                "datekey": [pd.Timestamp("2020-02-15")],
                "calendardate": [pd.Timestamp("2019-12-31")],
                "lastupdated": [pd.Timestamp("2020-02-16")],
                "revenue": [100.0],
            }
        ).to_parquet(sf1, index=False)
        with pytest.raises(ValueError, match="none of the requested"):
            load_fundamentals_from_sharadar(["AAA"], fields=["does_not_exist"], path=sf1)


def _dec(value: str) -> Decimal:
    return Decimal(value)
