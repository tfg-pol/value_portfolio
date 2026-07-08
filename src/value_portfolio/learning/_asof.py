
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np


class AsOfSeries:

    def __init__(self, series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self.series = series

    @classmethod
    def from_parquet(
        cls, path: Path | str, value_column: str, *, symbols: Sequence[str] | None = None
    ) -> AsOfSeries:
        """Load ``ticker, date, <value_column>`` from a Sharadar Parquet, grouped
        by ticker and sorted by date (rows with a null value dropped).
        """
        import pandas as pd

        frame = pd.read_parquet(path, columns=["ticker", "date", value_column])
        if symbols is not None:
            frame = frame[frame["ticker"].isin(set(symbols))]
        frame = frame.dropna(subset=[value_column]).sort_values(["ticker", "date"])

        series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ticker, group in frame.groupby("ticker", sort=False):
            series[str(ticker)] = (
                group["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
                group[value_column].to_numpy(dtype=np.float64),
            )
        return cls(series)

    def lookup(self, symbol: str, as_of: datetime) -> tuple[int, float] | None:
        """``(timestamp_ns, value)`` of the most recent observation at or before
        `as_of`, or ``None`` if the symbol is absent or has no prior observation.
        """
        series = self.series.get(symbol)
        if series is None:
            return None
        dates, values = series
        stamp = np.datetime64(as_of.replace(tzinfo=None), "ns").astype(np.int64)
        idx = int(np.searchsorted(dates, stamp, side="right")) - 1
        if idx < 0:
            return None
        return int(dates[idx]), float(values[idx])
