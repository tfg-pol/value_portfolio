"""Smoke test for `load_bars_from_alpaca` (real endpoint). Skipped without
credentials; run with ``uv run pytest -m smoke``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from value_portfolio import AlpacaSettings, InMemoryMarketData
from value_portfolio.data.loaders import load_bars_from_alpaca

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def settings() -> AlpacaSettings:
    try:
        return AlpacaSettings()  # type: ignore[call-arg]
    except ValidationError:
        pytest.skip("Alpaca credentials not configured (env or .env); skipping smoke tests.")


def test_loads_bars_into_in_memory_source(settings: AlpacaSettings) -> None:
    symbols = ["AAPL", "MSFT"]
    data = load_bars_from_alpaca(
        symbols,
        start=datetime(2023, 1, 1, tzinfo=UTC),
        end=datetime(2023, 3, 1, tzinfo=UTC),
        settings=settings,
    )

    assert isinstance(data, InMemoryMarketData)
    assert data.symbols == frozenset(symbols)
    # A two-month window of daily bars should hold a few dozen trading days.
    assert len(data.timeline) > 20
