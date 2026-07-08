"""Smoke tests for `AlpacaMarketData` (real endpoint). Skipped without
credentials; run with ``uv run pytest -m smoke``.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from value_portfolio import AlpacaMarketData, AlpacaSettings, Quote

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def data() -> AlpacaMarketData:
    try:
        settings = AlpacaSettings()  # type: ignore[call-arg]
    except ValidationError:
        pytest.skip("Alpaca credentials not configured (env or .env); skipping smoke tests.")
    return AlpacaMarketData(settings)


def test_get_quote_returns_sane_bid_ask(data: AlpacaMarketData) -> None:
    quote = data.get_quote("AAPL")
    assert isinstance(quote, Quote)
    assert quote.symbol == "AAPL"
    assert quote.bid_price > Decimal("0")
    assert quote.ask_price >= quote.bid_price
