
from __future__ import annotations

from typing import Any

from value_portfolio.learning.cost_aware_allocator import CostAwareAllocator
from value_portfolio.learning.score_proportional_top_k import ScoreProportionalTopK
from value_portfolio.learning.score_top_k import ScoreTopK
from value_portfolio.learning.selection import select_top_scored

_LAZY: dict[str, str] = {
    "CrossSection": "value_portfolio.learning.features",
    "DailyMarketCap": "value_portfolio.learning.features",
    "build_cross_sections": "value_portfolio.learning.features",
    "month_end_dates": "value_portfolio.learning.features",
    "rank_normalize": "value_portfolio.learning.features",
    "ValuationConfig": "value_portfolio.learning.valuation",
    "fit_predict_expanding": "value_portfolio.learning.valuation",
    "write_scores_parquet": "value_portfolio.learning.valuation",
}

__all__ = [
    "CostAwareAllocator",
    "ScoreProportionalTopK",
    "ScoreTopK",
    "select_top_scored",
    *sorted(_LAZY),
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
