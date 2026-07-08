
from __future__ import annotations

from value_portfolio.baselines.buy_and_hold import BuyAndHold
from value_portfolio.baselines.equal_weight import EqualWeight
from value_portfolio.baselines.mean_variance import MeanVariance
from value_portfolio.baselines.momentum import Momentum

__all__ = [
    "BuyAndHold",
    "EqualWeight",
    "MeanVariance",
    "Momentum",
]
