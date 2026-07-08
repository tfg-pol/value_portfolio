from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from value_portfolio.broker.types import AccountSnapshot
from value_portfolio.data.base import MarketDataSource
from value_portfolio.data.fundamentals import FundamentalsDataSource
from value_portfolio.data.scores import ScoreSource
from value_portfolio.data.universe import Universe


@dataclass(frozen=True, slots=True)
class DecisionContext:
    now: datetime
    account: AccountSnapshot
    data: MarketDataSource
    universe: Universe | None = None
    fundamentals: FundamentalsDataSource | None = None
    scores: ScoreSource | None = None


class Agent(ABC):
    @abstractmethod
    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        """Outputs Ticker : Weight dictionary"""


class RebalancingAgent(Agent):
    """Base for agents that act once every `rebalance_every` decision steps."""

    def __init__(self, rebalance_every: int) -> None:
        if rebalance_every < 1:
            raise ValueError(f"rebalance_every must be >= 1, got {rebalance_every}")
        self._rebalance_every = rebalance_every
        self._step = 0

    def _should_rebalance(self) -> bool:
        """Whether this step rebalances; advances the step counter."""
        is_step = self._step % self._rebalance_every == 0
        self._step += 1
        return is_step


def require_unique_symbols(symbols: Sequence[str], owner: str) -> list[str]:
    if not symbols:
        raise ValueError(f"{owner} requires at least one symbol")
    if len(set(symbols)) != len(symbols):
        raise ValueError(f"{owner} symbols must be unique, got {list(symbols)}")
    return list(symbols)
