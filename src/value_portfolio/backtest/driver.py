
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal

from value_portfolio.agent import Agent, DecisionContext
from value_portfolio.backtest.benchmark import BenchmarkSeries
from value_portfolio.backtest.report import BacktestReport
from value_portfolio.broker.simulated import SimulatedBroker
from value_portfolio.broker.types import AccountSnapshot, Order, OrderStatus
from value_portfolio.data.fundamentals import FundamentalsDataSource
from value_portfolio.data.in_memory import InMemoryMarketData
from value_portfolio.data.scores import ScoreSource
from value_portfolio.data.universe import Universe
from value_portfolio.rebalancer import Rebalancer


def run_backtest(
    agent: Agent,
    broker: SimulatedBroker,
    data: InMemoryMarketData,
    timeline: Iterable[datetime] | None = None,
    rebalancer: Rebalancer | None = None,
    benchmark: BenchmarkSeries | None = None,
    universe: Universe | None = None,
    fundamentals: FundamentalsDataSource | None = None,
    scores: ScoreSource | None = None,
) -> BacktestReport:
    steps = list(timeline) if timeline is not None else list(data.timeline)
    if not steps:
        raise ValueError("run_backtest requires a non-empty timeline")
    rebalancer = rebalancer or Rebalancer()

    snapshots: list[AccountSnapshot] = []
    fills: list[Order] = []
    benchmark_levels: list[Decimal | None] = []
    for t in steps:
        terminal = broker.advance_to(t)
        fills.extend(order for order in terminal if order.status is OrderStatus.FILLED)
        ctx = DecisionContext(
            now=t,
            account=broker.get_account(),
            data=data,
            universe=universe,
            fundamentals=fundamentals,
            scores=scores,
        )
        target = agent.decide(ctx)
        if target is not None:
            rebalancer.rebalance(target, broker, data)
        snapshots.append(broker.get_account())
        if benchmark is not None:
            benchmark_levels.append(benchmark.level_at(t))

    return BacktestReport(
        snapshots=tuple(snapshots),
        fills=tuple(fills),
        benchmark_levels=tuple(benchmark_levels) if benchmark is not None else None,
    )
