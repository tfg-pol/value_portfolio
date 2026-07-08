
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal

from value_portfolio.data.scores import ScoreSource
from value_portfolio.data.universe import Universe


def select_top_scored(
    scores: ScoreSource,
    now: datetime,
    candidates: Iterable[str],
    top_k: int,
    universe: Universe | None = None,
) -> list[tuple[str, Decimal]]:

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")

    eligible = set(candidates)
    if universe is not None:
        eligible &= universe.members_at(now)

    scored: list[tuple[str, Decimal]] = []
    for symbol in eligible:
        value = scores.score(symbol, now)
        if value is not None:
            scored.append((symbol, value))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]
