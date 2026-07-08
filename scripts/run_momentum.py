
from __future__ import annotations

from _results import record_run, single_run_payload
from _sharadar_inputs import (
    DEFAULT_END,
    DEFAULT_MAX_SYMBOLS,
    DEFAULT_START,
    build_sharadar_backtest,
    run_and_report,
)

from value_portfolio import Momentum

LOOKBACK = 126  # ~6 trading months
SKIP = 0  # set to 21 for 12-1-style momentum
TOP_K = 10  # hold the top 10 of the basket
REBALANCE_EVERY = 21  # ~one trading month


def main() -> None:
    backtest = build_sharadar_backtest()
    top_k = min(TOP_K, len(backtest.symbols))
    agent = Momentum(
        backtest.symbols,
        lookback=LOOKBACK,
        skip=SKIP,
        top_k=top_k,
        rebalance_every=REBALANCE_EVERY,
    )
    report = run_and_report(
        backtest,
        agent,
        "=== Momentum on point-in-time S&P 500 (Sharadar) ===",
        extra_lines=[
            f"Lookback / skip   : {LOOKBACK} / {SKIP} bars",
            f"Top-K             : {top_k} of {len(backtest.symbols)}",
            f"Rebalance cadence : every {REBALANCE_EVERY} trading days",
        ],
    )

    payload = single_run_payload(report)
    metrics = payload["metrics"]
    record_run(
        kind="single_run",
        slug="momentum",
        params={
            "agent": "momentum",
            "start": DEFAULT_START.date().isoformat(),
            "end": DEFAULT_END.date().isoformat(),
            "max_symbols": DEFAULT_MAX_SYMBOLS,
            "lookback": LOOKBACK,
            "skip": SKIP,
            "top_k": top_k,
            "rebalance_every": REBALANCE_EVERY,
        },
        payload=payload,
        headline={
            "information_ratio": metrics["information_ratio"],
            "active_return": metrics["active_return"],
        },
    )


if __name__ == "__main__":
    main()
