"""Equal-weight on the point-in-time S&P 500 (Sharadar data), rebalanced monthly

Run with: ``uv run python scripts/run_equal_weight.py``
"""

from __future__ import annotations

from _results import record_run, single_run_payload
from _sharadar_inputs import (
    DEFAULT_END,
    DEFAULT_MAX_SYMBOLS,
    DEFAULT_START,
    build_sharadar_backtest,
    run_and_report,
)

from value_portfolio import EqualWeight

REBALANCE_EVERY = 21  # ~one trading month on a daily timeline


def main() -> None:
    backtest = build_sharadar_backtest()
    agent = EqualWeight(backtest.symbols, rebalance_every=REBALANCE_EVERY)
    report = run_and_report(
        backtest,
        agent,
        "=== Equal-weight on point-in-time S&P 500 (Sharadar) ===",
        extra_lines=[f"Rebalance cadence : every {REBALANCE_EVERY} trading days"],
    )

    payload = single_run_payload(report)
    metrics = payload["metrics"]
    record_run(
        kind="single_run",
        slug="equal_weight",
        params={
            "agent": "equal_weight",
            "start": DEFAULT_START.date().isoformat(),
            "end": DEFAULT_END.date().isoformat(),
            "max_symbols": DEFAULT_MAX_SYMBOLS,
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
