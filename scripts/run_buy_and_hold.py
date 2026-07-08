
from __future__ import annotations

from _results import record_run, single_run_payload
from _sharadar_inputs import (
    DEFAULT_END,
    DEFAULT_MAX_SYMBOLS,
    DEFAULT_START,
    build_sharadar_backtest,
    run_and_report,
)

from value_portfolio import BuyAndHold


def main() -> None:
    backtest = build_sharadar_backtest()
    agent = BuyAndHold(backtest.symbols)
    report = run_and_report(
        backtest, agent, "=== Buy-and-hold on point-in-time S&P 500 (Sharadar) ==="
    )

    payload = single_run_payload(report)
    metrics = payload["metrics"]
    record_run(
        kind="single_run",
        slug="buy_and_hold",
        params={
            "agent": "buy_and_hold",
            "start": DEFAULT_START.date().isoformat(),
            "end": DEFAULT_END.date().isoformat(),
            "max_symbols": DEFAULT_MAX_SYMBOLS,
        },
        payload=payload,
        headline={
            "information_ratio": metrics["information_ratio"],
            "active_return": metrics["active_return"],
        },
    )


if __name__ == "__main__":
    main()
