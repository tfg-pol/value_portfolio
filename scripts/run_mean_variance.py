
from __future__ import annotations

from decimal import Decimal

from _results import record_run, single_run_payload
from _sharadar_inputs import (
    DEFAULT_END,
    DEFAULT_MAX_SYMBOLS,
    DEFAULT_START,
    build_sharadar_backtest,
    run_and_report,
)

from value_portfolio import MeanVariance
from value_portfolio.baselines.mean_variance import Mode

LOOKBACK = 252  # ~1 trading year for the covariance window
MODE: Mode = "min_var"  # switch to "mean_var" to use the risk-aversion knob
RISK_AVERSION = Decimal("1")
REBALANCE_EVERY = 21  # ~one trading month


def main() -> None:
    backtest = build_sharadar_backtest()
    agent = MeanVariance(
        backtest.symbols,
        lookback=LOOKBACK,
        mode=MODE,
        risk_aversion=RISK_AVERSION,
        rebalance_every=REBALANCE_EVERY,
    )
    extra_lines = [f"Mode              : {MODE}"]
    if MODE == "mean_var":
        extra_lines.append(f"Risk aversion     : {RISK_AVERSION}")
    extra_lines.append(f"Lookback          : {LOOKBACK} bars")
    extra_lines.append(f"Rebalance cadence : every {REBALANCE_EVERY} trading days")
    report = run_and_report(
        backtest,
        agent,
        "=== Mean-variance on point-in-time S&P 500 (Sharadar) ===",
        extra_lines=extra_lines,
    )

    payload = single_run_payload(report)
    metrics = payload["metrics"]
    record_run(
        kind="single_run",
        slug="mean_variance",
        params={
            "agent": "mean_variance",
            "start": DEFAULT_START.date().isoformat(),
            "end": DEFAULT_END.date().isoformat(),
            "max_symbols": DEFAULT_MAX_SYMBOLS,
            "mode": MODE,
            "risk_aversion": str(RISK_AVERSION) if MODE == "mean_var" else None,
            "lookback": LOOKBACK,
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
