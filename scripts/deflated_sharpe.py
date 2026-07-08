
from __future__ import annotations

import argparse
import json
from typing import Any

from _results import latest_artifact, record_run, slugs_of_kind

from value_portfolio.backtest import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    sample_kurtosis,
    sample_skewness,
    series_from_dict,
)

_TRADING_DAYS_PER_YEAR = 252


def _load_record(slug: str, kind: str) -> dict[str, Any]:
    record = json.loads(latest_artifact(slug, kind).read_text())
    if not isinstance(record, dict):
        raise SystemExit(f"artifact for '{slug}' is not a JSON object")
    return record


def _window_of(record: dict[str, Any]) -> tuple[str, str]:
    params = record.get("params", {})
    return str(params.get("start")), str(params.get("end"))


def _periodic_returns(record: dict[str, Any], slug: str) -> list[float]:
    if "series" not in record:
        raise SystemExit(
            f"'{slug}' artifact has no time series (it predates series persistence); "
            "re-run the strategy to produce a deflatable artifact."
        )
    equity = [float(value) for value in series_from_dict(record["series"]).equity]
    return [
        (equity[i] - equity[i - 1]) / equity[i - 1]
        for i in range(1, len(equity))
        if equity[i - 1] > 0.0
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy", required=True, help="ledger slug of the run to deflate (the selected winner)"
    )
    parser.add_argument(
        "--family",
        nargs="+",
        default=None,
        metavar="SLUG",
        help="the trial family deflated against (default: every single_run slug over the same "
        "window). The strategy is always included.",
    )
    parser.add_argument(
        "--kind",
        default="single_run",
        help="ledger kind of the artifacts (default: single_run; see module docstring)",
    )
    args = parser.parse_args()

    strategy_record = _load_record(args.strategy, args.kind)
    window = _window_of(strategy_record)

    explicit = args.family is not None
    family = list(args.family) if explicit else slugs_of_kind(args.kind)
    if args.strategy not in family:
        family.append(args.strategy)
    sharpes: dict[str, float] = {}
    mismatched: list[str] = []
    for slug in family:
        record = strategy_record if slug == args.strategy else _load_record(slug, args.kind)
        if _window_of(record) != window:
            mismatched.append(f"{slug} {_window_of(record)[0]}..{_window_of(record)[1]}")
            continue
        sharpes[slug] = per_period_sharpe(_periodic_returns(record, slug))
    if explicit and mismatched:
        raise SystemExit(
            f"trial family must share the strategy's window {window[0]}..{window[1]}; "
            f"these differ: {', '.join(mismatched)}."
        )
    if mismatched:
        print(
            f"note: skipped {len(mismatched)} run(s) outside the {window[0]}..{window[1]} window "
            f"({', '.join(mismatched)})."
        )

    n_trials = len(sharpes)
    if n_trials < 2:
        raise SystemExit(
            f"the Deflated Sharpe needs >= 2 trials (the variants the winner beat); found "
            f"{n_trials} over window {window[0]}..{window[1]}. Run more variants or pass --family."
        )

    trial_values = list(sharpes.values())
    mean_sr = sum(trial_values) / n_trials
    sr_variance = sum((sr - mean_sr) ** 2 for sr in trial_values) / (n_trials - 1)

    returns = _periodic_returns(strategy_record, args.strategy)
    observed_sr = sharpes[args.strategy]
    skew = sample_skewness(returns)
    kurt = sample_kurtosis(returns)
    n_obs = len(returns)

    benchmark_sr = expected_max_sharpe(n_trials, sr_variance=sr_variance)
    psr = probabilistic_sharpe_ratio(
        observed_sr, sr_benchmark=0.0, n_obs=n_obs, skewness=skew, kurtosis=kurt
    )
    dsr = deflated_sharpe_ratio(
        observed_sr,
        n_trials=n_trials,
        sr_variance=sr_variance,
        n_obs=n_obs,
        skewness=skew,
        kurtosis=kurt,
    )
    annualised = observed_sr * _TRADING_DAYS_PER_YEAR**0.5

    print(f"=== Deflated Sharpe Ratio — {args.strategy} ({window[0]} -> {window[1]}) ===")
    print(f"Trials (n_trials)        : {n_trials}  [{', '.join(sorted(sharpes))}]")
    print(f"Observed Sharpe / period : {observed_sr:+.4f}  (annualised {annualised:+.3f})")
    print(f"Return moments           : skew {skew:+.3f}, kurtosis {kurt:.3f}, n_obs {n_obs}")
    print(f"Cross-trial SR variance  : {sr_variance:.6f}")
    print(f"Expected best-of-N SR    : {benchmark_sr:+.4f} / period")
    print(f"PSR  P[SR>0]             : {psr:.4f}  (before deflation)")
    print(f"DSR  P[SR>E(max)]        : {dsr:.4f}  (after deflation across {n_trials} trials)")
    verdict = "survives" if dsr > 0.95 else "does NOT survive" if dsr < 0.5 else "marginal under"
    print(f"-> {args.strategy} {verdict} deflation.")

    record_run(
        kind="deflated_sharpe",
        slug=args.strategy,
        params={
            "strategy": args.strategy,
            "source_kind": args.kind,
            "start": window[0],
            "end": window[1],
            "n_trials": n_trials,
            "family": sorted(sharpes),
        },
        payload={
            "observed_sharpe_per_period": observed_sr,
            "observed_sharpe_annualised": annualised,
            "skewness": skew,
            "kurtosis": kurt,
            "n_obs": n_obs,
            "sr_variance": sr_variance,
            "expected_max_sharpe": benchmark_sr,
            "probabilistic_sharpe_ratio": psr,
            "deflated_sharpe_ratio": dsr,
            "trial_sharpes": dict(sorted(sharpes.items())),
        },
        headline={
            "deflated_sharpe_ratio": round(dsr, 4),
            "probabilistic_sharpe_ratio": round(psr, 4),
            "n_trials": n_trials,
        },
    )


if __name__ == "__main__":
    main()
