
from __future__ import annotations

import argparse
from pathlib import Path

from _cli import parse_date
from _results import record_run
from _sharadar_inputs import build_sharadar_backtest

from value_portfolio import Rebalancer, run_backtest
from value_portfolio.backtest import BacktestReport, Window, evaluate_windows, rolling_windows
from value_portfolio.backtest.evaluation import evaluation_to_dict
from value_portfolio.learning import CostAwareAllocator, ScoreProportionalTopK, ScoreTopK
from value_portfolio.learning.features import FEATURE_NAMES

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Maps the --weighting flag to (allocator class, slug prefix); the prefix keeps
# the variants from colliding in the results ledger.
_ALLOCATORS = {
    "equal": (ScoreTopK, "top"),
    "proportional": (ScoreProportionalTopK, "prop"),
    "costaware": (CostAwareAllocator, "cost"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("ridge", "gbt"), default="ridge")
    parser.add_argument(
        "--target",
        choices=("cap", "mb", "ma", "ret"),
        default="cap",
        help="which stage-1 label's scores to load: cap (log market cap), "
        "mb (log market-to-book) or ma (log market-to-assets). Scores are trained "
        "on the broad panel; the traded universe stays the point-in-time S&P 500.",
    )
    parser.add_argument(
        "--industry", action="store_true", help="load the industry-aware (_ind) scores"
    )
    parser.add_argument(
        "--weighting",
        choices=("equal", "proportional", "costaware"),
        default="equal",
        help="equal-weight the top-K (ScoreTopK), weight by mispricing magnitude "
        "(ScoreProportionalTopK), or mean-variance with a turnover penalty toward "
        "current holdings (CostAwareAllocator)",
    )
    parser.add_argument("--start", type=parse_date, default=parse_date("2008-01-01"))
    parser.add_argument("--end", type=parse_date, default=parse_date("2025-12-31"))
    parser.add_argument("--window-years", type=int, default=5)
    parser.add_argument("--step-months", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--rebalance-every", type=int, default=21)
    # costaware-only knobs (ignored by the other allocators).
    parser.add_argument("--lookback", type=int, default=252, help="covariance window (costaware)")
    parser.add_argument("--risk-aversion", type=float, default=1.0, help="lambda (costaware)")
    parser.add_argument(
        "--turnover-aversion", type=float, default=1.0, help="gamma, spring stiffness (costaware)"
    )
    args = parser.parse_args()

    target_tag = "" if args.target == "cap" else f"_{args.target}"
    score_tag = f"_broad{target_tag}{'_ind' if args.industry else ''}"
    scores_path = _REPO_ROOT / "data" / "scores" / f"valuation_{args.model}{score_tag}.parquet"

    allocator_cls, slug_prefix = _ALLOCATORS[args.weighting]

    windows = rolling_windows(
        args.start, args.end, window_years=args.window_years, step_months=args.step_months
    )
    if not windows:
        raise SystemExit("No window of the requested length fits in the requested range.")
    print(
        f"{len(windows)} rolling windows of {args.window_years}y stepping "
        f"{args.step_months}m over {args.start.date()} -> {args.end.date()}.\n"
    )

    def run_window(window: Window) -> BacktestReport | None:
        print(f"--- window {window.label} ---")
        backtest = build_sharadar_backtest(
            start=window.start, end=window.end, scores_path=scores_path
        )
        if len(backtest.data.timeline) < 2 * args.rebalance_every:
            print("Skipped: not enough trading days in the window.")
            return None
        agent_kwargs: dict[str, object] = {
            "symbols": backtest.symbols,
            "top_k": args.top_k,
            "rebalance_every": args.rebalance_every,
        }
        if args.weighting == "costaware":
            agent_kwargs["lookback"] = args.lookback
            agent_kwargs["risk_aversion"] = args.risk_aversion
            agent_kwargs["turnover_aversion"] = args.turnover_aversion
        return run_backtest(
            allocator_cls(**agent_kwargs),
            backtest.broker,
            backtest.data,
            rebalancer=Rebalancer(),
            benchmark=backtest.benchmark,
            universe=backtest.universe,
            fundamentals=backtest.fundamentals,
            scores=backtest.scores,
        )

    title = (
        f"{slug_prefix}{args.top_k} ({args.weighting} weighting, {args.model} "
        f"{args.target}{'+ind' if args.industry else ''} scores, broad estimation panel)"
    )
    params = {
        "agent": f"{slug_prefix}{args.top_k}",
        "weighting": args.weighting,
        "model": args.model,
        "score_universe": "broad",
        "target": args.target,
        "industry": args.industry,
        "start": args.start.date().isoformat(),
        "end": args.end.date().isoformat(),
        "window_years": args.window_years,
        "step_months": args.step_months,
        "top_k": args.top_k,
        "rebalance_every": args.rebalance_every,
        "n_features": len(FEATURE_NAMES),
    }
    if args.weighting == "costaware":
        params["lookback"] = args.lookback
        params["risk_aversion"] = args.risk_aversion
        params["turnover_aversion"] = args.turnover_aversion

    evaluation = evaluate_windows(windows, run_window)
    print(f"\n=== Multi-window distribution — {title} ===")
    print(evaluation.summary())
    for metric in ("active_return", "information_ratio", "sharpe_ratio"):
        print(f"\n{metric} per window:")
        print("\n".join(evaluation.per_window_lines(metric)))
    payload = evaluation_to_dict(evaluation)

    summary = payload["summary"]
    assert isinstance(summary, dict)
    record_run(
        kind="eval_windows",
        slug=f"{slug_prefix}{args.top_k}_{args.model}{score_tag}",
        params=params,
        payload=payload,
        headline={
            "n_windows": payload["n_windows"],
            "information_ratio": summary.get("information_ratio"),
            "sharpe_ratio": summary.get("sharpe_ratio"),
        },
    )


if __name__ == "__main__":
    main()
