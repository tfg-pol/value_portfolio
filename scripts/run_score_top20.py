
from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path

from _cli import parse_date
from _results import record_run, single_run_payload
from _sharadar_inputs import build_sharadar_backtest, run_and_report

from value_portfolio.agent import Agent, DecisionContext
from value_portfolio.learning import CostAwareAllocator, ScoreProportionalTopK, ScoreTopK

_REPO_ROOT = Path(__file__).resolve().parent.parent

_ALLOCATORS = {
    "equal": (ScoreTopK, "top"),
    "proportional": (ScoreProportionalTopK, "prop"),
    "costaware": (CostAwareAllocator, "cost"),
}


class _WeightRecorder(Agent):

    def __init__(self, inner: Agent) -> None:
        self._inner = inner
        self.weight_vectors: list[list[float]] = []

    def decide(self, context: DecisionContext) -> dict[str, Decimal] | None:
        target = self._inner.decide(context)
        if target:
            self.weight_vectors.append([float(w) for w in target.values()])
        return target


def _mean_effective_holdings(weight_vectors: list[list[float]]) -> float:
    if not weight_vectors:
        return 0.0
    effective = [1.0 / sum(w * w for w in vec) for vec in weight_vectors if vec]
    return sum(effective) / len(effective)


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
    parser.add_argument("--start", type=parse_date, default=parse_date("2018-01-01"))
    parser.add_argument("--end", type=parse_date, default=parse_date("2019-12-31"))
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
    backtest = build_sharadar_backtest(start=args.start, end=args.end, scores_path=scores_path)

    allocator_cls, slug_prefix = _ALLOCATORS[args.weighting]
    agent_kwargs: dict[str, object] = {
        "symbols": backtest.symbols,
        "top_k": args.top_k,
        "rebalance_every": args.rebalance_every,
    }
    if args.weighting == "costaware":
        agent_kwargs["lookback"] = args.lookback
        agent_kwargs["risk_aversion"] = args.risk_aversion
        agent_kwargs["turnover_aversion"] = args.turnover_aversion
    recorder = _WeightRecorder(allocator_cls(**agent_kwargs))
    report = run_and_report(
        backtest,
        recorder,
        f"=== {allocator_cls.__name__} (top {args.top_k}, {args.model} {args.target}"
        f"{'+ind' if args.industry else ''} valuation scores, {args.weighting} weighting) ===",
        extra_lines=[f"Window {args.start.date()} -> {args.end.date()}; monthly rebalance."],
    )

    effective_holdings = _mean_effective_holdings(recorder.weight_vectors)
    print(f"Mean effective holdings (1 / Σ wᵢ²): {effective_holdings:.2f} of {args.top_k}")

    payload = single_run_payload(report)
    metrics = payload["metrics"]
    params: dict[str, object] = {
        "agent": f"{slug_prefix}{args.top_k}",
        "weighting": args.weighting,
        "model": args.model,
        "score_universe": "broad",
        "target": args.target,
        "industry": args.industry,
        "start": args.start.date().isoformat(),
        "end": args.end.date().isoformat(),
        "top_k": args.top_k,
        "rebalance_every": args.rebalance_every,
    }
    if args.weighting == "costaware":
        params["lookback"] = args.lookback
        params["risk_aversion"] = args.risk_aversion
        params["turnover_aversion"] = args.turnover_aversion
    record_run(
        kind="single_run",
        slug=f"{slug_prefix}{args.top_k}_{args.model}{score_tag}",
        params=params,
        payload=payload,
        headline={
            "information_ratio": metrics["information_ratio"],
            "active_return": metrics["active_return"],
            "mean_effective_holdings": f"{effective_holdings:.2f}",
        },
    )


if __name__ == "__main__":
    main()
