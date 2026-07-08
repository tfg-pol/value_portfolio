
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from value_portfolio.backtest import (
    EquitySeries,
    drawdown_levels,
    normalized_levels,
    series_from_dict,
)

try:
    import matplotlib

    matplotlib.use("Agg")  # headless: write files, never open a window
    from matplotlib import pyplot as plt
except ImportError:
    raise SystemExit(
        "matplotlib is required for plotting: install with `uv sync --extra viz` "
        "(or `uv sync --all-extras`)."
    ) from None

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LEDGER = _REPO_ROOT / "results" / "ledger.jsonl"
_DEFAULT_OUT = _REPO_ROOT / "results" / "figures"

_DEFAULT_WINDOW_METRICS = ("information_ratio", "sharpe_ratio", "active_return")
# scripts/run_*.py entry point per recorded `params["agent"]`, for rerun hints.
_AGENT_SCRIPTS = {
    "top20": "run_score_top20.py",
    "buy_and_hold": "run_buy_and_hold.py",
    "equal_weight": "run_equal_weight.py",
    "momentum": "run_momentum.py",
    "mean_variance": "run_mean_variance.py",
}


# Artifact loading (no matplotlib below this section's callers)


def _load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"artifact not found: {path}")
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise SystemExit(f"artifact is not a JSON object: {path}")
    return record


def _latest_artifacts(slugs: list[str], kind: str) -> list[Path]:
    """The most recent ledger entry per slug (restricted to `kind`), resolved
    to artifact paths, in the order the slugs were given.
    """
    if not _LEDGER.exists():
        raise SystemExit(f"no ledger at {_LEDGER}; run a strategy script first")
    latest: dict[str, str] = {}
    for line in _LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("kind") == kind and entry.get("slug") in slugs:
            latest[entry["slug"]] = entry["artifact"]  # later lines win
    missing = [slug for slug in slugs if slug not in latest]
    if missing:
        raise SystemExit(
            f"no ledger entry of kind '{kind}' for slug(s): {', '.join(missing)}. "
            "Run the corresponding strategy script first."
        )
    return [_REPO_ROOT / latest[slug] for slug in slugs]


def _rerun_hint(record: dict[str, Any]) -> str:
    params = record.get("params", {})
    script = _AGENT_SCRIPTS.get(params.get("agent", ""), "the strategy script")
    hint = f"uv run python scripts/{script}"
    for flag in ("model", "seed", "start", "end"):
        if params.get(flag) is not None:
            hint += f" --{flag} {params[flag]}"
    return hint


def _extract_series(record: dict[str, Any], path: Path) -> EquitySeries:
    if "series" not in record:
        raise SystemExit(
            f"{path} has no time series: it predates series persistence. "
            f"Re-run the strategy to produce a plottable artifact, e.g.:\n  {_rerun_hint(record)}"
        )
    return series_from_dict(record["series"])


def _window_metric_values(record: dict[str, Any], metric: str) -> list[float]:
    windows = record.get("windows")
    if not isinstance(windows, list) or not windows:
        raise SystemExit("artifact has no per-window results (is it an eval_windows run?)")
    return [float(w[metric]) for w in windows if w.get(metric) is not None]


def _label(record: dict[str, Any]) -> str:
    return str(record.get("slug", "unnamed"))


def _window_text(record: dict[str, Any]) -> str:
    params = record.get("params", {})
    return f"{params.get('start', '?')} -> {params.get('end', '?')}"


def _warn_on_mismatched_windows(records: list[dict[str, Any]]) -> None:
    spans = {_window_text(record) for record in records}
    if len(spans) > 1:
        print(
            "WARNING: artifacts cover different windows "
            f"({'; '.join(sorted(spans))}) — normalized curves are not directly comparable."
        )


def _save(
    fig: Any, subcommand: str, records: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    slugs = "_".join(_label(record) for record in records)
    path = args.out / f"{stamp}_{subcommand}_{slugs}.{args.format}"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written to {path}")


# Figures


def _plot_growth(
    records: list[dict[str, Any]], paths: list[Path], args: argparse.Namespace
) -> None:
    _warn_on_mismatched_windows(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    benchmark_drawn = False
    for record, path in zip(records, paths, strict=True):
        series = _extract_series(record, path)
        timestamps = series.timestamps
        # Equity has no None entries, so the normalized curve is fully defined.
        growth = [float(v) for v in normalized_levels(series.equity) if v is not None]
        ax.plot(timestamps, growth, label=_label(record))
        if not benchmark_drawn and series.benchmark is not None:
            known = [
                (stamp, level)
                for stamp, level in zip(
                    timestamps, normalized_levels(series.benchmark), strict=True
                )
                if level is not None
            ]
            if known:
                ax.plot(
                    [stamp for stamp, _ in known],
                    [float(level) for _, level in known],
                    label="SP500TR",
                    color="black",
                    linestyle="--",
                )
                benchmark_drawn = True
    ax.set_title(
        f"Growth of 1 — {_window_text(records[0])} (commit {records[0].get('git_commit', '?')})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 (start = 1.0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, "growth", records, args)


def _plot_drawdown(
    records: list[dict[str, Any]], paths: list[Path], args: argparse.Namespace
) -> None:
    _warn_on_mismatched_windows(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    benchmark_drawn = False
    for record, path in zip(records, paths, strict=True):
        series = _extract_series(record, path)
        drawdowns = drawdown_levels(series.equity)
        ax.plot(
            series.timestamps,
            [float(d) * 100 for d in drawdowns],
            label=_label(record),
        )
        if not benchmark_drawn and series.benchmark is not None:
            known = [
                (stamp, level)
                for stamp, level in zip(series.timestamps, series.benchmark, strict=True)
                if level is not None
            ]
            if known:
                bench_dd = drawdown_levels([level for _, level in known])
                ax.plot(
                    [stamp for stamp, _ in known],
                    [float(d) * 100 for d in bench_dd],
                    label="SP500TR",
                    color="black",
                    linestyle="--",
                )
                benchmark_drawn = True
    ax.set_title(
        f"Drawdown — {_window_text(records[0])} (commit {records[0].get('git_commit', '?')})"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    _save(fig, "drawdown", records, args)


def _plot_windows(
    records: list[dict[str, Any]], paths: list[Path], args: argparse.Namespace
) -> None:
    metrics: list[str] = args.metrics
    for record, path in zip(records, paths, strict=True):
        for metric in metrics:
            if not _window_metric_values(record, metric):
                raise SystemExit(f"metric '{metric}' has no values in {path}")
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6), squeeze=False)
    labels = [_label(record) for record in records]
    for ax, metric in zip(axes[0], metrics, strict=True):
        values_per_strategy = [_window_metric_values(record, metric) for record in records]
        ax.boxplot(values_per_strategy, tick_labels=labels)
        for i, values in enumerate(values_per_strategy, start=1):
            ax.scatter([i] * len(values), values, alpha=0.5, color="tab:blue", zorder=3)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    n_windows = ", ".join(str(record.get("n_windows", "?")) for record in records)
    fig.suptitle(f"Per-window metric distributions ({n_windows} windows per strategy)")
    _save(fig, "windows", records, args)


def _plot_signal(
    records: list[dict[str, Any]], paths: list[Path], args: argparse.Namespace
) -> None:
    window = args.rolling_months
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for record, path in zip(records, paths, strict=True):
        months = record.get("months")
        if not isinstance(months, list) or not months:
            raise SystemExit(f"{path} has no monthly series (is it a signal_diagnostics run?)")
        dates = [datetime.fromisoformat(m["date"]) for m in months]
        ics = [float(m["rank_ic"]) for m in months]
        rolling = [sum(ics[i - window + 1 : i + 1]) / window for i in range(window - 1, len(ics))]
        axes[0].plot(dates[window - 1 :], rolling, label=_label(record))

        level = 1.0
        growth: list[tuple[datetime, float]] = []
        for date, month in zip(dates, months, strict=True):
            spread = month.get("decile_spread_gross")
            if spread is not None:
                level *= 1.0 + float(spread)
                growth.append((date, level))
        axes[1].plot([d for d, _ in growth], [g for _, g in growth], label=_label(record))

    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title(f"Rank IC, {window}-month rolling mean")
    axes[0].set_ylabel("Spearman rank IC")
    axes[1].set_title("Cumulative top-minus-bottom decile spread (gross, not investable)")
    axes[1].set_ylabel("Growth of 1 (gross)")
    for ax in axes:
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.autofmt_xdate()
    fig.suptitle(f"Signal diagnostics within S&P 500 — {_window_text(records[0])}")
    _save(fig, "signal", records, args)


# CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    ledger_kinds = {
        "growth": "single_run",
        "drawdown": "single_run",
        "windows": "eval_windows",
        "signal": "signal_diagnostics",
    }
    for name, help_text in (
        ("growth", "growth-of-1 equity curves vs SP500TR"),
        ("drawdown", "drawdown curves"),
        ("windows", "per-window metric distributions across strategies"),
        ("signal", "rolling rank IC + cumulative gross decile spread per score variant"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("artifacts", nargs="*", type=Path, help="result artifact JSON paths")
        sub.add_argument(
            "--latest",
            nargs="+",
            default=[],
            metavar="SLUG",
            help=f"resolve each slug's latest '{ledger_kinds[name]}' ledger entry",
        )
        sub.add_argument("--out", type=Path, default=_DEFAULT_OUT)
        sub.add_argument("--format", choices=("png", "pdf"), default="png")
        sub.add_argument("--dpi", type=int, default=300)
        if name == "windows":
            sub.add_argument(
                "--metrics",
                nargs="+",
                default=list(_DEFAULT_WINDOW_METRICS),
                help=f"per-window metrics to plot (default: {' '.join(_DEFAULT_WINDOW_METRICS)})",
            )
        if name == "signal":
            sub.add_argument(
                "--rolling-months",
                type=int,
                default=12,
                help="window of the rolling mean rank IC (default: 12)",
            )
    args = parser.parse_args()

    paths = list(args.artifacts)
    if args.latest:
        paths.extend(_latest_artifacts(args.latest, ledger_kinds[args.subcommand]))
    if not paths:
        raise SystemExit("no artifacts given: pass paths and/or --latest SLUG [SLUG ...]")
    records = [_load_artifact(path) for path in paths]

    plot = {
        "growth": _plot_growth,
        "drawdown": _plot_drawdown,
        "windows": _plot_windows,
        "signal": _plot_signal,
    }
    plot[args.subcommand](records, paths, args)


if __name__ == "__main__":
    main()
