from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from value_portfolio.backtest import BacktestReport, series_from_report, series_to_dict
from value_portfolio.backtest.evaluation import report_metrics

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_LEDGER = _RESULTS_DIR / "ledger.jsonl"


def iter_ledger() -> list[dict[str, object]]:
    """Every ledger entry in append order (oldest first); empty if no ledger."""
    if not _LEDGER.exists():
        return []
    return [json.loads(line) for line in _LEDGER.read_text().splitlines() if line.strip()]


def latest_artifact(slug: str, kind: str) -> Path:
    """Absolute path of the most recent ``kind`` artifact for `slug`.

    Later ledger lines win (the registry is append-only and chronological).
    Raises ``SystemExit`` with an actionable message when none is found.
    """
    artifact: str | None = None
    for entry in iter_ledger():
        if entry.get("kind") == kind and entry.get("slug") == slug:
            artifact = str(entry["artifact"])  # later lines override
    if artifact is None:
        raise SystemExit(
            f"no ledger entry of kind '{kind}' for slug '{slug}'. "
            "Run the corresponding strategy script first."
        )
    return _RESULTS_DIR.parent / artifact


def slugs_of_kind(kind: str, *, prefix: str = "") -> list[str]:
    """Distinct slugs recorded under `kind` (optionally filtered by `prefix`),
    in first-seen order — the candidate trial family for a deflation.
    """
    seen: dict[str, None] = {}
    for entry in iter_ledger():
        slug = entry.get("slug")
        if entry.get("kind") == kind and isinstance(slug, str) and slug.startswith(prefix):
            seen.setdefault(slug, None)
    return list(seen)


def single_run_payload(report: BacktestReport) -> dict[str, object]:
    """The standard single-run payload: the full metric catalog (Decimal
    strings, full precision) plus the plottable time series for
    ``scripts/plot_results.py``.
    """
    metrics = {k: str(v) if v is not None else None for k, v in report_metrics(report).items()}
    return {"metrics": metrics, "series": series_to_dict(series_from_report(report))}


def _git_commit() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_RESULTS_DIR.parent,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_RESULTS_DIR.parent,
        ).stdout.strip()
        return f"{head}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def record_run(
    kind: str,
    slug: str,
    params: dict[str, object],
    payload: dict[str, object],
    headline: dict[str, object] | None = None,
) -> Path:
    """Write the run's JSON artifact and append its ledger line; returns the
    artifact path. `headline` is the small metric set echoed into the ledger.
    """
    stamp = datetime.now(tz=UTC)
    record = {
        "timestamp": stamp.isoformat(timespec="seconds"),
        "kind": kind,
        "slug": slug,
        "git_commit": _git_commit(),
        "params": params,
        **payload,
    }

    out_dir = _RESULTS_DIR / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}_{slug}.json"
    artifact.write_text(json.dumps(record, indent=2) + "\n")

    ledger_line = {
        "timestamp": record["timestamp"],
        "kind": kind,
        "slug": slug,
        "git_commit": record["git_commit"],
        "params": params,
        "headline": headline or {},
        "artifact": str(artifact.relative_to(_RESULTS_DIR.parent)),
    }
    with (_RESULTS_DIR / "ledger.jsonl").open("a") as ledger:
        ledger.write(json.dumps(ledger_line) + "\n")

    print(f"\nResults written to {artifact} (ledger: results/ledger.jsonl).")
    return artifact
