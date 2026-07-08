
from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nasdaqdatalink as ndl

from value_portfolio.config import SharadarSettings

BUNDLE_TABLES = [
    "INDICATORS",  # data dictionary
    "SP500",       # index membership history
    "TICKERS",     # ticker metadata
    "ACTIONS",     # corporate actions
    "METRICS",     # daily metrics for funds
    "SF3A",        # institutional holdings, aggregated by ticker
    "SF3B",        # institutional holdings, aggregated by investor
    "EVENTS",      # fundamental event flags
    "SF1",         # core fundamentals
    "SFP",         # fund/ETF prices
    "SF2",         # insider transactions
    "DAILY",       # daily valuation metrics
    "SEP",         # equity prices
    "SF3",         # institutional holdings, full detail
]

OUT_DIR = Path(__file__).resolve().parents[2] / "core_us_bundle"
MANIFEST = OUT_DIR / "manifest.jsonl"

_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 60.0


def _zip_is_sound(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(table: str, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
    entry = {
        "table": f"SHARADAR/{table}",
        "file": zip_path.name,
        "bytes": zip_path.stat().st_size,
        "sha256": _sha256(zip_path),
        "csv_members": [
            {"name": info.filename, "uncompressed_bytes": info.file_size} for info in members
        ],
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with MANIFEST.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def fetch_table(table: str) -> None:
    zip_path = OUT_DIR / f"SHARADAR_{table}.zip"
    if zip_path.exists():
        if _zip_is_sound(zip_path):
            print(f"[skip] {zip_path.name} already present and CRC-clean", flush=True)
            return
        print(f"[redo] {zip_path.name} exists but is corrupt; re-downloading", flush=True)
        zip_path.unlink()

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            print(f"[get ] SHARADAR/{table} (attempt {attempt}/{_MAX_RETRIES}) ...", flush=True)
            started = time.monotonic()
            ndl.export_table(f"SHARADAR/{table}", filename=str(zip_path))
            if not _zip_is_sound(zip_path):
                raise RuntimeError(f"{zip_path.name} failed the post-download CRC check")
            elapsed = time.monotonic() - started
            size_mb = zip_path.stat().st_size / 1e6
            _record(table, zip_path)
            print(f"[ ok ] {zip_path.name}: {size_mb:,.0f} MB in {elapsed:,.0f}s", flush=True)
            return
        except Exception as exc:  # transport, API, or integrity failure
            last_exc = exc
            zip_path.unlink(missing_ok=True)
            wait = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            print(f"[fail] SHARADAR/{table}: {exc}; retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"SHARADAR/{table} failed after {_MAX_RETRIES} attempts") from last_exc


def main(argv: list[str]) -> int:
    tables = [name.upper() for name in argv] or BUNDLE_TABLES
    unknown = [name for name in tables if name not in BUNDLE_TABLES]
    if unknown:
        print(f"Unknown table(s): {unknown}; expected any of {BUNDLE_TABLES}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ndl.ApiConfig.api_key = SharadarSettings().us_bundle_api_key

    failures: list[str] = []
    for table in tables:
        try:
            fetch_table(table)
        except Exception as exc:
            print(f"[GIVE UP] SHARADAR/{table}: {exc}", flush=True)
            failures.append(table)

    done = [t for t in tables if t not in failures]
    print(f"\nFinished: {len(done)}/{len(tables)} tables mirrored under {OUT_DIR}", flush=True)
    if failures:
        print(f"FAILED tables (re-run to retry just these): {' '.join(failures)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
