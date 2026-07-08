"""Shared CLI helpers for the runnable scripts."""

from __future__ import annotations

from datetime import UTC, datetime


def parse_date(text: str) -> datetime:
    """Parse a ``YYYY-MM-DD`` CLI argument into a UTC-aware datetime."""
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
