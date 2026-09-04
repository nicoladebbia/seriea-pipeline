"""Shared helpers for tracking failed scraper downloads.

Used by scrape_sofascore, scrape_understat_matches
to persist/load a JSON dict of failed URLs for retry.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_failed(path: Path) -> dict:
    """Load failed-download log from *path*, returning empty dict on error."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def save_failed(path: Path, failed: dict) -> None:
    """Persist *failed* dict as JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failed, indent=2), encoding="utf-8")
