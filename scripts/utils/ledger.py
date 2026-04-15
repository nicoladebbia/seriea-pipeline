"""Shared JSON ledger load/save helpers.

Used by substitution_features, fair_odds_tracker, prop_tracker, and
substitution_tracker to read/write persistent JSON ledger files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)


def load_json_ledger(path: Path) -> List[dict]:
    """Load a JSON array from *path*.  Returns ``[]`` on any error."""
    if not path.exists():
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            log.warning("%s is not a list — returning empty ledger", path)
            return []
        return data
    except (json.JSONDecodeError, IOError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return []


def save_json_ledger(path: Path, data: List[dict]) -> None:
    """Write *data* as a JSON array to *path* with indent=2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
