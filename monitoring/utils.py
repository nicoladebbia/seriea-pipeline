"""Shared utilities for monitoring modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json_history(path: Path) -> List[Dict[str, Any]]:
    """Load JSON history from a file.

    Returns an empty list if the file does not exist or is unreadable.
    """
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def save_json_history(
    path: Path,
    data: List[Dict[str, Any]],
    max_entries: Optional[int] = None,
) -> None:
    """Save JSON history to a file.

    Args:
        path: File path to write to.
        data: List of history entries.
        max_entries: If set, only keep the last N entries.
    """
    to_write = data[-max_entries:] if max_entries else data
    with open(path, "w") as f:
        json.dump(to_write, f, indent=2)
