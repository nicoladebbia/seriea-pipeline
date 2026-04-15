"""Shared JSON I/O utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MISSING = object()


def load_json_safe(path: Path, default: Any = _MISSING) -> Any:
    """Load JSON file safely. Returns default on failure (empty dict if not specified)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    if default is _MISSING:
        return {}
    return default
