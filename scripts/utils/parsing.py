"""Shared parsing and caching utilities used across scripts."""

import hashlib
import re
from pathlib import Path
from typing import Optional


def extract_line(s: str, default: Optional[float] = None) -> Optional[float]:
    """Extract numeric line from a market/selection string.

    Examples:
        'Over 2.5'  -> 2.5
        'HOME -1'   -> -1.0
        'Shots Over 1.5' -> 1.5

    Args:
        s: The market or selection string to parse.
        default: Value to return if no number is found (default: None).

    Returns:
        The extracted line as a float, or *default* if no number is found.
    """
    match = re.search(r"[-+]?\d+\.?\d*", s)
    if match:
        return float(match.group())
    return default


def get_cache_path(cache_dir: Path, key: str) -> Path:
    """Return an MD5-hashed JSON cache file path inside *cache_dir*.

    Creates *cache_dir* if it does not exist.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
