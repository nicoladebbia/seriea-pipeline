"""
Module: config/settings.py
Purpose: Global configuration constants for FBref scraping, HTTP requests, seasons, paths, and league parameters
Inputs:  None (static configuration)
Outputs: Constants for FBREF_BASE_URL, DATA_DIR, SEASONS, REQUEST_DELAY_SECONDS, HEADERS, etc.
Called by: All modules needing path resolution, HTTP config, or season lists
Depends on: None (pure config module)
"""

from datetime import date
from pathlib import Path


def get_current_season() -> str:
    """Return the current season string (e.g. '2025-2026').

    Season boundary: August 1. Before Aug 1 → previous season is current.
    """
    today = date.today()
    if today.month >= 8:
        return f"{today.year}-{today.year + 1}"
    return f"{today.year - 1}-{today.year}"

# FBref
FBREF_BASE_URL = "https://fbref.com"
SERIE_A_COMP_ID = 11

# HTTP
REQUEST_DELAY_SECONDS = 12  # Conservative for historical backfill (revert to 6 after)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 2  # exponential: 4s, 8s, 16s, 32s, 64s
MAX_BACKOFF = 60
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Seasons to process (20 seasons of data for better training)
SEASONS = [
    "2005-2006",
    "2006-2007",
    "2007-2008",
    "2008-2009",
    "2009-2010",
    "2010-2011",
    "2011-2012",
    "2012-2013",
    "2013-2014",
    "2014-2015",
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

# League codes for football-data.co.uk
# Maps league name -> (code, display_name)
LEAGUES = {
    "serie_a": ("I1", "Serie A"),
    "premier_league": ("E0", "Premier League"),
    "la_liga": ("SP1", "La Liga"),
    "bundesliga": ("D1", "Bundesliga"),
    "ligue_1": ("F1", "Ligue 1"),
}

# Default league
DEFAULT_LEAGUE = "serie_a"

# Feature engineering
ROLLING_WINDOWS = [3, 5, 10]

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_FIXTURES_DIR = DATA_DIR / "raw" / "fixtures"
PARSED_DIR = DATA_DIR / "parsed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
UPCOMING_DIR = DATA_DIR / "upcoming"
BETTING_DIR = DATA_DIR / "betting"
BANKROLL_DIR = DATA_DIR / "bankroll"
LIVE_DIR = DATA_DIR / "live"
REGISTRY_PATH = DATA_DIR / "registry.json"

# Existing data (from old project)
OLD_PROJECT_ROOT = PROJECT_ROOT.parent
OLD_HTML_DIR = OLD_PROJECT_ROOT / "season_2024_2025"


# ---------------------------------------------------------------------------
# Atomic file writes — prevent corruption from concurrent/interrupted writes
# ---------------------------------------------------------------------------

import json as _json
import tempfile as _tempfile


def atomic_write_json(path: Path, data, indent: int = 2, cls=None):
    """Write JSON atomically: write to temp file, then rename.

    When cls is provided, it takes full control of serialization (json.dump
    ignores `default` when `cls` is set). When cls is None, `default=str`
    handles datetime/Path/etc.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = _tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w") as f:
            if cls is not None:
                _json.dump(data, f, indent=indent, cls=cls)
            else:
                _json.dump(data, f, indent=indent, default=str)
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_parquet(path: Path, df, **kwargs):
    """Write parquet atomically: write to temp file, then rename."""
    import pandas as _pd

    # Safety net: normalize match_date to datetime before every write
    if hasattr(df, "columns") and "match_date" in df.columns:
        df["match_date"] = _pd.to_datetime(df["match_date"], errors="coerce")
    if hasattr(df, "columns") and "matchweek" in df.columns:
        df["matchweek"] = _pd.to_numeric(df["matchweek"], errors="coerce")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    try:
        df.to_parquet(tmp, **kwargs)
        tmp.rename(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
