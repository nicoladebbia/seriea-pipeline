"""Centralized path logic for all data artifacts."""

from pathlib import Path

from config.settings import (
    DATA_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    PARSED_DIR,
    RAW_FIXTURES_DIR,
    RAW_HTML_DIR,
    REGISTRY_PATH,
)


def ensure_dirs() -> None:
    """Create all required data directories."""
    for d in [RAW_HTML_DIR, RAW_FIXTURES_DIR, PARSED_DIR, FEATURES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def html_season_dir(season: str) -> Path:
    d = RAW_HTML_DIR / season
    d.mkdir(parents=True, exist_ok=True)
    return d


def html_match_path(season: str, match_id: str) -> Path:
    return html_season_dir(season) / f"{match_id}.html"


def fixtures_csv_path(season: str) -> Path:
    RAW_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_FIXTURES_DIR / f"{season}.csv"


def parsed_path(table_name: str) -> Path:
    """Return path for a parsed Parquet table (e.g. 'matches', 'player_stats')."""
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    return PARSED_DIR / f"{table_name}.parquet"


def features_path(league: str | None = None) -> Path:
    """Return path for the ML feature table.

    Args:
        league: If specified, return per-league path (e.g. features_serie_a.parquet).
                None returns the combined all-league file (backward compat).
    """
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    if league is None:
        return FEATURES_DIR / "features.parquet"
    return FEATURES_DIR / f"features_{league}.parquet"


def registry_path() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return REGISTRY_PATH
