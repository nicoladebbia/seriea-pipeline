"""Write parsed match data to structured Parquet tables."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import pandas as pd

from models.schemas import MatchData
from parser.events import events_to_records
from parser.lineups import lineups_to_records
from storage.paths import parsed_path

log = logging.getLogger(__name__)


def save_all(matches: list[MatchData]) -> None:
    """Save all parsed match data to Parquet tables.

    Appends to existing tables if they exist.
    """
    if not matches:
        log.warning("No matches to save")
        return

    _save_matches(matches)
    _save_player_stats(matches)
    _save_goalkeeper_stats(matches)
    _save_shots(matches)
    _save_lineups(matches)
    _save_events(matches)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns that should be numeric but may have mixed types.

    FBref sometimes embeds string values like '86' in numeric columns.
    This causes ArrowInvalid errors when writing parquet after concat.
    We force-coerce known problematic columns and any object columns
    that look fully numeric.
    """
    KNOWN_NUMERIC = {
        "minutes", "goals", "assists", "shots", "shots_on_target",
        "yellow_cards", "red_cards", "xg", "npxg", "xa",
        "passes", "passes_completed", "progressive_passes",
        "carries", "progressive_carries", "touches",
        "tackles", "interceptions", "blocks", "clearances",
        "home_score", "away_score", "home_xg", "away_xg",
        "attendance", "matchweek", "shirt_number",
    }
    for col in df.columns:
        if col in KNOWN_NUMERIC or (df[col].dtype == object):
            try:
                coerced = pd.to_numeric(df[col], errors="coerce")
                # Only replace if at least 80% converted (avoid text columns)
                if coerced.notna().sum() >= 0.8 * df[col].notna().sum():
                    df[col] = coerced
            except Exception:
                pass
    return df


def _append_or_create(df: pd.DataFrame, table_name: str) -> None:
    """Append df to an existing Parquet table, or create it."""
    path = parsed_path(table_name)

    if path.exists():
        existing = pd.read_parquet(path)
        # Remove any rows with match_ids already present (in case of reparse)
        if "match_id" in df.columns and "match_id" in existing.columns:
            new_ids = set(df["match_id"].unique())
            existing = existing[~existing["match_id"].isin(new_ids)]
        df = pd.concat([existing, df], ignore_index=True)

    # Coerce mixed-type columns to avoid ArrowInvalid on parquet write
    df = _coerce_numeric_columns(df)
    df.to_parquet(path, index=False)
    log.info("Saved %s: %d rows to %s", table_name, len(df), path)


def _save_matches(matches: list[MatchData]) -> None:
    """Save match-level metadata + team stats."""
    rows: list[dict[str, Any]] = []

    for m in matches:
        row = {
            "match_id": m.metadata.match_id,
            "season": m.metadata.season,
            "match_date": str(m.metadata.match_date),
            "kickoff_time": m.metadata.kickoff_time,
            "matchweek": m.metadata.matchweek,
            "home_team": m.metadata.home_team,
            "away_team": m.metadata.away_team,
            "home_score": m.metadata.home_score,
            "away_score": m.metadata.away_score,
            "home_xg": m.metadata.home_xg,
            "away_xg": m.metadata.away_xg,
            "venue": m.metadata.venue,
            "attendance": m.metadata.attendance,
            "referee": m.metadata.referee,
            "home_manager": m.metadata.home_manager,
            "away_manager": m.metadata.away_manager,
            "home_captain": m.metadata.home_captain,
            "away_captain": m.metadata.away_captain,
        }
        # Add lineup formations
        if m.home_lineup:
            row["home_formation"] = m.home_lineup.formation or ""
        if m.away_lineup:
            row["away_formation"] = m.away_lineup.formation or ""

        # Merge team stats
        row.update(m.team_stats)

        rows.append(row)

    df = pd.DataFrame(rows)
    _append_or_create(df, "matches")


def _save_player_stats(matches: list[MatchData]) -> None:
    """Save player-level stats (all players, all matches)."""
    all_records: list[dict] = []
    for m in matches:
        all_records.extend(m.home_players)
        all_records.extend(m.away_players)

    if not all_records:
        log.warning("No player stats to save")
        return

    df = pd.DataFrame(all_records)

    # Add season and match_date for convenience
    meta_map = {m.metadata.match_id: m.metadata for m in matches}
    if "match_id" in df.columns:
        df["season"] = df["match_id"].map(lambda mid: meta_map.get(mid, None) and meta_map[mid].season)
        df["match_date"] = df["match_id"].map(
            lambda mid: str(meta_map[mid].match_date) if mid in meta_map else ""
        )

    _append_or_create(df, "player_stats")


def _save_goalkeeper_stats(matches: list[MatchData]) -> None:
    """Save goalkeeper stats."""
    all_records: list[dict] = []
    for m in matches:
        all_records.extend(m.home_gk)
        all_records.extend(m.away_gk)

    if not all_records:
        return

    df = pd.DataFrame(all_records)
    _append_or_create(df, "goalkeeper_stats")


def _save_shots(matches: list[MatchData]) -> None:
    """Save shot-level data."""
    all_records: list[dict] = []
    for m in matches:
        all_records.extend(m.shots)

    if not all_records:
        return

    df = pd.DataFrame(all_records)

    # Add season
    meta_map = {m.metadata.match_id: m.metadata for m in matches}
    if "match_id" in df.columns:
        df["season"] = df["match_id"].map(lambda mid: meta_map.get(mid, None) and meta_map[mid].season)

    _append_or_create(df, "shots")


def _save_lineups(matches: list[MatchData]) -> None:
    """Save lineup data."""
    all_records: list[dict] = []
    for m in matches:
        records = lineups_to_records(
            m.home_lineup,
            m.away_lineup,
            m.metadata.match_id,
            m.metadata.home_team,
            m.metadata.away_team,
        )
        all_records.extend(records)

    if not all_records:
        return

    df = pd.DataFrame(all_records)
    _append_or_create(df, "lineups")


def _save_events(matches: list[MatchData]) -> None:
    """Save match events."""
    all_records: list[dict] = []
    for m in matches:
        records = events_to_records(m.events, m.metadata.match_id, m.metadata.season)
        all_records.extend(records)

    if not all_records:
        return

    df = pd.DataFrame(all_records)
    _append_or_create(df, "events")
