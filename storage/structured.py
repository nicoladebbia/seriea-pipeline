"""Write parsed match data to structured Parquet tables."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.schemas import MatchData
from parser.events import events_to_records
from parser.lineups import lineups_to_records
from storage.paths import parsed_path

log = logging.getLogger(__name__)

_BASE = Path(__file__).parent.parent
_SOFASCORE_FIXTURES = _BASE / "data" / "external" / "sofascore"


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


def _fill_missing_matchweeks(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN matchweeks using Sofascore fixtures, then chronological fallback.

    Strategy (in priority order):
    1. Sofascore fixtures — authoritative matchweek assignments from official data
    2. Chronological inference — for each team, count matches in date order
       and assign matchweek = ceil(match_number_for_home_team)

    Only operates on rows where matchweek is NaN.
    """
    if "matchweek" not in df.columns:
        return df

    nan_mask = df["matchweek"].isna()
    n_missing = nan_mask.sum()
    if n_missing == 0:
        return df

    log.info("Filling %d missing matchweek values", n_missing)

    # --- Strategy 1: Sofascore fixtures lookup ---
    try:
        from config.team_names import normalize_team
    except ImportError:
        normalize_team = None

    sofascore_lookup: dict[tuple[str, str, str], int] = {}
    for fixture_file in sorted(_SOFASCORE_FIXTURES.glob("fixtures_*.json")):
        try:
            with open(fixture_file) as f:
                fixtures = json.load(f)
            if not isinstance(fixtures, list):
                continue
            for m in fixtures:
                round_info = m.get("roundInfo", {})
                mw = round_info.get("round") if isinstance(round_info, dict) else None
                if mw is None:
                    continue
                home_raw = m.get("homeTeam", {}).get("name", "")
                away_raw = m.get("awayTeam", {}).get("name", "")
                # Normalize team names (Sofascore uses "Hellas Verona" etc.)
                home = normalize_team(home_raw) if normalize_team else home_raw
                away = normalize_team(away_raw) if normalize_team else away_raw
                # Key by home_team + away_team (season-unique fixture)
                sofascore_lookup[(home, away)] = int(mw)
        except Exception as e:
            log.debug("Failed to load Sofascore fixtures %s: %s", fixture_file.name, e)

    if sofascore_lookup:
        filled_sofa = 0
        for idx in df.index[nan_mask]:
            home = df.at[idx, "home_team"]
            away = df.at[idx, "away_team"]
            mw = sofascore_lookup.get((home, away))
            if mw is not None:
                df.at[idx, "matchweek"] = mw
                filled_sofa += 1
        if filled_sofa:
            log.info("Filled %d matchweeks from Sofascore fixtures", filled_sofa)
        nan_mask = df["matchweek"].isna()

    # --- Strategy 2: Chronological inference ---
    # For remaining NaN: count each team's matches in date order within the season
    still_missing = nan_mask.sum()
    if still_missing > 0:
        log.info("Inferring %d remaining matchweeks chronologically", still_missing)
        for season in df.loc[nan_mask, "season"].unique():
            season_mask = df["season"] == season
            season_df = df[season_mask].sort_values("match_date").reset_index()

            # Count matches per team in chronological order
            team_match_num: dict[str, int] = {}
            for _, row in season_df.iterrows():
                orig_idx = row["index"]
                if pd.notna(df.at[orig_idx, "matchweek"]):
                    # Already has a matchweek — still count for team tracking
                    team_match_num[row["home_team"]] = team_match_num.get(row["home_team"], 0) + 1
                    team_match_num[row["away_team"]] = team_match_num.get(row["away_team"], 0) + 1
                    continue
                h = row["home_team"]
                a = row["away_team"]
                team_match_num[h] = team_match_num.get(h, 0) + 1
                team_match_num[a] = team_match_num.get(a, 0) + 1
                # Use the average of both teams' match counts
                inferred_mw = round((team_match_num[h] + team_match_num[a]) / 2)
                df.at[orig_idx, "matchweek"] = inferred_mw

        filled_chrono = still_missing - df["matchweek"].isna().sum()
        if filled_chrono:
            log.info("Filled %d matchweeks via chronological inference", filled_chrono)

    remaining = df["matchweek"].isna().sum()
    if remaining:
        log.warning("Still %d rows with NaN matchweek after all strategies", remaining)

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

    # Fill any NaN matchweeks before saving (matches table only)
    if table_name == "matches" and "matchweek" in df.columns:
        df = _fill_missing_matchweeks(df)

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
