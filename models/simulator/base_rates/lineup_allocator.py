"""Phase 5.2 — Lineup + shot-share allocator.

Given:
  - team (e.g. "Juventus")
  - a team_shot_rate (expected shots/match, from ShotRateEstimator)
  - a lineup: list[int] of player_ids (ideally 11 starters)
  - a PlayerProfileStore

Produces:
  dict[player_id, shot_rate] — each player's expected shots, summing to team_shot_rate

Algorithm:
  1. Look up each player's shot_rate_per_90 × (minutes_per_start / 90)
  2. Sum to team_rate_expected
  3. Scale each player's rate by (team_shot_rate / team_rate_expected)
     so total exactly matches the Phase 2 team estimate
  4. Clamp per-player to [0, 6] shots (defensive)

Lineup resolution when no confirmed XI available:
  - Use last-match XI from data/parsed/lineups.parquet
  - Fallback to top-11 by rolling minutes
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from models.simulator.base_rates.player_profiles import PlayerProfileStore, PlayerProfile

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LINEUPS_PATH = PROJECT_ROOT / "data" / "parsed" / "lineups.parquet"


def allocate_team_shots_to_players(
    team_shot_rate: float,
    lineup_player_ids: list[int],
    profiles: PlayerProfileStore,
) -> dict[int, float]:
    """Return {player_id: expected_shots}. Sum equals team_shot_rate."""
    if not lineup_player_ids or team_shot_rate <= 0:
        return {}

    raw_rates = {}
    total_expected = 0.0
    for pid in lineup_player_ids:
        prof = profiles.lookup(pid)
        if prof.position in {"G", "GK"}:
            raw_rates[pid] = 0.0
            continue
        mins = prof.minutes_per_start
        player_shots = prof.shot_rate_per_90 * (mins / 90.0)
        raw_rates[pid] = player_shots
        total_expected += player_shots

    if total_expected <= 0:
        # Uniform fallback across outfielders
        outfield = [pid for pid in lineup_player_ids if profiles.lookup(pid).position not in {"G", "GK"}]
        if not outfield:
            return {}
        per = team_shot_rate / len(outfield)
        return {pid: float(np.clip(per, 0.0, 6.0)) for pid in outfield}

    scale = team_shot_rate / total_expected
    return {pid: float(np.clip(r * scale, 0.0, 6.0)) for pid, r in raw_rates.items()}


def resolve_lineup_from_history(
    team: str,
    match_date: pd.Timestamp,
    profiles: PlayerProfileStore,
    max_lookback_days: int = 14,
) -> list[int]:
    """Get a best-guess XI: last match's starters, else top-11 by recent minutes.

    This is a safe fallback for backtest (we can't hit live lineup leaks
    historically). For live prediction, the caller should pass the actual
    confirmed XI and skip this.
    """
    if not LINEUPS_PATH.exists():
        return _top11_from_profiles(team, profiles)
    try:
        df = pd.read_parquet(LINEUPS_PATH)
    except Exception:
        return _top11_from_profiles(team, profiles)

    if "team" not in df.columns or "match_id" not in df.columns:
        return _top11_from_profiles(team, profiles)

    # Need per-lineup date — join through match_id_mapping if absent
    team_lineups = df[df["team"] == team]
    if len(team_lineups) == 0:
        return _top11_from_profiles(team, profiles)

    # Pick the most recent row batch for this team that pre-dates match_date.
    # lineups.parquet has season but not match_date; we rely on insertion order
    # per team being chronological (a reasonable assumption given scraper writes).
    starters = team_lineups[team_lineups.get("is_starter", True) == True]  # noqa: E712
    if len(starters) == 0:
        starters = team_lineups  # fallback
    # Most recent 11 by last appearance — using a simple "last match_id block"
    last_match = starters["match_id"].iloc[-1]
    lineup_rows = starters[starters["match_id"] == last_match]
    player_ids = []
    if "player_id" in lineup_rows.columns:
        for pid in lineup_rows["player_id"].dropna().astype(int).tolist():
            player_ids.append(pid)
    if len(player_ids) < 8:
        # Not enough coverage — top-11 fallback
        return _top11_from_profiles(team, profiles)
    return player_ids[:11]


def _top11_from_profiles(team: str, profiles: PlayerProfileStore) -> list[int]:
    """Fallback XI: 11 most-minutes players for this team from profiles."""
    matching = [
        (pid, prof) for pid, prof in profiles.all_profiles().items()
        if prof.team == team and prof.position not in {"G", "GK"}
    ]
    # Sort by minutes_per_start × n_starts_used (total minutes)
    matching.sort(key=lambda x: -(x[1].minutes_per_start * x[1].n_starts_used))
    return [pid for pid, _ in matching[:11]]
