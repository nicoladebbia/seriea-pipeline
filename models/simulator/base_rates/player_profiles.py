"""Phase 5.1 — Per-player profile extractor for player-prop simulation.

For each player, rolling last-N-starts profile:
  shot_rate_per_90       — expected shots per 90 minutes when starting
  sot_per_shot           — SOT / total_shots empirical ratio
  goals_per_sot          — goals / SOT empirical ratio (conversion rate)
  xg_per_shot            — mean xG per shot (quality proxy)
  foul_rate_per_90       — expected fouls conceded per 90 (→ carding)
  minutes_per_start      — expected minutes when starting (ceiling 90)

Missing data fallback: positional prior
  - STRIKER/Forward: shot_rate 2.5, sot_per_shot 0.40, goals_per_sot 0.28
  - Winger (W): shot_rate 1.8, 0.35, 0.18
  - Midfielder (M): shot_rate 0.9, 0.30, 0.12
  - Defender (D): shot_rate 0.3, 0.28, 0.08
  - Goalkeeper (G): skipped (no shot markets)

Written as a self-contained class so Phase 5.5 SimulatorPredictor can call
  profiles.fit(train_df_with_player_match_stats)
  profiles.lookup(player_id) -> PlayerProfile
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PLAYER_STATS_PATH = PROJECT_ROOT / "data" / "external" / "sofascore" / "player_match_stats.parquet"


# Positional priors (Serie A defaults, empirically similar across European leagues)
POSITIONAL_PRIORS: dict[str, "PlayerProfile"] = {}


@dataclass(frozen=True)
class PlayerProfile:
    player_id: int
    player_name: str
    position: str
    team: str | None
    n_starts_used: int
    shot_rate_per_90: float
    sot_per_shot: float
    goals_per_sot: float
    xg_per_shot: float
    foul_rate_per_90: float
    minutes_per_start: float
    is_fallback: bool = False

    def is_outfielder(self) -> bool:
        return self.position not in {"G", "Goalkeeper"}


def _positional_prior(position: str, player_id: int = 0, player_name: str = "", team: str | None = None) -> PlayerProfile:
    position = (position or "").upper()[:1]
    defaults = {
        "F": (2.5, 0.40, 0.28, 0.12, 1.0),   # Forward
        "M": (0.9, 0.30, 0.12, 0.06, 1.8),   # Midfielder
        "D": (0.3, 0.28, 0.08, 0.03, 1.5),   # Defender
        "G": (0.0, 0.0, 0.0, 0.0, 0.1),      # Goalkeeper
    }
    if position == "W":
        shot, sot, goals, xg, foul = (1.8, 0.35, 0.18, 0.08, 1.3)
    else:
        shot, sot, goals, xg, foul = defaults.get(position, (0.5, 0.30, 0.10, 0.05, 1.2))
    return PlayerProfile(
        player_id=int(player_id),
        player_name=player_name or f"prior_{position}",
        position=position or "M",
        team=team,
        n_starts_used=0,
        shot_rate_per_90=shot,
        sot_per_shot=sot,
        goals_per_sot=goals,
        xg_per_shot=xg,
        foul_rate_per_90=foul,
        minutes_per_start=85.0,  # typical minutes for a start (subs at ~70-85)
        is_fallback=True,
    )


class PlayerProfileStore:
    """Builds + caches PlayerProfile per player_id from rolling last-N starts."""

    MIN_STARTS_FOR_PROFILE = 3
    ROLLING_WINDOW = 10

    def __init__(self, min_starts: int = 3, rolling_window: int = 10):
        self.min_starts = min_starts
        self.rolling_window = rolling_window
        self._profiles: dict[int, PlayerProfile] = {}
        self._position_by_player: dict[int, str] = {}
        self._team_by_player: dict[int, str] = {}
        self._name_by_player: dict[int, str] = {}

    def fit(self, player_match_stats: pd.DataFrame, as_of_date: pd.Timestamp | None = None) -> None:
        """Compute profiles from player_match_stats up to as_of_date.

        `as_of_date` = None means use ALL rows (OK for backtest holdouts where
        the caller has already filtered by season boundary). For live
        predictions, pass the match date.
        """
        df = player_match_stats.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if as_of_date is not None:
                df = df[df["date"] < pd.to_datetime(as_of_date)]
        if len(df) == 0:
            log.warning("PlayerProfileStore: empty training data")
            return

        # Filter to starters with meaningful minutes
        starters = df[(df["is_starter"] == True) & (df["minutes"] >= 45)]  # noqa: E712
        if len(starters) == 0:
            log.warning("PlayerProfileStore: no starters with minutes>=45")
            return

        # Sort chronologically per player, take last N starts
        starters = starters.sort_values(["player_id", "date"])
        counts = starters.groupby("player_id", observed=True).size()

        profiles: dict[int, PlayerProfile] = {}

        for player_id, sub in starters.groupby("player_id", observed=True, sort=False):
            n = min(len(sub), self.rolling_window)
            recent = sub.tail(n)
            n_starts = int(len(recent))

            # Basic descriptors
            position = str(recent["position"].mode().iloc[0]) if len(recent["position"].mode()) > 0 else "M"
            team = str(recent["team"].iloc[-1])
            name = str(recent["player_name"].iloc[-1])
            self._position_by_player[int(player_id)] = position
            self._team_by_player[int(player_id)] = team
            self._name_by_player[int(player_id)] = name

            if n_starts < self.min_starts:
                # Too few starts — positional prior with player meta
                profiles[int(player_id)] = _positional_prior(position, int(player_id), name, team)
                continue

            total_mins = recent["minutes"].sum()
            if total_mins < 90:  # Essentially no data
                profiles[int(player_id)] = _positional_prior(position, int(player_id), name, team)
                continue

            total_shots = recent["total_shots"].fillna(0).sum()
            total_sot = recent["shots_on_target"].fillna(0).sum()
            total_goals = recent["goals"].fillna(0).sum()
            total_xg = recent["xg"].fillna(0).sum()
            total_fouls = recent["fouls"].fillna(0).sum() if "fouls" in recent.columns else 0.0

            shot_rate_90 = float(total_shots / total_mins * 90.0)
            sot_per_shot = float(total_sot / total_shots) if total_shots > 0 else 0.30
            goals_per_sot = float(total_goals / total_sot) if total_sot > 0 else 0.15
            xg_per_shot = float(total_xg / total_shots) if total_shots > 0 else 0.08
            foul_rate_90 = float(total_fouls / total_mins * 90.0)
            mins_per_start = float(total_mins / n_starts)

            # Defensive clamps (no player has shot_rate > 6 per 90 in Serie A)
            shot_rate_90 = float(np.clip(shot_rate_90, 0.0, 6.0))
            sot_per_shot = float(np.clip(sot_per_shot, 0.10, 0.60))
            goals_per_sot = float(np.clip(goals_per_sot, 0.02, 0.50))
            xg_per_shot = float(np.clip(xg_per_shot, 0.01, 0.35))

            profiles[int(player_id)] = PlayerProfile(
                player_id=int(player_id),
                player_name=name,
                position=position,
                team=team,
                n_starts_used=n_starts,
                shot_rate_per_90=shot_rate_90,
                sot_per_shot=sot_per_shot,
                goals_per_sot=goals_per_sot,
                xg_per_shot=xg_per_shot,
                foul_rate_per_90=foul_rate_90,
                minutes_per_start=min(mins_per_start, 95.0),
                is_fallback=False,
            )

        self._profiles = profiles
        log.info("PlayerProfileStore: fit %d player profiles (%d with real data, %d fallback)",
                 len(profiles),
                 sum(1 for p in profiles.values() if not p.is_fallback),
                 sum(1 for p in profiles.values() if p.is_fallback))

    def lookup(self, player_id: int, fallback_position: str | None = None) -> PlayerProfile:
        pid = int(player_id)
        if pid in self._profiles:
            return self._profiles[pid]
        # Unknown player — use positional prior
        pos = fallback_position or self._position_by_player.get(pid, "M")
        return _positional_prior(pos, pid)

    def all_profiles(self) -> dict[int, PlayerProfile]:
        return dict(self._profiles)

    @property
    def n_profiles(self) -> int:
        return len(self._profiles)


def load_player_match_stats() -> pd.DataFrame | None:
    if not PLAYER_STATS_PATH.exists():
        return None
    return pd.read_parquet(PLAYER_STATS_PATH)
