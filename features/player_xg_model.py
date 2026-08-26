#!/usr/bin/env python3
"""Player-Level xG Modeling - Phase 2 Implementation

Creates individual player xG profiles and lineup-based team xG predictions.

Key features:
1. Player xG/90 profiles - career and recent form
2. Lineup-based team xG aggregation
3. Key player absence impact modeling
4. Player vs opponent strength adjustments

This enables more accurate xG predictions by:
- Accounting for which specific players are playing
- Modeling the impact of injuries/suspensions
- Identifying team dependency on star players
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.paths import parsed_path
from config.settings import DATA_DIR
from config.team_names import normalize_team
from scraper.lineup_fetcher import normalize_player_name

log = logging.getLogger(__name__)


class PlayerXGProfile:
    """Individual player xG profile with career and form stats."""

    def __init__(self, player_name: str, team: str):
        self.player_name = player_name
        self.team = team
        self.position: Optional[str] = None
        self.matches_played = 0
        self.total_minutes = 0
        self.total_xg = 0.0
        self.total_xa = 0.0
        self.total_goals = 0
        self.total_assists = 0
        self.recent_matches: List[Dict] = []  # Last 10 matches

    @property
    def xg_per_90(self) -> float:
        """Expected goals per 90 minutes."""
        if self.total_minutes < 90:
            return 0.0
        return (self.total_xg / self.total_minutes) * 90

    @property
    def xa_per_90(self) -> float:
        """Expected assists per 90 minutes."""
        if self.total_minutes < 90:
            return 0.0
        return (self.total_xa / self.total_minutes) * 90

    @property
    def xg_xa_per_90(self) -> float:
        """Combined xG+xA per 90."""
        return self.xg_per_90 + self.xa_per_90

    @property
    def recent_form_xg(self) -> float:
        """xG/90 over last 5 matches with exponential recency weighting.

        Most recent match gets weight 5, then 4, 3, 2, 1.
        This captures hot/cold streaks better than equal weighting.
        """
        if len(self.recent_matches) < 3:
            return self.xg_per_90

        recent = self.recent_matches[-5:]
        n = len(recent)

        # Exponential decay weights: most recent = highest
        weights = list(range(1, n + 1))  # [1, 2, 3, 4, 5]
        total_weighted_xg = 0.0
        total_weighted_mins = 0.0

        for i, m in enumerate(recent):
            mins = m.get("minutes", 0)
            xg = m.get("xg", 0)
            w = weights[i]
            total_weighted_xg += xg * w
            total_weighted_mins += mins * w

        if total_weighted_mins < 200:
            return self.xg_per_90
        form_xg = (total_weighted_xg / total_weighted_mins) * 90
        # Cap at 3x career average to prevent tiny-sample outliers
        career = self.xg_per_90
        if career > 0:
            form_xg = min(form_xg, career * 3.0)
        return form_xg

    @property
    def recent_form_xa(self) -> float:
        """xA/90 over last 5 matches with exponential recency weighting."""
        if len(self.recent_matches) < 3:
            return self.xa_per_90

        recent = self.recent_matches[-5:]
        n = len(recent)
        weights = list(range(1, n + 1))
        total_weighted_xa = 0.0
        total_weighted_mins = 0.0

        for i, m in enumerate(recent):
            mins = m.get("minutes", 0)
            xa = m.get("xa", 0)
            w = weights[i]
            total_weighted_xa += xa * w
            total_weighted_mins += mins * w

        if total_weighted_mins < 200:
            return self.xa_per_90
        form_xa = (total_weighted_xa / total_weighted_mins) * 90
        # Cap at 3x career average to prevent tiny-sample outliers
        career = self.xa_per_90
        if career > 0:
            form_xa = min(form_xa, career * 3.0)
        return form_xa

    def add_match(self, match_data: Dict):
        """Add a match to the player's profile."""
        self.matches_played += 1
        minutes = match_data.get("minutes", 0)
        self.total_minutes += minutes
        self.total_xg += match_data.get("xg", 0)
        self.total_xa += match_data.get("xa", match_data.get("xg_assist", 0))
        self.total_goals += match_data.get("goals", 0)
        self.total_assists += match_data.get("assists", 0)

        # Keep last 10 matches for form
        self.recent_matches.append({
            "match_id": match_data.get("match_id"),
            "minutes": minutes,
            "xg": match_data.get("xg", 0),
            "xa": match_data.get("xa", match_data.get("xg_assist", 0)),
            "goals": match_data.get("goals", 0),
        })
        if len(self.recent_matches) > 10:
            self.recent_matches.pop(0)

    def infer_position(self) -> str:
        """Infer position from xG/xA patterns."""
        if self.total_minutes < 90:
            return "SUB"
        xg90 = self.xg_per_90
        xa90 = self.xa_per_90
        ga90 = xg90 + xa90
        if xg90 == 0 and xa90 == 0 and self.total_minutes > 2500:
            return "GK"
        if xg90 > 0.35:
            return "ST"
        if xg90 > 0.15 or ga90 > 0.4:
            return "AM"
        if ga90 > 0.15:
            return "CM"
        return "CB"

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        pos = self.position if self.position else self.infer_position()
        return {
            "player_name": self.player_name,
            "team": self.team,
            "position": pos,
            "matches_played": self.matches_played,
            "total_minutes": self.total_minutes,
            "total_xg": round(self.total_xg, 3),
            "total_xa": round(self.total_xa, 3),
            "xg_per_90": round(self.xg_per_90, 3),
            "xa_per_90": round(self.xa_per_90, 3),
            "xg_xa_per_90": round(self.xg_xa_per_90, 3),
            "recent_form_xg": round(self.recent_form_xg, 3),
            "recent_form_xa": round(self.recent_form_xa, 3),
        }


class PlayerXGDatabase:
    """Database of all player xG profiles."""

    def __init__(self):
        self.profiles: Dict[str, PlayerXGProfile] = {}  # key: "player_name|team"
        self.team_profiles: Dict[str, Dict[str, PlayerXGProfile]] = defaultdict(dict)
        self.loaded = False

    def _player_key(self, player: str, team: str) -> str:
        return f"{player}|{team}"

    def get_profile(self, player: str, team: str) -> Optional[PlayerXGProfile]:
        """Get a player's profile."""
        key = self._player_key(player, team)
        return self.profiles.get(key)

    def get_or_create(self, player: str, team: str) -> PlayerXGProfile:
        """Get or create a player profile."""
        key = self._player_key(player, team)
        if key not in self.profiles:
            self.profiles[key] = PlayerXGProfile(player, team)
            self.team_profiles[team][player] = self.profiles[key]
        return self.profiles[key]

    def build_from_match_data(self, player_stats_path: Path = None) -> bool:
        """Build profiles from match-level player stats."""
        if player_stats_path is None:
            player_stats_path = parsed_path("player_stats")

        if not player_stats_path.exists():
            log.warning(f"Player stats not found: {player_stats_path}")
            return False

        df = pd.read_parquet(player_stats_path)
        log.info(f"Building player profiles from {len(df):,} records")

        # Convert numeric columns
        for col in ["minutes", "goals", "assists", "xg", "npxg", "xg_assist"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Sort by match to ensure chronological order
        if "match_id" in df.columns:
            df = df.sort_values("match_id")

        for _, row in df.iterrows():
            player = row.get("player")
            team = row.get("team")

            if not player or not team or pd.isna(player):
                continue

            profile = self.get_or_create(player, team)
            profile.add_match({
                "match_id": row.get("match_id"),
                "minutes": row.get("minutes", 0),
                "xg": row.get("xg", 0),
                "xa": row.get("xg_assist", 0),
                "goals": row.get("goals", 0),
                "assists": row.get("assists", 0),
            })

        self.loaded = True
        log.info(f"Built {len(self.profiles)} player profiles for {len(self.team_profiles)} teams")
        self.validate_against_squads()
        return True

    def get_team_top_contributors(self, team: str, n: int = 5, metric: str = "xg_xa_per_90") -> List[Tuple[str, float]]:
        """Get top N contributors for a team by specified metric."""
        team_players = self.team_profiles.get(team, {})
        if not team_players:
            return []

        scored = []
        for player, profile in team_players.items():
            if profile.total_minutes >= 270:  # At least 3 full matches
                if metric == "xg_per_90":
                    score = profile.xg_per_90
                elif metric == "xa_per_90":
                    score = profile.xa_per_90
                else:
                    score = profile.xg_xa_per_90
                scored.append((player, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)[:n]

    def get_team_total_xg_capacity(self, team: str) -> float:
        """Get total team xG capacity (sum of all player xG/90)."""
        team_players = self.team_profiles.get(team, {})
        return sum(p.xg_per_90 for p in team_players.values() if p.total_minutes >= 180)

    def save(self, path: Path = None):
        """Save profiles to JSON."""
        if path is None:
            path = DATA_DIR / "features" / "player_xg_profiles.json"

        data = {
            key: profile.to_dict()
            for key, profile in self.profiles.items()
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        log.info(f"Saved {len(data)} player profiles to {path}")

    def load(self, path: Path = None) -> bool:
        """Load profiles from JSON."""
        if path is None:
            path = DATA_DIR / "features" / "player_xg_profiles.json"

        if not path.exists():
            return False

        with open(path) as f:
            data = json.load(f)

        for key, profile_data in data.items():
            player = profile_data["player_name"]
            team = profile_data["team"]
            profile = self.get_or_create(player, team)
            profile.matches_played = profile_data["matches_played"]
            profile.total_minutes = profile_data["total_minutes"]
            profile.total_xg = profile_data["total_xg"]
            profile.total_xa = profile_data["total_xa"]
            stored_pos = profile_data.get("position")
            if stored_pos and stored_pos not in (None, "None", "SUB"):
                profile.position = stored_pos

        self.loaded = True
        log.info(f"Loaded {len(self.profiles)} player profiles")
        self.validate_against_squads()
        return True


    def build_from_sofascore(self, sofascore_path: Path = None) -> bool:
        """Build/enrich profiles from SofaScore per-match player stats.

        SofaScore has much better xG coverage for recent seasons (2022+)
        and provides position data (G/D/M/F), xG, xA, shots, rating.
        This enriches the FBref-based profiles with more recent/complete data.
        """
        if sofascore_path is None:
            sofascore_path = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"

        if not sofascore_path.exists():
            log.warning("SofaScore data not found: %s", sofascore_path)
            return False

        df = pd.read_parquet(sofascore_path)
        log.info("Enriching player profiles from %d SofaScore records", len(df))

        # Convert numeric columns
        for col in ["minutes", "goals", "assists", "xg", "xa", "total_shots",
                     "rating", "key_passes", "is_starter"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Sort by date for chronological processing
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")

        # Position mapping (SofaScore G/D/M/F → our position labels)
        pos_map = {"F": "ST", "M": "CM", "D": "CB", "G": "GK"}

        added = 0
        for _, row in df.iterrows():
            player = row.get("player_name")
            team = row.get("team")
            if not player or not team or pd.isna(player):
                continue

            # Normalize team name using canonical mapping
            team_norm = normalize_team(team)

            profile = self.get_or_create(player, team_norm)

            # Set position from SofaScore if not already set
            ss_pos = row.get("position", "")
            if ss_pos and (not profile.position or profile.position in (None, "SUB")):
                profile.position = pos_map.get(ss_pos, ss_pos)

            # Add match data (xG from SofaScore is more reliable for recent seasons)
            xg_val = row.get("xg", 0)
            xa_val = row.get("xa", 0)

            # Only add if this match has meaningful data
            if row.get("minutes", 0) > 0:
                profile.add_match({
                    "match_id": row.get("match_id"),
                    "minutes": row.get("minutes", 0),
                    "xg": xg_val if not pd.isna(xg_val) else 0,
                    "xa": xa_val if not pd.isna(xa_val) else 0,
                    "goals": row.get("goals", 0),
                    "assists": row.get("assists", 0),
                })
                added += 1

        log.info("Added %d SofaScore match records to player profiles "
                 "(%d total profiles, %d teams)",
                 added, len(self.profiles), len(self.team_profiles))
        self.loaded = True
        return True

    def validate_against_squads(self, squads_path: Path = None) -> int:
        """Remove profiles for players no longer at their listed team.

        Cross-references current squad rosters to prune departed players.

        Args:
            squads_path: Path to current_squads.json. Uses default if None.

        Returns:
            Number of profiles removed.
        """
        if squads_path is None:
            squads_path = DATA_DIR / "squads" / "current_squads.json"

        if not squads_path.exists():
            return 0

        try:
            with open(squads_path) as f:
                squad_data = json.load(f)
        except Exception:
            return 0

        teams = squad_data.get("teams", {})
        if not teams:
            return 0

        # Build set of (normalized_name, team) for current players
        current_players = set()
        for team_name, team_data in teams.items():
            for player in team_data.get("players", []):
                norm = normalize_player_name(player.get("name", ""))
                if norm:
                    current_players.add((norm, team_name))

        # Find profiles to remove
        to_remove = []
        for key, profile in self.profiles.items():
            team = profile.team
            # Only validate teams that appear in squad data
            if team not in teams:
                continue
            norm = normalize_player_name(profile.player_name)
            if (norm, team) not in current_players:
                to_remove.append(key)

        # Remove departed players
        for key in to_remove:
            profile = self.profiles.pop(key)
            team_dict = self.team_profiles.get(profile.team, {})
            team_dict.pop(profile.player_name, None)

        if to_remove:
            log.info(f"Removed {len(to_remove)} departed player profiles from xG database")

        return len(to_remove)


class LineupXGPredictor:
    """Predicts team xG based on expected lineup.

    Improvements over v1:
    - Learned team-level scaling factor (replaces hardcoded 0.65)
    - Understat xG_chain integration (captures build-up play, not just finishing)
    - Opponent defensive strength adjustment
    """

    # Understat team_id -> our team_name mapping
    UNDERSTAT_TEAM_MAP = {
        111: "Milan", 107: "Atalanta", 222: "Benevento", 97: "Bologna",
        242: "Brescia", 116: "Cagliari", 109: "Chievo", 286: "Como",
        272: "Cremonese", 115: "Crotone", 108: "Empoli", 110: "Fiorentina",
        112: "Frosinone", 101: "Genoa", 106: "Inter", 98: "Juventus",
        96: "Lazio", 243: "Lecce", 271: "Monza", 105: "Napoli",
        230: "Parma", 95: "Roma", 221: "SPAL", 264: "Salernitana",
        102: "Sampdoria", 104: "Sassuolo", 260: "Spezia", 113: "Torino",
        99: "Udinese", 265: "Venezia", 94: "Verona",
    }

    def __init__(self, player_db: PlayerXGDatabase):
        self.player_db = player_db
        self.team_scaling_factors: Dict[str, float] = {}
        self.team_xg_chain_per_90: Dict[str, float] = {}
        self.team_xg_conceded_per_90: Dict[str, float] = {}
        self.league_avg_xg_per_90 = 1.38  # Serie A average

        self._load_understat_data()
        self._compute_team_scaling_factors()

    def _load_understat_data(self):
        """Load understat xG_chain data and team defensive stats."""
        try:
            us_path = DATA_DIR / "understat" / "all_player_stats.parquet"
            if not us_path.exists():
                return

            df = pd.read_parquet(us_path)
            # Use most recent season available
            latest_season = df["season_code"].max()
            season_df = df[df["season_code"] == latest_season].copy()

            # Map team_id -> team_name (UNDERSTAT_TEAM_MAP is already id->name)
            id_to_name = self.UNDERSTAT_TEAM_MAP

            # --- Load actual team xG and xG conceded from schedules ---
            # (xG_chain sums across players double-count because multiple
            #  players participate in each possession chain, inflating by ~3x.
            #  Use actual team xG from schedules instead.)
            sched_path = DATA_DIR / "understat" / "all_schedules.parquet"
            if sched_path.exists():
                sched = pd.read_parquet(sched_path)
                sched_season = sched[sched["season_code"] == latest_season]
                for raw_name in sched_season["home_team"].unique():
                    team_name = normalize_team(raw_name)
                    home_matches = sched_season[sched_season["home_team"] == raw_name]
                    away_matches = sched_season[sched_season["away_team"] == raw_name]
                    n_matches = len(home_matches) + len(away_matches)
                    if n_matches > 0:
                        total_xg_for = home_matches["home_xg"].sum() + away_matches["away_xg"].sum()
                        self.team_xg_chain_per_90[team_name] = total_xg_for / n_matches
                        total_conceded = home_matches["away_xg"].sum() + away_matches["home_xg"].sum()
                        self.team_xg_conceded_per_90[team_name] = total_conceded / n_matches

            if self.team_xg_chain_per_90:
                log.info(f"Loaded understat xG_chain for {len(self.team_xg_chain_per_90)} teams")

        except Exception as e:
            log.debug(f"Could not load understat data: {e}")

    def _compute_team_scaling_factors(self):
        """Compute learned team-level xG scaling factors.

        For each team: actual_team_xg_per_match / sum_of_player_xg_per_90
        This replaces the hardcoded 0.65 with team-specific values.
        """
        try:
            sched_path = DATA_DIR / "understat" / "all_schedules.parquet"
            if not sched_path.exists():
                return

            sched = pd.read_parquet(sched_path)
            latest_season = sched["season_code"].max()
            season_sched = sched[sched["season_code"] == latest_season]

            for raw_name in season_sched["home_team"].unique():
                team_name = normalize_team(raw_name)
                # Actual team xG per match
                home_xg = season_sched[season_sched["home_team"] == raw_name]["home_xg"].sum()
                away_xg = season_sched[season_sched["away_team"] == raw_name]["away_xg"].sum()
                n_matches = len(season_sched[season_sched["home_team"] == raw_name]) + \
                            len(season_sched[season_sched["away_team"] == raw_name])
                if n_matches == 0:
                    continue
                actual_xg_per_match = (home_xg + away_xg) / n_matches

                # Sum of player xG/90 from our database
                player_sum = self.player_db.get_team_total_xg_capacity(team_name)
                if player_sum > 0.5:  # Need meaningful player data
                    factor = actual_xg_per_match / player_sum
                    # Clamp to reasonable range (0.3 to 1.2)
                    self.team_scaling_factors[team_name] = max(0.3, min(1.2, factor))

            if self.team_scaling_factors:
                avg = np.mean(list(self.team_scaling_factors.values()))
                log.info(f"Computed scaling factors for {len(self.team_scaling_factors)} teams "
                         f"(avg={avg:.3f}, range={min(self.team_scaling_factors.values()):.3f}-"
                         f"{max(self.team_scaling_factors.values()):.3f})")

        except Exception as e:
            log.debug(f"Could not compute scaling factors: {e}")

    def predict_team_xg(
        self,
        team: str,
        lineup: List[str] = None,
        missing_players: List[str] = None,
        use_form: bool = True,
        opponent: str = None,
    ) -> Dict:
        """Predict team xG based on lineup.

        Args:
            team: Team name
            lineup: List of player names in expected lineup (11 players)
            missing_players: List of players known to be unavailable
            use_form: Use recent form instead of career averages
            opponent: Opponent team name (for defensive strength adjustment)

        Returns:
            Dict with predicted_xg, predicted_xa, player_contributions, etc.
        """
        team_players = self.player_db.team_profiles.get(team, {})

        if not team_players:
            return {
                "predicted_xg": 1.4,
                "predicted_xa": 0.4,
                "confidence": "low",
                "method": "league_average",
            }

        # Get top contributors if no lineup provided
        if lineup is None:
            top_players = self.player_db.get_team_top_contributors(team, n=11)
            lineup = [p[0] for p in top_players]

        # Remove missing players
        if missing_players:
            lineup = [p for p in lineup if p not in missing_players]

        # Calculate xG contribution from lineup
        total_xg = 0.0
        total_xa = 0.0
        player_contributions = []

        for player in lineup:
            profile = self.player_db.get_profile(player, team)

            if profile and profile.total_minutes >= 180:
                if use_form:
                    xg = profile.recent_form_xg
                    xa = profile.recent_form_xa
                else:
                    xg = profile.xg_per_90
                    xa = profile.xa_per_90

                total_xg += xg
                total_xa += xa
                player_contributions.append({
                    "player": player,
                    "xg_90": round(xg, 3),
                    "xa_90": round(xa, 3),
                })

        if len(player_contributions) >= 7:
            # Use learned team-specific scaling factor (fallback to 0.65)
            scale_factor = self.team_scaling_factors.get(team, 0.65)
            predicted_xg = total_xg * scale_factor
            predicted_xa = total_xa * scale_factor

            # Blend with xG_chain signal (60% player xG/90, 40% xG_chain)
            xg_chain = self.team_xg_chain_per_90.get(team)
            if xg_chain is not None:
                predicted_xg = 0.6 * predicted_xg + 0.4 * xg_chain

            confidence = "high" if len(player_contributions) >= 10 else "medium"
        else:
            team_avg = self._get_team_historical_avg(team)
            predicted_xg = team_avg.get("xg", 1.4)
            predicted_xa = team_avg.get("xa", 0.4)
            confidence = "low"

        # Opponent strength adjustment
        if opponent and self.team_xg_conceded_per_90:
            opp_conceded = self.team_xg_conceded_per_90.get(opponent)
            if opp_conceded is not None and self.league_avg_xg_per_90 > 0:
                opp_def_strength = opp_conceded / self.league_avg_xg_per_90
                # Clamp to avoid extreme adjustments
                opp_def_strength = max(0.7, min(1.4, opp_def_strength))
                predicted_xg *= opp_def_strength

        # Clip to reasonable ranges
        predicted_xg = max(0.5, min(3.5, predicted_xg))
        predicted_xa = max(0.1, min(1.5, predicted_xa))

        return {
            "predicted_xg": round(predicted_xg, 2),
            "predicted_xa": round(predicted_xa, 2),
            "total_raw_xg": round(total_xg, 2),
            "total_raw_xa": round(total_xa, 2),
            "players_used": len(player_contributions),
            "confidence": confidence,
            "method": "lineup_based",
            "top_contributors": sorted(
                player_contributions,
                key=lambda x: x["xg_90"] + x["xa_90"],
                reverse=True
            )[:5],
        }

    def estimate_absence_impact(
        self,
        team: str,
        missing_players: List[str],
    ) -> Dict:
        """Estimate impact of missing players on team xG.

        Returns:
            Dict with xg_reduction, xa_reduction, impact_level
        """
        if not missing_players:
            return {"xg_reduction": 0, "xa_reduction": 0, "impact_level": "none"}

        total_xg_loss = 0.0
        total_xa_loss = 0.0
        player_impacts = []

        team_total = self.player_db.get_team_total_xg_capacity(team)

        for player in missing_players:
            profile = self.player_db.get_profile(player, team)
            if profile and profile.total_minutes >= 180:
                xg_loss = profile.xg_per_90
                xa_loss = profile.xa_per_90

                # Calculate player's share of team xG
                share = xg_loss / team_total if team_total > 0 else 0

                total_xg_loss += xg_loss
                total_xa_loss += xa_loss
                player_impacts.append({
                    "player": player,
                    "xg_loss": round(xg_loss, 3),
                    "xa_loss": round(xa_loss, 3),
                    "team_share": round(share, 3),
                })

        # Impact level
        if total_xg_loss > 0.4:
            impact_level = "critical"
        elif total_xg_loss > 0.25:
            impact_level = "high"
        elif total_xg_loss > 0.1:
            impact_level = "medium"
        else:
            impact_level = "low"

        return {
            "xg_reduction": round(total_xg_loss * 0.5, 3),  # 50% replacement
            "xa_reduction": round(total_xa_loss * 0.5, 3),
            "impact_level": impact_level,
            "missing_players": player_impacts,
        }

    def _get_team_historical_avg(self, team: str) -> Dict:
        """Get team's historical average xG."""
        team_players = self.player_db.team_profiles.get(team, {})
        if not team_players:
            return {"xg": 1.4, "xa": 0.4}

        total_xg = sum(p.total_xg for p in team_players.values())
        total_mins = sum(p.total_minutes for p in team_players.values())

        if total_mins < 900:  # Less than 10 matches worth
            return {"xg": 1.4, "xa": 0.4}

        # Approximate team xG per match
        matches_approx = total_mins / (11 * 90)
        avg_xg = total_xg / matches_approx if matches_approx > 0 else 1.4

        return {"xg": round(avg_xg, 2), "xa": 0.4}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 70)
    print("PLAYER XG MODEL - Phase 2")
    print("=" * 70)

    # Build player database
    player_db = PlayerXGDatabase()
    player_db.build_from_match_data()
    player_db.save()

    # Show top players by xG/90
    print("\n=== Top xG/90 Players (min 500 mins) ===")
    all_profiles = [p for p in player_db.profiles.values() if p.total_minutes >= 500]
    top_by_xg = sorted(all_profiles, key=lambda x: x.xg_per_90, reverse=True)[:20]

    for i, p in enumerate(top_by_xg, 1):
        print(f"{i:2}. {p.player_name:25} ({p.team:15}) - xG/90: {p.xg_per_90:.3f}, xA/90: {p.xa_per_90:.3f}")

    # Test lineup prediction
    print("\n=== Lineup-Based Predictions ===")
    predictor = LineupXGPredictor(player_db)

    for team in ["Inter", "Roma", "Napoli", "Juventus"]:
        pred = predictor.predict_team_xg(team)
        print(f"\n{team}:")
        print(f"  Predicted xG: {pred['predicted_xg']}")
        print(f"  Confidence: {pred['confidence']}")
        if pred.get("top_contributors"):
            print(f"  Top contributors:")
            for tc in pred["top_contributors"][:3]:
                print(f"    - {tc['player']}: xG/90={tc['xg_90']}, xA/90={tc['xa_90']}")

    # Test absence impact
    print("\n=== Absence Impact Test ===")
    # Simulate Lautaro Martinez missing for Inter
    inter_top = player_db.get_team_top_contributors("Inter", n=3)
    if inter_top:
        top_player = inter_top[0][0]
        impact = predictor.estimate_absence_impact("Inter", [top_player])
        print(f"Impact of losing {top_player} for Inter:")
        print(f"  xG reduction: {impact['xg_reduction']}")
        print(f"  Impact level: {impact['impact_level']}")
