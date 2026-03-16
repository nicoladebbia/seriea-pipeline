"""Lineup-level stat aggregation for corners, cards, and fouls.

Aggregates per-player stats from player_stats.parquet into lineup-level
features that can be used by the corners and cards models.

Functions:
    get_lineup_expected_corners(team, lineup=None)
    get_lineup_expected_cards(team, lineup=None)
    get_lineup_foul_rate(team, lineup=None)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from storage.paths import parsed_path

log = logging.getLogger(__name__)


class LineupStatsDatabase:
    """Aggregates per-player stats for lineup-level features."""

    def __init__(self):
        self.player_stats: Dict[str, Dict[str, Dict]] = {}  # team -> player -> stats
        self.team_averages: Dict[str, Dict] = {}  # team -> avg stats
        self.loaded = False

    def load(self) -> bool:
        """Load player_stats.parquet and build per-player aggregates."""
        path = parsed_path("player_stats")
        if not path.exists():
            log.warning(f"player_stats.parquet not found at {path}")
            return False

        df = pd.read_parquet(path)

        # Convert numeric columns
        stat_cols = [
            "passing_types_corner_kicks", "passing_types_corner_kicks_in",
            "passing_types_corner_kicks_out", "passing_types_corner_kicks_straight",
            "misc_cards_yellow", "misc_cards_red", "misc_fouls", "misc_fouled",
            "minutes",
        ]
        for col in stat_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if "minutes" not in df.columns or "team" not in df.columns:
            log.warning("Required columns missing from player_stats.parquet")
            return False

        # Aggregate per player per team (across all matches)
        for (team, player), group in df.groupby(["team", "player"]):
            if pd.isna(player) or pd.isna(team):
                continue

            total_mins = group["minutes"].sum()
            if total_mins < 90:  # Need at least 1 full match
                continue

            stats = {
                "minutes": total_mins,
                "matches": len(group),
            }

            # Corner kicks taken per 90
            if "passing_types_corner_kicks" in group.columns:
                total_corners = group["passing_types_corner_kicks"].sum()
                stats["corners_per_90"] = (total_corners / total_mins) * 90 if total_mins > 0 else 0
                stats["total_corners"] = total_corners

            # Yellow cards per 90
            if "misc_cards_yellow" in group.columns:
                total_yellows = group["misc_cards_yellow"].sum()
                stats["yellows_per_90"] = (total_yellows / total_mins) * 90 if total_mins > 0 else 0
                stats["total_yellows"] = total_yellows

            # Red cards per 90
            if "misc_cards_red" in group.columns:
                total_reds = group["misc_cards_red"].sum()
                stats["reds_per_90"] = (total_reds / total_mins) * 90 if total_mins > 0 else 0

            # Fouls per 90
            if "misc_fouls" in group.columns:
                total_fouls = group["misc_fouls"].sum()
                stats["fouls_per_90"] = (total_fouls / total_mins) * 90 if total_mins > 0 else 0
                stats["total_fouls"] = total_fouls

            if team not in self.player_stats:
                self.player_stats[team] = {}
            self.player_stats[team][player] = stats

        # Compute team averages
        for team, players in self.player_stats.items():
            all_corners = [p["corners_per_90"] for p in players.values() if "corners_per_90" in p]
            all_yellows = [p["yellows_per_90"] for p in players.values() if "yellows_per_90" in p]
            all_fouls = [p["fouls_per_90"] for p in players.values() if "fouls_per_90" in p]

            self.team_averages[team] = {
                "avg_corners_per_90": np.mean(all_corners) if all_corners else 0,
                "avg_yellows_per_90": np.mean(all_yellows) if all_yellows else 0,
                "avg_fouls_per_90": np.mean(all_fouls) if all_fouls else 0,
                "n_players": len(players),
            }

        self.loaded = True
        log.info(f"Loaded lineup stats for {len(self.player_stats)} teams, "
                 f"{sum(len(p) for p in self.player_stats.values())} players")
        return True

    def get_corner_takers(self, team: str, n: int = 2) -> List[Dict]:
        """Identify top N corner kick takers for a team."""
        players = self.player_stats.get(team, {})
        if not players:
            return []

        corner_players = [
            {"player": name, "corners_per_90": stats["corners_per_90"], "total_corners": stats.get("total_corners", 0)}
            for name, stats in players.items()
            if stats.get("corners_per_90", 0) > 0.1  # At least some corners
        ]

        return sorted(corner_players, key=lambda x: x["corners_per_90"], reverse=True)[:n]

    def get_lineup_expected_corners(
        self, team: str, lineup: Optional[List[str]] = None
    ) -> Dict:
        """Estimate expected corners contribution from a lineup.

        If a corner taker is absent (not in lineup), reduce expected corners.
        """
        players = self.player_stats.get(team, {})
        if not players:
            return {"expected_corners_adj": 0.0, "corner_taker_present": True, "confidence": "low"}

        corner_takers = self.get_corner_takers(team, n=2)
        if not corner_takers:
            return {"expected_corners_adj": 0.0, "corner_taker_present": True, "confidence": "low"}

        # Check if primary corner taker is in lineup
        primary_taker = corner_takers[0]["player"]
        corner_taker_present = True

        if lineup:
            # Normalize names for matching
            lineup_lower = [p.lower().strip() for p in lineup]
            primary_lower = primary_taker.lower().strip()
            corner_taker_present = any(primary_lower in l or l in primary_lower for l in lineup_lower)

        # Adjustment: if primary corner taker absent, reduce expected corners
        adj = 0.0
        if not corner_taker_present:
            # Reduction proportional to how dominant they are
            taker_corners = corner_takers[0]["corners_per_90"]
            # Typical team gets ~5 corners/match; taker handles most of them
            adj = -min(1.5, taker_corners * 0.3)  # Cap at -1.5 corners

        return {
            "expected_corners_adj": round(adj, 2),
            "corner_taker_present": corner_taker_present,
            "primary_taker": primary_taker,
            "primary_taker_corners_90": round(corner_takers[0]["corners_per_90"], 2),
            "confidence": "high" if lineup else "medium",
        }

    def get_lineup_expected_cards(
        self, team: str, lineup: Optional[List[str]] = None
    ) -> Dict:
        """Estimate expected yellow cards from a lineup.

        Aggregates yellows/90 for players in the lineup.
        """
        players = self.player_stats.get(team, {})
        if not players:
            avg = self.team_averages.get(team, {}).get("avg_yellows_per_90", 0.18)
            return {
                "expected_yellows": round(avg * 11, 2),
                "lineup_foul_rate": 0.0,
                "confidence": "low",
            }

        if lineup:
            # Sum yellows/90 for lineup players
            total_yellows = 0.0
            matched = 0
            for player_name in lineup:
                pstats = players.get(player_name)
                if pstats and "yellows_per_90" in pstats:
                    total_yellows += pstats["yellows_per_90"]
                    matched += 1

            if matched < 5:
                # Not enough matches, use team average
                avg = self.team_averages.get(team, {}).get("avg_yellows_per_90", 0.18)
                total_yellows = avg * 11
                confidence = "low"
            else:
                # Scale up for unmatched players (assume average)
                if matched < 11:
                    avg = self.team_averages.get(team, {}).get("avg_yellows_per_90", 0.18)
                    total_yellows += avg * (11 - matched)
                confidence = "high" if matched >= 8 else "medium"
        else:
            # Use team average
            avg = self.team_averages.get(team, {}).get("avg_yellows_per_90", 0.18)
            total_yellows = avg * 11
            confidence = "low"

        return {
            "expected_yellows": round(total_yellows, 2),
            "confidence": confidence,
        }

    def get_lineup_foul_rate(
        self, team: str, lineup: Optional[List[str]] = None
    ) -> Dict:
        """Estimate total fouls per 90 for a lineup."""
        players = self.player_stats.get(team, {})
        if not players:
            return {"fouls_per_90": 0.0, "confidence": "low"}

        if lineup:
            total_fouls = 0.0
            matched = 0
            for player_name in lineup:
                pstats = players.get(player_name)
                if pstats and "fouls_per_90" in pstats:
                    total_fouls += pstats["fouls_per_90"]
                    matched += 1

            if matched < 5:
                avg = self.team_averages.get(team, {}).get("avg_fouls_per_90", 1.2)
                total_fouls = avg * 11
                confidence = "low"
            else:
                if matched < 11:
                    avg = self.team_averages.get(team, {}).get("avg_fouls_per_90", 1.2)
                    total_fouls += avg * (11 - matched)
                confidence = "high" if matched >= 8 else "medium"
        else:
            avg = self.team_averages.get(team, {}).get("avg_fouls_per_90", 1.2)
            total_fouls = avg * 11
            confidence = "low"

        return {
            "fouls_per_90": round(total_fouls, 2),
            "confidence": confidence,
        }


# Module-level singleton for reuse
_db: Optional[LineupStatsDatabase] = None


def _get_db() -> LineupStatsDatabase:
    """Get or create the singleton database."""
    global _db
    if _db is None or not _db.loaded:
        _db = LineupStatsDatabase()
        _db.load()
    return _db


def get_lineup_expected_corners(team: str, lineup: Optional[List[str]] = None) -> Dict:
    """Get expected corner adjustment for a team/lineup."""
    return _get_db().get_lineup_expected_corners(team, lineup)


def get_lineup_expected_cards(team: str, lineup: Optional[List[str]] = None) -> Dict:
    """Get expected yellow cards for a team/lineup."""
    return _get_db().get_lineup_expected_cards(team, lineup)


def get_lineup_foul_rate(team: str, lineup: Optional[List[str]] = None) -> Dict:
    """Get fouls per 90 for a team/lineup."""
    return _get_db().get_lineup_foul_rate(team, lineup)
