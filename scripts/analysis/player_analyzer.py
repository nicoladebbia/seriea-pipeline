#!/usr/bin/env python3
"""PLAYER-LEVEL ANALYSIS SYSTEM - Deep Player Intelligence

Provides player-level analysis for Serie A matches using free data sources.

Features:
- Player stats from FBref (free)
- Key player identification
- Player form tracking
- Injury impact quantification
- Formation-player fit analysis
- Individual matchup assessment

Data Sources (Free):
- FBref: Comprehensive stats (xG, xA, passing, defense)
- Understat: Shot data, expected metrics
- Transfermarkt: Market values, squad info
- Web scraping for lineups and news

Usage:
    python player_analyzer.py --team "Inter"
    python player_analyzer.py --match "Lecce vs Udinese"
    python player_analyzer.py --all
"""

import os
import sys
import json
import time
import re
import hashlib
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, PROJECT_ROOT, latest_season_with_results
from config.team_names import normalize_team
from scripts.utils.parsing import get_cache_path
from scraper.lineup_fetcher import normalize_player_name

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPING = True
except ImportError:
    HAS_SCRAPING = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Cache settings
CACHE_DIR = DATA_DIR / "cache" / "players"
CACHE_DURATION_DAYS = 1  # Player stats refresh daily

# FBref URLs
FBREF_BASE = "https://fbref.com"
FBREF_SERIE_A = f"{FBREF_BASE}/en/comps/11/Serie-A-Stats"

# Position importance weights (for team strength calculation)
POSITION_WEIGHTS = {
    "GK": 0.12,
    "CB": 0.10,
    "LB": 0.08,
    "RB": 0.08,
    "DM": 0.10,
    "CM": 0.10,
    "AM": 0.10,
    "LW": 0.08,
    "RW": 0.08,
    "ST": 0.12,
    "CF": 0.12,
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PlayerStats:
    """Individual player statistics."""
    name: str
    team: str
    position: str
    age: int

    # Playing time
    matches: int
    starts: int
    minutes: int
    minutes_per_match: float

    # Performance
    goals: int
    assists: int
    xg: float  # Expected goals
    xa: float  # Expected assists
    xg_per_90: float
    xa_per_90: float

    # Passing
    pass_completion: float
    progressive_passes: int
    key_passes: int

    # Defense
    tackles: int
    interceptions: int
    blocks: int
    clearances: int

    # Other
    yellow_cards: int
    red_cards: int

    # Calculated metrics
    overall_rating: float = 0.0
    form_rating: float = 0.0  # Recent 5 game form
    importance_score: float = 0.0  # How important to the team


@dataclass
class TeamSquad:
    """Team squad analysis."""
    team: str
    players: List[PlayerStats] = field(default_factory=list)

    # Key players by position
    key_players: Dict[str, str] = field(default_factory=dict)

    # Squad metrics
    avg_age: float = 0.0
    total_xg: float = 0.0
    total_xa: float = 0.0
    squad_depth: float = 0.0  # 0-100 rating

    # Analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    generated_at: str = ""


@dataclass
class PlayerMatchup:
    """Head-to-head player matchup analysis."""
    home_player: str
    away_player: str
    position: str

    home_rating: float
    away_rating: float
    advantage: str  # "home", "away", "even"
    advantage_margin: float

    analysis: str


@dataclass
class MatchPlayerAnalysis:
    """Complete player-level analysis for a match."""
    match: str
    date: str
    home_team: str
    away_team: str

    # Squad analysis
    home_squad: Optional[TeamSquad]
    away_squad: Optional[TeamSquad]

    # Key player comparisons
    key_matchups: List[PlayerMatchup]

    # Team strength ratings
    home_strength: float  # 0-100
    away_strength: float  # 0-100

    # Injury/absence impact
    home_injury_impact: float  # 0 to -50
    away_injury_impact: float  # 0 to -50

    # Key findings
    key_factors: List[str]
    analysis_summary: str

    generated_at: str


# =============================================================================
# WEB SCRAPING (FREE SOURCES)
# =============================================================================

class PlayerDataScraper:
    """Scrapes player data from free sources."""

    def __init__(self):
        if HAS_SCRAPING:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
        else:
            self.session = None

    def _normalize_team(self, team: str) -> str:
        """Normalize team name."""
        return normalize_team(team)

    def _load_from_cache(self, key: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Load data from cache if valid."""
        cache_path = get_cache_path(CACHE_DIR, key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                return None

            return data.get("data")
        except Exception:
            return None

    def _save_to_cache(self, key: str, data: any):
        """Save data to cache."""
        cache_path = get_cache_path(CACHE_DIR, key)
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "cached_at": datetime.now().isoformat(),
                    "data": data
                }, f)
        except Exception as e:
            log.warning(f"Cache save error: {e}")

    def get_serie_a_players(self) -> List[Dict]:
        """Get all Serie A player stats from FBref."""
        cache_key = "fbref_serie_a_players"
        cached = self._load_from_cache(cache_key)
        if cached:
            log.info("Using cached Serie A player data")
            return cached

        if not self.session:
            log.warning("Scraping not available (install requests and bs4)")
            return self._get_fallback_players()

        log.info("Fetching Serie A player data from FBref...")

        try:
            # Get the main Serie A page
            response = self.session.get(FBREF_SERIE_A, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the player stats table
            players = []

            # Look for standard stats table
            stats_table = soup.find('table', {'id': 'stats_standard'})
            if not stats_table:
                # Try alternative approach
                stats_table = soup.find('table', {'class': 'stats_table'})

            if stats_table:
                rows = stats_table.find('tbody').find_all('tr')
                for row in rows:
                    if 'thead' in row.get('class', []):
                        continue

                    cols = row.find_all(['td', 'th'])
                    if len(cols) < 10:
                        continue

                    try:
                        player_data = {
                            "name": cols[0].get_text(strip=True),
                            "team": self._normalize_team(cols[3].get_text(strip=True)) if len(cols) > 3 else "",
                            "position": cols[4].get_text(strip=True) if len(cols) > 4 else "",
                            "age": int(cols[5].get_text(strip=True).split('-')[0]) if len(cols) > 5 and cols[5].get_text(strip=True) else 25,
                            "matches": int(cols[6].get_text(strip=True) or 0) if len(cols) > 6 else 0,
                            "starts": int(cols[7].get_text(strip=True) or 0) if len(cols) > 7 else 0,
                            "minutes": int(cols[8].get_text(strip=True).replace(',', '') or 0) if len(cols) > 8 else 0,
                            "goals": int(cols[9].get_text(strip=True) or 0) if len(cols) > 9 else 0,
                            "assists": int(cols[10].get_text(strip=True) or 0) if len(cols) > 10 else 0,
                        }
                        if player_data["name"] and player_data["minutes"] > 0:
                            players.append(player_data)
                    except (ValueError, IndexError) as e:
                        continue

            if players:
                self._save_to_cache(cache_key, players)
                log.info(f"Fetched {len(players)} players from FBref")
            else:
                players = self._get_fallback_players()

            time.sleep(3)  # Rate limiting
            return players

        except Exception as e:
            log.warning(f"FBref scraping failed: {e}")
            players = self._get_fallback_players()
            # Cache fallback data to prevent repeated 403s on subsequent calls
            if players:
                self._save_to_cache(cache_key, players)
            return players

    def _get_fallback_players(self) -> List[Dict]:
        """Return player data from best available source.

        Priority:
        1. player_props.json — current season xG profiles with minutes/goals (best)
        2. current_squads.json + xG enrichment
        3. Emergency hardcoded list (last resort)
        """
        # 1. Try player_props.json — most current and comprehensive
        props_path = DATA_DIR / "upcoming" / "player_props.json"
        if props_path.exists():
            try:
                with open(props_path) as f:
                    props = json.load(f)
                players = self._convert_props_to_player_list(props)
                if players:
                    log.info(f"Using player_props data: {len(players)} players")
                    return players
            except Exception as e:
                log.warning(f"Failed to load player_props: {e}")

        # 2. Try cached squad data + xG enrichment
        squads_path = DATA_DIR / "squads" / "current_squads.json"
        if squads_path.exists():
            try:
                with open(squads_path) as f:
                    squad_data = json.load(f)
                players = self._convert_squad_to_player_list(squad_data)
                if players:
                    players = self._enrich_with_xg_profiles(players)
                    log.info(f"Using cached squad data: {len(players)} players from {len(squad_data.get('teams', {}))} teams")
                    return players
            except Exception as e:
                log.warning(f"Failed to load cached squads: {e}")

        log.warning("No player data available, using emergency fallback")
        return self._get_emergency_fallback_players()

    def _convert_props_to_player_list(self, props: Dict) -> List[Dict]:
        """Convert player_props.json to the flat player list format.

        Strategy:
        1. Collect all (name, team, data) entries from all matches
        2. Load current roster from current_squads.json
        3. For each player with multiple team associations, prefer the one
           that matches the current roster (handles transfers)
        4. Enrich with age and position from roster
        """
        # Step 1: Collect all entries per player name, grouped by team
        by_name = defaultdict(dict)  # name -> {team: best_entry}
        for mk, match_data in props.get("matches", {}).items():
            for p in match_data.get("players", []):
                name = p.get("name", "")
                team = p.get("team", "")
                if not name or not team:
                    continue
                mins = p.get("minutes", 0)
                if team not in by_name[name] or mins > by_name[name][team].get("minutes", 0):
                    by_name[name][team] = p

        # Step 2: Load roster for validation
        squads_path = DATA_DIR / "squads" / "current_squads.json"
        roster_lastname_team = {}  # (last_name_lower, team) -> {age, position}
        if squads_path.exists():
            try:
                with open(squads_path) as f:
                    squads = json.load(f)
                for team_name, team_data in squads.get("teams", {}).items():
                    for sp in team_data.get("players", []):
                        sname = sp.get("name", "")
                        if not sname:
                            continue
                        parts = sname.strip().split()
                        last = parts[-1].lower() if parts else ""
                        if last:
                            roster_lastname_team[(last, team_name)] = {
                                "age": sp.get("age"),
                                "position": sp.get("position", ""),
                            }
            except Exception:
                pass

        # Step 3: For each player, pick the correct team
        seen = {}
        for name, team_entries in by_name.items():
            parts = name.strip().split()
            last = parts[-1].lower() if parts else ""

            # Prefer the team validated by roster
            chosen_team = None
            chosen_entry = None
            for team, entry in team_entries.items():
                if (last, team) in roster_lastname_team:
                    chosen_team = team
                    chosen_entry = entry
                    break

            # If no roster match for the listed team, check if the player
            # is in ANY current Serie A team's roster (handles incomplete
            # roster for a specific team).
            if not chosen_entry and roster_lastname_team:
                # Check all teams in the roster
                roster_team = None
                for (rlast, rteam), _ in roster_lastname_team.items():
                    if rlast == last:
                        roster_team = rteam
                        break
                if roster_team:
                    # Player found in a different team's roster — use that team
                    # and the entry with the most minutes
                    best = max(team_entries.values(),
                               key=lambda x: x.get("minutes", 0))
                    chosen_team = roster_team
                    chosen_entry = best
                else:
                    # Not in ANY roster — likely left Serie A entirely
                    # (Giroud, Džeko, Osimhen, etc.)
                    best = max(team_entries.values(),
                               key=lambda x: x.get("minutes", 0))
                    # Only skip if >60 career matches (clear multi-season data)
                    if best.get("matches", 0) >= 60:
                        continue
                    # Otherwise keep — could be a new signing not in squads yet
                    chosen_team = max(team_entries,
                                      key=lambda t: team_entries[t].get("minutes", 0))
                    chosen_entry = best

            # No roster available at all: fall back to most minutes
            if not chosen_entry:
                for team, entry in sorted(team_entries.items(),
                                          key=lambda x: x[1].get("minutes", 0),
                                          reverse=True):
                    chosen_team = team
                    chosen_entry = entry
                    break

            if not chosen_entry:
                continue

            total_xg = chosen_entry.get("total_xg", 0)
            total_xa = chosen_entry.get("total_xa", 0)
            player_dict = {
                "name": name,
                "team": chosen_team,
                "position": self._infer_position_from_props(chosen_entry),
                "age": 25,
                "matches": chosen_entry.get("matches", 0),
                "starts": max(0, chosen_entry.get("matches", 0) - 3),
                "minutes": chosen_entry.get("minutes", 0),
                "goals": int(round(total_xg * 1.05)) if total_xg > 0 else 0,
                "assists": int(round(total_xa)) if total_xa > 0 else 0,
                "xg": round(total_xg, 2),
                "xa": round(total_xa, 2),
                "xg_per_90": round(chosen_entry.get("xg_per_90", 0), 3),
                "xa_per_90": round(chosen_entry.get("xa_per_90", 0), 3),
                "anytime_goal_prob": round(chosen_entry.get("anytime_goal_prob", 0), 3),
                "sofascore_rating": chosen_entry.get("sofascore_rating", 0),
            }

            # Step 4: Enrich from roster
            roster_info = roster_lastname_team.get((last, chosen_team), {})
            if roster_info.get("age"):
                player_dict["age"] = roster_info["age"]
            pos = roster_info.get("position", "")
            if pos in ("GK", "CB", "LB", "RB", "DM", "CM", "AM", "LW", "RW", "ST", "CF"):
                player_dict["position"] = pos

            seen[name] = player_dict

        return list(seen.values())

    @staticmethod
    def _infer_position_from_props(p: Dict) -> str:
        """Infer position from xG/xA profile and performance data.

        Uses a combination of xG, xA, goal probability, tackles, and shots
        to classify players. player_props.json mostly contains goal-threat
        players, so the hierarchy is biased toward offensive positions.
        """
        xg90 = p.get("xg_per_90", 0)
        xa90 = p.get("xa_per_90", 0)
        tackles = p.get("tackles_expected", 0)
        shots = p.get("shots_expected", 0)
        goal_prob = p.get("anytime_goal_prob", 0)

        # Pure striker: high xG, high goal probability, lots of shots
        if xg90 >= 0.30 or (xg90 >= 0.20 and goal_prob >= 0.20):
            return "ST"
        # Second striker / forward: moderate xG with some creativity
        if xg90 >= 0.15 and xa90 < 0.10:
            return "CF"
        # Attacking midfielder / winger: creative with moderate goal threat
        if xa90 >= 0.12 or (xg90 >= 0.10 and xa90 >= 0.06):
            return "AM"
        # Defensive midfielder: tackles-heavy, low offensive output
        if tackles >= 2.0 and xg90 < 0.10:
            return "DM"
        # Central midfielder: some offensive output
        if xg90 >= 0.05 or xa90 >= 0.04:
            return "CM"
        # Defender: low everything offensively, some tackles
        if tackles >= 1.0 and xg90 < 0.05:
            return "CB"
        # Goalkeeper: no offensive stats at all
        if xg90 == 0 and xa90 == 0 and goal_prob < 0.05:
            return "GK"
        return "CM"  # default

    def _convert_squad_to_player_list(self, squad_data: Dict) -> List[Dict]:
        """Convert nested squad JSON to flat player list format."""
        players = []
        for team_name, team_data in squad_data.get("teams", {}).items():
            for p in team_data.get("players", []):
                players.append({
                    "name": p.get("name", ""),
                    "team": team_name,
                    "position": p.get("position", "CM"),
                    "age": p.get("age") or 25,
                    "matches": 0,
                    "starts": 0,
                    "minutes": 0,
                    "goals": 0,
                    "assists": 0,
                })
        return players

    def _enrich_with_xg_profiles(self, players: List[Dict]) -> List[Dict]:
        """Cross-reference xG profiles to add goals/minutes/xG stats."""
        profiles_path = DATA_DIR / "features" / "player_xg_profiles.json"
        if not profiles_path.exists():
            return players

        try:
            with open(profiles_path) as f:
                profiles = json.load(f)
        except Exception:
            return players

        # Build normalized lookup: (normalized_name, team) -> profile
        profile_lookup = {}
        for key, profile in profiles.items():
            pname = normalize_player_name(profile.get("player_name", ""))
            pteam = profile.get("team", "")
            if pname and pteam:
                profile_lookup[(pname, pteam)] = profile

        enriched = 0
        for player in players:
            norm_name = normalize_player_name(player["name"])
            team = player["team"]
            profile = profile_lookup.get((norm_name, team))
            if profile:
                mins = profile.get("total_minutes", 0)
                matches = profile.get("matches_played", 0)
                xg = profile.get("total_xg", 0)
                xa = profile.get("total_xa", 0)
                # Estimate goals/assists from xG (rough: goals ~ xg * 1.05)
                player["minutes"] = mins
                player["matches"] = matches
                player["starts"] = max(0, matches - 2)  # Approximate
                player["goals"] = int(round(xg * 1.05)) if xg > 0 else 0
                player["assists"] = int(round(xa * 1.0)) if xa > 0 else 0
                player["xg"] = round(xg, 2)
                player["xa"] = round(xa, 2)
                enriched += 1

        log.info(f"Enriched {enriched}/{len(players)} players with xG profiles")
        return players

    def _get_emergency_fallback_players(self) -> List[Dict]:
        """Minimal emergency fallback with only verified 2025-26 season players."""
        return [
            # Inter
            {"name": "Lautaro Martinez", "team": "Inter", "position": "ST", "age": 28, "matches": 20, "starts": 18, "minutes": 1500, "goals": 12, "assists": 4},
            {"name": "Marcus Thuram", "team": "Inter", "position": "ST", "age": 27, "matches": 20, "starts": 16, "minutes": 1400, "goals": 8, "assists": 5},
            {"name": "Hakan Calhanoglu", "team": "Inter", "position": "CM", "age": 31, "matches": 18, "starts": 17, "minutes": 1400, "goals": 5, "assists": 6},
            # Napoli
            {"name": "Romelu Lukaku", "team": "Napoli", "position": "ST", "age": 32, "matches": 18, "starts": 16, "minutes": 1300, "goals": 8, "assists": 3},
            {"name": "Stanislav Lobotka", "team": "Napoli", "position": "DM", "age": 30, "matches": 19, "starts": 18, "minutes": 1550, "goals": 0, "assists": 3},
            {"name": "Scott McTominay", "team": "Napoli", "position": "CM", "age": 29, "matches": 18, "starts": 16, "minutes": 1350, "goals": 4, "assists": 2},
            # Juventus
            {"name": "Dusan Vlahovic", "team": "Juventus", "position": "ST", "age": 26, "matches": 20, "starts": 18, "minutes": 1500, "goals": 10, "assists": 2},
            {"name": "Manuel Locatelli", "team": "Juventus", "position": "CM", "age": 27, "matches": 19, "starts": 17, "minutes": 1400, "goals": 2, "assists": 4},
            # Milan
            {"name": "Rafael Leao", "team": "Milan", "position": "LW", "age": 26, "matches": 20, "starts": 18, "minutes": 1500, "goals": 7, "assists": 8},
            {"name": "Christian Pulisic", "team": "Milan", "position": "RW", "age": 26, "matches": 19, "starts": 17, "minutes": 1400, "goals": 8, "assists": 5},
            {"name": "Santiago Gimenez", "team": "Milan", "position": "ST", "age": 23, "matches": 12, "starts": 10, "minutes": 850, "goals": 5, "assists": 2},
            # Roma
            {"name": "Paulo Dybala", "team": "Roma", "position": "AM", "age": 31, "matches": 17, "starts": 15, "minutes": 1200, "goals": 6, "assists": 5},
            {"name": "Artem Dovbyk", "team": "Roma", "position": "ST", "age": 27, "matches": 18, "starts": 16, "minutes": 1350, "goals": 7, "assists": 2},
            # Atalanta
            {"name": "Ademola Lookman", "team": "Atalanta", "position": "LW", "age": 27, "matches": 20, "starts": 18, "minutes": 1500, "goals": 9, "assists": 6},
            {"name": "Charles De Ketelaere", "team": "Atalanta", "position": "AM", "age": 24, "matches": 19, "starts": 17, "minutes": 1400, "goals": 7, "assists": 5},
            {"name": "Mateo Retegui", "team": "Atalanta", "position": "ST", "age": 26, "matches": 19, "starts": 17, "minutes": 1400, "goals": 10, "assists": 2},
            # Lazio
            {"name": "Boulaye Dia", "team": "Lazio", "position": "ST", "age": 28, "matches": 18, "starts": 15, "minutes": 1200, "goals": 6, "assists": 3},
            {"name": "Mattia Zaccagni", "team": "Lazio", "position": "LW", "age": 30, "matches": 19, "starts": 17, "minutes": 1400, "goals": 5, "assists": 5},
            # Fiorentina
            {"name": "Moise Kean", "team": "Fiorentina", "position": "ST", "age": 25, "matches": 19, "starts": 17, "minutes": 1400, "goals": 10, "assists": 2},
            {"name": "Lucas Beltran", "team": "Fiorentina", "position": "ST", "age": 23, "matches": 19, "starts": 16, "minutes": 1300, "goals": 6, "assists": 4},
            # Bologna
            {"name": "Riccardo Orsolini", "team": "Bologna", "position": "RW", "age": 28, "matches": 19, "starts": 17, "minutes": 1400, "goals": 5, "assists": 4},
            {"name": "Santiago Castro", "team": "Bologna", "position": "ST", "age": 21, "matches": 18, "starts": 15, "minutes": 1200, "goals": 5, "assists": 4},
            # Torino
            {"name": "Che Adams", "team": "Torino", "position": "ST", "age": 28, "matches": 18, "starts": 16, "minutes": 1300, "goals": 6, "assists": 3},
            # Lecce
            {"name": "Nikola Krstovic", "team": "Lecce", "position": "ST", "age": 24, "matches": 17, "starts": 14, "minutes": 1100, "goals": 4, "assists": 3},
            # Udinese
            {"name": "Florian Thauvin", "team": "Udinese", "position": "RW", "age": 32, "matches": 16, "starts": 14, "minutes": 1100, "goals": 3, "assists": 4},
            # Genoa
            {"name": "Andrea Pinamonti", "team": "Genoa", "position": "ST", "age": 26, "matches": 18, "starts": 15, "minutes": 1200, "goals": 5, "assists": 2},
            # Cagliari
            {"name": "Roberto Piccoli", "team": "Cagliari", "position": "ST", "age": 25, "matches": 24, "starts": 20, "minutes": 1700, "goals": 6, "assists": 2},
            # Parma
            {"name": "Dennis Man", "team": "Parma", "position": "RW", "age": 26, "matches": 18, "starts": 16, "minutes": 1300, "goals": 4, "assists": 3},
            # Como
            {"name": "Patrick Cutrone", "team": "Como", "position": "ST", "age": 27, "matches": 17, "starts": 14, "minutes": 1100, "goals": 4, "assists": 2},
            # Verona
            {"name": "Casper Tengstedt", "team": "Verona", "position": "ST", "age": 25, "matches": 17, "starts": 14, "minutes": 1100, "goals": 5, "assists": 2},
            # Sassuolo (promoted 2025-26)
            {"name": "Domenico Berardi", "team": "Sassuolo", "position": "RW", "age": 31, "matches": 15, "starts": 13, "minutes": 1050, "goals": 5, "assists": 4},
            # Pisa (promoted 2025-26)
            {"name": "Nicholas Bonfanti", "team": "Pisa", "position": "ST", "age": 23, "matches": 16, "starts": 14, "minutes": 1100, "goals": 4, "assists": 2},
            # Cremonese (promoted 2025-26)
            {"name": "Franco Vazquez", "team": "Cremonese", "position": "AM", "age": 32, "matches": 15, "starts": 12, "minutes": 950, "goals": 3, "assists": 3},
        ]


# =============================================================================
# PLAYER ANALYSIS
# =============================================================================

_EPL_FULL_TO_SHORT = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Tottenham Hotspur": "Tottenham",
    "Leeds United": "Leeds",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Nottingham Forest": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
}


def _resolve_sofascore_source(team: str) -> tuple:
    """Return (parquet_path, normalized_team_name) for a team.

    Picks the SA parquet for Serie A teams and the EPL parquet for EPL teams,
    based on infer_league(). Normalizes Odds-API full names like
    'Manchester City' to Sofascore short names like 'Man City'.
    """
    from pathlib import Path
    try:
        from config.leagues import infer_league
        league = infer_league(team)
    except Exception:
        league = "serie_a"

    base = DATA_DIR / ".." / "data" / "external" / "sofascore"
    if league == "premier_league":
        path = base / "player_match_stats_premier_league.parquet"
        normalized = _EPL_FULL_TO_SHORT.get(team, team)
    else:
        path = base / "player_match_stats.parquet"
        normalized = team
        if team == "Verona":
            normalized = "Hellas Verona"  # Sofascore name; we'll fall back to "Verona" if not found
    return Path(path), normalized


class PlayerAnalyzer:
    """Analyzes players and generates insights."""

    def __init__(self):
        self.scraper = PlayerDataScraper()
        self.players_cache: Dict[str, List[PlayerStats]] = {}

    def _calculate_player_rating(self, player: Dict) -> float:
        """Calculate overall player rating (0-100)."""
        minutes = player.get("minutes", 0)
        if minutes < 100:
            return 30.0  # Low rating for players with minimal playing time

        goals = player.get("goals", 0)
        assists = player.get("assists", 0)
        matches = player.get("matches", 1)

        # Goals + assists per 90 minutes
        ga_per_90 = (goals + assists) / (minutes / 90) if minutes > 0 else 0

        # Position-adjusted scoring
        position = player.get("position", "").upper()
        if "GK" in position:
            # Goalkeepers rated differently
            rating = 60 + min(minutes / 100, 20)  # More minutes = higher rating
        elif any(p in position for p in ["CB", "LB", "RB", "DM"]):
            # Defenders: fewer goals expected
            rating = 50 + (ga_per_90 * 30) + min(minutes / 100, 20)
        elif any(p in position for p in ["CM", "AM"]):
            # Midfielders: moderate contribution
            rating = 45 + (ga_per_90 * 25) + min(minutes / 100, 15)
        else:
            # Attackers: goals are key
            rating = 40 + (ga_per_90 * 20) + min(minutes / 100, 10)

        return min(100, max(0, rating))

    def _calculate_importance(self, player: Dict, team_players: List[Dict]) -> float:
        """Calculate how important a player is to the team (0-100)."""
        minutes = player.get("minutes", 0)
        total_team_minutes = sum(p.get("minutes", 0) for p in team_players)

        if total_team_minutes == 0:
            return 50.0

        # Minutes share
        minutes_share = (minutes / total_team_minutes) * 100

        # Goal contribution share
        goals = player.get("goals", 0) + player.get("assists", 0)
        total_goals = sum(p.get("goals", 0) + p.get("assists", 0) for p in team_players)
        goal_share = (goals / total_goals * 100) if total_goals > 0 else 0

        # Combined importance
        importance = (minutes_share * 0.4) + (goal_share * 0.6)

        return min(100, max(0, importance * 2))  # Scale to 0-100

    def _load_squad_positions(self) -> Dict[str, str]:
        """Load position data from current_squads.json, keyed by normalized (name, team)."""
        squads_path = DATA_DIR / "squads" / "current_squads.json"
        if not squads_path.exists():
            return {}
        try:
            with open(squads_path) as f:
                sq = json.load(f)
            lookup = {}
            for team_name, team_data in sq.get("teams", {}).items():
                for p in team_data.get("players", []):
                    norm = normalize_player_name(p.get("name", ""))
                    if norm:
                        lookup[(norm, team_name)] = p.get("position", "")
            return lookup
        except Exception:
            return {}

    def _infer_position(self, xg_per_90: float, xa_per_90: float, minutes: int) -> str:
        """Infer position from xG/xA patterns when no position data is available."""
        if minutes < 90:
            return "SUB"
        ga_per_90 = xg_per_90 + xa_per_90
        if xg_per_90 == 0 and xa_per_90 == 0 and minutes > 2500:
            return "GK"
        if xg_per_90 > 0.35:
            return "ST"
        if xg_per_90 > 0.15 or ga_per_90 > 0.4:
            return "AM"
        if ga_per_90 > 0.15:
            return "CM"
        return "CB"

    def _load_xg_profiles_for_team(self, team: str) -> List[Dict]:
        """Load player data directly from xG profiles for a given team."""
        profiles_path = DATA_DIR / "features" / "player_xg_profiles.json"
        if not profiles_path.exists():
            return []

        try:
            with open(profiles_path) as f:
                profiles = json.load(f)
        except Exception:
            return []

        # Load squad positions for cross-reference
        squad_positions = self._load_squad_positions()

        players = []
        for key, profile in profiles.items():
            if profile.get("team", "") != team:
                continue
            mins = profile.get("total_minutes", 0)
            if mins < 90:
                continue
            xg = profile.get("total_xg", 0)
            xa = profile.get("total_xa", 0)
            xg90 = profile.get("xg_per_90", 0)
            xa90 = profile.get("xa_per_90", 0)
            recent_xg = profile.get("recent_form_xg", xg90)
            # Blend career and recent form
            blended_xg90 = 0.7 * xg90 + 0.3 * recent_xg if recent_xg else xg90
            matches = profile.get("matches_played", max(1, int(mins / 80)))

            # Determine position: squad data first, then inference
            pname = profile.get("player_name", key)
            norm_name = normalize_player_name(pname)
            position = squad_positions.get((norm_name, team), "")
            if not position or position == "CM":
                stored_pos = profile.get("position")
                if stored_pos and stored_pos != "None":
                    position = stored_pos
                else:
                    position = self._infer_position(xg90, xa90, mins)

            players.append({
                "name": pname,
                "team": team,
                "position": position,
                "age": profile.get("age", 25),
                "matches": matches,
                "starts": max(0, matches - 2),
                "minutes": mins,
                "goals": int(round(xg * 1.05)) if xg > 0 else 0,
                "assists": int(round(xa * 1.0)) if xa > 0 else 0,
                "xg": round(xg, 2),
                "xa": round(xa, 2),
                "xg_per_90": round(blended_xg90, 3),
                "xa_per_90": round(xa90, 3),
            })
        return players

    def _validate_against_sofascore(self, team_players: List[Dict], team: str) -> List[Dict]:
        """Filter out players who haven't actually played for this team this season.

        Cross-references against:
        1. Sofascore player_match_stats.parquet (actual match appearances)
        2. current_squads.json (official roster from Transfermarkt)

        A player is valid if they appear in EITHER source for this team.
        This handles both mid-season transfers and new signings.
        """
        try:
            import pandas as pd

            # --- Source 1: Sofascore match data (league-aware) ---
            sofascore_names = set()  # lowercase last names of players who played for this team
            sofascore_full = set()   # lowercase full names
            sofascore_path, sofascore_team = _resolve_sofascore_source(team)
            if sofascore_path.exists():
                pms = pd.read_parquet(sofascore_path)
                # Latest season with data, not the calendar season — player rows
                # exist only for played matches. See config.settings.
                _season = latest_season_with_results(pms, result_col="minutes")
                current = pms[pms["season"] == _season] if _season else pms.iloc[0:0]
                if not current.empty:
                    if sofascore_team not in current["team"].unique() and team in current["team"].unique():
                        sofascore_team = team  # fall back to original if normalized name doesn't match
                    team_data = current[current["team"] == sofascore_team]
                    for name in team_data["player_name"].unique():
                        sofascore_full.add(name.lower().strip())
                        parts = name.strip().split()
                        if parts:
                            sofascore_names.add(parts[-1].lower())

            # --- Source 2: Current squads roster ---
            roster_names = set()
            roster_full = set()
            squads_path = DATA_DIR / "squads" / "current_squads.json"
            if squads_path.exists():
                try:
                    import json as _json
                    with open(squads_path) as f:
                        squads = _json.load(f)
                    for sp in squads.get("teams", {}).get(team, {}).get("players", []):
                        sname = sp.get("name", "")
                        if sname:
                            roster_full.add(sname.lower().strip())
                            parts = sname.strip().split()
                            if parts:
                                roster_names.add(parts[-1].lower())
                except Exception:
                    pass

            # Combine both sources
            all_valid_lastnames = sofascore_names | roster_names
            all_valid_full = sofascore_full | roster_full

            if not all_valid_lastnames and not all_valid_full:
                return team_players  # No validation data available

            # Filter: keep players who match either source
            validated = []
            removed = []
            for p in team_players:
                pname = p.get("name", "")
                pname_lower = pname.lower().strip()
                parts = pname.strip().split()
                last = parts[-1].lower() if parts else ""

                if pname_lower in all_valid_full or last in all_valid_lastnames:
                    validated.append(p)
                else:
                    removed.append(pname)

            if removed:
                log.info(f"[{team}] Removed {len(removed)} stale players: {', '.join(removed)}")

            return validated if validated else team_players  # Don't return empty
        except Exception as e:
            log.warning(f"Sofascore validation failed for {team}: {e}")
            return team_players

    def _build_squad_from_sofascore(self, team: str) -> List[Dict]:
        """Build squad data directly from Sofascore player_match_stats.parquet.

        This is the most reliable source: actual match appearances this season.
        Uses real goals, real xG, real minutes — no stale FBref data.
        """
        try:
            import pandas as pd
            sofascore_path, sofascore_team = _resolve_sofascore_source(team)
            if not sofascore_path.exists():
                return []

            pms = pd.read_parquet(sofascore_path)
            # Latest season with data, not the calendar season — player rows
            # exist only for played matches. See config.settings.
            _season = latest_season_with_results(pms, result_col="minutes")
            current = pms[pms["season"] == _season] if _season else pms.iloc[0:0]
            if current.empty:
                return []

            if sofascore_team not in current["team"].unique() and team in current["team"].unique():
                sofascore_team = team

            team_data = current[current["team"] == sofascore_team]
            if team_data.empty:
                return []

            # Aggregate per player
            players = []
            for (pid, pname), grp in team_data.groupby(["player_id", "player_name"]):
                apps = len(grp)
                starts = int(grp["is_starter"].sum())
                total_mins = int(grp["minutes"].sum())
                total_goals = int(grp["goals"].sum()) if "goals" in grp.columns else 0
                total_assists = int(grp["assists"].sum()) if "assists" in grp.columns else 0
                total_xg = float(grp["xg"].sum()) if "xg" in grp.columns and grp["xg"].notna().any() else 0
                total_xa = float(grp["xa"].sum()) if "xa" in grp.columns and grp["xa"].notna().any() else 0
                avg_rating = grp["rating"].mean() if "rating" in grp.columns and grp["rating"].notna().any() else 0
                avg_rating = float(avg_rating) if pd.notna(avg_rating) else 0.0
                position = grp["position"].mode().iloc[0] if len(grp["position"].mode()) > 0 else "?"
                latest_date = grp["date"].max()

                # Infer specific position from Sofascore base position + stats
                xg_per_90 = round(total_xg / (total_mins / 90), 3) if total_mins > 0 else 0
                xa_per_90 = round(total_xa / (total_mins / 90), 3) if total_mins > 0 else 0
                tackles_per_90 = round(float(grp["tackles"].sum()) / (total_mins / 90), 2) if total_mins > 0 and "tackles" in grp.columns else 0

                if position == "G":
                    std_pos = "GK"
                elif position == "D":
                    std_pos = "CB"  # Will be enriched from roster (LB, RB, etc.)
                elif position == "M":
                    # Distinguish CM/DM/AM based on offensive vs defensive output
                    if xg_per_90 >= 0.15 or xa_per_90 >= 0.15:
                        std_pos = "AM"
                    elif tackles_per_90 >= 2.5:
                        std_pos = "DM"
                    else:
                        std_pos = "CM"
                elif position == "F":
                    # Distinguish ST/CF/RW/LW based on goal threat vs creativity
                    if xg_per_90 >= 0.30 or (total_goals > 0 and xa_per_90 < 0.05):
                        std_pos = "ST"
                    elif xa_per_90 >= 0.12:
                        std_pos = "RW"  # Creative winger
                    else:
                        std_pos = "CF"
                else:
                    std_pos = position

                players.append({
                    "name": pname,
                    "team": team,
                    "position": std_pos,
                    "age": 25,  # Will be enriched from roster
                    "matches": apps,
                    "starts": starts,
                    "minutes": total_mins,
                    "goals": total_goals,
                    "assists": total_assists,
                    "xg": round(total_xg, 2),
                    "xa": round(total_xa, 2),
                    "xg_per_90": xg_per_90,
                    "xa_per_90": xa_per_90,
                    "sofascore_rating": round(avg_rating, 2),
                    "pass_completion": 80,
                    "progressive_passes": 0,
                    "key_passes": 0,
                    "tackles": 0,
                    "interceptions": 0,
                    "blocks": 0,
                    "clearances": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "latest_date": str(latest_date),
                })

            # Enrich ages from roster
            squads_path = DATA_DIR / "squads" / "current_squads.json"
            if squads_path.exists():
                try:
                    import json as _json
                    with open(squads_path) as f:
                        squads = _json.load(f)
                    roster = squads.get("teams", {}).get(team, {}).get("players", [])
                    roster_by_lastname = {}
                    for sp in roster:
                        sname = sp.get("name", "")
                        parts = sname.strip().split()
                        if parts:
                            roster_by_lastname[parts[-1].lower()] = sp
                    for p in players:
                        parts = p["name"].strip().split()
                        last = parts[-1].lower() if parts else ""
                        if last in roster_by_lastname:
                            age = roster_by_lastname[last].get("age")
                            if age:
                                p["age"] = age
                            pos = roster_by_lastname[last].get("position", "")
                            if pos in ("GK", "CB", "LB", "RB", "DM", "CM", "AM", "LW", "RW", "ST", "CF"):
                                p["position"] = pos
                except Exception:
                    pass

            # Sort by minutes played (most active first)
            players.sort(key=lambda p: p["minutes"], reverse=True)
            log.info(f"[{team}] Built squad from Sofascore: {len(players)} players")
            return players
        except Exception as e:
            log.warning(f"Failed to build Sofascore squad for {team}: {e}")
            return []

    def get_team_squad(self, team: str) -> TeamSquad:
        """Get complete squad analysis for a team.

        Data priority:
        1. Sofascore player_match_stats.parquet (real match data — most reliable)
        2. FBref/player_props.json (fallback, may have stale transfers)
        """
        team = normalize_team(team)

        # Primary source: Sofascore actual match data
        team_players = self._build_squad_from_sofascore(team)

        # Fallback: FBref/props data (only if Sofascore has nothing)
        if not team_players:
            log.info(f"[{team}] No Sofascore data, falling back to FBref/props")
            all_players = self.scraper.get_serie_a_players()
            team_players = [p for p in all_players if p.get("team", "") == team]

            # Supplement from xG profiles if needed
            players_with_data = [p for p in team_players if p.get("minutes", 0) > 200]
            if len(players_with_data) < 5:
                xg_players = self._load_xg_profiles_for_team(team)
                if xg_players:
                    existing_names = {normalize_player_name(p["name"]) for p in xg_players}
                    for p in team_players:
                        if normalize_player_name(p["name"]) not in existing_names:
                            xg_players.append(p)
                    team_players = xg_players

            # Validate against Sofascore + roster
            team_players = self._validate_against_sofascore(team_players, team)

        if not team_players:
            log.warning(f"No players found for {team}")
            return TeamSquad(team=team, generated_at=datetime.now().isoformat())

        # Convert to PlayerStats
        player_stats = []
        for p in team_players:
            minutes = p.get("minutes", 0)
            matches = p.get("matches", 1)

            # Use xG data directly if available, otherwise estimate
            xg = p.get("xg", p.get("goals", 0) * 0.9)
            xa = p.get("xa", p.get("assists", 0) * 0.8)
            xg_per_90 = p.get("xg_per_90", xg / (minutes / 90) if minutes > 0 else 0)
            xa_per_90 = p.get("xa_per_90", xa / (minutes / 90) if minutes > 0 else 0)

            stats = PlayerStats(
                name=p.get("name", ""),
                team=team,
                position=p.get("position", ""),
                age=p.get("age", 25),
                matches=p.get("matches", 0),
                starts=p.get("starts", 0),
                minutes=minutes,
                minutes_per_match=minutes / matches if matches > 0 else 0,
                goals=p.get("goals", 0),
                assists=p.get("assists", 0),
                xg=xg,
                xa=xa,
                xg_per_90=xg_per_90,
                xa_per_90=xa_per_90,
                pass_completion=p.get("pass_completion", 80),
                progressive_passes=p.get("progressive_passes", 0),
                key_passes=p.get("key_passes", 0),
                tackles=p.get("tackles", 0),
                interceptions=p.get("interceptions", 0),
                blocks=p.get("blocks", 0),
                clearances=p.get("clearances", 0),
                yellow_cards=p.get("yellow_cards", 0),
                red_cards=p.get("red_cards", 0),
                overall_rating=self._calculate_player_rating(p),
                importance_score=self._calculate_importance(p, team_players)
            )
            player_stats.append(stats)

        # Sort by importance
        player_stats.sort(key=lambda x: x.importance_score, reverse=True)

        # Identify key players by position — use minutes > 200 threshold (xG profiles
        # have real data, so 500 is too strict for teams with partial coverage)
        key_players = {}
        positions_found = set()
        min_threshold = 200
        for p in player_stats:
            pos = p.position.upper()
            if pos and pos not in positions_found and p.minutes > min_threshold:
                key_players[pos] = p.name
                positions_found.add(pos)

        # If still empty, take top 3 by rating regardless of minutes
        if not key_players:
            for p in sorted(player_stats, key=lambda x: x.overall_rating, reverse=True)[:3]:
                pos = p.position.upper() or "FW"
                if pos not in key_players:
                    key_players[pos] = p.name

        # Calculate squad metrics
        avg_age = sum(p.age for p in player_stats) / len(player_stats) if player_stats else 0
        total_xg = sum(p.xg for p in player_stats)
        total_xa = sum(p.xa for p in player_stats)

        # Squad depth: count players with significant minutes (>500), scale to 0-100
        # ~20 players with 500+ min = 100% depth, <10 = thin squad
        players_with_minutes = len([p for p in player_stats if p.minutes > 500])
        squad_depth = min(100, int(players_with_minutes * 5))

        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []

        if total_xg > 30:
            strengths.append("Strong attacking output")
        elif total_xg < 15:
            weaknesses.append("Weak attacking output")

        if avg_age < 26:
            strengths.append("Young, energetic squad")
        elif avg_age > 29:
            weaknesses.append("Aging squad")

        if squad_depth > 70:
            strengths.append("Good squad depth")
        elif squad_depth < 50:
            weaknesses.append("Limited squad depth")

        return TeamSquad(
            team=team,
            players=player_stats[:20],  # Top 20 players
            key_players=key_players,
            avg_age=round(avg_age, 1),
            total_xg=round(total_xg, 1),
            total_xa=round(total_xa, 1),
            squad_depth=squad_depth,
            strengths=strengths,
            weaknesses=weaknesses,
            generated_at=datetime.now().isoformat()
        )

    def analyze_match(self, home_team: str, away_team: str, match_date: str) -> MatchPlayerAnalysis:
        """Perform complete player-level analysis for a match."""
        match_name = f"{home_team} vs {away_team}"
        log.info(f"Analyzing players for {match_name}...")

        # Get squad analysis
        home_squad = self.get_team_squad(home_team)
        away_squad = self.get_team_squad(away_team)

        # Calculate team strengths
        home_strength = self._calculate_team_strength(home_squad)
        away_strength = self._calculate_team_strength(away_squad)

        # Analyze key matchups
        key_matchups = self._analyze_matchups(home_squad, away_squad)

        # Calculate injury impact (placeholder - would use real injury data)
        home_injury_impact = 0
        away_injury_impact = 0

        # Generate key factors
        key_factors = []

        if home_strength > away_strength + 10:
            key_factors.append("home_squad_advantage")
        elif away_strength > home_strength + 10:
            key_factors.append("away_squad_advantage")

        if home_squad.total_xg > away_squad.total_xg * 1.3:
            key_factors.append("home_attacking_advantage")
        elif away_squad.total_xg > home_squad.total_xg * 1.3:
            key_factors.append("away_attacking_advantage")

        if home_squad.squad_depth > away_squad.squad_depth + 15:
            key_factors.append("home_depth_advantage")
        elif away_squad.squad_depth > home_squad.squad_depth + 15:
            key_factors.append("away_depth_advantage")

        # Build summary
        summary = f"""
Player Analysis for {match_name}:

HOME ({home_team}):
- Squad Strength: {home_strength:.0f}/100
- Key Players: {', '.join(list(home_squad.key_players.values())[:3])}
- Total xG: {home_squad.total_xg:.1f}
- Squad Depth: {home_squad.squad_depth:.0f}%
- Strengths: {', '.join(home_squad.strengths) or 'None identified'}

AWAY ({away_team}):
- Squad Strength: {away_strength:.0f}/100
- Key Players: {', '.join(list(away_squad.key_players.values())[:3])}
- Total xG: {away_squad.total_xg:.1f}
- Squad Depth: {away_squad.squad_depth:.0f}%
- Strengths: {', '.join(away_squad.strengths) or 'None identified'}

KEY MATCHUPS:
{self._format_matchups(key_matchups)}

CONCLUSION: {'Home advantage' if home_strength > away_strength else 'Away advantage' if away_strength > home_strength else 'Even match'}
(Strength difference: {abs(home_strength - away_strength):.0f} points)
"""

        return MatchPlayerAnalysis(
            match=match_name,
            date=match_date,
            home_team=home_team,
            away_team=away_team,
            home_squad=home_squad,
            away_squad=away_squad,
            key_matchups=key_matchups,
            home_strength=home_strength,
            away_strength=away_strength,
            home_injury_impact=home_injury_impact,
            away_injury_impact=away_injury_impact,
            key_factors=key_factors,
            analysis_summary=summary.strip(),
            generated_at=datetime.now().isoformat()
        )

    def _calculate_team_strength(self, squad: TeamSquad) -> float:
        """Calculate overall team strength (0-100)."""
        if not squad.players:
            return 50.0

        # Weighted average of top 11 players' ratings
        top_11 = squad.players[:11]
        if not top_11:
            return 50.0

        avg_rating = sum(p.overall_rating for p in top_11) / len(top_11)

        # Adjust for squad depth
        depth_bonus = (squad.squad_depth - 50) / 10

        # Adjust for xG production
        xg_bonus = (squad.total_xg - 20) / 5 if squad.total_xg > 20 else 0

        return min(100, max(0, avg_rating + depth_bonus + xg_bonus))

    def _analyze_matchups(self, home_squad: TeamSquad, away_squad: TeamSquad) -> List[PlayerMatchup]:
        """Analyze key player matchups."""
        matchups = []

        # Compare key players at each position
        for pos in ["ST", "AM", "CM", "CB"]:
            home_player = next((p for p in home_squad.players if pos in p.position.upper()), None)
            away_player = next((p for p in away_squad.players if pos in p.position.upper()), None)

            if home_player and away_player:
                advantage_margin = home_player.overall_rating - away_player.overall_rating
                if advantage_margin > 5:
                    advantage = "home"
                elif advantage_margin < -5:
                    advantage = "away"
                else:
                    advantage = "even"

                matchups.append(PlayerMatchup(
                    home_player=home_player.name,
                    away_player=away_player.name,
                    position=pos,
                    home_rating=home_player.overall_rating,
                    away_rating=away_player.overall_rating,
                    advantage=advantage,
                    advantage_margin=abs(advantage_margin),
                    analysis=f"{pos}: {home_player.name} ({home_player.overall_rating:.0f}) vs {away_player.name} ({away_player.overall_rating:.0f})"
                ))

        return matchups

    def _format_matchups(self, matchups: List[PlayerMatchup]) -> str:
        """Format matchups for display."""
        lines = []
        for m in matchups:
            indicator = "←" if m.advantage == "home" else "→" if m.advantage == "away" else "="
            lines.append(f"  {m.position}: {m.home_player} {indicator} {m.away_player} (diff: {m.advantage_margin:.0f})")
        return "\n".join(lines) if lines else "  No significant matchups identified"


# =============================================================================
# INTEGRATION
# =============================================================================

def analyze_all_upcoming_matches() -> List[MatchPlayerAnalysis]:
    """Analyze players for all upcoming matches across all active leagues."""
    from config.leagues import ACTIVE_LEAGUES

    matches: List[dict] = []
    upcoming_dir = DATA_DIR / "upcoming"
    for _league in ACTIVE_LEAGUES:
        fname = "predictions.json" if _league == "serie_a" else f"predictions_{_league}.json"
        path = upcoming_dir / fname
        if not path.exists():
            log.warning("No predictions file for %s at %s", _league, path)
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            league_matches = data.get("predictions", []) if isinstance(data, dict) else data
            for m in league_matches:
                if isinstance(m, dict):
                    m.setdefault("league", _league)
            matches.extend(league_matches)
            log.info("Loaded %d upcoming matches for %s", len(league_matches), _league)
        except Exception as e:
            log.warning("Failed to load %s: %s", path, e)

    if not matches:
        log.error("No predictions found across any league")
        return []

    analyzer = PlayerAnalyzer()
    results = []

    for match in matches:
        home_team = match.get("home_team", "")
        away_team = match.get("away_team", "")

        # Fallback: parse from "match" field (e.g. "Roma vs Cagliari")
        if not home_team or not away_team:
            match_str = match.get("match", "")
            if " vs " in match_str:
                parts = match_str.split(" vs ")
                home_team = parts[0].strip()
                away_team = parts[1].strip()

        match_date = match.get("date", "")

        if not home_team or not away_team:
            continue

        try:
            analysis = analyzer.analyze_match(home_team, away_team, match_date)
            results.append(analysis)
            log.info(f"Analyzed: {home_team} vs {away_team} -> H:{analysis.home_strength:.0f} A:{analysis.away_strength:.0f}")
        except Exception as e:
            log.error(f"Failed to analyze {home_team} vs {away_team}: {e}")

    # Save results
    output_path = DATA_DIR / "upcoming" / "player_analysis.json"

    # Convert to serializable format
    output_data = {
        "generated_at": datetime.now().isoformat(),
        "matches": []
    }

    for analysis in results:
        match_data = {
            "match": analysis.match,
            "date": analysis.date,
            "home_team": analysis.home_team,
            "away_team": analysis.away_team,
            "home_strength": analysis.home_strength,
            "away_strength": analysis.away_strength,
            "key_factors": analysis.key_factors,
            "analysis_summary": analysis.analysis_summary,
            "home_squad": {
                "team": analysis.home_squad.team,
                "key_players": analysis.home_squad.key_players,
                "total_xg": analysis.home_squad.total_xg,
                "squad_depth": analysis.home_squad.squad_depth,
                "strengths": analysis.home_squad.strengths,
                "weaknesses": analysis.home_squad.weaknesses,
            } if analysis.home_squad else None,
            "away_squad": {
                "team": analysis.away_squad.team,
                "key_players": analysis.away_squad.key_players,
                "total_xg": analysis.away_squad.total_xg,
                "squad_depth": analysis.away_squad.squad_depth,
                "strengths": analysis.away_squad.strengths,
                "weaknesses": analysis.away_squad.weaknesses,
            } if analysis.away_squad else None,
            "matchups": [asdict(m) for m in analysis.key_matchups],
        }
        output_data["matches"].append(match_data)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    log.info(f"Saved player analysis to {output_path}")
    return results


def get_player_factors(analysis: MatchPlayerAnalysis) -> Dict[str, any]:
    """Convert player analysis to prediction factors."""
    factors = {}

    # Squad strength factors
    strength_diff = analysis.home_strength - analysis.away_strength
    if strength_diff > 15:
        factors["home_squad_advantage"] = True
        factors["squad_advantage_margin"] = round(strength_diff / 100, 2)
    elif strength_diff < -15:
        factors["away_squad_advantage"] = True
        factors["squad_advantage_margin"] = round(abs(strength_diff) / 100, 2)

    # Key factors from analysis
    for factor in analysis.key_factors:
        factors[factor] = True

    return factors


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Player-Level Analysis System")
    parser.add_argument("--team", type=str, help="Analyze specific team")
    parser.add_argument("--match", type=str, help="Analyze specific match (e.g., 'Lecce vs Udinese')")
    parser.add_argument("--all", action="store_true", help="Analyze all upcoming matches")

    args = parser.parse_args()

    analyzer = PlayerAnalyzer()

    if args.all:
        results = analyze_all_upcoming_matches()
        print(f"\nAnalyzed {len(results)} matches")
        for r in results:
            print(f"  {r.match}: H:{r.home_strength:.0f} A:{r.away_strength:.0f}")

    elif args.team:
        squad = analyzer.get_team_squad(args.team)
        print(f"\n{squad.team} Squad Analysis:")
        print(f"  Average Age: {squad.avg_age}")
        print(f"  Total xG: {squad.total_xg}")
        print(f"  Squad Depth: {squad.squad_depth}%")
        print(f"  Strengths: {', '.join(squad.strengths)}")
        print(f"  Key Players:")
        for pos, name in squad.key_players.items():
            print(f"    {pos}: {name}")

    elif args.match:
        if " vs " in args.match:
            home, away = args.match.split(" vs ")
            date = datetime.now().strftime("%Y-%m-%d")
            analysis = analyzer.analyze_match(home.strip(), away.strip(), date)
            print(analysis.analysis_summary)
        else:
            print("Invalid match format. Use: 'Home Team vs Away Team'")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
