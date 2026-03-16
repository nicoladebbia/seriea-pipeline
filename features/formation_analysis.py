#!/usr/bin/env python3
"""Formation Analysis System - Phase 3 Implementation

Analyzes team formations and creates matchup-based predictions.

Key features:
1. Formation detection and classification
2. Formation matchup matrix (historical win rates)
3. Tactical flexibility index
4. Formation change detection
5. Style-based matchup advantages

Formation categories:
- Back 3: 3-4-3, 3-5-2, 3-4-1-2, 3-5-1-1, 3-1-4-2
- Back 4: 4-2-3-1, 4-3-3, 4-4-2, 4-4-1-1, 4-5-1, 4-1-4-1
- Back 5: 5-3-2, 5-4-1

Style categories:
- Attacking: 3-4-3, 4-3-3, 4-2-3-1
- Balanced: 3-5-2, 4-4-2, 4-2-3-1
- Defensive: 5-3-2, 5-4-1, 4-5-1
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.paths import parsed_path
from config.settings import DATA_DIR

log = logging.getLogger(__name__)


# =============================================================================
# FORMATION CLASSIFICATION
# =============================================================================

FORMATION_CATEGORIES = {
    # Back 3 formations
    "3-4-3": {"defenders": 3, "style": "attacking", "width": "wide"},
    "3-5-2": {"defenders": 3, "style": "balanced", "width": "narrow"},
    "3-4-1-2": {"defenders": 3, "style": "attacking", "width": "narrow"},
    "3-5-1-1": {"defenders": 3, "style": "defensive", "width": "narrow"},
    "3-1-4-2": {"defenders": 3, "style": "attacking", "width": "wide"},

    # Back 4 formations
    "4-2-3-1": {"defenders": 4, "style": "balanced", "width": "wide"},
    "4-3-3": {"defenders": 4, "style": "attacking", "width": "wide"},
    "4-4-2": {"defenders": 4, "style": "balanced", "width": "wide"},
    "4-4-1-1": {"defenders": 4, "style": "defensive", "width": "narrow"},
    "4-5-1": {"defenders": 4, "style": "defensive", "width": "narrow"},
    "4-1-4-1": {"defenders": 4, "style": "balanced", "width": "wide"},
    "4-2-2-2": {"defenders": 4, "style": "attacking", "width": "narrow"},
    "4-3-2-1": {"defenders": 4, "style": "attacking", "width": "narrow"},
    "4-3-1-2": {"defenders": 4, "style": "attacking", "width": "narrow"},
    "4-1-3-2": {"defenders": 4, "style": "attacking", "width": "narrow"},

    # Back 5 formations
    "5-3-2": {"defenders": 5, "style": "defensive", "width": "narrow"},
    "5-4-1": {"defenders": 5, "style": "defensive", "width": "narrow"},
}

# Default for unknown formations
DEFAULT_FORMATION = {"defenders": 4, "style": "balanced", "width": "wide"}


def classify_formation(formation: str) -> Dict:
    """Get formation characteristics."""
    return FORMATION_CATEGORIES.get(formation, DEFAULT_FORMATION)


def get_formation_style(formation: str) -> str:
    """Get formation style (attacking/balanced/defensive)."""
    return classify_formation(formation).get("style", "balanced")


def get_defender_count(formation: str) -> int:
    """Get number of defenders in formation."""
    return classify_formation(formation).get("defenders", 4)


# =============================================================================
# FORMATION MATCHUP MATRIX
# =============================================================================

class FormationMatchupMatrix:
    """Builds and queries historical formation matchup data."""

    def __init__(self):
        self.matchups: Dict[str, Dict[str, Dict]] = {}  # home_form -> away_form -> stats
        self.style_matchups: Dict[str, Dict[str, Dict]] = {}  # home_style -> away_style -> stats
        self.loaded = False

    def build_from_matches(self, matches_path: Path = None, lineups_path: Path = None) -> bool:
        """Build matchup matrix from historical data."""
        if matches_path is None:
            matches_path = parsed_path("matches")
        if lineups_path is None:
            lineups_path = parsed_path("lineups")

        if not matches_path.exists() or not lineups_path.exists():
            log.warning("Matches or lineups data not found")
            return False

        matches = pd.read_parquet(matches_path)
        lineups = pd.read_parquet(lineups_path)

        # Get formation per team per match
        formations = lineups.groupby(["match_id", "team"])["formation"].first().reset_index()

        log.info(f"Building matchup matrix from {len(formations)} team-match formations")

        # Merge with match results
        for _, match in matches.iterrows():
            match_id = match.get("match_id")
            home_team = match.get("home_team")
            away_team = match.get("away_team")
            home_score = match.get("home_score")
            away_score = match.get("away_score")

            if pd.isna(home_score) or pd.isna(away_score):
                continue

            # Get formations
            home_form = formations[
                (formations["match_id"] == match_id) & (formations["team"] == home_team)
            ]["formation"].values
            away_form = formations[
                (formations["match_id"] == match_id) & (formations["team"] == away_team)
            ]["formation"].values

            if len(home_form) == 0 or len(away_form) == 0:
                continue

            home_form = home_form[0]
            away_form = away_form[0]

            # Determine result
            if home_score > away_score:
                result = "H"
            elif home_score < away_score:
                result = "A"
            else:
                result = "D"

            # Update formation matchup stats
            self._update_matchup(home_form, away_form, result)

            # Update style matchup stats
            home_style = get_formation_style(home_form)
            away_style = get_formation_style(away_form)
            self._update_style_matchup(home_style, away_style, result)

        self.loaded = True
        log.info(f"Built matchup matrix with {len(self.matchups)} formation pairs")
        return True

    def _update_matchup(self, home_form: str, away_form: str, result: str):
        """Update formation matchup statistics."""
        if home_form not in self.matchups:
            self.matchups[home_form] = {}
        if away_form not in self.matchups[home_form]:
            self.matchups[home_form][away_form] = {
                "matches": 0, "home_wins": 0, "draws": 0, "away_wins": 0
            }

        self.matchups[home_form][away_form]["matches"] += 1
        if result == "H":
            self.matchups[home_form][away_form]["home_wins"] += 1
        elif result == "D":
            self.matchups[home_form][away_form]["draws"] += 1
        else:
            self.matchups[home_form][away_form]["away_wins"] += 1

    def _update_style_matchup(self, home_style: str, away_style: str, result: str):
        """Update style matchup statistics."""
        if home_style not in self.style_matchups:
            self.style_matchups[home_style] = {}
        if away_style not in self.style_matchups[home_style]:
            self.style_matchups[home_style][away_style] = {
                "matches": 0, "home_wins": 0, "draws": 0, "away_wins": 0
            }

        self.style_matchups[home_style][away_style]["matches"] += 1
        if result == "H":
            self.style_matchups[home_style][away_style]["home_wins"] += 1
        elif result == "D":
            self.style_matchups[home_style][away_style]["draws"] += 1
        else:
            self.style_matchups[home_style][away_style]["away_wins"] += 1

    def get_matchup_stats(self, home_form: str, away_form: str) -> Dict:
        """Get historical stats for a formation matchup."""
        if home_form in self.matchups and away_form in self.matchups[home_form]:
            stats = self.matchups[home_form][away_form]
            n = stats["matches"]
            if n >= 5:  # Minimum sample size
                return {
                    "matches": n,
                    "home_win_rate": stats["home_wins"] / n,
                    "draw_rate": stats["draws"] / n,
                    "away_win_rate": stats["away_wins"] / n,
                    "confidence": "high" if n >= 20 else "medium",
                }

        # Fall back to style matchup
        return self.get_style_matchup_stats(
            get_formation_style(home_form),
            get_formation_style(away_form)
        )

    def get_style_matchup_stats(self, home_style: str, away_style: str) -> Dict:
        """Get historical stats for a style matchup."""
        if home_style in self.style_matchups and away_style in self.style_matchups[home_style]:
            stats = self.style_matchups[home_style][away_style]
            n = stats["matches"]
            if n >= 10:
                return {
                    "matches": n,
                    "home_win_rate": stats["home_wins"] / n,
                    "draw_rate": stats["draws"] / n,
                    "away_win_rate": stats["away_wins"] / n,
                    "confidence": "medium",
                }

        # Default rates
        return {
            "matches": 0,
            "home_win_rate": 0.45,
            "draw_rate": 0.27,
            "away_win_rate": 0.28,
            "confidence": "low",
        }

    def save(self, path: Path = None):
        """Save matchup matrix to JSON."""
        if path is None:
            path = DATA_DIR / "features" / "formation_matchups.json"

        data = {
            "formations": self.matchups,
            "styles": self.style_matchups,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        log.info(f"Saved formation matchups to {path}")

    def load(self, path: Path = None) -> bool:
        """Load matchup matrix from JSON."""
        if path is None:
            path = DATA_DIR / "features" / "formation_matchups.json"

        if not path.exists():
            return False

        with open(path) as f:
            data = json.load(f)

        self.matchups = data.get("formations", {})
        self.style_matchups = data.get("styles", {})
        self.loaded = True
        log.info(f"Loaded formation matchups")
        return True


# =============================================================================
# TEAM FORMATION PROFILE
# =============================================================================

class TeamFormationProfile:
    """Tracks a team's formation history and preferences."""

    def __init__(self, team: str):
        self.team = team
        self.formation_history: List[Dict] = []  # [{match_id, formation, result}]
        self.formation_counts: Dict[str, int] = defaultdict(int)

    @property
    def preferred_formation(self) -> str:
        """Get team's most used formation."""
        if not self.formation_counts:
            return "4-2-3-1"
        return max(self.formation_counts, key=self.formation_counts.get)

    @property
    def formation_flexibility(self) -> float:
        """Formation flexibility index (0 = always same, 1 = always different).

        Calculated as entropy of formation distribution normalized by max entropy.
        """
        if not self.formation_counts:
            return 0.5

        total = sum(self.formation_counts.values())
        if total < 5:
            return 0.5

        probs = [c / total for c in self.formation_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(len(self.formation_counts)) if len(self.formation_counts) > 1 else 1

        return entropy / max_entropy if max_entropy > 0 else 0

    @property
    def recent_formation(self) -> str:
        """Get formation from most recent match."""
        if not self.formation_history:
            return self.preferred_formation
        return self.formation_history[-1].get("formation", self.preferred_formation)

    @property
    def formation_changed_recently(self) -> bool:
        """Check if formation changed in last 3 matches."""
        if len(self.formation_history) < 3:
            return False

        recent = [h["formation"] for h in self.formation_history[-3:]]
        return len(set(recent)) > 1

    def add_match(self, match_id: str, formation: str, result: str = None):
        """Add a match to the team's history."""
        self.formation_history.append({
            "match_id": match_id,
            "formation": formation,
            "result": result,
        })
        self.formation_counts[formation] += 1

        # Keep only last 38 matches
        if len(self.formation_history) > 38:
            old = self.formation_history.pop(0)
            self.formation_counts[old["formation"]] -= 1
            if self.formation_counts[old["formation"]] <= 0:
                del self.formation_counts[old["formation"]]

    def get_formation_results(self, formation: str) -> Dict:
        """Get results when playing a specific formation."""
        matches = [h for h in self.formation_history if h["formation"] == formation]
        if not matches:
            return {"matches": 0, "wins": 0, "draws": 0, "losses": 0}

        wins = sum(1 for m in matches if m.get("result") == "W")
        draws = sum(1 for m in matches if m.get("result") == "D")
        losses = sum(1 for m in matches if m.get("result") == "L")

        return {
            "matches": len(matches),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / len(matches) if matches else 0,
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "team": self.team,
            "preferred_formation": self.preferred_formation,
            "recent_formation": self.recent_formation,
            "flexibility": round(self.formation_flexibility, 3),
            "formation_changed_recently": self.formation_changed_recently,
            "formation_counts": dict(self.formation_counts),
        }


# =============================================================================
# FORMATION DATABASE
# =============================================================================

class FormationDatabase:
    """Database of team formation profiles."""

    def __init__(self):
        self.profiles: Dict[str, TeamFormationProfile] = {}
        self.matchup_matrix = FormationMatchupMatrix()
        self.loaded = False

    def get_profile(self, team: str) -> TeamFormationProfile:
        """Get or create a team profile."""
        if team not in self.profiles:
            self.profiles[team] = TeamFormationProfile(team)
        return self.profiles[team]

    def build_from_data(self) -> bool:
        """Build database from lineup and match data."""
        lineups_path = parsed_path("lineups")
        matches_path = parsed_path("matches")

        if not lineups_path.exists():
            log.warning("Lineups data not found")
            return False

        lineups = pd.read_parquet(lineups_path)
        formations = lineups.groupby(["match_id", "team"])["formation"].first().reset_index()

        # Load matches for results
        matches = pd.read_parquet(matches_path) if matches_path.exists() else pd.DataFrame()

        log.info(f"Building formation database from {len(formations)} entries")

        for _, row in formations.iterrows():
            match_id = row["match_id"]
            team = row["team"]
            formation = row["formation"]

            # Get result for this team in this match
            result = None
            if not matches.empty:
                match_row = matches[matches["match_id"] == match_id]
                if not match_row.empty:
                    match_row = match_row.iloc[0]
                    home = match_row.get("home_team")
                    away = match_row.get("away_team")
                    home_score = match_row.get("home_score")
                    away_score = match_row.get("away_score")

                    if pd.notna(home_score) and pd.notna(away_score):
                        if team == home:
                            result = "W" if home_score > away_score else ("D" if home_score == away_score else "L")
                        elif team == away:
                            result = "W" if away_score > home_score else ("D" if away_score == home_score else "L")

            profile = self.get_profile(team)
            profile.add_match(match_id, formation, result)

        # Build matchup matrix
        self.matchup_matrix.build_from_matches()

        self.loaded = True
        log.info(f"Built formation database for {len(self.profiles)} teams")
        return True

    def predict_formations(self, home_team: str, away_team: str) -> Dict:
        """Predict likely formations for an upcoming match."""
        home_profile = self.get_profile(home_team)
        away_profile = self.get_profile(away_team)

        return {
            "home_formation": home_profile.recent_formation,
            "away_formation": away_profile.recent_formation,
            "home_preferred": home_profile.preferred_formation,
            "away_preferred": away_profile.preferred_formation,
            "home_flexibility": home_profile.formation_flexibility,
            "away_flexibility": away_profile.formation_flexibility,
        }

    def get_matchup_advantage(self, home_team: str, away_team: str) -> Dict:
        """Get formation-based matchup advantage."""
        formations = self.predict_formations(home_team, away_team)
        home_form = formations["home_formation"]
        away_form = formations["away_formation"]

        # Get matchup stats
        matchup_stats = self.matchup_matrix.get_matchup_stats(home_form, away_form)

        # Calculate advantage
        base_home_rate = 0.45
        home_advantage = matchup_stats["home_win_rate"] - base_home_rate

        # Style-based factors
        home_style = get_formation_style(home_form)
        away_style = get_formation_style(away_form)

        # Attacking vs defensive advantage
        style_advantage = 0
        if home_style == "attacking" and away_style == "defensive":
            style_advantage = 0.03  # Attacking can break down defensive
        elif home_style == "defensive" and away_style == "attacking":
            style_advantage = -0.02  # Counter-attack opportunity for away

        # Width mismatch
        home_width = classify_formation(home_form).get("width", "wide")
        away_width = classify_formation(away_form).get("width", "wide")
        width_advantage = 0
        if home_width == "wide" and away_width == "narrow":
            width_advantage = 0.02  # Wide team can exploit narrow defense

        total_advantage = home_advantage + style_advantage + width_advantage

        return {
            "home_formation": home_form,
            "away_formation": away_form,
            "matchup_home_win_rate": matchup_stats["home_win_rate"],
            "matchup_draw_rate": matchup_stats["draw_rate"],
            "matchup_away_win_rate": matchup_stats["away_win_rate"],
            "formation_advantage": round(home_advantage, 3),
            "style_advantage": round(style_advantage, 3),
            "width_advantage": round(width_advantage, 3),
            "total_advantage": round(total_advantage, 3),
            "confidence": matchup_stats["confidence"],
        }

    def save(self, path: Path = None):
        """Save database to JSON."""
        if path is None:
            path = DATA_DIR / "features" / "formation_database.json"

        data = {
            team: profile.to_dict()
            for team, profile in self.profiles.items()
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # Also save matchup matrix
        self.matchup_matrix.save()

        log.info(f"Saved formation database to {path}")

    def load(self, path: Path = None) -> bool:
        """Load database from JSON."""
        if path is None:
            path = DATA_DIR / "features" / "formation_database.json"

        if not path.exists():
            return False

        with open(path) as f:
            data = json.load(f)

        for team, profile_data in data.items():
            profile = self.get_profile(team)
            profile.formation_counts = defaultdict(int, profile_data.get("formation_counts", {}))

        self.matchup_matrix.load()
        self.loaded = True
        log.info(f"Loaded formation database for {len(self.profiles)} teams")
        return True


# =============================================================================
# FORMATION FEATURES FOR MATCHES
# =============================================================================

def add_formation_features(matches: pd.DataFrame, formation_db: FormationDatabase = None) -> pd.DataFrame:
    """Add formation-based features to match DataFrame.

    Features added:
    - home_formation, away_formation (predicted)
    - home_formation_flexibility, away_formation_flexibility
    - formation_matchup_home_rate, formation_matchup_draw_rate
    - formation_total_advantage
    - style_mismatch (attacking vs defensive)
    """
    if formation_db is None:
        formation_db = FormationDatabase()
        if not formation_db.load():
            formation_db.build_from_data()
            formation_db.save()

    df = matches.copy()

    # Initialize columns
    df["home_formation"] = ""
    df["away_formation"] = ""
    df["home_formation_flexibility"] = 0.5
    df["away_formation_flexibility"] = 0.5
    df["formation_matchup_home_rate"] = 0.45
    df["formation_matchup_draw_rate"] = 0.27
    df["formation_total_advantage"] = 0.0
    df["formation_confidence"] = "low"
    df["formation_style_mismatch"] = 0
    df["formation_width_mismatch"] = 0

    for idx, row in df.iterrows():
        home = row.get("home_team")
        away = row.get("away_team")

        if not home or not away:
            continue

        # Get formations
        formations = formation_db.predict_formations(home, away)
        df.at[idx, "home_formation"] = formations["home_formation"]
        df.at[idx, "away_formation"] = formations["away_formation"]
        df.at[idx, "home_formation_flexibility"] = formations["home_flexibility"]
        df.at[idx, "away_formation_flexibility"] = formations["away_flexibility"]

        # Get matchup advantage
        advantage = formation_db.get_matchup_advantage(home, away)
        df.at[idx, "formation_matchup_home_rate"] = advantage["matchup_home_win_rate"]
        df.at[idx, "formation_matchup_draw_rate"] = advantage["matchup_draw_rate"]
        df.at[idx, "formation_total_advantage"] = advantage["total_advantage"]
        df.at[idx, "formation_confidence"] = advantage["confidence"]

        # Style mismatch: 1 if attacking vs defensive (or vice versa)
        home_style = get_formation_style(formations["home_formation"])
        away_style = get_formation_style(formations["away_formation"])
        style_pair = {home_style, away_style}
        if "attacking" in style_pair and "defensive" in style_pair:
            df.at[idx, "formation_style_mismatch"] = 1

        # Width mismatch: 1 if wide vs narrow
        home_cls = classify_formation(formations["home_formation"])
        away_cls = classify_formation(formations["away_formation"])
        width_pair = {home_cls.get("width", "wide"), away_cls.get("width", "wide")}
        if "wide" in width_pair and "narrow" in width_pair:
            df.at[idx, "formation_width_mismatch"] = 1

    log.info(f"Added formation features to {len(df)} matches")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 70)
    print("FORMATION ANALYSIS SYSTEM - Phase 3")
    print("=" * 70)

    # Build database
    db = FormationDatabase()
    db.build_from_data()
    db.save()

    # Show team profiles
    print("\n=== Team Formation Profiles ===")
    for team in sorted(db.profiles.keys()):
        profile = db.profiles[team]
        print(f"{team:20} Preferred: {profile.preferred_formation:10} "
              f"Flexibility: {profile.formation_flexibility:.2f} "
              f"Changed: {profile.formation_changed_recently}")

    # Show matchup matrix samples
    print("\n=== Formation Matchup Samples ===")
    matchups = db.matchup_matrix.matchups
    for home_form in list(matchups.keys())[:5]:
        for away_form in list(matchups[home_form].keys())[:3]:
            stats = matchups[home_form][away_form]
            if stats["matches"] >= 5:
                print(f"{home_form} vs {away_form}: "
                      f"H={stats['home_wins']/stats['matches']:.1%} "
                      f"D={stats['draws']/stats['matches']:.1%} "
                      f"A={stats['away_wins']/stats['matches']:.1%} "
                      f"(n={stats['matches']})")

    # Test predictions
    print("\n=== Matchup Predictions ===")
    test_matches = [
        ("Inter", "Milan"),
        ("Juventus", "Napoli"),
        ("Roma", "Lazio"),
        ("Atalanta", "Fiorentina"),
    ]

    for home, away in test_matches:
        advantage = db.get_matchup_advantage(home, away)
        print(f"\n{home} ({advantage['home_formation']}) vs {away} ({advantage['away_formation']})")
        print(f"  Matchup rates: H={advantage['matchup_home_win_rate']:.1%} "
              f"D={advantage['matchup_draw_rate']:.1%} A={advantage['matchup_away_win_rate']:.1%}")
        print(f"  Total advantage: {advantage['total_advantage']:+.3f} ({advantage['confidence']})")
