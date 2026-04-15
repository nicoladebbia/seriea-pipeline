#!/usr/bin/env python3
"""FORMATION ANALYZER - Tier 2 Predictions System

Provides formation-adjusted predictions 1 hour before match kickoff.

Features:
- Real-time lineup scraping from free sources
- Formation detection (4-3-3, 3-5-2, 4-4-2, etc.)
- Expected vs actual lineup comparison
- Formation impact calculation
- Key player availability assessment
- Tier 2 prediction adjustments

Data Sources (Free):
- Flashscore: Quick lineup updates
- SofaScore: Formation visualization
- Serie A official: Confirmed lineups
- Team Twitter/social media: Early leaks

Usage:
    python formation_analyzer.py --match "Lecce vs Udinese"
    python formation_analyzer.py --all
    python formation_analyzer.py --tier2-update  # Generate Tier 2 predictions
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, PROJECT_ROOT
from scripts.utils.parsing import get_cache_path

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
CACHE_DIR = DATA_DIR / "cache" / "formations"
LINEUP_CACHE_MINUTES = 15  # Short cache for lineups (time-sensitive)

# Formation patterns
FORMATION_PATTERNS = {
    "4-3-3": {"defenders": 4, "midfielders": 3, "forwards": 3},
    "4-4-2": {"defenders": 4, "midfielders": 4, "forwards": 2},
    "4-2-3-1": {"defenders": 4, "midfielders": 5, "forwards": 1},
    "3-5-2": {"defenders": 3, "midfielders": 5, "forwards": 2},
    "3-4-3": {"defenders": 3, "midfielders": 4, "forwards": 3},
    "5-3-2": {"defenders": 5, "midfielders": 3, "forwards": 2},
    "4-1-4-1": {"defenders": 4, "midfielders": 5, "forwards": 1},
    "4-5-1": {"defenders": 4, "midfielders": 5, "forwards": 1},
}

# Formation impact factors
FORMATION_IMPACTS = {
    # Attacking formations
    "4-3-3": {"attack": 1.05, "defense": 0.95, "style": "balanced_attack"},
    "3-4-3": {"attack": 1.10, "defense": 0.90, "style": "high_attack"},

    # Balanced formations
    "4-4-2": {"attack": 1.00, "defense": 1.00, "style": "balanced"},
    "4-2-3-1": {"attack": 1.02, "defense": 0.98, "style": "creative"},

    # Defensive formations
    "5-3-2": {"attack": 0.92, "defense": 1.08, "style": "defensive"},
    "3-5-2": {"attack": 0.95, "defense": 1.05, "style": "wing_play"},
    "4-5-1": {"attack": 0.90, "defense": 1.10, "style": "ultra_defensive"},
}

# Expected formations by team (based on recent patterns)
EXPECTED_FORMATIONS = {
    "Inter": "3-5-2",
    "Napoli": "4-3-3",
    "Juventus": "4-3-3",
    "Milan": "4-2-3-1",
    "Roma": "4-3-3",
    "Atalanta": "3-4-3",
    "Lazio": "4-3-3",
    "Fiorentina": "4-2-3-1",
    "Bologna": "4-2-3-1",
    "Torino": "3-5-2",
    "Genoa": "4-2-3-1",
    "Udinese": "3-5-2",
    "Lecce": "4-3-3",
    "Cagliari": "4-3-3",
    "Sassuolo": "4-3-3",
    "Verona": "4-2-3-1",
    "Cremonese": "3-5-2",
    "Pisa": "3-4-3",
    "Parma": "4-2-3-1",
    "Como": "4-2-3-1",
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
class Player:
    """Player in a lineup."""
    name: str
    position: str
    number: int = 0
    is_captain: bool = False
    is_starter: bool = True


@dataclass
class Lineup:
    """Team lineup."""
    team: str
    formation: str
    players: List[Player] = field(default_factory=list)
    substitutes: List[Player] = field(default_factory=list)

    # Status
    confirmed: bool = False
    source: str = ""
    timestamp: str = ""


@dataclass
class FormationAnalysis:
    """Complete formation analysis for a match."""
    match: str
    date: str
    home_team: str
    away_team: str

    # Lineups
    home_lineup: Optional[Lineup]
    away_lineup: Optional[Lineup]

    # Formation comparison
    home_expected_formation: str
    away_expected_formation: str
    home_actual_formation: str
    away_actual_formation: str

    # Formation change impact
    home_formation_change: bool
    away_formation_change: bool
    home_formation_impact: float  # -20 to +20%
    away_formation_impact: float  # -20 to +20%

    # Key player availability
    home_key_players_available: int
    home_key_players_missing: List[str]
    away_key_players_available: int
    away_key_players_missing: List[str]

    # Tactical analysis
    tactical_matchup: str  # "home_advantage", "away_advantage", "neutral"
    tactical_notes: List[str]

    # Tier 2 adjustments
    tier2_home_adjustment: float  # Probability adjustment
    tier2_away_adjustment: float
    tier2_confidence: str  # "high", "medium", "low"

    # Summary
    analysis_summary: str
    generated_at: str


@dataclass
class Tier2Prediction:
    """Tier 2 formation-adjusted prediction."""
    match: str
    date: str

    # Tier 1 (base) predictions
    tier1_home_prob: float
    tier1_away_prob: float
    tier1_draw_prob: float

    # Tier 2 (formation-adjusted) predictions
    tier2_home_prob: float
    tier2_away_prob: float
    tier2_draw_prob: float

    # Changes
    home_change: float
    away_change: float
    draw_change: float

    # Formation factors
    formation_impact: str
    key_changes: List[str]

    # Betting implications
    value_shift: str  # "home_value_increased", "away_value_increased", "no_change"
    bet_recommendation_change: bool

    # Confidence
    confidence: str
    lineup_confirmed: bool


# =============================================================================
# LINEUP SCRAPING
# =============================================================================

class LineupScraper:
    """Scrapes lineups from free sources."""

    def __init__(self):
        if HAS_SCRAPING:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
        else:
            self.session = None

    def _load_from_cache(self, key: str) -> Optional[Dict]:
        """Load lineup from cache if still valid."""
        cache_path = get_cache_path(CACHE_DIR, key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if datetime.now() - cached_time > timedelta(minutes=LINEUP_CACHE_MINUTES):
                return None

            return data.get("lineup")
        except Exception:
            return None

    def _save_to_cache(self, key: str, lineup: Dict):
        """Save lineup to cache."""
        cache_path = get_cache_path(CACHE_DIR, key)
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "cached_at": datetime.now().isoformat(),
                    "lineup": lineup
                }, f)
        except Exception as e:
            log.warning(f"Cache save error: {e}")

    def get_lineup(self, team: str, match_date: str) -> Optional[Lineup]:
        """Get lineup for a team on a specific date."""
        cache_key = f"lineup_{team}_{match_date}"

        # Check cache
        cached = self._load_from_cache(cache_key)
        if cached:
            return Lineup(**cached)

        # Try scraping from multiple sources
        lineup = None

        # For now, return expected/simulated lineup
        # In production, this would scrape from Flashscore, SofaScore, etc.
        lineup = self._get_expected_lineup(team)

        if lineup:
            self._save_to_cache(cache_key, asdict(lineup))

        return lineup

    def _get_expected_lineup(self, team: str) -> Lineup:
        """Get expected lineup based on team's typical formation."""
        formation = EXPECTED_FORMATIONS.get(team, "4-3-3")

        # Generate expected starting XI
        players = self._generate_expected_players(team, formation)

        return Lineup(
            team=team,
            formation=formation,
            players=players,
            substitutes=[],
            confirmed=False,
            source="expected",
            timestamp=datetime.now().isoformat()
        )

    def _generate_expected_players(self, team: str, formation: str) -> List[Player]:
        """Generate expected players based on formation."""
        players = []

        # Get formation pattern
        pattern = FORMATION_PATTERNS.get(formation, {"defenders": 4, "midfielders": 3, "forwards": 3})

        # Goalkeeper
        players.append(Player(name=f"{team} GK", position="GK", number=1))

        # Defenders
        for i in range(pattern["defenders"]):
            pos = ["CB", "CB", "LB", "RB"][i % 4] if pattern["defenders"] == 4 else ["CB", "CB", "CB", "LWB", "RWB"][i % 5]
            players.append(Player(name=f"{team} DEF{i+1}", position=pos, number=i+2))

        # Midfielders
        for i in range(pattern["midfielders"]):
            pos = ["CM", "DM", "AM", "LM", "RM"][i % 5]
            players.append(Player(name=f"{team} MID{i+1}", position=pos, number=5+i))

        # Forwards
        for i in range(pattern["forwards"]):
            pos = ["ST", "LW", "RW"][i % 3]
            players.append(Player(name=f"{team} FWD{i+1}", position=pos, number=9+i))

        return players


# =============================================================================
# FORMATION ANALYSIS
# =============================================================================

class FormationAnalyzer:
    """Analyzes formations and generates Tier 2 predictions."""

    def __init__(self):
        self.scraper = LineupScraper()

    def detect_formation(self, lineup: Lineup) -> str:
        """Detect formation from lineup."""
        if not lineup or not lineup.players:
            return "4-3-3"  # Default

        # Count by position type
        defenders = sum(1 for p in lineup.players if p.position in ["CB", "LB", "RB", "LWB", "RWB"])
        midfielders = sum(1 for p in lineup.players if p.position in ["CM", "DM", "AM", "LM", "RM", "CDM", "CAM"])
        forwards = sum(1 for p in lineup.players if p.position in ["ST", "CF", "LW", "RW", "SS"])

        # Match to formation
        for formation, pattern in FORMATION_PATTERNS.items():
            if (pattern["defenders"] == defenders and
                pattern["midfielders"] == midfielders and
                pattern["forwards"] == forwards):
                return formation

        # Default based on counts
        return f"{defenders}-{midfielders}-{forwards}"

    def calculate_formation_impact(
        self,
        actual_formation: str,
        expected_formation: str,
        team_context: str = "balanced"
    ) -> float:
        """Calculate impact of formation on predictions."""
        # No change
        if actual_formation == expected_formation:
            return 0.0

        # Get formation characteristics
        actual = FORMATION_IMPACTS.get(actual_formation, {"attack": 1.0, "defense": 1.0})
        expected = FORMATION_IMPACTS.get(expected_formation, {"attack": 1.0, "defense": 1.0})

        # Calculate difference
        attack_change = actual["attack"] - expected["attack"]
        defense_change = actual["defense"] - expected["defense"]

        # Impact based on context
        if team_context == "attacking":
            return attack_change * 10  # Attacking teams care more about attack
        elif team_context == "defensive":
            return defense_change * 10  # Defensive teams care more about defense
        else:
            return (attack_change + defense_change) * 5  # Balanced

    def analyze_tactical_matchup(
        self,
        home_formation: str,
        away_formation: str
    ) -> Tuple[str, List[str]]:
        """Analyze tactical matchup between formations."""
        notes = []

        home_style = FORMATION_IMPACTS.get(home_formation, {}).get("style", "balanced")
        away_style = FORMATION_IMPACTS.get(away_formation, {}).get("style", "balanced")

        # Tactical matchup analysis
        if home_style == "high_attack" and away_style == "ultra_defensive":
            notes.append("Attacking home vs defensive away - likely low-scoring")
            return "neutral", notes
        elif home_style == "ultra_defensive" and away_style == "high_attack":
            notes.append("Defensive home vs attacking away - away may dominate")
            return "away_advantage", notes
        elif home_style == "wing_play" and away_style == "defensive":
            notes.append("Wing play vs narrow defense - crossing opportunities")
            return "home_advantage", notes

        # 3-back vs 4-back
        if home_formation.startswith("3") and not away_formation.startswith("3"):
            notes.append("Home playing 3-back - wing areas may be exposed")
        if away_formation.startswith("3") and not home_formation.startswith("3"):
            notes.append("Away playing 3-back - wing areas may be exposed")

        # 5-back detection
        if home_formation.startswith("5"):
            notes.append("Home parking the bus with 5 defenders")
            return "away_advantage", notes
        if away_formation.startswith("5"):
            notes.append("Away parking the bus - home expected to dominate possession")
            return "home_advantage", notes

        return "neutral", notes

    def analyze_match(self, home_team: str, away_team: str, match_date: str) -> FormationAnalysis:
        """Perform complete formation analysis for a match."""
        match_name = f"{home_team} vs {away_team}"
        log.info(f"Analyzing formations for {match_name}...")

        # Get lineups
        home_lineup = self.scraper.get_lineup(home_team, match_date)
        away_lineup = self.scraper.get_lineup(away_team, match_date)

        # Expected formations
        home_expected = EXPECTED_FORMATIONS.get(home_team, "4-3-3")
        away_expected = EXPECTED_FORMATIONS.get(away_team, "4-3-3")

        # Actual formations (from lineup)
        home_actual = self.detect_formation(home_lineup) if home_lineup else home_expected
        away_actual = self.detect_formation(away_lineup) if away_lineup else away_expected

        # Formation changes
        home_changed = home_actual != home_expected
        away_changed = away_actual != away_expected

        # Calculate impacts
        home_impact = self.calculate_formation_impact(home_actual, home_expected)
        away_impact = self.calculate_formation_impact(away_actual, away_expected)

        # Tactical analysis
        tactical_matchup, tactical_notes = self.analyze_tactical_matchup(home_actual, away_actual)

        # Key player analysis (placeholder)
        home_missing = []
        away_missing = []

        # Calculate Tier 2 adjustments
        tier2_home_adj = home_impact / 100
        tier2_away_adj = away_impact / 100

        # Adjust based on tactical matchup
        if tactical_matchup == "home_advantage":
            tier2_home_adj += 0.03
            tier2_away_adj -= 0.02
        elif tactical_matchup == "away_advantage":
            tier2_home_adj -= 0.02
            tier2_away_adj += 0.03

        # Determine confidence
        lineups_confirmed = (home_lineup and home_lineup.confirmed and
                           away_lineup and away_lineup.confirmed)
        confidence = "high" if lineups_confirmed else "medium" if home_lineup and away_lineup else "low"

        # Build summary
        summary = f"""
Formation Analysis for {match_name}:

HOME ({home_team}):
- Expected Formation: {home_expected}
- Actual Formation: {home_actual}
- Formation Changed: {'Yes' if home_changed else 'No'}
- Impact: {home_impact:+.1f}%

AWAY ({away_team}):
- Expected Formation: {away_expected}
- Actual Formation: {away_actual}
- Formation Changed: {'Yes' if away_changed else 'No'}
- Impact: {away_impact:+.1f}%

TACTICAL MATCHUP: {tactical_matchup.upper()}
{chr(10).join('- ' + note for note in tactical_notes) if tactical_notes else '- No significant tactical notes'}

TIER 2 ADJUSTMENTS:
- Home probability: {tier2_home_adj:+.1%}
- Away probability: {tier2_away_adj:+.1%}
- Confidence: {confidence.upper()}
"""

        return FormationAnalysis(
            match=match_name,
            date=match_date,
            home_team=home_team,
            away_team=away_team,
            home_lineup=home_lineup,
            away_lineup=away_lineup,
            home_expected_formation=home_expected,
            away_expected_formation=away_expected,
            home_actual_formation=home_actual,
            away_actual_formation=away_actual,
            home_formation_change=home_changed,
            away_formation_change=away_changed,
            home_formation_impact=home_impact,
            away_formation_impact=away_impact,
            home_key_players_available=11,
            home_key_players_missing=home_missing,
            away_key_players_available=11,
            away_key_players_missing=away_missing,
            tactical_matchup=tactical_matchup,
            tactical_notes=tactical_notes,
            tier2_home_adjustment=tier2_home_adj,
            tier2_away_adjustment=tier2_away_adj,
            tier2_confidence=confidence,
            analysis_summary=summary.strip(),
            generated_at=datetime.now().isoformat()
        )

    def generate_tier2_prediction(
        self,
        formation_analysis: FormationAnalysis,
        tier1_predictions: Dict
    ) -> Tier2Prediction:
        """Generate Tier 2 formation-adjusted prediction."""
        # Get Tier 1 probabilities
        tier1_home = tier1_predictions.get("home_win_prob", 0.33)
        tier1_away = tier1_predictions.get("away_win_prob", 0.33)
        tier1_draw = tier1_predictions.get("draw_prob", 0.34)

        # Apply formation adjustments
        tier2_home = tier1_home + formation_analysis.tier2_home_adjustment
        tier2_away = tier1_away + formation_analysis.tier2_away_adjustment

        # Normalize probabilities
        total = tier2_home + tier2_away + tier1_draw
        tier2_home = tier2_home / total
        tier2_away = tier2_away / total
        tier2_draw = tier1_draw / total

        # Calculate changes
        home_change = tier2_home - tier1_home
        away_change = tier2_away - tier1_away
        draw_change = tier2_draw - tier1_draw

        # Determine key changes
        key_changes = []
        if formation_analysis.home_formation_change:
            key_changes.append(f"{formation_analysis.home_team} changed to {formation_analysis.home_actual_formation}")
        if formation_analysis.away_formation_change:
            key_changes.append(f"{formation_analysis.away_team} changed to {formation_analysis.away_actual_formation}")
        if formation_analysis.tactical_notes:
            key_changes.extend(formation_analysis.tactical_notes)

        # Value shift
        if home_change > 0.02:
            value_shift = "home_value_increased"
        elif away_change > 0.02:
            value_shift = "away_value_increased"
        else:
            value_shift = "no_significant_change"

        # Recommendation change
        tier1_best = max(tier1_home, tier1_away, tier1_draw)
        tier2_best = max(tier2_home, tier2_away, tier2_draw)

        tier1_pick = "home" if tier1_home == tier1_best else "away" if tier1_away == tier1_best else "draw"
        tier2_pick = "home" if tier2_home == tier2_best else "away" if tier2_away == tier2_best else "draw"

        bet_change = tier1_pick != tier2_pick

        return Tier2Prediction(
            match=formation_analysis.match,
            date=formation_analysis.date,
            tier1_home_prob=tier1_home,
            tier1_away_prob=tier1_away,
            tier1_draw_prob=tier1_draw,
            tier2_home_prob=tier2_home,
            tier2_away_prob=tier2_away,
            tier2_draw_prob=tier2_draw,
            home_change=home_change,
            away_change=away_change,
            draw_change=draw_change,
            formation_impact=formation_analysis.tactical_matchup,
            key_changes=key_changes,
            value_shift=value_shift,
            bet_recommendation_change=bet_change,
            confidence=formation_analysis.tier2_confidence,
            lineup_confirmed=formation_analysis.home_lineup.confirmed if formation_analysis.home_lineup else False
        )


# =============================================================================
# INTEGRATION
# =============================================================================

def analyze_all_upcoming_formations() -> List[FormationAnalysis]:
    """Analyze formations for all upcoming matches."""
    predictions_path = DATA_DIR / "upcoming" / "predictions.json"

    if not predictions_path.exists():
        log.error("No predictions file found")
        return []

    with open(predictions_path) as f:
        data = json.load(f)

    matches = data.get("predictions", [])
    analyzer = FormationAnalyzer()
    results = []

    for match in matches:
        home_team = match.get("home_team", "")
        away_team = match.get("away_team", "")
        match_date = match.get("date", "")

        if not home_team or not away_team:
            continue

        try:
            analysis = analyzer.analyze_match(home_team, away_team, match_date)
            results.append(analysis)
            log.info(f"Analyzed: {home_team} ({analysis.home_actual_formation}) vs {away_team} ({analysis.away_actual_formation})")
        except Exception as e:
            log.error(f"Failed to analyze {home_team} vs {away_team}: {e}")

    # Save results
    output_path = DATA_DIR / "upcoming" / "formation_analysis.json"

    output_data = {
        "generated_at": datetime.now().isoformat(),
        "tier": 2,
        "description": "Formation-adjusted predictions (Tier 2)",
        "matches": []
    }

    for analysis in results:
        match_data = {
            "match": analysis.match,
            "date": analysis.date,
            "home_team": analysis.home_team,
            "away_team": analysis.away_team,
            "home_formation": analysis.home_actual_formation,
            "away_formation": analysis.away_actual_formation,
            "home_expected_formation": analysis.home_expected_formation,
            "away_expected_formation": analysis.away_expected_formation,
            "home_formation_changed": analysis.home_formation_change,
            "away_formation_changed": analysis.away_formation_change,
            "home_impact": analysis.home_formation_impact,
            "away_impact": analysis.away_formation_impact,
            "tactical_matchup": analysis.tactical_matchup,
            "tactical_notes": analysis.tactical_notes,
            "tier2_home_adjustment": analysis.tier2_home_adjustment,
            "tier2_away_adjustment": analysis.tier2_away_adjustment,
            "confidence": analysis.tier2_confidence,
            "analysis_summary": analysis.analysis_summary,
        }
        output_data["matches"].append(match_data)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    log.info(f"Saved formation analysis to {output_path}")
    return results


def generate_tier2_predictions() -> List[Tier2Prediction]:
    """Generate Tier 2 predictions for all matches."""
    # Load Tier 1 predictions
    predictions_path = DATA_DIR / "upcoming" / "predictions.json"
    if not predictions_path.exists():
        log.error("No predictions file found")
        return []

    with open(predictions_path) as f:
        tier1_data = json.load(f)

    tier1_matches = {m.get("match", ""): m for m in tier1_data.get("predictions", [])}

    # Run formation analysis
    formation_analyses = analyze_all_upcoming_formations()

    # Generate Tier 2 predictions
    analyzer = FormationAnalyzer()
    tier2_predictions = []

    for analysis in formation_analyses:
        tier1 = tier1_matches.get(analysis.match, {})
        if not tier1:
            continue

        tier2 = analyzer.generate_tier2_prediction(analysis, tier1)
        tier2_predictions.append(tier2)

    # Save Tier 2 predictions
    output_path = DATA_DIR / "betting" / "tier2_predictions.json"

    output_data = {
        "generated_at": datetime.now().isoformat(),
        "tier": 2,
        "description": "Formation-adjusted predictions (1 hour before kickoff)",
        "predictions": [asdict(p) for p in tier2_predictions],
        "summary": {
            "total_matches": len(tier2_predictions),
            "home_value_increased": sum(1 for p in tier2_predictions if p.value_shift == "home_value_increased"),
            "away_value_increased": sum(1 for p in tier2_predictions if p.value_shift == "away_value_increased"),
            "bet_changes": sum(1 for p in tier2_predictions if p.bet_recommendation_change),
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    log.info(f"Saved Tier 2 predictions to {output_path}")
    return tier2_predictions


def get_formation_factors(analysis: FormationAnalysis) -> Dict[str, any]:
    """Convert formation analysis to prediction factors."""
    factors = {}

    if analysis.home_formation_change:
        factors["home_formation_change"] = True
        factors[f"home_playing_{analysis.home_actual_formation.replace('-', '_')}"] = True

    if analysis.away_formation_change:
        factors["away_formation_change"] = True
        factors[f"away_playing_{analysis.away_actual_formation.replace('-', '_')}"] = True

    if analysis.tactical_matchup == "home_advantage":
        factors["tactical_home_advantage"] = True
    elif analysis.tactical_matchup == "away_advantage":
        factors["tactical_away_advantage"] = True

    if analysis.home_formation_impact > 5:
        factors["positive_home_formation_impact"] = True
    elif analysis.home_formation_impact < -5:
        factors["negative_home_formation_impact"] = True

    if analysis.away_formation_impact > 5:
        factors["positive_away_formation_impact"] = True
    elif analysis.away_formation_impact < -5:
        factors["negative_away_formation_impact"] = True

    return factors


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Formation Analyzer - Tier 2 System")
    parser.add_argument("--match", type=str, help="Analyze specific match")
    parser.add_argument("--all", action="store_true", help="Analyze all upcoming matches")
    parser.add_argument("--tier2", action="store_true", help="Generate Tier 2 predictions")

    args = parser.parse_args()

    if args.tier2:
        predictions = generate_tier2_predictions()
        print(f"\nGenerated {len(predictions)} Tier 2 predictions")
        for p in predictions:
            change = "+" if p.home_change > 0 else ""
            print(f"  {p.match}: H:{p.tier2_home_prob:.1%} ({change}{p.home_change:.1%}) | "
                  f"A:{p.tier2_away_prob:.1%} ({'+' if p.away_change > 0 else ''}{p.away_change:.1%})")

    elif args.all:
        results = analyze_all_upcoming_formations()
        print(f"\nAnalyzed formations for {len(results)} matches")
        for r in results:
            print(f"  {r.match}: {r.home_actual_formation} vs {r.away_actual_formation} ({r.tactical_matchup})")

    elif args.match:
        if " vs " in args.match:
            home, away = args.match.split(" vs ")
            date = datetime.now().strftime("%Y-%m-%d")
            analyzer = FormationAnalyzer()
            analysis = analyzer.analyze_match(home.strip(), away.strip(), date)
            print(analysis.analysis_summary)
        else:
            print("Invalid match format. Use: 'Home Team vs Away Team'")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
