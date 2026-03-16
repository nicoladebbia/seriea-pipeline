#!/usr/bin/env python3
"""LIVE BETTING SYSTEM - Tier 3 Real-Time Analysis

Provides in-play betting opportunities during matches.

Features:
- Real-time score and event tracking
- Momentum analysis
- Live value opportunity detection
- In-play probability adjustments
- Next goal predictions
- Over/under live updates

Data Sources (Free):
- Flashscore: Live scores and events
- SofaScore: Live statistics
- Serie A official: Official updates
- BBC Sport: Live text commentary

Usage:
    python live_betting.py --match "Lecce vs Udinese"
    python live_betting.py --monitor  # Monitor all live matches
    python live_betting.py --simulate  # Simulate a match for testing
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, PROJECT_ROOT

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Live update settings
UPDATE_INTERVAL_SECONDS = 30  # Check for updates every 30 seconds
MOMENTUM_WINDOW_MINUTES = 10  # Analyze last 10 minutes for momentum

# Live betting thresholds
LIVE_VALUE_THRESHOLD = 0.05  # 5% minimum edge for live bets
MOMENTUM_THRESHOLD = 0.6  # 60% momentum in one direction = significant

# Data directory
LIVE_DATA_DIR = DATA_DIR / "live"
LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class EventType(Enum):
    GOAL = "goal"
    RED_CARD = "red_card"
    YELLOW_CARD = "yellow_card"
    SUBSTITUTION = "substitution"
    PENALTY = "penalty"
    VAR_DECISION = "var"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"
    KICKOFF = "kickoff"


@dataclass
class MatchEvent:
    """Single match event."""
    minute: int
    event_type: str
    team: str  # "home" or "away"
    player: str
    detail: str
    timestamp: str


@dataclass
class LiveStats:
    """Live match statistics."""
    possession_home: float
    possession_away: float
    shots_home: int
    shots_away: int
    shots_on_target_home: int
    shots_on_target_away: int
    corners_home: int
    corners_away: int
    fouls_home: int
    fouls_away: int
    xg_home: float  # Expected goals if available
    xg_away: float


@dataclass
class LiveMatchData:
    """Complete live match data."""
    match: str
    home_team: str
    away_team: str

    # Score
    home_score: int
    away_score: int

    # Time
    minute: int
    status: str  # "not_started", "first_half", "half_time", "second_half", "finished"
    is_live: bool

    # Events
    events: List[MatchEvent] = field(default_factory=list)

    # Stats
    stats: Optional[LiveStats] = None

    # Metadata
    last_updated: str = ""


@dataclass
class MomentumAnalysis:
    """Momentum analysis for a match."""
    home_momentum: float  # -1 to +1 (negative = away momentum)
    momentum_trend: str  # "home_rising", "away_rising", "stable"
    recent_events: List[str]
    pressure_team: str  # "home", "away", "neutral"
    analysis: str


@dataclass
class LiveBetOpportunity:
    """Live betting opportunity."""
    match: str
    minute: int

    # Bet details
    market: str  # "next_goal", "match_result", "over_under", "corners"
    selection: str
    current_odds: float
    fair_odds: float

    # Value
    our_probability: float
    implied_probability: float
    value_edge: float

    # Confidence
    confidence: str  # "high", "medium", "low"
    momentum_support: bool

    # Reasoning
    reasoning: str
    factors: List[str]

    # Timing
    valid_until: str  # Opportunity may expire


@dataclass
class Tier3Analysis:
    """Complete Tier 3 live analysis for a match."""
    match: str
    match_data: LiveMatchData

    # Pre-match predictions for comparison
    tier1_home_prob: float
    tier1_away_prob: float
    tier1_draw_prob: float

    # Live adjusted probabilities
    live_home_prob: float
    live_away_prob: float
    live_draw_prob: float

    # Changes from pre-match
    home_change: float
    away_change: float

    # Momentum
    momentum: MomentumAnalysis

    # Live opportunities
    opportunities: List[LiveBetOpportunity]

    # Summary
    summary: str
    generated_at: str


# =============================================================================
# LIVE DATA FETCHING
# =============================================================================

class LiveDataFetcher:
    """Fetches live match data from free sources."""

    def __init__(self):
        if HAS_REQUESTS:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            })
        else:
            self.session = None

    def get_live_matches(self) -> List[LiveMatchData]:
        """Get all currently live Serie A matches."""
        # In production, this would scrape from Flashscore/SofaScore
        # For now, return simulated/cached data
        live_file = LIVE_DATA_DIR / "current_live.json"

        if live_file.exists():
            try:
                with open(live_file) as f:
                    data = json.load(f)
                return [LiveMatchData(**m) for m in data.get("matches", [])]
            except Exception as e:
                log.warning(f"Error loading live data: {e}")

        return []

    def get_match_data(self, home_team: str, away_team: str) -> Optional[LiveMatchData]:
        """Get live data for a specific match."""
        live_matches = self.get_live_matches()

        for match in live_matches:
            if match.home_team == home_team and match.away_team == away_team:
                return match

        return None

    def simulate_match(self, home_team: str, away_team: str, minute: int = 45) -> LiveMatchData:
        """Simulate a live match for testing."""
        import random

        # Simulate score based on minute
        avg_goals_per_minute = 2.5 / 90  # ~2.5 goals per match
        expected_goals = minute * avg_goals_per_minute

        home_score = int(random.gauss(expected_goals * 0.55, 0.8))  # Home slight advantage
        away_score = int(random.gauss(expected_goals * 0.45, 0.7))
        home_score = max(0, home_score)
        away_score = max(0, away_score)

        # Generate events
        events = []
        for i in range(home_score):
            events.append(MatchEvent(
                minute=random.randint(1, minute),
                event_type=EventType.GOAL.value,
                team="home",
                player=f"{home_team} Player",
                detail="Goal",
                timestamp=datetime.now().isoformat()
            ))
        for i in range(away_score):
            events.append(MatchEvent(
                minute=random.randint(1, minute),
                event_type=EventType.GOAL.value,
                team="away",
                player=f"{away_team} Player",
                detail="Goal",
                timestamp=datetime.now().isoformat()
            ))

        # Sort by minute
        events.sort(key=lambda x: x.minute)

        # Simulate stats
        possession_home = random.uniform(40, 60)
        stats = LiveStats(
            possession_home=possession_home,
            possession_away=100 - possession_home,
            shots_home=random.randint(3, 10),
            shots_away=random.randint(2, 8),
            shots_on_target_home=random.randint(1, 5),
            shots_on_target_away=random.randint(0, 4),
            corners_home=random.randint(2, 7),
            corners_away=random.randint(1, 6),
            fouls_home=random.randint(5, 15),
            fouls_away=random.randint(4, 14),
            xg_home=home_score + random.uniform(-0.5, 1.0),
            xg_away=away_score + random.uniform(-0.5, 0.8)
        )

        # Determine status
        if minute < 45:
            status = "first_half"
        elif minute == 45:
            status = "half_time"
        elif minute < 90:
            status = "second_half"
        else:
            status = "finished"

        return LiveMatchData(
            match=f"{home_team} vs {away_team}",
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            minute=minute,
            status=status,
            is_live=status not in ["not_started", "finished"],
            events=events,
            stats=stats,
            last_updated=datetime.now().isoformat()
        )


# =============================================================================
# LIVE ANALYSIS
# =============================================================================

class LiveAnalyzer:
    """Analyzes live match data and generates betting opportunities."""

    def __init__(self):
        self.fetcher = LiveDataFetcher()

    def calculate_live_probabilities(
        self,
        match_data: LiveMatchData,
        tier1_home: float,
        tier1_away: float
    ) -> Tuple[float, float, float]:
        """Calculate live-adjusted probabilities based on current state."""
        home_score = match_data.home_score
        away_score = match_data.away_score
        minute = match_data.minute
        remaining = 90 - minute

        # Base adjustment for current score
        if home_score > away_score:
            # Home is winning
            lead = home_score - away_score
            home_boost = 0.15 * lead * (1 - remaining / 90)  # Larger boost as time runs out
            away_penalty = 0.10 * lead * (1 - remaining / 90)
        elif away_score > home_score:
            # Away is winning
            lead = away_score - home_score
            away_boost = 0.15 * lead * (1 - remaining / 90)
            home_penalty = 0.10 * lead * (1 - remaining / 90)
            home_boost = -home_penalty
            away_penalty = -away_boost
        else:
            # Draw
            home_boost = 0
            away_penalty = 0

        # Adjust for stats if available
        if match_data.stats:
            stats = match_data.stats

            # Possession adjustment
            poss_diff = (stats.possession_home - 50) / 100
            home_boost += poss_diff * 0.03

            # Shots adjustment
            if stats.shots_home + stats.shots_away > 0:
                shot_ratio = stats.shots_home / (stats.shots_home + stats.shots_away)
                home_boost += (shot_ratio - 0.5) * 0.05

            # xG adjustment
            if stats.xg_home + stats.xg_away > 0:
                xg_diff = (stats.xg_home - stats.xg_away) / (stats.xg_home + stats.xg_away)
                home_boost += xg_diff * 0.08

        # Calculate live probabilities
        live_home = tier1_home + home_boost
        live_away = tier1_away - home_boost  # Inverse adjustment

        # Ensure valid probabilities
        live_home = max(0.01, min(0.98, live_home))
        live_away = max(0.01, min(0.98, live_away))
        live_draw = 1 - live_home - live_away
        live_draw = max(0.01, live_draw)

        # Normalize
        total = live_home + live_away + live_draw
        return live_home / total, live_away / total, live_draw / total

    def analyze_momentum(self, match_data: LiveMatchData) -> MomentumAnalysis:
        """Analyze current match momentum."""
        if not match_data.events:
            return MomentumAnalysis(
                home_momentum=0,
                momentum_trend="stable",
                recent_events=[],
                pressure_team="neutral",
                analysis="No events to analyze momentum"
            )

        minute = match_data.minute

        # Analyze recent events (last 10 minutes)
        recent = [e for e in match_data.events if e.minute >= minute - MOMENTUM_WINDOW_MINUTES]

        home_events = sum(1 for e in recent if e.team == "home" and e.event_type in ["goal", "shot_on_target"])
        away_events = sum(1 for e in recent if e.team == "away" and e.event_type in ["goal", "shot_on_target"])

        # Calculate momentum (-1 to +1)
        total_events = home_events + away_events
        if total_events > 0:
            momentum = (home_events - away_events) / total_events
        else:
            momentum = 0

        # Add stats-based momentum if available
        if match_data.stats:
            stats = match_data.stats

            # Recent possession (weighted toward current state)
            poss_momentum = (stats.possession_home - 50) / 50

            # Shot momentum
            total_shots = stats.shots_home + stats.shots_away
            if total_shots > 0:
                shot_momentum = (stats.shots_home - stats.shots_away) / total_shots
            else:
                shot_momentum = 0

            # Combine
            momentum = momentum * 0.5 + poss_momentum * 0.25 + shot_momentum * 0.25

        # Determine trend
        if momentum > 0.3:
            trend = "home_rising"
            pressure = "home"
        elif momentum < -0.3:
            trend = "away_rising"
            pressure = "away"
        else:
            trend = "stable"
            pressure = "neutral"

        # Build analysis
        recent_event_strs = [f"{e.minute}' {e.event_type} ({e.team})" for e in recent[-5:]]

        analysis = f"Momentum: {momentum:+.2f} ({trend}). "
        if match_data.stats:
            analysis += f"Possession: {match_data.stats.possession_home:.0f}%-{match_data.stats.possession_away:.0f}%. "
            analysis += f"Shots: {match_data.stats.shots_home}-{match_data.stats.shots_away}."

        return MomentumAnalysis(
            home_momentum=momentum,
            momentum_trend=trend,
            recent_events=recent_event_strs,
            pressure_team=pressure,
            analysis=analysis
        )

    def find_live_opportunities(
        self,
        match_data: LiveMatchData,
        live_home: float,
        live_away: float,
        live_draw: float,
        momentum: MomentumAnalysis
    ) -> List[LiveBetOpportunity]:
        """Find live betting opportunities."""
        opportunities = []

        minute = match_data.minute
        remaining = 90 - minute

        # Skip if match nearly over
        if remaining < 5:
            return opportunities

        # === NEXT GOAL MARKET ===
        # Calculate next goal probabilities
        avg_goals_remaining = 2.5 * (remaining / 90)  # Scale by remaining time

        # Adjust for momentum
        if momentum.home_momentum > 0.3:
            home_next_goal = 0.55 + momentum.home_momentum * 0.15
        elif momentum.home_momentum < -0.3:
            home_next_goal = 0.45 + momentum.home_momentum * 0.15
        else:
            home_next_goal = 0.50

        away_next_goal = 1 - home_next_goal - 0.15  # 15% no more goals
        no_goal = 0.15

        # Simulated odds (would come from live odds API)
        home_next_odds = 1 / home_next_goal if home_next_goal > 0 else 10
        away_next_odds = 1 / away_next_goal if away_next_goal > 0 else 10

        # Check for value
        if momentum.home_momentum > 0.4:
            # Home momentum - check if odds offer value
            implied = 1 / home_next_odds
            edge = home_next_goal - implied
            if edge > LIVE_VALUE_THRESHOLD:
                opportunities.append(LiveBetOpportunity(
                    match=match_data.match,
                    minute=minute,
                    market="next_goal",
                    selection="Home",
                    current_odds=home_next_odds,
                    fair_odds=1 / home_next_goal,
                    our_probability=home_next_goal,
                    implied_probability=implied,
                    value_edge=edge,
                    confidence="high" if momentum.home_momentum > 0.5 else "medium",
                    momentum_support=True,
                    reasoning=f"Home has strong momentum ({momentum.home_momentum:.2f}). {momentum.analysis}",
                    factors=["home_momentum", "recent_pressure"],
                    valid_until=(datetime.now() + timedelta(minutes=5)).isoformat()
                ))

        if momentum.home_momentum < -0.4:
            # Away momentum
            implied = 1 / away_next_odds
            edge = away_next_goal - implied
            if edge > LIVE_VALUE_THRESHOLD:
                opportunities.append(LiveBetOpportunity(
                    match=match_data.match,
                    minute=minute,
                    market="next_goal",
                    selection="Away",
                    current_odds=away_next_odds,
                    fair_odds=1 / away_next_goal,
                    our_probability=away_next_goal,
                    implied_probability=implied,
                    value_edge=edge,
                    confidence="high" if momentum.home_momentum < -0.5 else "medium",
                    momentum_support=True,
                    reasoning=f"Away has strong momentum ({-momentum.home_momentum:.2f}). {momentum.analysis}",
                    factors=["away_momentum", "recent_pressure"],
                    valid_until=(datetime.now() + timedelta(minutes=5)).isoformat()
                ))

        # === OVER/UNDER MARKET ===
        total_goals = match_data.home_score + match_data.away_score
        expected_remaining = avg_goals_remaining

        # Over 0.5 more goals
        over_prob = 1 - (0.5 ** (expected_remaining * 1.2))  # Poisson approximation
        over_odds = 1 / over_prob if over_prob > 0 else 5

        if over_prob > 0.7 and remaining > 20:  # High confidence only
            opportunities.append(LiveBetOpportunity(
                match=match_data.match,
                minute=minute,
                market="over_under",
                selection=f"Over {total_goals + 0.5}",
                current_odds=over_odds,
                fair_odds=1 / over_prob,
                our_probability=over_prob,
                implied_probability=1 / over_odds,
                value_edge=over_prob - (1 / over_odds),
                confidence="medium",
                momentum_support=momentum.pressure_team != "neutral",
                reasoning=f"Expected {expected_remaining:.1f} more goals. Current: {total_goals}",
                factors=["time_remaining", "attacking_momentum"],
                valid_until=(datetime.now() + timedelta(minutes=10)).isoformat()
            ))

        return opportunities

    def analyze_live_match(
        self,
        home_team: str,
        away_team: str,
        tier1_home: float = 0.4,
        tier1_away: float = 0.35,
        tier1_draw: float = 0.25,
        simulated_minute: int = None
    ) -> Tier3Analysis:
        """Perform complete Tier 3 live analysis."""
        # Get or simulate match data
        if simulated_minute is not None:
            match_data = self.fetcher.simulate_match(home_team, away_team, simulated_minute)
        else:
            match_data = self.fetcher.get_match_data(home_team, away_team)
            if not match_data:
                # Return empty analysis if match not live
                return None

        # Calculate live probabilities
        live_home, live_away, live_draw = self.calculate_live_probabilities(
            match_data, tier1_home, tier1_away
        )

        # Analyze momentum
        momentum = self.analyze_momentum(match_data)

        # Find opportunities
        opportunities = self.find_live_opportunities(
            match_data, live_home, live_away, live_draw, momentum
        )

        # Build summary
        summary = f"""
TIER 3 LIVE ANALYSIS: {match_data.match}
========================================
Score: {match_data.home_score} - {match_data.away_score} ({match_data.minute}')
Status: {match_data.status.upper()}

PRE-MATCH (Tier 1):
  Home: {tier1_home:.1%} | Away: {tier1_away:.1%} | Draw: {tier1_draw:.1%}

LIVE ADJUSTED (Tier 3):
  Home: {live_home:.1%} ({live_home - tier1_home:+.1%}) | Away: {live_away:.1%} ({live_away - tier1_away:+.1%}) | Draw: {live_draw:.1%}

MOMENTUM:
  {momentum.analysis}
  Pressure: {momentum.pressure_team.upper()}

LIVE OPPORTUNITIES: {len(opportunities)}
"""
        for opp in opportunities:
            summary += f"\n  - {opp.market}: {opp.selection} @ {opp.current_odds:.2f} (Value: {opp.value_edge:.1%})"

        return Tier3Analysis(
            match=match_data.match,
            match_data=match_data,
            tier1_home_prob=tier1_home,
            tier1_away_prob=tier1_away,
            tier1_draw_prob=tier1_draw,
            live_home_prob=live_home,
            live_away_prob=live_away,
            live_draw_prob=live_draw,
            home_change=live_home - tier1_home,
            away_change=live_away - tier1_away,
            momentum=momentum,
            opportunities=opportunities,
            summary=summary.strip(),
            generated_at=datetime.now().isoformat()
        )


# =============================================================================
# INTEGRATION
# =============================================================================

def monitor_live_matches():
    """Monitor all live Serie A matches."""
    analyzer = LiveAnalyzer()

    log.info("=" * 60)
    log.info("LIVE MATCH MONITOR STARTED")
    log.info("=" * 60)

    while True:
        try:
            live_matches = analyzer.fetcher.get_live_matches()

            if not live_matches:
                log.info("No live matches currently")
            else:
                for match_data in live_matches:
                    if not match_data.is_live:
                        continue

                    # Load Tier 1 predictions
                    tier1_home = 0.4  # Would load from predictions file
                    tier1_away = 0.35

                    analysis = analyzer.analyze_live_match(
                        match_data.home_team,
                        match_data.away_team,
                        tier1_home,
                        tier1_away
                    )

                    if analysis:
                        print(analysis.summary)

                        # Save opportunities
                        if analysis.opportunities:
                            save_live_opportunities(analysis.opportunities)

            time.sleep(UPDATE_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log.info("Monitor stopped")
            break
        except Exception as e:
            log.error(f"Monitor error: {e}")
            time.sleep(10)


def save_live_opportunities(opportunities: List[LiveBetOpportunity]):
    """Save live betting opportunities to file."""
    output_path = LIVE_DATA_DIR / "opportunities.json"

    # Load existing
    existing = []
    if output_path.exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
                existing = data.get("opportunities", [])
        except Exception:
            pass

    # Add new (avoiding duplicates)
    existing_keys = set(f"{o['match']}_{o['minute']}_{o['selection']}" for o in existing)

    for opp in opportunities:
        key = f"{opp.match}_{opp.minute}_{opp.selection}"
        if key not in existing_keys:
            existing.append(asdict(opp))

    # Save
    with open(output_path, "w") as f:
        json.dump({
            "updated_at": datetime.now().isoformat(),
            "opportunities": existing[-50:]  # Keep last 50
        }, f, indent=2)


def get_live_factors(analysis: Tier3Analysis) -> Dict[str, any]:
    """Convert live analysis to prediction factors."""
    factors = {}

    if analysis.home_change > 0.05:
        factors["live_home_advantage"] = True
        factors["live_home_boost"] = round(analysis.home_change, 2)
    elif analysis.away_change > 0.05:
        factors["live_away_advantage"] = True
        factors["live_away_boost"] = round(analysis.away_change, 2)

    if analysis.momentum.pressure_team == "home":
        factors["home_momentum"] = True
    elif analysis.momentum.pressure_team == "away":
        factors["away_momentum"] = True

    if analysis.opportunities:
        factors["live_value_available"] = True
        factors["live_opportunities"] = len(analysis.opportunities)

    return factors


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Live Betting System - Tier 3")
    parser.add_argument("--match", type=str, help="Analyze specific match")
    parser.add_argument("--monitor", action="store_true", help="Monitor all live matches")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    parser.add_argument("--minute", type=int, default=60, help="Simulated minute")

    args = parser.parse_args()

    analyzer = LiveAnalyzer()

    if args.monitor:
        monitor_live_matches()

    elif args.simulate or args.match:
        if args.match and " vs " in args.match:
            home, away = args.match.split(" vs ")
        else:
            home, away = "Lecce", "Udinese"

        analysis = analyzer.analyze_live_match(
            home.strip(), away.strip(),
            tier1_home=0.45, tier1_away=0.30, tier1_draw=0.25,
            simulated_minute=args.minute
        )

        if analysis:
            print(analysis.summary)

            # Save analysis
            output_path = LIVE_DATA_DIR / "current_analysis.json"
            with open(output_path, "w") as f:
                json.dump({
                    "match": analysis.match,
                    "minute": analysis.match_data.minute,
                    "score": f"{analysis.match_data.home_score}-{analysis.match_data.away_score}",
                    "live_probs": {
                        "home": analysis.live_home_prob,
                        "away": analysis.live_away_prob,
                        "draw": analysis.live_draw_prob
                    },
                    "momentum": asdict(analysis.momentum),
                    "opportunities": [asdict(o) for o in analysis.opportunities],
                    "generated_at": analysis.generated_at
                }, f, indent=2)
            print(f"\nSaved to {output_path}")
        else:
            print("No live data available")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
