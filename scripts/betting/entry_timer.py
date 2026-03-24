"""Bet entry timing optimizer.

Analyzes historical odds snapshots to recommend WHEN to place bets
for maximum closing line value (CLV).

Strategy:
  1. Build per-match odds timeline from snapshots
  2. Track Pinnacle (sharp) vs soft book divergence over time
  3. Identify entry windows where:
     - Sharp money has moved but soft books haven't caught up yet
     - Our model edge is still available at soft book prices
     - Odds are NOT drifting against our selection
  4. Recommend: "Place bet NOW" vs "Wait — odds likely to improve"

Usage:
    from scripts.betting.entry_timer import EntryTimingAnalyzer
    analyzer = EntryTimingAnalyzer()
    analyzer.load_snapshots()
    recommendation = analyzer.recommend_entry("Milan vs Roma", "Draw")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
SNAPSHOTS_DIR = DATA_DIR / "odds_snapshots"

# Sharp bookmakers (move first, most accurate closing lines)
SHARP_BOOKS = {"Pinnacle", "Betfair", "Matchbook"}
# Minimum snapshots to analyze a match
MIN_SNAPSHOTS = 3


class OddsTimeline:
    """Per-match odds timeline built from historical snapshots."""

    def __init__(self, match_key: str, commence_time: datetime | None = None):
        self.match_key = match_key
        self.commence_time = commence_time
        # List of (timestamp, {bookmaker: {home, draw, away}})
        self.snapshots: List[Tuple[datetime, Dict[str, Dict[str, float]]]] = []

    def add_snapshot(self, ts: datetime, bookmaker_odds: Dict[str, Dict[str, float]]):
        """Add a timestamped snapshot of all bookmaker odds for this match."""
        self.snapshots.append((ts, bookmaker_odds))

    def sort(self):
        """Sort snapshots chronologically."""
        self.snapshots.sort(key=lambda x: x[0])

    @property
    def n_snapshots(self) -> int:
        return len(self.snapshots)

    @property
    def hours_tracked(self) -> float:
        if len(self.snapshots) < 2:
            return 0
        delta = self.snapshots[-1][0] - self.snapshots[0][0]
        return delta.total_seconds() / 3600

    def get_sharp_line(self, ts_idx: int, outcome: str) -> float | None:
        """Get Pinnacle/sharp implied probability at a given snapshot index."""
        _, bm_odds = self.snapshots[ts_idx]
        for bm_name, odds in bm_odds.items():
            if bm_name in SHARP_BOOKS and outcome in odds:
                val = odds[outcome]
                if val and val > 1.0:
                    return 1.0 / val
        return None

    def get_market_avg(self, ts_idx: int, outcome: str) -> float | None:
        """Get average implied probability across all bookmakers."""
        _, bm_odds = self.snapshots[ts_idx]
        probs = []
        for bm_name, odds in bm_odds.items():
            if outcome in odds:
                val = odds[outcome]
                if val and val > 1.0:
                    probs.append(1.0 / val)
        return np.mean(probs) if probs else None

    def get_best_odds(self, ts_idx: int, outcome: str) -> Tuple[float, str]:
        """Get best available odds and bookmaker for an outcome."""
        _, bm_odds = self.snapshots[ts_idx]
        best_odds = 0
        best_bm = ""
        for bm_name, odds in bm_odds.items():
            if outcome in odds:
                val = odds[outcome]
                if val and val > best_odds:
                    best_odds = val
                    best_bm = bm_name
        return best_odds, best_bm

    def compute_velocity(self, outcome: str, window: int = 3) -> List[Dict]:
        """Compute odds velocity (rate of change) per snapshot.

        Returns list of {ts, sharp_prob, market_prob, divergence, velocity, hours_to_kick}.
        """
        results = []
        for i in range(len(self.snapshots)):
            ts, _ = self.snapshots[i]
            sharp = self.get_sharp_line(i, outcome)
            market = self.get_market_avg(i, outcome)
            best_odds, best_bm = self.get_best_odds(i, outcome)

            hours_to_kick = None
            if self.commence_time:
                hours_to_kick = (self.commence_time - ts).total_seconds() / 3600

            # Velocity: change in sharp prob per hour
            velocity = 0
            if i >= window and sharp is not None:
                prev_sharp = self.get_sharp_line(i - window, outcome)
                if prev_sharp is not None:
                    ts_prev = self.snapshots[i - window][0]
                    dt_hours = (ts - ts_prev).total_seconds() / 3600
                    if dt_hours > 0:
                        velocity = (sharp - prev_sharp) / dt_hours

            divergence = (sharp - market) if sharp and market else 0

            results.append({
                "ts": ts.isoformat(),
                "sharp_prob": round(sharp, 4) if sharp else None,
                "market_prob": round(market, 4) if market else None,
                "best_odds": round(best_odds, 3),
                "best_bookmaker": best_bm,
                "divergence": round(divergence, 4),
                "velocity": round(velocity, 6),
                "hours_to_kick": round(hours_to_kick, 2) if hours_to_kick else None,
            })
        return results


class EntryTimingAnalyzer:
    """Analyzes odds timelines to recommend optimal bet entry points."""

    def __init__(self):
        self.timelines: Dict[str, OddsTimeline] = {}
        self.commence_times: Dict[str, datetime] = {}

    def load_snapshots(self) -> int:
        """Load all historical bookmaker snapshots and build timelines."""
        bm_files = sorted(SNAPSHOTS_DIR.glob("bookmakers_*.json"))
        if not bm_files:
            log.warning("No bookmaker snapshots found in %s", SNAPSHOTS_DIR)
            return 0

        # Load commence times from odds_full.json
        odds_full_path = DATA_DIR / "upcoming" / "odds_full.json"
        if odds_full_path.exists():
            try:
                full_data = json.load(open(odds_full_path))
                matches = full_data.get("matches", full_data)
                for mk, mdata in matches.items():
                    ct = mdata.get("commence_time")
                    if ct:
                        try:
                            self.commence_times[mk] = datetime.fromisoformat(
                                ct.replace("Z", "+00:00")
                            ).replace(tzinfo=None)
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                log.debug("Could not load commence times: %s", e)

        n_loaded = 0
        for fpath in bm_files:
            try:
                data = json.load(open(fpath))
                ts_str = data.get("timestamp", "")
                if not ts_str:
                    # Extract from filename
                    name = fpath.stem  # bookmakers_20260324_104944
                    parts = name.split("_")
                    if len(parts) >= 3:
                        ts_str = f"{parts[1][:4]}-{parts[1][4:6]}-{parts[1][6:8]}T{parts[2][:2]}:{parts[2][2:4]}:{parts[2][4:6]}"

                ts = datetime.fromisoformat(ts_str.split(".")[0]) if ts_str else None
                if not ts:
                    continue

                matches = data.get("matches", {})
                for match_key, markets in matches.items():
                    if match_key not in self.timelines:
                        ct = self.commence_times.get(match_key)
                        self.timelines[match_key] = OddsTimeline(match_key, ct)

                    # Extract h2h odds per bookmaker
                    h2h = markets.get("h2h", [])
                    bm_odds = {}
                    for entry in h2h:
                        bm = entry.get("bookmaker", "")
                        if bm:
                            bm_odds[bm] = {
                                "home": entry.get("home", 0),
                                "draw": entry.get("draw", 0),
                                "away": entry.get("away", 0),
                            }

                    if bm_odds:
                        self.timelines[match_key].add_snapshot(ts, bm_odds)

                n_loaded += 1
            except Exception as e:
                log.debug("Failed to load %s: %s", fpath.name, e)

        # Sort all timelines
        for tl in self.timelines.values():
            tl.sort()

        log.info("Loaded %d snapshots → %d match timelines", n_loaded, len(self.timelines))
        return n_loaded

    def recommend_entry(
        self,
        match_key: str,
        selection: str,
        model_prob: float = 0,
    ) -> Dict:
        """Recommend whether to enter NOW or WAIT for better odds.

        Args:
            match_key: "Home vs Away" match identifier
            selection: "home", "draw", or "away"
            model_prob: Our model's probability for this outcome

        Returns dict with:
            action: "ENTER_NOW" | "WAIT" | "AVOID"
            reason: Human-readable explanation
            current_best_odds: Best available odds right now
            current_best_bookmaker: Which bookmaker
            sharp_direction: "shortening" | "drifting" | "stable"
            edge_trend: "improving" | "declining" | "stable"
            estimated_closing_odds: Where we expect odds to end up
            confidence: 0-1 how confident in the recommendation
        """
        tl = self.timelines.get(match_key)
        if not tl or tl.n_snapshots < MIN_SNAPSHOTS:
            return {
                "action": "ENTER_NOW",
                "reason": "Insufficient odds history — place bet at current best odds",
                "confidence": 0.3,
                "current_best_odds": 0,
                "current_best_bookmaker": "",
            }

        outcome = selection.lower()
        velocity_data = tl.compute_velocity(outcome)
        if not velocity_data:
            return {"action": "ENTER_NOW", "reason": "No velocity data", "confidence": 0.2}

        latest = velocity_data[-1]
        current_best = latest["best_odds"]
        current_bm = latest["best_bookmaker"]
        current_sharp = latest["sharp_prob"]
        current_market = latest["market_prob"]
        current_divergence = latest["divergence"]
        hours_to_kick = latest.get("hours_to_kick")

        # Compute recent velocity trend (last 3 snapshots)
        recent_velocities = [v["velocity"] for v in velocity_data[-5:] if v["velocity"] != 0]
        avg_velocity = np.mean(recent_velocities) if recent_velocities else 0

        # Compute divergence trend
        recent_divergences = [v["divergence"] for v in velocity_data[-5:]]
        divergence_expanding = len(recent_divergences) >= 2 and recent_divergences[-1] > recent_divergences[0]

        # Sharp direction
        if avg_velocity > 0.001:
            sharp_direction = "shortening"  # Odds getting shorter (more money on this outcome)
        elif avg_velocity < -0.001:
            sharp_direction = "drifting"    # Odds getting longer (money moving away)
        else:
            sharp_direction = "stable"

        # Edge trend (using model probability)
        edge_trend = "stable"
        if model_prob > 0 and current_sharp:
            current_edge = model_prob - current_sharp
            if len(velocity_data) >= 5:
                earlier_sharp = velocity_data[-5].get("sharp_prob")
                if earlier_sharp:
                    earlier_edge = model_prob - earlier_sharp
                    if current_edge > earlier_edge + 0.01:
                        edge_trend = "improving"
                    elif current_edge < earlier_edge - 0.01:
                        edge_trend = "declining"

        # Decision logic
        action = "ENTER_NOW"
        reason = ""
        confidence = 0.5

        # Rule 1: If sharp money is moving TOWARD our selection (shortening),
        # enter NOW before soft books catch up
        if sharp_direction == "shortening" and selection.lower() in ["home", "draw", "away"]:
            if divergence_expanding:
                action = "ENTER_NOW"
                reason = (f"Sharp money moving toward {selection} (velocity {avg_velocity:+.4f}/h). "
                         f"Soft books haven't caught up (divergence {current_divergence:+.4f}). "
                         f"Place now at {current_bm} @{current_best:.2f}")
                confidence = 0.8
            else:
                action = "ENTER_NOW"
                reason = (f"Sharp money confirming {selection}. "
                         f"Best odds @{current_best:.2f} at {current_bm}")
                confidence = 0.6

        # Rule 2: If sharp money is moving AGAINST our selection, WAIT or AVOID
        elif sharp_direction == "drifting":
            if abs(avg_velocity) > 0.003:
                action = "AVOID"
                reason = (f"Sharp money moving AGAINST {selection} (velocity {avg_velocity:+.4f}/h). "
                         f"Pinnacle odds drifting — edge may not be real.")
                confidence = 0.7
            else:
                action = "WAIT"
                reason = (f"Slight drift against {selection}. Monitor for 1-2h. "
                         f"If drift continues, skip this bet.")
                confidence = 0.5

        # Rule 3: If stable and we have edge, enter based on time to kickoff
        elif sharp_direction == "stable":
            if hours_to_kick is not None:
                if hours_to_kick < 2:
                    action = "ENTER_NOW"
                    reason = (f"Odds stable, kickoff in {hours_to_kick:.1f}h. "
                             f"No further movement expected. Enter at {current_bm} @{current_best:.2f}")
                    confidence = 0.7
                elif hours_to_kick < 6:
                    action = "ENTER_NOW"
                    reason = (f"Odds stable for {tl.hours_tracked:.0f}h. "
                             f"Sharp books not moving — current price is fair. "
                             f"Enter at {current_bm} @{current_best:.2f}")
                    confidence = 0.6
                else:
                    action = "WAIT"
                    reason = (f"Kickoff in {hours_to_kick:.0f}h. Odds stable. "
                             f"Wait until 3-6h before kickoff for tighter lines.")
                    confidence = 0.4

        # Estimate closing odds based on velocity trend
        estimated_closing = current_best
        if avg_velocity != 0 and hours_to_kick and hours_to_kick > 0:
            # Project: closing_prob ≈ current_market_prob + velocity × hours_remaining
            if current_market:
                closing_prob = current_market + avg_velocity * min(hours_to_kick, 6)
                closing_prob = max(0.05, min(0.95, closing_prob))
                estimated_closing = round(1.0 / closing_prob, 2)

        return {
            "action": action,
            "reason": reason,
            "confidence": round(confidence, 2),
            "current_best_odds": round(current_best, 3),
            "current_best_bookmaker": current_bm,
            "sharp_direction": sharp_direction,
            "edge_trend": edge_trend,
            "estimated_closing_odds": estimated_closing,
            "hours_to_kickoff": round(hours_to_kick, 1) if hours_to_kick else None,
            "snapshots_analyzed": tl.n_snapshots,
            "hours_tracked": round(tl.hours_tracked, 1),
            "avg_velocity": round(avg_velocity, 6),
            "sharp_soft_divergence": round(current_divergence, 4),
        }

    def analyze_all_matches(self, bets: List[Dict] = None) -> List[Dict]:
        """Analyze entry timing for all upcoming matches or specific bet candidates.

        Args:
            bets: Optional list of bet dicts with {match, selection, model_prob}.
                  If None, analyzes all tracked matches for all outcomes.
        """
        results = []

        if bets:
            for bet in bets:
                match = bet.get("match", "")
                selection = bet.get("selection", "").lower()
                # Map selection names to outcome keys
                outcome = selection
                if "home" in selection.lower() or "1" in selection:
                    outcome = "home"
                elif "draw" in selection.lower() or "x" in selection.lower():
                    outcome = "draw"
                elif "away" in selection.lower() or "2" in selection:
                    outcome = "away"

                rec = self.recommend_entry(
                    match, outcome,
                    model_prob=bet.get("model_prob", 0),
                )
                rec["match"] = match
                rec["selection"] = selection
                results.append(rec)
        else:
            for match_key, tl in self.timelines.items():
                if tl.n_snapshots < MIN_SNAPSHOTS:
                    continue
                for outcome in ["home", "draw", "away"]:
                    rec = self.recommend_entry(match_key, outcome)
                    rec["match"] = match_key
                    rec["selection"] = outcome
                    results.append(rec)

        return results

    def get_match_summary(self, match_key: str) -> Dict:
        """Get a complete odds movement summary for a match."""
        tl = self.timelines.get(match_key)
        if not tl or tl.n_snapshots < 2:
            return {"match": match_key, "error": "insufficient data"}

        summary = {
            "match": match_key,
            "snapshots": tl.n_snapshots,
            "hours_tracked": round(tl.hours_tracked, 1),
            "commence_time": tl.commence_time.isoformat() if tl.commence_time else None,
            "outcomes": {},
        }

        for outcome in ["home", "draw", "away"]:
            first_sharp = tl.get_sharp_line(0, outcome)
            last_sharp = tl.get_sharp_line(-1, outcome)
            first_best, _ = tl.get_best_odds(0, outcome)
            last_best, last_bm = tl.get_best_odds(-1, outcome)

            movement = 0
            if first_best > 0 and last_best > 0:
                movement = last_best - first_best

            summary["outcomes"][outcome] = {
                "opening_best": round(first_best, 3),
                "current_best": round(last_best, 3),
                "current_bookmaker": last_bm,
                "movement": round(movement, 3),
                "direction": "shortening" if movement < -0.05 else "drifting" if movement > 0.05 else "stable",
                "sharp_opening": round(1/first_sharp, 3) if first_sharp and first_sharp > 0 else None,
                "sharp_current": round(1/last_sharp, 3) if last_sharp and last_sharp > 0 else None,
            }

        return summary


def run_timing_analysis(bets: List[Dict] = None) -> Dict:
    """Run entry timing analysis for upcoming bets.

    Args:
        bets: Optional list of bet candidates from the betting engine.

    Returns:
        Dict with per-match recommendations and summary.
    """
    analyzer = EntryTimingAnalyzer()
    n_loaded = analyzer.load_snapshots()

    if n_loaded == 0:
        return {"error": "No odds snapshots available", "recommendations": []}

    recommendations = analyzer.analyze_all_matches(bets)

    # Summary stats
    enter_now = sum(1 for r in recommendations if r["action"] == "ENTER_NOW")
    wait = sum(1 for r in recommendations if r["action"] == "WAIT")
    avoid = sum(1 for r in recommendations if r["action"] == "AVOID")

    return {
        "snapshots_loaded": n_loaded,
        "matches_analyzed": len(analyzer.timelines),
        "recommendations": recommendations,
        "summary": {
            "enter_now": enter_now,
            "wait": wait,
            "avoid": avoid,
            "total": len(recommendations),
        },
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_timing_analysis()
    print(f"\nAnalyzed {result['matches_analyzed']} matches from {result['snapshots_loaded']} snapshots")
    print(f"Summary: {result['summary']['enter_now']} ENTER NOW, "
          f"{result['summary']['wait']} WAIT, {result['summary']['avoid']} AVOID")

    print("\n" + "=" * 70)
    for rec in result["recommendations"]:
        if rec.get("action") in ("AVOID", "WAIT"):
            emoji = "\u26a0\ufe0f" if rec["action"] == "WAIT" else "\u274c"
        else:
            emoji = "\u2705"
        print(f"{emoji} {rec['match']} — {rec['selection'].upper()}: "
              f"{rec['action']} ({rec.get('confidence', 0):.0%})")
        print(f"   {rec.get('reason', '')}")
        if rec.get("current_best_odds"):
            print(f"   Best: @{rec['current_best_odds']:.2f} ({rec.get('current_best_bookmaker', '')})")
        print()
