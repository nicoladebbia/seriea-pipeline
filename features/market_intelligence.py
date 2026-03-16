#!/usr/bin/env python3
"""MARKET INTELLIGENCE AGGREGATOR

Aggregates signals from:
- Bookmaker analysis (sharp/soft divergence) — Phase 2
- Odds movement tracking (temporal patterns) — Phase 3
- Cross-market correlations (h2h/totals/spreads) — Phase 4

Produces a composite market_intelligence_score per match.

Reads:
  data/upcoming/bookmaker_analysis.json
  data/upcoming/odds_movement.json
  data/upcoming/cross_market_signals.json
Writes:
  data/upcoming/market_intelligence.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

log = logging.getLogger(__name__)


class MarketIntelligence:
    """Aggregate all market intelligence signals into a unified analysis."""

    def __init__(self):
        self.bookmaker_analysis = {}   # from bookmaker_analysis.json
        self.movement_data = {}        # from odds_movement.json
        self.cross_market = {}         # from cross_market_signals.json
        self._loaded = False

    def load(self) -> bool:
        """Load all upstream signal files."""
        self._loaded = True

        # Bookmaker analysis (sharp/soft)
        ba_path = DATA_DIR / "upcoming" / "bookmaker_analysis.json"
        if ba_path.exists():
            try:
                with open(ba_path) as f:
                    data = json.load(f)
                self.bookmaker_analysis = data.get("matches", {})
                log.info(f"Loaded bookmaker analysis for {len(self.bookmaker_analysis)} matches")
            except Exception as e:
                log.debug(f"Could not load bookmaker analysis: {e}")

        # Odds movement
        mv_path = DATA_DIR / "upcoming" / "odds_movement.json"
        if mv_path.exists():
            try:
                with open(mv_path) as f:
                    data = json.load(f)
                self.movement_data = data.get("matches", {})
                log.info(f"Loaded odds movement for {len(self.movement_data)} matches")
            except Exception as e:
                log.debug(f"Could not load odds movement: {e}")

        # Cross-market signals
        cm_path = DATA_DIR / "upcoming" / "cross_market_signals.json"
        if cm_path.exists():
            try:
                with open(cm_path) as f:
                    data = json.load(f)
                self.cross_market = data.get("matches", {})
                log.info(f"Loaded cross-market signals for {len(self.cross_market)} matches")
            except Exception as e:
                log.debug(f"Could not load cross-market signals: {e}")

        return True

    def analyze_match(self, match_key: str) -> Dict:
        """Produce unified market intelligence for a single match.

        Returns:
            Dict with sharp_direction, divergence, steam_detected,
            offensive_dominance, market_confidence, composite score.
        """
        if not self._loaded:
            self.load()

        result = {
            "match": match_key,
            "sharp_direction": "neutral",
            "divergence": 0.0,
            "steam_detected": False,
            "offensive_dominance": None,
            "defensive_signal": False,
            "hc_inconsistency": False,
            "market_confidence": 0.5,
            "composite_score": 0.0,
            "signals_available": 0,
        }

        score_components = []

        # --- Bookmaker analysis signals ---
        ba = self.bookmaker_analysis.get(match_key, {})
        if ba:
            result["signals_available"] += 1
            result["sharp_direction"] = ba.get("sharp_direction", "neutral")
            result["divergence"] = ba.get("divergence", 0) or 0
            result["has_sharp_data"] = ba.get("has_sharp_data", False)

            # Sharp consensus probabilities
            sharp_cons = ba.get("sharp_consensus")
            if sharp_cons:
                result["sharp_prob_H"] = sharp_cons.get("prob_H", 0)
                result["sharp_prob_D"] = sharp_cons.get("prob_D", 0)
                result["sharp_prob_A"] = sharp_cons.get("prob_A", 0)

            # Divergence contributes to composite score
            div = result["divergence"]
            if div > 0.03:
                score_components.append(("sharp_divergence", min(div * 10, 1.0)))
            elif div > 0.01:
                score_components.append(("sharp_divergence", div * 5))

        # --- Movement signals ---
        mv = self.movement_data.get(match_key, {})
        if mv:
            result["signals_available"] += 1
            result["steam_detected"] = mv.get("is_steam_move", False)
            result["line_move"] = mv.get("is_line_move", False)
            result["movement_direction"] = mv.get("direction", "stable")

            if result["steam_detected"]:
                score_components.append(("steam_move", 0.8))
            elif result["line_move"]:
                score_components.append(("line_move", 0.4))

        # --- Cross-market signals ---
        cm = self.cross_market.get(match_key, {})
        if cm:
            result["signals_available"] += 1

            off = cm.get("offensive_dominance", {})
            if off.get("detected"):
                result["offensive_dominance"] = off.get("side")
                score_components.append(("offensive_dominance", off.get("strength", 0.3)))

            dfs = cm.get("defensive_signal", {})
            if dfs.get("detected"):
                result["defensive_signal"] = True
                score_components.append(("defensive_signal", dfs.get("strength", 0.3)))

            hc = cm.get("handicap_consistency", {})
            if not hc.get("consistent", True):
                result["hc_inconsistency"] = True
                score_components.append(("hc_inconsistency", hc.get("strength", 0.2)))

        # --- Compute market confidence ---
        # Higher confidence when: sharp data available, low divergence, consistent signals
        confidence = 0.5
        if ba.get("has_sharp_data"):
            confidence += 0.15
        if ba.get("bookmakers_total", 0) > 15:
            confidence += 0.10
        if result["divergence"] < 0.02:
            confidence += 0.10
        if cm and cm.get("handicap_consistency", {}).get("consistent", True):
            confidence += 0.05
        result["market_confidence"] = min(0.95, confidence)

        # --- Composite score (0-1, higher = more actionable intelligence) ---
        if score_components:
            result["composite_score"] = round(
                sum(v for _, v in score_components) / len(score_components), 3
            )
            result["score_components"] = {k: round(v, 3) for k, v in score_components}
        else:
            result["composite_score"] = 0.0

        return result

    def analyze_all(self) -> Dict:
        """Analyze all matches present in any data source."""
        if not self._loaded:
            self.load()

        all_matches = set()
        all_matches.update(self.bookmaker_analysis.keys())
        all_matches.update(self.movement_data.keys())
        all_matches.update(self.cross_market.keys())

        results = {}
        for match_key in all_matches:
            results[match_key] = self.analyze_match(match_key)

        return results

    def save(self, results: Dict, path: Path = None) -> Path:
        """Save aggregated market intelligence."""
        if path is None:
            path = DATA_DIR / "upcoming" / "market_intelligence.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "analyzed_at": datetime.now().isoformat(),
            "matches": results,
            "summary": {
                "total_matches": len(results),
                "with_sharp_data": sum(1 for r in results.values() if r.get("has_sharp_data")),
                "steam_moves": sum(1 for r in results.values() if r.get("steam_detected")),
                "actionable": sum(1 for r in results.values() if r.get("composite_score", 0) > 0.3),
            },
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        log.info(f"Saved market intelligence: {output['summary']}")
        return path


def run_market_intelligence() -> Dict:
    """Run the full market intelligence aggregation pipeline."""
    mi = MarketIntelligence()
    mi.load()
    results = mi.analyze_all()
    mi.save(results)
    return results


def get_match_intelligence(match_key: str) -> Dict:
    """Quick helper: get intelligence for a single match from saved file."""
    path = DATA_DIR / "upcoming" / "market_intelligence.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("matches", {}).get(match_key, {})
    except Exception:
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    results = run_market_intelligence()
    for match, data in results.items():
        score = data.get("composite_score", 0)
        direction = data.get("sharp_direction", "neutral")
        div = data.get("divergence", 0)
        flags = []
        if data.get("steam_detected"):
            flags.append("STEAM")
        if data.get("offensive_dominance"):
            flags.append(f"OFF-{data['offensive_dominance']}")
        if data.get("defensive_signal"):
            flags.append("DEF")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"  {match}: score={score:.2f} sharp={direction} div={div:.3f}{flag_str}")
