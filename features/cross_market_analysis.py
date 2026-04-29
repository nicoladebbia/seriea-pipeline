#!/usr/bin/env python3
"""CROSS-MARKET CORRELATION ANALYSIS

Detects signals by correlating h2h, totals, and spreads markets:
1. Offensive dominance: H2H shortening + Over increasing
2. Defensive signal: Both teams drifting + Under shortening
3. Handicap-1X2 consistency checks
4. BTTS vs totals agreement

Reads: data/upcoming/odds_full.json, odds_movement.json
Writes: data/upcoming/cross_market_signals.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

log = logging.getLogger(__name__)


class CrossMarketAnalyzer:
    """Analyzes correlations between different betting markets."""

    def __init__(self):
        self.odds_data = {}  # match -> full odds data
        self.movement_data = {}  # match -> movement data

    def load(self) -> bool:
        """Load odds and movement data."""
        full_path = DATA_DIR / "upcoming" / "odds_full.json"
        if full_path.exists():
            try:
                with open(full_path) as f:
                    data = json.load(f)
                self.odds_data = data.get("matches", {})
            except Exception as e:
                log.error(f"Failed to load odds_full.json: {e}")
                return False

        movement_path = DATA_DIR / "upcoming" / "odds_movement.json"
        if movement_path.exists():
            try:
                with open(movement_path) as f:
                    data = json.load(f)
                self.movement_data = data.get("matches", {})
            except Exception as e:
                log.debug(f"No movement data: {e}")

        log.info(f"Loaded cross-market data: {len(self.odds_data)} matches")
        return bool(self.odds_data)

    def _get_main_totals(self, match_data: Dict) -> Optional[Dict]:
        """Get the 2.5 line totals (primary for Serie A)."""
        for total in match_data.get("totals", []):
            if total.get("line") == 2.5:
                return total
        # Fallback to first available
        totals = match_data.get("totals", [])
        return totals[0] if totals else None

    def _get_main_spread(self, match_data: Dict) -> Optional[Dict]:
        """Get the primary spread line."""
        spreads = match_data.get("spreads", [])
        if not spreads:
            return None
        # Prefer the 0.5 or 1.0 line
        for spread in spreads:
            if spread.get("line") in (0.5, 1.0):
                return spread
        return spreads[0]

    def _implied_prob(self, odds: float) -> float:
        """Convert decimal odds to implied probability."""
        return 1.0 / odds if odds > 1.0 else 0.0

    def detect_offensive_dominance(self, match_key: str, match_data: Dict) -> Dict:
        """Detect offensive dominance: favorite shortening + over shortening.

        When a team's h2h odds shorten AND over 2.5 odds shorten,
        the market expects that team to score goals.
        """
        h2h = match_data.get("h2h", {})
        totals = self._get_main_totals(match_data)
        movement = self.movement_data.get(match_key, {})

        signal = {
            "detected": False,
            "side": "neutral",
            "strength": 0.0,
            "reasoning": "",
        }

        if not h2h or not totals:
            return signal

        home_odds = h2h.get("home", 0)
        away_odds = h2h.get("away", 0)
        over_odds = totals.get("over", 0)

        if not (home_odds > 1 and away_odds > 1 and over_odds > 1):
            return signal

        home_prob = self._implied_prob(home_odds)
        away_prob = self._implied_prob(away_odds)
        over_prob = self._implied_prob(over_odds)

        # Strong favorite + high over probability = offensive dominance
        if home_prob > 0.45 and over_prob > 0.52:
            signal["detected"] = True
            signal["side"] = "home"
            signal["strength"] = min(1.0, (home_prob - 0.40) * 5 * (over_prob - 0.48) * 10)
            signal["reasoning"] = f"Home strong ({home_prob:.0%}) + over likely ({over_prob:.0%})"
        elif away_prob > 0.45 and over_prob > 0.52:
            signal["detected"] = True
            signal["side"] = "away"
            signal["strength"] = min(1.0, (away_prob - 0.40) * 5 * (over_prob - 0.48) * 10)
            signal["reasoning"] = f"Away strong ({away_prob:.0%}) + over likely ({over_prob:.0%})"

        # Enhance with movement data
        if movement and signal["detected"]:
            h_move = movement.get("home_movement", 0)
            a_move = movement.get("away_movement", 0)
            if signal["side"] == "home" and h_move < -0.05:
                signal["strength"] = min(1.0, signal["strength"] + 0.2)
                signal["reasoning"] += " + home shortening"
            elif signal["side"] == "away" and a_move < -0.05:
                signal["strength"] = min(1.0, signal["strength"] + 0.2)
                signal["reasoning"] += " + away shortening"

        return signal

    def detect_defensive_signal(self, match_key: str, match_data: Dict) -> Dict:
        """Detect defensive match: both teams drifting + under shortening."""
        h2h = match_data.get("h2h", {})
        totals = self._get_main_totals(match_data)

        signal = {
            "detected": False,
            "strength": 0.0,
            "reasoning": "",
        }

        if not h2h or not totals:
            return signal

        home_odds = h2h.get("home", 0)
        away_odds = h2h.get("away", 0)
        draw_odds = h2h.get("draw", 0)
        under_odds = totals.get("under", 0)

        if not all(o > 1 for o in [home_odds, away_odds, draw_odds, under_odds]):
            return signal

        home_prob = self._implied_prob(home_odds)
        away_prob = self._implied_prob(away_odds)
        draw_prob = self._implied_prob(draw_odds)
        under_prob = self._implied_prob(under_odds)

        # High draw probability + under probability = defensive match
        if draw_prob > 0.28 and under_prob > 0.50:
            signal["detected"] = True
            signal["strength"] = min(1.0, (draw_prob - 0.25) * 5 * (under_prob - 0.45) * 5)
            signal["reasoning"] = f"Draw likely ({draw_prob:.0%}) + under likely ({under_prob:.0%})"

        # Close match (no clear favorite) + high under
        if abs(home_prob - away_prob) < 0.08 and under_prob > 0.48:
            if not signal["detected"]:
                signal["detected"] = True
                signal["strength"] = min(1.0, (0.10 - abs(home_prob - away_prob)) * 10 * (under_prob - 0.44) * 8)
                signal["reasoning"] = f"Close match + under lean ({under_prob:.0%})"

        return signal

    def check_handicap_consistency(self, match_data: Dict) -> Dict:
        """Check if handicap and 1X2 markets agree.

        Inconsistency: Handicap -1.5 favors home at short odds
        but 1X2 home odds are drifting.
        """
        h2h = match_data.get("h2h", {})
        spread = self._get_main_spread(match_data)

        signal = {
            "consistent": True,
            "inconsistency_type": None,
            "strength": 0.0,
            "reasoning": "",
        }

        if not h2h or not spread:
            return signal

        home_odds = h2h.get("home", 0)
        away_odds = h2h.get("away", 0)

        if not (home_odds > 1 and away_odds > 1):
            return signal

        home_prob = self._implied_prob(home_odds)
        away_prob = self._implied_prob(away_odds)

        # Check handicap direction
        home_hc = spread.get("home_handicap", 0)
        home_hc_odds = spread.get("home_odds", 0)
        away_hc_odds = spread.get("away_odds", 0)

        if not (home_hc_odds > 1 and away_hc_odds > 1):
            return signal

        home_hc_prob = self._implied_prob(home_hc_odds)

        # Inconsistency: Home is h2h underdog but handicap favorite
        if home_prob < 0.35 and home_hc < 0 and home_hc_prob > 0.50:
            signal["consistent"] = False
            signal["inconsistency_type"] = "h2h_underdog_hc_favorite"
            signal["strength"] = min(1.0, (0.40 - home_prob) * 5)
            signal["reasoning"] = f"Home h2h underdog ({home_prob:.0%}) but HC-{abs(home_hc)} favorite ({home_hc_prob:.0%})"

        # Inconsistency: Strong h2h favorite but handicap doesn't support
        if home_prob > 0.55 and home_hc >= 0:
            signal["consistent"] = False
            signal["inconsistency_type"] = "strong_favorite_no_hc"
            signal["strength"] = min(1.0, (home_prob - 0.50) * 5)
            signal["reasoning"] = f"Home strong h2h ({home_prob:.0%}) but no handicap support"

        if away_prob > 0.55 and home_hc <= 0:
            # Away is strong favorite but has no handicap disadvantage to home
            signal["consistent"] = False
            signal["inconsistency_type"] = "away_favorite_hc_mismatch"
            signal["strength"] = min(1.0, (away_prob - 0.50) * 5)
            signal["reasoning"] = f"Away strong ({away_prob:.0%}) but HC misaligned"

        return signal

    def check_btts_totals_agreement(self, match_data: Dict) -> Dict:
        """Check BTTS vs totals agreement.

        Unusual: BTTS Yes shortening but Over lengthening (possible 1-1 / 2-1 expectation).
        """
        btts = match_data.get("btts", {})
        totals = self._get_main_totals(match_data)

        signal = {
            "agreement": True,
            "anomaly_type": None,
            "strength": 0.0,
            "reasoning": "",
        }

        if not btts or not totals:
            return signal

        btts_yes_odds = btts.get("yes", 0)
        over_odds = totals.get("over", 0)
        under_odds = totals.get("under", 0)

        if not (btts_yes_odds > 1 and over_odds > 1 and under_odds > 1):
            return signal

        btts_prob = self._implied_prob(btts_yes_odds)
        over_prob = self._implied_prob(over_odds)
        under_prob = self._implied_prob(under_odds)

        # BTTS likely + Under likely = 1-1 or 2-1 territory
        if btts_prob > 0.55 and under_prob > 0.50:
            signal["agreement"] = False
            signal["anomaly_type"] = "btts_yes_but_under"
            signal["strength"] = min(1.0, (btts_prob - 0.50) * 5 * (under_prob - 0.45) * 5)
            signal["reasoning"] = f"BTTS Yes ({btts_prob:.0%}) but Under ({under_prob:.0%}) - expect low-scoring game with both teams scoring"

        # BTTS unlikely + Over likely = one-sided goal fest
        if btts_prob < 0.40 and over_prob > 0.55:
            signal["agreement"] = False
            signal["anomaly_type"] = "btts_no_but_over"
            signal["strength"] = min(1.0, (0.45 - btts_prob) * 5 * (over_prob - 0.50) * 5)
            signal["reasoning"] = f"BTTS No ({btts_prob:.0%}) but Over ({over_prob:.0%}) - expect one-sided high-scoring"

        return signal

    def analyze_match(self, match_key: str) -> Dict:
        """Run full cross-market analysis for a match."""
        match_data = self.odds_data.get(match_key, {})

        result = {
            "match": match_key,
        }

        # Offensive dominance
        offense = self.detect_offensive_dominance(match_key, match_data)
        result["offensive_dominance"] = offense

        # Defensive signal
        defense = self.detect_defensive_signal(match_key, match_data)
        result["defensive_signal"] = defense

        # Handicap consistency
        hc = self.check_handicap_consistency(match_data)
        result["handicap_consistency"] = hc

        # BTTS vs totals
        btts_totals = self.check_btts_totals_agreement(match_data)
        result["btts_totals"] = btts_totals

        # Composite score: how many cross-market signals fire?
        signals_detected = sum([
            offense["detected"],
            defense["detected"],
            not hc["consistent"],
            not btts_totals["agreement"],
        ])
        result["signals_count"] = signals_detected
        result["has_anomaly"] = signals_detected > 0

        return result

    def analyze_all(self) -> Dict:
        """Analyze all matches."""
        results = {}
        for match_key in self.odds_data:
            r = self.analyze_match(match_key)
            if isinstance(r, dict):
                r.setdefault("league", "serie_a")
            results[match_key] = r
        return results

    def save(self, results: Dict, path: Path = None) -> Path:
        """Save analysis results."""
        if path is None:
            path = DATA_DIR / "upcoming" / "cross_market_signals.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        output = {
            "analyzed_at": datetime.now().isoformat(),
            "matches": results,
            "summary": {
                "total_matches": len(results),
                "offensive_dominance": sum(
                    1 for r in results.values()
                    if r.get("offensive_dominance", {}).get("detected")
                ),
                "defensive_signals": sum(
                    1 for r in results.values()
                    if r.get("defensive_signal", {}).get("detected")
                ),
                "hc_inconsistencies": sum(
                    1 for r in results.values()
                    if not r.get("handicap_consistency", {}).get("consistent", True)
                ),
                "btts_anomalies": sum(
                    1 for r in results.values()
                    if not r.get("btts_totals", {}).get("agreement", True)
                ),
            },
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        log.info(f"Saved cross-market analysis: {output['summary']}")
        return path


def run_cross_market_analysis() -> Dict:
    """Run cross-market analysis pipeline."""
    analyzer = CrossMarketAnalyzer()
    if not analyzer.load():
        log.warning("No odds data for cross-market analysis")
        return {}

    results = analyzer.analyze_all()
    analyzer.save(results)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    results = run_cross_market_analysis()
    for match, data in results.items():
        flags = []
        if data.get("offensive_dominance", {}).get("detected"):
            flags.append(f"OFF({data['offensive_dominance']['side']})")
        if data.get("defensive_signal", {}).get("detected"):
            flags.append("DEF")
        if not data.get("handicap_consistency", {}).get("consistent", True):
            flags.append(f"HC({data['handicap_consistency']['inconsistency_type']})")
        if not data.get("btts_totals", {}).get("agreement", True):
            flags.append(f"BTTS({data['btts_totals']['anomaly_type']})")
        flag_str = " | ".join(flags) if flags else "no signals"
        print(f"  {match}: {flag_str}")
