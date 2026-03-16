#!/usr/bin/env python3
"""BOOKMAKER ANALYSIS - Sharp vs Soft Bookmaker Classification & Divergence

Analyzes per-bookmaker odds to detect:
- Sharp bookmaker consensus (Pinnacle, BetCRIS, Matchbook)
- Soft bookmaker consensus (bet365, William Hill, Unibet, etc.)
- Sharp-soft divergence (>3% = actionable signal)
- Outlier bookmakers (>5% from consensus)

Reads: data/upcoming/odds_bookmakers.json (per-bookmaker prices)
Writes: data/upcoming/bookmaker_analysis.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

log = logging.getLogger(__name__)

# Bookmaker classification registry
# Sharp books: low margin (2-3%), fast line movement, professional bettors welcome
SHARP_BOOKMAKERS = {
    "Pinnacle": {"tier": "sharp", "typical_margin": 0.02},
    "BetCRIS": {"tier": "sharp", "typical_margin": 0.025},
    "Matchbook": {"tier": "sharp", "typical_margin": 0.02},
    "Betfair Exchange": {"tier": "sharp", "typical_margin": 0.02},
    "Betfair": {"tier": "sharp", "typical_margin": 0.025},
    "Smarkets": {"tier": "sharp", "typical_margin": 0.02},
}

# Soft books: higher margin (6-11%), limit winning accounts, recreational bettors
SOFT_BOOKMAKERS = {
    "bet365": {"tier": "soft", "typical_margin": 0.07},
    "Bet365": {"tier": "soft", "typical_margin": 0.07},
    "William Hill": {"tier": "soft", "typical_margin": 0.06},
    "Unibet": {"tier": "soft", "typical_margin": 0.06},
    "Betway": {"tier": "soft", "typical_margin": 0.06},
    "1xBet": {"tier": "soft", "typical_margin": 0.08},
    "888sport": {"tier": "soft", "typical_margin": 0.07},
    "Ladbrokes": {"tier": "soft", "typical_margin": 0.06},
    "Coral": {"tier": "soft", "typical_margin": 0.06},
    "Paddy Power": {"tier": "soft", "typical_margin": 0.06},
    "BetVictor": {"tier": "soft", "typical_margin": 0.06},
    "BoyleSports": {"tier": "soft", "typical_margin": 0.07},
    "Betfred": {"tier": "soft", "typical_margin": 0.07},
    "SNAI": {"tier": "soft", "typical_margin": 0.08},
    "Lottomatica": {"tier": "soft", "typical_margin": 0.08},
    "Eurobet": {"tier": "soft", "typical_margin": 0.07},
    "Sisal": {"tier": "soft", "typical_margin": 0.08},
    "Betsson": {"tier": "soft", "typical_margin": 0.06},
    "LeoVegas": {"tier": "soft", "typical_margin": 0.07},
}

# Mid-tier books: moderate margin, somewhat sharp-tolerant
MID_BOOKMAKERS = {
    "DraftKings": {"tier": "mid", "typical_margin": 0.05},
    "FanDuel": {"tier": "mid", "typical_margin": 0.05},
    "BetMGM": {"tier": "mid", "typical_margin": 0.05},
    "Caesars": {"tier": "mid", "typical_margin": 0.05},
    "PointsBet": {"tier": "mid", "typical_margin": 0.05},
    "BetRivers": {"tier": "mid", "typical_margin": 0.05},
    "Marathon Bet": {"tier": "mid", "typical_margin": 0.04},
    "Marathonbet": {"tier": "mid", "typical_margin": 0.04},
    "Coolbet": {"tier": "mid", "typical_margin": 0.04},
    "Nordicbet": {"tier": "mid", "typical_margin": 0.05},
}


def _to_vig_free(home: float, draw: float, away: float) -> Dict[str, float]:
    """Convert decimal odds to vig-free implied probabilities."""
    if home <= 1.0 or draw <= 1.0 or away <= 1.0:
        return {}
    raw_h = 1.0 / home
    raw_d = 1.0 / draw
    raw_a = 1.0 / away
    total = raw_h + raw_d + raw_a
    if total <= 0:
        return {}
    return {
        "prob_H": raw_h / total,
        "prob_D": raw_d / total,
        "prob_A": raw_a / total,
        "margin": round((total - 1.0) * 100, 2),
    }


class BookmakerAnalyzer:
    """Analyzes per-bookmaker odds for sharp/soft divergence."""

    def __init__(self):
        self.bookmaker_data = {}  # match -> {h2h: [{bookmaker, home, draw, away}]}
        self.analysis = {}  # match -> analysis result

    def load(self, path: Path = None) -> bool:
        """Load per-bookmaker odds data."""
        if path is None:
            path = DATA_DIR / "upcoming" / "odds_bookmakers.json"

        if not path.exists():
            log.warning(f"No bookmaker data at {path}")
            return False

        try:
            with open(path) as f:
                data = json.load(f)
            self.bookmaker_data = data.get("matches", {})
            log.info(f"Loaded bookmaker data for {len(self.bookmaker_data)} matches")
            return bool(self.bookmaker_data)
        except Exception as e:
            log.error(f"Failed to load bookmaker data: {e}")
            return False

    def classify_bookmakers(self, bookmakers: List[Dict]) -> Dict[str, List[Dict]]:
        """Split bookmaker entries into sharp/mid/soft/unknown tiers."""
        classified = {"sharp": [], "mid": [], "soft": [], "unknown": []}
        for entry in bookmakers:
            name = entry.get("bookmaker", "")
            if name in SHARP_BOOKMAKERS:
                classified["sharp"].append(entry)
            elif name in MID_BOOKMAKERS:
                classified["mid"].append(entry)
            elif name in SOFT_BOOKMAKERS:
                classified["soft"].append(entry)
            else:
                classified["unknown"].append(entry)
        return classified

    def compute_sharp_consensus(self, bookmakers: List[Dict]) -> Optional[Dict[str, float]]:
        """Compute weighted vig-free average from sharp bookmakers."""
        classified = self.classify_bookmakers(bookmakers)
        sharp = classified["sharp"]
        if not sharp:
            return None

        total_weight = 0.0
        prob_H, prob_D, prob_A = 0.0, 0.0, 0.0

        for entry in sharp:
            probs = _to_vig_free(entry.get("home", 0), entry.get("draw", 0), entry.get("away", 0))
            if not probs:
                continue
            # Weight by inverse margin (lower margin = more weight)
            margin = max(probs["margin"], 0.5) / 100.0
            w = 1.0 / margin
            prob_H += w * probs["prob_H"]
            prob_D += w * probs["prob_D"]
            prob_A += w * probs["prob_A"]
            total_weight += w

        if total_weight == 0:
            return None

        return {
            "prob_H": prob_H / total_weight,
            "prob_D": prob_D / total_weight,
            "prob_A": prob_A / total_weight,
            "source_count": len(sharp),
        }

    def compute_soft_consensus(self, bookmakers: List[Dict]) -> Optional[Dict[str, float]]:
        """Compute weighted vig-free average from soft bookmakers."""
        classified = self.classify_bookmakers(bookmakers)
        soft = classified["soft"]
        if not soft:
            return None

        total_weight = 0.0
        prob_H, prob_D, prob_A = 0.0, 0.0, 0.0

        for entry in soft:
            probs = _to_vig_free(entry.get("home", 0), entry.get("draw", 0), entry.get("away", 0))
            if not probs:
                continue
            w = 1.0  # Equal weight for soft books
            prob_H += w * probs["prob_H"]
            prob_D += w * probs["prob_D"]
            prob_A += w * probs["prob_A"]
            total_weight += w

        if total_weight == 0:
            return None

        return {
            "prob_H": prob_H / total_weight,
            "prob_D": prob_D / total_weight,
            "prob_A": prob_A / total_weight,
            "source_count": len(soft),
        }

    def compute_market_consensus(self, bookmakers: List[Dict]) -> Optional[Dict[str, float]]:
        """Compute overall market consensus from all bookmakers."""
        if not bookmakers:
            return None

        total = 0.0
        prob_H, prob_D, prob_A = 0.0, 0.0, 0.0

        for entry in bookmakers:
            probs = _to_vig_free(entry.get("home", 0), entry.get("draw", 0), entry.get("away", 0))
            if not probs:
                continue
            prob_H += probs["prob_H"]
            prob_D += probs["prob_D"]
            prob_A += probs["prob_A"]
            total += 1.0

        if total == 0:
            return None

        return {
            "prob_H": prob_H / total,
            "prob_D": prob_D / total,
            "prob_A": prob_A / total,
            "source_count": int(total),
        }

    def compute_sharp_soft_divergence(
        self, sharp: Dict[str, float], soft: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute implied probability difference between sharp and soft consensus.

        Positive divergence on a side = sharps think that side is more likely.
        >3% divergence is actionable.
        """
        div_H = sharp["prob_H"] - soft["prob_H"]
        div_D = sharp["prob_D"] - soft["prob_D"]
        div_A = sharp["prob_A"] - soft["prob_A"]

        max_div = max(abs(div_H), abs(div_D), abs(div_A))

        # Determine sharp direction
        if div_H > 0.02:
            direction = "home"
        elif div_A > 0.02:
            direction = "away"
        elif div_D > 0.02:
            direction = "draw"
        else:
            direction = "neutral"

        return {
            "home_divergence": round(div_H, 4),
            "draw_divergence": round(div_D, 4),
            "away_divergence": round(div_A, 4),
            "max_divergence": round(max_div, 4),
            "is_actionable": max_div > 0.03,
            "sharp_direction": direction,
        }

    def detect_outlier_bookmakers(
        self, bookmakers: List[Dict], consensus: Dict[str, float]
    ) -> List[Dict]:
        """Find bookmakers >5% from consensus (>10% = extreme)."""
        outliers = []
        for entry in bookmakers:
            probs = _to_vig_free(entry.get("home", 0), entry.get("draw", 0), entry.get("away", 0))
            if not probs:
                continue

            dev_H = abs(probs["prob_H"] - consensus["prob_H"])
            dev_D = abs(probs["prob_D"] - consensus["prob_D"])
            dev_A = abs(probs["prob_A"] - consensus["prob_A"])
            max_dev = max(dev_H, dev_D, dev_A)

            if max_dev > 0.05:
                outliers.append({
                    "bookmaker": entry["bookmaker"],
                    "max_deviation": round(max_dev, 4),
                    "is_extreme": max_dev > 0.10,
                    "home_dev": round(dev_H, 4),
                    "draw_dev": round(dev_D, 4),
                    "away_dev": round(dev_A, 4),
                })

        return sorted(outliers, key=lambda x: x["max_deviation"], reverse=True)

    def analyze_match(self, match_key: str) -> Dict:
        """Run full bookmaker analysis for a match."""
        match_data = self.bookmaker_data.get(match_key, {})
        h2h_bookmakers = match_data.get("h2h", [])

        result = {
            "match": match_key,
            "bookmakers_total": len(h2h_bookmakers),
            "sharp_consensus": None,
            "soft_consensus": None,
            "market_consensus": None,
            "divergence": None,
            "sharp_direction": "neutral",
            "outliers": [],
            "has_sharp_data": False,
        }

        if not h2h_bookmakers:
            return result

        classified = self.classify_bookmakers(h2h_bookmakers)
        result["sharp_count"] = len(classified["sharp"])
        result["soft_count"] = len(classified["soft"])
        result["mid_count"] = len(classified["mid"])
        result["unknown_count"] = len(classified["unknown"])

        # Market consensus (all bookmakers)
        market = self.compute_market_consensus(h2h_bookmakers)
        if market:
            result["market_consensus"] = {
                "prob_H": round(market["prob_H"], 4),
                "prob_D": round(market["prob_D"], 4),
                "prob_A": round(market["prob_A"], 4),
            }

        # Sharp consensus
        sharp = self.compute_sharp_consensus(h2h_bookmakers)
        if sharp:
            result["has_sharp_data"] = True
            result["sharp_consensus"] = {
                "prob_H": round(sharp["prob_H"], 4),
                "prob_D": round(sharp["prob_D"], 4),
                "prob_A": round(sharp["prob_A"], 4),
                "source_count": sharp["source_count"],
            }

        # Soft consensus
        soft = self.compute_soft_consensus(h2h_bookmakers)
        if soft:
            result["soft_consensus"] = {
                "prob_H": round(soft["prob_H"], 4),
                "prob_D": round(soft["prob_D"], 4),
                "prob_A": round(soft["prob_A"], 4),
                "source_count": soft["source_count"],
            }

        # Divergence
        if sharp and soft:
            div = self.compute_sharp_soft_divergence(sharp, soft)
            result["divergence"] = round(div["max_divergence"], 4)
            result["divergence_detail"] = {
                "home": round(div["home_divergence"], 4),
                "draw": round(div["draw_divergence"], 4),
                "away": round(div["away_divergence"], 4),
            }
            result["sharp_direction"] = div["sharp_direction"]
            result["divergence_actionable"] = div["is_actionable"]

        # Outliers
        consensus = market or sharp or soft
        if consensus:
            result["outliers"] = self.detect_outlier_bookmakers(h2h_bookmakers, consensus)

        return result

    def analyze_all(self) -> Dict:
        """Analyze all matches and return results."""
        results = {}
        for match_key in self.bookmaker_data:
            results[match_key] = self.analyze_match(match_key)

        self.analysis = results
        return results

    def save(self, path: Path = None) -> Path:
        """Save analysis to JSON."""
        if path is None:
            path = DATA_DIR / "upcoming" / "bookmaker_analysis.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        output = {
            "analyzed_at": datetime.now().isoformat(),
            "matches": self.analysis,
            "summary": {
                "total_matches": len(self.analysis),
                "with_sharp_data": sum(1 for m in self.analysis.values() if m.get("has_sharp_data")),
                "actionable_divergence": sum(
                    1 for m in self.analysis.values()
                    if m.get("divergence_actionable")
                ),
            },
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        log.info(f"Saved bookmaker analysis: {output['summary']}")
        return path


def run_bookmaker_analysis() -> Dict:
    """Run bookmaker analysis pipeline. Returns analysis dict."""
    analyzer = BookmakerAnalyzer()
    if not analyzer.load():
        log.warning("No bookmaker data available for analysis")
        return {}

    results = analyzer.analyze_all()
    analyzer.save()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    results = run_bookmaker_analysis()
    if results:
        for match, data in results.items():
            div = data.get("divergence", 0) or 0
            direction = data.get("sharp_direction", "neutral")
            flag = " ***" if data.get("divergence_actionable") else ""
            print(f"  {match}: sharp={direction} div={div:.3f}{flag}")
