#!/usr/bin/env python3
"""ADVANCED MULTI-MARKET BETTING ENGINE - Phase 7 Implementation

Integrates all prediction models and generates unified betting recommendations.

Features:
- Cross-market analysis and correlation handling
- Portfolio optimization across all markets
- Risk-adjusted return calculations
- Automatic stake allocation
- Real-time value opportunity identification
- Comprehensive reporting

Markets Covered:
- 1X2 (Match Winner)
- Over/Under Goals
- Point Spreads/Handicaps
- Cards/Bookings
- BTTS (Both Teams To Score)
- Corners
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default bankroll
DEFAULT_BANKROLL = 100.0

# Market priorities (higher = better liquidity/reliability)
MARKET_PRIORITY = {
    "h2h": 5,       # Best liquidity
    "totals": 4,    # Good liquidity
    "spreads": 3,   # Moderate
    "btts": 2,      # Variable
    "corners": 1,   # Low liquidity
    "cards": 1,     # Low liquidity
}

# Value thresholds by market
VALUE_THRESHOLDS = {
    "h2h": {"strong": 0.15, "moderate": 0.08, "minimum": 0.03},
    "totals": {"strong": 0.12, "moderate": 0.06, "minimum": 0.03},
    "spreads": {"strong": 0.10, "moderate": 0.05, "minimum": 0.03},
    "btts": {"strong": 0.12, "moderate": 0.06, "minimum": 0.03},
    "corners": {"strong": 0.12, "moderate": 0.06, "minimum": 0.03},
    "cards": {"strong": 0.10, "moderate": 0.05, "minimum": 0.03},
}

# Stake allocation by value tier
STAKE_ALLOCATION = {
    "strong": 0.03,    # 3% of bankroll
    "moderate": 0.02,  # 2%
    "minimum": 0.01,   # 1%
}

# Maximum exposure limits
MAX_DAILY_EXPOSURE = 0.15  # 15% of bankroll
MAX_MATCH_EXPOSURE = 0.07  # 7% per match

# Odds sanity ranges — odds outside these are almost certainly data errors
ODDS_SANITY = {
    "h2h":     {"min": 1.05, "max": 25.0},
    "totals":  {"min": 1.05, "max": 6.0},   # Over 1.5 at 12.0 = broken data
    "spreads": {"min": 1.05, "max": 6.0},   # HOME +0 at 8.6 = broken data
    "btts":    {"min": 1.10, "max": 8.0},
    "corners": {"min": 1.10, "max": 10.0},
    "cards":   {"min": 1.10, "max": 10.0},
}

# Maximum credible value edge — above this, the odds are likely wrong, not mispriced
MAX_CREDIBLE_VALUE_PCT = 50.0  # No real edge exceeds 50% (data error, not mispricing)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UnifiedBet:
    """Unified bet representation across all markets."""
    id: str
    match: str
    date: str
    market: str
    selection: str
    odds: float
    our_probability: float
    implied_probability: float
    value: float
    value_pct: float
    expected_value: float
    stake: float
    potential_profit: float
    confidence: str
    risk_level: str
    factors: List[str]
    source: str  # Which model generated this


@dataclass
class MatchAnalysis:
    """Complete analysis for a single match."""
    match: str
    date: str
    venue: str

    # Predictions by market
    h2h_prediction: Optional[Dict] = None
    totals_prediction: Optional[Dict] = None
    spreads_prediction: Optional[Dict] = None
    btts_prediction: Optional[Dict] = None
    corners_prediction: Optional[Dict] = None
    cards_prediction: Optional[Dict] = None

    # Recommended bets
    recommended_bets: List[UnifiedBet] = field(default_factory=list)

    # Totals
    total_stake: float = 0.0
    total_potential_profit: float = 0.0
    total_value: float = 0.0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_h2h_bets() -> List[Dict]:
    """Load 1X2 betting recommendations."""
    path = DATA_DIR / "betting" / "betting_slip.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("recommended_singles", []) + data.get("consider_singles", [])


def load_totals_bets() -> List[Dict]:
    """Load over/under betting recommendations."""
    path = DATA_DIR / "upcoming" / "over_under_bets.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("recommended", []) + data.get("consider", [])


def load_handicap_bets() -> List[Dict]:
    """Load handicap betting recommendations."""
    path = DATA_DIR / "upcoming" / "handicap_bets.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("recommended", []) + data.get("consider", [])


def load_cards_bets() -> List[Dict]:
    """Load cards betting recommendations."""
    path = DATA_DIR / "upcoming" / "cards_bets.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("recommended", []) + data.get("consider", [])


def load_btts_corners_bets() -> List[Dict]:
    """Load BTTS and corners betting recommendations."""
    path = DATA_DIR / "upcoming" / "btts_corners_bets.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data.get("recommended", []) + data.get("consider", [])


def load_all_predictions() -> Dict[str, List[Dict]]:
    """Load all predictions from all models."""
    predictions = {}

    # Goal predictions
    goal_path = DATA_DIR / "upcoming" / "goal_predictions.json"
    if goal_path.exists():
        with open(goal_path) as f:
            predictions["goals"] = json.load(f).get("predictions", [])

    # Margin predictions
    margin_path = DATA_DIR / "upcoming" / "margin_predictions.json"
    if margin_path.exists():
        with open(margin_path) as f:
            predictions["margins"] = json.load(f).get("predictions", [])

    # Cards predictions
    cards_path = DATA_DIR / "upcoming" / "cards_predictions.json"
    if cards_path.exists():
        with open(cards_path) as f:
            predictions["cards"] = json.load(f).get("predictions", [])

    # BTTS predictions
    btts_path = DATA_DIR / "upcoming" / "btts_predictions.json"
    if btts_path.exists():
        with open(btts_path) as f:
            predictions["btts"] = json.load(f).get("predictions", [])

    # Corners predictions
    corners_path = DATA_DIR / "upcoming" / "corners_predictions.json"
    if corners_path.exists():
        with open(corners_path) as f:
            predictions["corners"] = json.load(f).get("predictions", [])

    # Main predictions
    main_path = DATA_DIR / "upcoming" / "predictions.json"
    if main_path.exists():
        with open(main_path) as f:
            predictions["main"] = json.load(f).get("predictions", [])

    return predictions


# =============================================================================
# ODDS SANITY CHECKS
# =============================================================================

def check_odds_sanity(odds: float, market: str, selection: str = "",
                      our_prob: float = 0.0) -> tuple:
    """Validate odds are within reasonable ranges for the market.

    Returns (is_sane, reason) — False means the bet should be rejected.
    """
    limits = ODDS_SANITY.get(market, {"min": 1.05, "max": 25.0})

    if odds < limits["min"]:
        return False, f"Odds {odds} below minimum {limits['min']} for {market}"

    if odds > limits["max"]:
        return False, f"Odds {odds} above maximum {limits['max']} for {market} — likely data error"

    # Check if implied value is absurdly high (indicates broken odds, not a real edge)
    if our_prob > 0 and odds > 0:
        implied_prob = 1.0 / odds
        value_pct = (our_prob - implied_prob) / implied_prob * 100 if implied_prob > 0 else 0
        if value_pct > MAX_CREDIBLE_VALUE_PCT:
            return False, (f"Value {value_pct:.0f}% exceeds {MAX_CREDIBLE_VALUE_PCT}% cap "
                           f"(prob={our_prob:.1%}, odds={odds}) — likely bad odds data")

    return True, ""


# =============================================================================
# BET NORMALIZATION
# =============================================================================

def normalize_h2h_bet(bet: Dict, bankroll: float) -> UnifiedBet:
    """Normalize 1X2 bet to unified format."""
    value = bet.get("value", 0)
    value_pct = bet.get("value_pct", value * 100)
    odds = bet.get("odds", 1.5)

    # Determine value tier
    if value >= VALUE_THRESHOLDS["h2h"]["strong"]:
        tier = "strong"
        risk = "medium"
    elif value >= VALUE_THRESHOLDS["h2h"]["moderate"]:
        tier = "moderate"
        risk = "low"
    else:
        tier = "minimum"
        risk = "low"

    stake = bankroll * STAKE_ALLOCATION[tier]

    return UnifiedBet(
        id=f"h2h_{bet['match'].replace(' ', '_')}",
        match=bet["match"],
        date=bet.get("date", ""),
        market="h2h",
        selection=bet.get("prediction", ""),
        odds=odds,
        our_probability=bet.get("our_probability", 0.5),
        implied_probability=bet.get("implied_probability", 1/odds),
        value=value,
        value_pct=value_pct,
        expected_value=bet.get("our_probability", 0.5) * odds - 1,
        stake=round(stake, 2),
        potential_profit=round(stake * (odds - 1), 2),
        confidence=bet.get("confidence", "MEDIUM"),
        risk_level=risk,
        factors=bet.get("factors", []),
        source="realtime_prediction_engine"
    )


def normalize_totals_bet(bet: Dict, bankroll: float) -> UnifiedBet:
    """Normalize over/under bet."""
    value_pct = bet.get("value_pct", 0)
    value = value_pct / 100
    odds = bet.get("odds", 1.8)

    if abs(value) >= VALUE_THRESHOLDS["totals"]["strong"]:
        tier = "strong"
        risk = "medium"
    elif abs(value) >= VALUE_THRESHOLDS["totals"]["moderate"]:
        tier = "moderate"
        risk = "low"
    else:
        tier = "minimum"
        risk = "low"

    stake = bankroll * STAKE_ALLOCATION[tier]

    return UnifiedBet(
        id=f"totals_{bet['match'].replace(' ', '_')}_{bet.get('bet', '')}",
        match=bet["match"],
        date=bet.get("date", ""),
        market="totals",
        selection=bet.get("bet", ""),
        odds=odds,
        our_probability=bet.get("our_probability", 0.5),
        implied_probability=1/odds if odds > 0 else 0,
        value=value,
        value_pct=value_pct,
        expected_value=bet.get("our_probability", 0.5) * odds - 1,
        stake=round(stake, 2),
        potential_profit=round(stake * (odds - 1), 2),
        confidence="HIGH" if abs(value) > 0.10 else "MEDIUM",
        risk_level=risk,
        factors=bet.get("factors", []),
        source="over_under_model"
    )


def normalize_handicap_bet(bet: Dict, bankroll: float) -> UnifiedBet:
    """Normalize handicap bet."""
    value_pct = bet.get("value_pct", 0)
    value = value_pct / 100
    odds = bet.get("odds", 1.9)

    if abs(value) >= VALUE_THRESHOLDS["spreads"]["strong"]:
        tier = "strong"
        risk = "medium"
    elif abs(value) >= VALUE_THRESHOLDS["spreads"]["moderate"]:
        tier = "moderate"
        risk = "low"
    else:
        tier = "minimum"
        risk = "low"

    stake = bankroll * STAKE_ALLOCATION[tier]

    return UnifiedBet(
        id=f"spreads_{bet['match'].replace(' ', '_')}_{bet.get('bet', '')}",
        match=bet["match"],
        date=bet.get("date", ""),
        market="spreads",
        selection=bet.get("bet", ""),
        odds=odds,
        our_probability=bet.get("our_probability", 0.5),
        implied_probability=1/odds if odds > 0 else 0,
        value=value,
        value_pct=value_pct,
        expected_value=bet.get("our_probability", 0.5) * odds - 1,
        stake=round(stake, 2),
        potential_profit=round(stake * (odds - 1), 2),
        confidence="HIGH" if abs(value) > 0.15 else "MEDIUM",
        risk_level=risk,
        factors=bet.get("factors", []),
        source="handicap_model"
    )


def normalize_btts_corners_bet(bet: Dict, bankroll: float) -> UnifiedBet:
    """Normalize BTTS or corners bet."""
    market = bet.get("market", "btts")
    fair_odds = bet.get("fair_odds", 2.0)
    our_prob = bet.get("our_probability", 0.5)

    # Estimate value based on typical bookmaker margins (~5%)
    implied_prob = 1 / fair_odds
    bookmaker_odds = fair_odds * 0.95  # Assume 5% margin
    value = our_prob - implied_prob

    if abs(value) >= VALUE_THRESHOLDS.get(market, VALUE_THRESHOLDS["btts"])["strong"]:
        tier = "strong"
        risk = "high"  # Lower liquidity markets
    elif abs(value) >= VALUE_THRESHOLDS.get(market, VALUE_THRESHOLDS["btts"])["moderate"]:
        tier = "moderate"
        risk = "medium"
    else:
        tier = "minimum"
        risk = "medium"

    stake = bankroll * STAKE_ALLOCATION[tier] * 0.7  # Reduce stake for lower liquidity

    bet_str = bet.get("bet", "")

    return UnifiedBet(
        id=f"{market}_{bet['match'].replace(' ', '_')}_{bet_str}",
        match=bet["match"],
        date=bet.get("date", ""),
        market=market,
        selection=bet_str,
        odds=fair_odds,
        our_probability=our_prob,
        implied_probability=implied_prob,
        value=value,
        value_pct=round(value * 100, 1),
        expected_value=our_prob * fair_odds - 1,
        stake=round(stake, 2),
        potential_profit=round(stake * (fair_odds - 1), 2),
        confidence="MEDIUM",  # External odds - less certain
        risk_level=risk,
        factors=bet.get("factors", []),
        source="btts_corners_model"
    )


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

def calculate_correlation(bet1: UnifiedBet, bet2: UnifiedBet) -> float:
    """Calculate correlation between two bets."""
    # Same match = high correlation
    if bet1.match == bet2.match:
        # Same market = very high
        if bet1.market == bet2.market:
            return 0.95

        # Related markets
        related_pairs = [
            ("h2h", "spreads"),  # Both depend on winner
            ("totals", "btts"),   # Both depend on scoring
        ]
        if (bet1.market, bet2.market) in related_pairs or \
           (bet2.market, bet1.market) in related_pairs:
            return 0.50

        # Less related
        return 0.20

    # Different matches = low correlation
    return 0.05


def _load_market_adjustments() -> dict:
    """Load per-market stake multipliers from ROI feedback loop."""
    adj_path = DATA_DIR / "feedback" / "market_adjustments.json"
    if not adj_path.exists():
        return {}
    try:
        with open(adj_path) as f:
            data = json.load(f)
        return data.get("market_multipliers", {})
    except Exception:
        return {}


def optimize_portfolio(bets: List[UnifiedBet], bankroll: float) -> List[UnifiedBet]:
    """Optimize bet portfolio considering correlations, limits, and ROI feedback."""

    if not bets:
        return []

    # Load ROI-based market adjustments (from learning loop)
    market_multipliers = _load_market_adjustments()

    # Sort by expected value
    sorted_bets = sorted(bets, key=lambda x: x.value, reverse=True)

    optimized = []
    total_exposure = 0
    match_exposures = {}
    max_daily = bankroll * MAX_DAILY_EXPOSURE

    for bet in sorted_bets:
        # Check daily limit
        if total_exposure >= max_daily:
            continue

        # Check match limit
        match_exp = match_exposures.get(bet.match, 0)
        max_match = bankroll * MAX_MATCH_EXPOSURE

        if match_exp >= max_match:
            continue

        # Check for conflicting bets
        conflict = False
        for existing in optimized:
            if existing.match == bet.match and existing.market == bet.market:
                # Same market, same match - skip
                conflict = True
                break

        if conflict:
            continue

        # Adjust stake for remaining capacity
        remaining_daily = max_daily - total_exposure
        remaining_match = max_match - match_exp

        adjusted_stake = min(bet.stake, remaining_daily, remaining_match)

        # Minimum bet = 0.5% of bankroll (e.g., $0.50 for $100 bankroll)
        min_bet = max(0.50, bankroll * 0.005)
        if adjusted_stake < min_bet:
            continue

        # Apply ROI-based market multiplier (from learning loop feedback)
        mkt_mult = market_multipliers.get(bet.market, 1.0)
        if mkt_mult != 1.0:
            adjusted_stake *= mkt_mult

        # Apply correlation penalty
        correlation_penalty = 0
        for existing in optimized:
            corr = calculate_correlation(bet, existing)
            correlation_penalty += corr * 0.1

        adjusted_stake *= max(0.5, 1 - correlation_penalty)
        adjusted_stake = round(adjusted_stake, 2)

        # Update bet
        bet.stake = adjusted_stake
        bet.potential_profit = round(adjusted_stake * (bet.odds - 1), 2)

        optimized.append(bet)
        total_exposure += adjusted_stake
        match_exposures[bet.match] = match_exposures.get(bet.match, 0) + adjusted_stake

    return optimized


# =============================================================================
# ANALYSIS & REPORTING
# =============================================================================

def build_match_analysis(
    match: str,
    bets: List[UnifiedBet],
    predictions: Dict
) -> MatchAnalysis:
    """Build comprehensive analysis for a single match."""

    match_bets = [b for b in bets if b.match == match]

    # Find predictions for this match
    goal_pred = None
    for p in predictions.get("goals", []):
        if p.get("match") == match:
            goal_pred = p
            break

    margin_pred = None
    for p in predictions.get("margins", []):
        if p.get("match") == match:
            margin_pred = p
            break

    cards_pred = None
    for p in predictions.get("cards", []):
        if p.get("match") == match:
            cards_pred = p
            break

    btts_pred = None
    for p in predictions.get("btts", []):
        if p.get("match") == match:
            btts_pred = p
            break

    corners_pred = None
    for p in predictions.get("corners", []):
        if p.get("match") == match:
            corners_pred = p
            break

    main_pred = None
    for p in predictions.get("main", []):
        if p.get("match") == match:
            main_pred = p
            break

    date = match_bets[0].date if match_bets else ""
    venue = main_pred.get("venue", "") if main_pred else ""

    return MatchAnalysis(
        match=match,
        date=date,
        venue=venue,
        h2h_prediction=main_pred,
        totals_prediction=goal_pred,
        spreads_prediction=margin_pred,
        btts_prediction=btts_pred,
        corners_prediction=corners_pred,
        cards_prediction=cards_pred,
        recommended_bets=match_bets,
        total_stake=sum(b.stake for b in match_bets),
        total_potential_profit=sum(b.potential_profit for b in match_bets),
        total_value=sum(b.value for b in match_bets) / len(match_bets) if match_bets else 0
    )


def generate_unified_report(bankroll: float = None) -> Dict:
    """Generate comprehensive unified betting report."""

    if bankroll is None:
        bankroll = DEFAULT_BANKROLL

    log.info("=" * 60)
    log.info("ADVANCED MULTI-MARKET BETTING ENGINE")
    log.info("=" * 60)

    # Load all bets
    all_raw_bets = []

    # H2H bets
    h2h_bets = load_h2h_bets()
    for bet in h2h_bets:
        all_raw_bets.append(("h2h", bet))
    log.info(f"Loaded {len(h2h_bets)} H2H bets")

    # Totals bets
    totals_bets = load_totals_bets()
    for bet in totals_bets:
        all_raw_bets.append(("totals", bet))
    log.info(f"Loaded {len(totals_bets)} totals bets")

    # Handicap bets
    handicap_bets = load_handicap_bets()
    for bet in handicap_bets:
        all_raw_bets.append(("spreads", bet))
    log.info(f"Loaded {len(handicap_bets)} handicap bets")

    # BTTS/Corners bets
    btts_corners = load_btts_corners_bets()
    for bet in btts_corners:
        all_raw_bets.append(("btts_corners", bet))
    log.info(f"Loaded {len(btts_corners)} BTTS/corners bets")

    # Load anomaly flags from predictions to skip problematic matches
    anomalous_matches = set()
    pred_path = DATA_DIR / "upcoming" / "predictions.json"
    if pred_path.exists():
        try:
            with open(pred_path) as f:
                pred_data = json.load(f)
            for p in pred_data.get("predictions", []):
                anomaly = p.get("market_anomaly", {})
                if anomaly.get("severity", 0) >= 2:
                    anomalous_matches.add(p.get("match", ""))
            if anomalous_matches:
                log.warning(f"Excluding {len(anomalous_matches)} anomalous matches: {anomalous_matches}")
        except Exception:
            pass

    # Normalize all bets with sanity checks
    unified_bets = []
    rejected_count = 0
    for bet_type, bet in all_raw_bets:
        # Skip matches flagged as anomalous
        if bet.get("match", "") in anomalous_matches:
            log.warning(f"REJECTED {bet.get('match', '?')}: market anomaly detected")
            rejected_count += 1
            continue
        try:
            # Pre-normalization odds sanity check
            odds = bet.get("odds", bet.get("fair_odds", 0))
            our_prob = bet.get("our_probability", 0)
            sane, reason = check_odds_sanity(odds, bet_type, bet.get("bet", ""), our_prob)
            if not sane:
                log.warning(f"REJECTED {bet.get('match', '?')} {bet.get('bet', '')}: {reason}")
                rejected_count += 1
                continue

            if bet_type == "h2h":
                unified = normalize_h2h_bet(bet, bankroll)
            elif bet_type == "totals":
                unified = normalize_totals_bet(bet, bankroll)
            elif bet_type == "spreads":
                unified = normalize_handicap_bet(bet, bankroll)
            else:
                unified = normalize_btts_corners_bet(bet, bankroll)

            if unified.value > 0:  # Only positive value bets
                unified_bets.append(unified)
        except Exception as e:
            log.warning(f"Error normalizing bet: {e}")

    if rejected_count:
        log.warning(f"Rejected {rejected_count} bets with suspicious odds")
    log.info(f"Normalized {len(unified_bets)} positive value bets")

    # Optimize portfolio
    optimized = optimize_portfolio(unified_bets, bankroll)
    log.info(f"Optimized portfolio to {len(optimized)} bets")

    # Load predictions for context
    predictions = load_all_predictions()

    # Build match analyses
    matches = list(set(b.match for b in optimized))
    analyses = [build_match_analysis(m, optimized, predictions) for m in matches]
    analyses.sort(key=lambda x: x.total_value, reverse=True)

    # Calculate totals
    total_stake = sum(b.stake for b in optimized)
    total_potential_profit = sum(b.potential_profit for b in optimized)
    avg_odds = sum(b.odds for b in optimized) / len(optimized) if optimized else 0
    avg_value = sum(b.value for b in optimized) / len(optimized) if optimized else 0

    # Build report
    report = {
        "generated_at": datetime.now().isoformat(),
        "bankroll": bankroll,
        "summary": {
            "total_bets": len(optimized),
            "total_stake": round(total_stake, 2),
            "stake_pct": round(total_stake / bankroll * 100, 1),
            "total_potential_profit": round(total_potential_profit, 2),
            "average_odds": round(avg_odds, 2),
            "average_value": round(avg_value * 100, 1),
            "by_market": {},
            "by_confidence": {},
        },
        "bets": [],
        "match_analyses": []
    }

    # Count by market
    for market in MARKET_PRIORITY.keys():
        market_bets = [b for b in optimized if b.market == market]
        if market_bets:
            report["summary"]["by_market"][market] = {
                "count": len(market_bets),
                "stake": round(sum(b.stake for b in market_bets), 2),
                "potential_profit": round(sum(b.potential_profit for b in market_bets), 2)
            }

    # Count by confidence
    for conf in ["HIGH", "MEDIUM-HIGH", "MEDIUM", "LOW"]:
        conf_bets = [b for b in optimized if b.confidence == conf]
        if conf_bets:
            report["summary"]["by_confidence"][conf] = len(conf_bets)

    # Add bets
    for bet in optimized:
        report["bets"].append({
            "match": bet.match,
            "date": bet.date,
            "market": bet.market,
            "selection": bet.selection,
            "odds": bet.odds,
            "stake": bet.stake,
            "potential_profit": bet.potential_profit,
            "our_probability": round(bet.our_probability, 3),
            "value_pct": bet.value_pct,
            "confidence": bet.confidence,
            "risk_level": bet.risk_level,
            "factors": bet.factors
        })

    # Add match analyses
    for analysis in analyses:
        report["match_analyses"].append({
            "match": analysis.match,
            "date": analysis.date,
            "venue": analysis.venue,
            "bets_count": len(analysis.recommended_bets),
            "total_stake": round(analysis.total_stake, 2),
            "total_potential_profit": round(analysis.total_potential_profit, 2)
        })

    return report


def save_report(report: Dict):
    """Save unified report."""
    output_path = DATA_DIR / "betting" / "unified_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"Saved unified report to {output_path}")


def print_report(report: Dict):
    """Print formatted report."""

    print("\n" + "=" * 80)
    print(" MULTI-MARKET BETTING ANALYSIS")
    print("=" * 80)

    summary = report["summary"]
    print(f"\n  Bankroll: ${report['bankroll']:.2f}")
    print(f"  Total Bets: {summary['total_bets']}")
    print(f"  Total Stake: ${summary['total_stake']:.2f} ({summary['stake_pct']:.1f}%)")
    print(f"  Potential Profit: ${summary['total_potential_profit']:.2f}")
    print(f"  Average Odds: {summary['average_odds']:.2f}")
    print(f"  Average Value: {summary['average_value']:+.1f}%")

    print(f"\n  BY MARKET:")
    for market, data in summary.get("by_market", {}).items():
        print(f"    {market.upper()}: {data['count']} bets, ${data['stake']:.2f} stake, "
              f"${data['potential_profit']:.2f} profit")

    print(f"\n  BY CONFIDENCE:")
    for conf, count in summary.get("by_confidence", {}).items():
        print(f"    {conf}: {count} bets")

    print("\n" + "=" * 80)
    print(" RECOMMENDED BETS")
    print("=" * 80)

    print(f"\n  {'Match':<25} {'Market':>8} {'Selection':>12} {'Odds':>6} "
          f"{'Stake':>8} {'Profit':>8} {'Value':>7}")
    print("  " + "-" * 80)

    for bet in report["bets"]:
        print(f"  {bet['match']:<25} {bet['market']:>8} {bet['selection']:>12} "
              f"{bet['odds']:>6.2f} ${bet['stake']:>7.2f} ${bet['potential_profit']:>7.2f} "
              f"{bet['value_pct']:>+6.1f}%")

    # Match analysis
    print("\n" + "=" * 80)
    print(" MATCH-BY-MATCH SUMMARY")
    print("=" * 80)

    for analysis in report["match_analyses"]:
        print(f"\n  {analysis['match']} ({analysis['date']})")
        print(f"    Bets: {analysis['bets_count']} | Stake: ${analysis['total_stake']:.2f} | "
              f"Potential: ${analysis['total_potential_profit']:.2f}")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Multi-Market Betting Engine")
    parser.add_argument("--bankroll", type=float, default=1000, help="Bankroll amount")
    args = parser.parse_args()

    report = generate_unified_report(args.bankroll)
    save_report(report)
    print_report(report)


if __name__ == "__main__":
    main()
