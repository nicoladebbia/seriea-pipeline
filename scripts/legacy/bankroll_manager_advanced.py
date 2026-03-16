#!/usr/bin/env python3
"""ADVANCED MULTI-MARKET BANKROLL MANAGER - Phase 4 Implementation

Sophisticated bankroll management for multi-market betting.

Features:
- Market-specific bankroll allocation
- Kelly Criterion with correlation adjustments
- Cross-market dependency analysis
- Maximum exposure limits
- Daily/weekly loss limits
- Position sizing optimization
- Performance tracking and ROI analysis

Bankroll Allocation Strategy:
- 1X2 markets: 40% ($400 of $1,000)
- Over/Under: 30% ($300)
- Point spreads: 20% ($200)
- Speculative: 10% ($100)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default bankroll settings
DEFAULT_BANKROLL = 100.0
MIN_BET = 5.0
MAX_BET_PCT = 5.0  # Maximum 5% per bet
MAX_DAILY_EXPOSURE = 15.0  # Maximum 15% per day
MAX_MATCH_EXPOSURE = 7.0  # Maximum 7% per match

# Market allocation (percentage of total bankroll)
MARKET_ALLOCATION = {
    "h2h": 0.40,      # 40% for 1X2
    "totals": 0.30,   # 30% for Over/Under
    "spreads": 0.20,  # 20% for Handicaps
    "speculative": 0.10  # 10% for high-risk bets
}

# Kelly Criterion settings
KELLY_FRACTION = 0.25  # Use 25% Kelly (fractional)
MIN_KELLY_EDGE = 0.05  # Minimum 5% edge for Kelly
MAX_KELLY_BET = 0.05   # Cap Kelly at 5% of bankroll

# Market correlation matrix (how correlated are same-match markets)
# 1.0 = perfectly correlated, 0.0 = independent
MARKET_CORRELATIONS = {
    ("h2h", "h2h"): 1.0,
    ("h2h", "totals"): 0.15,  # Low correlation
    ("h2h", "spreads"): 0.65,  # High correlation (win implies margin)
    ("totals", "totals"): 1.0,
    ("totals", "spreads"): 0.20,  # Moderate
    ("spreads", "spreads"): 1.0,
}

# Value thresholds for market types
VALUE_THRESHOLDS = {
    "h2h": {"bet": 0.10, "consider": 0.03},
    "totals": {"bet": 0.08, "consider": 0.03},
    "spreads": {"bet": 0.08, "consider": 0.03},
}

# Data file
BANKROLL_FILE = DATA_DIR / "bankroll.json"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Bet:
    """Individual bet record."""
    id: str
    match: str
    date: str
    market: str  # h2h, totals, spreads
    selection: str  # e.g., "HOME", "OVER 2.5", "HOME -1.5"
    odds: float
    stake: float
    our_probability: float
    implied_probability: float
    value: float
    status: str = "pending"  # pending, won, lost, void
    profit: float = 0.0
    placed_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BankrollState:
    """Current bankroll state."""
    total: float
    available: float
    allocated: Dict[str, float]  # By market
    pending: float  # In active bets
    today_exposed: float
    today_profit: float
    streak: int  # Positive = wins, negative = losses
    max_drawdown: float


@dataclass
class StakingRecommendation:
    """Stake recommendation for a bet."""
    match: str
    market: str
    selection: str
    odds: float
    our_probability: float
    kelly_stake: float
    recommended_stake: float
    max_allowed: float
    reason: str
    risk_level: str  # low, medium, high


# =============================================================================
# BANKROLL DATA MANAGEMENT
# =============================================================================

def load_bankroll() -> Dict:
    """Load bankroll data from file."""
    if BANKROLL_FILE.exists():
        try:
            with open(BANKROLL_FILE) as f:
                return json.load(f)
        except Exception:
            pass

    # Default bankroll structure
    return {
        "initial_bankroll": DEFAULT_BANKROLL,
        "current_bankroll": DEFAULT_BANKROLL,
        "peak_bankroll": DEFAULT_BANKROLL,
        "allocated": {
            "h2h": DEFAULT_BANKROLL * MARKET_ALLOCATION["h2h"],
            "totals": DEFAULT_BANKROLL * MARKET_ALLOCATION["totals"],
            "spreads": DEFAULT_BANKROLL * MARKET_ALLOCATION["spreads"],
            "speculative": DEFAULT_BANKROLL * MARKET_ALLOCATION["speculative"],
        },
        "pending_bets": [],
        "bet_history": [],
        "daily_stats": {},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }


def save_bankroll(data: Dict):
    """Save bankroll data to file."""
    data["updated_at"] = datetime.now().isoformat()
    BANKROLL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BANKROLL_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_bankroll_state() -> BankrollState:
    """Get current bankroll state."""
    data = load_bankroll()
    today = datetime.now().strftime("%Y-%m-%d")

    # Calculate pending exposure
    pending_stakes = sum(b.get("stake", 0) for b in data.get("pending_bets", []))

    # Today's stats
    today_stats = data.get("daily_stats", {}).get(today, {})
    today_exposed = today_stats.get("total_staked", 0)
    today_profit = today_stats.get("profit", 0)

    # Calculate streak
    history = data.get("bet_history", [])[-20:]
    streak = 0
    for bet in reversed(history):
        if bet["status"] == "won":
            if streak >= 0:
                streak += 1
            else:
                break
        elif bet["status"] == "lost":
            if streak <= 0:
                streak -= 1
            else:
                break

    # Max drawdown
    peak = data.get("peak_bankroll", DEFAULT_BANKROLL)
    current = data.get("current_bankroll", DEFAULT_BANKROLL)
    max_drawdown = (peak - current) / peak if peak > 0 else 0

    return BankrollState(
        total=data.get("current_bankroll", DEFAULT_BANKROLL),
        available=data.get("current_bankroll", DEFAULT_BANKROLL) - pending_stakes,
        allocated=data.get("allocated", {}),
        pending=pending_stakes,
        today_exposed=today_exposed,
        today_profit=today_profit,
        streak=streak,
        max_drawdown=max_drawdown
    )


# =============================================================================
# KELLY CRITERION
# =============================================================================

def calculate_kelly(probability: float, odds: float) -> float:
    """Calculate Kelly Criterion stake as fraction of bankroll.

    Kelly formula: f* = (bp - q) / b
    where:
        b = decimal odds - 1 (net odds)
        p = probability of winning
        q = probability of losing (1 - p)
    """
    if odds <= 1 or probability <= 0 or probability >= 1:
        return 0

    b = odds - 1
    p = probability
    q = 1 - p

    kelly = (b * p - q) / b

    # Apply fractional Kelly
    kelly *= KELLY_FRACTION

    # Cap at maximum
    kelly = min(kelly, MAX_KELLY_BET)

    return max(0, kelly)


def calculate_adjusted_kelly(
    probability: float,
    odds: float,
    correlation_factor: float = 1.0
) -> float:
    """Calculate Kelly with correlation adjustment for portfolio.

    When bets are correlated, reduce stake to manage risk.
    """
    base_kelly = calculate_kelly(probability, odds)

    # Reduce stake based on correlation
    # Higher correlation = lower individual stake
    adjusted = base_kelly * (1 - (correlation_factor * 0.3))

    return max(0, adjusted)


# =============================================================================
# STAKE SIZING
# =============================================================================

def calculate_stake(
    market: str,
    our_probability: float,
    odds: float,
    match: str,
    bankroll_state: BankrollState,
    existing_bets: List[Dict] = None
) -> StakingRecommendation:
    """Calculate recommended stake for a bet."""

    existing_bets = existing_bets or []

    # Get market-specific bankroll
    market_bankroll = bankroll_state.allocated.get(market, 0)

    # Calculate Kelly stake
    kelly_fraction = calculate_kelly(our_probability, odds)
    kelly_stake = kelly_fraction * bankroll_state.total

    # Check for same-match bets and calculate correlation
    same_match_bets = [b for b in existing_bets if b.get("match") == match]
    correlation = 0
    for bet in same_match_bets:
        bet_market = bet.get("market", "")
        pair = tuple(sorted([market, bet_market]))
        corr = MARKET_CORRELATIONS.get(pair, 0.2)
        correlation = max(correlation, corr)

    # Adjust Kelly for correlation
    if correlation > 0:
        kelly_stake = calculate_adjusted_kelly(our_probability, odds, correlation) * bankroll_state.total

    # Calculate maximum allowed stake
    max_by_total = bankroll_state.total * (MAX_BET_PCT / 100)
    max_by_market = market_bankroll * (MAX_BET_PCT / 100) * 2  # Allow 2x normal for market
    max_by_daily = (bankroll_state.total * (MAX_DAILY_EXPOSURE / 100)) - bankroll_state.today_exposed

    # Match exposure limit
    same_match_exposure = sum(b.get("stake", 0) for b in same_match_bets)
    max_by_match = (bankroll_state.total * (MAX_MATCH_EXPOSURE / 100)) - same_match_exposure

    max_allowed = min(max_by_total, max_by_market, max_by_daily, max_by_match)
    max_allowed = max(0, max_allowed)

    # Determine recommended stake
    if kelly_stake <= 0:
        recommended = 0
        reason = "No positive expectation"
        risk_level = "none"
    elif max_allowed < MIN_BET:
        recommended = 0
        reason = "Below minimum bet size"
        risk_level = "none"
    else:
        recommended = min(kelly_stake, max_allowed)
        recommended = max(recommended, MIN_BET)
        recommended = round(recommended, 2)

        # Determine reason and risk level
        value = our_probability - (1 / odds)

        if value > 0.15:
            reason = f"High value edge ({value:.1%})"
            risk_level = "medium"
        elif value > 0.10:
            reason = f"Good value edge ({value:.1%})"
            risk_level = "low"
        elif value > 0.05:
            reason = f"Moderate value ({value:.1%})"
            risk_level = "low"
        else:
            reason = f"Small edge ({value:.1%})"
            risk_level = "high"

        # Adjust risk based on correlation
        if correlation > 0.5:
            risk_level = "high"
            reason += " (correlated bet)"

    return StakingRecommendation(
        match=match,
        market=market,
        selection="",  # To be filled by caller
        odds=odds,
        our_probability=our_probability,
        kelly_stake=round(kelly_stake, 2),
        recommended_stake=recommended,
        max_allowed=round(max_allowed, 2),
        reason=reason,
        risk_level=risk_level
    )


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

def optimize_portfolio(
    bets: List[Dict],
    bankroll_state: BankrollState
) -> List[Dict]:
    """Optimize stakes for a portfolio of bets.

    Considers:
    - Cross-market correlations
    - Maximum exposure limits
    - Risk-adjusted returns
    """
    if not bets:
        return []

    # Group bets by match
    by_match = {}
    for bet in bets:
        match = bet.get("match", "")
        if match not in by_match:
            by_match[match] = []
        by_match[match].append(bet)

    optimized = []
    total_exposure = 0
    max_daily = bankroll_state.total * (MAX_DAILY_EXPOSURE / 100)

    for match, match_bets in by_match.items():
        match_exposure = 0
        max_match = bankroll_state.total * (MAX_MATCH_EXPOSURE / 100)

        # Sort by value (highest first)
        match_bets.sort(key=lambda x: x.get("value", 0), reverse=True)

        for bet in match_bets:
            if total_exposure >= max_daily:
                bet["optimized_stake"] = 0
                bet["optimization_reason"] = "Daily limit reached"
            elif match_exposure >= max_match:
                bet["optimized_stake"] = 0
                bet["optimization_reason"] = "Match limit reached"
            else:
                # Calculate optimal stake
                rec = calculate_stake(
                    market=bet.get("market", "h2h"),
                    our_probability=bet.get("our_probability", 0.5),
                    odds=bet.get("odds", 1.5),
                    match=match,
                    bankroll_state=bankroll_state,
                    existing_bets=optimized
                )

                remaining_daily = max_daily - total_exposure
                remaining_match = max_match - match_exposure

                final_stake = min(rec.recommended_stake, remaining_daily, remaining_match)
                final_stake = max(0, round(final_stake, 2))

                bet["optimized_stake"] = final_stake
                bet["kelly_stake"] = rec.kelly_stake
                bet["optimization_reason"] = rec.reason
                bet["risk_level"] = rec.risk_level

                total_exposure += final_stake
                match_exposure += final_stake

            optimized.append(bet)

    return optimized


# =============================================================================
# BET TRACKING
# =============================================================================

def place_bet(
    match: str,
    market: str,
    selection: str,
    odds: float,
    stake: float,
    our_probability: float
) -> Optional[Bet]:
    """Record a placed bet."""

    data = load_bankroll()

    # Create bet record
    bet_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{market[:3]}_{len(data['pending_bets'])}"

    bet = Bet(
        id=bet_id,
        match=match,
        date=datetime.now().strftime("%Y-%m-%d"),
        market=market,
        selection=selection,
        odds=odds,
        stake=stake,
        our_probability=our_probability,
        implied_probability=1/odds,
        value=our_probability - 1/odds,
        status="pending"
    )

    # Add to pending bets
    data["pending_bets"].append({
        "id": bet.id,
        "match": bet.match,
        "date": bet.date,
        "market": bet.market,
        "selection": bet.selection,
        "odds": bet.odds,
        "stake": bet.stake,
        "our_probability": bet.our_probability,
        "value": bet.value,
        "status": "pending",
        "placed_at": bet.placed_at
    })

    # Update daily stats
    today = datetime.now().strftime("%Y-%m-%d")
    if today not in data["daily_stats"]:
        data["daily_stats"][today] = {"total_staked": 0, "bets": 0, "profit": 0}
    data["daily_stats"][today]["total_staked"] += stake
    data["daily_stats"][today]["bets"] += 1

    save_bankroll(data)
    log.info(f"Bet placed: {bet.id} - {selection} @ {odds:.2f} (${stake:.2f})")

    return bet


def settle_bet(bet_id: str, result: str):
    """Settle a pending bet.

    Args:
        bet_id: The bet ID to settle
        result: "won", "lost", or "void"
    """
    data = load_bankroll()

    # Find and settle the bet
    for i, bet in enumerate(data["pending_bets"]):
        if bet["id"] == bet_id:
            bet["status"] = result
            bet["settled_at"] = datetime.now().isoformat()

            if result == "won":
                profit = bet["stake"] * (bet["odds"] - 1)
                bet["profit"] = profit
                data["current_bankroll"] += profit
            elif result == "lost":
                profit = -bet["stake"]
                bet["profit"] = profit
                data["current_bankroll"] -= bet["stake"]
            else:  # void
                bet["profit"] = 0

            # Move to history
            data["bet_history"].append(bet)
            data["pending_bets"].pop(i)

            # Update daily stats
            today = datetime.now().strftime("%Y-%m-%d")
            if today in data["daily_stats"]:
                data["daily_stats"][today]["profit"] += bet["profit"]

            # Update peak
            if data["current_bankroll"] > data.get("peak_bankroll", 0):
                data["peak_bankroll"] = data["current_bankroll"]

            # Re-allocate bankroll
            data["allocated"] = {
                k: data["current_bankroll"] * v
                for k, v in MARKET_ALLOCATION.items()
            }

            save_bankroll(data)
            log.info(f"Bet settled: {bet_id} - {result} (${bet['profit']:+.2f})")
            return

    log.warning(f"Bet not found: {bet_id}")


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================

def get_performance_stats() -> Dict:
    """Calculate performance statistics."""
    data = load_bankroll()

    history = data.get("bet_history", [])
    if not history:
        return {"message": "No betting history"}

    # Basic stats
    total_bets = len(history)
    won = sum(1 for b in history if b["status"] == "won")
    lost = sum(1 for b in history if b["status"] == "lost")
    void = sum(1 for b in history if b["status"] == "void")

    win_rate = won / (won + lost) if (won + lost) > 0 else 0

    total_staked = sum(b["stake"] for b in history if b["status"] != "void")
    total_profit = sum(b["profit"] for b in history)
    roi = total_profit / total_staked if total_staked > 0 else 0

    # By market
    by_market = {}
    for market in MARKET_ALLOCATION.keys():
        market_bets = [b for b in history if b["market"] == market]
        if market_bets:
            m_won = sum(1 for b in market_bets if b["status"] == "won")
            m_total = sum(1 for b in market_bets if b["status"] != "void")
            m_staked = sum(b["stake"] for b in market_bets if b["status"] != "void")
            m_profit = sum(b["profit"] for b in market_bets)

            by_market[market] = {
                "bets": m_total,
                "won": m_won,
                "win_rate": m_won / m_total if m_total > 0 else 0,
                "staked": m_staked,
                "profit": m_profit,
                "roi": m_profit / m_staked if m_staked > 0 else 0
            }

    # Max drawdown
    peak = data.get("initial_bankroll", DEFAULT_BANKROLL)
    current = data.get("current_bankroll", DEFAULT_BANKROLL)
    running = peak
    max_drawdown = 0

    for bet in history:
        running += bet["profit"]
        if running > peak:
            peak = running
        drawdown = (peak - running) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

    return {
        "total_bets": total_bets,
        "won": won,
        "lost": lost,
        "void": void,
        "win_rate": win_rate,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "current_bankroll": data.get("current_bankroll", DEFAULT_BANKROLL),
        "peak_bankroll": data.get("peak_bankroll", DEFAULT_BANKROLL),
        "max_drawdown": max_drawdown,
        "by_market": by_market
    }


# =============================================================================
# REPORTING
# =============================================================================

def print_bankroll_summary():
    """Print current bankroll summary."""
    state = get_bankroll_state()
    stats = get_performance_stats()

    print("\n" + "=" * 60)
    print(" BANKROLL MANAGEMENT DASHBOARD")
    print("=" * 60)

    print(f"\n  CURRENT STATE:")
    print(f"    Total Bankroll: ${state.total:.2f}")
    print(f"    Available: ${state.available:.2f}")
    print(f"    Pending Bets: ${state.pending:.2f}")

    print(f"\n  MARKET ALLOCATION:")
    for market, amount in state.allocated.items():
        pct = amount / state.total * 100 if state.total > 0 else 0
        print(f"    {market.upper()}: ${amount:.2f} ({pct:.0f}%)")

    print(f"\n  TODAY'S ACTIVITY:")
    print(f"    Exposure: ${state.today_exposed:.2f} / ${state.total * MAX_DAILY_EXPOSURE / 100:.2f}")
    print(f"    Profit: ${state.today_profit:+.2f}")

    print(f"\n  RISK METRICS:")
    print(f"    Streak: {state.streak:+d}")
    print(f"    Max Drawdown: {state.max_drawdown:.1%}")

    if isinstance(stats, dict) and "total_bets" in stats:
        print(f"\n  OVERALL PERFORMANCE:")
        print(f"    Total Bets: {stats['total_bets']} ({stats['won']}W/{stats['lost']}L)")
        print(f"    Win Rate: {stats['win_rate']:.1%}")
        print(f"    Total Staked: ${stats['total_staked']:.2f}")
        print(f"    Total Profit: ${stats['total_profit']:+.2f}")
        print(f"    ROI: {stats['roi']:+.1%}")


def generate_betting_slip(bankroll: float = None) -> Dict:
    """Generate comprehensive betting slip with all markets."""

    if bankroll is None:
        state = get_bankroll_state()
        bankroll = state.total
    else:
        state = BankrollState(
            total=bankroll,
            available=bankroll,
            allocated={k: bankroll * v for k, v in MARKET_ALLOCATION.items()},
            pending=0,
            today_exposed=0,
            today_profit=0,
            streak=0,
            max_drawdown=0
        )

    # Load all bets from different models
    all_bets = []

    # Load 1X2 bets
    h2h_path = DATA_DIR / "betting" / "betting_slip.json"
    if h2h_path.exists():
        with open(h2h_path) as f:
            h2h_data = json.load(f)
        for bet in h2h_data.get("recommended_singles", []):
            all_bets.append({
                "match": bet["match"],
                "date": bet["date"],
                "market": "h2h",
                "selection": bet["prediction"],
                "odds": bet["odds"],
                "our_probability": bet["our_probability"],
                "value": bet["value"],
                "original_stake": bet.get("recommended_stake", 0)
            })

    # Load Over/Under bets
    totals_path = DATA_DIR / "upcoming" / "over_under_bets.json"
    if totals_path.exists():
        with open(totals_path) as f:
            totals_data = json.load(f)
        for bet in totals_data.get("recommended", []):
            all_bets.append({
                "match": bet["match"],
                "date": bet["date"],
                "market": "totals",
                "selection": bet["bet"],
                "odds": bet["odds"],
                "our_probability": bet["our_probability"],
                "value": bet["value_pct"] / 100,
                "original_stake": bet.get("stake_pct", 0) * bankroll / 100
            })

    # Load Handicap bets
    handicap_path = DATA_DIR / "upcoming" / "handicap_bets.json"
    if handicap_path.exists():
        with open(handicap_path) as f:
            handicap_data = json.load(f)
        for bet in handicap_data.get("recommended", []):
            all_bets.append({
                "match": bet["match"],
                "date": bet["date"],
                "market": "spreads",
                "selection": bet["bet"],
                "odds": bet["odds"],
                "our_probability": bet["our_probability"],
                "value": bet["value_pct"] / 100,
                "original_stake": bet.get("stake_pct", 0) * bankroll / 100
            })

    # Optimize portfolio
    optimized = optimize_portfolio(all_bets, state)

    # Filter to bets with positive stakes
    final_bets = [b for b in optimized if b.get("optimized_stake", 0) > 0]

    # Calculate totals
    total_stake = sum(b["optimized_stake"] for b in final_bets)
    total_potential_profit = sum(
        b["optimized_stake"] * (b["odds"] - 1) for b in final_bets
    )

    slip = {
        "generated_at": datetime.now().isoformat(),
        "bankroll": bankroll,
        "summary": {
            "total_bets": len(final_bets),
            "by_market": {},
            "total_stake": round(total_stake, 2),
            "stake_pct": round(total_stake / bankroll * 100, 1),
            "potential_profit": round(total_potential_profit, 2)
        },
        "bets": []
    }

    # Count by market
    for market in MARKET_ALLOCATION.keys():
        market_bets = [b for b in final_bets if b["market"] == market]
        if market_bets:
            slip["summary"]["by_market"][market] = {
                "count": len(market_bets),
                "stake": sum(b["optimized_stake"] for b in market_bets)
            }

    # Add bets
    for bet in final_bets:
        slip["bets"].append({
            "match": bet["match"],
            "date": bet["date"],
            "market": bet["market"],
            "selection": bet["selection"],
            "odds": bet["odds"],
            "stake": bet["optimized_stake"],
            "potential_profit": round(bet["optimized_stake"] * (bet["odds"] - 1), 2),
            "value_pct": round(bet["value"] * 100, 1),
            "risk_level": bet.get("risk_level", "medium"),
            "reason": bet.get("optimization_reason", "")
        })

    # Save
    output_path = DATA_DIR / "betting" / "optimized_slip.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(slip, f, indent=2)

    return slip


def print_betting_slip(slip: Dict):
    """Print formatted betting slip."""

    print("\n" + "=" * 80)
    print(" OPTIMIZED BETTING SLIP")
    print("=" * 80)

    summary = slip["summary"]
    print(f"\n  Bankroll: ${slip['bankroll']:.2f}")
    print(f"  Total Bets: {summary['total_bets']}")
    print(f"  Total Stake: ${summary['total_stake']:.2f} ({summary['stake_pct']:.1f}%)")
    print(f"  Potential Profit: ${summary['potential_profit']:.2f}")

    print(f"\n  BY MARKET:")
    for market, data in summary.get("by_market", {}).items():
        print(f"    {market.upper()}: {data['count']} bets, ${data['stake']:.2f}")

    print(f"\n  {'Match':<25} {'Selection':>15} {'Odds':>6} {'Stake':>8} {'Profit':>8} {'Value':>7}")
    print("  " + "-" * 75)

    for bet in slip["bets"]:
        print(f"  {bet['match']:<25} {bet['selection']:>15} {bet['odds']:>6.2f} "
              f"${bet['stake']:>7.2f} ${bet['potential_profit']:>7.2f} {bet['value_pct']:>+6.1f}%")


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Bankroll Management")
    parser.add_argument("--status", action="store_true", help="Show bankroll status")
    parser.add_argument("--slip", action="store_true", help="Generate betting slip")
    parser.add_argument("--bankroll", type=float, help="Set bankroll amount")
    parser.add_argument("--reset", action="store_true", help="Reset bankroll")
    args = parser.parse_args()

    if args.reset:
        bankroll = args.bankroll or DEFAULT_BANKROLL
        data = {
            "initial_bankroll": bankroll,
            "current_bankroll": bankroll,
            "peak_bankroll": bankroll,
            "allocated": {k: bankroll * v for k, v in MARKET_ALLOCATION.items()},
            "pending_bets": [],
            "bet_history": [],
            "daily_stats": {},
            "created_at": datetime.now().isoformat(),
        }
        save_bankroll(data)
        print(f"Bankroll reset to ${bankroll:.2f}")
        return

    if args.status:
        print_bankroll_summary()
        return

    if args.slip or True:  # Default action
        slip = generate_betting_slip(args.bankroll)
        print_betting_slip(slip)


if __name__ == "__main__":
    main()
