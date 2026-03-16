#!/usr/bin/env python3
"""BANKROLL MANAGEMENT SYSTEM - Professional Money Management

Complete bankroll tracking and stake sizing:
1. Kelly Criterion (optimal sizing)
2. Flat Staking (conservative)
3. Bet History Tracking
4. ROI Analysis
5. Performance Reports

This is CRITICAL for long-term profitability.

Usage:
    python scripts/bankroll_manager.py status           # Show bankroll status
    python scripts/bankroll_manager.py place BET_ID     # Record a bet
    python scripts/bankroll_manager.py settle BET_ID    # Settle a bet
    python scripts/bankroll_manager.py report           # Performance report
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BANKROLL = 1000.0  # Starting bankroll
MAX_SINGLE_BET_PCT = 0.05  # Maximum 5% on single bet
KELLY_FRACTION = 0.25      # Use 1/4 Kelly for safety

# Stake sizing by confidence level (% of bankroll)
FLAT_STAKES = {
    "VERY_HIGH": 0.03,    # 3% for 80%+ confidence
    "HIGH": 0.02,         # 2% for 70-79%
    "MEDIUM_HIGH": 0.015, # 1.5% for 65-69%
    "MEDIUM": 0.01,       # 1% for 60-64%
    "LOW": 0.005,         # 0.5% for below 60%
}

# =============================================================================
# DATA PATHS
# =============================================================================

BANKROLL_FILE = DATA_DIR / "betting" / "bankroll.json"
BETS_FILE = DATA_DIR / "betting" / "bets.json"
HISTORY_FILE = DATA_DIR / "betting" / "history.json"


def ensure_data_dir():
    """Ensure betting data directory exists."""
    (DATA_DIR / "betting").mkdir(parents=True, exist_ok=True)


def load_bankroll() -> Dict:
    """Load bankroll data."""
    ensure_data_dir()

    if BANKROLL_FILE.exists():
        with open(BANKROLL_FILE) as f:
            return json.load(f)

    # Initialize new bankroll
    bankroll = {
        "initial_balance": DEFAULT_BANKROLL,
        "current_balance": DEFAULT_BANKROLL,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "total_deposited": DEFAULT_BANKROLL,
        "total_withdrawn": 0.0,
        "peak_balance": DEFAULT_BANKROLL,
        "lowest_balance": DEFAULT_BANKROLL,
    }

    save_bankroll(bankroll)
    return bankroll


def save_bankroll(bankroll: Dict):
    """Save bankroll data."""
    ensure_data_dir()
    bankroll["updated_at"] = datetime.now().isoformat()
    with open(BANKROLL_FILE, "w") as f:
        json.dump(bankroll, f, indent=2)


def load_bets() -> List[Dict]:
    """Load pending bets."""
    ensure_data_dir()

    if BETS_FILE.exists():
        with open(BETS_FILE) as f:
            return json.load(f)
    return []


def save_bets(bets: List[Dict]):
    """Save pending bets."""
    ensure_data_dir()
    with open(BETS_FILE, "w") as f:
        json.dump(bets, f, indent=2)


def load_history() -> List[Dict]:
    """Load bet history."""
    ensure_data_dir()

    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: List[Dict]):
    """Save bet history."""
    ensure_data_dir()
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# =============================================================================
# STAKE SIZING ALGORITHMS
# =============================================================================

def calculate_kelly_stake(
    bankroll: float,
    win_probability: float,
    decimal_odds: float,
    fraction: float = KELLY_FRACTION
) -> float:
    """Calculate optimal stake using Kelly Criterion.

    Kelly formula: f* = (bp - q) / b
    where:
        b = decimal odds - 1 (net odds)
        p = probability of winning
        q = probability of losing (1 - p)

    Args:
        bankroll: Current bankroll
        win_probability: Our estimated win probability (0-1)
        decimal_odds: Decimal odds (e.g., 1.50, 2.00)
        fraction: Kelly fraction (0.25 = quarter Kelly)

    Returns:
        Recommended stake amount
    """
    if win_probability <= 0 or win_probability >= 1:
        return 0

    if decimal_odds <= 1:
        return 0

    b = decimal_odds - 1  # Net odds
    p = win_probability
    q = 1 - p

    # Kelly formula
    kelly = (b * p - q) / b

    # If negative, don't bet (no edge)
    if kelly <= 0:
        return 0

    # Apply fraction and bankroll
    stake = bankroll * kelly * fraction

    # Cap at maximum
    max_stake = bankroll * MAX_SINGLE_BET_PCT
    stake = min(stake, max_stake)

    return round(stake, 2)


def calculate_flat_stake(
    bankroll: float,
    confidence: str
) -> float:
    """Calculate stake using flat staking system.

    Simpler approach with fixed percentages by confidence level.

    Args:
        bankroll: Current bankroll
        confidence: Confidence level string

    Returns:
        Recommended stake amount
    """
    # Normalize confidence (handle numeric from ensemble)
    if isinstance(confidence, (int, float)):
        confidence = "HIGH" if confidence >= 0.55 else "MEDIUM_HIGH" if confidence >= 0.50 else "MEDIUM"
    conf_key = confidence.upper().replace("-", "_").replace(" ", "_")

    # Find matching stake percentage
    stake_pct = FLAT_STAKES.get(conf_key, FLAT_STAKES["LOW"])

    stake = bankroll * stake_pct

    # Cap at maximum
    max_stake = bankroll * MAX_SINGLE_BET_PCT
    stake = min(stake, max_stake)

    return round(stake, 2)


def calculate_value(
    win_probability: float,
    decimal_odds: float
) -> float:
    """Calculate betting value (edge).

    Value = (probability × odds) - 1

    Args:
        win_probability: Our estimated win probability
        decimal_odds: Decimal odds offered

    Returns:
        Value as decimal (0.10 = 10% edge)
    """
    return (win_probability * decimal_odds) - 1


def get_stake_recommendation(
    prediction: Dict,
    bankroll: float,
    odds: float = None
) -> Dict:
    """Get comprehensive stake recommendation for a prediction.

    Args:
        prediction: Prediction dict from prediction engine
        bankroll: Current bankroll amount
        odds: Decimal odds (if available)

    Returns:
        Dict with stake recommendations
    """
    confidence = prediction.get("confidence_level", prediction.get("confidence", "LOW"))
    if isinstance(confidence, (int, float)):
        confidence = "HIGH" if confidence >= 0.55 else "MEDIUM-HIGH" if confidence >= 0.50 else "MEDIUM"
    probs = prediction.get("probabilities", {})
    outcome = prediction.get("predicted_outcome", "HOME")

    # Get win probability for predicted outcome
    if outcome == "HOME":
        win_prob = probs.get("home", 0.5)
    elif outcome == "AWAY":
        win_prob = probs.get("away", 0.3)
    else:
        win_prob = probs.get("draw", 0.25)

    # Calculate stakes
    flat_stake = calculate_flat_stake(bankroll, confidence)

    # Kelly calculation (if odds provided)
    kelly_stake = 0
    value = 0
    implied_prob = 0

    if odds and odds > 1:
        kelly_stake = calculate_kelly_stake(bankroll, win_prob, odds)
        value = calculate_value(win_prob, odds)
        implied_prob = 1 / odds

    # Determine recommendation
    recommendation = {
        "match": prediction.get("match", "Unknown"),
        "prediction": outcome,
        "confidence": confidence,
        "our_probability": round(win_prob, 3),
        "flat_stake": flat_stake,
        "kelly_stake": kelly_stake,
        "recommended_stake": flat_stake,  # Default to flat
        "bankroll": bankroll,
        "stake_pct": round(flat_stake / bankroll * 100, 2),
    }

    if odds:
        recommendation.update({
            "odds": odds,
            "implied_probability": round(implied_prob, 3),
            "value": round(value, 3),
            "has_value": value > 0.05,  # Require 5% minimum edge
        })

        # Use Kelly if we have edge and Kelly suggests less
        if value > 0.05 and kelly_stake > 0:
            recommendation["recommended_stake"] = min(flat_stake, kelly_stake)
            recommendation["stake_pct"] = round(recommendation["recommended_stake"] / bankroll * 100, 2)

        # Warning if no value
        if value <= 0:
            recommendation["warning"] = "NO VALUE - odds too low for our probability"
            recommendation["recommended_stake"] = 0

    return recommendation


# =============================================================================
# BET MANAGEMENT
# =============================================================================

def create_bet(
    prediction: Dict,
    stake: float,
    odds: float,
    bookmaker: str = "Unknown"
) -> Dict:
    """Create a new bet record."""
    bet_id = str(uuid.uuid4())[:8]

    bet = {
        "id": bet_id,
        "match": prediction.get("match", "Unknown"),
        "date": prediction.get("date", "Unknown"),
        "prediction": prediction.get("predicted_outcome", "Unknown"),
        "confidence": prediction.get("confidence", "Unknown"),
        "our_probability": prediction.get("probabilities", {}).get(
            prediction.get("predicted_outcome", "home").lower(), 0.5
        ),
        "stake": stake,
        "odds": odds,
        "potential_return": round(stake * odds, 2),
        "potential_profit": round(stake * (odds - 1), 2),
        "bookmaker": bookmaker,
        "status": "pending",
        "placed_at": datetime.now().isoformat(),
        "factors": prediction.get("home_factors", []) + prediction.get("away_factors", []),
    }

    return bet


def place_bet(bet: Dict) -> bool:
    """Record a placed bet and update bankroll."""
    bankroll = load_bankroll()
    bets = load_bets()

    # Check sufficient funds
    if bet["stake"] > bankroll["current_balance"]:
        log.error(f"Insufficient funds: {bet['stake']} > {bankroll['current_balance']}")
        return False

    # Deduct stake from bankroll
    bankroll["current_balance"] -= bet["stake"]
    bankroll["lowest_balance"] = min(bankroll["lowest_balance"], bankroll["current_balance"])

    # Add to pending bets
    bets.append(bet)

    save_bankroll(bankroll)
    save_bets(bets)

    log.info(f"Placed bet {bet['id']}: {bet['stake']} on {bet['match']} @ {bet['odds']}")
    return True


def settle_bet(bet_id: str, won: bool, actual_result: str = None) -> bool:
    """Settle a bet and update bankroll."""
    bankroll = load_bankroll()
    bets = load_bets()
    history = load_history()

    # Find the bet
    bet = None
    for i, b in enumerate(bets):
        if b["id"] == bet_id:
            bet = bets.pop(i)
            break

    if not bet:
        log.error(f"Bet {bet_id} not found")
        return False

    # Update bet status
    bet["status"] = "won" if won else "lost"
    bet["settled_at"] = datetime.now().isoformat()
    bet["actual_result"] = actual_result

    # Update bankroll
    if won:
        bankroll["current_balance"] += bet["potential_return"]
        bet["profit"] = bet["potential_profit"]
    else:
        bet["profit"] = -bet["stake"]

    bankroll["peak_balance"] = max(bankroll["peak_balance"], bankroll["current_balance"])
    bankroll["lowest_balance"] = min(bankroll["lowest_balance"], bankroll["current_balance"])

    # Move to history
    history.append(bet)

    save_bankroll(bankroll)
    save_bets(bets)
    save_history(history)

    status = "WON" if won else "LOST"
    log.info(f"Settled bet {bet_id}: {status} | Profit: {bet['profit']}")
    return True


# =============================================================================
# REPORTING
# =============================================================================

def get_performance_stats(history: List[Dict] = None) -> Dict:
    """Calculate comprehensive performance statistics."""
    if history is None:
        history = load_history()

    if not history:
        return {"error": "No bet history"}

    total_bets = len(history)
    wins = [b for b in history if b["status"] == "won"]
    losses = [b for b in history if b["status"] == "lost"]

    total_staked = sum(b["stake"] for b in history)
    total_profit = sum(b.get("profit", 0) for b in history)
    total_returns = sum(b.get("potential_return", 0) for b in wins)

    # By confidence level
    by_confidence = {}
    for conf in ["VERY_HIGH", "HIGH", "MEDIUM_HIGH", "MEDIUM", "LOW"]:
        conf_bets = [b for b in history if b.get("confidence", "").upper().replace("-", "_").replace(" ", "_") == conf]
        if conf_bets:
            conf_wins = len([b for b in conf_bets if b["status"] == "won"])
            conf_staked = sum(b["stake"] for b in conf_bets)
            conf_profit = sum(b.get("profit", 0) for b in conf_bets)
            by_confidence[conf] = {
                "bets": len(conf_bets),
                "wins": conf_wins,
                "win_rate": round(conf_wins / len(conf_bets) * 100, 1),
                "staked": round(conf_staked, 2),
                "profit": round(conf_profit, 2),
                "roi": round(conf_profit / conf_staked * 100, 1) if conf_staked > 0 else 0,
            }

    # Time-based analysis
    recent_30d = [b for b in history if datetime.fromisoformat(b["settled_at"]) > datetime.now() - timedelta(days=30)]
    recent_7d = [b for b in history if datetime.fromisoformat(b["settled_at"]) > datetime.now() - timedelta(days=7)]

    return {
        "total_bets": total_bets,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / total_bets * 100, 1) if total_bets > 0 else 0,
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0,
        "average_stake": round(total_staked / total_bets, 2) if total_bets > 0 else 0,
        "average_odds": round(sum(b["odds"] for b in history) / total_bets, 2) if total_bets > 0 else 0,
        "by_confidence": by_confidence,
        "last_30_days": {
            "bets": len(recent_30d),
            "profit": round(sum(b.get("profit", 0) for b in recent_30d), 2),
        },
        "last_7_days": {
            "bets": len(recent_7d),
            "profit": round(sum(b.get("profit", 0) for b in recent_7d), 2),
        },
    }


def print_status():
    """Print current bankroll status."""
    bankroll = load_bankroll()
    bets = load_bets()

    print("\n" + "=" * 60)
    print(" BANKROLL STATUS")
    print("=" * 60)

    print(f"\n  Current Balance: ${bankroll['current_balance']:.2f}")
    print(f"  Initial Balance: ${bankroll['initial_balance']:.2f}")

    profit = bankroll['current_balance'] - bankroll['initial_balance']
    roi = (profit / bankroll['initial_balance']) * 100

    if profit >= 0:
        print(f"  Total Profit:    ${profit:.2f} (+{roi:.1f}%)")
    else:
        print(f"  Total Loss:      ${profit:.2f} ({roi:.1f}%)")

    print(f"\n  Peak Balance:    ${bankroll['peak_balance']:.2f}")
    print(f"  Lowest Balance:  ${bankroll['lowest_balance']:.2f}")

    if bets:
        print(f"\n  Pending Bets:    {len(bets)}")
        pending_stake = sum(b["stake"] for b in bets)
        print(f"  At Risk:         ${pending_stake:.2f}")

        print("\n  Active Bets:")
        for bet in bets:
            print(f"    [{bet['id']}] {bet['match']} | {bet['prediction']} @ {bet['odds']} | ${bet['stake']}")
    else:
        print("\n  No pending bets")


def print_report():
    """Print performance report."""
    stats = get_performance_stats()

    if "error" in stats:
        print(f"\n{stats['error']}")
        return

    print("\n" + "=" * 70)
    print(" PERFORMANCE REPORT")
    print("=" * 70)

    print(f"\n  OVERALL:")
    print(f"    Total Bets:    {stats['total_bets']}")
    print(f"    Win Rate:      {stats['win_rate']}% ({stats['wins']}/{stats['total_bets']})")
    print(f"    ROI:           {stats['roi']}%")
    print(f"    Total Profit:  ${stats['total_profit']:.2f}")
    print(f"    Total Staked:  ${stats['total_staked']:.2f}")

    print(f"\n  BY CONFIDENCE LEVEL:")
    for conf, data in stats.get("by_confidence", {}).items():
        print(f"    {conf}: {data['win_rate']}% win rate | {data['roi']}% ROI | {data['bets']} bets")

    print(f"\n  RECENT PERFORMANCE:")
    print(f"    Last 7 days:   {stats['last_7_days']['bets']} bets | ${stats['last_7_days']['profit']:.2f}")
    print(f"    Last 30 days:  {stats['last_30_days']['bets']} bets | ${stats['last_30_days']['profit']:.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Bankroll Management System")
    parser.add_argument("command", nargs="?", default="status",
                       choices=["status", "report", "init", "deposit", "withdraw"],
                       help="Command to execute")
    parser.add_argument("--amount", type=float, help="Amount for deposit/withdraw")
    args = parser.parse_args()

    if args.command == "status":
        print_status()

    elif args.command == "report":
        print_report()

    elif args.command == "init":
        # Force re-initialize
        bankroll = {
            "initial_balance": DEFAULT_BANKROLL,
            "current_balance": DEFAULT_BANKROLL,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_deposited": DEFAULT_BANKROLL,
            "total_withdrawn": 0.0,
            "peak_balance": DEFAULT_BANKROLL,
            "lowest_balance": DEFAULT_BANKROLL,
        }
        save_bankroll(bankroll)
        print(f"Initialized bankroll with ${DEFAULT_BANKROLL:.2f}")

    elif args.command == "deposit":
        if not args.amount:
            print("Error: --amount required for deposit")
            return

        bankroll = load_bankroll()
        bankroll["current_balance"] += args.amount
        bankroll["total_deposited"] += args.amount
        bankroll["peak_balance"] = max(bankroll["peak_balance"], bankroll["current_balance"])
        save_bankroll(bankroll)
        print(f"Deposited ${args.amount:.2f} | New balance: ${bankroll['current_balance']:.2f}")

    elif args.command == "withdraw":
        if not args.amount:
            print("Error: --amount required for withdraw")
            return

        bankroll = load_bankroll()
        if args.amount > bankroll["current_balance"]:
            print(f"Error: Cannot withdraw ${args.amount:.2f} from ${bankroll['current_balance']:.2f}")
            return

        bankroll["current_balance"] -= args.amount
        bankroll["total_withdrawn"] += args.amount
        save_bankroll(bankroll)
        print(f"Withdrew ${args.amount:.2f} | New balance: ${bankroll['current_balance']:.2f}")


if __name__ == "__main__":
    main()
