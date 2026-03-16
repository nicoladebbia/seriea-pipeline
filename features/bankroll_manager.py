#!/usr/bin/env python3
"""Bankroll Management System - Phase 4.1 & 4.2

Implements:
- Bankroll tracking with position sizing
- Drawdown limits and stop-loss triggers
- Bet logging and P&L tracking
- Risk management rules

Usage:
    from features.bankroll_manager import BankrollManager, BettingTracker

    manager = BankrollManager(initial_bankroll=1000)
    tracker = BettingTracker()
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# BANKROLL MANAGER (Phase 4.1)
# =============================================================================

class BankrollManager:
    """Manages betting bankroll with risk controls.

    Risk Management Rules:
    1. Never bet more than max_bet_pct of bankroll on single bet
    2. Stop betting if drawdown exceeds max_drawdown_pct
    3. Daily loss limit: max_daily_loss_pct
    4. Reduce bet size during losing streaks
    """

    DEFAULT_CONFIG = {
        "max_bet_pct": 0.05,          # Max 5% of bankroll per bet
        "max_drawdown_pct": 0.20,     # Stop at 20% drawdown
        "max_daily_loss_pct": 0.10,   # Max 10% daily loss
        "kelly_fraction": 0.15,       # Quarter Kelly
        "min_bet_units": 5,           # Minimum bet in units
        "losing_streak_reduce": 3,    # Reduce bets after 3 losses
        "losing_streak_factor": 0.5,  # Reduce to 50% size
    }

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        config: Dict = None,
    ):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Tracking
        self.peak_bankroll = initial_bankroll
        self.daily_start = initial_bankroll
        self.daily_pnl = 0.0
        self.current_streak = 0  # Positive = wins, negative = losses
        self.total_bets = 0
        self.total_wins = 0

        # State file
        self.state_file = DATA_DIR / "bankroll" / "state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> bool:
        """Load bankroll state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                self.current_bankroll = state.get("current_bankroll", self.initial_bankroll)
                self.peak_bankroll = state.get("peak_bankroll", self.current_bankroll)
                self.daily_start = state.get("daily_start", self.current_bankroll)
                self.daily_pnl = state.get("daily_pnl", 0)
                self.current_streak = state.get("current_streak", 0)
                self.total_bets = state.get("total_bets", 0)
                self.total_wins = state.get("total_wins", 0)
                log.info(f"Loaded bankroll state: {self.current_bankroll:.2f}")
                return True
            except Exception as e:
                log.error(f"Failed to load state: {e}")
        return False

    def save_state(self):
        """Save bankroll state to file."""
        state = {
            "current_bankroll": self.current_bankroll,
            "initial_bankroll": self.initial_bankroll,
            "peak_bankroll": self.peak_bankroll,
            "daily_start": self.daily_start,
            "daily_pnl": self.daily_pnl,
            "current_streak": self.current_streak,
            "total_bets": self.total_bets,
            "total_wins": self.total_wins,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def reset_daily(self):
        """Reset daily tracking (call at start of each day)."""
        self.daily_start = self.current_bankroll
        self.daily_pnl = 0.0
        self.save_state()

    def get_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_bankroll <= 0:
            return 0
        return (self.peak_bankroll - self.current_bankroll) / self.peak_bankroll

    def get_daily_loss(self) -> float:
        """Calculate daily loss percentage."""
        if self.daily_start <= 0:
            return 0
        return max(0, (self.daily_start - self.current_bankroll) / self.daily_start)

    def can_bet(self) -> Tuple[bool, str]:
        """Check if betting is allowed based on risk rules."""
        # Check drawdown limit
        drawdown = self.get_drawdown()
        if drawdown >= self.config["max_drawdown_pct"]:
            return False, f"Drawdown limit reached: {drawdown:.1%}"

        # Check daily loss limit
        daily_loss = self.get_daily_loss()
        if daily_loss >= self.config["max_daily_loss_pct"]:
            return False, f"Daily loss limit reached: {daily_loss:.1%}"

        # Check minimum bankroll
        if self.current_bankroll < self.config["min_bet_units"]:
            return False, f"Bankroll too low: {self.current_bankroll:.2f}"

        return True, "OK"

    def calculate_bet_size(
        self,
        edge: float,
        odds: float,
        confidence: str = "MEDIUM",
    ) -> float:
        """Calculate optimal bet size based on Kelly and risk rules.

        Args:
            edge: Model edge (probability - implied probability)
            odds: Decimal odds
            confidence: Confidence level (LOW, MEDIUM, HIGH)

        Returns:
            Recommended bet size in units
        """
        can_bet, reason = self.can_bet()
        if not can_bet:
            log.warning(f"Betting blocked: {reason}")
            return 0

        # Base Kelly calculation
        if edge <= 0 or odds <= 1:
            return 0

        b = odds - 1
        p = min(0.99, max(0.01, edge + (1 / odds)))  # Model probability, bounded
        q = 1 - p

        full_kelly = (b * p - q) / b if b > 0 else 0
        full_kelly = max(0, full_kelly)  # Never negative
        kelly_bet = full_kelly * self.config["kelly_fraction"]

        # Adjust for confidence (applied to Kelly fraction before bankroll scaling)
        confidence_multiplier = {
            "LOW": 0.5,
            "MEDIUM": 1.0,
            "HIGH": 1.5,
        }.get(confidence, 1.0)

        # Adjust for losing streak
        if self.current_streak <= -self.config["losing_streak_reduce"]:
            streak_multiplier = self.config["losing_streak_factor"]
        else:
            streak_multiplier = 1.0

        # Apply multipliers to Kelly fraction, then cap, then scale to bankroll
        adjusted_kelly = kelly_bet * confidence_multiplier * streak_multiplier
        adjusted_kelly = min(adjusted_kelly, self.config["max_bet_pct"])
        bet_size = adjusted_kelly * self.current_bankroll

        # Apply minimum
        if bet_size < self.config["min_bet_units"]:
            return 0

        return round(bet_size, 2)

    def record_bet(self, stake: float, won: bool, returns: float = 0):
        """Record a bet result and update bankroll.

        Args:
            stake: Amount staked
            won: Whether bet won
            returns: Total returns if won (stake + profit)
        """
        self.total_bets += 1

        if won:
            profit = returns - stake
            self.current_bankroll += profit
            self.total_wins += 1
            self.current_streak = max(1, self.current_streak + 1)
        else:
            self.current_bankroll -= stake
            self.current_streak = min(-1, self.current_streak - 1)

        # Update daily P&L
        self.daily_pnl = self.current_bankroll - self.daily_start

        # Update peak
        if self.current_bankroll > self.peak_bankroll:
            self.peak_bankroll = self.current_bankroll

        self.save_state()

    def get_status(self) -> Dict:
        """Get current bankroll status."""
        return {
            "current_bankroll": round(self.current_bankroll, 2),
            "initial_bankroll": self.initial_bankroll,
            "peak_bankroll": round(self.peak_bankroll, 2),
            "drawdown": round(self.get_drawdown(), 4),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_loss": round(self.get_daily_loss(), 4),
            "current_streak": self.current_streak,
            "total_bets": self.total_bets,
            "total_wins": self.total_wins,
            "win_rate": self.total_wins / self.total_bets if self.total_bets > 0 else 0,
            "roi": (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll,
            "can_bet": self.can_bet()[0],
        }


# =============================================================================
# BETTING TRACKER (Phase 4.2)
# =============================================================================

class BettingTracker:
    """Tracks all bets with detailed logging and reporting."""

    def __init__(self):
        self.bets_file = DATA_DIR / "bankroll" / "bets.json"
        self.bets_file.parent.mkdir(parents=True, exist_ok=True)
        self.bets = []
        self.load_bets()

    def load_bets(self):
        """Load bet history from file."""
        if self.bets_file.exists():
            try:
                with open(self.bets_file) as f:
                    self.bets = json.load(f)
                log.info(f"Loaded {len(self.bets)} bet records")
            except Exception as e:
                log.error(f"Failed to load bets: {e}")
                self.bets = []

    def save_bets(self):
        """Save bet history to file."""
        with open(self.bets_file, "w") as f:
            json.dump(self.bets, f, indent=2, default=str)

    def log_bet(
        self,
        match: str,
        prediction: str,
        odds: float,
        stake: float,
        confidence: float,
        edge: float,
        strategy: str = "default",
        notes: str = "",
    ) -> int:
        """Log a new bet.

        Returns:
            Bet ID
        """
        bet_id = len(self.bets) + 1

        bet = {
            "id": bet_id,
            "timestamp": datetime.now().isoformat(),
            "match": match,
            "prediction": prediction,
            "odds": odds,
            "stake": stake,
            "confidence": confidence,
            "edge": edge,
            "strategy": strategy,
            "notes": notes,
            "status": "pending",
            "actual_result": None,
            "won": None,
            "returns": None,
            "profit": None,
        }

        self.bets.append(bet)
        self.save_bets()

        return bet_id

    def settle_bet(
        self,
        bet_id: int,
        actual_result: str,
        won: bool,
        returns: float = 0,
    ):
        """Settle a pending bet.

        Args:
            bet_id: Bet ID to settle
            actual_result: Actual match result (HOME/DRAW/AWAY)
            won: Whether bet won
            returns: Total returns if won
        """
        for bet in self.bets:
            if bet["id"] == bet_id:
                bet["status"] = "settled"
                bet["actual_result"] = actual_result
                bet["won"] = won
                bet["returns"] = returns
                bet["profit"] = returns - bet["stake"] if won else -bet["stake"]
                bet["settled_at"] = datetime.now().isoformat()
                break

        self.save_bets()

    def get_pending_bets(self) -> List[Dict]:
        """Get all pending (unsettled) bets."""
        return [b for b in self.bets if b["status"] == "pending"]

    def get_settled_bets(self, days: int = None) -> List[Dict]:
        """Get settled bets, optionally filtered by days."""
        settled = [b for b in self.bets if b["status"] == "settled"]

        if days:
            cutoff = datetime.now() - timedelta(days=days)
            settled = [
                b for b in settled
                if datetime.fromisoformat(b["timestamp"]) >= cutoff
            ]

        return settled

    def get_stats(self, days: int = None) -> Dict:
        """Get betting statistics."""
        settled = self.get_settled_bets(days)

        if not settled:
            return {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_stake": 0,
                "total_returns": 0,
                "total_profit": 0,
                "roi": 0,
                "avg_odds": 0,
                "avg_stake": 0,
            }

        wins = [b for b in settled if b["won"]]
        losses = [b for b in settled if not b["won"]]

        total_stake = sum(b["stake"] for b in settled)
        total_returns = sum(b["returns"] or 0 for b in settled)
        total_profit = sum(b["profit"] or 0 for b in settled)

        return {
            "total_bets": len(settled),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(settled) if settled else 0,
            "total_stake": round(total_stake, 2),
            "total_returns": round(total_returns, 2),
            "total_profit": round(total_profit, 2),
            "roi": round(total_profit / total_stake * 100, 2) if total_stake > 0 else 0,
            "avg_odds": round(np.mean([b["odds"] for b in settled]), 2),
            "avg_stake": round(np.mean([b["stake"] for b in settled]), 2),
        }

    def get_stats_by_prediction(self) -> Dict:
        """Get stats grouped by prediction type."""
        settled = self.get_settled_bets()

        stats = {}
        for pred_type in ["HOME", "DRAW", "AWAY"]:
            bets = [b for b in settled if b["prediction"] == pred_type]
            if bets:
                wins = [b for b in bets if b["won"]]
                total_stake = sum(b["stake"] for b in bets)
                total_profit = sum(b["profit"] or 0 for b in bets)

                stats[pred_type] = {
                    "bets": len(bets),
                    "wins": len(wins),
                    "win_rate": len(wins) / len(bets),
                    "profit": round(total_profit, 2),
                    "roi": round(total_profit / total_stake * 100, 2) if total_stake > 0 else 0,
                }

        return stats

    def generate_report(self, days: int = 30) -> str:
        """Generate a text report of betting performance."""
        stats = self.get_stats(days)
        by_pred = self.get_stats_by_prediction()

        report = []
        report.append("=" * 60)
        report.append(f"BETTING REPORT (Last {days} days)")
        report.append("=" * 60)

        report.append(f"\nOverall Performance:")
        report.append(f"  Total Bets: {stats['total_bets']}")
        report.append(f"  Win Rate: {stats['win_rate']:.1%}")
        report.append(f"  Total Profit: {stats['total_profit']:+.2f} units")
        report.append(f"  ROI: {stats['roi']:+.1f}%")

        report.append(f"\nBy Prediction Type:")
        for pred_type, data in by_pred.items():
            report.append(f"  {pred_type}: {data['wins']}/{data['bets']} wins "
                         f"({data['win_rate']:.1%}), ROI {data['roi']:+.1f}%")

        return "\n".join(report)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_bankroll_manager(initial: float = 1000.0) -> BankrollManager:
    """Get or create bankroll manager with state loading."""
    manager = BankrollManager(initial_bankroll=initial)
    manager.load_state()
    return manager


def get_betting_tracker() -> BettingTracker:
    """Get betting tracker with history loaded."""
    return BettingTracker()


# =============================================================================
# BACKWARD-COMPAT FUNCTIONAL API
# Migrated from scripts/bankroll_manager.py so callers (tests, alert_system)
# use the same Kelly fraction as production (0.15, not 0.25).
# =============================================================================

KELLY_FRACTION = BankrollManager.DEFAULT_CONFIG["kelly_fraction"]  # 0.15
MAX_SINGLE_BET_PCT = BankrollManager.DEFAULT_CONFIG["max_bet_pct"]  # 0.05


def calculate_kelly_stake(
    bankroll: float,
    win_probability: float,
    decimal_odds: float,
    fraction: float = KELLY_FRACTION,
) -> float:
    """Calculate optimal stake using fractional Kelly Criterion.

    Kelly formula: f* = (bp - q) / b
    where b = decimal_odds - 1, p = win_probability, q = 1 - p.
    Result is scaled by *fraction* and capped at MAX_SINGLE_BET_PCT of bankroll.
    """
    if win_probability <= 0 or win_probability >= 1:
        return 0
    if decimal_odds <= 1:
        return 0

    b = decimal_odds - 1
    p = win_probability
    q = 1 - p

    kelly = (b * p - q) / b
    if kelly <= 0:
        return 0

    stake = bankroll * kelly * fraction
    max_stake = bankroll * MAX_SINGLE_BET_PCT
    stake = min(stake, max_stake)
    return round(stake, 2)


def calculate_value(win_probability: float, decimal_odds: float) -> float:
    """Calculate betting value (edge). Value = (probability * odds) - 1."""
    return (win_probability * decimal_odds) - 1


def load_history() -> List[Dict]:
    """Load bet history from the betting data directory."""
    history_file = DATA_DIR / "betting" / "history.json"
    if history_file.exists():
        try:
            with open(history_file) as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load history: {e}")
    return []


def get_performance_stats(history: List[Dict] = None) -> Dict:
    """Calculate comprehensive performance statistics from bet history."""
    if history is None:
        history = load_history()
    if not history:
        return {"error": "No bet history"}

    total_bets = len(history)
    wins = [b for b in history if b.get("status") == "won"]
    losses = [b for b in history if b.get("status") == "lost"]
    total_staked = sum(b.get("stake", 0) for b in history)
    total_profit = sum(b.get("profit", 0) for b in history)

    return {
        "total_bets": total_bets,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / total_bets * 100, 1) if total_bets > 0 else 0,
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0,
    }


if __name__ == "__main__":
    # Demo
    print("Bankroll Management System Demo")
    print("=" * 60)

    # Create manager
    manager = get_bankroll_manager(initial=1000)
    tracker = get_betting_tracker()

    # Show status
    status = manager.get_status()
    print(f"\nBankroll Status:")
    print(f"  Current: {status['current_bankroll']:.2f}")
    print(f"  Drawdown: {status['drawdown']:.1%}")
    print(f"  Can bet: {status['can_bet']}")

    # Calculate bet size
    bet_size = manager.calculate_bet_size(edge=0.05, odds=2.0, confidence="MEDIUM")
    print(f"\nRecommended bet size (5% edge, 2.0 odds): {bet_size:.2f}")

    # Show tracker stats
    stats = tracker.get_stats()
    print(f"\nBetting Stats:")
    print(f"  Total bets: {stats['total_bets']}")
    print(f"  Win rate: {stats['win_rate']:.1%}")
    print(f"  ROI: {stats['roi']:+.1f}%")
