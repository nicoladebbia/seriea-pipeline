#!/usr/bin/env python3
"""Risk Controls — Pre-betting gates that protect the bankroll.

Checks run BEFORE generating new bets. If any check fails, betting is
paused with a clear reason and notification.

Controls:
  1. Max drawdown limit — pause if bankroll drops >X% from peak
  2. Consecutive loss breaker — pause after N straight losses
  3. Market kill switch — disable market if live ROI diverges from backtest
  4. Bankroll floor — hard stop below minimum bankroll

Usage:
    from scripts.betting.risk_controls import check_risk_gates

    gates = check_risk_gates(bankroll=1000.0)
    if not gates["allow_betting"]:
        print(f"BETTING PAUSED: {gates['reason']}")
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RiskConfig:
    """Risk control thresholds. Conservative defaults for early-stage system."""

    # Drawdown limits (% of starting bankroll)
    max_drawdown_pct: float = 25.0       # Pause if down 25% from peak
    warning_drawdown_pct: float = 15.0   # Warn at 15%

    # Consecutive losses
    max_consecutive_losses: int = 8      # Pause after 8 straight losses
    warning_consecutive_losses: int = 5  # Warn at 5

    # Bankroll floor
    min_bankroll_pct: float = 50.0       # Hard stop if bankroll < 50% of start
    starting_bankroll: float = 1000.0    # Reference starting bankroll

    # Market kill switches (live ROI thresholds to disable a market)
    # If live ROI is worse than threshold with >= min_bets, disable the market
    market_kill_min_bets: int = 15       # Need 15+ bets before killing a market
    market_kill_roi_threshold: float = -30.0  # Kill if live ROI < -30%

    # Stake reduction under stress
    # When drawdown > warning but < max, reduce stakes by this factor
    stress_stake_multiplier: float = 0.5


# =============================================================================
# CORE RISK CHECKS
# =============================================================================

def _load_settled_bets() -> List[Dict]:
    """Load settled bets from journal, ordered by settlement date."""
    journal_path = DATA_DIR / "betting" / "bet_journal.json"
    if not journal_path.exists():
        return []
    try:
        with open(journal_path) as f:
            journal = json.load(f)
        bets = journal.get("bets", {})
        settled = [
            b for b in bets.values()
            if isinstance(b, dict) and b.get("status") in ("won", "lost", "push", "void")
        ]
        return sorted(settled, key=lambda b: b.get("settled_at", b.get("date", "")))
    except Exception as e:
        log.warning("Failed to load journal: %s", e)
        return []


def check_drawdown(settled: List[Dict], cfg: RiskConfig) -> Dict:
    """Check if current drawdown exceeds limits.

    Drawdown = (peak bankroll - current bankroll) / peak bankroll.
    Standard industry formula: measures decline from highest achieved balance.
    """
    if len(settled) < 5:
        return {"status": "ok", "drawdown_pct": 0, "message": "insufficient data"}

    running = 0.0
    peak = 0.0
    max_dd = 0.0

    for bet in settled:
        running += bet.get("profit", 0) or 0
        peak = max(peak, running)
        dd = peak - running
        max_dd = max(max_dd, dd)

    # Standard drawdown: % decline from peak bankroll (not from starting bankroll).
    # Old formula divided by starting_bankroll, inflating drawdown (25.4% vs 19.8%).
    peak_bankroll = cfg.starting_bankroll + peak
    current_bankroll = cfg.starting_bankroll + running
    dd_pct = (max_dd / peak_bankroll * 100) if peak_bankroll > 0 else 0
    current_dd = ((peak_bankroll - current_bankroll) / peak_bankroll * 100) if peak_bankroll > 0 else 0

    result = {
        "current_pnl": round(running, 2),
        "peak_pnl": round(peak, 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(dd_pct, 1),
        "current_drawdown_pct": round(current_dd, 1),
    }

    if current_dd >= cfg.max_drawdown_pct:
        result["status"] = "BLOCKED"
        result["message"] = (
            f"Drawdown {current_dd:.1f}% exceeds limit {cfg.max_drawdown_pct}%. "
            f"Betting paused until recovery."
        )
    elif current_dd >= cfg.warning_drawdown_pct:
        result["status"] = "WARNING"
        result["message"] = (
            f"Drawdown {current_dd:.1f}% approaching limit {cfg.max_drawdown_pct}%. "
            f"Stakes reduced by {(1-cfg.stress_stake_multiplier)*100:.0f}%."
        )
    else:
        result["status"] = "ok"
        result["message"] = f"Drawdown {current_dd:.1f}% within limits"

    return result


def check_consecutive_losses(settled: List[Dict], cfg: RiskConfig) -> Dict:
    """Check for consecutive loss streaks."""
    if not settled:
        return {"status": "ok", "streak": 0, "message": "no data"}

    # Count current streak (from most recent)
    current_streak = 0
    for bet in reversed(settled):
        if bet.get("status") == "lost":
            current_streak += 1
        else:
            break

    # Find max historical streak
    max_streak = 0
    streak = 0
    for bet in settled:
        if bet.get("status") == "lost":
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    result = {
        "current_streak": current_streak,
        "max_streak": max_streak,
    }

    if current_streak >= cfg.max_consecutive_losses:
        result["status"] = "BLOCKED"
        result["message"] = (
            f"{current_streak} consecutive losses — circuit breaker triggered. "
            f"Betting paused. Review model calibration."
        )
    elif current_streak >= cfg.warning_consecutive_losses:
        result["status"] = "WARNING"
        result["message"] = (
            f"{current_streak} consecutive losses — stakes reduced. "
            f"Expected variance for value betting, but monitoring."
        )
    else:
        result["status"] = "ok"
        result["message"] = f"Current loss streak: {current_streak}"

    return result


def check_bankroll_floor(settled: List[Dict], cfg: RiskConfig) -> Dict:
    """Check if bankroll has fallen below minimum threshold."""
    total_pnl = sum(b.get("profit", 0) or 0 for b in settled)
    current_bankroll = cfg.starting_bankroll + total_pnl
    floor = cfg.starting_bankroll * (cfg.min_bankroll_pct / 100)
    bankroll_pct = (current_bankroll / cfg.starting_bankroll) * 100

    result = {
        "current_bankroll": round(current_bankroll, 2),
        "bankroll_pct": round(bankroll_pct, 1),
        "floor": round(floor, 2),
    }

    if current_bankroll <= floor:
        result["status"] = "BLOCKED"
        result["message"] = (
            f"Bankroll €{current_bankroll:.0f} below floor €{floor:.0f} "
            f"({bankroll_pct:.0f}% of start). Hard stop."
        )
    else:
        result["status"] = "ok"
        result["message"] = f"Bankroll €{current_bankroll:.0f} ({bankroll_pct:.0f}% of start)"

    return result


def check_market_health(settled: List[Dict], cfg: RiskConfig) -> Dict:
    """Check per-market live performance and flag failing markets."""
    by_market = {}
    for bet in settled:
        mkt = bet.get("market", "?")
        # Normalize market names
        if "O/U" in mkt or "totals" in mkt:
            mkt_key = "O/U"
        elif mkt in ("1X2", "h2h"):
            mkt_key = "1X2"
        elif mkt == "DC":
            mkt_key = "DC"
        elif mkt == "DNB":
            mkt_key = "DNB"
        elif mkt == "BTTS":
            mkt_key = "BTTS"
        elif "AH" in mkt or "spread" in mkt.lower():
            mkt_key = "AH"
        else:
            mkt_key = mkt

        if mkt_key not in by_market:
            by_market[mkt_key] = {"bets": 0, "profit": 0, "staked": 0}
        by_market[mkt_key]["bets"] += 1
        by_market[mkt_key]["profit"] += bet.get("profit", 0) or 0
        by_market[mkt_key]["staked"] += bet.get("stake", 0) or 0

    kill_markets = []
    warn_markets = []

    for mkt, stats in by_market.items():
        if stats["bets"] < cfg.market_kill_min_bets:
            continue
        roi = (stats["profit"] / stats["staked"] * 100) if stats["staked"] > 0 else 0
        stats["roi"] = round(roi, 1)

        if roi < cfg.market_kill_roi_threshold:
            kill_markets.append(f"{mkt} ({stats['bets']} bets, {roi:+.1f}% ROI)")
        elif roi < -15:
            warn_markets.append(f"{mkt} ({stats['bets']} bets, {roi:+.1f}% ROI)")

    result = {
        "markets": {k: v for k, v in by_market.items()},
        "kill_markets": kill_markets,
        "warn_markets": warn_markets,
    }

    if kill_markets:
        result["status"] = "WARNING"
        result["message"] = f"Kill switch: {', '.join(kill_markets)}"
    else:
        result["status"] = "ok"
        result["message"] = "All markets within tolerance"

    return result


# =============================================================================
# MAIN GATE
# =============================================================================

def check_risk_gates(
    bankroll: float = 1000.0,
    cfg: RiskConfig = None,
) -> Dict:
    """Run all risk checks and return a unified verdict.

    Returns:
        {
            "allow_betting": bool,
            "reduce_stakes": bool,
            "stake_multiplier": float (1.0 = normal, 0.5 = reduced),
            "reason": str (if blocked),
            "checks": {drawdown, streak, bankroll, markets},
            "timestamp": str,
        }
    """
    if cfg is None:
        cfg = RiskConfig(starting_bankroll=bankroll)

    settled = _load_settled_bets()

    dd = check_drawdown(settled, cfg)
    streak = check_consecutive_losses(settled, cfg)
    floor = check_bankroll_floor(settled, cfg)
    markets = check_market_health(settled, cfg)

    checks = {
        "drawdown": dd,
        "consecutive_losses": streak,
        "bankroll_floor": floor,
        "market_health": markets,
    }

    # Determine overall verdict
    blocked = [k for k, v in checks.items() if v.get("status") == "BLOCKED"]
    warnings = [k for k, v in checks.items() if v.get("status") == "WARNING"]

    if blocked:
        allow = False
        reason = "; ".join(checks[k]["message"] for k in blocked)
        multiplier = 0.0
    elif warnings:
        allow = True
        reason = "; ".join(checks[k]["message"] for k in warnings)
        multiplier = cfg.stress_stake_multiplier
    else:
        allow = True
        reason = "All checks passed"
        multiplier = 1.0

    result = {
        "allow_betting": allow,
        "reduce_stakes": multiplier < 1.0,
        "stake_multiplier": multiplier,
        "reason": reason,
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
        "n_settled": len(settled),
    }

    # Save risk state for monitoring
    _save_risk_state(result)

    return result


def _save_risk_state(state: Dict):
    """Persist risk state for monitoring/dashboard."""
    path = DATA_DIR / "betting" / "risk_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        log.warning("Failed to save risk state: %s", e)


def print_risk_report(gates: Dict):
    """Print a formatted risk report."""
    print("=" * 60)
    print("RISK CONTROLS REPORT")
    print("=" * 60)

    status = "BETTING ALLOWED" if gates["allow_betting"] else "BETTING PAUSED"
    if gates["reduce_stakes"]:
        status = f"BETTING ALLOWED (stakes at {gates['stake_multiplier']:.0%})"
    print(f"\n  Status: {status}")
    print(f"  Reason: {gates['reason']}")
    print(f"  Settled bets: {gates['n_settled']}")

    checks = gates["checks"]

    print(f"\n  Drawdown:")
    dd = checks["drawdown"]
    print(f"    Current P&L:    €{dd.get('current_pnl', 0):+.2f}")
    print(f"    Peak P&L:       €{dd.get('peak_pnl', 0):+.2f}")
    print(f"    Max drawdown:   €{dd.get('max_drawdown', 0):.2f} ({dd.get('max_drawdown_pct', 0):.1f}%)")
    print(f"    Current DD:     {dd.get('current_drawdown_pct', 0):.1f}%")
    print(f"    [{dd.get('status', '?')}] {dd.get('message', '')}")

    print(f"\n  Loss streak:")
    ls = checks["consecutive_losses"]
    print(f"    Current streak: {ls.get('current_streak', 0)}")
    print(f"    Max streak:     {ls.get('max_streak', 0)}")
    print(f"    [{ls.get('status', '?')}] {ls.get('message', '')}")

    print(f"\n  Bankroll:")
    bf = checks["bankroll_floor"]
    print(f"    Current:  €{bf.get('current_bankroll', 0):.2f} ({bf.get('bankroll_pct', 0):.0f}%)")
    print(f"    Floor:    €{bf.get('floor', 0):.2f}")
    print(f"    [{bf.get('status', '?')}] {bf.get('message', '')}")

    print(f"\n  Market health:")
    mh = checks["market_health"]
    for mkt, stats in mh.get("markets", {}).items():
        roi = stats.get("roi", "N/A")
        print(f"    {mkt:8s}: {stats['bets']:2d} bets, €{stats['profit']:+7.2f}, ROI={roi}")
    if mh.get("kill_markets"):
        print(f"    KILL: {', '.join(mh['kill_markets'])}")
    if mh.get("warn_markets"):
        print(f"    WARN: {', '.join(mh['warn_markets'])}")
    print(f"    [{mh.get('status', '?')}] {mh.get('message', '')}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Risk controls for betting system")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    args = parser.parse_args()

    gates = check_risk_gates(bankroll=args.bankroll)
    print_risk_report(gates)
    print(f"\n  Result: {'ALLOW' if gates['allow_betting'] else 'BLOCK'}")
