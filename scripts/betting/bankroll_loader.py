#!/usr/bin/env python3
"""Bankroll Loader — single source of truth for current bankroll.

Computes balance from bet_journal.json (the journal IS the ledger).
Reads staking config from config/bankroll.yaml.

Usage:
    from scripts.betting.bankroll_loader import get_effective_bankroll
    bankroll = get_effective_bankroll()  # e.g. 1093.66
"""

import json
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "bankroll.yaml"
_JOURNAL_PATH = _PROJECT_ROOT / "data" / "betting" / "bet_journal.json"
_BANKROLL_JSON = _PROJECT_ROOT / "data" / "betting" / "bankroll.json"

# Hardcoded defaults (fallback if YAML missing)
_DEFAULTS = {
    "initial_balance": 1000.0,
    "kelly_fraction": 0.10,
    "max_stake_pct": 2.5,
    "min_stake_pct": 0.5,
    "max_drawdown_pct": 25.0,
    "warning_drawdown_pct": 15.0,
    "max_consecutive_losses": 8,
    "warning_consecutive_losses": 5,
    "recovery_wins_to_reset": 2,
    "min_bankroll_pct": 50.0,
    "stress_stake_multiplier": 0.5,
}


def load_bankroll_config() -> Dict:
    """Read config/bankroll.yaml, fall back to hardcoded defaults."""
    if _CONFIG_PATH.exists():
        try:
            import yaml
            with open(_CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
            merged = {**_DEFAULTS, **cfg}
            return merged
        except ImportError:
            log.debug("PyYAML not installed, using defaults")
        except Exception as e:
            log.warning("Failed to load bankroll.yaml: %s — using defaults", e)
    return dict(_DEFAULTS)


def compute_current_bankroll(config: Dict = None) -> Dict:
    """Compute balance from bet_journal.json.

    Formula: initial + sum(settled_profits) - pending_stakes

    Safety guards:
    - Returns initial_balance if journal missing
    - Returns initial_balance if balance negative
    - Returns initial_balance if balance > 10x initial (data error)

    Returns dict with current_balance, pending_stakes, initial_balance, etc.
    """
    if config is None:
        config = load_bankroll_config()

    initial = config["initial_balance"]

    if not _JOURNAL_PATH.exists():
        log.info("No bet journal found — using initial balance %.2f", initial)
        return {
            "initial_balance": initial,
            "current_balance": initial,
            "pending_stakes": 0.0,
            "settled_profit": 0.0,
            "peak_balance": initial,
        }

    try:
        with open(_JOURNAL_PATH) as f:
            journal = json.load(f)
    except Exception as e:
        log.warning("Failed to load journal: %s — using initial balance", e)
        return {
            "initial_balance": initial,
            "current_balance": initial,
            "pending_stakes": 0.0,
            "settled_profit": 0.0,
            "peak_balance": initial,
        }

    bets = journal.get("bets", {})

    # Sum settled profits
    settled_profit = 0.0
    peak_profit = 0.0
    running = 0.0
    for bet in bets.values():
        if not isinstance(bet, dict):
            continue
        status = bet.get("status", "")
        if status in ("won", "lost", "push", "void"):
            profit = bet.get("profit", 0) or 0
            running += profit
            peak_profit = max(peak_profit, running)
    settled_profit = running

    # Sum pending stakes
    pending_stakes = 0.0
    for bet in bets.values():
        if not isinstance(bet, dict):
            continue
        if bet.get("status") == "pending":
            pending_stakes += bet.get("stake", 0) or 0

    balance = initial + settled_profit - pending_stakes
    peak_balance = initial + peak_profit

    # Safety guards
    if balance <= 0 or balance > initial * 10:
        log.warning(
            "Computed balance %.2f looks wrong (initial=%.2f) — using initial",
            balance, initial,
        )
        balance = initial

    return {
        "initial_balance": initial,
        "current_balance": round(balance, 2),
        "pending_stakes": round(pending_stakes, 2),
        "settled_profit": round(settled_profit, 2),
        "peak_balance": round(peak_balance, 2),
    }


def get_effective_bankroll() -> float:
    """High-level: load config + compute balance, return the number for stake sizing."""
    config = load_bankroll_config()
    info = compute_current_bankroll(config)
    balance = info["current_balance"]
    log.info("Effective bankroll: €%.2f (initial: €%.2f, P&L: %+.2f, pending: €%.2f)",
             balance, info["initial_balance"], info["settled_profit"], info["pending_stakes"])
    return balance


def update_bankroll_json(balance_info: Dict = None):
    """Write derived state to data/betting/bankroll.json for dashboard consumers.

    This is a DERIVED file — the journal is the source of truth.
    """
    if balance_info is None:
        balance_info = compute_current_bankroll()

    from datetime import datetime

    # Count pending bets from journal
    pending_count = 0
    if _JOURNAL_PATH.exists():
        try:
            with open(_JOURNAL_PATH) as f:
                journal = json.load(f)
            pending_count = sum(
                1 for b in journal.get("bets", {}).values()
                if isinstance(b, dict) and b.get("status") == "pending"
            )
        except Exception:
            pass

    bankroll_data = {
        "initial_balance": balance_info["initial_balance"],
        "current_balance": balance_info["current_balance"],
        "peak_balance": balance_info["peak_balance"],
        "lowest_balance": balance_info["initial_balance"],  # Not tracked here; kept for compat
        "pending_bets": pending_count,
        "pending_stakes": balance_info["pending_stakes"],
        "updated_at": datetime.now().isoformat(),
    }

    _BANKROLL_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_BANKROLL_JSON, "w") as f:
        json.dump(bankroll_data, f, indent=2)

    log.info("Updated bankroll.json: €%.2f", balance_info["current_balance"])
