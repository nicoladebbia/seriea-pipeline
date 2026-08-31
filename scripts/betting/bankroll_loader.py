#!/usr/bin/env python3
"""Bankroll Loader — staking config + thin shim over scripts.betting.ledger.

Balance numbers come from ledger.get_metrics() (bet_journal.json is the
truth). This module only adds config/bankroll.yaml staking parameters.

Usage:
    from scripts.betting.bankroll_loader import get_effective_bankroll
    bankroll = get_effective_bankroll()  # e.g. 1093.66
"""

import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "bankroll.yaml"

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
    """Balance numbers from ledger.get_metrics() — the one computation.

    Key names kept for existing callers (auto_settle, betting_unified).
    Kelly sizing uses available_balance (current minus pending stakes).
    """
    if config is None:
        config = load_bankroll_config()
    initial = config["initial_balance"]
    try:
        from scripts.betting.ledger import get_metrics
        mb = get_metrics(include_alerts=False)["bankroll"]
    except (ImportError, OSError, ValueError, KeyError, TypeError) as e:
        log.warning("ledger.get_metrics failed (%s) — using initial balance", e)
        return {
            "initial_balance": initial,
            "current_balance": initial,
            "available_balance": initial,
            "pending_stakes": 0.0,
            "settled_profit": 0.0,
            "peak_balance": initial,
        }
    return {
        "initial_balance": mb["initial"],
        "current_balance": mb["current"],
        "available_balance": mb["available"],
        "pending_stakes": mb["pending_stakes"],
        "settled_profit": round(mb["current"] - mb["initial"], 2),
        "peak_balance": mb["peak"],
    }


def get_effective_bankroll() -> float:
    """High-level: load config + compute balance, return the number for stake sizing.

    Returns available_balance (current minus pending stakes) so that new bets
    are sized against capital not already committed.
    """
    config = load_bankroll_config()
    info = compute_current_bankroll(config)
    balance = info["available_balance"]
    log.info("Effective bankroll: €%.2f (current: €%.2f, pending: €%.2f, P&L: %+.2f)",
             balance, info["current_balance"], info["pending_stakes"], info["settled_profit"])
    return balance


def update_bankroll_json(balance_info: Dict = None):
    """Regenerate bankroll.json via ledger.rebuild_caches() — the one writer.

    `balance_info` is ignored (was a source of drift — callers could pass
    stale values that overwrote correct ones).
    """
    try:
        from scripts.betting import ledger
        result = ledger.rebuild_caches()
        log.info("Updated bankroll.json via ledger: €%.2f",
                 result["bankroll"]["current_balance"])
    except Exception as e:
        log.error("ledger.rebuild_caches failed: %s — bankroll.json NOT updated", e)
