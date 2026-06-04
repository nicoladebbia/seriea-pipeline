"""Loader for `config/comparative_markets.json`.

Centralizes the validated model coefficients for the comparative team-stat
markets, the referee-aware total markets, and the odds-comparison engine — so
they live in one editable place instead of inline literals. These are model
HYPERPARAMETERS (validated on held-out data), not data: base rates are computed
live elsewhere. Mirrors the config/betting_rules.py loading pattern.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "comparative_markets.json"


@lru_cache(maxsize=1)
def load() -> dict:
    """Load and cache the comparative-markets config."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def comparative_stats() -> dict:
    return load().get("comparative_stats", {})


def referee_total_markets() -> dict:
    return load().get("referee_total_markets", {})


def odds_comparison() -> dict:
    return load().get("odds_comparison", {})


def market_trust() -> dict:
    return load().get("market_trust", {})


def best_bets_cfg() -> dict:
    return load().get("best_bets", {})
