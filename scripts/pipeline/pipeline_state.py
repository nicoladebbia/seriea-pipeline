#!/usr/bin/env python3
"""
Module: scripts/pipeline_state.py
Purpose: Persistent state tracker for incremental pipeline runs to avoid re-fetching/re-predicting already-processed matches
Inputs:  data/pipeline_state.json (previous run state), match results, fixture odds
Outputs: data/pipeline_state.json (updated with predicted_matches, settled_matches, last_run timestamps)
Called by: scheduler.py (incremental runs), scripts that need to check what's been processed
Depends on: config.settings (DATA_DIR)
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

log = logging.getLogger(__name__)

STATE_FILE = DATA_DIR / "pipeline_state.json"

# Default empty state
_DEFAULT_STATE = {
    "last_run": None,
    "last_odds_fetch": None,
    "last_results_fetch": None,
    "last_features_rebuild": None,
    "predicted_matches": [],       # ["Inter vs Juventus_2026-02-14", ...]
    "settled_matches": [],         # ["Genoa vs Napoli_2026-02-07", ...]
    "last_predictions_hash": None, # detect if predictions.json changed
}


def load_state() -> Dict:
    """Load pipeline state from disk. Returns default state if file missing."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            # Merge with defaults for any missing keys
            for key, default in _DEFAULT_STATE.items():
                if key not in state:
                    state[key] = default
            return state
        except Exception as e:
            log.warning(f"Failed to load pipeline state: {e}")
    return dict(_DEFAULT_STATE)


def save_state(state: Dict) -> Path:
    """Save pipeline state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    return STATE_FILE


def get_new_results(results: Dict[str, Dict], state: Dict) -> Dict[str, Dict]:
    """Filter results to only those not yet settled.

    Args:
        results: Dict mapping "Home vs Away" -> result data (from parse_scores)
        state: Current pipeline state

    Returns:
        Dict of results not in settled_matches
    """
    settled_set = set(state.get("settled_matches", []))
    new = {}
    for match_key, result in results.items():
        # Build state key: "Home vs Away_YYYY-MM-DD"
        date = result.get("commence_time", "")[:10]
        state_key = f"{match_key}_{date}" if date else match_key
        if state_key not in settled_set and match_key not in settled_set:
            new[match_key] = result
    return new


def get_new_fixtures(odds: Dict, state: Dict) -> Dict:
    """Filter fixtures to only those not yet predicted.

    Args:
        odds: Dict mapping "Home vs Away" -> odds data (from odds_fetcher)
        state: Current pipeline state

    Returns:
        Dict of fixtures not in predicted_matches
    """
    predicted_set = set(state.get("predicted_matches", []))
    new = {}
    for match_key, match_data in odds.items():
        # Check both with and without date suffix
        if match_key not in predicted_set:
            # Also check with date prefix
            commence = ""
            if isinstance(match_data, dict):
                commence = match_data.get("commence_time", "")[:10] if match_data.get("commence_time") else ""
            state_key = f"{match_key}_{commence}" if commence else match_key
            if state_key not in predicted_set:
                new[match_key] = match_data
    return new


def needs_odds_refresh(state: Dict, max_age_hours: float = 6.0) -> bool:
    """Check if odds data is stale and needs refreshing.

    Args:
        state: Current pipeline state
        max_age_hours: Maximum age before odds are considered stale

    Returns:
        True if odds should be re-fetched
    """
    last_fetch = state.get("last_odds_fetch")
    if not last_fetch:
        return True
    try:
        last_dt = datetime.fromisoformat(last_fetch)
        # Normalize to UTC-aware: last_odds_fetch is written tz-aware by
        # odds_fetcher.py but tz-naive by update_timestamp(); compare both
        # against a UTC-aware now so the subtraction never raises (the bug
        # that made this gate always return True → wake-storm credit burn).
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_dt
        return age > timedelta(hours=max_age_hours)
    except (ValueError, TypeError):
        return True


def mark_predicted(state: Dict, match_keys: List[str], dates: Optional[List[str]] = None) -> Dict:
    """Mark matches as predicted in state.

    Args:
        state: Current pipeline state
        match_keys: List of "Home vs Away" match keys
        dates: Optional list of date strings (same length as match_keys)

    Returns:
        Updated state
    """
    predicted = set(state.get("predicted_matches", []))
    for i, key in enumerate(match_keys):
        date = dates[i] if dates and i < len(dates) else ""
        state_key = f"{key}_{date}" if date else key
        predicted.add(state_key)
    state["predicted_matches"] = sorted(predicted)
    return state


def mark_settled(state: Dict, match_keys: List[str], dates: Optional[List[str]] = None) -> Dict:
    """Mark matches as settled in state.

    Args:
        state: Current pipeline state
        match_keys: List of "Home vs Away" match keys
        dates: Optional list of date strings

    Returns:
        Updated state
    """
    settled = set(state.get("settled_matches", []))
    for i, key in enumerate(match_keys):
        date = dates[i] if dates and i < len(dates) else ""
        state_key = f"{key}_{date}" if date else key
        settled.add(state_key)
    state["settled_matches"] = sorted(settled)
    return state


def update_timestamp(state: Dict, field: str) -> Dict:
    """Update a timestamp field in state to now.

    Args:
        state: Current pipeline state
        field: One of 'last_run', 'last_odds_fetch', 'last_results_fetch', 'last_features_rebuild'

    Returns:
        Updated state
    """
    # UTC-aware to match odds_fetcher.py's writer — all timestamps in this repo
    # are UTC-aware ISO strings (CLAUDE.md prevention rule).
    state[field] = datetime.now(timezone.utc).isoformat()
    return state


def get_state_summary(state: Dict) -> Dict:
    """Get a human-readable summary of pipeline state."""
    now = datetime.now(timezone.utc)

    def _age_str(iso_str):
        if not iso_str:
            return "never"
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = now - dt
            if delta.total_seconds() < 60:
                return "just now"
            if delta.total_seconds() < 3600:
                return f"{int(delta.total_seconds() / 60)}m ago"
            if delta.total_seconds() < 86400:
                return f"{delta.total_seconds() / 3600:.1f}h ago"
            return f"{delta.days}d ago"
        except (ValueError, TypeError):
            return "unknown"

    return {
        "last_run": _age_str(state.get("last_run")),
        "last_odds_fetch": _age_str(state.get("last_odds_fetch")),
        "last_results_fetch": _age_str(state.get("last_results_fetch")),
        "predicted_count": len(state.get("predicted_matches", [])),
        "settled_count": len(state.get("settled_matches", [])),
        "odds_stale": needs_odds_refresh(state),
    }
