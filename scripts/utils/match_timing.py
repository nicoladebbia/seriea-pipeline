#!/usr/bin/env python3
"""Match timing classification and live-match filtering.

Classifies matches into timing windows based on commence_time and filters
out live/completed matches from odds processing.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

# Match window thresholds (hours relative to kickoff)
WINDOW_FAR = 6.0         # >6h = far
WINDOW_APPROACHING = 1.0  # 1-6h = approaching
WINDOW_IMMINENT = 0.0     # 0-1h = imminent (lineups available)
WINDOW_LIVE_END = -2.0    # -2h to 0 = likely live
# < -2h = completed


def classify_match_window(
    commence_time_iso: str,
    now: Optional[datetime] = None,
) -> str:
    """Classify a match into a timing window.

    Args:
        commence_time_iso: ISO 8601 timestamp (e.g. "2026-02-07T19:45:00Z")
        now: Override current time (for testing)

    Returns:
        One of: "far", "approaching", "imminent", "live", "completed"
    """
    if now is None:
        now = datetime.now(timezone.utc)

    try:
        commence_dt = datetime.fromisoformat(
            commence_time_iso.replace("Z", "+00:00")
        )
    except (ValueError, AttributeError):
        log.warning(f"Invalid commence_time: {commence_time_iso}, treating as far")
        return "far"

    hours_until = (commence_dt - now).total_seconds() / 3600

    if hours_until > WINDOW_FAR:
        return "far"
    elif hours_until > WINDOW_APPROACHING:
        return "approaching"
    elif hours_until > WINDOW_IMMINENT:
        return "imminent"
    elif hours_until > WINDOW_LIVE_END:
        return "live"
    else:
        return "completed"


def get_hours_until_kickoff(
    commence_time_iso: str,
    now: Optional[datetime] = None,
) -> float:
    """Return hours until kickoff (negative = already started)."""
    if now is None:
        now = datetime.now(timezone.utc)
    try:
        commence_dt = datetime.fromisoformat(
            commence_time_iso.replace("Z", "+00:00")
        )
        return (commence_dt - now).total_seconds() / 3600
    except (ValueError, AttributeError):
        return 999.0


def filter_prematch_only(matches: Dict) -> Tuple[Dict, int]:
    """Filter out live and completed matches.

    Args:
        matches: Dict of match_key -> match_data, each with "commence_time"

    Returns:
        (filtered_matches, num_filtered)
    """
    filtered = {}
    num_filtered = 0

    for match_key, match_data in matches.items():
        commence = match_data.get("commence_time", "")
        window = classify_match_window(commence)

        # Add window classification to match data
        match_data["match_window"] = window

        if window in ("live", "completed"):
            hours = get_hours_until_kickoff(commence)
            log.warning(
                f"Filtering {window} match: {match_key} "
                f"(commence_time={commence}, {hours:+.1f}h)"
            )
            num_filtered += 1
        else:
            filtered[match_key] = match_data

    if num_filtered:
        log.info(
            f"Filtered {num_filtered} live/completed match(es), "
            f"{len(filtered)} pre-match remaining"
        )

    return filtered, num_filtered
