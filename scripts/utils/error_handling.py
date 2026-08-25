#!/usr/bin/env python3
"""
Module: scripts/error_handling.py
Purpose: Production-ready error management with data validation, API retry logic, graceful degradation, and health checks
Inputs:  Match data dictionaries (for validation), API responses (for retry logic)
Outputs: Validated data, retry-wrapped API calls, error logs, health check status
Called by: All scripts that need robust error handling (scrapers, prediction engine, betting system)
Depends on: config.settings (DATA_DIR)
"""

import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class DataValidationError(Exception):
    """Raised when data fails validation."""
    pass


class APIError(Exception):
    """Raised when an API call fails."""
    def __init__(self, message: str, status_code: int = None, retryable: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class FeatureUnavailableError(Exception):
    """Raised when a feature/data source is unavailable."""
    pass


# =============================================================================
# DATA VALIDATION
# =============================================================================

def _valid_team_names() -> set[str]:
    """Every alias that resolves to a club playing in an active league THIS season.

    This used to be a hand-typed 20-name Serie A set. It was wrong twice over:
    frozen on the previous season (no Venezia / Frosinone / Monza, still listing
    relegated clubs) and completely EPL-blind, so validation rejected 13 of the
    20 live fixtures — Crystal Palace and Man City included. Two of the repo's
    own documented traps at once: the season rollover and the league-parity
    rule.

    The obvious repair — ``set(TEAM_NAME_MAP)`` — over-corrects. That map is the
    union of FIVE leagues (406 entries, La Liga / Bundesliga / Ligue 1 included)
    plus every club either active league has ever fielded, so it validated
    "Real Madrid", "Bayern Munich" and "Blackpool" clean. This gate exists to
    catch rows that do not belong in an upcoming-fixture file at all, so the
    scope must be the CURRENT-SEASON rosters, with aliases folded in so
    "Liverpool FC" and "Coventry City" still resolve. It rolls over by itself
    when the two season constants are updated.

    Deliberately narrower than the map in one direction: a relegated club is
    rejected. That is correct here — the only caller, ``safe_load_matches``,
    validates UPCOMING fixtures. Do not reuse this on historical rows.
    """
    from config.team_names import (
        PREMIER_LEAGUE_2026_27,
        SERIE_A_2026_27,
        TEAM_NAME_MAP,
    )

    active = set(SERIE_A_2026_27) | set(PREMIER_LEAGUE_2026_27)
    return active | {k for k, v in TEAM_NAME_MAP.items() if v in active}


def validate_team_name(team: str) -> bool:
    """Validate that a team name resolves to a known club in an active league."""
    if not team or not isinstance(team, str):
        return False
    return team.strip() in _valid_team_names()


def validate_match_data(match: Dict) -> tuple[bool, List[str]]:
    """Validate match data structure and content.

    Args:
        match: Dictionary containing match data

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    required = ["home_team", "away_team"]
    for field in required:
        if field not in match:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Validate team names
    home = match.get("home_team", "")
    away = match.get("away_team", "")

    if not validate_team_name(home):
        errors.append(f"Invalid home team: '{home}'")

    if not validate_team_name(away):
        errors.append(f"Invalid away team: '{away}'")

    if home == away:
        errors.append(f"Home and away team are the same: '{home}'")

    # Validate date if present
    if "date" in match:
        try:
            date = match["date"]
            if isinstance(date, str):
                datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            errors.append(f"Invalid date format: '{match.get('date')}' (expected YYYY-MM-DD)")

    # Validate odds if present (sanity check: must be > 1.0 and < 100)
    odds_fields = ["home_odds", "draw_odds", "away_odds",
                   "avg_home_odds", "avg_draw_odds", "avg_away_odds"]
    for field in odds_fields:
        val = match.get(field)
        if val is not None:
            try:
                odds_val = float(val)
                if odds_val < 1.0:
                    errors.append(f"Odds below 1.0: {field}={odds_val}")
                elif odds_val > 100:
                    errors.append(f"Odds suspiciously high: {field}={odds_val}")
            except (TypeError, ValueError):
                errors.append(f"Non-numeric odds: {field}={val}")

    return len(errors) == 0, errors


def validate_probability(prob: float, name: str = "probability") -> tuple[bool, str]:
    """Validate that a probability is between 0 and 1."""
    if not isinstance(prob, (int, float)):
        return False, f"{name} must be a number, got {type(prob)}"

    if prob < 0 or prob > 1:
        return False, f"{name} must be between 0 and 1, got {prob}"

    return True, ""


def validate_odds(odds: float, name: str = "odds") -> tuple[bool, str]:
    """Validate that odds are reasonable (1.01 to 100.0)."""
    if not isinstance(odds, (int, float)):
        return False, f"{name} must be a number, got {type(odds)}"

    if odds < 1.01 or odds > 100:
        return False, f"{name} must be between 1.01 and 100, got {odds}"

    return True, ""


def validate_predictions(predictions: List[Dict]) -> tuple[bool, List[str]]:
    """Validate a list of predictions."""
    errors = []

    if not predictions:
        errors.append("Empty predictions list")
        return False, errors

    for i, pred in enumerate(predictions):
        # Check required fields
        if "match" not in pred and ("home_team" not in pred or "away_team" not in pred):
            errors.append(f"Prediction {i}: missing match identifier")

        # Check probabilities
        probs = pred.get("probabilities", {})
        if probs:
            total = probs.get("home", 0) + probs.get("draw", 0) + probs.get("away", 0)
            if abs(total - 1.0) > 0.05:  # Allow 5% tolerance
                errors.append(f"Prediction {i}: probabilities sum to {total:.3f}, expected ~1.0")

    return len(errors) == 0, errors


# =============================================================================
# API CALL HANDLING
# =============================================================================

def safe_api_call(
    func: Callable[..., T],
    *args,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: int = 30,
    default: T = None,
    **kwargs
) -> T:
    """Execute an API call with retries and error handling.

    Args:
        func: The function to call
        max_retries: Maximum number of retry attempts
        retry_delay: Seconds to wait between retries (doubles each retry)
        timeout: Request timeout in seconds
        default: Default value to return on failure
        *args, **kwargs: Arguments to pass to func

    Returns:
        Result of func or default value on failure
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            result = func(*args, timeout=timeout, **kwargs)
            return result

        except requests.exceptions.Timeout as e:
            last_error = e
            log.warning(f"API timeout (attempt {attempt + 1}/{max_retries}): {e}")

        except requests.exceptions.ConnectionError as e:
            last_error = e
            log.warning(f"API connection error (attempt {attempt + 1}/{max_retries}): {e}")

        except requests.exceptions.HTTPError as e:
            last_error = e
            status_code = e.response.status_code if e.response else None

            # Don't retry client errors (4xx) except 429 (rate limit)
            if status_code and 400 <= status_code < 500 and status_code != 429:
                log.error(f"API client error (not retrying): {status_code} - {e}")
                break

            log.warning(f"API HTTP error (attempt {attempt + 1}/{max_retries}): {status_code}")

        except Exception as e:
            last_error = e
            log.warning(f"API error (attempt {attempt + 1}/{max_retries}): {e}")

        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            wait_time = retry_delay * (2 ** attempt)
            log.info(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    log.error(f"API call failed after {max_retries} attempts: {last_error}")
    return default


def api_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for API calls with automatic retry."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return safe_api_call(func, *args, max_retries=max_retries, retry_delay=delay, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# GRACEFUL DEGRADATION
# =============================================================================

def with_fallback(
    primary: Callable[..., T],
    fallback: Callable[..., T],
    *args,
    feature_name: str = "feature",
    **kwargs
) -> tuple[T, bool]:
    """Try primary function, fall back to alternative if it fails.

    Args:
        primary: Primary function to try
        fallback: Fallback function if primary fails
        feature_name: Name for logging
        *args, **kwargs: Arguments to pass to both functions

    Returns:
        Tuple of (result, used_primary_flag)
    """
    try:
        result = primary(*args, **kwargs)
        if result is not None:
            return result, True
    except Exception as e:
        log.warning(f"{feature_name} primary failed: {e}")

    try:
        result = fallback(*args, **kwargs)
        log.info(f"{feature_name} using fallback data")
        return result, False
    except Exception as e:
        log.error(f"{feature_name} fallback also failed: {e}")
        return None, False


def get_cached_or_fetch(
    cache_path: Path,
    fetch_func: Callable[[], T],
    max_age_hours: int = 24,
    feature_name: str = "data"
) -> tuple[T, str]:
    """Get data from cache or fetch if stale/missing.

    Args:
        cache_path: Path to cached data file
        fetch_func: Function to fetch fresh data
        max_age_hours: Maximum cache age in hours
        feature_name: Name for logging

    Returns:
        Tuple of (data, source: "cache"|"fresh"|"stale_cache"|"empty")
    """
    import json
    from datetime import timedelta

    # Check cache
    if cache_path.exists():
        try:
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age = datetime.now() - mtime
            is_stale = age > timedelta(hours=max_age_hours)

            with open(cache_path) as f:
                cached_data = json.load(f)

            if not is_stale:
                log.info(f"{feature_name}: Using cached data ({age.seconds // 3600}h old)")
                return cached_data, "cache"

        except Exception as e:
            log.warning(f"{feature_name}: Cache read failed: {e}")

    # Try to fetch fresh data
    try:
        fresh_data = fetch_func()
        if fresh_data:
            # Save to cache
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(fresh_data, f, indent=2, default=str)
                log.info(f"{feature_name}: Fetched fresh data and cached")
            except Exception as e:
                log.warning(f"{feature_name}: Cache write failed: {e}")

            return fresh_data, "fresh"

    except Exception as e:
        log.warning(f"{feature_name}: Fetch failed: {e}")

    # Fall back to stale cache if available
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                stale_data = json.load(f)
            log.warning(f"{feature_name}: Using stale cache (fetch failed)")
            return stale_data, "stale_cache"
        except Exception as e:
            log.warning(f"Failed to load stale cache for {feature_name}: {e}")

    log.error(f"{feature_name}: No data available")
    return None, "empty"


# =============================================================================
# FEATURE AVAILABILITY
# =============================================================================

class FeatureStatus:
    """Track availability of prediction features."""

    def __init__(self):
        self.status = {}
        self.last_check = {}

    def set_available(self, feature: str, available: bool, details: str = ""):
        """Set feature availability status."""
        self.status[feature] = {
            "available": available,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.last_check[feature] = datetime.now()

    def is_available(self, feature: str) -> bool:
        """Check if a feature is available."""
        return self.status.get(feature, {}).get("available", False)

    def get_summary(self) -> Dict:
        """Get summary of all feature statuses."""
        return {
            name: info["available"]
            for name, info in self.status.items()
        }

    def to_dict(self) -> Dict:
        """Export full status to dict."""
        return {
            "features": self.status,
            "last_updated": datetime.now().isoformat()
        }


# Global feature status tracker
feature_status = FeatureStatus()


def check_feature_availability() -> Dict:
    """Check availability of all prediction features."""
    features = {
        "weather": DATA_DIR / "upcoming" / "weather.json",
        "referee": DATA_DIR / "upcoming" / "referees.json",
        "form": DATA_DIR / "upcoming" / "current_form.json",
        "odds": DATA_DIR / "upcoming" / "odds.json",
        "predictions": DATA_DIR / "upcoming" / "predictions.json",
    }

    results = {}
    for name, path in features.items():
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                is_valid = bool(data)
                feature_status.set_available(name, is_valid, f"Loaded from {path.name}")
                results[name] = is_valid
            except Exception as e:
                feature_status.set_available(name, False, str(e))
                results[name] = False
        else:
            feature_status.set_available(name, False, "File not found")
            results[name] = False

    return results


# =============================================================================
# HEALTH CHECKS
# =============================================================================

def health_check() -> Dict:
    """Perform system health check."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "issues": []
    }

    # Check data directory
    if DATA_DIR.exists():
        health["checks"]["data_dir"] = True
    else:
        health["checks"]["data_dir"] = False
        health["issues"].append("Data directory not found")

    # Check feature availability
    features = check_feature_availability()
    health["checks"]["features"] = features

    missing_critical = []
    if not features.get("form"):
        missing_critical.append("form")
    if not features.get("predictions"):
        missing_critical.append("predictions")

    if missing_critical:
        health["status"] = "degraded"
        health["issues"].append(f"Missing critical features: {missing_critical}")

    # Check model files
    model_path = DATA_DIR / "models" / "universal" / "model_h2h.cbm"
    if model_path.exists():
        health["checks"]["model"] = True
    else:
        health["checks"]["model"] = False
        health["issues"].append("Prediction model not found")

    if health["issues"]:
        if health["status"] == "healthy":
            health["status"] = "warning"

    return health


# =============================================================================
# SAFE WRAPPERS FOR PIPELINE COMPONENTS
# =============================================================================

def safe_load_matches() -> List[Dict]:
    """Safely load UPCOMING matches with validation.

    "Safe" has to include fresh. This read manual_matches.json with no date
    filter, so once the Odds API key lapsed it returned the previous season's
    final matchweek forever — validated, well-formed, and three months stale.
    Nothing in production calls this today, which is exactly why it needed
    fixing: a wrapper named ``safe_*`` is the one a future caller trusts
    without reading. Fresh source first, manual file only as a date-filtered
    fallback, and a stale source degrades to EMPTY rather than to wrong.
    """
    from scripts.utils.match_timing import _is_future, _load_sofascore_fixtures

    now = datetime.now(timezone.utc)
    matches = []

    for match in _load_sofascore_fixtures(now):
        is_valid, errors = validate_match_data(match)
        if is_valid:
            matches.append(match)
        else:
            log.warning(f"Invalid match data: {errors}")
    if matches:
        return matches

    manual_path = DATA_DIR / "upcoming" / "manual_matches.json"
    if manual_path.exists():
        try:
            with open(manual_path) as f:
                data = json.load(f)
            raw_matches = [m for m in (data.get("matches") or [])
                           if _is_future(m, now)]

            for match in raw_matches:
                is_valid, errors = validate_match_data(match)
                if is_valid:
                    matches.append(match)
                else:
                    log.warning(f"Invalid match data: {errors}")

        except json.JSONDecodeError as e:
            log.error(f"Invalid JSON in manual_matches.json: {e}")
        except Exception as e:
            log.error(f"Error loading matches: {e}")

    if not matches:
        log.warning("No valid matches loaded - using empty list")

    return matches


def safe_load_predictions() -> List[Dict]:
    """Safely load predictions with validation."""
    predictions_path = DATA_DIR / "upcoming" / "predictions.json"

    if not predictions_path.exists():
        log.warning("predictions.json not found")
        return []

    try:
        with open(predictions_path) as f:
            data = json.load(f)

        predictions = data.get("predictions", data) if isinstance(data, dict) else data

        is_valid, errors = validate_predictions(predictions)
        if not is_valid:
            log.warning(f"Prediction validation warnings: {errors}")

        return predictions

    except json.JSONDecodeError as e:
        log.error(f"Invalid JSON in predictions.json: {e}")
        return []
    except Exception as e:
        log.error(f"Error loading predictions: {e}")
        return []


if __name__ == "__main__":
    print("=" * 60)
    print("SYSTEM HEALTH CHECK")
    print("=" * 60)

    health = health_check()
    print(f"\nStatus: {health['status'].upper()}")
    print(f"\nFeature Availability:")
    for feature, available in health["checks"].get("features", {}).items():
        status = "[OK]" if available else "[MISSING]"
        print(f"  {status} {feature}")

    if health["issues"]:
        print(f"\nIssues:")
        for issue in health["issues"]:
            print(f"  - {issue}")
