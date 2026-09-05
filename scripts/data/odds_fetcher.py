#!/usr/bin/env python3
"""ADVANCED MULTI-MARKET ODDS FETCHER - Phase 1 Implementation

Fetches REAL betting odds from actual bookmakers using The Odds API.
Supports multiple markets: 1X2, Over/Under, Point Spreads (Handicaps)

Features:
- Multi-market fetching (h2h, totals, spreads)
- Smart caching to reduce API calls
- Rate limiting (100K plan = 3,333/day)
- Data validation and cleaning
- Usage monitoring and tracking
- Error handling with retry logic

API: The Odds API (https://the-odds-api.com/)
- 100K plan: $59/month, 100,000 requests/month
- Covers Serie A and all major leagues
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass  # dotenv not required if env vars set externally

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "https://api.the-odds-api.com/v4"
SERIE_A_KEY = "soccer_italy_serie_a"
REGIONS = "eu"  # EU only — Nicola bets on CISL (Italian books), which mirror EU pricing.
# The Odds API charges credits = (# regions) × (# markets). Keeping regions=1
# collapses the per-call cost to (# markets). Do NOT expand without a quota plan.

# League key mapping for The Odds API
ODDS_API_LEAGUE_KEYS: Dict[str, str] = {
    "serie_a": "soccer_italy_serie_a",
    "premier_league": "soccer_epl",
    "la_liga": "soccer_spain_la_liga",
    "bundesliga": "soccer_germany_bundesliga",
    "ligue_1": "soccer_france_ligue_one",
}


def _resolve_sport_key(league: str = "serie_a") -> str:
    """Resolve a league identifier to an Odds API sport key."""
    key = ODDS_API_LEAGUE_KEYS.get(league)
    if not key:
        raise ValueError(
            f"Unknown league '{league}'. Valid: {', '.join(ODDS_API_LEAGUE_KEYS)}"
        )
    return key
ODDS_FORMAT = "decimal"

# Markets to fetch (core = always available, extra = available closer to kickoff)
MARKETS = {
    "h2h": "1X2 Match Winner",
    "totals": "Over/Under Goals",
    "spreads": "Point Spread/Handicap",
}

EXTRA_MARKETS = {
    # btts and draw_no_bet require per-event endpoint (not bulk /odds/)
    # Fetching them here returns 422 "Invalid market". Skip to avoid
    # wasted API calls. BTTS model calculates its own probabilities.
}

# Per-event markets (only available via /events/{id}/odds endpoint)
PER_EVENT_MARKETS = {
    "btts": "Both Teams To Score",
    "double_chance": "Double Chance",
    "team_totals": "Team Totals",
    "alternate_totals": "Alternate Totals",
    "draw_no_bet": "Draw No Bet",
    "alternate_spreads": "Alternate Spreads",
}

# Cache settings
CACHE_DIR = DATA_DIR / "cache"
CACHE_DURATION_MINUTES = 60  # Per-event extras (btts, alt_totals, etc.) don't move
# enough pre-kickoff to justify a 10-min refresh; this is layer 1 of the wake-storm
# fix in CLAUDE.md ("Daily Odds API spend spikes after Mac wake/launchctl reload"),
# where a duplicate pipeline run re-fetched identical extras 30-40 min apart.
# Layer 2 (use_cache=True on the non-critical run_full_pipeline callers) shipped;
# this line was stashed and lost, so the catalogue claimed a fix that was not live.
# SAFE for closing lines: fetch_tagged_snapshot() passes use_cache=False, and the
# cache is gated on use_cache — NOT on critical — so T-5min snapshots never read it.

# Rate limiting — monthly quota is the binding constraint, not daily
MONTHLY_LIMIT = 100000  # The Odds API 100K plan ($59/mo)
DAILY_LIMIT = 5000      # Fallback guard; real budget is enforced via remaining-credits tiers
CALLS_PER_MINUTE = 10   # Conservative rate limit

# Circuit-breaker thresholds (checked against the API's own x-requests-remaining header)
# These apply to non-critical fetches; callers can pass critical=True to override the
# upper tier only. Below HARD_STOP_REMAINING, nothing goes through — full stop.
SOFT_STOP_REMAINING = 2000   # Block historical + player props + extra-markets refresh
HARD_STOP_REMAINING = 500    # Block everything except explicit critical=True (T-2h final snapshot)

# Usage tracking file
USAGE_FILE = DATA_DIR / "api_usage.json"

# Team name normalization — uses the central system (400+ mappings, all leagues)
from config.team_names import normalize_team as _central_normalize_team


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================

from config.api_keys import get_odds_api_key

API_KEY = get_odds_api_key()


def check_api_key() -> bool:
    """Check if API key is configured."""
    if not API_KEY:
        log.error("ODDS_API_KEY not set!")
        log.error("Get your API key at: https://the-odds-api.com/")
        return False
    return True


# =============================================================================
# USAGE TRACKING & RATE LIMITING
# =============================================================================

def _load_usage() -> Dict:
    """Load API usage tracking data."""
    if USAGE_FILE.exists():
        try:
            with open(USAGE_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load API usage tracking data: {e}")

    return {
        "total_calls": 0,            # Kept for back-compat; now counts real credits, not requests
        "request_count": 0,          # NEW: distinct HTTP requests (separate from credit cost)
        "daily_calls": {},           # Daily REAL credits (delta-derived)
        "monthly_calls": {},         # Monthly REAL credits (delta-derived)
        "undercount_events": 0,      # Times header-delta disagreed with caller's estimate
        "last_call": None,
        "remaining_credits": None,
        "history": []
    }


def _save_usage(usage: Dict):
    """Save API usage tracking data."""
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f, indent=2, default=str)


def track_api_call(
    credits_remaining: Optional[int] = None,
    estimated_cost: int = 1,
    endpoint: str = "",
) -> int:
    """Track an API call using the API's own header as source of truth.

    The Odds API bills `(# regions) × (# markets)` per request, so a naive
    `credits_used=1` undercounts by up to ~24×. This function computes the
    TRUE cost from the delta between the previous and current `x-requests-remaining`
    header values. The caller's `estimated_cost` is only used as a sanity check
    and as a fallback for the very first call (when no previous value exists).

    Args:
        credits_remaining: Value of `x-requests-remaining` header from the response.
            Required — if None, we can't derive true cost.
        estimated_cost: Caller's guess (regions × markets) for sanity-check logging.
        endpoint: Short label for the history entry ("h2h", "per_event_btts", etc.).

    Returns:
        The real credits consumed by this call (delta from header).
    """
    usage = _load_usage()

    today = datetime.now().strftime("%Y-%m-%d")
    month = datetime.now().strftime("%Y-%m")

    prev_remaining = usage.get("remaining_credits")

    if credits_remaining is None:
        # Header wasn't present or wasn't parsed — fall back to estimate.
        real_cost = estimated_cost
        log.warning(
            "track_api_call: no x-requests-remaining header; falling back to estimate=%d",
            estimated_cost,
        )
    elif prev_remaining is None:
        # First call in this usage file — can't compute a delta yet. Use estimate.
        real_cost = estimated_cost
    else:
        delta = prev_remaining - credits_remaining
        if delta < 0:
            # Quota reset (new month) — don't record a negative delta.
            real_cost = estimated_cost
            log.info(
                "track_api_call: remaining went UP (%d → %d) — likely monthly reset",
                prev_remaining, credits_remaining,
            )
        else:
            real_cost = delta
            if delta > estimated_cost:
                # True undercount: the API billed MORE than the caller estimated.
                # (delta < estimate is a harmless overestimate — do not count it.)
                usage["undercount_events"] = usage.get("undercount_events", 0) + 1
                log.warning(
                    "track_api_call: header delta=%d but caller estimated=%d on %s "
                    "(undercount events: %d)",
                    delta, estimated_cost, endpoint,
                    usage["undercount_events"],
                )

    usage["total_calls"] += real_cost
    usage["request_count"] = usage.get("request_count", 0) + 1
    usage["daily_calls"][today] = usage["daily_calls"].get(today, 0) + real_cost
    usage["monthly_calls"][month] = usage["monthly_calls"].get(month, 0) + real_cost
    usage["last_call"] = datetime.now().isoformat()

    if credits_remaining is not None:
        usage["remaining_credits"] = credits_remaining

    usage["history"].append({
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "credits_used": real_cost,
        "estimated": estimated_cost,
        "remaining": credits_remaining,
    })
    usage["history"] = usage["history"][-200:]  # Keep a bit more for debugging

    _save_usage(usage)
    return real_cost


def check_quota_budget(critical: bool = False) -> Tuple[bool, str]:
    """Circuit breaker against The Odds API monthly quota.

    Uses the authoritative `remaining_credits` value from the last API response.
    Tiers:
      - remaining >= SOFT_STOP_REMAINING: all calls allowed
      - HARD_STOP_REMAINING <= remaining < SOFT_STOP_REMAINING: only critical=True allowed
      - remaining < HARD_STOP_REMAINING: nothing allowed
      - remaining is None (never called yet this month): allow, we'll learn on first response

    Args:
        critical: True for mandatory calls (e.g. T-2h closing snapshot on a match we've bet).
                  False for refreshes, extra-markets sweeps, player props, historical backfill.

    Returns:
        (allowed, human-readable reason).
    """
    usage = _load_usage()
    remaining = usage.get("remaining_credits")

    if remaining is None:
        return True, "OK (no prior remaining value; first call of the cycle)"

    if remaining < HARD_STOP_REMAINING:
        return False, (
            f"HARD STOP: {remaining} credits remaining (< {HARD_STOP_REMAINING}). "
            f"Quota nearly exhausted. All fetches blocked until monthly reset."
        )

    if remaining < SOFT_STOP_REMAINING and not critical:
        return False, (
            f"SOFT STOP: {remaining} credits remaining (< {SOFT_STOP_REMAINING}). "
            f"Only critical=True fetches allowed until monthly reset."
        )

    return True, f"OK ({remaining} credits remaining)"


# Priority hierarchy for pacing-based throttling.
# Lower number = higher importance (kept longer when budget tightens).
# Stages drop only when actual pacing exceeds the per-priority drop threshold.
PRIORITY_CRITICAL    = 1   # T-5m closing line — never drops (critical=True)
PRIORITY_EXTRAS      = 2   # T-6h / T-3h with extra markets — drops at 95% pacing
PRIORITY_PROPS       = 3   # Player props (per-event sweep) — drops at 90% pacing
PRIORITY_T24H        = 4   # T-24h h2h snapshot — drops at 80% pacing
PRIORITY_T72H        = 5   # T-72h opening line — drops at 70% pacing
PRIORITY_BACKFILL    = 6   # Historical backfill — drops at 60% pacing

# Drop threshold per priority. Stage is allowed iff (used/budget) <= (time_elapsed/days_in_month) + tolerance,
# where tolerance is per-priority. Higher tolerance = stage allowed to run further over expected pacing.
# Critical (1) has no threshold — bypasses controller entirely.
PRIORITY_TOLERANCE = {
    PRIORITY_EXTRAS:    0.02,   # allow 2% over time-expected pacing — small slack to absorb burst
    PRIORITY_PROPS:     0.00,   # exactly on pace
    PRIORITY_T24H:     -0.05,   # cut if pacing > time - 5%
    PRIORITY_T72H:     -0.10,
    PRIORITY_BACKFILL: -0.20,
}


def check_budget_pacing(priority: int = PRIORITY_EXTRAS, critical: bool = False) -> Tuple[bool, str]:
    """Adaptive controller: lands monthly spend at MONTHLY_LIMIT regardless of league count.

    Logic:
      pacing_ratio = (credits_used_this_month) / MONTHLY_LIMIT
      time_ratio   = (day_of_month) / days_in_month
      slack        = time_ratio - pacing_ratio
      slack > 0  → under-spending vs schedule (allow everything)
      slack < 0  → over-spending — drop priorities from the bottom up

    Each non-critical priority gets its own tolerance. Higher-priority stages
    survive longer over-pacing. Critical stages always pass.

    Self-balancing: as ACTIVE_LEAGUES grows from 2→3→5, the per-stage cost rises,
    pacing tightens earlier in the month, low-priority stages drop sooner — total
    spend converges to MONTHLY_LIMIT.
    """
    if critical:
        return True, "OK (critical=True bypasses pacing controller)"

    # Hard quota first — if circuit breaker says no, no pacing math saves us.
    ok, msg = check_quota_budget(critical=False)
    if not ok:
        return False, msg

    usage = _load_usage()
    now = datetime.now()
    month_key = now.strftime("%Y-%m")
    month_used = usage.get("monthly_calls", {}).get(month_key, 0)

    # Compute pacing
    import calendar as _cal
    days_in_month = _cal.monthrange(now.year, now.month)[1]
    time_ratio = now.day / days_in_month
    pacing_ratio = month_used / MONTHLY_LIMIT if MONTHLY_LIMIT > 0 else 0.0
    slack = time_ratio - pacing_ratio

    # Tolerance for this priority
    tol = PRIORITY_TOLERANCE.get(priority, 0.0)

    if slack >= -tol:
        return True, (
            f"OK pacing={pacing_ratio:.1%} time={time_ratio:.1%} slack={slack:+.1%} "
            f"priority={priority} tol={tol:+.1%}"
        )

    return False, (
        f"PACING DROP: priority={priority}, pacing={pacing_ratio:.1%} ahead of "
        f"time={time_ratio:.1%} (slack={slack:+.1%}, tol={tol:+.1%}). "
        f"Spent {month_used:,} of {MONTHLY_LIMIT:,} this month."
    )


def check_rate_limit(critical: bool = False) -> Tuple[bool, str]:
    """Check per-minute rate limit + daily cap + quota circuit breaker.

    Daily cap is a belt-and-braces guard; the real enforcement is the monthly
    quota checked via `check_quota_budget`.
    """
    # 1) Hard quota circuit breaker first — if we're out, no math can save us.
    ok, msg = check_quota_budget(critical=critical)
    if not ok:
        return False, msg

    usage = _load_usage()
    today = datetime.now().strftime("%Y-%m-%d")
    daily_used = usage["daily_calls"].get(today, 0)

    # 2) Daily guard (only matters if circuit breaker didn't already trip)
    if daily_used >= DAILY_LIMIT and not critical:
        return False, f"Daily limit reached ({daily_used}/{DAILY_LIMIT})"

    # 3) Per-minute courtesy throttle
    if usage["last_call"]:
        last_call = datetime.fromisoformat(usage["last_call"])
        seconds_since = (datetime.now() - last_call).total_seconds()
        if seconds_since < (60 / CALLS_PER_MINUTE):
            wait_time = (60 / CALLS_PER_MINUTE) - seconds_since
            time.sleep(wait_time)

    return True, f"OK ({daily_used}/{DAILY_LIMIT} today)"


def get_usage_summary() -> Dict:
    """Get API usage summary."""
    usage = _load_usage()
    today = datetime.now().strftime("%Y-%m-%d")
    month = datetime.now().strftime("%Y-%m")

    monthly_used = usage["monthly_calls"].get(month, 0)
    api_remaining = usage.get("remaining_credits")

    # Monthly reset = first day of next month, 00:00 UTC (Odds API convention)
    now = datetime.now(timezone.utc)
    if now.month == 12:
        reset = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        reset = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
    days_to_reset = max(0, (reset - now).days)

    return {
        "daily_used": usage["daily_calls"].get(today, 0),
        "daily_limit": DAILY_LIMIT,
        "daily_remaining": DAILY_LIMIT - usage["daily_calls"].get(today, 0),
        "monthly_used": monthly_used,
        "monthly_limit": MONTHLY_LIMIT,
        "monthly_remaining_tracked": MONTHLY_LIMIT - monthly_used,
        "api_remaining": api_remaining,  # Authoritative if present
        "request_count": usage.get("request_count", 0),
        "undercount_events": usage.get("undercount_events", 0),  # lifetime, debugging only
        # Recent TRUE undercounts (billed more than estimated) from the rolling
        # 200-entry history — this is what the dashboard badge should show, not
        # the lifetime counter, which never resets and goes stale.
        "undercount_recent": sum(
            1 for h in usage.get("history", [])
            if (h.get("credits_used") or 0) > (h.get("estimated") or 0)
        ),
        "last_call": usage.get("last_call"),
        "reset_at": reset.isoformat(),
        "days_to_reset": days_to_reset,
    }


# =============================================================================
# CACHING SYSTEM
# =============================================================================

def _get_cache_key(market: str, sport_key: str = SERIE_A_KEY) -> str:
    """Generate cache key for a market."""
    return hashlib.md5(f"{sport_key}_{market}_{REGIONS}".encode()).hexdigest()[:12]


def _get_cache_path(market: str, sport_key: str = SERIE_A_KEY) -> Path:
    """Get cache file path for a market."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"odds_{market}_{_get_cache_key(market, sport_key)}.json"


def _is_cache_valid(cache_path: Path) -> bool:
    """Check if cache is still valid."""
    if not cache_path.exists():
        return False

    try:
        with open(cache_path) as f:
            cache = json.load(f)

        cached_time = datetime.fromisoformat(cache.get("cached_at", "2000-01-01"))
        age_minutes = (datetime.now() - cached_time).total_seconds() / 60

        return age_minutes < CACHE_DURATION_MINUTES
    except Exception:
        return False


def _load_from_cache(market: str, sport_key: str = SERIE_A_KEY) -> Optional[List[Dict]]:
    """Load data from cache if valid."""
    cache_path = _get_cache_path(market, sport_key)

    if not _is_cache_valid(cache_path):
        return None

    try:
        with open(cache_path) as f:
            cache = json.load(f)
        log.info(f"Using cached data for {market} (age: {cache.get('age_minutes', '?')} min)")
        return cache.get("data", [])
    except Exception:
        return None


def _save_to_cache(market: str, data: List[Dict], sport_key: str = SERIE_A_KEY):
    """Save data to cache."""
    cache_path = _get_cache_path(market, sport_key)

    cache = {
        "cached_at": datetime.now().isoformat(),
        "market": market,
        "data": data
    }

    with open(cache_path, "w") as f:
        json.dump(cache, f)


def clear_cache():
    """Clear all cached data."""
    if CACHE_DIR.exists():
        for cache_file in CACHE_DIR.glob("odds_*.json"):
            cache_file.unlink()
        log.info("Cache cleared")


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_odds(odds: float) -> bool:
    """Validate that odds are reasonable."""
    return 1.01 <= odds <= 100.0


def validate_match_data(match: Dict) -> Tuple[bool, List[str]]:
    """Validate match data completeness and correctness."""
    errors = []

    if not match.get("home_team"):
        errors.append("Missing home team")
    if not match.get("away_team"):
        errors.append("Missing away team")
    if not match.get("commence_time"):
        errors.append("Missing commence time")

    # Validate h2h odds if present
    if match.get("h2h"):
        h2h = match["h2h"]
        for key in ["home", "draw", "away"]:
            if key in h2h and not validate_odds(h2h[key]):
                errors.append(f"Invalid {key} odds: {h2h.get(key)}")

    # Validate totals if present
    if match.get("totals"):
        for total in match["totals"]:
            if not validate_odds(total.get("over", 0)):
                errors.append(f"Invalid over odds: {total.get('over')}")
            if not validate_odds(total.get("under", 0)):
                errors.append(f"Invalid under odds: {total.get('under')}")

    return len(errors) == 0, errors


def detect_market_anomaly(match: Dict) -> Dict:
    """Detect anomalous market structures that suggest non-standard matches.

    Flags matches where the odds pattern is unusual for a standard Serie A
    league match (e.g., Coppa Italia second legs, or corrupted data).

    Returns dict with 'is_anomalous', 'reasons', 'severity'.
    """
    reasons = []
    severity = 0  # 0=normal, 1=warning, 2=reject

    h2h = match.get("h2h", {})
    home_odds = h2h.get("home", 0)
    draw_odds = h2h.get("draw", 0)
    away_odds = h2h.get("away", 0)

    if not (home_odds and draw_odds and away_odds):
        return {"is_anomalous": False, "reasons": [], "severity": 0}

    # Convert to implied probabilities (with vig)
    total = (1/home_odds + 1/draw_odds + 1/away_odds) if all([home_odds, draw_odds, away_odds]) else 0
    if total <= 0:
        return {"is_anomalous": True, "reasons": ["Cannot compute overround"], "severity": 2}

    home_imp = (1/home_odds) / total
    draw_imp = (1/draw_odds) / total
    away_imp = (1/away_odds) / total
    overround = (total - 1) * 100

    # 1. Draw is heavy favorite (>55%) — very unusual for a league match
    if draw_imp > 0.55:
        reasons.append(f"Draw is heavy favorite ({draw_imp:.0%}) — possible cup 2nd leg or special market")
        severity = max(severity, 2)

    # 2. Extreme overround (>20% or <1%) suggests non-standard market
    if overround > 20:
        reasons.append(f"Overround {overround:.1f}% is abnormally high")
        severity = max(severity, 1)
    elif overround < 1:
        reasons.append(f"Overround {overround:.1f}% is abnormally low")
        severity = max(severity, 1)

    # 3. Any single outcome with implied prob > 80% is unusual for Serie A
    if max(home_imp, draw_imp, away_imp) > 0.80:
        reasons.append(f"Single outcome has {max(home_imp, draw_imp, away_imp):.0%} implied prob")
        severity = max(severity, 1)

    # 4. All three outcomes very close to 33% but one is drastically different
    #    in the raw odds (e.g., 1.27 draw but 7+ home/away) = anomalous structure
    min_odds = min(home_odds, draw_odds, away_odds)
    max_odds = max(home_odds, draw_odds, away_odds)
    if max_odds / min_odds > 10:
        reasons.append(f"Odds ratio {max_odds/min_odds:.1f}x between outcomes — extreme disparity")
        severity = max(severity, 2)

    return {
        "is_anomalous": len(reasons) > 0,
        "reasons": reasons,
        "severity": severity,
        "implied_probs": {"home": round(home_imp, 3), "draw": round(draw_imp, 3), "away": round(away_imp, 3)},
        "overround": round(overround, 1),
    }


# =============================================================================
# API FETCHING
# =============================================================================

def normalize_team(name: str) -> str:
    """Normalize team name using the central mapping system."""
    return _central_normalize_team(name)


def fetch_market_odds(market: str, use_cache: bool = True, league: str = "serie_a") -> List[Dict]:
    """Fetch odds for a specific market.

    Args:
        market: One of 'h2h', 'totals', 'spreads'
        use_cache: Whether to use cached data if available
        league: League identifier (default: "serie_a").
                Valid values: serie_a, premier_league, la_liga, bundesliga, ligue_1

    Returns:
        List of match data with odds
    """
    sport_key = _resolve_sport_key(league)

    if not check_api_key():
        return []

    if not HAS_REQUESTS:
        log.error("requests library required: pip install requests")
        return []

    # Check cache first
    if use_cache:
        cached = _load_from_cache(market, sport_key)
        if cached:
            return cached

    # Check rate limit
    ok, msg = check_rate_limit()
    if not ok:
        log.error(f"Rate limit: {msg}")
        return []

    url = f"{API_BASE_URL}/sports/{sport_key}/odds/"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS,
        "markets": market,
        "oddsFormat": ODDS_FORMAT,
    }

    try:
        log.info(f"Fetching {MARKETS.get(market, market)} odds...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Track usage — cost is (# regions) × (# markets in this call) = len(REGIONS.split(',')) × 1
        remaining_hdr = response.headers.get("x-requests-remaining")
        remaining = int(remaining_hdr) if remaining_hdr is not None else None
        est_cost = len(REGIONS.split(",")) * 1
        real_cost = track_api_call(
            credits_remaining=remaining,
            estimated_cost=est_cost,
            endpoint=f"bulk_{market}_{sport_key}",
        )

        log.info(
            f"Fetched {len(data)} events | Cost: {real_cost} credits | "
            f"Remaining: {remaining if remaining is not None else '?'}"
        )

        # Cache the result
        _save_to_cache(market, data, sport_key)

        return data

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            log.error(
                "QUOTA EXHAUSTED or API key invalid (HTTP 401). "
                "The Odds API auto-disables keys when the monthly quota is reached. "
                "Check https://the-odds-api.com/account/ — reset at start of next month."
            )
        elif e.response.status_code == 429:
            log.error("Rate limit exceeded (HTTP 429) — backing off")
        elif e.response.status_code == 422:
            log.error(f"Invalid market: {market}")
        else:
            log.error(f"HTTP error: {e}")
        return []
    except Exception as e:
        log.error(f"Failed to fetch odds: {e}")
        return []


def fetch_all_markets(use_cache: bool = True, league: str = "serie_a") -> Dict[str, List[Dict]]:
    """Fetch odds for all supported markets.

    Args:
        use_cache: Whether to use cached data if available
        league: League identifier (default: "serie_a")

    Returns:
        Dict mapping market name to list of match data
    """
    all_data = {}

    for market in MARKETS.keys():
        data = fetch_market_odds(market, use_cache=use_cache, league=league)
        if data:
            all_data[market] = data
        time.sleep(0.5)  # Small delay between calls

    return all_data


def fetch_extra_markets_per_event(use_cache: bool = True, league: str = "serie_a") -> Dict[str, Dict]:
    """Fetch additional markets (btts, double_chance, team_totals, etc.) via
    the per-event endpoint which supports more markets than the bulk /odds endpoint.

    Args:
        use_cache: Whether to use cached data if available
        league: League identifier (default: "serie_a")

    Returns:
        Dict keyed by normalized "Home vs Away" with market odds per match.
    """
    sport_key = _resolve_sport_key(league)

    if not check_api_key():
        return {}
    if not HAS_REQUESTS:
        log.error("requests library required: pip install requests")
        return {}

    # Check cache
    cache_key = "extra_markets_per_event"
    if use_cache:
        cached = _load_from_cache(cache_key, sport_key)
        if cached:
            log.info(f"Using cached extra markets ({len(cached)} matches)")
            return cached

    # Step 1: Get event list
    ok, msg = check_rate_limit()
    if not ok:
        log.error(f"Rate limit: {msg}")
        return {}

    events_url = f"{API_BASE_URL}/sports/{sport_key}/events"
    try:
        resp = requests.get(events_url, params={"apiKey": API_KEY}, timeout=30)
        resp.raise_for_status()
        events = resp.json()
        # /events listing is billed 0 credits by The Odds API
        remaining_hdr = resp.headers.get("x-requests-remaining")
        remaining = int(remaining_hdr) if remaining_hdr is not None else None
        track_api_call(credits_remaining=remaining, estimated_cost=0,
                       endpoint=f"events_list_{sport_key}")
        log.info(f"Found {len(events)} upcoming events for extra markets")
    except Exception as e:
        log.error(f"Failed to fetch events list: {e}")
        return {}

    if not events:
        return {}

    # Step 2: For each event, fetch all per-event markets in one call
    markets_csv = ",".join(PER_EVENT_MARKETS.keys())
    all_extra = {}

    for event in events:
        event_id = event.get("id", "")
        home_raw = event.get("home_team", "")
        away_raw = event.get("away_team", "")
        home = normalize_team(home_raw)
        away = normalize_team(away_raw)
        match_key = f"{home} vs {away}"
        commence = event.get("commence_time", "")

        ok, msg = check_rate_limit()
        if not ok:
            log.warning(f"Rate limit hit after {len(all_extra)} events: {msg}")
            break

        try:
            url = f"{API_BASE_URL}/sports/{sport_key}/events/{event_id}/odds"
            resp = requests.get(url, params={
                "apiKey": API_KEY,
                "regions": REGIONS,
                "markets": markets_csv,
                "oddsFormat": ODDS_FORMAT,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            # Per-event endpoint is billed (# regions) × (# markets in request).
            # markets_csv currently contains len(PER_EVENT_MARKETS) keys.
            remaining_hdr = resp.headers.get("x-requests-remaining")
            remaining = int(remaining_hdr) if remaining_hdr is not None else None
            est_cost = len(REGIONS.split(",")) * len(markets_csv.split(","))
            track_api_call(
                credits_remaining=remaining,
                estimated_cost=est_cost,
                endpoint=f"per_event_{sport_key}",
            )
        except Exception as e:
            log.warning(f"Failed to fetch extra markets for {match_key}: {e}")
            time.sleep(0.3)
            continue

        # Parse all markets from this event
        match_markets = {
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
        }

        for bm in data.get("bookmakers", []):
            bookie_name = bm.get("title", "Unknown")
            for mkt in bm.get("markets", []):
                mkt_key = mkt.get("key", "")
                if mkt_key not in match_markets:
                    match_markets[mkt_key] = []

                outcomes = []
                for o in mkt.get("outcomes", []):
                    outcomes.append({
                        "name": o.get("name", ""),
                        "price": o.get("price", 0),
                        "point": o.get("point"),
                    })

                match_markets[mkt_key].append({
                    "bookmaker": bookie_name,
                    "outcomes": outcomes,
                })

        all_extra[match_key] = match_markets
        time.sleep(0.3)  # Rate limit courtesy

    log.info(f"Fetched extra markets for {len(all_extra)} events "
             f"[credits remaining: {remaining if events else '?'}]")

    # Cache results
    _save_to_cache(cache_key, all_extra, sport_key)

    return all_extra


SCORER_ODDS_FILE = Path(__file__).resolve().parents[2] / "data" / "fantacalcio" / "scorer_odds_raw.json"
SCORER_REFRESH_MIN = 45.0   # a per-event fetch younger than this is not repeated


def fetch_anytime_scorer_odds(hours_ahead: float = 1.25,
                              league: str = "serie_a") -> dict:
    """Fanta T-60 fetch: player_goal_scorer_anytime for events kicking off
    within `hours_ahead`. Probe-verified 2026-09-03: region eu carries the
    market (2 books, 41-45 players, FULL names in outcome `description`,
    name="Yes", decimal price), billed 1 credit per event (1 region x 1
    market). Budget ceiling: 10 SA events/round x 1 credit — negligible;
    the free /events listing gates everything else.

    Merge-writes data/fantacalcio/scorer_odds_raw.json keyed by event id;
    an event fetched < SCORER_REFRESH_MIN ago is skipped, so the scheduler
    can call this every cycle for free. All timestamps UTC ISO (display
    layers render local time — America/New_York on this machine)."""
    sport_key = _resolve_sport_key(league)
    if not check_api_key() or not HAS_REQUESTS:
        return {}
    store: dict = {"events": {}}
    try:
        store = json.loads(SCORER_ODDS_FILE.read_text())
    except (OSError, ValueError):
        pass
    ok, msg = check_rate_limit()
    if not ok:
        log.error(f"Scorer props: rate limit: {msg}")
        return store
    try:
        resp = requests.get(f"{API_BASE_URL}/sports/{sport_key}/events",
                            params={"apiKey": API_KEY}, timeout=30)
        resp.raise_for_status()
        events = resp.json()
        remaining_hdr = resp.headers.get("x-requests-remaining")
        track_api_call(credits_remaining=int(remaining_hdr) if remaining_hdr
                       else None, estimated_cost=0,
                       endpoint=f"events_list_{sport_key}")
    except Exception as e:
        log.error(f"Scorer props: events list failed: {e}")
        return store
    now = datetime.now(timezone.utc)  # noqa: UP017 — module style
    due = _scorer_events_due(events, store, now, hours_ahead)
    for ev in due:
        eid = ev["id"]
        try:
            r = requests.get(
                f"{API_BASE_URL}/sports/{sport_key}/events/{eid}/odds",
                params={"apiKey": API_KEY, "regions": REGIONS,
                        "markets": "player_goal_scorer_anytime",
                        "oddsFormat": ODDS_FORMAT}, timeout=30)
            r.raise_for_status()
            data = r.json()
            remaining_hdr = r.headers.get("x-requests-remaining")
            track_api_call(credits_remaining=int(remaining_hdr)
                           if remaining_hdr else None,
                           estimated_cost=len(REGIONS.split(",")),
                           endpoint=f"scorer_props_{sport_key}")
        except Exception as e:
            log.warning(f"Scorer props fetch failed for {eid}: {e}")
            continue
        prices: dict = {}
        for bm in data.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                if mkt.get("key") != "player_goal_scorer_anytime":
                    continue
                for o in mkt.get("outcomes", []):
                    if o.get("name") != "Yes" or not o.get("description"):
                        continue
                    prices.setdefault(o["description"], []).append(
                        float(o.get("price") or 0))
        store["events"][eid] = {
            "home": normalize_team(ev.get("home_team", "")),
            "away": normalize_team(ev.get("away_team", "")),
            "commence": ev.get("commence_time"),
            "fetched_at": now.isoformat(),
            "prices": prices,
        }
        log.info(f"Scorer props: {ev.get('home_team')} vs "
                 f"{ev.get('away_team')}: {len(prices)} players priced")
        time.sleep(0.3)
    if due:
        store["fetched_at"] = now.isoformat()
        SCORER_ODDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCORER_ODDS_FILE.write_text(
            json.dumps(store, indent=1, ensure_ascii=False))
    return store


def _scorer_events_due(events: list, store: dict, now, hours_ahead: float,
                       refresh_min: float = SCORER_REFRESH_MIN) -> list:
    """Pure selection: events kicking off within the window whose last fetch
    (if any) is older than `refresh_min` (default SCORER_REFRESH_MIN) and
    predates kickoff. Shared by the scorer fetch and fetch_pick_markets."""
    due = []
    for ev in events:
        try:
            ko = datetime.fromisoformat(
                ev["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError, AttributeError):
            continue
        if not (now - timedelta(minutes=15) <= ko
                <= now + timedelta(hours=hours_ahead)):
            continue
        prev = (store.get("events") or {}).get(ev.get("id"))
        if prev:
            try:
                age_min = (now - datetime.fromisoformat(
                    prev["fetched_at"])).total_seconds() / 60
                if age_min < refresh_min:
                    continue
            except (KeyError, ValueError):
                pass
        due.append(ev)
    return due


# ---- Pick markets: every per-event market the pick engine can price ----------
# Specimen-verified 2026-09-05 on Juventus vs AC Milan (region eu): every key
# below returned prices; alternate_team_totals / h2h_h2 / totals_h2 did NOT and
# are deliberately absent (a 422 on the whole request still costs credits).
# Billed 1 credit per market per event: len(PICK_EVENT_MARKETS) per event.
PICK_EVENT_MARKETS = (
    "h2h_h1", "totals_h1", "btts_h1", "halftime_fulltime", "double_chance_h1",
    "alternate_totals_corners", "alternate_totals_cards", "correct_score",
    "player_goal_scorer_anytime", "player_first_goal_scorer", "player_last_goal_scorer",
    "player_to_receive_card", "player_to_receive_red_card",
    "player_shots_on_target", "player_shots", "player_assists",
)
PICK_MARKETS_FILE = DATA_DIR / "upcoming" / "pick_markets_raw.json"
PICK_REFRESH_MIN = 45.0


def fetch_pick_markets(hours_ahead: float = 6.5, league: str = "serie_a",
                       refresh_min: float = PICK_REFRESH_MIN) -> dict:
    """Per-event odds for the pick engine (scripts/betting/picks.py): every
    market in PICK_EVENT_MARKETS for events kicking off within `hours_ahead`,
    RAW (bookmaker -> market -> outcomes, names verbatim) so the parser is
    testable against the saved specimen. Merge-writes PICK_MARKETS_FILE keyed
    by event id; an event fetched < `refresh_min` ago is skipped, so the T-6h,
    T-3h and every T-30 pre-kickoff cycle can call this and only the first of
    each window pays. Budget: len(PICK_EVENT_MARKETS) credits per event,
    gated by check_budget_pacing(PRIORITY_EXTRAS) — never critical."""
    sport_key = _resolve_sport_key(league)
    if not check_api_key() or not HAS_REQUESTS:
        return {}
    store: dict = {"events": {}}
    try:
        store = json.loads(PICK_MARKETS_FILE.read_text())
        store.setdefault("events", {})
    except (OSError, ValueError):
        pass
    ok, msg = check_budget_pacing(PRIORITY_EXTRAS)
    if not ok:
        log.warning(f"Pick markets: budget pacing blocked the fetch: {msg}")
        return store
    try:
        resp = requests.get(f"{API_BASE_URL}/sports/{sport_key}/events",
                            params={"apiKey": API_KEY}, timeout=30)
        resp.raise_for_status()
        events = resp.json()
        remaining_hdr = resp.headers.get("x-requests-remaining")
        track_api_call(credits_remaining=int(remaining_hdr) if remaining_hdr
                       else None, estimated_cost=0,
                       endpoint=f"events_list_{sport_key}")
    except Exception as e:
        log.error(f"Pick markets: events list failed: {e}")
        return store
    now = datetime.now(timezone.utc)  # noqa: UP017 — module style
    due = _scorer_events_due(events, store, now, hours_ahead, refresh_min)
    markets_csv = ",".join(PICK_EVENT_MARKETS)
    for ev in due:
        eid = ev["id"]
        ok, msg = check_rate_limit()
        if not ok:
            log.warning(f"Pick markets: rate limit after {eid}: {msg}")
            break
        try:
            r = requests.get(
                f"{API_BASE_URL}/sports/{sport_key}/events/{eid}/odds",
                params={"apiKey": API_KEY, "regions": REGIONS,
                        "markets": markets_csv, "oddsFormat": ODDS_FORMAT},
                timeout=30)
            r.raise_for_status()
            data = r.json()
            remaining_hdr = r.headers.get("x-requests-remaining")
            track_api_call(credits_remaining=int(remaining_hdr)
                           if remaining_hdr else None,
                           estimated_cost=len(PICK_EVENT_MARKETS)
                           * len(REGIONS.split(",")),
                           endpoint=f"pick_markets_{sport_key}")
        except Exception as e:
            log.warning(f"Pick markets fetch failed for {eid}: {e}")
            continue
        store["events"][eid] = {
            "home": normalize_team(ev.get("home_team", "")),
            "away": normalize_team(ev.get("away_team", "")),
            "home_raw": ev.get("home_team", ""),
            "away_raw": ev.get("away_team", ""),
            "commence": ev.get("commence_time"),
            "fetched_at": now.isoformat(),
            "bookmakers": [
                {"title": bm.get("title", "Unknown"),
                 "markets": [{"key": m.get("key"),
                              "outcomes": [{"name": o.get("name"),
                                            "description": o.get("description"),
                                            "point": o.get("point"),
                                            "price": o.get("price")}
                                           for o in m.get("outcomes", [])]}
                             for m in bm.get("markets", [])]}
                for bm in data.get("bookmakers", [])],
        }
        n_mk = len({m["key"] for bm in store["events"][eid]["bookmakers"]
                    for m in bm["markets"]})
        log.info(f"Pick markets: {ev.get('home_team')} vs {ev.get('away_team')}: "
                 f"{n_mk} markets priced")
        time.sleep(0.3)
    if due:
        store["fetched_at"] = now.isoformat()
        store["league"] = league
        PICK_MARKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PICK_MARKETS_FILE.write_text(
            json.dumps(store, indent=1, ensure_ascii=False))
    return store


def process_extra_markets(raw_extra: Dict[str, Dict]) -> Dict[str, Dict]:
    """Process per-event extra markets into a clean odds structure.

    Returns:
        Dict keyed by "Home vs Away" with processed odds for each market.
    """
    processed = {}

    for match_key, mdata in raw_extra.items():
        home = mdata.get("home_team", "")
        away = mdata.get("away_team", "")
        result = {
            "home_team": home,
            "away_team": away,
            "commence_time": mdata.get("commence_time", ""),
        }

        # --- BTTS ---
        btts_entries = mdata.get("btts", [])
        if btts_entries:
            yes_prices, no_prices = [], []
            for bm in btts_entries:
                for o in bm.get("outcomes", []):
                    if o["name"].lower() == "yes":
                        yes_prices.append(o["price"])
                    elif o["name"].lower() == "no":
                        no_prices.append(o["price"])
            if yes_prices:
                result["btts"] = {
                    "yes": round(sum(yes_prices) / len(yes_prices), 2),
                    "no": round(sum(no_prices) / len(no_prices), 2) if no_prices else 0,
                    "best_yes": round(max(yes_prices), 2),
                    "best_no": round(max(no_prices), 2) if no_prices else 0,
                    "bookmakers_count": len(yes_prices),
                }

        # --- Double Chance ---
        dc_entries = mdata.get("double_chance", [])
        if dc_entries:
            # Normalize API names ("AC Milan or Draw") → standard keys ("1X", "X2", "12")
            dc_odds = {"1X": [], "X2": [], "12": []}
            dc_bookmakers = {"1X": [], "X2": [], "12": []}
            home_lower = home.lower()
            away_lower = away.lower()
            for bm in dc_entries:
                bookie_name = bm.get("bookmaker", "Unknown")
                for o in bm.get("outcomes", []):
                    raw_name = o.get("name", "")
                    price = o.get("price", 0)
                    if price <= 1:
                        continue
                    name_lower = raw_name.lower()
                    has_home = home_lower in name_lower
                    has_away = away_lower in name_lower
                    has_draw = "draw" in name_lower
                    # Classify: "Home or Draw" = 1X, "Away or Draw" = X2, "Home or Away" = 12
                    if has_home and has_draw and not has_away:
                        dc_odds["1X"].append(price)
                        dc_bookmakers["1X"].append({"bookmaker": bookie_name, "odds": price})
                    elif has_away and has_draw and not has_home:
                        dc_odds["X2"].append(price)
                        dc_bookmakers["X2"].append({"bookmaker": bookie_name, "odds": price})
                    elif has_home and has_away and not has_draw:
                        dc_odds["12"].append(price)
                        dc_bookmakers["12"].append({"bookmaker": bookie_name, "odds": price})
            
            dc_result = {}
            for sel_key in ["1X", "X2", "12"]:
                prices = dc_odds[sel_key]
                if prices:
                    dc_result[sel_key] = {
                        "avg": round(sum(prices) / len(prices), 2),
                        "best": round(max(prices), 2),
                        "bookmakers_count": len(prices),
                        "all_bookmakers": dc_bookmakers[sel_key],
                    }
            if dc_result:
                result["double_chance"] = dc_result

        # --- Team Totals ---
        tt_entries = mdata.get("team_totals", [])
        if tt_entries:
            # Outcomes like "Over" with point=0.5, team indicated by name prefix
            tt = {"home": {}, "away": {}}
            for bm in tt_entries:
                for o in bm.get("outcomes", []):
                    name = o.get("name", "")  # e.g., "Over", "Under"
                    point = o.get("point")
                    price = o.get("price", 0)
                    if point is None:
                        continue
                    # The Odds API splits by description: first set = home, second = away
                    # But typically outcomes alternate: home Over, home Under, away Over, away Under
                    # We key by (name, point) and group
                    line_key = f"{name}_{point}"
                    tt["home"].setdefault(line_key, []).append(price)

            # The team_totals market has outcomes like:
            # "Over" point=0.5 (for home), "Under" point=0.5 (for home),
            # "Over" point=0.5 (for away), "Under" point=0.5 (for away)
            # Since ordering may vary, store all outcomes raw
            result["team_totals_raw"] = tt_entries

        # --- Alternate Totals ---
        at_entries = mdata.get("alternate_totals", [])
        if at_entries:
            # Group by line (point)
            lines = {}  # {0.5: {"over": [prices], "under": [prices]}, ...}
            for bm in at_entries:
                for o in bm.get("outcomes", []):
                    name = o.get("name", "").lower()  # "Over" or "Under"
                    point = o.get("point")
                    price = o.get("price", 0)
                    if point is None or price <= 1:
                        continue
                    lines.setdefault(point, {}).setdefault(name, []).append(price)

            result["alternate_totals"] = {}
            for line, sides in sorted(lines.items()):
                over_prices = sides.get("over", [])
                under_prices = sides.get("under", [])
                entry = {}
                if over_prices:
                    entry["over"] = round(sum(over_prices) / len(over_prices), 2)
                    entry["best_over"] = round(max(over_prices), 2)
                if under_prices:
                    entry["under"] = round(sum(under_prices) / len(under_prices), 2)
                    entry["best_under"] = round(max(under_prices), 2)
                entry["bookmakers_count"] = max(len(over_prices), len(under_prices))
                result["alternate_totals"][str(line)] = entry

        # --- Draw No Bet ---
        dnb_entries = mdata.get("draw_no_bet", [])
        if dnb_entries:
            home_prices, away_prices = [], []
            for bm in dnb_entries:
                for o in bm.get("outcomes", []):
                    norm = normalize_team(o.get("name", ""))
                    if norm == home:
                        home_prices.append(o["price"])
                    elif norm == away:
                        away_prices.append(o["price"])
            if home_prices:
                result["draw_no_bet"] = {
                    "home": round(sum(home_prices) / len(home_prices), 2),
                    "away": round(sum(away_prices) / len(away_prices), 2) if away_prices else 0,
                    "best_home": round(max(home_prices), 2),
                    "best_away": round(max(away_prices), 2) if away_prices else 0,
                    "bookmakers_count": len(home_prices),
                }

        # --- Alternate Spreads ---
        as_entries = mdata.get("alternate_spreads", [])
        if as_entries:
            spreads = {}  # {-0.5: {"home": [prices], "away": [prices]}, ...}
            for bm in as_entries:
                for o in bm.get("outcomes", []):
                    name = o.get("name", "")
                    point = o.get("point")
                    price = o.get("price", 0)
                    if point is None or price <= 1:
                        continue
                    norm = normalize_team(name)
                    side = "home" if norm == home else "away" if norm == away else None
                    if side:
                        spreads.setdefault(point, {}).setdefault(side, []).append(price)

            result["alternate_spreads"] = {}
            for line, sides in sorted(spreads.items()):
                entry = {"line": line}
                for side in ("home", "away"):
                    prices = sides.get(side, [])
                    if prices:
                        entry[side] = round(sum(prices) / len(prices), 2)
                        entry[f"best_{side}"] = round(max(prices), 2)
                entry["bookmakers_count"] = max(
                    len(sides.get("home", [])), len(sides.get("away", []))
                )
                result["alternate_spreads"][str(line)] = entry

        processed[match_key] = result

    return processed


def save_extra_markets(extra_data: Dict[str, Dict], league: str = "serie_a") -> Path:
    """Save processed extra markets to JSON."""
    output_dir = DATA_DIR / "upcoming"
    output_dir.mkdir(parents=True, exist_ok=True)
    _suffix = f"_{league}" if league != "serie_a" else ""
    output_path = output_dir / f"odds_extra_markets{_suffix}.json"

    with open(output_path, "w") as f:
        json.dump({
            "fetched_at": datetime.now().isoformat(),
            "source": "the-odds-api.com (per-event endpoint)",
            "markets": list(PER_EVENT_MARKETS.keys()),
            "match_count": len(extra_data),
            "matches": extra_data,
        }, f, indent=2)

    log.info(f"Saved extra markets for {len(extra_data)} matches to {output_path}")
    return output_path


# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_h2h_odds(raw_data: List[Dict]) -> Dict[str, Dict]:
    """Process 1X2 (head-to-head) odds."""
    processed = {}

    for event in raw_data:
        home_team = normalize_team(event.get("home_team", ""))
        away_team = normalize_team(event.get("away_team", ""))

        if not home_team or not away_team:
            continue

        match_key = f"{home_team} vs {away_team}"
        commence_time = event.get("commence_time", "")

        bookmaker_odds = []
        best_home, best_draw, best_away = 0, 0, 0

        for bookmaker in event.get("bookmakers", []):
            bookie_name = bookmaker.get("title", "Unknown")

            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                odds = {"bookmaker": bookie_name, "home": 0, "draw": 0, "away": 0}

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)

                    normalized_name = normalize_team(name)

                    if normalized_name == home_team:
                        odds["home"] = price
                        best_home = max(best_home, price)
                    elif normalized_name == away_team:
                        odds["away"] = price
                        best_away = max(best_away, price)
                    elif name.lower() == "draw":
                        odds["draw"] = price
                        best_draw = max(best_draw, price)

                if odds["home"] > 0:
                    bookmaker_odds.append(odds)

        if not bookmaker_odds:
            continue

        avg_home = sum(o["home"] for o in bookmaker_odds) / len(bookmaker_odds)
        avg_draw = sum(o["draw"] for o in bookmaker_odds) / len(bookmaker_odds)
        avg_away = sum(o["away"] for o in bookmaker_odds) / len(bookmaker_odds)

        processed[match_key] = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "home": round(avg_home, 2),
            "draw": round(avg_draw, 2),
            "away": round(avg_away, 2),
            "best_home": round(best_home, 2),
            "best_draw": round(best_draw, 2),
            "best_away": round(best_away, 2),
            "bookmakers_count": len(bookmaker_odds),
            "all_bookmakers": bookmaker_odds,
        }

    return processed


def process_totals_odds(raw_data: List[Dict]) -> Dict[str, Dict]:
    """Process Over/Under (totals) odds."""
    processed = {}

    for event in raw_data:
        home_team = normalize_team(event.get("home_team", ""))
        away_team = normalize_team(event.get("away_team", ""))

        if not home_team or not away_team:
            continue

        match_key = f"{home_team} vs {away_team}"
        commence_time = event.get("commence_time", "")

        # Collect totals by line (2.5, 3.5, etc.)
        totals_by_line = {}
        # Per-bookmaker entries for each line
        totals_bookmaker_entries = {}

        for bookmaker in event.get("bookmakers", []):
            bookie_name = bookmaker.get("title", "Unknown")

            for market in bookmaker.get("markets", []):
                if market.get("key") != "totals":
                    continue

                # Collect per-bookmaker over/under per line
                bookie_line_data = {}

                for outcome in market.get("outcomes", []):
                    point = outcome.get("point", 0)
                    name = outcome.get("name", "").lower()
                    price = outcome.get("price", 0)

                    if point not in totals_by_line:
                        totals_by_line[point] = {"over": [], "under": [], "bookmakers": []}

                    if name == "over":
                        totals_by_line[point]["over"].append(price)
                        bookie_line_data.setdefault(point, {})["over"] = price
                    elif name == "under":
                        totals_by_line[point]["under"].append(price)
                        bookie_line_data.setdefault(point, {})["under"] = price

                    if bookie_name not in totals_by_line[point]["bookmakers"]:
                        totals_by_line[point]["bookmakers"].append(bookie_name)

                # Store per-bookmaker entry
                for line, prices in bookie_line_data.items():
                    if "over" in prices and "under" in prices:
                        totals_bookmaker_entries.setdefault(line, []).append({
                            "bookmaker": bookie_name,
                            "over": prices["over"],
                            "under": prices["under"],
                        })

        if not totals_by_line:
            continue

        # Calculate averages and best odds for each line
        totals = []
        for line, data in sorted(totals_by_line.items()):
            if data["over"] and data["under"]:
                entry = {
                    "line": line,
                    "over": round(sum(data["over"]) / len(data["over"]), 2),
                    "under": round(sum(data["under"]) / len(data["under"]), 2),
                    "best_over": round(max(data["over"]), 2),
                    "best_under": round(max(data["under"]), 2),
                    "bookmakers_count": len(data["bookmakers"]),
                }
                if line in totals_bookmaker_entries:
                    entry["all_bookmakers"] = totals_bookmaker_entries[line]
                totals.append(entry)

        processed[match_key] = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "totals": totals
        }

    return processed


def process_spreads_odds(raw_data: List[Dict]) -> Dict[str, Dict]:
    """Process Point Spread/Handicap odds."""
    processed = {}

    for event in raw_data:
        home_team = normalize_team(event.get("home_team", ""))
        away_team = normalize_team(event.get("away_team", ""))

        if not home_team or not away_team:
            continue

        match_key = f"{home_team} vs {away_team}"
        commence_time = event.get("commence_time", "")

        # Collect spreads by line
        spreads_by_line = {}
        spreads_bookmaker_entries = {}

        for bookmaker in event.get("bookmakers", []):
            bookie_name = bookmaker.get("title", "Unknown")

            for market in bookmaker.get("markets", []):
                if market.get("key") != "spreads":
                    continue

                bookie_spread_data = {}

                for outcome in market.get("outcomes", []):
                    point = outcome.get("point", 0)
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)

                    normalized_name = normalize_team(name)

                    # Create unique key for this spread line
                    line_key = abs(point)
                    if line_key not in spreads_by_line:
                        spreads_by_line[line_key] = {
                            "home": [], "away": [],
                            "home_point": 0, "away_point": 0,
                            "bookmakers": []
                        }

                    if normalized_name == home_team:
                        spreads_by_line[line_key]["home"].append(price)
                        spreads_by_line[line_key]["home_point"] = point
                        bookie_spread_data.setdefault(line_key, {})["home"] = price
                        bookie_spread_data.setdefault(line_key, {})["home_point"] = point
                    elif normalized_name == away_team:
                        spreads_by_line[line_key]["away"].append(price)
                        spreads_by_line[line_key]["away_point"] = point
                        bookie_spread_data.setdefault(line_key, {})["away"] = price
                        bookie_spread_data.setdefault(line_key, {})["away_point"] = point

                    if bookie_name not in spreads_by_line[line_key]["bookmakers"]:
                        spreads_by_line[line_key]["bookmakers"].append(bookie_name)

                for lk, prices in bookie_spread_data.items():
                    if "home" in prices and "away" in prices:
                        spreads_bookmaker_entries.setdefault(lk, []).append({
                            "bookmaker": bookie_name,
                            "home": prices["home"],
                            "away": prices["away"],
                            "home_point": prices.get("home_point", 0),
                            "away_point": prices.get("away_point", 0),
                        })

        if not spreads_by_line:
            continue

        # Calculate averages
        spreads = []
        for line, data in sorted(spreads_by_line.items()):
            if data["home"] and data["away"]:
                entry = {
                    "line": line,
                    "home_handicap": data["home_point"],
                    "away_handicap": data["away_point"],
                    "home_odds": round(sum(data["home"]) / len(data["home"]), 2),
                    "away_odds": round(sum(data["away"]) / len(data["away"]), 2),
                    "best_home": round(max(data["home"]), 2),
                    "best_away": round(max(data["away"]), 2),
                    "bookmakers_count": len(data["bookmakers"]),
                }
                if line in spreads_bookmaker_entries:
                    entry["all_bookmakers"] = spreads_bookmaker_entries[line]
                spreads.append(entry)

        processed[match_key] = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "spreads": spreads
        }

    return processed


def process_btts_odds(raw_data: List[Dict]) -> Dict[str, Dict]:
    """Process Both Teams To Score odds."""
    processed = {}

    for event in raw_data:
        home_team = normalize_team(event.get("home_team", ""))
        away_team = normalize_team(event.get("away_team", ""))

        if not home_team or not away_team:
            continue

        match_key = f"{home_team} vs {away_team}"
        commence_time = event.get("commence_time", "")

        yes_odds_list = []
        no_odds_list = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "btts":
                    continue

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "").lower()
                    price = outcome.get("price", 0)

                    if name == "yes":
                        yes_odds_list.append(price)
                    elif name == "no":
                        no_odds_list.append(price)

        if not yes_odds_list:
            continue

        processed[match_key] = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "yes": round(sum(yes_odds_list) / len(yes_odds_list), 2),
            "no": round(sum(no_odds_list) / len(no_odds_list), 2) if no_odds_list else 0,
            "best_yes": round(max(yes_odds_list), 2),
            "best_no": round(max(no_odds_list), 2) if no_odds_list else 0,
            "bookmakers_count": len(yes_odds_list),
        }

    return processed


def process_dnb_odds(raw_data: List[Dict]) -> Dict[str, Dict]:
    """Process Draw No Bet odds."""
    processed = {}

    for event in raw_data:
        home_team = normalize_team(event.get("home_team", ""))
        away_team = normalize_team(event.get("away_team", ""))

        if not home_team or not away_team:
            continue

        match_key = f"{home_team} vs {away_team}"
        commence_time = event.get("commence_time", "")

        home_odds_list = []
        away_odds_list = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "draw_no_bet":
                    continue

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)
                    normalized_name = normalize_team(name)

                    if normalized_name == home_team:
                        home_odds_list.append(price)
                    elif normalized_name == away_team:
                        away_odds_list.append(price)

        if not home_odds_list:
            continue

        processed[match_key] = {
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
            "home": round(sum(home_odds_list) / len(home_odds_list), 2),
            "away": round(sum(away_odds_list) / len(away_odds_list), 2) if away_odds_list else 0,
            "best_home": round(max(home_odds_list), 2),
            "best_away": round(max(away_odds_list), 2) if away_odds_list else 0,
            "bookmakers_count": len(home_odds_list),
        }

    return processed


def process_all_markets(raw_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Process all markets into unified match data."""
    unified = {}

    # Process h2h first to establish base matches
    if "h2h" in raw_data:
        h2h_data = process_h2h_odds(raw_data["h2h"])
        for match_key, data in h2h_data.items():
            unified[match_key] = {
                "home_team": data["home_team"],
                "away_team": data["away_team"],
                "commence_time": data["commence_time"],
                "h2h": {
                    "home": data["home"],
                    "draw": data["draw"],
                    "away": data["away"],
                    "best_home": data["best_home"],
                    "best_draw": data["best_draw"],
                    "best_away": data["best_away"],
                    "bookmakers_count": data["bookmakers_count"],
                    "all_bookmakers": data.get("all_bookmakers", []),
                },
                "totals": [],
                "spreads": []
            }

    # Add totals
    if "totals" in raw_data:
        totals_data = process_totals_odds(raw_data["totals"])
        for match_key, data in totals_data.items():
            if match_key in unified:
                unified[match_key]["totals"] = data["totals"]
            else:
                unified[match_key] = {
                    "home_team": data["home_team"],
                    "away_team": data["away_team"],
                    "commence_time": data["commence_time"],
                    "h2h": None,
                    "totals": data["totals"],
                    "spreads": []
                }

    # Add spreads
    if "spreads" in raw_data:
        spreads_data = process_spreads_odds(raw_data["spreads"])
        for match_key, data in spreads_data.items():
            if match_key in unified:
                unified[match_key]["spreads"] = data["spreads"]
            else:
                unified[match_key] = {
                    "home_team": data["home_team"],
                    "away_team": data["away_team"],
                    "commence_time": data["commence_time"],
                    "h2h": None,
                    "totals": [],
                    "spreads": data["spreads"]
                }

    # Add btts
    if "btts" in raw_data:
        btts_data = process_btts_odds(raw_data["btts"])
        for match_key, data in btts_data.items():
            if match_key in unified:
                unified[match_key]["btts"] = {
                    "yes": data["yes"],
                    "no": data["no"],
                    "best_yes": data["best_yes"],
                    "best_no": data["best_no"],
                    "bookmakers_count": data["bookmakers_count"],
                }
            else:
                unified[match_key] = {
                    "home_team": data["home_team"],
                    "away_team": data["away_team"],
                    "commence_time": data["commence_time"],
                    "h2h": None, "totals": [], "spreads": [],
                    "btts": {"yes": data["yes"], "no": data["no"],
                             "best_yes": data["best_yes"], "best_no": data["best_no"],
                             "bookmakers_count": data["bookmakers_count"]},
                }

    # Add draw_no_bet
    if "draw_no_bet" in raw_data:
        dnb_data = process_dnb_odds(raw_data["draw_no_bet"])
        for match_key, data in dnb_data.items():
            if match_key in unified:
                unified[match_key]["draw_no_bet"] = {
                    "home": data["home"],
                    "away": data["away"],
                    "best_home": data["best_home"],
                    "best_away": data["best_away"],
                    "bookmakers_count": data["bookmakers_count"],
                }
            else:
                unified[match_key] = {
                    "home_team": data["home_team"],
                    "away_team": data["away_team"],
                    "commence_time": data["commence_time"],
                    "h2h": None, "totals": [], "spreads": [],
                    "draw_no_bet": {"home": data["home"], "away": data["away"],
                                    "best_home": data["best_home"], "best_away": data["best_away"],
                                    "bookmakers_count": data["bookmakers_count"]},
                }

    return unified


# =============================================================================
# SAVING DATA
# =============================================================================

def save_odds(odds_data: Dict[str, Dict], league: str = "serie_a") -> Dict[str, Path]:
    """Save odds to multiple JSON files. Non-serie_a leagues use suffixed filenames."""
    from scripts.utils.match_timing import filter_prematch_only

    # Filter out live and completed matches before saving
    odds_data, num_filtered = filter_prematch_only(odds_data)
    if num_filtered:
        log.info(f"Excluded {num_filtered} live/completed match(es) from odds output")

    output_paths = {}

    # Ensure output directory exists
    output_dir = DATA_DIR / "upcoming"
    output_dir.mkdir(parents=True, exist_ok=True)

    # League-specific suffix for non-default leagues (prevents overwriting Serie A data)
    _suffix = f"_{league}" if league != "serie_a" else ""

    # Save simple 1X2 odds (backwards compatible)
    simple_odds = {}
    for match_key, data in odds_data.items():
        if data.get("h2h"):
            simple_odds[match_key] = {
                "home": data["h2h"]["home"],
                "draw": data["h2h"]["draw"],
                "away": data["h2h"]["away"],
            }

    odds_path = output_dir / f"odds{_suffix}.json"
    with open(odds_path, "w") as f:
        json.dump(simple_odds, f, indent=2)
    output_paths["h2h"] = odds_path

    # Save over/under odds
    totals_odds = {}
    for match_key, data in odds_data.items():
        if data.get("totals"):
            totals_odds[match_key] = {
                "home_team": data["home_team"],
                "away_team": data["away_team"],
                "totals": data["totals"]
            }

    if totals_odds:
        totals_path = output_dir / f"odds_totals{_suffix}.json"
        with open(totals_path, "w") as f:
            json.dump(totals_odds, f, indent=2)
        output_paths["totals"] = totals_path

    # Save spreads odds
    spreads_odds = {}
    for match_key, data in odds_data.items():
        if data.get("spreads"):
            spreads_odds[match_key] = {
                "home_team": data["home_team"],
                "away_team": data["away_team"],
                "spreads": data["spreads"]
            }

    if spreads_odds:
        spreads_path = output_dir / f"odds_spreads{_suffix}.json"
        with open(spreads_path, "w") as f:
            json.dump(spreads_odds, f, indent=2)
        output_paths["spreads"] = spreads_path

    # Save BTTS odds
    btts_odds = {}
    for match_key, data in odds_data.items():
        if data.get("btts"):
            btts_odds[match_key] = {
                "home_team": data["home_team"],
                "away_team": data["away_team"],
                **data["btts"]
            }

    if btts_odds:
        btts_path = output_dir / f"odds_btts{_suffix}.json"
        with open(btts_path, "w") as f:
            json.dump(btts_odds, f, indent=2)
        output_paths["btts"] = btts_path

    # Save Draw No Bet odds
    dnb_odds = {}
    for match_key, data in odds_data.items():
        if data.get("draw_no_bet"):
            dnb_odds[match_key] = {
                "home_team": data["home_team"],
                "away_team": data["away_team"],
                **data["draw_no_bet"]
            }

    if dnb_odds:
        dnb_path = output_dir / f"odds_dnb{_suffix}.json"
        with open(dnb_path, "w") as f:
            json.dump(dnb_odds, f, indent=2)
        output_paths["draw_no_bet"] = dnb_path

    # Save full unified data
    full_path = output_dir / f"odds_full{_suffix}.json"
    with open(full_path, "w") as f:
        json.dump({
            "fetched_at": datetime.now().isoformat(),
            "source": "the-odds-api.com",
            "league": league,
            "markets": list(MARKETS.keys()) + list(EXTRA_MARKETS.keys()),
            "matches": odds_data
        }, f, indent=2)
    output_paths["full"] = full_path

    # Save per-bookmaker data (compact format for market intelligence)
    bookmaker_data = {}
    for match_key, data in odds_data.items():
        match_bookmakers = {}
        # H2H bookmakers
        h2h = data.get("h2h")
        if h2h and h2h.get("all_bookmakers"):
            match_bookmakers["h2h"] = h2h["all_bookmakers"]
        # Totals bookmakers (from the 2.5 line primarily)
        for total in data.get("totals", []):
            if total.get("all_bookmakers"):
                line_key = f"totals_{total['line']}"
                match_bookmakers[line_key] = total["all_bookmakers"]
        # Spreads bookmakers
        for spread in data.get("spreads", []):
            if spread.get("all_bookmakers"):
                line_key = f"spreads_{spread['line']}"
                match_bookmakers[line_key] = spread["all_bookmakers"]
        if match_bookmakers:
            bookmaker_data[match_key] = match_bookmakers

    if bookmaker_data:
        bookmaker_path = output_dir / f"odds_bookmakers{_suffix}.json"
        with open(bookmaker_path, "w") as f:
            json.dump({
                "fetched_at": datetime.now().isoformat(),
                "matches": bookmaker_data
            }, f, indent=2)
        output_paths["bookmakers"] = bookmaker_path

    log.info(f"Saved odds to {len(output_paths)} files")
    return output_paths


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def fetch_and_save_odds(markets: List[str] = None, use_cache: bool = True, league: str = "serie_a") -> Dict[str, Dict]:
    """Main function to fetch and save real odds.

    Args:
        markets: List of markets to fetch ('h2h', 'totals', 'spreads')
                If None, fetches all markets
        use_cache: Whether to use cached data
        league: League identifier (default: "serie_a").
                Valid values: serie_a, premier_league, la_liga, bundesliga, ligue_1

    Returns:
        Unified match data with all market odds
    """
    log.info("=" * 60)
    log.info("MULTI-MARKET ODDS FETCHER")
    log.info("=" * 60)

    # Show usage stats
    usage = get_usage_summary()
    log.info(f"API Usage: {usage['daily_used']}/{usage['daily_limit']} today, "
             f"{usage.get('api_remaining', '?')} credits remaining")

    if markets is None:
        markets = list(MARKETS.keys())

    # Fetch all requested markets
    raw_data = {}
    for market in markets:
        if market in MARKETS or market in EXTRA_MARKETS:
            data = fetch_market_odds(market, use_cache=use_cache, league=league)
            if data:
                raw_data[market] = data

    # Also try extra markets (btts, draw_no_bet) - available closer to kickoff
    for market in EXTRA_MARKETS:
        if market not in raw_data:
            data = fetch_market_odds(market, use_cache=use_cache, league=league)
            if data:
                raw_data[market] = data
                log.info(f"Extra market available: {EXTRA_MARKETS[market]}")

    if not raw_data:
        # Fallback: API returned nothing (quota hard-stopped, network issue, key rotated, etc.)
        # If we have a previously-saved odds_full*.json on disk, return it with a STALE warning
        # so the upstream data gate doesn't abort the whole pipeline. Bets generated against
        # this data must NOT auto-commit to the journal — handled upstream via
        # BETTING_DRY_RUN_FROM_MORNING (candidate-mode).
        try:
            from pathlib import Path
            output_dir = Path(__file__).resolve().parent.parent.parent / "data" / "upcoming"
            _suffix = "" if league == "serie_a" else f"_{league}"
            cache_path = output_dir / f"odds_full{_suffix}.json"
            if cache_path.exists():
                with open(cache_path) as fh:
                    cached = json.load(fh)
                fetched_at = cached.get("fetched_at", "?")
                matches = cached.get("matches", {})
                age_str = "?"
                try:
                    fetched_dt = datetime.fromisoformat(fetched_at)
                    age_hours = (datetime.now() - fetched_dt).total_seconds() / 3600
                    age_str = f"{age_hours:.1f}h"
                except Exception:
                    pass
                log.warning(
                    "STALE FALLBACK: API returned no data (likely quota hard-stop). "
                    "Loaded %d cached matches from %s (age: %s). "
                    "Upstream consumers MUST treat as candidate-only — do not commit bets at these prices.",
                    len(matches), cache_path.name, age_str,
                )
                return matches
        except Exception as e:
            log.warning("Stale fallback failed: %s", e)
        log.warning("No odds data received and no on-disk cache available")
        return {}

    # Process all markets
    unified_data = process_all_markets(raw_data)

    if not unified_data:
        log.warning("No matches processed")
        return {}

    log.info(f"Processed {len(unified_data)} matches with odds")

    # Validate data
    valid_count = 0
    for match_key, data in unified_data.items():
        is_valid, errors = validate_match_data(data)
        if is_valid:
            valid_count += 1
        else:
            log.warning(f"Validation errors for {match_key}: {errors}")

    log.info(f"Valid matches: {valid_count}/{len(unified_data)}")

    # Save (league-aware: non-serie_a writes to suffixed files)
    save_odds(unified_data, league=league)

    # Update pipeline_state so health-monitor sees a recent fetch.
    # Without this, the staleness check perpetually warns even when fetches succeed.
    try:
        from datetime import datetime as _dt, timezone as _tz
        state_path = DATA_DIR / "pipeline_state.json"
        state = {}
        if state_path.exists():
            with open(state_path) as fh:
                state = json.load(fh) or {}
        state["last_odds_fetch"] = _dt.now(_tz.utc).isoformat()
        with open(state_path, "w") as fh:
            json.dump(state, fh, indent=2)
    except Exception as e:
        log.debug(f"Failed to update last_odds_fetch state: {e}")

    # Fetch extra markets via per-event endpoint (btts, double_chance, team_totals, etc.)
    log.info("Fetching extra markets via per-event endpoint...")
    raw_extra = fetch_extra_markets_per_event(use_cache=use_cache, league=league)
    if raw_extra:
        extra_processed = process_extra_markets(raw_extra)
        save_extra_markets(extra_processed, league=league)
    else:
        log.warning("No extra markets data from per-event endpoint")

    return unified_data


def sync_matches_from_odds(odds_data: Dict[str, Dict]) -> List[Dict]:
    """Auto-populate upcoming matches from Odds API data.

    Extracts home_team, away_team, commence_time from odds response
    and writes to manual_matches.json, preserving manually-entered
    fields (venue, matchweek, referee).

    Args:
        odds_data: Unified odds data from fetch_and_save_odds()

    Returns:
        List of match dicts written to file
    """
    manual_path = DATA_DIR / "upcoming" / "manual_matches.json"

    # Load existing manual matches to preserve extra fields
    existing_matches = {}
    matchweek = 0
    season = ""
    if manual_path.exists():
        try:
            with open(manual_path) as f:
                manual_data = json.load(f)
            for m in manual_data.get("matches", []):
                key = f"{m.get('home_team', '')}_{m.get('away_team', '')}"
                existing_matches[key] = m
            matchweek = manual_data.get("matchweek", 0)
            season = manual_data.get("season", "")
        except Exception as e:
            log.warning(f"Failed to load manual fixtures data: {e}")

    synced_matches = []
    for match_key, data in odds_data.items():
        home = data.get("home_team", "")
        away = data.get("away_team", "")
        if not home or not away:
            continue

        key = f"{home}_{away}"

        # Parse commence_time
        commence = data.get("commence_time", "")
        match_date = ""
        match_time = ""
        if commence:
            try:
                dt_utc = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                # Convert UTC to Miami (America/New_York) time
                dt_miami = dt_utc.astimezone(ZoneInfo("America/New_York"))
                match_date = dt_miami.strftime("%Y-%m-%d")
                match_time = dt_miami.strftime("%H:%M")
            except Exception as e:
                log.debug(f"Failed to parse commence_time: {e}")

        # Merge with existing data (preserve venue, matchweek, etc.)
        if key in existing_matches:
            match = existing_matches[key].copy()
            if match_date:
                match["date"] = match_date
            if match_time:
                match["time"] = match_time
            if commence:
                match["commence_time"] = commence
        else:
            match = {
                "home_team": home,
                "away_team": away,
                "date": match_date,
                "time": match_time,
                "commence_time": commence,
                "venue": f"{home} Stadium",
                "matchweek": matchweek,
            }

        synced_matches.append(match)

    if not synced_matches:
        log.info("No matches to sync from odds data")
        return []

    output = {
        "matches": synced_matches,
        "matchweek": matchweek,
        "season": season or f"{datetime.now().year - 1}-{str(datetime.now().year)[2:]}",
        "updated_at": datetime.now().isoformat(),
        "source": "synced_from_odds_api",
    }

    with open(manual_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Synced {len(synced_matches)} matches from Odds API")
    return synced_matches


def print_odds_summary(odds_data: Dict[str, Dict]):
    """Print a summary of fetched odds."""
    print("\n" + "=" * 80)
    print(" MULTI-MARKET ODDS - SERIE A")
    print("=" * 80)

    if not odds_data:
        print("\n  No odds data available")
        return

    # Usage stats
    usage = get_usage_summary()
    print(f"\n  API Credits: {usage.get('api_remaining', '?')} remaining")
    print(f"  Today's usage: {usage['daily_used']}/{usage['daily_limit']}")

    print(f"\n  Matches with odds: {len(odds_data)}")

    # 1X2 Summary
    print(f"\n  {'='*70}")
    print(f"  1X2 ODDS (Match Winner)")
    print(f"  {'='*70}")
    print(f"  {'Match':<30} {'Home':>7} {'Draw':>7} {'Away':>7} {'Bookies':>8}")
    print("  " + "-" * 65)

    for match, data in odds_data.items():
        if data.get("h2h"):
            h2h = data["h2h"]
            print(f"  {match:<30} {h2h['home']:>7.2f} {h2h['draw']:>7.2f} "
                  f"{h2h['away']:>7.2f} {h2h['bookmakers_count']:>8}")

    # Over/Under Summary
    totals_count = sum(1 for d in odds_data.values() if d.get("totals"))
    if totals_count > 0:
        print(f"\n  {'='*70}")
        print(f"  OVER/UNDER ODDS (Totals)")
        print(f"  {'='*70}")
        print(f"  {'Match':<30} {'Line':>6} {'Over':>7} {'Under':>7}")
        print("  " + "-" * 55)

        for match, data in odds_data.items():
            if data.get("totals"):
                for total in data["totals"]:
                    if total["line"] == 2.5:  # Show main line
                        print(f"  {match:<30} {total['line']:>6.1f} "
                              f"{total['over']:>7.2f} {total['under']:>7.2f}")

    # Spreads Summary
    spreads_count = sum(1 for d in odds_data.values() if d.get("spreads"))
    if spreads_count > 0:
        print(f"\n  {'='*70}")
        print(f"  POINT SPREAD ODDS (Handicap)")
        print(f"  {'='*70}")
        print(f"  {'Match':<30} {'Home HC':>8} {'Odds':>7} {'Away HC':>8} {'Odds':>7}")
        print("  " + "-" * 65)

        for match, data in odds_data.items():
            if data.get("spreads"):
                for spread in data["spreads"][:1]:  # Show first line only
                    print(f"  {match:<30} {spread['home_handicap']:>+8.1f} "
                          f"{spread['home_odds']:>7.2f} {spread['away_handicap']:>+8.1f} "
                          f"{spread['away_odds']:>7.2f}")


def show_usage_dashboard():
    """Display API usage dashboard."""
    usage = _load_usage()
    summary = get_usage_summary()

    print("\n" + "=" * 60)
    print(" API USAGE DASHBOARD")
    print("=" * 60)

    print(f"\n  Plan: 100K ($59/month)")
    print(f"  Daily Limit: {DAILY_LIMIT} calls")

    print(f"\n  TODAY:")
    print(f"    Used: {summary['daily_used']}")
    print(f"    Remaining: {summary['daily_remaining']}")

    print(f"\n  THIS MONTH:")
    print(f"    Used: {summary['monthly_used']}")
    print(f"    Limit: {summary['monthly_limit']}")

    print(f"\n  API CREDITS:")
    print(f"    Remaining: {summary.get('api_remaining', 'Unknown')}")
    print(f"    Last call: {summary.get('last_call', 'Never')}")

    # Show recent history
    if usage.get("history"):
        print(f"\n  RECENT CALLS (last 5):")
        for entry in usage["history"][-5:]:
            print(f"    {entry['timestamp']}: +{entry['credits_used']} (remaining: {entry.get('remaining', '?')})")


# =============================================================================
# SNAPSHOT MODE (no cache, saves timestamped snapshot)
# =============================================================================

def list_upcoming_events(league: str = "serie_a") -> List[Dict]:
    """List upcoming events for a league via the FREE /events endpoint.

    The Odds API bills /events at 0 credits. This is our authoritative, always-current
    source for upcoming fixtures + kickoff times across any supported league (including
    EPL, which isn't maintained in manual_matches.json).

    Returns a list of dicts with keys: id, sport_key, commence_time (ISO8601 UTC),
    home_team, away_team. Empty list on any failure.

    The circuit breaker is bypassed for this call (it's free), but we still honor
    HTTP errors — if the API key is dead, we return [] and log.
    """
    sport_key = _resolve_sport_key(league)
    if not check_api_key() or not HAS_REQUESTS:
        return []

    url = f"{API_BASE_URL}/sports/{sport_key}/events"
    try:
        resp = requests.get(url, params={"apiKey": API_KEY}, timeout=15)
        resp.raise_for_status()
        events = resp.json() or []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            log.warning("list_upcoming_events: 401 — quota exhausted or key dead")
        else:
            log.warning("list_upcoming_events: HTTP %s", e.response.status_code)
        return []
    except Exception as e:
        log.warning("list_upcoming_events failed for %s: %s", league, e)
        return []

    # /events is billed at 0 credits; still log for visibility.
    remaining_hdr = resp.headers.get("x-requests-remaining")
    remaining = int(remaining_hdr) if remaining_hdr is not None else None
    track_api_call(credits_remaining=remaining, estimated_cost=0,
                   endpoint=f"events_list_{sport_key}")
    log.info("list_upcoming_events(%s): %d events, remaining=%s",
             league, len(events), remaining if remaining is not None else "?")
    return events


def fetch_snapshot(league: str = "serie_a") -> Dict[str, Dict]:
    """Fetch bulk h2h+totals+spreads without cache and save a timestamped snapshot.

    This is designed for high-frequency scheduled calls (every 10-15 min).
    Uses ~4 credits per call (1 per market × 4 regions counted as 1 call each
    for h2h, totals, spreads).

    Args:
        league: League identifier (default: "serie_a").
                Valid values: serie_a, premier_league, la_liga, bundesliga, ligue_1

    Returns:
        Unified odds data dict.
    """
    log.info("=== SNAPSHOT MODE (no cache) ===")

    usage = get_usage_summary()
    log.info(f"API Usage: {usage['daily_used']}/{usage['daily_limit']} today, "
             f"{usage['monthly_used']}/100000 this month")

    # Fetch all core markets without cache
    raw_data = {}
    for market in MARKETS.keys():
        data = fetch_market_odds(market, use_cache=False, league=league)
        if data:
            raw_data[market] = data
        time.sleep(0.3)

    if not raw_data:
        log.warning("Snapshot: no data received")
        return {}

    # Process
    unified_data = process_all_markets(raw_data)
    if not unified_data:
        log.warning("Snapshot: no matches processed")
        return {}

    log.info(f"Snapshot: {len(unified_data)} matches with odds")

    # Save standard odds files (league-aware)
    save_odds(unified_data, league=league)

    # Save timestamped snapshot via odds_tracker
    try:
        from scripts.data.odds_tracker import save_snapshot as save_ts_snapshot
        save_ts_snapshot(odds_data=None)  # reads from saved odds.json
    except Exception as e:
        log.warning(f"Snapshot: failed to save timestamped snapshot: {e}")

    return unified_data


def fetch_tagged_snapshot(
    league: str = "serie_a",
    include_extra_markets: bool = False,
    critical: bool = False,
    tag: str = "",
) -> Dict[str, Dict]:
    """Match-clock-aware snapshot. Called by scheduler stages (T-72h, T-24h, T-5min…).

    Args:
        league: league identifier, must be in ACTIVE_LEAGUES (see config/leagues.py).
        include_extra_markets: if True, also fetches per-event markets (btts,
            double_chance, team_totals, alt_totals, dnb, alt_spreads). Costs
            (# per-event markets) credits × (# events) at eu-only.
        critical: if True, passes through the soft-stop circuit breaker. Use for
            T-5min closing snapshots on matches we're about to bet.
        tag: short label stored in usage history + snapshot filename (e.g. "T72h").

    Returns:
        Unified odds data dict (same shape as fetch_snapshot).
    """
    from config.leagues import ACTIVE_LEAGUES
    if league not in ACTIVE_LEAGUES:
        log.info("fetch_tagged_snapshot: %s is not in ACTIVE_LEAGUES; skipping", league)
        return {}

    ok, msg = check_quota_budget(critical=critical)
    if not ok:
        log.warning("fetch_tagged_snapshot[%s/%s]: quota breaker blocked — %s",
                    league, tag or "?", msg)
        return {}

    log.info("=== SNAPSHOT [%s] league=%s extra=%s critical=%s ===",
             tag or "ad-hoc", league, include_extra_markets, critical)

    raw_data: Dict[str, list] = {}
    for market in MARKETS.keys():
        data = fetch_market_odds(market, use_cache=False, league=league)
        if data:
            raw_data[market] = data
        time.sleep(0.3)

    if not raw_data:
        log.warning("Snapshot[%s]: no core data received", tag or "?")
        return {}

    unified_data = process_all_markets(raw_data)
    if not unified_data:
        return {}

    if include_extra_markets:
        try:
            extra = fetch_extra_markets_per_event(use_cache=False, league=league)
            if extra:
                processed_extra = process_extra_markets(extra)
                for match_key, ex in processed_extra.items():
                    if match_key in unified_data:
                        for market_key, market_data in ex.items():
                            if market_key not in ("home_team", "away_team", "commence_time"):
                                unified_data[match_key].setdefault("extra", {})[market_key] = market_data
        except Exception as e:
            log.warning("Snapshot[%s]: extra markets fetch failed: %s", tag or "?", e)

    log.info("Snapshot[%s]: %d matches for %s", tag or "ad-hoc", len(unified_data), league)

    save_odds(unified_data, league=league)

    try:
        from scripts.data.odds_tracker import save_snapshot as save_ts_snapshot
        # signature is save_snapshot(odds=None); the old odds_data=/tag=
        # kwargs raised TypeError on BOTH attempts, so tagged runs (incl.
        # the T-5m closing snapshot) never saved a tracker snapshot at all
        # (caught live 2026-09-04 14:41, Genoa-Como).
        save_ts_snapshot()
    except Exception as e:
        log.warning("Snapshot[%s]: tracker save failed: %s", tag or "?", e)

    return unified_data


def discover_kickoffs_via_events() -> List[Dict]:
    """Aggregate upcoming fixtures across ACTIVE_LEAGUES via the free /events endpoint.

    This is the scheduler's authoritative fixture discovery for leagues not
    maintained in manual_matches.json (primarily EPL). Costs 0 credits.

    Returns a list of dicts with keys: match (Home vs Away), home_team, away_team,
    kickoff_utc (datetime), league, source ("odds_api_events").
    """
    from config.leagues import ACTIVE_LEAGUES
    out: List[Dict] = []
    for league in ACTIVE_LEAGUES:
        try:
            events = list_upcoming_events(league)
        except Exception as e:
            log.warning("discover_kickoffs_via_events[%s]: %s", league, e)
            continue
        for ev in events:
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            ct = ev.get("commence_time", "")
            if not (home and away and ct):
                continue
            try:
                kickoff_utc = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                if kickoff_utc.tzinfo is None:
                    kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            # Normalize team names so match_key aligns with the rest of the pipeline.
            home_norm = normalize_team(home) if 'normalize_team' in globals() else home
            away_norm = normalize_team(away) if 'normalize_team' in globals() else away
            out.append({
                "match": f"{home_norm} vs {away_norm}",
                "home_team": home_norm,
                "away_team": away_norm,
                "kickoff_utc": kickoff_utc,
                "league": league,
                "source": "odds_api_events",
                "event_id": ev.get("id", ""),
            })
    return sorted(out, key=lambda x: x["kickoff_utc"])


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch multi-market betting odds")
    parser.add_argument("--markets", nargs="+", choices=["h2h", "totals", "spreads"],
                       help="Markets to fetch (default: all)")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh API call")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached data")
    parser.add_argument("--usage", action="store_true", help="Show API usage dashboard")
    parser.add_argument("--sports", action="store_true", help="List available sports")
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        return

    if args.usage:
        show_usage_dashboard()
        return

    if args.sports:
        # Show available sports
        if not check_api_key():
            return
        url = f"{API_BASE_URL}/sports/?apiKey={API_KEY}"
        response = requests.get(url, timeout=30)
        sports = response.json()
        remaining_hdr = response.headers.get("x-requests-remaining")
        remaining = int(remaining_hdr) if remaining_hdr is not None else None
        # /sports/ listing is billed 0 credits by The Odds API
        track_api_call(credits_remaining=remaining, estimated_cost=0, endpoint="sports_list")

        print("\nAvailable Soccer Leagues:")
        for s in sports:
            if s.get("group") == "Soccer":
                print(f"  {s.get('key')}: {s.get('title')}")
        return

    # Fetch odds
    odds_data = fetch_and_save_odds(
        markets=args.markets,
        use_cache=not args.no_cache
    )
    print_odds_summary(odds_data)


if __name__ == "__main__":
    main()
