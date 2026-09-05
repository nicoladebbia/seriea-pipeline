#!/usr/bin/env python3
"""
Module: scripts/scheduler.py
Purpose: Automated betting pipeline scheduler with match-day detection, adaptive scheduling, and retry logic
Inputs:  data/upcoming/predictions.json (for match-day detection), SCHEDULE_CONFIG (cron-like timing rules)
Outputs: Triggers pipeline runs via subprocess, logs to logs/scheduler.log, optional Slack/email notifications
Called by: systemd/cron (daemon mode), manual CLI invocation (once mode)
Depends on: config.settings (DATA_DIR, PROJECT_ROOT)
"""

import os
import sys
import json
import signal
import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict
import logging
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, PROJECT_ROOT

# Try to import APScheduler (optional)
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import from central config — single source of truth
from config.leagues import ACTIVE_LEAGUES
from config.team_names import normalize_team
from scripts.utils.match_timing import _entry_kickoff, _load_sofascore_fixtures

#: How far ahead the scheduler looks for kickoffs. The Sofascore files carry the
#: whole season, and an unbounded read would put ~740 fixtures in the kickoff map
#: every tick. Both callers window far tighter than this (72h and
#: MATCH_CLOCK_LOOKAHEAD_HOURS), so 14 days is slack, not a constraint.
SOFA_FIXTURE_HORIZON_DAYS = 14

# Schedule configuration (24h format, Italian timezone for Serie A)
SCHEDULE_CONFIG = {
    # Daily runs (regardless of match day)
    "daily_runs": [
        {"hour": 8, "minute": 0, "description": "Morning data refresh"},
        {"hour": 20, "minute": 0, "description": "Evening odds update"},
    ],

    # Additional runs on match days (when matches are scheduled)
    "match_day_runs": [
        {"hours_before_match": 3, "description": "Pre-match analysis"},
        {"hours_before_match": 1, "description": "Final pre-match update"},
    ],

    # Typical match times by league (for match day detection fallback)
    # Times in local timezone: Italy (CET/CEST) for Serie A, UK (GMT/BST) for EPL
    "typical_match_times": [
        # Serie A
        {"hour": 12, "minute": 30},  # Early kickoff
        {"hour": 15, "minute": 0},   # Afternoon
        {"hour": 18, "minute": 0},   # Early evening
        {"hour": 20, "minute": 45},  # Prime time
    ],

    # EPL-specific typical kickoff times (GMT — Saturday + some midweek)
    "epl_match_times": [
        {"hour": 12, "minute": 30},  # Early kickoff (BT Sport)
        {"hour": 15, "minute": 0},   # 3pm blackout traditional slots
        {"hour": 17, "minute": 30},  # Late afternoon (Sky)
        {"hour": 20, "minute": 0},   # Evening (midweek / Monday night)
    ],

    # Post-match settlement (runs after last match of the day finishes)
    "settlement_runs": [
        {"hour": 23, "minute": 30, "description": "Post-match auto-settlement"},
    ],
}

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_MINUTES = 5

# Log rotation
LOG_MAX_BYTES = 500_000       # 500 KB per log file
LOG_BACKUP_COUNT = 3          # Keep 3 rotated backups (.1, .2, .3)

# Logging setup
LOG_FILE = PROJECT_ROOT / "logs" / "scheduler.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

from logging.handlers import RotatingFileHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT
        ),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def rotate_launchd_logs():
    """Rotate launchd stdout/stderr logs that exceed the size cap.

    launchd writes directly to log files (not via Python logging), so they
    grow unbounded. This function truncates them by keeping only the last
    LOG_MAX_BYTES bytes, preserving the tail (most recent entries).
    """
    log_dir = PROJECT_ROOT / "logs"
    for log_path in log_dir.glob("launchd-*.log"):
        try:
            size = log_path.stat().st_size
            if size > LOG_MAX_BYTES:
                # Keep the last LOG_MAX_BYTES bytes (most recent log lines)
                data = log_path.read_bytes()
                # Find first newline after the trim point to avoid partial lines
                trim_start = len(data) - LOG_MAX_BYTES
                newline_pos = data.find(b"\n", trim_start)
                if newline_pos == -1:
                    newline_pos = trim_start
                log_path.write_bytes(data[newline_pos + 1:])
                log.debug("Rotated %s: %d KB → %d KB",
                          log_path.name, size // 1024, log_path.stat().st_size // 1024)
        except Exception:
            pass  # Non-critical, don't block pipeline


# =============================================================================
# MATCH DAY DETECTION
# =============================================================================

def get_upcoming_matches(leagues: List[str] = None) -> List[Dict]:
    """Load upcoming matches from predictions files for all active leagues.

    Args:
        leagues: List of leagues to load. Defaults to ACTIVE_LEAGUES.
    """
    if leagues is None:
        leagues = ACTIVE_LEAGUES

    all_matches = []

    for league in leagues:
        if league == "serie_a":
            pred_path = DATA_DIR / "upcoming" / "predictions.json"
        else:
            pred_path = DATA_DIR / "upcoming" / f"predictions_{league}.json"

        if not pred_path.exists():
            continue

        try:
            with open(pred_path) as f:
                data = json.load(f)
            preds = data.get("predictions", [])
            for p in preds:
                p.setdefault("league", league)
            all_matches.extend(preds)
        except Exception as e:
            log.warning(f"Could not load {league} predictions: {e}")

    return all_matches


def is_match_day(date: datetime = None, leagues: List[str] = None) -> bool:
    """Check if today has matches scheduled for any active league.

    Args:
        date: Date to check (default: now).
        leagues: Leagues to check (default: ACTIVE_LEAGUES).
    """
    if date is None:
        date = datetime.now()

    matches = get_upcoming_matches(leagues)
    today_str = date.strftime("%Y-%m-%d")

    for match in matches:
        match_date = match.get("date", "")
        if match_date == today_str:
            return True

    return False


def get_today_matches(leagues: List[str] = None) -> List[Dict]:
    """Get matches scheduled for today across all active leagues."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    matches = get_upcoming_matches(leagues)
    return [m for m in matches if m.get("date", "") == today_str]


def get_next_match_time() -> Optional[datetime]:
    """Get the time of the next match (if today).

    Checks both Serie A and EPL typical match times from schedule config.
    """
    today_matches = get_today_matches()
    if not today_matches:
        return None

    now = datetime.now()

    # Collect all typical match times (Serie A + EPL)
    all_match_times = list(SCHEDULE_CONFIG["typical_match_times"])
    all_match_times.extend(SCHEDULE_CONFIG.get("epl_match_times", []))

    # Deduplicate and sort by time
    seen = set()
    unique_times = []
    for mt in all_match_times:
        key = (mt["hour"], mt["minute"])
        if key not in seen:
            seen.add(key)
            unique_times.append(mt)
    unique_times.sort(key=lambda t: (t["hour"], t["minute"]))

    # Find next upcoming match time
    for match_time in unique_times:
        match_dt = now.replace(
            hour=match_time["hour"],
            minute=match_time["minute"],
            second=0
        )
        if match_dt > now:
            return match_dt

    return None


def _utc_now() -> datetime:
    """UTC-aware now — single source of truth for current time."""
    return datetime.now(timezone.utc)


def _to_italy_date(utc_dt: datetime) -> str:
    """Convert UTC datetime to Italy date string (CET/CEST).

    Serie A matches are in Italy, so we group by Italy date.
    CET = UTC+1, CEST = UTC+2 (last Sun March → last Sun October).
    """
    # Approximate DST: use +2 for Apr-Oct, +1 for Nov-Mar
    month = utc_dt.month
    offset = 2 if 4 <= month <= 10 else 1
    italy_dt = utc_dt + timedelta(hours=offset)
    return italy_dt.strftime("%Y-%m-%d")


def get_kickoff_times() -> List[Dict]:
    """Get actual kickoff times from data files.

    Reads commence_time from manual_matches.json and odds API cache.
    Returns list of {match, home_team, away_team, kickoff_utc, date}.

    All times are UTC-aware. The 'date' field is the Italy date (CET/CEST)
    for grouping purposes. Time comparisons must use kickoff_utc vs _utc_now().
    """
    results = []
    seen = set()

    anon = [0]

    def _seen_key(home: str, away: str, fallback: str = "") -> str:
        """Dedup on CANONICAL names. Sofascore says "Milan" where the Odds API
        says "AC Milan"; a raw-name key lets the same fixture in twice, and the
        T-30 monitor then fires twice for one match.

        A row missing either name CANNOT be identified by name: normalising two
        empty strings yields ONE constant key, so the first such row would be
        kept and every later one silently dropped. Source 2 feeds the -3h
        settlement tail, so that failure mode is settled matches going
        unprocessed. Fall back to the row's own unique id, or a per-call
        sentinel that can never collide.
        """
        h, a = normalize_team(home or ""), normalize_team(away or "")
        if not h or not a:
            if fallback:
                return f"?id:{fallback}"
            anon[0] += 1
            return f"?anon:{anon[0]}"
        return f"{h}_{a}"

    # Source 0: Sofascore season fixtures. Leads because it is the ONLY source
    # here that does not depend on the Odds API — sources 1 and 3 are both
    # Odds-API-derived, so a lapsed key left the scheduler blind to every
    # kickoff and the T-30 pre-kickoff monitors simply never fired (2026-08-24).
    # It also carries a real `league` tag, which sources 1 and 2 do not: without
    # one, caller `run_line_movement` falls back to "serie_a" and mis-tags EPL.
    try:
        now_sofa = _utc_now()
        # -6h so a match that has already kicked off stays visible: the match
        # clock wants a 3h settlement tail, and dropping started matches here
        # would silently break settlement rather than pre-kickoff.
        lo = now_sofa - timedelta(hours=6)
        hi = now_sofa + timedelta(days=SOFA_FIXTURE_HORIZON_DAYS)
        for entry in _load_sofascore_fixtures(
            lo, horizon_days=SOFA_FIXTURE_HORIZON_DAYS
        ):
            ko = _entry_kickoff(entry)
            if ko is None or ko > hi:
                continue
            home, away = entry["home_team"], entry["away_team"]
            sk = _seen_key(home, away)
            if sk in seen:
                continue
            seen.add(sk)
            results.append({
                "match": f"{home} vs {away}",
                "home_team": home,
                "away_team": away,
                "kickoff_utc": ko,
                "date": _to_italy_date(ko),
                "league": entry.get("league", ""),
            })
    except (OSError, ValueError, TypeError, KeyError) as e:
        # Narrow deliberately: _load_sofascore_fixtures already degrades quietly
        # on a missing/corrupt file, so anything reaching here is a parse or
        # shape problem in a row. A blind catch here would hide a real bug in
        # the reader while the scheduler keeps reporting "source unavailable".
        log.warning("Sofascore fixture source unavailable: %s", e)

    # Source 1: manual_matches.json (most reliable, has commence_time from Odds API)
    mm_path = DATA_DIR / "upcoming" / "manual_matches.json"
    if mm_path.exists():
        try:
            with open(mm_path) as f:
                data = json.load(f)
            matches = data.get("matches", data if isinstance(data, list) else [])
            for m in matches:
                ct = m.get("commence_time", "")
                if not ct:
                    continue
                home = m.get("home_team", "")
                away = m.get("away_team", "")
                key = f"{home} vs {away}"
                sk = _seen_key(home, away)
                if sk in seen:
                    continue
                seen.add(sk)
                try:
                    kickoff_utc = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    if kickoff_utc.tzinfo is None:
                        kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
                    results.append({
                        "match": key,
                        "home_team": home,
                        "away_team": away,
                        "kickoff_utc": kickoff_utc,
                        "date": _to_italy_date(kickoff_utc),
                    })
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            log.warning("Could not read manual_matches.json: %s", e)

    # Source 3: The Odds API /events endpoint (FREE, 0 credits) — authoritative for EPL
    # and any league not maintained in manual_matches.json. Only called once per
    # scheduler tick; failures are non-fatal.
    try:
        from scripts.data.odds_fetcher import discover_kickoffs_via_events
        for ev in discover_kickoffs_via_events():
            key = ev.get("match", "")
            sk = _seen_key(ev.get("home_team", ""), ev.get("away_team", ""), key)
            if sk in seen:
                continue
            seen.add(sk)
            results.append({
                "match": key,
                "home_team": ev.get("home_team", ""),
                "away_team": ev.get("away_team", ""),
                "kickoff_utc": ev["kickoff_utc"],
                "date": _to_italy_date(ev["kickoff_utc"]),
                "league": ev.get("league", ""),
            })
    except Exception as e:
        log.debug("discover_kickoffs_via_events unavailable: %s", e)

    # Source 2: results.json (for recently completed matches — has commence_time)
    results_path = DATA_DIR / "upcoming" / "results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                data = json.load(f)
            for key, m in data.get("results", {}).items():
                ct = m.get("commence_time", "")
                sk = _seen_key(m.get("home_team", ""), m.get("away_team", ""), key)
                if not ct or sk in seen:
                    continue
                seen.add(sk)
                try:
                    kickoff_utc = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    if kickoff_utc.tzinfo is None:
                        kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
                    results.append({
                        "match": key,
                        "home_team": m.get("home_team", ""),
                        "away_team": m.get("away_team", ""),
                        "kickoff_utc": kickoff_utc,
                        "date": _to_italy_date(kickoff_utc),
                    })
                except (ValueError, TypeError):
                    continue
        except Exception:
            pass

    return sorted(results, key=lambda x: x["kickoff_utc"])


PRE_KICKOFF_STATE_FILE = PROJECT_ROOT / "data" / "pipeline" / "pre_kickoff_state.json"


def _load_pre_kickoff_state() -> Dict:
    """Load state tracking which matches have had pre-kickoff runs."""
    if PRE_KICKOFF_STATE_FILE.exists():
        try:
            with open(PRE_KICKOFF_STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"processed": {}}


def _sheet_landed_after_prediction(match_state: dict) -> bool:
    """True when prediction_update already fired for this match and the lineup
    stage was still on retry, i.e. the T-30 pricing used a predicted XI."""
    stages = (match_state or {}).get("stages") or {}
    pred = stages.get("prediction_update")
    lf = stages.get("lineup_fetch") or {}
    return bool(pred) and bool(lf.get("needs_retry"))


def _save_pre_kickoff_state(state: Dict):
    """Save pre-kickoff state (atomic write)."""
    from config.settings import atomic_write_json
    atomic_write_json(PRE_KICKOFF_STATE_FILE, state)



# Multi-stage match clock events — each fires once per match per day
# (except lineup_fetch which retries if lineups weren't confirmed)
#
# kind="odds"          → dispatched via _dispatch_odds_stage (odds_fetcher.fetch_tagged_snapshot)
# kind="player_props"  → dispatched via _dispatch_player_props_stage (requires confirmed lineups)
# kind missing         → existing in-code dispatch (lineup_fetch / prediction_update / settlement_check)
MATCH_CLOCK_STAGES = [
    # --- Odds snapshots: bulk per league, cheap (1 credit per market, eu region) ---
    # `priority` field gates dispatch through `check_budget_pacing` so spend self-balances
    # to MONTHLY_LIMIT regardless of how many leagues are active. Lower number = higher priority.
    {"name": "odds_T72h", "minutes_before": 72*60, "window": (70*60, 74*60),
     "description": "Opening-line bulk snapshot",
     "kind": "odds", "include_extra_markets": False, "critical": False, "priority": 5},
    {"name": "odds_T24h", "minutes_before": 24*60, "window": (22*60, 26*60),
     "description": "T-24h bulk snapshot",
     "kind": "odds", "include_extra_markets": False, "critical": False, "priority": 4},
    {"name": "odds_T6h",  "minutes_before":  6*60, "window": (5*60, 7*60),
     "description": "T-6h bulk + extra markets",
     "kind": "odds", "include_extra_markets": True, "critical": False, "priority": 2},
    {"name": "odds_T3h",  "minutes_before":  3*60, "window": (2*60+30, 3*60+30),
     "description": "T-3h bulk + extra markets",
     "kind": "odds", "include_extra_markets": True, "critical": False, "priority": 2},
    {"name": "odds_T5m",  "minutes_before":        5, "window": (0, 15),
     "description": "Closing-line bulk (CRITICAL — bypasses soft quota cap)",
     "kind": "odds", "include_extra_markets": False, "critical": True, "priority": 1},

    # --- Player props: only after confirmed_lineups.json has the match ---
    {"name": "player_props_T60", "minutes_before": 60, "window": (40, 70),
     "description": "Player-prop odds snapshot (lineup-gated)",
     "kind": "player_props", "critical": False, "retry_if_empty": True, "priority": 3},

    # --- Existing match-day stages (unchanged) ---
    {
        "name": "lineup_fetch",
        "minutes_before": 55,   # T-55: lineups drop at T-60, give 5 min buffer
        "window": (5, 58),      # retry every cycle down to T-5 (was 20: a sheet
                                # published at T-25 was never picked up, 2026-09-05)
        "description": "Fetch confirmed lineups",
        "retry_if_empty": True, # Re-trigger if lineups weren't found
    },
    {
        "name": "prediction_update",
        "minutes_before": 30,   # T-30: re-run predictions with real lineups
        "window": (10, 45),     # Trigger when kickoff is 10-45 min away
        "description": "Re-predict with confirmed lineups",
    },
    {
        "name": "settlement_check",
        "minutes_before": -120, # T+120: check if match ended, settle
        "window": (-150, -105), # Trigger 105-150 min after kickoff
        "description": "Post-match settlement check",
    },
    {
        # Fill-verification tier (two-tier ledger). (0, 15) guarantees exactly
        # one 15-min monitor tick lands in the window; the nudge itself only
        # fires if ticket lines are still unanswered.
        "name": "fill_nudge_T10",
        "minutes_before": 10,
        "window": (0, 15),
        "description": "T-10 unconfirmed-fill nudge",
    },
    {
        # After kickoff: unanswered ticket lines get flagged "unverified" in
        # the journal (fill tier only -- model tier and settlement untouched).
        "name": "fill_verify_close",
        "minutes_before": -10,
        "window": (-95, -3),
        "description": "Flag unanswered fills unverified",
    },
]

# How far ahead the match clock looks. Widened from "today only" to 80h so the
# T-72h odds stage can fire on Tuesday for a match on Friday.
MATCH_CLOCK_LOOKAHEAD_HOURS = 80


def _dispatch_odds_stage(stage: Dict, matches: List[Dict]) -> bool:
    """Fire an odds-snapshot stage for one or more matches. Returns True on success."""
    if not matches:
        return True
    from config.leagues import ACTIVE_LEAGUES
    # Group by league so we make one bulk call per league, not per match.
    leagues_to_fetch = {m.get("league") for m in matches if m.get("league")}
    if not leagues_to_fetch:
        leagues_to_fetch = set(ACTIVE_LEAGUES)

    try:
        from scripts.data.odds_fetcher import fetch_tagged_snapshot, check_budget_pacing
    except Exception as e:
        log.warning("odds stage %s: odds_fetcher unavailable — %s", stage["name"], e)
        return False

    # Adaptive budget pacing: drop low-priority stages if monthly spend is ahead of schedule.
    # Critical stages (T-5m closing line) bypass via critical=True.
    priority = int(stage.get("priority", 5))
    is_critical = bool(stage.get("critical"))
    ok_pacing, pacing_msg = check_budget_pacing(priority=priority, critical=is_critical)
    if not ok_pacing:
        log.info("odds stage %s SKIPPED by budget controller: %s",
                 stage["name"], pacing_msg)
        return True  # Returning True so retry_if_empty doesn't fire forever

    ok_all = True
    for league in sorted(leagues_to_fetch):
        try:
            fetch_tagged_snapshot(
                league=league,
                include_extra_markets=bool(stage.get("include_extra_markets")),
                critical=is_critical,
                tag=stage["name"],
            )
        except Exception as e:
            log.warning("odds stage %s: fetch failed for %s — %s",
                        stage["name"], league, e)
            ok_all = False
    if stage.get("include_extra_markets") and "serie_a" in leagues_to_fetch:
        # Pick markets ride the T-6h / T-3h extra-market stages (same pacing
        # priority); the fetcher's own 45-min gate dedups repeats.
        try:
            from scripts.data.odds_fetcher import fetch_pick_markets
            fetch_pick_markets(hours_ahead=stage["minutes_before"] / 60 + 1.0)
        except Exception as e:
            log.warning("odds stage %s: pick markets fetch failed — %s", stage["name"], e)
    return ok_all


def _dispatch_player_props_stage(stage: Dict, matches: List[Dict]) -> bool:
    """Fire player-prop odds fetch, but only for matches whose lineups have posted."""
    if not matches:
        return True
    try:
        lineups_path = DATA_DIR / "upcoming" / "confirmed_lineups.json"
        confirmed = set()
        if lineups_path.exists():
            with open(lineups_path) as f:
                lineups_data = json.load(f)
            for mk, mdata in lineups_data.get("matches", {}).items():
                home_xi = mdata.get("home_lineup", [])
                away_xi = mdata.get("away_lineup", [])
                if (isinstance(home_xi, list) and isinstance(away_xi, list)
                        and len(home_xi) >= 7 and len(away_xi) >= 7):
                    confirmed.add(mk)
    except Exception as e:
        log.warning("player_props stage: lineup read failed — %s", e)
        confirmed = set()

    gated = [m for m in matches if m["match"] in confirmed]
    skipped = [m["match"] for m in matches if m["match"] not in confirmed]
    if skipped:
        log.info("player_props stage: deferring %d match(es) — lineups not confirmed: %s",
                 len(skipped), ", ".join(skipped))
    if not gated:
        return False  # returning False keeps retry_if_empty alive

    # Budget pacing: drop player-props if monthly spend is ahead of schedule.
    try:
        from scripts.data.odds_fetcher import check_budget_pacing
        priority = int(stage.get("priority", 3))
        ok_pacing, pacing_msg = check_budget_pacing(priority=priority, critical=False)
        if not ok_pacing:
            log.info("player_props stage SKIPPED by budget controller: %s", pacing_msg)
            return True
    except Exception as e:
        log.debug("player_props stage: pacing check unavailable — %s", e)

    try:
        from scripts.betting.player_prop_odds import fetch_player_prop_odds
    except Exception as e:
        log.warning("player_props stage: module unavailable — %s", e)
        return False

    by_league: Dict[str, List[Dict]] = {}
    for m in gated:
        by_league.setdefault(m.get("league") or "serie_a", []).append(m)

    ok_all = True
    for league, mlist in by_league.items():
        try:
            fetch_player_prop_odds(league=league, use_cache=False)
            log.info("player_props[%s]: fetched for %d match(es)", league, len(mlist))
        except TypeError:
            try:
                fetch_player_prop_odds()
            except Exception as e:
                log.warning("player_props[%s]: legacy fallback failed — %s", league, e)
                ok_all = False
        except Exception as e:
            log.warning("player_props[%s]: fetch failed — %s", league, e)
            ok_all = False
    return ok_all


def run_line_movement(leagues: list = None) -> bool:
    """Hourly line-movement snapshot during match-day windows.

    Cheap bulk h2h+totals snapshot for each active league — but ONLY if any
    match is within a 72h look-ahead window. Otherwise exits at 0 credits.

    Output: timestamped snapshot file via odds_tracker; feeds the line_vel_*
    feature family. Use to compute intra-day line velocity outside of the
    match-clock T-X stages.

    Launchd cadence: every 60 min.
    """
    if leagues is None:
        leagues = ACTIVE_LEAGUES

    try:
        from scripts.data.odds_fetcher import fetch_tagged_snapshot
    except Exception as e:
        log.error("line_movement: odds_fetcher unavailable — %s", e)
        return False

    now = _utc_now()
    kickoffs = get_kickoff_times()
    horizon = timedelta(hours=72)

    relevant_leagues = set()
    for k in kickoffs:
        if 0 <= (k["kickoff_utc"] - now).total_seconds() <= horizon.total_seconds():
            lg = k.get("league")
            # Fallback: if league wasn't tagged (manual_matches.json entries), assume SA
            relevant_leagues.add(lg if lg in leagues else "serie_a")

    relevant_leagues &= set(leagues)
    if not relevant_leagues:
        log.info("line_movement: no matches within 72h in %s — skipping (0 credits)",
                 ",".join(leagues))
        return True

    log.info("line_movement: capturing for %s", sorted(relevant_leagues))
    ok_all = True
    for lg in sorted(relevant_leagues):
        try:
            fetch_tagged_snapshot(
                league=lg,
                include_extra_markets=False,  # bulk h2h+totals+spreads only
                critical=False,
                tag="linemove",
            )
        except Exception as e:
            log.warning("line_movement[%s]: %s", lg, e)
            ok_all = False
    return ok_all


def run_refresh(bankroll: float = 0, leagues: list = None) -> bool:
    """Lightweight incremental refresh — keeps odds + predictions fresh.

    Designed to run every 3 hours between full pipeline runs. Uses
    run_incremental() which only fetches fresh data when stale (>4h for odds,
    >6h for market data) and only predicts unseen fixtures.

    API cost: 2-4 credits per run (~20 credits/day for 7 runs).
    Duration: 30-90 seconds typical.
    """
    log.info("=" * 60)
    log.info("INCREMENTAL REFRESH — keeping odds & predictions fresh")
    log.info("=" * 60)

    try:
        from scripts.pipeline.run_full_pipeline import run_incremental
        from scripts.betting.bankroll_loader import get_effective_bankroll

        br = bankroll if bankroll > 0 else get_effective_bankroll()
        summary = run_incremental(bankroll=br, leagues=leagues)

        status = summary.get("status", "unknown")
        credits = summary.get("credits_used", 0)
        odds_updated = summary.get("odds_updated", False)
        preds_updated = summary.get("predictions_updated", False)

        log.info("Refresh complete: status=%s, credits=%d, odds=%s, predictions=%s",
                 status, credits, odds_updated, preds_updated)

        if summary.get("errors"):
            for err in summary["errors"]:
                log.warning("Refresh error: %s", err)

        return status != "error"

    except Exception as e:
        log.error("Incremental refresh failed: %s", e, exc_info=True)
        try:
            from scripts.pipeline.notify import notify
            notify(f"Incremental refresh failed: {e}",
                   title="Refresh Error", level="warning", category="system")
        except Exception:
            pass
        return False


CAFFEINATE_PID = DATA_DIR / "monitoring" / "caffeinate.pid"


def _match_norm(mk: str) -> str:
    """Canonical form of a 'Home vs Away' key. kickoff sources disagree on
    names (Sofascore 'Milan' vs Odds API 'AC Milan'), and during a Sofascore
    ban a match can reappear under the other source's spelling — comparing
    raw keys would then false-alarm on a commit that actually happened."""
    parts = mk.split(" vs ")
    if len(parts) != 2:
        return mk.strip().lower()
    return f"{normalize_team(parts[0])} vs {normalize_team(parts[1])}".lower()


def _missed_commits(kickoffs: List[Dict], processed: Dict,
                    alerted: Dict, now) -> List[str]:
    """Match keys whose kickoff passed within the last 12h and whose T-30
    prediction_update stage was NEVER ENTERED — i.e. the Mac slept through
    the whole window. Scope is deliberately narrow: the stage marker is
    written on window entry (before dispatch, to prevent double-triggers),
    so an entered-but-failed commit is invisible here — that failure lands
    in the scheduler log/notify path instead. Pure; skips already-alerted."""
    committed = {_match_norm(k) for k, v in processed.items()
                 if "prediction_update" in (v or {}).get("stages", {})}
    seen_alerts = {_match_norm(k) for k in alerted}
    out = []
    for k in kickoffs:
        mins_past = (now - k["kickoff_utc"]).total_seconds() / 60
        if not (0 < mins_past <= 12 * 60):
            continue
        mk = k["match"]
        nk = _match_norm(mk)
        if nk in seen_alerts or nk in committed:
            continue
        out.append(mk)
        seen_alerts.add(nk)
    return out


def _caffeinate_alive(pid: int) -> bool:
    """The pid must exist AND still be caffeinate — a recycled pid must not
    satisfy the singleton check."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        comm = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "comm="],
                              capture_output=True, text=True,
                              timeout=5).stdout.strip()
        return "caffeinate" in comm
    except Exception:
        return True                      # pid alive, name unknowable: assume held


def _ensure_awake_hold(kickoffs: List[Dict], now, spawn=None,
                       pidfile: Path = CAFFEINATE_PID) -> bool:
    """Idle-sleep guard for match days: when a kickoff is within 2h, hold the
    system awake (caffeinate -s — AC power only, by macOS design) through the
    LAST kickoff of the next 6h + 45 min, so back-to-back Sunday slots do not
    leave an unheld seam between holds. Singleton via pidfile, pid verified
    to still be caffeinate. The child is spawned with start_new_session=True:
    this job is a short-lived launchd tick (KeepAlive=false) and launchd
    kills the tick's whole process group on exit — without its own session
    the hold would die seconds after the tick that spawned it. Cannot WAKE a
    sleeping Mac — that needs a sudo pmset schedule, which is Nicola's call."""
    trigger = [k["kickoff_utc"] for k in kickoffs
               if now <= k["kickoff_utc"] <= now + timedelta(hours=2)]
    if not trigger:
        return False
    try:
        pid = int(pidfile.read_text().strip())
        if _caffeinate_alive(pid):
            return True                  # hold already active
    except (OSError, ValueError):
        pass
    horizon = [k["kickoff_utc"] for k in kickoffs
               if now <= k["kickoff_utc"] <= now + timedelta(hours=6)]
    secs = int((max(horizon) - now).total_seconds()) + 45 * 60
    if spawn is None:
        def spawn(s: int) -> int:
            return subprocess.Popen(
                ["/usr/bin/caffeinate", "-s", "-t", str(s)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True).pid
    pid = spawn(secs)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pidfile.write_text(str(pid))
    log.info("Awake hold: caffeinate -s for %ds (PID %s)", secs, pid)
    return True


def run_pre_kickoff_monitor(bankroll: float = 0) -> bool:
    """Multi-stage match clock — orchestrates all match-day events.

    Called every 30 minutes by launchd. For each match today (Italy date),
    checks which stage should fire based on time-to-kickoff in UTC:

    Stages:
        T-60min  lineup_fetch       → Fetch confirmed lineups from API-Football
        T-30min  prediction_update  → Re-run predictions with confirmed lineups + fresh odds
        T+120min settlement_check   → Check if match ended, auto-settle bets

    Each stage fires at most once per match per day (tracked in state file).
    Works offline — based on UTC time vs known kickoff times.
    Timezone-safe — works correctly regardless of where the machine is located.
    """
    now = _utc_now()
    today_italy = _to_italy_date(now)

    kickoffs = get_kickoff_times()

    # Component ledger: freeze the current per-component forecasts ex-ante.
    # Fail-soft — a ledger error must never delay the T-30 bet commit.
    try:
        from scripts.prediction import component_ledger
        component_ledger.snapshot()
    except Exception as e:
        log.debug("component ledger snapshot skipped: %s", e)

    # Blind-spots guards — BEFORE the horizon early-return, so a Mac that
    # slept through kickoff+3h still reports the missed commit on wake, and
    # the awake hold arms as soon as a kickoff is near. Both fully guarded.
    try:
        gstate = _load_pre_kickoff_state()
        alerted = gstate.setdefault("missed_alerts", {})
        stale = [k for k, v in alerted.items()
                 if str(v)[:10] < (now - timedelta(days=7)).strftime("%Y-%m-%d")]
        for k in stale:
            del alerted[k]
        missed = _missed_commits(kickoffs, gstate.get("processed", {}),
                                 alerted, now)
        if missed:
            from scripts.pipeline.notify import notify
            notify("T-30 window NEVER ENTERED for: " + ", ".join(missed)
                   + " — the Mac was asleep through it; no prediction "
                   "refresh or bet commit was attempted.",
                   title="Pre-kickoff window missed", level="warning",
                   category="alert",
                   tg_html="<b>⚠️ Finestra T-30 mai aperta</b> "
                           "(Mac in sleep?): " + ", ".join(missed))
            for mk in missed:
                alerted[mk] = now.isoformat()
            _save_pre_kickoff_state(gstate)
    except Exception as e:
        log.warning("missed-commit detector failed: %s", e)
    try:
        _ensure_awake_hold(kickoffs, now)
    except Exception as e:
        log.warning("awake hold failed: %s", e)

    # Widened horizon: include any match kicking off within LOOKAHEAD hours AND
    # any match that kicked off up to 3h ago (so settlement_check still fires).
    lookahead = timedelta(hours=MATCH_CLOCK_LOOKAHEAD_HOURS)
    settlement_tail = timedelta(hours=3)
    horizon_matches = [
        k for k in kickoffs
        if (now - settlement_tail) <= k["kickoff_utc"] <= (now + lookahead)
    ]

    if not horizon_matches:
        log.info("Match clock: no matches in +%dh / -3h window (Italy date: %s)",
                 MATCH_CLOCK_LOOKAHEAD_HOURS, today_italy)
        return True

    state = _load_pre_kickoff_state()
    processed = state.get("processed", {})

    # Clean old entries (keep only last 7 days)
    cutoff = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    processed = {k: v for k, v in processed.items()
                 if isinstance(v, dict) and v.get("date", "") >= cutoff}

    # actions_needed maps stage_name → list of match dicts (full objects, for dispatch)
    actions_needed: Dict[str, List[Dict]] = {s["name"]: [] for s in MATCH_CLOCK_STAGES}

    for match in horizon_matches:
        match_key = match["match"]
        kickoff_utc = match["kickoff_utc"]
        match_date = match.get("date", today_italy)
        # Both are UTC-aware — subtraction gives correct timedelta anywhere
        minutes_until = (kickoff_utc - now).total_seconds() / 60

        # Initialize match state (scoped by each match's own date, not today's)
        match_state = processed.get(match_key, {"date": match_date, "stages": {}})
        if match_state.get("date") != match_date:
            match_state = {"date": match_date, "stages": {}}
        stages_done = match_state.get("stages", {})

        for stage in MATCH_CLOCK_STAGES:
            stage_name = stage["name"]

            # Check if already done — but allow retry for stages that support it
            if stage_name in stages_done:
                prev = stages_done[stage_name]
                if stage.get("retry_if_empty") and prev.get("needs_retry"):
                    pass  # Allow re-trigger
                else:
                    continue  # Already fired successfully

            win_lo, win_hi = stage["window"]
            if win_lo <= minutes_until <= win_hi:
                log.info("Match clock [%s] %s: kickoff in %.0f min — TRIGGERING %s",
                         stage_name, match_key, minutes_until, stage["description"])
                actions_needed[stage_name].append(match)
                stages_done[stage_name] = {
                    "triggered_at": now.isoformat(),
                    "minutes_until_kickoff": round(minutes_until),
                }
            elif minutes_until > win_hi:
                log.debug("Match clock [%s] %s: kickoff in %.0f min — too early",
                          stage_name, match_key, minutes_until)
            elif minutes_until < win_lo:
                log.debug("Match clock [%s] %s: kickoff in %.0f min — window passed",
                          stage_name, match_key, minutes_until)

        match_state["stages"] = stages_done
        processed[match_key] = match_state

    # Save state before executing actions (prevents double-triggers on crash)
    state["processed"] = processed
    state["last_check"] = now.isoformat()
    _save_pre_kickoff_state(state)

    # Execute actions
    success = True

    # --- Dispatch kind=odds and kind=player_props stages via the new dispatchers ---
    for stage in MATCH_CLOCK_STAGES:
        kind = stage.get("kind")
        matches = actions_needed.get(stage["name"], [])
        if not matches:
            continue
        if kind == "odds":
            ok = _dispatch_odds_stage(stage, matches)
        elif kind == "player_props":
            ok = _dispatch_player_props_stage(stage, matches)
        else:
            continue  # in-code stages below handle themselves
        if not ok and stage.get("retry_if_empty"):
            # Mark for retry on next tick
            for m in matches:
                pkey = m["match"]
                ps = processed.get(pkey, {}).setdefault("stages", {})
                if stage["name"] in ps:
                    ps[stage["name"]]["needs_retry"] = True
            state["processed"] = processed
            _save_pre_kickoff_state(state)

    # The legacy stage dispatchers below key off the dict's "match" string — rebuild
    # a flat-key view so we don't have to rewrite the three downstream blocks.
    _actions_flat = {k: [m["match"] for m in v] for k, v in actions_needed.items()}
    actions_needed = _actions_flat  # back-compat for legacy dispatcher blocks

    # Stage 1: Lineup fetch (retries if lineups not yet available)
    if actions_needed["lineup_fetch"]:
        matches_str = ", ".join(actions_needed["lineup_fetch"])
        log.info("Match clock: fetching lineups for %s", matches_str)
        try:
            cmd = [sys.executable, "-c",
                   "from scraper.lineup_fetcher import fetch_and_save_lineups; "
                   "fetch_and_save_lineups()"]
            # 60s starved this on matchday afternoons (killed at T-51 on
            # 2026-08-28); and on 2026-08-30 a Sofascore 403 wave burned ~42s
            # of retries PER MATCH, so 5 matches blew even 180s and the kill
            # discarded the lineups that WERE confirmed. The fetcher now
            # self-bounds at 150s (deadline + blocked-endpoint breaker) and
            # saves partial results, so 180s here is a guaranteed fit.
            _lf = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=180)
            # The fetcher's own log lines live in the child's stderr; dropping
            # them hid a fully dead source chain for a whole matchday (2026-09-05)
            _tail = [ln for ln in (_lf.stderr or "").splitlines() if "WARNING" in ln or "ERROR" in ln][-6:]
            for ln in _tail:
                log.warning("lineup fetcher: %s", ln[-300:])
            if _lf.returncode:
                log.warning("lineup fetcher exited %s", _lf.returncode)

            # Check which matches actually got confirmed lineups
            try:
                from config.settings import DATA_DIR
                import json as _json
                lineups_path = DATA_DIR / "upcoming" / "confirmed_lineups.json"
                confirmed_matches = set()
                if lineups_path.exists():
                    with open(lineups_path) as _f:
                        lineups_data = _json.load(_f)
                    for mk, mdata in lineups_data.get("matches", {}).items():
                        home_xi = mdata.get("home_lineup", [])
                        away_xi = mdata.get("away_lineup", [])
                        both_confirmed = (
                            isinstance(home_xi, list) and len(home_xi) >= 7
                            and isinstance(away_xi, list) and len(away_xi) >= 7
                        )
                        if both_confirmed:
                            confirmed_matches.add(mk)

                # No standalone lineup blast (2026-08-27): full XIs are
                # dashboard material; the order ticket carries an "XI ✓" flag
                # per match instead.
                if confirmed_matches:
                    log.info("Lineups confirmed for %s (no notification — ticket carries the flag)",
                             ", ".join(confirmed_matches))
                    # Fantacalcio T-60: official XIs are the biggest p_play
                    # update of the week — rebuild the advice and push a diff
                    # while the league deadline (first kickoff) is still open.
                    # Fully guarded: must never touch the betting path.
                    try:
                        from scripts.fantacalcio.lineup_check import (
                            run_official_lineup_check,
                        )
                        log.info("Fanta lineup check: %s",
                                 run_official_lineup_check())
                    except Exception as e:
                        log.warning(
                            "Fanta lineup check failed (betting unaffected): %s", e)

                # Fantacalcio T-60 scorer props: independent of lineup
                # confirmation (a Sofascore ban must not starve it). One
                # Odds API credit per event, self-deduped inside. Fully
                # guarded: must never touch the betting path.
                try:
                    from scripts.fantacalcio.lineup_check import (
                        run_scorer_props_check,
                        run_screenshot_reminder,
                    )
                    log.info("Fanta scorer props: %s",
                             run_scorer_props_check())
                    log.info("Fanta screenshot reminder: %s",
                             run_screenshot_reminder())
                except Exception as e:
                    log.warning(
                        "Fanta scorer props failed (betting unaffected): %s", e)

                # Mark matches WITHOUT confirmed lineups for retry
                for mk in actions_needed["lineup_fetch"]:
                    match_state = processed.get(mk, {})
                    stage_data = match_state.get("stages", {}).get("lineup_fetch", {})
                    if mk not in confirmed_matches:
                        stage_data["needs_retry"] = True
                        log.info("Lineup NOT confirmed for %s — will retry next cycle", mk)
                        # Once per match: say WHY, from the fetcher's chain report,
                        # so "XI prob." on /picks is a known state, not a mystery
                        if not stage_data.get("unavailable_notified"):
                            try:
                                import json as _jj

                                from config.settings import DATA_DIR as _dd
                                from scripts.pipeline.notify import notify
                                _rep = _jj.loads((_dd / "upcoming" / "lineup_chain_status.json").read_text())
                                _why = _rep.get("reason") or "nessuna fonte ha risposto"
                                notify(f"Formazioni ufficiali non disponibili per {mk}: {_why}. "
                                       "I props giocatore in /picks restano su XI probabile.",
                                       title="Formazioni", level="warning", category="alert")
                                stage_data["unavailable_notified"] = True
                            except Exception as _e:  # noqa: BLE001 - never block the clock
                                log.debug("lineup-unavailable notice skipped: %s", _e)
                    else:
                        stage_data.pop("needs_retry", None)
                        log.info("Lineup CONFIRMED for %s", mk)
                        if _sheet_landed_after_prediction(match_state):
                            # the T-30 run already priced this match off the
                            # PREDICTED XI: run it again on the team sheet
                            match_state["stages"].pop("prediction_update", None)
                            if mk not in actions_needed["prediction_update"]:
                                actions_needed["prediction_update"].append(mk)
                            log.info("Sheet landed after the T-30 run for %s — re-predicting on the XI", mk)
                    match_state.setdefault("stages", {})["lineup_fetch"] = stage_data
                    processed[mk] = match_state

                # Save updated state with retry flags
                state["processed"] = processed
                _save_pre_kickoff_state(state)

            except Exception as e:
                log.warning("Lineup confirmation check failed: %s", e)
        except Exception as e:
            log.warning("Lineup fetch failed: %s", e)

    # Stage 2: Prediction update (pre-kickoff pipeline)
    if actions_needed["prediction_update"]:
        matches_str = ", ".join(actions_needed["prediction_update"])
        log.info("Match clock: re-predicting for %s", matches_str)
        success = run_pre_kickoff(bankroll)
        if success:
            # Notifications moved INTO run_pre_kickoff (2026-08-27): the order
            # ticket fires at the journal-commit moment covering EVERY match in
            # the window, and the no-action notice covers imminent matches that
            # produced no bets. The old poller-side briefing raced the commit
            # and briefed only the most imminent match.
            log.info("Pre-kickoff run complete for %s (ticket/notice sent from the run itself)",
                     ", ".join(actions_needed["prediction_update"]))

    # Stage 3: Settlement check
    if actions_needed["settlement_check"]:
        matches_str = ", ".join(actions_needed["settlement_check"])
        log.info("Match clock: checking settlement for %s", matches_str)
        try:
            settle_success = run_settle()
            if settle_success:
                # Routine machine status: log only. The FT card carries the
                # result; the day wrap carries the reconciliation.
                log.info("Settlement check completed for: %s", matches_str)
        except Exception as e:
            log.warning("Settlement check failed: %s", e)
        # Player prop outcomes settle on the same clock (forward sample for the
        # Player_Props betting gate — predictions are useless evidence unsettled)
        try:
            from scripts.betting.prop_tracker import settle_props
            prop_result = settle_props()
            if prop_result.get("total_settled", 0):
                log.info("Prop settlement: %s props settled", prop_result["total_settled"])
        except Exception as e:
            log.warning("Prop settlement failed (non-fatal): %s", e)

    # Stage 4: T-10 unconfirmed-fill nudge (two-tier ledger).
    # Only ticketed bets count -- the T-30 marker maps ticket numbers to
    # bet_ids per match. If every line is answered, the stage stays silent.
    if actions_needed.get("fill_nudge_T10"):
        try:
            from scripts.betting.bet_journal import get_pending_bets
            from scripts.pipeline.notify import notify_fill_nudge
            from scripts.pipeline.run_full_pipeline import _t30_state
            ticketed = _t30_state().get("bets", {})
            pending = get_pending_bets(include_superseded=False)
            mins_by_match = {m["match"]: (m["kickoff_utc"] - now).total_seconds() / 60
                            for m in horizon_matches}
            for mk in actions_needed["fill_nudge_T10"]:
                ids = {v.get("bet_id") for v in ticketed.values()
                       if isinstance(v, dict) and v.get("match") == mk}
                unconfirmed = [b for b in pending
                               if b.get("bet_id") in ids and not b.get("fill_status")]
                if unconfirmed:
                    notify_fill_nudge(mk, len(unconfirmed),
                                      minutes=int(mins_by_match.get(mk, 10)))
                    log.info("Fill nudge sent: %s, %d unconfirmed", mk, len(unconfirmed))
                else:
                    log.info("Fill nudge skipped for %s: all lines answered", mk)
        except Exception as e:
            log.warning("Fill nudge stage failed (non-fatal): %s", e)

    # Stage 5: post-kickoff unverified sweep. Unanswered ticket lines are
    # flagged in the journal's fill tier; explicit answers are never touched.
    if actions_needed.get("fill_verify_close"):
        try:
            from scripts.betting.bet_journal import sweep_unverified_fills
            from scripts.pipeline.run_full_pipeline import _t30_state
            ticketed = _t30_state().get("bets", {})
            for mk in actions_needed["fill_verify_close"]:
                ids = [v.get("bet_id") for v in ticketed.values()
                       if isinstance(v, dict) and v.get("match") == mk]
                if ids:
                    n = sweep_unverified_fills(ids)
                    if n:
                        log.info("Fill sweep: %d bet(s) unverified for %s", n, mk)
        except Exception as e:
            log.warning("Fill sweep stage failed (non-fatal): %s", e)

    # Log summary
    total_triggered = sum(len(v) for v in actions_needed.values())
    if total_triggered == 0:
        pending = []
        for match in horizon_matches:
            match_key = match["match"]
            mins = (match["kickoff_utc"] - now).total_seconds() / 60
            ms = processed.get(match_key, {}).get("stages", {})
            remaining = [s["name"] for s in MATCH_CLOCK_STAGES if s["name"] not in ms]
            if remaining and mins > -180:
                pending.append(f"  {match_key}: {mins:+.0f}min — pending: {', '.join(remaining)}")
        if pending:
            log.info("Match clock: waiting. Next events:\n%s", "\n".join(pending[:5]))
        else:
            log.info("Match clock: all stages complete for today's matches")

    return success


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_pipeline(bankroll: float = 0, quick: bool = False, leagues: list = None) -> bool:
    """Execute the betting pipeline with health check + risk controls gates.

    Args:
        bankroll: Bankroll amount (0 = auto-load from journal).
        quick: If True, use cached data.
        leagues: List of leagues to run. Defaults to ACTIVE_LEAGUES.
    """
    if leagues is None:
        leagues = ACTIVE_LEAGUES

    if bankroll <= 0:
        try:
            from scripts.betting.bankroll_loader import get_effective_bankroll
            bankroll = get_effective_bankroll()
        except Exception as e:
            log.warning("Failed to auto-load bankroll: %s — using 1000", e)
            bankroll = 1000.0

    log.info("=" * 60)
    log.info("STARTING SCHEDULED PIPELINE RUN (leagues: %s)", ", ".join(leagues))
    log.info("=" * 60)

    # Refresh fixtures if stale (>24h) — pipeline needs fresh fixture list
    try:
        import os
        from datetime import datetime as _dt
        _fixtures_path = DATA_DIR / "upcoming" / "matches.json"
        _fixtures_age_h = 999
        if _fixtures_path.exists():
            _fixtures_age_h = (_dt.now() - _dt.fromtimestamp(os.path.getmtime(_fixtures_path))).total_seconds() / 3600
        if _fixtures_age_h > 24:
            log.info("Fixtures are %.0fh old — refreshing...", _fixtures_age_h)
            from scripts.data.fetch_upcoming_matches import get_upcoming_matches, save_upcoming_matches
            matches = get_upcoming_matches()
            if matches:
                save_upcoming_matches(matches)
                log.info("Refreshed %d upcoming matches", len(matches))
            log.info("Fixtures refreshed")
    except Exception as e:
        log.warning("Fixture refresh failed: %s — continuing with stale data", e)

    # Health check gate — abort on critical system issues
    try:
        from scripts.pipeline.health_check import run_health_check
        health = run_health_check()
        status = health.get("overall_status", "HEALTHY")
        issues = health.get("issues", [])

        if status == "CRITICAL":
            critical_msgs = [msg for lvl, msg in issues if lvl == "CRITICAL"]
            log.warning("HEALTH CHECK CRITICAL: %s", "; ".join(critical_msgs))
            # Only notify if the message changed (prevent daily spam for same issue)
            _alert_key = ";".join(sorted(critical_msgs))
            _alert_cache_path = DATA_DIR / "cache" / ".last_health_alert"
            _alert_cache_path.parent.mkdir(parents=True, exist_ok=True)
            _last_alert = _alert_cache_path.read_text().strip() if _alert_cache_path.exists() else ""
            if _alert_key != _last_alert:
                send_notification(
                    "\n".join(critical_msgs) + "\n\nPipeline will run predictions but skip betting.",
                    title="HEALTH WARNING: Issues Detected"
                )
                _alert_cache_path.write_text(_alert_key)
            # Don't abort — still run predictions and odds updates.
            # The risk gate below will handle betting pause separately.
        elif status == "WARNING":
            warning_msgs = [msg for lvl, msg in issues if lvl == "WARNING"]
            log.warning("HEALTH CHECK WARNING: %s", "; ".join(warning_msgs))
        else:
            log.info("Health check: HEALTHY")
    except Exception as e:
        log.warning("Health check failed: %s — proceeding anyway", e)

    # Risk controls gate — check before running
    try:
        from scripts.betting.risk_controls import check_risk_gates
        gates = check_risk_gates(bankroll=bankroll)
        if not gates["allow_betting"]:
            log.warning("RISK GATE BLOCKED: %s", gates["reason"])
            send_notification(
                f"Betting paused: {gates['reason']}",
                title="RISK CONTROL: Betting Paused"
            )
            return True  # Not a pipeline failure, just a risk pause
        if gates["reduce_stakes"]:
            log.warning("RISK: Stakes reduced to %.0f%%", gates["stake_multiplier"] * 100)
            # Adjust bankroll to reduce stakes
            bankroll = bankroll * gates["stake_multiplier"]
            log.info("Effective bankroll reduced to %.0f for this run", bankroll)
    except Exception as e:
        log.warning("Risk controls check failed: %s — proceeding with caution", e)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "pipeline" / "run_full_pipeline.py"),
        "--bankroll", str(bankroll),
        "--leagues", ",".join(leagues),
    ]

    if quick:
        cmd.append("--quick")

    for attempt in range(1, MAX_RETRIES + 1):
        log.info(f"Attempt {attempt}/{MAX_RETRIES}")

        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=2400  # 40 min timeout (2 leagues × ~8 min features + data + predictions)
            )

            if result.returncode == 0:
                log.info("Pipeline completed successfully")
                log.debug(f"Output: {result.stdout[-500:]}")
                # Ensure pipeline_state is updated even if the subprocess
                # didn't write it (belt-and-suspenders)
                try:
                    import json as _json
                    from datetime import datetime as _dt
                    state_path = PROJECT_ROOT / "data" / "pipeline_state.json"
                    state = {}
                    if state_path.exists():
                        with open(state_path) as _f:
                            state = _json.load(_f)
                    state["last_run"] = _dt.now().isoformat()
                    state["last_run_status"] = "success"
                    from config.settings import atomic_write_json as _awj
                    _awj(state_path, state)
                except Exception:
                    pass
                return True
            else:
                log.error(f"Pipeline failed with return code {result.returncode}")
                log.error(f"Error output: {result.stderr[-500:]}")

        except subprocess.TimeoutExpired:
            log.error("Pipeline timed out after 20 minutes")
        except Exception as e:
            log.error(f"Pipeline execution error: {e}")

        if attempt < MAX_RETRIES:
            log.info(f"Retrying in {RETRY_DELAY_MINUTES} minutes...")
            import time
            time.sleep(RETRY_DELAY_MINUTES * 60)

    log.error("All retry attempts failed")

    # Send critical alert — pipeline has been failing for all retries
    try:
        send_notification(
            f"Pipeline failed {MAX_RETRIES} times. Last error: {result.stderr[-200:] if 'result' in dir() else 'unknown'}",
            title="CRITICAL: Pipeline Down"
        )
    except Exception:
        pass

    return False


# The pre-kickoff subprocess is NOT a ~25s flow on a matchday: the per-event
# odds extras loop alone runs ~5 min (20 events x ~7s x 2 leagues) plus the
# ensemble re-predict. A 120s timeout killed the T-30 order-ticket commit
# TWICE on go-live day (2026-08-28, 14:12 and 14:29 — no ticket, no
# no-action, no journal write). 900s = one launchd tick; launchd will not
# start a second monitor instance while one runs, so this cannot overlap.
PRE_KICKOFF_TIMEOUT_SEC = 900


def run_pre_kickoff(bankroll: float = 0) -> bool:
    """Execute the pre-kickoff pipeline (confirmed lineups + re-prediction + CLV capture).

    Lightweight ~25s flow:
    1. Quick odds refresh
    2. Fetch confirmed lineups from API-Football (~60-90 min pre-match)
    3. Regenerate predictions with confirmed lineups (40% lineup xG weight for O/U)
    4. Regenerate betting recommendations
    5. Capture closing odds for pending bets → CLV tracking

    This is the highest-ROI automated step: backtest shows O/U Over ROI jumps
    from +4.6% (predicted lineups) to +25.2% (confirmed lineups).
    """
    if not is_match_day():
        log.info("Not a match day — skipping pre-kickoff run")
        return True

    if bankroll <= 0:
        try:
            from scripts.betting.bankroll_loader import get_effective_bankroll
            bankroll = get_effective_bankroll()
        except Exception as e:
            log.warning("Failed to auto-load bankroll: %s — using 1000", e)
            bankroll = 1000.0

    log.info("=" * 60)
    log.info("STARTING PRE-KICKOFF PIPELINE (confirmed lineups)")
    log.info("=" * 60)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "pipeline" / "run_full_pipeline.py"),
        "--pre-kickoff",
        "--bankroll", str(bankroll),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=PRE_KICKOFF_TIMEOUT_SEC
        )

        if result.returncode == 0:
            log.info("Pre-kickoff pipeline completed successfully")
            # Check if confirmed lineups were fetched
            if "Confirmed lineups for" in result.stdout:
                log.info("CONFIRMED LINEUPS FETCHED — predictions updated with 40% lineup xG weight")
                send_notification(
                    "Confirmed lineups fetched! Predictions updated with player-level xG.",
                    "Betting Pipeline Pre-Kickoff"
                )
            else:
                log.info("No confirmed lineups available yet (may be too early)")

            # Capture CLV from the fresh odds snapshot
            try:
                from scripts.betting.clv_capture import capture_clv
                clv_result = capture_clv(from_cache=True)  # odds already fetched by pre-kickoff
                n_captured = clv_result.get("captured", 0)
                if n_captured > 0:
                    log.info("CLV captured for %d pending bets", n_captured)
                else:
                    log.debug("No CLV captured this run")
            except Exception as e:
                log.warning("CLV capture failed: %s", e)

            return True
        else:
            log.error(f"Pre-kickoff pipeline failed: {result.stderr[-500:]}")
            return False

    except subprocess.TimeoutExpired as e:
        # TimeoutExpired carries the partial captured output — log the tail so
        # the next miss is diagnosable from the log alone (today's was not).
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        tail = out[-800:]
        log.error("Pre-kickoff pipeline timed out after %ds — partial stdout tail:\n%s",
                  PRE_KICKOFF_TIMEOUT_SEC, tail)
        return False
    except Exception as e:
        log.error(f"Pre-kickoff execution error: {e}")
        return False


# =============================================================================
# NOTIFICATION (Optional)
# =============================================================================

def send_notification(message: str, title: str = "Betting Pipeline"):
    """Send notification via all configured channels."""
    try:
        from scripts.pipeline.notify import notify
        # Infer level from title
        title_lower = title.lower()
        if "error" in title_lower or "fail" in title_lower:
            level = "error"
        elif "warning" in title_lower or "warn" in title_lower:
            level = "warning"
        else:
            level = "info"
        notify(message, title=title, level=level)
    except Exception as e:
        # Fallback to macOS if import fails
        if sys.platform == "darwin":
            try:
                subprocess.run(["osascript", "-e", f'display notification "{message}" with title "{title}"'], capture_output=True, timeout=5)
            except Exception:
                pass


# =============================================================================
# POST-MATCH SETTLEMENT
# =============================================================================

def run_post_matchday_ingest() -> dict:
    """Ingest new Sofascore match data + rebuild features after matchday.

    Runs matchday_updater to detect finished matches, fetch stats from
    Sofascore, merge into all parquets, then optionally rebuild features.

    Returns summary dict from matchday_updater.
    """
    log.info("=" * 60)
    log.info("POST-MATCHDAY DATA INGEST")
    log.info("=" * 60)

    try:
        from scripts.data.matchday_updater import run_matchday_update
        result = run_matchday_update(
            rebuild_features=True,
            regenerate_standings=True,
        )

        new_matches = result.get("new_matches_detected", 0)
        fetched = result.get("matches_fetched", 0)

        if new_matches == 0:
            log.info("No new matches to ingest")
        else:
            # Routine success is log-only (signal-only Telegram doctrine,
            # 2026-08-27). The error path below still notifies.
            log.info("Ingested %d/%d new matches", fetched, new_matches)

        return result

    except Exception as e:
        log.error("Post-matchday ingest failed: %s", e)
        send_notification(f"Data ingest failed: {e}", title="Betting Pipeline Data Update ERROR")
        return {"error": str(e)}


_DAY_WRAP_MARKER = DATA_DIR / "pipeline" / "day_wrap_state.json"
_PROOF_MARKER = DATA_DIR / "pipeline" / "proof_of_edge_state.json"


def _post_settlement_wrap(result: dict) -> None:
    """RECONCILIATION consolidation: per-batch settlement cards stay silent;
    ONE day-wrap card fires when the day's last open bet settles. A late
    settlement after the wrap (postponed finish, void correction) falls back
    to the classic settlement card so it is never silent. Never raises.
    """
    from scripts.pipeline.notify import notify_day_wrap, notify_settlement
    from scripts.betting.bet_journal import get_journal_stats, get_pending_bets

    today = datetime.now().strftime("%Y-%m-%d")
    balance = result.get("settlement", {}).get("balance", 0)
    stats = get_journal_stats()
    settled_today = stats.get("settled_today", []) or []

    state = {}
    try:
        state = json.loads(_DAY_WRAP_MARKER.read_text())
    except (OSError, ValueError):
        pass

    if state.get("date") == today and state.get("wrapped"):
        s = result.get("settlement", {})
        notify_settlement(
            settled=s.get("settled", 0), won=s.get("won", 0),
            lost=s.get("lost", 0), push=s.get("push", 0),
            profit=s.get("profit", 0), balance=balance,
            settled_bets=settled_today,
        )
        return

    open_today = [b for b in get_pending_bets(include_superseded=False)
                  if (b.get("date") or "9999") <= today]
    if open_today:
        log.info("Settlement: %d bet(s) still open today \u2014 day wrap deferred",
                 len(open_today))
        return

    notify_day_wrap(settled_today, balance=balance)
    try:
        _DAY_WRAP_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _DAY_WRAP_MARKER.write_text(json.dumps({"date": today, "wrapped": True}))
    except OSError as e:
        log.debug("Day-wrap marker write failed: %s", e)


def _maybe_proof_of_edge() -> None:
    """Sunday >=22:00: the weekly proof-of-edge card, once per ISO week.

    Hosted on the 5-min settlement tick because it is the one job guaranteed
    alive on a Sunday evening. A week with no settled bets sends nothing but
    still marks the week (no retry loop). Never raises.
    """
    try:
        now = datetime.now()
        if now.weekday() != 6 or now.hour < 22:
            return
        week = now.strftime("%G-W%V")
        try:
            if json.loads(_PROOF_MARKER.read_text()).get("week") == week:
                return
        except (OSError, ValueError):
            pass
        from scripts.pipeline.notify import notify_proof_of_edge
        notify_proof_of_edge(days=7)
        _PROOF_MARKER.parent.mkdir(parents=True, exist_ok=True)
        _PROOF_MARKER.write_text(json.dumps({"week": week}))
    except Exception as e:
        log.debug("Proof-of-edge check failed: %s", e)


def run_settle() -> bool:
    """Run post-match auto-settlement: fetch results, settle bets, track P&L.

    After settlement, triggers post-matchday data ingest (Sofascore stats +
    feature rebuild) so the pipeline always has fresh data for the next run.

    Only runs on match days. Uses scripts.betting.auto_settle orchestrator.
    """
    _maybe_proof_of_edge()

    # Component ledger: snapshot + settle vs matches.parquet + rot alarm +
    # floor-gated weight refit. Fail-soft — settlement must never block on it.
    try:
        from scripts.prediction import component_ledger
        log.info("component ledger: %s", component_ledger.run())
    except Exception as e:
        log.debug("component ledger run skipped: %s", e)

    # Check for pending bets even on non-match days (late finishes, postponed games)
    has_pending = False
    try:
        from scripts.betting.bet_journal import get_pending_bets
        pending = get_pending_bets()
        has_pending = len(pending) > 0
    except Exception:
        pass

    if not is_match_day() and not has_pending:
        log.info("Not a match day and no pending bets — skipping settlement")
        return True

    log.info("=" * 60)
    log.info("RUNNING POST-MATCH SETTLEMENT%s", " (pending bets from previous matchday)" if not is_match_day() else "")
    log.info("=" * 60)

    try:
        from scripts.betting.auto_settle import run as run_auto_settle
        result = run_auto_settle(days_from=3)

        settled = result.get("settlement", {}).get("settled", 0)
        profit = result.get("settlement", {}).get("profit", 0)
        alerts = result.get("alerts", [])

        if settled > 0:
            # Day-wrap consolidation: silent while bets remain open today,
            # ONE reconciliation card when the last one settles (the FT card
            # already carried each match's P&L in real time).
            try:
                _post_settlement_wrap(result)
            except Exception:
                # Fallback: never let a settlement pass unannounced
                send_notification(
                    f"Settled {settled} bets | P&L: {'+'if profit >= 0 else ''}\u20ac{profit:.2f}",
                    title="Betting Pipeline Settlement"
                )

        # Alert on critical drift
        critical = [a for a in alerts if a["level"] == "CRITICAL"]
        if critical:
            send_notification(
                "\n".join(a["message"] for a in critical),
                title="CRITICAL: Betting Drift Alert"
            )

        # Post-settlement: loss streak check (per-bet notifications removed —
        # batch notify_settlement() already shows each bet with full details)
        if settled > 0:
            try:
                from scripts.pipeline.notify import notify_loss_streak
                from scripts.betting.bet_journal import get_journal_stats

                stats = get_journal_stats()
                streak = stats.get("current_streak", 0)
                if streak < -4:  # 5+ consecutive losses (raised from 3 to reduce noise)
                    streak_loss = stats.get("streak_loss", 0)
                    recent = stats.get("recent_losses", [])
                    try:
                        notify_loss_streak(abs(streak), total_loss=abs(streak_loss),
                                           recent_bets=recent)
                    except Exception:
                        pass
            except Exception as e:
                log.debug("Post-settlement notification extras failed: %s", e)

            # bankroll.json / history.json / state.json are regenerated by
            # ledger.rebuild_caches() inside the settle flow (auto_settle ->
            # update_bankroll_json). No out-of-band sync here — one writer.

        # Run post-settlement reconciliation
        if settled > 0:
            try:
                from scripts.analysis.live_reconciliation import run as run_reconciliation
                recon = run_reconciliation()
                live_roi = recon.get("market_comparison", {})
                log.info("Post-settlement reconciliation complete")

                # Run risk controls check after settlement
                from scripts.betting.risk_controls import check_risk_gates
                gates = check_risk_gates()
                if not gates["allow_betting"]:
                    send_notification(
                        f"Post-settlement risk check: {gates['reason']}",
                        title="RISK CONTROL: Betting Paused"
                    )
                    log.warning("RISK GATE: %s", gates["reason"])
            except Exception as e:
                log.warning("Post-settlement analysis failed: %s", e)

        log.info("Settlement complete: %d bets settled", settled)

    except Exception as e:
        log.error("Settlement failed: %s", e)
        send_notification(f"Settlement failed: {e}", title="Betting Pipeline Settlement ERROR")
        return False

    # --- Update rolling correction layer with settled results ---
    if settled > 0:
        try:
            from ml.correction_layer import CorrectionLayer, read_ledger
            from storage.paths import features_path

            layer = CorrectionLayer()
            layer.load()

            if layer.active:
                ledger = read_ledger()
                watermark = layer.rolling.processed_count

                # Only process NEW ledger entries (skip already-incorporated ones)
                new_entries = ledger[watermark:]
                if new_entries:
                    # Read actual results from features.parquet
                    feat_df = pd.read_parquet(str(features_path()),
                                             columns=["home_team", "match_date", "result"])
                    feat_df["match_date"] = pd.to_datetime(feat_df["match_date"])
                    feat_df["_key"] = (feat_df["home_team"] + "_" +
                                       feat_df["match_date"].dt.strftime("%Y-%m-%d"))
                    results_map = dict(zip(feat_df["_key"], feat_df["result"]))

                    updated = 0
                    for entry in new_entries:
                        key = entry.get("home_team", "") + "_" + entry.get("match_date", "")[:10]
                        actual = results_map.get(key)
                        if actual and actual in ("H", "D", "A"):
                            layer.update_rolling(
                                entry["prob_H"], entry["prob_D"], entry["prob_A"],
                                actual, entry.get("match_date"),
                            )
                            updated += 1

                    # Advance watermark to total ledger length
                    layer.rolling.processed_count = len(ledger)
                    if updated > 0:
                        layer.save_rolling()
                        log.info("Rolling correction updated with %d new settled matches "
                                 "(watermark %d → %d)", updated, watermark, len(ledger))
                    else:
                        # Still save to advance watermark even if no matches settled yet
                        layer.save_rolling()
        except Exception as e:
            log.warning("Rolling correction update failed (non-fatal): %s", e)

    # --- Post-matchday data ingest (Sofascore stats + features rebuild) ---
    # Runs after settlement so the morning pipeline has fresh data.
    try:
        run_post_matchday_ingest()
    except Exception as e:
        log.warning("Post-matchday ingest failed (non-fatal): %s", e)

    return True


# =============================================================================
# WEEKLY MONITORING + MODEL RETRAINING
# =============================================================================

def run_weekly_monitoring() -> bool:
    """Weekly CLV trend check.

    Runs Monday 06:00 via the weekly-monitor plist. The former drift /
    calibration / retrain cycle (monitoring/ package) was removed 2026-09-04:
    its retrainer was a placeholder that trained nothing and its validator
    scored random predictions. Real retraining is scripts/pipeline/weekly_retrain.py.
    """
    import time as _t
    log.info("=" * 60)
    log.info("WEEKLY CLV TREND CHECK")
    log.info("=" * 60)

    t0 = _t.time()
    details: dict = {}
    status = "ok"
    error_msg = None

    try:
        from scripts.analysis.clv_analysis import analyze_trends
        trends = analyze_trends(window_weeks=2)
        if trends.get("periods") and len(trends["periods"]) >= 2:
            latest_clv = trends["periods"][-1]["avg_clv"]
            prev_clv = trends["periods"][-2]["avg_clv"]
            details["CLV (latest)"] = f"{latest_clv:.2f}%"
            if prev_clv - latest_clv > 1.5:
                from scripts.pipeline.notify import notify_clv_degradation
                notify_clv_degradation(latest_clv, prev_clv, period="2 weeks")
                log.warning("CLV degradation: %.2f%% -> %.2f%%", prev_clv, latest_clv)
                status = "warn"
        else:
            details["CLV"] = "not enough settled periods"

        success = True

    except Exception as e:
        log.error("Weekly monitoring failed: %s", e)
        status = "fail"
        error_msg = str(e)
        success = False

    elapsed = _t.time() - t0
    try:
        from scripts.pipeline.notify import notify_scheduler_run
        notify_scheduler_run(
            name="weekly-monitor",
            status=status,
            duration_sec=elapsed,
            details=details,
            error=error_msg,
        )
    except Exception as e:
        log.debug("Notify failed: %s", e)

    return success


def run_model_retrain() -> bool:
    """Force model retraining with walk-forward validation.

    Called monthly by launchd or on-demand. Retrains all models using latest
    data and validates via walk-forward CV before promoting.
    """
    import time as _t
    log.info("=" * 60)
    log.info("MODEL RETRAINING")
    log.info("=" * 60)

    t0 = _t.time()
    status = "fail"
    details: dict = {}
    error_msg = None
    success = False

    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "models" / "train_unified.py"),
            "--exclude-odds",
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        if result.returncode == 0:
            log.info("Model retraining completed successfully")
            details["Ensemble retrain"] = "promoted"

            try:
                from scripts.models.train_over_under import train_over_under
                log.info("Retraining O/U classifiers...")
                ou_report = train_over_under(lines=[1.5, 2.5], top_k=60, n_tune_trials=0)
                # The trainer gates promotion vs the incumbent; say what happened.
                ou_summary = ", ".join(
                    f"{ln} {'promoted' if info.get('promoted') else 'held'}"
                    for ln, info in ou_report.get("lines", {}).items()
                ) or "no lines trained"
                log.info("O/U classifiers: %s", ou_summary)
                details["O/U classifiers"] = ou_summary
            except Exception as ou_e:
                log.error("O/U retrain failed (non-fatal): %s", ou_e)
                details["O/U classifiers"] = f"failed: {ou_e}"
                status = "warn"

            if status != "warn":
                status = "success"
            success = True
        else:
            tail = result.stderr[-500:]
            log.error("Model retraining failed: %s", tail)
            error_msg = tail
            success = False

    except subprocess.TimeoutExpired:
        log.error("Model retraining timed out (30 min)")
        error_msg = "Timed out after 30 minutes"
        success = False
    except Exception as e:
        log.error("Model retraining error: %s", e)
        error_msg = str(e)
        success = False

    elapsed = _t.time() - t0
    try:
        from scripts.pipeline.notify import notify_scheduler_run
        notify_scheduler_run(
            name="monthly-retrain",
            status=status if success else "fail",
            duration_sec=elapsed,
            details=details,
            error=error_msg,
        )
    except Exception as e:
        log.debug("Notify failed: %s", e)

    return success


# =============================================================================
# SCHEDULER MODES
# =============================================================================

def run_daemon(bankroll: float = 0, leagues: list = None):
    """Run as daemon with APScheduler.

    Args:
        bankroll: Bankroll amount.
        leagues: Active leagues to run. Defaults to ACTIVE_LEAGUES.
    """
    if leagues is None:
        leagues = ACTIVE_LEAGUES

    if not HAS_APSCHEDULER:
        log.error("APScheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    scheduler = BlockingScheduler()

    # Add daily runs (full pipeline for all active leagues)
    for run in SCHEDULE_CONFIG["daily_runs"]:
        scheduler.add_job(
            lambda: run_pipeline(bankroll, leagues=leagues),
            CronTrigger(hour=run["hour"], minute=run["minute"]),
            id=f"daily_{run['hour']}_{run['minute']}",
            name=run["description"]
        )
        log.info(f"Scheduled: {run['description']} at {run['hour']:02d}:{run['minute']:02d}")

    # Add pre-kickoff runs for typical Serie A match times.
    # Runs 1h before each typical kickoff to fetch confirmed lineups.
    # Skips automatically on non-match days.
    for mt in SCHEDULE_CONFIG["typical_match_times"]:
        pre_h = mt["hour"] - 1
        pre_m = mt["minute"]
        if pre_h < 0:
            pre_h = 23
        scheduler.add_job(
            lambda: run_pre_kickoff(bankroll),
            CronTrigger(hour=pre_h, minute=pre_m),
            id=f"pre_kickoff_{mt['hour']}_{mt['minute']}",
            name=f"Pre-kickoff for {mt['hour']:02d}:{mt['minute']:02d} matches"
        )
        log.info(f"Scheduled: Pre-kickoff at {pre_h:02d}:{pre_m:02d} "
                 f"(for {mt['hour']:02d}:{mt['minute']:02d} kickoff)")

    # Add EPL pre-kickoff runs (if premier_league is active)
    if "premier_league" in leagues:
        for mt in SCHEDULE_CONFIG.get("epl_match_times", []):
            pre_h = mt["hour"] - 1
            pre_m = mt["minute"]
            if pre_h < 0:
                pre_h = 23
            job_id = f"epl_pre_kickoff_{mt['hour']}_{mt['minute']}"
            # Avoid duplicate job IDs if Serie A and EPL share a time slot
            if scheduler.get_job(job_id) is None:
                scheduler.add_job(
                    lambda: run_pre_kickoff(bankroll),
                    CronTrigger(hour=pre_h, minute=pre_m),
                    id=job_id,
                    name=f"EPL Pre-kickoff for {mt['hour']:02d}:{mt['minute']:02d} matches"
                )
                log.info(f"Scheduled: EPL Pre-kickoff at {pre_h:02d}:{pre_m:02d} "
                         f"(for {mt['hour']:02d}:{mt['minute']:02d} kickoff)")

    # Add post-match settlement runs
    for run in SCHEDULE_CONFIG.get("settlement_runs", []):
        scheduler.add_job(
            run_settle,
            CronTrigger(hour=run["hour"], minute=run["minute"]),
            id=f"settle_{run['hour']}_{run['minute']}",
            name=run["description"]
        )
        log.info(f"Scheduled: {run['description']} at {run['hour']:02d}:{run['minute']:02d}")

    # Graceful shutdown
    def shutdown(signum, frame):
        log.info("Shutting down scheduler...")
        scheduler.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("=" * 60)
    log.info("SCHEDULER STARTED")
    log.info("=" * 60)
    log.info(f"Bankroll: ${bankroll}")
    log.info(f"Active leagues: {', '.join(leagues)}")
    log.info(f"Match day: {'Yes' if is_match_day(leagues=leagues) else 'No'}")
    log.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")


def run_once(bankroll: float = 0, quick: bool = False, leagues: list = None):
    """Single pipeline run — morning or evening.

    Success is silent (2026-08-27): the old picks card advertised stale-odds
    candidates 12h before the T-30 commit — an invitation to place early, the
    journal-measured −5% ROI path. Candidates surface as a count in the daily
    digest; selections appear only on the T-30 order ticket. On failure the
    standard scheduler_run card fires with the error.
    """
    if leagues is None:
        leagues = ACTIVE_LEAGUES

    import time as _t
    t0 = _t.time()
    success = run_pipeline(bankroll, quick, leagues=leagues)
    elapsed = _t.time() - t0

    # Component ledger sweep after the pipeline refreshed predictions.json.
    try:
        from scripts.prediction import component_ledger
        log.info("component ledger: %s", component_ledger.run())
    except Exception as e:
        log.debug("component ledger run skipped: %s", e)

    # Infer which schedule this is (morning vs evening) from wall-clock hour
    hour = datetime.now().hour
    if 5 <= hour < 14:
        sched_name = "morning"
    elif 17 <= hour < 23:
        sched_name = "evening"
    else:
        sched_name = "morning" if hour < 5 else "evening"

    try:
        from scripts.pipeline.notify import notify_scheduler_run
        notify_scheduler_run(
            name=sched_name,
            status="success" if success else "fail",
            duration_sec=elapsed,
            error=None if success else "Pipeline retries exhausted",
        )
    except Exception as e:
        log.error("Scheduler notification failed: %s", e)

    sys.exit(0 if success else 1)


def show_cron_setup(bankroll: float = 100.0):
    """Print cron entries for manual cron setup."""
    script_path = Path(__file__).resolve()
    pipeline_path = PROJECT_ROOT / "scripts" / "pipeline" / "run_full_pipeline.py"
    python_path = sys.executable
    log_path = PROJECT_ROOT / "logs" / "cron.log"

    print("=" * 60)
    print("CRON SETUP FOR BETTING PIPELINE")
    print("=" * 60)
    print()
    print("Add these lines to your crontab (crontab -e):")
    print()

    print("# --- Daily runs (full pipeline) ---")
    for run in SCHEDULE_CONFIG["daily_runs"]:
        cron_line = f"{run['minute']} {run['hour']} * * * {python_path} {script_path} once --bankroll {bankroll}"
        print(f"# {run['description']}")
        print(cron_line)
        print()

    print("# --- Pre-kickoff runs (confirmed lineups + re-prediction) ---")
    print("# Runs 1h before each typical kickoff. Fetches confirmed lineups")
    print("# from API-Football and re-runs predictions with 40% lineup xG weight.")
    print("# Backtest: O/U Over ROI jumps from +4.6% to +25.2% with confirmed lineups.")
    print("# Requires: APIFOOTBALL_KEY env variable set.")
    for mt in SCHEDULE_CONFIG["typical_match_times"]:
        pre_h = mt["hour"] - 1
        pre_m = mt["minute"]
        if pre_h < 0:
            pre_h = 23
        cron_line = (
            f"{pre_m} {pre_h} * * * "
            f"cd {PROJECT_ROOT} && {python_path} {pipeline_path} --pre-kickoff "
            f"--bankroll {bankroll} >> {log_path} 2>&1"
        )
        print(f"# Pre-kickoff for {mt['hour']:02d}:{mt['minute']:02d} matches")
        print(cron_line)
        print()

    print("# --- Post-match settlement (auto-settle + reconciliation + risk check) ---")
    for run in SCHEDULE_CONFIG.get("settlement_runs", []):
        cron_line = (
            f"{run['minute']} {run['hour']} * * * "
            f"cd {PROJECT_ROOT} && {python_path} {script_path} settle "
            f">> {log_path} 2>&1"
        )
        print(f"# {run['description']}")
        print(cron_line)
        print()

    print("# --- Manual one-off ---")
    print(f"# Full pipeline:  {python_path} {script_path} once --bankroll {bankroll}")
    print(f"# Pre-kickoff:    {python_path} {script_path} pre-kickoff --bankroll {bankroll}")
    print(f"# Settlement:     {python_path} {script_path} settle")
    print(f"# Risk check:     {python_path} -m scripts.betting.risk_controls --bankroll {bankroll}")
    print(f"# Health check:   {python_path} {script_path} health")
    print(f"# Status:         {python_path} {script_path} status")
    print()


def show_status():
    """Show scheduler and pipeline status."""
    print("=" * 60)
    print("BETTING PIPELINE STATUS")
    print("=" * 60)
    print()

    # Check last run — unified_bet_slip.json is the only bet artifact
    report_path = DATA_DIR / "upcoming" / "unified_bet_slip.json"
    if report_path.exists():
        with open(report_path) as f:
            data = json.load(f)
        generated = data.get("generated_at", "Unknown")
        num_bets = len(data.get("selected_bets", data.get("bets", [])))
        print(f"Last run: {generated}")
        print(f"Bets generated: {num_bets}")
        print(f"Source: {report_path.name}")
    else:
        print("No previous runs found")

    print()

    # Match day status
    if is_match_day():
        print("Today is a MATCH DAY")
        today_matches = get_today_matches()
        print(f"Matches scheduled: {len(today_matches)}")
        for match in today_matches[:3]:
            print(f"  - {match.get('match', 'Unknown')}")
    else:
        print("No matches scheduled today")

    print()

    # Settlement status
    try:
        from scripts.betting.bet_journal import get_journal_stats
        stats = get_journal_stats()
        if stats.get("total_bets", 0) > 0:
            print(f"Betting: {stats['total_bets']} total ({stats['pending']} pending, "
                  f"{stats['settled']} settled)")
            print(f"P&L: ${stats['total_profit']:+,.2f} | ROI: {stats['roi_pct']:+.1f}%")
        print()
    except Exception:
        pass

    # Risk controls
    try:
        from scripts.betting.risk_controls import check_risk_gates
        gates = check_risk_gates()
        status = "ALLOWED" if gates["allow_betting"] else "PAUSED"
        if gates["reduce_stakes"]:
            status = f"REDUCED ({gates['stake_multiplier']:.0%})"
        print(f"Risk status: {status}")
        dd = gates["checks"]["drawdown"]
        print(f"  Drawdown: {dd.get('current_drawdown_pct', 0):.1f}% "
              f"(max {dd.get('max_drawdown_pct', 0):.1f}%)")
        ls = gates["checks"]["consecutive_losses"]
        print(f"  Loss streak: {ls.get('current_streak', 0)} current, "
              f"{ls.get('max_streak', 0)} max")
        if not gates["allow_betting"]:
            print(f"  REASON: {gates['reason']}")
        print()
    except Exception:
        pass

    # Next scheduled run (if daemon mode)
    print("Scheduled runs (daily):")
    for run in SCHEDULE_CONFIG["daily_runs"]:
        print(f"  - {run['hour']:02d}:{run['minute']:02d} - {run['description']}")
    for run in SCHEDULE_CONFIG.get("settlement_runs", []):
        print(f"  - {run['hour']:02d}:{run['minute']:02d} - {run['description']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Betting Pipeline Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "mode",
        choices=["daemon", "once", "refresh", "pre-kickoff", "pre-kickoff-monitor",
                 "settle", "health", "monitor", "retrain", "cron-setup", "status",
                 "line-movement"],
        help="Run mode (monitor: weekly drift/calibration check, retrain: force model retraining)"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=0,
        help="Bankroll amount (default: 0 = auto-load from journal)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (skip heavy computations)"
    )
    parser.add_argument(
        "--leagues",
        type=str,
        default=None,
        help="Comma-separated leagues to run (default: ACTIVE_LEAGUES config). "
             "E.g. --leagues serie_a,premier_league"
    )

    args = parser.parse_args()

    # Resolve leagues
    if args.leagues:
        active_leagues = [l.strip() for l in args.leagues.split(",") if l.strip()]
    else:
        active_leagues = ACTIVE_LEAGUES

    # Rotate launchd logs on every invocation (keeps them under 500KB)
    rotate_launchd_logs()

    if args.mode == "daemon":
        run_daemon(args.bankroll, leagues=active_leagues)
    elif args.mode == "once":
        run_once(args.bankroll, args.quick, leagues=active_leagues)
    elif args.mode == "refresh":
        run_refresh(args.bankroll, leagues=active_leagues)
    elif args.mode == "pre-kickoff":
        success = run_pre_kickoff(args.bankroll)
        sys.exit(0 if success else 1)
    elif args.mode == "pre-kickoff-monitor":
        success = run_pre_kickoff_monitor(args.bankroll)
        sys.exit(0 if success else 1)
    elif args.mode == "settle":
        success = run_settle()
        sys.exit(0 if success else 1)
    elif args.mode == "health":
        from scripts.pipeline.health_check import run_health_check, print_health_check
        result = run_health_check()
        print_health_check(result)
        sys.exit(0 if result["overall_status"] == "HEALTHY" else 1)
    elif args.mode == "monitor":
        success = run_weekly_monitoring()
        sys.exit(0 if success else 1)
    elif args.mode == "retrain":
        success = run_model_retrain()
        sys.exit(0 if success else 1)
    elif args.mode == "cron-setup":
        show_cron_setup(args.bankroll)
    elif args.mode == "status":
        show_status()
    elif args.mode == "line-movement":
        success = run_line_movement(leagues=active_leagues)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
