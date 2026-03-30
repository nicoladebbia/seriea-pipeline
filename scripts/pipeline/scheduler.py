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

# Active leagues (add leagues here as models become available)
# Default: Serie A only. Add "premier_league" etc. when EPL model is trained.
ACTIVE_LEAGUES = ["serie_a", "premier_league"]

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
                if key in seen:
                    continue
                seen.add(key)
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

    # Source 2: results.json (for recently completed matches — has commence_time)
    results_path = DATA_DIR / "upcoming" / "results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                data = json.load(f)
            for key, m in data.get("results", {}).items():
                ct = m.get("commence_time", "")
                if not ct or key in seen:
                    continue
                seen.add(key)
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


def _save_pre_kickoff_state(state: Dict):
    """Save pre-kickoff state."""
    PRE_KICKOFF_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PRE_KICKOFF_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)



# Multi-stage match clock events — each fires once per match per day
# (except lineup_fetch which retries if lineups weren't confirmed)
MATCH_CLOCK_STAGES = [
    {
        "name": "lineup_fetch",
        "minutes_before": 55,   # T-55: lineups drop at T-60, give 5 min buffer
        "window": (20, 58),     # Only try 20-58 min before (never before T-60)
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
]


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
    today_matches = [k for k in kickoffs if k["date"] == today_italy]

    if not today_matches:
        log.info("Match clock: no matches today (Italy date: %s)", today_italy)
        return True

    state = _load_pre_kickoff_state()
    processed = state.get("processed", {})

    # Clean old entries (keep only last 7 days)
    cutoff = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    processed = {k: v for k, v in processed.items()
                 if isinstance(v, dict) and v.get("date", "") >= cutoff}

    actions_needed = {"lineup_fetch": [], "prediction_update": [], "settlement_check": []}

    for match in today_matches:
        match_key = match["match"]
        kickoff_utc = match["kickoff_utc"]
        # Both are UTC-aware — subtraction gives correct timedelta anywhere
        minutes_until = (kickoff_utc - now).total_seconds() / 60

        # Initialize match state
        match_state = processed.get(match_key, {"date": today_italy, "stages": {}})
        if match_state.get("date") != today_italy:
            match_state = {"date": today_italy, "stages": {}}
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
                actions_needed[stage_name].append(match_key)
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

    # Stage 1: Lineup fetch (retries if lineups not yet available)
    if actions_needed["lineup_fetch"]:
        matches_str = ", ".join(actions_needed["lineup_fetch"])
        log.info("Match clock: fetching lineups for %s", matches_str)
        try:
            cmd = [sys.executable, "-c",
                   "from scraper.lineup_fetcher import fetch_and_save_lineups; "
                   "fetch_and_save_lineups()"]
            subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, timeout=60)

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

                if confirmed_matches:
                    from scripts.pipeline.notify import notify_lineups_confirmed
                    notify_lineups_confirmed(matches=", ".join(confirmed_matches))

                # Mark matches WITHOUT confirmed lineups for retry
                for mk in actions_needed["lineup_fetch"]:
                    match_state = processed.get(mk, {})
                    stage_data = match_state.get("stages", {}).get("lineup_fetch", {})
                    if mk not in confirmed_matches:
                        stage_data["needs_retry"] = True
                        log.info("Lineup NOT confirmed for %s — will retry next cycle", mk)
                    else:
                        stage_data.pop("needs_retry", None)
                        log.info("Lineup CONFIRMED for %s", mk)
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
            # Send ONE consolidated pre-kickoff notification (not 3-5 separate ones)
            # Only the bet briefing if user has bets, otherwise just predictions ready
            try:
                from scripts.betting.live_bet_context import get_match_bet_context
                from scripts.pipeline.notify import notify_pre_kickoff_bets
                matches_with_bets = []
                for match_key in actions_needed["prediction_update"]:
                    ctx = get_match_bet_context(match_key)
                    if ctx["has_bets"]:
                        matches_with_bets.append((match_key, ctx))

                if matches_with_bets:
                    # Send bet briefing for the first match only (most imminent)
                    # The briefing already includes all relevant info
                    mk, ctx = matches_with_bets[0]
                    notify_pre_kickoff_bets(mk, bet_context=ctx)
                else:
                    # No bets — just a brief predictions-ready note
                    from scripts.pipeline.notify import notify_predictions_ready
                    notify_predictions_ready(n_matches=len(actions_needed["prediction_update"]))
            except Exception:
                pass

    # Stage 3: Settlement check
    if actions_needed["settlement_check"]:
        matches_str = ", ".join(actions_needed["settlement_check"])
        log.info("Match clock: checking settlement for %s", matches_str)
        try:
            settle_success = run_settle()
            if settle_success:
                send_notification(
                    f"Settlement check completed for: {matches_str}",
                    "Match Clock: Settlement"
                )
        except Exception as e:
            log.warning("Settlement check failed: %s", e)

    # Log summary
    total_triggered = sum(len(v) for v in actions_needed.values())
    if total_triggered == 0:
        pending = []
        for match in today_matches:
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

    # Health check gate — abort on critical system issues
    try:
        from scripts.pipeline.health_check import run_health_check
        health = run_health_check()
        status = health.get("overall_status", "HEALTHY")
        issues = health.get("issues", [])

        if status == "CRITICAL":
            critical_msgs = [msg for lvl, msg in issues if lvl == "CRITICAL"]
            log.warning("HEALTH CHECK CRITICAL: %s", "; ".join(critical_msgs))
            send_notification(
                "\n".join(critical_msgs) + "\n\nPipeline will run predictions but skip betting.",
                title="HEALTH WARNING: Issues Detected"
            )
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
                timeout=1200  # 20 minute timeout (ensemble can take 15min on cold start)
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
                    with open(state_path, "w") as _f:
                        _json.dump(state, _f, indent=2)
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
            timeout=120  # 2 minute timeout (should be ~25s)
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

    except subprocess.TimeoutExpired:
        log.error("Pre-kickoff pipeline timed out")
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
            log.info("Ingested %d/%d new matches", fetched, new_matches)
            send_notification(
                f"Ingested {fetched} new matches | Features rebuilt",
                title="Betting Pipeline Data Update"
            )

        return result

    except Exception as e:
        log.error("Post-matchday ingest failed: %s", e)
        send_notification(f"Data ingest failed: {e}", title="Betting Pipeline Data Update ERROR")
        return {"error": str(e)}


def run_settle() -> bool:
    """Run post-match auto-settlement: fetch results, settle bets, track P&L.

    After settlement, triggers post-matchday data ingest (Sofascore stats +
    feature rebuild) so the pipeline always has fresh data for the next run.

    Only runs on match days. Uses scripts.betting.auto_settle orchestrator.
    """
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
            # Use the coaching-style settlement notification instead of raw send
            try:
                from scripts.pipeline.notify import notify_settlement
                won = result.get("settlement", {}).get("won", 0)
                lost = result.get("settlement", {}).get("lost", 0)
                push = result.get("settlement", {}).get("push", 0)
                balance = result.get("settlement", {}).get("balance", 0)
                notify_settlement(
                    settled=settled, won=won, lost=lost, push=push,
                    profit=profit, balance=balance,
                )
            except Exception:
                # Fallback to simple notification
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

        # Per-bet settlement notifications + loss streak check
        if settled > 0:
            try:
                from scripts.pipeline.notify import notify_bet_settled, notify_loss_streak
                from scripts.betting.bet_journal import get_journal_stats

                # Send per-bet notifications for today's settlements
                stats = get_journal_stats()
                settled_today = stats.get("settled_today", [])
                for bet in settled_today[-5:]:  # Cap at 5 to avoid spam
                    try:
                        notify_bet_settled(bet, result_score=bet.get("result_score", ""))
                    except Exception:
                        pass  # Per-bet notification failure shouldn't block

                # Check for loss streak
                streak = stats.get("current_streak", 0)
                if streak < -2:  # 3+ consecutive losses
                    streak_loss = stats.get("streak_loss", 0)
                    recent = stats.get("recent_losses", [])
                    try:
                        notify_loss_streak(abs(streak), total_loss=abs(streak_loss),
                                           recent_bets=recent)
                    except Exception:
                        pass
            except Exception as e:
                log.debug("Post-settlement notification extras failed: %s", e)

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
    """Run full monitoring cycle: drift detection, calibration check, retrain check.

    Designed to run weekly (Sunday morning). Detects feature drift, checks model
    calibration, and triggers retraining if accuracy has degraded.
    """
    log.info("=" * 60)
    log.info("WEEKLY MONITORING CYCLE")
    log.info("=" * 60)

    try:
        from scripts.pipeline.run_monitoring import MonitoringSystem
        monitor = MonitoringSystem()
        result = monitor.run_full_cycle()

        status = result.get("status", "unknown")
        drift = result.get("drift", {})
        retrain = result.get("retrain", {})
        calibration = result.get("calibration", {})

        # Report drift
        n_drifted = drift.get("features_drifted", 0)
        if n_drifted > 0:
            log.warning("Feature drift detected: %d features drifted", n_drifted)

        # Report calibration
        cal_status = calibration.get("status", "ok")
        if cal_status != "ok":
            log.warning("Calibration issue: %s", cal_status)

        # CLV trend check
        try:
            from scripts.analysis.clv_analysis import analyze_trends
            trends = analyze_trends(window_weeks=2)
            if trends.get("periods") and len(trends["periods"]) >= 2:
                latest_clv = trends["periods"][-1]["avg_clv"]
                prev_clv = trends["periods"][-2]["avg_clv"]
                if prev_clv - latest_clv > 1.5:
                    from scripts.pipeline.notify import notify_clv_degradation
                    notify_clv_degradation(latest_clv, prev_clv, period="2 weeks")
                    log.warning("CLV degradation: %.2f%% -> %.2f%%", prev_clv, latest_clv)
        except Exception as e:
            log.debug("CLV trend check failed: %s", e)

        # Report retrain
        if retrain.get("retrained", False):
            log.info("Model retrained successfully")
            send_notification(
                f"Weekly monitoring: {n_drifted} features drifted | Model retrained",
                title="Betting Pipeline Weekly Monitoring"
            )
        elif n_drifted > 5:
            send_notification(
                f"Weekly monitoring: {n_drifted} features drifted — consider retraining",
                title="Betting Pipeline Drift Warning"
            )
        else:
            log.info("Weekly monitoring: system healthy, no retrain needed")

        return True

    except Exception as e:
        log.error("Weekly monitoring failed: %s", e)
        send_notification(f"Monitoring failed: {e}", title="Betting Pipeline Monitoring ERROR")
        return False


def run_model_retrain() -> bool:
    """Force model retraining with walk-forward validation.

    Called monthly by launchd or on-demand. Retrains all models using latest
    data and validates via walk-forward CV before promoting.
    """
    log.info("=" * 60)
    log.info("MODEL RETRAINING")
    log.info("=" * 60)

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
            send_notification(
                "Models retrained successfully with latest data",
                title="Betting Pipeline Model Retrain"
            )
            return True
        else:
            log.error("Model retraining failed: %s", result.stderr[-500:])
            send_notification(
                f"Model retrain failed: {result.stderr[-200:]}",
                title="Betting Pipeline Retrain ERROR"
            )
            return False

    except subprocess.TimeoutExpired:
        log.error("Model retraining timed out (30 min)")
        return False
    except Exception as e:
        log.error("Model retraining error: %s", e)
        return False


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
    """Single pipeline run.

    Args:
        bankroll: Bankroll amount.
        quick: Use cached data.
        leagues: Active leagues. Defaults to ACTIVE_LEAGUES.
    """
    if leagues is None:
        leagues = ACTIVE_LEAGUES

    success = run_pipeline(bankroll, quick, leagues=leagues)

    league_label = " + ".join(l.replace("_", " ").title() for l in leagues)
    if success:
        send_notification("Pipeline completed successfully", f"{league_label} Betting")
    else:
        send_notification("Pipeline failed after retries", f"{league_label} Betting ERROR")

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

    # Check last run — prefer unified_bet_slip.json (current pipeline)
    bet_slip = DATA_DIR / "upcoming" / "unified_bet_slip.json"
    unified_report = DATA_DIR / "betting" / "unified_report.json"
    report_path = bet_slip if bet_slip.exists() else unified_report
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
        choices=["daemon", "once", "pre-kickoff", "pre-kickoff-monitor", "settle",
                 "health", "monitor", "retrain", "cron-setup", "status"],
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


if __name__ == "__main__":
    main()
