#!/usr/bin/env python3
"""Automated health monitor — runs every 30 min via launchd.

Combines the existing health_check with live API validation and staleness
detection so broken keys / stalled pipelines are caught within the hour
instead of after 16 days.

Checks:
  1. All existing health_check checks (data freshness, models, betting, system)
  2. Odds API key alive? (GET /v4/sports/ — costs 0 requests)
  3. Perplexity API key alive? (tiny chat completion)
  4. Last successful pipeline run age (from pipeline_state.json)
  5. Last successful odds fetch age (from api_usage.json)
  6. Pending bets past match time (placed_bets.json entries > 3h old)

Outputs:
  - data/monitoring/health_status.json  (always written)
  - logs/monitor.log                    (always appended)
  - macOS notification                  (on CRITICAL only)

Usage:
    python -m scripts.pipeline.monitor          # Run all checks
    python -m scripts.pipeline.monitor --json   # Machine-readable output
    python -m scripts.pipeline.monitor --quiet  # Only output on failure

Exit codes: 0=HEALTHY, 1=WARNING, 2=CRITICAL
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR, PROJECT_ROOT
from scripts.pipeline.health_check import run_health_check

# ─── Paths ───
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "monitor.log"
STATUS_FILE = DATA_DIR / "monitoring" / "health_status.json"

# ─── Thresholds ───
MAX_PIPELINE_RUN_AGE_HOURS = 36      # Pipeline should run at least daily
MAX_ODDS_FETCH_AGE_HOURS = 24        # Odds fetched at least once per day
PENDING_BET_STALE_HOURS = 3          # Bets older than 3h should be settled

# ─── Logging ───
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("monitor")
log.setLevel(logging.INFO)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
log.addHandler(_file_handler)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
log.addHandler(_console_handler)


# ─── Helpers ───

def _load_env_key(name: str) -> str:
    """Load an API key from environment or .env file."""
    val = os.environ.get(name)
    if val:
        return val
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip() == name:
                        return v.strip()
    return ""


def _load_json(path: Path) -> dict:
    """Safely load a JSON file, returning empty dict on failure."""
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _iso_age_hours(iso_str: str) -> float:
    """Return hours since an ISO timestamp, or -1 if unparseable."""
    if not iso_str:
        return -1
    try:
        dt = datetime.fromisoformat(iso_str)
        return (datetime.now() - dt).total_seconds() / 3600
    except (ValueError, TypeError):
        return -1


# ─── Check functions ───

def check_odds_api_key() -> Dict:
    """Validate the Odds API key with a zero-cost /v4/sports/ call."""
    import urllib.request
    import urllib.error

    key = _load_env_key("ODDS_API_KEY")
    if not key:
        return {"status": "CRITICAL", "detail": "ODDS_API_KEY not set"}

    url = f"https://api.the-odds-api.com/v4/sports/?apiKey={key}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                remaining = resp.headers.get("x-requests-remaining", "?")
                return {
                    "status": "OK",
                    "detail": f"Key valid, {remaining} requests remaining",
                    "requests_remaining": remaining,
                }
            return {"status": "WARNING", "detail": f"HTTP {resp.status}"}
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"status": "CRITICAL", "detail": "Odds API key INVALID (401)"}
        return {"status": "WARNING", "detail": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"status": "WARNING", "detail": f"Connection failed: {e}"}


def check_perplexity_api_key() -> Dict:
    """Validate the Perplexity API key with a minimal request."""
    import urllib.request
    import urllib.error

    key = _load_env_key("PERPLEXITY_API_KEY")
    if not key:
        return {"status": "WARNING", "detail": "PERPLEXITY_API_KEY not set (optional)"}

    url = "https://api.perplexity.ai/chat/completions"
    payload = json.dumps({
        "model": "sonar",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }).encode()

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                return {"status": "OK", "detail": "Key valid"}
            return {"status": "WARNING", "detail": f"HTTP {resp.status}"}
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return {"status": "CRITICAL", "detail": f"Perplexity key INVALID ({e.code})"}
        # 429 = rate limited, which means key is valid
        if e.code == 429:
            return {"status": "OK", "detail": "Key valid (rate limited)"}
        return {"status": "WARNING", "detail": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"status": "WARNING", "detail": f"Connection failed: {e}"}


def check_pipeline_staleness() -> Dict:
    """Check when the pipeline last ran successfully."""
    state = _load_json(DATA_DIR / "pipeline_state.json")

    last_run = state.get("last_run")
    age = _iso_age_hours(last_run)

    if age < 0:
        return {"status": "WARNING", "detail": "No pipeline run recorded", "last_run": None}

    if age > MAX_PIPELINE_RUN_AGE_HOURS:
        return {
            "status": "CRITICAL",
            "detail": f"Last pipeline run {age:.1f}h ago (threshold: {MAX_PIPELINE_RUN_AGE_HOURS}h)",
            "last_run": last_run,
            "age_hours": round(age, 1),
        }

    return {
        "status": "OK",
        "detail": f"Last run {age:.1f}h ago",
        "last_run": last_run,
        "age_hours": round(age, 1),
    }


def check_odds_fetch_staleness() -> Dict:
    """Check when odds were last fetched, using both pipeline_state and api_usage."""
    state = _load_json(DATA_DIR / "pipeline_state.json")
    usage = _load_json(DATA_DIR / "api_usage.json")

    # Best source: pipeline_state.last_odds_fetch
    last_fetch = state.get("last_odds_fetch")
    age = _iso_age_hours(last_fetch)

    # Fallback: most recent date in api_usage daily_calls
    if age < 0 and usage.get("daily_calls"):
        try:
            latest_day = max(usage["daily_calls"].keys())
            # Approximate: end of that day
            last_fetch = f"{latest_day}T23:59:59"
            age = _iso_age_hours(last_fetch)
        except Exception:
            pass

    if age < 0:
        return {"status": "WARNING", "detail": "No odds fetch recorded", "last_fetch": None}

    if age > MAX_ODDS_FETCH_AGE_HOURS:
        return {
            "status": "WARNING",
            "detail": f"Last odds fetch {age:.1f}h ago (threshold: {MAX_ODDS_FETCH_AGE_HOURS}h)",
            "last_fetch": last_fetch,
            "age_hours": round(age, 1),
        }

    return {
        "status": "OK",
        "detail": f"Last fetch {age:.1f}h ago",
        "last_fetch": last_fetch,
        "age_hours": round(age, 1),
    }


def check_pending_bets() -> Dict:
    """Check for placed bets whose matches should have finished (>3h past kickoff)."""
    placed = _load_json(DATA_DIR / "betting" / "placed_bets.json")
    bets = placed.get("bets", [])

    if not bets:
        return {"status": "OK", "detail": "No placed bets to check", "stale_count": 0}

    now = datetime.now()
    stale = []

    for bet in bets:
        match_date_str = bet.get("date", "")
        if not match_date_str:
            continue
        try:
            # Match dates are YYYY-MM-DD; assume ~21:00 kickoff (Serie A evening)
            match_dt = datetime.strptime(match_date_str, "%Y-%m-%d").replace(hour=21, minute=0)
            hours_since = (now - match_dt).total_seconds() / 3600
            if hours_since > PENDING_BET_STALE_HOURS:
                stale.append({
                    "match": bet.get("match", "?"),
                    "date": match_date_str,
                    "hours_since": round(hours_since, 1),
                })
        except (ValueError, TypeError):
            continue

    if stale:
        return {
            "status": "WARNING",
            "detail": f"{len(stale)} bet(s) past settlement window",
            "stale_count": len(stale),
            "stale_bets": stale[:10],  # Cap output
        }

    return {"status": "OK", "detail": "All placed bets within settlement window", "stale_count": 0}


# ─── Notification ───

from scripts.pipeline.notify import notify as _unified_notify


def send_macos_notification(title: str, message: str):
    """Send notification via all configured channels (macOS + Telegram)."""
    _unified_notify(message, title=title, level="critical", category="alert")


# ─── Main monitor ───

def run_monitor() -> Dict:
    """Run all monitoring checks and return unified result."""
    log.info("=" * 50)
    log.info("MONITOR RUN STARTED")
    log.info("=" * 50)

    result = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "HEALTHY",
        "checks": {},
        "issues": [],
    }

    # 1. Run existing health_check
    try:
        hc = run_health_check()
        result["checks"]["health_check"] = {
            "status": hc["overall_status"],
            "issues": hc.get("issues", []),
        }
        # Propagate issues
        for level, msg in hc.get("issues", []):
            result["issues"].append((level, f"[health_check] {msg}"))
    except Exception as e:
        log.error(f"health_check failed: {e}")
        result["checks"]["health_check"] = {"status": "ERROR", "detail": str(e)}
        result["issues"].append(("CRITICAL", f"health_check threw exception: {e}"))

    # 2. Odds API key
    log.info("Checking Odds API key...")
    odds_api = check_odds_api_key()
    result["checks"]["odds_api_key"] = odds_api
    if odds_api["status"] == "CRITICAL":
        result["issues"].append(("CRITICAL", f"[odds_api] {odds_api['detail']}"))
    elif odds_api["status"] == "WARNING":
        result["issues"].append(("WARNING", f"[odds_api] {odds_api['detail']}"))
    log.info(f"  Odds API: {odds_api['status']} — {odds_api['detail']}")

    # 3. Perplexity API key
    log.info("Checking Perplexity API key...")
    pplx = check_perplexity_api_key()
    result["checks"]["perplexity_api_key"] = pplx
    if pplx["status"] == "CRITICAL":
        result["issues"].append(("CRITICAL", f"[perplexity] {pplx['detail']}"))
    elif pplx["status"] == "WARNING":
        result["issues"].append(("WARNING", f"[perplexity] {pplx['detail']}"))
    log.info(f"  Perplexity: {pplx['status']} — {pplx['detail']}")

    # 4. Pipeline staleness
    log.info("Checking pipeline staleness...")
    pipeline = check_pipeline_staleness()
    result["checks"]["pipeline_staleness"] = pipeline
    if pipeline["status"] in ("CRITICAL", "WARNING"):
        result["issues"].append((pipeline["status"], f"[pipeline] {pipeline['detail']}"))
    log.info(f"  Pipeline: {pipeline['status']} — {pipeline['detail']}")

    # 5. Odds fetch staleness
    log.info("Checking odds fetch staleness...")
    odds_fetch = check_odds_fetch_staleness()
    result["checks"]["odds_fetch_staleness"] = odds_fetch
    if odds_fetch["status"] in ("CRITICAL", "WARNING"):
        result["issues"].append((odds_fetch["status"], f"[odds_fetch] {odds_fetch['detail']}"))
    log.info(f"  Odds fetch: {odds_fetch['status']} — {odds_fetch['detail']}")

    # 6. Pending bets
    log.info("Checking pending bets...")
    pending = check_pending_bets()
    result["checks"]["pending_bets"] = pending
    if pending["status"] in ("CRITICAL", "WARNING"):
        result["issues"].append((pending["status"], f"[pending_bets] {pending['detail']}"))
    log.info(f"  Pending bets: {pending['status']} — {pending['detail']}")

    # ─── Determine overall status ───
    if any(level == "CRITICAL" for level, _ in result["issues"]):
        result["overall_status"] = "CRITICAL"
    elif any(level == "WARNING" for level, _ in result["issues"]):
        result["overall_status"] = "WARNING"

    # ─── Persist status ───
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # ─── Notification on CRITICAL (deduped: max once per 6 hours) ───
    if result["overall_status"] == "CRITICAL":
        _dedup_path = DATA_DIR / ".monitor_alert_dedup.json"
        should_notify = True
        try:
            if _dedup_path.exists():
                with open(_dedup_path) as _f:
                    _dedup = json.load(_f)
                last_alert = _dedup.get("last_critical_alert", "")
                if last_alert:
                    from datetime import datetime as _dt
                    hours_since = (
                        _dt.now() - _dt.fromisoformat(last_alert)
                    ).total_seconds() / 3600
                    if hours_since < 6:
                        should_notify = False
                        log.info("Critical alert suppressed (last sent %.1fh ago)", hours_since)
        except Exception:
            pass

        if should_notify:
            critical_msgs = [msg for level, msg in result["issues"] if level == "CRITICAL"]
            send_macos_notification(
                "Serie A Pipeline CRITICAL",
                "; ".join(critical_msgs[:3]),
            )
            try:
                _dedup_path.parent.mkdir(parents=True, exist_ok=True)
                with open(_dedup_path, "w") as _f:
                    json.dump({"last_critical_alert": datetime.now().isoformat()}, _f)
            except Exception:
                pass

    log.info(f"Overall: {result['overall_status']} ({len(result['issues'])} issues)")
    log.info(f"Status written to {STATUS_FILE}")
    log.info("")

    return result


def print_monitor_result(result: Dict):
    """Print formatted monitor result to console."""
    status = result["overall_status"]
    icon = {"HEALTHY": "[OK]", "WARNING": "[!!]", "CRITICAL": "[XX]"}.get(status, "[??]")

    print()
    print("=" * 60)
    print(f"  MONITOR STATUS  {icon} {status}")
    print(f"  {result['timestamp'][:19]}")
    print("=" * 60)

    for name, check in result["checks"].items():
        check_status = check.get("status", "?")
        detail = check.get("detail", "")
        check_icon = {"OK": "[OK]", "HEALTHY": "[OK]", "WARNING": "[!!]", "CRITICAL": "[XX]"}.get(
            check_status, "[??]"
        )
        label = name.replace("_", " ").title()
        print(f"  {check_icon} {label:<28} {detail}")

    issues = result.get("issues", [])
    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for level, msg in issues:
            issue_icon = "[!!]" if level == "WARNING" else "[XX]"
            print(f"    {issue_icon} {msg}")
    else:
        print(f"\n  No issues detected")

    print()
    print("=" * 60)


# ─── CLI ───

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated health monitor")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--quiet", action="store_true", help="Only output on failure")
    args = parser.parse_args()

    result = run_monitor()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif args.quiet:
        if result["overall_status"] != "HEALTHY":
            print_monitor_result(result)
    else:
        print_monitor_result(result)

    # Exit codes
    if result["overall_status"] == "CRITICAL":
        sys.exit(2)
    elif result["overall_status"] == "WARNING":
        sys.exit(1)
    sys.exit(0)
