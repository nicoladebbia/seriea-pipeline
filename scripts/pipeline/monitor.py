#!/usr/bin/env python3
"""Automated health monitor — runs every 30 min via launchd.

Combines the existing health_check with live API validation and staleness
detection so broken keys / stalled pipelines are caught within the hour
instead of after 16 days.

Checks:
  1. All existing health_check checks (data freshness, models, betting, system)
  2. Odds API key alive? (GET /v4/sports/ — costs 0 requests)
  3. Gemini + Groq API keys alive? (minimal ping requests)
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
from scripts.utils.json_utils import load_json_safe

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
log.propagate = False

if not log.handlers:
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
    """Load an API key from environment or .env file.

    Delegates to scripts.pipeline.notify._load_env_key.
    """
    from scripts.pipeline.notify import _load_env_key as _impl

    return _impl(name)


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
                try:
                    remaining_int = int(remaining)
                except (TypeError, ValueError):
                    remaining_int = None
                if remaining_int is not None and remaining_int <= 0:
                    status = "CRITICAL"
                    detail = f"Odds API quota EXHAUSTED ({remaining} requests remaining)"
                elif remaining_int is not None and remaining_int < 50:
                    status = "WARNING"
                    detail = f"Odds API quota low ({remaining} requests remaining)"
                else:
                    status = "OK"
                    detail = f"Key valid, {remaining} requests remaining"
                return {
                    "status": status,
                    "detail": detail,
                    "requests_remaining": remaining,
                }
            return {"status": "WARNING", "detail": f"HTTP {resp.status}"}
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"status": "CRITICAL", "detail": "Odds API key INVALID (401)"}
        return {"status": "WARNING", "detail": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"status": "WARNING", "detail": f"Connection failed: {e}"}


def check_gemini_api_key() -> Dict:
    """Validate the Google Gemini API key with a minimal request."""
    key = _load_env_key("GOOGLE_GEMINI_KEY")
    if not key:
        return {"status": "WARNING", "detail": "GOOGLE_GEMINI_KEY not set (optional)"}

    try:
        from google import genai
        client = genai.Client(api_key=key)
        from google.genai import types as genai_types
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say OK.",
            config=genai_types.GenerateContentConfig(
                max_output_tokens=10,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        if response.text is not None:
            return {"status": "OK", "detail": "Gemini key valid"}
        return {"status": "WARNING", "detail": "Gemini returned empty response"}
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err or "INVALID" in err.upper() or "API_KEY" in err.upper():
            return {"status": "CRITICAL", "detail": f"Gemini key INVALID: {err[:80]}"}
        if "429" in err:
            return {"status": "OK", "detail": "Gemini key valid (rate limited)"}
        return {"status": "WARNING", "detail": f"Gemini check failed: {err[:80]}"}


def check_groq_api_key() -> Dict:
    """Validate the Groq API key with a minimal request."""
    key = _load_env_key("GROQ_API_KEY")
    if not key:
        return {"status": "WARNING", "detail": "GROQ_API_KEY not set (optional)"}

    try:
        from groq import Groq
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        if response.choices:
            return {"status": "OK", "detail": "Groq key valid"}
        return {"status": "WARNING", "detail": "Groq returned empty response"}
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err:
            return {"status": "CRITICAL", "detail": f"Groq key INVALID: {err[:80]}"}
        if "429" in err:
            return {"status": "OK", "detail": "Groq key valid (rate limited)"}
        return {"status": "WARNING", "detail": f"Groq check failed: {err[:80]}"}


def check_pipeline_staleness() -> Dict:
    """Check when the pipeline last ran successfully."""
    state = load_json_safe(DATA_DIR / "pipeline_state.json")

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
    state = load_json_safe(DATA_DIR / "pipeline_state.json")
    usage = load_json_safe(DATA_DIR / "api_usage.json")

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
    """Check for placed bets whose matches should have finished (>3h past kickoff).

    Uses bet_journal.json (source of truth for bet status) instead of
    placed_bets.json (which is a CLV tracking log that persists after settlement).
    """
    # Use journal for actual pending status
    journal = load_json_safe(DATA_DIR / "betting" / "bet_journal.json")
    journal_bets = journal.get("bets", {})
    bets = [v for v in journal_bets.values() if v.get("status") == "pending"]

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



def check_ledger_invariants() -> Dict:
    """Ensure journal ↔ bankroll ↔ history agree. Drift = CRITICAL alert.

    Detects silently-broken state from any code path that bypassed the ledger:
    - direct writes to history.json/bankroll.json
    - a new `void` (vs 'voided') status appearing
    - superseded bets leaking into history
    - settlement-cache bug recurrence (pre-2026 dates)
    """
    try:
        from scripts.betting import ledger
    except Exception as e:
        return {"status": "WARNING", "detail": f"ledger import failed: {e}"}
    try:
        r = ledger.verify_invariants()
    except Exception as e:
        return {"status": "WARNING", "detail": f"verify_invariants crashed: {e}"}
    if r["ok"]:
        return {
            "status": "OK",
            "detail": (
                f"Journal ↔ caches agree "
                f"(balance €{r['journal']['current_balance']:.2f}, "
                f"{r['journal']['settled_count']} settled)"
            ),
            "violations": 0,
        }
    return {
        "status": "CRITICAL",
        "detail": f"Ledger drift detected: {len(r['violations'])} violation(s)",
        "violations": r["violations"][:5],  # cap for payload size
    }


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

    # 3. Web Search API keys (Gemini primary, Groq fallback)
    log.info("Checking Gemini API key...")
    gemini = check_gemini_api_key()
    result["checks"]["gemini_api_key"] = gemini
    if gemini["status"] == "CRITICAL":
        result["issues"].append(("CRITICAL", f"[gemini] {gemini['detail']}"))
    elif gemini["status"] == "WARNING":
        result["issues"].append(("WARNING", f"[gemini] {gemini['detail']}"))
    log.info(f"  Gemini: {gemini['status']} — {gemini['detail']}")

    log.info("Checking Groq API key...")
    groq_check = check_groq_api_key()
    result["checks"]["groq_api_key"] = groq_check
    if groq_check["status"] == "CRITICAL":
        result["issues"].append(("CRITICAL", f"[groq] {groq_check['detail']}"))
    elif groq_check["status"] == "WARNING":
        result["issues"].append(("WARNING", f"[groq] {groq_check['detail']}"))
    log.info(f"  Groq: {groq_check['status']} — {groq_check['detail']}")

    # 3b. API usage limits
    log.info("Checking API usage limits...")
    try:
        from scripts.prediction.sentiment_analyzer import _usage_tracker
        usage = _usage_tracker.get_status()
        result["checks"]["api_usage"] = usage
        for provider, info in usage.items():
            pct = info.get("pct_used", 0)
            remaining = info.get("remaining", 0)
            period = info.get("period", "?")
            if pct >= 95:
                result["issues"].append(("CRITICAL",
                    f"[api_usage] {provider} {period} limit EXHAUSTED: "
                    f"{info['used']}/{info['limit']} ({pct}%) — sentiment will use fallback"))
            elif pct >= 80:
                result["issues"].append(("WARNING",
                    f"[api_usage] {provider} {period} usage high: "
                    f"{info['used']}/{info['limit']} ({pct}%), {remaining} remaining"))
            log.info(f"  {provider}: {info['used']}/{info['limit']} {period} ({pct}% used)")
    except Exception as e:
        log.debug(f"API usage check skipped: {e}")

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

    # 7. Ledger invariants — journal ↔ bankroll.json ↔ history.json agreement
    log.info("Checking ledger invariants...")
    inv = check_ledger_invariants()
    result["checks"]["ledger_invariants"] = inv
    if inv["status"] in ("CRITICAL", "WARNING"):
        result["issues"].append((inv["status"], f"[ledger] {inv['detail']}"))
        for v in inv.get("violations", []):
            result["issues"].append(("WARNING", f"[ledger] {v}"))
    log.info(f"  Ledger: {inv['status']} — {inv['detail']}")

    # ─── Auto-recovery: if pipeline stale >12h, try to run it ───
    if pipeline.get("status") == "CRITICAL":
        age = pipeline.get("age_hours", 0)
        if age > 12:
            _recovery_dedup = DATA_DIR / ".pipeline_recovery_dedup.json"
            should_recover = True
            try:
                if _recovery_dedup.exists():
                    rd = load_json_safe(_recovery_dedup)
                    last_attempt = rd.get("last_attempt", "")
                    if last_attempt:
                        hours_since = _iso_age_hours(last_attempt)
                        if hours_since < 6:  # Don't retry more than once per 6 hours
                            should_recover = False
            except Exception:
                pass

            if should_recover:
                log.info("AUTO-RECOVERY: Pipeline stale >12h — triggering emergency run")
                try:
                    import subprocess
                    cmd = [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "pipeline" / "run_full_pipeline.py"),
                        "--quick",
                    ]
                    proc = subprocess.Popen(
                        cmd, cwd=str(PROJECT_ROOT),
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    log.info(f"Emergency pipeline started (PID {proc.pid})")
                    result["auto_recovery"] = {"triggered": True, "pid": proc.pid}

                    # Write dedup AFTER Popen succeeds — if Popen fails, we can
                    # retry on the next monitor cycle instead of being blocked for 6h
                    _recovery_dedup.parent.mkdir(parents=True, exist_ok=True)
                    with open(_recovery_dedup, "w") as _f:
                        json.dump({"last_attempt": datetime.now().isoformat()}, _f)

                    _unified_notify(
                        f"Pipeline was stale ({age:.0f}h). Auto-recovery triggered.",
                        title="Auto-Recovery: Pipeline Restart",
                        level="warning", category="alert",
                    )
                except Exception as e:
                    log.error(f"Auto-recovery failed: {e}")
                    result["auto_recovery"] = {"triggered": False, "error": str(e)}

    # ─── Determine overall status ───
    if any(level == "CRITICAL" for level, _ in result["issues"]):
        result["overall_status"] = "CRITICAL"
    elif any(level == "WARNING" for level, _ in result["issues"]):
        result["overall_status"] = "WARNING"

    # ─── Persist status ───
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # NOTE (2026-04-24): The old "Serie A Pipeline CRITICAL" macOS alert path
    # was removed. It duplicated the far-better `notify_health_state_change`
    # card (fires with proper issue-key dedup + flap suppression) and was
    # causing redundant noise. State-change alerts are now handled exclusively
    # by notify_health_state_change() below.

    log.info(f"Overall: {result['overall_status']} ({len(result['issues'])} issues)")
    log.info(f"Status written to {STATUS_FILE}")
    log.info("")

    # Fire state-change alert (silent on first-run and on no-change)
    try:
        from scripts.pipeline.notify import notify_health_state_change
        notify_health_state_change(result)
    except Exception as e:
        log.debug(f"Health state-change notify skipped: {e}")

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
