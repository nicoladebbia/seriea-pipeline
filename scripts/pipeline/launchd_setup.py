#!/usr/bin/env python3
"""Generate macOS launchd plists for automated Serie A pipeline.

Launchd is more reliable than cron on macOS — it handles sleep/wake,
retries missed jobs, and integrates with the system scheduler.

Usage:
    python -m scripts.pipeline.launchd_setup               # Generate plists
    python -m scripts.pipeline.launchd_setup --install      # Generate + install
    python -m scripts.pipeline.launchd_setup --uninstall    # Remove plists
    python -m scripts.pipeline.launchd_setup --status       # Check if loaded
"""

import argparse
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_PATH = sys.executable
SCHEDULER_PATH = PROJECT_ROOT / "scripts" / "pipeline" / "scheduler.py"
PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
LOG_DIR = PROJECT_ROOT / "logs"

LABEL_PREFIX = "com.seriea-pipeline"

# Define the jobs
JOBS = [
    {
        "name": "morning",
        "label": f"{LABEL_PREFIX}.morning",
        "description": "Morning data refresh + predictions",
        "mode": "once",
        "hour": 8,
        "minute": 0,
        "bankroll": 1000,
    },
    {
        "name": "evening",
        "label": f"{LABEL_PREFIX}.evening",
        "description": "Evening odds update + betting slip",
        "mode": "once",
        "hour": 20,
        "minute": 0,
        "bankroll": 1000,
    },
    {
        "name": "pre-kickoff-monitor",
        "label": f"{LABEL_PREFIX}.pre-kickoff-monitor",
        "description": "Smart pre-kickoff: checks actual match times every 30 min",
        "mode": "pre-kickoff-monitor",
        "interval_seconds": 1800,  # Every 30 minutes
        "bankroll": 1000,
    },
    {
        "name": "settlement",
        "label": f"{LABEL_PREFIX}.settlement",
        "description": "Post-match auto-settlement + reconciliation + risk check",
        "mode": "settle",
        "hour": 23,
        "minute": 30,
        "bankroll": 1000,
    },
    {
        "name": "weekly-monitor",
        "label": f"{LABEL_PREFIX}.weekly-monitor",
        "description": "Weekly drift detection, calibration check, retrain evaluation",
        "mode": "monitor",
        "weekday": 0,  # Sunday (0=Sun in launchd)
        "hour": 9,
        "minute": 0,
    },
    {
        "name": "matchweek-retrain",
        "label": f"{LABEL_PREFIX}.matchweek-retrain",
        "description": "Nightly check: if matchweek complete, retrain models (quick or full auto-detect)",
        "mode": "matchweek-retrain",
        "hour": 0,
        "minute": 15,
        "custom_args": [str(PYTHON_PATH), "-m", "scripts.pipeline.weekly_retrain"],
    },
    {
        "name": "health-monitor",
        "label": f"{LABEL_PREFIX}.health-monitor",
        "description": "Health monitor: API keys, pipeline staleness, pending bets (every 30 min)",
        "mode": "health-monitor",
        "interval_seconds": 1800,  # Every 30 minutes
        "custom_args": [str(PYTHON_PATH), "-m", "scripts.pipeline.monitor", "--quiet"],
    },
    {
        "name": "telegram-bot",
        "label": f"{LABEL_PREFIX}.telegram-bot",
        "description": "Telegram chat bot — AI advisor on your phone",
        "mode": "telegram-bot",
        "custom_args": [str(PYTHON_PATH), "-m", "scripts.pipeline.telegram_bot"],
        "keep_alive": True,  # Restart if it crashes
    },
    {
        "name": "daily-digest",
        "label": f"{LABEL_PREFIX}.daily-digest",
        "description": "Daily betting summary notification at 23:00",
        "mode": "daily-digest",
        "hour": 23,
        "minute": 0,
        "custom_args": [str(PYTHON_PATH), "-m", "scripts.pipeline.notify", "--digest"],
    },
]


def generate_plist(job: dict) -> dict:
    """Generate a launchd plist dict for a job."""
    log_out = str(LOG_DIR / f"launchd-{job['name']}.log")
    log_err = str(LOG_DIR / f"launchd-{job['name']}-err.log")

    if job.get("custom_args"):
        args = job["custom_args"]
    else:
        args = [
            str(PYTHON_PATH),
            str(SCHEDULER_PATH),
            job["mode"],
        ]
        if job.get("bankroll"):
            args.extend(["--bankroll", str(job["bankroll"])])

    plist = {
        "Label": job["label"],
        "ProgramArguments": args,
        "WorkingDirectory": str(PROJECT_ROOT),
        "StandardOutPath": log_out,
        "StandardErrorPath": log_err,
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(Path.home()),
            "PYTHONPATH": str(PROJECT_ROOT),
        },
        # Run even if user is not logged in (requires load with -w flag)
        "RunAtLoad": bool(job.get("keep_alive")),
        # Retry on failure — keep_alive jobs auto-restart, others don't
        "KeepAlive": bool(job.get("keep_alive")),
        "ThrottleInterval": 10 if job.get("keep_alive") else 300,
    }

    # Schedule: keep_alive daemons run continuously (no schedule needed),
    # interval-based jobs use StartInterval, others use StartCalendarInterval
    if job.get("keep_alive"):
        # Continuous daemon — RunAtLoad + KeepAlive handles it
        pass
    elif "interval_seconds" in job:
        plist["StartInterval"] = job["interval_seconds"]
    else:
        cal = {
            "Hour": job["hour"],
            "Minute": job["minute"],
        }
        # Weekly jobs (Weekday: 0=Sunday in launchd)
        if "weekday" in job:
            cal["Weekday"] = job["weekday"]
        # Monthly jobs (Day: 1-31)
        if "day" in job:
            cal["Day"] = job["day"]
        plist["StartCalendarInterval"] = cal

    # Add API keys from environment, falling back to .env file
    env_file = PROJECT_ROOT / ".env"
    dotenv_vars = {}
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    dotenv_vars[k.strip()] = v.strip()

    for key in ("ODDS_API_KEY", "APIFOOTBALL_KEY", "SLACK_WEBHOOK_URL",
                "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "ANTHROPIC_API_KEY"):
        val = os.environ.get(key) or dotenv_vars.get(key)
        if val:
            plist["EnvironmentVariables"][key] = val

    return plist


def generate_all():
    """Generate all plist files."""
    PLIST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Generated launchd plists:")
    print()

    for job in JOBS:
        plist = generate_plist(job)
        plist_path = PLIST_DIR / f"{job['label']}.plist"

        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)

        print(f"  {plist_path.name}")
        print(f"    {job['description']}")
        if job.get("keep_alive"):
            print(f"    Schedule: always-on (auto-restart)")
        elif "interval_seconds" in job:
            print(f"    Schedule: every {job['interval_seconds'] // 60} minutes")
        elif "weekday" in job:
            days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            print(f"    Schedule: {days[job['weekday']]}s at {job['hour']:02d}:{job['minute']:02d}")
        elif "day" in job:
            print(f"    Schedule: {job['day']}st of each month at {job['hour']:02d}:{job['minute']:02d}")
        else:
            print(f"    Schedule: {job['hour']:02d}:{job['minute']:02d} daily")
        print(f"    Command: {job['mode']}")
        print()

    return [PLIST_DIR / f"{job['label']}.plist" for job in JOBS]


def install():
    """Generate and load all plists."""
    paths = generate_all()

    print("Loading plists...")
    for path in paths:
        label = path.stem
        # Unload first (ignore errors if not loaded)
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
            capture_output=True
        )
        # Load
        result = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Loaded: {label}")
        else:
            # Fallback to legacy load
            result2 = subprocess.run(
                ["launchctl", "load", str(path)],
                capture_output=True, text=True
            )
            if result2.returncode == 0:
                print(f"  Loaded (legacy): {label}")
            else:
                print(f"  FAILED: {label} — {result.stderr.strip() or result2.stderr.strip()}")

    print()
    print("Done. Check status with: python -m scripts.pipeline.launchd_setup --status")


def uninstall():
    """Unload and remove all plists."""
    for job in JOBS:
        label = job["label"]
        plist_path = PLIST_DIR / f"{label}.plist"

        # Unload
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
            capture_output=True
        )

        # Remove file
        if plist_path.exists():
            plist_path.unlink()
            print(f"  Removed: {label}")
        else:
            print(f"  Not found: {label}")

    print("\nAll pipeline jobs uninstalled.")


def status():
    """Check status of all pipeline launchd jobs."""
    print("Launchd job status:")
    print()

    result = subprocess.run(
        ["launchctl", "list"],
        capture_output=True, text=True
    )

    for job in JOBS:
        label = job["label"]
        plist_path = PLIST_DIR / f"{label}.plist"
        installed = plist_path.exists()
        loaded = label in result.stdout

        status_str = "LOADED" if loaded else ("installed" if installed else "not installed")
        if job.get("keep_alive"):
            schedule = "always-on"
        elif "interval_seconds" in job:
            schedule = f"every {job['interval_seconds'] // 60}m"
        elif "weekday" in job:
            days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            schedule = f"{days[job['weekday']]} {job['hour']:02d}:{job['minute']:02d}"
        elif "day" in job:
            schedule = f"D{job['day']:02d} {job['hour']:02d}:{job['minute']:02d}"
        else:
            schedule = f"{job['hour']:02d}:{job['minute']:02d}"
        print(f"  {label:45s} {status_str:15s} {schedule:12s} [{job['mode']}]")

        # Check last log
        log_file = LOG_DIR / f"launchd-{job['name']}.log"
        if log_file.exists():
            stat = log_file.stat()
            from datetime import datetime
            mtime = datetime.fromtimestamp(stat.st_mtime)
            size = stat.st_size
            print(f"    Last log: {mtime.strftime('%Y-%m-%d %H:%M')} ({size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="macOS launchd setup for Serie A pipeline")
    parser.add_argument("--install", action="store_true", help="Generate and load plists")
    parser.add_argument("--uninstall", action="store_true", help="Remove all plists")
    parser.add_argument("--status", action="store_true", help="Check job status")
    args = parser.parse_args()

    if args.uninstall:
        uninstall()
    elif args.install:
        install()
    elif args.status:
        status()
    else:
        generate_all()
        print("To install, run with --install flag.")
        print("To check cron alternative: python -m scripts.pipeline.scheduler cron-setup")


if __name__ == "__main__":
    main()
