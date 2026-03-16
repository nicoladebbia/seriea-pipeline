#!/usr/bin/env python3
"""PRODUCTION HEALTH CHECK — Quick system-wide status in one command.

Lightweight check that runs in seconds (no API calls, no heavy computation).
Designed for daily monitoring and CI integration.

Checks:
  1. Data freshness — features.parquet, odds, predictions, results
  2. Model freshness — are models stale?
  3. Betting performance — ROI, win rate, drift alerts from auto_settle
  4. System integrity — missing files, broken imports

Usage:
    python -m scripts.pipeline.health_check          # Full health check
    python -m scripts.pipeline.health_check --json   # Machine-readable output

Exit codes: 0=HEALTHY, 1=WARNING, 2=CRITICAL
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, MODELS_DIR

# ─── Staleness thresholds ───
MAX_FEATURES_AGE_DAYS = 7       # Features should be rebuilt weekly
MAX_MODEL_AGE_DAYS = 30         # Models should be retrained monthly
MAX_ODDS_AGE_HOURS = 48         # Odds should be fresh if matches upcoming
MAX_PREDICTIONS_AGE_HOURS = 48  # Predictions should be recent before matchday


def _file_age(path: Path) -> Tuple[float, str]:
    """Return (age_hours, human_readable_age) for a file, or (-1, 'missing')."""
    if not path.exists():
        return -1, "MISSING"
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    delta = datetime.now() - mtime
    hours = delta.total_seconds() / 3600

    if hours < 1:
        return hours, f"{int(delta.total_seconds() / 60)}m ago"
    elif hours < 24:
        return hours, f"{hours:.1f}h ago"
    else:
        days = hours / 24
        return hours, f"{days:.1f}d ago"


def check_data_freshness() -> Dict:
    """Check freshness of key data files."""
    checks = {}

    files = {
        "features.parquet": DATA_DIR / "features" / "features.parquet",
        "odds_data": DATA_DIR / "upcoming" / "odds_data.json",
        "predictions": DATA_DIR / "upcoming" / "predictions.json",
        "results": DATA_DIR / "upcoming" / "results.json",
        "unified_report": DATA_DIR / "betting" / "unified_report.json",
        "bet_journal": DATA_DIR / "betting" / "bet_journal.json",
    }

    for name, path in files.items():
        age_hours, age_str = _file_age(path)
        status = "OK"

        if age_hours < 0:
            status = "MISSING"
        elif name == "features.parquet" and age_hours > MAX_FEATURES_AGE_DAYS * 24:
            status = "STALE"
        elif name == "odds_data" and age_hours > MAX_ODDS_AGE_HOURS:
            status = "STALE"
        elif name == "predictions" and age_hours > MAX_PREDICTIONS_AGE_HOURS:
            status = "STALE"

        checks[name] = {
            "path": str(path),
            "exists": path.exists(),
            "age": age_str,
            "age_hours": round(age_hours, 1),
            "status": status,
        }

    return checks


def check_model_freshness() -> Dict:
    """Check model files exist and aren't stale."""
    checks = {}

    models = {
        "catboost_no_odds": MODELS_DIR / "universal" / "catboost_no_odds.cbm",
        "ensemble_calibrator": MODELS_DIR / "universal" / "ensemble_calibrator.pkl",
    }

    # Check for any .cbm/.xgb/.lgb files in models dir
    universal_dir = MODELS_DIR / "universal"
    if universal_dir.exists():
        for ext in ["*.cbm", "*.xgb", "*.lgb"]:
            for f in universal_dir.glob(ext):
                models[f.stem] = f

    for name, path in models.items():
        age_hours, age_str = _file_age(path)
        status = "OK"

        if age_hours < 0:
            status = "MISSING"
        elif age_hours > MAX_MODEL_AGE_DAYS * 24:
            status = "STALE"

        checks[name] = {
            "exists": path.exists(),
            "age": age_str,
            "status": status,
        }

    return checks


def check_betting_health() -> Dict:
    """Check betting system health from journal stats and drift alerts."""
    result = {"status": "OK", "details": {}}

    # Journal stats
    try:
        from scripts.betting.bet_journal import get_journal_stats
        stats = get_journal_stats()
        result["details"]["journal"] = {
            "total_bets": stats.get("total_bets", 0),
            "pending": stats.get("pending", 0),
            "settled": stats.get("settled", 0),
            "roi_pct": stats.get("roi_pct", 0),
            "total_profit": stats.get("total_profit", 0),
            "clv_avg_pct": stats.get("clv_avg_pct", 0),
        }
    except Exception as e:
        result["details"]["journal"] = {"error": str(e)}

    # Drift alerts from auto_settle
    drift_file = DATA_DIR / "betting" / "drift_alerts.json"
    if drift_file.exists():
        try:
            with open(drift_file) as f:
                drift_data = json.load(f)
            result["details"]["drift"] = {
                "checked_at": drift_data.get("checked_at", "?"),
                "has_critical": drift_data.get("has_critical", False),
                "has_warning": drift_data.get("has_warning", False),
                "alerts": [
                    {"level": a["level"], "message": a["message"]}
                    for a in drift_data.get("alerts", [])
                    if a["level"] in ("CRITICAL", "WARNING")
                ],
            }
            if drift_data.get("has_critical"):
                result["status"] = "CRITICAL"
            elif drift_data.get("has_warning"):
                result["status"] = "WARNING"
        except Exception:
            pass

    # Bankroll
    bankroll_file = DATA_DIR / "betting" / "bankroll.json"
    if bankroll_file.exists():
        try:
            with open(bankroll_file) as f:
                bankroll = json.load(f)
            result["details"]["bankroll"] = {
                "current": bankroll.get("current_balance", 0),
                "initial": bankroll.get("initial_balance", 0),
            }
        except Exception:
            pass

    return result


def check_system_integrity() -> Dict:
    """Check critical imports and file system state."""
    checks = {}

    # Check critical imports
    imports_ok = True
    for module_name in [
        "catboost",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
    ]:
        try:
            __import__(module_name)
            checks[module_name] = "OK"
        except ImportError:
            checks[module_name] = "MISSING"
            imports_ok = False

    # Check config
    try:
        from config.settings import DATA_DIR, MODELS_DIR, PROJECT_ROOT
        checks["config"] = "OK"
    except Exception as e:
        checks["config"] = f"ERROR: {e}"

    # Check features.parquet has expected columns
    features_path = DATA_DIR / "features" / "features.parquet"
    if features_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(features_path, columns=["match_date", "home_team", "away_team"])
            checks["features_readable"] = f"OK ({len(df)} rows)"
        except Exception as e:
            checks["features_readable"] = f"ERROR: {e}"

    return {"imports_ok": imports_ok, "checks": checks}


def run_health_check() -> Dict:
    """Run all health checks and return unified result."""
    result = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "HEALTHY",
        "data_freshness": check_data_freshness(),
        "model_freshness": check_model_freshness(),
        "betting_health": check_betting_health(),
        "system_integrity": check_system_integrity(),
        "issues": [],
    }

    # Determine overall status
    issues = []

    # Data issues
    for name, check in result["data_freshness"].items():
        if check["status"] == "MISSING":
            issues.append(("WARNING", f"Data file missing: {name}"))
        elif check["status"] == "STALE":
            issues.append(("WARNING", f"Data file stale: {name} ({check['age']})"))

    # Model issues
    for name, check in result["model_freshness"].items():
        if check["status"] == "MISSING":
            issues.append(("CRITICAL", f"Model missing: {name}"))
        elif check["status"] == "STALE":
            issues.append(("WARNING", f"Model stale: {name} ({check['age']})"))

    # Betting issues
    betting_status = result["betting_health"].get("status", "OK")
    if betting_status == "CRITICAL":
        drift_alerts = result["betting_health"]["details"].get("drift", {}).get("alerts", [])
        for a in drift_alerts:
            issues.append(("CRITICAL", a["message"]))
    elif betting_status == "WARNING":
        drift_alerts = result["betting_health"]["details"].get("drift", {}).get("alerts", [])
        for a in drift_alerts:
            issues.append(("WARNING", a["message"]))

    # System issues
    if not result["system_integrity"].get("imports_ok", True):
        issues.append(("CRITICAL", "Missing critical Python packages"))

    result["issues"] = issues

    # Overall status
    if any(level == "CRITICAL" for level, _ in issues):
        result["overall_status"] = "CRITICAL"
    elif any(level == "WARNING" for level, _ in issues):
        result["overall_status"] = "WARNING"

    return result


def print_health_check(result: Dict):
    """Print formatted health check to console."""
    status = result["overall_status"]
    status_icon = {"HEALTHY": "[OK]", "WARNING": "[!!]", "CRITICAL": "[XX]"}.get(status, "[??]")

    print()
    print("=" * 60)
    print(f"  SYSTEM HEALTH CHECK  {status_icon} {status}")
    print(f"  {result['timestamp'][:19]}")
    print("=" * 60)

    # Data freshness
    print("\n  DATA FRESHNESS:")
    for name, check in result["data_freshness"].items():
        icon = "[OK]" if check["status"] == "OK" else "[!!]" if check["status"] == "STALE" else "[XX]"
        print(f"    {icon} {name:<22} {check['age']:>12}")

    # Models
    print("\n  MODELS:")
    for name, check in result["model_freshness"].items():
        icon = "[OK]" if check["status"] == "OK" else "[!!]" if check["status"] == "STALE" else "[XX]"
        print(f"    {icon} {name:<22} {check['age']:>12}")

    # Betting
    print("\n  BETTING:")
    journal = result["betting_health"].get("details", {}).get("journal", {})
    if journal and not journal.get("error"):
        print(f"    Bets: {journal.get('total_bets', 0)} total "
              f"({journal.get('pending', 0)} pending, {journal.get('settled', 0)} settled)")
        print(f"    ROI:  {journal.get('roi_pct', 0):+.1f}%  |  "
              f"P&L: ${journal.get('total_profit', 0):+,.2f}")
        clv = journal.get("clv_avg_pct", 0)
        if clv:
            print(f"    CLV:  {clv:+.1f}%")
    bankroll = result["betting_health"].get("details", {}).get("bankroll", {})
    if bankroll:
        current = bankroll.get("current", 0)
        initial = bankroll.get("initial", 0)
        growth = ((current / initial) - 1) * 100 if initial > 0 else 0
        print(f"    Bank: ${current:,.2f} (${initial:,.2f} initial, {growth:+.1f}%)")

    # System
    print("\n  SYSTEM:")
    integrity = result["system_integrity"]
    for pkg, status in integrity.get("checks", {}).items():
        if status == "OK" or status.startswith("OK"):
            continue  # Only show issues
        print(f"    [!!] {pkg}: {status}")
    if all(s == "OK" or s.startswith("OK") for s in integrity.get("checks", {}).values()):
        print(f"    [OK] All packages and configs verified")

    # Issues summary
    issues = result.get("issues", [])
    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for level, msg in issues:
            icon = "[!!]" if level == "WARNING" else "[XX]"
            print(f"    {icon} {msg}")
    else:
        print(f"\n  No issues detected")

    print()
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Production health check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = run_health_check()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_health_check(result)

    # Exit codes
    if result["overall_status"] == "CRITICAL":
        sys.exit(2)
    elif result["overall_status"] == "WARNING":
        sys.exit(1)
    sys.exit(0)
