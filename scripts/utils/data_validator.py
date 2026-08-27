"""Post-pipeline data validator — catches data integrity issues before they reach users.

Run automatically after every pipeline execution. Checks:
  1. League consistency (predictions ↔ odds same league)
  2. Team name coverage (all predictions have matching odds)
  3. Probability integrity (sum to 1.0, range [0,1])
  4. Odds sanity (> 1.0, < 100)
  5. File staleness (critical files not too old)
  6. Field completeness (required fields present)
  7. Cross-file coverage (supplementary data covers all predictions)

Usage:
    from scripts.utils.data_validator import validate_pipeline_output
    issues = validate_pipeline_output()
    # issues = {"critical": [...], "warning": [...], "info": [...]}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.utils.json_utils import load_json_safe

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _BASE / "data"
UPCOMING_DIR = DATA_DIR / "upcoming"

# Maximum age in hours before data is considered stale
FRESHNESS_LIMITS = {
    "predictions.json": 24,
    "predictions_premier_league.json": 24,
    "odds_full.json": 12,
    "odds_full_premier_league.json": 12,
    "odds_movement.json": 12,
    "current_form.json": 48,
    "standings.json": 48,
    "market_intelligence.json": 24,
    "sentiment_analysis.json": 48,
    "lineup_predictions.json": 48,
    "weather.json": 48,
    "bookmaker_analysis.json": 24,
}

# Required fields per prediction entry
REQUIRED_PREDICTION_FIELDS = [
    "home_team", "away_team", "predicted_outcome",
    "probabilities", "confidence", "date",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_match_keys(data: Any) -> set[str]:
    """Extract match keys from various JSON formats."""
    keys = set()
    if isinstance(data, dict):
        if "matches" in data:
            m = data["matches"]
            if isinstance(m, dict):
                keys = set(m.keys())
            elif isinstance(m, list):
                for item in m:
                    if isinstance(item, dict):
                        mk = item.get("match", "")
                        if not mk:
                            h = item.get("home_team", "")
                            a = item.get("away_team", "")
                            if h and a:
                                mk = f"{h} vs {a}"
                        if mk:
                            keys.add(mk)
        elif "predictions" in data:
            for item in data["predictions"]:
                if isinstance(item, dict):
                    mk = item.get("match", "")
                    if mk:
                        keys.add(mk)
        elif "matchups" in data:
            keys = set(data["matchups"].keys())
        else:
            for k in data:
                if isinstance(k, str) and " vs " in k:
                    keys.add(k)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                mk = item.get("match", "")
                if not mk:
                    h = item.get("home_team", "")
                    a = item.get("away_team", "")
                    if h and a:
                        mk = f"{h} vs {a}"
                if mk:
                    keys.add(mk)
    return keys


def _classify_league(match_keys: set[str]) -> dict[str, set[str]]:
    """Classify match keys into leagues using team name registry."""
    try:
        from config.team_names import SERIE_A_NAMES, PREMIER_LEAGUE_NAMES
        sa_canon = set(SERIE_A_NAMES.values())
        epl_canon = set(PREMIER_LEAGUE_NAMES.values())
    except ImportError:
        return {"unknown": match_keys}

    from config.leagues import ACTIVE_LEAGUES
    result: dict[str, set[str]] = {lg: set() for lg in ACTIVE_LEAGUES}
    result["unknown"] = set()
    for mk in match_keys:
        parts = mk.split(" vs ")
        if len(parts) != 2:
            result["unknown"].add(mk)
            continue
        h, a = parts
        if h in sa_canon or a in sa_canon:
            result["serie_a"].add(mk)
        elif h in epl_canon or a in epl_canon:
            result["premier_league"].add(mk)
        else:
            result["unknown"].add(mk)
    return result


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _check_league_consistency(issues: dict[str, list]) -> None:
    """Check that odds_full.json matches the same league as predictions.json."""
    preds = load_json_safe(UPCOMING_DIR / "predictions.json")
    odds = load_json_safe(UPCOMING_DIR / "odds_full.json")

    pred_keys = _extract_match_keys(preds)
    odds_keys = _extract_match_keys(odds)

    if not pred_keys or not odds_keys:
        return

    overlap = pred_keys & odds_keys
    if not overlap:
        pred_leagues = _classify_league(pred_keys)
        odds_leagues = _classify_league(odds_keys)
        pred_main = max(pred_leagues, key=lambda k: len(pred_leagues[k]))
        odds_main = max(odds_leagues, key=lambda k: len(odds_leagues[k]))
        if pred_main != odds_main:
            issues["critical"].append(
                f"LEAGUE MISMATCH: predictions.json is {pred_main} "
                f"but odds_full.json is {odds_main}. Zero match overlap."
            )
        else:
            issues["warning"].append(
                f"predictions.json and odds_full.json have 0 overlapping match keys "
                f"despite both being {pred_main}. Check team name normalization."
            )
    elif len(overlap) < len(pred_keys) * 0.5:
        issues["warning"].append(
            f"Only {len(overlap)}/{len(pred_keys)} prediction matches found in odds_full.json."
        )


def _check_probability_integrity(issues: dict[str, list]) -> None:
    """Check all predictions have valid probabilities."""
    for fname in ["predictions.json", "predictions_premier_league.json"]:
        data = load_json_safe(UPCOMING_DIR / fname)
        if not data:
            continue
        preds = data.get("predictions", data) if isinstance(data, dict) else data
        if not isinstance(preds, list):
            continue

        for p in preds:
            mk = p.get("match", "?")
            probs = p.get("probabilities", {})
            h = probs.get("home", 0)
            d = probs.get("draw", 0)
            a = probs.get("away", 0)
            s = h + d + a

            if abs(s - 1.0) > 0.02:
                issues["critical"].append(f"{fname}: {mk} probability sum = {s:.4f}")

            for label, val in [("home", h), ("draw", d), ("away", a)]:
                if val < 0 or val > 1:
                    issues["critical"].append(f"{fname}: {mk} invalid {label} prob = {val}")


def _check_odds_sanity(issues: dict[str, list]) -> None:
    """Check odds are in valid range."""
    for fname in ["odds_full.json", "odds_full_premier_league.json"]:
        data = load_json_safe(UPCOMING_DIR / fname)
        matches = data.get("matches", {}) if isinstance(data, dict) else {}
        if not isinstance(matches, dict):
            continue

        for mk, m in matches.items():
            h2h = m.get("h2h", {})
            if not isinstance(h2h, dict):
                continue
            for label in ["home", "draw", "away"]:
                val = h2h.get(label)
                if val is not None:
                    if val < 1.0:
                        issues["critical"].append(f"{fname}: {mk} odds.{label} = {val} (below 1.0)")
                    elif val > 100:
                        issues["warning"].append(f"{fname}: {mk} odds.{label} = {val} (>100, suspicious)")


def _check_staleness(issues: dict[str, list]) -> None:
    """Check critical files are not too old."""
    now = datetime.now()
    for fname, max_age_h in FRESHNESS_LIMITS.items():
        fpath = UPCOMING_DIR / fname
        if not fpath.exists():
            if fname in ("predictions.json", "odds_full.json"):
                issues["critical"].append(f"{fname}: FILE MISSING")
            continue

        mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
        age_h = (now - mtime).total_seconds() / 3600
        if age_h > max_age_h:
            level = "critical" if fname in ("predictions.json", "odds_full.json") else "warning"
            issues[level].append(f"{fname}: {age_h:.1f}h old (limit {max_age_h}h)")


def _check_field_completeness(issues: dict[str, list]) -> None:
    """Check predictions have all required fields."""
    for fname in ["predictions.json", "predictions_premier_league.json"]:
        data = load_json_safe(UPCOMING_DIR / fname)
        if not data:
            continue
        preds = data.get("predictions", data) if isinstance(data, dict) else data
        if not isinstance(preds, list):
            continue

        for p in preds:
            mk = p.get("match", p.get("home_team", "?"))
            for field in REQUIRED_PREDICTION_FIELDS:
                if not p.get(field):
                    issues["warning"].append(f"{fname}: {mk} missing field '{field}'")


def _check_cross_coverage(issues: dict[str, list]) -> None:
    """Check supplementary files cover all prediction matches — per league.

    Odds files are per-league; comparing a mixed predictions file against the
    Serie A odds file alone is how the Aug-24 league-contamination bug showed
    up as a permanent (and misread) "missing 18/32" warning. Each prediction
    file is checked against ITS league's odds file.
    """
    league_pairs = [
        ("predictions.json", "odds_full.json"),
        ("predictions_premier_league.json", "odds_full_premier_league.json"),
    ]
    supplementary = {
        "weather.json": "info",
        "current_form.json": "warning",
        "market_intelligence.json": "info",
        "sentiment_analysis.json": "info",
    }

    for pred_fname, odds_fname in league_pairs:
        preds = load_json_safe(UPCOMING_DIR / pred_fname)
        pred_keys = _extract_match_keys(preds)
        if not pred_keys:
            continue

        checks = {odds_fname: "critical"}
        if pred_fname == "predictions.json":
            checks.update(supplementary)  # supplementary files are SA-shaped

        for fname, level in checks.items():
            data = load_json_safe(UPCOMING_DIR / fname)
            file_keys = _extract_match_keys(data)
            missing = pred_keys - file_keys
            if missing and len(missing) > len(pred_keys) * 0.25:
                issues[level].append(
                    f"{fname}: missing {len(missing)}/{len(pred_keys)} "
                    f"{pred_fname} matches"
                )


def _check_odds_league_tag(issues: dict[str, list]) -> None:
    """Ensure odds files have explicit league tags to prevent future mix-ups."""
    for fname in ["odds_full.json", "odds_full_premier_league.json"]:
        data = load_json_safe(UPCOMING_DIR / fname)
        if isinstance(data, dict) and data.get("matches"):
            league = data.get("league", "")
            if not league:
                issues["warning"].append(f"{fname}: no 'league' field — risk of cross-league contamination")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_pipeline_output() -> dict[str, list[str]]:
    """Run all validators and return categorized issues.

    Returns:
        {"critical": [...], "warning": [...], "info": [...]}
    """
    issues: dict[str, list[str]] = {"critical": [], "warning": [], "info": []}

    _check_league_consistency(issues)
    _check_probability_integrity(issues)
    _check_odds_sanity(issues)
    _check_staleness(issues)
    _check_field_completeness(issues)
    _check_cross_coverage(issues)
    _check_odds_league_tag(issues)

    # Log results
    total = sum(len(v) for v in issues.values())
    if issues["critical"]:
        log.error("DATA VALIDATION: %d critical issues found!", len(issues["critical"]))
        for msg in issues["critical"]:
            log.error("  CRITICAL: %s", msg)
    if issues["warning"]:
        log.warning("DATA VALIDATION: %d warnings", len(issues["warning"]))
        for msg in issues["warning"]:
            log.warning("  WARNING: %s", msg)
    if issues["info"]:
        for msg in issues["info"]:
            log.info("  INFO: %s", msg)
    if total == 0:
        log.info("DATA VALIDATION: All checks passed ✓")

    return issues


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    issues = validate_pipeline_output()
    print(f"\nSummary: {len(issues['critical'])} critical, {len(issues['warning'])} warnings, {len(issues['info'])} info")
