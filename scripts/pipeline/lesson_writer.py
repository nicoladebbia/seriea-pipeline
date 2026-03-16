#!/usr/bin/env python3
"""LESSON WRITER - Persistent Corrections from Post-Match Analysis

Reads prediction_audit.json and roi_report.json (from learning_loop.py),
generates structured lesson entries that the ensemble reads at prediction time.

Lesson types:
  - xg_bias: team-specific xG correction
  - confidence_shift: recalibrate confidence band
  - market_penalty: downweight market method for specific conditions
  - method_boost: temporarily boost a method's weight

Safety caps enforced. Lessons auto-expire after N matches.
Lesson effectiveness tracked and underperformers auto-deactivated.

Usage:
    python3 -m scripts.lesson_writer
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEEDBACK_DIR = DATA_DIR / "feedback"
LESSONS_PATH = FEEDBACK_DIR / "lessons.json"

# Safety caps
MAX_XG_ADJUST = 0.5
MAX_CONFIDENCE_SHIFT = 0.10  # 10pp
MAX_WEIGHT_ADJUST = 0.05     # 5pp
DEFAULT_EXPIRY = 20           # matches
MIN_APPLICATIONS_FOR_REVIEW = 10
MIN_HELP_RATE = 0.4


def _load_lessons() -> dict:
    """Load existing lessons file."""
    if LESSONS_PATH.exists():
        try:
            with open(LESSONS_PATH) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load lessons file: {e}")
    return {"lessons": [], "metadata": {"version": 1, "created": datetime.now().isoformat()}}


def _save_lessons(data: dict):
    """Save lessons file."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    data["metadata"]["updated"] = datetime.now().isoformat()
    with open(LESSONS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _next_lesson_id(lessons: list) -> str:
    """Generate next lesson ID."""
    existing_ids = [l.get("id", "") for l in lessons]
    max_num = 0
    for lid in existing_ids:
        if lid.startswith("lesson_"):
            try:
                max_num = max(max_num, int(lid.split("_")[1]))
            except (ValueError, IndexError) as e:
                log.debug(f"Failed to parse lesson ID number: {e}")
    return f"lesson_{max_num + 1:03d}"


def _generate_xg_bias_lessons(audit: dict, existing_ids: set) -> list:
    """Generate xG bias lessons from per-team calibration data."""
    new_lessons = []
    calibration = audit.get("per_team_calibration", {})

    for team, stats in calibration.items():
        n = stats.get("matches", 0)
        if n < 3:
            continue

        mean_bias = stats.get("xg_mean_bias", 0)
        mae = stats.get("xg_mae", 0)

        # Only create lesson if bias is significant (>0.25 xG) and MAE is notably worse than average
        if abs(mean_bias) < 0.25:
            continue

        # Check if a lesson for this team already exists
        scope_key = f"xg_bias_{team}"
        if scope_key in existing_ids:
            continue

        correction = -mean_bias  # Counter the bias
        correction = max(-MAX_XG_ADJUST, min(MAX_XG_ADJUST, correction))

        # Determine if bias is for home or away xG (or both)
        # If positive bias -> we overestimate this team -> reduce xG
        lesson = {
            "id": None,  # Filled by caller
            "created": datetime.now().isoformat()[:10],
            "type": "xg_bias",
            "scope": {"team": team},
            "correction": {"team_xg_adjust": round(correction, 2)},
            "evidence": f"{n} matches, MAE {mae:.2f}, mean bias {mean_bias:+.2f}",
            "confidence": min(0.9, 0.3 + n * 0.05),  # Confidence grows with sample size
            "expires_after_n": DEFAULT_EXPIRY,
            "applied_count": 0,
            "helped_count": 0,
            "active": True,
            "_scope_key": scope_key,
        }
        new_lessons.append(lesson)

    return new_lessons


def _generate_confidence_lessons(audit: dict, existing_ids: set) -> list:
    """Generate confidence band recalibration lessons."""
    new_lessons = []
    bands = audit.get("confidence_bands", {})

    # Expected hit rates for each band
    expected = {
        "VERY HIGH": 0.65,
        "HIGH": 0.55,
        "MEDIUM-HIGH": 0.45,
        "MEDIUM": 0.38,
        "LOW": 0.30,
    }

    for level, stats in bands.items():
        n = stats.get("total", 0)
        if n < 5:
            continue

        hit_rate = stats.get("hit_rate", 0)
        exp_rate = expected.get(level, 0.40)
        gap = hit_rate - exp_rate

        # Only flag if band is off by >10pp
        if abs(gap) < 0.10:
            continue

        scope_key = f"confidence_shift_{level}"
        if scope_key in existing_ids:
            continue

        shift = max(-MAX_CONFIDENCE_SHIFT, min(MAX_CONFIDENCE_SHIFT, gap / 2))

        lesson = {
            "id": None,
            "created": datetime.now().isoformat()[:10],
            "type": "confidence_shift",
            "scope": {"confidence_level": level},
            "correction": {"confidence_adjust": round(shift, 3)},
            "evidence": f"{n} predictions, hit rate {hit_rate:.1%} vs expected {exp_rate:.1%}",
            "confidence": min(0.8, 0.3 + n * 0.03),
            "expires_after_n": DEFAULT_EXPIRY * 2,  # Confidence lessons last longer
            "applied_count": 0,
            "helped_count": 0,
            "active": True,
            "_scope_key": scope_key,
        }
        new_lessons.append(lesson)

    return new_lessons


def _generate_market_lessons(roi_report: dict, existing_ids: set) -> list:
    """Generate market-specific lessons from ROI data."""
    new_lessons = []
    by_market = roi_report.get("by_market", {})

    for market, stats in by_market.items():
        n = stats.get("bets", 0)
        if n < 5:
            continue

        roi = stats.get("roi_pct", 0)

        # If a market is consistently losing badly, add a penalty lesson
        if roi < -15:
            scope_key = f"market_penalty_{market}"
            if scope_key in existing_ids:
                continue

            # Penalty proportional to how bad ROI is, capped
            penalty = min(MAX_WEIGHT_ADJUST, abs(roi) / 100 * 0.05)

            lesson = {
                "id": None,
                "created": datetime.now().isoformat()[:10],
                "type": "market_penalty",
                "scope": {"market_type": market},
                "correction": {"confidence_reduce": round(penalty, 3)},
                "evidence": f"{n} bets in {market}, ROI {roi:+.1f}%",
                "confidence": min(0.7, 0.3 + n * 0.03),
                "expires_after_n": DEFAULT_EXPIRY,
                "applied_count": 0,
                "helped_count": 0,
                "active": True,
                "_scope_key": scope_key,
            }
            new_lessons.append(lesson)

    return new_lessons


def generate_lessons() -> dict:
    """Generate new lessons from audit and ROI data. Update effectiveness of existing lessons.

    Returns status dict for the learning loop.
    """
    log.info("Generating lessons from feedback data...")

    # Load inputs
    audit_path = FEEDBACK_DIR / "prediction_audit.json"
    roi_path = FEEDBACK_DIR / "roi_report.json"

    audit = {}
    if audit_path.exists():
        try:
            with open(audit_path) as f:
                audit = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load audit data for lesson writing: {e}")

    roi_report = {}
    if roi_path.exists():
        try:
            with open(roi_path) as f:
                roi_report = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load ROI report for lesson writing: {e}")

    if not audit and not roi_report:
        return {"status": "ok", "summary": "no feedback data available yet"}

    # Load existing lessons
    lessons_data = _load_lessons()
    existing = lessons_data["lessons"]

    # Build set of existing scope keys (to avoid duplicates)
    existing_scope_keys = set()
    for l in existing:
        sk = l.get("_scope_key", "")
        if sk:
            existing_scope_keys.add(sk)

    # Generate new lessons
    new_lessons = []
    if audit:
        new_lessons.extend(_generate_xg_bias_lessons(audit, existing_scope_keys))
        new_lessons.extend(_generate_confidence_lessons(audit, existing_scope_keys))
    if roi_report:
        new_lessons.extend(_generate_market_lessons(roi_report, existing_scope_keys))

    # Assign IDs and add to existing
    for lesson in new_lessons:
        lesson["id"] = _next_lesson_id(existing)
        existing.append(lesson)

    # Expire old lessons
    expired_count = 0
    for l in existing:
        if not l.get("active", True):
            continue
        if l.get("applied_count", 0) >= l.get("expires_after_n", DEFAULT_EXPIRY):
            l["active"] = False
            l["deactivated_reason"] = "expired"
            expired_count += 1

    # Deactivate underperformers
    deactivated_count = 0
    for l in existing:
        if not l.get("active", True):
            continue
        applied = l.get("applied_count", 0)
        helped = l.get("helped_count", 0)
        if applied >= MIN_APPLICATIONS_FOR_REVIEW:
            help_rate = helped / applied
            if help_rate < MIN_HELP_RATE:
                l["active"] = False
                l["deactivated_reason"] = f"low help rate ({help_rate:.1%})"
                deactivated_count += 1

    lessons_data["lessons"] = existing
    _save_lessons(lessons_data)

    active_count = sum(1 for l in existing if l.get("active", True))
    summary_parts = [f"{len(new_lessons)} new"]
    if expired_count:
        summary_parts.append(f"{expired_count} expired")
    if deactivated_count:
        summary_parts.append(f"{deactivated_count} deactivated")
    summary_parts.append(f"{active_count} active total")

    log.info(f"  Lessons: {', '.join(summary_parts)}")

    return {
        "status": "ok",
        "summary": ", ".join(summary_parts),
        "new_count": len(new_lessons),
        "active_count": active_count,
    }


def update_lesson_effectiveness(match_name: str, predicted_outcome: str,
                                actual_outcome: str, lessons_applied: list):
    """After results settle, update helped_count for applied lessons.

    Called by the learning loop after settlement.
    """
    if not lessons_applied:
        return

    lessons_data = _load_lessons()
    lesson_map = {l["id"]: l for l in lessons_data["lessons"]}

    was_correct = predicted_outcome.upper() == actual_outcome.upper()

    for lid in lessons_applied:
        if lid in lesson_map:
            if was_correct:
                lesson_map[lid]["helped_count"] = lesson_map[lid].get("helped_count", 0) + 1

    _save_lessons(lessons_data)


if __name__ == "__main__":
    result = generate_lessons()
    print(json.dumps(result, indent=2))
