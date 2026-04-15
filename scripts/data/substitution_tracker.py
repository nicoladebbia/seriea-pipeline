"""Extract and persist substitution data from live matchday JSONs.

Reads completed match data from data/live/*.json, extracts substitution events,
and appends to a historical ledger. This data feeds future feature engineering:
- Team substitution patterns (avg minute, count per game)
- Manager tactical flexibility metrics
- Player fatigue/rotation signals

Called from _run_settle() after match completion.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from scripts.utils.ledger import load_json_ledger, save_json_ledger

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent.parent
LIVE_DIR = BASE / "data" / "live"
SUBS_FILE = BASE / "data" / "lineup_history" / "substitutions.json"


def extract_substitutions(date_str: str = None) -> dict:
    """Extract substitution data from a matchday JSON.

    Args:
        date_str: Date string like '2026-02-20'. If None, uses today.

    Returns:
        dict with 'extracted' count and 'matches' summary.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    live_file = LIVE_DIR / f"{date_str}.json"
    if not live_file.exists():
        log.debug("No live data for %s", date_str)
        return {"extracted": 0, "date": date_str}

    with open(live_file) as f:
        matchday = json.load(f)

    matches = matchday.get("matches", {})
    if not matches:
        return {"extracted": 0, "date": date_str}

    # Load existing ledger
    ledger = load_json_ledger(SUBS_FILE)
    existing_keys = {r["match_key"] for r in ledger}

    new_records = []
    for mk, mdata in matches.items():
        if mk in existing_keys:
            continue

        status = mdata.get("status", "")
        if status != "completed":
            continue

        events = mdata.get("live_events", [])
        subs = [e for e in events if e.get("type") == "substitution"]

        if not subs:
            continue

        home_team = mk.split(" vs ")[0] if " vs " in mk else ""
        away_team = mk.split(" vs ")[1] if " vs " in mk else ""

        home_subs = []
        away_subs = []
        for s in subs:
            sub_record = {
                "player_out": s.get("player_out", ""),
                "player_in": s.get("player_in", ""),
                "minute": s.get("minute", 0),
                "added_time": s.get("added_time", 0),
            }
            if s.get("is_home"):
                home_subs.append(sub_record)
            else:
                away_subs.append(sub_record)

        # Final score
        score = mdata.get("final_score") or mdata.get("score", [0, 0])

        record = {
            "match_key": mk,
            "date": date_str,
            "home_team": home_team,
            "away_team": away_team,
            "score": score,
            "home_subs": home_subs,
            "away_subs": away_subs,
            "home_sub_count": len(home_subs),
            "away_sub_count": len(away_subs),
            "home_avg_sub_minute": (
                round(sum(s["minute"] for s in home_subs) / len(home_subs), 1)
                if home_subs else None
            ),
            "away_avg_sub_minute": (
                round(sum(s["minute"] for s in away_subs) / len(away_subs), 1)
                if away_subs else None
            ),
            "extracted_at": datetime.now().isoformat(),
        }

        # Also capture confirmed lineups if available
        player_stats = mdata.get("live_player_stats", {})
        if player_stats:
            for side in ("home", "away"):
                starters = [
                    p["name"] for p in player_stats.get(side, [])
                    if not p.get("substitute", False)
                ]
                bench_used = [
                    p["name"] for p in player_stats.get(side, [])
                    if p.get("substitute", False)
                    and p.get("minutes_played", 0) > 0
                ]
                record[f"{side}_starters"] = starters
                record[f"{side}_bench_used"] = bench_used

        new_records.append(record)

    if new_records:
        ledger.extend(new_records)
        save_json_ledger(SUBS_FILE, ledger)
        log.info("Persisted %d substitution records for %s", len(new_records), date_str)

    return {
        "extracted": len(new_records),
        "date": date_str,
        "matches": [r["match_key"] for r in new_records],
    }


def get_team_sub_patterns(team: str, last_n: int = 10) -> dict:
    """Get substitution patterns for a team (for feature engineering).

    Returns:
        dict with avg_subs_per_game, avg_first_sub_minute, typical_sub_minutes.
    """
    ledger = load_json_ledger(SUBS_FILE)
    records = []
    for r in reversed(ledger):
        if r["home_team"] == team:
            records.append({
                "count": r["home_sub_count"],
                "avg_minute": r["home_avg_sub_minute"],
                "subs": r["home_subs"],
            })
        elif r["away_team"] == team:
            records.append({
                "count": r["away_sub_count"],
                "avg_minute": r["away_avg_sub_minute"],
                "subs": r["away_subs"],
            })
        if len(records) >= last_n:
            break

    if not records:
        return {}

    counts = [r["count"] for r in records]
    avg_minutes = [r["avg_minute"] for r in records if r["avg_minute"] is not None]
    first_sub_minutes = []
    for r in records:
        if r["subs"]:
            first_sub_minutes.append(min(s["minute"] for s in r["subs"]))

    return {
        "avg_subs_per_game": round(sum(counts) / len(counts), 1),
        "avg_sub_minute": round(sum(avg_minutes) / len(avg_minutes), 1) if avg_minutes else None,
        "avg_first_sub_minute": round(sum(first_sub_minutes) / len(first_sub_minutes), 1) if first_sub_minutes else None,
        "games_tracked": len(records),
    }
