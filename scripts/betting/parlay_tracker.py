#!/usr/bin/env python3
"""Parlay Performance Tracker — records generated parlays and tracks outcomes.

Tracks top parlay picks over time, matches them against settlement results,
and provides stats that feed back into the historical combo analysis engine.

Usage:
    from scripts.betting.parlay_tracker import record_parlays, settle_parlays, get_parlay_stats
"""

import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

HISTORY_FILE = DATA_DIR / "betting" / "parlay_history.jsonl"
STATS_FILE = DATA_DIR / "betting" / "parlay_stats.json"


def record_parlays(top_picks: list):
    """Save top picks to parlay history (JSONL, one record per pick).

    Called by generate_parlay_report() after selecting top picks.
    """
    if not top_picks:
        return

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    with open(HISTORY_FILE, "a") as f:
        for pick in top_picks:
            parlay = pick.get("parlay", {})
            legs = parlay.get("legs", [])
            record = {
                "timestamp": timestamp,
                "rank": pick.get("rank"),
                "recommendation_score": pick.get("recommendation_score"),
                "category": pick.get("category", ""),
                "parlay_id": parlay.get("id", ""),
                "n_legs": parlay.get("n_legs", len(legs)),
                "combined_odds": parlay.get("combined_odds", 1),
                "stake": parlay.get("stake", 0),
                "hit_probability": parlay.get("hit_probability", {}),
                "dc_anchored": parlay.get("dc_anchored", False),
                "legs": [
                    {
                        "match": l.get("match", ""),
                        "market": l.get("market", ""),
                        "selection": l.get("selection", ""),
                        "odds": l.get("odds", 0),
                        "probability": l.get("probability", 0),
                    }
                    for l in legs
                ],
                "why": pick.get("why", []),
                "status": "pending",
                "profit": None,
            }
            f.write(json.dumps(record) + "\n")


def _load_history() -> list:
    """Load all parlay history records."""
    if not HISTORY_FILE.exists():
        return []
    records = []
    with open(HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _save_history(records: list):
    """Overwrite history file with updated records."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def settle_parlays(results: dict):
    """Match settled results against saved parlays and record outcomes.

    Args:
        results: dict mapping match_key -> {"score": "2-1", "home_goals": 2, "away_goals": 1}

    Called by auto_settle.py after processing results.
    """
    records = _load_history()
    if not records:
        return 0

    settled_count = 0

    for record in records:
        if record.get("status") != "pending":
            continue

        legs = record.get("legs", [])
        if not legs:
            continue

        # Check if all legs in this parlay have results
        all_settled = True
        all_won = True

        for leg in legs:
            match = leg.get("match", "")
            result = results.get(match)
            if not result:
                all_settled = False
                break

            # Normalize score field names (API uses home_score, tracker uses home_goals)
            normalized = {
                "home_goals": result.get("home_goals") or result.get("home_score"),
                "away_goals": result.get("away_goals") or result.get("away_score"),
            }

            won = _check_leg_result(leg, normalized)
            if won is None:
                all_settled = False
                break
            if not won:
                all_won = False

        if all_settled:
            stake = record.get("stake", 0)
            combined_odds = record.get("combined_odds", 1)
            if all_won:
                record["status"] = "won"
                record["profit"] = round(stake * (combined_odds - 1), 2)
            else:
                record["status"] = "lost"
                record["profit"] = -stake
            record["settled_at"] = datetime.now().isoformat()
            settled_count += 1

    if settled_count > 0:
        _save_history(records)
        _update_stats(records)

    return settled_count


def _check_leg_result(leg: dict, result: dict) -> bool | None:
    """Check if a single leg won given the match result.

    Returns True (won), False (lost), or None (can't determine).
    """
    selection = leg.get("selection", "").upper()
    market = leg.get("market", "")
    home_goals = result.get("home_goals")
    away_goals = result.get("away_goals")

    if home_goals is None or away_goals is None:
        return None

    total_goals = home_goals + away_goals

    if market == "h2h":
        if "HOME" in selection:
            return home_goals > away_goals
        elif "AWAY" in selection:
            return away_goals > home_goals
        elif "DRAW" in selection:
            return home_goals == away_goals

    elif market == "double_chance":
        if "1X" in selection:
            return home_goals >= away_goals
        elif "X2" in selection:
            return away_goals >= home_goals
        elif "12" in selection:
            return home_goals != away_goals

    elif market == "totals":
        if "OVER" in selection:
            # Extract line number
            parts = selection.split()
            for p in parts:
                try:
                    line = float(p)
                    return total_goals > line
                except ValueError:
                    continue
        elif "UNDER" in selection:
            parts = selection.split()
            for p in parts:
                try:
                    line = float(p)
                    return total_goals < line
                except ValueError:
                    continue

    elif market == "btts":
        both_scored = home_goals > 0 and away_goals > 0
        if "YES" in selection:
            return both_scored
        elif "NO" in selection:
            return not both_scored

    return None


def _update_stats(records: list):
    """Compute and save aggregate parlay stats."""
    settled = [r for r in records if r.get("status") in ("won", "lost")]
    if not settled:
        return

    total = len(settled)
    won = sum(1 for r in settled if r["status"] == "won")
    total_profit = sum(r.get("profit", 0) for r in settled)
    total_staked = sum(r.get("stake", 0) for r in settled)

    # Stats by category
    cat_stats = {}
    for r in settled:
        cat = r.get("category", "unknown")
        if cat not in cat_stats:
            cat_stats[cat] = {"won": 0, "total": 0, "profit": 0, "staked": 0}
        cat_stats[cat]["total"] += 1
        cat_stats[cat]["staked"] += r.get("stake", 0)
        if r["status"] == "won":
            cat_stats[cat]["won"] += 1
        cat_stats[cat]["profit"] += r.get("profit", 0)

    for cat, s in cat_stats.items():
        s["win_rate"] = round(s["won"] / s["total"] * 100, 1) if s["total"] > 0 else 0
        s["roi"] = round(s["profit"] / s["staked"] * 100, 1) if s["staked"] > 0 else 0

    # Stats by combo type (market combinations)
    combo_stats = {}
    for r in settled:
        markets = sorted(set(l.get("market", "") for l in r.get("legs", [])))
        combo_key = "+".join(markets)
        if combo_key not in combo_stats:
            combo_stats[combo_key] = {"won": 0, "total": 0, "profit": 0}
        combo_stats[combo_key]["total"] += 1
        if r["status"] == "won":
            combo_stats[combo_key]["won"] += 1
        combo_stats[combo_key]["profit"] += r.get("profit", 0)

    for ck, s in combo_stats.items():
        s["win_rate"] = round(s["won"] / s["total"] * 100, 1) if s["total"] > 0 else 0

    # DC anchor stats
    dc_settled = [r for r in settled if r.get("dc_anchored")]
    dc_won = sum(1 for r in dc_settled if r["status"] == "won")

    stats = {
        "updated_at": datetime.now().isoformat(),
        "overall": {
            "total": total,
            "won": won,
            "win_rate": round(won / total * 100, 1) if total > 0 else 0,
            "profit": round(total_profit, 2),
            "staked": round(total_staked, 2),
            "roi": round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0,
        },
        "by_category": cat_stats,
        "by_combo_type": combo_stats,
        "dc_anchored": {
            "total": len(dc_settled),
            "won": dc_won,
            "win_rate": round(dc_won / len(dc_settled) * 100, 1) if dc_settled else 0,
        },
    }

    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def get_parlay_stats() -> dict:
    """Load and return parlay performance stats."""
    if STATS_FILE.exists():
        try:
            with open(STATS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


if __name__ == "__main__":
    stats = get_parlay_stats()
    if stats:
        overall = stats.get("overall", {})
        print(f"Parlay Stats: {overall.get('total', 0)} settled, "
              f"{overall.get('win_rate', 0)}% WR, "
              f"ROI {overall.get('roi', 0)}%")
        print(f"  P&L: {overall.get('profit', 0):+.2f} on {overall.get('staked', 0):.2f} staked")

        by_cat = stats.get("by_category", {})
        if by_cat:
            print("\nBy Category:")
            for cat, s in by_cat.items():
                print(f"  {cat}: {s['won']}/{s['total']} ({s['win_rate']}% WR)")

        dc = stats.get("dc_anchored", {})
        if dc.get("total", 0) > 0:
            print(f"\nDC Anchored: {dc['won']}/{dc['total']} ({dc['win_rate']}% WR)")
    else:
        print("No parlay stats yet. Run the pipeline to start tracking.")
