#!/usr/bin/env python3
"""Player prop outcome tracker — settlement, ROI tracking, and calibration.

After matches complete, evaluates all player prop value bets against actual
Sofascore per-player stats. Maintains a persistent ledger of outcomes for:
  - Hit rate tracking per market type and tier
  - ROI calculation (Kelly-staked)
  - Model calibration (predicted prob vs actual hit rate)
  - Edge decay analysis (did high-edge bets actually hit more?)

Data flow:
  1. Match completes → live_monitor has final player stats in matchday JSON
  2. settle_props() reads player_prop_value_bets.json + live player stats
  3. Evaluates each prop → HIT / LOST / NO_DATA
  4. Appends results to data/betting/prop_ledger.json (persistent)
  5. Generates summary stats → data/betting/prop_performance.json

Called from:
  - web/app.py settle endpoint (auto-triggered on match completion)
  - CLI: python scripts/betting/prop_tracker.py --settle
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import DATA_DIR, UPCOMING_DIR

log = logging.getLogger(__name__)

BETTING_DIR = DATA_DIR / "betting"
LIVE_DIR = DATA_DIR / "live"

LEDGER_PATH = BETTING_DIR / "prop_ledger.json"
PERFORMANCE_PATH = BETTING_DIR / "prop_performance.json"


def _load_ledger() -> List[Dict]:
    """Load persistent prop ledger."""
    if LEDGER_PATH.exists():
        try:
            with open(LEDGER_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save_ledger(ledger: List[Dict]):
    """Save prop ledger."""
    BETTING_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def _evaluate_prop_outcome(market: str, pstats: dict) -> dict:
    """Evaluate a prop bet outcome given final player stats.

    Returns {"status": "hit"|"lost"|"void", "actual": value, "line": line}
    """
    market_lower = market.lower()
    import re

    def _extract_line(s, default=0.5):
        m = re.search(r"(\d+\.?\d*)", s)
        return float(m.group(1)) if m else default

    # Anytime Goalscorer
    if "anytime" in market_lower and "goal" in market_lower:
        goals = pstats.get("goals", 0)
        return {"status": "hit" if goals >= 1 else "lost", "actual": goals, "line": 0.5}

    # 2+ Goals
    if ("2+" in market_lower or "two" in market_lower) and "goal" in market_lower:
        goals = pstats.get("goals", 0)
        return {"status": "hit" if goals >= 2 else "lost", "actual": goals, "line": 1.5}

    # First / Last Goalscorer — can't settle from stats alone, mark void
    if ("first" in market_lower or "last" in market_lower) and "goal" in market_lower:
        return {"status": "void", "actual": pstats.get("goals", 0), "line": None}

    # SOT Over X.5
    if "sot" in market_lower or ("shot" in market_lower and "target" in market_lower):
        sot = pstats.get("shots_on_target", 0)
        line = _extract_line(market_lower)
        return {"status": "hit" if sot > line else "lost", "actual": sot, "line": line}

    # Shots Over X.5
    if "shot" in market_lower and "target" not in market_lower:
        shots = pstats.get("shots", 0)
        line = _extract_line(market_lower)
        return {"status": "hit" if shots > line else "lost", "actual": shots, "line": line}

    # Tackles Over X.5
    if "tackle" in market_lower:
        tackles = pstats.get("tackles", 0)
        line = _extract_line(market_lower)
        return {"status": "hit" if tackles > line else "lost", "actual": tackles, "line": line}

    # Fouls Over X.5
    if "foul" in market_lower:
        fouls = pstats.get("fouls_committed", 0)
        line = _extract_line(market_lower)
        return {"status": "hit" if fouls > line else "lost", "actual": fouls, "line": line}

    # To Be Carded
    if "card" in market_lower or "booked" in market_lower:
        cards = pstats.get("yellow_cards", 0) + pstats.get("red_cards", 0)
        return {"status": "hit" if cards >= 1 else "lost", "actual": cards, "line": 0.5}

    # Assists
    if "assist" in market_lower:
        assists = pstats.get("assists", 0)
        line = _extract_line(market_lower)
        return {"status": "hit" if assists > line else "lost", "actual": assists, "line": line}

    # Key Passes
    if "key" in market_lower and "pass" in market_lower:
        kp = pstats.get("key_passes", 0)
        line = _extract_line(market_lower)
        return {"status": "hit" if kp > line else "lost", "actual": kp, "line": line}

    # Crosses
    if "cross" in market_lower:
        crosses = pstats.get("crosses", 0)
        line = _extract_line(market_lower)
        return {"status": "hit" if crosses > line else "lost", "actual": crosses, "line": line}

    return {"status": "void", "actual": None, "line": None}


def settle_props(date_str: str = None) -> Dict:
    """Settle all player prop bets for completed matches on a given date.

    Reads live matchday data for final player stats, evaluates all prop value
    bets, and appends results to the persistent ledger.

    Returns summary dict with hit counts, ROI, etc.
    """
    import unicodedata

    def _strip_accents(s):
        return "".join(c for c in unicodedata.normalize("NFD", s)
                       if unicodedata.category(c) != "Mn")

    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Load live matchday data
    live_path = LIVE_DIR / f"{date_str}.json"
    if not live_path.exists():
        return {"settled": 0, "message": f"No live data for {date_str}"}

    with open(live_path) as f:
        live_data = json.load(f)

    # Load prop value bets
    prop_vb_path = UPCOMING_DIR / "player_prop_value_bets.json"
    if not prop_vb_path.exists():
        return {"settled": 0, "message": "No prop value bets file"}

    with open(prop_vb_path) as f:
        prop_vb = json.load(f)
    all_bets = prop_vb.get("bets", [])

    if not all_bets:
        return {"settled": 0, "message": "No prop bets to settle"}

    # Load existing ledger to check for already-settled bets
    ledger = _load_ledger()
    settled_keys = set()
    for entry in ledger:
        k = f"{entry.get('date')}|{entry.get('player')}|{entry.get('market')}|{entry.get('match')}"
        settled_keys.add(k)

    # Normalize team names for matching
    try:
        from config.team_names import normalize_team as _nt
    except ImportError:
        def _nt(s): return s

    # Process completed matches
    completed_matches = {}
    for mk, md in live_data.get("matches", {}).items():
        if md.get("status") != "completed":
            continue
        player_stats = md.get("live_player_stats", {})
        if not player_stats:
            continue
        completed_matches[mk] = {
            "player_stats": player_stats,
            "final_score": md.get("final_score", [0, 0]),
        }

    if not completed_matches:
        return {"settled": 0, "message": "No completed matches with player stats"}

    # Build player stat lookups for each completed match
    match_lookups = {}
    for mk, md in completed_matches.items():
        lookup = {}
        for side in ("home", "away"):
            for p in md["player_stats"].get(side, []):
                name = p.get("name", "")
                if name:
                    lookup[name.lower()] = p
                    lookup[_strip_accents(name).lower()] = p
                    short = p.get("short_name", "")
                    if short:
                        lookup[short.lower()] = p
                        lookup[_strip_accents(short).lower()] = p
                    parts = name.split()
                    if len(parts) > 1:
                        lookup[parts[-1].lower()] = p
                        lookup[_strip_accents(parts[-1]).lower()] = p
        # Store lookup with both raw and normalized match keys
        match_lookups[mk] = lookup
        parts = mk.split(" vs ", 1)
        if len(parts) == 2:
            norm_mk = f"{_nt(parts[0].strip())} vs {_nt(parts[1].strip())}"
            match_lookups[norm_mk] = lookup

    # Settle each prop bet
    new_entries = []
    for bet in all_bets:
        bet_match = bet.get("match", "")
        player_name = bet.get("player", "")
        market = bet.get("market", "")

        # Check if already settled
        dedup_key = f"{date_str}|{player_name}|{market}|{bet_match}"
        if dedup_key in settled_keys:
            continue

        # Find player stats lookup for this match
        lookup = match_lookups.get(bet_match)
        if not lookup:
            # Try normalized match name
            bm_parts = bet_match.split(" vs ", 1)
            if len(bm_parts) == 2:
                norm_bm = f"{_nt(bm_parts[0].strip())} vs {_nt(bm_parts[1].strip())}"
                lookup = match_lookups.get(norm_bm)
        if not lookup:
            continue

        # Find player
        pstats = lookup.get(player_name.lower())
        if not pstats:
            pstats = lookup.get(_strip_accents(player_name).lower())
        if not pstats:
            parts = player_name.split()
            if len(parts) > 1:
                pstats = lookup.get(parts[-1].lower())
                if not pstats:
                    pstats = lookup.get(_strip_accents(parts[-1]).lower())

        if not pstats:
            # Player not in lineup — mark as void
            new_entries.append({
                "date": date_str,
                "match": bet_match,
                "player": player_name,
                "team": bet.get("team", ""),
                "position": bet.get("position", ""),
                "market": market,
                "model_prob": bet.get("model_prob", 0),
                "implied_prob": bet.get("implied_prob", 0),
                "best_odds": bet.get("best_odds", 0),
                "edge_pct": bet.get("edge_pct", 0),
                "kelly_fraction": bet.get("kelly_fraction", 0),
                "tier": bet.get("tier", ""),
                "best_bookmaker": bet.get("best_bookmaker", ""),
                "outcome": "void",
                "actual": None,
                "line": None,
                "settled_at": datetime.now(timezone.utc).isoformat(),
            })
            continue

        # Evaluate outcome
        result = _evaluate_prop_outcome(market, pstats)

        entry = {
            "date": date_str,
            "match": bet_match,
            "player": player_name,
            "team": bet.get("team", ""),
            "position": bet.get("position", ""),
            "market": market,
            "model_prob": bet.get("model_prob", 0),
            "implied_prob": bet.get("implied_prob", 0),
            "best_odds": bet.get("best_odds", 0),
            "edge_pct": bet.get("edge_pct", 0),
            "kelly_fraction": bet.get("kelly_fraction", 0),
            "tier": bet.get("tier", ""),
            "best_bookmaker": bet.get("best_bookmaker", ""),
            "outcome": result["status"],
            "actual": result["actual"],
            "line": result["line"],
            "settled_at": datetime.now(timezone.utc).isoformat(),
        }
        new_entries.append(entry)

    # Append to ledger and save
    if new_entries:
        ledger.extend(new_entries)
        _save_ledger(ledger)
        log.info("Settled %d prop bets for %s", len(new_entries), date_str)

    # Generate summary
    hits = [e for e in new_entries if e["outcome"] == "hit"]
    losses = [e for e in new_entries if e["outcome"] == "lost"]
    voids = [e for e in new_entries if e["outcome"] == "void"]

    # Calculate ROI (assuming $1 unit stake per bet)
    total_staked = len(hits) + len(losses)  # void doesn't count
    total_return = sum(e["best_odds"] for e in hits) if hits else 0
    roi = ((total_return - total_staked) / total_staked * 100) if total_staked > 0 else 0

    summary = {
        "date": date_str,
        "total_settled": len(new_entries),
        "hits": len(hits),
        "losses": len(losses),
        "voids": len(voids),
        "hit_rate": round(len(hits) / total_staked * 100, 1) if total_staked > 0 else 0,
        "roi_pct": round(roi, 1),
        "total_staked": total_staked,
        "total_return": round(total_return, 2),
    }

    # Update performance summary
    _update_performance(ledger)

    return summary


def _update_performance(ledger: List[Dict]):
    """Generate aggregated performance stats from the full ledger."""
    if not ledger:
        return

    # Filter to non-void entries
    settled = [e for e in ledger if e["outcome"] in ("hit", "lost")]
    if not settled:
        return

    # Overall stats
    hits = [e for e in settled if e["outcome"] == "hit"]
    total = len(settled)
    total_return = sum(e.get("best_odds", 0) for e in hits)
    roi = (total_return - total) / total * 100 if total > 0 else 0

    # By market type
    by_market = {}
    for e in settled:
        m = e.get("market", "Unknown")
        # Normalize market name
        m_lower = m.lower()
        if "anytime" in m_lower and "goal" in m_lower:
            cat = "Anytime Goalscorer"
        elif "shot" in m_lower and ("target" in m_lower or "sot" in m_lower.split()):
            cat = "Shots On Target"
        elif "shot" in m_lower:
            cat = "Shots"
        elif "tackle" in m_lower:
            cat = "Tackles"
        elif "foul" in m_lower:
            cat = "Fouls"
        elif "card" in m_lower or "booked" in m_lower:
            cat = "Cards"
        elif "assist" in m_lower:
            cat = "Assists"
        else:
            cat = m

        if cat not in by_market:
            by_market[cat] = {"total": 0, "hits": 0, "return": 0}
        by_market[cat]["total"] += 1
        if e["outcome"] == "hit":
            by_market[cat]["hits"] += 1
            by_market[cat]["return"] += e.get("best_odds", 0)

    for cat, stats in by_market.items():
        t = stats["total"]
        h = stats["hits"]
        stats["hit_rate"] = round(h / t * 100, 1) if t > 0 else 0
        stats["roi_pct"] = round((stats["return"] - t) / t * 100, 1) if t > 0 else 0

    # By tier
    by_tier = {}
    for e in settled:
        tier = e.get("tier", "?")
        if tier not in by_tier:
            by_tier[tier] = {"total": 0, "hits": 0, "return": 0}
        by_tier[tier]["total"] += 1
        if e["outcome"] == "hit":
            by_tier[tier]["hits"] += 1
            by_tier[tier]["return"] += e.get("best_odds", 0)

    for tier, stats in by_tier.items():
        t = stats["total"]
        h = stats["hits"]
        stats["hit_rate"] = round(h / t * 100, 1) if t > 0 else 0
        stats["roi_pct"] = round((stats["return"] - t) / t * 100, 1) if t > 0 else 0

    # By edge bucket
    edge_buckets = {"0-10%": [], "10-20%": [], "20-30%": [], "30%+": []}
    for e in settled:
        edge = e.get("edge_pct", 0)
        if edge >= 30:
            edge_buckets["30%+"].append(e)
        elif edge >= 20:
            edge_buckets["20-30%"].append(e)
        elif edge >= 10:
            edge_buckets["10-20%"].append(e)
        else:
            edge_buckets["0-10%"].append(e)

    by_edge = {}
    for bucket, entries in edge_buckets.items():
        if not entries:
            continue
        h = len([e for e in entries if e["outcome"] == "hit"])
        t = len(entries)
        r = sum(e.get("best_odds", 0) for e in entries if e["outcome"] == "hit")
        by_edge[bucket] = {
            "total": t,
            "hits": h,
            "hit_rate": round(h / t * 100, 1),
            "roi_pct": round((r - t) / t * 100, 1),
        }

    # Calibration: model_prob buckets vs actual hit rate
    cal_buckets = {}
    for e in settled:
        prob = e.get("model_prob", 0)
        bucket = f"{int(prob * 10) * 10}-{int(prob * 10) * 10 + 10}%"
        if bucket not in cal_buckets:
            cal_buckets[bucket] = {"total": 0, "hits": 0, "avg_prob": 0}
        cal_buckets[bucket]["total"] += 1
        cal_buckets[bucket]["avg_prob"] += prob
        if e["outcome"] == "hit":
            cal_buckets[bucket]["hits"] += 1

    for bucket, stats in cal_buckets.items():
        t = stats["total"]
        stats["avg_prob"] = round(stats["avg_prob"] / t * 100, 1) if t > 0 else 0
        stats["actual_rate"] = round(stats["hits"] / t * 100, 1) if t > 0 else 0
        stats["calibration_error"] = round(stats["actual_rate"] - stats["avg_prob"], 1)

    performance = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_bets": total,
        "total_hits": len(hits),
        "hit_rate": round(len(hits) / total * 100, 1),
        "roi_pct": round(roi, 1),
        "total_return": round(total_return, 2),
        "by_market": by_market,
        "by_tier": by_tier,
        "by_edge": by_edge,
        "calibration": cal_buckets,
    }

    BETTING_DIR.mkdir(parents=True, exist_ok=True)
    with open(PERFORMANCE_PATH, "w") as f:
        json.dump(performance, f, indent=2)

    log.info("Prop performance: %d bets, %.1f%% hit rate, %.1f%% ROI",
             total, performance["hit_rate"], performance["roi_pct"])


def get_calibration_adjustments(min_samples: int = 20) -> Dict[str, float]:
    """Compute per-market calibration multipliers from the settlement ledger.

    Returns a dict like:
      {"Anytime Goalscorer": 0.92, "Shots": 1.05, ...}

    Meaning: multiply model probability by this factor to get calibrated probability.
    Only returns adjustments for markets with >= min_samples settled bets.

    Used by player_prop_odds.py value scanner to correct systematic model biases.
    """
    ledger = _load_ledger()
    settled = [e for e in ledger if e["outcome"] in ("hit", "lost")]
    if len(settled) < min_samples:
        return {}

    # Group by market category
    from collections import defaultdict
    market_groups: Dict[str, List[Dict]] = defaultdict(list)

    for e in settled:
        m_lower = e.get("market", "").lower()
        if "anytime" in m_lower and "goal" in m_lower:
            cat = "goalscorer"
        elif "sot" in m_lower or ("shot" in m_lower and "target" in m_lower):
            cat = "sot"
        elif "shot" in m_lower:
            cat = "shots"
        elif "tackle" in m_lower:
            cat = "tackles"
        elif "foul" in m_lower:
            cat = "fouls"
        elif "card" in m_lower or "booked" in m_lower:
            cat = "carded"
        elif "assist" in m_lower:
            cat = "assists"
        elif "key" in m_lower and "pass" in m_lower:
            cat = "keypasses"
        else:
            continue
        market_groups[cat].append(e)

    adjustments = {}
    for cat, entries in market_groups.items():
        if len(entries) < min_samples:
            continue

        # Actual hit rate
        hits = sum(1 for e in entries if e["outcome"] == "hit")
        actual_rate = hits / len(entries)

        # Average model probability
        avg_model_prob = sum(e.get("model_prob", 0) for e in entries) / len(entries)

        if avg_model_prob > 0.01:
            # Calibration multiplier: actual/predicted
            # Capped to [0.5, 2.0] to prevent wild swings from small samples
            ratio = actual_rate / avg_model_prob
            ratio = max(0.5, min(2.0, ratio))
            adjustments[cat] = round(ratio, 3)

    return adjustments


def get_hypothetical_kelly_pnl(bankroll: float = 1000.0) -> Dict:
    """Calculate hypothetical P&L if Kelly sizing was applied to all value prop bets.

    Uses quarter-Kelly (conservative) on all settled prop bets with positive edge.
    Simulates sequential bet placement from the ledger in chronological order.

    Args:
        bankroll: Starting hypothetical bankroll (default 1000).

    Returns:
        dict with total_pnl, final_bankroll, roi_pct, bet_count, and per-bet breakdown.
    """
    ledger = _load_ledger()
    settled = [e for e in ledger if e["outcome"] in ("hit", "lost")]
    if not settled:
        return {
            "starting_bankroll": bankroll,
            "final_bankroll": bankroll,
            "total_pnl": 0,
            "roi_pct": 0,
            "bet_count": 0,
            "bets": [],
        }

    # Sort chronologically
    settled.sort(key=lambda e: e.get("settled_at", e.get("date", "")))

    current = bankroll
    total_staked = 0
    bets = []

    for e in settled:
        odds = e.get("best_odds", 0)
        model_prob = e.get("model_prob", 0)
        kelly = e.get("kelly_fraction", 0)
        if odds <= 1 or model_prob <= 0:
            continue

        # Use quarter-Kelly, capped at 5% of bankroll
        quarter_kelly = kelly * 0.25
        stake_pct = min(quarter_kelly, 0.05)
        if stake_pct <= 0:
            continue

        stake = round(current * stake_pct, 2)
        if stake < 0.5:
            continue  # Skip micro-bets

        if e["outcome"] == "hit":
            profit = round(stake * (odds - 1), 2)
            current = round(current + profit, 2)
        else:
            profit = -stake
            current = round(current - stake, 2)

        total_staked += stake

        bets.append({
            "player": e.get("player", ""),
            "market": e.get("market", ""),
            "match": e.get("match", ""),
            "date": e.get("date", ""),
            "odds": odds,
            "edge_pct": e.get("edge_pct", 0),
            "stake": stake,
            "outcome": e["outcome"],
            "profit": profit,
            "running_bankroll": current,
        })

    total_pnl = round(current - bankroll, 2)
    roi = round(total_pnl / total_staked * 100, 1) if total_staked > 0 else 0

    return {
        "starting_bankroll": bankroll,
        "final_bankroll": current,
        "total_pnl": total_pnl,
        "total_staked": round(total_staked, 2),
        "roi_pct": roi,
        "bet_count": len(bets),
        "hits": len([b for b in bets if b["outcome"] == "hit"]),
        "losses": len([b for b in bets if b["outcome"] == "lost"]),
        "hit_rate": round(
            len([b for b in bets if b["outcome"] == "hit"]) / len(bets) * 100, 1
        ) if bets else 0,
        "bets": bets,
    }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Player prop outcome tracker")
    parser.add_argument("--settle", action="store_true",
                        help="Settle props for today (or --date)")
    parser.add_argument("--date", type=str,
                        help="Date to settle (YYYY-MM-DD)")
    parser.add_argument("--stats", action="store_true",
                        help="Show performance stats")
    args = parser.parse_args()

    if args.settle:
        result = settle_props(args.date)
        print(json.dumps(result, indent=2))
    elif args.stats:
        if PERFORMANCE_PATH.exists():
            with open(PERFORMANCE_PATH) as f:
                perf = json.load(f)
            print(json.dumps(perf, indent=2))
        else:
            print("No performance data yet. Run --settle first.")
    else:
        parser.print_help()
