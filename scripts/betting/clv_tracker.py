#!/usr/bin/env python3
"""CLOSING LINE VALUE (CLV) TRACKER

The single best predictor of long-term betting profitability.

At settlement time, compares the odds at which a bet was placed
against the closing line (last snapshot before kickoff).

CLV = (placement_implied_prob / closing_implied_prob) - 1
  Positive CLV = we got better odds than the closing line (sharp)
  Negative CLV = market moved against us (likely -EV long term)

Supports all markets: 1X2, BTTS, DC, O/U, AH.

Workflow:
  1. record_bet_placement() — called when bet slip is generated
  2. Pipeline takes odds snapshots periodically (odds_tracker.py)
  3. track_clv_from_slip() — called after matchday to compute CLV
  4. generate_clv_report() — analysis of CLV by market, bookmaker, etc.

Reads:
  data/betting/placed_bets.json (recorded at placement time)
  data/odds_snapshots/ (historical odds snapshots)
  data/upcoming/ultimate_bet_slip.json
Writes:
  data/betting/clv_history.json
  data/betting/placed_bets.json
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

log = logging.getLogger(__name__)

CLV_HISTORY_FILE = DATA_DIR / "betting" / "clv_history.json"
PLACED_BETS_FILE = DATA_DIR / "betting" / "placed_bets.json"
SNAPSHOTS_DIR = DATA_DIR / "odds_snapshots"


def _resolve_league(item: dict) -> str:
    """Return league for a bet dict, inferring from team names if absent."""
    lg = item.get("league")
    if lg:
        return lg
    home = item.get("home_team")
    away = item.get("away_team")
    if (not home or not away) and item.get("match"):
        try:
            home, away = item["match"].split(" vs ", 1)
        except ValueError:
            pass
    try:
        from config.leagues import infer_league
        return infer_league(home, away)
    except Exception:
        return "serie_a"


def _load_clv_history() -> Dict:
    """Load existing CLV tracking data."""
    if CLV_HISTORY_FILE.exists():
        try:
            with open(CLV_HISTORY_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load CLV history: {e}")
    return {
        "bets": [],
        "running_clv": 0.0,
        "total_bets_tracked": 0,
        "positive_clv_count": 0,
        "negative_clv_count": 0,
    }


def _save_clv_history(data: Dict):
    """Save CLV history."""
    CLV_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = datetime.now().isoformat()
    with open(CLV_HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# BET PLACEMENT RECORDING
# =============================================================================

def record_bet_placement(slip_path: Path = None) -> int:
    """Record placement odds from a bet slip for future CLV comparison.

    Called automatically when the ultimate betting system generates a slip.
    Saves each bet's odds, bookmaker, market, and selection.

    Returns number of new bets recorded.
    """
    if slip_path is None:
        slip_path = DATA_DIR / "upcoming" / "ultimate_bet_slip.json"

    if not slip_path.exists():
        log.warning("No bet slip found at %s", slip_path)
        return 0

    with open(slip_path) as f:
        slip = json.load(f)

    selected = slip.get("selected_bets", [])
    if not selected:
        log.info("No selected bets in slip")
        return 0

    # Load existing placed bets
    placed = _load_placed_bets()
    existing_keys = {
        (b["match"], b["market"], b["selection"])
        for b in placed.get("bets", [])
    }

    new_count = 0
    now = datetime.now().isoformat()

    for bet in selected:
        key = (bet.get("match", ""), bet.get("market", ""), bet.get("selection", ""))
        if key in existing_keys:
            continue

        entry = {
            "match": bet.get("match", ""),
            "date": bet.get("date", ""),
            "league": _resolve_league(bet),
            "market": bet.get("market", ""),
            "selection": bet.get("selection", ""),
            "best_odds": bet.get("best_odds", 0),
            "best_bookmaker": bet.get("best_bookmaker", ""),
            "avg_odds": bet.get("avg_odds", 0),
            "pinnacle_odds": bet.get("pinnacle_odds", 0),
            "model_prob": bet.get("model_prob", 0),
            "sharp_implied_prob": bet.get("sharp_implied_prob", 0),
            "edge_pct": bet.get("edge_pct", 0),
            "stake_amount": bet.get("stake_amount", 0),
            "placed_at": now,
            "slip_generated_at": slip.get("generated_at", now),
        }
        placed["bets"].append(entry)
        new_count += 1
        existing_keys.add(key)

    if new_count > 0:
        placed["updated_at"] = now
        _save_placed_bets(placed)
        log.info("Recorded %d new bet placements (total: %d)", new_count, len(placed["bets"]))

    return new_count


def _load_placed_bets() -> Dict:
    if PLACED_BETS_FILE.exists():
        try:
            with open(PLACED_BETS_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load placed bets: {e}")
    return {"bets": [], "updated_at": None}


def _save_placed_bets(data: Dict):
    PLACED_BETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PLACED_BETS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# =============================================================================
# CLOSING LINE LOOKUP
# =============================================================================

def get_closing_odds(match_key: str) -> Optional[Dict]:
    """Get the last recorded odds snapshot for a match (closing line proxy).

    Looks for the most recent snapshot that contains this match.
    """
    if not SNAPSHOTS_DIR.exists():
        return None

    # Walk backwards through snapshots to find the latest with this match
    for f in sorted(SNAPSHOTS_DIR.glob("odds_*.json"), reverse=True):
        try:
            with open(f) as fh:
                snapshot = json.load(fh)
            matches = snapshot.get("matches", {})
            if match_key in matches:
                return matches[match_key]
        except (json.JSONDecodeError, OSError):
            continue

    return None


def get_closing_odds_bookmaker(match_key: str, bookmaker: str = None) -> Optional[Dict]:
    """Get closing odds from per-bookmaker snapshots for more accurate CLV.

    Bookmaker snapshots have structure:
      {"h2h": [{"bookmaker": "X", "home": 2.3, "draw": 3.0, "away": 3.1}, ...],
       "totals_2.5": [{"bookmaker": "X", "over": 1.9, "under": 1.9}, ...],
       "spreads_0.0": [{"bookmaker": "X", "home": 1.8, "away": 2.0}, ...]}

    Returns a flat dict with best available closing odds across all markets.
    """
    if not SNAPSHOTS_DIR.exists():
        return get_closing_odds(match_key)

    for f in sorted(SNAPSHOTS_DIR.glob("bookmakers_*.json"), reverse=True):
        try:
            with open(f) as fh:
                snapshot = json.load(fh)
            matches = snapshot.get("matches", {})
            match_data = matches.get(match_key)
            if not match_data:
                continue

            # Build flat closing odds dict from nested structure
            result = {}

            # 1X2 from h2h
            h2h_list = match_data.get("h2h", [])
            if h2h_list:
                if bookmaker:
                    bm_entry = next(
                        (e for e in h2h_list if e.get("bookmaker", "").lower() == bookmaker.lower()),
                        None
                    )
                    if bm_entry:
                        result["home"] = bm_entry.get("home", 0)
                        result["draw"] = bm_entry.get("draw", 0)
                        result["away"] = bm_entry.get("away", 0)
                if not result:
                    # Use average across all bookmakers as closing line
                    homes = [e["home"] for e in h2h_list if e.get("home", 0) > 1]
                    draws = [e["draw"] for e in h2h_list if e.get("draw", 0) > 1]
                    aways = [e["away"] for e in h2h_list if e.get("away", 0) > 1]
                    if homes:
                        result["home"] = round(sum(homes) / len(homes), 3)
                    if draws:
                        result["draw"] = round(sum(draws) / len(draws), 3)
                    if aways:
                        result["away"] = round(sum(aways) / len(aways), 3)

            # O/U from totals_2.5 (most common line)
            for tkey in ["totals_2.5", "totals_2.0", "totals_3.0"]:
                totals_list = match_data.get(tkey, [])
                if totals_list:
                    overs = [e["over"] for e in totals_list if e.get("over", 0) > 1]
                    unders = [e["under"] for e in totals_list if e.get("under", 0) > 1]
                    if overs:
                        result["over"] = round(sum(overs) / len(overs), 3)
                    if unders:
                        result["under"] = round(sum(unders) / len(unders), 3)
                    result["ou_line"] = float(tkey.split("_")[1])
                    break

            # AH from spreads
            for skey in sorted(match_data.keys()):
                if skey.startswith("spreads_"):
                    spreads_list = match_data.get(skey, [])
                    if spreads_list:
                        sh = [e["home"] for e in spreads_list if e.get("home", 0) > 1]
                        sa = [e["away"] for e in spreads_list if e.get("away", 0) > 1]
                        if sh:
                            result["spread_home"] = round(sum(sh) / len(sh), 3)
                        if sa:
                            result["spread_away"] = round(sum(sa) / len(sa), 3)
                        result["spread_line"] = float(skey.split("_")[1])
                        break

            if result:
                return result
        except (json.JSONDecodeError, OSError):
            continue

    return get_closing_odds(match_key)


def get_closing_odds_extra(match_key: str) -> Optional[Dict]:
    """Get closing odds from extra market snapshots (BTTS, DC, DNB, alt totals).

    These snapshots contain markets not available in the h2h/totals/spreads snapshots.
    """
    if not SNAPSHOTS_DIR.exists():
        return None

    for f in sorted(SNAPSHOTS_DIR.glob("extra_*.json"), reverse=True):
        try:
            with open(f) as fh:
                snapshot = json.load(fh)
            matches = snapshot.get("matches", {})
            match_data = matches.get(match_key)
            if not match_data:
                continue

            result = {}

            # BTTS
            btts = match_data.get("btts", {})
            if btts:
                if btts.get("yes", 0) > 1:
                    result["btts_yes"] = btts["yes"]
                if btts.get("no", 0) > 1:
                    result["btts_no"] = btts["no"]

            # Double Chance
            dc = match_data.get("double_chance", {})
            if dc:
                for sel_key in ["1X", "X2", "12"]:
                    sel_data = dc.get(sel_key, {})
                    avg = sel_data.get("avg", 0)
                    if avg > 1:
                        result[f"dc_{sel_key.lower()}"] = avg

            # Draw No Bet
            dnb = match_data.get("draw_no_bet", {})
            if dnb:
                if dnb.get("home", 0) > 1:
                    result["dnb_home"] = dnb["home"]
                if dnb.get("away", 0) > 1:
                    result["dnb_away"] = dnb["away"]

            # Alternate totals (store all lines)
            alt_totals = match_data.get("alternate_totals", {})
            if alt_totals:
                for line_str, line_data in alt_totals.items():
                    over = line_data.get("over", 0)
                    under = line_data.get("under", 0)
                    if over > 1:
                        result[f"ou_{line_str}_over"] = over
                    if under > 1:
                        result[f"ou_{line_str}_under"] = under

            if result:
                return result
        except (json.JSONDecodeError, OSError):
            continue

    return None


def compute_clv(bet_odds: float, closing_odds: float) -> float:
    """Compute Closing Line Value.

    CLV = (bet_odds / closing_odds) - 1
    Positive = we got better odds than the closing line (we beat the market).
    Negative = closing line was better than our odds (market moved against us).

    Example: bet at 3.50, closes at 3.30 → CLV = +6.1% (we got better price)
    Example: bet at 3.50, closes at 3.70 → CLV = -5.4% (market moved against us)
    """
    if bet_odds <= 1.0 or closing_odds <= 1.0:
        return 0.0
    clv = (bet_odds / closing_odds) - 1.0
    log.debug("CLV: bet_odds=%.4f, closing_odds=%.4f, clv=%+.4f (%+.2f%%)",
              bet_odds, closing_odds, clv, clv * 100)
    return clv


# =============================================================================
# CLV COMPUTATION
# =============================================================================

def _map_selection_to_closing(selection: str, market: str, closing: Dict,
                              extra_closing: Dict = None) -> float:
    """Map a bet selection to the corresponding closing odds.

    Handles all markets: 1X2, BTTS, DC, O/U, AH, DNB.
    Uses extra_closing (from extra market snapshots) for BTTS, DC, DNB.
    """
    sel = selection.upper()
    mkt = market.upper()
    extra = extra_closing or {}

    # 1X2 market
    if mkt == "1X2":
        if "DRAW" in sel or sel == "X":
            return closing.get("draw", 0)
        elif "HOME" in sel or sel == "1":
            return closing.get("home", 0)
        elif "AWAY" in sel or sel == "2":
            return closing.get("away", 0)

    # Double Chance — prefer real DC closing odds from extra snapshots
    # Validate that DC odds from extra snapshots are actual odds (> 1.0)
    # and within reasonable range of the synthetic calculation from 1X2 odds.
    if mkt == "DC" or "DOUBLE" in mkt:
        def _validate_dc_odds(real_dc: float, synthetic_dc: float) -> bool:
            """Check that DC odds from extra snapshots are reasonable."""
            if real_dc <= 1.0:
                return False
            if synthetic_dc <= 1.0:
                return True  # Can't validate without synthetic, trust real
            # Allow up to 15% deviation from synthetic
            ratio = real_dc / synthetic_dc
            if 0.85 <= ratio <= 1.15:
                return True
            log.debug("DC odds validation failed: real=%.3f synthetic=%.3f ratio=%.3f",
                      real_dc, synthetic_dc, ratio)
            return False

        if "1X" in sel:
            h = closing.get("home", 0)
            d = closing.get("draw", 0)
            synthetic = (1.0 / (1.0/h + 1.0/d)) if h > 1 and d > 1 else 0
            real_dc = extra.get("dc_1x", 0)
            if _validate_dc_odds(real_dc, synthetic):
                return real_dc
            if synthetic > 1:
                return synthetic
        elif "X2" in sel:
            d = closing.get("draw", 0)
            a = closing.get("away", 0)
            synthetic = (1.0 / (1.0/d + 1.0/a)) if d > 1 and a > 1 else 0
            real_dc = extra.get("dc_x2", 0)
            if _validate_dc_odds(real_dc, synthetic):
                return real_dc
            if synthetic > 1:
                return synthetic
        elif "12" in sel:
            h = closing.get("home", 0)
            a = closing.get("away", 0)
            synthetic = (1.0 / (1.0/h + 1.0/a)) if h > 1 and a > 1 else 0
            real_dc = extra.get("dc_12", 0)
            if _validate_dc_odds(real_dc, synthetic):
                return real_dc
            if synthetic > 1:
                return synthetic

    # Over/Under — match by line number in market string (e.g., "O/U 2.5")
    if mkt.startswith("O/U") or mkt in ("OU", "OVER/UNDER", "TOTALS"):
        # Extract line from market string (e.g., "O/U 2.5" → 2.5)
        import re
        line_match = re.search(r'(\d+\.?\d*)', market)
        line = float(line_match.group(1)) if line_match else 2.5

        # Try extra closing odds for alternate lines
        line_str = str(line)
        if "OVER" in sel:
            alt_key = f"ou_{line_str}_over"
            if extra.get(alt_key, 0) > 1:
                return extra[alt_key]
            return closing.get("over", 0)
        elif "UNDER" in sel:
            alt_key = f"ou_{line_str}_under"
            if extra.get(alt_key, 0) > 1:
                return extra[alt_key]
            return closing.get("under", 0)

    # BTTS — use real closing odds from extra snapshots
    if "BTTS" in mkt:
        if "YES" in sel:
            return extra.get("btts_yes", closing.get("btts_yes", 0))
        elif "NO" in sel:
            return extra.get("btts_no", closing.get("btts_no", 0))

    # Draw No Bet
    if mkt == "DNB" or "DRAW NO BET" in mkt:
        if "HOME" in sel:
            return extra.get("dnb_home", 0)
        elif "AWAY" in sel:
            return extra.get("dnb_away", 0)

    # Asian Handicap — match by line in selection
    if mkt.startswith("AH") or mkt in ("ASIAN HANDICAP", "SPREADS"):
        if "HOME" in sel or "-" in sel:
            return closing.get("spread_home", 0)
        elif "AWAY" in sel or "+" in sel:
            return closing.get("spread_away", 0)

    return 0


def track_clv_for_settled_bets(settled_bets: List[Dict]) -> Dict:
    """Compute CLV for a batch of settled bets.

    Args:
        settled_bets: List of bet dicts with at least {match, odds, result}.

    Returns:
        Summary of CLV tracking results.
    """
    history = _load_clv_history()
    new_tracked = 0

    # Track which bets we've already processed
    processed_keys = {
        (b.get("match", ""), b.get("market", ""), b.get("selection", ""))
        for b in history.get("bets", [])
    }

    for bet in settled_bets:
        match_key = bet.get("match", "")
        market = bet.get("market", "1X2")
        selection = bet.get("selection", "")
        bet_key = (match_key, market, selection)

        if bet_key in processed_keys:
            continue

        bet_odds = bet.get("odds", bet.get("best_odds", 0))
        if bet_odds <= 1.0:
            continue

        # Get closing odds — try per-bookmaker first, then aggregate
        closing = get_closing_odds_bookmaker(
            match_key, bookmaker=bet.get("best_bookmaker")
        )
        if not closing:
            closing = get_closing_odds(match_key)
        if not closing:
            closing = {}

        # Get extra market closing odds (BTTS, DC, DNB, alt totals)
        extra_closing = get_closing_odds_extra(match_key)

        if not closing and not extra_closing:
            continue

        closing_odds = _map_selection_to_closing(
            selection, market, closing, extra_closing=extra_closing
        )
        if closing_odds <= 1.0:
            continue

        clv = compute_clv(bet_odds, closing_odds)

        clv_entry = {
            "match": match_key,
            "league": _resolve_league(bet),
            "market": market,
            "selection": selection,
            "bet_odds": bet_odds,
            "closing_odds": round(closing_odds, 3),
            "clv": round(clv, 4),
            "clv_pct": round(clv * 100, 2),
            "result": bet.get("result", "unknown"),
            "placed_at": bet.get("placed_at", bet.get("date", "")),
            "tracked_at": datetime.now().isoformat(),
        }

        history["bets"].append(clv_entry)
        new_tracked += 1

        if clv > 0:
            history["positive_clv_count"] += 1
        else:
            history["negative_clv_count"] += 1

        # Update journal with CLV data
        try:
            from scripts.betting.bet_journal import update_clv, _generate_bet_id
            bet_id = _generate_bet_id(
                bet.get("date", ""),
                match_key,
                market,
                selection,
            )
            update_clv(bet_id, round(closing_odds, 3), round(clv * 100, 2))
        except Exception as e:
            log.debug("Failed to update journal CLV: %s", e)

    # Update running averages
    history["total_bets_tracked"] = len(history["bets"])
    if history["bets"]:
        history["running_clv"] = round(
            sum(b["clv"] for b in history["bets"]) / len(history["bets"]), 4
        )
        history["running_clv_pct"] = round(history["running_clv"] * 100, 2)

    _save_clv_history(history)

    result = {
        "new_tracked": new_tracked,
        "total_tracked": history["total_bets_tracked"],
        "running_clv": history["running_clv"],
        "running_clv_pct": history.get("running_clv_pct", 0),
        "positive_rate": (
            history["positive_clv_count"] / history["total_bets_tracked"]
            if history["total_bets_tracked"] > 0 else 0
        ),
    }

    if new_tracked > 0:
        log.info("CLV tracked: %d new bets, running CLV: %+.2f%%",
                 new_tracked, result['running_clv_pct'])

    return result


def track_clv_from_slip(slip_path: Path = None) -> Dict:
    """Auto-track CLV from placed bets file.

    Reads placed_bets.json (recorded at bet time) and computes CLV
    using the latest available closing odds.
    """
    placed = _load_placed_bets()
    bets = placed.get("bets", [])
    if not bets:
        log.info("No placed bets to track CLV for")
        return {"new_tracked": 0, "total_tracked": 0}

    # Convert placed bets format to settled bets format
    settled = []
    for b in bets:
        settled.append({
            "match": b.get("match", ""),
            "league": _resolve_league(b),
            "market": b.get("market", "1X2"),
            "selection": b.get("selection", ""),
            "odds": b.get("best_odds", 0),
            "best_bookmaker": b.get("best_bookmaker", ""),
            "placed_at": b.get("placed_at", ""),
            "date": b.get("date", ""),
        })

    return track_clv_for_settled_bets(settled)


# =============================================================================
# CLV REPORTING
# =============================================================================

def get_clv_summary() -> Dict:
    """Get CLV tracking summary."""
    history = _load_clv_history()
    return {
        "total_tracked": history.get("total_bets_tracked", 0),
        "running_clv": history.get("running_clv", 0),
        "running_clv_pct": history.get("running_clv_pct", 0),
        "positive_clv_count": history.get("positive_clv_count", 0),
        "negative_clv_count": history.get("negative_clv_count", 0),
        "positive_rate": round(
            history.get("positive_clv_count", 0) / max(history.get("total_bets_tracked", 1), 1), 3
        ),
    }


def generate_clv_report() -> Dict:
    """Generate comprehensive CLV analysis report.

    Breaks down CLV by market, selection type, and edge tier.
    """
    history = _load_clv_history()
    bets = history.get("bets", [])

    if not bets:
        return {"status": "no_data", "message": "No CLV data yet. Run pipeline + settle bets first."}

    # Per-market breakdown
    by_market = {}
    for b in bets:
        mkt = b.get("market", "unknown")
        by_market.setdefault(mkt, []).append(b)

    market_stats = {}
    for mkt, mkt_bets in by_market.items():
        clvs = [b["clv"] for b in mkt_bets]
        pos = sum(1 for c in clvs if c > 0)
        market_stats[mkt] = {
            "count": len(mkt_bets),
            "avg_clv_pct": round(sum(clvs) / len(clvs) * 100, 2),
            "positive_rate": round(pos / len(mkt_bets), 3),
            "best_clv_pct": round(max(clvs) * 100, 2),
            "worst_clv_pct": round(min(clvs) * 100, 2),
        }

    # Overall stats
    all_clvs = [b["clv"] for b in bets]
    pos_count = sum(1 for c in all_clvs if c > 0)

    report = {
        "status": "ok",
        "generated_at": datetime.now().isoformat(),
        "total_bets": len(bets),
        "avg_clv_pct": round(sum(all_clvs) / len(all_clvs) * 100, 2),
        "positive_clv_rate": round(pos_count / len(bets), 3),
        "by_market": market_stats,
        "interpretation": _interpret_clv(sum(all_clvs) / len(all_clvs), pos_count / len(bets)),
    }

    # Save report
    report_path = DATA_DIR / "betting" / "clv_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def _interpret_clv(avg_clv: float, pos_rate: float) -> str:
    """Human-readable interpretation of CLV metrics."""
    if avg_clv > 0.02:
        quality = "Excellent — consistently beating closing lines by 2%+"
    elif avg_clv > 0.005:
        quality = "Good — positive CLV indicates sharp betting"
    elif avg_clv > -0.005:
        quality = "Neutral — roughly matching the closing line"
    elif avg_clv > -0.02:
        quality = "Below average — market moves against us"
    else:
        quality = "Poor — significant negative CLV, likely -EV long term"

    if pos_rate > 0.6:
        consistency = "Consistent edge (>60% of bets beat closing line)"
    elif pos_rate > 0.5:
        consistency = "Slight edge (>50% beat closing line)"
    else:
        consistency = "Market moves against us more often than not"

    return f"{quality}. {consistency}."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    parser = argparse.ArgumentParser(description="CLV Tracker")
    parser.add_argument("action", nargs="?", default="summary",
                        choices=["summary", "record", "track", "report"],
                        help="Action: summary, record (placement), track (CLV), report")
    args = parser.parse_args()

    if args.action == "record":
        n = record_bet_placement()
        print(f"Recorded {n} new bet placements")
    elif args.action == "track":
        result = track_clv_from_slip()
        print(f"Tracked {result['new_tracked']} new bets")
        print(f"Running CLV: {result.get('running_clv_pct', 0):+.2f}%")
    elif args.action == "report":
        report = generate_clv_report()
        if report.get("status") == "ok":
            print(f"\nCLV Report ({report['total_bets']} bets):")
            print(f"  Average CLV: {report['avg_clv_pct']:+.2f}%")
            print(f"  Positive CLV rate: {report['positive_clv_rate']:.1%}")
            print(f"  {report['interpretation']}")
            print("\n  By Market:")
            for mkt, stats in report["by_market"].items():
                print(f"    {mkt}: {stats['avg_clv_pct']:+.2f}% CLV "
                      f"({stats['count']} bets, {stats['positive_rate']:.0%} positive)")
        else:
            print(report.get("message", "No data"))
    else:
        summary = get_clv_summary()
        print("\nCLV Tracking Summary:")
        print(f"  Total bets tracked: {summary['total_tracked']}")
        print(f"  Running CLV: {summary['running_clv_pct']:+.2f}%")
        print(f"  Positive CLV rate: {summary['positive_rate']:.1%}")
