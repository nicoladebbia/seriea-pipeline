#!/usr/bin/env python3
"""PLAYER PROP ODDS FETCHER & VALUE SCANNER

Fetches real player prop odds from The Odds API and compares them against
our CatBoost player-level model predictions to find value bets.

Available markets (Serie A via The Odds API):
  - player_goal_scorer_anytime  (6+ bookmakers)
  - player_first_goal_scorer    (5+ bookmakers)
  - player_last_goal_scorer     (3+ bookmakers)

Usage:
    python scripts/player_prop_odds.py              # fetch + scan for value
    python scripts/player_prop_odds.py --fetch-only # just fetch odds
    python scripts/player_prop_odds.py --scan-only  # scan existing odds vs model
"""

import json
import logging
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR
from config.team_names import normalize_team

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────────────
API_BASE_URL = "https://api.the-odds-api.com/v4"
SERIE_A_KEY = "soccer_italy_serie_a"
REGIONS = "eu,uk,us,au"
ODDS_FORMAT = "decimal"

PLAYER_PROP_MARKETS = {
    "player_goal_scorer_anytime": "Anytime Goalscorer",
    "player_first_goal_scorer": "First Goalscorer",
    "player_last_goal_scorer": "Last Goalscorer",
    "player_shots_on_target": "Shots on Target O/U",
    "player_shots_on_target_alternate": "Shots on Target Alt O/U",
    "player_shots": "Total Shots O/U",
}

# Value scanning thresholds
MIN_VALUE_PCT = 3.0       # minimum edge % to flag as value
MAX_EDGE_PCT = 50.0       # cap edge — anything higher is likely model artifact
MIN_PROB = 0.05           # ignore players with <5% model probability
MIN_PROB_DEFENDER = 0.08  # higher bar for defenders (model overestimates them)
MAX_ODDS = 20.0           # ignore extreme longshots (>20.0 = <5% implied)
MIN_ODDS = 1.10           # ignore near-certainties
MIN_BOOKMAKERS = 2        # require ≥2 bookmakers for goalscorer markets
MIN_BOOKMAKERS_OU = 1     # O/U markets have standardized lines, 1 bookmaker OK

CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "upcoming"

from config.api_keys import get_odds_api_key


# =============================================================================
# FETCHING
# =============================================================================

def fetch_player_prop_odds(use_cache: bool = True) -> Dict:
    """Fetch player prop odds for all upcoming Serie A matches.

    Returns:
        Dict keyed by "Home vs Away" with player prop odds per market.
    """
    api_key = get_odds_api_key()
    if not api_key:
        log.error("No ODDS_API_KEY found. Set it in .env or config/api_keys.json")
        return {}
    if not HAS_REQUESTS:
        log.error("requests library required: pip install requests")
        return {}

    # Check cache
    cache_path = CACHE_DIR / "player_props.json"
    if use_cache and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            age_mins = (datetime.now() - datetime.fromisoformat(cached.get("fetched_at", "2000-01-01"))).total_seconds() / 60
            if age_mins < 15:
                log.info(f"Using cached player props ({age_mins:.0f}min old, {len(cached.get('matches', {}))} matches)")
                return cached.get("matches", {})
        except Exception as e:
            log.debug(f"Failed to load cached player prop odds: {e}")

    # Step 1: Get events
    log.info("Fetching Serie A events for player props...")
    try:
        resp = requests.get(
            f"{API_BASE_URL}/sports/{SERIE_A_KEY}/events",
            params={"apiKey": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
        log.info(f"Found {len(events)} upcoming events")
    except Exception as e:
        log.error(f"Failed to fetch events: {e}")
        return {}

    if not events:
        return {}

    # Step 2: For each event, fetch player prop markets
    all_props = {}
    credits_remaining = "?"

    for event in events:
        event_id = event["id"]
        home = normalize_team(event.get("home_team", ""))
        away = normalize_team(event.get("away_team", ""))
        match_key = f"{home} vs {away}"
        commence = event.get("commence_time", "")

        markets_csv = ",".join(PLAYER_PROP_MARKETS.keys())

        try:
            resp = requests.get(
                f"{API_BASE_URL}/sports/{SERIE_A_KEY}/events/{event_id}/odds",
                params={
                    "apiKey": api_key,
                    "regions": REGIONS,
                    "markets": markets_csv,
                    "oddsFormat": ODDS_FORMAT,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            credits_remaining = resp.headers.get("x-requests-remaining", "?")
        except Exception as e:
            log.warning(f"  Failed for {match_key}: {e}")
            time.sleep(0.3)
            continue

        # Parse player props
        match_props = {
            "home_team": home,
            "away_team": away,
            "commence_time": commence,
            "markets": {},
        }

        for bm in data.get("bookmakers", []):
            bookie = bm.get("title", "Unknown")
            for mkt in bm.get("markets", []):
                mkt_key = mkt.get("key", "")
                if mkt_key not in PLAYER_PROP_MARKETS:
                    continue

                if mkt_key not in match_props["markets"]:
                    match_props["markets"][mkt_key] = {}

                is_ou_market = ("over_under" in mkt_key or "shots_on_target" in mkt_key
                               or mkt_key == "player_shots")

                for outcome in mkt.get("outcomes", []):
                    price = outcome.get("price", 0)
                    if price <= 1:
                        continue

                    if is_ou_market:
                        # O/U format: name=Over/Under, description=player, point=line
                        side = outcome.get("name", "")
                        if side != "Over":
                            continue  # only track Over outcomes
                        player_name = outcome.get("description", "")
                        point = outcome.get("point", 0.5)
                        # Key: "PlayerName O 0.5"
                        entry_key = f"{player_name} O {point}"
                    else:
                        # Goalscorer format: description=player name
                        player_name = outcome.get("description", outcome.get("name", ""))
                        entry_key = player_name
                        point = None

                    if not player_name:
                        continue

                    if entry_key not in match_props["markets"][mkt_key]:
                        match_props["markets"][mkt_key][entry_key] = {
                            "player_name": player_name,
                            "best_odds": price,
                            "avg_odds": price,
                            "bookmakers": [],
                            "n_bookmakers": 0,
                        }
                        if point is not None:
                            match_props["markets"][mkt_key][entry_key]["line"] = point

                    entry = match_props["markets"][mkt_key][entry_key]
                    entry["bookmakers"].append({"bookmaker": bookie, "odds": price})
                    entry["n_bookmakers"] = len(entry["bookmakers"])
                    entry["best_odds"] = max(entry["best_odds"], price)
                    prices = [b["odds"] for b in entry["bookmakers"]]
                    entry["avg_odds"] = round(sum(prices) / len(prices), 2)

        # Count players
        n_players = sum(len(players) for players in match_props["markets"].values())
        if n_players > 0:
            all_props[match_key] = match_props
            log.info(f"  ✅ {match_key}: {n_players} player props across {len(match_props['markets'])} markets")
        else:
            log.info(f"  ⚪ {match_key}: no player props available")

        time.sleep(0.3)

    log.info(f"Fetched player props for {len(all_props)}/{len(events)} matches "
             f"[credits remaining: {credits_remaining}]")

    # Cache results
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"fetched_at": datetime.now().isoformat(), "matches": all_props}, f, indent=2)

    return all_props


def save_player_prop_odds(props: Dict) -> Path:
    """Save player prop odds to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "player_prop_odds.json"
    with open(out_path, "w") as f:
        json.dump({
            "fetched_at": datetime.now().isoformat(),
            "source": "the-odds-api.com",
            "markets": list(PLAYER_PROP_MARKETS.keys()),
            "match_count": len(props),
            "matches": props,
        }, f, indent=2)
    log.info(f"Saved player prop odds to {out_path}")
    return out_path


# =============================================================================
# VALUE SCANNING — Compare model predictions vs bookmaker odds
# =============================================================================

def _load_model_predictions() -> Dict:
    """Load player model predictions from player_predictions.json."""
    pred_path = OUTPUT_DIR / "player_predictions.json"
    if not pred_path.exists():
        log.warning(f"No player predictions at {pred_path}. Run player_predictions.py predict first.")
        return {}
    with open(pred_path) as f:
        return json.load(f)


def _fuzzy_match_player(model_name: str, odds_name: str) -> bool:
    """Fuzzy match player names between model and odds API."""
    # Normalize: lowercase, remove accents approximation
    m = model_name.lower().strip()
    o = odds_name.lower().strip()

    if m == o:
        return True

    # Last name match (most reliable)
    m_parts = m.split()
    o_parts = o.split()
    if m_parts and o_parts and m_parts[-1] == o_parts[-1]:
        # Also check first initial if available
        if len(m_parts) >= 2 and len(o_parts) >= 2:
            if m_parts[0][0] == o_parts[0][0]:
                return True
        # Just last name match for single-name players
        if len(m_parts) == 1 or len(o_parts) == 1:
            return True

    # Substring match (handles "Lautaro Martínez" vs "Lautaro Martinez")
    # Remove common accent variations
    import unicodedata
    def _strip_accents(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    m_clean = _strip_accents(m)
    o_clean = _strip_accents(o)

    if m_clean == o_clean:
        return True

    # Last name without accents
    m_last = _strip_accents(m_parts[-1]) if m_parts else ""
    o_last = _strip_accents(o_parts[-1]) if o_parts else ""
    if m_last and o_last and m_last == o_last:
        return True

    return False


def scan_for_value(
    props: Dict,
    model_preds: Optional[Dict] = None,
    min_value_pct: float = MIN_VALUE_PCT,
) -> List[Dict]:
    """Compare model predictions against bookmaker odds to find value bets.

    Returns:
        List of value bet opportunities sorted by edge.
    """
    if model_preds is None:
        model_preds = _load_model_predictions()

    # If no pre-computed predictions, generate on the fly
    pms = None
    predict_fn = None
    if not model_preds:
        log.info("No pre-computed predictions — generating on the fly from player database...")
        try:
            from scripts.betting.player_predictions import (
                load_player_data, build_player_features, predict_player_markets
            )
            pms = load_player_data()
            pms = build_player_features(pms)
            predict_fn = predict_player_markets
        except Exception as e:
            log.error(f"Cannot load player data: {e}")
            return []

    # Load calibration adjustments from prop tracker (if available)
    cal_adjustments = {}
    try:
        from scripts.betting.prop_tracker import get_calibration_adjustments
        cal_adjustments = get_calibration_adjustments(min_samples=20)
        if cal_adjustments:
            log.info(f"  Calibration adjustments loaded: {cal_adjustments}")
    except ImportError:
        pass

    # Map odds market keys → model target keys
    MARKET_TO_MODEL = {
        "player_goal_scorer_anytime": {"model_key": "goalscorer", "label": "Anytime Goalscorer"},
        "player_shots_on_target": {"line_map": {0.5: "sot_o05"}, "label": "SOT Over"},
        "player_shots_on_target_alternate": {"line_map": {0.5: "sot_o05"}, "label": "SOT Over"},
        "player_shots": {"line_map": {0.5: "shots_o05", 1.5: "shots_o15", 2.5: "shots_o25"}, "label": "Shots Over"},
    }

    value_bets = []
    n_matched = 0
    n_total_odds_players = 0

    for match_key, match_data in props.items():
        home = match_data.get("home_team", "")
        away = match_data.get("away_team", "")
        commence = match_data.get("commence_time", "")

        all_markets = match_data.get("markets", {})
        if not all_markets:
            continue

        # Get model predictions for this match (pre-computed or on-the-fly)
        all_players = []
        if model_preds:
            match_preds = model_preds.get(match_key, {})
            all_players = (
                match_preds.get("home_players", []) + match_preds.get("away_players", [])
            )

        # On-the-fly: build a lookup of player names → full predictions from pms
        fly_cache = {}
        if pms is not None and predict_fn is not None and not all_players:
            for team, opponent, is_home in [(home, away, True), (away, home, False)]:
                team_players = pms[pms["team"] == team].drop_duplicates("player_id", keep="last")
                for _, row in team_players.iterrows():
                    pname = row.get("player_name", "")
                    pid = int(row.get("player_id", 0))
                    pos = row.get("position", "M")
                    if not pname or pid == 0:
                        continue
                    fly_cache[pname] = {
                        "player_name": pname,
                        "player_id": pid,
                        "team": team,
                        "opponent": opponent,
                        "position": pos,
                        "is_home": is_home,
                        "_pred": None,  # lazy-loaded
                    }

        # Scan each available market
        for mkt_key, mkt_entries in all_markets.items():
            mkt_cfg = MARKET_TO_MODEL.get(mkt_key)
            if not mkt_cfg:
                continue  # skip markets we don't have models for

            for entry_key, odds_data in mkt_entries.items():
                best_odds = odds_data.get("best_odds", 0)
                avg_odds = odds_data.get("avg_odds", 0)
                n_bookmakers = odds_data.get("n_bookmakers", 0)
                odds_player_name = odds_data.get("player_name", entry_key)
                n_total_odds_players += 1

                if best_odds < MIN_ODDS or best_odds > MAX_ODDS:
                    continue
                # O/U markets: 1 bookmaker OK (standardized lines)
                # Goalscorer: require 2+ for price reliability
                is_ou_scan = "line_map" in mkt_cfg
                min_bkm = MIN_BOOKMAKERS_OU if is_ou_scan else MIN_BOOKMAKERS
                if n_bookmakers < min_bkm:
                    continue

                # Determine which model target to use
                if "model_key" in mkt_cfg:
                    model_target = mkt_cfg["model_key"]
                elif "line_map" in mkt_cfg:
                    line = odds_data.get("line", 0.5)
                    model_target = mkt_cfg["line_map"].get(line)
                    if not model_target:
                        continue
                else:
                    continue

                market_label = mkt_cfg["label"]
                if "line" in odds_data:
                    market_label = f"{mkt_cfg['label']} {odds_data['line']}"

                # Find matching model prediction
                model_prob = None
                matched_player = None

                # Try pre-computed predictions first
                for p in all_players:
                    if _fuzzy_match_player(p.get("player_name", ""), odds_player_name):
                        target_market = p.get("markets", {}).get(model_target, {})
                        model_prob = target_market.get("prob", None)
                        matched_player = p
                        break

                # Try on-the-fly prediction
                if model_prob is None and predict_fn is not None:
                    for db_name, db_info in fly_cache.items():
                        if _fuzzy_match_player(db_name, odds_player_name):
                            # Lazy-load prediction
                            if db_info["_pred"] is None:
                                db_info["_pred"] = predict_fn(
                                    player_name=db_info["player_name"],
                                    player_id=db_info["player_id"],
                                    team=db_info["team"],
                                    opponent=db_info["opponent"],
                                    position=db_info["position"],
                                    is_home=db_info["is_home"],
                                    pms=pms,
                                )
                            pred = db_info["_pred"]
                            if pred and pred.get("markets", {}).get(model_target):
                                model_prob = pred["markets"][model_target]["prob"]
                                matched_player = {
                                    "player_name": db_info["player_name"],
                                    "team": db_info["team"],
                                    "position": db_info["position"],
                                }
                            break

                if model_prob is None or model_prob < MIN_PROB:
                    continue

                pos = matched_player.get("position", "M") if matched_player else "M"
                if pos == "D" and model_prob < MIN_PROB_DEFENDER:
                    continue

                n_matched += 1

                # Apply calibration adjustment from settlement history
                raw_model_prob = model_prob
                if cal_adjustments and model_target in cal_adjustments:
                    model_prob = min(0.95, model_prob * cal_adjustments[model_target])

                # Calculate value
                implied_prob = 1.0 / best_odds
                edge_pct = (model_prob - implied_prob) / implied_prob * 100

                if edge_pct > MAX_EDGE_PCT:
                    continue

                if edge_pct >= min_value_pct:
                    kelly_f = (model_prob * best_odds - 1) / (best_odds - 1)
                    kelly_f = max(0, min(kelly_f, 0.10))

                    # Confidence tier
                    if edge_pct <= 10 and n_bookmakers >= 3:
                        tier = "A"
                    elif edge_pct <= 20 and n_bookmakers >= 2:
                        tier = "B"
                    else:
                        tier = "C"

                    value_bets.append({
                        "match": match_key,
                        "commence_time": commence,
                        "player": odds_player_name,
                        "team": matched_player.get("team", ""),
                        "position": pos,
                        "market": market_label,
                        "model_prob": round(model_prob, 4),
                        "implied_prob": round(implied_prob, 4),
                        "best_odds": best_odds,
                        "avg_odds": avg_odds,
                        "edge_pct": round(edge_pct, 1),
                        "kelly_fraction": round(kelly_f, 4),
                        "n_bookmakers": n_bookmakers,
                        "tier": tier,
                        "bookmakers": odds_data.get("bookmakers", []),
                    })

    log.info(f"  Matched {n_matched}/{n_total_odds_players} odds players to model predictions")

    # Sort by edge descending
    value_bets.sort(key=lambda x: x["edge_pct"], reverse=True)

    log.info(f"\n{'='*70}")
    log.info(f"PLAYER PROP VALUE SCANNER — {len(value_bets)} value bets found")
    log.info(f"{'='*70}")

    for vb in value_bets[:20]:
        log.info(f"  {vb['player']:25s} ({vb['position']}) | {vb['match']}")
        log.info(f"    Model: {vb['model_prob']:.1%} | Odds: {vb['best_odds']:.2f} "
                 f"(implied {vb['implied_prob']:.1%}) | Edge: {vb['edge_pct']:+.1f}% "
                 f"| Kelly: {vb['kelly_fraction']:.2%}")

    return value_bets


def save_value_bets(value_bets: List[Dict]) -> Path:
    """Save value bets to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "player_prop_value_bets.json"

    # Strip bookmaker details for cleaner output
    clean_bets = []
    for vb in value_bets:
        clean = {k: v for k, v in vb.items() if k != "bookmakers"}
        clean["best_bookmaker"] = max(
            vb.get("bookmakers", [{}]),
            key=lambda b: b.get("odds", 0),
            default={}
        ).get("bookmaker", "Unknown")
        clean_bets.append(clean)

    with open(out_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "min_edge_pct": MIN_VALUE_PCT,
            "total_value_bets": len(clean_bets),
            "bets": clean_bets,
        }, f, indent=2)

    log.info(f"Saved {len(clean_bets)} value bets to {out_path}")
    return out_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Player Prop Odds Fetcher & Value Scanner")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch odds, don't scan")
    parser.add_argument("--scan-only", action="store_true", help="Only scan existing odds")
    parser.add_argument("--no-cache", action="store_true", help="Force fresh fetch")
    parser.add_argument("--min-edge", type=float, default=MIN_VALUE_PCT, help="Minimum edge %%")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("PLAYER PROP ODDS — FETCHER & VALUE SCANNER")
    log.info("=" * 70)

    if args.scan_only:
        # Load existing odds
        odds_path = OUTPUT_DIR / "player_prop_odds.json"
        if not odds_path.exists():
            log.error("No player_prop_odds.json found. Run with --fetch-only first.")
            return
        with open(odds_path) as f:
            props = json.load(f).get("matches", {})
    else:
        # Fetch odds
        props = fetch_player_prop_odds(use_cache=not args.no_cache)
        if props:
            save_player_prop_odds(props)

    if args.fetch_only:
        return

    if not props:
        log.warning("No player prop odds available")
        return

    # Scan for value
    value_bets = scan_for_value(props, min_value_pct=args.min_edge)
    if value_bets:
        save_value_bets(value_bets)

    # Summary
    log.info(f"\n{'='*70}")
    log.info(f"SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"  Matches with player props: {len(props)}")
    log.info(f"  Value bets found: {len(value_bets)}")
    if value_bets:
        avg_edge = sum(vb["edge_pct"] for vb in value_bets) / len(value_bets)
        log.info(f"  Average edge: {avg_edge:+.1f}%")
        log.info(f"  Top edge: {value_bets[0]['edge_pct']:+.1f}% ({value_bets[0]['player']})")


if __name__ == "__main__":
    main()
