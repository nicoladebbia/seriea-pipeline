"""Track 2 — Shadow ROI calculator.

Joins:
  1. simulator_shadow_log.json (predictions)
  2. manual_odds_log.json (odds you entered)
  3. shadow_results_history.json (actual outcomes)

Produces per-market, per-edge-threshold ROI with bootstrap CI using the same
stake policies as Phase 3b harness. Output: data/diagnostics/shadow_roi_report.json

Which markets report? Any market where:
  (a) simulator emitted a probability,
  (b) user entered odds for both sides (or just one + the other is implied 1/(p−overround)),
  (c) the outcome has settled.

Usage:
    python3 scripts/prediction/compute_shadow_roi.py
    python3 scripts/prediction/compute_shadow_roi.py --edge-thresholds 0,3,5,7,10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.simulator.backtests.roi_bootstrap import compute_roi_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SHADOW_LOG_PATH = PROJECT_ROOT / "data" / "upcoming" / "simulator_shadow_log.json"
MANUAL_ODDS_PATH = PROJECT_ROOT / "data" / "upcoming" / "manual_odds_log.json"
RESULTS_HISTORY_PATH = PROJECT_ROOT / "data" / "betting" / "shadow_results_history.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "diagnostics" / "shadow_roi_report.json"


# ---------------------------------------------------------------------------
# Market routing: (odds_key_pair, simulator_market_key, outcome_key, fn_to_settle)
#
# For each market, we need:
#   - the two odds keys in manual_odds_log (yes/no pair)
#   - the simulator market label in shadow_log
#   - how to extract the actual outcome from settled fixture
# ---------------------------------------------------------------------------

def _total_goals(actual: dict) -> int:
    return actual["total_goals"]


def _resolve_1x2(actual: dict) -> str:
    return actual["result"]


def _build_bets(shadow_fixtures: list[dict], odds_fixtures: list[dict],
                settled_fixtures: list[dict], edge_thresholds: list[float]) -> list[dict]:
    """For each (fixture, market), attempt to build a bet record."""
    shadow_map = {f["match_key"]: f for f in shadow_fixtures}
    odds_map = {f["match_key"]: f for f in odds_fixtures}
    settled_map = {f["match_key"]: f for f in settled_fixtures}

    bet_records: list[dict] = []

    for match_key, shadow in shadow_map.items():
        if match_key not in odds_map or match_key not in settled_map:
            continue
        odds_entry = odds_map[match_key]
        odds = odds_entry.get("odds", {}) or {}
        settled = settled_map[match_key]
        actual = settled["actual"]

        gm = shadow["goal_markets"]

        # Binary markets: lookup (yes_odds_key, no_odds_key, sim_prob_key, outcome_fn)
        binary_markets = [
            # (yes_key, no_key, sim_key, outcome_value_fn)
            ("odds_over_0_5", "odds_under_0_5", "over_0_5", lambda a: int(_total_goals(a) > 0)),
            ("odds_over_1_5", "odds_under_1_5", "over_1_5", lambda a: int(_total_goals(a) > 1)),
            ("odds_over_2_5", "odds_under_2_5", "over_2_5", lambda a: int(_total_goals(a) > 2)),
            ("odds_over_3_5", "odds_under_3_5", "over_3_5", lambda a: int(_total_goals(a) > 3)),
            ("odds_over_4_5", "odds_under_4_5", "over_4_5", lambda a: int(_total_goals(a) > 4)),
            ("odds_btts_yes", "odds_btts_no", "btts", lambda a: int(a["btts"])),
        ]
        for yes_key, no_key, sim_key, outcome_fn in binary_markets:
            yes_odds = odds.get(yes_key)
            no_odds = odds.get(no_key)
            if yes_odds is None:
                continue
            p = gm.get(sim_key)
            if p is None:
                continue
            # Deoverround if we have both sides; else assume 5% vig
            if no_odds is not None and no_odds > 1.0:
                inv_y = 1.0 / yes_odds
                inv_n = 1.0 / no_odds
                fair_yes = inv_y / (inv_y + inv_n)
                fair_no = 1.0 - fair_yes
            else:
                fair_yes = 1.0 / yes_odds / 1.05  # approximate 5% vig
                fair_no = 1.0 - fair_yes
            edge_yes = (p - fair_yes) / fair_yes * 100.0
            edge_no = ((1 - p) - fair_no) / fair_no * 100.0
            outcome = outcome_fn(actual)
            if edge_yes >= edge_no:
                side, bet_prob, bet_odds, edge, won = "yes", p, yes_odds, edge_yes, outcome == 1
            else:
                if no_odds is None:
                    continue  # can't bet "no" if we don't have odds
                side, bet_prob, bet_odds, edge, won = "no", 1 - p, no_odds, edge_no, outcome == 0
            bet_records.append({
                "match_key": match_key,
                "market": sim_key,
                "side": side,
                "prob": bet_prob,
                "odds": bet_odds,
                "edge_pct": edge,
                "won": won,
            })

        # 1X2
        if all(odds.get(k) is not None for k in ("odds_1", "odds_X", "odds_2")):
            o = [float(odds["odds_1"]), float(odds["odds_X"]), float(odds["odds_2"])]
            inv = [1 / x for x in o]
            s = sum(inv)
            fair = [i / s for i in inv]
            probs = [gm["1X2"]["H"], gm["1X2"]["D"], gm["1X2"]["A"]]
            classes = ["H", "D", "A"]
            best_idx, best_edge = 0, -1e9
            for i in range(3):
                e = (probs[i] - fair[i]) / fair[i] * 100.0
                if e > best_edge:
                    best_edge = e
                    best_idx = i
            bet_records.append({
                "match_key": match_key,
                "market": "1X2",
                "side": classes[best_idx],
                "prob": probs[best_idx],
                "odds": o[best_idx],
                "edge_pct": best_edge,
                "won": classes[best_idx] == actual["result"],
            })

        # Corners — match Phase 2 probabilities against entered odds
        cms = shadow.get("corner_card_shot_markets", {})
        corner_markets = [
            ("odds_corners_over_7_5", "odds_corners_under_7_5", "corners_over_7_5", 7),
            ("odds_corners_over_8_5", "odds_corners_under_8_5", "corners_over_8_5", 8),
            ("odds_corners_over_9_5", "odds_corners_under_9_5", "corners_over_9_5", 9),
            ("odds_corners_over_10_5", "odds_corners_under_10_5", "corners_over_10_5", 10),
            ("odds_corners_over_11_5", "odds_corners_under_11_5", "corners_over_11_5", 11),
        ]
        total_corners = actual.get("total_corners")
        if total_corners is not None:
            for yes_key, no_key, sim_key, line in corner_markets:
                yes_odds = odds.get(yes_key)
                no_odds = odds.get(no_key)
                if yes_odds is None or sim_key not in cms:
                    continue
                p = cms[sim_key]
                if no_odds is not None and no_odds > 1.0:
                    inv_y, inv_n = 1 / yes_odds, 1 / no_odds
                    fair_yes = inv_y / (inv_y + inv_n)
                else:
                    fair_yes = 1.0 / yes_odds / 1.05
                fair_no = 1.0 - fair_yes
                outcome_over = int(total_corners > line)
                edge_yes = (p - fair_yes) / fair_yes * 100.0
                edge_no = ((1 - p) - fair_no) / fair_no * 100.0
                if edge_yes >= edge_no:
                    side, bp, bo, edge, won = "yes", p, yes_odds, edge_yes, outcome_over == 1
                else:
                    if no_odds is None:
                        continue
                    side, bp, bo, edge, won = "no", 1 - p, no_odds, edge_no, outcome_over == 0
                bet_records.append({
                    "match_key": match_key,
                    "market": sim_key,
                    "side": side,
                    "prob": bp,
                    "odds": bo,
                    "edge_pct": edge,
                    "won": won,
                })

        # Cards — same pattern
        total_cards = actual.get("total_cards")
        card_markets = [
            ("odds_cards_over_2_5", "odds_cards_under_2_5", "cards_over_2_5", 2),
            ("odds_cards_over_3_5", "odds_cards_under_3_5", "cards_over_3_5", 3),
            ("odds_cards_over_4_5", "odds_cards_under_4_5", "cards_over_4_5", 4),
            ("odds_cards_over_5_5", "odds_cards_under_5_5", "cards_over_5_5", 5),
        ]
        if total_cards is not None:
            for yes_key, no_key, sim_key, line in card_markets:
                yes_odds = odds.get(yes_key)
                no_odds = odds.get(no_key)
                if yes_odds is None or sim_key not in cms:
                    continue
                p = cms[sim_key]
                if no_odds is not None and no_odds > 1.0:
                    inv_y, inv_n = 1 / yes_odds, 1 / no_odds
                    fair_yes = inv_y / (inv_y + inv_n)
                else:
                    fair_yes = 1.0 / yes_odds / 1.05
                fair_no = 1.0 - fair_yes
                outcome_over = int(total_cards > line)
                edge_yes = (p - fair_yes) / fair_yes * 100.0
                edge_no = ((1 - p) - fair_no) / fair_no * 100.0
                if edge_yes >= edge_no:
                    side, bp, bo, edge, won = "yes", p, yes_odds, edge_yes, outcome_over == 1
                else:
                    if no_odds is None:
                        continue
                    side, bp, bo, edge, won = "no", 1 - p, no_odds, edge_no, outcome_over == 0
                bet_records.append({
                    "match_key": match_key,
                    "market": sim_key,
                    "side": side,
                    "prob": bp,
                    "odds": bo,
                    "edge_pct": edge,
                    "won": won,
                })

    return bet_records


def compute_roi_by_market_and_threshold(bets: list[dict], thresholds: list[float],
                                         stake_flat: float = 10.0) -> dict:
    out: dict[str, dict] = {}
    markets = sorted(set(b["market"] for b in bets))
    for mk in markets:
        out[mk] = {}
        market_bets = [b for b in bets if b["market"] == mk]
        for t in thresholds:
            filtered = [b for b in market_bets if b["edge_pct"] >= t]
            if not filtered:
                out[mk][f"thresh_{int(t)}pct"] = {"n": 0}
                continue
            stakes = np.array([stake_flat] * len(filtered))
            profits = np.array([
                (b["odds"] - 1.0) * stake_flat if b["won"] else -stake_flat
                for b in filtered
            ])
            stats = compute_roi_stats(stakes, profits, n_resample=500, seed_salt=0)
            out[mk][f"thresh_{int(t)}pct"] = {
                "n": stats.n_bets,
                "profit_flat_eur": stats.total_profit_eur,
                "roi_pct_point": stats.roi_pct_point,
                "roi_ci_lower": stats.roi_pct_ci_lower,
                "roi_ci_upper": stats.roi_pct_ci_upper,
                "wins": int(sum(1 for b in filtered if b["won"])),
                "losses": int(sum(1 for b in filtered if not b["won"])),
            }
    return out


def run(shadow_log_path: Path, odds_path: Path, results_path: Path,
        output_path: Path, edge_thresholds: list[float]) -> dict:
    with open(shadow_log_path) as f:
        shadow = json.load(f)
    shadow_fixtures = shadow.get("fixtures", [])

    if not odds_path.exists():
        log.warning("Manual odds file missing — no ROI possible")
        return {"status": "missing_odds_file"}
    with open(odds_path) as f:
        odds_data = json.load(f)
    odds_fixtures = odds_data.get("fixtures", [])

    if not results_path.exists():
        log.warning("Shadow results history missing — run settlement first")
        return {"status": "missing_settlement"}
    with open(results_path) as f:
        history = json.load(f)
    # Flatten settled fixtures across runs
    settled_fixtures = []
    for run_entry in history.get("runs", []):
        for fx in run_entry.get("fixtures", []):
            settled_fixtures.append(fx)

    bets = _build_bets(shadow_fixtures, odds_fixtures, settled_fixtures, edge_thresholds)
    log.info("Built %d candidate bets", len(bets))

    roi_by_market = compute_roi_by_market_and_threshold(bets, edge_thresholds)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_shadow_run_id": shadow.get("run_id"),
        "n_bets_total": len(bets),
        "edge_thresholds_pct": edge_thresholds,
        "bets": bets,
        "per_market": roi_by_market,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shadow-log", default=str(SHADOW_LOG_PATH))
    ap.add_argument("--odds", default=str(MANUAL_ODDS_PATH))
    ap.add_argument("--results", default=str(RESULTS_HISTORY_PATH))
    ap.add_argument("--output", default=str(OUTPUT_PATH))
    ap.add_argument("--edge-thresholds", default="0,3,5,7,10")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.edge_thresholds.split(",")]
    rep = run(Path(args.shadow_log), Path(args.odds), Path(args.results),
              Path(args.output), thresholds)

    if rep.get("status"):
        print(f"Status: {rep['status']}")
        return 1

    print(f"\n=== Shadow ROI Report ===")
    print(f"n bets candidate:  {rep['n_bets_total']}")
    if not rep.get("bets"):
        print("No bets yet — need settled matches + odds + predictions to overlap.")
        return 0
    print(f"\nPer market, per edge threshold (flat €10):")
    for mk in sorted(rep["per_market"]):
        print(f"\n  {mk}")
        for thresh_key, stats in rep["per_market"][mk].items():
            if stats["n"] == 0:
                continue
            ci_l = stats.get("roi_ci_lower")
            ci_u = stats.get("roi_ci_upper")
            ci_str = f"  CI[{ci_l},{ci_u}]" if ci_l is not None else ""
            print(f"    {thresh_key}: n={stats['n']:3d}  profit=€{stats.get('profit_flat_eur',0)}  "
                  f"ROI={stats.get('roi_pct_point','?')}%{ci_str}  "
                  f"({stats.get('wins',0)}W/{stats.get('losses',0)}L)")

    print(f"\nWrote: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
