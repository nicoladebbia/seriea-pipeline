"""Track 1 — Shadow prediction settlement.

After matches finish + Sofascore + matches.parquet refresh, this script joins
shadow predictions from simulator_shadow_log.json against actual outcomes
and computes per-market Brier scores, accuracy, and calibration stats.

Outputs a growing ledger: data/betting/shadow_results_history.json

Run Monday morning (or Tuesday if Monday-night matches are still pending).

Usage:
    python3 scripts/prediction/settle_shadow_log.py
    python3 scripts/prediction/settle_shadow_log.py --shadow-log path --no-update-history
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SHADOW_LOG_PATH = PROJECT_ROOT / "data" / "upcoming" / "simulator_shadow_log.json"
SETTLED_PATH = PROJECT_ROOT / "data" / "betting" / "shadow_results_history.json"
MATCHES_PATH = PROJECT_ROOT / "data" / "parsed" / "matches.parquet"
TEAM_STATS_PATH = PROJECT_ROOT / "data" / "external" / "sofascore" / "match_team_stats.parquet"
MAPPING_PATH = PROJECT_ROOT / "data" / "parsed" / "match_id_mapping.parquet"
EVENTS_PATH = PROJECT_ROOT / "data" / "parsed" / "events.parquet"


def _load_actuals_for_fixtures(fixtures: list[dict]) -> dict[str, dict]:
    """Build {match_key: {actual outcomes}} from matches.parquet + Sofascore + events."""
    matches = pd.read_parquet(MATCHES_PATH)
    matches = matches[matches["league"] == "serie_a"].copy()
    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")

    team_stats = None
    if TEAM_STATS_PATH.exists():
        ts = pd.read_parquet(TEAM_STATS_PATH)
        # Aggregate per match for total corners + cards
        team_stats = ts.copy()

    mapping = None
    if MAPPING_PATH.exists():
        m = pd.read_parquet(MAPPING_PATH)[["match_id", "sofascore_id"]].dropna()
        m["sofascore_id"] = m["sofascore_id"].astype(str)
        mapping = m

    events = None
    if EVENTS_PATH.exists():
        events = pd.read_parquet(EVENTS_PATH)

    results: dict[str, dict] = {}
    for fx in fixtures:
        mk = fx["match_key"]
        home = fx["home_team"]
        away = fx["away_team"]
        date_str = fx["match_date"]
        mdate = pd.to_datetime(date_str, errors="coerce")

        row_mask = ((matches["home_team"] == home)
                    & (matches["away_team"] == away)
                    & (matches["match_date"] == mdate))
        row = matches[row_mask]
        if len(row) == 0 or pd.isna(row["home_score"].iloc[0]):
            # Not settled yet
            continue
        r = row.iloc[0]
        actual = {
            "match_id_canonical": str(r.get("match_id", mk)),
            "home_score": int(r["home_score"]),
            "away_score": int(r["away_score"]),
            "result": str(r.get("result", "")),
            "home_corners": int(r["home_corners"]) if "home_corners" in r and pd.notna(r.get("home_corners")) else None,
            "away_corners": int(r["away_corners"]) if "away_corners" in r and pd.notna(r.get("away_corners")) else None,
            "home_yellow": int(r["home_yellow_cards"]) if "home_yellow_cards" in r and pd.notna(r.get("home_yellow_cards")) else None,
            "away_yellow": int(r["away_yellow_cards"]) if "away_yellow_cards" in r and pd.notna(r.get("away_yellow_cards")) else None,
            "home_red": int(r.get("home_red_cards", 0)) if pd.notna(r.get("home_red_cards")) else 0,
            "away_red": int(r.get("away_red_cards", 0)) if pd.notna(r.get("away_red_cards")) else 0,
            "home_shots": int(r["home_shots_total"]) if "home_shots_total" in r and pd.notna(r.get("home_shots_total")) else None,
            "away_shots": int(r["away_shots_total"]) if "away_shots_total" in r and pd.notna(r.get("away_shots_total")) else None,
        }

        # Goal scorers from events.parquet (match via mapping or canonical id)
        scorers: set[str] = set()
        if events is not None and mapping is not None:
            canonical_id = actual["match_id_canonical"]
            fbref_row = mapping[mapping["match_id"] == canonical_id]
            fbref_hash = None
            if len(fbref_row) > 0:
                fbref_hash = fbref_row.get("fbref_hash")
                if fbref_hash is not None and len(fbref_hash) > 0:
                    fbref_hash = fbref_hash.iloc[0]
            goal_rows = events[events["event_type"] == "goal"]
            # Events match_id may be fbref_hash or canonical
            for candidate in filter(None, [canonical_id, fbref_hash]):
                match_events = goal_rows[goal_rows["match_id"] == candidate]
                for p in match_events["player"].dropna():
                    if isinstance(p, str):
                        scorers.add(p.strip())
        actual["scorers"] = sorted(scorers)

        results[mk] = actual
    return results


def _settle_binary(predicted_prob: float, outcome: int) -> dict:
    """Return Brier, log-loss contribution, correct/incorrect."""
    p = float(np.clip(predicted_prob, 1e-4, 1 - 1e-4))
    brier = (p - outcome) ** 2
    # Log loss
    ll = -np.log(p) if outcome == 1 else -np.log(1 - p)
    return {"prob": round(p, 4), "outcome": int(outcome), "brier": round(brier, 4), "log_loss": round(ll, 4),
            "correct": int((p > 0.5) == bool(outcome))}


def _settle_multiclass(pred: dict, actual: str) -> dict:
    """1X2-style: predicted is {H,D,A}, actual is one of them."""
    ks = list(pred.keys())
    probs = [float(pred[k]) for k in ks]
    outcomes = [1 if k == actual else 0 for k in ks]
    brier = sum((p - o) ** 2 for p, o in zip(probs, outcomes))
    predicted_idx = int(np.argmax(probs))
    correct = int(ks[predicted_idx] == actual)
    return {"predicted_class": ks[predicted_idx], "actual_class": actual,
            "probs": {k: round(float(v), 4) for k, v in zip(ks, probs)},
            "brier": round(brier, 4), "correct": correct,
            "max_prob": round(max(probs), 4)}


def settle_fixture(prediction: dict, actual: dict) -> dict:
    """Score every market in a prediction against the actual outcome."""
    total_goals = actual["home_score"] + actual["away_score"]
    btts = int(actual["home_score"] > 0 and actual["away_score"] > 0)

    gm = prediction["goal_markets"]
    settled: dict[str, dict] = {}

    # 1X2
    settled["1X2"] = _settle_multiclass(gm["1X2"], actual["result"])
    # Double chance
    settled["double_chance_1X"] = _settle_binary(gm["double_chance_1X"], int(actual["result"] in ("H", "D")))
    settled["double_chance_X2"] = _settle_binary(gm["double_chance_X2"], int(actual["result"] in ("D", "A")))
    settled["double_chance_12"] = _settle_binary(gm["double_chance_12"], int(actual["result"] in ("H", "A")))
    # Over/under
    for line_str, line in [("over_0_5", 0.5), ("over_1_5", 1.5), ("over_2_5", 2.5),
                            ("over_3_5", 3.5), ("over_4_5", 4.5)]:
        settled[line_str] = _settle_binary(gm[line_str], int(total_goals > line))
    settled["btts"] = _settle_binary(gm["btts"], btts)
    settled["home_clean_sheet"] = _settle_binary(gm["home_clean_sheet"], int(actual["away_score"] == 0))
    settled["away_clean_sheet"] = _settle_binary(gm["away_clean_sheet"], int(actual["home_score"] == 0))

    # AH — home side
    diff = actual["home_score"] - actual["away_score"]
    for key, line in [("AH_home_-1.5", -1.5), ("AH_home_-0.5", -0.5),
                      ("AH_home_+0.5", +0.5), ("AH_home_+1.5", +1.5)]:
        settled[key] = _settle_binary(gm[key], int(diff + line > 0))

    # Phase 2 markets (corners/cards/shots)
    cms = prediction.get("corner_card_shot_markets", {})
    total_corners = None
    if actual.get("home_corners") is not None and actual.get("away_corners") is not None:
        total_corners = actual["home_corners"] + actual["away_corners"]
    total_cards = None
    if actual.get("home_yellow") is not None:
        total_cards = (actual["home_yellow"] + actual["away_yellow"]
                       + actual.get("home_red", 0) + actual.get("away_red", 0))
    total_shots = None
    if actual.get("home_shots") is not None and actual.get("away_shots") is not None:
        total_shots = actual["home_shots"] + actual["away_shots"]

    for label, pred_prob in cms.items():
        if "corners_over_" in label and not label.startswith(("home_", "away_")):
            line_str = label.split("corners_over_")[-1].replace("_", ".")
            line = float(line_str)
            if total_corners is not None:
                settled[label] = _settle_binary(pred_prob, int(total_corners > line))
        elif "cards_over_" in label and not label.startswith(("home_", "away_")):
            line_str = label.split("cards_over_")[-1].replace("_", ".")
            line = float(line_str)
            if total_cards is not None:
                settled[label] = _settle_binary(pred_prob, int(total_cards > line))
        elif label.startswith("shots_over_"):
            line = float(label.split("shots_over_")[-1].replace("_", "."))
            if total_shots is not None:
                settled[label] = _settle_binary(pred_prob, int(total_shots > line))

    # Top scorer hit/miss
    top_scorer_hits = []
    for ps in prediction.get("top_scorers", []):
        player_name = ps["player_name"]
        predicted_prob = ps["anytime_scorer_prob"]
        scored = int(player_name in actual.get("scorers", []))
        top_scorer_hits.append({
            "player_name": player_name,
            "predicted_prob": round(predicted_prob, 4),
            "scored": scored,
            "brier": round((predicted_prob - scored) ** 2, 4),
        })

    return {
        "match_key": prediction["match_key"],
        "home_team": prediction["home_team"],
        "away_team": prediction["away_team"],
        "match_date": prediction["match_date"],
        "actual": {
            "score": f"{actual['home_score']}-{actual['away_score']}",
            "result": actual["result"],
            "total_goals": int(total_goals),
            "btts": btts,
            "total_corners": total_corners,
            "total_cards": total_cards,
            "total_shots": total_shots,
            "scorers": actual.get("scorers", []),
        },
        "markets": settled,
        "top_scorers": top_scorer_hits,
    }


def run(shadow_log_path: Path, output_path: Path, update_history: bool = True) -> dict:
    log.info("Loading shadow predictions from %s", shadow_log_path)
    with open(shadow_log_path) as f:
        shadow = json.load(f)
    fixtures = shadow.get("fixtures", [])
    log.info("  %d fixtures in shadow log", len(fixtures))

    log.info("Building actuals from matches.parquet + Sofascore + events")
    actuals = _load_actuals_for_fixtures(fixtures)
    log.info("  %d fixtures with settled outcomes", len(actuals))

    settled_fixtures = []
    for fx in fixtures:
        mk = fx["match_key"]
        if mk not in actuals:
            continue
        settled_fixtures.append(settle_fixture(fx, actuals[mk]))

    # Aggregate Brier + accuracy per market
    per_market_summary: dict[str, dict] = {}
    for sf in settled_fixtures:
        for label, m in sf["markets"].items():
            agg = per_market_summary.setdefault(label, {"n": 0, "brier_sum": 0.0, "correct_sum": 0})
            agg["n"] += 1
            agg["brier_sum"] += float(m["brier"])
            agg["correct_sum"] += int(m.get("correct", 0))

    market_stats = {}
    for label, agg in per_market_summary.items():
        market_stats[label] = {
            "n": agg["n"],
            "brier_mean": round(agg["brier_sum"] / agg["n"], 4) if agg["n"] else None,
            "accuracy": round(agg["correct_sum"] / agg["n"], 4) if agg["n"] else None,
        }

    settlement = {
        "run_id": shadow.get("run_id"),
        "settled_at": datetime.now(timezone.utc).isoformat(),
        "n_predicted": len(fixtures),
        "n_settled": len(settled_fixtures),
        "fixtures": settled_fixtures,
        "per_market": market_stats,
    }

    if update_history:
        # Append to rolling ledger
        history = {"runs": []}
        if output_path.exists():
            try:
                history = json.loads(output_path.read_text())
            except Exception:
                pass
        runs = history.get("runs", [])
        # Dedupe by run_id
        existing_ids = {r.get("run_id") for r in runs}
        if settlement["run_id"] not in existing_ids:
            runs.append(settlement)
        history["runs"] = runs
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(history, indent=2, default=str))
        log.info("Wrote settlement to %s (total runs: %d)", output_path, len(runs))
    else:
        log.info("--no-update-history flag set; returning settlement without write")

    return settlement


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shadow-log", default=str(SHADOW_LOG_PATH))
    ap.add_argument("--output", default=str(SETTLED_PATH))
    ap.add_argument("--no-update-history", action="store_true")
    args = ap.parse_args()

    settlement = run(Path(args.shadow_log), Path(args.output), update_history=not args.no_update_history)

    print(f"\n=== Shadow Settlement ===")
    print(f"Predicted: {settlement['n_predicted']}  Settled: {settlement['n_settled']}")
    if settlement["n_settled"] == 0:
        print("No fixtures have settled outcomes yet. Re-run after matches finish + data refresh.")
        return 0

    print("\nPer-match results:")
    for f in settlement["fixtures"]:
        a = f["actual"]
        r_1x2 = f["markets"].get("1X2", {})
        btts = f["markets"].get("btts", {})
        ou = f["markets"].get("over_2_5", {})
        print(f"  {f['home_team']:12s} {a['score']:5s} {f['away_team']:12s}  "
              f"1X2 pred={r_1x2.get('predicted_class','?')} actual={r_1x2.get('actual_class','?')} "
              f"[{'✓' if r_1x2.get('correct') else '✗'}]  "
              f"BTTS {'✓' if btts.get('correct') else '✗'}  "
              f"O2.5 {'✓' if ou.get('correct') else '✗'}")

    print(f"\nPer-market aggregates:")
    for label, stats in sorted(settlement["per_market"].items(), key=lambda x: -x[1]["n"]):
        if stats["n"] < 3:
            continue
        print(f"  {label:30s} n={stats['n']:3d}  Brier={stats['brier_mean']}  acc={stats['accuracy']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
