#!/usr/bin/env python3
"""Advanced Parlay Engine — 7-engine architecture for correlation-adjusted,
Monte-Carlo-validated multi-leg bet generation.

Engines:
  1. Universal Leg Collector — loads from 12+ market types
  2. Bivariate Poisson Score Matrix — same-game parlay joint probabilities
  3. Gaussian Copula — cross-match correlation adjustment
  4. Multi-Signal Leg Quality Scoring — composite quality 0-100
  5. Smart Combination Generator — tier-based filtering, conflict detection
  6. Parlay Valuation & Kelly Sizing — Monte Carlo hit-rate bands
  7. Categorization & Ranking — 6 categories, composite parlay quality

Usage:
    python3 scripts/parlay_generator.py                # Generate parlay report
    python3 scripts/parlay_generator.py --bankroll 2000 # Custom bankroll
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import poisson, norm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR

UPCOMING_DIR = DATA_DIR / "upcoming"
BETTING_DIR = DATA_DIR / "betting"
BANKROLL_DIR = DATA_DIR / "bankroll"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_LEG_VALUE_PCT = 2.0
MIN_LEG_QUALITY = 30
QUALITY_MIN_VALUE = 2.0
POISSON_CORRELATION_RHO = 0.10
COPULA_RHO_SAME_DAY = 0.03
COPULA_RHO_SAME_ROUND = 0.01
MONTE_CARLO_SIMS = 5000
KELLY_FRACTION = 0.10
MAX_STAKE_PCT = 0.02
TOP_N_PER_CATEGORY = 8
MAX_TOTAL_PARLAYS = 50
SCORE_MATRIX_SIZE = 8  # 0-7 goals per team

CONFIDENCE_MAP = {
    "VERY HIGH": 100, "Very High": 100,
    "HIGH": 75, "High": 75,
    "MEDIUM-HIGH": 50, "Medium-High": 50,
    "MEDIUM": 25, "Medium": 25,
    "LOW": 10, "Low": 10,
}

CONFIDENCE_K = {
    "VERY HIGH": 50, "Very High": 50,
    "HIGH": 30, "High": 30,
    "MEDIUM-HIGH": 20, "Medium-High": 20,
    "MEDIUM": 15, "Medium": 15,
    "LOW": 8, "Low": 8,
}

# Leg types that can be derived from the score matrix
SCORE_MATRIX_MARKETS = {
    "h2h", "totals", "btts", "team_totals", "spreads",
    "double_chance", "first_half", "multi_goal",
}

# Conflicting leg pairs (same match)
CONFLICTING_PAIRS = [
    (("totals", "UNDER"), ("btts", "YES")),
    (("totals", "UNDER 1.5"), ("btts", "YES")),
    (("totals", "UNDER 2.5"), ("btts", "YES")),
    (("h2h", "HOME"), ("h2h", "AWAY")),
    (("h2h", "HOME"), ("h2h", "DRAW")),
    (("h2h", "AWAY"), ("h2h", "DRAW")),
    (("h2h", "HOME"), ("double_chance", "X2")),
    (("h2h", "AWAY"), ("double_chance", "1X")),
    (("draw_no_bet", "HOME"), ("draw_no_bet", "AWAY")),
    (("draw_no_bet", "HOME"), ("h2h", "AWAY")),
    (("draw_no_bet", "AWAY"), ("h2h", "HOME")),
]

# ---------------------------------------------------------------------------
# Calibration artifacts (loaded from file if available)
# ---------------------------------------------------------------------------

def _load_calibrated_beta_k() -> dict:
    """Load calibrated Beta K values from JSON, fall back to CONFIDENCE_K."""
    cal_path = DATA_DIR / "calibration" / "beta_k.json"
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


_CALIBRATED_BETA_K = _load_calibrated_beta_k()


def calibrate_beta_k(backtest_path: str | Path | None = None) -> dict:
    """Calibrate Beta K parameters from walk-forward backtest predictions.

    Bins predictions by confidence level, computes empirical variance of
    (predicted_prob - actual_outcome) per bin, and maps to K:
        K = p*(1-p)/variance - 1

    Higher K means tighter Beta distribution (more confident), which is
    appropriate when empirical variance is low (model is well-calibrated).
    Lower K means wider Beta (more uncertainty), used when model predictions
    vary a lot from actuals in that confidence bin.

    Saves to data/calibration/beta_k.json and returns the K dict.
    """
    import glob as glob_mod

    # Find most recent backtest file
    if backtest_path is None:
        opt_dir = DATA_DIR / "optimization"
        if not opt_dir.exists():
            print("No optimization directory found, using default K values")
            return dict(CONFIDENCE_K)
        files = sorted(opt_dir.glob("backtest_unified_*.json"), reverse=True)
        if not files:
            files = sorted(opt_dir.glob("backtest_*.json"), reverse=True)
        if not files:
            print("No backtest files found, using default K values")
            return dict(CONFIDENCE_K)
        backtest_path = files[0]
    else:
        backtest_path = Path(backtest_path)

    try:
        with open(backtest_path) as f:
            bt_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Failed to load backtest: {e}")
        return dict(CONFIDENCE_K)

    # Extract (predicted_prob, actual_outcome, confidence_level) triples
    triples = []

    # Check for backtest_unified output: accuracy.prediction_records
    if "accuracy" in bt_data and "prediction_records" in bt_data.get("accuracy", {}):
        predictions = bt_data["accuracy"]["prediction_records"]
    else:
        predictions = bt_data.get("predictions", bt_data.get("results", []))
    for pred in predictions:
        prob = pred.get("predicted_prob", pred.get("probability"))
        actual = pred.get("actual_outcome", pred.get("correct"))
        conf = pred.get("confidence_level", pred.get("confidence", "MEDIUM"))
        if prob is not None and actual is not None:
            triples.append((float(prob), float(actual), str(conf).upper()))

    if len(triples) < 50:
        print(f"Only {len(triples)} predictions found, need 50+. Using defaults.")
        return dict(CONFIDENCE_K)

    # Bin by confidence level
    from collections import defaultdict
    bins = defaultdict(list)
    for prob, actual, conf in triples:
        bins[conf].append((prob, actual))

    calibrated = {}
    for conf, pairs in bins.items():
        if len(pairs) < 10:
            # Too few samples, keep default
            calibrated[conf] = CONFIDENCE_K.get(conf, 15)
            continue
        probs = np.array([p for p, _ in pairs])
        actuals = np.array([a for _, a in pairs])
        n = len(probs)
        mean_prob = np.mean(probs)
        emp_accuracy = np.mean(actuals)

        # Use CALIBRATION ERROR, not binary outcome variance.
        #
        # The old formula (np.var(probs - actuals)) is dominated by aleatoric
        # variance (≈ p*(1-p)) which is irreducible — even a perfect model
        # produces var ≈ 0.25 on binary outcomes, giving K ≈ 0.
        #
        # Instead we measure EPISTEMIC uncertainty: how far is the model's
        # predicted probability from the empirical accuracy in this bin?
        #   cal_error²  = systematic model bias (the thing K should capture)
        #   sampling_var = noise from finite sample (prevents K explosion
        #                  when cal_error ≈ 0 in a small bin)
        cal_error_sq = (mean_prob - emp_accuracy) ** 2
        sampling_var = emp_accuracy * (1 - emp_accuracy) / n
        effective_var = cal_error_sq + sampling_var

        # K = p*(1-p)/var - 1, clamped to [5, 200]
        theoretical_var = mean_prob * (1 - mean_prob)
        if effective_var > 0:
            k = theoretical_var / effective_var - 1
            k = float(np.clip(k, 5, 200))
        else:
            k = 200.0  # perfect calibration → very tight Beta
        calibrated[conf] = round(k, 1)
        print(
            f"  {conf:>12s}: n={n:3d}, pred={mean_prob:.3f}, "
            f"actual={emp_accuracy:.3f}, cal_err={mean_prob - emp_accuracy:+.3f}, K={calibrated[conf]}"
        )

    # Also add lowercase versions
    full_calibrated = {}
    for conf, k in calibrated.items():
        full_calibrated[conf] = k
        # Add title-case version
        title = conf.title().replace("-", "-")
        full_calibrated[title] = k

    # Save
    cal_dir = DATA_DIR / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cal_dir / "beta_k.json"
    with open(cal_path, "w") as f:
        json.dump(full_calibrated, f, indent=2)
    print(f"Calibrated Beta K saved to {cal_path}: {full_calibrated}")

    # Update module-level cache
    global _CALIBRATED_BETA_K
    _CALIBRATED_BETA_K = full_calibrated

    return full_calibrated


class _NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _get_bankroll():
    state = _load_json(BANKROLL_DIR / "state.json")
    return state.get("current_bankroll", 1000)


def _normalize_match(m):
    """Normalize match key for comparison."""
    return m.strip() if m else ""


# ---------------------------------------------------------------------------
# Engine 1: Universal Leg Collector
# ---------------------------------------------------------------------------

def _load_enrichment_data():
    """Load predictions, bookmaker analysis, market intelligence, odds movement."""
    predictions = {}
    preds_raw = _load_json(UPCOMING_DIR / "predictions.json")
    for p in preds_raw.get("predictions", []):
        mk = _normalize_match(p.get("match", ""))
        if mk:
            predictions[mk] = p

    bookmaker = {}
    bk_raw = _load_json(UPCOMING_DIR / "bookmaker_analysis.json")
    for mk, data in bk_raw.get("matches", {}).items():
        bookmaker[_normalize_match(mk)] = data

    market_intel = {}
    mi_raw = _load_json(UPCOMING_DIR / "market_intelligence.json")
    for mk, data in mi_raw.get("matches", {}).items():
        market_intel[_normalize_match(mk)] = data

    odds_movement = {}
    om_raw = _load_json(UPCOMING_DIR / "odds_movement.json")
    for mk, data in om_raw.get("matches", {}).items():
        odds_movement[_normalize_match(mk)] = data

    goal_preds = {}
    gp_raw = _load_json(UPCOMING_DIR / "goal_predictions.json")
    for p in gp_raw.get("predictions", []):
        mk = _normalize_match(p.get("match", ""))
        if mk:
            goal_preds[mk] = p

    return predictions, bookmaker, market_intel, odds_movement, goal_preds


def _enrich_leg(leg, predictions, bookmaker, market_intel, odds_movement):
    """Add quality signals to a leg."""
    mk = _normalize_match(leg["match"])
    pred = predictions.get(mk, {})
    bk = bookmaker.get(mk, {})
    mi = market_intel.get(mk, {})
    om = odds_movement.get(mk, {})

    # Confidence from ensemble predictions
    conf_str = pred.get("confidence_level") or leg.get("confidence", "MEDIUM")
    if isinstance(conf_str, (int, float)):
        conf_str = "HIGH" if conf_str >= 0.6 else "MEDIUM" if conf_str >= 0.4 else "LOW"
    leg["confidence_level"] = conf_str
    leg["confidence_score"] = CONFIDENCE_MAP.get(conf_str, 25) / 100.0

    # Method agreement: std dev of component predictions
    comp = pred.get("component_predictions", {})
    if comp:
        sel = leg.get("selection", "").upper()
        key = "prob_H" if "HOME" in sel or "1X" in sel else "prob_A" if "AWAY" in sel or "X2" in sel else "prob_D" if "DRAW" in sel else None
        if key:
            vals = []
            for method_data in comp.values():
                if isinstance(method_data, dict) and key in method_data:
                    vals.append(method_data[key])
            if len(vals) >= 2:
                leg["method_agreement"] = float(np.std(vals))
            else:
                leg["method_agreement"] = 0.10
        else:
            leg["method_agreement"] = 0.10
    else:
        leg["method_agreement"] = 0.10

    # Sharp alignment
    sharp_cons = bk.get("sharp_consensus", {})
    sel_upper = leg.get("selection", "").upper()
    if sharp_cons:
        sharp_dir = bk.get("sharp_direction", "neutral")
        if ("HOME" in sel_upper and sharp_dir == "home") or \
           ("AWAY" in sel_upper and sharp_dir == "away"):
            leg["sharp_aligned"] = True
        elif sharp_dir == "neutral":
            leg["sharp_aligned"] = None  # neutral
        else:
            leg["sharp_aligned"] = False
    else:
        leg["sharp_aligned"] = None

    # Market edge
    implied_prob = 1.0 / leg["odds"] if leg["odds"] > 1 else 0
    leg["market_edge"] = leg["probability"] - implied_prob

    # Composite intelligence
    leg["composite_intel"] = mi.get("composite_score", 0)

    # Steam alignment
    steam = om.get("is_steam_move", False)
    steam_dir = om.get("direction", "stable")
    if steam:
        if ("HOME" in sel_upper and "home" in steam_dir) or \
           ("AWAY" in sel_upper and "away" in steam_dir) or \
           ("OVER" in sel_upper and "over" in steam_dir):
            leg["steam_aligned"] = True
        else:
            leg["steam_aligned"] = False
    else:
        leg["steam_aligned"] = False

    # Form/momentum from factors
    factors = leg.get("factors", [])
    if isinstance(factors, list):
        hot = any("hot" in f for f in factors)
        cold = any("cold" in f for f in factors)
        leg["momentum"] = "hot" if hot and not cold else "cold" if cold and not hot else "neutral"
    else:
        leg["momentum"] = "neutral"

    return leg


def _load_extra_market_odds():
    """Load real bookmaker odds from per-event extra markets file."""
    raw = _load_json(UPCOMING_DIR / "odds_extra_markets.json")
    return raw.get("matches", {})


def load_all_value_legs():
    """Aggregate legs from all 12+ market types, enriched with quality signals."""
    legs = []
    predictions, bookmaker, market_intel, odds_movement, goal_preds = _load_enrichment_data()
    extra_odds = _load_extra_market_odds()

    # --- 1X2 from unified_report ---
    unified = _load_json(BETTING_DIR / "unified_report.json")
    for bet in unified.get("bets", []):
        prob = bet.get("our_probability", 0) or 0
        odds = bet.get("odds", 0) or 0
        val = bet.get("value_pct", 0) or 0
        if prob > 0 and odds > 1 and val >= MIN_LEG_VALUE_PCT:
            legs.append({
                "match": bet.get("match", ""),
                "date": bet.get("date", ""),
                "market": bet.get("market", "h2h"),
                "selection": bet.get("selection", ""),
                "odds": round(odds, 2),
                "probability": round(prob, 4),
                "value_pct": round(val, 1),
                "confidence": bet.get("confidence", "MEDIUM"),
                "factors": bet.get("factors", []),
                "source": "unified",
            })

    # --- Over/Under value bets ---
    ou = _load_json(UPCOMING_DIR / "over_under_bets.json")
    for bet in ou.get("recommended", []) + ou.get("consider", []):
        prob = bet.get("our_probability", 0) or 0
        odds = bet.get("odds", 0) or 0
        val = bet.get("value_pct", 0) or 0
        if prob > 0 and odds > 1 and val >= MIN_LEG_VALUE_PCT:
            legs.append({
                "match": bet.get("match", ""),
                "date": bet.get("date", ""),
                "market": "totals",
                "selection": bet.get("bet", ""),
                "odds": round(odds, 2),
                "probability": round(prob, 4),
                "value_pct": round(val, 1),
                "confidence": bet.get("confidence", "MEDIUM"),
                "factors": bet.get("factors", []),
                "source": "over_under",
            })

    # --- Over/Under from goal_predictions (raw probabilities) ---
    # Use real bookmaker odds from extra_odds when available (especially for Over 0.5/1.5
    # which are excellent high-probability parlay legs).
    for mk, gp in goal_preds.items():
        ext_totals = extra_odds.get(mk, {}).get("alternate_totals", {})
        for line_key in ["over_0_5", "over_1_5", "over_2_5", "over_3_5", "over_4_5"]:
            prob = gp.get(line_key, 0)
            if prob and prob > 0.50:
                line_label = line_key.replace("over_", "OVER ").replace("_", ".")
                fair_odds = round(1.0 / prob, 2) if prob > 0 else 99
                # Try to get real bookmaker odds for this line
                line_num = line_key.replace("over_", "").replace("_", ".")
                bk_data = ext_totals.get(line_num, {})
                bk_odds = bk_data.get("best_over", 0)
                use_odds = bk_odds if bk_odds > 1 else fair_odds
                val = ((prob * use_odds) - 1) * 100 if use_odds > 1 else 0
                # Only add if not already present from over_under_bets
                already_has = any(
                    l["match"] == mk and l["market"] == "totals" and line_label in l["selection"]
                    for l in legs
                )
                if not already_has and use_odds > 1:
                    legs.append({
                        "match": mk,
                        "date": gp.get("date", ""),
                        "market": "totals",
                        "selection": line_label,
                        "odds": round(use_odds, 2),
                        "probability": round(prob, 4),
                        "value_pct": round(val, 1),
                        "confidence": gp.get("confidence", "MEDIUM"),
                        "factors": gp.get("factors", []),
                        "source": "goal_predictions" + ("_live_odds" if bk_odds > 1 else ""),
                    })

    # --- Handicap value bets ---
    hc = _load_json(UPCOMING_DIR / "handicap_bets.json")
    for bet in hc.get("recommended", []) + hc.get("consider", []):
        prob = bet.get("our_probability", 0) or 0
        odds = bet.get("odds", 0) or 0
        val = bet.get("value_pct", 0) or 0
        if prob > 0 and odds > 1 and val >= MIN_LEG_VALUE_PCT:
            legs.append({
                "match": bet.get("match", ""),
                "date": bet.get("date", ""),
                "market": "spreads",
                "selection": bet.get("bet", ""),
                "odds": round(odds, 2),
                "probability": round(prob, 4),
                "value_pct": round(val, 1),
                "confidence": bet.get("confidence", "MEDIUM"),
                "factors": bet.get("factors", []),
                "source": "handicap",
            })

    # --- Handicap from margin_predictions (raw probs, all 7 lines) ---
    margin_raw = _load_json(UPCOMING_DIR / "margin_predictions.json")
    for pred_entry in margin_raw.get("predictions", []):
        mk = pred_entry.get("match", "")
        hc_probs = pred_entry.get("handicap_probs", {})
        for line, sides in hc_probs.items():
            if not isinstance(sides, dict):
                continue
            for side in ["home", "away"]:
                prob = sides.get(side, 0)
                if prob > 0.50:
                    fair_odds = round(1.0 / prob, 2)
                    sel = f"{side.upper()} {line}"
                    already = any(
                        l["match"] == mk and l["market"] == "spreads" and l["selection"] == sel
                        for l in legs
                    )
                    if not already and fair_odds > 1:
                        legs.append({
                            "match": mk,
                            "date": pred_entry.get("date", ""),
                            "market": "spreads",
                            "selection": sel,
                            "odds": fair_odds,
                            "probability": round(prob, 4),
                            "value_pct": 0,
                            "confidence": pred_entry.get("confidence", "MEDIUM"),
                            "factors": pred_entry.get("factors", []),
                            "source": "margin_predictions",
                        })

    # --- BTTS & Corners value bets ---
    bc = _load_json(UPCOMING_DIR / "btts_corners_bets.json")
    for bet in bc.get("recommended", []) + bc.get("consider", []):
        prob = bet.get("our_probability", 0) or 0
        val = bet.get("value_pct", 0) or 0
        mkt = bet.get("market", "btts")
        odds = bet.get("odds", 0) or 0
        fair_odds = bet.get("fair_odds", 0) or 0
        if odds <= 1 and fair_odds > 1:
            odds = fair_odds
        if prob > 0 and odds > 1 and val >= MIN_LEG_VALUE_PCT:
            legs.append({
                "match": bet.get("match", ""),
                "date": bet.get("date", ""),
                "market": mkt,
                "selection": bet.get("bet", ""),
                "odds": round(odds, 2),
                "probability": round(prob, 4),
                "value_pct": round(val, 1),
                "confidence": bet.get("confidence", "MEDIUM"),
                "factors": bet.get("factors", []),
                "source": "btts_corners",
            })

    # --- BTTS from btts_predictions (raw) — enhanced with real bookmaker odds ---
    btts_raw = _load_json(UPCOMING_DIR / "btts_predictions.json")
    # Handle both list and dict formats (file may be a plain list or {"predictions": [...]})
    btts_preds = btts_raw if isinstance(btts_raw, list) else btts_raw.get("predictions", [])
    for bp in btts_preds:
        mk = bp.get("match", "")
        ext_match = extra_odds.get(mk, {})
        ext_btts = ext_match.get("btts", {})
        for side_key, sel_label in [("btts_yes", "YES"), ("btts_no", "NO")]:
            prob = bp.get(side_key, 0)
            if prob > 0.50:
                fair = round(1.0 / prob, 2) if prob > 0 else 99
                # Use real bookmaker odds if available
                bk_odds = ext_btts.get("best_yes" if sel_label == "YES" else "best_no", 0)
                use_odds = bk_odds if bk_odds > 1 else fair
                val = ((prob * use_odds) - 1) * 100 if bk_odds > 1 else 0
                already = any(
                    l["match"] == mk and l["market"] == "btts" and sel_label in l["selection"]
                    for l in legs
                )
                if not already and use_odds > 1:
                    legs.append({
                        "match": mk,
                        "date": bp.get("date", ""),
                        "market": "btts",
                        "selection": sel_label,
                        "odds": round(use_odds, 2),
                        "probability": round(prob, 4),
                        "value_pct": round(max(val, 0), 1),
                        "confidence": bp.get("confidence", "MEDIUM"),
                        "factors": bp.get("factors", []),
                        "source": "btts_predictions" + ("_live_odds" if bk_odds > 1 else ""),
                    })

    # --- Corners from corners_predictions ---
    corners_raw = _load_json(UPCOMING_DIR / "corners_predictions.json")
    corners_preds = corners_raw if isinstance(corners_raw, list) else corners_raw.get("predictions", [])
    for cp in corners_preds:
        mk = cp.get("match", "")
        for line_key, sel_label in [("over_9_5", "OVER 9.5"), ("over_10_5", "OVER 10.5")]:
            prob = cp.get(line_key, 0)
            if prob > 0.50:
                fair = round(1.0 / prob, 2) if prob > 0 else 99
                already = any(
                    l["match"] == mk and l["market"] == "corners" and sel_label in l["selection"]
                    for l in legs
                )
                if not already and fair > 1:
                    legs.append({
                        "match": mk,
                        "date": cp.get("date", ""),
                        "market": "corners",
                        "selection": sel_label,
                        "odds": fair,
                        "probability": round(prob, 4),
                        "value_pct": 0,
                        "confidence": cp.get("confidence", "MEDIUM"),
                        "factors": cp.get("factors", []),
                        "source": "corners_predictions",
                    })

    # --- Cards value bets ---
    cards = _load_json(UPCOMING_DIR / "cards_bets.json")
    for bet in cards.get("recommended", []) + cards.get("consider", []):
        prob = bet.get("our_probability", 0) or 0
        fair = bet.get("fair_odds", 0) or 0
        odds = bet.get("odds", 0) or fair
        if odds <= 1 and fair > 1:
            odds = fair
        val = ((prob * odds) - 1) * 100 if prob > 0 and odds > 1 else 0
        if prob > 0 and odds > 1 and val >= MIN_LEG_VALUE_PCT:
            legs.append({
                "match": bet.get("match", ""),
                "date": bet.get("date", ""),
                "market": "cards",
                "selection": bet.get("bet", ""),
                "odds": round(odds, 2),
                "probability": round(prob, 4),
                "value_pct": round(val, 1),
                "confidence": bet.get("confidence", "MEDIUM"),
                "factors": bet.get("factors", []),
                "source": "cards",
            })

    # --- Extended markets (double chance, team totals, first half, multi-goal) ---
    # Now enhanced with real bookmaker odds from per-event endpoint
    ext = _load_json(UPCOMING_DIR / "extended_markets.json")

    for match_key, mdata in ext.get("matches", {}).items():
        ext_match = extra_odds.get(match_key, {})

        # Double chance — use real bookmaker odds when available
        dc = mdata.get("double_chance", {})
        ext_dc = ext_match.get("double_chance", {})
        # Build lookup: our "1X"/"X2"/"12" -> API's "Home or Draw"/"Away or Draw"/"Home or Away"
        home_t = ext_match.get("home_team", mdata.get("home_team", ""))
        away_t = ext_match.get("away_team", mdata.get("away_team", ""))
        dc_api_names = {
            "1X": [f"{home_t} or Draw", "1X"],
            "X2": [f"{away_t} or Draw", "X2"],
            "12": [f"{home_t} or {away_t}", "12"],
        }
        for sel, info in dc.items():
            prob = info.get("prob", 0)
            fair = info.get("fair_odds", 0)
            if prob > 0 and fair > 1:
                # Look up real bookmaker odds by trying API name variants
                bk_odds = 0
                for variant in dc_api_names.get(sel, [sel]):
                    entry = ext_dc.get(variant, {})
                    if entry.get("best", 0) > 1:
                        bk_odds = entry["best"]
                        break
                use_odds = bk_odds if bk_odds > 1 else fair
                val = ((prob * use_odds) - 1) * 100
                if val >= MIN_LEG_VALUE_PCT:
                    legs.append({
                        "match": match_key, "date": "",
                        "market": "double_chance", "selection": sel,
                        "odds": round(use_odds, 2),
                        "probability": round(prob, 4),
                        "value_pct": round(val, 1),
                        "confidence": "MEDIUM",
                        "factors": [],
                        "source": "extended" + ("_live_odds" if bk_odds > 1 else ""),
                    })

        # Team totals
        tt = mdata.get("team_totals", {})
        for side in ("home", "away"):
            for line, info in tt.get(side, {}).items():
                prob = info.get("prob", 0)
                fair = info.get("fair_odds", 0)
                if prob > 0.35 and fair > 1:
                    val = ((prob * fair) - 1) * 100
                    if val >= MIN_LEG_VALUE_PCT:
                        team = mdata.get(f"{side}_team", side.title())
                        legs.append({
                            "match": match_key, "date": "",
                            "market": "team_totals",
                            "selection": f"{team} {line.replace('_', ' ')}",
                            "odds": round(fair, 2),
                            "probability": round(prob, 4),
                            "value_pct": round(val, 1),
                            "confidence": "MEDIUM",
                            "factors": [],
                            "source": "extended",
                        })

        # First half
        fh = mdata.get("first_half", {})
        ou_fh = fh.get("over_under", {})
        for line, info in ou_fh.items():
            if not isinstance(info, dict):
                continue
            prob = info.get("prob", 0)
            fair = info.get("fair_odds", 0)
            if prob > 0.35 and fair > 1:
                val = ((prob * fair) - 1) * 100
                if val >= MIN_LEG_VALUE_PCT:
                    legs.append({
                        "match": match_key, "date": "",
                        "market": "first_half",
                        "selection": line.replace("_", " "),
                        "odds": round(fair, 2),
                        "probability": round(prob, 4),
                        "value_pct": round(val, 1),
                        "confidence": "MEDIUM",
                        "factors": [],
                        "source": "extended",
                    })

        # Multi-goal ranges
        mg = mdata.get("multi_goal", [])
        for g in mg:
            prob = g.get("prob", 0)
            fair = g.get("fair_odds", 0)
            if prob > 0.40 and fair > 1:
                val = ((prob * fair) - 1) * 100
                if val >= MIN_LEG_VALUE_PCT:
                    legs.append({
                        "match": match_key, "date": "",
                        "market": "multi_goal",
                        "selection": f"{g['range']} goals",
                        "odds": round(fair, 2),
                        "probability": round(prob, 4),
                        "value_pct": round(val, 1),
                        "confidence": "MEDIUM",
                        "factors": [],
                        "source": "extended",
                    })

    # --- Alternate totals from extra odds (real bookmaker odds for O/U lines) ---
    # Cross-reference our model probabilities with real bookmaker prices
    # Only use standard half-integer lines (0.5, 1.5, 2.5, 3.5, 4.5)
    STANDARD_OU_LINES = {0.5, 1.5, 2.5, 3.5, 4.5}
    for mk, ext_match in extra_odds.items():
        alt_totals = ext_match.get("alternate_totals", {})
        gp = goal_preds.get(mk, {})
        pred = predictions.get(mk, {})
        if not alt_totals:
            continue
        for line_str, odds_info in alt_totals.items():
            try:
                line_val = float(line_str)
            except (ValueError, TypeError):
                continue
            if line_val not in STANDARD_OU_LINES:
                continue  # Skip non-standard lines (2.0, 2.75, etc.)
            # Map line to our model probability (try pre-computed first)
            line_key = f"over_{str(line_val).replace('.', '_')}"
            our_prob = gp.get(line_key, 0) if gp else 0
            # If no pre-computed prob, compute from Poisson using ensemble xG
            if our_prob <= 0 and pred:
                h_xg = pred.get("home_xg", 0)
                a_xg = pred.get("away_xg", 0)
                if h_xg > 0 and a_xg > 0:
                    total_xg = h_xg + a_xg
                    our_prob = 1.0 - sum(
                        poisson.pmf(g, total_xg) for g in range(int(line_val) + 1)
                    ) if line_val == int(line_val) else 1.0 - sum(
                        poisson.pmf(g, total_xg) for g in range(math.ceil(line_val))
                    )
                    our_prob = max(0, min(1, our_prob))
            bk_over = odds_info.get("best_over", 0)
            bk_under = odds_info.get("best_under", 0)
            # Over side
            if our_prob > 0.30 and bk_over > 1:
                val = ((our_prob * bk_over) - 1) * 100
                sel = f"OVER {line_val}"
                already = any(
                    l["match"] == mk and l["market"] == "totals" and l["selection"] == sel
                    for l in legs
                )
                if not already and val >= MIN_LEG_VALUE_PCT:
                    legs.append({
                        "match": mk, "date": gp.get("date", ""),
                        "market": "totals", "selection": sel,
                        "odds": round(bk_over, 2),
                        "probability": round(our_prob, 4),
                        "value_pct": round(val, 1),
                        "confidence": gp.get("confidence", "MEDIUM"),
                        "factors": gp.get("factors", []),
                        "source": "alt_totals_live_odds",
                    })
            # Under side
            under_prob = 1.0 - our_prob if our_prob > 0 else 0
            if under_prob > 0.30 and bk_under > 1:
                val = ((under_prob * bk_under) - 1) * 100
                sel = f"UNDER {line_val}"
                already = any(
                    l["match"] == mk and l["market"] == "totals" and l["selection"] == sel
                    for l in legs
                )
                if not already and val >= MIN_LEG_VALUE_PCT:
                    legs.append({
                        "match": mk, "date": gp.get("date", ""),
                        "market": "totals", "selection": sel,
                        "odds": round(bk_under, 2),
                        "probability": round(under_prob, 4),
                        "value_pct": round(val, 1),
                        "confidence": gp.get("confidence", "MEDIUM"),
                        "factors": gp.get("factors", []),
                        "source": "alt_totals_live_odds",
                    })

    # --- Double chance from extra odds (with real bookmaker odds) ---
    # Add legs for matches where we have real DC odds but no extended_markets entry
    for mk, ext_match in extra_odds.items():
        ext_dc = ext_match.get("double_chance", {})
        pred = predictions.get(mk, {})
        if not ext_dc or not pred:
            continue
        home_xg = pred.get("home_xg", 0)
        away_xg = pred.get("away_xg", 0)
        if not (home_xg > 0 and away_xg > 0):
            continue
        # Compute 1X2 probs from Poisson for DC probs
        p_h = sum(poisson.pmf(h, home_xg) * sum(poisson.pmf(a, away_xg) for a in range(h)) for h in range(8))
        p_d = sum(poisson.pmf(g, home_xg) * poisson.pmf(g, away_xg) for g in range(8))
        p_a = 1.0 - p_h - p_d
        dc_probs = {
            "1X": p_h + p_d,
            "X2": p_d + p_a,
            "12": p_h + p_a,
        }
        # Alternative name formats from the Odds API
        dc_name_map = {
            "1X": ["1X", f"{ext_match.get('home_team', '')} or Draw"],
            "X2": ["X2", f"{ext_match.get('away_team', '')} or Draw"],
            "12": ["12", f"{ext_match.get('home_team', '')} or {ext_match.get('away_team', '')}"],
        }
        for dc_key, prob in dc_probs.items():
            # Find matching odds entry
            bk_odds = 0
            for name_variant in dc_name_map.get(dc_key, [dc_key]):
                entry = ext_dc.get(name_variant, {})
                if entry.get("best", 0) > 1:
                    bk_odds = entry["best"]
                    break
            if bk_odds > 1 and prob > 0.30:
                val = ((prob * bk_odds) - 1) * 100
                already = any(
                    l["match"] == mk and l["market"] == "double_chance" and l["selection"] == dc_key
                    for l in legs
                )
                if not already and val >= MIN_LEG_VALUE_PCT:
                    legs.append({
                        "match": mk, "date": "",
                        "market": "double_chance", "selection": dc_key,
                        "odds": round(bk_odds, 2),
                        "probability": round(prob, 4),
                        "value_pct": round(val, 1),
                        "confidence": pred.get("confidence_level", "MEDIUM"),
                        "factors": [],
                        "source": "extra_odds_dc",
                    })

    # --- Draw No Bet from extra odds (real bookmaker odds) ---
    for mk, ext_match in extra_odds.items():
        dnb = ext_match.get("draw_no_bet", {})
        pred = predictions.get(mk, {})
        if not dnb or not pred:
            continue
        home_xg = pred.get("home_xg", 0)
        away_xg = pred.get("away_xg", 0)
        if not (home_xg > 0 and away_xg > 0):
            continue
        # Compute win probs from Poisson (DNB removes draw)
        p_h = sum(poisson.pmf(h, home_xg) * sum(poisson.pmf(a, away_xg) for a in range(h)) for h in range(8))
        p_d = sum(poisson.pmf(g, home_xg) * poisson.pmf(g, away_xg) for g in range(8))
        p_a = max(0, 1.0 - p_h - p_d)
        # DNB returns stake on draw, so effective prob = win_prob / (1 - draw_prob)
        dnb_p_h = p_h / (1 - p_d) if p_d < 1 else 0
        dnb_p_a = p_a / (1 - p_d) if p_d < 1 else 0
        for side, sel_label, prob in [("home", "HOME", dnb_p_h), ("away", "AWAY", dnb_p_a)]:
            bk_odds = dnb.get(f"best_{side}", 0)
            # Cap at 15.0 to filter distorted lines from low-volume bookmakers
            if bk_odds > 15.0:
                bk_odds = dnb.get(side, 0)  # Try avg instead of best
            if bk_odds > 1 and bk_odds <= 15.0 and prob > 0.25:
                val = ((prob * bk_odds) - 1) * 100
                already = any(
                    l["match"] == mk and l["market"] == "draw_no_bet" and l["selection"] == sel_label
                    for l in legs
                )
                if not already and val >= MIN_LEG_VALUE_PCT:
                    legs.append({
                        "match": mk, "date": "",
                        "market": "draw_no_bet", "selection": sel_label,
                        "odds": round(bk_odds, 2),
                        "probability": round(prob, 4),
                        "value_pct": round(val, 1),
                        "confidence": pred.get("confidence_level", "MEDIUM"),
                        "factors": [],
                        "source": "extra_odds_dnb",
                    })

    # Deduplicate
    seen = set()
    unique = []
    for leg in legs:
        key = (_normalize_match(leg["match"]), leg["market"], leg["selection"])
        if key not in seen:
            seen.add(key)
            unique.append(leg)

    # Enrich each leg with quality signals
    for leg in unique:
        _enrich_leg(leg, predictions, bookmaker, market_intel, odds_movement)

    return unique


# ---------------------------------------------------------------------------
# Engine 2: Bivariate Poisson Score Matrix (Same-Game Parlays)
# ---------------------------------------------------------------------------

def _build_score_matrix(home_xg, away_xg, rho=POISSON_CORRELATION_RHO):
    """Build NxN score probability matrix via bivariate Poisson.

    Uses Karlis-Ntzoufras decomposition for proper correlation modeling
    instead of the ad-hoc multiplicative adjustment.
    """
    from scripts.betting.extended_markets import BivariatePoissonParams, bivariate_score_matrix

    params = BivariatePoissonParams(
        lambda_home=max(home_xg, 0.1),
        lambda_away=max(away_xg, 0.1),
        rho=rho,
    )
    return bivariate_score_matrix(params, max_goals=SCORE_MATRIX_SIZE - 1)


def _sgp_joint_probability(legs, matrix, home_xg, away_xg):
    """Compute joint probability for same-game parlay legs from score matrix."""
    n = SCORE_MATRIX_SIZE
    # Start with all scores possible
    mask = np.ones((n, n), dtype=bool)

    for leg in legs:
        sel = leg["selection"].upper()
        market = leg["market"]
        leg_mask = np.zeros((n, n), dtype=bool)

        if market == "h2h":
            if "HOME" in sel or sel == "1":
                for h in range(n):
                    for a in range(n):
                        if h > a:
                            leg_mask[h, a] = True
            elif "DRAW" in sel or sel == "X":
                for h in range(n):
                    leg_mask[h, h] = True
            elif "AWAY" in sel or sel == "2":
                for h in range(n):
                    for a in range(n):
                        if a > h:
                            leg_mask[h, a] = True

        elif market == "totals":
            line = _extract_line(sel)
            if line is not None:
                is_over = "OVER" in sel
                for h in range(n):
                    for a in range(n):
                        total = h + a
                        if is_over and total > line:
                            leg_mask[h, a] = True
                        elif not is_over and total < line:
                            leg_mask[h, a] = True
                        elif not is_over and "UNDER" in sel and total < line:
                            leg_mask[h, a] = True

        elif market == "btts":
            if "YES" in sel:
                for h in range(1, n):
                    for a in range(1, n):
                        leg_mask[h, a] = True
            else:
                for h in range(n):
                    for a in range(n):
                        if h == 0 or a == 0:
                            leg_mask[h, a] = True

        elif market == "team_totals":
            line = _extract_line(sel)
            if line is not None:
                is_over = "OVER" in sel or "over" in sel
                # Determine which team
                is_home = not any(kw in sel.lower() for kw in ["away"])
                for h in range(n):
                    for a in range(n):
                        team_goals = h if is_home else a
                        if is_over and team_goals > line:
                            leg_mask[h, a] = True
                        elif not is_over and team_goals < line:
                            leg_mask[h, a] = True

        elif market == "spreads":
            line = _extract_line(sel)
            if line is not None:
                is_home = "HOME" in sel
                for h in range(n):
                    for a in range(n):
                        margin = (h - a) if is_home else (a - h)
                        if margin > line:
                            leg_mask[h, a] = True

        elif market == "double_chance":
            if "1X" in sel:
                for h in range(n):
                    for a in range(n):
                        if h >= a:
                            leg_mask[h, a] = True
            elif "X2" in sel:
                for h in range(n):
                    for a in range(n):
                        if a >= h:
                            leg_mask[h, a] = True
            elif "12" in sel:
                for h in range(n):
                    for a in range(n):
                        if h != a:
                            leg_mask[h, a] = True

        elif market == "multi_goal":
            low, high = _extract_range(sel)
            for h in range(n):
                for a in range(n):
                    total = h + a
                    if low is not None and high is not None:
                        if low <= total <= high:
                            leg_mask[h, a] = True
                    elif low is not None:
                        if total >= low:
                            leg_mask[h, a] = True

        else:
            # Unknown market — use independence
            return None

        mask &= leg_mask

    joint = (matrix * mask).sum()
    return float(joint) if joint > 0 else None


def _extract_line(sel):
    """Extract numeric line from selection string like 'OVER 2.5' or 'HOME -1'."""
    import re
    match = re.search(r'[-+]?\d+\.?\d*', sel)
    if match:
        return float(match.group())
    return None


def _extract_range(sel):
    """Extract range from multi-goal selection like '1-3 goals' or '3+ goals'."""
    import re
    m = re.search(r'(\d+)-(\d+)', sel)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d+)\+', sel)
    if m:
        return int(m.group(1)), None
    return None, None


# ---------------------------------------------------------------------------
# Engine 3: Gaussian Copula (Cross-Match Correlation)
# ---------------------------------------------------------------------------

def _copula_joint_probability(legs):
    """Compute joint probability adjusted for cross-match correlation.

    Same-day matches get ρ=0.03, same-round ρ=0.01, different rounds ρ=0.
    """
    n = len(legs)
    if n < 2:
        return legs[0]["probability"] if legs else 0

    probs = [leg["probability"] for leg in legs]
    dates = [leg.get("date", "") for leg in legs]

    # Build correlation matrix
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if dates[i] and dates[j] and dates[i] == dates[j]:
                rho = COPULA_RHO_SAME_DAY
            elif dates[i] and dates[j]:
                rho = COPULA_RHO_SAME_ROUND
            else:
                rho = 0.0
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # If no correlation, use independence
    if np.allclose(corr_matrix, np.eye(n)):
        return float(np.prod(probs))

    # Transform marginals to standard normal via inverse CDF
    z = np.array([norm.ppf(p) for p in probs])

    # For small correlations, use analytic approximation:
    # P(all hit) ≈ prod(p_i) * (1 - sum of pairwise adjustments)
    # This avoids expensive multivariate CDF computation
    naive = float(np.prod(probs))
    adjustment = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rho = corr_matrix[i, j]
            if rho > 0:
                # For small rho: adjustment ≈ -rho * phi(z_i) * phi(z_j) / (p_i * p_j)
                # where phi is standard normal PDF
                phi_i = norm.pdf(z[i])
                phi_j = norm.pdf(z[j])
                adjustment += rho * phi_i * phi_j

    # The copula adjustment reduces the joint probability slightly
    adjusted = naive * (1.0 - adjustment / max(naive, 1e-10))
    # Ensure it doesn't exceed naive or go negative
    adjusted = max(adjusted * 0.95, adjusted)  # at minimum, small reduction
    adjusted = min(adjusted, naive)
    adjusted = max(adjusted, naive * 0.80)  # don't reduce more than 20%

    return float(adjusted)


# ---------------------------------------------------------------------------
# Engine 4: Multi-Signal Leg Quality Scoring
# ---------------------------------------------------------------------------

def _compute_leg_quality(leg):
    """Compute quality score (0-100) for a leg."""
    scores = {}

    # 1. Value edge (25%)
    vp = leg.get("value_pct", 0)
    scores["value_edge"] = min(vp / 30.0, 1.0) * 100

    # 2. Ensemble confidence (20%)
    conf = leg.get("confidence_level", "MEDIUM")
    scores["confidence"] = CONFIDENCE_MAP.get(conf, 25)

    # 3. Method agreement (15%) — lower std dev = higher quality
    ma = leg.get("method_agreement", 0.10)
    scores["method_agreement"] = max(0, (1 - ma / 0.15)) * 100

    # 4. Sharp alignment (15%)
    sa = leg.get("sharp_aligned")
    if sa is True:
        scores["sharp_alignment"] = 100
    elif sa is None:
        scores["sharp_alignment"] = 50
    else:
        scores["sharp_alignment"] = 0

    # 5. Market edge (10%)
    me = leg.get("market_edge", 0)
    scores["market_edge"] = min(me * 100 / 10.0, 1.0) * 100

    # 6. Intelligence signals (10%)
    ci = leg.get("composite_intel", 0)
    intel = ci * 100
    if leg.get("steam_aligned"):
        intel = min(100, intel + 20)
    scores["intel_signals"] = intel

    # 7. Momentum/form (5%)
    momentum = leg.get("momentum", "neutral")
    scores["momentum"] = 80 if momentum == "hot" else 50 if momentum == "neutral" else 20

    # 8. Probability bonus (for parlay legs: high-prob legs like Over 0.5/1.5 are
    # inherently high quality even without explicit bookmaker value comparison)
    prob = leg.get("probability", 0)
    if prob >= 0.80:
        scores["prob_bonus"] = 100  # Very high prob leg
    elif prob >= 0.70:
        scores["prob_bonus"] = 70
    elif prob >= 0.60:
        scores["prob_bonus"] = 40
    else:
        scores["prob_bonus"] = 0

    # Weighted composite (prob_bonus gets 15%, taken from value_edge 25%→15% and confidence 20%→15%)
    quality = (
        scores["value_edge"] * 0.15 +
        scores["confidence"] * 0.15 +
        scores["method_agreement"] * 0.15 +
        scores["sharp_alignment"] * 0.15 +
        scores["market_edge"] * 0.10 +
        scores["intel_signals"] * 0.10 +
        scores["prob_bonus"] * 0.15 +
        scores["momentum"] * 0.05
    )

    leg["quality_score"] = round(quality, 1)
    leg["quality_breakdown"] = {k: round(v, 1) for k, v in scores.items()}
    return quality


# ---------------------------------------------------------------------------
# Engine 5: Smart Combination Generator
# ---------------------------------------------------------------------------

def _legs_conflict(leg_a, leg_b):
    """Check if two legs from the same match are contradictory."""
    if _normalize_match(leg_a["match"]) != _normalize_match(leg_b["match"]):
        return False

    # Same market same match (redundant)
    if leg_a["market"] == leg_b["market"]:
        return True

    sel_a = leg_a["selection"].upper()
    sel_b = leg_b["selection"].upper()
    mkt_a = leg_a["market"]
    mkt_b = leg_b["market"]

    for (m1, s1), (m2, s2) in CONFLICTING_PAIRS:
        if ((mkt_a == m1 and s1 in sel_a and mkt_b == m2 and s2 in sel_b) or
                (mkt_a == m2 and s2 in sel_a and mkt_b == m1 and s1 in sel_b)):
            return True

    # Under totals + BTTS YES (general check)
    if mkt_a == "totals" and "UNDER" in sel_a and mkt_b == "btts" and "YES" in sel_b:
        line = _extract_line(sel_a)
        if line is not None and line <= 2.5:
            return True
    if mkt_b == "totals" and "UNDER" in sel_b and mkt_a == "btts" and "YES" in sel_a:
        line = _extract_line(sel_b)
        if line is not None and line <= 2.5:
            return True

    return False


def _compute_adaptive_tiers(valid_legs):
    """Compute tier thresholds from actual quality distribution.

    Uses percentile-based thresholds so tiers work regardless of
    how many signals are available in the data.
    """
    scores = sorted([l.get("quality_score", 0) for l in valid_legs], reverse=True)
    if not scores:
        return 70, 50  # defaults

    # T1 = top 25%, T2 = top 60%, T3 = rest
    t1_threshold = scores[max(0, len(scores) // 4 - 1)] if len(scores) >= 4 else scores[0]
    t2_threshold = scores[max(0, int(len(scores) * 0.6) - 1)] if len(scores) >= 2 else scores[-1]

    # Floor: T1 must be at least 40, T2 at least 30
    t1_threshold = max(t1_threshold, 40)
    t2_threshold = max(t2_threshold, MIN_LEG_QUALITY)

    return t1_threshold, t2_threshold


def _get_tier(quality, t1_thresh=70, t2_thresh=50):
    if quality >= t1_thresh:
        return 1
    elif quality >= t2_thresh:
        return 2
    return 3


def generate_combinations(legs, max_legs=4):
    """Generate smart combinations with tier-based filtering."""
    # Pre-filter
    # Standard quality + value filter, PLUS high-probability legs (>= 70% model prob)
    # that may lack explicit value_pct (e.g. Over 0.5 from goal_predictions with no
    # bookmaker odds comparison) but are still excellent parlay legs due to high hit rate.
    valid = [l for l in legs
             if (l.get("quality_score", 0) >= MIN_LEG_QUALITY
                 and l.get("value_pct", 0) >= QUALITY_MIN_VALUE)
             or l.get("probability", 0) >= 0.70]

    if len(valid) < 2:
        return []

    # Adaptive tier thresholds based on actual data distribution
    t1_thresh, t2_thresh = _compute_adaptive_tiers(valid)

    # Tier classification
    for leg in valid:
        leg["_tier"] = _get_tier(leg.get("quality_score", 0), t1_thresh, t2_thresh)

    combos = []
    combo_id = 0

    # Limit input to top legs to prevent combinatorial explosion
    # C(25,4) = 12650 which is manageable; C(40,4) = 91390 which is slow
    max_valid = min(len(valid), 25)
    # Sort by quality, breaking ties with probability (high-prob legs make great parlay legs)
    valid_sorted = sorted(valid, key=lambda l: (l.get("quality_score", 0), l.get("probability", 0)),
                          reverse=True)[:max_valid]

    MAX_COMBOS_PER_SIZE = 200  # cap per leg count

    for n in range(2, min(max_legs + 1, len(valid_sorted) + 1)):
        n_combos = 0
        for combo in combinations(valid_sorted, n):
            if n_combos >= MAX_COMBOS_PER_SIZE:
                break

            combo = list(combo)

            # Skip if any pair conflicts
            has_conflict = False
            for i, j in combinations(range(len(combo)), 2):
                if _legs_conflict(combo[i], combo[j]):
                    has_conflict = True
                    break
            if has_conflict:
                continue

            tiers = [l["_tier"] for l in combo]
            t1_count = tiers.count(1)
            t12_count = tiers.count(1) + tiers.count(2)
            # High-prob legs (>= 70%) count as honorary T1 for tier rules
            high_prob_count = sum(1 for l in combo if l.get("probability", 0) >= 0.70)

            # Tier rules: at least 1 T1 or T2 leg, higher combos need more quality
            # High-prob legs bypass tier requirements (they're excellent parlay legs)
            if n == 2 and t12_count < 1 and high_prob_count < 1:
                continue
            elif n == 3 and (t1_count + high_prob_count < 1 or t12_count + high_prob_count < 2):
                continue
            elif n == 4 and t1_count + high_prob_count < 2:
                continue

            combo_id += 1
            n_combos += 1
            combos.append({
                "id": f"PRL-{combo_id:03d}",
                "legs": combo,
                "n_legs": n,
            })

    # Also generate SGP combos (same match, different markets)
    matches = defaultdict(list)
    for leg in valid:
        matches[_normalize_match(leg["match"])].append(leg)

    for match_key, match_legs in matches.items():
        if len(match_legs) < 2:
            continue
        for n in range(2, min(4, len(match_legs) + 1)):
            for combo in combinations(match_legs, n):
                combo = list(combo)
                # All different markets
                markets = [l["market"] for l in combo]
                if len(set(markets)) < len(markets):
                    continue
                # SGP quality threshold
                if all(l.get("quality_score", 0) >= 40 for l in combo):
                    # Check conflicts
                    has_conflict = False
                    for i, j in combinations(range(len(combo)), 2):
                        if _legs_conflict(combo[i], combo[j]):
                            has_conflict = True
                            break
                    if has_conflict:
                        continue
                    combo_id += 1
                    combos.append({
                        "id": f"PRL-{combo_id:03d}",
                        "legs": combo,
                        "n_legs": n,
                        "_is_sgp": True,
                    })

    return combos


# ---------------------------------------------------------------------------
# Engine 6: Parlay Valuation & Kelly Sizing
# ---------------------------------------------------------------------------

def _compute_hit_probability(combo, predictions_data):
    """Compute hit probability using appropriate method per combo type."""
    legs = combo["legs"]
    matches = set(_normalize_match(l["match"]) for l in legs)
    is_sgp = len(matches) == 1
    combo["is_same_game"] = is_sgp

    naive_prob = float(np.prod([l["probability"] for l in legs]))
    combo["hit_probability"] = {"naive": round(naive_prob, 6)}

    if is_sgp:
        # Use bivariate Poisson score matrix
        match_key = _normalize_match(legs[0]["match"])
        pred = predictions_data.get(match_key, {})
        comp = pred.get("component_predictions", {})
        xg_details = comp.get("xg_details", {})
        home_xg = xg_details.get("home_xg", 0)
        away_xg = xg_details.get("away_xg", 0)

        if home_xg <= 0 or away_xg <= 0:
            # Fallback: try from extended markets
            ext = _load_json(UPCOMING_DIR / "extended_markets.json")
            ext_match = ext.get("matches", {}).get(match_key, {})
            home_xg = ext_match.get("home_xg", 1.3)
            away_xg = ext_match.get("away_xg", 1.0)

        # Store xG for MC engine
        combo["_home_xg"] = home_xg
        combo["_away_xg"] = away_xg

        # Only use score matrix for markets it supports
        matrix_legs = [l for l in legs if l["market"] in SCORE_MATRIX_MARKETS]
        other_legs = [l for l in legs if l["market"] not in SCORE_MATRIX_MARKETS]

        if matrix_legs:
            matrix = _build_score_matrix(home_xg, away_xg)
            joint = _sgp_joint_probability(matrix_legs, matrix, home_xg, away_xg)
            if joint is not None:
                # Multiply by independent probs for non-matrix legs
                for ol in other_legs:
                    joint *= ol["probability"]
                combo["hit_probability"]["copula_adjusted"] = round(joint, 6)
                combo["hit_probability"]["method"] = "bivariate_poisson"
            else:
                combo["hit_probability"]["copula_adjusted"] = round(naive_prob, 6)
                combo["hit_probability"]["method"] = "independence_fallback"
        else:
            combo["hit_probability"]["copula_adjusted"] = round(naive_prob, 6)
            combo["hit_probability"]["method"] = "independence"
    else:
        # Cross-match: use Gaussian copula
        adj = _copula_joint_probability(legs)
        combo["hit_probability"]["copula_adjusted"] = round(adj, 6)
        combo["hit_probability"]["method"] = "gaussian_copula"

    return combo["hit_probability"].get("copula_adjusted", naive_prob)


def _adaptive_sim_count(base_prob: float, n_legs: int) -> int:
    """Choose sim count to target CV < 5% of hit rate estimator.

    For a Bernoulli proportion p, Var(p_hat) = p(1-p)/n.
    CV = sqrt(Var) / p = sqrt((1-p)/(n*p)).
    Solve for n: n >= (1-p) / (CV_target^2 * p).
    With CV_target = 0.05: n >= (1-p) / (0.0025 * p).
    """
    p = max(base_prob, 1e-6)
    n_needed = int(math.ceil((1 - p) / (0.0025 * p)))
    # Scale up slightly for multi-leg combos (higher variance)
    n_needed = int(n_needed * (1 + 0.1 * (n_legs - 1)))
    return max(5000, min(n_needed, 100000))


def _leg_to_outcome_key(leg: dict) -> str | None:
    """Map a leg dict to a derive_market_outcomes() key.

    Returns None for markets not supported by the score matrix sampler,
    which signals the MC engine to use Beta-Bernoulli fallback for that leg.
    """
    import re

    market = leg["market"]
    sel = leg["selection"].upper()

    if market == "h2h":
        if "HOME" in sel or sel == "1":
            return "h2h_home"
        elif "DRAW" in sel or sel == "X":
            return "h2h_draw"
        elif "AWAY" in sel or sel == "2":
            return "h2h_away"

    elif market == "btts":
        return "btts_yes" if "YES" in sel else "btts_no"

    elif market == "totals":
        line_match = re.search(r'(\d+\.?\d*)', sel)
        if line_match:
            line = line_match.group(1)
            direction = "over" if "OVER" in sel else "under"
            return f"totals_{direction}_{line}"

    elif market == "team_totals":
        line_match = re.search(r'(\d+\.?\d*)', sel)
        if line_match:
            line = line_match.group(1)
            is_away = any(kw in sel.lower() for kw in ["away"])
            team = "away" if is_away else "home"
            direction = "over" if "OVER" in sel else "under"
            return f"team_totals_{team}_{direction}_{line}"

    elif market == "double_chance":
        if "1X" in sel:
            return "double_chance_1x"
        elif "X2" in sel:
            return "double_chance_x2"
        elif "12" in sel:
            return "double_chance_12"

    elif market == "spreads":
        line_match = re.search(r'([-+]?\d+\.?\d*)', sel)
        if line_match:
            line = line_match.group(1)
            team = "home" if "HOME" in sel else "away"
            return f"spreads_{team}_{line}"

    elif market == "multi_goal":
        range_match = re.search(r'(\d+)-(\d+)', sel)
        if range_match:
            return f"multi_goal_{range_match.group(1)}-{range_match.group(2)}"
        plus_match = re.search(r'(\d+)\+', sel)
        if plus_match:
            return f"multi_goal_{plus_match.group(1)}+"

    return None


def _mc_sgp(combo: dict, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo for same-game parlays via bivariate Poisson sampling.

    Samples correlated (home_goals, away_goals) and derives all market
    outcomes from the score, naturally capturing intra-match correlation
    without any copula.

    Returns boolean hit array of length n_sims.
    """
    from scripts.betting.extended_markets import (
        BivariatePoissonParams, sample_scores, derive_market_outcomes,
    )
    from scipy.stats import beta as beta_dist

    legs = combo["legs"]
    home_xg = combo.get("_home_xg", 1.3)
    away_xg = combo.get("_away_xg", 1.0)

    params = BivariatePoissonParams(
        lambda_home=max(home_xg, 0.1),
        lambda_away=max(away_xg, 0.1),
        rho=POISSON_CORRELATION_RHO,
    )

    # Sample scores
    hg, ag = sample_scores(params, n_samples=n_sims, rng=rng)
    outcomes = derive_market_outcomes(hg, ag)

    # Check each leg
    all_hit = np.ones(n_sims, dtype=bool)
    for leg in legs:
        key = _leg_to_outcome_key(leg)
        if key is not None and key in outcomes:
            all_hit &= outcomes[key]
        else:
            # Fallback: Beta-Bernoulli for unsupported markets
            prob = leg["probability"]
            conf = leg.get("confidence_level", "MEDIUM")
            k = _get_beta_k(conf)
            a = max(prob * k, 0.5)
            b = max((1 - prob) * k, 0.5)
            sampled_probs = rng.beta(a, b, size=n_sims)
            all_hit &= (rng.random(n_sims) < sampled_probs)

    return all_hit


def _mc_cross_match(combo: dict, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo for cross-match parlays via Gaussian copula.

    Builds correlation matrix from dates (same-day → ρ=0.03),
    uses Cholesky decomposition to generate correlated standard normals,
    transforms through CDF to get correlated uniform → Beta → Bernoulli.

    Returns boolean hit array of length n_sims.
    """
    from scipy.stats import beta as beta_dist

    legs = combo["legs"]
    n = len(legs)

    probs = np.array([leg["probability"] for leg in legs])
    dates = [leg.get("date", "") for leg in legs]

    # Build correlation matrix
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if dates[i] and dates[j] and dates[i] == dates[j]:
                rho = COPULA_RHO_SAME_DAY
            elif dates[i] and dates[j]:
                rho = COPULA_RHO_SAME_ROUND
            else:
                rho = 0.0
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho

    # If no correlation, use independent Beta-Bernoulli
    has_correlation = not np.allclose(corr_matrix, np.eye(n))

    if has_correlation:
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            # Cholesky failed — fall back to identity (independent)
            import logging
            logging.getLogger(__name__).warning(
                "Cholesky decomposition failed for cross-match MC, "
                "falling back to independent simulation"
            )
            has_correlation = False

    all_hit = np.ones(n_sims, dtype=bool)

    if has_correlation:
        # Correlated standard normals → uniform via Phi → Beta ppf → Bernoulli
        Z_indep = rng.standard_normal((n_sims, n))
        Z_corr = Z_indep @ L.T  # (n_sims, n) correlated normals
        U = norm.cdf(Z_corr)  # transform to uniform [0,1]

        for i, leg in enumerate(legs):
            prob = leg["probability"]
            conf = leg.get("confidence_level", "MEDIUM")
            k = _get_beta_k(conf)
            a = max(prob * k, 0.5)
            b = max((1 - prob) * k, 0.5)
            # Map uniform through Beta inverse CDF to get correlated Beta samples
            sampled_probs = beta_dist.ppf(np.clip(U[:, i], 1e-10, 1 - 1e-10), a, b)
            all_hit &= (rng.random(n_sims) < sampled_probs)
    else:
        # Independent Beta-Bernoulli
        for leg in legs:
            prob = leg["probability"]
            conf = leg.get("confidence_level", "MEDIUM")
            k = _get_beta_k(conf)
            a = max(prob * k, 0.5)
            b = max((1 - prob) * k, 0.5)
            sampled_probs = rng.beta(a, b, size=n_sims)
            all_hit &= (rng.random(n_sims) < sampled_probs)

    return all_hit


def _get_beta_k(confidence_level: str) -> float:
    """Get Beta K parameter, using calibrated values if available."""
    return _CALIBRATED_BETA_K.get(confidence_level,
                                  CONFIDENCE_K.get(confidence_level, 15))


def _monte_carlo_bands(combo):
    """Run Monte Carlo simulation for uncertainty bands.

    Dispatches to SGP (bivariate Poisson) or cross-match (Gaussian copula)
    MC engine. Uses adaptive sim count and bootstrap for percentile bands.

    Output contract unchanged: sets combo["hit_probability"]["median/p10/p90"]
    and combo["expected_roi"]. Also adds combo["hit_probability"]["mc_method"]
    and combo["hit_probability"]["n_sims"].
    """
    legs = combo["legs"]
    matches = set(_normalize_match(l["match"]) for l in legs)
    is_sgp = len(matches) == 1
    combined_odds = float(np.prod([l["odds"] for l in legs]))

    # Adaptive sim count based on expected hit rate
    base_prob = combo["hit_probability"].get("copula_adjusted",
                                              combo["hit_probability"].get("naive", 0.01))
    n_sims = _adaptive_sim_count(base_prob, len(legs))

    rng = np.random.default_rng()

    if is_sgp:
        hit_mask = _mc_sgp(combo, n_sims, rng)
        mc_method = "bivariate_poisson_sgp"
    else:
        hit_mask = _mc_cross_match(combo, n_sims, rng)
        mc_method = "gaussian_copula_cross"

    # Compute payout array
    results = np.where(hit_mask, combined_odds, 0.0)

    # Bootstrap for percentile bands
    boot_rates = []
    boot_size = min(500, n_sims)
    n_boot = 200
    for _ in range(n_boot):
        sample = rng.choice(hit_mask.astype(float), size=boot_size, replace=True)
        boot_rates.append(sample.mean())
    boot_rates = np.array(boot_rates)

    combo["hit_probability"]["median"] = round(float(np.median(boot_rates)), 6)
    combo["hit_probability"]["p10"] = round(float(np.percentile(boot_rates, 10)), 6)
    combo["hit_probability"]["p90"] = round(float(np.percentile(boot_rates, 90)), 6)
    combo["hit_probability"]["mc_method"] = mc_method
    combo["hit_probability"]["n_sims"] = n_sims

    # Expected ROI
    avg_payout = results.mean()
    combo["expected_roi"] = round(float(avg_payout - 1.0), 4)


def _kelly_parlay_stake(combo, bankroll):
    """Compute Kelly-based stake for a parlay."""
    adj_prob = combo["hit_probability"].get("copula_adjusted", 0)
    combined_odds = float(np.prod([l["odds"] for l in combo["legs"]]))

    b = combined_odds - 1
    if b <= 0 or adj_prob <= 0:
        return 0, combined_odds

    full_kelly = (b * adj_prob - (1 - adj_prob)) / b
    if full_kelly <= 0:
        return 0, combined_odds

    # Quality-weighted confidence
    avg_quality = np.mean([l.get("quality_score", 50) for l in combo["legs"]])
    confidence_factor = avg_quality / 100.0

    # Variance penalty for multi-leg
    n_legs = combo["n_legs"]
    variance_penalty = 1.0 / math.sqrt(n_legs)

    stake = bankroll * full_kelly * KELLY_FRACTION * confidence_factor * variance_penalty
    max_stake = bankroll * MAX_STAKE_PCT
    stake = min(stake, max_stake)
    stake = max(stake, 0)

    return round(stake, 2), round(combined_odds, 2)


# ---------------------------------------------------------------------------
# Engine 7: Categorization & Ranking
# ---------------------------------------------------------------------------

def _diversification_score(combo):
    """Compute diversification metrics."""
    legs = combo["legs"]
    unique_markets = len(set(l["market"] for l in legs))
    unique_matches = len(set(_normalize_match(l["match"]) for l in legs))

    # Score 0-100
    market_div = min(unique_markets / 3.0, 1.0) * 50
    match_div = min(unique_matches / 3.0, 1.0) * 50
    score = market_div + match_div

    # Bonus for diversity
    avg_quality = np.mean([l.get("quality_score", 50) for l in legs])
    if unique_markets >= 3:
        avg_quality += 5
    if unique_matches >= 3:
        avg_quality += 3
    # Penalty for concentration
    if unique_markets == 1:
        avg_quality -= 10

    return {
        "markets": unique_markets,
        "matches": unique_matches,
        "score": round(score, 1),
    }, avg_quality


def _parlay_quality_score(combo):
    """Compute composite parlay quality (0-100)."""
    ev = combo.get("expected_roi", 0)
    ev_score = min(max(ev * 100, 0), 100)

    avg_leg_quality = np.mean([l.get("quality_score", 50) for l in combo["legs"]])

    hit_median = combo.get("hit_probability", {}).get("median", 0)
    hit_score = min(hit_median * 500, 100)  # scale: 20% hit = 100

    div = combo.get("diversification", {}).get("score", 50)

    sharp_pct = combo.get("sharp_alignment_pct", 50)

    quality = (
        ev_score * 0.30 +
        avg_leg_quality * 0.25 +
        hit_score * 0.20 +
        div * 0.15 +
        sharp_pct * 0.10
    )

    return round(quality, 1)


def categorize_parlays(combos):
    """Sort parlays into 6 categories."""
    categories = {
        "safe_doubles": [],
        "value_trebles": [],
        "long_shots": [],
        "same_game": [],
        "sharp_specials": [],
        "banker_combos": [],
    }

    for p in combos:
        if not p.get("stake") or p["stake"] <= 0:
            continue

        n = p["n_legs"]
        hit_med = p.get("hit_probability", {}).get("median", 0)
        avg_q = np.mean([l.get("quality_score", 50) for l in p["legs"]])
        combined_odds = p.get("combined_odds", 1)
        sharp_pct = p.get("sharp_alignment_pct", 0)
        is_sgp = p.get("is_same_game", False)
        vp = p.get("value_pct", 0)

        assigned = False

        # Same-game parlays
        if is_sgp:
            categories["same_game"].append(p)
            p["category"] = "same_game"
            assigned = True

        # Banker combos FIRST — high-probability legs are the "safe parlay" concept.
        # Must be checked BEFORE sharp_specials because many high-prob legs are also
        # sharp-aligned, and we want to surface them as banker combos for the user.
        max_prob = max(l["probability"] for l in p["legs"])
        min_prob = min(l["probability"] for l in p["legs"])
        avg_prob = np.mean([l["probability"] for l in p["legs"]])
        if not assigned and (
            (max_prob >= 0.70 and n <= 4) or  # 1 banker leg + up to 3 others
            (min_prob >= 0.60 and avg_prob >= 0.65 and n <= 4)  # all legs high-prob
        ):
            categories["banker_combos"].append(p)
            p["category"] = "banker_combos"
            assigned = True

        # Sharp specials: all legs sharp-aligned (only if not already a banker combo)
        if not assigned and sharp_pct >= 90 and avg_q >= 50:
            p["category"] = "sharp_specials"
            categories["sharp_specials"].append(p)
            assigned = True

        if assigned:
            continue

        # Safe doubles
        if n == 2 and hit_med >= 0.10 and avg_q >= 55:
            categories["safe_doubles"].append(p)
            p["category"] = "safe_doubles"
        # Value trebles
        elif n == 3 and vp >= 10:
            categories["value_trebles"].append(p)
            p["category"] = "value_trebles"
        # Long shots
        elif combined_odds >= 8.0 or n >= 4:
            categories["long_shots"].append(p)
            p["category"] = "long_shots"
        # Default: by legs
        elif n == 2:
            categories["safe_doubles"].append(p)
            p["category"] = "safe_doubles"
        elif n == 3:
            categories["value_trebles"].append(p)
            p["category"] = "value_trebles"
        else:
            categories["long_shots"].append(p)
            p["category"] = "long_shots"

    # Sort and trim each category
    sort_keys = {
        "safe_doubles": lambda x: x.get("hit_probability", {}).get("median", 0),
        "value_trebles": lambda x: x.get("expected_roi", 0),
        "long_shots": lambda x: x.get("expected_roi", 0),
        "same_game": lambda x: x.get("value_pct", 0),
        "sharp_specials": lambda x: x.get("parlay_quality", 0),
        "banker_combos": lambda x: x.get("hit_probability", {}).get("median", 0),
    }

    for cat_key, items in categories.items():
        if cat_key == "banker_combos" and len(items) > TOP_N_PER_CATEGORY:
            # Ensure variety in banker combos: top 5 by hit rate + top 3 multi-leg by EV.
            # Without this, 2-folds with 75%+ hit rate always dominate and 3-4 fold
            # "safe parlays" (like 3x Over 1.5 @2.32) get cut.
            items.sort(key=sort_keys["banker_combos"], reverse=True)
            top_by_hit = items[:5]
            remaining = [p for p in items if p not in top_by_hit]
            # Prefer multi-leg parlays: prioritize "safest" (highest min_prob) first,
            # then fill with highest EV. This ensures the user's "safe parlay" concept
            # (3-4 legs all at 70%+ prob) appears even if their EV is lower than
            # mixed combos with one longshot leg.
            multi_leg = [p for p in remaining if p.get("n_legs", 0) >= 3]
            multi_leg.sort(key=lambda x: (
                min(l.get("probability", 0) for l in x.get("legs", [{"probability": 0}])),
                x.get("expected_roi", 0)
            ), reverse=True)
            top_multi = multi_leg[:3]
            # Fill remaining slots from leftover
            used = set(id(p) for p in top_by_hit + top_multi)
            leftover = [p for p in items if id(p) not in used]
            leftover.sort(key=sort_keys["banker_combos"], reverse=True)
            final = top_by_hit + top_multi + leftover
            items[:] = final[:TOP_N_PER_CATEGORY + 2]  # Allow 10 for banker combos
        else:
            items.sort(key=sort_keys.get(cat_key, lambda x: 0), reverse=True)
            del items[TOP_N_PER_CATEGORY:]

    return categories


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def _format_parlay_output(combo):
    """Format a combo into the output schema."""
    legs_out = []
    for leg in combo["legs"]:
        legs_out.append({
            "match": leg["match"],
            "market": leg["market"],
            "selection": leg["selection"],
            "odds": leg["odds"],
            "probability": leg["probability"],
            "value_pct": leg.get("value_pct", 0),
            "quality_score": leg.get("quality_score", 0),
            "quality_breakdown": leg.get("quality_breakdown", {}),
            "sharp_aligned": leg.get("sharp_aligned"),
            "confidence_level": leg.get("confidence_level", "MEDIUM"),
            "source": leg.get("source", ""),
        })

    combined_odds = combo.get("combined_odds", 1)
    stake = combo.get("stake", 0)

    # Sharp alignment percentage
    sharp_count = sum(1 for l in combo["legs"] if l.get("sharp_aligned") is True)
    total_with_data = sum(1 for l in combo["legs"] if l.get("sharp_aligned") is not None)
    sharp_pct = round((sharp_count / total_with_data) * 100) if total_with_data > 0 else 50

    return {
        "id": combo["id"],
        "n_legs": combo["n_legs"],
        "legs": legs_out,
        "combined_odds": combined_odds,
        "hit_probability": combo.get("hit_probability", {}),
        "value_pct": combo.get("value_pct", 0),
        "expected_roi": combo.get("expected_roi", 0),
        "stake": stake,
        "potential_profit": round(stake * (combined_odds - 1), 2) if stake > 0 else 0,
        "parlay_quality": combo.get("parlay_quality", 0),
        "diversification": combo.get("diversification", {}),
        "correlation_warning": combo.get("correlation_warning"),
        "is_same_game": combo.get("is_same_game", False),
        "sharp_alignment_pct": sharp_pct,
        "category": combo.get("category", ""),
    }


def generate_parlay_report(bankroll=None):
    """Main entry point — full 7-engine parlay generation pipeline."""
    if bankroll is None:
        bankroll = _get_bankroll()

    print("  [Engine 1] Loading value legs from all markets...")
    legs = load_all_value_legs()
    markets_found = set(l["market"] for l in legs)
    print(f"  Found {len(legs)} legs across {len(markets_found)} markets: {', '.join(sorted(markets_found))}")

    if len(legs) < 2:
        print("  Not enough value legs to generate parlays")
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_parlays": 0,
            "total_legs_available": len(legs),
            "legs_by_market": {},
            "bankroll": bankroll,
            "model_info": _model_info(),
            "categories": {k: [] for k in [
                "safe_doubles", "value_trebles", "long_shots",
                "same_game", "sharp_specials", "banker_combos"
            ]},
        }
        _save_report(report)
        return report

    # Engine 4: Score all legs
    print("  [Engine 4] Computing leg quality scores...")
    for leg in legs:
        _compute_leg_quality(leg)
    qualified = [l for l in legs if l.get("quality_score", 0) >= MIN_LEG_QUALITY]
    print(f"  {len(qualified)}/{len(legs)} legs passed quality threshold (>={MIN_LEG_QUALITY})")

    # Legs by market count
    legs_by_market = defaultdict(int)
    for l in legs:
        legs_by_market[l["market"]] += 1

    # Engine 5: Generate combinations
    print("  [Engine 5] Generating smart combinations...")
    combos = generate_combinations(legs)
    print(f"  Generated {len(combos)} candidate parlays")

    if not combos:
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_parlays": 0,
            "total_legs_available": len(legs),
            "legs_by_market": dict(legs_by_market),
            "bankroll": bankroll,
            "model_info": _model_info(),
            "categories": {k: [] for k in [
                "safe_doubles", "value_trebles", "long_shots",
                "same_game", "sharp_specials", "banker_combos"
            ]},
        }
        _save_report(report)
        return report

    # Engine 2 & 3: Compute hit probabilities
    print("  [Engine 2/3] Computing hit probabilities (Poisson + Copula)...")
    predictions_data = {}
    preds_raw = _load_json(UPCOMING_DIR / "predictions.json")
    for p in preds_raw.get("predictions", []):
        predictions_data[_normalize_match(p.get("match", ""))] = p

    for combo in combos:
        _compute_hit_probability(combo, predictions_data)

    # Engine 6: Monte Carlo + Kelly sizing
    print(f"  [Engine 6] Running Monte Carlo ({MONTE_CARLO_SIMS} sims) + Kelly sizing...")
    valued_combos = []
    for combo in combos:
        adj_prob = combo["hit_probability"].get("copula_adjusted", 0)
        combined_odds = float(np.prod([l["odds"] for l in combo["legs"]]))
        combo["combined_odds"] = round(combined_odds, 2)

        value = (adj_prob * combined_odds - 1) * 100
        combo["value_pct"] = round(value, 1)
        if value < 3.0:
            continue

        _monte_carlo_bands(combo)

        stake, _ = _kelly_parlay_stake(combo, bankroll)
        combo["stake"] = stake
        if stake <= 0:
            continue

        # Diversification
        div, adj_quality = _diversification_score(combo)
        combo["diversification"] = div

        # Sharp alignment
        sharp_count = sum(1 for l in combo["legs"] if l.get("sharp_aligned") is True)
        total_data = sum(1 for l in combo["legs"] if l.get("sharp_aligned") is not None)
        combo["sharp_alignment_pct"] = round((sharp_count / total_data) * 100) if total_data > 0 else 50

        # Parlay quality
        combo["parlay_quality"] = _parlay_quality_score(combo)

        # Correlation warning
        match_set = set(_normalize_match(l["match"]) for l in combo["legs"])
        if len(match_set) < combo["n_legs"] and not combo.get("is_same_game"):
            same_pairs = sum(
                1 for i, j in combinations(range(combo["n_legs"]), 2)
                if _normalize_match(combo["legs"][i]["match"]) == _normalize_match(combo["legs"][j]["match"])
            )
            combo["correlation_warning"] = f"{same_pairs} correlated leg pair(s) — probability adjusted via score matrix"
        else:
            combo["correlation_warning"] = None

        valued_combos.append(combo)

    print(f"  {len(valued_combos)} parlays with positive value and stake")

    # Engine 7: Categorize
    print("  [Engine 7] Categorizing and ranking...")
    categories = categorize_parlays(valued_combos)

    # Format output
    formatted_categories = {}
    for cat_key, items in categories.items():
        formatted_categories[cat_key] = [_format_parlay_output(c) for c in items]

    total = sum(len(v) for v in formatted_categories.values())

    # Print summary
    for cat_key, items in formatted_categories.items():
        if items:
            print(f"    {cat_key}: {len(items)}")

    report = {
        "generated_at": datetime.now().isoformat(),
        "total_parlays": total,
        "total_legs_available": len(legs),
        "legs_by_market": dict(legs_by_market),
        "bankroll": bankroll,
        "model_info": _model_info(),
        "categories": formatted_categories,
    }

    _save_report(report)
    return report


def _model_info():
    return {
        "copula_rho_same_day": COPULA_RHO_SAME_DAY,
        "poisson_correlation": POISSON_CORRELATION_RHO,
        "monte_carlo_sims": MONTE_CARLO_SIMS,
        "kelly_fraction": KELLY_FRACTION,
        "quality_threshold": MIN_LEG_QUALITY,
        "min_value_pct": MIN_LEG_VALUE_PCT,
    }


def _save_report(report):
    BETTING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BETTING_DIR / "parlay_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, cls=_NumpySafeEncoder)
    print(f"  Saved {report['total_parlays']} parlays -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bk = None
    if "--bankroll" in sys.argv:
        idx = sys.argv.index("--bankroll")
        if idx + 1 < len(sys.argv):
            bk = float(sys.argv[idx + 1])

    report = generate_parlay_report(bankroll=bk)
    print(f"\nDone: {report['total_parlays']} parlays generated across "
          f"{len([k for k, v in report['categories'].items() if v])} categories")
