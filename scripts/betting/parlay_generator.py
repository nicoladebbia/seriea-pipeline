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

import hashlib
import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import poisson, norm

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, UPCOMING_DIR

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

# Realistic value cap: any single leg claiming >80% value is almost certainly
# a model error, not a real edge. Sharp markets don't leave 80%+ on the table.
MAX_LEG_VALUE_PCT = 80.0

# Minimum odds for parlay legs: legs below this add vig without selectivity.
# Over 0.5 goals @1.06 adds 6% vig for a 95% prob event — pure filler.
MIN_PARLAY_LEG_ODDS = 1.15

# Maximum single-leg concentration: no leg should appear in >40% of parlays.
# Prevents a single "value" leg from dominating the entire output.
MAX_LEG_CONCENTRATION = 0.40

# Maximum combined odds for "realistic" parlays in top picks.
# 20x+ parlays are lottery tickets, not recommendations.
MAX_TOP_PICK_ODDS = 15.0

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
# Includes logical implication chains: if leg A winning guarantees leg B wins,
# they should never appear together (the parlay is just paying extra vig for
# a correlated outcome, not adding real diversification).
CONFLICTING_PAIRS = [
    # --- 1X2 mutual exclusion ---
    (("h2h", "HOME"), ("h2h", "AWAY")),
    (("h2h", "HOME"), ("h2h", "DRAW")),
    (("h2h", "AWAY"), ("h2h", "DRAW")),
    # --- 1X2 ↔ DC implication ---
    (("h2h", "HOME"), ("double_chance", "X2")),   # H win → X2 loses
    (("h2h", "AWAY"), ("double_chance", "1X")),   # A win → 1X loses
    (("h2h", "HOME"), ("double_chance", "1X")),   # H win IMPLIES 1X wins (redundant)
    (("h2h", "AWAY"), ("double_chance", "X2")),   # A win IMPLIES X2 wins (redundant)
    # --- DNB mutual exclusion + implication ---
    (("draw_no_bet", "HOME"), ("draw_no_bet", "AWAY")),
    (("draw_no_bet", "HOME"), ("h2h", "AWAY")),
    (("draw_no_bet", "AWAY"), ("h2h", "HOME")),
    (("draw_no_bet", "HOME"), ("h2h", "HOME")),   # H win IMPLIES DNB H (redundant)
    (("draw_no_bet", "AWAY"), ("h2h", "AWAY")),   # A win IMPLIES DNB A (redundant)
    # --- DC ↔ DNB implication chains ---
    (("double_chance", "1X"), ("draw_no_bet", "HOME")),  # DNB H ⊂ DC 1X (redundant)
    (("double_chance", "X2"), ("draw_no_bet", "AWAY")),  # DNB A ⊂ DC X2 (redundant)
    (("double_chance", "1X"), ("draw_no_bet", "AWAY")),  # Contradictory only on A win
    (("double_chance", "X2"), ("draw_no_bet", "HOME")),  # Contradictory only on H win
    # --- DC ↔ DC on same match (redundant overlap) ---
    (("double_chance", "1X"), ("double_chance", "X2")),  # Overlap on draw
    (("double_chance", "1X"), ("double_chance", "12")),  # Overlap on home
    (("double_chance", "X2"), ("double_chance", "12")),  # Overlap on away
    # --- Spreads ↔ result implication ---
    # Note: handled dynamically in _legs_conflict() because the direction
    # of the spread line matters (negative = implies win, positive = doesn't).
    # --- Totals ↔ BTTS ---
    (("totals", "UNDER"), ("btts", "YES")),
    (("totals", "UNDER 1.5"), ("btts", "YES")),
    (("totals", "UNDER 2.5"), ("btts", "YES")),
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
        title = conf.title()
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
# Historical Combo Analysis (Step 1)
# ---------------------------------------------------------------------------

# Market normalization: bet_journal uses display names, parlay engine uses internal names
_MARKET_NORMALIZE = {
    "1X2": "h2h", "DC": "double_chance", "DNB": "draw_no_bet",
    "BTTS": "btts", "PARLAY": "parlay",
}


def _normalize_market_name(market: str) -> str:
    """Normalize market name from bet journal to parlay engine format."""
    if market in _MARKET_NORMALIZE:
        return _MARKET_NORMALIZE[market]
    if market.startswith("O/U") or market.startswith("OVER") or market.startswith("UNDER"):
        return "totals"
    if market.startswith("AH"):
        return "spreads"
    return market.lower()


def _analyze_historical_combos() -> dict:
    """Analyze bet journal to compute win rate multipliers per market-pair combo.

    Returns dict of combo-type (e.g. "double_chance+totals") -> multiplier.
    Multiplier > 1.0 means the combo historically outperforms, < 1.0 underperforms.
    """
    journal = _load_json(BETTING_DIR / "bet_journal.json")
    bets = journal.get("bets", {})
    if isinstance(bets, dict):
        bets = list(bets.values())

    # Only settled bets
    settled = [b for b in bets if b.get("status") in ("won", "lost", "push")]
    if len(settled) < 20:
        return {}

    # Overall win rate as baseline
    total_won = sum(1 for b in settled if b.get("status") == "won")
    baseline_wr = total_won / len(settled) if settled else 0.5

    # Group by normalized market
    market_stats = defaultdict(lambda: {"won": 0, "total": 0, "profit": 0})
    for b in settled:
        mkt = _normalize_market_name(b.get("market", ""))
        market_stats[mkt]["total"] += 1
        if b.get("status") == "won":
            market_stats[mkt]["won"] += 1
        market_stats[mkt]["profit"] += b.get("profit", 0)

    # Compute per-market win rate
    market_wr = {}
    for mkt, stats in market_stats.items():
        if stats["total"] >= 5:
            market_wr[mkt] = stats["won"] / stats["total"]

    # Generate pair combo multipliers from market win rates
    # Combo multiplier = geometric mean of both market WRs / baseline WR
    combo_multipliers = {}
    markets = sorted(market_wr.keys())
    for i, m1 in enumerate(markets):
        for m2 in markets[i:]:
            combo_key = f"{m1}+{m2}" if m1 <= m2 else f"{m2}+{m1}"
            geo_mean = math.sqrt(market_wr[m1] * market_wr[m2])
            multiplier = geo_mean / baseline_wr if baseline_wr > 0 else 1.0
            # Clamp to [0.5, 2.0]
            multiplier = max(0.5, min(2.0, multiplier))
            combo_multipliers[combo_key] = round(multiplier, 3)

    return combo_multipliers


# ---------------------------------------------------------------------------
# Staleness Detection (Step 4)
# ---------------------------------------------------------------------------

_INPUT_FILES_FOR_HASH = [
    "upcoming/predictions.json",
    "upcoming/odds_full.json",
    "upcoming/extended_markets.json",
    "upcoming/bookmaker_analysis.json",
    "upcoming/sentiment_analysis.json",
]


def _compute_input_hash() -> str:
    """Hash the content of all parlay input files for staleness detection."""
    h = hashlib.md5()
    for rel_path in _INPUT_FILES_FOR_HASH:
        fpath = DATA_DIR / rel_path
        if fpath.exists():
            h.update(fpath.read_bytes())
    return h.hexdigest()


def _check_staleness() -> tuple:
    """Check if inputs changed since last generation.

    Returns (is_stale, cached_report_or_None).
    is_stale=True means we should regenerate.
    """
    hash_path = BETTING_DIR / ".parlay_input_hash.json"
    report_path = BETTING_DIR / "parlay_report.json"
    current_hash = _compute_input_hash()

    if hash_path.exists() and report_path.exists():
        try:
            stored = json.loads(hash_path.read_text())
            if stored.get("hash") == current_hash:
                cached = json.loads(report_path.read_text())
                cached["regenerated"] = False
                return False, cached
        except (json.JSONDecodeError, OSError):
            pass

    return True, None


def _save_input_hash():
    """Store current input hash after successful generation."""
    hash_path = BETTING_DIR / ".parlay_input_hash.json"
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(json.dumps({
        "hash": _compute_input_hash(),
        "generated_at": datetime.now().isoformat(),
    }))


# ---------------------------------------------------------------------------
# Engine 1: Universal Leg Collector
# ---------------------------------------------------------------------------

def _load_enrichment_data():
    """Load predictions, bookmaker analysis, market intelligence, odds movement.

    Loads from ALL league prediction files (Serie A + Premier League).
    """
    predictions = {}
    for pred_file in ["predictions.json", "predictions_premier_league.json"]:
        preds_raw = _load_json(UPCOMING_DIR / pred_file)
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
            leg["steam_against"] = False
        else:
            leg["steam_aligned"] = False
            leg["steam_against"] = True  # Betting against sharp money
    else:
        leg["steam_aligned"] = False
        leg["steam_against"] = False

    # Form/momentum from factors
    factors = leg.get("factors", [])
    if isinstance(factors, list):
        hot = any("hot" in f for f in factors)
        cold = any("cold" in f for f in factors)
        leg["momentum"] = "hot" if hot and not cold else "cold" if cold and not hot else "neutral"
    else:
        leg["momentum"] = "neutral"

    return leg


def _get_ensemble_1x2(pred: dict) -> tuple:
    """Extract average 1X2 probabilities from ensemble component predictions.

    Returns (prob_H, prob_D, prob_A) or (0, 0, 0) if no components available.
    """
    comp = pred.get("component_predictions", {})
    probs = []
    for method_data in comp.values():
        if isinstance(method_data, dict):
            h = method_data.get("prob_H", 0)
            d = method_data.get("prob_D", 0)
            a = method_data.get("prob_A", 0)
            if h > 0 and d > 0 and a > 0 and abs(h + d + a - 1.0) < 0.05:
                probs.append((h, d, a))
    if not probs:
        return 0, 0, 0
    avg_h = sum(p[0] for p in probs) / len(probs)
    avg_d = sum(p[1] for p in probs) / len(probs)
    avg_a = sum(p[2] for p in probs) / len(probs)
    total = avg_h + avg_d + avg_a
    if total <= 0:
        return 0.4, 0.3, 0.3  # Fallback uninformative prior
    return avg_h / total, avg_d / total, avg_a / total


def _xg_is_placeholder(home_xg: float, away_xg: float) -> bool:
    """Detect if xG values are placeholders (equal xG = likely no real data)."""
    return (home_xg == away_xg) or (home_xg <= 0) or (away_xg <= 0)


def _reverse_xg_from_ensemble(pred: dict) -> tuple:
    """Reverse-engineer plausible xG from ensemble 1X2 probabilities.

    When raw xG is a placeholder (e.g. 1.3 vs 1.3 for Milan-Torino),
    the score matrix produces garbage. Instead, find xG that produce
    Poisson 1X2 probs close to the ensemble's view.

    Uses binary search on home_xg (keeping total xG fixed) to match
    the ensemble's home win probability.
    """
    ens_h, ens_d, ens_a = _get_ensemble_1x2(pred)
    if ens_h <= 0:
        return None, None

    # Total xG from predictions (or reasonable default of 2.5 for Serie A)
    raw_home = pred.get("home_xg", 0)
    raw_away = pred.get("away_xg", 0)
    total_xg = raw_home + raw_away if (raw_home > 0 and raw_away > 0) else 2.5

    # Binary search: find home_xg such that Poisson P(H) ≈ ensemble P(H)
    lo, hi = 0.3, total_xg - 0.3
    for _ in range(20):
        mid = (lo + hi) / 2
        h_xg, a_xg = mid, total_xg - mid
        p_h = sum(
            poisson.pmf(h, h_xg) * sum(poisson.pmf(a, a_xg) for a in range(h))
            for h in range(8)
        )
        if p_h < ens_h:
            lo = mid
        else:
            hi = mid

    home_xg = round((lo + hi) / 2, 2)
    away_xg = round(total_xg - home_xg, 2)
    return home_xg, away_xg


def _get_blended_1x2_probs(pred: dict) -> tuple:
    """Get 1X2 probabilities by blending ensemble components with Poisson.

    Raw Poisson from xG alone can produce wildly wrong results (e.g. 50/50
    for Milan-Torino when xG happens to be equal). The ensemble's component
    predictions incorporate form, home advantage, market data, and ML — they
    are far more reliable for directional probability.

    Blend: 70% ensemble average, 30% Poisson from xG.
    Falls back to Poisson-only if no ensemble components available.
    """
    ens_h, ens_d, ens_a = _get_ensemble_1x2(pred)

    # Poisson from xG — use reverse-engineered xG if raw is a placeholder
    home_xg = pred.get("home_xg", 0)
    away_xg = pred.get("away_xg", 0)

    if _xg_is_placeholder(home_xg, away_xg) and ens_h > 0:
        rev_h, rev_a = _reverse_xg_from_ensemble(pred)
        if rev_h is not None:
            home_xg, away_xg = rev_h, rev_a

    if home_xg > 0 and away_xg > 0:
        poi_h = sum(poisson.pmf(h, home_xg) * sum(poisson.pmf(a, away_xg) for a in range(h)) for h in range(8))
        poi_d = sum(poisson.pmf(g, home_xg) * poisson.pmf(g, away_xg) for g in range(8))
        poi_a = max(0, 1.0 - poi_h - poi_d)
    else:
        poi_h, poi_d, poi_a = 0, 0, 0

    if ens_h > 0:
        if poi_h > 0:
            # Blend: 70% ensemble, 30% Poisson
            p_h = ens_h * 0.7 + poi_h * 0.3
            p_d = ens_d * 0.7 + poi_d * 0.3
            p_a = ens_a * 0.7 + poi_a * 0.3
        else:
            p_h, p_d, p_a = ens_h, ens_d, ens_a
        total = p_h + p_d + p_a
        if total > 0:
            p_h, p_d, p_a = p_h / total, p_d / total, p_a / total
        return p_h, p_d, p_a
    elif poi_h > 0:
        return poi_h, poi_d, poi_a
    else:
        return 0, 0, 0


def _load_extra_market_odds():
    """Load real bookmaker odds from per-event extra markets files (all leagues)."""
    result = {}
    for fname in ["odds_extra_markets.json", "odds_extra_markets_premier_league.json"]:
        raw = _load_json(UPCOMING_DIR / fname)
        matches = raw.get("matches", {})
        result.update(matches)
    return result


def load_all_value_legs():
    """Aggregate legs from all 12+ market types across ALL leagues."""
    legs = []
    predictions, bookmaker, market_intel, odds_movement, goal_preds = _load_enrichment_data()
    extra_odds = _load_extra_market_odds()

    # --- Load goal predictions from ALL leagues ---
    for gp_file in ["goal_predictions_premier_league.json"]:
        gp_path = UPCOMING_DIR / gp_file
        if gp_path.exists():
            _extra_gp = _load_json(gp_path)
            if isinstance(_extra_gp, dict):
                for k, v in _extra_gp.items():
                    if isinstance(v, dict) and k not in goal_preds:
                        goal_preds[k] = v

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

        # Get 1X2 probs: prefer ensemble component average over raw Poisson.
        # Raw Poisson from xG alone ignores home advantage, form, and all non-xG
        # signals, producing garbage like 50/50 for Milan-Torino when ensemble says 70-30.
        p_h, p_d, p_a = _get_blended_1x2_probs(pred)
        if p_h <= 0:
            continue

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

        # Use blended ensemble probs instead of raw Poisson
        p_h, p_d, p_a = _get_blended_1x2_probs(pred)
        if p_h <= 0:
            continue

        # DNB returns stake on draw, so effective prob = win_prob / (1 - draw_prob)
        _dnb_denom = max(0.01, 1.0 - min(p_d, 0.99))
        dnb_p_h = p_h / _dnb_denom
        dnb_p_a = p_a / _dnb_denom
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

    # --- Reality filters ---
    # Remove legs that no sharp bettor would include in a parlay
    filtered = []
    for leg in unique:
        # Odds floor: legs below MIN_PARLAY_LEG_ODDS add vig without selectivity
        if leg.get("odds", 0) < MIN_PARLAY_LEG_ODDS:
            continue
        # Value cap: >80% claimed value on a single leg is a model error
        if leg.get("value_pct", 0) > MAX_LEG_VALUE_PCT:
            continue

        # Market-model divergence shrinkage: when our model probability is
        # >2x the implied market probability, shrink toward the market.
        # Sharp bookmakers (especially Pinnacle) have seen more data than us.
        # If we think 28% and the market says 14%, truth is likely in between.
        odds = leg.get("odds", 0)
        prob = leg.get("probability", 0)
        if odds > 1 and prob > 0:
            implied = 1.0 / odds
            ratio = prob / implied if implied > 0 else 1.0
            if ratio > 1.5:
                # Shrink: new_prob = 60% model + 40% market (with vig removed)
                # Approximate vig removal: implied * 0.95 (typical ~5% overround per leg)
                fair_implied = implied * 0.95
                shrunk_prob = prob * 0.60 + fair_implied * 0.40
                new_val = ((shrunk_prob * odds) - 1) * 100
                leg["probability"] = round(shrunk_prob, 4)
                leg["value_pct"] = round(new_val, 1)
                leg["_shrinkage_applied"] = True
                # Re-check value after shrinkage
                if new_val < MIN_LEG_VALUE_PCT:
                    continue

        filtered.append(leg)

    # Enforce market_rules: remove legs from disabled markets
    # (unless explicitly allowed for parlays via acca_parlay_only_markets)
    try:
        from scripts.betting.betting_unified import BettingConfig
        cfg = BettingConfig()
        _parlay_only = {m.upper() for m in getattr(cfg, "acca_parlay_only_markets", [])}
        _market_map = {
            "h2h": "1X2", "1x2": "1X2",
            "totals": "O/U_Over", "spreads": "AH",
            "double_chance": "DC", "btts": "BTTS",
            "corners": "Corners", "cards": "Cards",
        }
        pre_filter = len(filtered)
        kept = []
        for leg in filtered:
            raw_mkt = (leg.get("market") or "").lower()
            rule_key = _market_map.get(raw_mkt, raw_mkt.upper())

            # Check if this market category is enabled in market_rules
            rule = cfg.market_rules.get(rule_key, {})
            is_enabled = rule.get("enabled", True)
            is_parlay_only = raw_mkt.upper() in _parlay_only

            if is_enabled or is_parlay_only:
                kept.append(leg)
            # else: silently drop — market is disabled for both singles and parlays

        if len(kept) < pre_filter:
            log.info("Market-rules filter: %d -> %d legs (%d disabled-market legs removed)",
                     pre_filter, len(kept), pre_filter - len(kept))
        filtered = kept
    except Exception as e:
        log.debug("Market-rules filter skipped: %s", e)

    # Enrich each leg with quality signals
    for leg in filtered:
        _enrich_leg(leg, predictions, bookmaker, market_intel, odds_movement)

    return filtered


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

    # The copula adjustment reduces the joint probability slightly.
    # adjustment is already in probability units (product of PDF values × rho).
    # Correct formula: multiply naive by (1 - adjustment), not divide.
    adjusted = naive * max(0, 1.0 - adjustment)
    # Bounds: never exceed naive, never reduce more than 20%
    adjusted = min(adjusted, naive)
    adjusted = max(adjusted, naive * 0.80)

    return float(adjusted)


# ---------------------------------------------------------------------------
# Engine 4: Multi-Signal Leg Quality Scoring
# ---------------------------------------------------------------------------

def _compute_leg_quality(leg, sentiment_data=None):
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
    elif leg.get("steam_against"):
        intel = max(0, intel - 30)  # Penalize legs betting against sharp money
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

    # 9. Sentiment & injury adjustment (Step 7)
    sentiment_adj = 0
    if sentiment_data:
        mk = _normalize_match(leg.get("match", ""))
        sent = sentiment_data.get(mk, {})
        if sent:
            sel_upper = leg.get("selection", "").upper()
            # Determine which side we're betting on
            betting_home = any(x in sel_upper for x in ("HOME", "1X", "OVER"))
            betting_away = any(x in sel_upper for x in ("AWAY", "X2"))

            # Injury impact: negative = team hurt by injuries
            if betting_home:
                injury = sent.get("home_injury_impact", 0)
                composite = sent.get("home_composite", 0)
            elif betting_away:
                injury = sent.get("away_injury_impact", 0)
                composite = sent.get("away_composite", 0)
            else:
                injury = 0
                composite = 0

            # Strong injuries against our pick: penalty
            if injury < -40:
                sentiment_adj -= 10
            elif injury < -20:
                sentiment_adj -= 5

            # Strong positive sentiment for our pick: bonus
            if composite > 30:
                sentiment_adj += 5
            elif composite > 15:
                sentiment_adj += 3

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
    quality = max(0, min(100, quality + sentiment_adj))

    leg["quality_score"] = round(quality, 1)
    leg["sentiment_adj"] = sentiment_adj
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

    # --- Dynamic spreads conflict detection ---
    # Negative spread (e.g. "AWAY -1.5") implies the team wins by 2+.
    # This IMPLIES: h2h win, DNB win, DC for that side.
    # This CONTRADICTS: h2h/DNB/DC for the opposite side.
    spread_leg = spread_other = None
    if mkt_a == "spreads":
        spread_leg, spread_other = leg_a, leg_b
    elif mkt_b == "spreads":
        spread_leg, spread_other = leg_b, leg_a

    if spread_leg:
        spread_sel = spread_leg["selection"].upper()
        spread_line = _extract_line(spread_sel)
        other_mkt = spread_other["market"]
        other_sel = spread_other["selection"].upper()

        if spread_line is not None and spread_line < 0:
            # Negative spread: team must win by margin > |line|
            spread_is_home = "HOME" in spread_sel
            # This side is implied to win → redundant with same-side result bets
            same_side = (
                (spread_is_home and ("HOME" in other_sel or "1X" in other_sel or "1" == other_sel)) or
                (not spread_is_home and ("AWAY" in other_sel or "X2" in other_sel or "2" == other_sel))
            )
            # Opposite side contradicts
            opp_side = (
                (spread_is_home and ("AWAY" in other_sel or "X2" in other_sel)) or
                (not spread_is_home and ("HOME" in other_sel or "1X" in other_sel))
            )
            if other_mkt in ("h2h", "draw_no_bet", "double_chance"):
                if same_side or opp_side:
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

            # DC Anchor tagging (Step 2)
            has_dc_anchor = any(l.get("market") == "double_chance" for l in combo)

            combo_id += 1
            n_combos += 1
            combos.append({
                "id": f"PRL-{combo_id:03d}",
                "legs": combo,
                "n_legs": n,
                "dc_anchored": has_dc_anchor,
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

    # --- Leg concentration limiter ---
    # Count how often each leg appears. If a single leg dominates >40% of
    # combos, randomly drop excess combos containing it. This prevents the
    # entire output from being "42 variations of the same Milan DC bet."
    if combos:
        leg_counts = defaultdict(int)
        for c in combos:
            for leg in c["legs"]:
                lk = (_normalize_match(leg["match"]), leg["market"], leg["selection"])
                leg_counts[lk] += 1

        max_appearances = int(len(combos) * MAX_LEG_CONCENTRATION)
        over_represented = {lk for lk, cnt in leg_counts.items() if cnt > max_appearances}

        if over_represented:
            import random as _rng
            _rng.seed(42)  # Deterministic for reproducibility
            # For each over-represented leg, keep only max_appearances combos
            for lk in over_represented:
                containing = [i for i, c in enumerate(combos)
                              if any((_normalize_match(l["match"]), l["market"], l["selection"]) == lk
                                     for l in c["legs"])]
                if len(containing) > max_appearances:
                    to_drop = set(_rng.sample(containing, len(containing) - max_appearances))
                    combos = [c for i, c in enumerate(combos) if i not in to_drop]

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
            home_xg = pred.get("home_xg", 0)
            away_xg = pred.get("away_xg", 0)

        if home_xg <= 0 or away_xg <= 0:
            ext = _load_json(UPCOMING_DIR / "extended_markets.json")
            ext_match = ext.get("matches", {}).get(match_key, {})
            home_xg = ext_match.get("home_xg", 1.3)
            away_xg = ext_match.get("away_xg", 1.0)

        # If xG is a placeholder (equal values), reverse-engineer from ensemble
        # to prevent the score matrix from producing garbage SGP probabilities
        if _xg_is_placeholder(home_xg, away_xg) and pred:
            rev_h, rev_a = _reverse_xg_from_ensemble(pred)
            if rev_h is not None:
                home_xg, away_xg = rev_h, rev_a

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
            prob = np.clip(leg.get("probability", 0.5), 0.001, 0.999)
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
            prob = np.clip(leg.get("probability", 0.5), 0.001, 0.999)
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
            prob = np.clip(leg.get("probability", 0.5), 0.001, 0.999)
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
    """Compute Kelly-based stake for a parlay.

    Parlay Kelly uses much smaller fractions than singles because:
    1. Each leg's probability estimate has error; errors MULTIPLY in parlays
    2. Variance is much higher (you lose the full stake most of the time)
    3. The edge estimate is less reliable (compounded model uncertainty)

    Uses fractional Kelly with aggressive variance discounting:
    - Base fraction: 3% (vs 10% for singles) — standard for props/parlays
    - Probability uncertainty penalty: each leg adds model error
    - Quality-weighted confidence factor
    """
    adj_prob = combo["hit_probability"].get("copula_adjusted", 0)
    combined_odds = float(np.prod([l["odds"] for l in combo["legs"]]))

    b = combined_odds - 1
    if b <= 0 or adj_prob <= 0:
        return 0, combined_odds

    full_kelly = (b * adj_prob - (1 - adj_prob)) / b
    if full_kelly <= 0:
        return 0, combined_odds

    n_legs = combo["n_legs"]

    # --- Parlay-specific Kelly adjustments ---

    # 1. Base fraction: 5% Kelly for parlays (vs 10% for singles).
    # Industry standard for multi-leg bets where edge estimation is noisier.
    parlay_kelly_fraction = 0.05

    # 2. Quality-weighted confidence
    avg_quality = np.mean([l.get("quality_score", 50) for l in combo["legs"]])
    confidence_factor = avg_quality / 100.0

    # 3. Probability uncertainty compounding penalty.
    # Each leg introduces ~5% relative error in probability estimate.
    # For N legs, the joint probability error grows geometrically.
    prob_uncertainty = 0.95 ** n_legs

    # 4. Variance penalty: parlays lose more often, so Kelly must be smaller.
    # 1/sqrt(n) is gentler than 1/n but still meaningful for 3-4 leg parlays.
    variance_penalty = 1.0 / math.sqrt(n_legs)

    stake = (bankroll * full_kelly * parlay_kelly_fraction
             * confidence_factor * prob_uncertainty * variance_penalty)

    # Cap: 2% bankroll for 2-legs, 1% for 3+, 0.5% for 4+
    if n_legs >= 4:
        max_pct = 0.005
    elif n_legs >= 3:
        max_pct = 0.01
    else:
        max_pct = MAX_STAKE_PCT
    max_stake = bankroll * max_pct
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
    """Compute composite parlay quality (0-100).

    Rebuilt to prioritize what actually predicts parlay success:
    1. Individual leg quality (35%) — bad legs = bad parlay, period
    2. Hit probability (25%) — realistic chance of hitting
    3. Minimum leg probability (15%) — weakest link penalty
    4. Sharp alignment (15%) — are bookmakers agreeing?
    5. Expected value (10%) — is the edge real?
    """
    legs = combo.get("legs", [])
    if not legs:
        return 0.0

    # 1. Average leg quality (0-100) — the foundation
    avg_leg_quality = np.mean([l.get("quality_score", 50) for l in legs])

    # 2. Hit probability score — realistic scaling
    # 50% hit = 100, 30% = 60, 10% = 20 (linear, not the old 500x)
    hit_median = combo.get("hit_probability", {}).get("median", 0)
    hit_score = min(hit_median * 200, 100)

    # 3. Weakest link penalty — a parlay is only as good as its worst leg
    min_leg_prob = min(l.get("probability", 0) for l in legs)
    # Contrarian legs (prob < 30%) get heavily penalized
    if min_leg_prob < 0.30:
        min_leg_score = min_leg_prob * 100  # 20% prob → 20 score
    elif min_leg_prob < 0.50:
        min_leg_score = 30 + (min_leg_prob - 0.30) * 200  # 30-70 range
    else:
        min_leg_score = 70 + (min_leg_prob - 0.50) * 60  # 70-100 range

    # 4. Sharp alignment
    sharp_pct = combo.get("sharp_alignment_pct", 50)

    # 5. Expected value (capped sensibly)
    ev = combo.get("expected_roi", 0)
    ev_score = min(max(ev * 200, 0), 100)  # 50% EV → 100

    quality = (
        avg_leg_quality * 0.35 +
        hit_score * 0.25 +
        min_leg_score * 0.15 +
        sharp_pct * 0.15 +
        ev_score * 0.10
    )

    # DC Anchor boost: +5 (reduced from +10 — don't over-reward DC)
    if combo.get("dc_anchored"):
        quality += 5

    return round(min(quality, 100), 1)


def _is_draw_leg(leg: dict) -> bool:
    """Check if a leg is draw-related (1X2 Draw, DC 1X/X2, DNB)."""
    sel = leg.get("selection", "").upper()
    market = leg.get("market", "").lower()
    if market == "h2h" and sel in ("DRAW", "X"):
        return True
    if market == "double_chance" and "X" in sel:
        return True
    return False


def _draw_quality_boost(leg: dict, predictions_data: dict = None) -> float:
    """Compute quality boost for draw-related legs based on draw indicators.

    Returns bonus points (0-25) to add to the leg's quality score.
    """
    match_key = _normalize_match(leg.get("match", ""))
    bonus = 0

    # Check draw probability from predictions
    if predictions_data and match_key in predictions_data:
        pred = predictions_data[match_key]
        probs = pred.get("probabilities", pred.get("betting_probabilities", {}))
        draw_prob = probs.get("draw", 0) if isinstance(probs, dict) else 0

        # High draw probability bonus
        if draw_prob >= 0.30:
            bonus += 10
        elif draw_prob >= 0.25:
            bonus += 5

        # Draw analysis from ensemble
        da = pred.get("draw_analysis", {})
        if da.get("is_draw_candidate"):
            bonus += 8

        # Evenly matched teams (close Elo)
        comp = pred.get("component_predictions", {})
        # Check if multiple methods agree on draw
        draw_agreement = 0
        for method_name, method_pred in comp.items():
            if isinstance(method_pred, dict):
                mp_d = method_pred.get("prob_D", 0)
                if mp_d >= 0.28:
                    draw_agreement += 1
        if draw_agreement >= 3:
            bonus += 7  # 3+ methods agree on draw

    # Market-specific bonus
    sel = leg.get("selection", "").upper()
    if sel in ("DRAW", "X"):
        # Pure draw picks get extra bonus if odds are in sweet spot (2.8-3.8)
        odds = leg.get("odds", 0)
        if 2.8 <= odds <= 3.8:
            bonus += 5  # Sweet spot for draw parlays

    return min(bonus, 25)  # Cap at 25


def generate_draw_parlays(legs: list, predictions_data: dict = None,
                          bankroll: float = 1000) -> list:
    """Generate draw-focused parlay combinations.

    Strategy: Combine 2-3 high-confidence draw picks into parlays.
    Draw at 3.0-3.5 odds × 2-3 legs = 9-42x combined odds.
    Model F1 Draw = 0.328 means draws are now detectable — exploit this.

    Returns list of parlay combo dicts ready for categorization.
    """
    # Filter to draw-related legs
    draw_legs = [l for l in legs if _is_draw_leg(l)]
    if len(draw_legs) < 2:
        return []

    # Apply draw quality boost
    for leg in draw_legs:
        boost = _draw_quality_boost(leg, predictions_data)
        leg["_draw_quality_boost"] = boost
        # Temporarily boost quality for draw combo selection
        leg["_orig_quality"] = leg.get("quality_score", 50)
        leg["quality_score"] = leg.get("quality_score", 50) + boost

    # Sort by boosted quality
    draw_legs.sort(key=lambda l: l.get("quality_score", 0), reverse=True)

    # Take top 6 draw legs (enough for interesting combos)
    top_draw = draw_legs[:6]

    combos = []

    # Generate 2-leg and 3-leg draw parlays
    for n_legs in [2, 3]:
        if len(top_draw) < n_legs:
            continue
        for leg_combo in combinations(top_draw, n_legs):
            # Check no same-match conflicts
            matches = [_normalize_match(l["match"]) for l in leg_combo]
            if len(set(matches)) < n_legs:
                continue  # Same match twice — skip

            combined_odds = float(np.prod([l["odds"] for l in leg_combo]))
            combined_prob = float(np.prod([l["probability"] for l in leg_combo]))

            # Draw-specific correlation: draws are slightly positively correlated
            # within the same league round (if one match is 0-0 at HT, nervous
            # play spreads to other matches). Apply small boost: 1.05x per leg pair.
            n_pairs = n_legs * (n_legs - 1) // 2
            corr_adj = 1.0 + 0.02 * n_pairs
            adj_prob = combined_prob * corr_adj

            ev = adj_prob * combined_odds - 1
            if ev < 0.03:  # Need 3% EV minimum
                continue

            avg_quality = np.mean([l.get("quality_score", 50) for l in leg_combo])
            avg_draw_boost = np.mean([l.get("_draw_quality_boost", 0) for l in leg_combo])

            # Generate unique ID
            leg_ids = "_".join(sorted(
                f"{_normalize_match(l['match'])}_{l['selection']}" for l in leg_combo
            ))
            combo_id = hashlib.md5(f"draw_{leg_ids}".encode()).hexdigest()[:12]

            combo = {
                "id": f"draw_{combo_id}",
                "legs": [dict(l) for l in leg_combo],  # Copy to avoid mutation
                "n_legs": n_legs,
                "combined_odds": round(combined_odds, 2),
                "hit_probability": {
                    "naive": round(combined_prob, 4),
                    "copula_adjusted": round(adj_prob, 4),
                    "median": round(adj_prob, 4),
                },
                "value_pct": round(ev * 100, 1),
                "expected_roi": round(ev, 4),
                "is_same_game": False,
                "is_draw_parlay": True,
                "avg_draw_quality_boost": round(avg_draw_boost, 1),
                "parlay_quality": round(min(avg_quality, 100), 1),
                "sharp_alignment_pct": 50,
                "correlation_warning": None,
                "diversification": 1.0,
            }

            # Kelly sizing (more conservative for draw parlays: 3% Kelly)
            kelly_frac = 0.03
            if combined_odds <= 1.01 or adj_prob <= 0:
                combo["stake"] = 0
                draw_combos.append(combo)
                continue
            full_kelly = (adj_prob * combined_odds - 1) / (combined_odds - 1)
            if full_kelly > 0:
                stake = bankroll * full_kelly * kelly_frac
                stake = max(2.0, min(stake, bankroll * 0.015))  # 0.2% - 1.5% of bankroll
            else:
                stake = 0
            combo["stake"] = round(stake, 2)

            if stake > 0:
                combos.append(combo)

    # Restore original quality scores
    for leg in draw_legs:
        leg["quality_score"] = leg.pop("_orig_quality", leg.get("quality_score", 50))

    return combos


def categorize_parlays(combos):
    """Sort parlays into 7 categories (6 standard + draw_specials)."""
    categories = {
        "draw_specials": [],
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
        if not p.get("legs"):
            continue

        n = p["n_legs"]
        hit_med = p.get("hit_probability", {}).get("median", 0)
        avg_q = np.mean([l.get("quality_score", 50) for l in p["legs"]])
        combined_odds = p.get("combined_odds", 1)
        sharp_pct = p.get("sharp_alignment_pct", 0)
        is_sgp = p.get("is_same_game", False)
        vp = p.get("value_pct", 0)

        assigned = False

        # Draw specials — parlays where all legs are draw-related
        if p.get("is_draw_parlay"):
            categories["draw_specials"].append(p)
            p["category"] = "draw_specials"
            assigned = True

        # Same-game parlays
        if not assigned and is_sgp:
            categories["same_game"].append(p)
            p["category"] = "same_game"
            assigned = True

        # Banker combos — ALL legs must be high probability. A "safe" parlay
        # with one contrarian leg isn't safe at all.
        max_prob = max(l["probability"] for l in p["legs"])
        min_prob = min(l["probability"] for l in p["legs"])
        avg_prob = np.mean([l["probability"] for l in p["legs"]])
        if not assigned and (
            min_prob >= 0.55 and avg_prob >= 0.65 and n <= 4  # ALL legs must be 55%+
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
        "draw_specials": lambda x: x.get("parlay_quality", 0),
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
# Exposure Coordination (Step 6)
# ---------------------------------------------------------------------------

def _load_todays_singles() -> set:
    """Load today's placed singles as (match, market, selection) tuples."""
    placed = _load_json(BETTING_DIR / "placed_bets.json")
    bets = placed.get("bets", [])
    if isinstance(bets, dict):
        bets = list(bets.values())

    today = datetime.now().strftime("%Y-%m-%d")
    tuples = set()
    for b in bets:
        if b.get("date", "") >= today:
            match = _normalize_match(b.get("match", ""))
            market = b.get("market", "").lower()
            selection = b.get("selection", "").upper()
            tuples.add((match, market, selection))
    return tuples


def _apply_exposure_penalties(combos, active_singles):
    """Apply quality penalty and stake reduction for overlapping exposure."""
    if not active_singles:
        return

    for combo in combos:
        overlap_count = 0
        for leg in combo["legs"]:
            mk = _normalize_match(leg.get("match", ""))
            mkt = leg.get("market", "").lower()
            sel = leg.get("selection", "").upper()
            if (mk, mkt, sel) in active_singles:
                overlap_count += 1

        if overlap_count > 0:
            # Penalize quality and halve stake
            combo["parlay_quality"] = max(0, combo.get("parlay_quality", 0) - 15 * overlap_count)
            combo["stake"] = round(combo.get("stake", 0) * 0.5, 2)
            combo["exposure_overlap"] = overlap_count


# ---------------------------------------------------------------------------
# Top Parlay Selection (Step 3)
# ---------------------------------------------------------------------------

def select_top_parlays(categories, combo_multipliers=None, n=3):
    """Select the top N parlays across all categories with diversity constraint.

    Returns list of dicts with full parlay data + human-readable explanation.
    """
    if combo_multipliers is None:
        combo_multipliers = {}

    # Collect all parlays, filtering out lottery tickets from top picks
    all_parlays = []
    for cat_key, items in categories.items():
        for p in items:
            # No lottery tickets in top picks — max 15x combined odds
            if p.get("combined_odds", 0) > MAX_TOP_PICK_ODDS:
                continue
            all_parlays.append((cat_key, p))

    if not all_parlays:
        return []

    # Score each parlay for top pick selection
    scored = []
    for cat_key, p in all_parlays:
        base_quality = p.get("parlay_quality", 0)
        legs = p.get("legs", [])

        # Combo multiplier from historical analysis (capped to avoid over-weighting)
        markets = sorted(set(l.get("market", "") for l in legs))
        combo_key = "+".join(markets)
        combo_mult = combo_multipliers.get(combo_key, 1.0)
        if len(markets) >= 2:
            pair_mults = []
            for i, m1 in enumerate(markets):
                for m2 in markets[i + 1:]:
                    pk = f"{m1}+{m2}" if m1 <= m2 else f"{m2}+{m1}"
                    pair_mults.append(combo_multipliers.get(pk, 1.0))
            if pair_mults:
                combo_mult = max(combo_mult, sum(pair_mults) / len(pair_mults))
        combo_mult = min(combo_mult, 1.3)  # Cap at 1.3x — don't let history dominate

        # Penalty for contrarian legs (picking against >70% favorites)
        contrarian_penalty = 1.0
        if legs:
            min_prob = min(l.get("probability", 0.5) for l in legs)
            if min_prob < 0.25:
                contrarian_penalty = 0.6  # Harsh — picking against 75%+ favorites
            elif min_prob < 0.35:
                contrarian_penalty = 0.8  # Moderate — risky picks

        rec_score = base_quality * combo_mult * contrarian_penalty
        scored.append((rec_score, cat_key, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top N with diversity: max 1 per category in top N
    selected = []
    used_categories = set()
    for rec_score, cat_key, p in scored:
        if len(selected) >= n:
            break
        if cat_key in used_categories:
            continue
        used_categories.add(cat_key)

        # Generate "why" explanation
        why_parts = []
        legs = p.get("legs", [])

        # Signal alignment
        sharp_count = sum(1 for l in legs if l.get("sharp_aligned") is True)
        if sharp_count > 0:
            why_parts.append(f"{sharp_count}/{len(legs)} legs sharp-aligned")

        # DC anchor — compute actual WR from journal instead of hardcoding
        dc_legs = [l for l in legs if l.get("market") == "double_chance"]
        if dc_legs:
            try:
                from scripts.betting.bet_journal import _load_journal
                j = _load_journal()
                dc_bets = [b for b in j["bets"].values()
                           if b.get("status") in ("won", "lost", "push")
                           and "DC" in (b.get("market") or "").upper()]
                if len(dc_bets) >= 5:
                    dc_wr = sum(1 for b in dc_bets if b["status"] == "won") / len(dc_bets)
                    why_parts.append(f"DC anchor ({dc_wr:.0%} live WR on {len(dc_bets)} bets)")
                else:
                    why_parts.append("Double Chance anchor")
            except Exception:
                why_parts.append("Double Chance anchor")

        # High-prob legs — clarify this is MODEL probability, not implied
        high_prob = [l for l in legs if l.get("probability", 0) >= 0.70]
        if high_prob:
            why_parts.append(f"{len(high_prob)} high-probability legs (over 70% each)")

        # Form/momentum
        hot_legs = [l for l in legs if l.get("momentum") == "hot"]
        if hot_legs:
            why_parts.append(f"{len(hot_legs)} legs on hot form")

        # Historical combo performance
        if combo_mult > 1.1:
            why_parts.append(f"combo type historically strong ({combo_mult:.2f}x)")
        elif combo_mult < 0.9:
            why_parts.append(f"combo type historically weak ({combo_mult:.2f}x)")

        # Risk factors
        risks = []
        if p.get("exposure_overlap"):
            risks.append(f"overlaps {p['exposure_overlap']} active single(s)")
        cold_legs = [l for l in legs if l.get("momentum") == "cold"]
        if cold_legs:
            risks.append(f"{len(cold_legs)} legs on cold form")
        low_q = [l for l in legs if l.get("quality_score", 100) < 40]
        if low_q:
            risks.append(f"{len(low_q)} low-quality legs")

        selected.append({
            "rank": len(selected) + 1,
            "recommendation_score": round(rec_score, 1),
            "category": cat_key,
            "parlay": p,
            "why": why_parts or ["solid overall quality score"],
            "risks": risks or ["no significant risks identified"],
        })

    # If we didn't fill N due to diversity constraint, relax it
    if len(selected) < n:
        for rec_score, cat_key, p in scored:
            if len(selected) >= n:
                break
            if any(s["parlay"]["id"] == p["id"] for s in selected):
                continue
            selected.append({
                "rank": len(selected) + 1,
                "recommendation_score": round(rec_score, 1),
                "category": cat_key,
                "parlay": p,
                "why": ["additional high-quality pick"],
                "risks": [],
            })

    return selected


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
        "dc_anchored": combo.get("dc_anchored", False),
        "sharp_alignment_pct": sharp_pct,
        "category": combo.get("category", ""),
        "exposure_overlap": combo.get("exposure_overlap", 0),
    }


def generate_parlay_report(bankroll=None):
    """Main entry point — full 7-engine parlay generation pipeline."""
    if bankroll is None:
        bankroll = _get_bankroll()

    # Step 4: Staleness detection
    is_stale, cached = _check_staleness()
    if not is_stale and cached:
        print("  [Staleness] Input data unchanged — returning cached report")
        return cached

    # Step 1: Historical combo analysis
    print("  [Historical] Analyzing bet journal for combo multipliers...")
    combo_multipliers = _analyze_historical_combos()
    if combo_multipliers:
        top_combos = sorted(combo_multipliers.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Found {len(combo_multipliers)} combo types, strongest: {', '.join(f'{k}={v}' for k,v in top_combos)}")

    print("  [Engine 1] Loading value legs from all markets...")
    legs = load_all_value_legs()
    markets_found = set(l["market"] for l in legs)
    print(f"  Found {len(legs)} legs across {len(markets_found)} markets: {', '.join(sorted(markets_found))}")

    empty_cats = {k: [] for k in [
        "draw_specials", "safe_doubles", "value_trebles", "long_shots",
        "same_game", "sharp_specials", "banker_combos"
    ]}

    if len(legs) < 2:
        print("  Not enough value legs to generate parlays")
        report = {
            "generated_at": datetime.now().isoformat(),
            "regenerated": True,
            "total_parlays": 0,
            "total_legs_available": len(legs),
            "legs_by_market": {},
            "bankroll": bankroll,
            "model_info": _model_info(),
            "categories": empty_cats,
            "top_picks": [],
        }
        _save_report(report)
        _save_input_hash()
        return report

    # Step 7: Load sentiment data for quality scoring
    sentiment_data = {}
    sent_raw = _load_json(UPCOMING_DIR / "sentiment_analysis.json")
    sent_matches = sent_raw.get("matches", [])
    if isinstance(sent_matches, list):
        for s in sent_matches:
            mk = _normalize_match(s.get("match", ""))
            if mk:
                sentiment_data[mk] = s

    # Engine 4: Score all legs (with sentiment integration)
    print("  [Engine 4] Computing leg quality scores (with sentiment)...")
    for leg in legs:
        _compute_leg_quality(leg, sentiment_data=sentiment_data)
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
            "regenerated": True,
            "total_parlays": 0,
            "total_legs_available": len(legs),
            "legs_by_market": dict(legs_by_market),
            "bankroll": bankroll,
            "model_info": _model_info(),
            "categories": empty_cats,
            "top_picks": [],
        }
        _save_report(report)
        _save_input_hash()
        return report

    # Engine 2 & 3: Compute hit probabilities
    print("  [Engine 2/3] Computing hit probabilities (Poisson + Copula)...")
    predictions_data = {}
    preds_raw = _load_json(UPCOMING_DIR / "predictions.json")
    for p in preds_raw.get("predictions", []):
        predictions_data[_normalize_match(p.get("match", ""))] = p

    for combo in combos:
        _compute_hit_probability(combo, predictions_data)

    # Engine 5b: Draw-focused parlays (leverages F1_D=0.328 model capability)
    print("  [Engine 5b] Generating draw-focused parlays...")
    draw_combos = generate_draw_parlays(legs, predictions_data, bankroll)
    if draw_combos:
        combos.extend(draw_combos)
        valued_draw = [c for c in draw_combos if c.get("stake", 0) > 0]
        print(f"  Added {len(valued_draw)} draw-focused parlays ({len(draw_combos)} candidates)")
    else:
        print("  No draw-focused parlays generated (need 2+ draw legs)")

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

        # Parlay EV reality gate: compound model error makes extreme values
        # unreliable. Cap per-parlay value based on number of legs.
        # 2-leg: max 150% value, 3-leg: max 200%, 4-leg: max 300%
        # Anything beyond this is almost certainly model overconfidence.
        max_parlay_value = {2: 150, 3: 200, 4: 300}.get(combo["n_legs"], 400)
        if value > max_parlay_value:
            combo["value_pct"] = max_parlay_value
            combo["_value_capped"] = True

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

    # Step 6: Exposure coordination
    print("  [Exposure] Checking overlap with active singles...")
    active_singles = _load_todays_singles()
    if active_singles:
        _apply_exposure_penalties(valued_combos, active_singles)
        overlap_count = sum(1 for c in valued_combos if c.get("exposure_overlap", 0) > 0)
        print(f"  {overlap_count} parlays penalized for exposure overlap")

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

    # Step 3: Select top 3 picks
    print("  [Top Picks] Selecting best parlays across categories...")
    top_picks = select_top_parlays(formatted_categories, combo_multipliers, n=3)
    if top_picks:
        for pick in top_picks:
            p = pick["parlay"]
            cat_label = pick["category"].replace("_", " ").title()
            print(f"    #{pick['rank']} {cat_label} (score={pick['recommendation_score']}) "
                  f"— {p.get('combined_odds', 0):.2f}x, {len(p.get('legs', []))} legs")

    report = {
        "generated_at": datetime.now().isoformat(),
        "regenerated": True,
        "total_parlays": total,
        "total_legs_available": len(legs),
        "legs_by_market": dict(legs_by_market),
        "bankroll": bankroll,
        "model_info": _model_info(),
        "categories": formatted_categories,
        "top_picks": top_picks,
        "combo_multipliers": combo_multipliers,
    }

    _save_report(report)
    _save_input_hash()

    # Step 8: Save to parlay history for tracking
    try:
        from scripts.betting.parlay_tracker import record_parlays
        record_parlays(top_picks)
    except Exception:
        pass

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
