#!/usr/bin/env python3
"""M2 gate test: did rich-features EPL retrain produce better holdout ROI?

After M1 produces an updated player_stats_epl.parquet with the rich FBref
schema (~141 cols vs the original 30), this script:

  1. Rebuilds EPL features (cache-bust the affected plugins).
  2. Trains a fresh walkforward fold on EPL with the new feature set.
  3. Re-runs the subset analysis to measure 2025-26 holdout ROI.
  4. Compares to the prior baseline -22.6% (rich-features-MISSING) result.

Decision gate: if new holdout ROI > -10%, M3 (multi-market) and M4 (re-enable)
are unlocked. If still -15%+, EPL is structurally hard to bet — report and
stop the chain.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ACCEPTANCE_HOLDOUT_ROI_PCT = -10.0  # better than -10% = proceed; worse = stop
PRIOR_BASELINE_HOLDOUT_ROI = -22.64  # what we got with sparse features


def step_1_invalidate_cache():
    """Cache-bust the EPL feature caches that depend on player_stats_epl.parquet.

    The data_inputs annotations (F5 / G3) should auto-invalidate, but we'll
    nuke the relevant ones to be safe.
    """
    cache_dir = PROJECT_ROOT / "data" / "cache" / "features" / "premier_league"
    if not cache_dir.exists():
        log.warning("No EPL cache dir at %s — assuming fresh build", cache_dir)
        return
    targets = [
        "advanced_player_v1.1.parquet",
        "advanced_player_v1.1.fingerprint",
        "advanced_player_v1.0.parquet",
        "advanced_player_v1.0.fingerprint",
        "advanced_shots_v1.0.parquet",
        "advanced_shots_v1.0.fingerprint",
    ]
    for t in targets:
        p = cache_dir / t
        if p.exists():
            p.unlink()
            log.info("Removed cache: %s", p.name)


def step_2_rebuild_features():
    """Run cli.py features to rebuild EPL parquet from updated player_stats_epl."""
    log.info("Rebuilding features (this regenerates SA + EPL)...")
    result = subprocess.run(
        ["python3", "cli.py", "features"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour cap
    )
    if result.returncode != 0:
        log.error("Feature rebuild failed:\n%s", result.stderr[-2000:])
        return False
    log.info("Feature rebuild complete")
    return True


def step_3_check_feature_count():
    """Verify EPL features now has more columns (was 909, target ~1300+)."""
    import pandas as pd
    epl = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "features_premier_league.parquet")
    log.info("EPL features now: %d rows × %d cols (was 909 cols pre-M1)", len(epl), len(epl.columns))
    adv_cols = [c for c in epl.columns if "adv_roll5" in c]
    log.info("  EPL adv_roll5 cols: %d (was 0 pre-M1)", len(adv_cols))
    return len(epl.columns), len(adv_cols)


def step_4_train_fresh_epl_2025():
    """Train walkforward fold for EPL 2025-26 with new feature set."""
    log.info("Training fresh EPL 2025-26 fold with rich features...")
    env = os.environ.copy()
    env["TRAIN_SEED"] = "42"
    result = subprocess.run(
        [
            "python3", "-m", "scripts.models.train_walkforward",
            "--league", "premier_league",
            "--market", "1x2",
            "--eval-seasons", "2025-2026",
            "--output-suffix", "rich_features_2025_seed42",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min
    )
    if result.returncode != 0:
        log.error("Training failed:\n%s", result.stderr[-2000:])
        return False
    log.info("Training complete")
    log.info(result.stdout[-1000:])
    return True


def step_5_compute_holdout_roi():
    """Score 2025-26 EPL with rich-features model and compute paper-trade ROI.

    Reuses the existing subset_alpha_fresh_epl.py logic but points at the
    rich-features model instead of fresh_2025_seed42.
    """
    import pandas as pd
    import numpy as np

    log.info("Computing holdout ROI on 2025-26 with rich-features model...")
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "parsed" / "matches.parquet")
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    matches = matches[matches["league"] == "premier_league"].copy()
    matches = matches.dropna(subset=["home_score", "away_score"]).copy()
    matches = matches[matches["season"] == "2025-2026"].copy()
    matches = matches[matches["odds_MaxH"].notna()].copy()
    log.info("EPL 2025-26 matches with odds: %d", len(matches))

    feats = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "features_premier_league.parquet")
    feats["match_date"] = pd.to_datetime(feats["match_date"])
    feats = feats[feats["season"] == "2025-2026"].copy()

    from catboost import CatBoostClassifier
    import pickle as pk
    from ml.feature_selection import exclude_odds
    from scripts.models.train_walkforward import LEAKY_COLUMNS, META_COLUMNS

    model_dir = PROJECT_ROOT / "data" / "models" / "walkforward" / "premier_league" / "1x2__rich_features_2025_seed42"
    model = CatBoostClassifier()
    model.load_model(str(model_dir / "season_2025-2026.cbm"))
    with open(model_dir / "season_2025-2026_calibrators.pkl", "rb") as f:
        cals = pk.load(f)

    # Re-derive feature set
    numeric = feats.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [
        c for c in numeric
        if c not in META_COLUMNS and c not in LEAKY_COLUMNS and not c.startswith("_")
    ]
    feat_cols = [f for f in feat_cols if not feats[feat_cols].isna().all(axis=0).get(f, False)]
    _, feat_cols = exclude_odds(feats[feat_cols].copy(), feat_cols)

    # Align to model's feature names
    model_features = list(model.feature_names_)
    feat_cols = [f for f in model_features if f in feats.columns]
    log.info("Aligned features: %d", len(feat_cols))

    X = feats[feat_cols]
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    cal_proba = np.zeros_like(proba)
    for i, cls in enumerate(classes):
        if str(cls) in cals:
            cal_proba[:, i] = cals[str(cls)].transform(proba[:, i])
        elif cls in cals:
            cal_proba[:, i] = cals[cls].transform(proba[:, i])
        else:
            cal_proba[:, i] = proba[:, i]
    rs = cal_proba.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
    cal_proba /= rs

    # Map class indices to H/D/A
    meta_path = model_dir / "season_2025-2026_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        spec_classes = meta.get("classes", ["A", "D", "H"])
    else:
        spec_classes = ["A", "D", "H"]
    idx_a = spec_classes.index("A") if "A" in spec_classes else 0
    idx_d = spec_classes.index("D") if "D" in spec_classes else 1
    idx_h = spec_classes.index("H") if "H" in spec_classes else 2
    feats = feats.reset_index(drop=True)
    feats["prob_H"] = cal_proba[:, idx_h]
    feats["prob_D"] = cal_proba[:, idx_d]
    feats["prob_A"] = cal_proba[:, idx_a]

    # Baseline paper-trade
    EDGE_THRESHOLD = 0.05
    KELLY_FRACTION = 0.25
    MAX_STAKE = 0.02
    INITIAL = 1000.0

    bets = []
    for _, row in matches.iterrows():
        prob_row = feats[
            (feats["home_team"] == row["home_team"])
            & (feats["away_team"] == row["away_team"])
            & (feats["match_date"] == row["match_date"])
        ]
        if prob_row.empty:
            continue
        probs = {
            "H": float(prob_row.iloc[0]["prob_H"]),
            "D": float(prob_row.iloc[0]["prob_D"]),
            "A": float(prob_row.iloc[0]["prob_A"]),
        }
        odds = {
            "H": float(row["odds_MaxH"]) if pd.notna(row["odds_MaxH"]) else 0.0,
            "D": float(row["odds_MaxD"]) if pd.notna(row["odds_MaxD"]) else 0.0,
            "A": float(row["odds_MaxA"]) if pd.notna(row["odds_MaxA"]) else 0.0,
        }
        best = None
        for outcome in ["H", "D", "A"]:
            o = odds[outcome]
            if o <= 1.01:
                continue
            edge = probs[outcome] - (1.0 / o)
            if edge < EDGE_THRESHOLD:
                continue
            if best is None or edge > best[3]:
                best = (outcome, probs[outcome], o, edge)
        if best is None:
            continue
        outcome, p, op_odds, edge = best
        b = op_odds - 1.0
        full_kelly = (b * p - (1 - p)) / b
        if full_kelly <= 0:
            continue
        stake_frac = min(full_kelly * KELLY_FRACTION, MAX_STAKE)
        stake = stake_frac * INITIAL
        if stake < 0.01:
            continue
        h_score, a_score = int(row["home_score"]), int(row["away_score"])
        actual = "H" if h_score > a_score else "A" if h_score < a_score else "D"
        won = (outcome == actual)
        pnl = stake * b if won else -stake
        bets.append({"outcome": outcome, "actual": actual, "won": won, "stake": stake, "pnl": pnl})

    if not bets:
        log.warning("No bets generated")
        return None
    n = len(bets)
    wins = sum(1 for b in bets if b["won"])
    total_stake = sum(b["stake"] for b in bets)
    total_pnl = sum(b["pnl"] for b in bets)
    roi_pct = (total_pnl / total_stake) * 100 if total_stake > 0 else 0
    return {
        "n_bets": n,
        "win_rate": wins / n,
        "total_staked": total_stake,
        "total_pnl": total_pnl,
        "roi_pct": roi_pct,
    }


def main() -> int:
    log.info("=" * 70)
    log.info("M2 GATE TEST — EPL Rich Features Retrain + Retest")
    log.info("=" * 70)

    log.info("\nStep 1: Cache-busting EPL feature caches...")
    step_1_invalidate_cache()

    log.info("\nStep 2: Rebuilding features...")
    if not step_2_rebuild_features():
        log.error("Step 2 failed; aborting.")
        return 1

    log.info("\nStep 3: Verifying feature count...")
    n_cols, n_adv = step_3_check_feature_count()
    if n_cols < 1000:
        log.warning("EPL features still only %d cols — rich features may not have flowed through", n_cols)
        log.warning("Continuing anyway, but expect modest improvement only.")

    log.info("\nStep 4: Training fresh EPL 2025-26 fold with rich features...")
    if not step_4_train_fresh_epl_2025():
        log.error("Step 4 failed; aborting.")
        return 1

    log.info("\nStep 5: Computing holdout ROI...")
    result = step_5_compute_holdout_roi()
    if result is None:
        log.error("Step 5 failed.")
        return 1

    print()
    print("=" * 70)
    print("M2 GATE RESULTS")
    print("=" * 70)
    print(f"  Bets placed:      {result['n_bets']}")
    print(f"  Win rate:         {result['win_rate']*100:.1f}%")
    print(f"  Total staked:     €{result['total_staked']:.2f}")
    print(f"  Total P/L:        €{result['total_pnl']:+.2f}")
    print(f"  Flat ROI:         {result['roi_pct']:+.2f}%  (acceptance gate: > {ACCEPTANCE_HOLDOUT_ROI_PCT:.0f}%)")
    print(f"  Prior baseline:   {PRIOR_BASELINE_HOLDOUT_ROI:.2f}% (sparse features)")
    print(f"  Improvement:      {result['roi_pct'] - PRIOR_BASELINE_HOLDOUT_ROI:+.2f}pp")
    print()

    if result["roi_pct"] > ACCEPTANCE_HOLDOUT_ROI_PCT:
        print(">>> GATE PASSED. M3 (multi-market) and M4 (re-enable) unlocked.")
        return 0
    else:
        print(">>> GATE FAILED. EPL still structurally negative. M3/M4 ABORTED.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
