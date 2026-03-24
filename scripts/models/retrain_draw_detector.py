#!/usr/bin/env python3
"""Retrain + ablation for draw_detector.cbm.

Retrains the binary draw detector on fresh data, evaluates it against
the current production ensemble, and decides whether to enable blending.

Steps:
  1. Load features + recent seasons for walk-forward eval
  2. Train binary draw detector (CatBoost, draw vs non-draw)
  3. Fit isotonic calibrator on validation fold
  4. Test blending vs. no blending over last 3 seasons
  5. If blending improves log_loss by >= 0.001: save + enable
     Otherwise: save model (for monitoring) but keep blending disabled

Usage:
    python scripts/models/retrain_draw_detector.py              # Retrain + ablation
    python scripts/models/retrain_draw_detector.py --dry-run    # Evaluate only
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, MODELS_DIR
from storage.paths import features_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UNIVERSAL_DIR = MODELS_DIR / "universal"
LABEL_MAP = {"H": 0, "D": 1, "A": 2}
# Minimum improvement in log_loss to justify enabling draw blending
MIN_LL_IMPROVEMENT = 0.001


def _load_features_data() -> Tuple[pd.DataFrame, list]:
    """Load features.parquet and return (df, feature_columns)."""
    fp = features_path()
    df = pd.read_parquet(fp)
    log.info("Loaded %d rows from %s", len(df), fp)

    # Get ML feature columns (exclude meta, target, odds for no-odds model)
    from features.build import get_ml_feature_columns
    all_features = get_ml_feature_columns(df)

    # Filter to numeric, non-null
    numeric = df[all_features].select_dtypes(include=[np.number]).columns.tolist()
    usable = [c for c in numeric if df[c].notna().mean() > 0.2]
    log.info("Features: %d total -> %d numeric -> %d usable", len(all_features), len(numeric), len(usable))

    return df, usable


def _inject_draw_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic draw-specific features (must match production inference)."""
    X = X.copy()

    # Elo-based
    elo = X.get("elo_diff", pd.Series(0, index=X.index)).fillna(0).astype(float)
    X["abs_elo_diff"] = elo.abs()
    X["elo_close"] = (elo.abs() < 50).astype(float)

    # Form
    hf = X.get("home_roll_5_points", pd.Series(0, index=X.index)).fillna(0).astype(float)
    af = X.get("away_roll_5_points", pd.Series(0, index=X.index)).fillna(0).astype(float)
    X["form_diff_abs"] = (hf - af).abs()

    hgs = X.get("home_roll_5_goals_scored", pd.Series(1.3, index=X.index)).fillna(1.3).astype(float)
    ags = X.get("away_roll_5_goals_scored", pd.Series(1.1, index=X.index)).fillna(1.1).astype(float)
    hgc = X.get("home_roll_5_goals_conceded", pd.Series(1.1, index=X.index)).fillna(1.1).astype(float)
    agc = X.get("away_roll_5_goals_conceded", pd.Series(1.3, index=X.index)).fillna(1.3).astype(float)
    X["total_goals_form"] = hgs + ags + hgc + agc
    X["goals_form_diff"] = hgs - ags

    # Defense/attack
    hds = X.get("home_defense_strength", pd.Series(1.0, index=X.index)).fillna(1.0).astype(float)
    ads = X.get("away_defense_strength", pd.Series(1.0, index=X.index)).fillna(1.0).astype(float)
    X["defense_parity"] = 1.0 / (1.0 + (hds - ads).abs())

    has_ = X.get("home_attack_strength", pd.Series(1.0, index=X.index)).fillna(1.0).astype(float)
    aas_ = X.get("away_attack_strength", pd.Series(1.0, index=X.index)).fillna(1.0).astype(float)
    X["attack_weakness"] = (1.0 - pd.concat([has_, aas_], axis=1).max(axis=1)).clip(lower=0)

    # Draw tendency
    hdt = X.get("home_draw_tendency_10", pd.Series(0.28, index=X.index)).fillna(0.28).astype(float)
    adt = X.get("away_draw_tendency_10", pd.Series(0.28, index=X.index)).fillna(0.28).astype(float)
    X["combined_draw_tendency"] = (hdt + adt) / 2.0
    X["draw_tendency_product"] = hdt * adt

    # Odds-based
    bh = X.get("odds_B365H", pd.Series(0, index=X.index)).fillna(0).astype(float)
    ba = X.get("odds_B365A", pd.Series(0, index=X.index)).fillna(0).astype(float)
    X["odds_spread"] = np.where((bh > 0) & (ba > 0), (bh - ba).abs(), 0)

    bd = X.get("odds_B365D", pd.Series(0, index=X.index)).fillna(0).astype(float)
    X["market_draw_prob"] = np.where(bd > 0, 1.0 / bd.clip(lower=1.01), 0.28)

    pd_ = X.get("odds_PSD", pd.Series(0, index=X.index)).fillna(0).astype(float)
    X["pin_draw_prob"] = np.where(pd_ > 0, 1.0 / pd_.clip(lower=1.01), 0.28)

    return X


def train_and_evaluate(dry_run: bool = False) -> Dict:
    """Retrain draw detector and ablation-test against current ensemble."""
    df, base_features = _load_features_data()

    # Filter to valid results
    df = df[df["result"].isin(["H", "D", "A"])].copy()
    seasons = sorted(df["season"].unique())
    log.info("Seasons available: %s", seasons)

    # We'll do walk-forward on last 3 seasons
    test_seasons = seasons[-3:]
    log.info("Test seasons: %s", test_seasons)

    all_results = []

    for test_season in test_seasons:
        test_idx = seasons.index(test_season)
        if test_idx < 2:
            continue

        val_season = seasons[test_idx - 1]
        train_seasons = seasons[:test_idx - 1]

        train_m = df["season"].isin(train_seasons)
        val_m = df["season"] == val_season
        test_m = df["season"] == test_season

        # Prepare features with draw-specific additions
        X_all = _inject_draw_features(df[base_features].fillna(0))
        all_feature_cols = [c for c in X_all.columns if X_all[c].dtype in [np.float64, np.int64, float, int]]

        X_tr = X_all.loc[train_m, all_feature_cols]
        X_val = X_all.loc[val_m, all_feature_cols]
        X_te = X_all.loc[test_m, all_feature_cols]

        y_tr = (df.loc[train_m, "result"] == "D").astype(int)
        y_val = (df.loc[val_m, "result"] == "D").astype(int)
        y_te = (df.loc[test_m, "result"] == "D").astype(int)
        y_te_3class = df.loc[test_m, "result"].map(LABEL_MAP).values

        log.info("\n--- %s: train=%d, val=%d, test=%d ---",
                 test_season, len(X_tr), len(X_val), len(X_te))

        # Train binary draw detector
        model = CatBoostClassifier(
            iterations=1500, depth=6, learning_rate=0.02,
            l2_leaf_reg=5, min_data_in_leaf=20,
            random_seed=42, verbose=0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                  early_stopping_rounds=150, verbose=0)

        # Calibrate on validation
        raw_val = model.predict_proba(X_val)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_val, y_val.values)

        # Predict on test
        raw_te = model.predict_proba(X_te)[:, 1]
        cal_te = calibrator.predict(raw_te)

        draw_binary_acc = accuracy_score(y_te, (cal_te > 0.5).astype(int))
        draw_brier = brier_score_loss(y_te, cal_te)
        log.info("  Binary draw: acc=%.1f%%, brier=%.4f", draw_binary_acc * 100, draw_brier)

        # Ablation: train a 3-class CatBoost (simulating the production ensemble),
        # then compare log_loss with and without draw detector blending.
        log.info("  Running ablation (3-class baseline vs. draw-blended)...")

        cls_3class = CatBoostClassifier(
            iterations=2000, depth=6, learning_rate=0.02,
            l2_leaf_reg=3.0, min_data_in_leaf=30,
            random_seed=42, verbose=0,
            loss_function="MultiClass", classes_count=3,
            auto_class_weights="Balanced",
        )
        y_tr_3 = df.loc[train_m, "result"].map(LABEL_MAP)
        y_val_3 = df.loc[val_m, "result"].map(LABEL_MAP)
        cls_3class.fit(X_tr, y_tr_3, eval_set=(X_val, y_val_3),
                       early_stopping_rounds=150, verbose=0)

        probs_no_blend = cls_3class.predict_proba(X_te)
        probs_no_blend = probs_no_blend / probs_no_blend.sum(axis=1, keepdims=True)

        # Blend draw detector into 3-class probs
        alpha = 0.32
        probs_with_blend = probs_no_blend.copy()
        old_d = probs_with_blend[:, 1]
        new_d = (1 - alpha) * old_d + alpha * cal_te
        old_ha = probs_with_blend[:, 0] + probs_with_blend[:, 2]
        new_ha = np.maximum(1.0 - new_d, 0.05)
        ratio_h = np.where(old_ha > 0, probs_with_blend[:, 0] / old_ha, 0.5)
        probs_with_blend = np.column_stack([ratio_h * new_ha, new_d, (1 - ratio_h) * new_ha])

        ll_no = log_loss(y_te_3class, probs_no_blend, labels=[0, 1, 2])
        ll_with = log_loss(y_te_3class, probs_with_blend, labels=[0, 1, 2])
        acc_no = accuracy_score(y_te_3class, probs_no_blend.argmax(axis=1))
        acc_with = accuracy_score(y_te_3class, probs_with_blend.argmax(axis=1))

        improvement = ll_no - ll_with  # positive = blending helps
        log.info("  3-class LL:  no_blend=%.4f  with_blend=%.4f  diff=%+.4f",
                 ll_no, ll_with, -improvement)
        log.info("  3-class Acc: no_blend=%.1f%%  with_blend=%.1f%%",
                 acc_no * 100, acc_with * 100)

        all_results.append({
            "season": test_season,
            "n_matches": len(y_te_3class),
            "ll_no_blend": round(ll_no, 5),
            "ll_with_blend": round(ll_with, 5),
            "ll_improvement": round(improvement, 5),
            "acc_no_blend": round(acc_no, 4),
            "acc_with_blend": round(acc_with, 4),
            "draw_binary_acc": round(draw_binary_acc, 4),
            "draw_brier": round(draw_brier, 4),
        })

    # ------- Summary + Decision -------
    log.info("\n" + "=" * 60)
    log.info("DRAW DETECTOR ABLATION SUMMARY")
    log.info("=" * 60)

    if all_results:
        avg_improvement = np.mean([r["ll_improvement"] for r in all_results])
        avg_acc_diff = np.mean([r["acc_with_blend"] - r["acc_no_blend"] for r in all_results])

        for r in all_results:
            log.info("  %s: LL %+.4f, Acc %+.1fpp (%d matches)",
                     r["season"], r["ll_improvement"],
                     (r["acc_with_blend"] - r["acc_no_blend"]) * 100,
                     r["n_matches"])

        log.info("\n  Average LL improvement: %+.4f", avg_improvement)
        log.info("  Average Acc change: %+.2fpp", avg_acc_diff * 100)

        enable_blend = avg_improvement >= MIN_LL_IMPROVEMENT
    else:
        log.info("  No ablation results — training model only")
        avg_improvement = 0
        enable_blend = False

    # ------- Save model (always, even if blending disabled) -------
    if not dry_run:
        # Retrain on ALL data up to latest season for production use
        val_season = seasons[-1]
        train_seasons_full = seasons[:-1]

        X_all_full = _inject_draw_features(df[base_features].fillna(0))
        all_cols = [c for c in X_all_full.columns if X_all_full[c].dtype in [np.float64, np.int64, float, int]]

        X_tr_full = X_all_full.loc[df["season"].isin(train_seasons_full), all_cols]
        y_tr_full = (df.loc[df["season"].isin(train_seasons_full), "result"] == "D").astype(int)
        X_val_full = X_all_full.loc[df["season"] == val_season, all_cols]
        y_val_full = (df.loc[df["season"] == val_season, "result"] == "D").astype(int)

        log.info("\nTraining final model: train=%d, val=%d", len(X_tr_full), len(X_val_full))

        final_model = CatBoostClassifier(
            iterations=1500, depth=6, learning_rate=0.02,
            l2_leaf_reg=5, min_data_in_leaf=20,
            random_seed=42, verbose=0,
            loss_function="Logloss",
            auto_class_weights="Balanced",
        )
        final_model.fit(X_tr_full, y_tr_full, eval_set=(X_val_full, y_val_full),
                        early_stopping_rounds=150, verbose=0)

        # Calibrate on val
        raw_val_full = final_model.predict_proba(X_val_full)[:, 1]
        final_calibrator = IsotonicRegression(out_of_bounds="clip")
        final_calibrator.fit(raw_val_full, y_val_full.values)

        # Save
        model_path = UNIVERSAL_DIR / "draw_detector.cbm"
        cal_path = UNIVERSAL_DIR / "draw_detector_calibrator.pkl"
        meta_path = UNIVERSAL_DIR / "draw_detector_metadata.json"

        final_model.save_model(str(model_path))
        with open(cal_path, "wb") as f:
            pickle.dump(final_calibrator, f)

        metadata = {
            "trained_at": pd.Timestamp.now().isoformat(),
            "n_train": len(X_tr_full),
            "n_features": len(all_cols),
            "feature_names": all_cols,
            "blend_alpha": 0.32,
            "ablation_results": all_results,
            "avg_ll_improvement": round(avg_improvement, 5),
            "blend_enabled": enable_blend,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        log.info("Saved draw_detector.cbm (%d features)", len(all_cols))
        log.info("Saved draw_detector_calibrator.pkl")
        log.info("Saved draw_detector_metadata.json")

    # ------- Decision -------
    if enable_blend:
        log.info("\n  DECISION: ENABLE draw blending (avg LL improvement: %+.4f >= %.4f threshold)",
                 avg_improvement, MIN_LL_IMPROVEMENT)
    else:
        log.info("\n  DECISION: KEEP draw blending DISABLED (avg LL improvement: %+.4f < %.4f threshold)",
                 avg_improvement, MIN_LL_IMPROVEMENT)

    return {
        "enable_blend": enable_blend,
        "avg_ll_improvement": round(avg_improvement, 5),
        "results": all_results,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrain + ablation for draw detector")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate only, don't save")
    args = parser.parse_args()

    result = train_and_evaluate(dry_run=args.dry_run)

    print(f"\n{'=' * 60}")
    print(f"DRAW DETECTOR: {'ENABLE' if result['enable_blend'] else 'DISABLED'}")
    print(f"Avg LL improvement: {result['avg_ll_improvement']:+.4f}")
    print(f"{'=' * 60}")
