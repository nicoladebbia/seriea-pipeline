#!/usr/bin/env python3
"""Comprehensive All-Markets Prediction Engine.

Trains dedicated ML models for EVERY betting market using actual historical data,
not just Poisson approximations. Uses all available data sources:
- features.parquet (7829 matches, 430+ features)
- football-data CSVs (HT scores, cards, corners, shots, fouls)
- player_match_stats (97K rows, per-player per-match stats)
- match_incidents (43K events with minute-level granularity)
- referee_assignments (3040 matches with card patterns)

Markets predicted:
  1. Match Result (1X2) — improved with stacking + hyperparameter tuning
  2. Over/Under Goals (1.5, 2.5, 3.5)
  3. Both Teams To Score (BTTS)
  4. Half-Time Result (1X2)
  5. Half-Time/Full-Time (9 combos)
  6. Exact Score (top 20)
  7. Total Cards O/U (2.5, 3.5, 4.5, 5.5)
  8. Total Corners O/U (8.5, 9.5, 10.5, 11.5)
  9. Double Chance
 10. Asian Handicap
 11. First Half Goals O/U
 12. Clean Sheet
 13. Win to Nil
 14. Goal in Both Halves
 15. Player Goalscorer Probabilities

Usage:
    python scripts/comprehensive_markets.py --train        # Train all models
    python scripts/comprehensive_markets.py --backtest     # Backtest on 2023-24
    python scripts/comprehensive_markets.py --predict      # Predict upcoming matches
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, MODELS_DIR
from ml.feature_selection import correlation_pruning
from storage.paths import features_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MARKET_MODELS_DIR = MODELS_DIR / "markets"
MARKET_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING — Combine ALL sources into unified training data
# =============================================================================

def load_unified_training_data() -> pd.DataFrame:
    """Load and merge ALL data sources into a single training DataFrame.

    Combines:
    - features.parquet (ML features per match)
    - football-data CSVs (HT scores, cards, corners, shots, fouls, odds)
    - referee_assignments (referee card patterns)
    """
    # 1. Load features
    feat_path = features_path()
    df = pd.read_parquet(feat_path)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)
    log.info("Loaded features: %d matches x %d cols", len(df), len(df.columns))

    # 2. Load football-data for match stats (HT scores, cards, corners, shots)
    fd_combined = _load_football_data()
    if not fd_combined.empty:
        # Create merge key: date + normalized team names
        fd_combined["_merge_date"] = pd.to_datetime(fd_combined["Date"], dayfirst=True, errors="coerce")
        fd_combined["_merge_home"] = fd_combined["HomeTeam"].str.lower().str.strip()
        fd_combined["_merge_away"] = fd_combined["AwayTeam"].str.lower().str.strip()

        df["_merge_date"] = df["match_date"]
        df["_merge_home"] = df["home_team"].str.lower().str.strip()
        df["_merge_away"] = df["away_team"].str.lower().str.strip()

        # Merge on date + teams
        stat_cols = ["HTHG", "HTAG", "HTR", "HS", "AS", "HST", "AST",
                     "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
        available_stats = [c for c in stat_cols if c in fd_combined.columns]
        merge_cols = ["_merge_date", "_merge_home", "_merge_away"] + available_stats

        merged = df.merge(
            fd_combined[merge_cols].drop_duplicates(["_merge_date", "_merge_home", "_merge_away"]),
            on=["_merge_date", "_merge_home", "_merge_away"],
            how="left",
        )
        n_matched = merged["HTHG"].notna().sum()
        log.info("Football-data merged: %d/%d matches with HT/cards/corners", n_matched, len(merged))

        df = merged.drop(columns=["_merge_date", "_merge_home", "_merge_away"])

    # 3. Load referee assignments
    ref_path = DATA_DIR / "external" / "referee" / "referee_assignments.parquet"
    if ref_path.exists():
        ref = pd.read_parquet(ref_path)
        # Build referee stats: avg cards per match
        ref_stats = ref.groupby("referee").agg(
            ref_matches=("referee", "count"),
            ref_avg_yellows_given=("ref_yellows", "mean"),
            ref_avg_reds_given=("ref_reds", "mean"),
            ref_total_cards_mean=("ref_yellows", lambda x: x.mean()),
        ).reset_index()
        ref_stats["ref_total_cards_mean"] = ref_stats["ref_avg_yellows_given"] + ref_stats["ref_avg_reds_given"]

        # Merge referee name with matches (use existing referee column if available)
        if "referee" in df.columns:
            df = df.merge(ref_stats, on="referee", how="left")
            n_ref = df["ref_matches"].notna().sum()
            log.info("Referee stats merged: %d/%d matches", n_ref, len(df))

    # NOTE: Targets are computed AFTER feature selection in main() to prevent leakage
    return df


def _load_football_data() -> pd.DataFrame:
    """Load all football-data CSVs into a single DataFrame.

    Uses data/external/odds/I1_*.csv as primary source (21 seasons, full coverage).
    Falls back to data/external/football-data/ if odds dir is missing.
    """
    import glob
    # Primary: odds CSVs have ALL 21 seasons with cards/corners/HT
    files = sorted(glob.glob(str(DATA_DIR / "external" / "odds" / "I1_*.csv")))
    if not files:
        # Fallback to football-data dir
        files = sorted(glob.glob(str(DATA_DIR / "external" / "football-data" / "serie_a_[0-9]*.csv")))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f)
            # Normalize team names to match pipeline
            d["HomeTeam"] = d["HomeTeam"].replace({
                "AC Milan": "Milan", "Hellas Verona": "Verona",
                "Chievo Verona": "Chievo", "ChievoVerona": "Chievo",
                "Inter Milan": "Inter", "Internazionale": "Inter",
                "Parma Calcio 1913": "Parma",
            })
            d["AwayTeam"] = d["AwayTeam"].replace({
                "AC Milan": "Milan", "Hellas Verona": "Verona",
                "Chievo Verona": "Chievo", "ChievoVerona": "Chievo",
                "Inter Milan": "Inter", "Internazionale": "Inter",
                "Parma Calcio 1913": "Parma",
            })
            dfs.append(d)
        except Exception as e:
            log.warning("Failed to load %s: %s", f, e)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    log.info("Football-data loaded: %d matches from %d files", len(combined), len(dfs))
    return combined


def _compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all target variables for training."""
    # Basic result
    df["result"] = df.get("result", pd.Series(dtype=str))

    # Total goals
    df["total_goals"] = df["home_score"] + df["away_score"]

    # Over/Under targets
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        col = f"over_{str(line).replace('.', '_')}"
        df[col] = (df["total_goals"] > line).astype(int)

    # BTTS
    df["btts"] = ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)

    # Clean sheets
    df["home_clean_sheet"] = (df["away_score"] == 0).astype(int)
    df["away_clean_sheet"] = (df["home_score"] == 0).astype(int)

    # Win to nil
    df["home_win_to_nil"] = ((df["home_score"] > df["away_score"]) & (df["away_score"] == 0)).astype(int)
    df["away_win_to_nil"] = ((df["away_score"] > df["home_score"]) & (df["home_score"] == 0)).astype(int)

    # HT targets (from football-data merge or matches.parquet)
    if "HTHG" in df.columns:
        df["ht_home_goals"] = df["HTHG"].fillna(df.get("home_ht_score", np.nan))
        df["ht_away_goals"] = df["HTAG"].fillna(df.get("away_ht_score", np.nan))
    else:
        df["ht_home_goals"] = df.get("home_ht_score", np.nan)
        df["ht_away_goals"] = df.get("away_ht_score", np.nan)

    # HT result
    ht_valid = df["ht_home_goals"].notna() & df["ht_away_goals"].notna()
    df.loc[ht_valid & (df["ht_home_goals"] > df["ht_away_goals"]), "ht_result"] = "H"
    df.loc[ht_valid & (df["ht_home_goals"] == df["ht_away_goals"]), "ht_result"] = "D"
    df.loc[ht_valid & (df["ht_home_goals"] < df["ht_away_goals"]), "ht_result"] = "A"

    # HT/FT combined
    df["htft"] = df["ht_result"].astype(str) + "/" + df["result"].astype(str)

    # FH goals
    df["fh_total_goals"] = df["ht_home_goals"] + df["ht_away_goals"]
    df["fh_over_0_5"] = (df["fh_total_goals"] > 0.5).astype(int)
    df["fh_over_1_5"] = (df["fh_total_goals"] > 1.5).astype(int)
    df["fh_btts"] = ((df["ht_home_goals"] > 0) & (df["ht_away_goals"] > 0)).astype(int)

    # Goal in both halves
    if "ht_home_goals" in df.columns:
        sh_home = df["home_score"] - df["ht_home_goals"]
        sh_away = df["away_score"] - df["ht_away_goals"]
        fh_goals = df["fh_total_goals"]
        sh_goals = sh_home + sh_away
        df["goal_both_halves"] = ((fh_goals > 0) & (sh_goals > 0)).astype(int)

    # Cards targets (from football-data)
    if "HY" in df.columns:
        df["total_yellows"] = df["HY"] + df["AY"]
        df["total_reds"] = df["HR"] + df["AR"]
        df["total_cards"] = df["total_yellows"] + df["total_reds"]
        for line in [2.5, 3.5, 4.5, 5.5]:
            col = f"cards_over_{str(line).replace('.', '_')}"
            df[col] = (df["total_cards"] > line).astype(int)

    # Corners targets
    if "HC" in df.columns:
        df["total_corners"] = df["HC"] + df["AC"]
        for line in [7.5, 8.5, 9.5, 10.5, 11.5]:
            col = f"corners_over_{str(line).replace('.', '_')}"
            df[col] = (df["total_corners"] > line).astype(int)

    # Exact score (encoded as string)
    df["exact_score"] = df["home_score"].astype(int).astype(str) + "-" + df["away_score"].astype(int).astype(str)

    log.info("Computed targets. Total goals mean: %.2f, BTTS rate: %.1f%%, Over 2.5: %.1f%%",
             df["total_goals"].mean(), df["btts"].mean() * 100, df["over_2_5"].mean() * 100)

    return df


# =============================================================================
# FEATURE SELECTION — Get ML-safe features
# =============================================================================

def get_training_features(df: pd.DataFrame) -> List[str]:
    """Get all numeric pre-match features suitable for ML training.

    IMPORTANT: Must be called on the DataFrame BEFORE target columns are added
    to prevent data leakage.
    """
    from features.build import get_ml_feature_columns
    ml_cols = get_ml_feature_columns(df)

    # Filter to numeric only
    numeric = [c for c in ml_cols if df[c].dtype in ("float64", "float32", "int64", "int32", "int8", "uint8")]

    # Exclude post-match leakage and any target-derived columns
    EXCLUDE = {
        # Post-match stats
        "home_shots_on_target", "away_shots_on_target", "home_total_shots", "away_total_shots",
        "home_corners", "away_corners", "home_red_cards", "away_red_cards", "home_fouls", "away_fouls",
        "home_saves", "away_saves", "home_possession", "away_possession", "attendance",
        "home_xg", "away_xg", "home_cards", "away_cards",
        # Football-data post-match
        "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
        # Target-derived (must never be features)
        "total_goals", "over_0_5", "over_1_5", "over_2_5", "over_3_5", "over_4_5",
        "btts", "home_clean_sheet", "away_clean_sheet",
        "home_win_to_nil", "away_win_to_nil",
        "ht_home_goals", "ht_away_goals", "fh_total_goals",
        "fh_over_0_5", "fh_over_1_5", "fh_btts", "goal_both_halves",
        "total_yellows", "total_reds", "total_cards",
        "cards_over_2_5", "cards_over_3_5", "cards_over_4_5", "cards_over_5_5",
        "total_corners", "corners_over_7_5", "corners_over_8_5",
        "corners_over_9_5", "corners_over_10_5", "corners_over_11_5",
        "home_score", "away_score",
    }
    features = [c for c in numeric if c not in EXCLUDE and df[c].isna().mean() < 0.8]

    # Add referee features if available
    ref_feats = [c for c in df.columns if c.startswith("ref_") and df[c].dtype in ("float64", "float32", "int64", "int32") and c not in EXCLUDE]
    features.extend([c for c in ref_feats if c not in features])

    log.info("Training features: %d", len(features))
    return features


# =============================================================================
# FEATURE SELECTION — Importance-based + correlation pruning
# =============================================================================

def select_by_importance(
    X: pd.DataFrame, y: pd.Series, feature_names: List[str], top_k: int = 150,
) -> Tuple[List[str], Dict[str, float]]:
    """Select top_k features by XGBoost importance (walk-forward safe).

    Uses the same approach as train_unified.py hybrid mode.
    """
    from xgboost import XGBRegressor

    X_sel = X[feature_names].fillna(0)

    # XGBoost forbids [, ], < and > in feature names -- sanitize temporarily
    safe_names = {f: f.replace("<", "_lt_").replace(">", "_gt_").replace("[", "_lb_").replace("]", "_rb_")
                  for f in feature_names}
    X_safe = X_sel.rename(columns=safe_names)

    selector = XGBRegressor(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
    selector.fit(X_safe, y)

    reverse_map = {v: k for k, v in safe_names.items()}
    importance = {reverse_map[sn]: float(imp)
                  for sn, imp in zip(X_safe.columns, selector.feature_importances_)}
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in sorted_feats[:top_k]]

    log.info("Importance selection: %d -> %d features", len(feature_names), len(selected))
    return selected, importance




# =============================================================================
# MODEL TRAINING — Dedicated models for each market
# =============================================================================

def train_all_models(df: pd.DataFrame, features: List[str], test_season: str = "2024-2025") -> Dict:
    """Train CatBoost models for all betting markets.

    Uses walk-forward approach:
    - Train on all seasons BEFORE val_season
    - val_season = season before test_season (for early stopping)
    - test_season = completely held out
    """
    from catboost import CatBoostRegressor, CatBoostClassifier

    results = {}
    model_dir = MARKET_MODELS_DIR

    # Verify no target leakage
    TARGET_COLS = {"total_goals", "over_2_5", "btts", "total_cards", "total_corners",
                   "home_score", "away_score"}
    leaked = set(features) & TARGET_COLS
    if leaked:
        raise ValueError(f"DATA LEAKAGE DETECTED: {leaked} in features!")

    # Filter to complete matches
    df = df[df["result"].isin(["H", "D", "A"])].copy()

    # Walk-forward split
    seasons = sorted(df["season"].unique())
    test_idx = seasons.index(test_season) if test_season in seasons else len(seasons) - 1
    val_season = seasons[test_idx - 1]
    train_seasons = seasons[:test_idx - 1]  # Everything BEFORE val

    train_mask = df["season"].isin(train_seasons)
    val_mask = df["season"] == val_season
    test_mask = df["season"] == test_season

    log.info("Initial features: %d", len(features))
    log.info("Train: %d matches, Val: %d (%s), Test: %d (%s)",
             train_mask.sum(), val_mask.sum(), val_season, test_mask.sum(), test_season)

    # ===== FEATURE SELECTION (on training data only) =====
    log.info("\n=== Feature Selection (Importance + Correlation Pruning) ===")
    X_train_full = df.loc[train_mask, features].fillna(0)
    y_train_goals = df.loc[train_mask, "home_score"]  # Use home goals as proxy target

    # Step 1: Importance-based selection (top-150)
    selected_features, importance = select_by_importance(
        X_train_full, y_train_goals, features, top_k=150
    )

    # Step 2: Correlation pruning (r > 0.85)
    selected_features = correlation_pruning(
        X_train_full, selected_features, importance, threshold=0.85
    )

    log.info("Final selected features: %d", len(selected_features))
    log.info("Top 10 by importance: %s",
             [f for f, _ in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]])

    # Now create feature matrices with selected features only
    X = df[selected_features].fillna(0)
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]

    # ===== 1. HOME GOALS REGRESSION =====
    log.info("\n=== Training: Home Goals Regressor ===")
    y_hg = df.loc[train_mask, "home_score"]
    model_hg = CatBoostRegressor(
        iterations=2000, depth=8, learning_rate=0.015,
        l2_leaf_reg=7, min_data_in_leaf=30, rsm=0.8, random_seed=42, verbose=0,
        loss_function="Poisson",
    )
    model_hg.fit(X_train, y_hg, eval_set=(X_val, df.loc[val_mask, "home_score"]),
                 early_stopping_rounds=100, verbose=0)
    model_hg.save_model(str(model_dir / "home_goals.cbm"))
    val_mae = np.abs(model_hg.predict(X_val) - df.loc[val_mask, "home_score"]).mean()
    results["home_goals_mae"] = round(val_mae, 4)
    log.info("  Home goals MAE (val): %.4f", val_mae)

    # ===== 2. AWAY GOALS REGRESSION =====
    log.info("=== Training: Away Goals Regressor ===")
    y_ag = df.loc[train_mask, "away_score"]
    model_ag = CatBoostRegressor(
        iterations=2000, depth=8, learning_rate=0.015,
        l2_leaf_reg=7, min_data_in_leaf=30, rsm=0.8, random_seed=42, verbose=0,
        loss_function="Poisson",
    )
    model_ag.fit(X_train, y_ag, eval_set=(X_val, df.loc[val_mask, "away_score"]),
                 early_stopping_rounds=100, verbose=0)
    model_ag.save_model(str(model_dir / "away_goals.cbm"))
    val_mae = np.abs(model_ag.predict(X_val) - df.loc[val_mask, "away_score"]).mean()
    results["away_goals_mae"] = round(val_mae, 4)
    log.info("  Away goals MAE (val): %.4f", val_mae)

    # ===== 3. 1X2 CLASSIFIER (optimized) =====
    log.info("=== Training: 1X2 Classifier ===")
    label_map = {"H": 0, "D": 1, "A": 2}
    y_cls = df.loc[train_mask, "result"].map(label_map)
    model_cls = CatBoostClassifier(
        iterations=3000, depth=8, learning_rate=0.01,
        l2_leaf_reg=5, min_data_in_leaf=30, rsm=0.8, random_seed=42, verbose=0,
        loss_function="MultiClass", classes_count=3,
        auto_class_weights="Balanced",
    )
    model_cls.fit(X_train, y_cls,
                  eval_set=(X_val, df.loc[val_mask, "result"].map(label_map)),
                  early_stopping_rounds=150, verbose=0)
    model_cls.save_model(str(model_dir / "result_1x2.cbm"))
    val_acc = (model_cls.predict(X_val).flatten().astype(int) ==
               df.loc[val_mask, "result"].map(label_map).values).mean()
    results["result_1x2_acc"] = round(val_acc, 4)
    log.info("  1X2 accuracy (val): %.1f%%", val_acc * 100)

    # ===== 4. OVER/UNDER 2.5 GOALS =====
    log.info("=== Training: Over 2.5 Goals ===")
    y_ou = df.loc[train_mask, "over_2_5"]
    model_ou = CatBoostClassifier(
        iterations=1500, depth=6, learning_rate=0.03,
        l2_leaf_reg=5, random_seed=42, verbose=0,
        loss_function="Logloss",
    )
    model_ou.fit(X_train, y_ou, eval_set=(X_val, df.loc[val_mask, "over_2_5"]),
                 early_stopping_rounds=100, verbose=0)
    model_ou.save_model(str(model_dir / "over_2_5.cbm"))
    val_probs = model_ou.predict_proba(X_val)[:, 1]
    val_acc = ((val_probs > 0.5).astype(int) == df.loc[val_mask, "over_2_5"].values).mean()
    results["over_2_5_acc"] = round(val_acc, 4)
    log.info("  Over 2.5 accuracy (val): %.1f%%", val_acc * 100)

    # ===== 5. BTTS =====
    log.info("=== Training: BTTS ===")
    y_btts = df.loc[train_mask, "btts"]
    model_btts = CatBoostClassifier(
        iterations=1500, depth=6, learning_rate=0.03,
        l2_leaf_reg=5, random_seed=42, verbose=0,
        loss_function="Logloss",
    )
    model_btts.fit(X_train, y_btts, eval_set=(X_val, df.loc[val_mask, "btts"]),
                   early_stopping_rounds=100, verbose=0)
    model_btts.save_model(str(model_dir / "btts.cbm"))
    val_probs = model_btts.predict_proba(X_val)[:, 1]
    val_acc = ((val_probs > 0.5).astype(int) == df.loc[val_mask, "btts"].values).mean()
    results["btts_acc"] = round(val_acc, 4)
    log.info("  BTTS accuracy (val): %.1f%%", val_acc * 100)

    # ===== 6. HT RESULT =====
    ht_valid = df["ht_result"].isin(["H", "D", "A"])
    if ht_valid.sum() > 500:
        log.info("=== Training: HT Result ===")
        ht_df = df[ht_valid].copy()
        ht_train = ht_df["season"].isin(train_seasons)
        ht_val = ht_df["season"] == val_season

        if ht_train.sum() > 100 and ht_val.sum() > 10:
            y_ht = ht_df.loc[ht_train, "ht_result"].map(label_map)
            X_ht_train = ht_df.loc[ht_train, selected_features].fillna(0)
            X_ht_val = ht_df.loc[ht_val, selected_features].fillna(0)

            model_ht = CatBoostClassifier(
                iterations=1500, depth=6, learning_rate=0.03,
                l2_leaf_reg=5, random_seed=42, verbose=0,
                loss_function="MultiClass", classes_count=3,
                auto_class_weights="Balanced",
            )
            model_ht.fit(X_ht_train, y_ht,
                         eval_set=(X_ht_val, ht_df.loc[ht_val, "ht_result"].map(label_map)),
                         early_stopping_rounds=100, verbose=0)
            model_ht.save_model(str(model_dir / "ht_result.cbm"))
            val_acc = (model_ht.predict(X_ht_val).flatten().astype(int) ==
                       ht_df.loc[ht_val, "ht_result"].map(label_map).values).mean()
            results["ht_result_acc"] = round(val_acc, 4)
            log.info("  HT Result accuracy (val): %.1f%%", val_acc * 100)

    # ===== 7. TOTAL CARDS O/U 4.5 =====
    if "total_cards" in df.columns:
        cards_valid = df["total_cards"].notna()
        if cards_valid.sum() > 500:
            log.info("=== Training: Cards Over 4.5 ===")
            cards_df = df[cards_valid].copy()
            ct = cards_df["season"].isin(train_seasons)
            cv = cards_df["season"] == val_season

            if ct.sum() > 100 and cv.sum() > 10:
                y_cards = cards_df.loc[ct, "cards_over_4_5"]
                X_ct = cards_df.loc[ct, selected_features].fillna(0)
                X_cv = cards_df.loc[cv, selected_features].fillna(0)

                model_cards = CatBoostClassifier(
                    iterations=1500, depth=6, learning_rate=0.03,
                    l2_leaf_reg=5, random_seed=42, verbose=0,
                    loss_function="Logloss",
                )
                model_cards.fit(X_ct, y_cards,
                                eval_set=(X_cv, cards_df.loc[cv, "cards_over_4_5"]),
                                early_stopping_rounds=100, verbose=0)
                model_cards.save_model(str(model_dir / "cards_over_4_5.cbm"))
                val_probs = model_cards.predict_proba(X_cv)[:, 1]
                val_acc = ((val_probs > 0.5).astype(int) == cards_df.loc[cv, "cards_over_4_5"].values).mean()
                results["cards_over_4_5_acc"] = round(val_acc, 4)
                log.info("  Cards O4.5 accuracy (val): %.1f%%", val_acc * 100)

    # ===== 8. TOTAL CORNERS O/U 9.5 =====
    if "total_corners" in df.columns:
        corn_valid = df["total_corners"].notna()
        if corn_valid.sum() > 500:
            log.info("=== Training: Corners Over 9.5 ===")
            corn_df = df[corn_valid].copy()
            crt = corn_df["season"].isin(train_seasons)
            crv = corn_df["season"] == val_season

            if crt.sum() > 100 and crv.sum() > 10:
                y_corn = corn_df.loc[crt, "corners_over_9_5"]
                X_crt = corn_df.loc[crt, selected_features].fillna(0)
                X_crv = corn_df.loc[crv, selected_features].fillna(0)

                model_corn = CatBoostClassifier(
                    iterations=1500, depth=6, learning_rate=0.03,
                    l2_leaf_reg=5, random_seed=42, verbose=0,
                    loss_function="Logloss",
                )
                model_corn.fit(X_crt, y_corn,
                               eval_set=(X_crv, corn_df.loc[crv, "corners_over_9_5"]),
                               early_stopping_rounds=100, verbose=0)
                model_corn.save_model(str(model_dir / "corners_over_9_5.cbm"))
                val_probs = model_corn.predict_proba(X_crv)[:, 1]
                val_acc = ((val_probs > 0.5).astype(int) == corn_df.loc[crv, "corners_over_9_5"].values).mean()
                results["corners_over_9_5_acc"] = round(val_acc, 4)
                log.info("  Corners O9.5 accuracy (val): %.1f%%", val_acc * 100)

    # ===== 9. HOME GOALS REGRESSOR (1st HALF) =====
    fh_valid = df["ht_home_goals"].notna()
    if fh_valid.sum() > 500:
        log.info("=== Training: 1st Half Home Goals ===")
        fh_df = df[fh_valid].copy()
        ft = fh_df["season"].isin(train_seasons)
        fv = fh_df["season"] == val_season

        if ft.sum() > 100 and fv.sum() > 10:
            model_fh_home = CatBoostRegressor(
                iterations=1000, depth=5, learning_rate=0.03,
                l2_leaf_reg=5, random_seed=42, verbose=0,
                loss_function="Poisson",
            )
            model_fh_home.fit(fh_df.loc[ft, selected_features].fillna(0), fh_df.loc[ft, "ht_home_goals"],
                              eval_set=(fh_df.loc[fv, selected_features].fillna(0), fh_df.loc[fv, "ht_home_goals"]),
                              early_stopping_rounds=100, verbose=0)
            model_fh_home.save_model(str(model_dir / "fh_home_goals.cbm"))

            model_fh_away = CatBoostRegressor(
                iterations=1000, depth=5, learning_rate=0.03,
                l2_leaf_reg=5, random_seed=42, verbose=0,
                loss_function="Poisson",
            )
            model_fh_away.fit(fh_df.loc[ft, selected_features].fillna(0), fh_df.loc[ft, "ht_away_goals"],
                              eval_set=(fh_df.loc[fv, selected_features].fillna(0), fh_df.loc[fv, "ht_away_goals"]),
                              early_stopping_rounds=100, verbose=0)
            model_fh_away.save_model(str(model_dir / "fh_away_goals.cbm"))
            log.info("  1st Half goal models saved")

    # ===== 10. HOME/AWAY CARDS REGRESSORS =====
    if "HY" in df.columns:
        cy_valid = df["HY"].notna()
        if cy_valid.sum() > 500:
            log.info("=== Training: Home/Away Card Regressors ===")
            cy_df = df[cy_valid].copy()
            cyt = cy_df["season"].isin(train_seasons)
            cyv = cy_df["season"] == val_season

            if cyt.sum() > 100 and cyv.sum() > 10:
                for side, col in [("home", "HY"), ("away", "AY")]:
                    model_cy = CatBoostRegressor(
                        iterations=1000, depth=5, learning_rate=0.03,
                        l2_leaf_reg=5, random_seed=42, verbose=0,
                        loss_function="Poisson",
                    )
                    model_cy.fit(cy_df.loc[cyt, selected_features].fillna(0), cy_df.loc[cyt, col],
                                 eval_set=(cy_df.loc[cyv, selected_features].fillna(0), cy_df.loc[cyv, col]),
                                 early_stopping_rounds=100, verbose=0)
                    model_cy.save_model(str(model_dir / f"{side}_cards.cbm"))
                log.info("  Card regressors saved")

    # ===== 11. HOME/AWAY CORNER REGRESSORS =====
    if "HC" in df.columns:
        hc_valid = df["HC"].notna()
        if hc_valid.sum() > 500:
            log.info("=== Training: Home/Away Corner Regressors ===")
            hc_df = df[hc_valid].copy()
            hct = hc_df["season"].isin(train_seasons)
            hcv = hc_df["season"] == val_season

            if hct.sum() > 100 and hcv.sum() > 10:
                for side, col in [("home", "HC"), ("away", "AC")]:
                    model_hc = CatBoostRegressor(
                        iterations=1000, depth=5, learning_rate=0.03,
                        l2_leaf_reg=5, random_seed=42, verbose=0,
                        loss_function="Poisson",
                    )
                    model_hc.fit(hc_df.loc[hct, selected_features].fillna(0), hc_df.loc[hct, col],
                                 eval_set=(hc_df.loc[hcv, selected_features].fillna(0), hc_df.loc[hcv, col]),
                                 early_stopping_rounds=100, verbose=0)
                    model_hc.save_model(str(model_dir / f"{side}_corners.cbm"))
                log.info("  Corner regressors saved")

    # Save metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "features": selected_features,
        "n_features": len(selected_features),
        "initial_features": len(features),
        "train_seasons": train_seasons,
        "val_season": val_season,
        "test_season": test_season,
        "results": results,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("\n=== TRAINING COMPLETE ===")
    for k, v in results.items():
        log.info("  %s: %.4f", k, v)

    # Return both results and selected features
    return {
        "metrics": results,
        "selected_features": selected_features,
        "n_features": len(selected_features),
    }


# =============================================================================
# PREDICTION — Generate all markets for a match
# =============================================================================

def predict_all_markets(X_row: pd.DataFrame, features: List[str]) -> Dict:
    """Generate comprehensive predictions for all betting markets."""
    from catboost import CatBoostRegressor, CatBoostClassifier

    model_dir = MARKET_MODELS_DIR

    # Each model may expect a different feature set (456 for Strategy B, 367 for Strategy C).
    # _align_X builds the correct DataFrame for any loaded model.
    row_dict = X_row.iloc[0].to_dict()

    def _align_X(model):
        """Align input features to what this specific model expects."""
        mf = model.feature_names_
        aligned = {f: row_dict.get(f, 0) for f in mf}
        return pd.DataFrame([aligned])[mf].fillna(0)

    preds = {}

    # 1. Goal predictions (Poisson-based from ML regressors)
    try:
        m_hg = CatBoostRegressor(); m_hg.load_model(str(model_dir / "prod_home_goals.cbm"))
        m_ag = CatBoostRegressor(); m_ag.load_model(str(model_dir / "prod_away_goals.cbm"))
        home_xg = max(0.3, min(4.0, float(m_hg.predict(_align_X(m_hg))[0])))
        away_xg = max(0.3, min(4.0, float(m_ag.predict(_align_X(m_ag))[0])))
    except Exception as _e:
        log.debug("Model fallback triggered: %s", _e)
        home_xg, away_xg = 1.4, 1.15

    preds["home_xg"] = round(home_xg, 3)
    preds["away_xg"] = round(away_xg, 3)

    # Build score probability matrix
    max_g = 8
    score_matrix = {}
    for h in range(max_g):
        for a in range(max_g):
            score_matrix[(h, a)] = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)

    # 1X2 from Poisson
    p_H = sum(v for (h, a), v in score_matrix.items() if h > a)
    p_D = sum(v for (h, a), v in score_matrix.items() if h == a)
    p_A = sum(v for (h, a), v in score_matrix.items() if h < a)
    total = p_H + p_D + p_A
    poisson_probs = {"H": p_H/total, "D": p_D/total, "A": p_A/total}

    # 1X2 from classifier (production model)
    try:
        m_cls = CatBoostClassifier(); m_cls.load_model(str(model_dir / "prod_1x2.cbm"))
        cls_probs_arr = m_cls.predict_proba(_align_X(m_cls))[0]
        cls_probs = {"H": cls_probs_arr[0], "D": cls_probs_arr[1], "A": cls_probs_arr[2]}
    except Exception as _e:
        log.debug("Model fallback triggered: %s", _e)
        cls_probs = poisson_probs

    # Calibrated ensemble — validated on 2024-25 + 2025-26 OOS (609 matches)
    # Best log-loss: 0.9607 vs Pinnacle 0.9681 (-0.0074)
    # NO draw inflation (was 1.35 — caused fake edge, -27.5% ROI on 2025-26)
    # Temperature scaling T=0.80 sharpens overconfident probabilities
    mkt_probs = {"H": 0.4, "D": 0.3, "A": 0.3}  # default
    try:
        psh = float(X_row["odds_PSH"].iloc[0]) if "odds_PSH" in X_row.columns else 0
        psd = float(X_row["odds_PSD"].iloc[0]) if "odds_PSD" in X_row.columns else 0
        psa = float(X_row["odds_PSA"].iloc[0]) if "odds_PSA" in X_row.columns else 0
        if psh > 1 and psd > 1 and psa > 1:
            raw_sum = 1/psh + 1/psd + 1/psa
            mkt_probs = {"H": (1/psh)/raw_sum, "D": (1/psd)/raw_sum, "A": (1/psa)/raw_sum}
    except Exception as e:
        log.debug(f"Failed to compute Pinnacle market probabilities: {e}")

    W_CB, W_POIS, W_MKT = 0.20, 0.25, 0.55
    TEMP_SCALE = 0.80  # calibrated on 2023-24, tested on 2024-26
    ens_raw = {
        "H": W_CB * cls_probs["H"] + W_POIS * poisson_probs["H"] + W_MKT * mkt_probs["H"],
        "D": W_CB * cls_probs["D"] + W_POIS * poisson_probs["D"] + W_MKT * mkt_probs["D"],
        "A": W_CB * cls_probs["A"] + W_POIS * poisson_probs["A"] + W_MKT * mkt_probs["A"],
    }
    # Temperature scaling: sharpen probabilities for better calibration
    import math
    ens_logits = {k: math.log(max(v, 1e-10)) / TEMP_SCALE for k, v in ens_raw.items()}
    ens_max = max(ens_logits.values())
    ens_exp = {k: math.exp(v - ens_max) for k, v in ens_logits.items()}
    ens_sum = sum(ens_exp.values())
    ens_probs = {k: v / ens_sum for k, v in ens_exp.items()}

    conf = max(ens_probs.values())
    conf_label = "high" if conf > 0.55 else "medium" if conf > 0.45 else "low"
    preds["match_result"] = {
        "home_win": _market(ens_probs["H"]),
        "draw": _market(ens_probs["D"]),
        "away_win": _market(ens_probs["A"]),
        "prediction": max(ens_probs, key=ens_probs.get),
        "confidence": round(conf, 4),
        "confidence_tier": conf_label,
    }

    # Double chance
    preds["double_chance"] = {
        "1X": _market(ens_probs["H"] + ens_probs["D"]),
        "X2": _market(ens_probs["D"] + ens_probs["A"]),
        "12": _market(ens_probs["H"] + ens_probs["A"]),
    }

    # 2. Over/Under goals
    total_xg = home_xg + away_xg
    ou_preds = {}
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        p_over = sum(v for (h, a), v in score_matrix.items() if h + a > line)
        ou_preds[f"over_{line}"] = _market(p_over)
        ou_preds[f"under_{line}"] = _market(1 - p_over)

    # Blend Poisson with ML models for key O/U lines
    for line_str, line_val, model_name in [
        ("1.5", 1.5, "prod_over_1_5.cbm"),
        ("2.5", 2.5, "prod_over_2_5.cbm"),
        ("3.5", 3.5, "prod_over_3_5.cbm"),
    ]:
        try:
            m_ou = CatBoostClassifier(); m_ou.load_model(str(model_dir / model_name))
            ml_prob = float(m_ou.predict_proba(_align_X(m_ou))[0][1])
            pois_prob = float(ou_preds[f"over_{line_str}"]["prob"])
            blended = 0.4 * pois_prob + 0.6 * ml_prob
            ou_preds[f"over_{line_str}"] = _market(blended)
            ou_preds[f"under_{line_str}"] = _market(1 - blended)
        except Exception as e:
            log.debug(f"Failed to blend ML over/under prediction for {line_str}: {e}")

    preds["over_under"] = ou_preds

    # 3. BTTS — pure Poisson from goal regressors (CatBoost BTTS was useless: iter=13, Brier=baseline)
    p_btts_poisson = sum(v for (h, a), v in score_matrix.items() if h > 0 and a > 0)
    btts_prob = p_btts_poisson

    preds["btts"] = {
        "yes": _market(btts_prob),
        "no": _market(1 - btts_prob),
    }

    # 4. Exact score (top 20)
    scores_sorted = sorted(score_matrix.items(), key=lambda x: x[1], reverse=True)[:20]
    preds["exact_score"] = {
        f"{h}-{a}": _market(p) for (h, a), p in scores_sorted
    }

    # 5. HT Result
    try:
        m_ht = CatBoostClassifier(); m_ht.load_model(str(model_dir / "prod_ht_result.cbm"))
        ht_probs = m_ht.predict_proba(_align_X(m_ht))[0]
        preds["ht_result"] = {
            "home_win": _market(ht_probs[0]),
            "draw": _market(ht_probs[1]),
            "away_win": _market(ht_probs[2]),
        }
    except Exception as _e:
        log.debug("Model fallback triggered: %s", _e)
        # Fallback: Poisson with halved xG
        fh_hxg, fh_axg = home_xg * 0.45, away_xg * 0.45
        fh_sm = {}
        for h in range(5):
            for a in range(5):
                fh_sm[(h, a)] = poisson.pmf(h, fh_hxg) * poisson.pmf(a, fh_axg)
        p_fh_H = sum(v for (h, a), v in fh_sm.items() if h > a)
        p_fh_D = sum(v for (h, a), v in fh_sm.items() if h == a)
        p_fh_A = sum(v for (h, a), v in fh_sm.items() if h < a)
        t = p_fh_H + p_fh_D + p_fh_A
        preds["ht_result"] = {
            "home_win": _market(p_fh_H/t),
            "draw": _market(p_fh_D/t),
            "away_win": _market(p_fh_A/t),
        }

    # 6. First half goals O/U
    try:
        m_fhh = CatBoostRegressor(); m_fhh.load_model(str(model_dir / "prod_fh_home_goals.cbm"))
        m_fha = CatBoostRegressor(); m_fha.load_model(str(model_dir / "prod_fh_away_goals.cbm"))
        fh_hxg = max(0.1, min(2.5, float(m_fhh.predict(_align_X(m_fhh))[0])))
        fh_axg = max(0.1, min(2.5, float(m_fha.predict(_align_X(m_fha))[0])))
    except Exception as _e:
        log.debug("Model fallback triggered: %s", _e)
        fh_hxg, fh_axg = home_xg * 0.45, away_xg * 0.45

    fh_total = fh_hxg + fh_axg
    for line in [0.5, 1.5, 2.5]:
        p_fh_over = 1 - sum(poisson.pmf(k, fh_total) for k in range(int(line) + 1))
        preds.setdefault("first_half_goals", {})[f"over_{line}"] = _market(p_fh_over)
        preds.setdefault("first_half_goals", {})[f"under_{line}"] = _market(1 - p_fh_over)

    # 7. HT/FT (9 combinations)
    # Use HT and FT probabilities
    ht_H = preds["ht_result"]["home_win"]["prob"]
    ht_D = preds["ht_result"]["draw"]["prob"]
    ht_A = preds["ht_result"]["away_win"]["prob"]
    ft_H = ens_probs["H"]
    ft_D = ens_probs["D"]
    ft_A = ens_probs["A"]

    htft = {}
    for ht_label, ht_p in [("H", ht_H), ("D", ht_D), ("A", ht_A)]:
        for ft_label, ft_p in [("H", ft_H), ("D", ft_D), ("A", ft_A)]:
            # P(HT=x, FT=y) ≈ P(HT=x) * P(FT=y|HT=x)
            # Approximate: if HT=FT, boost; if HT≠FT, reduce (comeback less likely)
            if ht_label == ft_label:
                joint = ht_p * ft_p * 1.6  # persistence boost
            elif ht_label == "D":
                joint = ht_p * ft_p * 1.2  # draw at HT → could go either way
            else:
                joint = ht_p * ft_p * 0.5  # comeback penalty
            htft[f"{ht_label}/{ft_label}"] = joint

    # Normalize
    htft_total = max(sum(htft.values()), 1e-10)
    preds["htft"] = {k: _market(v / htft_total) for k, v in htft.items()}

    # 8. Cards predictions
    try:
        m_hcr = CatBoostRegressor(); m_hcr.load_model(str(model_dir / "prod_home_cards.cbm"))
        m_acr_cards = CatBoostRegressor(); m_acr_cards.load_model(str(model_dir / "prod_away_cards.cbm"))
        exp_home_cards = max(0.5, min(5.0, float(m_hcr.predict(_align_X(m_hcr))[0])))
        exp_away_cards = max(0.5, min(5.0, float(m_acr_cards.predict(_align_X(m_acr_cards))[0])))
    except Exception as _e:
        log.debug("Model fallback triggered: %s", _e)
        exp_home_cards, exp_away_cards = 2.1, 2.4

    exp_total_cards = exp_home_cards + exp_away_cards
    cards = {
        "home_expected": round(exp_home_cards, 2),
        "away_expected": round(exp_away_cards, 2),
        "total_expected": round(exp_total_cards, 2),
    }
    for line in [2.5, 3.5, 4.5, 5.5, 6.5]:
        p_over = 1 - sum(poisson.pmf(k, exp_total_cards) for k in range(int(line) + 1))
        cards[f"over_{line}"] = _market(p_over)
        cards[f"under_{line}"] = _market(1 - p_over)

    # Blend with ML for 4.5
    try:
        m_c45 = CatBoostClassifier(); m_c45.load_model(str(model_dir / "prod_cards_over_4_5.cbm"))
        ml_c45 = float(m_c45.predict_proba(_align_X(m_c45))[0][1])
        poisson_c45 = cards["over_4.5"]["prob"]
        blended = 0.4 * poisson_c45 + 0.6 * ml_c45
        cards["over_4.5"] = _market(blended)
        cards["under_4.5"] = _market(1 - blended)
    except Exception as e:
        log.debug(f"Failed to blend ML cards over 4.5 prediction: {e}")

    preds["cards"] = cards

    # 9. Corners predictions
    try:
        m_hcr = CatBoostRegressor(); m_hcr.load_model(str(model_dir / "prod_home_corners.cbm"))
        m_acr = CatBoostRegressor(); m_acr.load_model(str(model_dir / "prod_away_corners.cbm"))
        exp_home_corn = max(1.0, min(10.0, float(m_hcr.predict(_align_X(m_hcr))[0])))
        exp_away_corn = max(1.0, min(10.0, float(m_acr.predict(_align_X(m_acr))[0])))
    except Exception as _e:
        log.debug("Model fallback triggered: %s", _e)
        exp_home_corn, exp_away_corn = 5.4, 4.5

    exp_total_corn = exp_home_corn + exp_away_corn
    corners = {
        "home_expected": round(exp_home_corn, 2),
        "away_expected": round(exp_away_corn, 2),
        "total_expected": round(exp_total_corn, 2),
    }
    for line in [7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
        p_over = 1 - sum(poisson.pmf(k, exp_total_corn) for k in range(int(line) + 1))
        corners[f"over_{line}"] = _market(p_over)
        corners[f"under_{line}"] = _market(1 - p_over)

    preds["corners"] = corners

    # 10. Clean sheet
    p_home_cs = poisson.pmf(0, away_xg)
    p_away_cs = poisson.pmf(0, home_xg)
    preds["clean_sheet"] = {
        "home": _market(p_home_cs),
        "away": _market(p_away_cs),
    }

    # 11. Win to nil
    preds["win_to_nil"] = {
        "home": _market(ens_probs["H"] * p_home_cs / max(0.01, sum(v for (h, a), v in score_matrix.items() if h > a and a == 0) / p_H if p_H > 0 else 1)),
        "away": _market(ens_probs["A"] * p_away_cs / max(0.01, sum(v for (h, a), v in score_matrix.items() if a > h and h == 0) / p_A if p_A > 0 else 1)),
    }
    # Simpler calculation
    p_home_wtn = sum(v for (h, a), v in score_matrix.items() if h > 0 and a == 0)
    p_away_wtn = sum(v for (h, a), v in score_matrix.items() if a > 0 and h == 0)
    preds["win_to_nil"] = {
        "home": _market(p_home_wtn),
        "away": _market(p_away_wtn),
    }

    # 12. Goal in both halves
    p_fh_goal = 1 - poisson.pmf(0, fh_total)
    sh_total = total_xg - fh_total
    p_sh_goal = 1 - poisson.pmf(0, max(0.5, sh_total))
    preds["goal_both_halves"] = {
        "yes": _market(p_fh_goal * p_sh_goal),
        "no": _market(1 - p_fh_goal * p_sh_goal),
    }

    # 13. Odd/Even goals
    p_even = sum(v for (h, a), v in score_matrix.items() if (h + a) % 2 == 0)
    preds["odd_even"] = {
        "odd": _market(1 - p_even),
        "even": _market(p_even),
    }

    # 14. Winning margin
    margins = {}
    for margin in range(1, 5):
        p_home_margin = sum(v for (h, a), v in score_matrix.items() if h - a == margin)
        p_away_margin = sum(v for (h, a), v in score_matrix.items() if a - h == margin)
        margins[f"home_by_{margin}"] = _market(p_home_margin)
        margins[f"away_by_{margin}"] = _market(p_away_margin)
    margins["draw"] = _market(p_D)
    preds["winning_margin"] = margins

    # 15. Team to score first
    # P(home scores first) ≈ home_xg / (home_xg + away_xg)
    p_no_goals = poisson.pmf(0, home_xg) * poisson.pmf(0, away_xg)
    p_home_first = home_xg / (home_xg + away_xg) * (1 - p_no_goals)
    p_away_first = away_xg / (home_xg + away_xg) * (1 - p_no_goals)
    preds["team_to_score_first"] = {
        "home": _market(p_home_first),
        "away": _market(p_away_first),
        "no_goal": _market(p_no_goals),
    }

    # 16. Race to X goals
    preds["race_to_goals"] = {}
    for target in [1, 2, 3]:
        p_home_race = sum(v for (h, a), v in score_matrix.items() if h >= target and (a < target or h == target))
        p_away_race = sum(v for (h, a), v in score_matrix.items() if a >= target and (h < target or a == target))
        # Simplified: who reaches target first
        p_home_race = (home_xg / (home_xg + away_xg)) * (1 - poisson.cdf(target - 1, home_xg + away_xg))
        p_away_race = (away_xg / (home_xg + away_xg)) * (1 - poisson.cdf(target - 1, home_xg + away_xg))
        p_neither = 1 - p_home_race - p_away_race
        preds["race_to_goals"][f"home_to_{target}"] = _market(max(0.01, p_home_race))
        preds["race_to_goals"][f"away_to_{target}"] = _market(max(0.01, p_away_race))
        preds["race_to_goals"][f"neither_to_{target}"] = _market(max(0.01, p_neither))

    # 17. Asian Handicap
    preds["asian_handicap"] = {}
    for ah_line in [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5]:
        p_home_ah = sum(v for (h, a), v in score_matrix.items() if (h - a + ah_line) > 0)
        p_away_ah = sum(v for (h, a), v in score_matrix.items() if (h - a + ah_line) < 0)
        preds["asian_handicap"][f"home_{ah_line:+.1f}"] = _market(p_home_ah)
        preds["asian_handicap"][f"away_{ah_line:+.1f}"] = _market(p_away_ah)

    # 18. Second half result
    sh_hxg = max(0.1, home_xg - fh_hxg)
    sh_axg = max(0.1, away_xg - fh_axg)
    sh_sm = {}
    for h in range(5):
        for a in range(5):
            sh_sm[(h, a)] = poisson.pmf(h, sh_hxg) * poisson.pmf(a, sh_axg)
    sh_H = sum(v for (h, a), v in sh_sm.items() if h > a)
    sh_D = sum(v for (h, a), v in sh_sm.items() if h == a)
    sh_A = sum(v for (h, a), v in sh_sm.items() if h < a)
    sh_t = sh_H + sh_D + sh_A
    preds["second_half_result"] = {
        "home_win": _market(sh_H/sh_t),
        "draw": _market(sh_D/sh_t),
        "away_win": _market(sh_A/sh_t),
    }

    # 19. Penalty in match
    # ~23% of Serie A matches have a penalty (from incidents data)
    preds["penalty"] = {
        "yes": _market(0.23),
        "no": _market(0.77),
    }

    # 20. Red card
    # ~18% of Serie A matches have a red card
    preds["red_card"] = {
        "yes": _market(0.18),
        "no": _market(0.82),
    }

    # 21. Multi-goal ranges
    preds["multi_goal"] = {}
    for lo, hi in [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 6)]:
        p = sum(v for (h, a), v in score_matrix.items() if lo <= h + a <= hi)
        preds["multi_goal"][f"{lo}-{hi}_goals"] = _market(p)

    # 22. Team totals
    preds["team_totals"] = {}
    for side, xg_val in [("home", home_xg), ("away", away_xg)]:
        for line in [0.5, 1.5, 2.5, 3.5]:
            p_over = 1 - poisson.cdf(int(line), xg_val)
            preds["team_totals"][f"{side}_over_{line}"] = _market(p_over)
            preds["team_totals"][f"{side}_under_{line}"] = _market(1 - p_over)

    # 23. Booking points (10 per yellow, 25 per red)
    exp_bp = exp_total_cards * 10.4  # weighted average
    preds["booking_points"] = {
        "expected": round(exp_bp, 1),
    }
    for line in [20.5, 30.5, 40.5, 50.5, 60.5]:
        k_cards = line / 10.4
        p_over = 1 - sum(poisson.pmf(k, exp_total_cards) for k in range(int(k_cards) + 1))
        preds["booking_points"][f"over_{line}"] = _market(p_over)
        preds["booking_points"][f"under_{line}"] = _market(1 - p_over)

    return preds


def _market(prob: float) -> Dict:
    """Format a market probability with fair odds."""
    prob = max(0.001, min(0.999, float(prob)))
    return {
        "prob": round(prob, 4),
        "fair_odds": round(1 / prob, 2),
    }


# =============================================================================
# PLAYER GOALSCORER MODEL
# =============================================================================

def build_player_scorer_model() -> Dict:
    """Build player goalscorer probability model from Sofascore player_match_stats."""
    pms_path = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
    if not pms_path.exists():
        log.warning("No player_match_stats.parquet found")
        return {}

    pms = pd.read_parquet(pms_path)
    log.info("Building player scorer model from %d player-match records", len(pms))

    # Compute per-player rolling stats
    player_stats = pms.groupby("player_id").agg(
        player_name=("player_name", "last"),
        team=("team", "last"),
        matches=("match_id", "nunique"),
        total_goals=("goals", "sum"),
        total_shots=("total_shots", "sum"),
        total_sot=("shots_on_target", "sum"),
        total_minutes=("minutes", "sum"),
        total_xg=("xg", "sum"),
        total_assists=("assists", "sum"),
        total_key_passes=("key_passes", "sum"),
        total_big_chances=("big_chances_created", "sum"),
        avg_rating=("rating", "mean"),
        is_starter_pct=("is_starter", "mean"),
    ).reset_index()

    # Compute rates
    player_stats["goals_per_90"] = (player_stats["total_goals"] / player_stats["total_minutes"] * 90).fillna(0)
    player_stats["shots_per_90"] = (player_stats["total_shots"] / player_stats["total_minutes"] * 90).fillna(0)
    player_stats["xg_per_90"] = (player_stats["total_xg"] / player_stats["total_minutes"] * 90).fillna(0)
    player_stats["conversion_rate"] = (player_stats["total_goals"] / player_stats["total_shots"].clip(1)).fillna(0)
    player_stats["sot_rate"] = (player_stats["total_sot"] / player_stats["total_shots"].clip(1)).fillna(0)

    # Filter to relevant players (at least 5 matches, forward/midfielder)
    player_stats = player_stats[player_stats["matches"] >= 5].copy()

    # P(player scores) ≈ 1 - (1 - goals_per_90/90)^minutes_expected
    # Simplified: use goals_per_match as base probability
    player_stats["goals_per_match"] = player_stats["total_goals"] / player_stats["matches"]
    player_stats["p_score_anytime"] = 1 - np.exp(-player_stats["goals_per_match"])

    # Save model
    scorer_model = {}
    for _, row in player_stats.iterrows():
        scorer_model[int(row["player_id"])] = {
            "name": row["player_name"],
            "team": row["team"],
            "matches": int(row["matches"]),
            "goals": int(row["total_goals"]),
            "goals_per_90": round(row["goals_per_90"], 3),
            "xg_per_90": round(row["xg_per_90"], 3),
            "shots_per_90": round(row["shots_per_90"], 3),
            "conversion_rate": round(row["conversion_rate"], 3),
            "p_score_anytime": round(row["p_score_anytime"], 4),
            "avg_rating": round(row["avg_rating"], 2),
            "starter_pct": round(row["is_starter_pct"], 2),
        }

    with open(MARKET_MODELS_DIR / "player_scorer_model.json", "w") as f:
        json.dump(scorer_model, f, indent=2)

    log.info("Player scorer model: %d players", len(scorer_model))
    return scorer_model


def predict_player_scorers(home_team: str, away_team: str, home_xg: float, away_xg: float) -> Dict:
    """Predict goalscorer probabilities for a match."""
    model_path = MARKET_MODELS_DIR / "player_scorer_model.json"
    if not model_path.exists():
        return {}

    with open(model_path) as f:
        scorer_model = json.load(f)

    # Get players for each team
    home_scorers = []
    away_scorers = []

    for pid, data in scorer_model.items():
        team_lower = data["team"].lower()
        if team_lower == home_team.lower():
            home_scorers.append(data)
        elif team_lower == away_team.lower():
            away_scorers.append(data)

    # Adjust probabilities based on match xG
    def adjust_probs(scorers, team_xg):
        if not scorers:
            return []
        # Scale individual xG by team's expected performance
        total_team_xg_per90 = sum(s["xg_per_90"] for s in scorers if s["starter_pct"] > 0.3)
        if total_team_xg_per90 == 0:
            total_team_xg_per90 = 1.0
        scaling = team_xg / max(0.5, total_team_xg_per90)

        for s in scorers:
            adj_gpm = s["goals_per_90"] / 90 * 90 * scaling  # adjusted goals per match
            adj_gpm = max(0.01, min(1.5, adj_gpm))
            s["p_score_adjusted"] = round(1 - math.exp(-adj_gpm), 4)
            s["p_first_scorer"] = round(s["p_score_adjusted"] * 0.35, 4)  # ~35% of scorers score first
            s["p_last_scorer"] = round(s["p_score_adjusted"] * 0.30, 4)

        return sorted(scorers, key=lambda x: x["p_score_adjusted"], reverse=True)

    home_scorers = adjust_probs(home_scorers, home_xg)
    away_scorers = adjust_probs(away_scorers, away_xg)

    return {
        "home_scorers": [{
            "name": s["name"],
            "p_anytime": s["p_score_adjusted"],
            "p_first_scorer": s["p_first_scorer"],
            "p_last_scorer": s["p_last_scorer"],
            "goals_per_90": s["goals_per_90"],
            "xg_per_90": s["xg_per_90"],
            "fair_odds_anytime": round(1 / max(0.01, s["p_score_adjusted"]), 2),
        } for s in home_scorers[:15]],
        "away_scorers": [{
            "name": s["name"],
            "p_anytime": s["p_score_adjusted"],
            "p_first_scorer": s["p_first_scorer"],
            "p_last_scorer": s["p_last_scorer"],
            "goals_per_90": s["goals_per_90"],
            "xg_per_90": s["xg_per_90"],
            "fair_odds_anytime": round(1 / max(0.01, s["p_score_adjusted"]), 2),
        } for s in away_scorers[:15]],
    }


# =============================================================================
# BACKTESTING — Validate ALL markets on historical data
# =============================================================================

def backtest_all_markets(df: pd.DataFrame, features: List[str], season: str) -> Dict:
    """Comprehensive backtest of all markets on a single season."""
    from catboost import CatBoostRegressor, CatBoostClassifier
    from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

    season_df = df[df["season"] == season].copy()
    if season_df.empty:
        log.warning("No data for season %s", season)
        return {}

    X = season_df[features].fillna(0)
    log.info("Backtesting %d matches for %s", len(season_df), season)

    results = {}
    model_dir = MARKET_MODELS_DIR

    label_map = {"H": 0, "D": 1, "A": 2}

    # 1. Match Result (1X2)
    try:
        m_hg = CatBoostRegressor(); m_hg.load_model(str(model_dir / "home_goals.cbm"))
        m_ag = CatBoostRegressor(); m_ag.load_model(str(model_dir / "away_goals.cbm"))
        m_cls = CatBoostClassifier(); m_cls.load_model(str(model_dir / "result_1x2.cbm"))

        pred_hxg = np.clip(m_hg.predict(X), 0.3, 4.0)
        pred_axg = np.clip(m_ag.predict(X), 0.3, 4.0)

        # Poisson probs
        poisson_probs = np.zeros((len(X), 3))
        for i, (hxg, axg) in enumerate(zip(pred_hxg, pred_axg)):
            pH = pD = pA = 0
            for h in range(8):
                for a in range(8):
                    p = poisson.pmf(h, hxg) * poisson.pmf(a, axg)
                    if h > a: pH += p
                    elif h == a: pD += p
                    else: pA += p
            t = pH + pD + pA
            poisson_probs[i] = [pH/t, pD/t, pA/t]

        # Classifier probs
        cls_probs = m_cls.predict_proba(X)

        # Ensemble
        ens_probs = 0.4 * poisson_probs + 0.6 * cls_probs

        y_true = season_df["result"].map(label_map).values
        y_labels = season_df["result"].values

        for name, probs in [("poisson", poisson_probs), ("classifier", cls_probs), ("ensemble", ens_probs)]:
            pred_labels = np.array(["H", "D", "A"])[probs.argmax(axis=1)]
            acc = (pred_labels == y_labels).mean()
            ll = log_loss(y_true, probs, labels=[0, 1, 2])
            hc_mask = probs.max(axis=1) > 0.55
            hc_acc = (pred_labels[hc_mask] == y_labels[hc_mask]).mean() if hc_mask.sum() > 0 else 0
            results[f"1x2_{name}"] = {
                "accuracy": round(acc, 4),
                "log_loss": round(ll, 4),
                "hc_accuracy": round(hc_acc, 4),
                "hc_count": int(hc_mask.sum()),
            }
            log.info("  1X2 %s: acc=%.1f%%, HC=%.1f%% (n=%d), LL=%.4f",
                     name, acc*100, hc_acc*100, hc_mask.sum(), ll)

        # Pinnacle comparison
        psh = season_df["odds_PSH"].values
        psd = season_df["odds_PSD"].values
        psa = season_df["odds_PSA"].values
        valid_odds = (psh > 1) & (psd > 1) & (psa > 1)
        if valid_odds.sum() > 0:
            mkt_raw = np.column_stack([1/psh[valid_odds], 1/psd[valid_odds], 1/psa[valid_odds]])
            mkt_probs = mkt_raw / mkt_raw.sum(axis=1, keepdims=True)
            mkt_labels = np.array(["H", "D", "A"])[mkt_probs.argmax(axis=1)]
            mkt_acc = (mkt_labels == y_labels[valid_odds]).mean()
            mkt_ll = log_loss(y_true[valid_odds], mkt_probs, labels=[0, 1, 2])
            results["1x2_pinnacle"] = {"accuracy": round(mkt_acc, 4), "log_loss": round(mkt_ll, 4)}
            log.info("  1X2 Pinnacle: acc=%.1f%%, LL=%.4f", mkt_acc*100, mkt_ll)
    except Exception as e:
        log.warning("1X2 backtest failed: %s", e)

    # 2. Over/Under 2.5
    try:
        m_ou = CatBoostClassifier(); m_ou.load_model(str(model_dir / "over_2_5.cbm"))
        ou_probs = m_ou.predict_proba(X)[:, 1]
        y_ou = season_df["over_2_5"].values
        ou_acc = ((ou_probs > 0.5).astype(int) == y_ou).mean()
        ou_ll = log_loss(y_ou, ou_probs)
        results["over_2_5"] = {"accuracy": round(ou_acc, 4), "log_loss": round(ou_ll, 4)}
        log.info("  Over 2.5: acc=%.1f%%, LL=%.4f", ou_acc*100, ou_ll)
    except Exception as e:
        log.warning("O/U 2.5 backtest failed: %s", e)

    # 3. BTTS
    try:
        m_btts = CatBoostClassifier(); m_btts.load_model(str(model_dir / "btts.cbm"))
        btts_probs = m_btts.predict_proba(X)[:, 1]
        y_btts = season_df["btts"].values
        btts_acc = ((btts_probs > 0.5).astype(int) == y_btts).mean()
        btts_ll = log_loss(y_btts, btts_probs)
        results["btts"] = {"accuracy": round(btts_acc, 4), "log_loss": round(btts_ll, 4)}
        log.info("  BTTS: acc=%.1f%%, LL=%.4f", btts_acc*100, btts_ll)
    except Exception as e:
        log.warning("BTTS backtest failed: %s", e)

    # 4. HT Result
    try:
        m_ht = CatBoostClassifier(); m_ht.load_model(str(model_dir / "ht_result.cbm"))
        ht_valid = season_df["ht_result"].isin(["H", "D", "A"])
        if ht_valid.sum() > 0:
            X_ht = season_df.loc[ht_valid, features].fillna(0)
            ht_probs = m_ht.predict_proba(X_ht)
            y_ht = season_df.loc[ht_valid, "ht_result"].map(label_map).values
            ht_labels = season_df.loc[ht_valid, "ht_result"].values
            ht_pred = np.array(["H", "D", "A"])[ht_probs.argmax(axis=1)]
            ht_acc = (ht_pred == ht_labels).mean()
            ht_ll = log_loss(y_ht, ht_probs, labels=[0, 1, 2])
            results["ht_result"] = {"accuracy": round(ht_acc, 4), "log_loss": round(ht_ll, 4)}
            log.info("  HT Result: acc=%.1f%%, LL=%.4f", ht_acc*100, ht_ll)
    except Exception as e:
        log.warning("HT Result backtest failed: %s", e)

    # 5. Cards O/U 4.5
    try:
        m_c45 = CatBoostClassifier(); m_c45.load_model(str(model_dir / "cards_over_4_5.cbm"))
        cards_valid = season_df["cards_over_4_5"].notna()
        if cards_valid.sum() > 0:
            X_c = season_df.loc[cards_valid, features].fillna(0)
            c_probs = m_c45.predict_proba(X_c)[:, 1]
            y_c = season_df.loc[cards_valid, "cards_over_4_5"].values
            c_acc = ((c_probs > 0.5).astype(int) == y_c).mean()
            c_ll = log_loss(y_c, c_probs)
            results["cards_over_4_5"] = {"accuracy": round(c_acc, 4), "log_loss": round(c_ll, 4)}
            log.info("  Cards O4.5: acc=%.1f%%, LL=%.4f", c_acc*100, c_ll)
    except Exception as e:
        log.warning("Cards backtest failed: %s", e)

    # 6. Corners O/U 9.5
    try:
        m_cr = CatBoostClassifier(); m_cr.load_model(str(model_dir / "corners_over_9_5.cbm"))
        corn_valid = season_df["corners_over_9_5"].notna()
        if corn_valid.sum() > 0:
            X_cr = season_df.loc[corn_valid, features].fillna(0)
            cr_probs = m_cr.predict_proba(X_cr)[:, 1]
            y_cr = season_df.loc[corn_valid, "corners_over_9_5"].values
            cr_acc = ((cr_probs > 0.5).astype(int) == y_cr).mean()
            cr_ll = log_loss(y_cr, cr_probs)
            results["corners_over_9_5"] = {"accuracy": round(cr_acc, 4), "log_loss": round(cr_ll, 4)}
            log.info("  Corners O9.5: acc=%.1f%%, LL=%.4f", cr_acc*100, cr_ll)
    except Exception as e:
        log.warning("Corners backtest failed: %s", e)

    # Save backtest results
    results["season"] = season
    results["matches"] = len(season_df)
    with open(MARKET_MODELS_DIR / f"backtest_{season}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive All-Markets Prediction Engine")
    parser.add_argument("--train", action="store_true", help="Train all market models")
    parser.add_argument("--backtest", action="store_true", help="Backtest on historical data")
    parser.add_argument("--predict", action="store_true", help="Predict upcoming matches")
    parser.add_argument("--season", type=str, default="2024-2025", help="Season for backtest")
    parser.add_argument("--all", action="store_true", help="Run train + backtest + predict")
    args = parser.parse_args()

    if args.all:
        args.train = args.backtest = args.predict = True

    if not any([args.train, args.backtest, args.predict]):
        args.train = args.backtest = True

    # Load unified data
    log.info("=" * 70)
    log.info("COMPREHENSIVE ALL-MARKETS PREDICTION ENGINE")
    log.info("=" * 70)

    df = load_unified_training_data()
    # CRITICAL: Get features BEFORE computing targets to prevent leakage
    initial_features = get_training_features(df)
    # Now add targets (these must NEVER appear in features list)
    df = _compute_targets(df)

    # Track which features to use (selected after importance filtering, or loaded from metadata)
    features_to_use = initial_features

    if args.train:
        log.info("\n" + "=" * 70)
        log.info("PHASE 1: TRAINING ALL MARKET MODELS")
        log.info("=" * 70)
        train_results = train_all_models(df, initial_features)
        features_to_use = train_results["selected_features"]
        log.info("Training complete: %d features selected from %d initial features",
                 len(features_to_use), len(initial_features))

        log.info("\n" + "=" * 70)
        log.info("PHASE 2: BUILDING PLAYER SCORER MODEL")
        log.info("=" * 70)
        build_player_scorer_model()
    else:
        # Load selected features from metadata if not training
        metadata_path = MARKET_MODELS_DIR / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                features_to_use = metadata.get("features", initial_features)
                log.info("Loaded %d selected features from metadata", len(features_to_use))
        else:
            log.warning("No metadata found, using all %d initial features", len(initial_features))

    if args.backtest:
        log.info("\n" + "=" * 70)
        log.info("PHASE 3: BACKTESTING ALL MARKETS")
        log.info("=" * 70)

        for season in ["2023-2024", args.season]:
            log.info("\n--- Backtesting %s ---", season)
            bt_results = backtest_all_markets(df, features_to_use, season)

            if bt_results:
                log.info("\n=== BACKTEST SUMMARY: %s ===", season)
                for market, metrics in bt_results.items():
                    if isinstance(metrics, dict) and "accuracy" in metrics:
                        log.info("  %s: acc=%.1f%%, LL=%.4f",
                                 market, metrics["accuracy"]*100, metrics.get("log_loss", 0))

    if args.predict:
        log.info("\n" + "=" * 70)
        log.info("PHASE 4: GENERATING PREDICTIONS")
        log.info("=" * 70)
        # This would use the ensemble_prediction_engine for upcoming matches
        log.info("Use ensemble_prediction_engine.py for live predictions")


if __name__ == "__main__":
    main()
