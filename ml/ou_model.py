"""Over/Under & BTTS Prediction — 3-model ensemble for goal markets.

First-class prediction system for:
  - Over/Under 1.5, 2.5, 3.5 (binary: did total goals exceed the line?)
  - BTTS (binary: did both teams score?)

Architecture mirrors the 1X2 ensemble: XGBoost + LightGBM + CatBoost with
walk-forward CV, time-decay sample weights, and calibration.

Key innovation: Poisson analytical probabilities are fed AS FEATURES so the
tree models learn when Poisson is right and when to deviate.

Usage:
    from ml.ou_model import train_goals_model, predict_ou, predict_btts
    results = train_goals_model(league="serie_a")
    probs = predict_ou(features_df, league="serie_a", line=2.5)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss

from config.settings import MODELS_DIR
from ml.config import ValidationConfig
from ml.data import DataLoader, TimeSeriesSplitter
from ml.tuning import _compute_sample_weights
from storage.paths import features_path

log = logging.getLogger(__name__)

# Goal-relevant features prioritized for O/U and BTTS models.
# Poisson priors (poisson_over_*) are included as features for the ML to correct.
GOAL_FEATURE_PRIORITIES = [
    # Poisson analytical priors (ML learns corrections)
    "poisson_over_1_5", "poisson_over_2_5", "poisson_over_3_5", "poisson_btts",
    "total_xg_expected", "poisson_home_xg", "poisson_away_xg",
    # Goal-specific features
    "combined_goals_per_game", "combined_goals_conceded",
    "attack_mismatch_score", "defense_mismatch_score",
    "scoring_variance", "pressing_intensity_sum",
    "clean_sheet_matchup", "h2h_goals_vs_avg", "derby_goal_suppression",
    # BTTS features
    "btts_probability_naive", "home_scoring_rate_5", "away_scoring_rate_5",
    "home_conceding_rate_5", "away_conceding_rate_5",
    # Core attacking/defensive stats
    "home_attack_strength", "away_attack_strength", "attack_strength_diff",
    "home_defense_strength", "away_defense_strength", "defense_strength_diff",
    "home_roll_5_goals_scored", "away_roll_5_goals_scored",
    "home_roll_10_goals_scored", "away_roll_10_goals_scored",
    "home_roll_5_goals_conceded", "away_roll_5_goals_conceded",
    "home_roll_10_goals_conceded", "away_roll_10_goals_conceded",
    # xG data
    "us_xg_diff", "home_roll_5_xg_for", "away_roll_5_xg_for",
    # Shots (proxy for attacking intent)
    "home_roll_10_shots_on_target", "away_roll_10_shots_on_target",
    "home_roll_5_shots_on_target", "away_roll_5_shots_on_target",
    # Match context
    "elo_diff", "matchup_competitiveness",
    # H2H
    "h2h_goals_avg", "h2h_recent_3_total_goals",
    # Form
    "home_form_points_5", "away_form_points_5",
    "home_roll_5_goals_scored_std", "away_roll_5_goals_scored_std",
    "home_goals_scored_trend", "away_goals_scored_trend",
    # Pressing
    "home_ppda", "away_ppda",
    # Clean sheets
    "home_roll_5_clean_sheet", "away_roll_5_clean_sheet",
    # Rest/congestion
    "rest_advantage",
    # Venue
    "home_stadium_capacity",
]

# Targets we can train
TARGETS = {
    "ou_1.5": lambda total: (total > 1.5).astype(int),
    "ou_2.5": lambda total: (total > 2.5).astype(int),
    "ou_3.5": lambda total: (total > 3.5).astype(int),
    "btts": None,  # Handled separately (needs both scores, not total)
}

MODEL_TYPES = ["xgboost", "lightgbm", "catboost"]


def _get_model(model_type: str, params: dict = None):
    """Create a model instance."""
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        default = {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42,
                    "eval_metric": "logloss", "use_label_encoder": False}
        default.update(params or {})
        return XGBClassifier(**default)
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        default = {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8, "random_seed": 42,
                    "verbose": -1}
        default.update(params or {})
        return LGBMClassifier(**default)
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        default = {"iterations": 500, "depth": 5, "learning_rate": 0.05,
                    "l2_leaf_reg": 3.0, "random_seed": 42, "verbose": 0,
                    "auto_class_weights": "Balanced"}
        default.update(params or {})
        return CatBoostClassifier(**default)
    raise ValueError(f"Unknown model: {model_type}")


def train_goals_model(
    league: str = "serie_a",
    targets: list[str] | None = None,
) -> Dict:
    """Train O/U and BTTS models for a league.

    Trains a 3-model ensemble (XGB + LGB + CB) per target.
    Walk-forward CV with time-decay sample weights.

    Args:
        league: "serie_a" or "premier_league"
        targets: List of targets to train. Default: all (ou_1.5, ou_2.5, ou_3.5, btts)

    Returns:
        Dict with per-target CV metrics, model paths, feature importances
    """
    if targets is None:
        targets = list(TARGETS.keys())

    log.info("Training goals models for %s: %s", league, targets)

    # Load features
    fp = str(features_path(league=league))
    loader = DataLoader(fp)
    X_full, _, feature_names = loader.get_universal_dataset(league=league)

    # Get scores for target computation
    raw_df = loader.df
    mask = raw_df["result"].isin(["H", "D", "A"])
    val_cfg = ValidationConfig()
    if val_cfg.min_train_season and "season" in raw_df.columns:
        mask = mask & (raw_df["season"] >= val_cfg.min_train_season)
    if league and "league" in raw_df.columns:
        mask = mask & (raw_df["league"] == league)

    scores = raw_df.loc[mask, ["home_score", "away_score"]].reset_index(drop=True)
    total_goals = scores["home_score"].astype(float) + scores["away_score"].astype(float)
    btts = ((scores["home_score"].astype(float) > 0) & (scores["away_score"].astype(float) > 0)).astype(int)

    # Select goal-relevant features
    available = [f for f in GOAL_FEATURE_PRIORITIES if f in feature_names]
    # Add extras with goal-related keywords
    extra = [f for f in feature_names
             if any(kw in f.lower() for kw in ["goal", "xg", "shot", "attack", "defen", "scor"])
             and f not in available]
    available.extend(extra[:20])
    log.info("Goal features: %d priority + %d extra = %d total",
             len(GOAL_FEATURE_PRIORITIES), len(extra[:20]), len(available))

    X = X_full[available].copy()
    seasons = X_full["_season"] if "_season" in X_full.columns else None

    # Walk-forward splits
    splitter = TimeSeriesSplitter(val_cfg)
    splits = splitter.generate_splits(seasons) if seasons is not None else []

    all_results = {}
    save_dir = MODELS_DIR / league / "ou_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    for target_name in targets:
        log.info("--- Training %s ---", target_name)

        # Create binary target
        if target_name == "btts":
            y = btts
        else:
            line = float(target_name.split("_")[1].replace("_", "."))
            y = (total_goals > line).astype(int)

        over_rate = y.mean()
        log.info("Target %s: %d/%d positive (%.1f%%)", target_name, y.sum(), len(y), over_rate * 100)

        # Walk-forward CV with 3-model ensemble
        fold_metrics = []
        oof_probs = np.zeros(len(y))
        oof_mask = np.zeros(len(y), dtype=bool)

        for fold_idx, (train_seasons, test_seasons) in enumerate(splits):
            train_mask = seasons.isin(train_seasons)
            test_mask = seasons.isin(test_seasons)

            X_train = X[train_mask].values
            y_train = y[train_mask].values
            X_test = X[test_mask].values
            y_test = y[test_mask].values

            # Sample weights with time decay
            train_seasons_s = seasons[train_mask]
            sw = _compute_sample_weights(
                pd.Series(["H"] * len(y_train)),  # Dummy — class weights from auto_class_weights
                seasons=train_seasons_s,
            )
            # For binary, just use time decay (no class rebalancing via sample_weight)
            # CatBoost auto_class_weights handles class imbalance

            # Train 3 models and blend
            fold_probs = []
            for mt in MODEL_TYPES:
                model = _get_model(mt)
                try:
                    if mt == "catboost":
                        # CatBoost with early stopping
                        split_pt = int(len(X_train) * 0.85)
                        model.fit(
                            X_train[:split_pt], y_train[:split_pt],
                            eval_set=(X_train[split_pt:], y_train[split_pt:]),
                            early_stopping_rounds=50, verbose=0,
                        )
                    else:
                        model.fit(X_train, y_train, sample_weight=sw)
                    proba = model.predict_proba(X_test)[:, 1]
                    fold_probs.append(proba)
                except Exception as e:
                    log.warning("Fold %d %s failed: %s", fold_idx, mt, e)

            if not fold_probs:
                continue

            # Equal-weight blend
            blended = np.mean(fold_probs, axis=0)
            oof_probs[test_mask.values] = blended
            oof_mask[test_mask.values] = True

            # Fold metrics
            y_pred = (blended > 0.5).astype(int)
            fold_metrics.append({
                "fold": fold_idx,
                "test": ", ".join(test_seasons),
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "log_loss": round(log_loss(y_test, blended), 4),
                "brier": round(brier_score_loss(y_test, blended), 4),
                "f1": round(f1_score(y_test, y_pred), 4),
                "n_test": len(y_test),
                "over_rate_actual": round(y_test.mean(), 4),
                "over_rate_predicted": round(blended.mean(), 4),
            })

            log.info("Fold %d [%s]: acc=%.3f ll=%.4f brier=%.4f",
                     fold_idx, test_seasons, fold_metrics[-1]["accuracy"],
                     fold_metrics[-1]["log_loss"], fold_metrics[-1]["brier"])

        # Train final models on ALL data
        final_models = {}
        for mt in MODEL_TYPES:
            model = _get_model(mt)
            try:
                fit_kw = {"verbose": 0} if mt == "catboost" else {}
                model.fit(X.values, y.values, **fit_kw)
                final_models[mt] = model
            except Exception as e:
                log.warning("Final %s training failed: %s", mt, e)

        # Save models
        for mt, model in final_models.items():
            if mt == "catboost":
                model.save_model(str(save_dir / f"{target_name}_catboost.cbm"))
            elif mt == "xgboost":
                model.save_model(str(save_dir / f"{target_name}_xgboost.json"))
            elif mt == "lightgbm":
                model.booster_.save_model(str(save_dir / f"{target_name}_lightgbm.txt"))

        # Compute averages
        avg_metrics = {
            "accuracy": round(np.mean([m["accuracy"] for m in fold_metrics]), 4),
            "log_loss": round(np.mean([m["log_loss"] for m in fold_metrics]), 4),
            "brier": round(np.mean([m["brier"] for m in fold_metrics]), 4),
            "f1": round(np.mean([m["f1"] for m in fold_metrics]), 4),
        } if fold_metrics else {}

        # Feature importance (from CatBoost)
        feat_imp = {}
        if "catboost" in final_models:
            imp = final_models["catboost"].feature_importances_
            feat_imp = dict(zip(available, [round(float(x), 4) for x in imp]))

        all_results[target_name] = {
            "cv_metrics": fold_metrics,
            "avg": avg_metrics,
            "n_features": len(available),
            "features": available,
            "feature_importance": feat_imp,
            "base_rate": round(over_rate, 4),
        }

        log.info("%s avg: acc=%.3f ll=%.4f brier=%.4f (base=%.1f%%)",
                 target_name, avg_metrics.get("accuracy", 0),
                 avg_metrics.get("log_loss", 0), avg_metrics.get("brier", 0),
                 over_rate * 100)

    # Save combined report
    report = {
        "league": league,
        "targets": all_results,
        "features_used": available,
        "n_features": len(available),
        "model_types": MODEL_TYPES,
    }
    report_path = save_dir / "goals_model_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Goals model report saved to %s", report_path)

    return report


def predict_ou(
    features_df: pd.DataFrame,
    league: str = "serie_a",
    line: float = 2.5,
) -> float | None:
    """Predict P(Over line) for a single match using 3-model ensemble.

    Returns probability of Over, or None if models not available.
    """
    target_name = f"ou_{str(line).replace('.', '_')}"
    save_dir = MODELS_DIR / league / "ou_model"

    probs = []
    for mt in MODEL_TYPES:
        if mt == "catboost":
            from catboost import CatBoostClassifier
            path = save_dir / f"{target_name}_catboost.cbm"
            if not path.exists():
                continue
            model = CatBoostClassifier()
            model.load_model(str(path))
        elif mt == "xgboost":
            from xgboost import XGBClassifier
            path = save_dir / f"{target_name}_xgboost.json"
            if not path.exists():
                continue
            model = XGBClassifier()
            model.load_model(str(path))
        elif mt == "lightgbm":
            from lightgbm import LGBMClassifier, Booster
            path = save_dir / f"{target_name}_lightgbm.txt"
            if not path.exists():
                continue
            booster = Booster(model_file=str(path))
            # LightGBM raw predict returns raw score, need sigmoid for probability
            raw = booster.predict(features_df.values)
            prob = 1 / (1 + np.exp(-raw))
            probs.append(float(prob[0]) if len(prob.shape) == 1 else float(prob[0, 1]))
            continue

        try:
            report_path = save_dir / "goals_model_report.json"
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                feat_names = report.get("features_used", [])
            else:
                feat_names = GOAL_FEATURE_PRIORITIES

            X = pd.DataFrame(index=[0])
            for f in feat_names:
                X[f] = features_df[f].values[0] if f in features_df.columns else 0.0

            prob = model.predict_proba(X.values)[:, 1]
            probs.append(float(prob[0]))
        except Exception as e:
            log.debug("Predict %s %s failed: %s", mt, target_name, e)

    if not probs:
        return None

    return round(float(np.mean(probs)), 4)


def predict_btts(
    features_df: pd.DataFrame,
    league: str = "serie_a",
) -> float | None:
    """Predict P(BTTS Yes) using 3-model ensemble."""
    return predict_ou(features_df, league=league, line=0)  # Reuse with btts target


if __name__ == "__main__":
    import sys
    league = sys.argv[1] if len(sys.argv) > 1 else "serie_a"

    results = train_goals_model(league=league)
    print(f"\n{'='*60}")
    print(f"GOALS MODEL RESULTS — {league}")
    print(f"{'='*60}")
    for target, data in results.get("targets", {}).items():
        avg = data.get("avg", {})
        base = data.get("base_rate", 0)
        print(f"  {target}: acc={avg.get('accuracy',0):.3f} (base={base:.1%}), "
              f"brier={avg.get('brier',0):.4f}, ll={avg.get('log_loss',0):.4f}")
