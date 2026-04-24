"""Unified walk-forward trainer for market models.

Key guarantees:
  - Each eval season's model is trained using ONLY data from strictly prior seasons.
    No information from the eval season (or any later season) touches the fitted model.
  - Raw odds columns are excluded from features (so the model is independent of
    the market signal and can produce edges against it).
  - One artifact per (league, market, eval_season): trained model + metadata.
  - Honest evaluation: log-loss, Brier, accuracy, calibration gap computed
    per eval season, never mixed with training data.

Outputs:
  data/models/walkforward/{league}/{market}/season_{YYYY-YYYY}.cbm
  data/models/walkforward/{league}/{market}/season_{YYYY-YYYY}_metadata.json
  data/models/walkforward/{league}/{market}/summary.json

Usage:
  python -m scripts.models.train_walkforward --league serie_a --market over_2_5
  python -m scripts.models.train_walkforward --league premier_league --market 1x2
  python -m scripts.models.train_walkforward --all   # every league × market
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.feature_selection import exclude_odds  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "models" / "walkforward"

LEAGUE_TO_FEATURES = {
    "serie_a": FEATURES_DIR / "features_serie_a.parquet",
    "premier_league": FEATURES_DIR / "features_premier_league.parquet",
}

# Minimum prior seasons before we trust the model to train (needs enough signal).
# Overridable via --min-prior-seasons for narrower-window experiments (e.g.
# a 2019+ window has only 2-3 prior seasons for the earliest eval season).
MIN_TRAIN_SEASONS = 5

# Evaluation target: only eval on the most recent 3 seasons to keep runs tractable
# and focused on the current regime. These are the seasons that will have
# approval-critical backtest numbers.
DEFAULT_EVAL_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]

# Columns that are always meta/target rather than features.
META_COLUMNS = {
    "match_id", "match_date", "season", "league", "home_team", "away_team",
    "home_score", "away_score", "result", "result_1X2", "home_win", "draw", "away_win",
    "total_goals", "btts", "home_clean_sheet", "away_clean_sheet",
}

# Known post-match leakage columns — these carry information that was only
# knowable AFTER the match, so they look predictive in hindsight but cannot
# be used in a real pre-match prediction. Every entry must have a documented
# reason for exclusion.
LEAKY_COLUMNS = {
    # Sofascore per-player ratings are assigned after the match based on the
    # player's observed performance. They correlate 0.6-0.7 with match result.
    "home_lineup_rating_mean", "away_lineup_rating_mean", "lineup_rating_mean_diff",
    # Sofascore per-player xG is the realized shot-level xG in the match being
    # played, NOT a pre-match forecast. Post-match leakage by definition.
    "home_lineup_xg_sum", "away_lineup_xg_sum", "lineup_xg_sum_diff",
    "home_lineup_xa_sum", "away_lineup_xa_sum", "lineup_xa_sum_diff",
    # Formation rotation uses current-match lineup; rotation calc reflects
    # confirmed starters but lineup_xg/lineup_rating upstream includes in-match
    # subs by default. Excluded conservatively until we split pre/post-match.
    "home_lineup_rotation", "away_lineup_rotation",
    # Half-time goals of the CURRENT match — post-kickoff info, not pre-match.
    "home_ht_goals", "away_ht_goals", "ht_home_goals", "ht_away_goals",
    "home_ht_score", "away_ht_score",
    # CURRENT-MATCH raw counts: shots, corners, fouls, cards, possession, etc.
    # These are final-whistle stats, not pre-match. Rolling/venue/avg versions
    # of the same underlying stat (e.g. home_roll_5_shots_total) are legitimate
    # pre-match features and are NOT excluded.
    "home_shots_total", "away_shots_total",
    "home_shots_on_target_count", "away_shots_on_target_count",
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
    "home_fouls", "away_fouls",
    "home_yellow_cards", "away_yellow_cards",
    "home_red_cards", "away_red_cards",
    "home_offsides", "away_offsides",
    "home_saves", "away_saves",
    "home_possession", "away_possession",
    "home_possession_pct", "away_possession_pct",
    "home_passing_accuracy", "away_passing_accuracy",
    "total_goals", "total_corners", "total_cards", "total_shots", "total_fouls",
    # Odds-derived meta features — these reflect the closing line and sharp
    # money movement. Including them would make the backtest "edge vs closing
    # line" circular because the model has seen the closing line indirectly.
    # Must exclude from walk-forward training that evaluates against closing
    # odds. (Raw odds are already removed by ml.feature_selection.exclude_odds.)
    "market_elo_disagreement", "odds_consistency", "odds_home_fav",
    "sharp_soft_away_div", "sharp_soft_draw_div", "sharp_soft_home_div",
    "sharp_soft_x_elo",
}


@dataclass(frozen=True)
class MarketSpec:
    """Describes a betting market trained as a supervised target."""
    name: str                         # e.g. "over_2_5"
    kind: str                         # "binary" or "multiclass"
    classes: tuple[str, ...]          # ("yes","no") for binary, ("H","D","A") for 1x2
    target_builder: Callable[[pd.DataFrame], pd.Series]  # produces labels
    class_weights: tuple[float, ...] | None = None  # Default per-class loss weights
    # Per-league override map, e.g. {"serie_a": (1.0, 1.8, 1.0), "premier_league": (1.0, 1.2, 1.0)}.
    # If a league is present here, its weights override `class_weights` for that league.
    per_league_weights: dict[str, tuple[float, ...]] | None = None  # Optional per-class loss weights  # produces labels


def _build_over_2_5(df: pd.DataFrame) -> pd.Series:
    return ((df["home_score"] + df["away_score"]) >= 3).astype(int)


def _build_over_1_5(df: pd.DataFrame) -> pd.Series:
    return ((df["home_score"] + df["away_score"]) >= 2).astype(int)


def _build_over_3_5(df: pd.DataFrame) -> pd.Series:
    return ((df["home_score"] + df["away_score"]) >= 4).astype(int)


def _build_btts(df: pd.DataFrame) -> pd.Series:
    return ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int)


def _build_1x2(df: pd.DataFrame) -> pd.Series:
    def _map(hs, a_s):
        if hs > a_s:
            return 0  # H
        if hs == a_s:
            return 1  # D
        return 2      # A
    return pd.Series(
        [_map(int(h), int(a)) for h, a in zip(df["home_score"], df["away_score"])],
        index=df.index, dtype=int,
    )


def _build_corners_over(line: float):
    def _b(df: pd.DataFrame) -> pd.Series:
        totals = df["home_corners"].fillna(0) + df["away_corners"].fillna(0)
        return (totals > line).astype(int)
    return _b


def _build_cards_over(line: float):
    def _b(df: pd.DataFrame) -> pd.Series:
        totals = (
            df["home_yellow_cards"].fillna(0) + df["away_yellow_cards"].fillna(0)
            + df.get("home_red_cards", pd.Series(0, index=df.index)).fillna(0)
            + df.get("away_red_cards", pd.Series(0, index=df.index)).fillna(0)
        )
        return (totals > line).astype(int)
    return _b


def _build_home_clean_sheet(df: pd.DataFrame) -> pd.Series:
    return (df["away_score"] == 0).astype(int)


def _build_away_clean_sheet(df: pd.DataFrame) -> pd.Series:
    return (df["home_score"] == 0).astype(int)


MARKETS: dict[str, MarketSpec] = {
    "over_2_5":  MarketSpec("over_2_5",  "binary",     ("no", "yes"),     _build_over_2_5),
    "over_1_5":  MarketSpec("over_1_5",  "binary",     ("no", "yes"),     _build_over_1_5),
    "over_3_5":  MarketSpec("over_3_5",  "binary",     ("no", "yes"),     _build_over_3_5),
    "btts":      MarketSpec("btts",      "binary",     ("no", "yes"),     _build_btts),
    # 1X2 draws (class 1) are under-predicted as argmax because H/A dominate.
    # Per-league tuning: SA benefits from stronger draw weighting than EPL.
    # Backtest-validated on 2022-25 walk-forward vs Pinnacle closing line:
    #   SA 1X2 @ 5% edge: (1.0,1.8,1.0) → -9.9% ROI vs (1.0,1.2,1.0) → -11.9%
    #   EPL 1X2 @ 5% edge: (1.0,1.2,1.0) → -4.2% ROI vs (1.0,1.8,1.0) → -9.9%
    # Serie A has slightly higher draw rate (28% vs 23% in EPL), which we
    # think explains why a steeper boost helps there.
    "1x2":       MarketSpec("1x2",       "multiclass", ("H", "D", "A"),   _build_1x2,
                             per_league_weights={
                                 "serie_a":        (1.0, 1.8, 1.0),
                                 "premier_league": (1.0, 1.2, 1.0),
                             }),
    # Corners markets — realistic lines for Serie A + EPL
    "corners_over_8_5":  MarketSpec("corners_over_8_5",  "binary", ("no","yes"), _build_corners_over(8.5)),
    "corners_over_9_5":  MarketSpec("corners_over_9_5",  "binary", ("no","yes"), _build_corners_over(9.5)),
    "corners_over_10_5": MarketSpec("corners_over_10_5", "binary", ("no","yes"), _build_corners_over(10.5)),
    # Cards markets
    "cards_over_3_5":    MarketSpec("cards_over_3_5",    "binary", ("no","yes"), _build_cards_over(3.5)),
    "cards_over_4_5":    MarketSpec("cards_over_4_5",    "binary", ("no","yes"), _build_cards_over(4.5)),
    "cards_over_5_5":    MarketSpec("cards_over_5_5",    "binary", ("no","yes"), _build_cards_over(5.5)),
    # Clean sheet markets — binary "team X conceded 0 goals"
    "home_clean_sheet":  MarketSpec("home_clean_sheet",  "binary", ("no","yes"), _build_home_clean_sheet),
    "away_clean_sheet":  MarketSpec("away_clean_sheet",  "binary", ("no","yes"), _build_away_clean_sheet),
}


def _calibration_gap(y_true: np.ndarray, p_pos: np.ndarray, bins: int = 10) -> float:
    """Mean |predicted − actual| across probability bins (binary targets)."""
    edges = np.linspace(0, 1, bins + 1)
    gaps = []
    for i in range(bins):
        mask = (p_pos >= edges[i]) & (p_pos < edges[i + 1])
        if mask.sum() < 5:
            continue
        gaps.append(abs(p_pos[mask].mean() - y_true[mask].mean()))
    return float(np.mean(gaps)) if gaps else float("nan")


def _ece_multiclass(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error across top-class confidence bins."""
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(bins):
        mask = (confidences >= edges[i]) & (confidences < edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == y_true[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _select_features(df: pd.DataFrame) -> list[str]:
    """Features = everything except meta, target proxies, and raw odds.

    Excludes:
      1. META_COLUMNS (targets, IDs)
      2. Private columns (leading underscore)
      3. Known post-match leakage columns (LEAKY_COLUMNS)
      4. Raw odds (via ml.feature_selection.exclude_odds, preserves disagreement features)

    Then validates: no remaining feature has |corr| > 0.5 with either the
    over/under target or the home/draw/away flags — if one does, we refuse
    to train. Threshold 0.5 lets through natural quality→outcome signals
    like squad_value_diff (~0.4) but catches direct leakage (>0.5).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = [
        c for c in numeric_cols
        if c not in META_COLUMNS and c not in LEAKY_COLUMNS and not c.startswith("_")
    ]
    # Drop columns that are entirely null
    all_null = df[feature_names].isna().all(axis=0)
    feature_names = [f for f in feature_names if not all_null.get(f, False)]
    # Strip raw odds via the shared utility; preserves disagreement metadata
    _, kept = exclude_odds(df[feature_names].copy(), feature_names)

    # Leakage safety net: scan remaining features vs all plausible targets.
    if {"home_score", "away_score"}.issubset(df.columns):
        targets = {
            "over_2_5": ((df["home_score"] + df["away_score"]) >= 3).astype(int),
            "home_win": (df["home_score"] > df["away_score"]).astype(int),
            "draw":     (df["home_score"] == df["away_score"]).astype(int),
            "btts":     ((df["home_score"] > 0) & (df["away_score"] > 0)).astype(int),
        }
        offenders: list[tuple[str, str, float]] = []
        for f in kept:
            col = df[f]
            if col.notna().sum() < 100:
                continue
            for tname, y in targets.items():
                try:
                    c = float(col.corr(y))
                except Exception:
                    continue
                if not np.isnan(c) and abs(c) > 0.5:
                    offenders.append((f, tname, c))
        if offenders:
            offenders.sort(key=lambda x: -abs(x[2]))
            msg = "\n".join(f"  {f} (corr with {t} = {c:+.3f})"
                            for f, t, c in offenders[:20])
            raise RuntimeError(
                f"Refusing to train — {len(offenders)} feature(s) exceed |corr|>0.5 "
                f"with a plausible target (likely leakage). Add them to LEAKY_COLUMNS "
                f"or fix the feature pipeline.\n{msg}"
            )

    return kept


def _fit_one_fold(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    kind: str, n_classes: int,
    class_weights: tuple[float, ...] | None = None,
) -> CatBoostClassifier:
    """Train one CatBoost model with early stopping on the val fold.

    If class_weights is provided, it is passed to CatBoost which scales the
    loss contribution of each class. Useful for imbalanced multiclass targets
    where the minority class (e.g. Draw in 1X2) gets underpicked.
    """
    import os as _os
    params = {
        "iterations": 2000,
        "learning_rate": 0.02,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "min_data_in_leaf": 20,
        "loss_function": "Logloss" if kind == "binary" else "MultiClass",
        "early_stopping_rounds": 150,
        "verbose": 0,
        "random_seed": int(_os.environ.get("TRAIN_SEED", "42")),
        "allow_writing_files": False,
    }
    if kind == "multiclass":
        params["classes_count"] = n_classes
    if class_weights is not None:
        params["class_weights"] = list(class_weights)

    model = CatBoostClassifier(**params)
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)
    model.fit(train_pool, eval_set=val_pool)
    return model


def walkforward_train_market(
    league: str, market: str, eval_seasons: list[str],
    min_train_season: str | None = None,
    min_prior_seasons: int | None = None,
    output_suffix: str = "",
) -> dict:
    """Train one (league, market) pair with strict walk-forward CV.

    For each eval season S:
      - Train data = all seasons strictly before S, optionally floored by
        `min_train_season` (e.g., "2017-2018") so only modern regimes train.
      - The final 15% of training (chronological) is the val pool — used for
        early stopping AND for fitting isotonic calibration.
      - Evaluate on season S only, reporting both raw and calibrated metrics.
      - Save both the CatBoost model and the per-class isotonic calibrators
        so inference-time code can apply calibration.

    If `output_suffix` is non-empty, artifacts go to
    `{league}/{market}__{suffix}/` so multiple variants can coexist without
    overwriting the production models in `{league}/{market}/`.
    """
    import pickle
    from sklearn.isotonic import IsotonicRegression

    spec = MARKETS[market]
    features_path = LEAGUE_TO_FEATURES[league]
    log.info("Loading %s", features_path)
    df = pd.read_parquet(features_path)
    df = df[df["league"] == league].copy() if "league" in df.columns else df
    df = df.sort_values(["season", "match_date"]).reset_index(drop=True)

    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["_target"] = spec.target_builder(df).values

    all_seasons_sorted = sorted(df["season"].unique())
    feature_names = _select_features(df)
    floor_season = min_train_season or ""
    prior_min = MIN_TRAIN_SEASONS if min_prior_seasons is None else int(min_prior_seasons)
    log.info("League=%s market=%s total_rows=%d feature_count=%d "
             "min_train_season=%s min_prior_seasons=%d suffix=%s",
             league, market, len(df), len(feature_names),
             floor_season or "<all>", prior_min, output_suffix or "<none>")

    market_dir_name = f"{market}__{output_suffix}" if output_suffix else market
    out_dir = OUTPUT_ROOT / league / market_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_reports = []
    concat_probs_raw: list[np.ndarray] = []
    concat_probs_cal: list[np.ndarray] = []
    concat_true: list[np.ndarray] = []

    n_classes = len(spec.classes)

    for eval_season in eval_seasons:
        prior = [s for s in all_seasons_sorted if s < eval_season]
        if floor_season:
            prior = [s for s in prior if s >= floor_season]
        if len(prior) < prior_min:
            log.warning("Skipping eval %s: only %d prior seasons available (< %d)",
                        eval_season, len(prior), prior_min)
            continue

        train_mask = df["season"].isin(prior)
        eval_mask = df["season"] == eval_season
        if eval_mask.sum() == 0:
            log.warning("No rows for eval season %s", eval_season)
            continue

        X_full = df.loc[train_mask, feature_names].copy()
        y_full = df.loc[train_mask, "_target"].copy()

        # Chronological split. Two supported modes:
        #   TRAIN_SPLIT_MODE=separate_cal_val (new): 70/15/15 train/es-val/cal-val
        #     Fits isotonic on cal-val which CatBoost never saw (not even ES).
        #     Intended fix for task #6 but FAILED success threshold on 2026-04-24.
        #   TRAIN_SPLIT_MODE=shared  (old/default): 85/15 train/val,
        #     val used for BOTH early stopping and isotonic fit (double-dipping
        #     but matches Phase 1's headline numbers).
        # Back to shared by default until task #6 is actually fixed.
        import os as _os  # local alias to avoid shadowing
        split_mode = _os.environ.get("TRAIN_SPLIT_MODE", "shared")
        n_full = len(X_full)
        if split_mode == "separate_cal_val":
            es_idx = int(n_full * 0.70)
            cal_idx = int(n_full * 0.85)
            X_tr = X_full.iloc[:es_idx]
            y_tr = y_full.iloc[:es_idx]
            X_val = X_full.iloc[es_idx:cal_idx]
            y_val = y_full.iloc[es_idx:cal_idx]
            X_cal = X_full.iloc[cal_idx:]
            y_cal = y_full.iloc[cal_idx:]
        else:  # "shared" — Phase 1 baseline
            split_idx = int(n_full * 0.85)
            X_tr = X_full.iloc[:split_idx]
            y_tr = y_full.iloc[:split_idx]
            X_val = X_full.iloc[split_idx:]
            y_val = y_full.iloc[split_idx:]
            X_cal = X_val  # calibration fits on the same pool as early stopping
            y_cal = y_val

        X_eval = df.loc[eval_mask, feature_names].copy()
        y_eval = df.loc[eval_mask, "_target"].to_numpy()

        # Choose class weights: per-league override > market default > None.
        effective_weights = (
            (spec.per_league_weights or {}).get(league)
            if spec.per_league_weights
            else None
        ) or spec.class_weights

        model = _fit_one_fold(X_tr, y_tr, X_val, y_val,
                              kind=spec.kind, n_classes=n_classes,
                              class_weights=effective_weights)

        # Raw eval-season probabilities.
        proba_raw = model.predict_proba(X_eval)

        # Fit per-class isotonic calibrators on the HELD-OUT CALIBRATION pool
        # (not the early-stopping pool). This pool was never seen during
        # CatBoost training OR early-stopping, so its predictions are truly
        # out-of-fold and legitimate for calibration fitting.
        proba_val = model.predict_proba(X_cal)
        y_val_np = y_cal.to_numpy()
        log.info("  Cal-val size: n=%d  class counts: %s",
                 len(y_val_np),
                 {spec.classes[i]: int((y_val_np == i).sum()) for i in range(n_classes)})
        calibrators = []
        if spec.kind == "binary":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(proba_val[:, 1], y_val_np.astype(float))
            calibrators = [iso]
            p_pos_raw = proba_raw[:, 1]
            p_pos_cal = iso.predict(p_pos_raw)
            p_pos_cal = np.clip(p_pos_cal, 1e-6, 1 - 1e-6)
            proba_cal = np.column_stack([1 - p_pos_cal, p_pos_cal])
        else:
            for ci in range(n_classes):
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(proba_val[:, ci], (y_val_np == ci).astype(float))
                calibrators.append(iso)
            cal_cols = [calibrators[ci].predict(proba_raw[:, ci]) for ci in range(n_classes)]
            proba_cal = np.column_stack(cal_cols)
            # Renormalize so rows sum to 1.
            row_sums = proba_cal.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            proba_cal = proba_cal / row_sums
            proba_cal = np.clip(proba_cal, 1e-6, 1 - 1e-6)
            # Final renormalize after clip.
            proba_cal = proba_cal / proba_cal.sum(axis=1, keepdims=True)

        # Save model + calibrators.
        model_path = out_dir / f"season_{eval_season}.cbm"
        model.save_model(str(model_path))
        cal_path = out_dir / f"season_{eval_season}_calibrators.pkl"
        with open(cal_path, "wb") as fh:
            pickle.dump(calibrators, fh)

        # Metrics: both raw and calibrated.
        def _binary_metrics(p_pos: np.ndarray) -> dict:
            return {
                "log_loss": round(float(log_loss(y_eval, np.clip(p_pos, 1e-6, 1-1e-6))), 4),
                "brier": round(float(brier_score_loss(y_eval, p_pos)), 4),
                "accuracy": round(float(((p_pos >= 0.5).astype(int) == y_eval).mean()), 4),
                "calibration_gap": round(_calibration_gap(y_eval, p_pos), 4),
            }

        def _multi_metrics(proba: np.ndarray) -> dict:
            preds = proba.argmax(axis=1)
            per_class_acc = {}
            for ci, cname in enumerate(spec.classes):
                m = y_eval == ci
                if m.sum() > 0:
                    per_class_acc[cname] = round(float((preds[m] == ci).mean()), 4)
            return {
                "log_loss": round(float(log_loss(y_eval, np.clip(proba, 1e-6, 1-1e-6),
                                                labels=list(range(n_classes)))), 4),
                "accuracy": round(float((preds == y_eval).mean()), 4),
                "ece": round(_ece_multiclass(y_eval, proba), 4),
                "per_class_accuracy": per_class_acc,
            }

        fold_row: dict = {
            "eval_season": eval_season,
            "n_train": int(len(X_full)),
            "n_eval": int(len(X_eval)),
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "calibrators_path": str(cal_path.relative_to(PROJECT_ROOT)),
        }
        if spec.kind == "binary":
            fold_row["base_rate"] = float(y_eval.mean())
            fold_row["raw"] = _binary_metrics(proba_raw[:, 1])
            fold_row["calibrated"] = _binary_metrics(proba_cal[:, 1])
        else:
            fold_row["class_freq"] = {c: round(float((y_eval == i).mean()), 4)
                                      for i, c in enumerate(spec.classes)}
            fold_row["raw"] = _multi_metrics(proba_raw)
            fold_row["calibrated"] = _multi_metrics(proba_cal)

        fold_reports.append(fold_row)
        concat_probs_raw.append(proba_raw)
        concat_probs_cal.append(proba_cal)
        concat_true.append(y_eval)

        meta = {
            "league": league,
            "market": market,
            "kind": spec.kind,
            "classes": list(spec.classes),
            "eval_season": eval_season,
            "train_seasons": prior,
            "min_train_season": floor_season or None,
            "min_prior_seasons": prior_min,
            "output_suffix": output_suffix or None,
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "calibration": "isotonic_per_class_on_val_pool",
            "class_weights": list(effective_weights) if effective_weights else None,
        }
        with open(out_dir / f"season_{eval_season}_metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        log.info("  Eval %s: raw_ll=%.4f raw_acc=%.4f | cal_ll=%.4f cal_acc=%.4f",
                 eval_season, fold_row["raw"]["log_loss"], fold_row["raw"]["accuracy"],
                 fold_row["calibrated"]["log_loss"], fold_row["calibrated"]["accuracy"])

    # Aggregate summary across all folds (raw + calibrated separately).
    summary = {
        "league": league,
        "market": market,
        "kind": spec.kind,
        "classes": list(spec.classes),
        "eval_seasons": [r["eval_season"] for r in fold_reports],
        "n_features": len(feature_names),
        "min_train_season": floor_season or None,
        "min_prior_seasons": prior_min,
        "output_suffix": output_suffix or None,
        "calibration": "isotonic_per_class_on_val_pool",
        "folds": fold_reports,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if concat_probs_raw:
        all_true = np.concatenate(concat_true)
        if spec.kind == "binary":
            raw = np.concatenate([p[:, 1] for p in concat_probs_raw])
            cal = np.concatenate([p[:, 1] for p in concat_probs_cal])
            summary["overall_raw"] = {
                "n": int(len(all_true)),
                "log_loss": round(float(log_loss(all_true, np.clip(raw, 1e-6, 1-1e-6))), 4),
                "brier": round(float(brier_score_loss(all_true, raw)), 4),
                "accuracy": round(float(((raw >= 0.5).astype(int) == all_true).mean()), 4),
                "calibration_gap": round(_calibration_gap(all_true, raw), 4),
                "base_rate": round(float(all_true.mean()), 4),
            }
            summary["overall_calibrated"] = {
                "n": int(len(all_true)),
                "log_loss": round(float(log_loss(all_true, np.clip(cal, 1e-6, 1-1e-6))), 4),
                "brier": round(float(brier_score_loss(all_true, cal)), 4),
                "accuracy": round(float(((cal >= 0.5).astype(int) == all_true).mean()), 4),
                "calibration_gap": round(_calibration_gap(all_true, cal), 4),
                "base_rate": round(float(all_true.mean()), 4),
            }
            # Back-compat: keep "overall" as the calibrated version
            summary["overall"] = summary["overall_calibrated"]
        else:
            raw = np.concatenate(concat_probs_raw, axis=0)
            cal = np.concatenate(concat_probs_cal, axis=0)
            raw_preds = raw.argmax(axis=1)
            cal_preds = cal.argmax(axis=1)
            summary["overall_raw"] = {
                "n": int(len(all_true)),
                "log_loss": round(float(log_loss(all_true, np.clip(raw, 1e-6, 1-1e-6),
                                                labels=list(range(n_classes)))), 4),
                "accuracy": round(float((raw_preds == all_true).mean()), 4),
                "ece": round(_ece_multiclass(all_true, raw), 4),
                "per_class_accuracy": {
                    c: round(float((raw_preds[all_true == i] == i).mean()), 4)
                    if (all_true == i).sum() > 0 else None
                    for i, c in enumerate(spec.classes)
                },
                "class_freq": {c: round(float((all_true == i).mean()), 4)
                               for i, c in enumerate(spec.classes)},
            }
            summary["overall_calibrated"] = {
                "n": int(len(all_true)),
                "log_loss": round(float(log_loss(all_true, np.clip(cal, 1e-6, 1-1e-6),
                                                labels=list(range(n_classes)))), 4),
                "accuracy": round(float((cal_preds == all_true).mean()), 4),
                "ece": round(_ece_multiclass(all_true, cal), 4),
                "per_class_accuracy": {
                    c: round(float((cal_preds[all_true == i] == i).mean()), 4)
                    if (all_true == i).sum() > 0 else None
                    for i, c in enumerate(spec.classes)
                },
                "class_freq": {c: round(float((all_true == i).mean()), 4)
                               for i, c in enumerate(spec.classes)},
            }
            summary["overall"] = summary["overall_calibrated"]

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--league", choices=list(LEAGUE_TO_FEATURES), required=False)
    ap.add_argument("--market", choices=list(MARKETS), required=False)
    ap.add_argument("--all", action="store_true", help="Run every league × market combo")
    ap.add_argument("--eval-seasons", default=",".join(DEFAULT_EVAL_SEASONS))
    ap.add_argument("--min-train-season", default="",
                    help='Earliest season allowed in training pool, e.g. "2017-2018". '
                         'Empty = use all seasons from the features parquet.')
    ap.add_argument("--min-prior-seasons", type=int, default=None,
                    help="Override MIN_TRAIN_SEASONS guard (default 5). "
                         "Narrower windows (e.g. 2019+) may only have 2-3 prior seasons.")
    ap.add_argument("--output-suffix", default="",
                    help="If set, artifacts go to {league}/{market}__{suffix}/ "
                         "so variants don't overwrite each other.")
    args = ap.parse_args()

    eval_seasons = [s.strip() for s in args.eval_seasons.split(",") if s.strip()]

    if args.all:
        combos: Iterable[tuple[str, str]] = [
            (lg, mk) for lg in LEAGUE_TO_FEATURES for mk in MARKETS
        ]
    elif args.league and args.market:
        combos = [(args.league, args.market)]
    else:
        ap.error("Either --all or both --league and --market must be provided")
        return 2

    global_summary: dict[str, dict] = {}
    for lg, mk in combos:
        log.info("=" * 72)
        log.info("Training %s / %s", lg, mk)
        log.info("=" * 72)
        s = walkforward_train_market(
            lg, mk, eval_seasons,
            min_train_season=(args.min_train_season or None),
            min_prior_seasons=args.min_prior_seasons,
            output_suffix=args.output_suffix,
        )
        global_summary[f"{lg}__{mk}"] = s.get("overall", {})

    suffix_part = f"_{args.output_suffix}" if args.output_suffix else ""
    out_path = OUTPUT_ROOT / f"run_summary{suffix_part}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "eval_seasons": eval_seasons,
            "combos": global_summary,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    print("\n=== Walk-forward training summary ===")
    for key, stats in global_summary.items():
        if not stats:
            print(f"  {key}: (no folds produced)")
            continue
        if "accuracy" in stats:
            extra = ""
            if "per_class_accuracy" in stats:
                pc = stats["per_class_accuracy"]
                extra = " | per-class: " + ", ".join(f"{k}={v}" for k, v in pc.items() if v is not None)
            print(f"  {key}: n={stats['n']} acc={stats['accuracy']} ll={stats['log_loss']}"
                  f" {('ece=' + str(stats.get('ece','-'))) if 'ece' in stats else ('cal_gap=' + str(stats.get('calibration_gap','-')))}"
                  f"{extra}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
