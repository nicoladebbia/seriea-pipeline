"""
Player-Level Prediction System for Serie A
==========================================
Predicts per-player per-match outcomes:
  1. Anytime Goalscorer (P(goals > 0))
  2. Player Shots O/U 0.5, 1.5, 2.5
  3. Player Shots on Target O/U 0.5
  4. Player to be Carded (P(yellow card))
  5. Player Tackles O/U 1.5
  6. Player Fouls Committed O/U 0.5

Data sources:
  - Sofascore player_match_stats (97K rows, 64 cols)
  - Sofascore match_incidents (14.6K card events)
  - Sofascore lineups / player info

Architecture:
  - 1 CatBoost classifier per target
  - Rolling player features (last 5/10 matches) + opponent team features
  - Chronological walk-forward split (no leakage)
"""

import numpy as np
import pandas as pd
import logging
import json
import gc
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    log_loss, brier_score_loss, roc_auc_score, accuracy_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SS_DIR   = DATA_DIR / "external" / "sofascore"
MODEL_DIR = DATA_DIR / "models" / "player"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TEAM_MAP = {
    "Hellas Verona": "Verona", "ChievoVerona": "Chievo",
    "AC Milan": "Milan", "Inter Milan": "Inter",
    "Internazionale": "Inter", "Parma Calcio 1913": "Parma",
}

# ── target definitions ───────────────────────────────────────────────────────
TARGETS = {
    "goalscorer":   {"col": "goals",           "op": ">", "line": 0, "label": "Anytime Goalscorer"},
    "shots_o05":    {"col": "total_shots",      "op": ">", "line": 0, "label": "Shots Over 0.5"},
    "shots_o15":    {"col": "total_shots",      "op": ">", "line": 1, "label": "Shots Over 1.5"},
    "shots_o25":    {"col": "total_shots",      "op": ">", "line": 2, "label": "Shots Over 2.5"},
    "sot_o05":      {"col": "shots_on_target",  "op": ">", "line": 0, "label": "SOT Over 0.5"},
    "carded":       {"col": "n_yellows",        "op": ">", "line": 0, "label": "To Be Carded"},
    "tackles_o15":  {"col": "tackles",          "op": ">", "line": 1, "label": "Tackles Over 1.5"},
    "fouls_o05":    {"col": "fouls",            "op": ">", "line": 0, "label": "Fouls Over 0.5"},
    "keypasses_o05":{"col": "key_passes",       "op": ">", "line": 0, "label": "Key Passes Over 0.5"},
    "keypasses_o15":{"col": "key_passes",       "op": ">", "line": 1, "label": "Key Passes Over 1.5"},
}

# ── rolling window config ────────────────────────────────────────────────────
ROLL_STATS = [
    "goals", "total_shots", "shots_on_target", "key_passes",
    "accurate_passes", "total_passes", "tackles", "tackles_won",
    "interceptions", "fouls", "was_fouled", "duels_won", "duels_lost",
    "aerial_won", "aerial_lost", "clearances", "blocks",
    "progressive_carries", "big_chances_created", "big_chances_missed",
    "rating", "minutes", "n_yellows", "dispossessed", "ball_recoveries",
    "total_crosses", "carries", "opp_half_passes",
    "assists", "accurate_crosses", "n_reds",
    "xg_total", "xg_max", "xg_mean", "xg_per_shot", "xg_overperformance",
    "xg_npxg", "xg_big_chances", "xg_conversion", "xg_n_shots",
]
WINDOWS = [5, 10]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_player_data() -> pd.DataFrame:
    """Load and merge Sofascore player stats with card incidents."""
    log.info("Loading Sofascore player_match_stats …")
    pms = pd.read_parquet(SS_DIR / "player_match_stats.parquet")
    pms["team"] = pms["team"].replace(TEAM_MAP)
    pms["opponent"] = pms["opponent"].replace(TEAM_MAP)
    pms["date"] = pd.to_datetime(pms["date"])
    pms["match_id_str"] = pms["match_id"].astype(str)

    # ── merge card incidents ─────────────────────────────────────────────
    log.info("Loading match incidents for card data …")
    inc = pd.read_parquet(SS_DIR / "match_incidents.parquet")
    card_inc = inc[inc["incident_type"] == "card"].copy()
    card_inc["match_id_str"] = card_inc["match_id"].astype(str)
    card_inc["player_id"] = pd.to_numeric(card_inc["player_id"], errors="coerce")
    card_inc = card_inc.dropna(subset=["player_id"])
    card_inc["player_id"] = card_inc["player_id"].astype(int)

    cards_agg = (
        card_inc.groupby(["match_id_str", "player_id"])
        .agg(
            n_yellows=("card_type", lambda x: (x == "yellow").sum()),
            n_reds=("card_type", lambda x: x.isin(["red", "yellowRed"]).sum()),
        )
        .reset_index()
    )

    pms = pms.merge(cards_agg, on=["match_id_str", "player_id"], how="left")
    pms["n_yellows"] = pms["n_yellows"].fillna(0).astype(int)
    pms["n_reds"] = pms["n_reds"].fillna(0).astype(int)

    # ── merge per-player xG from shot data ───────────────────────────
    shots_path = SS_DIR / "all_shots_with_xg.parquet"
    if shots_path.exists():
        log.info("Loading per-player xG from shot data …")
        shots = pd.read_parquet(shots_path)
        shots["match_id"] = pd.to_numeric(shots["match_id"], errors="coerce")
        shots = shots.dropna(subset=["match_id", "player_id"])
        shots["match_id"] = shots["match_id"].astype(int)
        shots["player_id"] = shots["player_id"].astype(int)

        # Use xg_predicted (82K full coverage) with xg as fallback
        shots["xg_val"] = shots["xg_predicted"].fillna(shots["xg"]).fillna(0)

        shot_agg = shots.groupby(["match_id", "player_id"]).agg(
            xg_total=("xg_val", "sum"),
            xg_max=("xg_val", "max"),
            xg_mean=("xg_val", "mean"),
            xg_n_shots=("xg_val", "count"),
            xg_goals=("is_goal", "sum"),
            xg_big_chances=("xg_val", lambda x: (x > 0.25).sum()),
            xg_headers=("is_header", "sum"),
            xg_penalties=("is_penalty", "sum"),
            xg_fast_breaks=("is_fast_break", "sum"),
            xg_npxg=("xg_val", lambda x: x[~shots.loc[x.index, "is_penalty"].astype(bool)].sum()),
        ).reset_index()

        shot_agg["xg_per_shot"] = shot_agg["xg_total"] / shot_agg["xg_n_shots"].clip(lower=1)
        shot_agg["xg_overperformance"] = shot_agg["xg_goals"] - shot_agg["xg_total"]
        shot_agg["xg_conversion"] = shot_agg["xg_goals"] / shot_agg["xg_n_shots"].clip(lower=1)

        pms = pms.merge(shot_agg, on=["match_id", "player_id"], how="left")
        xg_cols = ["xg_total", "xg_max", "xg_mean", "xg_n_shots", "xg_goals",
                   "xg_big_chances", "xg_headers", "xg_penalties", "xg_fast_breaks",
                   "xg_npxg", "xg_per_shot", "xg_overperformance", "xg_conversion"]
        for c in xg_cols:
            pms[c] = pms[c].fillna(0)
        n_xg = pms["xg_total"].gt(0).sum()
        log.info(f"  xG features merged: {n_xg}/{len(pms)} rows ({n_xg/len(pms)*100:.1f}%)")
    else:
        log.warning(f"No shot data at {shots_path}")

    # ── merge referee data (critical for cards model) ──────────────────
    ref_path = DATA_DIR / "external" / "referee" / "referee_assignments.parquet"
    if ref_path.exists():
        log.info("Loading referee assignment data …")
        ref = pd.read_parquet(ref_path)
        ref["match_date"] = pd.to_datetime(ref["match_date"])
        ref["home_team"] = ref["home_team"].replace(TEAM_MAP)
        ref["away_team"] = ref["away_team"].replace(TEAM_MAP)

        # Build referee rolling stats (avg yellows/reds per match, lagged)
        ref = ref.sort_values("match_date")
        ref_roll = []
        for rname, rgrp in ref.groupby("referee"):
            rgrp = rgrp.sort_values("match_date")
            rgrp["ref_avg_yellows"] = rgrp["ref_yellows"].shift(1).expanding(min_periods=3).mean()
            rgrp["ref_avg_reds"] = rgrp["ref_reds"].shift(1).expanding(min_periods=3).mean()
            rgrp["ref_avg_cards_total"] = (rgrp["ref_yellows"] + rgrp["ref_reds"]).shift(1).expanding(min_periods=3).mean()
            rgrp["ref_card_std"] = rgrp["ref_yellows"].shift(1).expanding(min_periods=5).std()
            rgrp["ref_r5_yellows"] = rgrp["ref_yellows"].shift(1).rolling(5, min_periods=3).mean()
            rgrp["ref_matches"] = range(len(rgrp))
            ref_roll.append(rgrp)
        ref_rolling = pd.concat(ref_roll, ignore_index=True)

        ref_cols = ["ref_avg_yellows", "ref_avg_reds", "ref_avg_cards_total",
                    "ref_card_std", "ref_r5_yellows", "ref_matches"]

        # Merge to player data via date + home/away team
        ref_merge = ref_rolling[["match_date", "home_team", "away_team"] + ref_cols].copy()
        pms = pms.merge(
            ref_merge,
            left_on=["date", "home_team", "away_team"],
            right_on=["match_date", "home_team", "away_team"],
            how="left",
        )
        if "match_date" in pms.columns:
            pms.drop(columns=["match_date"], inplace=True)
        for c in ref_cols:
            pms[c] = pms[c].fillna(0)
        n_ref = pms["ref_avg_yellows"].gt(0).sum()
        log.info(f"  Referee features merged: {n_ref}/{len(pms)} rows ({n_ref/len(pms)*100:.1f}%)")
    else:
        log.warning(f"No referee data at {ref_path}")
        for c in ["ref_avg_yellows", "ref_avg_reds", "ref_avg_cards_total",
                  "ref_card_std", "ref_r5_yellows", "ref_matches"]:
            pms[c] = 0.0

    # ── merge Transfermarkt market values per player ─────────────────
    tm_dir = DATA_DIR / "external" / "transfermarkt"
    tm_files = sorted(tm_dir.glob("market_values_*.parquet"))
    if tm_files:
        log.info("Loading Transfermarkt market values …")
        import unicodedata as _ud
        from collections import defaultdict as _dd

        def _norm(s):
            if not isinstance(s, str):
                return ""
            s = _ud.normalize("NFD", s)
            return "".join(c for c in s if _ud.category(c) != "Mn").lower().strip()

        all_mv = []
        for f in tm_files:
            mv = pd.read_parquet(f)
            season = f.stem.replace("market_values_", "").replace("_", "-")
            mv["season"] = season
            all_mv.append(mv)
        tm = pd.concat(all_mv, ignore_index=True)
        tm["team"] = tm["team"].replace(TEAM_MAP)
        tm["name_norm"] = tm["player_name"].apply(_norm)

        # Build lookups: (name_norm, team, season), (name_norm, season), (lastname, team, season)
        tm_exact = {}
        tm_name_season = {}
        tm_lastname = _dd(list)
        for _, row in tm.iterrows():
            val = row.get("market_value_eur", 0) or 0
            key1 = (row["name_norm"], row["team"], row["season"])
            tm_exact[key1] = val
            key2 = (row["name_norm"], row["season"])
            if val > tm_name_season.get(key2, 0):
                tm_name_season[key2] = val
            parts = row["name_norm"].split()
            if parts:
                tm_lastname[(parts[-1], row["team"], row["season"])].append(val)

        pms["_name_norm"] = pms["player_name"].apply(_norm)
        mv_vals = []
        for _, row in pms.iterrows():
            nn = row["_name_norm"]
            team = row.get("team", "")
            season = row["season"]
            val = tm_exact.get((nn, team, season)) or tm_name_season.get((nn, season))
            if not val:
                parts = nn.split()
                if parts:
                    cands = tm_lastname.get((parts[-1], team, season), [])
                    if len(cands) == 1:
                        val = cands[0]
            mv_vals.append(val or 0)

        pms["market_value_eur"] = mv_vals
        pms["market_value_log"] = np.log1p(pms["market_value_eur"])
        pms.drop(columns=["_name_norm"], inplace=True)
        n_mv = (pms["market_value_eur"] > 0).sum()
        log.info(f"  Market values merged: {n_mv}/{len(pms)} rows ({n_mv/len(pms)*100:.1f}%)")
    else:
        log.warning(f"No Transfermarkt data at {tm_dir}")
        pms["market_value_eur"] = 0
        pms["market_value_log"] = 0

    log.info(f"Player data: {len(pms)} rows, {pms['player_id'].nunique()} players, "
             f"{pms['season'].nunique()} seasons")
    return pms


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_player_features(pms: pd.DataFrame) -> pd.DataFrame:
    """
    Build rolling per-player features + opponent team features.
    All features are LAGGED (shift=1) to prevent leakage.
    """
    log.info("Building player rolling features …")
    pms = pms.sort_values(["player_id", "date"]).reset_index(drop=True)

    # ── per-90 normalization for key stats ───────────────────────────────
    mins_safe = pms["minutes"].clip(lower=1)
    for col in ["goals", "total_shots", "shots_on_target", "key_passes",
                "tackles", "fouls", "was_fouled", "interceptions",
                "duels_won", "duels_lost", "xg_total", "xg_npxg"]:
        if col in pms.columns:
            pms[f"{col}_p90"] = pms[col] / mins_safe * 90

    # ── pass accuracy ────────────────────────────────────────────────────
    pms["pass_pct"] = pms["accurate_passes"] / pms["total_passes"].clip(lower=1)
    pms["duel_win_pct"] = pms["duels_won"] / (pms["duels_won"] + pms["duels_lost"]).clip(lower=1)

    # ── rolling features per player ──────────────────────────────────────
    roll_cols = [c for c in ROLL_STATS if c in pms.columns]
    p90_cols = [c for c in pms.columns if c.endswith("_p90")]
    derived_cols = ["pass_pct", "duel_win_pct"]
    all_roll = roll_cols + p90_cols + derived_cols

    feat_frames = []
    for pid, grp in pms.groupby("player_id"):
        grp = grp.sort_values("date")
        for w in WINDOWS:
            for col in all_roll:
                grp[f"pr{w}_{col}"] = grp[col].shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
            # std for key stats (captures consistency)
            for col in ["goals", "total_shots", "rating", "fouls", "tackles",
                        "xg_total", "xg_overperformance"]:
                if col in grp.columns:
                    grp[f"pr{w}_{col}_std"] = grp[col].shift(1).rolling(w, min_periods=max(2, w // 2)).std()
        # career stats (expanding, shifted)
        for col in ["goals", "total_shots", "n_yellows", "fouls", "tackles", "rating",
                    "key_passes", "assists", "xg_total", "xg_npxg", "xg_per_shot",
                    "xg_overperformance"]:
            if col in grp.columns:
                grp[f"career_{col}_mean"] = grp[col].shift(1).expanding(min_periods=3).mean()
        # games played (proxy for experience)
        grp["career_games"] = range(len(grp))
        feat_frames.append(grp)

    pms = pd.concat(feat_frames, ignore_index=True)
    log.info(f"  Rolling features built: {len([c for c in pms.columns if c.startswith('pr')])} cols")

    # ── opponent team rolling features ───────────────────────────────────
    log.info("Building opponent team features …")
    # Aggregate team-match stats (goals conceded, fouls, cards)
    team_match = (
        pms[pms["is_starter"] & (pms["minutes"] >= 45)]
        .groupby(["season", "date", "team"])
        .agg(
            team_goals_scored=("goals", "sum"),
            team_shots=("total_shots", "sum"),
            team_fouls=("fouls", "sum"),
            team_tackles=("tackles", "sum"),
            team_yellows=("n_yellows", "sum"),
            team_avg_rating=("rating", "mean"),
        )
        .reset_index()
    )
    team_match = team_match.sort_values(["team", "date"])

    # Rolling team stats (opponent perspective)
    opp_frames = []
    for team_name, grp in team_match.groupby("team"):
        grp = grp.sort_values("date")
        for col in ["team_goals_scored", "team_shots", "team_fouls",
                     "team_tackles", "team_yellows", "team_avg_rating"]:
            grp[f"opp_r5_{col}"] = grp[col].shift(1).rolling(5, min_periods=2).mean()
        opp_frames.append(grp)
    opp_rolling = pd.concat(opp_frames, ignore_index=True)
    opp_cols = [c for c in opp_rolling.columns if c.startswith("opp_r5_")]

    # Merge: opponent's rolling stats → each player row
    pms = pms.merge(
        opp_rolling[["team", "date"] + opp_cols].rename(columns={"team": "opponent"}),
        on=["opponent", "date"],
        how="left",
    )

    # ── static / contextual features ─────────────────────────────────────
    pms["is_forward"] = (pms["position"] == "F").astype(int)
    pms["is_midfielder"] = (pms["position"] == "M").astype(int)
    pms["is_defender"] = (pms["position"] == "D").astype(int)
    pms["is_goalkeeper"] = (pms["position"] == "G").astype(int)
    pms["is_home_int"] = pms["is_home"].astype(int)

    log.info(f"  Total columns after feature engineering: {len(pms.columns)}")
    return pms


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return the list of feature columns for modeling."""
    prefixes = ("pr5_", "pr10_", "career_", "opp_r5_", "is_forward",
                "is_midfielder", "is_defender", "is_goalkeeper", "is_home_int",
                "ref_", "market_value_log")
    return [c for c in df.columns
            if any(c.startswith(p) for p in prefixes)
            and df[c].dtype in ("float64", "float32", "int64", "int32", "int8")]


# =============================================================================
# TRAINING
# =============================================================================

def train_player_models(
    pms: pd.DataFrame,
    val_season: str = "2023-2024",
    test_seasons: Optional[List[str]] = None,
) -> Dict:
    """Train one CatBoost model per target, evaluate on OOS data."""
    if test_seasons is None:
        test_seasons = ["2024-2025", "2025-2026"]

    features = get_feature_cols(pms)
    log.info(f"Feature columns: {len(features)}")

    # Only starters with meaningful minutes
    df = pms[(pms["is_starter"]) & (pms["minutes"] >= 45)].copy()
    log.info(f"Training set (starters 45+ min): {len(df)}")

    seasons = sorted(df["season"].unique())
    train_seasons = [s for s in seasons if s < val_season]

    train_m = df["season"].isin(train_seasons)
    val_m = df["season"] == val_season
    test_m = df["season"].isin(test_seasons)

    log.info(f"Train: {train_m.sum()}, Val: {val_m.sum()}, Test: {test_m.sum()}")

    results = {}

    for target_name, target_cfg in TARGETS.items():
        col = target_cfg["col"]
        line = target_cfg["line"]
        label = target_cfg["label"]

        if col not in df.columns:
            log.warning(f"  SKIP {target_name}: column '{col}' not found")
            continue

        y = (df[col] > line).astype(int)
        pos_rate = y[test_m].mean()

        log.info(f"\n{'='*60}")
        log.info(f"  {label} (target: {col} > {line}, base rate: {pos_rate:.1%})")
        log.info(f"{'='*60}")

        X_tr = df.loc[train_m, features].fillna(0)
        X_val = df.loc[val_m, features].fillna(0)
        X_te = df.loc[test_m, features].fillna(0)
        y_tr = y[train_m]
        y_val = y[val_m]
        y_te = y[test_m]

        model = CatBoostClassifier(
            iterations=2000,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=0,
            loss_function="Logloss",
            eval_metric="Logloss",
        )
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                  early_stopping_rounds=100, verbose=0)

        # Platt scaling (logistic calibration) on validation set
        # Preserves smooth probability spread unlike isotonic (staircase)
        raw_val = model.predict_proba(X_val)[:, 1]
        log_odds_val = np.log(np.clip(raw_val, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_val, 1e-7, 1 - 1e-7)))
        platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        platt.fit(log_odds_val.reshape(-1, 1), y_val)

        # Evaluate with calibrated probabilities
        raw_te = model.predict_proba(X_te)[:, 1]
        log_odds_te = np.log(np.clip(raw_te, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_te, 1e-7, 1 - 1e-7)))
        probs = platt.predict_proba(log_odds_te.reshape(-1, 1))[:, 1]
        pred = (probs > 0.5).astype(int)
        acc = accuracy_score(y_te, pred)
        brier = brier_score_loss(y_te, probs)
        base_brier = brier_score_loss(y_te, np.full(len(y_te), pos_rate))
        try:
            auc = roc_auc_score(y_te, probs)
        except ValueError:
            auc = 0.5
        ll = log_loss(y_te, probs)
        base_ll = log_loss(y_te, np.full(len(y_te), pos_rate))

        brier_improvement = (base_brier - brier) / base_brier * 100
        ll_improvement = (base_ll - ll) / base_ll * 100

        log.info(f"  Acc={acc:.1%}, AUC={auc:.3f}, Brier={brier:.4f} "
                 f"(base={base_brier:.4f}, {brier_improvement:+.1f}%)")
        log.info(f"  LogLoss={ll:.4f} (base={base_ll:.4f}, {ll_improvement:+.1f}%)")
        log.info(f"  Iterations: {model.best_iteration_}")

        # Top features
        imp = model.get_feature_importance()
        top_idx = np.argsort(imp)[-10:][::-1]
        top_feats = [(features[i], round(imp[i], 1)) for i in top_idx]
        log.info(f"  Top features: {top_feats[:5]}")

        # Save model + calibrator
        model_path = MODEL_DIR / f"player_{target_name}.cbm"
        model.save_model(str(model_path))
        cal_path = MODEL_DIR / f"player_{target_name}_platt.pkl"
        with open(cal_path, "wb") as fp:
            pickle.dump(platt, fp)
        log.info(f"  Saved: {model_path} + {cal_path}")

        results[target_name] = {
            "label": label,
            "base_rate": round(float(pos_rate), 4),
            "accuracy": round(float(acc), 4),
            "auc": round(float(auc), 4),
            "brier": round(float(brier), 4),
            "base_brier": round(float(base_brier), 4),
            "brier_improvement_pct": round(float(brier_improvement), 1),
            "log_loss": round(float(ll), 4),
            "base_log_loss": round(float(base_ll), 4),
            "ll_improvement_pct": round(float(ll_improvement), 1),
            "iterations": model.best_iteration_,
            "top_features": top_feats[:5],
            "n_features": len(features),
        }

        del model
        gc.collect()

    # Save metadata
    meta_path = MODEL_DIR / "player_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "n_features": len(features),
            "feature_names": features,
            "val_season": val_season,
            "test_seasons": test_seasons,
            "calibration": "platt_scaling",
            "results": results,
        }, f, indent=2)
    log.info(f"\nMetadata saved: {meta_path}")

    return results


# =============================================================================
# PREDICTION — For upcoming matches
# =============================================================================

def predict_player_markets(
    player_name: str,
    player_id: int,
    team: str,
    opponent: str,
    position: str,
    is_home: bool,
    is_starter: bool = True,
    pms: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Generate predictions for a single player in an upcoming match.
    Uses the player's historical rolling stats + opponent features.
    """
    from catboost import CatBoostClassifier

    if pms is None:
        pms = load_player_data()
        pms = build_player_features(pms)

    # Get this player's latest rolling features
    player_df = pms[pms["player_id"] == player_id].sort_values("date")
    if len(player_df) == 0:
        log.warning(f"No data for player {player_name} (ID={player_id})")
        return {}

    latest = player_df.iloc[-1]
    features = get_feature_cols(pms)

    # Build feature row
    row = {}
    for f in features:
        if f in latest.index and pd.notna(latest[f]):
            row[f] = float(latest[f])
        else:
            row[f] = 0.0

    # Override contextual features
    row["is_forward"] = int(position == "F")
    row["is_midfielder"] = int(position == "M")
    row["is_defender"] = int(position == "D")
    row["is_goalkeeper"] = int(position == "G")
    row["is_home_int"] = int(is_home)

    # Get opponent's latest rolling stats
    opp_df = pms[(pms["team"] == opponent)].sort_values("date")
    if len(opp_df) > 0:
        opp_latest = opp_df.iloc[-1]
        for f in features:
            if f.startswith("opp_r5_") and f in opp_latest.index and pd.notna(opp_latest[f]):
                row[f] = float(opp_latest[f])

    X = pd.DataFrame([row])[features].fillna(0)

    # Run all models
    predictions = {
        "player_name": player_name,
        "player_id": player_id,
        "team": team,
        "opponent": opponent,
        "position": position,
        "is_home": is_home,
        "markets": {},
    }

    for target_name, target_cfg in TARGETS.items():
        model_path = MODEL_DIR / f"player_{target_name}.cbm"
        iso_path = MODEL_DIR / f"player_{target_name}_iso.pkl"
        if not model_path.exists():
            continue

        model = CatBoostClassifier()
        model.load_model(str(model_path))

        # Align features to model
        model_feats = model.feature_names_
        aligned = {f: row.get(f, 0) for f in model_feats}
        X_aligned = pd.DataFrame([aligned])[model_feats].fillna(0)

        raw_prob = float(model.predict_proba(X_aligned)[0][1])

        # Apply Platt scaling calibration if available
        cal_path = MODEL_DIR / f"player_{target_name}_platt.pkl"
        if cal_path.exists():
            with open(cal_path, "rb") as fp:
                platt = pickle.load(fp)
            log_odds = np.log(max(raw_prob, 1e-7) / max(1 - raw_prob, 1e-7))
            prob = float(platt.predict_proba([[log_odds]])[0][1])
        else:
            prob = raw_prob

        predictions["markets"][target_name] = {
            "label": target_cfg["label"],
            "prob": round(prob, 4),
            "odds_implied": round(1 / max(prob, 0.01), 2),
        }

    return predictions


def predict_match_players(
    home_team: str,
    away_team: str,
    home_lineup: List[Dict],
    away_lineup: List[Dict],
    pms: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Predict all player markets for both teams in a match.
    Each lineup entry: {"player_name": str, "player_id": int, "position": str}
    """
    if pms is None:
        pms = load_player_data()
        pms = build_player_features(pms)

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "home_players": [],
        "away_players": [],
    }

    for player in home_lineup:
        pred = predict_player_markets(
            player_name=player["player_name"],
            player_id=player["player_id"],
            team=home_team,
            opponent=away_team,
            position=player.get("position", "M"),
            is_home=True,
            pms=pms,
        )
        if pred:
            result["home_players"].append(pred)

    for player in away_lineup:
        pred = predict_player_markets(
            player_name=player["player_name"],
            player_id=player["player_id"],
            team=away_team,
            opponent=home_team,
            position=player.get("position", "M"),
            is_home=False,
            pms=pms,
        )
        if pred:
            result["away_players"].append(pred)

    return result


# =============================================================================
# MAIN — Train and evaluate
# =============================================================================

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        pms = load_player_data()
        pms = build_player_features(pms)
        results = train_player_models(pms)

        log.info(f"\n{'='*70}")
        log.info("PLAYER MODEL RESULTS SUMMARY")
        log.info(f"{'='*70}")
        for name, r in results.items():
            log.info(f"  {r['label']:25s}: AUC={r['auc']:.3f}, "
                     f"Brier={r['brier']:.4f} ({r['brier_improvement_pct']:+.1f}% vs base), "
                     f"LL={r['log_loss']:.4f} ({r['ll_improvement_pct']:+.1f}%)")

    elif mode == "predict":
        # Example: predict for a specific match
        pms = load_player_data()
        pms = build_player_features(pms)

        # Get latest lineup for a team
        latest_season = pms[pms["season"] == "2025-2026"]
        if len(latest_season) > 0:
            # Pick a recent match as example
            latest_match = latest_season.sort_values("date").iloc[-1]
            team = latest_match["team"]
            opp = latest_match["opponent"]
            log.info(f"Example prediction for {team} vs {opp}")

            # Get starters from that team's last match
            team_starters = latest_season[
                (latest_season["team"] == team) &
                (latest_season["date"] == latest_match["date"]) &
                (latest_season["is_starter"])
            ]

            for _, player in team_starters.iterrows():
                pred = predict_player_markets(
                    player_name=player["player_name"],
                    player_id=player["player_id"],
                    team=team,
                    opponent=opp,
                    position=player["position"],
                    is_home=player["is_home"],
                    pms=pms,
                )
                if pred and pred.get("markets"):
                    gs = pred["markets"].get("goalscorer", {})
                    cd = pred["markets"].get("carded", {})
                    sh = pred["markets"].get("shots_o15", {})
                    log.info(f"  {player['player_name']:25s} ({player['position']}): "
                             f"Goal={gs.get('prob', 0):.1%}, "
                             f"Card={cd.get('prob', 0):.1%}, "
                             f"Shots>1.5={sh.get('prob', 0):.1%}")
