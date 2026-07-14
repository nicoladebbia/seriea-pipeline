"""A/B backtest: does the leak-free net_squad_delta_diff add skill to the 1X2 model?

Faithful to the production protocol: imports the SAME walk_forward_validate,
TimeSeriesSplitter, compute_metrics, sample weights, CatBoost params and locked
126-feature set that scripts/models/retrain_no_odds_catboost.py uses. The ONLY
difference between the two arms is whether net_squad_delta_diff is appended to
the feature list — same folds, same seeds, same everything else. Anything else
would make the delta a fabricated number (cross-condition comparison).

The feature is computed FRESH here (not read from features_serie_a.parquet, which
predates the column) using the FIXED compute_net_squad_delta:
  - winter-window transfers excluded (leak-free; pre-season delta only)
  - season-matched market values (no future-price contamination)

Restricted to seasons where BOTH weight halves are populated (delta uses the
prior season's on-pitch importance, which starts 2019-20 → earliest usable delta
season is 2020-21). Earlier folds would score the 0.15-floor noise.

Run: python3 -m scripts.analysis.backtest_net_squad_delta
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR
from features.transfer_impact_analysis import compute_net_squad_delta
from ml.config import LABEL_MAP, ValidationConfig
from scripts.models.retrain_no_odds_catboost import (
    load_current_feature_set,
    walk_forward_validate,
)
from storage.paths import features_path

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger(__name__)

# Delta seasons whose weights are fully populated (prior-season importance exists
# from 2019-20; season-matched market values exist for all). Matches trained on
# earlier seasons keep net_squad_delta_diff = 0 (no signal, no harm).
DELTA_SEASONS = [
    "2020-2021", "2021-2022", "2022-2023",
    "2023-2024", "2024-2025", "2025-2026",
]

BASE_PARAMS = {
    "iterations": 1000, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0,
    "loss_function": "MultiClass", "eval_metric": "MultiClass",
    "random_seed": 42, "verbose": False, "thread_count": -1,
}


def _attach_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-free net_squad_delta per (club, season) and join onto matches.

    home/away_net_squad_delta default 0.0; net_squad_delta_diff = home - away.
    Only DELTA_SEASONS get real values — the rest stay 0 (no importance data).
    """
    df = df.copy()
    df["home_net_squad_delta"] = 0.0
    df["away_net_squad_delta"] = 0.0
    for season in DELTA_SEASONS:
        per_club = compute_net_squad_delta(season=season, league="serie_a")
        if not per_club:
            log.warning("no delta for %s (transfers file missing?) — zeros", season)
            continue
        mask = df["season"] == season
        for prefix in ("home", "away"):
            teams = df.loc[mask, f"{prefix}_team"].astype(str).str.lower().str.strip()
            df.loc[mask, f"{prefix}_net_squad_delta"] = teams.map(
                lambda t: per_club.get(t, {}).get("net_squad_delta", 0.0)
            ).astype(float)
    df["net_squad_delta_diff"] = df["home_net_squad_delta"] - df["away_net_squad_delta"]
    return df


def _run_arm(df: pd.DataFrame, feature_names: list[str], label: str) -> pd.DataFrame:
    """Run the production walk-forward CV on a given feature list; return cv_df."""
    y_str = df["result"].copy()
    y = y_str.map(LABEL_MAP)
    if "_season" not in df.columns:
        df = df.copy()
        df["_season"] = df["season"]
    keep = [c for c in feature_names + ["_season"] if c in df.columns]
    X = df[keep].copy()
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        for f in missing:
            X[f] = 0.0
        log.warning("%s: zero-imputed %d missing features: %s", label, len(missing), missing[:6])
    cv_df = walk_forward_validate(X, y, y_str, BASE_PARAMS)
    return cv_df


def _last3(cv_df: pd.DataFrame) -> dict:
    last3 = cv_df.tail(3)
    return {
        "acc": round(last3["accuracy"].mean(), 4),
        "logloss": round(last3["log_loss"].mean(), 4),
        "brier": round(last3["brier_score"].mean(), 4),
        "draw_f1": round(last3["f1_D"].mean(), 4),
    }


def main() -> int:
    feature_names = load_current_feature_set()
    # the locked set may already list net_squad_delta_diff — strip it so the
    # baseline arm is truly without it, then add it only to the treatment arm.
    baseline_feats = [f for f in feature_names if f != "net_squad_delta_diff"]
    treatment_feats = baseline_feats + ["net_squad_delta_diff"]

    df = pd.read_parquet(features_path())
    min_season = ValidationConfig().min_train_season
    if min_season:
        df = df[df["season"] >= min_season].copy()
    # Drop rows whose result is not a real H/D/A outcome (e.g. "U" = unplayed):
    # LABEL_MAP has no "U", so y = result.map(LABEL_MAP) would be NaN and
    # compute_metrics rejects it. Production filters these upstream; match that.
    n_before = len(df)
    df = df[df["result"].isin(LABEL_MAP)].copy()
    dropped = n_before - len(df)
    if dropped:
        log.warning("dropped %d rows with non-H/D/A result (unplayed)", dropped)
    df = _attach_delta(df)

    nz = (df["net_squad_delta_diff"] != 0).sum()
    print(f"\nnet_squad_delta_diff nonzero on {nz}/{len(df)} rows "
          f"({100 * nz / len(df):.0f}%), seasons {DELTA_SEASONS[0]}..{DELTA_SEASONS[-1]}")

    print("\n=== BASELINE (locked features, no delta) ===")
    cv_base = _run_arm(df, baseline_feats, "baseline")
    print("\n=== TREATMENT (+ net_squad_delta_diff) ===")
    cv_treat = _run_arm(df, treatment_feats, "treatment")

    b, t = _last3(cv_base), _last3(cv_treat)
    d_acc = round(t["acc"] - b["acc"], 4)
    d_ll = round(t["logloss"] - b["logloss"], 4)   # negative = better
    d_brier = round(t["brier"] - b["brier"], 4)     # negative = better
    # skill vs baseline on the primary metric (log-loss): 1 - t_ll/b_ll
    skill = round(1 - (t["logloss"] / b["logloss"]), 4)

    print("\n" + "=" * 64)
    print("  RESULT — last-3-folds walk-forward (same folds, same params)")
    print("=" * 64)
    print(f"  metric      baseline   +delta     delta")
    print(f"  accuracy    {b['acc']:.4f}    {t['acc']:.4f}    {d_acc:+.4f}")
    print(f"  log-loss    {b['logloss']:.4f}    {t['logloss']:.4f}    {d_ll:+.4f}  (neg=better)")
    print(f"  brier       {b['brier']:.4f}    {t['brier']:.4f}    {d_brier:+.4f}  (neg=better)")
    print(f"  draw F1     {b['draw_f1']:.4f}    {t['draw_f1']:.4f}")
    print(f"\n  skill vs baseline (log-loss): {skill:+.4f}  "
          f"({'PASS >0' if skill > 0 else 'FAIL <=0'})")
    print("=" * 64)

    out = {
        "delta_seasons": DELTA_SEASONS,
        "nonzero_rows": int(nz),
        "total_rows": int(len(df)),
        "baseline_last3": b,
        "treatment_last3": t,
        "delta_accuracy": d_acc,
        "delta_logloss": d_ll,
        "delta_brier": d_brier,
        "skill_vs_baseline_logloss": skill,
        "verdict": "PASS" if skill > 0 else "FAIL",
    }
    outpath = MODELS_DIR / "universal" / "net_squad_delta_backtest.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
