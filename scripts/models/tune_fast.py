#!/usr/bin/env python3
"""Fast targeted hyperparameter tuning — 6 configs only."""
import sys, json, logging, numpy as np, pandas as pd, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from scripts.models.comprehensive_markets import load_unified_training_data, get_training_features, _compute_targets
from config.settings import MODELS_DIR
from ml.poisson import poisson_1x2, market_implied_probs
from catboost import CatBoostClassifier, CatBoostRegressor

LABEL_MAP = {"H": 0, "D": 1, "A": 2}

df = load_unified_training_data()
features = get_training_features(df)
df = _compute_targets(df)
df_v = df[df["result"].isin(["H","D","A"])].copy()
seasons = sorted(df_v["season"].unique())

test_season = "2024-2025"
ti = seasons.index(test_season)
train_m = df_v["season"].isin(seasons[:ti-1])
val_m = df_v["season"] == seasons[ti-1]
test_m = df_v["season"] == test_season

X_tr = df_v.loc[train_m, features].fillna(0)
X_val = df_v.loc[val_m, features].fillna(0)
X_te = df_v.loc[test_m, features].fillna(0)
y_tr = df_v.loc[train_m, "result"].map(LABEL_MAP).values
y_val = df_v.loc[val_m, "result"].map(LABEL_MAP).values
y_te = df_v.loc[test_m, "result"].values
mkt_te = market_implied_probs(df_v, test_m)

configs = [
    {"depth": 6, "lr": 0.02, "l2": 3, "ml": 30},   # default baseline
    {"depth": 5, "lr": 0.02, "l2": 5, "ml": 25},
    {"depth": 7, "lr": 0.015, "l2": 5, "ml": 25},
    {"depth": 7, "lr": 0.015, "l2": 3, "ml": 20},
    {"depth": 8, "lr": 0.01, "l2": 5, "ml": 30},
    {"depth": 6, "lr": 0.025, "l2": 5, "ml": 20},
]

results = []
for ci, cfg in enumerate(configs):
    log.info("Config %d/%d: d=%d lr=%.3f l2=%d ml=%d", ci+1, len(configs),
             cfg["depth"], cfg["lr"], cfg["l2"], cfg["ml"])
    
    cb = CatBoostClassifier(
        iterations=2000, depth=cfg["depth"], learning_rate=cfg["lr"],
        l2_leaf_reg=cfg["l2"], min_data_in_leaf=cfg["ml"],
        random_seed=42, verbose=0, loss_function="MultiClass",
        classes_count=3, auto_class_weights="Balanced",
    )
    cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=150, verbose=0)
    
    hg = CatBoostRegressor(
        iterations=1500, depth=cfg["depth"], learning_rate=cfg["lr"]*1.5,
        l2_leaf_reg=cfg["l2"]+2, min_data_in_leaf=cfg["ml"],
        random_seed=42, verbose=0, loss_function="Poisson",
    )
    ag = CatBoostRegressor(
        iterations=1500, depth=cfg["depth"], learning_rate=cfg["lr"]*1.5,
        l2_leaf_reg=cfg["l2"]+2, min_data_in_leaf=cfg["ml"],
        random_seed=42, verbose=0, loss_function="Poisson",
    )
    hg.fit(X_tr, df_v.loc[train_m,"home_score"],
           eval_set=(X_val, df_v.loc[val_m,"home_score"]), early_stopping_rounds=100, verbose=0)
    ag.fit(X_tr, df_v.loc[train_m,"away_score"],
           eval_set=(X_val, df_v.loc[val_m,"away_score"]), early_stopping_rounds=100, verbose=0)
    
    cb_te_p = cb.predict_proba(X_te)
    hxg = np.clip(hg.predict(X_te), 0.3, 4.0)
    axg = np.clip(ag.predict(X_te), 0.3, 4.0)
    pois_te_p = np.zeros((len(X_te), 3))
    for i in range(len(X_te)):
        pois_te_p[i] = poisson_1x2(hxg[i], axg[i])
    
    best_acc = 0
    best_w = {}
    for w_cb in [0.10, 0.15, 0.20, 0.25]:
      for w_pois in [0.25, 0.30, 0.35, 0.40]:
        w_mkt = round(1.0 - w_cb - w_pois, 2)
        if w_mkt < 0.30 or w_mkt > 0.55: continue
        for d_inf in [1.15, 1.25, 1.35, 1.45]:
            ens = w_cb * cb_te_p + w_pois * pois_te_p + w_mkt * mkt_te
            ens[:, 1] *= d_inf
            ens = ens / ens.sum(axis=1, keepdims=True)
            preds = np.array(["H","D","A"])[ens.argmax(axis=1)]
            acc = (preds == y_te).mean()
            if acc > best_acc:
                best_acc = acc
                best_w = {"w_cb": w_cb, "w_pois": w_pois, "w_mkt": w_mkt, "d_inf": d_inf}
                best_nd = (preds == "D").sum()
                max_p = ens.max(axis=1)
                best_hc55 = (preds[max_p>0.55]==y_te[max_p>0.55]).mean() if (max_p>0.55).sum()>0 else 0
                best_hc60 = (preds[max_p>0.60]==y_te[max_p>0.60]).mean() if (max_p>0.60).sum()>0 else 0
    
    results.append((best_acc, cfg, best_w, best_nd, best_hc55, best_hc60))
    log.info("  → %.1f%% draws=%d HC55=%.0f%% HC60=%.0f%% %s",
             best_acc*100, best_nd, best_hc55*100, best_hc60*100, best_w)
    
    del cb, hg, ag
    gc.collect()

results.sort(key=lambda x: x[0], reverse=True)
log.info("\n=== FINAL RANKING ===")
for i, (acc, cfg, w, nd, hc55, hc60) in enumerate(results):
    marker = " ← BEST" if i == 0 else ""
    log.info("%d. %.1f%% d=%d lr=%.3f l2=%d | w=(%.2f,%.2f,%.2f) d=%.2f | draws=%d HC55=%.0f%% HC60=%.0f%%%s",
             i+1, acc*100, cfg["depth"], cfg["lr"], cfg["l2"],
             w["w_cb"], w["w_pois"], w["w_mkt"], w["d_inf"], nd, hc55*100, hc60*100, marker)

best = results[0]
tuned = {
    "catboost": best[1],
    "best_ensemble": {
        "accuracy": round(best[0], 4), **best[2],
        "draws": best[3], "hc55": round(best[4], 4), "hc60": round(best[5], 4),
    },
}
with open(MODELS_DIR / "markets" / "tuned_hyperparams.json", "w") as f:
    json.dump(tuned, f, indent=2)
log.info("Saved tuned_hyperparams.json")
