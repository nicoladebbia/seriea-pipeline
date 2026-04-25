"""Compare current production model vs Phase 5 walkforward model.

Both models are evaluated on the same 2022-2025 eval seasons. The current
prod model has in-sample contamination (trained on these seasons), so this
comparison is biased FOR the old model. If the new model still beats the
old, it's an unambiguous win. If close, the new model probably wins on
clean data.

Outputs:
    data/diagnostics/phase5_comparison.json  — full per-fold metrics
    Console table with key adoption metrics

Usage:
    PYTHONPATH=. python3 scripts/diagnostics/compare_production_vs_phase5.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import LABEL_MAP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models" / "universal"
EVAL_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]


def load_model_with_calibrator(model_name: str, cal_name: str | None) -> tuple:
    model = CatBoostClassifier()
    model.load_model(str(MODELS_DIR / model_name))
    cal = None
    if cal_name and (MODELS_DIR / cal_name).exists():
        with open(MODELS_DIR / cal_name, "rb") as fh:
            cal = pickle.load(fh)
        log.info("Loaded calibrator from %s", cal_name)
    return model, cal


def apply_calibration(proba: np.ndarray, cal: dict | None) -> np.ndarray:
    if cal is None:
        return proba
    out = np.zeros_like(proba)
    for cls in range(proba.shape[1]):
        out[:, cls] = cal[cls].predict(proba[:, cls])
    out = np.clip(out, 1e-6, 1 - 1e-6)
    return out / out.sum(axis=1, keepdims=True)


def evaluate(name: str, model: CatBoostClassifier, cal: dict | None,
             X: pd.DataFrame, y: np.ndarray) -> dict:
    feats = list(model.feature_names_)
    Xa = X.copy()
    for f in feats:
        if f not in Xa.columns:
            Xa[f] = 0.0
    Xa = Xa[feats].fillna(0.0)
    raw = model.predict_proba(Xa)
    cal_p = apply_calibration(raw, cal)
    raw_pred = raw.argmax(axis=1)
    cal_pred = cal_p.argmax(axis=1)
    metrics = {
        "name": name,
        "n": int(len(y)),
        "raw_acc": float(accuracy_score(y, raw_pred)),
        "raw_log_loss": float(log_loss(y, np.clip(raw, 1e-6, 1 - 1e-6), labels=[0, 1, 2])),
        "raw_brier_H": float(brier_score_loss((y == 0).astype(int), raw[:, 0])),
        "raw_brier_D": float(brier_score_loss((y == 1).astype(int), raw[:, 1])),
        "raw_brier_A": float(brier_score_loss((y == 2).astype(int), raw[:, 2])),
        "cal_acc": float(accuracy_score(y, cal_pred)),
        "cal_log_loss": float(log_loss(y, cal_p, labels=[0, 1, 2])),
        "cal_brier_H": float(brier_score_loss((y == 0).astype(int), cal_p[:, 0])),
        "cal_brier_D": float(brier_score_loss((y == 1).astype(int), cal_p[:, 1])),
        "cal_brier_A": float(brier_score_loss((y == 2).astype(int), cal_p[:, 2])),
        # ECE on calibrated probs (top-1 confidence vs accuracy)
        "cal_ece": float(_ece(y, cal_p)),
        "raw_ece": float(_ece(y, raw)),
    }
    metrics["raw_brier_avg"] = (metrics["raw_brier_H"] + metrics["raw_brier_D"] + metrics["raw_brier_A"]) / 3
    metrics["cal_brier_avg"] = (metrics["cal_brier_H"] + metrics["cal_brier_D"] + metrics["cal_brier_A"]) / 3
    return metrics


def _ece(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    n = len(y)
    ece = 0.0
    for i in range(bins):
        mask = (confidences >= edges[i]) & (confidences < edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == y[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def main(variant_suffix: str = "phase5_v1") -> None:
    df = pd.read_parquet(PROJECT_ROOT / "data" / "features" / "features.parquet")
    log.info("Loaded features: %s", df.shape)

    df = df[df["season"].isin(EVAL_SEASONS)].copy()
    log.info("Filtered to eval seasons %s: %d rows", EVAL_SEASONS, len(df))

    df = df.dropna(subset=["result"]).copy()
    y_str = df["result"]
    y = y_str.map(LABEL_MAP).values

    # Load both models
    log.info("Loading current production model...")
    prod_model, prod_cal = load_model_with_calibrator(
        "catboost_no_odds.cbm", "lean_calibrators.pkl",
    )
    log.info("Loading Phase 5 model (suffix=%s)...", variant_suffix)
    new_model, new_cal = load_model_with_calibrator(
        f"catboost_no_odds__{variant_suffix}.cbm",
        f"lean_calibrators__{variant_suffix}.pkl",
    )

    prod_metrics = evaluate("production (April 18, in-sample)", prod_model, prod_cal, df, y)
    new_metrics = evaluate(f"phase5_{variant_suffix} (walkforward)", new_model, new_cal, df, y)

    # Per-season breakdown
    per_season = {}
    for season in EVAL_SEASONS:
        mask = df["season"] == season
        if mask.sum() == 0:
            continue
        sub_y = y[mask]
        per_season[season] = {
            "production": evaluate(f"prod_{season}", prod_model, prod_cal, df[mask], sub_y),
            "phase5": evaluate(f"phase5_{season}", new_model, new_cal, df[mask], sub_y),
        }

    # Print headline
    print("\n" + "=" * 92)
    print(f"{'Metric':30s}  {'Production (in-sample)':>22s}  {'Phase 5 (walkforward)':>22s}  {'Δ':>10s}")
    print("=" * 92)
    for k in ["raw_acc", "raw_log_loss", "raw_brier_avg", "raw_ece",
              "cal_acc", "cal_log_loss", "cal_brier_avg", "cal_ece"]:
        p, n = prod_metrics[k], new_metrics[k]
        delta = n - p
        # For acc, higher is better; for log_loss/brier/ece, lower is better
        winner = ""
        if k.endswith("_acc"):
            winner = " ★new" if delta > 0.005 else (" ★prod" if delta < -0.005 else " ~tied")
        else:
            winner = " ★new" if delta < -0.005 else (" ★prod" if delta > 0.005 else " ~tied")
        print(f"{k:30s}  {p:>22.4f}  {n:>22.4f}  {delta:>+10.4f}{winner}")
    print("=" * 92)

    # Per-season
    print("\nPER-SEASON:")
    for season, both in per_season.items():
        print(f"\n  {season}: n={both['production']['n']}")
        print(f"    raw_log_loss   prod={both['production']['raw_log_loss']:.4f}  phase5={both['phase5']['raw_log_loss']:.4f}")
        print(f"    raw_acc        prod={both['production']['raw_acc']:.4f}  phase5={both['phase5']['raw_acc']:.4f}")
        print(f"    cal_log_loss   prod={both['production']['cal_log_loss']:.4f}  phase5={both['phase5']['cal_log_loss']:.4f}")
        print(f"    cal_acc        prod={both['production']['cal_acc']:.4f}  phase5={both['phase5']['cal_acc']:.4f}")

    # Save
    out_path = PROJECT_ROOT / "data" / "diagnostics" / "phase5_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "production": prod_metrics,
            f"phase5_{variant_suffix}": new_metrics,
            "per_season": per_season,
            "note": "Production model trained on all data through April 2026, including 2022-2025 eval seasons (in-sample). Phase 5 model uses walkforward last-fold (no in-sample on most-recent season).",
        }, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant-suffix", default="phase5_v1")
    args = p.parse_args()
    main(variant_suffix=args.variant_suffix)
