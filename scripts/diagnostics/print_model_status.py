#!/usr/bin/env python3
"""Print live model performance from production metadata.

Single source of truth: data/models/universal/catboost_no_odds_metadata.json.
Never edit numbers in markdown — run this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA = PROJECT_ROOT / "data" / "models" / "universal" / "catboost_no_odds_metadata.json"

REALISTIC_CEILING = 0.55  # Pinnacle closing line / academic SOTA


def main() -> int:
    if not METADATA.exists():
        print(f"ERROR: {METADATA} not found. Has the model been trained?", file=sys.stderr)
        return 1

    with METADATA.open() as f:
        meta = json.load(f)

    cv = meta.get("cv_summary", {})
    metrics = meta.get("metrics", {})
    variant = meta.get("variant", "unknown")
    saved_at = meta.get("saved_at", "unknown")
    n_features = meta.get("n_features", "?")

    last3_acc = cv.get("last3_accuracy")
    last3_ll = cv.get("last3_logloss")
    last3_brier = cv.get("last3_brier")
    all_acc = cv.get("all_folds_accuracy")
    all_ll = cv.get("all_folds_logloss")
    ece = metrics.get("ece")
    kelly = metrics.get("kelly_roi")
    held_acc = metrics.get("accuracy")

    print()
    print("=" * 64)
    print("  MODEL STATUS — read from production metadata")
    print("=" * 64)
    print(f"  Variant      : {variant}")
    print(f"  Features     : {n_features}")
    print(f"  Saved at     : {saved_at}")
    print()
    print("  Walk-forward CV (last 3 folds — primary)")
    print("  " + "-" * 60)
    if last3_acc is not None:
        gate = "PASS" if last3_acc >= REALISTIC_CEILING else "below ceiling"
        print(f"    Accuracy   : {last3_acc:.2%}   ({gate} vs {REALISTIC_CEILING:.0%} realistic ceiling)")
    if last3_ll is not None:
        print(f"    Log-loss   : {last3_ll:.4f}   (random = 1.0986; lower is better)")
    if last3_brier is not None:
        print(f"    Brier      : {last3_brier:.4f}")
    print()
    print("  Walk-forward CV (all folds)")
    print("  " + "-" * 60)
    if all_acc is not None:
        print(f"    Accuracy   : {all_acc:.2%}")
    if all_ll is not None:
        print(f"    Log-loss   : {all_ll:.4f}")
    print()
    print("  Held-out fold")
    print("  " + "-" * 60)
    if held_acc is not None:
        print(f"    Accuracy   : {held_acc:.2%}")
    if ece is not None:
        print(f"    ECE        : {ece:.4f}   (calibration error; lower is better)")
    if kelly is not None:
        print(f"    Kelly ROI  : {kelly:+.2f}%   (noisy at small N)")
    print()
    print("=" * 64)
    print("  Reality check: 1X2 academic ceiling is 53-55%. Any markdown")
    print("  file claiming 60%+ is fiction. Trust this output, not docs.")
    print("=" * 64)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
