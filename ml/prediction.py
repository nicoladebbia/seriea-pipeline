"""Predict upcoming match outcomes using trained models."""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from ml.config import CLASS_LABELS, LABEL_NAMES, MODEL_TYPES, N_CLASSES
from ml.persistence import load_model

log = logging.getLogger(__name__)


def predict_upcoming(
    upcoming: pd.DataFrame,
    variant: str = "universal",
    model_type: str = "xgboost",
) -> pd.DataFrame:
    """Predict outcomes for upcoming matches.

    Parameters
    ----------
    upcoming : DataFrame
        Must contain the same feature columns the model was trained on,
        plus 'home_team' and 'away_team' for display.
    variant : str
        Model variant ('universal' or 'rich').
    model_type : str
        Model type ('xgboost' or 'lightgbm').

    Returns
    -------
    DataFrame with columns: home_team, away_team, prob_H, prob_D, prob_A,
    predicted, confidence.
    """
    wrapper, metadata = load_model(variant, model_type)
    feature_names = metadata["feature_names"]

    # Ensure all expected features exist; fill missing with NaN
    missing = set(feature_names) - set(upcoming.columns)
    if missing:
        log.warning("%d features missing from input; filling with NaN: %s",
                    len(missing), sorted(missing)[:10])
        for col in missing:
            upcoming[col] = np.nan

    X = upcoming[feature_names]
    y_proba = wrapper.predict_proba(X)

    results = pd.DataFrame({
        "home_team": upcoming["home_team"].values if "home_team" in upcoming.columns else range(len(X)),
        "away_team": upcoming["away_team"].values if "away_team" in upcoming.columns else range(len(X)),
    })

    # Dynamically build prob columns from CLASS_LABELS
    for idx, cls in enumerate(CLASS_LABELS):
        results[f"prob_{cls}"] = y_proba[:, idx]

    results["predicted"] = np.argmax(y_proba, axis=1)
    results["predicted"] = results["predicted"].map(LABEL_NAMES)
    results["confidence"] = np.max(y_proba, axis=1)

    return results


def predict_with_ensemble(
    upcoming: pd.DataFrame,
    variant: str = "universal",
    model_types: List[str] | None = None,
    weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Ensemble prediction averaging probabilities across models.

    Parameters
    ----------
    upcoming : DataFrame
        Match feature data.
    variant : str
        Model variant.
    model_types : list
        Model types to ensemble (default: all MODEL_TYPES).
    weights : dict
        Optional model_type -> weight mapping (default: equal weights).
    """
    model_types = model_types or list(MODEL_TYPES)
    weights = weights or {mt: 1.0 / len(model_types) for mt in model_types}

    all_probas = []

    for mt in model_types:
        try:
            wrapper, metadata = load_model(variant, mt)
        except FileNotFoundError:
            log.warning("Model %s/%s not found; skipping", variant, mt)
            continue

        feature_names = metadata["feature_names"]
        missing = set(feature_names) - set(upcoming.columns)
        for col in missing:
            upcoming[col] = np.nan

        X = upcoming[feature_names]
        y_proba = wrapper.predict_proba(X)
        all_probas.append((mt, y_proba))

    if not all_probas:
        raise FileNotFoundError("No trained models found for ensemble.")

    # Weighted average
    total_weight = sum(weights.get(mt, 0) for mt, _ in all_probas)
    avg_proba = sum(
        weights.get(mt, 1.0 / len(all_probas)) / total_weight * proba
        for mt, proba in all_probas
    )

    results = pd.DataFrame({
        "home_team": upcoming["home_team"].values if "home_team" in upcoming.columns else range(len(upcoming)),
        "away_team": upcoming["away_team"].values if "away_team" in upcoming.columns else range(len(upcoming)),
    })

    for idx, cls in enumerate(CLASS_LABELS):
        results[f"prob_{cls}"] = avg_proba[:, idx]

    results["predicted"] = np.argmax(avg_proba, axis=1)
    results["predicted"] = results["predicted"].map(LABEL_NAMES)
    results["confidence"] = np.max(avg_proba, axis=1)

    # Add per-model probabilities for transparency
    for mt, proba in all_probas:
        for idx, cls in enumerate(CLASS_LABELS):
            results[f"{mt}_prob_{cls}"] = proba[:, idx]

    return results


def format_predictions(preds: pd.DataFrame) -> str:
    """Format prediction DataFrame as a readable table."""
    lines = [
        f"{'Match':<35} {'Pred':>4} {'Conf':>5}  {'%H':>5} {'%D':>5} {'%A':>5}",
        "-" * 70,
    ]
    for _, row in preds.iterrows():
        match_name = f"{row['home_team']} vs {row['away_team']}"
        lines.append(
            f"{match_name:<35} {row['predicted']:>4} {row['confidence']:>5.1%}"
            f"  {row['prob_H']:>5.1%} {row['prob_D']:>5.1%} {row['prob_A']:>5.1%}"
        )
    return "\n".join(lines)
