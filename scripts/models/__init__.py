"""Model training, hyperparameter tuning, and experimental architectures."""

import json
import logging
from typing import Dict, List

from config.settings import DATA_DIR

log = logging.getLogger(__name__)


def load_predictions() -> List[Dict]:
    """Load match predictions from all league files."""
    all_preds: List[Dict] = []
    for fname in ["predictions.json", "predictions_premier_league.json"]:
        p = DATA_DIR / "upcoming" / fname
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                preds = data.get("predictions", [])
                all_preds.extend(preds)
                log.info("Loaded %d predictions from %s", len(preds), fname)
            except Exception as e:
                log.warning("Failed to load %s: %s", fname, e)
    if not all_preds:
        log.warning("No predictions files found")
    return all_preds
