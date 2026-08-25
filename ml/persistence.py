"""Model persistence: save and load trained models with metadata and versioning."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

from config.settings import MODELS_DIR
from ml.config import MODEL_EXTENSIONS

log = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def save_model(
    model: Any,
    variant: str,
    model_type: str,
    feature_names: list[str],
    metrics: Dict[str, Any] | None = None,
) -> Path:
    """Save model artifact + metadata JSON.

    Saves both a versioned copy (with timestamp) and a _latest symlink/copy.
    """
    save_dir = MODELS_DIR / variant
    save_dir.mkdir(parents=True, exist_ok=True)

    ext = MODEL_EXTENSIONS.get(model_type, ".pkl")
    ts = _timestamp()

    # Save versioned model
    versioned_path = save_dir / f"{model_type}_{ts}{ext}"
    model.save(str(versioned_path))

    # Save _latest (overwrite)
    latest_path = save_dir / f"{model_type}_latest{ext}"
    model.save(str(latest_path))

    saved_at = datetime.now(timezone.utc).isoformat()
    # Safely extract feature importance (not all wrappers support this)
    try:
        feat_imp = model.get_feature_importance()
    except (AttributeError, TypeError):
        feat_imp = {}
    metadata = {
        "model_type": model_type,
        "variant": variant,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "saved_at": saved_at,
        "version": ts,
        "metrics": metrics or {},
        "feature_importance": feat_imp,
    }

    # Save versioned metadata
    meta_versioned = save_dir / f"{model_type}_{ts}_metadata.json"
    with open(meta_versioned, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save _latest metadata (overwrite)
    meta_latest = save_dir / f"{model_type}_metadata.json"
    with open(meta_latest, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Saved %s (%s) v%s to %s", model_type, variant, ts, latest_path)
    return latest_path


def load_model(
    variant: str, model_type: str, version: str | None = None,
) -> Tuple[Any, Dict]:
    """Load model + metadata from disk.

    If version is None, loads the latest version.
    """
    from ml.models import get_model

    load_dir = MODELS_DIR / variant
    ext = MODEL_EXTENSIONS.get(model_type, ".pkl")

    if version:
        meta_path = load_dir / f"{model_type}_{version}_metadata.json"
        model_path = load_dir / f"{model_type}_{version}{ext}"
    else:
        meta_path = load_dir / f"{model_type}_metadata.json"
        model_path = load_dir / f"{model_type}_latest{ext}"

    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata at {meta_path}. Train first.")
    with open(meta_path) as f:
        metadata = json.load(f)

    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Train first.")

    wrapper = get_model(model_type)
    wrapper.load(str(model_path))
    wrapper.feature_names = metadata["feature_names"]

    log.info("Loaded %s (%s) v%s from %s",
             model_type, variant, metadata.get("version", "unknown"), model_path)
    return wrapper, metadata
