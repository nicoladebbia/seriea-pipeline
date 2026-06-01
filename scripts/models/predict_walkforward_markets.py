"""Per-match corners + cards predictions from the trained walk-forward CatBoost models.

Loads `season_2024-2025.cbm` for each over-line market, applies the per-class
isotonic calibrators, and writes per-match probabilities to
`data/upcoming/corners_predictions.json` and `data/upcoming/cards_predictions.json`.

Replaces the constant-broadcast fallback in `comprehensive_markets.predict_all_markets`
(which silently emitted `expected_corners=10.0` for every match because the
`data/models/markets/` directory was emptied during the Apr 27 cleanup).

Run:
    python3 -m scripts.models.predict_walkforward_markets
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WF_DIR = DATA_DIR / "models" / "walkforward"
UPCOMING_DIR = DATA_DIR / "upcoming"

# Markets per league. Add EPL/other-league markets here as they get trained.
LEAGUE_MARKETS: dict[str, dict[str, list[float]]] = {
    "serie_a": {
        "corners": [8.5, 9.5, 10.5],
        "cards": [3.5, 4.5, 5.5],
    },
}

# Use the most recent fold for production prediction (trained on the most data).
_PROD_FOLD = "season_2024-2025"


@dataclass
class _MarketModel:
    line: float
    booster: CatBoostClassifier
    calibrators: list  # list of one IsotonicRegression for binary markets
    feature_names: list[str]


def _load_market_models(league: str, market: str, lines: list[float]) -> list[_MarketModel]:
    """Load CatBoost + isotonic calibrators for every over-line of one market."""
    out: list[_MarketModel] = []
    for line in lines:
        market_key = f"{market}_over_{str(line).replace('.', '_')}"
        market_dir = WF_DIR / league / market_key
        cbm_path = market_dir / f"{_PROD_FOLD}.cbm"
        cal_path = market_dir / f"{_PROD_FOLD}_calibrators.pkl"
        meta_path = market_dir / f"{_PROD_FOLD}_metadata.json"
        if not cbm_path.exists() or not cal_path.exists() or not meta_path.exists():
            log.warning("Skipping %s/%s: artifacts missing under %s",
                        league, market_key, market_dir)
            continue
        booster = CatBoostClassifier()
        booster.load_model(str(cbm_path))
        with open(cal_path, "rb") as fh:
            calibrators = pickle.load(fh)
        with open(meta_path) as fh:
            meta = json.load(fh)
        out.append(_MarketModel(
            line=line,
            booster=booster,
            calibrators=calibrators,
            feature_names=list(meta["feature_names"]),
        ))
    return out


def _align_row(row_dict: dict, feature_names: list[str]) -> pd.DataFrame:
    """Align an upcoming-feature row to the columns this model expects."""
    aligned = {f: row_dict.get(f, np.nan) for f in feature_names}
    return pd.DataFrame([aligned])[feature_names]


def _predict_calibrated(model: _MarketModel, X: pd.DataFrame) -> float:
    """Return the calibrated P(yes) for a binary market on a single row."""
    proba = model.booster.predict_proba(X)
    p_pos = proba[:, 1]
    p_pos_cal = model.calibrators[0].predict(p_pos)
    p_pos_cal = np.clip(p_pos_cal, 1e-4, 1 - 1e-4)
    return float(p_pos_cal[0])


def _fit_lambda_to_overs(line_to_prob: dict[float, float]) -> float:
    """Given calibrated P(over X.5) for several lines, fit one Poisson rate λ.

    Minimizes squared error between Poisson tail probabilities and the
    classifier-derived probabilities. This converts three calibrated binary
    outputs into one count-style "expected total" without inventing a
    separate regressor.
    """
    if not line_to_prob:
        return float("nan")
    lines = np.array(list(line_to_prob.keys()), dtype=float)
    probs = np.array(list(line_to_prob.values()), dtype=float)

    def loss(lam: float) -> float:
        # P(N > k) where line = k + 0.5
        ks = lines - 0.5
        pred = 1.0 - poisson.cdf(ks.astype(int), lam)
        return float(np.sum((pred - probs) ** 2))

    res = minimize_scalar(loss, bounds=(0.5, 25.0), method="bounded")
    return float(res.x) if res.success else float("nan")


def _load_upcoming_fixtures() -> list[dict]:
    """Pull SA upcoming fixtures from data/upcoming/matches.json.

    The file lacks a `league` tag; fall back to inferring per match.
    """
    from config.leagues import infer_league
    path = UPCOMING_DIR / "matches.json"
    raw = json.loads(path.read_text())
    matches = raw.get("matches", [])
    out = []
    for m in matches:
        if not m.get("home_team") or not m.get("away_team"):
            continue
        league = m.get("league") or infer_league(m["home_team"], m["away_team"])
        m["league"] = league
        out.append(m)
    return out


def _merge_into_existing(path: Path, fresh: list[dict]) -> list[dict]:
    """Replace per-match entries with the fresh ones, keep entries we didn't touch.

    Mirrors `_merge_predictions_by_match` in `ml_market_predictions.py` so we
    don't wipe out the EPL or other-league entries when overwriting SA.
    """
    if not path.exists():
        return fresh
    try:
        existing = json.loads(path.read_text())
        existing_preds = existing.get("predictions", []) if isinstance(existing, dict) else existing
    except Exception:
        return fresh
    fresh_keys = {(p.get("match"), p.get("date")) for p in fresh}
    kept = [p for p in existing_preds
            if (p.get("match"), p.get("date")) not in fresh_keys]
    return kept + fresh


def _load_features_for_fixtures(
    fixtures: list[dict], league: str, fast_only: bool = False,
) -> pd.DataFrame:
    """Resolve feature rows for the given fixtures.

    Fast path: read pre-built rows from `data/features/features_{league}.parquet`.
    Fixtures inserted upstream by `inject_upcoming_fixtures.py` will be there
    after the nightly features build runs (~30 min, but only once per day).

    Slow path: if any fixture is missing from the cached features parquet,
    fall back to `features.build.build_upcoming_features()` — the full
    on-demand pipeline. This is heavy (10-30 min, can OOM on 16 GB) and
    should only ever fire when the user calls the predictor faster than the
    nightly job can run.

    If `fast_only=True`, the slow path is skipped and the function returns
    only fixtures already present in the cached features parquet (callers
    can use this for crash-free demo runs before the nightly build catches up).
    """
    feats_path = DATA_DIR / "features" / f"features_{league}.parquet"
    fix_keys = {
        (f["home_team"], f["away_team"], pd.Timestamp(f["date"]))
        for f in fixtures
    }

    if feats_path.exists():
        feats = pd.read_parquet(feats_path)
        feats["match_date"] = pd.to_datetime(feats["match_date"])
        present = feats[
            feats[["home_team", "away_team", "match_date"]]
            .apply(tuple, axis=1)
            .isin(fix_keys)
        ].copy()
        if len(present) == len(fix_keys):
            log.info("Fast path: all %d fixtures found in %s",
                     len(present), feats_path.name)
            return present.reset_index(drop=True)
        missing = len(fix_keys) - len(present)
        if fast_only:
            log.warning("Fast-only mode: %d/%d fixtures missing from %s — "
                        "skipping them. Run scripts/data/inject_upcoming_fixtures.py "
                        "+ features build to fill the cache.",
                        missing, len(fix_keys), feats_path.name)
            return present.reset_index(drop=True)
        log.warning("Fast path incomplete: %d fixtures missing from %s — "
                    "falling back to on-demand pipeline build (slow). "
                    "Run scripts/data/inject_upcoming_fixtures.py + features "
                    "build to populate the cache.",
                    missing, feats_path.name)

    if fast_only:
        log.error("Fast-only mode but no features parquet at %s. "
                  "Cannot predict — bail.", feats_path)
        return pd.DataFrame()

    log.warning("Slow path: building features on-demand for %d fixtures. "
                "This can take 15-30 minutes and may OOM on <32 GB RAM.",
                len(fix_keys))
    from features.build import build_upcoming_features
    fix_df = pd.DataFrame([
        {
            "home_team": f["home_team"],
            "away_team": f["away_team"],
            "match_date": pd.Timestamp(f["date"]),
            "season": f.get("season") or "2025-2026",
        }
        for f in fixtures
    ])
    return build_upcoming_features(fix_df, league=league)


def predict_walkforward_markets(league: str = "serie_a", fast_only: bool = False) -> dict:
    """Run all configured walkforward markets for one league's upcoming fixtures.

    Returns a dict {"corners": [...], "cards": [...]} of per-match entries
    matching the existing JSON schemas in data/upcoming/.
    """
    markets = LEAGUE_MARKETS.get(league)
    if not markets:
        log.warning("No walkforward markets configured for league=%s", league)
        return {}

    all_fixtures = _load_upcoming_fixtures()
    fixtures = [m for m in all_fixtures if m["league"] == league]
    if not fixtures:
        log.warning("No upcoming fixtures for league=%s", league)
        return {}
    log.info("Predicting %d markets x %d fixtures for league=%s",
             sum(len(v) for v in markets.values()), len(fixtures), league)

    feats = _load_features_for_fixtures(fixtures, league, fast_only=fast_only)
    if feats.empty:
        log.warning("No feature rows resolved — nothing to predict for %s", league)
        return {"corners": [], "cards": []}
    log.info("Got %d feature rows (cols=%d)", len(feats), feats.shape[1])

    # Pre-load every model once.
    loaded: dict[str, list[_MarketModel]] = {
        name: _load_market_models(league, name, lines)
        for name, lines in markets.items()
    }

    feats["_match_key"] = list(zip(feats["home_team"], feats["away_team"], feats["match_date"]))

    corners_out: list[dict] = []
    cards_out: list[dict] = []

    for f in fixtures:
        key = (f["home_team"], f["away_team"], pd.Timestamp(f["date"]))
        feat_rows = feats[feats["_match_key"] == key]
        if feat_rows.empty:
            log.warning("No feature row for %s vs %s on %s — skipping",
                        f["home_team"], f["away_team"], f["date"])
            continue
        row_dict = feat_rows.iloc[0].to_dict()

        # ---- corners ----
        corners_models = loaded.get("corners", [])
        if corners_models:
            line_probs: dict[float, float] = {}
            for mdl in corners_models:
                X = _align_row(row_dict, mdl.feature_names)
                line_probs[mdl.line] = _predict_calibrated(mdl, X)
            lam = _fit_lambda_to_overs(line_probs)
            entry = {
                "match": f"{f['home_team']} vs {f['away_team']}",
                "league": league,
                "date": f["date"],
                "expected_corners": round(lam, 2) if not np.isnan(lam) else None,
                "expected_home_corners": None,  # walkforward model is total-only
                "expected_away_corners": None,
                "source": "walkforward_catboost",
                "model_fold": _PROD_FOLD,
            }
            for line, p in line_probs.items():
                key_name = f"over_{str(line).replace('.', '_')}"
                entry[key_name] = round(p, 4)
            corners_out.append(entry)

        # ---- cards ----
        cards_models = loaded.get("cards", [])
        if cards_models:
            line_probs = {}
            for mdl in cards_models:
                X = _align_row(row_dict, mdl.feature_names)
                line_probs[mdl.line] = _predict_calibrated(mdl, X)
            lam = _fit_lambda_to_overs(line_probs)
            # Per-team eagerness: team's avg yellows over last 5 matches +
            # referee bias for that side (positive = ref tends to card this team
            # more than league average). Honest derivation from training-time
            # rolling features; not a separate model.
            home_roll5 = row_dict.get("home_roll_5_yellow_cards")
            away_roll5 = row_dict.get("away_roll_5_yellow_cards")
            ref_home_bias = row_dict.get("ref_home_team_cards") or 0.0
            ref_away_bias = row_dict.get("ref_away_team_cards") or 0.0
            home_eager = (
                round(float(home_roll5) + float(ref_home_bias) * 0.2, 2)
                if home_roll5 is not None and not pd.isna(home_roll5) else None
            )
            away_eager = (
                round(float(away_roll5) + float(ref_away_bias) * 0.2, 2)
                if away_roll5 is not None and not pd.isna(away_roll5) else None
            )
            entry = {
                "match": f"{f['home_team']} vs {f['away_team']}",
                "league": league,
                "date": f["date"],
                "expected_cards": round(lam, 2) if not np.isnan(lam) else None,
                "expected_home_cards": home_eager,
                "expected_away_cards": away_eager,
                "home_card_eagerness": home_eager,
                "away_card_eagerness": away_eager,
                "source": "walkforward_catboost",
                "model_fold": _PROD_FOLD,
            }
            for line, p in line_probs.items():
                key_name = f"over_{str(line).replace('.', '_')}"
                entry[key_name] = round(p, 4)
            cards_out.append(entry)

    return {"corners": corners_out, "cards": cards_out}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-match corners + cards predictions from the trained "
                    "walkforward CatBoost models. Writes per-match probabilities "
                    "to data/upcoming/corners_predictions.json and "
                    "data/upcoming/cards_predictions.json."
    )
    parser.add_argument(
        "--fast-only", action="store_true",
        help="Skip the slow on-demand feature build. Predicts only fixtures "
             "already present in features_serie_a.parquet (typically the ones "
             "the nightly job has already processed).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    UPCOMING_DIR.mkdir(parents=True, exist_ok=True)

    all_corners: list[dict] = []
    all_cards: list[dict] = []
    for league in LEAGUE_MARKETS:
        out = predict_walkforward_markets(league, fast_only=args.fast_only)
        all_corners.extend(out.get("corners", []))
        all_cards.extend(out.get("cards", []))

    if all_corners:
        path = UPCOMING_DIR / "corners_predictions.json"
        merged = _merge_into_existing(path, all_corners)
        path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": merged,
        }, indent=2))
        log.info("Wrote %s (%d total entries; %d fresh from walkforward)",
                 path, len(merged), len(all_corners))

    if all_cards:
        path = UPCOMING_DIR / "cards_predictions.json"
        merged = _merge_into_existing(path, all_cards)
        path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": merged,
        }, indent=2))
        log.info("Wrote %s (%d total entries; %d fresh from walkforward)",
                 path, len(merged), len(all_cards))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
