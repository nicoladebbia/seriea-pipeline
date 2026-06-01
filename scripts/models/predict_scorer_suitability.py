"""Per-team top-scorer suitability for upcoming fixtures.

Reads the predicted XI from `data/upcoming/lineup_predictions.json`, runs the
existing `player_goalscorer.cbm` (with isotonic calibrator) on each starter,
and writes per-team top-3 scorers + their goal probabilities to
`data/upcoming/scorers_predictions.json`.

Reuses `scripts.betting.player_predictions.{load_player_data,
build_player_features}` to assemble the player-rolling features (~190 s for
all 100k player-matches) but calls the CatBoost goalscorer model directly
to dodge a Platt pickle / sklearn version mismatch in `predict_player_markets`.

Run:
    python3 -m scripts.models.predict_scorer_suitability
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

UPCOMING_DIR = DATA_DIR / "upcoming"
LINEUP_PATH = UPCOMING_DIR / "lineup_predictions.json"
PLAYER_STATS_PATH = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
PLAYER_STATS_EPL_PATH = DATA_DIR / "external" / "sofascore" / "player_match_stats_premier_league.parquet"
OUT_PATH = UPCOMING_DIR / "scorers_predictions.json"


def _load_name_to_id_map(league: str) -> dict[str, int]:
    """Latest player_id seen for each player_name. Most-recent wins for transfers."""
    paths = {
        "serie_a": PLAYER_STATS_PATH,
        "premier_league": PLAYER_STATS_EPL_PATH,
    }
    p = paths.get(league)
    if not p or not p.exists():
        log.warning("No player_match_stats for %s at %s", league, p)
        return {}
    df = pd.read_parquet(p, columns=["player_id", "player_name", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").drop_duplicates("player_name", keep="last")
    return dict(zip(df["player_name"].astype(str), df["player_id"].astype(int)))


def _team_top_scorers(
    team: str, opponent: str, lineup: list[dict], is_home: bool,
    name_to_id: dict[str, int], pms: pd.DataFrame,
    goalscorer_model, isotonic_calibrator, feature_cols: list[str],
    top_n: int = 3,
) -> list[dict]:
    """Return top-N most-likely goalscorers from this team's predicted XI.

    Bypasses `predict_player_markets` to avoid its Platt-pickle sklearn-version
    bug. We call the CatBoost model directly and use the isotonic calibrator
    (saved to player_goalscorer_iso.pkl) for the probability calibration.
    """
    starters = []
    for player in lineup:
        name = player.get("name") or player.get("player_name")
        pos = player.get("position", "M")
        if not name or pos == "G":
            continue  # skip GKs
        pid = name_to_id.get(name)
        if not pid:
            log.debug("No player_id for %s on %s", name, team)
            continue

        # Pull the player's most-recent feature row
        player_df = pms[pms["player_id"] == pid].sort_values("date")
        if len(player_df) == 0:
            continue
        latest = player_df.iloc[-1]

        # Build aligned feature vector for the goalscorer model
        row = {}
        for f in feature_cols:
            if f in latest.index and pd.notna(latest[f]):
                row[f] = float(latest[f])
            else:
                row[f] = 0.0
        # Override contextual features for THIS upcoming match
        row["is_forward"] = int(pos == "F")
        row["is_midfielder"] = int(pos == "M")
        row["is_defender"] = int(pos == "D")
        row["is_goalkeeper"] = 0
        row["is_home_int"] = int(is_home)

        # Inject opponent's latest rolling stats
        opp_df = pms[pms["team"] == opponent].sort_values("date")
        if len(opp_df) > 0:
            opp_latest = opp_df.iloc[-1]
            for f in feature_cols:
                if f.startswith("opp_r5_") and f in opp_latest.index and pd.notna(opp_latest[f]):
                    row[f] = float(opp_latest[f])

        try:
            X = pd.DataFrame([row])[feature_cols].fillna(0)
            raw_prob = float(goalscorer_model.predict_proba(X)[0][1])
            # Apply isotonic if available; fall back to raw
            if isotonic_calibrator is not None:
                prob = float(isotonic_calibrator.predict([raw_prob])[0])
                prob = float(np.clip(prob, 1e-4, 1 - 1e-4))
            else:
                prob = raw_prob
        except Exception as e:
            log.debug("Goalscorer prediction failed for %s: %s", name, e)
            continue

        starters.append({
            "player": name,
            "position": pos,
            "goal_prob": round(float(prob), 4),
        })

    starters.sort(key=lambda x: x["goal_prob"], reverse=True)
    return starters[:top_n]


def predict_scorers_for_league(league: str = "serie_a") -> list[dict]:
    """Generate per-fixture top-scorer entries for one league."""
    if not LINEUP_PATH.exists():
        log.warning("No lineup_predictions.json at %s", LINEUP_PATH)
        return []

    import pickle
    from catboost import CatBoostClassifier
    from scripts.betting.player_predictions import (
        load_player_data, build_player_features, get_feature_cols, MODEL_DIR,
    )
    from config.leagues import infer_league

    raw = json.loads(LINEUP_PATH.read_text())
    matches = raw.get("matches", {})
    if not matches:
        return []

    log.info("Loading player_match_stats for league=%s", league)
    pms = load_player_data()
    pms = build_player_features(pms)
    name_to_id = _load_name_to_id_map(league)
    log.info("name→id map: %d entries", len(name_to_id))

    # Load goalscorer model + isotonic calibrator once. We bypass
    # predict_player_markets because its Platt pickle is incompatible with
    # the installed sklearn (saved on 1.8, runtime 1.6 lacks `multi_class`).
    gs_path = MODEL_DIR / "player_goalscorer.cbm"
    iso_path = MODEL_DIR / "player_goalscorer_iso.pkl"
    if not gs_path.exists():
        log.error("player_goalscorer.cbm not found at %s — bail", gs_path)
        return []
    gs_model = CatBoostClassifier()
    gs_model.load_model(str(gs_path))
    gs_iso = None
    if iso_path.exists():
        try:
            with open(iso_path, "rb") as fh:
                gs_iso = pickle.load(fh)
            log.info("Loaded isotonic calibrator (%s)", type(gs_iso).__name__)
        except Exception as e:
            log.warning("Failed to load %s: %s — using raw probabilities", iso_path, e)
    feature_cols = list(gs_model.feature_names_)
    log.info("goalscorer model: %d features", len(feature_cols))

    out = []
    for match_name, match in matches.items():
        home = match.get("home_team")
        away = match.get("away_team")
        if not home or not away:
            continue
        match_league = infer_league(home, away) or "serie_a"
        if match_league != league:
            continue

        home_xi = match.get("home_lineup", {}).get("predicted_xi", [])
        away_xi = match.get("away_lineup", {}).get("predicted_xi", [])
        if not home_xi or not away_xi:
            log.debug("No predicted XI for %s — skipping", match_name)
            continue

        home_top = _team_top_scorers(
            home, away, home_xi, is_home=True,
            name_to_id=name_to_id, pms=pms,
            goalscorer_model=gs_model, isotonic_calibrator=gs_iso,
            feature_cols=feature_cols,
        )
        away_top = _team_top_scorers(
            away, home, away_xi, is_home=False,
            name_to_id=name_to_id, pms=pms,
            goalscorer_model=gs_model, isotonic_calibrator=gs_iso,
            feature_cols=feature_cols,
        )

        out.append({
            "match": match_name,
            "league": league,
            "date": match.get("date", ""),
            "home_team": home,
            "away_team": away,
            "home_top_scorers": home_top,
            "away_top_scorers": away_top,
            "source": "player_goalscorer_cbm",
        })
        log.info(
            "  %s: home top = %s | away top = %s",
            match_name,
            ", ".join(f"{p['player']}({p['goal_prob']:.2f})" for p in home_top) or "—",
            ", ".join(f"{p['player']}({p['goal_prob']:.2f})" for p in away_top) or "—",
        )

    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    UPCOMING_DIR.mkdir(parents=True, exist_ok=True)

    all_entries = []
    for league in ("serie_a",):  # extend when EPL XI predictions are reliable
        all_entries.extend(predict_scorers_for_league(league))

    if not all_entries:
        log.warning("No scorer predictions generated.")
        return 0

    OUT_PATH.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": all_entries,
    }, indent=2))
    log.info("Wrote %s (%d matches)", OUT_PATH, len(all_entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
