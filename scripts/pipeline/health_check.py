#!/usr/bin/env python3
"""PRODUCTION HEALTH CHECK — Quick system-wide status in one command.

Lightweight check that runs in seconds (no API calls, no heavy computation).
Designed for daily monitoring and CI integration.

Checks:
  1. Data freshness — features.parquet, odds, predictions, results
  2. Model freshness — are models stale?
  3. Betting performance — ROI, win rate, drift alerts from auto_settle
  4. System integrity — missing files, broken imports

Usage:
    python -m scripts.pipeline.health_check          # Full health check
    python -m scripts.pipeline.health_check --json   # Machine-readable output

Exit codes: 0=HEALTHY, 1=WARNING, 2=CRITICAL
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, MODELS_DIR, get_current_season

# ─── Staleness thresholds ───
MAX_FEATURES_AGE_DAYS = 7       # Features should be rebuilt weekly
MAX_MODEL_AGE_DAYS = 30         # Models should be retrained monthly
MIN_UNSEEN_MATCHES_FOR_STALE = 10   # ...but only if a matchweek of new
                                    # results exists that they never saw.
MAX_ODDS_AGE_HOURS = 48         # Odds should be fresh if matches upcoming
MAX_PREDICTIONS_AGE_HOURS = 48  # Predictions should be recent before matchday


def _file_age(path: Path) -> Tuple[float, str]:
    """Return (age_hours, human_readable_age) for a file, or (-1, 'missing')."""
    if not path.exists():
        return -1, "MISSING"
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    delta = datetime.now() - mtime
    hours = delta.total_seconds() / 3600

    if hours < 1:
        return hours, f"{int(delta.total_seconds() / 60)}m ago"
    elif hours < 24:
        return hours, f"{hours:.1f}h ago"
    else:
        days = hours / 24
        return hours, f"{days:.1f}d ago"


def _iso_age_days(stamp: str) -> Optional[float]:
    """Age in days of an ISO timestamp, or None if it cannot be read.

    Naive stamps are read as UTC. That is the convention the project's bug
    catalogue settled on after ``_iso_age_hours`` mixed naive and aware
    datetimes, raised TypeError, swallowed it and reported -1 — a staleness
    check that silently returns "fine" is worse than one that is absent.
    ``predictions_archive.json`` writes naive stamps, so this is the branch
    that actually runs; a few hours of local-vs-UTC skew is immaterial against
    a threshold measured in weeks.
    """
    if not stamp:
        return None
    try:
        dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400


def _freshest_fixture_source() -> Path:
    """The newest file that can actually tell us about upcoming matches.

    ``upcoming/matches.json`` is the raw Odds API schedule, and it used to be the
    only thing this check looked at. That made the check lie once the Odds API
    key lapsed: it reported "fixtures stale" while the Sofascore season files —
    which predict_unified now loads first, and which refresh independently of
    the Odds API — were hours old. Being blind to upcoming matches is the
    condition worth alerting on, not one source of them being cold.
    """
    season = get_current_season().replace("-", "_")
    sofa = DATA_DIR / "external" / "sofascore"
    candidates = [
        DATA_DIR / "upcoming" / "matches.json",
        sofa / f"fixtures_{season}.json",
        sofa / f"fixtures_{season}_premier_league.json",
    ]
    existing = [c for c in candidates if c.exists()]
    if not existing:
        return candidates[0]
    return max(existing, key=lambda c: c.stat().st_mtime)


def check_data_freshness() -> Dict:
    """Check freshness of key data files."""
    checks = {}

    files = {
        "features.parquet": DATA_DIR / "features" / "features.parquet",
        "fixtures": _freshest_fixture_source(),
        "odds_data": DATA_DIR / "upcoming" / "odds.json",
        "predictions": DATA_DIR / "upcoming" / "predictions.json",
        "results": DATA_DIR / "upcoming" / "results.json",
        "unified_report": DATA_DIR / "upcoming" / "unified_bet_slip.json",  # live slip (key kept)
        "bet_journal": DATA_DIR / "betting" / "bet_journal.json",
    }

    for name, path in files.items():
        age_hours, age_str = _file_age(path)
        status = "OK"

        if age_hours < 0:
            status = "MISSING"
        elif name == "features.parquet" and age_hours > MAX_FEATURES_AGE_DAYS * 24:
            status = "STALE"
        elif name == "fixtures" and age_hours > 48:
            status = "STALE"  # Fixtures older than 48h = blind to upcoming matches
        elif name == "odds_data" and age_hours > MAX_ODDS_AGE_HOURS:
            status = "STALE"
        elif name == "predictions" and age_hours > MAX_PREDICTIONS_AGE_HOURS:
            status = "STALE"

        checks[name] = {
            "path": str(path),
            "exists": path.exists(),
            "age": age_str,
            "age_hours": round(age_hours, 1),
            "status": status,
        }

    return checks


def _labeled_matches_since_model() -> Dict[str, Optional[int]]:
    """How many matches with a RESULT postdate each model's training run.

    Uses the model file's mtime as the training instant — the same signal the
    age check already trusts, and the only one available: six of the seven
    active models ship no ``saved_at`` metadata. Zero means the model has seen
    everything there is to learn from, however old the file is. Returns
    ``None`` per model when the count cannot be established, so the caller
    fails loud rather than silently reporting OK.

    Two granularity traps, both handled by comparing whole DAYS:

    * ``match_date`` is date-only (every row is midnight), so comparing it
      against a mid-afternoon timestamp counts same-day matches as already
      seen. Truncating the model side too makes "strictly later day" the test.
    * ``datetime.fromtimestamp(mtime)`` is naive-LOCAL, while the parquet dates
      are naive-UTC. On this machine (EDT) that shifted the training instant
      four hours early — enough to flip a whole day of fixtures into "unseen".
      This is the ``_iso_age_hours`` naive/aware bug in the project's own bug
      catalogue; convert with an explicit ``timezone.utc`` and keep BOTH sides
      naive-UTC. Do not make one of them aware without doing the other.

    The count is deliberately league-agnostic: these are the ``universal``
    models, trained across Serie A and the Premier League together (roughly
    7,990 / 7,909 rows), so a matchweek of either league is genuinely unseen
    training data for them.
    """
    import pandas as pd  # local, matching this module's lazy-import convention

    out: Dict[str, Optional[int]] = {}
    universal_dir = MODELS_DIR / "universal"
    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    if not matches_path.exists():
        return out
    try:
        m = pd.read_parquet(matches_path, columns=["match_date", "home_score"])
        played = m[m["home_score"].notna()].copy()
        played["match_date"] = pd.to_datetime(
            played["match_date"], errors="coerce"
        ).dt.normalize()
        played = played.dropna(subset=["match_date"])
    except (OSError, ValueError, KeyError):
        # No logger in this module by design — it is a CLI that prints its
        # report. Returning the empty mapping is how the failure is surfaced:
        # every model then reads None, which the caller turns into STALE, so an
        # unreadable parquet fails loud instead of passing silently as healthy.
        return out

    for path in universal_dir.glob("*"):
        if not path.is_file():
            continue
        try:
            trained_on = pd.Timestamp(
                datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                .replace(tzinfo=None)
            ).normalize()
        except OSError:
            continue
        out[path.stem] = int((played["match_date"] > trained_on).sum())
    return out


def check_model_freshness() -> Dict:
    """Check model files exist and aren't stale.

    Only checks ACTIVE production models (not archived checkpoints or experiments).
    Stale artifacts live in data/models/archive/ and are not monitored.
    """
    checks = {}

    universal_dir = MODELS_DIR / "universal"

    # Active production models — these are the only ones that matter
    active_models = {
        "catboost_no_odds": universal_dir / "catboost_no_odds.cbm",
        "catboost_latest": universal_dir / "catboost_latest.cbm",
        "xg_home": universal_dir / "xg_home.cbm",
        "xg_away": universal_dir / "xg_away.cbm",
        "draw_detector": universal_dir / "draw_detector.cbm",
        "lightgbm_latest": universal_dir / "lightgbm_latest.txt",
        "xgboost_latest": universal_dir / "xgboost_latest.json",
    }

    unseen = _labeled_matches_since_model()

    for name, path in active_models.items():
        age_hours, age_str = _file_age(path)
        status = "OK"

        if age_hours < 0:
            status = "MISSING"
        elif age_hours > MAX_MODEL_AGE_DAYS * 24:
            # Calendar age alone is NOT staleness. A model goes stale because
            # labeled matches exist that it never trained on — not because the
            # off-season is long. Blind calendar age fired on all seven models
            # for the whole of June/July and drowned the three real signals in
            # the issues list, which is how a monitor stops being read.
            n = unseen.get(name)
            if n is None:
                status = "STALE"          # can't measure — fail loud, not quiet
            elif n >= MIN_UNSEEN_MATCHES_FOR_STALE:
                status = "STALE"

        entry = {
            "exists": path.exists(),
            "age": age_str,
            "status": status,
        }
        if unseen.get(name) is not None:
            entry["unseen_matches"] = unseen[name]
        checks[name] = entry

    return checks


def check_betting_health() -> Dict:
    """Check betting system health from journal stats and drift alerts."""
    result = {"status": "OK", "details": {}}

    # Journal stats — ledger.get_metrics() is the one computation
    try:
        from scripts.betting.ledger import get_metrics
        m = get_metrics(include_alerts=False)
        result["details"]["journal"] = {
            "total_bets": m["record"]["settled_n"] + m["bankroll"]["pending_n"],
            "pending": m["bankroll"]["pending_n"],
            "settled": m["record"]["settled_n"],
            "roi_pct": m["roi"]["all_time_pct"],
            "total_profit": m["roi"]["all_time_profit"],
            "clv_avg_pct": m["clv"]["avg_pct"],
        }
    except Exception as e:
        result["details"]["journal"] = {"error": str(e)}
        m = None

    # Drift alerts from auto_settle
    drift_file = DATA_DIR / "betting" / "drift_alerts.json"
    if drift_file.exists():
        try:
            with open(drift_file) as f:
                drift_data = json.load(f)
            result["details"]["drift"] = {
                "checked_at": drift_data.get("checked_at", "?"),
                "has_critical": drift_data.get("has_critical", False),
                "has_warning": drift_data.get("has_warning", False),
                "alerts": [
                    {"level": a["level"], "message": a["message"]}
                    for a in drift_data.get("alerts", [])
                    if a["level"] in ("CRITICAL", "WARNING")
                ],
            }
            if drift_data.get("has_critical"):
                result["status"] = "CRITICAL"
            elif drift_data.get("has_warning"):
                result["status"] = "WARNING"
        except Exception:
            pass

    # Bankroll — from the same payload
    if m is not None:
        result["details"]["bankroll"] = {
            "current": m["bankroll"]["current"],
            "initial": m["bankroll"]["initial"],
        }

    return result


def check_system_integrity() -> Dict:
    """Check critical imports and file system state."""
    checks = {}

    # Check critical imports
    imports_ok = True
    for module_name in [
        "catboost",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
    ]:
        try:
            __import__(module_name)
            checks[module_name] = "OK"
        except ImportError:
            checks[module_name] = "MISSING"
            imports_ok = False

    # Check config
    try:
        from config.settings import DATA_DIR, MODELS_DIR, PROJECT_ROOT
        checks["config"] = "OK"
    except Exception as e:
        checks["config"] = f"ERROR: {e}"

    # Check features.parquet has expected columns
    features_path = DATA_DIR / "features" / "features.parquet"
    if features_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(features_path, columns=["match_date", "home_team", "away_team"])
            checks["features_readable"] = f"OK ({len(df)} rows)"
        except Exception as e:
            checks["features_readable"] = f"ERROR: {e}"

    return {"imports_ok": imports_ok, "checks": checks}


def check_model_metadata_consistency() -> Dict:
    """Verify that deployed models match their metadata files.

    Catches KB#19: deployment state uncertainty where model feature counts
    disagree between the model, metadata, and training report.
    """
    result = {"status": "OK", "checks": []}
    universal_dir = MODELS_DIR / "universal"

    # Check catboost_no_odds consistency
    no_odds_model = universal_dir / "catboost_no_odds.cbm"
    no_odds_meta = universal_dir / "catboost_no_odds_metadata.json"
    if no_odds_model.exists() and no_odds_meta.exists():
        try:
            import json
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(str(no_odds_model))
            model_n_features = len(model.feature_names_)

            with open(no_odds_meta) as f:
                meta = json.load(f)
            meta_n_features = meta.get("n_features", -1)

            if model_n_features != meta_n_features:
                result["status"] = "WARNING"
                result["checks"].append({
                    "model": "catboost_no_odds",
                    "issue": f"Model has {model_n_features} features but metadata says {meta_n_features}",
                })
            else:
                result["checks"].append({
                    "model": "catboost_no_odds",
                    "status": "OK",
                    "n_features": model_n_features,
                })
        except Exception as e:
            result["checks"].append({
                "model": "catboost_no_odds",
                "issue": f"Could not verify: {e}",
            })

    # Check ensemble consistency
    ensemble_meta = universal_dir / "ensemble" / "ensemble_metadata.json"
    if ensemble_meta.exists():
        try:
            import json
            with open(ensemble_meta) as f:
                meta = json.load(f)
            ens_n_features = len(meta.get("feature_names", []))
            result["checks"].append({
                "model": "ensemble",
                "status": "OK",
                "n_features": ens_n_features,
            })
        except Exception as e:
            result["checks"].append({
                "model": "ensemble",
                "issue": f"Could not verify: {e}",
            })

    # Check deployment_state.json exists and has active_ml_model
    deploy_path = MODELS_DIR / "deployment_state.json"
    if deploy_path.exists():
        try:
            import json
            with open(deploy_path) as f:
                state = json.load(f)
            if "active_ml_model" not in state:
                result["checks"].append({
                    "model": "deployment_state",
                    "issue": "No active_ml_model field — run retrain to populate",
                })
        except Exception:
            pass

    return result


def check_data_quality() -> Dict:
    """Validate content quality of key data files (NaN rates, row counts, schema)."""
    checks = {}

    # Check features.parquet
    features_path = DATA_DIR / "features" / "features.parquet"
    if features_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(features_path)
            row_count = len(df)

            # Known-sparse column families: advanced-event rollups only exist
            # post-2017, so computing NaN rate over full history is misleading.
            # Exclude from both the avg-NaN and the sparse-col check.
            SPARSE_PREFIXES = (
                "adv_roll5_", "home_adv_roll5_", "away_adv_roll5_",
                "tagg_roll5_", "home_tagg_roll5_", "away_tagg_roll5_",
                # Squad-rotation / lineup-availability features: only populated for
                # matches with confirmed lineup data (recent seasons only). Across
                # 20 seasons of history they read >90% NaN by design.
                "home_squad_rotation", "away_squad_rotation",
                "home_key_players", "away_key_players",
                "home_top_scorer", "away_top_scorer",
                "squad_disruption", "suspended_count",
                # Cards/corners rollups: FBref stopped populating these post-Feb 2026
                # and pre-2017 data lacks them. Per-league feature build handles it;
                # the combined parquet is sparse by construction.
                "home_cards", "away_cards", "home_corners_roll",
                "away_corners_roll", "home_yellow_cards_roll",
                "away_yellow_cards_roll",
                # First-half rollups: only populated for matches with Sofascore
                # half-by-half stats (recent seasons / specific leagues). >90% NaN
                # across full history by design.
                "home_fh_", "away_fh_",
                # xG-share-by-zone rollups (open-play, penalty, counter, set-piece):
                # only populated for Sofascore-shotmap matches (post-2017). Sparse
                # by design across the full historical window.
                "home_xg_share_", "away_xg_share_",
                "home_xg_conceded_share_", "away_xg_conceded_share_",
                # Goalkeeper rollups: FBref GK tables only exist from 2024-25.
                # Measured 2026-08-24 on features_serie_a: 100% NaN for every
                # season through 2023-24, then 12.5% (2024-25) and 0.5%
                # (2025-26). Sparse across full history by construction, not a
                # regression — the underlying data did not exist.
                "home_gk_roll5_", "away_gk_roll5_",
            )
            dense_cols = [c for c in df.columns if not c.startswith(SPARSE_PREFIXES)]
            nan_rate = float(df[dense_cols].isna().mean().mean()) if dense_cols else 0.0

            status = "OK"
            issues = []

            if row_count < 100:
                status = "WARNING"
                issues.append(f"Low row count: {row_count}")
            if nan_rate > 0.5:
                status = "WARNING"
                issues.append(f"High avg NaN rate: {nan_rate:.1%}")

            # Per-column sparse detection: flag columns >90% NaN, excluding
            # known-sparse families.
            col_nan = df[dense_cols].isna().mean()
            sparse_cols = col_nan[col_nan > 0.90].sort_values(ascending=False)
            if len(sparse_cols) > 0:
                if status == "OK":
                    status = "WARNING"
                issues.append(
                    f"{len(sparse_cols)} sparse columns (>90% NaN): "
                    f"{', '.join(sparse_cols.index[:5])}{'...' if len(sparse_cols) > 5 else ''}"
                )

            required = ["home_team", "away_team"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                status = "CRITICAL"
                issues.append(f"Missing columns: {missing}")

            checks["features_quality"] = {
                "status": status,
                "rows": row_count,
                "columns": len(df.columns),
                "nan_rate": round(nan_rate, 3),
                "max_col_nan_rate": round(float(col_nan.max()), 3) if len(col_nan) else 0,
                "sparse_columns": len(sparse_cols),
                "issues": issues,
            }
        except Exception as e:
            checks["features_quality"] = {"status": "ERROR", "error": str(e)}
    else:
        checks["features_quality"] = {"status": "MISSING"}

    # Feature id-keying guard (2026-07-17): every per-league feature row must be
    # keyed by a canonical matches.parquet id. A re-minted Sofascore numeric id
    # (see DATA_CATALOG features row 2) breaks the join back to matches.parquet
    # AND silently empties the 64 shot features — features/shot_level_xg.py joins
    # canonical-only (_map_to_canonical), so a numeric-keyed row receives ZERO
    # shot columns, not partial NaN. Classify by membership in matches.parquet
    # (ground truth), NEVER by id shape: an FBref hash can be all-digits, so a
    # .isdigit() test would false-positive.
    try:
        import pandas as pd
        from config.leagues import ACTIVE_LEAGUES
        matches_path = DATA_DIR / "parsed" / "matches.parquet"
        if matches_path.exists():
            mdf = pd.read_parquet(matches_path, columns=["match_id", "league", "season"])
            cur = str(mdf["season"].dropna().astype(str).max())  # self-updating; no hardcoded season
            per_league = {}
            keying_issues = []
            levels = []
            for league in ACTIVE_LEAGUES:
                fpath = DATA_DIR / "features" / f"features_{league}.parquet"
                if not fpath.exists():
                    continue
                fdf = pd.read_parquet(fpath, columns=["match_id", "season"])
                canon = set(
                    mdf.loc[(mdf["league"] == league) & (mdf["season"] == cur), "match_id"].astype(str)
                )
                feats = set(fdf.loc[fdf["season"] == cur, "match_id"].astype(str))
                orphans = feats - canon
                per_league[league] = {"feature_rows": len(feats), "orphan_ids": len(orphans)}
                if orphans:
                    # >1 full matchweek (10) mis-keyed = systemic revert of the
                    # derived layer, not the documented transient new-match window
                    # (a match added from Sofascore before its weekly FBref report
                    # lands; self-heals at the Monday backfill).
                    lvl = "CRITICAL" if len(orphans) > 10 else "WARNING"
                    levels.append(lvl)
                    note = (
                        "systemic re-mint — derived layer reverted to Sofascore keys; "
                        "64 shot features silently empty for these rows"
                        if lvl == "CRITICAL"
                        else "likely transient new-match window before the weekly FBref backfill"
                    )
                    keying_issues.append(
                        f"{league}: {len(orphans)}/{len(feats)} current-season feature match_id(s) "
                        f"absent from matches.parquet — {note}: {', '.join(sorted(orphans)[:3])}"
                    )
            checks["feature_id_keying"] = {
                "status": "CRITICAL" if "CRITICAL" in levels else ("WARNING" if levels else "OK"),
                "current_season": cur,
                "per_league": per_league,
                "issues": keying_issues,
            }
    except Exception as e:
        checks["feature_id_keying"] = {"status": "ERROR", "error": str(e)}

    # --- Parsed-input keying guard -------------------------------------------
    # feature_id_keying above is BLIND to this class: feature-table ids come from
    # matches.parquet so they are always canonical, yet the per-player parsed
    # inputs can be hash-keyed and silently drop out of the canonical join that
    # builds those features. That is exactly what happened 2026-07-17 —
    # player_stats + goalkeeper_stats were 100% FBref-hash-keyed for the current
    # season, so player_impact / team_aggregates / advanced_player / gk_quality
    # went null for the whole season with NO loud signal. This guard reads the
    # inputs directly and classifies by membership in matches.parquet (ground
    # truth), NEVER by id shape (an FBref hash can be all-digits).
    try:
        import pandas as pd
        matches_path = DATA_DIR / "parsed" / "matches.parquet"
        if matches_path.exists():
            mdf = pd.read_parquet(matches_path, columns=["match_id", "season"])
            cur = str(mdf["season"].dropna().astype(str).max())
            canon = set(mdf.loc[mdf["season"] == cur, "match_id"].astype(str))
            per_file = {}
            issues = []
            levels = []
            for fname in ("player_stats.parquet", "goalkeeper_stats.parquet"):
                fpath = DATA_DIR / "parsed" / fname
                if not fpath.exists():
                    continue
                pdf = pd.read_parquet(fpath, columns=["match_id", "season"])
                ids = set(pdf.loc[pdf["season"] == cur, "match_id"].astype(str))
                if not ids:
                    continue
                orphans = ids - canon
                per_file[fname] = {"current_ids": len(ids), "orphan_ids": len(orphans)}
                if orphans:
                    # >1 matchweek mis-keyed = the writer regressed to html_path.stem
                    # (FBref {hash}.html); the whole current season joins to nothing.
                    lvl = "CRITICAL" if len(orphans) > 10 else "WARNING"
                    levels.append(lvl)
                    issues.append(
                        f"{fname}: {len(orphans)}/{len(ids)} current-season match_id(s) "
                        f"absent from matches.parquet — hash-keyed input, canonical joins "
                        f"(player_impact/gk_quality/team_aggregates) silently empty: "
                        f"{', '.join(sorted(orphans)[:3])}"
                    )
            checks["parsed_input_keying"] = {
                "status": "CRITICAL" if "CRITICAL" in levels else ("WARNING" if levels else "OK"),
                "current_season": cur,
                "per_file": per_file,
                "issues": issues,
            }
    except Exception as e:
        checks["parsed_input_keying"] = {"status": "ERROR", "error": str(e)}

    # Check predictions probability sums
    preds_path = DATA_DIR / "upcoming" / "predictions.json"
    if preds_path.exists():
        try:
            with open(preds_path) as f:
                preds = json.load(f)
            bad_sums = 0
            total = 0
            pred_list = preds.get("predictions", []) if isinstance(preds, dict) else preds
            for p in pred_list:
                if isinstance(p, dict):
                    # Support multiple formats: flat keys, nested "probabilities" dict
                    probs = p.get("probabilities", {}) if isinstance(p.get("probabilities"), dict) else {}
                    h = probs.get("home") or p.get("prob_home") or p.get("prob_H")
                    d = probs.get("draw") or p.get("prob_draw") or p.get("prob_D")
                    a = probs.get("away") or p.get("prob_away") or p.get("prob_A")
                    if h is not None and d is not None and a is not None:
                        total += 1
                        s = float(h) + float(d) + float(a)
                        if abs(s - 1.0) > 0.05:
                            bad_sums += 1
            checks["prediction_probs"] = {
                "status": "WARNING" if bad_sums > 0 else "OK",
                "total": total,
                "bad_probability_sums": bad_sums,
            }
        except Exception as e:
            checks["prediction_probs"] = {"status": "ERROR", "error": str(e)}

    # Check predictions-odds consistency
    odds_path = DATA_DIR / "upcoming" / "odds.json"
    if preds_path.exists() and odds_path.exists():
        try:
            with open(odds_path) as f:
                odds = json.load(f)
            # Re-load predictions to avoid NameError if probability check failed
            with open(preds_path) as f:
                _preds_for_consistency = json.load(f)
            odds_matches = set(odds.keys()) if isinstance(odds, dict) else set()
            pred_matches = set()
            pred_list = _preds_for_consistency.get("predictions", []) if isinstance(_preds_for_consistency, dict) else _preds_for_consistency
            for p in pred_list:
                if isinstance(p, dict) and p.get("match"):
                    pred_matches.add(p["match"])
            missing_odds = pred_matches - odds_matches
            if missing_odds:
                checks["data_consistency"] = {
                    "status": "WARNING",
                    "predictions_count": len(pred_matches),
                    "odds_count": len(odds_matches),
                    "missing_odds_for": len(missing_odds),
                    "examples": list(missing_odds)[:3],
                }
            else:
                checks["data_consistency"] = {
                    "status": "OK",
                    "predictions_count": len(pred_matches),
                    "odds_count": len(odds_matches),
                }
        except Exception as e:
            checks["data_consistency"] = {"status": "ERROR", "error": str(e)}

    return checks


def check_silent_failures() -> Dict:
    """Assert against the silent-failure classes found in the 2026-05-31 data-layer
    diagnostic — failures that left every other health check GREEN while the
    pipeline rotted. Each assertion here corresponds to a real failure observed:

      - mapping sofascore_id 0 per league   -> EPL features silently dropped
      - understat_id 0 (secondary canary)   -> builder read wrong columns
      - odds-gate timestamp unparseable      -> tz bug, swallowed, wake-storm burn
      - past-dated matches with null result  -> results pipeline silently stalled
      - predictions not varying across slate  -> model fell back to constants

    Severity rule (avoid alert fatigue): CRITICAL only for "the pipeline is
    lying about working" (dead model / swallowed-error timestamp). Coverage and
    staleness are WARNING (real but recoverable).
    """
    checks: Dict = {}

    # --- 1. Cross-source mapping coverage (the load-bearing key is sofascore_id) ---
    mapping_path = DATA_DIR / "parsed" / "match_id_mapping.parquet"
    if mapping_path.exists():
        try:
            import pandas as pd
            mp = pd.read_parquet(mapping_path)
            issues = []
            status = "OK"
            detail = {}
            leagues = mp["league"].unique() if "league" in mp.columns else ["(none)"]
            for lg in leagues:
                sub = mp[mp["league"] == lg] if "league" in mp.columns else mp
                sofa = int(sub["sofascore_id"].notna().sum()) if "sofascore_id" in sub else 0
                detail[f"{lg}_sofascore_id"] = sofa
                # sofascore_id is what the 5 feature builders inner-join on; 0 = stranded.
                if sofa == 0:
                    status = "WARNING"
                    issues.append(
                        f"{lg}: sofascore_id is 0 — Sofascore-derived features (shot xG, "
                        f"first-half, missing players) will silently drop on the inner join"
                    )
            # understat_id is a secondary canary (nothing joins on it yet, but 0 means
            # the builder's understat discovery broke — the 2026-05-31 bug).
            if "understat_id" in mp.columns and int(mp["understat_id"].notna().sum()) == 0:
                if status == "OK":
                    status = "WARNING"
                issues.append("understat_id is 0 across all rows — mapping builder understat discovery may be broken")
            checks["mapping_coverage"] = {"status": status, "issues": issues, **detail}
        except Exception as e:
            checks["mapping_coverage"] = {"status": "ERROR", "error": str(e), "issues": []}
    else:
        checks["mapping_coverage"] = {"status": "MISSING", "issues": ["match_id_mapping.parquet missing"]}

    # --- 2. Odds-gate timestamp must PARSE and COMPARE (the swallowed tz bug) ---
    # The tz regression made needs_odds_refresh() raise TypeError internally,
    # get caught, and silently always-refresh. Assert the freshness math actually
    # works rather than throwing — a populated timestamp must not render "unknown".
    state_path = DATA_DIR / "pipeline_state.json"
    if state_path.exists():
        try:
            from scripts.pipeline.pipeline_state import get_state_summary, needs_odds_refresh
            with open(state_path) as f:
                state = json.load(f)
            summary = get_state_summary(state)
            issues = []
            status = "OK"
            # A populated last_odds_fetch that summarizes to "unknown" == a parse/tz crash.
            if state.get("last_odds_fetch") and summary.get("last_odds_fetch") == "unknown":
                status = "CRITICAL"
                issues.append(
                    "last_odds_fetch is populated but un-summarizable — timestamp parse/tz "
                    "comparison is crashing (the swallowed bug that burns Odds API credits)"
                )
            # Exercise the gate itself; a raised+swallowed error would surface as a crash here.
            _ = needs_odds_refresh(state)
            checks["odds_gate_timestamp"] = {"status": status, "issues": issues}
        except Exception as e:
            checks["odds_gate_timestamp"] = {
                "status": "CRITICAL",
                "issues": [f"odds-gate freshness math raised {type(e).__name__}: {e}"],
            }

    # --- 3. Results staleness, season-aware (past-dated match with no result) ---
    # Calendar-age alarms fire every off-season; instead assert on the
    # unambiguous signal: a match whose date is in the past but has no score.
    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    if matches_path.exists():
        try:
            import pandas as pd
            m = pd.read_parquet(matches_path)
            m["match_date"] = pd.to_datetime(m["match_date"], errors="coerce")
            now = datetime.now()
            issues = []
            status = "OK"
            detail = {}
            leagues = m["league"].unique() if "league" in m.columns else ["(none)"]
            for lg in leagues:
                sub = m[m["league"] == lg] if "league" in m.columns else m
                past_no_result = sub[
                    (sub["match_date"] < now) & (sub["home_score"].isna())
                ]
                n = int(len(past_no_result))
                detail[f"{lg}_past_unresulted"] = n
                if n > 0:
                    status = "WARNING"
                    issues.append(
                        f"{lg}: {n} past-dated matches with no result — results pipeline stalled "
                        f"(latest missing: {past_no_result['match_date'].max().date() if n else '-'})"
                    )
            checks["results_staleness"] = {"status": status, "issues": issues, **detail}
        except Exception as e:
            checks["results_staleness"] = {"status": "ERROR", "error": str(e), "issues": []}

    # --- 4. Predictions must VARY (constant fallback = dead model) ---
    # prediction_probs already checks sums==1; a constant 0.33/0.33/0.33 fallback
    # passes that. Assert dispersion across the slate instead.
    preds_path = DATA_DIR / "upcoming" / "predictions.json"
    if preds_path.exists():
        try:
            import statistics
            with open(preds_path) as f:
                preds = json.load(f)
            pred_list = preds.get("predictions", []) if isinstance(preds, dict) else preds
            home_probs = []
            for p in pred_list:
                if isinstance(p, dict):
                    probs = p.get("probabilities", {}) if isinstance(p.get("probabilities"), dict) else {}
                    h = probs.get("home") or p.get("prob_home") or p.get("prob_H")
                    if h is not None:
                        home_probs.append(float(h))
            issues = []
            status = "OK"
            stdev = round(statistics.pstdev(home_probs), 4) if len(home_probs) > 1 else None
            # Need at least 3 matches to judge dispersion; below that it's not conclusive.
            if len(home_probs) >= 3 and stdev is not None and stdev < 0.01:
                status = "CRITICAL"
                issues.append(
                    f"predictions show ~no variance across {len(home_probs)} matches "
                    f"(home-prob stdev {stdev}) — model likely fell back to constants"
                )
            checks["predictions_vary"] = {
                "status": status, "issues": issues,
                "n": len(home_probs), "home_prob_stdev": stdev,
            }
        except Exception as e:
            checks["predictions_vary"] = {"status": "ERROR", "error": str(e), "issues": []}

    return checks


CALIB_BINS = 10
CALIB_MIN_BIN = 5


def _confidence_ece(probs: list, hits: list) -> Tuple[float, int]:
    """10-bin confidence-ECE on the argmax class; bins under CALIB_MIN_BIN skipped.

    Module level so the null floor below can be pinned against it in a test —
    the two implement the same binning and must not drift apart.
    """
    n = len(probs)
    total, bins_used = 0.0, 0
    for b in range(CALIB_BINS):
        sel = [i for i, q in enumerate(probs) if int(min(q, 0.999) * CALIB_BINS) == b]
        if len(sel) < CALIB_MIN_BIN:
            continue
        bins_used += 1
        conf = sum(probs[i] for i in sel) / len(sel)
        acc_ = sum(hits[i] for i in sel) / len(sel)
        total += abs(conf - acc_) * len(sel) / n
    return total, bins_used


def _calibration_null_floor(
    probs: list, sims: int = 2000, seed: int = 20260825
) -> Tuple[float, float, float]:
    """The ECE this estimator returns for a PERFECTLY calibrated model at this n.

    ECE sums |conf - acc|, so per-bin sampling noise always ADDS and never
    cancels: the estimator is biased upward, badly at small n. Measured
    2026-08-25 at n=100 with the live confidence distribution, a perfect model
    scores a median ECE of 0.079 — so the old fixed 0.06 WARNING fired on 76%
    of perfectly calibrated models and the 0.10 CRITICAL on 25%. Comparing
    against a fixed constant was measuring the window size, not the model.

    Drawing each outcome as Bernoulli(conf) IS a perfectly calibrated model by
    construction, so the resulting spread is exactly the floor at this n and
    this confidence distribution. Reusing the observed `probs` keeps the bin
    structure identical between null and observation.

    Seeded on purpose: an unattended monitor must not flap between runs on
    identical data.

    Returns (median, p90, p99).
    """
    import numpy as np

    p_arr = np.asarray(probs, dtype=float)
    n = p_arr.size
    bin_of = (np.minimum(p_arr, 0.999) * CALIB_BINS).astype(int)
    rng = np.random.default_rng(seed)
    # (sims, n) of perfectly-calibrated outcomes.
    draws = rng.random((sims, n)) < p_arr
    total = np.zeros(sims, dtype=float)
    for b in range(CALIB_BINS):
        sel = np.flatnonzero(bin_of == b)
        if sel.size < CALIB_MIN_BIN:
            continue
        conf = float(p_arr[sel].mean())
        acc = draws[:, sel].mean(axis=1)
        total += np.abs(conf - acc) * sel.size / n
    return (
        float(np.median(total)),
        float(np.percentile(total, 90)),
        float(np.percentile(total, 99)),
    )


def check_calibration_drift() -> Dict:
    """Rolling live-calibration check on archived pre-kickoff 1X2 predictions.

    Computes 10-bin confidence-ECE over the most recent CALIB_WINDOW archived
    predictions that joined to a real result (same leak-free join the Track
    Record page uses: archived_at < kickoff snapshots only). A model can stay
    accurate while its probabilities drift — this catches the drift between
    retrains.

    Thresholds are NOT constants. ECE is biased upward at small n, so the
    observation is compared against the floor a perfectly calibrated model
    produces at this same n and confidence distribution (see
    _calibration_null_floor): above the null p90 = WARNING, above the null p99
    = CRITICAL. Fewer than CALIB_MIN_N graded predictions = SKIP, and so does a
    window whose newest entry is over CALIB_MAX_AGE_DAYS old — an ECE computed
    entirely on last season is not live drift no matter how it is thresholded.

    Grades the MONEY PATH, not the display layer (isolated 2026-06-11):
    display "probabilities" carry a deliberate draw-boost + temperature
    sharpening (ECE ~0.11 by design, accuracy/draw-recall trade) while the
    raw blend the betting system prices with sat at ~0.06. Preference order:
    archived betting_probabilities -> raw blend reconstructed from
    component_predictions with ENSEMBLE_WEIGHTS -> display (last resort,
    labeled). display_ece is reported informationally either way.
    """
    CALIB_WINDOW = 100
    CALIB_MIN_N = 60
    # Gates the NEWEST graded entry, not the oldest: a 100-match window always
    # spans ~2.5 months of fixtures, but its newest entry going stale means the
    # archive stopped joining to results (off-season, or a broken pipeline).
    # Longest in-season gap is an international break, ~2 weeks.
    CALIB_MAX_AGE_DAYS = 30
    out: dict = {"status": "SKIP", "n": 0, "ece": None, "window": CALIB_WINDOW}
    try:
        import pandas as pd

        arch_path = DATA_DIR / "upcoming" / "predictions_archive.json"
        if not arch_path.exists():
            out["reason"] = "no predictions archive"
            return {"calibration_1x2": out}
        with open(arch_path) as f:
            arch = json.load(f)
        m = pd.read_parquet(
            DATA_DIR / "parsed" / "matches.parquet",
            columns=["match_date", "home_team", "away_team", "home_score", "away_score"],
        ).dropna(subset=["home_score", "away_score"])
        m["key"] = (m["home_team"] + " vs " + m["away_team"] + "_"
                    + pd.to_datetime(m["match_date"]).dt.strftime("%Y-%m-%d"))
        res = {r["key"]: (int(r["home_score"]), int(r["away_score"]))
               for _, r in m.iterrows()}

        # ENSEMBLE_WEIGHTS mirror (scripts/prediction/ensemble_prediction_engine.py)
        # for reconstructing the raw money-path blend from archived components.
        weights = {"factor": 0.035, "xg": 0.124, "ml": 0.605,
                   "player_xg": 0.032, "market": 0.205}

        def _money_probs(p: dict) -> tuple[dict | None, str]:
            bp = p.get("betting_probabilities") or {}
            if bp.get("home"):
                return bp, "betting_probabilities"
            cp = p.get("component_predictions") or {}
            acc, wsum = [0.0, 0.0, 0.0], 0.0
            for name, w in weights.items():
                c = cp.get(name) or {}
                if "prob_H" in c:
                    s = float(c["prob_H"]) + float(c["prob_D"]) + float(c["prob_A"])
                    if s > 0:
                        acc[0] += w * float(c["prob_H"]) / s
                        acc[1] += w * float(c["prob_D"]) / s
                        acc[2] += w * float(c["prob_A"]) / s
                        wsum += w
            if wsum >= 0.5:
                return ({"home": acc[0] / wsum, "draw": acc[1] / wsum,
                         "away": acc[2] / wsum}, "raw_blend")
            disp = p.get("probabilities") or {}
            return (disp or None), "display"

        graded, graded_disp, layers = [], [], {}
        for k, p in (arch or {}).items():
            if k not in res:
                continue
            hs, as_ = res[k]
            act = "home" if hs > as_ else ("away" if as_ > hs else "draw")
            money, layer = _money_probs(p)
            if money:
                call = max(money, key=money.get)
                graded.append((p.get("archived_at", ""), float(money[call]),
                               int(call == act)))
                layers[layer] = layers.get(layer, 0) + 1
            disp = p.get("probabilities") or {}
            if disp:
                dcall = max(disp, key=disp.get)
                graded_disp.append((p.get("archived_at", ""), float(disp[dcall]),
                                    int(dcall == act)))
        graded.sort()
        window = [(g[1], g[2]) for g in graded[-CALIB_WINDOW:]]
        out["n"] = len(window)
        out["layers"] = layers
        if len(window) < CALIB_MIN_N:
            out["reason"] = f"only {len(window)} graded predictions (< {CALIB_MIN_N})"
            return {"calibration_1x2": out}

        # How old is the freshest thing in this window? Without this the check
        # reported a May number as live drift every 30 minutes all summer.
        age_days = _iso_age_days(graded[-1][0])
        out["newest_graded_age_days"] = (
            round(age_days, 1) if age_days is not None else None
        )
        if age_days is not None and age_days > CALIB_MAX_AGE_DAYS:
            out["reason"] = (
                f"newest graded prediction is {age_days:.0f}d old "
                f"(> {CALIB_MAX_AGE_DAYS}d) — no recent calibration to measure"
            )
            return {"calibration_1x2": out}

        probs = [w[0] for w in window]
        hits = [w[1] for w in window]
        ece, bins_used = _confidence_ece(probs, hits)
        out["ece"] = round(ece, 4)
        out["bins_used"] = bins_used

        null_med, null_p90, null_p99 = _calibration_null_floor(probs)
        out["null_ece_median"] = round(null_med, 4)
        out["null_ece_p90"] = round(null_p90, 4)
        out["null_ece_p99"] = round(null_p99, 4)
        out["ece_excess"] = round(ece - null_med, 4)

        graded_disp.sort()
        dwin = [(g[1], g[2]) for g in graded_disp[-CALIB_WINDOW:]]
        if len(dwin) >= CALIB_MIN_N:
            out["display_ece"] = round(  # informational only
                _confidence_ece([g[0] for g in dwin], [g[1] for g in dwin])[0], 4
            )
        out["status"] = ("CRITICAL" if ece > null_p99 else
                         "WARNING" if ece > null_p90 else "OK")
    except Exception as e:
        out["status"] = "SKIP"
        out["reason"] = f"check failed: {e}"
    return {"calibration_1x2": out}


LINEUP_PROBE_HOURS = 30.0      # a matchday is "near" when a Serie A kickoff is inside this
LINEUP_DUE_MIN = 25.0          # a sheet is DUE when kickoff is this close: sources publish
                               # ~T-60..T-50 and the stage retries every 15 min, so T-45 was
                               # a false alarm on every match


def _upcoming_serie_a_kickoffs(now: datetime) -> List[Tuple[str, datetime]]:
    """[(match_key, kickoff_utc)] from odds_full.json, sorted."""
    out = []
    try:
        matches = json.loads((DATA_DIR / "upcoming" / "odds_full.json").read_text()).get("matches", {})
    except (OSError, ValueError):
        return out
    for mk, info in matches.items():
        ct = (info or {}).get("commence_time")
        if not ct:
            continue
        try:
            dt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
        except ValueError:
            continue
        out.append((mk, dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)))
    return sorted(out, key=lambda x: x[1])


def check_lineup_sources(now: Optional[datetime] = None, probe=None) -> Dict:
    """Is the lineup chain alive BEFORE it is needed, and did it deliver when it was?

    2026-09-05: every source was dead for a whole matchday and the only trace was a
    log line. Two questions, both answered here every health cycle:
      (a) reachability — one free GET to ESPN's scoreboard (the key-free backup) when a
          Serie A kickoff is inside LINEUP_PROBE_HOURS; a failure is CRITICAL on a
          matchday because the chain's other links are known-dead on this network;
      (b) delivery — data/upcoming/lineup_chain_status.json (written by every fetch
          run): a match inside LINEUP_DUE_MIN of kickoff, or kicked off in the last
          3h, with no team sheet is CRITICAL and carries the chain's own reason.
    `probe` is injectable for tests; default is the real ESPN request."""
    now = now or datetime.now(timezone.utc)
    kicks = _upcoming_serie_a_kickoffs(now)
    near = [(mk, dt) for mk, dt in kicks if -3 * 3600 <= (dt - now).total_seconds() <= LINEUP_PROBE_HOURS * 3600]
    out: Dict = {"status": "OK", "matchday_near": bool(near), "espn": None, "missing_sheets": [], "reason": None}
    if not near:
        return out
    # (a) reachability
    if probe is None:
        def probe():
            import requests
            r = requests.get("https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard",
                             params={"dates": now.strftime("%Y%m%d")}, timeout=15)
            return r.status_code
    try:
        code = probe()
    except Exception as e:  # noqa: BLE001 - a probe failure IS the finding
        code = f"error: {str(e)[:80]}"
    out["espn"] = code
    if code != 200:
        out["status"] = "CRITICAL"
        out["reason"] = f"ESPN scoreboard -> {code} with a Serie A kickoff inside {LINEUP_PROBE_HOURS:.0f}h"
    # (b) delivery
    try:
        rep = json.loads((DATA_DIR / "upcoming" / "lineup_chain_status.json").read_text())
    except (OSError, ValueError):
        rep = {}
    try:
        sheets = set(json.loads((DATA_DIR / "upcoming" / "confirmed_lineups.json").read_text()).get("matches", {}))
    except (OSError, ValueError):
        sheets = set()
    sheets |= set(rep.get("confirmed") or [])
    due = [mk for mk, dt in near if (dt - now).total_seconds() <= LINEUP_DUE_MIN * 60 and mk not in sheets]
    if due:
        out["missing_sheets"] = due
        out["status"] = "CRITICAL"
        why = rep.get("reason") or "nessun run del fetcher registrato"
        out["reason"] = (out["reason"] + " · " if out["reason"] else "") + \
            f"no team sheet for {', '.join(due)} ({why})"
    return out


STATS_GRACE_HOURS = 14.0   # the evening (20:00) and morning (08:00) runs both get a chance


def _serie_a_fixtures_kicked_off(now: datetime, min_age_s: float, max_age_s: float,
                                 with_id: bool = False) -> List[Tuple]:
    """[(date, home, away)] of Serie A fixtures whose kickoff lies between
    `min_age_s` and `max_age_s` ago, from the cached fixture file (known weeks
    ahead, so no check built on it depends on Sofascore being up). Canceled /
    postponed fixtures keep their original timestamp forever and are excluded."""
    from scripts.utils.match_timing import _sofascore_fixture_files
    fx_path = next(p for p, lg in _sofascore_fixture_files() if lg == "serie_a")
    raw = json.loads(fx_path.read_text())
    fixtures = raw if isinstance(raw, list) else next(v for v in raw.values() if isinstance(v, list))
    out = []
    for f in fixtures:
        ts = f.get("startTimestamp")
        if not ts:
            continue
        ko = datetime.fromtimestamp(int(ts), tz=timezone.utc)
        if (f.get("status") or {}).get("type") in ("canceled", "postponed"):
            continue
        if min_age_s <= (now - ko).total_seconds() <= max_age_s:
            item = (ko.strftime("%Y-%m-%d"), (f.get("homeTeam") or {}).get("name"), (f.get("awayTeam") or {}).get("name"))
            out.append(item + (f.get("id"),) if with_id else item)
    return out


PICKS_JOURNAL_GRACE_MIN = 30.0   # the T-30 run is the last chance to journal a pick


def check_picks_journal_activity(now: Optional[datetime] = None) -> Dict:
    """Did the T-30 path paper-journal anything for the Serie A matches that
    kicked off in the last 24h? The T-30 run is a child process
    (`run_full_pipeline --pre-kickoff`, stdout captured by the scheduler), so
    a pick engine that raises, or one that never ran, is invisible unless
    something reads the outcome; on 2026-09-05 Roma–Atalanta journaled
    nothing and only a by-hand log read said why. WARNING, never CRITICAL: a
    slate where no angle beats its price is legal, just rare with 40+ priced
    rows a match."""
    now = now or datetime.now(timezone.utc)
    try:
        due = _serie_a_fixtures_kicked_off(now, PICKS_JOURNAL_GRACE_MIN * 60, 24 * 3600)
    except (OSError, ValueError, StopIteration) as e:
        return {"status": "WARNING", "detail": f"fixture file unreadable: {e}"}
    if not due:
        return {"status": "OK", "detail": "no Serie A kickoff in the last 24h"}
    try:
        bets = json.loads((DATA_DIR / "betting" / "picks_journal.json").read_text()).get("bets", {})
    except (OSError, ValueError):
        bets = {}
    dates = sorted({d for d, _, _ in due})
    n = sum(1 for b in bets.values() if str(b.get("date") or "")[:10] in dates)
    if n == 0:
        return {"status": "WARNING", "dates": dates, "n_matches": len(due),
                "detail": f"{len(due)} Serie A match(es) kicked off on {', '.join(dates)} and the T-30 path "
                          "journaled no paper pick — grep logs/pipeline.log for 'Pick engine' / 'Picks:' "
                          "around T-30 (the child's output is captured, not in the monitor log)"}
    return {"status": "OK", "dates": dates, "n_matches": len(due), "n_journaled": n,
            "detail": f"{n} paper pick(s) journaled for {len(due)} Serie A match(es) on {', '.join(dates)}"}


def check_match_record_completeness(now: Optional[datetime] = None) -> Dict:
    """Does every Serie A match finished more than STATS_GRACE_HOURS ago (last
    7 days) have incident rows AND team stats on its ground-truth row? Under
    the Sofascore API challenge both endpoints answer nothing, the ingest
    writes a score-only row, the detector then sees the match as done, and
    nothing retries: nine of 21 finished 2026-27 matches had no incidents and
    six rows no possession before 2026-09-05. matchday_updater.heal_from_espn
    fills both from ESPN on every run; this check says when it did not.
    WARNING: the goal-process timeline, card counts and rolling shot/corner
    features are model inputs, not money inputs."""
    now = now or datetime.now(timezone.utc)
    try:
        import pandas as pd

        from config.team_names import normalize_team
        played = _serie_a_fixtures_kicked_off(now, STATS_GRACE_HOURS * 3600, 7 * 86400, with_id=True)
        if not played:
            return {"status": "OK", "detail": "no Serie A match past the stats grace in the last 7 days"}
        inc_path = DATA_DIR / "external" / "sofascore" / "match_incidents.parquet"
        with_incidents: set = set()
        if inc_path.exists():
            with_incidents = set(pd.read_parquet(inc_path, columns=["match_id"])["match_id"].unique())
        gt_path = DATA_DIR / "parsed" / "matches.parquet"
        gt = pd.read_parquet(gt_path, columns=["match_date", "home_team", "away_team", "league",
                                                "home_score", "home_possession"]) if gt_path.exists() else pd.DataFrame()
        rows = {}
        if len(gt):
            gt = gt[gt["league"] == "serie_a"]
            dates = gt["match_date"].astype(str).str[:10]
            for d, h, a, sc, poss in zip(dates, gt["home_team"], gt["away_team"], gt["home_score"], gt["home_possession"]):
                rows[(d, h, a)] = (sc, poss)
        gaps = []
        for d, h, a, fid in played:
            key = (d, normalize_team(h or ""), normalize_team(a or ""))
            what = []
            if fid not in with_incidents:
                what.append("no incidents")
            row = rows.get(key)
            if row is None or pd.isna(row[0]):
                what.append("no ground-truth row")
            elif pd.isna(row[1]):
                what.append("no team stats")
            if what:
                gaps.append(f"{key[1]}-{key[2]} {d} ({', '.join(what)})")
        if gaps:
            return {"status": "WARNING", "count": len(gaps), "matches": gaps,
                    "detail": f"{len(gaps)} of {len(played)} finished match(es) incomplete: "
                              f"{'; '.join(gaps[:4])}{' ...' if len(gaps) > 4 else ''} "
                              f"— run matchday_updater --heal-espn (Sofascore challenged?)"}
        return {"status": "OK", "detail": f"{len(played)}/{len(played)} finished matches have incidents and team stats"}
    except Exception as e:  # noqa: BLE001
        return {"status": "WARNING", "detail": f"match record check failed: {e}"}


def check_referee_coverage(now: Optional[datetime] = None) -> Dict:
    """Does every Serie A match finished more than STATS_GRACE_HOURS ago carry a
    referee in matches.parquet? The 1X2 ensemble reads three ref_* features;
    from 2026-08-23 to 2026-09-05 every row of the season had "" (the Sofascore
    fixture list names no referee, worldfootball had not published the season)
    and no check looked. ESPN fills it after the match (matchday_updater
    backfill_referees). WARNING: a display-model input, not a money input."""
    now = now or datetime.now(timezone.utc)
    try:
        played = _serie_a_fixtures_kicked_off(now, STATS_GRACE_HOURS * 3600, 7 * 86400)
    except (OSError, ValueError, StopIteration) as e:
        return {"status": "WARNING", "detail": f"fixture file unreadable: {e}"}
    if not played:
        return {"status": "OK", "detail": "no Serie A match finished in the last 7 days past the grace window"}
    try:
        import pandas as pd
        from config.team_names import normalize_team
        gt = pd.read_parquet(DATA_DIR / "parsed" / "matches.parquet",
                             columns=["match_date", "home_team", "away_team", "referee", "league"])
    except Exception as e:  # noqa: BLE001
        return {"status": "WARNING", "detail": f"matches.parquet unreadable: {e}"}
    gt = gt[gt["league"] == "serie_a"]
    named = {(str(d)[:10], h, a) for d, h, a, r in zip(gt["match_date"], gt["home_team"], gt["away_team"], gt["referee"])
             if isinstance(r, str) and r.strip()}
    missing = [f"{h} vs {a} ({d})" for d, h, a in played
               if (d, normalize_team(h or ""), normalize_team(a or "")) not in named]
    if missing:
        return {"status": "WARNING", "missing": missing,
                "detail": f"{len(missing)} of {len(played)} finished Serie A match(es) have no referee in "
                          f"matches.parquet: {', '.join(missing[:4])} — ref_* features are NaN for them; "
                          "run matchday_updater --backfill-referees (ESPN)"}
    return {"status": "OK", "detail": f"referee known for all {len(played)} Serie A match(es) finished >{STATS_GRACE_HOURS:.0f}h ago"}


def check_player_stats_coverage(now: Optional[datetime] = None) -> Dict:
    """Did the Sofascore player stats land for every Serie A match that finished
    more than STATS_GRACE_HOURS ago? Player-prop paper picks grade from
    player_match_stats.parquet; when the Sofascore API is challenged (2026-09-05)
    the ingestion fails silently and the record that gates real stakes never
    accrues. Fixture kickoffs come from the cached fixture file (known weeks
    ahead), so this check does not itself depend on Sofascore being up."""
    now = now or datetime.now(timezone.utc)
    try:
        played = _serie_a_fixtures_kicked_off(now, STATS_GRACE_HOURS * 3600, 7 * 86400)
    except (OSError, ValueError, StopIteration) as e:
        return {"status": "WARNING", "detail": f"fixture file unreadable: {e}"}
    if not played:
        return {"status": "OK", "detail": "no Serie A match finished in the last 7 days past the grace window"}
    pms = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
    try:
        import pandas as pd
        have = set(pd.read_parquet(pms, columns=["date"])["date"].astype(str).str[:10])
    except Exception as e:  # noqa: BLE001
        return {"status": "CRITICAL", "detail": f"player_match_stats unreadable: {e}"}
    missing = sorted({d for d, _, _ in played} - have)
    if missing:
        n = sum(1 for d, _, _ in played if d in missing)
        return {"status": "CRITICAL", "missing_dates": missing,
                "detail": f"{n} finished Serie A match(es) with no player stats on disk (dates {', '.join(missing)}) — "
                          "player-prop picks cannot be graded; Sofascore ingestion is failing"}
    return {"status": "OK", "detail": f"player stats cover every Serie A match finished >{STATS_GRACE_HOURS:.0f}h ago"}


def check_disk_space() -> Dict:
    """Check available disk space on the project partition."""
    import shutil
    try:
        stat = shutil.disk_usage(str(DATA_DIR))
        free_gb = stat.free / 1e9
        free_pct = (stat.free / stat.total) * 100
        # Absolute floors, not a bare percentage: on a ~1 TB drive "18.2%
        # free" is 168 GB — nothing is low, yet the pct threshold warned on
        # every health cycle. The pipeline's daily growth is well under a
        # GB, so 30/10 GB give weeks of runway.
        status = "OK"
        if free_gb < 10:
            status = "CRITICAL"
        elif free_gb < 30:
            status = "WARNING"
        return {
            "status": status,
            "free_gb": round(free_gb, 1),
            "total_gb": round(stat.total / 1e9, 1),
            "free_pct": round(free_pct, 1),
        }
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def check_log_sizes() -> Dict:
    """Check for oversized log files that could fill disk."""
    LOG_DIR = Path(__file__).parent.parent.parent / "logs"
    MAX_SIZE_MB = 100
    result = {"status": "OK", "large_logs": [], "total_mb": 0}
    if not LOG_DIR.exists():
        return result
    try:
        total = 0
        for log_file in LOG_DIR.glob("*.log*"):
            size_mb = log_file.stat().st_size / (1024 * 1024)
            total += size_mb
            if size_mb > MAX_SIZE_MB:
                result["large_logs"].append({
                    "file": log_file.name,
                    "size_mb": round(size_mb, 1),
                })
        result["total_mb"] = round(total, 1)
        if result["large_logs"]:
            result["status"] = "WARNING"
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
    return result


def check_feature_model_alignment() -> Dict:
    """Verify features.parquet columns align with deployed model expectations."""
    result = {"status": "OK", "issues": []}
    features_path = DATA_DIR / "features" / "features.parquet"

    if not features_path.exists():
        return {"status": "MISSING", "issues": ["features.parquet not found"]}

    # Check multiple metadata locations (different model naming conventions)
    meta_candidates = [
        MODELS_DIR / "universal" / "catboost_no_odds_metadata.json",
        MODELS_DIR / "universal" / "ensemble" / "ensemble_metadata.json",
    ]

    try:
        import pandas as pd
        import pyarrow.parquet as pq
        feature_cols = set(pq.ParquetFile(features_path).schema.names)
        result["data_columns"] = len(feature_cols)

        for meta_path in meta_candidates:
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            model_features = set(meta.get("feature_names", meta.get("features", [])))
            if not model_features:
                continue

            model_name = meta_path.stem.replace("_metadata", "")
            missing_in_data = model_features - feature_cols
            # Lineup-dependent features cannot be produced at runtime without
            # confirmed lineups (xi_quality plugin). CatBoost handles NaN; this
            # is expected, not critical. Downgrade to WARNING when the only
            # missing features are from this allowlist.
            _LINEUP_OPTIONAL = {
                "home_xi_minutes_continuity", "away_xi_minutes_continuity",
                "home_xi_avg_minutes", "away_xi_avg_minutes",
                "home_xi_quality_score", "away_xi_quality_score",
            }
            if missing_in_data:
                if missing_in_data <= _LINEUP_OPTIONAL:
                    # Only optional lineup features missing — informational.
                    if result["status"] == "OK":
                        result["status"] = "OK"  # don't downgrade
                    result["issues"].append(
                        f"{model_name}: {len(missing_in_data)} optional "
                        f"lineup-dependent features unavailable (expected): "
                        f"{', '.join(sorted(missing_in_data)[:5])}"
                    )
                else:
                    result["status"] = "CRITICAL"
                    result["issues"].append(
                        f"{model_name}: {len(missing_in_data)} features missing in data: "
                        f"{', '.join(sorted(missing_in_data)[:5])}"
                    )
            result[f"{model_name}_features"] = len(model_features)

    except Exception as e:
        result["status"] = "ERROR"
        result["issues"].append(str(e))
    return result


def check_preseason_coverage() -> Dict:
    """Which clubs will face matchweek 1 with NO pre-season signal.

    This is a prediction-QUALITY check, not a failure check. The pre-season
    friendly signal is worth +11.0pp of XI accuracy at matchweek 1 (measured over
    the two seasons with coverage: 60.6% with it against 49.6% without). A club
    with no friendly rows silently falls back to a league table a whole summer
    stale — the exact regime the signal exists to repair.

    Nothing is broken when this fires. Coverage depends on whether Sofascore
    lists a club's friendlies at all, and on whether they were PLAYED: measured
    2026-08-02, Brighton's only listed friendly was CANCELED (zero players on
    both sides, correctly refused by the writer) and Brentford had none listed.
    Both are real-world facts, not scrape bugs. The point of surfacing them is
    that the consequence — two clubs predicted from a stale table on opening day
    — is otherwise completely invisible.

    Only meaningful before the league table catches up, so it reports OK once the
    current season has played matches.
    """
    out: Dict = {"status": "OK", "season": None, "leagues": {}}
    try:
        import pandas as pd

        from config.leagues import ACTIVE_LEAGUES
        from scraper.sofascore_friendlies import (
            _in_friendly_window,
            current_friendly_season,
            load_club_roster,
        )

        today = datetime.now().date()
        if not _in_friendly_window(today):
            out["detail"] = "outside the friendly window — nothing to cover"
            return out

        # NOT get_current_season(): that rolls on 1 August, the window opens on
        # 1 June, so through June and July the calendar helper names the season
        # that just ended and would open the previous season's parquet.
        season = current_friendly_season(today)
        out["season"] = season
        fpath = (DATA_DIR / "external" / "sofascore"
                 / f"friendlies_{season.replace('-', '_')}.parquet")
        if not fpath.exists():
            out["status"] = "WARNING"
            out["detail"] = f"no friendlies parquet for {season} — nobody has any signal"
            return out

        # Once real league matches exist the stale-table problem is over and the
        # signal has faded by design (PRESEASON_FADE_MATCHES).
        mpath = DATA_DIR / "parsed" / "matches.parquet"
        if mpath.exists():
            m = pd.read_parquet(mpath, columns=["season", "home_score"])
            played = m[(m["season"] == season) & (m["home_score"].notna())]
            if len(played) >= 20:
                out["detail"] = f"{season} under way ({len(played)} played) — signal retired"
                return out

        fr = pd.read_parquet(fpath, columns=["club", "club_league", "is_our_club"])
        fr = fr[fr["is_our_club"]]

        # Read the roster the daily scrape persisted; never ask Sofascore from
        # here. This monitor runs every 30 minutes, so a live lookup would be
        # ~190 requests a day for three months against a source we get banned
        # from — to re-fetch a list that changes once a year.
        roster = load_club_roster()
        if not roster:
            out["status"] = "UNKNOWN"
            out["detail"] = ("club roster missing, stale or stamped to another "
                             "season — run scraper.sofascore_friendlies")
            return out

        for league in ACTIVE_LEAGUES:
            expected = set(roster.get(league, ()))
            have = set(fr[fr["club_league"] == league]["club"])
            # `load_club_roster` returns {} rather than a partial answer, but a
            # league key can still be short if that league 403'd during the
            # scrape — fetch_club_ids logs and CONTINUES, it does not raise. An
            # unguarded empty `expected` makes `expected - have` empty too, and
            # the check would report perfect coverage precisely when Sofascore
            # was blocking us. Guard on cardinality, never on the exception.
            if len(expected) < 18:
                out["status"] = "UNKNOWN"
                out["leagues"][league] = {
                    "clubs": len(expected),
                    "detail": "club list came back short — coverage not computable",
                }
                continue
            missing = sorted(expected - have)
            out["leagues"][league] = {
                "clubs": len(expected),
                "with_friendlies": len(expected) - len(missing),
                "without_friendlies": missing,
            }
            if missing and out["status"] == "OK":
                out["status"] = "WARNING"
        return out
    except Exception as exc:  # noqa: BLE001
        out["status"] = "UNKNOWN"
        out["detail"] = f"{type(exc).__name__}: {exc}"
        return out


def run_health_check() -> Dict:
    """Run all health checks and return unified result."""
    result = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "HEALTHY",
        "data_freshness": check_data_freshness(),
        "data_quality": check_data_quality(),
        "silent_failures": check_silent_failures(),
        "disk_space": check_disk_space(),
        "lineup_sources": check_lineup_sources(),
        "player_stats_coverage": check_player_stats_coverage(),
        "picks_journal_activity": check_picks_journal_activity(),
        "referee_coverage": check_referee_coverage(),
        "match_record_completeness": check_match_record_completeness(),
        "log_sizes": check_log_sizes(),
        "feature_model_alignment": check_feature_model_alignment(),
        "model_freshness": check_model_freshness(),
        "model_consistency": check_model_metadata_consistency(),
        "calibration_drift": check_calibration_drift(),
        "betting_health": check_betting_health(),
        "preseason_coverage": check_preseason_coverage(),
        "system_integrity": check_system_integrity(),
        "issues": [],
    }

    # Determine overall status
    issues = []

    # Data issues
    for name, check in result["data_freshness"].items():
        if check["status"] == "MISSING":
            issues.append(("WARNING", f"Data file missing: {name}"))
        elif check["status"] == "STALE":
            issues.append(("WARNING", f"Data file stale: {name} ({check['age']})"))

    # Model issues
    for name, check in result["model_freshness"].items():
        if check["status"] == "MISSING":
            issues.append(("CRITICAL", f"Model missing: {name}"))
        elif check["status"] == "STALE":
            issues.append(("WARNING", f"Model stale: {name} ({check['age']})"))

    # Model consistency issues (KB#19)
    consistency = result.get("model_consistency", {})
    if consistency.get("status") == "WARNING":
        for check in consistency.get("checks", []):
            if "issue" in check:
                issues.append(("WARNING", f"Model metadata mismatch: {check['issue']}"))

    # Betting issues
    betting_status = result["betting_health"].get("status", "OK")
    if betting_status == "CRITICAL":
        drift_alerts = result["betting_health"]["details"].get("drift", {}).get("alerts", [])
        for a in drift_alerts:
            issues.append(("CRITICAL", a["message"]))
    elif betting_status == "WARNING":
        drift_alerts = result["betting_health"]["details"].get("drift", {}).get("alerts", [])
        for a in drift_alerts:
            issues.append(("WARNING", a["message"]))

    # Pre-season XI coverage. WARNING, never CRITICAL: nothing is broken when a
    # club has no friendlies (Sofascore may not list them, or they may have been
    # canceled). What the line buys is visibility — otherwise those clubs quietly
    # fall back to a summer-stale table on opening day, ~11pp worse at MW1.
    pre = result.get("preseason_coverage", {})
    if pre.get("status") == "WARNING":
        for league, det in pre.get("leagues", {}).items():
            gap = det.get("without_friendlies") or []
            if gap:
                issues.append(("WARNING",
                               f"No pre-season friendlies for {len(gap)} {league} "
                               f"club(s): {', '.join(gap)} — MW1 XI falls back to "
                               f"the stale league table"))
        if not pre.get("leagues"):
            issues.append(("WARNING", f"Pre-season coverage: {pre.get('detail')}"))
    elif pre.get("status") == "UNKNOWN":
        issues.append(("WARNING",
                       f"Pre-season coverage not computable: "
                       f"{pre.get('detail') or 'club list unavailable'}"))

    # Live calibration drift (rolling ECE on archived 1X2 predictions)
    calib = result.get("calibration_drift", {}).get("calibration_1x2", {})
    if calib.get("status") in ("WARNING", "CRITICAL"):
        # Quote the null floor, not a constant: an ECE of 0.09 is alarming at
        # n=800 and unremarkable at n=100, and the alert text is where that
        # distinction was previously lost.
        issues.append((calib["status"],
                       f"1X2 calibration drift: rolling-{calib.get('n')} ECE "
                       f"{calib.get('ece')} vs perfectly-calibrated floor "
                       f"p90 {calib.get('null_ece_p90')} / "
                       f"p99 {calib.get('null_ece_p99')} "
                       f"({calib.get('newest_graded_age_days')}d old)"))

    # System issues
    if not result["system_integrity"].get("imports_ok", True):
        issues.append(("CRITICAL", "Missing critical Python packages"))

    # Disk space issues
    lu = result.get("lineup_sources", {})
    if lu.get("status") == "CRITICAL":
        issues.append(("CRITICAL", f"Lineup chain: {lu.get('reason')}"))
    ps = result.get("player_stats_coverage", {})
    if ps.get("status") in ("CRITICAL", "WARNING"):
        issues.append((ps["status"], f"Player stats: {ps.get('detail')}"))
    pj = result.get("picks_journal_activity", {})
    if pj.get("status") == "WARNING":
        issues.append(("WARNING", f"Picks journal: {pj.get('detail')}"))
    rc = result.get("referee_coverage", {})
    if rc.get("status") == "WARNING":
        issues.append(("WARNING", f"Referees: {rc.get('detail')}"))
    mr = result.get("match_record_completeness", {})
    if mr.get("status") == "WARNING":
        issues.append(("WARNING", f"Match record: {mr.get('detail')}"))

    disk = result.get("disk_space", {})
    if disk.get("status") == "CRITICAL":
        issues.append(("CRITICAL", f"Disk space critically low: {disk.get('free_pct', 0):.1f}% free ({disk.get('free_gb', 0)} GB)"))
    elif disk.get("status") == "WARNING":
        issues.append(("WARNING", f"Disk space low: {disk.get('free_pct', 0):.1f}% free"))

    # Log size issues
    logs = result.get("log_sizes", {})
    if logs.get("status") == "WARNING":
        for lg in logs.get("large_logs", []):
            issues.append(("WARNING", f"Large log file: {lg['file']} ({lg['size_mb']} MB)"))

    # Feature-model alignment issues. Optional lineup-dependent features
    # missing is expected (handled at predict time as NaN); demote to INFO.
    alignment = result.get("feature_model_alignment", {})
    align_status = alignment.get("status", "OK")
    for iss in alignment.get("issues", []):
        if "optional lineup-dependent" in iss:
            issues.append(("INFO", f"Feature note: {iss}"))
        elif align_status == "CRITICAL":
            issues.append(("CRITICAL", f"Feature-model mismatch: {iss}"))
        else:
            issues.append(("WARNING", f"Feature-model mismatch: {iss}"))

    # Data quality issues
    dq = result.get("data_quality", {})
    for check_name, check_data in dq.items():
        if isinstance(check_data, dict) and check_data.get("status") in ("WARNING", "CRITICAL"):
            for iss in check_data.get("issues", []):
                issues.append((check_data["status"], f"Data quality ({check_name}): {iss}"))

    # Silent-failure issues (the classes the 2026-05-31 diagnostic found —
    # things that left every other check GREEN while the pipeline rotted).
    sf = result.get("silent_failures", {})
    for check_name, check_data in sf.items():
        if isinstance(check_data, dict) and check_data.get("status") in ("WARNING", "CRITICAL"):
            for iss in check_data.get("issues", []):
                issues.append((check_data["status"], f"Silent failure ({check_name}): {iss}"))

    result["issues"] = issues

    # Overall status
    if any(level == "CRITICAL" for level, _ in issues):
        result["overall_status"] = "CRITICAL"
    elif any(level == "WARNING" for level, _ in issues):
        result["overall_status"] = "WARNING"

    # NOTE: Notifications are NOT sent here — callers (monitor.py, CLI) are
    # responsible for deciding when/how to alert. This prevents duplicate
    # notifications when run_health_check() is called from monitor.py which
    # has its own notification logic with deduplication.

    return result


def print_health_check(result: Dict):
    """Print formatted health check to console."""
    status = result["overall_status"]
    status_icon = {"HEALTHY": "[OK]", "WARNING": "[!!]", "CRITICAL": "[XX]"}.get(status, "[??]")

    print()
    print("=" * 60)
    print(f"  SYSTEM HEALTH CHECK  {status_icon} {status}")
    print(f"  {result['timestamp'][:19]}")
    print("=" * 60)

    # Data freshness
    print("\n  DATA FRESHNESS:")
    for name, check in result["data_freshness"].items():
        icon = "[OK]" if check["status"] == "OK" else "[!!]" if check["status"] == "STALE" else "[XX]"
        print(f"    {icon} {name:<22} {check['age']:>12}")

    # Models
    print("\n  MODELS:")
    for name, check in result["model_freshness"].items():
        icon = "[OK]" if check["status"] == "OK" else "[!!]" if check["status"] == "STALE" else "[XX]"
        print(f"    {icon} {name:<22} {check['age']:>12}")

    # Betting
    print("\n  BETTING:")
    journal = result["betting_health"].get("details", {}).get("journal", {})
    if journal and not journal.get("error"):
        print(f"    Bets: {journal.get('total_bets', 0)} total "
              f"({journal.get('pending', 0)} pending, {journal.get('settled', 0)} settled)")
        print(f"    ROI:  {journal.get('roi_pct', 0):+.1f}%  |  "
              f"P&L: ${journal.get('total_profit', 0):+,.2f}")
        clv = journal.get("clv_avg_pct", 0)
        if clv:
            print(f"    CLV:  {clv:+.1f}%")
    bankroll = result["betting_health"].get("details", {}).get("bankroll", {})
    if bankroll:
        current = bankroll.get("current", 0)
        initial = bankroll.get("initial", 0)
        growth = ((current / initial) - 1) * 100 if initial > 0 else 0
        print(f"    Bank: ${current:,.2f} (${initial:,.2f} initial, {growth:+.1f}%)")

    # System
    print("\n  SYSTEM:")
    integrity = result["system_integrity"]
    for pkg, status in integrity.get("checks", {}).items():
        if status == "OK" or status.startswith("OK"):
            continue  # Only show issues
        print(f"    [!!] {pkg}: {status}")
    if all(s == "OK" or s.startswith("OK") for s in integrity.get("checks", {}).values()):
        print(f"    [OK] All packages and configs verified")

    # Issues summary
    issues = result.get("issues", [])
    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for level, msg in issues:
            icon = "[!!]" if level == "WARNING" else "[XX]"
            print(f"    {icon} {msg}")
    else:
        print(f"\n  No issues detected")

    print()
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Production health check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = run_health_check()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_health_check(result)

    # Send notification for critical issues when run standalone (not via monitor)
    critical = [i for i in result.get("issues", []) if i[0] == "CRITICAL"]
    if critical:
        try:
            from scripts.pipeline.notify import notify
            notify("\n".join(i[1] for i in critical),
                   title="Health Check: CRITICAL", level="error", category="alert")
        except Exception:
            pass

    # Exit codes
    if result["overall_status"] == "CRITICAL":
        sys.exit(2)
    elif result["overall_status"] == "WARNING":
        sys.exit(1)
    sys.exit(0)
