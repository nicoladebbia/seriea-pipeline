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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, MODELS_DIR

# ─── Staleness thresholds ───
MAX_FEATURES_AGE_DAYS = 7       # Features should be rebuilt weekly
MAX_MODEL_AGE_DAYS = 30         # Models should be retrained monthly
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


def check_data_freshness() -> Dict:
    """Check freshness of key data files."""
    checks = {}

    files = {
        "features.parquet": DATA_DIR / "features" / "features.parquet",
        "fixtures": DATA_DIR / "upcoming" / "matches.json",
        "odds_data": DATA_DIR / "upcoming" / "odds.json",
        "predictions": DATA_DIR / "upcoming" / "predictions.json",
        "results": DATA_DIR / "upcoming" / "results.json",
        "unified_report": DATA_DIR / "betting" / "unified_report.json",
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

    for name, path in active_models.items():
        age_hours, age_str = _file_age(path)
        status = "OK"

        if age_hours < 0:
            status = "MISSING"
        elif age_hours > MAX_MODEL_AGE_DAYS * 24:
            status = "STALE"

        checks[name] = {
            "exists": path.exists(),
            "age": age_str,
            "status": status,
        }

    return checks


def check_betting_health() -> Dict:
    """Check betting system health from journal stats and drift alerts."""
    result = {"status": "OK", "details": {}}

    # Journal stats
    try:
        from scripts.betting.bet_journal import get_journal_stats
        stats = get_journal_stats()
        result["details"]["journal"] = {
            "total_bets": stats.get("total_bets", 0),
            "pending": stats.get("pending", 0),
            "settled": stats.get("settled", 0),
            "roi_pct": stats.get("roi_pct", 0),
            "total_profit": stats.get("total_profit", 0),
            "clv_avg_pct": stats.get("clv_avg_pct", 0),
        }
    except Exception as e:
        result["details"]["journal"] = {"error": str(e)}

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

    # Bankroll
    bankroll_file = DATA_DIR / "betting" / "bankroll.json"
    if bankroll_file.exists():
        try:
            with open(bankroll_file) as f:
                bankroll = json.load(f)
            result["details"]["bankroll"] = {
                "current": bankroll.get("current_balance", 0),
                "initial": bankroll.get("initial_balance", 0),
            }
        except Exception:
            pass

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


def check_calibration_drift() -> Dict:
    """Rolling live-calibration check on archived pre-kickoff 1X2 predictions.

    Computes 10-bin confidence-ECE over the most recent CALIB_WINDOW archived
    predictions that joined to a real result (same leak-free join the Track
    Record page uses: archived_at < kickoff snapshots only). A model can stay
    accurate while its probabilities drift — this catches the drift between
    retrains. Thresholds: ECE > 0.10 CRITICAL, > 0.06 WARNING. Fewer than
    CALIB_MIN_N graded predictions = SKIP (off-season, early season).

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

        def _ece(window: list) -> tuple[float, int]:
            probs = [w[0] for w in window]
            hits = [w[1] for w in window]
            total, bins_used = 0.0, 0
            for b in range(10):
                sel = [i for i, q in enumerate(probs) if int(min(q, 0.999) * 10) == b]
                if len(sel) < 5:
                    continue
                bins_used += 1
                conf = sum(probs[i] for i in sel) / len(sel)
                acc_ = sum(hits[i] for i in sel) / len(sel)
                total += abs(conf - acc_) * len(sel) / len(window)
            return total, bins_used

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

        ece, bins_used = _ece(window)
        out["ece"] = round(ece, 4)
        out["bins_used"] = bins_used
        graded_disp.sort()
        dwin = [(g[1], g[2]) for g in graded_disp[-CALIB_WINDOW:]]
        if len(dwin) >= CALIB_MIN_N:
            out["display_ece"] = round(_ece(dwin)[0], 4)  # informational only
        out["status"] = ("CRITICAL" if ece > 0.10 else
                         "WARNING" if ece > 0.06 else "OK")
    except Exception as e:
        out["status"] = "SKIP"
        out["reason"] = f"check failed: {e}"
    return {"calibration_1x2": out}


def check_disk_space() -> Dict:
    """Check available disk space on the project partition."""
    import shutil
    try:
        stat = shutil.disk_usage(str(DATA_DIR))
        free_pct = (stat.free / stat.total) * 100
        status = "OK"
        if free_pct < 5:
            status = "CRITICAL"
        elif free_pct < 20:
            status = "WARNING"
        return {
            "status": status,
            "free_gb": round(stat.free / 1e9, 1),
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


def run_health_check() -> Dict:
    """Run all health checks and return unified result."""
    result = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "HEALTHY",
        "data_freshness": check_data_freshness(),
        "data_quality": check_data_quality(),
        "silent_failures": check_silent_failures(),
        "disk_space": check_disk_space(),
        "log_sizes": check_log_sizes(),
        "feature_model_alignment": check_feature_model_alignment(),
        "model_freshness": check_model_freshness(),
        "model_consistency": check_model_metadata_consistency(),
        "calibration_drift": check_calibration_drift(),
        "betting_health": check_betting_health(),
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

    # Live calibration drift (rolling ECE on archived 1X2 predictions)
    calib = result.get("calibration_drift", {}).get("calibration_1x2", {})
    if calib.get("status") in ("WARNING", "CRITICAL"):
        issues.append((calib["status"],
                       f"1X2 calibration drift: rolling-{calib.get('n')} ECE "
                       f"{calib.get('ece')} (warn >0.06, crit >0.10)"))

    # System issues
    if not result["system_integrity"].get("imports_ok", True):
        issues.append(("CRITICAL", "Missing critical Python packages"))

    # Disk space issues
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
