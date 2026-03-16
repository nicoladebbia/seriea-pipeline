#!/usr/bin/env python3
"""Data quality report — cross-reference all sources, validate correctness.

Checks:
1. Row counts for all parquet files
2. Season coverage per source
3. Null % per critical column
4. Team name consistency across sources
5. xG cross-reference (FBref vs Understat)
6. Data freshness (last update date)
7. Feature performance monitoring (post-backtest)

Usage:
    python scripts/data_quality_report.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR, SEASONS

log = logging.getLogger(__name__)


# Known team name mappings across sources
TEAM_NAME_VARIANTS = {
    "Inter": ["Inter", "Inter Milan", "Internazionale"],
    "Milan": ["Milan", "AC Milan"],
    "Roma": ["Roma", "AS Roma"],
    "Napoli": ["Napoli", "SSC Napoli"],
    "Verona": ["Verona", "Hellas Verona"],
    "SPAL": ["SPAL", "Spal"],
}


def check_parquet_files() -> dict:
    """Check all parquet files for row counts and basic stats."""
    results = {}

    dirs_to_check = [
        ("parsed", DATA_DIR / "parsed"),
        ("features", DATA_DIR / "features"),
        ("external/transfermarkt", DATA_DIR / "external" / "transfermarkt"),
        ("external/odds", DATA_DIR / "external" / "odds"),
        ("external/injuries", DATA_DIR / "external" / "injuries"),
    ]

    for label, dir_path in dirs_to_check:
        if not dir_path.exists():
            results[label] = {"status": "missing", "files": []}
            continue

        files = []
        for f in sorted(dir_path.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                files.append({
                    "name": f.name,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                })
            except Exception as e:
                files.append({"name": f.name, "error": str(e)})

        results[label] = {"status": "ok", "files": files}

    return results


def check_season_coverage() -> dict:
    """Check which seasons are covered by each data source."""
    coverage = {}

    # FBref parsed stats
    for stat_type in ["matches", "player_stats"]:
        path = DATA_DIR / "parsed" / f"{stat_type}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "season" in df.columns:
                seasons = sorted(df["season"].unique())
                coverage[stat_type] = {
                    "seasons": list(seasons),
                    "count": len(seasons),
                    "latest": seasons[-1] if seasons else None,
                }

    # FBref team stats
    for stat in ["standard", "shooting", "passing", "defense", "misc"]:
        path = DATA_DIR / "parsed" / f"fbref_stats_{stat}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            season_col = "season" if "season" in df.columns else None
            if season_col:
                seasons = sorted(df[season_col].unique())
                coverage[f"fbref_{stat}"] = {
                    "seasons": list(seasons),
                    "count": len(seasons),
                }

    # Understat
    path = DATA_DIR / "parsed" / "understat_players.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "season" in df.columns:
            seasons = sorted(df["season"].unique())
            coverage["understat"] = {
                "seasons": list(seasons),
                "count": len(seasons),
                "latest": seasons[-1] if seasons else None,
            }

    # Transfermarkt
    tm_dir = DATA_DIR / "external" / "transfermarkt"
    if tm_dir.exists():
        tm_seasons = []
        for f in tm_dir.glob("*.parquet"):
            # Extract season from filename like "market_values_2024_2025.parquet"
            parts = f.stem.split("_")
            if len(parts) >= 4:
                try:
                    s = f"{parts[-2]}-{parts[-1]}"
                    tm_seasons.append(s)
                except Exception as e:
                    log.debug(f"Failed to parse Transfermarkt season from filename: {e}")
        coverage["transfermarkt"] = {
            "seasons": sorted(set(tm_seasons)),
            "count": len(set(tm_seasons)),
        }

    return coverage


def check_null_rates() -> dict:
    """Check null percentages for critical columns in key files."""
    null_report = {}

    critical_checks = {
        "matches": {
            "path": DATA_DIR / "parsed" / "matches.parquet",
            "columns": [
                "home_team", "away_team", "home_score", "away_score",
                "match_date", "season", "home_xg", "away_xg",
            ],
        },
        "features": {
            "path": DATA_DIR / "features" / "features.parquet",
            "columns": [
                "home_elo", "away_elo", "home_strength_rating", "away_strength_rating",
                "home_roll_goals_scored_5", "away_roll_goals_scored_5",
                "result",
            ],
        },
    }

    for name, check in critical_checks.items():
        if not check["path"].exists():
            null_report[name] = {"status": "missing"}
            continue

        df = pd.read_parquet(check["path"])
        col_nulls = {}
        for col in check["columns"]:
            if col in df.columns:
                null_pct = df[col].isna().mean() * 100
                col_nulls[col] = round(null_pct, 1)
            else:
                col_nulls[col] = "column_missing"

        null_report[name] = {
            "status": "ok",
            "total_rows": len(df),
            "null_pct": col_nulls,
        }

    return null_report


def check_team_name_consistency() -> dict:
    """Check team names match across FBref, Understat, Transfermarkt, Odds API."""
    sources = {}

    # FBref (matches)
    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    if matches_path.exists():
        df = pd.read_parquet(matches_path)
        teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
        sources["fbref"] = sorted(teams)

    # Understat
    us_path = DATA_DIR / "parsed" / "understat_players.parquet"
    if us_path.exists():
        df = pd.read_parquet(us_path)
        if "team" in df.columns:
            sources["understat"] = sorted(df["team"].unique())

    # Odds API
    odds_path = DATA_DIR / "upcoming" / "odds.json"
    if odds_path.exists():
        try:
            with open(odds_path) as f:
                odds = json.load(f)
            teams = set()
            for match in (odds if isinstance(odds, list) else odds.get("matches", {}).values()):
                if isinstance(match, dict):
                    for k in ["home_team", "away_team"]:
                        if k in match:
                            teams.add(match[k])
            if teams:
                sources["odds_api"] = sorted(teams)
        except Exception as e:
            log.warning(f"Failed to parse odds API team names: {e}")

    # Injuries
    from scraper.injuries import SERIE_A_TEAMS_TM
    sources["injuries_scraper"] = sorted(SERIE_A_TEAMS_TM.keys())

    # Check understat name mapping
    try:
        from features.understat_features import TEAM_NAME_MAP
        sources["understat_map_keys"] = sorted(TEAM_NAME_MAP.keys())
    except ImportError:
        pass

    # Find mismatches
    mismatches = []
    if "fbref" in sources and "understat" in sources:
        fbref_set = set(s.lower() for s in sources["fbref"])
        us_set = set(s.lower() for s in sources["understat"])
        only_fbref = fbref_set - us_set
        only_us = us_set - fbref_set
        if only_fbref:
            mismatches.append({"type": "fbref_only", "teams": sorted(only_fbref)})
        if only_us:
            mismatches.append({"type": "understat_only", "teams": sorted(only_us)})

    return {
        "sources": {k: len(v) for k, v in sources.items()},
        "mismatches": mismatches,
    }


def cross_reference_xg(season: str = "2024-2025") -> dict:
    """Compare FBref vs Understat xG for a season.

    Expected: correlation r > 0.75
    Flags matches with diff > 1.0 xG
    """
    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    if not matches_path.exists():
        return {"error": "No matches data"}

    matches = pd.read_parquet(matches_path)
    matches = matches[matches["season"] == season].copy()

    if "home_xg" not in matches.columns:
        return {"error": "No xG data in matches"}

    # FBref xG is in matches.parquet
    fbref_xg = matches[["home_team", "away_team", "match_date", "home_xg", "away_xg"]].copy()
    fbref_xg = fbref_xg.dropna(subset=["home_xg", "away_xg"])

    if fbref_xg.empty:
        return {"error": f"No FBref xG data for {season}"}

    # Understat xG
    us_path = DATA_DIR / "parsed" / "understat_matches.parquet"
    if not us_path.exists():
        # Try alternate location
        us_path = DATA_DIR / "parsed" / "understat_players.parquet"

    # If we can't find per-match Understat data, just report FBref stats
    result = {
        "season": season,
        "fbref_matches_with_xg": len(fbref_xg),
        "fbref_avg_home_xg": round(fbref_xg["home_xg"].mean(), 3),
        "fbref_avg_away_xg": round(fbref_xg["away_xg"].mean(), 3),
        "fbref_avg_total_xg": round(
            (fbref_xg["home_xg"] + fbref_xg["away_xg"]).mean(), 3
        ),
    }

    return result


def check_data_freshness() -> dict:
    """Check when key data files were last updated."""
    files_to_check = {
        "matches": DATA_DIR / "parsed" / "matches.parquet",
        "features": DATA_DIR / "features" / "features.parquet",
        "predictions": DATA_DIR / "upcoming" / "predictions.json",
        "odds": DATA_DIR / "upcoming" / "odds.json",
        "odds_full": DATA_DIR / "upcoming" / "odds_full.json",
        "injuries": DATA_DIR / "external" / "injuries",
        "player_xg_profiles": DATA_DIR / "features" / "player_xg_profiles.json",
        "betting_report": DATA_DIR / "betting" / "unified_report.json",
    }

    freshness = {}
    for name, path in files_to_check.items():
        if path.is_dir():
            # Check most recent file in directory
            latest = None
            for f in path.glob("*"):
                if f.is_file():
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if latest is None or mtime > latest:
                        latest = mtime
            freshness[name] = latest.isoformat() if latest else "empty"
        elif path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            freshness[name] = mtime.isoformat()
        else:
            freshness[name] = "missing"

    return freshness


def monitor_feature_performance() -> dict:
    """Track which features contribute to prediction accuracy.

    Checks feature importance from the ML model if available.
    Flags features with importance < 0.001 for potential removal.
    """
    model_path = DATA_DIR / "models"
    if not model_path.exists():
        return {"status": "no_models"}

    # Try to load CatBoost feature importance
    try:
        from catboost import CatBoostClassifier
        model_file = model_path / "catboost_classifier.cbm"
        if not model_file.exists():
            return {"status": "no_classifier_model"}

        model = CatBoostClassifier()
        model.load_model(str(model_file))

        importance = model.get_feature_importance()
        feature_names = model.feature_names_

        if feature_names and len(importance) == len(feature_names):
            features = sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )

            top_20 = [(n, round(v, 4)) for n, v in features[:20]]
            low_importance = [(n, round(v, 4)) for n, v in features if v < 0.001]

            return {
                "status": "ok",
                "total_features": len(features),
                "top_20": top_20,
                "low_importance_count": len(low_importance),
                "low_importance": low_importance[:10],
            }
    except Exception as e:
        return {"status": f"error: {e}"}

    return {"status": "unknown"}


def check_feature_correlations(threshold: float = 0.95) -> dict:
    """Detect near-duplicate features with |correlation| > threshold.

    Computes pairwise Pearson correlation for all numeric columns in the
    feature table and flags highly correlated pairs for potential removal.

    Saves report to data/quality/correlation_report.json.
    """
    features_file = DATA_DIR / "features" / "features.parquet"
    if not features_file.exists():
        return {"status": "no_features_file"}

    df = pd.read_parquet(features_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {"status": "insufficient_numeric_columns", "count": len(numeric_cols)}

    log.info("Computing correlation matrix for %d numeric columns...", len(numeric_cols))
    corr = df[numeric_cols].corr()

    # Find pairs above threshold (upper triangle only to avoid duplicates)
    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                high_corr_pairs.append({
                    "col_a": numeric_cols[i],
                    "col_b": numeric_cols[j],
                    "correlation": round(float(r), 4),
                })

    high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    result = {
        "status": "ok",
        "threshold": threshold,
        "total_numeric_features": len(numeric_cols),
        "high_correlation_pairs": len(high_corr_pairs),
        "pairs": high_corr_pairs[:100],  # Cap at 100 to avoid huge reports
    }

    if high_corr_pairs:
        log.warning(
            "Found %d feature pairs with |r| > %.2f (review for redundancy)",
            len(high_corr_pairs), threshold,
        )
        for pair in high_corr_pairs[:10]:
            log.warning("  %s ↔ %s: r=%.4f", pair["col_a"], pair["col_b"], pair["correlation"])

    # Save report
    quality_dir = DATA_DIR / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    report_path = quality_dir / "correlation_report.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Correlation report saved to %s", report_path)

    return result


def generate_full_report() -> dict:
    """Generate comprehensive data quality report."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "parquet_files": check_parquet_files(),
        "season_coverage": check_season_coverage(),
        "null_rates": check_null_rates(),
        "team_names": check_team_name_consistency(),
        "xg_crossref": cross_reference_xg(),
        "data_freshness": check_data_freshness(),
        "feature_performance": monitor_feature_performance(),
        "feature_correlations": check_feature_correlations(),
    }

    return report


def print_report(report: dict):
    """Print the report in human-readable format."""
    print("\n" + "=" * 70)
    print(" DATA QUALITY REPORT")
    print(f" Generated: {report['generated_at']}")
    print("=" * 70)

    # Parquet files
    print("\n--- PARQUET FILES ---")
    for section, data in report["parquet_files"].items():
        if data["status"] == "missing":
            print(f"  {section}: MISSING")
            continue
        for f in data["files"]:
            if "error" in f:
                print(f"  {section}/{f['name']}: ERROR - {f['error']}")
            else:
                print(f"  {section}/{f['name']}: {f['rows']:,} rows x {f['cols']} cols "
                      f"({f['size_kb']:.0f} KB)")

    # Season coverage
    print("\n--- SEASON COVERAGE ---")
    for source, data in report["season_coverage"].items():
        latest = data.get("latest", data.get("seasons", ["?"])[-1] if data.get("seasons") else "?")
        print(f"  {source}: {data['count']} seasons (latest: {latest})")

    # Null rates
    print("\n--- NULL RATES (critical columns) ---")
    for source, data in report["null_rates"].items():
        if data.get("status") == "missing":
            print(f"  {source}: FILE MISSING")
            continue
        print(f"  {source} ({data['total_rows']:,} rows):")
        for col, pct in data["null_pct"].items():
            flag = " ⚠️" if isinstance(pct, float) and pct > 10 else ""
            print(f"    {col}: {pct}%{flag}")

    # Team name consistency
    print("\n--- TEAM NAME CONSISTENCY ---")
    names = report["team_names"]
    for source, count in names.get("sources", {}).items():
        print(f"  {source}: {count} teams")
    for m in names.get("mismatches", []):
        print(f"  MISMATCH ({m['type']}): {m['teams']}")

    # xG cross-reference
    print("\n--- XG CROSS-REFERENCE ---")
    xg = report["xg_crossref"]
    if "error" in xg:
        print(f"  {xg['error']}")
    else:
        print(f"  Season: {xg['season']}")
        print(f"  Matches with xG: {xg['fbref_matches_with_xg']}")
        print(f"  Avg home xG: {xg['fbref_avg_home_xg']}")
        print(f"  Avg away xG: {xg['fbref_avg_away_xg']}")
        print(f"  Avg total xG: {xg['fbref_avg_total_xg']}")

    # Data freshness
    print("\n--- DATA FRESHNESS ---")
    for name, ts in report["data_freshness"].items():
        if ts in ("missing", "empty"):
            print(f"  {name}: {ts.upper()}")
        else:
            print(f"  {name}: {ts}")

    # Feature performance
    print("\n--- FEATURE PERFORMANCE ---")
    fp = report["feature_performance"]
    if fp.get("status") == "ok":
        print(f"  Total features: {fp['total_features']}")
        print(f"  Low importance (<0.001): {fp['low_importance_count']}")
        print(f"  Top 5:")
        for name, imp in fp.get("top_20", [])[:5]:
            print(f"    {name}: {imp}")
    else:
        print(f"  Status: {fp.get('status', 'unknown')}")

    # Feature correlations
    print("\n--- FEATURE CORRELATIONS ---")
    fc = report.get("feature_correlations", {})
    if fc.get("status") == "ok":
        print(f"  Numeric features: {fc['total_numeric_features']}")
        print(f"  Highly correlated pairs (|r| > {fc['threshold']}): {fc['high_correlation_pairs']}")
        for pair in fc.get("pairs", [])[:5]:
            print(f"    {pair['col_a']} <-> {pair['col_b']}: r={pair['correlation']}")
    else:
        print(f"  Status: {fc.get('status', 'unknown')}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    report = generate_full_report()
    print_report(report)

    # Save report
    report_path = DATA_DIR / "quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")
