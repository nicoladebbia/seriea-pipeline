#!/usr/bin/env python3
"""Rebuild features from clean Serie A data with quality validation.

This script:
1. Verifies data is Serie A only (no multi-league contamination)
2. Runs the full feature pipeline
3. Audits feature quality (NaN rates, data types)
4. Reports usable feature count
5. Saves clean features for training

Usage:
    python scripts/rebuild_features.py [--season SEASON] [--audit-only]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.build import build_features, get_ml_feature_columns
from storage.paths import parsed_path, features_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def verify_data_quality(matches_path: Path) -> dict:
    """Verify matches data is clean before building features."""
    df = pd.read_parquet(matches_path)

    issues = []

    # Check for multi-league contamination
    if "league" in df.columns:
        leagues = df["league"].unique()
        if len(leagues) > 1:
            issues.append(f"MULTI-LEAGUE CONTAMINATION: Found {len(leagues)} leagues: {list(leagues)}")
        elif leagues[0] != "serie_a":
            issues.append(f"WRONG LEAGUE: Expected 'serie_a', found '{leagues[0]}'")

    # Check date range
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    date_range = f"{df['match_date'].min().date()} to {df['match_date'].max().date()}"

    # Check for missing scores (incomplete matches)
    missing_scores = df["home_score"].isna().sum()

    # Check for duplicate match_ids
    duplicates = df["match_id"].duplicated().sum()

    return {
        "total_matches": len(df),
        "leagues": list(df.get("league", pd.Series(["unknown"])).unique()),
        "seasons": sorted(df["season"].unique().tolist()),
        "date_range": date_range,
        "missing_scores": missing_scores,
        "duplicate_ids": duplicates,
        "issues": issues,
    }


def audit_features(df: pd.DataFrame) -> dict:
    """Audit feature quality after building."""
    ml_cols = get_ml_feature_columns(df)

    # Calculate NaN rates
    nan_rates = {}
    for col in ml_cols:
        if col in df.columns:
            nan_rate = df[col].isna().sum() / len(df)
            nan_rates[col] = nan_rate

    # Categorize features
    excellent = [c for c, r in nan_rates.items() if r < 0.01]  # <1% NaN
    good = [c for c, r in nan_rates.items() if 0.01 <= r < 0.05]  # 1-5% NaN
    usable = [c for c, r in nan_rates.items() if 0.05 <= r < 0.20]  # 5-20% NaN
    poor = [c for c, r in nan_rates.items() if 0.20 <= r < 0.50]  # 20-50% NaN
    garbage = [c for c, r in nan_rates.items() if r >= 0.50]  # 50%+ NaN

    # Feature type breakdown
    numeric_cols = df[ml_cols].select_dtypes(include=[np.number]).columns.tolist()

    return {
        "total_features": len(ml_cols),
        "numeric_features": len(numeric_cols),
        "excellent_quality": len(excellent),
        "good_quality": len(good),
        "usable_quality": len(usable),
        "poor_quality": len(poor),
        "garbage_quality": len(garbage),
        "nan_rates": nan_rates,
        "excellent_features": excellent,
        "garbage_features": garbage,
    }


def print_audit_report(data_report: dict, feature_report: dict):
    """Print comprehensive audit report."""
    print("\n" + "=" * 70)
    print("FOOTBALL INTELLIGENCE SYSTEM - DATA QUALITY REPORT")
    print("=" * 70)

    # Data quality
    print("\n📊 DATA QUALITY")
    print("-" * 40)
    print(f"  Total matches: {data_report['total_matches']:,}")
    print(f"  Leagues: {data_report['leagues']}")
    print(f"  Seasons: {len(data_report['seasons'])} ({data_report['seasons'][0]} - {data_report['seasons'][-1]})")
    print(f"  Date range: {data_report['date_range']}")
    print(f"  Missing scores: {data_report['missing_scores']}")
    print(f"  Duplicate IDs: {data_report['duplicate_ids']}")

    if data_report["issues"]:
        print("\n  ⚠️ ISSUES:")
        for issue in data_report["issues"]:
            print(f"    - {issue}")
    else:
        print("\n  ✅ No data quality issues detected")

    # Feature quality
    print("\n📈 FEATURE QUALITY")
    print("-" * 40)
    print(f"  Total ML features: {feature_report['total_features']}")
    print(f"  Numeric features: {feature_report['numeric_features']}")
    print(f"\n  Quality breakdown:")
    print(f"    ✅ Excellent (<1% NaN):  {feature_report['excellent_quality']}")
    print(f"    ✅ Good (1-5% NaN):      {feature_report['good_quality']}")
    print(f"    ⚠️ Usable (5-20% NaN):  {feature_report['usable_quality']}")
    print(f"    ❌ Poor (20-50% NaN):    {feature_report['poor_quality']}")
    print(f"    💀 Garbage (50%+ NaN):   {feature_report['garbage_quality']}")

    usable_count = (feature_report['excellent_quality'] +
                   feature_report['good_quality'] +
                   feature_report['usable_quality'])
    print(f"\n  🎯 USABLE FEATURES: {usable_count} / {feature_report['total_features']}")

    if feature_report['garbage_features']:
        print(f"\n  💀 Garbage features (will be dropped):")
        for feat in sorted(feature_report['garbage_features'])[:20]:
            nan_rate = feature_report['nan_rates'][feat]
            print(f"    - {feat}: {nan_rate*100:.1f}% NaN")
        if len(feature_report['garbage_features']) > 20:
            print(f"    ... and {len(feature_report['garbage_features']) - 20} more")

    print("\n" + "=" * 70)


def clean_features(df: pd.DataFrame, feature_report: dict) -> pd.DataFrame:
    """Remove garbage features and fill remaining NaNs."""
    garbage = feature_report['garbage_features']

    # Drop garbage features
    df_clean = df.drop(columns=[c for c in garbage if c in df.columns], errors='ignore')
    log.info(f"Dropped {len(garbage)} garbage features")

    # Get remaining ML columns
    ml_cols = get_ml_feature_columns(df_clean)
    numeric_cols = df_clean[ml_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Fill NaNs in numeric columns with 0 (conservative)
    for col in numeric_cols:
        if col in df_clean.columns:
            nan_count = df_clean[col].isna().sum()
            if nan_count > 0:
                df_clean[col] = df_clean[col].fillna(0)

    return df_clean


def main():
    parser = argparse.ArgumentParser(description="Rebuild features with quality validation")
    parser.add_argument("--season", type=str, help="Build for specific season only")
    parser.add_argument("--audit-only", action="store_true", help="Only audit existing features")
    parser.add_argument("--no-clean", action="store_true", help="Don't remove garbage features")
    args = parser.parse_args()

    matches_path = parsed_path("matches")
    feat_path = features_path()

    log.info("=" * 70)
    log.info("REBUILDING FEATURES - FOOTBALL INTELLIGENCE SYSTEM")
    log.info("=" * 70)

    # Step 1: Verify source data
    log.info("\nStep 1: Verifying source data quality...")
    if not matches_path.exists():
        log.error(f"Matches file not found: {matches_path}")
        log.error("Run the scraper/parser first to create matches.parquet")
        return

    data_report = verify_data_quality(matches_path)

    if data_report["issues"]:
        log.warning("Data quality issues detected:")
        for issue in data_report["issues"]:
            log.warning(f"  - {issue}")

    if args.audit_only and feat_path.exists():
        # Audit existing features
        log.info("\nAuditing existing features...")
        df = pd.read_parquet(feat_path)
        feature_report = audit_features(df)
        print_audit_report(data_report, feature_report)
        return

    # Step 2: Build features
    log.info("\nStep 2: Building features (this may take a few minutes)...")
    df = build_features(season=args.season)

    if df.empty:
        log.error("Feature building failed - empty DataFrame")
        return

    # Step 3: Audit feature quality
    log.info("\nStep 3: Auditing feature quality...")
    feature_report = audit_features(df)

    # Step 4: Clean features (remove garbage)
    if not args.no_clean:
        log.info("\nStep 4: Cleaning features (removing garbage)...")
        df = clean_features(df, feature_report)

        # Re-audit after cleaning
        feature_report = audit_features(df)

    # Step 5: Save clean features
    log.info("\nStep 5: Saving clean features...")
    df.to_parquet(feat_path, index=False)
    log.info(f"Saved to {feat_path}")

    # Print report
    print_audit_report(data_report, feature_report)

    # Summary
    usable_count = (feature_report['excellent_quality'] +
                   feature_report['good_quality'] +
                   feature_report['usable_quality'])

    log.info("\n" + "=" * 70)
    log.info("REBUILD COMPLETE")
    log.info("=" * 70)
    log.info(f"Matches: {data_report['total_matches']:,}")
    log.info(f"Features: {usable_count} usable / {feature_report['total_features']} total")
    log.info(f"Output: {feat_path}")

    return df


if __name__ == "__main__":
    main()
