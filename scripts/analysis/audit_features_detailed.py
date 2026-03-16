"""Feature engineering pipeline audit script"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load features
features_path = project_root / 'data/features/features.parquet'
df = pd.read_parquet(features_path)

print("=" * 80)
print("FEATURE TABLE SUMMARY")
print("=" * 80)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}")

# Check what columns exist
print(f"\nFirst 20 columns: {list(df.columns[:20])}")

# Look for date column
date_cols = [c for c in df.columns if 'date' in c.lower()]
print(f"\nDate-related columns: {date_cols}")

# Use match_date if date doesn't exist
date_col = 'match_date' if 'match_date' in df.columns else ('date' if 'date' in df.columns else None)
if date_col:
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
print()

# Season breakdown
if 'season' in df.columns:
    print("Rows by season:")
    print(df['season'].value_counts().sort_index())
    print()

# NaN analysis
print("=" * 80)
print("NaN RATE ANALYSIS")
print("=" * 80)

nan_rates = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)

print(f"\nColumns with >50% NaN (unusable):")
high_nan = nan_rates[nan_rates > 50]
print(f"Count: {len(high_nan)}")
if len(high_nan) > 0:
    for col, rate in high_nan.head(30).items():
        print(f"  {col}: {rate:.1f}%")

print(f"\nColumns with 30-50% NaN (risky):")
medium_nan = nan_rates[(nan_rates > 30) & (nan_rates <= 50)]
print(f"Count: {len(medium_nan)}")
if len(medium_nan) > 0:
    for col, rate in medium_nan.head(20).items():
        print(f"  {col}: {rate:.1f}%")

print(f"\nColumns with <5% NaN (good quality):")
good_cols = nan_rates[nan_rates < 5]
print(f"Count: {len(good_cols)}")

# Constant columns
print("\n" + "=" * 80)
print("CONSTANT COLUMNS (zero variance)")
print("=" * 80)
constant_cols = []
for col in df.columns:
    if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
        if df[col].nunique() <= 1:
            constant_cols.append(col)

print(f"Count: {len(constant_cols)}")
if constant_cols:
    print("Columns:", constant_cols[:20])

# Recent seasons analysis (2023-2026)
if 'season' in df.columns:
    print("\n" + "=" * 80)
    print("RECENT SEASONS (2023-2026) NaN ANALYSIS")
    print("=" * 80)
    # Filter using string contains instead of exact match
    recent = df[df['season'].astype(str).str.contains('2023|2024|2025|2026')]
    print(f"Recent season rows: {len(recent):,}")

    if len(recent) > 0:
        recent_nan = (recent.isna().sum() / len(recent) * 100).sort_values(ascending=False)
        print(f"\nColumns with >30% NaN in recent seasons:")
        recent_high_nan = recent_nan[recent_nan > 30]
        print(f"Count: {len(recent_high_nan)}")
        if len(recent_high_nan) > 0:
            for col, rate in recent_high_nan.head(40).items():
                print(f"  {col}: {rate:.1f}%")

        # Check specific feature categories
        print("\n" + "=" * 80)
        print("FEATURE CATEGORY NaN RATES (Recent Seasons)")
        print("=" * 80)

        categories = {
            'h2h': [c for c in df.columns if 'h2h' in c.lower()],
            'sofascore': [c for c in df.columns if any(x in c.lower() for x in ['momentum', 'captain', 'card_timing', 'corner', 'dangerous_attack'])],
            'understat': [c for c in df.columns if 'xg' in c.lower() or 'npxg' in c.lower()],
            'player': [c for c in df.columns if 'player' in c.lower()],
            'weather': [c for c in df.columns if 'weather' in c.lower() or 'temp' in c.lower() or 'wind' in c.lower()],
            'referee': [c for c in df.columns if 'ref' in c.lower()],
            'tactical': [c for c in df.columns if any(x in c.lower() for x in ['formation', 'ppda'])],
        }

        for cat_name, cols in categories.items():
            if cols:
                avg_nan = recent[cols].isna().mean().mean() * 100
                print(f"\n{cat_name.upper()}: {len(cols)} columns, avg {avg_nan:.1f}% NaN")
                worst_cols = recent[cols].isna().mean().sort_values(ascending=False).head(5)
                for col, rate in worst_cols.items():
                    print(f"  - {col}: {rate*100:.1f}%")

        # Usable features summary
        print("\n" + "=" * 80)
        print("USABLE FEATURES SUMMARY")
        print("=" * 80)
        usable = recent_nan[recent_nan < 50]
        print(f"Columns with <50% NaN in recent seasons: {len(usable)}")

        non_constant_usable = [c for c in usable.index if c not in constant_cols]
        print(f"Non-constant, <50% NaN: {len(non_constant_usable)}")

# Check feature types
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"\nNumeric columns: {len(numeric_cols)}")
print(f"Non-numeric columns: {len(df.columns) - len(numeric_cols)}")
