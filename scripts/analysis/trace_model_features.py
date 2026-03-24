"""Trace where each model feature comes from and check prediction-time availability"""
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier

# Load model
model_path = Path(str(Path(__file__).parent.parent.parent) + '/data/models/universal/no_odds/catboost_latest.cbm')
model = CatBoostClassifier()
model.load_model(str(model_path))

features = model.feature_names_
print("=" * 80)
print("MODEL FEATURE TRACE")
print("=" * 80)
print(f"Total features used by CatBoost: {len(features)}\n")

# Load features.parquet
features_path = Path(str(Path(__file__).parent.parent.parent) + '/data/features/features.parquet')
df = pd.read_parquet(features_path)

# Filter recent seasons
recent = df[df['season'].astype(str).str.contains('2023|2024|2025|2026')]

# Categorize features
categories = {
    'Elo/Strength': ['elo', 'strength', 'attack', 'defense'],
    'H2H': ['h2h_'],
    'Form/Rolling': ['roll', 'gd_', '_per_match'],
    'Tactical': ['formation', 'pressing', 'ppda'],
    'Venue': ['venue_', 'stadium', 'travel'],
    'Manager': ['tenure', 'chemistry'],
    'Injury': ['injury'],
    'Congestion': ['congestion'],
    'League Position': ['league_', '_to_cl', '_to_rel'],
    'Match Context': ['is_late', 'is_derby', 'is_midweek'],
    'Referee': ['ref_'],
    'Metadata': ['_has_'],
}

def categorize_feature(feat):
    for cat, patterns in categories.items():
        if any(p in feat for p in patterns):
            return cat
    return 'Other/Derived'

print("FEATURE BREAKDOWN BY CATEGORY:")
print("-" * 80)

category_counts = {}
for feat in features:
    cat = categorize_feature(feat)
    if cat not in category_counts:
        category_counts[cat] = []
    category_counts[cat].append(feat)

for cat in sorted(category_counts.keys()):
    feats = category_counts[cat]
    print(f"\n{cat} ({len(feats)} features):")
    for f in sorted(feats):
        # Check if exists in features.parquet
        if f in df.columns:
            nan_rate = recent[f].isna().mean() * 100
            status = "OK" if nan_rate < 30 else f"WARN {nan_rate:.1f}% NaN"
        else:
            status = "MISSING from features.parquet"
        print(f"  - {f}: {status}")

# Check metadata features specifically
print("\n" + "=" * 80)
print("METADATA FEATURES (prediction-time injection)")
print("=" * 80)
metadata_features = [f for f in features if f.startswith('_has_')]
for f in metadata_features:
    if f in df.columns:
        value_counts = df[f].value_counts()
        print(f"\n{f}:")
        print(f"  In features.parquet: YES")
        print(f"  Value distribution:")
        for val, count in value_counts.items():
            print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print(f"\n{f}: MISSING from features.parquet (injected at prediction time)")

print("\n" + "=" * 80)
print("FEATURE AVAILABILITY CHECK")
print("=" * 80)

missing = [f for f in features if f not in df.columns]
high_nan = []
for f in features:
    if f in df.columns and recent[f].isna().mean() > 0.5:
        high_nan.append((f, recent[f].isna().mean() * 100))

if missing:
    print(f"\nMISSING from features.parquet ({len(missing)} features):")
    for f in missing:
        print(f"  - {f}")

if high_nan:
    print(f"\nHIGH NaN RATE in recent seasons ({len(high_nan)} features):")
    for f, rate in sorted(high_nan, key=lambda x: x[1], reverse=True):
        print(f"  - {f}: {rate:.1f}%")

if not missing and not high_nan:
    print("\nALL FEATURES AVAILABLE with <50% NaN rate")
