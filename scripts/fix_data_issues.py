"""
Fix data issues in matches.parquet and run sanity checks.
Issues addressed:
1. Newcastle vs West Ham 2021-08-15: away_shots_on_target (9) > away_shots_total (8)
2. Delete orphaned understat_players_epl.parquet
3. Flag Genoa vs Inter 2025-12-14 Max odds error (source CSV issue)
4. Sanity checks on the full dataset
"""

import pandas as pd
import os

PARQUET_PATH = "data/parsed/matches.parquet"
ORPHAN_PATH = "data/parsed/understat_players_epl.parquet"

df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df)} matches from {PARQUET_PATH}")

# --- Issue 1: Fix Newcastle vs West Ham shots ---
mask = (
    (df["match_date"] == "2021-08-15")
    & (df["home_team"] == "Newcastle")
    & (df["away_team"] == "West Ham")
)
matches_found = mask.sum()
print(f"\n=== Issue 1: Newcastle vs West Ham 2021-08-15 ===")
print(f"Matches found: {matches_found}")
if matches_found == 1:
    row = df.loc[mask].iloc[0]
    print(f"  BEFORE: away_shots_total={row['away_shots_total']}, away_shots_on_target={row['away_shots_on_target_count']}")
    df.loc[mask, "away_shots_total"] = 9.0
    row_after = df.loc[mask].iloc[0]
    print(f"  AFTER:  away_shots_total={row_after['away_shots_total']}, away_shots_on_target={row_after['away_shots_on_target_count']}")
else:
    print("  WARNING: Expected 1 match, skipping fix")

# --- Issue 2: Delete orphaned file ---
print(f"\n=== Issue 2: Delete orphaned {ORPHAN_PATH} ===")
if os.path.exists(ORPHAN_PATH):
    os.remove(ORPHAN_PATH)
    print(f"  DELETED: {ORPHAN_PATH}")
else:
    print(f"  File not found (already deleted?)")

# --- Issue 3: Flag Genoa vs Inter Max odds ---
print(f"\n=== Issue 3: Genoa vs Inter 2025-12-14 odds analysis ===")
mask3 = df["match_id"] == "2025-12-14_Genoa_Inter"
if mask3.sum() == 0:
    mask3 = (
        (df["match_date"] == "2025-12-14")
        & (df["home_team"].str.contains("Genoa", case=False))
        & (df["away_team"].str.contains("Inter", case=False))
    )
if mask3.sum() == 1:
    row = df.loc[mask3].iloc[0]
    avg_sum = 1/row["odds_AvgH"] + 1/row["odds_AvgD"] + 1/row["odds_AvgA"]
    max_sum = 1/row["odds_MaxH"] + 1/row["odds_MaxD"] + 1/row["odds_MaxA"]
    b365_sum = 1/row["odds_B365H"] + 1/row["odds_B365D"] + 1/row["odds_B365A"]
    print(f"  B365 odds: H={row['odds_B365H']}, D={row['odds_B365D']}, A={row['odds_B365A']}  (sum={b365_sum:.4f})")
    print(f"  Avg odds:  H={row['odds_AvgH']}, D={row['odds_AvgD']}, A={row['odds_AvgA']}  (sum={avg_sum:.4f})")
    print(f"  Max odds:  H={row['odds_MaxH']}, D={row['odds_MaxD']}, A={row['odds_MaxA']}  (sum={max_sum:.4f})")
    print()
    print(f"  DIAGNOSIS: Avg implied prob sum = {avg_sum:.4f} (slightly < 1.0)")
    print(f"  ROOT CAUSE: Max odds are corrupted in source CSV (I1_2526.csv).")
    print(f"    MaxD=6.8 is far too high (Avg draw = 4.2) — likely a data entry error.")
    print(f"    MaxA=4.0 is far too high (Avg away = 1.67) — likely swapped or corrupted.")
    print(f"    MaxH=6.75 looks plausible (AvgH=6.37).")
    print(f"  The Avg odds slight <1.0 sum ({avg_sum:.4f}) is marginal and within rounding.")
    print(f"  The Max odds are clearly wrong (sum={max_sum:.4f}), but cannot determine correct values.")
    print(f"  FLAG: odds_MaxD and odds_MaxA are corrupted for this match. Source: I1_2526.csv columns MaxD, MaxA.")
    print(f"  NO CHANGE MADE (correct values unknown).")
else:
    print(f"  Match not found or ambiguous ({mask3.sum()} matches)")

# --- Save fixed parquet ---
df.to_parquet(PARQUET_PATH, index=False)
print(f"\nSaved fixed data to {PARQUET_PATH}")

# --- Issue 4: Sanity checks ---
print(f"\n{'='*60}")
print(f"=== Sanity Checks ===")
print(f"{'='*60}")

# Check 1: shots_on_target <= shots_total
sot_cols = [
    ("home_shots_on_target_count", "home_shots_total"),
    ("away_shots_on_target_count", "away_shots_total"),
]
sot_violations = 0
for sot_col, total_col in sot_cols:
    valid = df[sot_col].notna() & df[total_col].notna()
    bad = df[valid & (df[sot_col] > df[total_col])]
    if len(bad) > 0:
        print(f"\n  FAIL: {len(bad)} matches with {sot_col} > {total_col}:")
        for _, r in bad.iterrows():
            print(f"    {r['match_date'].date()} {r['home_team']} vs {r['away_team']}: {sot_col}={r[sot_col]}, {total_col}={r[total_col]}")
        sot_violations += len(bad)
if sot_violations == 0:
    print("\n  PASS: No match has shots_on_target > shots_total")

# Check 2: ht_goals <= ft_goals
ht_violations = 0
for side in ["home", "away"]:
    ht_col = f"{side}_ht_goals"
    ft_col = f"{side}_score"
    valid = df[ht_col].notna() & df[ft_col].notna()
    bad = df[valid & (df[ht_col] > df[ft_col])]
    if len(bad) > 0:
        print(f"\n  FAIL: {len(bad)} matches with {ht_col} > {ft_col}:")
        for _, r in bad.head(5).iterrows():
            print(f"    {r['match_date'].date()} {r['home_team']} vs {r['away_team']}: ht={r[ht_col]}, ft={r[ft_col]}")
        ht_violations += len(bad)
if ht_violations == 0:
    print("  PASS: No match has ht_goals > ft_goals")

# Check 3: All odds > 1.0
odds_cols = [c for c in df.columns if c.startswith("odds_") and not c.endswith("_line")]
odds_violations = 0
for col in odds_cols:
    valid = df[col].notna()
    bad = df[valid & (df[col] <= 1.0)]
    if len(bad) > 0:
        print(f"\n  FAIL: {len(bad)} matches with {col} <= 1.0")
        for _, r in bad.head(3).iterrows():
            print(f"    {r['match_date'].date()} {r['home_team']} vs {r['away_team']}: {col}={r[col]}")
        odds_violations += len(bad)
if odds_violations == 0:
    print("  PASS: All odds > 1.0")

print(f"\n{'='*60}")
if sot_violations == 0 and ht_violations == 0 and odds_violations == 0:
    print("ALL SANITY CHECKS PASSED")
else:
    print(f"ISSUES FOUND: {sot_violations} SOT violations, {ht_violations} HT violations, {odds_violations} odds violations")
print(f"{'='*60}")
