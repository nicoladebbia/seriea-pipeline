#!/usr/bin/env python3
"""
Harmonize column names between Serie A and EPL in matches.parquet.

EPL data uses different column names for the same concepts as Serie A.
This script standardizes everything to the Serie A naming convention,
merges duplicate columns, computes missing derived columns, and drops
the EPL-only duplicates.

Column mapping (EPL -> Serie A):
  home_yellows / away_yellows       -> home_yellow_cards / away_yellow_cards
  home_reds / away_reds             -> home_red_cards / away_red_cards
  home_shots / away_shots           -> home_shots_on_target_total / away_shots_on_target_total  (total shots)
  home_shots_on_target (EPL)        -> home_shots_on_target_count / away_shots_on_target_count  (actual count)

Note on shots: Serie A has 3 shot columns:
  - home_shots_on_target: percentage (e.g. 33 = 33%)
  - home_shots_on_target_count: actual shots on target count
  - home_shots_on_target_total: total shots
  EPL's home_shots_on_target is a count (not percentage), mapping to _count.
  EPL's home_shots is total shots, mapping to _total.

Additional:
  - Compute 'result' for Serie A rows (H/D/A) from scores
  - Compute shots_on_target percentage for EPL rows where we have count+total
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "parsed" / "matches.parquet"


def harmonize():
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    sa_mask = df["league"] == "serie_a"
    epl_mask = df["league"] == "premier_league"

    print(f"  Serie A: {sa_mask.sum()} rows")
    print(f"  EPL:     {epl_mask.sum()} rows")

    # --- 1. Yellow cards: home_yellows -> home_yellow_cards ---
    for side in ("home", "away"):
        canon = f"{side}_yellow_cards"
        epl_col = f"{side}_yellows"
        # Fill canonical column where it's null but EPL column has data
        mask = df[canon].isna() & df[epl_col].notna()
        df.loc[mask, canon] = df.loc[mask, epl_col]
        filled = mask.sum()
        print(f"  {epl_col} -> {canon}: filled {filled} rows")

    # --- 2. Red cards: home_reds -> home_red_cards ---
    for side in ("home", "away"):
        canon = f"{side}_red_cards"
        epl_col = f"{side}_reds"
        mask = df[canon].isna() & df[epl_col].notna()
        df.loc[mask, canon] = df.loc[mask, epl_col]
        filled = mask.sum()
        print(f"  {epl_col} -> {canon}: filled {filled} rows")

    # --- 3. Shots: EPL home_shots -> home_shots_on_target_total (total shots) ---
    for side in ("home", "away"):
        canon = f"{side}_shots_on_target_total"
        epl_col = f"{side}_shots"
        mask = df[canon].isna() & df[epl_col].notna()
        df.loc[mask, canon] = df.loc[mask, epl_col]
        filled = mask.sum()
        print(f"  {epl_col} -> {canon}: filled {filled} rows")

    # --- 4. Shots on target: EPL home_shots_on_target is a COUNT, not percentage ---
    # For EPL rows, move the count value into _count, then compute percentage for _on_target
    for side in ("home", "away"):
        count_col = f"{side}_shots_on_target_count"
        pct_col = f"{side}_shots_on_target"
        total_col = f"{side}_shots_on_target_total"

        # EPL's shots_on_target is actually a count -> copy to _count
        mask = epl_mask & df[count_col].isna() & df[pct_col].notna()
        df.loc[mask, count_col] = df.loc[mask, pct_col]
        filled = mask.sum()
        print(f"  {pct_col} (EPL count) -> {count_col}: filled {filled} rows")

        # Now compute the percentage for EPL rows (to match Serie A's _on_target meaning)
        can_compute = epl_mask & df[count_col].notna() & df[total_col].notna() & (df[total_col] > 0)
        df.loc[can_compute, pct_col] = (
            df.loc[can_compute, count_col] / df.loc[can_compute, total_col] * 100
        ).round(0)
        computed = can_compute.sum()
        print(f"  Computed {pct_col} percentage for {computed} EPL rows")

    # --- 5. Result column: derive for Serie A from scores ---
    sa_scores = sa_mask & df["home_score"].notna() & df["away_score"].notna()
    conditions = [
        sa_scores & (df["home_score"] > df["away_score"]),
        sa_scores & (df["home_score"] == df["away_score"]),
        sa_scores & (df["home_score"] < df["away_score"]),
    ]
    choices = ["H", "D", "A"]
    df.loc[sa_scores, "result"] = np.select(
        [c[sa_scores] for c in conditions], choices, default=None
    )
    result_filled = df.loc[sa_mask, "result"].notna().sum()
    print(f"  Computed 'result' for {result_filled} Serie A rows")

    # --- 6. Drop the now-redundant EPL-only columns ---
    drop_cols = [
        "home_yellows", "away_yellows",
        "home_reds", "away_reds",
        "home_shots", "away_shots",
    ]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing_drops, inplace=True)
    print(f"  Dropped redundant columns: {existing_drops}")

    # --- 7. Save ---
    print(f"\nFinal shape: {len(df)} rows, {len(df.columns)} columns")
    df.to_parquet(DATA_PATH, index=False)
    print(f"Saved to {DATA_PATH}")

    # --- 8. Verification ---
    print("\n=== VERIFICATION ===")
    df2 = pd.read_parquet(DATA_PATH)
    sa2 = df2[df2["league"] == "serie_a"]
    epl2 = df2[df2["league"] == "premier_league"]

    key_cols = [
        "home_yellow_cards", "away_yellow_cards",
        "home_red_cards", "away_red_cards",
        "home_shots_on_target", "away_shots_on_target",
        "home_shots_on_target_count", "away_shots_on_target_count",
        "home_shots_on_target_total", "away_shots_on_target_total",
        "result",
    ]

    print(f"\n{'Column':<35} {'SA non-null':<15} {'EPL non-null':<15}")
    print("-" * 65)
    for c in key_cols:
        sa_nn = sa2[c].notna().sum()
        epl_nn = epl2[c].notna().sum()
        print(f"{c:<35} {sa_nn:<15} {epl_nn:<15}")

    # Confirm no duplicate columns remain
    for old in ["home_yellows", "away_yellows", "home_reds", "away_reds", "home_shots", "away_shots"]:
        assert old not in df2.columns, f"Column {old} should have been dropped!"
    print("\nAll duplicate columns successfully removed.")
    print("Harmonization complete.")


if __name__ == "__main__":
    harmonize()
