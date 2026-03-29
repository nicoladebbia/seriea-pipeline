"""Import Serie A odds from football-data.co.uk CSVs into matches.parquet.

Reads each I1_*.csv file, extracts odds columns, normalizes team names,
and merges into existing matches.parquet rows by (home_team, away_team, match_date).
"""

from __future__ import annotations

import glob
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.team_names import normalize_team

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PARQUET_PATH = ROOT / "data" / "parsed" / "matches.parquet"
ODDS_GLOB = str(ROOT / "data" / "external" / "odds" / "I1_*.csv")

# Mapping from CSV column names to parquet odds column names.
# The parquet uses odds_ prefix with cleaned names (no > < characters).
CSV_TO_PARQUET = {
    "B365H": "odds_B365H",
    "B365D": "odds_B365D",
    "B365A": "odds_B365A",
    "PSH": "odds_PSH",
    "PSD": "odds_PSD",
    "PSA": "odds_PSA",
    "AvgH": "odds_AvgH",
    "AvgD": "odds_AvgD",
    "AvgA": "odds_AvgA",
    "MaxH": "odds_MaxH",
    "MaxD": "odds_MaxD",
    "MaxA": "odds_MaxA",
    "B365>2.5": "odds_B365_over25",
    "B365<2.5": "odds_B365_under25",
    "Avg>2.5": "odds_Avg_over25",
    "Avg<2.5": "odds_Avg_under25",
}

# Old-format column names used in pre-2019 CSVs
OLD_TO_NEW = {
    "BbMxH": "MaxH",
    "BbAvH": "AvgH",
    "BbMxD": "MaxD",
    "BbAvD": "AvgD",
    "BbMxA": "MaxA",
    "BbAvA": "AvgA",
    "BbMx>2.5": "Max>2.5",
    "BbAv>2.5": "Avg>2.5",
    "BbMx<2.5": "Max<2.5",
    "BbAv<2.5": "Avg<2.5",
}


def load_all_odds_csvs() -> pd.DataFrame:
    """Load all Serie A odds CSVs into a single DataFrame."""
    csv_files = sorted(glob.glob(ODDS_GLOB))
    if not csv_files:
        log.error("No CSV files found matching %s", ODDS_GLOB)
        return pd.DataFrame()

    all_dfs = []
    for csv_path in csv_files:
        fname = Path(csv_path).name
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
        except Exception as e:
            log.warning("Failed to read %s: %s", fname, e)
            continue

        # Drop fully-empty rows
        df = df.dropna(how="all")

        if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
            log.warning("Skipping %s: no HomeTeam/AwayTeam columns", fname)
            continue

        # Drop rows with missing team names
        df = df.dropna(subset=["HomeTeam", "AwayTeam"])

        # Rename old-format columns to new-format
        renames = {old: new for old, new in OLD_TO_NEW.items()
                   if old in df.columns and new not in df.columns}
        if renames:
            df = df.rename(columns=renames)
            log.info("  %s: renamed %d old-format columns", fname, len(renames))

        # Normalize team names
        df["home_team_norm"] = df["HomeTeam"].apply(normalize_team)
        df["away_team_norm"] = df["AwayTeam"].apply(normalize_team)

        # Parse date
        if "Date" in df.columns:
            df["match_date_norm"] = pd.to_datetime(
                df["Date"], format="mixed", dayfirst=True, errors="coerce"
            ).dt.normalize()

        # Extract only the odds columns that exist in this CSV
        keep_cols = ["home_team_norm", "away_team_norm", "match_date_norm"]
        for csv_col, parquet_col in CSV_TO_PARQUET.items():
            if csv_col in df.columns:
                df[parquet_col] = pd.to_numeric(df[csv_col], errors="coerce")
                keep_cols.append(parquet_col)

        df = df[keep_cols].copy()
        log.info("  %s: %d matches loaded", fname, len(df))
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Total odds rows loaded: %d", len(combined))
    return combined


def main():
    log.info("Loading matches.parquet from %s", PARQUET_PATH)
    matches = pd.read_parquet(PARQUET_PATH)
    log.info("Loaded %d matches total", len(matches))

    # Count Serie A rows
    serie_a = matches[matches["league"] == "serie_a"]
    log.info("Serie A rows: %d", len(serie_a))
    log.info("Serie A rows with odds_B365H before import: %d",
             serie_a["odds_B365H"].notna().sum())

    # Load all odds CSVs
    odds = load_all_odds_csvs()
    if odds.empty:
        log.error("No odds data loaded, aborting")
        return

    # Prepare matches join key
    matches["_match_date_norm"] = pd.to_datetime(
        matches["match_date"], errors="coerce"
    ).dt.normalize()

    # Build a merge key for odds
    odds["_merge_key"] = (
        odds["home_team_norm"] + "|" +
        odds["away_team_norm"] + "|" +
        odds["match_date_norm"].astype(str)
    )

    matches["_merge_key"] = (
        matches["home_team"] + "|" +
        matches["away_team"] + "|" +
        matches["_match_date_norm"].astype(str)
    )

    # Only update Serie A rows
    serie_a_mask = matches["league"] == "serie_a"

    # Get the odds columns that exist in our data
    odds_cols = [c for c in odds.columns if c.startswith("odds_")]

    # Build a lookup dict from merge_key -> odds values
    # Drop duplicates keeping first (some CSVs may overlap)
    odds_deduped = odds.drop_duplicates(subset=["_merge_key"], keep="first")
    odds_lookup = odds_deduped.set_index("_merge_key")[odds_cols]

    # For each Serie A match, look up odds
    matched = 0
    for idx in matches.index[serie_a_mask]:
        key = matches.at[idx, "_merge_key"]
        if key in odds_lookup.index:
            for col in odds_cols:
                val = odds_lookup.at[key, col]
                if pd.notna(val):
                    matches.at[idx, col] = val
            matched += 1

    log.info("Matched %d/%d Serie A rows with odds data", matched, serie_a_mask.sum())

    # Cleanup temp columns
    matches.drop(columns=["_match_date_norm", "_merge_key"], errors="ignore", inplace=True)

    # Report coverage per season
    serie_a_updated = matches[matches["league"] == "serie_a"]
    log.info("\n--- Coverage per season ---")
    for season in sorted(serie_a_updated["season"].unique()):
        season_df = serie_a_updated[serie_a_updated["season"] == season]
        b365_count = season_df["odds_B365H"].notna().sum()
        ps_count = season_df["odds_PSH"].notna().sum()
        avg_count = season_df["odds_AvgH"].notna().sum()
        log.info("  %s: %d/%d B365, %d/%d PS, %d/%d Avg",
                 season, b365_count, len(season_df),
                 ps_count, len(season_df),
                 avg_count, len(season_df))

    # Save
    matches.to_parquet(PARQUET_PATH, index=False)
    log.info("Saved updated matches.parquet to %s", PARQUET_PATH)

    # Final verification
    total_sa = len(matches[matches["league"] == "serie_a"])
    total_with_odds = matches[(matches["league"] == "serie_a") & (matches["odds_B365H"].notna())].shape[0]
    log.info("\nFinal: Serie A has odds for %d/%d matches (%.1f%%)",
             total_with_odds, total_sa, 100 * total_with_odds / total_sa if total_sa else 0)


if __name__ == "__main__":
    main()
