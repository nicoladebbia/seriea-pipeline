"""Import match stats and closing odds from football-data.co.uk CSVs into matches.parquet.

Extracts from BOTH Serie A (I1_*.csv) and EPL (E0_*.csv):
  - Match stats: shots, shots on target, fouls, corners, half-time scores
  - Closing odds: B365 closing 1X2, B365 closing O/U 2.5, Pinnacle closing 1X2
  - Asian handicap: B365 AH prices, AH line (opening + closing)

Only fills NaN/null cells — never overwrites existing data.
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
ODDS_DIR = ROOT / "data" / "external" / "odds"

# ---------------------------------------------------------------------------
# Column mapping: CSV name -> parquet name
# ---------------------------------------------------------------------------
# Stats columns
STATS_MAP = {
    "HS": "home_shots_total",
    "AS": "away_shots_total",
    "HST": "home_shots_on_target_count",
    "AST": "away_shots_on_target_count",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HTHG": "home_ht_goals",
    "HTAG": "away_ht_goals",
    "HTR": "ht_result",
}

# Closing 1X2 odds
CLOSING_1X2_MAP = {
    "B365CH": "odds_B365_close_H",
    "B365CD": "odds_B365_close_D",
    "B365CA": "odds_B365_close_A",
    "PSCH": "odds_PS_close_H",
    "PSCD": "odds_PS_close_D",
    "PSCA": "odds_PS_close_A",
}

# Closing O/U 2.5 odds
CLOSING_OU_MAP = {
    "B365C>2.5": "odds_B365_close_over25",
    "B365C<2.5": "odds_B365_close_under25",
}

# Asian handicap
AH_MAP = {
    "B365AHH": "odds_B365_AH_H",
    "B365AHA": "odds_B365_AH_A",
    "AHh": "odds_AH_line",
    "AHCh": "odds_AH_close_line",
}

# Old-format column renames (pre-2019 CSVs)
OLD_TO_NEW = {
    "BbAHh": "AHh",  # Asian handicap line had different name
}

# Combine all mappings
ALL_CSV_TO_PARQUET = {}
ALL_CSV_TO_PARQUET.update(STATS_MAP)
ALL_CSV_TO_PARQUET.update(CLOSING_1X2_MAP)
ALL_CSV_TO_PARQUET.update(CLOSING_OU_MAP)
ALL_CSV_TO_PARQUET.update(AH_MAP)

# Columns that are numeric (everything except ht_result)
STRING_COLS = {"ht_result"}

# League prefix -> league identifier in parquet
LEAGUE_MAP = {
    "I1": "serie_a",
    "E0": "epl",
}


def load_csvs(league_prefix: str) -> pd.DataFrame:
    """Load all CSVs for a given league prefix (I1 or E0)."""
    pattern = str(ODDS_DIR / f"{league_prefix}_*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        log.warning("No CSV files found for %s", pattern)
        return pd.DataFrame()

    all_dfs = []
    for csv_path in csv_files:
        fname = Path(csv_path).name
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
        except Exception as e:
            log.warning("Failed to read %s: %s", fname, e)
            continue

        # Clean up BOM in column names
        df.columns = [c.replace("\ufeff", "") for c in df.columns]

        # Drop fully-empty rows
        df = df.dropna(how="all")

        if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
            log.warning("Skipping %s: no HomeTeam/AwayTeam columns", fname)
            continue

        df = df.dropna(subset=["HomeTeam", "AwayTeam"])

        # Rename old-format columns
        renames = {old: new for old, new in OLD_TO_NEW.items()
                   if old in df.columns and new not in df.columns}
        if renames:
            df = df.rename(columns=renames)

        # Normalize team names
        df["home_team_norm"] = df["HomeTeam"].apply(normalize_team)
        df["away_team_norm"] = df["AwayTeam"].apply(normalize_team)

        # Parse date (handles both dd/mm/yy and dd/mm/yyyy)
        if "Date" in df.columns:
            df["match_date_norm"] = pd.to_datetime(
                df["Date"], format="mixed", dayfirst=True, errors="coerce"
            ).dt.normalize()

        # Extract columns
        keep_cols = ["home_team_norm", "away_team_norm", "match_date_norm"]
        for csv_col, parquet_col in ALL_CSV_TO_PARQUET.items():
            if csv_col in df.columns:
                if parquet_col in STRING_COLS:
                    df[parquet_col] = df[csv_col].astype(str).where(df[csv_col].notna())
                else:
                    df[parquet_col] = pd.to_numeric(df[csv_col], errors="coerce")
                keep_cols.append(parquet_col)

        # Only keep columns that exist
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].copy()
        log.info("  %s: %d matches", fname, len(df))
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Total %s rows loaded: %d", league_prefix, len(combined))
    return combined


def main():
    log.info("Loading matches.parquet from %s", PARQUET_PATH)
    matches = pd.read_parquet(PARQUET_PATH)
    log.info("Loaded %d matches (%d columns)", len(matches), len(matches.columns))

    # Ensure all target columns exist in the DataFrame
    for parquet_col in ALL_CSV_TO_PARQUET.values():
        if parquet_col not in matches.columns:
            if parquet_col in STRING_COLS:
                matches[parquet_col] = pd.Series([None] * len(matches), dtype="object")
            else:
                matches[parquet_col] = pd.Series([float("nan")] * len(matches), dtype="float64")

    # Normalize match_date for joining
    matches["_match_date_norm"] = pd.to_datetime(
        matches["match_date"], errors="coerce"
    ).dt.normalize()

    matches["_merge_key"] = (
        matches["home_team"] + "|" +
        matches["away_team"] + "|" +
        matches["_match_date_norm"].astype(str)
    )

    # Track stats per league
    coverage: dict[str, dict[str, dict[str, int]]] = {}

    for prefix, league_id in LEAGUE_MAP.items():
        log.info("\n=== Processing %s (%s) ===", prefix, league_id)

        csv_data = load_csvs(prefix)
        if csv_data.empty:
            log.warning("No data for %s, skipping", prefix)
            continue

        # Build merge key for CSV data
        csv_data["_merge_key"] = (
            csv_data["home_team_norm"] + "|" +
            csv_data["away_team_norm"] + "|" +
            csv_data["match_date_norm"].astype(str)
        )

        # Deduplicate (keep first)
        csv_data = csv_data.drop_duplicates(subset=["_merge_key"], keep="first")

        # Get the new data columns present
        data_cols = [c for c in csv_data.columns
                     if c in ALL_CSV_TO_PARQUET.values()]

        # Build lookup
        csv_lookup = csv_data.set_index("_merge_key")

        # Filter to the league's matches
        league_mask = matches["league"] == league_id
        league_count = league_mask.sum()
        log.info("Parquet has %d %s matches", league_count, league_id)

        if league_count == 0:
            log.info("No %s matches in parquet — attempting to match ALL rows by key", league_id)
            # EPL rows don't exist yet — we can only fill existing rows
            # Skip if there are no matching rows
            target_mask = pd.Series(True, index=matches.index)
        else:
            target_mask = league_mask

        # Fill values
        filled_counts: dict[str, int] = {}
        matched = 0
        for idx in matches.index[target_mask]:
            key = matches.at[idx, "_merge_key"]
            if key not in csv_lookup.index:
                continue
            matched += 1
            for col in data_cols:
                if col not in csv_lookup.columns:
                    continue
                current_val = matches.at[idx, col]
                # Only fill if current value is NaN/null
                if col in STRING_COLS:
                    is_empty = pd.isna(current_val) or current_val is None
                else:
                    is_empty = pd.isna(current_val)
                if is_empty:
                    new_val = csv_lookup.at[key, col]
                    if col in STRING_COLS:
                        if pd.notna(new_val) and new_val != "nan":
                            matches.at[idx, col] = new_val
                            filled_counts[col] = filled_counts.get(col, 0) + 1
                    else:
                        if pd.notna(new_val):
                            matches.at[idx, col] = new_val
                            filled_counts[col] = filled_counts.get(col, 0) + 1

        log.info("Matched %d rows for %s", matched, league_id)
        coverage[league_id] = {"filled": filled_counts, "matched": matched,
                               "total": league_count}

    # Cleanup temp columns
    matches.drop(columns=["_match_date_norm", "_merge_key"], errors="ignore",
                 inplace=True)

    # Save
    matches.to_parquet(PARQUET_PATH, index=False)
    log.info("\nSaved updated matches.parquet (%d rows, %d columns)",
             len(matches), len(matches.columns))

    # ---------------------------------------------------------------------------
    # Coverage report
    # ---------------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("COVERAGE REPORT")
    log.info("=" * 70)

    for league_id, info in coverage.items():
        total = info["total"]
        matched_count = info["matched"]
        filled = info["filled"]
        log.info("\n%s — %d/%d rows matched from CSV", league_id.upper(),
                 matched_count, total if total > 0 else matched_count)

        if not filled:
            log.info("  (no new columns filled)")
            continue

        # Group by category
        categories = {
            "Match Stats": [v for v in STATS_MAP.values()],
            "Closing 1X2": [v for v in CLOSING_1X2_MAP.values()],
            "Closing O/U": [v for v in CLOSING_OU_MAP.values()],
            "Asian Handicap": [v for v in AH_MAP.values()],
        }

        for cat_name, cat_cols in categories.items():
            cat_filled = {c: filled.get(c, 0) for c in cat_cols if c in filled}
            if cat_filled:
                log.info("  %s:", cat_name)
                for col, count in cat_filled.items():
                    log.info("    %-35s %d rows filled", col, count)

    # Also report total non-null counts per new column
    log.info("\n" + "-" * 70)
    log.info("TOTAL NON-NULL COUNTS (all leagues combined)")
    log.info("-" * 70)
    for parquet_col in sorted(ALL_CSV_TO_PARQUET.values()):
        if parquet_col in matches.columns:
            total_nonnull = matches[parquet_col].notna().sum()
            # Break down by league
            parts = []
            for league_id in LEAGUE_MAP.values():
                league_df = matches[matches["league"] == league_id]
                if len(league_df) > 0:
                    cnt = league_df[parquet_col].notna().sum()
                    parts.append(f"{league_id}={cnt}/{len(league_df)}")
            # Also check rows with no league match (shouldn't happen)
            breakdown = ", ".join(parts) if parts else ""
            log.info("  %-35s %5d total   [%s]", parquet_col, total_nonnull, breakdown)


if __name__ == "__main__":
    main()
