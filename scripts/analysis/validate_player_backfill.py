"""Validate the backfilled player_stats.parquet data quality.

Checks:
  1. Record counts — ~12,000+ per season
  2. Season coverage — 8 seasons (2017-2025)
  3. Key players — known stars per era
  4. No duplicates — unique (match_id, team, player)
  5. Stat sanity — minutes ≤ 95, xG ≥ 0
  6. Team coverage — 20 teams per season
  7. Cross-ref — player goals vs matches.parquet totals

Usage:
    python3 -m scripts.validate_player_backfill
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from storage.paths import parsed_path

log = logging.getLogger(__name__)

# Known star players per era (for spot checks)
STAR_PLAYERS = {
    "2017-2018": ["Mauro Icardi", "Paulo Dybala", "Ciro Immobile", "Lorenzo Insigne"],
    "2018-2019": ["Cristiano Ronaldo", "Fabio Quagliarella", "Krzysztof Piątek"],
    "2019-2020": ["Cristiano Ronaldo", "Ciro Immobile", "Romelu Lukaku"],
    "2020-2021": ["Cristiano Ronaldo", "Romelu Lukaku", "Luis Muriel"],
    "2021-2022": ["Ciro Immobile", "Lautaro Martínez", "Dušan Vlahović"],
    "2022-2023": ["Victor Osimhen", "Lautaro Martínez", "Rafael Leão"],
    "2023-2024": ["Lautaro Martínez", "Marcus Thuram", "Dušan Vlahović"],
    "2024-2025": ["Lautaro Martínez", "Marcus Thuram", "Mateo Retegui"],
    "2025-2026": ["Lautaro Martínez", "Marcus Thuram", "Mateo Retegui"],
}


def check_record_counts(df: pd.DataFrame) -> list[str]:
    """Check per-season row counts."""
    issues = []
    for season, grp in df.groupby("season"):
        count = len(grp)
        if count < 8000:
            issues.append(f"LOW COUNT: {season} has only {count:,} rows (expected ~12,000+)")
        elif count < 10000:
            issues.append(f"WARNING: {season} has {count:,} rows (slightly below expected ~12,000)")
    return issues


def check_season_coverage(df: pd.DataFrame) -> list[str]:
    """Check that expected seasons are present."""
    issues = []
    seasons = set(df["season"].unique())
    expected = {
        "2017-2018", "2018-2019", "2019-2020", "2020-2021",
        "2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026",
    }
    missing = expected - seasons
    if missing:
        issues.append(f"MISSING SEASONS: {sorted(missing)}")
    extra = seasons - expected
    if extra:
        issues.append(f"EXTRA SEASONS (ok): {sorted(extra)}")
    return issues


def check_key_players(df: pd.DataFrame) -> list[str]:
    """Spot-check that known star players appear in the right seasons."""
    issues = []
    if "player" not in df.columns:
        return ["MISSING: 'player' column not found"]

    for season, stars in STAR_PLAYERS.items():
        season_df = df[df["season"] == season]
        if season_df.empty:
            continue
        players = set(season_df["player"].unique())
        for star in stars:
            # Fuzzy match: check if any player name contains the star's surname
            surname = star.split()[-1]
            found = any(surname.lower() in p.lower() for p in players)
            if not found:
                issues.append(f"MISSING PLAYER: {star} not found in {season} (surname: {surname})")
    return issues


def check_duplicates(df: pd.DataFrame) -> list[str]:
    """Check for duplicate (match_id, team, player) entries."""
    issues = []
    key_cols = [c for c in ["match_id", "team", "player"] if c in df.columns]
    if len(key_cols) < 3:
        return [f"MISSING COLUMNS for dedup check: have {key_cols}"]

    dupes = df.duplicated(subset=key_cols, keep=False)
    n_dupes = dupes.sum()
    if n_dupes > 0:
        n_groups = df[dupes].groupby(key_cols).ngroups
        issues.append(f"DUPLICATES: {n_dupes} duplicate rows in {n_groups} groups on ({', '.join(key_cols)})")
        # Show unique duplicated groups
        sample = df[dupes].drop_duplicates(subset=key_cols).head(5)[key_cols]
        for _, row in sample.iterrows():
            count = len(df[(df[key_cols] == row).all(axis=1)])
            issues.append(f"  e.g. {row.to_dict()} ({count}x)")
    return issues


def check_stat_sanity(df: pd.DataFrame) -> list[str]:
    """Check numeric stat ranges for sanity."""
    issues = []

    # Minutes played
    if "minutes" in df.columns:
        minutes = pd.to_numeric(df["minutes"], errors="coerce")
        if minutes.max() > 130:
            issues.append(f"MINUTES: max={minutes.max()} (expected ≤ ~120 for extra time)")
        if minutes.min() < 0:
            issues.append(f"MINUTES: min={minutes.min()} (negative!)")

    # xG
    for col in ["xg", "npxg", "xg_assist"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.min() < 0:
                issues.append(f"{col}: min={vals.min():.3f} (negative!)")
            if vals.max() > 5:
                issues.append(f"WARNING: {col}: max={vals.max():.3f} (unusually high for single match)")

    # Goals
    if "goals" in df.columns:
        goals = pd.to_numeric(df["goals"], errors="coerce")
        if goals.max() > 7:
            issues.append(f"WARNING: goals max={goals.max()} (unusually high)")
        if goals.min() < 0:
            issues.append(f"GOALS: min={goals.min()} (negative!)")

    return issues


def check_team_coverage(df: pd.DataFrame) -> list[str]:
    """Check that ~20 teams appear per season."""
    issues = []
    if "team" not in df.columns:
        return ["MISSING: 'team' column"]

    for season, grp in df.groupby("season"):
        n_teams = grp["team"].nunique()
        if n_teams < 18:
            issues.append(f"LOW TEAMS: {season} has {n_teams} teams (expected ~20)")
        elif n_teams > 22:
            issues.append(f"HIGH TEAMS: {season} has {n_teams} teams (expected ~20)")
    return issues


def check_cross_ref_goals(df: pd.DataFrame) -> list[str]:
    """Cross-reference player goals totals with matches.parquet."""
    issues = []
    matches_path = parsed_path("matches")
    if not matches_path.exists():
        return ["SKIP: matches.parquet not found for cross-reference"]

    if "goals" not in df.columns or "match_id" not in df.columns:
        return ["SKIP: missing 'goals' or 'match_id' column for cross-reference"]

    matches = pd.read_parquet(matches_path)
    df_goals = pd.to_numeric(df["goals"], errors="coerce").fillna(0)

    # Check a sample of seasons
    for season in df["season"].unique():
        season_ps = df[df["season"] == season].copy()
        season_ps["_goals"] = pd.to_numeric(season_ps["goals"], errors="coerce").fillna(0)

        season_m = matches[matches["season"] == season] if "season" in matches.columns else pd.DataFrame()
        if season_m.empty:
            continue

        # Total goals from player stats (home players only to avoid double counting)
        ps_home_goals = season_ps[season_ps["is_home"] == True]["_goals"].sum()
        match_home_goals = pd.to_numeric(season_m.get("home_score", pd.Series(dtype=float)), errors="coerce").sum()

        if match_home_goals > 0:
            ratio = ps_home_goals / match_home_goals
            if abs(ratio - 1.0) > 0.15:
                issues.append(
                    f"GOAL MISMATCH: {season} — player_stats home goals "
                    f"({ps_home_goals:.0f}) vs matches home_score "
                    f"({match_home_goals:.0f}), ratio={ratio:.2f}"
                )

    return issues


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ps_path = parsed_path("player_stats")
    if not ps_path.exists():
        log.error("player_stats.parquet not found at %s", ps_path)
        sys.exit(1)

    df = pd.read_parquet(ps_path)
    log.info("Loaded player_stats.parquet: %d rows × %d columns", len(df), len(df.columns))

    print("\n" + "=" * 70)
    print("  PLAYER STATS VALIDATION")
    print("=" * 70)

    all_issues = []
    checks = [
        ("Record counts", check_record_counts),
        ("Season coverage", check_season_coverage),
        ("Key players", check_key_players),
        ("Duplicates", check_duplicates),
        ("Stat sanity", check_stat_sanity),
        ("Team coverage", check_team_coverage),
        ("Cross-ref goals", check_cross_ref_goals),
    ]

    for name, check_fn in checks:
        print(f"\n{'─' * 50}")
        print(f"  {name}")
        issues = check_fn(df)
        if issues:
            for issue in issues:
                print(f"    {issue}")
            all_issues.extend(issues)
        else:
            print("    PASS")

    # Summary
    print(f"\n{'=' * 70}")
    errors = [i for i in all_issues if not i.startswith("WARNING") and not i.startswith("SKIP") and not i.startswith("EXTRA")]
    warnings = [i for i in all_issues if i.startswith("WARNING")]
    print(f"  RESULT: {len(errors)} errors, {len(warnings)} warnings")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    - {e}")

    # Per-season summary table
    print(f"\n{'─' * 50}")
    print(f"  Per-season summary:")
    print(f"  {'Season':<12} {'Rows':>8} {'Players':>8} {'Teams':>6} {'Matches':>8}")
    for season in sorted(df["season"].unique()):
        grp = df[df["season"] == season]
        print(
            f"  {season:<12} {len(grp):>8,} "
            f"{grp['player'].nunique() if 'player' in grp.columns else 0:>8} "
            f"{grp['team'].nunique() if 'team' in grp.columns else 0:>6} "
            f"{grp['match_id'].nunique() if 'match_id' in grp.columns else 0:>8}"
        )

    print("=" * 70 + "\n")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
