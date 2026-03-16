"""Download historical betting odds from football-data.co.uk.

football-data.co.uk provides FREE CSV files with historical match results
and betting odds from 15+ bookmakers for top European leagues from 1993/94 onwards.

URL pattern: https://www.football-data.co.uk/mmz4281/{YYMM}/{LEAGUE_CODE}.csv
  - I1 = Serie A, E0 = Premier League, SP1 = La Liga,
    D1 = Bundesliga, F1 = Ligue 1
  - YYMM = e.g. "2425" for 2024-2025

CSV columns include:
  - Match: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
  - Odds: B365H/D/A, BWH/D/A, IWH/D/A, PSH/D/A, WHH/D/A, VCH/D/A
  - Totals: B365>2.5, B365<2.5, P>2.5, P<2.5
  - Asian Handicap: B365AHH, B365AHA, PAHH, PAHA
  - Market: MaxH/D/A, AvgH/D/A, Max>2.5, Avg>2.5
"""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import requests

from config.settings import DATA_DIR, DEFAULT_LEAGUE, LEAGUES

log = logging.getLogger(__name__)

ODDS_DIR = DATA_DIR / "external" / "odds"
BASE_URL = "https://www.football-data.co.uk/mmz4281"


# Season code mapping: "2024-2025" -> "2425"
def _season_code(season: str) -> str:
    """Convert '2024-2025' to '2425'."""
    parts = season.split("-")
    return parts[0][-2:] + parts[1][-2:]


# Key odds columns to extract (subset of all available)
ODDS_COLUMNS = {
    # Match identification
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    # Bet365 1X2
    "B365H", "B365D", "B365A",
    # Pinnacle 1X2 (sharpest bookmaker)
    "PSH", "PSD", "PSA",
    # Market consensus
    "MaxH", "MaxD", "MaxA",
    "AvgH", "AvgD", "AvgA",
    # Over/Under 2.5 goals
    "B365>2.5", "B365<2.5",
    "P>2.5", "P<2.5",
    "Max>2.5", "Max<2.5",
    "Avg>2.5", "Avg<2.5",
    # Asian Handicap
    "AHh",  # handicap size
    "B365AHH", "B365AHA",
    "PAHH", "PAHA",
    "MaxAHH", "MaxAHA",
    "AvgAHH", "AvgAHA",
    # Betfair Exchange
    "BFH", "BFD", "BFA",
    # Closing odds (for line velocity: opening → closing movement)
    "B365CH", "B365CD", "B365CA",
    "PSCH", "PSCD", "PSCA",
    "MaxCH", "MaxCD", "MaxCA",
    "AvgCH", "AvgCD", "AvgCA",
    "B365C>2.5", "B365C<2.5",
    "PC>2.5", "PC<2.5",
    "MaxC>2.5", "MaxC<2.5",
    "AvgC>2.5", "AvgC<2.5",
    "AHCh",  # closing handicap size
    "B365CAHH", "B365CAHA",
    "PCAHH", "PCAHA",
}


def download_odds(season: str, league: str | None = None) -> pd.DataFrame:
    """Download odds CSV for a given season and league.

    Args:
        season: e.g. "2024-2025"
        league: league key from LEAGUES dict (e.g. "serie_a", "premier_league").
                Defaults to DEFAULT_LEAGUE.

    Returns:
        DataFrame with odds data for all matches in the season.
    """
    if league is None:
        league = DEFAULT_LEAGUE

    league_code, league_name = LEAGUES[league]
    code = _season_code(season)
    url = f"{BASE_URL}/{code}/{league_code}.csv"

    cache_path = ODDS_DIR / f"{league_code}_{code}.csv"
    if cache_path.exists():
        log.info("Loading cached odds from %s", cache_path)
        try:
            return pd.read_csv(cache_path)
        except Exception:
            # Corrupted cache, re-download
            cache_path.unlink()

    log.info("Downloading %s odds from %s", league_name, url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error("Failed to download odds for %s %s: %s", league_name, season, e)
        return pd.DataFrame()

    # Save raw CSV
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(resp.text)

    try:
        df = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        log.error("Failed to parse CSV for %s %s: %s", league_name, season, e)
        return pd.DataFrame()

    # Drop fully-empty rows (football-data sometimes has trailing empty rows)
    df = df.dropna(how="all")

    log.info("Downloaded %d matches with odds for %s %s", len(df), league_name, season)
    return df


def download_all_odds(
    seasons: list[str] | None = None,
    league: str | None = None,
) -> pd.DataFrame:
    """Download odds for all configured seasons for a league."""
    from config.settings import SEASONS
    if seasons is None:
        seasons = SEASONS
    if league is None:
        league = DEFAULT_LEAGUE

    all_dfs = []
    for season in seasons:
        df = download_odds(season, league=league)
        if not df.empty:
            df["season"] = season
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def merge_odds_with_matches(matches: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Merge odds data into the matches DataFrame.

    Matches on (home_team, away_team, date) since match_ids differ between sources.
    """
    if odds.empty:
        log.warning("No odds data to merge")
        return matches

    # Normalize team names in odds to match our pipeline names
    from config.team_names import normalize_team

    odds = odds.copy()

    # Normalize old-format column names (pre-2019 CSVs use BbMx/BbAv prefix)
    OLD_TO_NEW = {
        "BbMxH": "MaxH", "BbAvH": "AvgH",
        "BbMxD": "MaxD", "BbAvD": "AvgD",
        "BbMxA": "MaxA", "BbAvA": "AvgA",
        "BbMx>2.5": "Max>2.5", "BbAv>2.5": "Avg>2.5",
        "BbMx<2.5": "Max<2.5", "BbAv<2.5": "Avg<2.5",
        "BbAHh": "AHh",
        "BbMxAHH": "MaxAHH", "BbAvAHH": "AvgAHH",
        "BbMxAHA": "MaxAHA", "BbAvAHA": "AvgAHA",
    }
    renames = {old: new for old, new in OLD_TO_NEW.items()
               if old in odds.columns and new not in odds.columns}
    if renames:
        odds = odds.rename(columns=renames)
        log.info("Normalized %d old-format odds columns: %s",
                 len(renames), list(renames.keys()))

    # Drop rows with missing team names (trailing empty rows in CSVs)
    if "HomeTeam" in odds.columns:
        odds = odds.dropna(subset=["HomeTeam", "AwayTeam"])
        odds["home_team_norm"] = odds["HomeTeam"].apply(normalize_team)
    if "AwayTeam" in odds.columns:
        odds["away_team_norm"] = odds["AwayTeam"].apply(normalize_team)

    # Parse date
    if "Date" in odds.columns:
        odds["match_date_norm"] = pd.to_datetime(
            odds["Date"], format="mixed", dayfirst=True, errors="coerce"
        ).dt.date.astype(str)

    matches = matches.copy()
    matches["_match_date_str"] = pd.to_datetime(
        matches["match_date"], errors="coerce"
    ).dt.date.astype(str)

    # Select odds columns that exist
    available_odds = [c for c in ODDS_COLUMNS if c in odds.columns
                      and c not in ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")]
    odds_subset = odds[["home_team_norm", "away_team_norm", "match_date_norm"] + available_odds].copy()
    odds_subset = odds_subset.rename(columns={c: f"odds_{c}" for c in available_odds})

    result = matches.merge(
        odds_subset,
        left_on=["home_team", "away_team", "_match_date_str"],
        right_on=["home_team_norm", "away_team_norm", "match_date_norm"],
        how="left",
    )

    # Cleanup merge keys
    result.drop(
        columns=["home_team_norm", "away_team_norm", "match_date_norm", "_match_date_str"],
        errors="ignore",
        inplace=True,
    )

    matched = result[[c for c in result.columns if c.startswith("odds_")]].notna().any(axis=1).sum()
    log.info("Merged odds: %d/%d matches matched", matched, len(matches))

    return result


def compute_implied_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal odds to implied probabilities with overround removal.

    Uses Pinnacle odds (PSH/PSD/PSA) as they are the sharpest,
    falling back to market average (AvgH/AvgD/AvgA).

    Columns added:
      implied_home_prob, implied_draw_prob, implied_away_prob
    """
    df = df.copy()

    for prefix, h_col, d_col, a_col in [
        ("pinnacle", "odds_PSH", "odds_PSD", "odds_PSA"),
        ("market", "odds_AvgH", "odds_AvgD", "odds_AvgA"),
    ]:
        if all(c in df.columns for c in (h_col, d_col, a_col)):
            h = pd.to_numeric(df[h_col], errors="coerce")
            d = pd.to_numeric(df[d_col], errors="coerce")
            a = pd.to_numeric(df[a_col], errors="coerce")

            # Raw implied probabilities
            raw_h = 1 / h
            raw_d = 1 / d
            raw_a = 1 / a

            # Overround (should sum to >1)
            overround = raw_h + raw_d + raw_a

            # Normalize to true probabilities
            df[f"{prefix}_home_prob"] = raw_h / overround
            df[f"{prefix}_draw_prob"] = raw_d / overround
            df[f"{prefix}_away_prob"] = raw_a / overround
            df[f"{prefix}_overround"] = overround

    # Over/Under 2.5 goals implied probabilities
    for prefix, over_col, under_col in [
        ("pinnacle_ou", "odds_P>2.5", "odds_P<2.5"),
        ("market_ou", "odds_Avg>2.5", "odds_Avg<2.5"),
    ]:
        if all(c in df.columns for c in (over_col, under_col)):
            over = pd.to_numeric(df[over_col], errors="coerce")
            under = pd.to_numeric(df[under_col], errors="coerce")

            raw_over = 1 / over
            raw_under = 1 / under
            overround = raw_over + raw_under

            df[f"{prefix}_over_prob"] = raw_over / overround
            df[f"{prefix}_under_prob"] = raw_under / overround
            df[f"{prefix}_overround"] = overround

    return df
