"""Import historical match data from football-data.co.uk CSVs.

football-data.co.uk provides match results, shots, fouls, cards, corners,
and betting odds for all top European leagues. This module converts that data
into our matches.parquet schema so it can be used by the feature pipeline.

Supports: Serie A, Premier League, La Liga, Bundesliga, Ligue 1.

Columns available from football-data.co.uk:
  Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
  HTHG, HTAG, HTR (half-time),
  HS, AS (total shots), HST, AST (shots on target),
  HF, AF (fouls), HC, AC (corners),
  HY, AY (yellow cards), HR, AR (red cards)

Columns NOT available (will be NaN):
  xG, possession, passing_accuracy, saves, crosses, touches, tackles,
  interceptions, aerials_won, clearances, offsides, goal_kicks,
  throw_ins, long_balls, venue, attendance, referee, manager, captain,
  formation, player stats, shots detail, lineups, events
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from config.settings import DATA_DIR, DEFAULT_LEAGUE, LEAGUES, SEASONS
from config.team_names import normalize_team
from scraper.odds import download_odds
from storage.paths import parsed_path

log = logging.getLogger(__name__)


# Map venue cities by team (home team -> city for weather lookups)
# Only for Serie A — other leagues don't need weather features
TEAM_CITIES: dict[str, str] = {
    "Atalanta": "Bergamo",
    "Benevento": "Benevento",
    "Bologna": "Bologna",
    "Cagliari": "Cagliari",
    "Como": "Como",
    "Cremonese": "Cremona",
    "Crotone": "Crotone",
    "Empoli": "Empoli",
    "Fiorentina": "Firenze",
    "Frosinone": "Frosinone",
    "Genoa": "Genova",
    "Inter": "Milano",
    "Juventus": "Torino",
    "Lazio": "Roma",
    "Lecce": "Lecce",
    "Milan": "Milano",
    "Monza": "Monza",
    "Napoli": "Napoli",
    "Parma": "Parma",
    "Roma": "Roma",
    "Salernitana": "Salerno",
    "Sampdoria": "Genova",
    "Sassuolo": "Sassuolo",
    "Spezia": "La Spezia",
    "Torino": "Torino",
    "Udinese": "Udine",
    "Venezia": "Venezia",
    "Verona": "Verona",
    "Chievo": "Verona",
    "SPAL": "Ferrara",
    "Brescia": "Brescia",
    "Catania": "Catania",
    "Livorno": "Livorno",
    "Siena": "Siena",
    "Reggina": "Reggio Calabria",
    "Messina": "Messina",
    "Treviso": "Treviso",
    "Ascoli": "Ascoli Piceno",
    "Cesena": "Cesena",
    "Novara": "Novara",
    "Pescara": "Pescara",
    "Bari": "Bari",
    "Palermo": "Palermo",
    "Carpi": "Carpi",
    "Perugia": "Perugia",
}

# Stadium names by team (for venue column)
TEAM_VENUES: dict[str, str] = {
    "Atalanta": "Gewiss Stadium, Bergamo",
    "Benevento": "Stadio Ciro Vigorito, Benevento",
    "Bologna": "Stadio Renato Dall'Ara, Bologna",
    "Cagliari": "Unipol Domus, Cagliari",
    "Como": "Stadio Giuseppe Sinigaglia, Como",
    "Cremonese": "Stadio Giovanni Zini, Cremona",
    "Crotone": "Stadio Ezio Scida, Crotone",
    "Empoli": "Stadio Carlo Castellani, Empoli",
    "Fiorentina": "Stadio Artemio Franchi, Firenze",
    "Frosinone": "Stadio Benito Stirpe, Frosinone",
    "Genoa": "Stadio Luigi Ferraris, Genova",
    "Inter": "Stadio Giuseppe Meazza, Milano",
    "Juventus": "Allianz Stadium, Torino",
    "Lazio": "Stadio Olimpico, Roma",
    "Lecce": "Stadio Via del Mare, Lecce",
    "Milan": "Stadio Giuseppe Meazza, Milano",
    "Monza": "U-Power Stadium, Monza",
    "Napoli": "Stadio Diego Armando Maradona, Napoli",
    "Parma": "Stadio Ennio Tardini, Parma",
    "Roma": "Stadio Olimpico, Roma",
    "Salernitana": "Stadio Arechi, Salerno",
    "Sampdoria": "Stadio Luigi Ferraris, Genova",
    "Sassuolo": "Mapei Stadium, Sassuolo",
    "Spezia": "Stadio Alberto Picco, La Spezia",
    "Torino": "Stadio Olimpico Grande Torino, Torino",
    "Udinese": "Bluenergy Stadium, Udine",
    "Venezia": "Stadio Pier Luigi Penzo, Venezia",
    "Verona": "Stadio Marc'Antonio Bentegodi, Verona",
    "Chievo": "Stadio Marc'Antonio Bentegodi, Verona",
    "SPAL": "Stadio Paolo Mazza, Ferrara",
    "Brescia": "Stadio Mario Rigamonti, Brescia",
    "Catania": "Stadio Angelo Massimino, Catania",
    "Livorno": "Stadio Armando Picchi, Livorno",
    "Siena": "Stadio Artemio Franchi, Siena",
    "Reggina": "Stadio Oreste Granillo, Reggio Calabria",
    "Palermo": "Stadio Renzo Barbera, Palermo",
    "Bari": "Stadio San Nicola, Bari",
    "Pescara": "Stadio Adriatico, Pescara",
}


def import_historical_season(
    season: str,
    league: str | None = None,
) -> pd.DataFrame:
    """Import a single season from football-data.co.uk CSV.

    Downloads the CSV if not already cached, converts to matches.parquet schema.

    Args:
        season: e.g. "2024-2025"
        league: league key from LEAGUES dict. Defaults to DEFAULT_LEAGUE.

    Returns DataFrame compatible with existing matches.parquet.
    """
    if league is None:
        league = DEFAULT_LEAGUE

    league_code, league_name = LEAGUES[league]
    log.info("Importing %s %s from football-data.co.uk", league_name, season)

    # Download CSV (uses cache if available)
    raw = download_odds(season, league=league)
    if raw.empty:
        log.warning("No data available for %s %s", league_name, season)
        return pd.DataFrame()

    # Normalize team names
    raw["home_team"] = raw["HomeTeam"].apply(normalize_team)
    raw["away_team"] = raw["AwayTeam"].apply(normalize_team)

    # Parse dates
    raw["match_date"] = pd.to_datetime(
        raw["Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    raw = raw.sort_values("match_date").reset_index(drop=True)

    # Compute matchweek: count each team's matches chronologically
    team_match_count: dict[str, int] = {}
    matchweeks = []
    for _, row in raw.iterrows():
        h = row["home_team"]
        a = row["away_team"]
        team_match_count[h] = team_match_count.get(h, 0) + 1
        team_match_count[a] = team_match_count.get(a, 0) + 1
        matchweeks.append(team_match_count[h])
    raw["matchweek"] = matchweeks

    # Build matches DataFrame in our schema
    matches = pd.DataFrame()
    matches["match_id"] = raw.apply(
        lambda r: f"{r['match_date'].strftime('%Y-%m-%d')}_{r['home_team']}_{r['away_team']}",
        axis=1,
    )
    matches["season"] = season
    matches["league"] = league
    matches["league_name"] = league_name
    matches["match_date"] = pd.to_datetime(raw["match_date"])
    matches["kickoff_time"] = raw.get("Time", "")
    matches["matchweek"] = raw["matchweek"]
    matches["home_team"] = raw["home_team"]
    matches["away_team"] = raw["away_team"]
    matches["home_score"] = pd.to_numeric(raw["FTHG"], errors="coerce")
    matches["away_score"] = pd.to_numeric(raw["FTAG"], errors="coerce")

    # Venue (from team lookup, Serie A only)
    if league == "serie_a":
        matches["venue"] = raw["home_team"].map(TEAM_VENUES)
    else:
        matches["venue"] = None

    # No attendance, referee, manager, captain, formation from this source
    matches["attendance"] = None
    matches["referee"] = None
    matches["home_manager"] = None
    matches["away_manager"] = None
    matches["home_captain"] = None
    matches["away_captain"] = None
    matches["home_formation"] = None
    matches["away_formation"] = None

    # Map available stats
    matches["home_fouls"] = pd.to_numeric(raw.get("HF"), errors="coerce")
    matches["away_fouls"] = pd.to_numeric(raw.get("AF"), errors="coerce")
    matches["home_corners"] = pd.to_numeric(raw.get("HC"), errors="coerce")
    matches["away_corners"] = pd.to_numeric(raw.get("AC"), errors="coerce")
    matches["home_yellow_cards"] = pd.to_numeric(raw.get("HY"), errors="coerce")
    matches["away_yellow_cards"] = pd.to_numeric(raw.get("AY"), errors="coerce")
    matches["home_red_cards"] = pd.to_numeric(raw.get("HR"), errors="coerce")
    matches["away_red_cards"] = pd.to_numeric(raw.get("AR"), errors="coerce")

    # Shots
    hs = pd.to_numeric(raw.get("HS"), errors="coerce")
    as_ = pd.to_numeric(raw.get("AS"), errors="coerce")
    hst = pd.to_numeric(raw.get("HST"), errors="coerce")
    ast = pd.to_numeric(raw.get("AST"), errors="coerce")

    matches["home_shots_on_target_total"] = hs
    matches["home_shots_on_target_count"] = hst
    matches["away_shots_on_target_total"] = as_
    matches["away_shots_on_target_count"] = ast

    # SoT percentage (consistent with FBref data)
    matches["home_shots_on_target"] = (hst / hs * 100).where(hs > 0, None)
    matches["away_shots_on_target"] = (ast / as_ * 100).where(as_ > 0, None)

    # Half-time scores (bonus data not in FBref)
    matches["home_ht_score"] = pd.to_numeric(raw.get("HTHG"), errors="coerce")
    matches["away_ht_score"] = pd.to_numeric(raw.get("HTAG"), errors="coerce")

    # xG not available from this source
    matches["home_xg"] = None
    matches["away_xg"] = None

    # Stats not available from football-data.co.uk
    for col in [
        "home_possession", "away_possession",
        "home_passing_accuracy", "away_passing_accuracy",
        "home_saves", "away_saves",
        "home_cards", "away_cards",
        "home_crosses", "away_crosses",
        "home_touches", "away_touches",
        "home_tackles", "away_tackles",
        "home_interceptions", "away_interceptions",
        "home_aerials_won", "away_aerials_won",
        "home_clearances", "away_clearances",
        "home_offsides", "away_offsides",
        "home_goal_kicks", "away_goal_kicks",
        "home_throw_ins", "away_throw_ins",
        "home_long_balls", "away_long_balls",
        "home_passing_accuracy_count", "home_passing_accuracy_total",
        "away_passing_accuracy_count", "away_passing_accuracy_total",
        "home_saves_count", "home_saves_total",
        "away_saves_count", "away_saves_total",
    ]:
        matches[col] = None

    # Data source marker
    matches["data_source"] = "football-data.co.uk"

    log.info(
        "Imported %d matches for %s %s (%d unique teams)",
        len(matches),
        league_name,
        season,
        matches["home_team"].nunique(),
    )
    return matches


def import_all_historical(
    seasons: list[str] | None = None,
    leagues: list[str] | None = None,
    delay: float = 1.0,
) -> pd.DataFrame:
    """Import historical data for multiple seasons and leagues.

    Merges with existing matches.parquet, avoiding duplicates.

    Args:
        seasons: list of season strings. Defaults to SEASONS[:-1].
        leagues: list of league keys. Defaults to all LEAGUES.
        delay: seconds between downloads to be polite to server.
    """
    if seasons is None:
        # Import all seasons except the latest (which we may have from FBref)
        seasons = SEASONS[:-1]

    if leagues is None:
        leagues = list(LEAGUES.keys())

    # Load existing matches
    matches_path = parsed_path("matches")
    if matches_path.exists():
        existing = pd.read_parquet(matches_path)
        existing_ids = set(existing["match_id"])
        log.info("Existing matches: %d", len(existing))
    else:
        existing = pd.DataFrame()
        existing_ids = set()

    all_new = []
    for league in leagues:
        league_code, league_name = LEAGUES[league]
        log.info("--- Importing %s ---", league_name)

        for season in seasons:
            df = import_historical_season(season, league=league)
            if df.empty:
                continue
            # Remove any duplicates with existing data
            new_only = df[~df["match_id"].isin(existing_ids)]
            if len(new_only) > 0:
                all_new.append(new_only)
                log.info("  %s %s: %d new matches", league_name, season, len(new_only))
            else:
                log.info("  %s %s: already imported", league_name, season)

            # Be polite to the server
            time.sleep(delay)

    if not all_new:
        log.info("No new historical matches to import")
        return existing

    new_df = pd.concat(all_new, ignore_index=True)

    # Merge with existing
    if not existing.empty:
        # Add league/data_source to existing if missing
        if "data_source" not in existing.columns:
            existing["data_source"] = "fbref"
        if "league" not in existing.columns:
            existing["league"] = "serie_a"
        if "league_name" not in existing.columns:
            existing["league_name"] = "Serie A"
        result = pd.concat([existing, new_df], ignore_index=True)
    else:
        result = new_df

    # Sort by date
    result["_sort_date"] = pd.to_datetime(result["match_date"], errors="coerce")
    result = result.sort_values("_sort_date").reset_index(drop=True)
    result.drop(columns=["_sort_date"], inplace=True)

    # Save
    result.to_parquet(matches_path, index=False)
    log.info(
        "Matches table: %d total (%d new from %d leagues) saved to %s",
        len(result),
        len(new_df),
        len(leagues),
        matches_path,
    )

    return result
