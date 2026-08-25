"""Master league registry for multi-league expansion.

Central source of truth for all league-specific configuration:
  - External API identifiers (football-data, FBref, Understat, etc.)
  - Derby/rivalry definitions with intensity levels
  - Promoted teams per season
  - Helper to retrieve per-league config
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# A) League Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LeagueConfig:
    """Immutable configuration for a single league."""
    name: str
    country: str
    football_data_code: str          # football-data.co.uk code (I1, E0, etc.)
    fbref_comp_id: int               # FBref competition ID
    understat_slug: str              # Understat league slug
    odds_api_key: str                # The Odds API sport key
    sofascore_tournament_id: int     # Sofascore unique tournament ID
    transfermarkt_id: str            # Transfermarkt competition path segment
    worldfootball_slug: str          # worldfootball.net league slug
    matchweeks_per_season: int       # Standard number of matchweeks
    timezone: str                    # Primary timezone for match scheduling
    draw_rate: float = 0.25          # Historical draw rate (for stake sizing)
    var_introduced: str = "2017-2018"  # Season VAR was introduced
    avg_monthly_temps: tuple = ()    # (Jan..Dec) avg °C for weather features


LEAGUE_REGISTRY: dict[str, LeagueConfig] = {
    "serie_a": LeagueConfig(
        name="Serie A",
        country="Italy",
        football_data_code="I1",
        fbref_comp_id=11,
        understat_slug="Serie_A",
        odds_api_key="soccer_italy_serie_a",
        sofascore_tournament_id=23,
        transfermarkt_id="serie-a",
        worldfootball_slug="ita-serie-a",
        matchweeks_per_season=38,
        timezone="Europe/Rome",
        draw_rate=0.28,
        var_introduced="2017-2018",
        avg_monthly_temps=(7, 9, 12, 16, 20, 25, 28, 28, 24, 18, 12, 8),
    ),
    "premier_league": LeagueConfig(
        name="Premier League",
        country="England",
        football_data_code="E0",
        fbref_comp_id=9,
        understat_slug="EPL",
        odds_api_key="soccer_epl",
        sofascore_tournament_id=17,
        transfermarkt_id="premier-league",
        worldfootball_slug="eng-premier-league",
        matchweeks_per_season=38,
        timezone="Europe/London",
        draw_rate=0.22,
        var_introduced="2019-2020",
        avg_monthly_temps=(5, 5, 7, 10, 13, 16, 18, 18, 15, 11, 7, 5),
    ),
    "la_liga": LeagueConfig(
        name="La Liga",
        country="Spain",
        football_data_code="SP1",
        fbref_comp_id=12,
        understat_slug="La_Liga",
        odds_api_key="soccer_spain_la_liga",
        sofascore_tournament_id=8,
        transfermarkt_id="laliga",
        worldfootball_slug="esp-primera-division",
        matchweeks_per_season=38,
        timezone="Europe/Madrid",
        draw_rate=0.24,
        var_introduced="2018-2019",
        avg_monthly_temps=(6, 8, 11, 14, 18, 24, 28, 27, 22, 16, 10, 7),
    ),
    "bundesliga": LeagueConfig(
        name="Bundesliga",
        country="Germany",
        football_data_code="D1",
        fbref_comp_id=20,
        understat_slug="Bundesliga",
        odds_api_key="soccer_germany_bundesliga",
        sofascore_tournament_id=35,
        transfermarkt_id="1-bundesliga",
        worldfootball_slug="ger-1-bundesliga",
        matchweeks_per_season=34,
        timezone="Europe/Berlin",
        draw_rate=0.23,
        var_introduced="2017-2018",
        avg_monthly_temps=(1, 2, 6, 10, 15, 18, 20, 20, 16, 10, 5, 2),
    ),
    "ligue_1": LeagueConfig(
        name="Ligue 1",
        country="France",
        football_data_code="F1",
        fbref_comp_id=13,
        understat_slug="Ligue_1",
        odds_api_key="soccer_france_ligue_one",
        sofascore_tournament_id=34,
        transfermarkt_id="ligue-1",
        worldfootball_slug="fra-ligue-1",
        matchweeks_per_season=34,
        timezone="Europe/Paris",
        draw_rate=0.24,
        var_introduced="2018-2019",
        avg_monthly_temps=(4, 5, 9, 12, 16, 20, 22, 22, 18, 13, 8, 5),
    ),
}

# Leagues currently active for predictions + betting.
# To add a new league: register it in LEAGUE_REGISTRY above, then add its key here.
# All pipeline steps, web routes, and data loaders iterate over this list.
ACTIVE_LEAGUES: list[str] = ["serie_a", "premier_league"]


# ---------------------------------------------------------------------------
# B) Derby / Rivalry Definitions
# ---------------------------------------------------------------------------
# Intensity scale:
#   3 = city derby (shared stadium or same city, fierce rivalry)
#   2 = regional rivalry or major historical rivalry
#   1 = traditional rivalry / big-match atmosphere

DERBIES: dict[str, dict[frozenset[str], int]] = {
    "serie_a": {
        frozenset({"Inter", "Milan"}): 3,               # Derby della Madonnina
        frozenset({"Roma", "Lazio"}): 3,                 # Derby della Capitale
        frozenset({"Torino", "Juventus"}): 3,            # Derby della Mole
        frozenset({"Genoa", "Sampdoria"}): 3,            # Derby della Lanterna
        frozenset({"Napoli", "Roma"}): 2,                # Derby del Sole
        frozenset({"Inter", "Juventus"}): 2,             # Derby d'Italia
        frozenset({"Fiorentina", "Juventus"}): 2,        # Historical rivalry
        frozenset({"Napoli", "Juventus"}): 2,            # North-South rivalry
        frozenset({"Milan", "Juventus"}): 1,             # Top-table clash
        frozenset({"Lazio", "Napoli"}): 1,               # Central-South rivalry
        frozenset({"Fiorentina", "Roma"}): 1,            # Historical rivalry
        frozenset({"Verona", "Venezia"}): 2,             # Veneto derby
        frozenset({"Parma", "Bologna"}): 2,              # Emilia derby
    },
    "premier_league": {
        frozenset({"Arsenal", "Tottenham"}): 3,          # North London derby
        frozenset({"Liverpool", "Everton"}): 3,          # Merseyside derby
        frozenset({"Man United", "Man City"}): 3,        # Manchester derby
        frozenset({"Chelsea", "Tottenham"}): 2,          # London rivalry
        frozenset({"Chelsea", "Arsenal"}): 2,            # London rivalry
        frozenset({"Liverpool", "Man United"}): 2,       # Northwest derby
        frozenset({"Arsenal", "Man United"}): 2,         # Historical rivalry (90s-2000s)
        frozenset({"Newcastle", "Sunderland"}): 3,       # Tyne-Wear derby
        frozenset({"Crystal Palace", "Brighton"}): 2,    # M23 derby
        frozenset({"Aston Villa", "Birmingham"}): 3,     # Second City derby
        frozenset({"West Ham", "Tottenham"}): 2,         # London rivalry
        frozenset({"Leeds", "Man United"}): 2,           # Roses rivalry
        frozenset({"Wolves", "West Brom"}): 3,           # Black Country derby
        frozenset({"Nottingham Forest", "Derby"}): 3,    # East Midlands derby
        frozenset({"Southampton", "Portsmouth"}): 3,     # South Coast derby
    },
    "la_liga": {
        frozenset({"Barcelona", "Real Madrid"}): 3,      # El Clasico
        frozenset({"Atletico Madrid", "Real Madrid"}): 3, # Madrid derby
        frozenset({"Sevilla", "Betis"}): 3,              # Seville derby
        frozenset({"Barcelona", "Espanyol"}): 3,         # Barcelona derby
        frozenset({"Athletic Bilbao", "Real Sociedad"}): 3,  # Basque derby
        frozenset({"Barcelona", "Atletico Madrid"}): 2,  # Historical rivalry
        frozenset({"Valencia", "Villarreal"}): 2,        # Comunitat Valenciana derby
        frozenset({"Valencia", "Levante"}): 3,           # Valencia city derby
        frozenset({"Celta Vigo", "Deportivo"}): 3,       # Galician derby
    },
    "bundesliga": {
        frozenset({"Bayern Munich", "Dortmund"}): 3,     # Der Klassiker
        frozenset({"Dortmund", "Schalke 04"}): 3,        # Revierderby
        frozenset({"Hamburg", "Werder Bremen"}): 3,       # Nordderby
        frozenset({"Bayern Munich", "RB Leipzig"}): 2,   # Modern rivalry
        frozenset({"Monchengladbach", "FC Koln"}): 3,    # Rhineland derby
        frozenset({"Eintracht Frankfurt", "Mainz"}): 2,  # Rhein-Main derby
        frozenset({"Hertha Berlin", "Union Berlin"}): 3, # Berlin derby
        frozenset({"Stuttgart", "Karlsruhe"}): 3,        # Baden-Wurttemberg derby
        frozenset({"Dortmund", "Monchengladbach"}): 1,   # Historical rivalry
        frozenset({"Leverkusen", "FC Koln"}): 2,         # Rhine derby
    },
    "ligue_1": {
        frozenset({"Paris SG", "Marseille"}): 3,         # Le Classique
        frozenset({"Lyon", "Saint-Etienne"}): 3,         # Derby Rhone-Alpes
        frozenset({"Lyon", "Marseille"}): 2,             # Olympique rivalry
        frozenset({"Nice", "Monaco"}): 2,                # Cote d'Azur derby
        frozenset({"Marseille", "Nice"}): 2,             # Mediterranean rivalry
        frozenset({"Lille", "Lens"}): 3,                 # Derby du Nord
        frozenset({"Nantes", "Rennes"}): 3,              # Breton derby
        frozenset({"Bordeaux", "Marseille"}): 2,         # Historical rivalry
        frozenset({"Strasbourg", "Metz"}): 3,            # Derby de l'Est
    },
}


# ---------------------------------------------------------------------------
# C) Promoted Teams Per League Per Season
# ---------------------------------------------------------------------------
# Tracks which teams were promoted to the top flight each season.
# Used by creative_factors.py for the "promoted team" feature.

PROMOTED_TEAMS: dict[str, dict[str, set[str]]] = {
    "serie_a": {
        "2017-2018": {"Benevento", "SPAL", "Verona"},
        "2018-2019": {"Parma", "Empoli", "Frosinone"},
        "2019-2020": {"Brescia", "Lecce", "Verona"},
        "2020-2021": {"Benevento", "Crotone", "Spezia"},
        "2021-2022": {"Empoli", "Salernitana", "Venezia"},
        "2022-2023": {"Cremonese", "Lecce", "Monza"},
        "2023-2024": {"Cagliari", "Frosinone", "Genoa"},
        "2024-2025": {"Como", "Parma", "Venezia"},
        "2025-2026": {"Sassuolo", "Pisa", "Cremonese"},
        # Measured 2026-08-02 against the live Sofascore 26/27 club list
        # (unique-tournament seasons[0]) diffed against last season's
        # player_match_stats table — exactly 3-up-3-down, out: Cremonese, Pisa,
        # Verona. Not typed from memory.
        "2026-2027": {"Frosinone", "Monza", "Venezia"},
    },
    "premier_league": {
        "2005-2006": {"Sunderland", "Wigan", "West Ham"},
        "2006-2007": {"Reading", "Sheffield United", "Watford"},
        "2007-2008": {"Sunderland", "Birmingham", "Derby"},
        "2008-2009": {"West Brom", "Stoke", "Hull"},
        "2009-2010": {"Wolves", "Birmingham", "Burnley"},
        "2010-2011": {"Newcastle", "West Brom", "Blackpool"},
        "2011-2012": {"QPR", "Norwich", "Swansea"},
        "2012-2013": {"Reading", "Southampton", "West Ham"},
        "2013-2014": {"Cardiff", "Hull", "Crystal Palace"},
        "2014-2015": {"Leicester", "Burnley", "QPR"},
        "2015-2016": {"Bournemouth", "Watford", "Norwich"},
        "2016-2017": {"Burnley", "Middlesbrough", "Hull"},
        "2017-2018": {"Newcastle", "Brighton", "Huddersfield"},
        "2018-2019": {"Wolves", "Cardiff", "Fulham"},
        "2019-2020": {"Norwich", "Sheffield United", "Aston Villa"},
        "2020-2021": {"Leeds", "West Brom", "Fulham"},
        "2021-2022": {"Norwich", "Watford", "Brentford"},
        "2022-2023": {"Fulham", "Bournemouth", "Nottingham Forest"},
        "2023-2024": {"Burnley", "Sheffield United", "Luton"},
        "2024-2025": {"Leicester", "Ipswich", "Southampton"},
        "2025-2026": {"Leeds", "Sheffield United", "Burnley"},
        # Same measurement as Serie A above; out: Burnley, West Ham, Wolves.
        # Brentford and Brighton ARE in the 26/27 league (confirmed live) — they
        # merely have no pre-season friendly rows yet, which is a scrape gap and
        # must not be read as relegation.
        "2026-2027": {"Coventry City", "Hull", "Ipswich"},
    },
    # ⚠️ la_liga / bundesliga / ligue_1 have NO 2026-2027 entry. That is
    # deliberate: they are not in ACTIVE_LEAGUES, so nothing measured their
    # promotions, and inventing three club names from memory is exactly the kind
    # of plausible-looking fiction this file must not contain. Add them only
    # from a live source, the way the two above were.
    "la_liga": {
        "2023-2024": {"Granada", "Las Palmas", "Alaves"},
        "2024-2025": {"Leganes", "Valladolid", "Espanyol"},
        "2025-2026": {"Racing Santander", "Castellon", "Elche"},
    },
    "bundesliga": {
        "2023-2024": {"Heidenheim", "Darmstadt", "FC Koln"},
        "2024-2025": {"St Pauli", "Holstein Kiel"},
        "2025-2026": {"Kaiserslautern", "Dusseldorf", "Paderborn"},
    },
    "ligue_1": {
        "2023-2024": {"Le Havre", "Metz", "Clermont"},
        "2024-2025": {"Auxerre", "Angers", "Saint-Etienne"},
        "2025-2026": {"Caen", "Lorient", "Bastia"},
    },
}


# ---------------------------------------------------------------------------
# D) Lookup Helpers
# ---------------------------------------------------------------------------

def get_league_config(league_key: str) -> LeagueConfig:
    """Return the LeagueConfig for the given key.

    Raises KeyError with a helpful message if the league is not found.
    """
    if league_key not in LEAGUE_REGISTRY:
        valid = ", ".join(sorted(LEAGUE_REGISTRY))
        raise KeyError(
            f"Unknown league '{league_key}'. Valid keys: {valid}"
        )
    return LEAGUE_REGISTRY[league_key]


def get_derbies(league_key: str) -> dict[frozenset[str], int]:
    """Return the derby definitions for a league.

    Returns an empty dict for leagues without derby data.
    """
    return DERBIES.get(league_key, {})


def get_promoted_teams(league_key: str, season: str) -> set[str]:
    """Return the set of promoted teams for a league and season.

    Returns an empty set if no data is available.
    """
    return PROMOTED_TEAMS.get(league_key, {}).get(season, set())


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------
# Previously this module only exposed matchweek counts. Existing code
# (features/build.py, features/creative_factors.py) imports these names.

MATCHWEEKS_PER_SEASON: dict[str, int] = {
    cfg.name: cfg.matchweeks_per_season
    for cfg in LEAGUE_REGISTRY.values()
}

DEFAULT_MATCHWEEKS = 38




# ---------------------------------------------------------------------------
# Team → league mapping (derived from current and recent rosters)
# ---------------------------------------------------------------------------
# Used to tag prediction/bet outputs with the correct league when the match
# dict only carries home/away team names. Includes name variants seen in the
# wild (Odds API full names, FBref short names). Add new teams here at season
# start; missing teams fall back to "serie_a" (the historical default).

_SERIE_A_TEAMS: frozenset[str] = frozenset({
    "Atalanta", "Bologna", "Cagliari", "Como", "Cremonese", "Empoli",
    "Fiorentina", "Genoa", "Inter", "Internazionale", "Juventus", "Lazio",
    "Lecce", "Milan", "AC Milan", "Monza", "Napoli", "Parma", "Pisa",
    "Roma", "AS Roma", "Sassuolo", "Torino", "Udinese", "Venezia", "Verona",
    "Hellas Verona",
})

_PREMIER_LEAGUE_TEAMS: frozenset[str] = frozenset({
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Brighton and Hove Albion", "Burnley", "Chelsea", "Crystal Palace",
    "Everton", "Fulham", "Ipswich", "Ipswich Town", "Leeds", "Leeds United",
    "Leicester", "Leicester City", "Liverpool", "Man City", "Manchester City",
    "Man United", "Manchester United", "Newcastle", "Newcastle United",
    "Coventry", "Coventry City", "Hull", "Hull City",
    "Nottingham Forest", "Nott'm Forest", "Nott'ham Forest",
    "Southampton", "Sunderland",
    "Tottenham", "Tottenham Hotspur", "Spurs", "West Ham", "West Ham United",
    "Wolves", "Wolverhampton Wanderers",
})


def infer_league(home_team: str | None = None, away_team: str | None = None) -> str:
    """Return the league key implied by the team names. Defaults to serie_a.

    Normalises before matching. These sets are hand-maintained alias lists, so
    any spelling nobody thought to add fell through to the serie_a DEFAULT
    rather than failing loudly — measured 2026-08-24: Sofascore emits
    "Liverpool FC" and "Coventry City", both of which inferred as Serie A, and
    `run_line_movement` then mis-tagged those EPL fixtures. Normalising first
    fixes every alias the canonical map already knows, instead of chasing
    spellings one at a time.
    """
    from config.team_names import normalize_team

    for t in (home_team, away_team):
        if not t:
            continue
        for candidate in (t, normalize_team(t)):
            if candidate in _PREMIER_LEAGUE_TEAMS:
                return "premier_league"
            if candidate in _SERIE_A_TEAMS:
                return "serie_a"
    return "serie_a"


def get_matchweeks(league: str | None = None) -> int:
    """Return the number of matchweeks for *league*, defaulting to 38.

    Accepts either display names ("Serie A") or registry keys ("serie_a").
    """
    if league is None:
        return DEFAULT_MATCHWEEKS
    # Try display name first (backward compat)
    if league in MATCHWEEKS_PER_SEASON:
        return MATCHWEEKS_PER_SEASON[league]
    # Try registry key
    if league in LEAGUE_REGISTRY:
        return LEAGUE_REGISTRY[league].matchweeks_per_season
    return DEFAULT_MATCHWEEKS
