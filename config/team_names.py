"""Canonical team name mapping for all supported leagues.

Maps football-data.co.uk names (and FBref variants) to a single canonical name
per team. Organized by league for clarity, but all share one global lookup.
"""

import unicodedata

# ---------------------------------------------------------------------------
# Serie A (Italy) — including all teams from 2005 onwards
# ---------------------------------------------------------------------------
SERIE_A_NAMES: dict[str, str] = {
    # Current / recent teams
    "Internazionale": "Inter",
    "Inter Milan": "Inter",
    "FC Internazionale Milano": "Inter",
    "AC Milan": "Milan",
    "AC Milan1": "Milan",
    "Juventus": "Juventus",
    "Juventus FC": "Juventus",
    "Napoli": "Napoli",
    "SSC Napoli": "Napoli",
    "Roma": "Roma",
    "AS Roma": "Roma",
    "Lazio": "Lazio",
    "SS Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Atalanta BC": "Atalanta",
    "Fiorentina": "Fiorentina",
    "ACF Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Bologna FC 1909": "Bologna",
    "Bologna FC": "Bologna",
    "Torino": "Torino",
    "Torino FC": "Torino",
    "Udinese": "Udinese",
    "Udinese Calcio": "Udinese",
    "Genoa": "Genoa",
    "Genoa CFC": "Genoa",
    "Cagliari": "Cagliari",
    "Cagliari Calcio": "Cagliari",
    "Empoli": "Empoli",
    "Empoli FC": "Empoli",
    "Hellas Verona": "Verona",
    "Hellas Verona FC": "Verona",
    "Verona": "Verona",
    "Monza": "Monza",
    "AC Monza": "Monza",
    "Lecce": "Lecce",
    "US Lecce": "Lecce",
    "Sassuolo": "Sassuolo",
    "US Sassuolo Calcio": "Sassuolo",
    "US Sassuolo": "Sassuolo",
    "Salernitana": "Salernitana",
    "US Salernitana 1919": "Salernitana",
    "Frosinone": "Frosinone",
    "Frosinone Calcio": "Frosinone",
    "Spezia": "Spezia",
    "Spezia Calcio": "Spezia",
    "Sampdoria": "Sampdoria",
    "UC Sampdoria": "Sampdoria",
    "Venezia": "Venezia",
    "Venezia FC": "Venezia",
    "Benevento": "Benevento",
    "Benevento Calcio": "Benevento",
    "Crotone": "Crotone",
    "FC Crotone": "Crotone",
    "Parma": "Parma",
    "Parma Calcio 1913": "Parma",
    "Como": "Como",
    "Como 1907": "Como",
    "Brescia": "Brescia",
    "Brescia Calcio": "Brescia",
    "SPAL": "SPAL",
    "SPAL 2013": "SPAL",
    "Spal": "SPAL",
    "Cremonese": "Cremonese",
    "US Cremonese": "Cremonese",
    "Chievo": "Chievo",
    "ChievoVerona": "Chievo",
    "AC ChievoVerona": "Chievo",
    # Older Serie A teams (2005-2017)
    "Catania": "Catania",
    "Calcio Catania": "Catania",
    "Livorno": "Livorno",
    "AS Livorno": "Livorno",
    "AS Livorno Calcio": "Livorno",
    "Siena": "Siena",
    "AC Siena": "Siena",
    "Robur Siena": "Siena",
    "Reggina": "Reggina",
    "Reggina Calcio": "Reggina",
    "Messina": "Messina",
    "FC Messina": "Messina",
    "Treviso": "Treviso",
    "Ascoli": "Ascoli",
    "Ascoli Calcio": "Ascoli",
    "Cesena": "Cesena",
    "AC Cesena": "Cesena",
    "Novara": "Novara",
    "Novara Calcio": "Novara",
    "Pescara": "Pescara",
    "Delfino Pescara": "Pescara",
    "Bari": "Bari",
    "SSC Bari": "Bari",
    "AS Bari": "Bari",
    "Palermo": "Palermo",
    "US Palermo": "Palermo",
    "Catanzaro": "Catanzaro",
    "US Catanzaro 1929": "Catanzaro",
    "Cittadella": "Cittadella",
    "Pisa": "Pisa",
    "Pisa SC": "Pisa",
    "AC Pisa 1909": "Pisa",
    "Carpi": "Carpi",
    "Carpi FC": "Carpi",
    "Ternana": "Ternana",
    "Ternana Calcio": "Ternana",
    "Perugia": "Perugia",
    "AC Perugia Calcio": "Perugia",
    "Avellino": "Avellino",
    "US Avellino": "Avellino",
}

# ---------------------------------------------------------------------------
# Premier League (England)
# ---------------------------------------------------------------------------
PREMIER_LEAGUE_NAMES: dict[str, str] = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Birmingham": "Birmingham",
    "Birmingham City": "Birmingham",
    "Blackburn": "Blackburn",
    "Blackburn Rovers": "Blackburn",
    "Blackpool": "Blackpool",
    "Bolton": "Bolton",
    "Bolton Wanderers": "Bolton",
    "Bournemouth": "Bournemouth",
    "AFC Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brentford FC": "Brentford",
    "Brighton": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Burnley": "Burnley",
    "Burnley FC": "Burnley",
    "Cardiff": "Cardiff",
    "Cardiff City": "Cardiff",
    "Charlton": "Charlton",
    "Charlton Athletic": "Charlton",
    "Chelsea": "Chelsea",
    "Coventry": "Coventry",
    "Coventry City": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Derby": "Derby",
    "Derby County": "Derby",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Fulham FC": "Fulham",
    "Hull": "Hull",
    "Hull City": "Hull",
    "Huddersfield": "Huddersfield",
    "Huddersfield Town": "Huddersfield",
    "Ipswich": "Ipswich",
    "Ipswich Town": "Ipswich",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Liverpool FC": "Liverpool",
    "Luton": "Luton",
    "Luton Town": "Luton",
    "Man City": "Man City",
    "Manchester City": "Man City",
    "Man United": "Man United",
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",
    "Middlesbrough": "Middlesbrough",
    "Middlesboro": "Middlesbrough",
    "Newcastle": "Newcastle",
    "Newcastle United": "Newcastle",
    "Norwich": "Norwich",
    "Norwich City": "Norwich",
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "Portsmouth": "Portsmouth",
    "QPR": "QPR",
    "Queens Park Rangers": "QPR",
    "Reading": "Reading",
    "Sheffield United": "Sheffield United",
    "Sheffield Utd": "Sheffield United",
    "Southampton": "Southampton",
    "Stoke": "Stoke",
    "Stoke City": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea",
    "Swansea City": "Swansea",
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Watford": "Watford",
    "West Brom": "West Brom",
    "West Bromwich Albion": "West Brom",
    "West Ham": "West Ham",
    "West Ham United": "West Ham",
    "Wigan": "Wigan",
    "Wigan Athletic": "Wigan",
    "Wolves": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
}

# ---------------------------------------------------------------------------
# La Liga (Spain)
# ---------------------------------------------------------------------------
LA_LIGA_NAMES: dict[str, str] = {
    "Alaves": "Alaves",
    "Deportivo Alaves": "Alaves",
    "Almeria": "Almeria",
    "UD Almeria": "Almeria",
    "Ath Bilbao": "Athletic Bilbao",
    "Athletic Bilbao": "Athletic Bilbao",
    "Ath Madrid": "Atletico Madrid",
    "Atletico Madrid": "Atletico Madrid",
    "Atl. Madrid": "Atletico Madrid",
    "Barcelona": "Barcelona",
    "FC Barcelona": "Barcelona",
    "Betis": "Betis",
    "Real Betis": "Betis",
    "Cadiz": "Cadiz",
    "Cadiz CF": "Cadiz",
    "Celta": "Celta Vigo",
    "Celta Vigo": "Celta Vigo",
    "Cordoba": "Cordoba",
    "Deportivo La Coruna": "Deportivo",
    "La Coruna": "Deportivo",
    "Eibar": "Eibar",
    "SD Eibar": "Eibar",
    "Elche": "Elche",
    "Elche CF": "Elche",
    "Espanol": "Espanyol",
    "Espanyol": "Espanyol",
    "Getafe": "Getafe",
    "Getafe CF": "Getafe",
    "Girona": "Girona",
    "Girona FC": "Girona",
    "Granada": "Granada",
    "Granada CF": "Granada",
    "Gimnastic": "Gimnastic",
    "Hercules": "Hercules",
    "Huesca": "Huesca",
    "SD Huesca": "Huesca",
    "Las Palmas": "Las Palmas",
    "UD Las Palmas": "Las Palmas",
    "Leganes": "Leganes",
    "CD Leganes": "Leganes",
    "Levante": "Levante",
    "Levante UD": "Levante",
    "Mallorca": "Mallorca",
    "RCD Mallorca": "Mallorca",
    "Malaga": "Malaga",
    "Malaga CF": "Malaga",
    "Numancia": "Numancia",
    "Osasuna": "Osasuna",
    "CA Osasuna": "Osasuna",
    "Rayo Vallecano": "Rayo Vallecano",
    "Real Madrid": "Real Madrid",
    "Real Sociedad": "Real Sociedad",
    "Recreativo": "Recreativo",
    "Recreativo Huelva": "Recreativo",
    "Sevilla": "Sevilla",
    "Sevilla FC": "Sevilla",
    "Sp Gijon": "Sporting Gijon",
    "Sporting Gijon": "Sporting Gijon",
    "Tenerife": "Tenerife",
    "CD Tenerife": "Tenerife",
    "Valencia": "Valencia",
    "Valencia CF": "Valencia",
    "Valladolid": "Valladolid",
    "Real Valladolid": "Valladolid",
    "Villarreal": "Villarreal",
    "Villarreal CF": "Villarreal",
    "Xerez": "Xerez",
    "Zaragoza": "Zaragoza",
    "Real Zaragoza": "Zaragoza",
}

# ---------------------------------------------------------------------------
# Bundesliga (Germany)
# ---------------------------------------------------------------------------
BUNDESLIGA_NAMES: dict[str, str] = {
    "Augsburg": "Augsburg",
    "FC Augsburg": "Augsburg",
    "Bayern Munich": "Bayern Munich",
    "FC Bayern Munich": "Bayern Munich",
    "Bielefeld": "Bielefeld",
    "Arminia Bielefeld": "Bielefeld",
    "Bochum": "Bochum",
    "VfL Bochum": "Bochum",
    "Dortmund": "Dortmund",
    "Borussia Dortmund": "Dortmund",
    "Dusseldorf": "Dusseldorf",
    "Fortuna Dusseldorf": "Dusseldorf",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "FC Koln",
    "1. FC Koln": "FC Koln",
    "Freiburg": "Freiburg",
    "SC Freiburg": "Freiburg",
    "Furth": "Furth",
    "SpVgg Greuther Furth": "Furth",
    "Greuther Furth": "Furth",
    "Hamburg": "Hamburg",
    "Hamburger SV": "Hamburg",
    "Hannover": "Hannover",
    "Hannover 96": "Hannover",
    "Heidenheim": "Heidenheim",
    "1. FC Heidenheim": "Heidenheim",
    "Hertha": "Hertha Berlin",
    "Hertha Berlin": "Hertha Berlin",
    "Hertha BSC": "Hertha Berlin",
    "Hoffenheim": "Hoffenheim",
    "TSG Hoffenheim": "Hoffenheim",
    "Ingolstadt": "Ingolstadt",
    "FC Ingolstadt": "Ingolstadt",
    "Kaiserslautern": "Kaiserslautern",
    "1. FC Kaiserslautern": "Kaiserslautern",
    "Karlsruhe": "Karlsruhe",
    "Karlsruher SC": "Karlsruhe",
    "Leverkusen": "Leverkusen",
    "Bayer Leverkusen": "Leverkusen",
    "M'gladbach": "Monchengladbach",
    "Monchengladbach": "Monchengladbach",
    "Borussia Monchengladbach": "Monchengladbach",
    "Mainz": "Mainz",
    "1. FSV Mainz 05": "Mainz",
    "Mainz 05": "Mainz",
    "Nurnberg": "Nurnberg",
    "1. FC Nurnberg": "Nurnberg",
    "Paderborn": "Paderborn",
    "SC Paderborn": "Paderborn",
    "RB Leipzig": "RB Leipzig",
    "RasenBallsport Leipzig": "RB Leipzig",
    "Schalke 04": "Schalke 04",
    "FC Schalke 04": "Schalke 04",
    "St Pauli": "St Pauli",
    "FC St. Pauli": "St Pauli",
    "Stuttgart": "Stuttgart",
    "VfB Stuttgart": "Stuttgart",
    "Union Berlin": "Union Berlin",
    "1. FC Union Berlin": "Union Berlin",
    "Werder Bremen": "Werder Bremen",
    "SV Werder Bremen": "Werder Bremen",
    "Wolfsburg": "Wolfsburg",
    "VfL Wolfsburg": "Wolfsburg",
    "Darmstadt": "Darmstadt",
    "SV Darmstadt 98": "Darmstadt",
    "Cottbus": "Cottbus",
    "Energie Cottbus": "Cottbus",
    "Duisburg": "Duisburg",
    "MSV Duisburg": "Duisburg",
    "Holstein Kiel": "Holstein Kiel",
}

# ---------------------------------------------------------------------------
# Ligue 1 (France)
# ---------------------------------------------------------------------------
LIGUE_1_NAMES: dict[str, str] = {
    "Ajaccio": "Ajaccio",
    "AC Ajaccio": "Ajaccio",
    "GFC Ajaccio": "GFC Ajaccio",
    "Angers": "Angers",
    "Angers SCO": "Angers",
    "Auxerre": "Auxerre",
    "AJ Auxerre": "Auxerre",
    "Bastia": "Bastia",
    "SC Bastia": "Bastia",
    "Bordeaux": "Bordeaux",
    "Girondins de Bordeaux": "Bordeaux",
    "Brest": "Brest",
    "Stade Brestois": "Brest",
    "Caen": "Caen",
    "SM Caen": "Caen",
    "Clermont": "Clermont",
    "Clermont Foot": "Clermont",
    "Dijon": "Dijon",
    "Dijon FCO": "Dijon",
    "Evian Thonon Gaillard": "Evian",
    "Grenoble": "Grenoble",
    "Grenoble Foot": "Grenoble",
    "Guingamp": "Guingamp",
    "En Avant Guingamp": "Guingamp",
    "Le Havre": "Le Havre",
    "Le Mans": "Le Mans",
    "Lens": "Lens",
    "RC Lens": "Lens",
    "Lille": "Lille",
    "LOSC Lille": "Lille",
    "Lorient": "Lorient",
    "FC Lorient": "Lorient",
    "Lyon": "Lyon",
    "Olympique Lyonnais": "Lyon",
    "Marseille": "Marseille",
    "Olympique de Marseille": "Marseille",
    "Metz": "Metz",
    "FC Metz": "Metz",
    "Monaco": "Monaco",
    "AS Monaco": "Monaco",
    "Montpellier": "Montpellier",
    "Montpellier HSC": "Montpellier",
    "Nancy": "Nancy",
    "AS Nancy": "Nancy",
    "Nantes": "Nantes",
    "FC Nantes": "Nantes",
    "Nice": "Nice",
    "OGC Nice": "Nice",
    "Nimes": "Nimes",
    "Nimes Olympique": "Nimes",
    "Paris SG": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "PSG": "Paris SG",
    "Reims": "Reims",
    "Stade de Reims": "Reims",
    "Rennes": "Rennes",
    "Stade Rennais": "Rennes",
    "Sedan": "Sedan",
    "CS Sedan": "Sedan",
    "Sochaux": "Sochaux",
    "FC Sochaux": "Sochaux",
    "St Etienne": "Saint-Etienne",
    "Saint-Etienne": "Saint-Etienne",
    "AS Saint-Etienne": "Saint-Etienne",
    "Strasbourg": "Strasbourg",
    "RC Strasbourg": "Strasbourg",
    "Toulouse": "Toulouse",
    "Toulouse FC": "Toulouse",
    "Troyes": "Troyes",
    "ES Troyes AC": "Troyes",
    "Valenciennes": "Valenciennes",
    "Valenciennes FC": "Valenciennes",
}

# ---------------------------------------------------------------------------
# Combined lookup (all leagues)
# ---------------------------------------------------------------------------
TEAM_NAME_MAP: dict[str, str] = {}
TEAM_NAME_MAP.update(SERIE_A_NAMES)
TEAM_NAME_MAP.update(PREMIER_LEAGUE_NAMES)
TEAM_NAME_MAP.update(LA_LIGA_NAMES)
TEAM_NAME_MAP.update(BUNDESLIGA_NAMES)
TEAM_NAME_MAP.update(LIGUE_1_NAMES)
# Ensure every canonical name (map value) is also a key mapping to itself.
# This lets normalize_team("Inter") work even though "Inter" is only a value
# in the per-league dicts (keyed under "Internazionale", "Inter Milan", etc.).
for _canon in set(TEAM_NAME_MAP.values()):
    TEAM_NAME_MAP.setdefault(_canon, _canon)

# ---------------------------------------------------------------------------
# Current season canonical team lists (update each season)
# ---------------------------------------------------------------------------
# These are the 20 canonical team names for the active season.
# All data sources (FBref, Sofascore, Transfermarkt, Understat, Odds API)
# MUST normalize to these exact names via normalize_team().
#
# When promoted/relegated teams change between seasons, update this list
# and verify all scrapers produce the correct canonical names.
SERIE_A_2026_27: list[str] = [
    "Atalanta",
    "Bologna",
    "Cagliari",
    "Como",
    "Fiorentina",
    "Frosinone",    # promoted from Serie B (last in Serie A 2023-24)
    "Genoa",
    "Inter",
    "Juventus",
    "Lazio",
    "Lecce",
    "Milan",
    "Monza",        # promoted from Serie B (last in Serie A 2024-25)
    "Napoli",
    "Parma",
    "Roma",
    "Sassuolo",
    "Torino",
    "Udinese",
    "Venezia",      # promoted from Serie B (last in Serie A 2024-25)
]
# Relegated after 2025-26: Cremonese, Pisa, Verona. They keep their
# normalize_team() aliases above — historical rows still need resolving.

# The CURRENT-season Premier League 20. Derived from the Sofascore season
# fixture file (380 matches, primary source), not hand-typed: this list sat
# at 2025-26 while SERIE_A_2026_27 beside it had already rolled over, so the
# only EPL consumer resolved names against four relegated clubs and knew
# nothing of the four promoted ones. Roll BOTH lists every August.
PREMIER_LEAGUE_2026_27: list[str] = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Coventry",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull",
    "Ipswich",
    "Leeds",
    "Liverpool",
    "Man City",
    "Man United",
    "Newcastle",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham",
]


# Pre-built case-insensitive lookup: lowered variant -> canonical name.
# Covers sources that send lowercase names (Understat, Sofascore).
_TEAM_NAME_MAP_LOWER: dict[str, str] = {k.lower(): v for k, v in TEAM_NAME_MAP.items()}


def normalize_team(name) -> str:
    """Return the canonical team name for any known variant.

    Performs a case-sensitive lookup first (fast path), then falls back to a
    case-insensitive match.  This handles data sources that send lowercase
    names (e.g., Understat "inter" -> "Inter", Sofascore "ac milan" -> "Milan").
    """
    if not isinstance(name, str):
        return str(name) if name is not None and name == name else ""
    stripped = name.strip()
    # Fast path: exact match
    result = TEAM_NAME_MAP.get(stripped)
    if result is not None:
        return result
    # Fallback: case-insensitive
    result = _TEAM_NAME_MAP_LOWER.get(stripped.lower())
    if result is not None:
        return result
    return stripped


def normalize_team_safe(name: str) -> str:
    """NaN-safe wrapper around :func:`normalize_team`.

    Returns ``""`` for NaN / None inputs, otherwise delegates to
    :func:`normalize_team`.  Useful in pandas ``.apply()`` calls where
    the column may contain missing values.
    """
    try:
        import pandas as pd

        if pd.isna(name):
            return ""
    except Exception:
        pass
    return normalize_team(name)


def strip_accents(s: str) -> str:
    """Remove diacritical marks from a string.

    Useful for fuzzy matching player/team names across sources that differ
    only in accent usage (e.g. "Martínez" vs "Martinez").
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
