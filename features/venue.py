"""Venue Intelligence Module - Stadium-specific features.

This module captures venue-related factors that impact match outcomes:
- Stadium capacity and typical attendance
- Pitch dimensions (narrow vs wide)
- Altitude effects (e.g., Atalanta's Bergamo)
- Travel distance for away team
- Home team's historical performance at venue

These factors are often overlooked but can significantly impact results.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Multi-league stadium database.
# The canonical lookup is ALL_STADIUMS (union of every league dict).
# Individual league dicts are kept for clarity.

# Serie A stadium data (2024-25 season)
SERIE_A_STADIUMS = {
    "Inter": {
        "name": "San Siro",
        "city": "Milan",
        "capacity": 75923,
        "altitude_m": 122,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.478,
        "lon": 9.124,
    },
    "Milan": {
        "name": "San Siro",
        "city": "Milan",
        "capacity": 75923,
        "altitude_m": 122,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.478,
        "lon": 9.124,
    },
    "Juventus": {
        "name": "Allianz Stadium",
        "city": "Turin",
        "capacity": 41507,
        "altitude_m": 239,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.110,
        "lon": 7.641,
    },
    "Napoli": {
        "name": "Diego Armando Maradona",
        "city": "Naples",
        "capacity": 54726,
        "altitude_m": 5,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 40.828,
        "lon": 14.193,
    },
    "Roma": {
        "name": "Stadio Olimpico",
        "city": "Rome",
        "capacity": 72698,
        "altitude_m": 18,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 41.934,
        "lon": 12.455,
    },
    "Lazio": {
        "name": "Stadio Olimpico",
        "city": "Rome",
        "capacity": 72698,
        "altitude_m": 18,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 41.934,
        "lon": 12.455,
    },
    "Atalanta": {
        "name": "Gewiss Stadium",
        "city": "Bergamo",
        "capacity": 24950,
        "altitude_m": 352,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.709,
        "lon": 9.681,
    },
    "Fiorentina": {
        "name": "Artemio Franchi",
        "city": "Florence",
        "capacity": 43147,
        "altitude_m": 50,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 43.781,
        "lon": 11.282,
    },
    "Bologna": {
        "name": "Renato Dall'Ara",
        "city": "Bologna",
        "capacity": 36462,
        "altitude_m": 54,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 44.492,
        "lon": 11.310,
    },
    "Torino": {
        "name": "Stadio Olimpico Grande Torino",
        "city": "Turin",
        "capacity": 27994,
        "altitude_m": 239,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.042,
        "lon": 7.650,
    },
    "Udinese": {
        "name": "Dacia Arena",
        "city": "Udine",
        "capacity": 25144,
        "altitude_m": 113,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 46.082,
        "lon": 13.200,
    },
    "Genoa": {
        "name": "Luigi Ferraris",
        "city": "Genoa",
        "capacity": 36599,
        "altitude_m": 19,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 44.416,
        "lon": 8.952,
    },
    "Sampdoria": {
        "name": "Luigi Ferraris",
        "city": "Genoa",
        "capacity": 36599,
        "altitude_m": 19,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 44.416,
        "lon": 8.952,
    },
    "Cagliari": {
        "name": "Unipol Domus",
        "city": "Cagliari",
        "capacity": 16416,
        "altitude_m": 4,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 39.200,
        "lon": 9.137,
    },
    "Sassuolo": {
        "name": "Mapei Stadium",
        "city": "Reggio Emilia",
        "capacity": 23717,
        "altitude_m": 58,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 44.714,
        "lon": 10.650,
    },
    "Empoli": {
        "name": "Stadio Carlo Castellani",
        "city": "Empoli",
        "capacity": 16284,
        "altitude_m": 28,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 43.726,
        "lon": 10.946,
    },
    "Verona": {
        "name": "Stadio Bentegodi",
        "city": "Verona",
        "capacity": 39211,
        "altitude_m": 59,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.435,
        "lon": 10.969,
    },
    "Hellas Verona": {
        "name": "Stadio Bentegodi",
        "city": "Verona",
        "capacity": 39211,
        "altitude_m": 59,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.435,
        "lon": 10.969,
    },
    "Lecce": {
        "name": "Via del Mare",
        "city": "Lecce",
        "capacity": 40670,
        "altitude_m": 49,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 40.355,
        "lon": 18.190,
    },
    "Monza": {
        "name": "U-Power Stadium",
        "city": "Monza",
        "capacity": 16917,
        "altitude_m": 162,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.584,
        "lon": 9.280,
    },
    "Parma": {
        "name": "Stadio Tardini",
        "city": "Parma",
        "capacity": 22352,
        "altitude_m": 55,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 44.795,
        "lon": 10.339,
    },
    "Venezia": {
        "name": "Stadio Penzo",
        "city": "Venice",
        "capacity": 11150,
        "altitude_m": 1,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.432,
        "lon": 12.377,
    },
    "Como": {
        "name": "Stadio Sinigaglia",
        "city": "Como",
        "capacity": 13602,
        "altitude_m": 201,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.819,
        "lon": 9.069,
    },
}

# Premier League stadiums (all 20 current + recent teams, using canonical names)
EPL_STADIUMS = {
    "Man United": {
        "name": "Old Trafford",
        "city": "Manchester",
        "capacity": 74310,
        "altitude_m": 44,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.463,
        "lon": -2.291,
    },
    "Liverpool": {
        "name": "Anfield",
        "city": "Liverpool",
        "capacity": 61276,
        "altitude_m": 30,
        "pitch_width": 68,
        "pitch_length": 101,
        "lat": 53.431,
        "lon": -2.961,
    },
    "Arsenal": {
        "name": "Emirates Stadium",
        "city": "London",
        "capacity": 60704,
        "altitude_m": 41,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.555,
        "lon": -0.109,
    },
    "Man City": {
        "name": "Etihad Stadium",
        "city": "Manchester",
        "capacity": 53400,
        "altitude_m": 44,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.483,
        "lon": -2.200,
    },
    "Chelsea": {
        "name": "Stamford Bridge",
        "city": "London",
        "capacity": 40343,
        "altitude_m": 9,
        "pitch_width": 67,
        "pitch_length": 103,
        "lat": 51.482,
        "lon": -0.191,
    },
    "Tottenham": {
        "name": "Tottenham Hotspur Stadium",
        "city": "London",
        "capacity": 62850,
        "altitude_m": 36,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.604,
        "lon": -0.066,
    },
    "West Ham": {
        "name": "London Stadium",
        "city": "London",
        "capacity": 62500,
        "altitude_m": 5,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.539,
        "lon": -0.017,
    },
    "Newcastle": {
        "name": "St James' Park",
        "city": "Newcastle",
        "capacity": 52305,
        "altitude_m": 56,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 54.976,
        "lon": -1.622,
    },
    "Everton": {
        "name": "Goodison Park",
        "city": "Liverpool",
        "capacity": 39414,
        "altitude_m": 21,
        "pitch_width": 68,
        "pitch_length": 101,
        "lat": 53.439,
        "lon": -2.966,
    },
    "Aston Villa": {
        "name": "Villa Park",
        "city": "Birmingham",
        "capacity": 42657,
        "altitude_m": 143,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.509,
        "lon": -1.885,
    },
    "Bournemouth": {
        "name": "Vitality Stadium",
        "city": "Bournemouth",
        "capacity": 11364,
        "altitude_m": 27,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 50.735,
        "lon": -1.838,
    },
    "Brentford": {
        "name": "Gtech Community Stadium",
        "city": "London",
        "capacity": 17250,
        "altitude_m": 5,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.491,
        "lon": -0.289,
    },
    "Brighton": {
        "name": "Amex Stadium",
        "city": "Brighton",
        "capacity": 31800,
        "altitude_m": 60,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 50.862,
        "lon": -0.084,
    },
    "Crystal Palace": {
        "name": "Selhurst Park",
        "city": "London",
        "capacity": 25486,
        "altitude_m": 53,
        "pitch_width": 68,
        "pitch_length": 101,
        "lat": 51.398,
        "lon": -0.086,
    },
    "Fulham": {
        "name": "Craven Cottage",
        "city": "London",
        "capacity": 25700,
        "altitude_m": 5,
        "pitch_width": 65,
        "pitch_length": 100,
        "lat": 51.475,
        "lon": -0.222,
    },
    "Ipswich": {
        "name": "Portman Road",
        "city": "Ipswich",
        "capacity": 30311,
        "altitude_m": 10,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.055,
        "lon": 1.145,
    },
    "Leicester": {
        "name": "King Power Stadium",
        "city": "Leicester",
        "capacity": 32312,
        "altitude_m": 57,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.620,
        "lon": -1.142,
    },
    "Nottingham Forest": {
        "name": "City Ground",
        "city": "Nottingham",
        "capacity": 30455,
        "altitude_m": 30,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.940,
        "lon": -1.133,
    },
    "Southampton": {
        "name": "St Mary's Stadium",
        "city": "Southampton",
        "capacity": 32384,
        "altitude_m": 3,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 50.906,
        "lon": -1.391,
    },
    "Wolves": {
        "name": "Molineux Stadium",
        "city": "Wolverhampton",
        "capacity": 32050,
        "altitude_m": 165,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.590,
        "lon": -2.130,
    },
    # Recent EPL teams (promoted/relegated in recent seasons)
    "Leeds": {
        "name": "Elland Road",
        "city": "Leeds",
        "capacity": 37890,
        "altitude_m": 48,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.778,
        "lon": -1.572,
    },
    "Burnley": {
        "name": "Turf Moor",
        "city": "Burnley",
        "capacity": 21944,
        "altitude_m": 130,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.789,
        "lon": -2.230,
    },
    "Sheffield United": {
        "name": "Bramall Lane",
        "city": "Sheffield",
        "capacity": 32050,
        "altitude_m": 55,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.370,
        "lon": -1.471,
    },
    "Luton": {
        "name": "Kenilworth Road",
        "city": "Luton",
        "capacity": 10356,
        "altitude_m": 104,
        "pitch_width": 66,
        "pitch_length": 101,
        "lat": 51.884,
        "lon": -0.432,
    },
    "West Brom": {
        "name": "The Hawthorns",
        "city": "West Bromwich",
        "capacity": 26688,
        "altitude_m": 165,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.509,
        "lon": -1.964,
    },
    "Norwich": {
        "name": "Carrow Road",
        "city": "Norwich",
        "capacity": 27244,
        "altitude_m": 5,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.622,
        "lon": 1.309,
    },
    "Watford": {
        "name": "Vicarage Road",
        "city": "Watford",
        "capacity": 22220,
        "altitude_m": 72,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.650,
        "lon": -0.402,
    },
    # Historical EPL teams (promoted/relegated over 2005-2025)
    "Birmingham": {
        "name": "St Andrew's",
        "city": "Birmingham",
        "capacity": 29409,
        "altitude_m": 140,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.476,
        "lon": -1.868,
    },
    "Blackburn": {
        "name": "Ewood Park",
        "city": "Blackburn",
        "capacity": 31367,
        "altitude_m": 90,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.729,
        "lon": -2.489,
    },
    "Blackpool": {
        "name": "Bloomfield Road",
        "city": "Blackpool",
        "capacity": 17338,
        "altitude_m": 5,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.805,
        "lon": -3.048,
    },
    "Bolton": {
        "name": "Reebok Stadium",
        "city": "Bolton",
        "capacity": 28723,
        "altitude_m": 75,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.581,
        "lon": -2.536,
    },
    "Cardiff": {
        "name": "Cardiff City Stadium",
        "city": "Cardiff",
        "capacity": 33280,
        "altitude_m": 15,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.473,
        "lon": -3.203,
    },
    "Charlton": {
        "name": "The Valley",
        "city": "London",
        "capacity": 27111,
        "altitude_m": 5,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.486,
        "lon": 0.036,
    },
    "Derby": {
        "name": "Pride Park",
        "city": "Derby",
        "capacity": 33597,
        "altitude_m": 40,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.915,
        "lon": -1.447,
    },
    "Huddersfield": {
        "name": "John Smith's Stadium",
        "city": "Huddersfield",
        "capacity": 24500,
        "altitude_m": 120,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.654,
        "lon": -1.768,
    },
    "Hull": {
        "name": "MKM Stadium",
        "city": "Hull",
        "capacity": 25586,
        "altitude_m": 2,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.746,
        "lon": -0.368,
    },
    "Middlesbrough": {
        "name": "Riverside Stadium",
        "city": "Middlesbrough",
        "capacity": 34742,
        "altitude_m": 5,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 54.578,
        "lon": -1.217,
    },
    "Portsmouth": {
        "name": "Fratton Park",
        "city": "Portsmouth",
        "capacity": 20688,
        "altitude_m": 3,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 50.796,
        "lon": -1.064,
    },
    "QPR": {
        "name": "Loftus Road",
        "city": "London",
        "capacity": 18439,
        "altitude_m": 25,
        "pitch_width": 64,
        "pitch_length": 102,
        "lat": 51.509,
        "lon": -0.232,
    },
    "Reading": {
        "name": "Madejski Stadium",
        "city": "Reading",
        "capacity": 24161,
        "altitude_m": 40,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.422,
        "lon": -0.983,
    },
    "Stoke": {
        "name": "bet365 Stadium",
        "city": "Stoke-on-Trent",
        "capacity": 30089,
        "altitude_m": 126,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 52.988,
        "lon": -2.176,
    },
    "Sunderland": {
        "name": "Stadium of Light",
        "city": "Sunderland",
        "capacity": 48707,
        "altitude_m": 15,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 54.914,
        "lon": -1.388,
    },
    "Swansea": {
        "name": "Liberty Stadium",
        "city": "Swansea",
        "capacity": 21088,
        "altitude_m": 5,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.642,
        "lon": -3.935,
    },
    "Wigan": {
        "name": "DW Stadium",
        "city": "Wigan",
        "capacity": 25138,
        "altitude_m": 30,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 53.547,
        "lon": -2.654,
    },
}

# La Liga stadiums (5 major venues)
LA_LIGA_STADIUMS = {
    "Real Madrid": {
        "name": "Santiago Bernabeu",
        "city": "Madrid",
        "capacity": 81044,
        "altitude_m": 667,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 40.453,
        "lon": -3.688,
    },
    "Barcelona": {
        "name": "Estadi Olimpic Lluis Companys",
        "city": "Barcelona",
        "capacity": 55926,
        "altitude_m": 173,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 41.365,
        "lon": 2.156,
    },
    "Atletico Madrid": {
        "name": "Civitas Metropolitano",
        "city": "Madrid",
        "capacity": 70460,
        "altitude_m": 603,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 40.436,
        "lon": -3.600,
    },
    "Sevilla": {
        "name": "Ramon Sanchez-Pizjuan",
        "city": "Seville",
        "capacity": 43883,
        "altitude_m": 12,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 37.384,
        "lon": -5.970,
    },
    "Real Betis": {
        "name": "Benito Villamarin",
        "city": "Seville",
        "capacity": 60720,
        "altitude_m": 12,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 37.357,
        "lon": -5.982,
    },
}

# Bundesliga stadiums (5 major venues)
BUNDESLIGA_STADIUMS = {
    "Bayern Munich": {
        "name": "Allianz Arena",
        "city": "Munich",
        "capacity": 75024,
        "altitude_m": 519,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 48.219,
        "lon": 11.625,
    },
    "Dortmund": {
        "name": "Signal Iduna Park",
        "city": "Dortmund",
        "capacity": 81365,
        "altitude_m": 86,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.493,
        "lon": 7.452,
    },
    "Schalke 04": {
        "name": "Veltins-Arena",
        "city": "Gelsenkirchen",
        "capacity": 62271,
        "altitude_m": 58,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.554,
        "lon": 7.068,
    },
    "RB Leipzig": {
        "name": "Red Bull Arena",
        "city": "Leipzig",
        "capacity": 47069,
        "altitude_m": 113,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 51.346,
        "lon": 12.348,
    },
    "Eintracht Frankfurt": {
        "name": "Deutsche Bank Park",
        "city": "Frankfurt",
        "capacity": 51500,
        "altitude_m": 96,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 50.069,
        "lon": 8.645,
    },
}

# Ligue 1 stadiums (5 major venues)
LIGUE_1_STADIUMS = {
    "Paris S-G": {
        "name": "Parc des Princes",
        "city": "Paris",
        "capacity": 47929,
        "altitude_m": 34,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 48.842,
        "lon": 2.253,
    },
    "Marseille": {
        "name": "Stade Velodrome",
        "city": "Marseille",
        "capacity": 67394,
        "altitude_m": 16,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 43.270,
        "lon": 5.396,
    },
    "Lyon": {
        "name": "Groupama Stadium",
        "city": "Lyon",
        "capacity": 59186,
        "altitude_m": 198,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 45.765,
        "lon": 4.982,
    },
    "Monaco": {
        "name": "Stade Louis II",
        "city": "Monaco",
        "capacity": 18523,
        "altitude_m": 1,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 43.727,
        "lon": 7.415,
    },
    "Lille": {
        "name": "Stade Pierre-Mauroy",
        "city": "Lille",
        "capacity": 50157,
        "altitude_m": 22,
        "pitch_width": 68,
        "pitch_length": 105,
        "lat": 50.612,
        "lon": 3.131,
    },
}

# Union of all league stadium dicts — used as the canonical lookup.
ALL_STADIUMS: dict[str, dict] = {
    **SERIE_A_STADIUMS,
    **EPL_STADIUMS,
    **LA_LIGA_STADIUMS,
    **BUNDESLIGA_STADIUMS,
    **LIGUE_1_STADIUMS,
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in km."""
    R = 6371  # Radius of Earth in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def get_travel_distance(home_team: str, away_team: str) -> float:
    """Calculate travel distance for away team in km."""
    home_info = ALL_STADIUMS.get(home_team)
    away_info = ALL_STADIUMS.get(away_team)

    if not home_info or not away_info:
        return 0.0

    return haversine_distance(
        home_info["lat"], home_info["lon"],
        away_info["lat"], away_info["lon"]
    )


def get_altitude_difference(home_team: str, away_team: str) -> float:
    """Calculate altitude difference (home - away) in meters."""
    home_info = ALL_STADIUMS.get(home_team)
    away_info = ALL_STADIUMS.get(away_team)

    if not home_info or not away_info:
        return 0.0

    return home_info["altitude_m"] - away_info["altitude_m"]


def get_stadium_capacity(team: str) -> int:
    """Get stadium capacity for a team."""
    info = ALL_STADIUMS.get(team)
    return info["capacity"] if info else None


def add_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add venue-related features to match DataFrame.

    Features added:
    - travel_distance_km: Away team's travel distance
    - altitude_diff: Altitude difference (home - away)
    - home_stadium_capacity: Home team's stadium capacity
    - capacity_ratio: Attendance / Capacity (atmosphere indicator)
    - is_neutral_venue: Whether played at neutral venue
    - long_travel: Flag for travel > 500km
    - altitude_advantage: Flag for altitude > 200m difference
    """
    df = df.copy()

    log.info("Adding venue features...")

    # Initialize columns
    df["travel_distance_km"] = 0.0
    df["altitude_diff"] = 0.0
    df["home_stadium_capacity"] = np.nan
    df["capacity_ratio"] = 0.0
    df["is_neutral_venue"] = 0
    df["long_travel"] = 0
    df["altitude_advantage"] = 0

    for idx, row in df.iterrows():
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")

        if not home_team or not away_team:
            continue

        # Travel distance
        distance = get_travel_distance(home_team, away_team)
        df.at[idx, "travel_distance_km"] = round(distance, 1)
        df.at[idx, "long_travel"] = 1 if distance > 500 else 0

        # Altitude
        alt_diff = get_altitude_difference(home_team, away_team)
        df.at[idx, "altitude_diff"] = alt_diff
        df.at[idx, "altitude_advantage"] = 1 if alt_diff > 200 else 0

        # Capacity
        capacity = get_stadium_capacity(home_team)
        df.at[idx, "home_stadium_capacity"] = capacity

        # Capacity ratio (atmosphere)
        attendance = row.get("attendance", 0)
        if pd.notna(attendance) and capacity is not None and capacity > 0:
            attendance_val = pd.to_numeric(attendance, errors="coerce")
            if pd.notna(attendance_val) and attendance_val > 0:
                df.at[idx, "capacity_ratio"] = round(min(1.0, attendance_val / capacity), 2)

    # Normalize travel distance (0-1 scale, max ~1500km across top-5 leagues)
    df["travel_distance_norm"] = df["travel_distance_km"] / 1500

    n_long_travel = (df["long_travel"] > 0).sum()
    n_altitude = (df["altitude_advantage"] > 0).sum()

    log.info(f"Added venue features: {n_long_travel} long travels, {n_altitude} altitude advantages")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing venue module...")
    print("=" * 60)

    # Test travel distances
    print("\nTravel Distance Examples:")
    test_matches = [
        ("Napoli", "Inter"),  # Long distance
        ("Inter", "Milan"),  # Same city
        ("Lecce", "Napoli"),  # Southern teams
        ("Atalanta", "Napoli"),  # Altitude + distance
    ]

    for home, away in test_matches:
        dist = get_travel_distance(home, away)
        alt = get_altitude_difference(home, away)
        print(f"  {away} @ {home}: {dist:.0f} km, altitude diff: {alt:+.0f} m")

    # Stadium capacities
    print("\nStadium Capacities (Top 10):")
    capacities = [(team, info["capacity"]) for team, info in ALL_STADIUMS.items()]
    for team, cap in sorted(capacities, key=lambda x: -x[1])[:10]:
        print(f"  {team}: {cap:,}")
