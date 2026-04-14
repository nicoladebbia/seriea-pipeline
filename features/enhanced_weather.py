"""Consolidated Weather Features.

Combines basic weather impact scoring (formerly in weather.py) with
extended weather intelligence (Phase 4) into a single module.

=== BASIC WEATHER IMPACT (from former weather.py) ===
  - adverse_weather_score: overall bad-weather indicator
  - possession_handicap: impact on possession-based teams
  - direct_play_boost: benefit for direct-play teams
  - aerial_handicap: wind impact on aerial play
  - cold/hot/rain/wind condition flags
  - City-based weather estimation fallback

=== ENHANCED WEATHER (Phase 4) ===
  1. HUMIDITY EFFECT
     - High humidity affects ball control
     - Impacts passing accuracy and ball movement

  2. ALTITUDE EFFECT
     - Stadium altitude differences
     - Away teams less accustomed to conditions

  3. TIME OF DAY EFFECT
     - Night games vs day games
     - Floodlight conditions, atmosphere

  4. SEASONAL PATTERNS
     - Early season heat effect
     - Winter cold effect
     - Late season optimal conditions

  5. COMBINED WEATHER EFFECT
     - Weather goals effect, home boost
     - Full integration of all sub-effects
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional
import math

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# =============================================================================
# STADIUM DATA
# =============================================================================

# Stadium altitudes (meters above sea level)
STADIUM_ALTITUDES = {
    # ── Serie A (Italy) ──────────────────────────────────────────────
    "Inter": 120,
    "Milan": 120,
    "Juventus": 239,
    "Torino": 250,
    "Roma": 40,
    "Lazio": 40,
    "Napoli": 60,
    "Fiorentina": 50,
    "Atalanta": 250,
    "Bologna": 54,
    "Genoa": 10,
    "Sampdoria": 10,
    "Verona": 65,
    "Udinese": 113,
    "Sassuolo": 58,
    "Empoli": 28,
    "Cagliari": 10,
    "Lecce": 50,
    "Monza": 162,
    "Parma": 55,
    "Como": 200,
    "Venezia": 1,
    "Spezia": 10,
    "Cremonese": 45,
    "Frosinone": 291,
    "Pisa": 4,
    # ── Premier League (England) ─────────────────────────────────────
    # English stadiums are mostly at low altitude (10-100m)
    "Arsenal": 25,
    "Aston Villa": 100,
    "Bournemouth": 10,
    "Brentford": 15,
    "Brighton": 50,
    "Burnley": 190,
    "Chelsea": 10,
    "Crystal Palace": 55,
    "Everton": 10,
    "Fulham": 10,
    "Ipswich": 20,
    "Leeds": 50,
    "Leicester": 65,
    "Liverpool": 30,
    "Man City": 55,
    "Man United": 40,
    "Newcastle": 65,
    "Nottingham Forest": 30,
    "Sheffield United": 75,
    "Southampton": 15,
    "Tottenham": 30,
    "West Ham": 10,
    "Wolves": 160,
    "Luton": 130,
    "Sunderland": 25,
    "Watford": 70,
    "Norwich": 10,
    "West Brom": 170,
    "Huddersfield": 110,
    "Swansea": 10,
    "Stoke": 120,
    "Middlesbrough": 10,
}

# City classification for weather estimation.
# Italian cities use North/Central/South temperature profiles.
# English cities use a separate UK profile (see estimate_weather_from_month_location).
NORTHERN_CITIES = {
    "Milan", "Turin", "Bergamo", "Udine", "Venice", "Verona",
    "Genoa", "Como", "Monza", "Reggio Emilia",
}
CENTRAL_CITIES = {"Rome", "Florence", "Bologna", "Empoli"}
SOUTHERN_CITIES = {"Naples", "Lecce", "Cagliari"}
UK_CITIES = {
    "London", "Manchester", "Liverpool", "Birmingham", "Leeds",
    "Newcastle", "Brighton", "Southampton", "Wolverhampton",
    "Leicester", "Nottingham", "Bournemouth", "Burnley", "Sheffield",
    "Luton", "Brentford", "Ipswich", "Sunderland", "Fulham",
    "West Ham", "Stoke", "Watford", "Huddersfield", "Norwich",
    "Swansea", "West Bromwich", "Middlesbrough",
}

# Team-to-city mapping for weather estimation
TEAM_CITIES = {
    # ── Serie A (Italy) ──────────────────────────────────────────────
    "Inter": "Milan", "Milan": "Milan", "Monza": "Monza",
    "Juventus": "Turin", "Torino": "Turin",
    "Roma": "Rome", "Lazio": "Rome",
    "Napoli": "Naples",
    "Fiorentina": "Florence",
    "Atalanta": "Bergamo",
    "Bologna": "Bologna",
    "Genoa": "Genoa", "Sampdoria": "Genoa",
    "Verona": "Verona",
    "Udinese": "Udine",
    "Sassuolo": "Reggio Emilia",
    "Empoli": "Empoli",
    "Cagliari": "Cagliari",
    "Lecce": "Lecce",
    "Parma": "Parma",
    "Como": "Como",
    "Venezia": "Venice",
    "Spezia": "Spezia",
    "Cremonese": "Cremona",
    "Frosinone": "Frosinone",
    "Pisa": "Pisa",
    # ── Premier League (England) ─────────────────────────────────────
    "Arsenal": "London", "Chelsea": "London", "Tottenham": "London",
    "West Ham": "London", "Crystal Palace": "London", "Fulham": "London",
    "Brentford": "Brentford",
    "Man United": "Manchester", "Man City": "Manchester",
    "Liverpool": "Liverpool", "Everton": "Liverpool",
    "Aston Villa": "Birmingham",
    "Leeds": "Leeds",
    "Newcastle": "Newcastle", "Sunderland": "Sunderland",
    "Brighton": "Brighton",
    "Southampton": "Southampton",
    "Wolves": "Wolverhampton", "West Brom": "West Bromwich",
    "Leicester": "Leicester",
    "Nottingham Forest": "Nottingham",
    "Bournemouth": "Bournemouth",
    "Burnley": "Burnley",
    "Sheffield United": "Sheffield",
    "Luton": "Luton",
    "Ipswich": "Ipswich",
    "Watford": "Watford",
    "Norwich": "Norwich",
    "Stoke": "Stoke",
    "Huddersfield": "Huddersfield",
    "Swansea": "Swansea",
    "Middlesbrough": "Middlesbrough",
}


# =============================================================================
# BASIC WEATHER IMPACT SCORING (from former weather.py)
# =============================================================================

def get_weather_impact_score(
    temperature_c: float,
    precipitation_mm: float,
    wind_speed_kmh: float,
    humidity_pct: float,
) -> dict[str, float]:
    """Calculate weather impact scores on different play styles.

    Args:
        temperature_c: Temperature in Celsius
        precipitation_mm: Precipitation in mm
        wind_speed_kmh: Wind speed in km/h
        humidity_pct: Humidity percentage

    Returns:
        Dictionary of weather impact features
    """
    features = {}

    # Temperature impact
    # Extreme cold/heat affects stamina and technique
    if temperature_c < 5:
        temp_impact = 0.2  # Cold conditions
        features["cold_conditions"] = 1
        features["hot_conditions"] = 0
    elif temperature_c > 30:
        temp_impact = 0.3  # Hot conditions, higher impact
        features["cold_conditions"] = 0
        features["hot_conditions"] = 1
    else:
        temp_impact = 0.0  # Comfortable
        features["cold_conditions"] = 0
        features["hot_conditions"] = 0

    # Rain impact
    # Affects passing accuracy and ball control
    if precipitation_mm > 5:
        rain_impact = 0.3  # Heavy rain
        features["heavy_rain"] = 1
        features["light_rain"] = 0
    elif precipitation_mm > 1:
        rain_impact = 0.15  # Light rain
        features["heavy_rain"] = 0
        features["light_rain"] = 1
    else:
        rain_impact = 0.0  # Dry
        features["heavy_rain"] = 0
        features["light_rain"] = 0

    # Wind impact
    # Affects long balls and aerial play
    if wind_speed_kmh > 30:
        wind_impact = 0.25  # Strong wind
        features["strong_wind"] = 1
    elif wind_speed_kmh > 15:
        wind_impact = 0.1  # Moderate wind
        features["strong_wind"] = 0
    else:
        wind_impact = 0.0  # Calm
        features["strong_wind"] = 0

    # Humidity impact (less significant)
    humidity_impact = max(0, (humidity_pct - 70) / 100) * 0.1

    # Overall adverse weather score
    features["adverse_weather_score"] = round(
        temp_impact + rain_impact + wind_impact + humidity_impact, 2
    )

    # Style-specific impacts
    # Possession teams suffer more in bad weather
    features["possession_handicap"] = round(rain_impact + wind_impact * 0.5, 2)

    # Direct teams may benefit
    features["direct_play_boost"] = round(rain_impact * 0.5, 2)

    # Aerial play affected by wind
    features["aerial_handicap"] = round(wind_impact, 2)

    return features


def estimate_weather_from_month_location(
    month: int,
    city: str,
    hour: Optional[int] = None,
) -> dict[str, float]:
    """Estimate typical weather conditions based on month and location.

    This is a fallback when real weather data isn't available.
    Uses historical averages for Italian and English cities.
    """
    # Monthly temperature averages (simplified)
    temp_north = {1: 2, 2: 4, 3: 9, 4: 14, 5: 18, 6: 23, 7: 25, 8: 24, 9: 20, 10: 14, 11: 8, 12: 3}
    temp_central = {1: 7, 2: 8, 3: 11, 4: 14, 5: 19, 6: 23, 7: 26, 8: 26, 9: 22, 10: 17, 11: 12, 12: 8}
    temp_south = {1: 10, 2: 11, 3: 13, 4: 16, 5: 20, 6: 25, 7: 28, 8: 28, 9: 25, 10: 20, 11: 15, 12: 11}
    # UK: mild but wet; narrower temperature range than Italy
    temp_uk = {1: 5, 2: 5, 3: 7, 4: 10, 5: 13, 6: 16, 7: 18, 8: 18, 9: 15, 10: 11, 11: 8, 12: 5}

    # Monthly rain probability
    rain_prob_it = {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.4, 5: 0.35, 6: 0.2, 7: 0.15, 8: 0.2, 9: 0.3, 10: 0.4, 11: 0.5, 12: 0.35}
    # UK gets more frequent rain year-round
    rain_prob_uk = {1: 0.5, 2: 0.45, 3: 0.45, 4: 0.45, 5: 0.4, 6: 0.35, 7: 0.35, 8: 0.4, 9: 0.4, 10: 0.5, 11: 0.55, 12: 0.5}

    # Determine region
    is_uk = city in UK_CITIES
    if is_uk:
        temp = temp_uk.get(month, 10)
        rain_prob = rain_prob_uk
    elif city in NORTHERN_CITIES:
        temp = temp_north.get(month, 15)
        rain_prob = rain_prob_it
    elif city in CENTRAL_CITIES:
        temp = temp_central.get(month, 15)
        rain_prob = rain_prob_it
    elif city in SOUTHERN_CITIES:
        temp = temp_south.get(month, 18)
        rain_prob = rain_prob_it
    else:
        temp = 15  # Default
        rain_prob = rain_prob_it

    # Estimated values
    precipitation = 2 if rain_prob.get(month, 0.3) > 0.35 else 0
    # UK is generally windier than Italy
    if is_uk:
        wind = 20 if month in [1, 2, 3, 10, 11, 12] else 15
        humidity = 80 if month in [10, 11, 12, 1, 2] else 70
    else:
        wind = 15 if month in [1, 2, 3, 11, 12] else 10
        humidity = 70 if month in [10, 11, 12, 1, 2] else 55

    return {
        "estimated_temp_c": temp,
        "estimated_precip_mm": precipitation,
        "estimated_wind_kmh": wind,
        "estimated_humidity_pct": humidity,
    }


# =============================================================================
# ENHANCED WEATHER EFFECTS (Phase 4)
# =============================================================================

def compute_humidity_effect(humidity: float, temperature: float) -> Dict[str, float]:
    """Compute humidity impact on match.

    High humidity (>70%) + heat = reduced performance
    Low humidity (<30%) + cold = ball moves faster
    """
    if humidity is None:
        return {"humidity_effect": 0.0, "ball_control_penalty": 0.0}

    # Base effect
    effect = 0.0

    # High humidity penalty (especially with heat)
    if humidity > 70:
        effect -= 0.1  # Slight disadvantage for technical play
        if temperature and temperature > 25:
            effect -= 0.1  # Combined effect worse

    # Very high humidity (tropical conditions)
    if humidity > 85:
        effect -= 0.15

    # Low humidity can help ball movement
    if humidity < 40:
        effect += 0.05

    # Ball control penalty for high humidity
    ball_penalty = max(0, (humidity - 60) / 100) if humidity else 0

    return {
        "humidity_effect": effect,
        "ball_control_penalty": ball_penalty,
        "is_high_humidity": 1.0 if humidity and humidity > 70 else 0.0,
    }


def compute_altitude_effect(home_team: str, away_team: str) -> Dict[str, float]:
    """Compute altitude advantage/disadvantage.

    Away teams may struggle at higher altitudes if they're
    accustomed to sea level.
    """
    home_alt = STADIUM_ALTITUDES.get(home_team, 100)
    away_alt = STADIUM_ALTITUDES.get(away_team, 100)

    # Altitude difference (positive = home is higher)
    alt_diff = home_alt - away_alt

    # Effect is small but measurable above 150m difference
    effect = 0.0
    if alt_diff > 250:
        effect = 0.04
    elif alt_diff > 150:
        effect = 0.02  # Slight home advantage

    # Reverse for away altitude advantage
    if alt_diff < -250:
        effect = -0.04
    elif alt_diff < -150:
        effect = -0.02

    return {
        "altitude_diff": alt_diff,
        "altitude_effect": effect,
        "home_altitude": home_alt,
        "away_altitude": away_alt,
    }


def compute_time_of_day_effect(match_time: str, match_date: str = None) -> Dict[str, float]:
    """Compute time of day effects.

    Night games (after 20:00):
    - Floodlight conditions
    - Different atmosphere (louder fans)
    - Slightly favors home team

    Early games (12:30):
    - Teams may be less prepared
    - Less fan atmosphere

    Prime time (18:00-20:00):
    - Optimal conditions
    - Best atmosphere
    """
    try:
        hour = int(match_time.split(":")[0])
    except (ValueError, AttributeError):
        hour = 15  # Default to afternoon

    effect = 0.0
    is_night = 0.0
    is_early = 0.0

    # Night game (after 20:00)
    if hour >= 20:
        effect = 0.02  # Slight home boost from atmosphere
        is_night = 1.0

    # Early kickoff (before 14:00)
    elif hour < 14:
        effect = -0.01  # Slightly worse conditions
        is_early = 1.0

    # Late afternoon (18:00-20:00) - optimal
    elif 18 <= hour < 20:
        effect = 0.01

    # Seasonal adjustment
    if match_date:
        try:
            date_obj = datetime.strptime(match_date, "%Y-%m-%d")
            month = date_obj.month

            # Summer (June-August) - evening preferred
            if month in [6, 7, 8] and hour >= 20:
                effect += 0.01  # Better than hot afternoon

            # Winter (December-February) - earlier preferred
            if month in [12, 1, 2] and hour < 18:
                effect += 0.01  # Better than cold night
        except ValueError as e:
            log.debug(f"Failed to parse kickoff time for weather effect: {e}")

    return {
        "time_effect": effect,
        "is_night_game": is_night,
        "is_early_game": is_early,
        "kickoff_hour": hour,
    }


def compute_seasonal_effect(match_date: str) -> Dict[str, float]:
    """Compute seasonal effects on match.

    Early season (August-September):
    - Teams still finding form
    - Hot weather in some venues
    - More goals typically

    Mid-season (October-December):
    - Peak form for most teams
    - Weather starts to affect

    Winter (January-February):
    - Cold affects some teams
    - More physical play
    - Fewer goals typically

    Late season (March-May):
    - Motivation varies (title race vs nothing to play for)
    - Better weather returning
    """
    try:
        date_obj = datetime.strptime(match_date, "%Y-%m-%d")
        month = date_obj.month
    except (ValueError, TypeError):
        return {
            "seasonal_effect": 0.0,
            "is_early_season": 0.0,
            "is_winter": 0.0,
            "is_late_season": 0.0,
        }

    effect = 0.0
    is_early = 0.0
    is_winter = 0.0
    is_late = 0.0

    # Early season (August-September)
    if month in [8, 9]:
        effect = 0.05  # More goals, less defensive structure
        is_early = 1.0

    # Winter months (December-February)
    elif month in [12, 1, 2]:
        effect = -0.03  # Fewer goals, more physical
        is_winter = 1.0

    # Late season (April-May)
    elif month in [4, 5]:
        is_late = 1.0
        # Could add motivation factor here

    return {
        "seasonal_effect": effect,
        "is_early_season": is_early,
        "is_winter": is_winter,
        "is_late_season": is_late,
        "month": month,
    }


def compute_combined_weather_effect(
    weather_data: Dict,
    home_team: str,
    away_team: str,
    match_time: str = "15:00",
    match_date: str = None,
) -> Dict[str, float]:
    """Compute all weather effects combined.

    Args:
        weather_data: Weather dict with temperature, precipitation, wind_speed, humidity
        home_team: Home team name
        away_team: Away team name
        match_time: Kickoff time (HH:MM)
        match_date: Match date (YYYY-MM-DD)

    Returns:
        Dict with all weather effect features
    """
    result = {}

    # Basic weather effects
    temperature = weather_data.get("temperature")
    precipitation = weather_data.get("precipitation", 0)
    wind_speed = weather_data.get("wind_speed", 0)
    humidity = weather_data.get("humidity")

    # Temperature effects
    if temperature is not None:
        result["temperature"] = temperature
        result["is_cold"] = 1.0 if temperature < 10 else 0.0
        result["is_hot"] = 1.0 if temperature > 25 else 0.0
        result["is_extreme"] = 1.0 if temperature < 5 or temperature > 30 else 0.0

    # Precipitation effects
    result["precipitation"] = precipitation or 0
    result["is_rainy"] = 1.0 if precipitation and precipitation > 5 else 0.0
    result["is_heavy_rain"] = 1.0 if precipitation and precipitation > 15 else 0.0

    # Wind effects
    result["wind_speed"] = wind_speed or 0
    result["is_windy"] = 1.0 if wind_speed and wind_speed > 30 else 0.0
    result["is_very_windy"] = 1.0 if wind_speed and wind_speed > 50 else 0.0

    # Humidity effects
    humidity_effect = compute_humidity_effect(humidity, temperature)
    result.update(humidity_effect)

    # Altitude effects
    altitude_effect = compute_altitude_effect(home_team, away_team)
    result.update(altitude_effect)

    # Time of day effects
    time_effect = compute_time_of_day_effect(match_time, match_date)
    result.update(time_effect)

    # Seasonal effects
    if match_date:
        seasonal_effect = compute_seasonal_effect(match_date)
        result.update(seasonal_effect)

    # Style-specific impact scoring (from basic weather module)
    if temperature is not None and humidity is not None:
        style_impact = get_weather_impact_score(
            temperature, precipitation or 0, wind_speed or 0, humidity
        )
        # Add style-specific scores that aren't already covered
        result["adverse_weather_score"] = style_impact["adverse_weather_score"]
        result["possession_handicap"] = style_impact["possession_handicap"]
        result["direct_play_boost"] = style_impact["direct_play_boost"]
        result["aerial_handicap"] = style_impact["aerial_handicap"]

    # Combined impact score
    total_effect = 0.0

    # Weather impacts on goals
    if result.get("is_rainy"):
        total_effect += 0.15  # Rain increases goals slightly
    if result.get("is_windy"):
        total_effect -= 0.20  # Wind reduces goals
    if result.get("is_cold"):
        total_effect -= 0.05

    # Combined home advantage effect
    home_boost = (
        altitude_effect.get("altitude_effect", 0) +
        time_effect.get("time_effect", 0)
    )

    result["weather_goals_effect"] = total_effect
    result["weather_home_boost"] = home_boost

    return result


def get_enhanced_weather_features(
    home_team: str,
    away_team: str,
    match_date: str = None,
    match_time: str = "15:00",
    weather_data: Dict = None,
) -> Dict[str, float]:
    """Get all enhanced weather features for a match.

    If weather_data is not provided, returns only
    altitude, time, and seasonal effects.
    """
    if weather_data is None:
        weather_data = {}

    return compute_combined_weather_effect(
        weather_data=weather_data,
        home_team=home_team,
        away_team=away_team,
        match_time=match_time,
        match_date=match_date,
    )


# =============================================================================
# DATAFRAME-LEVEL ENTRY POINTS
# =============================================================================

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic weather-related features to match DataFrame.

    If real weather data isn't available, uses estimated values
    based on month, location, and time of year.

    Features added:
    - adverse_weather_score: Overall bad weather indicator
    - possession_handicap: Impact on possession teams
    - direct_play_boost: Benefit for direct teams
    - cold_conditions, hot_conditions: Temperature flags
    - heavy_rain, light_rain: Precipitation flags
    - strong_wind: Wind flag
    """
    df = df.copy()

    log.info("Adding weather features...")

    # Check if we have actual weather data
    has_weather = "temperature" in df.columns or "weather_temp" in df.columns

    # Initialize columns
    weather_cols = [
        "adverse_weather_score", "possession_handicap", "direct_play_boost",
        "aerial_handicap", "cold_conditions", "hot_conditions",
        "heavy_rain", "light_rain", "strong_wind",
    ]

    for col in weather_cols:
        df[col] = 0.0

    if has_weather:
        # Use actual weather data
        log.info("Using actual weather data")
        temp_col = "temperature" if "temperature" in df.columns else "weather_temp"
        precip_col = "precipitation" if "precipitation" in df.columns else "weather_precip"
        wind_col = "wind_speed" if "wind_speed" in df.columns else "weather_wind"
        humid_col = "humidity" if "humidity" in df.columns else "weather_humidity"

        for idx, row in df.iterrows():
            temp = pd.to_numeric(row.get(temp_col, 15), errors="coerce")
            precip = pd.to_numeric(row.get(precip_col, 0), errors="coerce")
            wind = pd.to_numeric(row.get(wind_col, 10), errors="coerce")
            humid = pd.to_numeric(row.get(humid_col, 60), errors="coerce")

            if pd.isna(temp):
                temp = 15
            if pd.isna(precip):
                precip = 0
            if pd.isna(wind):
                wind = 10
            if pd.isna(humid):
                humid = 60

            impact = get_weather_impact_score(temp, precip, wind, humid)

            for col, val in impact.items():
                if col in df.columns:
                    df.at[idx, col] = val
    else:
        # Estimate from date and location
        log.info("Estimating weather from date and location")

        # Use our own team-city mapping, with fallback to venue module
        try:
            from features.venue import SERIE_A_STADIUMS
        except ImportError:
            SERIE_A_STADIUMS = {}

        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

        for idx, row in df.iterrows():
            match_date = row["match_date"]
            home_team = row.get("home_team", "")

            if pd.isna(match_date):
                continue

            month = match_date.month

            # Get city: prefer our mapping, fall back to stadium data
            city = TEAM_CITIES.get(home_team)
            if city is None:
                stadium_info = SERIE_A_STADIUMS.get(home_team, {})
                city = stadium_info.get("city", "Rome")  # Default to Rome

            # Get estimated weather
            est_weather = estimate_weather_from_month_location(month, city)

            # Calculate impact
            impact = get_weather_impact_score(
                est_weather["estimated_temp_c"],
                est_weather["estimated_precip_mm"],
                est_weather["estimated_wind_kmh"],
                est_weather["estimated_humidity_pct"],
            )

            for col, val in impact.items():
                if col in df.columns:
                    df.at[idx, col] = val

    # Count matches with adverse weather
    n_adverse = (df["adverse_weather_score"] > 0.1).sum()
    n_rain = ((df["heavy_rain"] > 0) | (df["light_rain"] > 0)).sum()

    log.info(f"Added weather features: {n_adverse} adverse weather, {n_rain} rain matches")

    return df


def add_all_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ALL weather features (basic + enhanced) in one call.

    This is the recommended single entry point for the build pipeline.

    Applies:
    1. Basic weather impact scoring (adverse_weather_score, possession_handicap, etc.)
    2. Enhanced features (altitude, time-of-day, seasonal, humidity effects)

    Parameters
    ----------
    df : pd.DataFrame
        Match-level DataFrame with home_team, away_team, match_date columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with all weather columns appended.
    """
    log.info("Adding all weather features (basic + enhanced)...")

    # Step 1: Basic weather features (adverse score, style impacts, condition flags)
    result = add_weather_features(df)

    # Step 2: Enhanced features (altitude, time-of-day, seasonal)
    result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")

    # Altitude features
    alt_cols_added = 0
    alt_diffs = []
    alt_effects = []
    home_alts = []
    away_alts = []

    for _, row in result.iterrows():
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")
        alt = compute_altitude_effect(home_team, away_team)
        alt_diffs.append(alt["altitude_diff"])
        alt_effects.append(alt["altitude_effect"])
        home_alts.append(alt["home_altitude"])
        away_alts.append(alt["away_altitude"])

    result["altitude_diff"] = alt_diffs
    result["altitude_effect"] = alt_effects
    result["home_altitude"] = home_alts
    result["away_altitude"] = away_alts
    alt_cols_added = 4

    # Time-of-day features
    time_cols_added = 0
    if "kickoff_time" in result.columns:
        time_effects = []
        is_night_games = []
        is_early_games = []

        for _, row in result.iterrows():
            kt = row.get("kickoff_time", "15:00")
            if pd.isna(kt):
                kt = "15:00"
            md = None
            if pd.notna(row.get("match_date")):
                md = str(row["match_date"])[:10]
            tod = compute_time_of_day_effect(str(kt), md)
            time_effects.append(tod["time_effect"])
            is_night_games.append(tod["is_night_game"])
            is_early_games.append(tod["is_early_game"])

        result["time_effect"] = time_effects
        result["is_night_game"] = is_night_games
        result["is_early_game"] = is_early_games
        time_cols_added = 3

    # Seasonal features
    seasonal_effects = []
    is_early_seasons = []
    is_winters = []
    is_late_seasons = []

    for _, row in result.iterrows():
        md = row.get("match_date")
        if pd.notna(md):
            md_str = str(md)[:10]
            seasonal = compute_seasonal_effect(md_str)
        else:
            seasonal = {
                "seasonal_effect": 0.0,
                "is_early_season": 0.0,
                "is_winter": 0.0,
                "is_late_season": 0.0,
            }
        seasonal_effects.append(seasonal["seasonal_effect"])
        is_early_seasons.append(seasonal["is_early_season"])
        is_winters.append(seasonal["is_winter"])
        is_late_seasons.append(seasonal["is_late_season"])

    result["seasonal_effect"] = seasonal_effects
    result["weather_is_early_season"] = is_early_seasons
    result["weather_is_winter"] = is_winters
    result["weather_is_late_season"] = is_late_seasons
    seasonal_cols_added = 4

    total_enhanced = alt_cols_added + time_cols_added + seasonal_cols_added
    log.info(
        f"Added {total_enhanced} enhanced weather features "
        f"(altitude: {alt_cols_added}, time: {time_cols_added}, seasonal: {seasonal_cols_added})"
    )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Consolidated Weather Features")
    print("=" * 60)

    # Test basic weather impact
    print("\nWeather Impact Examples:")

    scenarios = [
        ("Perfect conditions", 18, 0, 10, 55),
        ("Cold winter match", 2, 0, 20, 75),
        ("Rainy evening", 12, 8, 15, 85),
        ("Hot summer day", 32, 0, 5, 50),
        ("Stormy conditions", 10, 15, 40, 90),
    ]

    for name, temp, precip, wind, humid in scenarios:
        impact = get_weather_impact_score(temp, precip, wind, humid)
        print(f"\n  {name} ({temp}C, {precip}mm rain, {wind}km/h wind):")
        print(f"    Adverse score: {impact['adverse_weather_score']}")
        print(f"    Possession handicap: {impact['possession_handicap']}")
        print(f"    Direct play boost: {impact['direct_play_boost']}")

    # Test enhanced features
    print("\n" + "=" * 60)
    print("\nEnhanced Weather Features (Atalanta vs Napoli):")

    weather = {
        "temperature": 8,
        "precipitation": 12,
        "wind_speed": 25,
        "humidity": 75,
    }

    features = get_enhanced_weather_features(
        home_team="Atalanta",
        away_team="Napoli",
        match_date="2026-01-15",
        match_time="20:45",
        weather_data=weather,
    )

    for key, value in sorted(features.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Test estimation
    print("\n" + "=" * 60)
    print("\nEstimated Weather by Month (Milan):")
    for month in [1, 4, 7, 10]:
        est = estimate_weather_from_month_location(month, "Milan")
        print(f"  Month {month}: {est['estimated_temp_c']}C, {est['estimated_precip_mm']}mm rain")
