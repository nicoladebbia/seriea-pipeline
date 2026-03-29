"""Fetch historical weather data for match venues using Open-Meteo.

Open-Meteo provides free historical weather data (1940–present) with no API key.
Endpoint: https://archive-api.open-meteo.com/v1/archive

We map each stadium venue to its city's latitude/longitude, then fetch
temperature, precipitation, wind, and humidity for the match date.

Supports: Serie A (Italy), Premier League (England).
"""

from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd
import requests

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

WEATHER_CACHE_PATH = DATA_DIR / "external" / "weather.parquet"

# City coordinates (lat, lon) for match venues.
# Keyed by city name as it appears after the comma in venue strings
# (e.g. "Stadio Olimpico, Roma" → "Roma").
VENUE_COORDS: dict[str, tuple[float, float]] = {
    # ── Serie A (Italy) ──────────────────────────────────────────────
    "Roma": (41.8967, 12.4822),
    "Milano": (45.4781, 9.1240),
    "Bergamo": (45.7094, 9.6808),
    "Udine": (46.0620, 13.2396),
    "Torino": (45.0416, 7.6500),
    "Parma": (44.7935, 10.3282),
    "Napoli": (40.8279, 14.1931),
    "Monza": (45.5845, 9.2744),
    "Lecce": (40.3526, 18.1759),
    "Bologna": (44.4939, 11.3097),
    "Verona": (45.4353, 10.9686),
    "Genova": (44.4159, 8.9525),
    "Firenze": (43.7808, 11.2824),
    "Empoli": (43.7237, 10.9537),
    "Como": (45.8111, 9.0847),
    "Cagliari": (39.2111, 9.1092),
    "Venezia": (45.4387, 12.3254),
    # Historical Serie A venues (promoted/relegated teams)
    "Sassuolo": (44.5349, 10.7924),
    "Salerno": (40.6831, 14.7997),
    "Benevento": (41.1297, 14.7826),
    "Crotone": (39.0843, 17.1175),
    "Cremona": (45.1333, 10.0167),
    "Frosinone": (41.6403, 13.3490),
    "La Spezia": (44.1025, 9.8241),
    "Ferrara": (44.8381, 11.6198),
    "Brescia": (45.5416, 10.2118),
    "Pisa": (43.7228, 10.3966),
    # ── Premier League (England) ─────────────────────────────────────
    "London": (51.5074, -0.1278),
    "Manchester": (53.4631, -2.2913),
    "Liverpool": (53.4308, -2.9608),
    "Birmingham": (52.5090, -1.8846),
    "Leeds": (53.7774, -1.5721),
    "Newcastle": (54.9756, -1.6143),
    "Brighton": (50.8615, -0.0833),
    "Southampton": (50.9058, -1.3911),
    "Wolverhampton": (52.5903, -2.1306),
    "Leicester": (52.6204, -1.1254),
    "Nottingham": (52.9400, -1.1329),
    "Bournemouth": (50.7352, -1.8388),
    "Burnley": (53.7890, -2.2301),
    "Sheffield": (53.3703, -1.4726),
    "Luton": (51.8843, -0.4316),
    "Brentford": (51.4907, -0.2888),
    "Ipswich": (52.0547, 1.1448),
    "Sunderland": (54.9146, -1.3882),
    "Fulham": (51.4749, -0.2217),
    "West Ham": (51.5387, 0.0166),
    "Stoke": (52.9884, -2.1756),
    "Watford": (51.6500, -0.4014),
    "Huddersfield": (53.6544, -1.7684),
    "Norwich": (52.6222, 1.3090),
    "Swansea": (51.6427, -3.9353),
    "West Bromwich": (52.5090, -1.9640),
    "Middlesbrough": (54.5782, -1.2169),
}

# Open-Meteo archive endpoint
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# City → timezone mapping.  Default is Europe/Rome (Serie A).
# English cities get Europe/London.
_CITY_TIMEZONE: dict[str, str] = {}
_UK_CITIES = {
    "London", "Manchester", "Liverpool", "Birmingham", "Leeds",
    "Newcastle", "Brighton", "Southampton", "Wolverhampton", "Leicester",
    "Nottingham", "Bournemouth", "Burnley", "Sheffield", "Luton",
    "Brentford", "Ipswich", "Sunderland", "Fulham", "West Ham",
    "Stoke", "Watford", "Huddersfield", "Norwich", "Swansea",
    "West Bromwich", "Middlesbrough",
}
for _c in _UK_CITIES:
    _CITY_TIMEZONE[_c] = "Europe/London"

# Weather variables to fetch (daily aggregates)
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",  # Available in newer API versions
]


def _extract_city(venue: str) -> str | None:
    """Extract the city name from a venue string like 'Stadio Olimpico, Roma'."""
    if not venue or pd.isna(venue):
        return None
    parts = venue.split(",")
    if len(parts) >= 2:
        return parts[-1].strip()
    return None


def _get_coords(city: str | None) -> tuple[float, float] | None:
    """Look up coordinates for a city."""
    if not city:
        return None
    return VENUE_COORDS.get(city)


def fetch_weather_for_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Fetch weather data for all matches.

    Args:
        matches: DataFrame with columns 'match_id', 'match_date', 'venue'

    Returns:
        DataFrame with match_id + weather columns
    """
    # Load cache if exists
    if WEATHER_CACHE_PATH.exists():
        cached = pd.read_parquet(WEATHER_CACHE_PATH)
        cached_ids = set(cached["match_id"])
    else:
        cached = pd.DataFrame()
        cached_ids = set()

    # Find matches that need weather data
    needed = matches[~matches["match_id"].isin(cached_ids)].copy()
    if needed.empty:
        log.info("All weather data already cached (%d matches)", len(cached))
        return cached

    log.info("Fetching weather for %d new matches", len(needed))

    rows = []
    # Group by (city, date) to batch requests
    needed["_city"] = needed["venue"].apply(_extract_city)
    needed["_date"] = pd.to_datetime(needed["match_date"], errors="coerce").dt.date

    # Deduplicate city+date pairs
    city_date_pairs = needed[["_city", "_date"]].drop_duplicates()
    weather_cache: dict[tuple, dict] = {}

    for _, pair in city_date_pairs.iterrows():
        city = pair["_city"]
        match_date = pair["_date"]
        coords = _get_coords(city)
        if coords is None or match_date is None:
            continue

        key = (city, str(match_date))
        if key in weather_cache:
            continue

        tz = _CITY_TIMEZONE.get(city, "Europe/Rome")
        weather = _fetch_single_day(coords[0], coords[1], match_date, timezone=tz)
        if weather:
            weather_cache[key] = weather
        time.sleep(0.3)  # Be respectful

    # Map weather back to matches
    for _, row in needed.iterrows():
        city = row["_city"]
        match_date = row["_date"]
        key = (city, str(match_date)) if city and match_date else None

        weather_row: dict = {"match_id": row["match_id"]}
        if key and key in weather_cache:
            weather_row.update(weather_cache[key])
        rows.append(weather_row)

    new_df = pd.DataFrame(rows)

    # Merge with cache
    if not cached.empty and not new_df.empty:
        result = pd.concat([cached, new_df], ignore_index=True)
    elif not new_df.empty:
        result = new_df
    else:
        result = cached

    # Save cache
    WEATHER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(WEATHER_CACHE_PATH, index=False)
    log.info("Weather data cached: %d matches at %s", len(result), WEATHER_CACHE_PATH)

    return result


def _fetch_single_day(
    lat: float,
    lon: float,
    d: date,
    retries: int = 3,
    timezone: str = "Europe/Rome",
) -> dict | None:
    """Fetch daily weather for a single location and date from Open-Meteo."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": d.isoformat(),
        "end_date": d.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": timezone,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as e:
            if attempt < retries:
                wait = 2 ** attempt
                log.warning(
                    "Weather fetch attempt %d/%d failed for (%s,%s) on %s: %s — retrying in %ds",
                    attempt, retries, lat, lon, d, e, wait,
                )
                time.sleep(wait)
            else:
                log.warning("Weather fetch failed after %d attempts for (%s,%s) on %s: %s", retries, lat, lon, d, e)
                return None

    daily = data.get("daily", {})
    if not daily:
        return None

    result = {}
    for var in DAILY_VARS:
        values = daily.get(var, [None])
        result[f"weather_{var}"] = values[0] if values else None

    return result
