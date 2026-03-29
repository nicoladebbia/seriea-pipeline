"""Backfill weather data for EPL matches using Open-Meteo archive API.

Maps EPL home teams to city coordinates, fetches daily weather,
and appends to data/external/weather.parquet matching the existing schema.
"""

import time
import sys
import requests
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WEATHER_PATH = PROJECT_ROOT / "data" / "external" / "weather.parquet"
MATCHES_PATH = PROJECT_ROOT / "data" / "parsed" / "matches.parquet"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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
    "relative_humidity_2m_mean",
]

# EPL team → city name (matching VENUE_COORDS keys in scraper/weather.py)
TEAM_CITY = {
    "Arsenal": "London",
    "Aston Villa": "Birmingham",
    "Birmingham": "Birmingham",
    "Blackburn": "Manchester",      # Blackburn ~25mi from Manchester, close enough
    "Blackpool": "Manchester",      # Blackpool ~50mi from Manchester
    "Bolton": "Manchester",         # Greater Manchester
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Cardiff": "Swansea",           # Cardiff ~40mi from Swansea, closest mapped city
    "Charlton": "London",
    "Chelsea": "London",
    "Crystal Palace": "London",
    "Derby": "Nottingham",          # Derby ~15mi from Nottingham
    "Everton": "Liverpool",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Hull": "Leeds",                # Hull ~60mi from Leeds, closest mapped city
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Man City": "Manchester",
    "Man United": "Manchester",
    "Middlesbrough": "Middlesbrough",
    "Newcastle": "Newcastle",
    "Norwich": "Norwich",
    "Nottingham Forest": "Nottingham",
    "Portsmouth": "Southampton",    # Portsmouth ~20mi from Southampton
    "QPR": "London",
    "Reading": "London",            # Reading ~40mi from London
    "Sheffield United": "Sheffield",
    "Southampton": "Southampton",
    "Stoke": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea",
    "Tottenham": "London",
    "Watford": "Watford",
    "West Brom": "West Bromwich",
    "West Ham": "West Ham",
    "Wigan": "Manchester",          # Greater Manchester
    "Wolves": "Wolverhampton",
}

# City coordinates (from scraper/weather.py VENUE_COORDS)
VENUE_COORDS = {
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


def fetch_single_day(lat: float, lon: float, date_str: str, retries: int = 5) -> dict | None:
    """Fetch daily weather for a single location and date."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/London",
    }
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
            if resp.status_code == 429:
                wait = min(10 * (2 ** (attempt - 1)), 120)  # 10s, 20s, 40s, 80s, 120s
                print(f"  Rate limited (429), waiting {wait}s (attempt {attempt}/{retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.HTTPError:
            raise  # re-raise non-429 HTTP errors after raise_for_status
        except Exception as e:
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Retry {attempt}/{retries} for ({lat},{lon}) on {date_str}: {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  FAILED after {retries} attempts for ({lat},{lon}) on {date_str}: {e}")
                return None
    else:
        print(f"  FAILED after {retries} attempts (rate limited) for ({lat},{lon}) on {date_str}")
        return None

    daily = data.get("daily", {})
    if not daily:
        return None

    result = {}
    for var in DAILY_VARS:
        values = daily.get(var, [None])
        result[f"weather_{var}"] = values[0] if values else None
    return result


WEATHER_COLS = [f"weather_{v}" for v in DAILY_VARS]


def _save_progress(epl: pd.DataFrame, weather_cache: dict, existing: pd.DataFrame):
    """Save current weather data (existing + new) to parquet."""
    rows = []
    for _, match in epl.iterrows():
        city = match["_city"]
        date_str = match["_date_str"]
        key = (city, date_str) if pd.notna(city) else None

        weather_row = {"match_id": match["match_id"]}
        if key and key in weather_cache:
            weather_row.update(weather_cache[key])
        else:
            for col in WEATHER_COLS:
                weather_row[col] = None
        rows.append(weather_row)

    new_df = pd.DataFrame(rows)
    expected_cols = ["match_id"] + WEATHER_COLS
    new_df = new_df[expected_cols]

    # Only keep rows that have actual data (drop NaN-only rows)
    new_with_data = new_df.dropna(subset=["weather_temperature_2m_max"])

    if not existing.empty and not new_with_data.empty:
        combined = pd.concat([existing, new_with_data], ignore_index=True)
    elif not new_with_data.empty:
        combined = new_with_data
    else:
        combined = existing

    for col in WEATHER_COLS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined.to_parquet(WEATHER_PATH, index=False)
    print(f"  Saved: {len(combined)} total rows ({len(new_with_data)} new EPL with data)")


def main():
    # Load matches
    matches = pd.read_parquet(MATCHES_PATH)
    epl = matches[matches["league"] == "premier_league"].copy()
    epl["match_date"] = pd.to_datetime(epl["match_date"])
    epl = epl[epl["match_date"].dt.year >= 2017].copy()
    print(f"EPL matches 2017+: {len(epl)}")

    # Load existing weather data to skip already-fetched match_ids
    if WEATHER_PATH.exists():
        existing = pd.read_parquet(WEATHER_PATH)
        # Only consider rows with actual weather data (not NaN placeholders) as "done"
        has_data = existing.dropna(subset=["weather_temperature_2m_max"])
        done_ids = set(has_data["match_id"])
        # Remove NaN-only EPL rows so we can re-fetch them
        nan_epl_ids = set(existing["match_id"]) - done_ids
        existing = existing[~existing["match_id"].isin(nan_epl_ids)]
        print(f"Existing weather rows with data: {len(existing)} (removed {len(nan_epl_ids)} NaN-only rows)")
    else:
        existing = pd.DataFrame()
        done_ids = set()

    # Filter to matches not yet successfully fetched
    epl = epl[~epl["match_id"].isin(done_ids)].copy()
    print(f"EPL matches needing weather: {len(epl)}")

    if epl.empty:
        print("Nothing to do!")
        return

    # Map teams to cities
    epl["_city"] = epl["home_team"].map(TEAM_CITY)
    unmapped = epl[epl["_city"].isna()]["home_team"].unique()
    if len(unmapped) > 0:
        print(f"WARNING: Unmapped teams: {unmapped}")

    epl["_date_str"] = epl["match_date"].dt.strftime("%Y-%m-%d")

    # Deduplicate city+date pairs for API efficiency
    city_date_pairs = epl[["_city", "_date_str"]].drop_duplicates()
    city_date_pairs = city_date_pairs.dropna(subset=["_city"])
    print(f"Unique city+date API calls needed: {len(city_date_pairs)}")

    # Fetch weather for each unique city+date
    weather_cache: dict[tuple, dict] = {}
    total = len(city_date_pairs)

    for i, (_, row) in enumerate(city_date_pairs.iterrows(), 1):
        city = row["_city"]
        date_str = row["_date_str"]
        coords = VENUE_COORDS.get(city)
        if coords is None:
            continue

        key = (city, date_str)
        result = fetch_single_day(coords[0], coords[1], date_str)
        if result:
            weather_cache[key] = result

        if i % 100 == 0 or i == total:
            print(f"  Progress: {i}/{total} ({len(weather_cache)} successful)")

        time.sleep(0.5)

        # Save intermediate progress every 500 calls
        if i % 500 == 0 and weather_cache:
            _save_progress(epl, weather_cache, existing)
            print(f"  [Checkpoint saved at {i}]")

    _save_progress(epl, weather_cache, existing)
    n_with_data = sum(1 for _, m in epl.iterrows()
                      if (m["_city"], m["_date_str"]) in weather_cache)
    print(f"\nDone! EPL matches with weather data: {n_with_data}/{len(epl)}")
    print(f"Re-run this script to fill any gaps from rate-limited requests.")


if __name__ == "__main__":
    main()
