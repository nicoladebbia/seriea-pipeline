#!/usr/bin/env python3
"""WEATHER INTEGRATION - Real Weather Data for Predictions

Fetches weather forecasts for match venues using:
1. Open-Meteo API (free, no key required)
2. Manual fallback for historical patterns

Weather factors affect:
- Goals: Rain +0.20 goals, Wind -0.22 goals
- Home win: Rain +3.2%
- Cards: Cold +5%
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Stadium coordinates for all supported leagues
STADIUM_COORDS = {
    # ── Serie A (Italy) ──────────────────────────────────────────────
    "Inter": {"lat": 45.4781, "lon": 9.1240, "city": "Milan", "tz": "Europe/Rome"},
    "Milan": {"lat": 45.4781, "lon": 9.1240, "city": "Milan", "tz": "Europe/Rome"},
    "Juventus": {"lat": 45.1096, "lon": 7.6413, "city": "Turin", "tz": "Europe/Rome"},
    "Torino": {"lat": 45.0413, "lon": 7.6497, "city": "Turin", "tz": "Europe/Rome"},
    "Roma": {"lat": 41.9341, "lon": 12.4547, "city": "Rome", "tz": "Europe/Rome"},
    "Lazio": {"lat": 41.9341, "lon": 12.4547, "city": "Rome", "tz": "Europe/Rome"},
    "Napoli": {"lat": 40.8280, "lon": 14.1930, "city": "Naples", "tz": "Europe/Rome"},
    "Fiorentina": {"lat": 43.7806, "lon": 11.2821, "city": "Florence", "tz": "Europe/Rome"},
    "Atalanta": {"lat": 45.7089, "lon": 9.6808, "city": "Bergamo", "tz": "Europe/Rome"},
    "Bologna": {"lat": 44.4924, "lon": 11.3096, "city": "Bologna", "tz": "Europe/Rome"},
    "Genoa": {"lat": 44.4163, "lon": 8.9527, "city": "Genoa", "tz": "Europe/Rome"},
    "Sampdoria": {"lat": 44.4163, "lon": 8.9527, "city": "Genoa", "tz": "Europe/Rome"},
    "Verona": {"lat": 45.4353, "lon": 10.9686, "city": "Verona", "tz": "Europe/Rome"},
    "Udinese": {"lat": 46.0819, "lon": 13.2003, "city": "Udine", "tz": "Europe/Rome"},
    "Sassuolo": {"lat": 44.7147, "lon": 10.6513, "city": "Reggio Emilia", "tz": "Europe/Rome"},
    "Empoli": {"lat": 43.7261, "lon": 10.9486, "city": "Empoli", "tz": "Europe/Rome"},
    "Cagliari": {"lat": 39.2001, "lon": 9.1375, "city": "Cagliari", "tz": "Europe/Rome"},
    "Lecce": {"lat": 40.3586, "lon": 18.1901, "city": "Lecce", "tz": "Europe/Rome"},
    "Monza": {"lat": 45.5845, "lon": 9.2767, "city": "Monza", "tz": "Europe/Rome"},
    "Parma": {"lat": 44.7953, "lon": 10.3381, "city": "Parma", "tz": "Europe/Rome"},
    "Como": {"lat": 45.8167, "lon": 9.0833, "city": "Como", "tz": "Europe/Rome"},
    "Venezia": {"lat": 45.4375, "lon": 12.3333, "city": "Venice", "tz": "Europe/Rome"},
    "Spezia": {"lat": 44.1024, "lon": 9.8200, "city": "La Spezia", "tz": "Europe/Rome"},
    "Salernitana": {"lat": 40.6806, "lon": 14.7681, "city": "Salerno", "tz": "Europe/Rome"},
    "Cremonese": {"lat": 45.1403, "lon": 10.0206, "city": "Cremona", "tz": "Europe/Rome"},
    "Pisa": {"lat": 43.7228, "lon": 10.3966, "city": "Pisa", "tz": "Europe/Rome"},
    "Frosinone": {"lat": 41.6406, "lon": 13.3403, "city": "Frosinone", "tz": "Europe/Rome"},
    # ── Premier League (England) ─────────────────────────────────────
    "Arsenal": {"lat": 51.5549, "lon": -0.1084, "city": "London", "tz": "Europe/London"},
    "Aston Villa": {"lat": 52.5092, "lon": -1.8847, "city": "Birmingham", "tz": "Europe/London"},
    "Bournemouth": {"lat": 50.7352, "lon": -1.8388, "city": "Bournemouth", "tz": "Europe/London"},
    "Brentford": {"lat": 51.4907, "lon": -0.2888, "city": "Brentford", "tz": "Europe/London"},
    "Brighton": {"lat": 50.8615, "lon": -0.0833, "city": "Brighton", "tz": "Europe/London"},
    "Burnley": {"lat": 53.7890, "lon": -2.2301, "city": "Burnley", "tz": "Europe/London"},
    "Chelsea": {"lat": 51.4817, "lon": -0.1910, "city": "London", "tz": "Europe/London"},
    "Crystal Palace": {"lat": 51.3983, "lon": -0.0856, "city": "London", "tz": "Europe/London"},
    "Everton": {"lat": 53.4389, "lon": -2.9664, "city": "Liverpool", "tz": "Europe/London"},
    "Fulham": {"lat": 51.4749, "lon": -0.2217, "city": "London", "tz": "Europe/London"},
    "Ipswich": {"lat": 52.0547, "lon": 1.1448, "city": "Ipswich", "tz": "Europe/London"},
    "Leeds": {"lat": 53.7774, "lon": -1.5721, "city": "Leeds", "tz": "Europe/London"},
    "Leicester": {"lat": 52.6204, "lon": -1.1420, "city": "Leicester", "tz": "Europe/London"},
    "Liverpool": {"lat": 53.4308, "lon": -2.9608, "city": "Liverpool", "tz": "Europe/London"},
    "Luton": {"lat": 51.8843, "lon": -0.4316, "city": "Luton", "tz": "Europe/London"},
    "Man City": {"lat": 53.4831, "lon": -2.2004, "city": "Manchester", "tz": "Europe/London"},
    "Man United": {"lat": 53.4631, "lon": -2.2913, "city": "Manchester", "tz": "Europe/London"},
    "Newcastle": {"lat": 54.9756, "lon": -1.6217, "city": "Newcastle", "tz": "Europe/London"},
    "Nottingham Forest": {"lat": 52.9400, "lon": -1.1329, "city": "Nottingham", "tz": "Europe/London"},
    "Sheffield United": {"lat": 53.3703, "lon": -1.4726, "city": "Sheffield", "tz": "Europe/London"},
    "Southampton": {"lat": 50.9058, "lon": -1.3911, "city": "Southampton", "tz": "Europe/London"},
    "Sunderland": {"lat": 54.9146, "lon": -1.3882, "city": "Sunderland", "tz": "Europe/London"},
    "Tottenham": {"lat": 51.6043, "lon": -0.0665, "city": "London", "tz": "Europe/London"},
    "Watford": {"lat": 51.6500, "lon": -0.4014, "city": "Watford", "tz": "Europe/London"},
    "West Brom": {"lat": 52.5090, "lon": -1.9640, "city": "West Bromwich", "tz": "Europe/London"},
    "West Ham": {"lat": 51.5387, "lon": 0.0166, "city": "London", "tz": "Europe/London"},
    "Wolves": {"lat": 52.5903, "lon": -2.1306, "city": "Wolverhampton", "tz": "Europe/London"},
    "Norwich": {"lat": 52.6222, "lon": 1.3090, "city": "Norwich", "tz": "Europe/London"},
    "Stoke": {"lat": 52.9884, "lon": -2.1756, "city": "Stoke", "tz": "Europe/London"},
    "Swansea": {"lat": 51.6427, "lon": -3.9353, "city": "Swansea", "tz": "Europe/London"},
    "Huddersfield": {"lat": 53.6544, "lon": -1.7684, "city": "Huddersfield", "tz": "Europe/London"},
    "Middlesbrough": {"lat": 54.5782, "lon": -1.2169, "city": "Middlesbrough", "tz": "Europe/London"},
}


def get_weather_forecast(team: str, date: str, time: str = "15:00") -> Optional[Dict]:
    """Fetch weather forecast for a team's stadium on a given date.

    Args:
        team: Home team name
        date: Match date (YYYY-MM-DD)
        time: Match time (HH:MM)

    Returns:
        Weather data dict or None
    """
    coords = STADIUM_COORDS.get(team)
    if not coords:
        log.warning(f"No coordinates for team: {team}")
        return None

    # Open-Meteo forecasts are unreliable beyond 5 days
    try:
        from datetime import datetime as _dt
        match_date = _dt.strptime(date, "%Y-%m-%d").date()
        days_until = (match_date - _dt.now().date()).days
        if days_until > 5:
            log.info(f"Skipping weather for {team} on {date} — {days_until} days away (max 5)")
            return None
    except ValueError:
        pass

    try:
        # Open-Meteo API (free, no key required)
        tz = coords.get("tz", "Europe/Rome")
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={coords['lat']}&longitude={coords['lon']}"
            f"&hourly=temperature_2m,precipitation,windspeed_10m,weathercode"
            f"&timezone={tz}"
        )

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Find the hour closest to match time
        target_datetime = f"{date}T{time}"
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        # Find index for match time
        hour = int(time.split(":")[0])
        target = f"{date}T{hour:02d}:00"

        if target in times:
            idx = times.index(target)
        else:
            # Find closest hour
            for i, t in enumerate(times):
                if t.startswith(date):
                    idx = i + hour
                    break
            else:
                idx = 0

        if idx >= len(hourly.get("temperature_2m", [])):
            idx = len(hourly.get("temperature_2m", [])) - 1

        weather = {
            "temperature": hourly.get("temperature_2m", [None])[idx],
            "precipitation": hourly.get("precipitation", [None])[idx],
            "wind_speed": hourly.get("windspeed_10m", [None])[idx],
            "weather_code": hourly.get("weathercode", [None])[idx],
            "city": coords["city"],
            "source": "open-meteo"
        }

        # Classify conditions
        weather["conditions"] = classify_weather(weather)

        log.info(f"Weather for {team} on {date}: {weather['temperature']}°C, "
                 f"{weather['precipitation']}mm rain, {weather['wind_speed']}km/h wind")

        return weather

    except Exception as e:
        log.warning(f"Failed to fetch weather for {team}: {e}")
        return None


def classify_weather(weather: Dict) -> Dict:
    """Classify weather conditions for prediction factors.

    Returns dict with:
    - is_rainy: precipitation > 5mm
    - is_windy: wind > 30 km/h
    - is_cold: temp < 10°C
    - is_hot: temp > 25°C
    """
    return {
        "is_rainy": (weather.get("precipitation") or 0) > 5,
        "is_windy": (weather.get("wind_speed") or 0) > 30,
        "is_cold": (weather.get("temperature") or 15) < 10,
        "is_hot": (weather.get("temperature") or 15) > 25,
    }


def get_weather_impact(weather: Dict) -> Dict:
    """Calculate prediction adjustments based on weather.

    Based on validated analysis:
    - Rain: +0.20 goals, +3.2% home win
    - Wind: -0.22 goals
    - Cold: +5% cards
    """
    if not weather or not weather.get("conditions"):
        return {"goals_adj": 0, "home_win_adj": 0, "cards_adj": 0, "factors": []}

    cond = weather["conditions"]
    impacts = {
        "goals_adj": 0,
        "home_win_adj": 0,
        "cards_adj": 0,
        "factors": []
    }

    if cond.get("is_rainy"):
        impacts["goals_adj"] += 0.20
        impacts["home_win_adj"] += 0.032  # +3.2%
        impacts["factors"].append("rain")

    if cond.get("is_windy"):
        impacts["goals_adj"] -= 0.22
        impacts["factors"].append("wind")

    if cond.get("is_cold"):
        impacts["cards_adj"] += 0.05  # +5%
        impacts["factors"].append("cold")

    return impacts


def fetch_all_match_weather(matches_file: str = None) -> Dict:
    """Fetch weather for all upcoming matches.

    Args:
        matches_file: Path to matches JSON file

    Returns:
        Dict mapping "home_team vs away_team" to weather data
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import DATA_DIR

    if matches_file is None:
        # No explicit file: use the shared upcoming-fixture source. This used to
        # read manual_matches.json directly with NO date filter, so once the Odds
        # API key lapsed it kept fetching forecasts for the previous season's
        # final matchweek (2026-08-24) — a forecast for a date three months past
        # is not just wasted API calls, it is garbage fed into the feature row.
        from scripts.utils.match_timing import _is_future, _load_sofascore_fixtures
        now = datetime.now(timezone.utc)
        matches = _load_sofascore_fixtures(now)
        if not matches:
            # Sofascore unavailable: fall back to the manual file, but date-filter
            # it this time — a stale source must degrade to EMPTY, never to wrong.
            mp = DATA_DIR / "upcoming" / "manual_matches.json"
            if mp.exists():
                try:
                    with open(mp) as f:
                        matches = [m for m in (json.load(f).get("matches") or [])
                                   if _is_future(m, now)]
                except (OSError, ValueError) as e:
                    log.warning(f"Could not read {mp.name}: {e}")
                    matches = []
        if not matches:
            log.warning("No upcoming fixtures available for weather fetch")
            return {}
    else:
        if not Path(matches_file).exists():
            log.warning(f"No matches file found at {matches_file}")
            return {}
        with open(matches_file) as f:
            data = json.load(f)
        matches = data.get("matches", [])
    weather_data = {}

    log.info(f"Fetching weather for {len(matches)} matches...")

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        date = match.get("date", "")
        time = match.get("time", "15:00")

        key = f"{home} vs {away}"
        weather = get_weather_forecast(home, date, time)

        if weather:
            weather["impact"] = get_weather_impact(weather)
            weather_data[key] = weather

    # Save weather data
    output_path = DATA_DIR / "upcoming" / "weather.json"
    with open(output_path, "w") as f:
        json.dump({
            "fetched_at": datetime.now().isoformat(),
            "matches": weather_data
        }, f, indent=2)

    log.info(f"Saved weather data to {output_path}")
    return weather_data


if __name__ == "__main__":
    weather = fetch_all_match_weather()

    print("\n" + "=" * 60)
    print("WEATHER SUMMARY")
    print("=" * 60)

    for match, data in weather.items():
        print(f"\n{match}")
        print(f"  City: {data.get('city')}")
        print(f"  Temp: {data.get('temperature')}°C")
        print(f"  Rain: {data.get('precipitation')}mm")
        print(f"  Wind: {data.get('wind_speed')} km/h")

        impact = data.get("impact", {})
        if impact.get("factors"):
            print(f"  Factors: {', '.join(impact['factors'])}")
            print(f"  Goals adj: {impact['goals_adj']:+.2f}")
            print(f"  Home win adj: {impact['home_win_adj']:+.1%}")
