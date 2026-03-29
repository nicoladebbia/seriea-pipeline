"""Scrape player market values and transfer data from Transfermarkt.

Uses the transfermarkt-datasets project (dcaribou/transfermarkt-datasets)
which publishes cleaned CSV/Parquet data on Kaggle. We download the
pre-built datasets rather than scraping Transfermarkt directly, since:
  1. No API key needed
  2. Data is already cleaned and structured
  3. Covers all major leagues including Serie A

Datasets:
  - players.csv: player profiles (name, DOB, position, foot, height, citizenship)
  - player_valuations.csv: historical market values per player over time
  - transfers.csv: all transfers with fee, clubs, dates
  - clubs.csv: club profiles
  - appearances.csv: player appearances per game

Kaggle dataset: https://www.kaggle.com/datasets/davidcariboo/player-scores
Direct download: https://github.com/dcaribou/transfermarkt-datasets

Since the automated updates are currently paused on that repo, we also
provide a direct Transfermarkt HTML scraper as fallback.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import DATA_DIR, HEADERS

log = logging.getLogger(__name__)

TM_DIR = DATA_DIR / "external" / "transfermarkt"
SERIE_A_TM_ID = "IT1"  # Transfermarkt competition code for Serie A

# Transfermarkt base
TM_BASE = "https://www.transfermarkt.com"

# Per-league Transfermarkt team IDs: canonical_name -> (slug, verein_id)
# Team names MUST match the canonical names from config/team_names.py

# All Premier League team IDs on Transfermarkt (canonical_name -> (slug, verein_id))
# Includes all teams from 2017-2026 seasons
EPL_TEAMS_TM: dict[str, tuple[str, int]] = {
    "Arsenal": ("arsenal-fc", 11),
    "Aston Villa": ("aston-villa", 405),
    "Bournemouth": ("afc-bournemouth", 989),
    "Brentford": ("brentford-fc", 1148),
    "Brighton": ("brighton-amp-hove-albion", 1237),
    "Burnley": ("burnley-fc", 1132),
    "Cardiff": ("cardiff-city", 2687),
    "Chelsea": ("fc-chelsea", 631),
    "Crystal Palace": ("crystal-palace", 873),
    "Everton": ("fc-everton", 29),
    "Fulham": ("fc-fulham", 931),
    "Huddersfield": ("huddersfield-town", 1110),
    "Hull": ("hull-city", 3008),
    "Ipswich": ("ipswich-town", 677),
    "Leeds": ("leeds-united", 399),
    "Leicester": ("leicester-city", 1003),
    "Liverpool": ("fc-liverpool", 31),
    "Luton": ("luton-town", 1031),
    "Man City": ("manchester-city", 281),
    "Man United": ("manchester-united", 985),
    "Middlesbrough": ("middlesbrough-fc", 432),
    "Newcastle": ("newcastle-united", 762),
    "Norwich": ("norwich-city", 1123),
    "Nottingham Forest": ("nottingham-forest", 703),
    "QPR": ("queens-park-rangers", 1039),
    "Sheffield United": ("sheffield-united", 350),
    "Southampton": ("fc-southampton", 180),
    "Stoke": ("stoke-city", 512),
    "Sunderland": ("afc-sunderland", 289),
    "Swansea": ("swansea-city", 2288),
    "Tottenham": ("tottenham-hotspur", 148),
    "Watford": ("fc-watford", 1010),
    "West Brom": ("west-bromwich-albion", 984),
    "West Ham": ("west-ham-united", 379),
    "Wigan": ("wigan-athletic", 1071),
    "Wolves": ("wolverhampton-wanderers", 543),
    # Historical EPL teams (pre-2017, in match data from 2005+)
    "Birmingham": ("birmingham-city", 337),
    "Blackburn": ("blackburn-rovers", 164),
    "Blackpool": ("fc-blackpool", 1181),
    "Bolton": ("bolton-wanderers", 355),
    "Charlton": ("charlton-athletic", 358),
    "Derby": ("derby-county", 22),
    "Portsmouth": ("fc-portsmouth", 1020),
    "Reading": ("fc-reading", 1032),
}

# All Serie A team IDs on Transfermarkt (team_name -> (slug, verein_id))
# Includes historical teams from 2017-2025
SERIE_A_TEAMS_TM: dict[str, tuple[str, int]] = {
    "Atalanta": ("atalanta-bc", 800),
    "Benevento": ("benevento-calcio", 4171),
    "Bologna": ("fc-bologna", 1025),
    "Brescia": ("brescia-calcio", 1006),
    "Cagliari": ("cagliari-calcio", 1390),
    "Chievo": ("chievo-verona", 1045),
    "Como": ("como-1907", 1047),
    "Cremonese": ("us-cremonese", 3898),
    "Crotone": ("fc-crotone", 2566),
    "Empoli": ("fc-empoli", 749),
    "Fiorentina": ("acf-fiorentina", 430),
    "Frosinone": ("frosinone-calcio", 8970),
    "Genoa": ("genoa-cfc", 252),
    "Inter": ("inter-milan", 46),
    "Juventus": ("juventus-fc", 506),
    "Lazio": ("ss-lazio", 398),
    "Lecce": ("us-lecce", 1005),
    "Milan": ("ac-milan", 5),
    "Monza": ("ac-monza", 2919),
    "Napoli": ("ssc-napoli", 6195),
    "Parma": ("parma-calcio-1913", 130),
    "Pisa": ("ac-pisa-1909", 1029),
    "Roma": ("as-roma", 12),
    "Salernitana": ("us-salernitana-1919", 380),
    "Sampdoria": ("uc-sampdoria", 1038),
    "Sassuolo": ("us-sassuolo-calcio", 6574),
    "SPAL": ("spal-2013", 3522),
    "Spezia": ("spezia-calcio", 3775),
    "Torino": ("torino-fc", 416),
    "Udinese": ("udinese-calcio", 410),
    "Venezia": ("venezia-fc", 607),
    "Verona": ("hellas-verona", 276),
}

TM_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# League -> team dict lookup
LEAGUE_TEAMS_TM: dict[str, dict[str, tuple[str, int]]] = {
    "serie_a": SERIE_A_TEAMS_TM,
    "premier_league": EPL_TEAMS_TM,
}


def _get_league_teams(league: str) -> dict[str, tuple[str, int]]:
    """Return the team dict for a league key. Defaults to Serie A."""
    teams = LEAGUE_TEAMS_TM.get(league)
    if teams is None:
        raise ValueError(
            f"No Transfermarkt team mapping for league '{league}'. "
            f"Available: {sorted(LEAGUE_TEAMS_TM)}"
        )
    return teams


def _league_cache_prefix(league: str) -> str:
    """Return file prefix for league-specific cache files.

    Serie A uses no prefix (backward compatible), other leagues use league key.
    """
    if league == "serie_a":
        return ""
    return f"{league}_"


def scrape_squad_market_values(
    season: str = "2024-2025",
    league: str = "serie_a",
) -> pd.DataFrame:
    """Scrape market values for all squads in a league from Transfermarkt.

    Args:
        season: Season string, e.g. "2024-2025"
        league: League key from LEAGUE_TEAMS_TM (e.g. "serie_a", "premier_league")

    Returns DataFrame with columns:
        team, player_name, position, market_value_eur, age, nationality
    """
    league_teams = _get_league_teams(league)
    prefix = _league_cache_prefix(league)
    cache_path = TM_DIR / f"{prefix}market_values_{season.replace('-', '_')}.parquet"
    cached_df = None
    teams_to_scrape = dict(league_teams)  # copy

    if cache_path.exists():
        cached_df = pd.read_parquet(cache_path)
        cached_teams = set(cached_df["team"].unique()) if "team" in cached_df.columns else set()
        all_teams = set(league_teams.keys())
        missing = all_teams - cached_teams
        if not missing:
            log.info("Loading cached market values from %s (%d teams)", cache_path, len(cached_teams))
            return cached_df
        # Only scrape the missing teams
        log.info("Cache exists but missing %d teams: %s — scraping those", len(missing), missing)
        teams_to_scrape = {k: v for k, v in league_teams.items() if k in missing}

    league_name = league.replace("_", " ").title()
    log.info("Scraping market values for %d %s teams (%s)", len(teams_to_scrape), league_name, season)
    all_rows = []

    # Convert season to TM format: "2024-2025" -> "2024"
    tm_season = season.split("-")[0]

    for team_name, (slug, verein_id) in teams_to_scrape.items():
        url = f"{TM_BASE}/{slug}/kader/verein/{verein_id}/saison_id/{tm_season}/plus/1"
        log.info("Fetching %s market values: %s", team_name, url)

        try:
            resp = requests.get(url, headers=TM_HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("Failed to fetch %s: %s", team_name, e)
            time.sleep(5)
            continue

        players = _parse_squad_page(resp.text, team_name)
        all_rows.extend(players)
        time.sleep(4)  # Respectful delay

    new_df = pd.DataFrame(all_rows)

    # Merge with cached data if we were backfilling
    if cached_df is not None and not new_df.empty:
        df = pd.concat([cached_df, new_df], ignore_index=True)
    elif cached_df is not None:
        df = cached_df
    else:
        df = new_df

    TM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("Saved %d player market values to %s", len(df), cache_path)
    return df


def _parse_squad_page(html: str, team_name: str) -> list[dict]:
    """Parse a Transfermarkt squad page for player market values."""
    soup = BeautifulSoup(html, "lxml")
    rows = []

    # Find player rows in the squad table
    for row in soup.select("table.items tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        player_data: dict = {"team": team_name}

        # Player name
        name_link = row.select_one("td.hauptlink a")
        if name_link:
            player_data["player_name"] = name_link.get_text(strip=True)
        else:
            continue

        # Position
        pos_td = row.select("td.posrela")
        if pos_td:
            pos_span = pos_td[0].select_one("tr:last-child td")
            if pos_span:
                player_data["position"] = pos_span.get_text(strip=True)

        # Age
        for td in cells:
            text = td.get_text(strip=True)
            if re.match(r"^\d{1,2}$", text):
                player_data["age"] = int(text)
                break

        # Market value (last cell typically)
        mv_cell = row.select_one("td.rechts.hauptlink")
        if mv_cell:
            player_data["market_value_eur"] = _parse_market_value(
                mv_cell.get_text(strip=True)
            )

        # Nationality
        flag_imgs = row.select("td.zentriert img.flaggenrahmen")
        if flag_imgs:
            player_data["nationality"] = flag_imgs[0].get("title", "")

        rows.append(player_data)

    return rows


def scrape_transfers(
    season: str = "2024-2025",
    league: str = "serie_a",
) -> pd.DataFrame:
    """Scrape all transfers for a league/season.

    Fetches each team's transfer page once and parses both the Arrivals
    and Departures tables separately (they appear as table[0] and table[1]).

    Args:
        season: Season string, e.g. "2024-2025"
        league: League key from LEAGUE_TEAMS_TM (e.g. "serie_a", "premier_league")

    Returns DataFrame with columns:
        team, player_name, age, transfer_type (in/out),
        from_club, to_club, fee_eur, fee_text, is_loan
    """
    league_teams = _get_league_teams(league)
    prefix = _league_cache_prefix(league)
    cache_path = TM_DIR / f"{prefix}transfers_{season.replace('-', '_')}.parquet"
    cached_df = None
    teams_to_scrape = dict(league_teams)

    if cache_path.exists():
        cached_df = pd.read_parquet(cache_path)
        cached_teams = set(cached_df["team"].unique()) if "team" in cached_df.columns else set()
        missing = set(league_teams.keys()) - cached_teams
        if not missing:
            log.info("Loading cached transfers from %s (%d teams)", cache_path, len(cached_teams))
            return cached_df
        log.info("Cache exists but missing %d teams: %s — scraping those", len(missing), missing)
        teams_to_scrape = {k: v for k, v in league_teams.items() if k in missing}

    league_name = league.replace("_", " ").title()
    log.info("Scraping transfers for %d %s teams (%s)", len(teams_to_scrape), league_name, season)
    all_rows = []
    tm_season = season.split("-")[0]

    for team_name, (slug, verein_id) in teams_to_scrape.items():
        url = f"{TM_BASE}/{slug}/transfers/verein/{verein_id}/saison_id/{tm_season}"
        log.info("Fetching %s transfers: %s", team_name, url)

        try:
            resp = requests.get(url, headers=TM_HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("Failed to fetch %s transfers: %s", team_name, e)
            time.sleep(5)
            continue

        arrivals, departures = _parse_transfers_page(resp.text, team_name)
        all_rows.extend(arrivals)
        all_rows.extend(departures)
        log.info("  %s: %d arrivals, %d departures", team_name, len(arrivals), len(departures))
        time.sleep(4)

    new_df = pd.DataFrame(all_rows)

    if cached_df is not None and not new_df.empty:
        df = pd.concat([cached_df, new_df], ignore_index=True)
    elif cached_df is not None:
        df = cached_df
    else:
        df = new_df

    TM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("Saved %d transfers to %s", len(df), cache_path)
    return df


def _parse_transfers_page(html: str, team_name: str) -> tuple[list[dict], list[dict]]:
    """Parse a Transfermarkt transfer page with both Arrivals and Departures tables.

    Returns (arrivals, departures) as two lists of dicts.
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.select("table.items")

    arrivals: list[dict] = []
    departures: list[dict] = []

    for table in tables:
        # Determine if this is arrivals or departures by checking preceding header
        prev_header = table.find_previous(["h2", "h3"])
        header_text = prev_header.get_text(strip=True).lower() if prev_header else ""

        if "arrival" in header_text or "zugäng" in header_text or "zugang" in header_text:
            direction = "in"
            target_list = arrivals
        elif "departure" in header_text or "abgäng" in header_text or "abgang" in header_text:
            direction = "out"
            target_list = departures
        else:
            # Skip unknown tables (e.g. transfer record summary)
            continue

        for row in table.select("tbody tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            transfer: dict = {"team": team_name, "transfer_type": direction}

            # Player name
            name_link = row.select_one("td.hauptlink a")
            if name_link:
                transfer["player_name"] = name_link.get_text(strip=True)
            else:
                continue

            # Age
            for td in cells:
                text = td.get_text(strip=True)
                if re.match(r"^\d{1,2}$", text):
                    transfer["age"] = int(text)
                    break

            # From/To club
            club_links = row.select("td.zentriert a[href*='/verein/']")
            if club_links:
                other_club = club_links[0].get_text(strip=True)
                if direction == "in":
                    transfer["from_club"] = other_club
                    transfer["to_club"] = team_name
                else:
                    transfer["from_club"] = team_name
                    transfer["to_club"] = other_club

            # Fee
            fee_cell = row.select_one("td.rechts")
            if fee_cell:
                fee_text = fee_cell.get_text(strip=True)
                transfer["fee_text"] = fee_text
                transfer["fee_eur"] = _parse_market_value(fee_text)
                transfer["is_loan"] = "loan" in fee_text.lower()

            target_list.append(transfer)

    return arrivals, departures


def _parse_market_value(text: str) -> Optional[float]:
    """Parse Transfermarkt market value strings like '€25.00m', '€800k', 'free transfer'.

    Returns value in EUR or None.
    """
    if not text or text == "-" or "free" in text.lower():
        return 0.0

    # Handle "End of loan" entries: may or may not have a fee after the date
    if "end of loan" in text.lower():
        # Look for a fee like "€250k" at the end
        loan_fee = re.search(r"€([\d.]+)\s*(m|bn|k)", text, re.I)
        if loan_fee:
            val = float(loan_fee.group(1))
            unit = loan_fee.group(2).lower()
            if unit == "bn":
                return val * 1_000_000_000
            elif unit == "m":
                return val * 1_000_000
            elif unit == "k":
                return val * 1_000
            return val
        return None  # End of loan with no fee

    # Handle "Loan fee:€X" and "loan transfer"
    if "loan transfer" in text.lower():
        return None

    # Extract the part after the last € sign to avoid date contamination
    if "€" in text:
        fee_part = text.rsplit("€", 1)[-1].strip()
    else:
        fee_part = text.replace("£", "").strip()

    m = re.search(r"([\d.]+)\s*(m|bn|k)", fee_part, re.I)
    if m:
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit == "bn":
            return val * 1_000_000_000
        elif unit == "m":
            return val * 1_000_000
        elif unit == "k":
            return val * 1_000
        return val

    # Plain number
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def compute_squad_value_features(
    matches: pd.DataFrame, market_values: pd.DataFrame
) -> pd.DataFrame:
    """Add squad market value features to the match table.

    Columns added:
      home_squad_value, away_squad_value, squad_value_ratio,
      home_avg_player_value, away_avg_player_value
    """
    if market_values.empty:
        return matches

    # Aggregate per team
    team_values = (
        market_values.groupby("team")["market_value_eur"]
        .agg(["sum", "mean", "median", "max", "count"])
        .rename(columns={
            "sum": "squad_total_value",
            "mean": "avg_player_value",
            "median": "median_player_value",
            "max": "max_player_value",
            "count": "squad_size",
        })
        .reset_index()
    )

    df = matches.copy()
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        tv = team_values.copy()
        tv = tv.rename(columns={
            c: f"{prefix}_{c}" for c in tv.columns if c != "team"
        })
        tv = tv.rename(columns={"team": team_col})
        df = df.merge(tv, on=team_col, how="left")

    # Ratio
    if "home_squad_total_value" in df.columns and "away_squad_total_value" in df.columns:
        h = pd.to_numeric(df["home_squad_total_value"], errors="coerce")
        a = pd.to_numeric(df["away_squad_total_value"], errors="coerce")
        df["squad_value_ratio"] = h / a.replace(0, float("nan"))

    return df


def compute_transfer_features(
    matches: pd.DataFrame, transfers: pd.DataFrame
) -> pd.DataFrame:
    """Add transfer activity features to the match table.

    Columns added:
      home_transfer_spend, away_transfer_spend,
      home_transfer_income, away_transfer_income,
      home_net_spend, away_net_spend,
      home_new_signings, away_new_signings
    """
    if transfers.empty:
        return matches

    df = matches.copy()

    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        for direction, label in [("in", "spend"), ("out", "income")]:
            t = transfers[transfers["transfer_type"] == direction].copy()
            team_spend = (
                t.groupby("team")["fee_eur"]
                .agg(["sum", "count"])
                .rename(columns={"sum": f"transfer_{label}", "count": f"transfers_{direction}"})
                .reset_index()
                .rename(columns={"team": team_col})
            )
            col_map = {
                c: f"{prefix}_{c}" for c in team_spend.columns if c != team_col
            }
            team_spend = team_spend.rename(columns=col_map)
            df = df.merge(team_spend, on=team_col, how="left")

    # Net spend
    for prefix in ("home", "away"):
        spend_col = f"{prefix}_transfer_spend"
        income_col = f"{prefix}_transfer_income"
        if spend_col in df.columns and income_col in df.columns:
            df[f"{prefix}_net_spend"] = (
                pd.to_numeric(df[spend_col], errors="coerce").fillna(0)
                - pd.to_numeric(df[income_col], errors="coerce").fillna(0)
            )

    return df
