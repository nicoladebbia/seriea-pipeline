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


_COMPETITION_IDS: dict[str, str] = {"serie_a": "IT1", "premier_league": "GB1"}


def current_league_teams(season: str, league: str = "serie_a") -> set[str] | None:
    """Team names actually in a league for a season, from TM's competition page.

    The static team maps are historical supersets; this resolves the ACTUAL
    member clubs for ``season`` by matching TM verein IDs on the competition
    page against the map. Returns None on any failure so callers fall back to
    the full map rather than silently scraping nothing.
    """
    comp = _COMPETITION_IDS.get(league)
    if comp is None:
        return None
    tm_season = season.split("-")[0]
    url = f"{TM_BASE}/x/startseite/wettbewerb/{comp}/saison_id/{tm_season}"
    try:
        resp = requests.get(url, headers=TM_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning("Could not fetch %s %s competition page (%s) — using full map",
                    league, season, e)
        return None
    ids_on_page = {
        int(cid)
        for cid in re.findall(rf"/startseite/verein/(\d+)/saison_id/{tm_season}", resp.text)
    }
    if not ids_on_page:
        log.warning("No club IDs parsed from %s competition page — using full map", league)
        return None
    names = {name for name, (_slug, cid) in _get_league_teams(league).items()
             if cid in ids_on_page}
    return names or None


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
    only_teams: set[str] | None = None,
) -> pd.DataFrame:
    """Scrape market values for all squads in a league from Transfermarkt.

    Args:
        season: Season string, e.g. "2024-2025"
        league: League key from LEAGUE_TEAMS_TM (e.g. "serie_a", "premier_league")
        only_teams: optional set of team names to restrict to (the map is a
            historical superset — pass the current clubs to avoid wasted fetches).

    Returns DataFrame with columns:
        team, player_name, position, market_value_eur, age, nationality
    """
    league_teams = _get_league_teams(league)
    if only_teams is not None:
        league_teams = {k: v for k, v in league_teams.items() if k in only_teams}
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

        # Age — on the squad page the age is the parenthesized number in the
        # date-of-birth cell, e.g. "01/07/2000 (26)". The OLD code grabbed the
        # first standalone 1-2 digit cell, which is the SHIRT NUMBER
        # (td.rueckennummer, e.g. "9") — so Scamacca (shirt 9) got age 9. Parse
        # "(NN)" instead; fall back to a plausible bare age only if no DOB found.
        player_data["age"] = None
        for td in cells:
            m = re.search(r"\((\d{1,2})\)", td.get_text(strip=True))
            if m:
                player_data["age"] = int(m.group(1))
                break
        if player_data["age"] is None:
            for td in cells:
                # skip the shirt-number cell explicitly
                if "rueckennummer" in (td.get("class") or []):
                    continue
                text = td.get_text(strip=True)
                if re.match(r"^\d{2}$", text) and 15 <= int(text) <= 45:
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
    only_teams: set[str] | None = None,
    window: str = "both",
) -> pd.DataFrame:
    """Scrape all transfers for a league/season.

    window: which transfer window(s) to fetch, and how to TAG each row.
        "both"   → fetch summer + winter separately and tag every row with a
                   ``window`` column ("summer"/"winter"). This is the default and
                   the form the daily cron uses so the file always carries the
                   window flag (the leak-free net_squad_delta filter and the live
                   jan_* features both depend on being able to tell them apart).
        "summer" → summer window only (w_s=s), rows tagged "summer".
        "winter" → winter window only (w_s=w), rows tagged "winter".
    Tagging matters: net_squad_delta must exclude winter (a January signing must
    not retroactively inflate that season's August matches — training leak), while
    the jan_* features specifically read winter rows. A single untagged file can
    serve neither correctly.

    Fetches each team's transfer page once and parses both the Arrivals
    and Departures tables separately (they appear as table[0] and table[1]).

    Args:
        season: Season string, e.g. "2024-2025"
        league: League key from LEAGUE_TEAMS_TM (e.g. "serie_a", "premier_league")
        only_teams: optional set of team names to restrict the scrape to. The
            Serie A map is a historical superset (relegated clubs kept for
            backfilling old seasons); pass the current season's clubs so a
            daily run doesn't waste ~12 requests on clubs no longer in the
            league (see current_league_teams()).

    Returns DataFrame with columns:
        team, player_name, age, transfer_type (in/out),
        from_club, to_club, fee_eur, fee_text, is_loan
    """
    league_teams = _get_league_teams(league)
    if only_teams is not None:
        league_teams = {k: v for k, v in league_teams.items() if k in only_teams}
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

    # (window label, w_s query value) pairs to fetch. TM's w_s param filters the
    # transfer window: s=summer, w=winter. "both" fetches each separately so every
    # row is unambiguously tagged (parsing a merged page's section order is
    # fragile; two explicit fetches are not).
    _WS = {"summer": "s", "winter": "w"}
    if window == "both":
        windows = [("summer", "s"), ("winter", "w")]
    elif window in _WS:
        windows = [(window, _WS[window])]
    else:
        raise ValueError(f"window must be 'both'/'summer'/'winter', got {window!r}")

    for team_name, (slug, verein_id) in teams_to_scrape.items():
        n_in = n_out = 0
        for win_label, ws in windows:
            url = (
                f"{TM_BASE}/{slug}/transfers/verein/{verein_id}/saison_id/{tm_season}"
                f"/pos//detailpos/0/w_s/{ws}/plus/1"
            )
            try:
                resp = requests.get(url, headers=TM_HEADERS, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as e:
                log.warning("Failed to fetch %s %s transfers: %s", team_name, win_label, e)
                time.sleep(5)
                continue

            arrivals, departures = _parse_transfers_page(resp.text, team_name)
            for r in arrivals + departures:
                r["window"] = win_label
            all_rows.extend(arrivals)
            all_rows.extend(departures)
            n_in += len(arrivals)
            n_out += len(departures)
            time.sleep(4)
        log.info("  %s: %d arrivals, %d departures", team_name, n_in, n_out)

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


def scrape_rumors(
    season: str = "2026-2027",
    league: str = "serie_a",
    only_teams: set[str] | None = None,
) -> pd.DataFrame:
    """Scrape UNCONFIRMED transfer rumors per club (Transfermarkt /geruechte).

    Rumors are SPECULATION, not done deals — this data is display-only and must
    NEVER feed the model (compute_net_squad_delta reads transfers_*, never
    rumors_*). Each rumor carries the honest signals TM actually provides:
    the player's market value, the linked club, the most-recent-source date and
    URL (freshness/traceability). No fabricated probability — TM's per-rumor
    "assessment" is frequently blank, so we do not invent a number.

    Rumors change and expire, so this OVERWRITES rumors_<season>.parquet each
    run (no incremental cache) and stamps scraped_at. All rows are confirmed=False.

    Returns columns: team, player_name, age, current_club, market_value_eur,
    market_value_text, source_date, source_url, confirmed, scraped_at.
    """
    league_teams = _get_league_teams(league)
    if only_teams is not None:
        league_teams = {k: v for k, v in league_teams.items() if k in only_teams}
    prefix = _league_cache_prefix(league)
    cache_path = TM_DIR / f"{prefix}rumors_{season.replace('-', '_')}.parquet"
    scraped_at = pd.Timestamp.utcnow().isoformat()

    all_rows: list[dict] = []
    for team_name, (slug, verein_id) in league_teams.items():
        url = f"{TM_BASE}/{slug}/geruechte/verein/{verein_id}"
        try:
            resp = requests.get(url, headers=TM_HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("Failed to fetch %s rumors: %s", team_name, e)
            time.sleep(5)
            continue
        rows = _parse_rumors_page(resp.text, team_name)
        for r in rows:
            r["confirmed"] = False
            r["scraped_at"] = scraped_at
        all_rows.extend(rows)
        log.info("  %s: %d rumors", team_name, len(rows))
        time.sleep(4)

    df = pd.DataFrame(all_rows)
    TM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info("Saved %d rumors to %s", len(df), cache_path)
    return df


def _parse_rumors_page(html: str, team_name: str) -> list[dict]:
    """Parse a Transfermarkt rumors (/geruechte) page → list of rumor dicts.

    Columns per row: Player, Nat., Age, Club (current club), Market value,
    Most recent source (a dated forum-thread link), Assessment (often '-').
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.items")
    if table is None:
        return []
    rows: list[dict] = []
    for tr in table.select("tbody tr"):
        name_link = tr.select_one("td.hauptlink a")
        if not name_link:
            continue
        player = name_link.get_text(strip=True)
        if not player:
            continue
        # TM renders a spacer/detail sub-row per player that also carries a
        # hauptlink; it has no age cell. Skip it to avoid duplicate all-NaN rows.
        has_age = any(
            re.match(r"^\d{1,2}$", td.get_text(strip=True)) for td in tr.find_all("td")
        )
        if not has_age:
            continue
        rumor: dict = {"team": team_name, "player_name": player}

        # Age — first standalone 1-2 digit cell
        for td in tr.find_all("td"):
            t = td.get_text(strip=True)
            if re.match(r"^\d{1,2}$", t):
                rumor["age"] = int(t)
                break

        # Current club (an image/link with a /verein/ href, title = club name)
        club_link = tr.select_one("td.zentriert a[href*='/verein/'] img")
        if club_link and club_link.get("alt"):
            rumor["current_club"] = club_link["alt"].strip()

        # Market value: a /marktwertverlauf/ link cell
        mv_cell = tr.select_one("td.rechts a[href*='/marktwertverlauf/']")
        if mv_cell:
            mv_text = mv_cell.get_text(strip=True)
            rumor["market_value_text"] = mv_text
            rumor["market_value_eur"] = _parse_market_value(mv_text)

        # Most-recent-source: dated forum link (freshness + traceability)
        for a in tr.select("td.zentriert a[href*='/thread/']"):
            date_txt = a.get_text(strip=True)
            if re.match(r"\d{2}/\d{2}/\d{4}", date_txt):
                rumor["source_date"] = date_txt
                rumor["source_url"] = a.get("href")
                break

        rows.append(rumor)
    return rows


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

            # Position — the 2nd row of the inline name table holds the role
            # subtext ("Centre-Forward"); the 1st row is the name itself.
            position = None
            inline = row.select_one("table.inline-table")
            if inline:
                inline_rows = inline.select("tr")
                if len(inline_rows) >= 2:
                    pos_text = inline_rows[1].get_text(strip=True)
                    if pos_text:
                        position = pos_text
            transfer["position"] = position

            # Nationality — first flag image title is the player's nation (later
            # flags are the club's country); skip if absent.
            flag = row.select_one("img.flaggenrahmen")
            transfer["nationality"] = flag.get("title") if flag else None

            # Age
            for td in cells:
                text = td.get_text(strip=True)
                if re.match(r"^\d{1,2}$", text):
                    transfer["age"] = int(text)
                    break

            # From/To club — on the /plus/1 layout the club link is NOT inside
            # td.zentriert (the sibling parser's selector finds 0 here). Each real
            # transfer row carries the other club TWICE: a crest <img> (full name,
            # e.g. "Manchester United") and a text link (short "Man Utd"), both
            # pointing at the same /verein/ id. Verified 2026-07-14 on a live page:
            # the two always resolve to the same club, so grabbing the crest's
            # full-name title is unambiguous. Prefer the FIRST crest img whose link
            # is a /verein/ link; fall back to the first verein link's text.
            other_club = None
            for a in row.select("a[href*='/verein/']"):
                img = a.select_one("img[title], img[alt]")
                if img is not None:
                    other_club = img.get("title") or img.get("alt")
                    break
            if not other_club:
                link = row.select_one("a[href*='/verein/']")
                other_club = link.get_text(strip=True) or None if link is not None else None
            if other_club:
                if direction == "in":
                    transfer["from_club"] = other_club
                    transfer["to_club"] = team_name
                else:
                    transfer["from_club"] = team_name
                    transfer["to_club"] = other_club

            # Market value AND fee both render as td.rechts. On the /plus/1 layout
            # the FIRST td.rechts is the player's market value at the move and the
            # LAST is the transfer fee. The old code took the first (select_one) →
            # it silently stored MARKET VALUE as the fee whenever both were present
            # (e.g. Højlund fee €44m was recorded as his €60m value). Take the last
            # for the fee, keep the first as market_value_at_transfer.
            rechts = row.select("td.rechts")
            if rechts:
                if len(rechts) >= 2:
                    mv_text = rechts[0].get_text(strip=True)
                    transfer["market_value_at_transfer"] = _parse_market_value(mv_text)
                    fee_text = rechts[-1].get_text(strip=True)
                else:
                    # single value → it's the fee column; no separate MV shown
                    transfer["market_value_at_transfer"] = None
                    fee_text = rechts[0].get_text(strip=True)
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
