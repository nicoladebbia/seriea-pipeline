"""Build a cross-source match_id mapping table.

Joins FBref (8-char hash) ↔ Sofascore (numeric id) ↔ Understat (numeric id) ↔
canonical (date_home_away) match IDs so downstream joins can unify
per-source data on a single canonical match row.

Writes to data/parsed/match_id_mapping.parquet with columns:
  match_id (canonical), fbref_hash, sofascore_id, understat_id,
  match_date, home_team, away_team, season, league

Scope: Serie A + Premier League. understat_id populates for SA only
(matches_xg.parquet is SA-only); EPL carries sofascore_id + fbref_hash.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.team_names import normalize_team

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = PROJECT / "data" / "parsed" / "match_id_mapping.parquet"


def canonical_id(home: str, away: str, date: pd.Timestamp) -> str:
    return f"{date.strftime('%Y-%m-%d')}_{home}_{away}".replace(" ", "-")


def load_matches() -> pd.DataFrame:
    """Base table — every Serie A and Premier League match with canonical match_id.

    Both leagues are mapped because the same Sofascore numeric ids key both SA and
    EPL parity files; without an EPL bridge here, EPL Sofascore-derived features
    (shot xG, first-half splits, missing players) drop on the inner join. The
    (home, away, date) key is collision-free across the two leagues (no shared
    team names). NOTE: understat_id only populates for SA — matches_xg.parquet is
    SA-only — so the EPL bridge carries sofascore_id + fbref_hash only.
    """
    df = pd.read_parquet(PROJECT / "data" / "parsed" / "matches.parquet")
    df = df[df["league"].isin(["serie_a", "premier_league"])].copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["home_team"] = df["home_team"].apply(normalize_team)
    df["away_team"] = df["away_team"].apply(normalize_team)
    return df[["match_id", "match_date", "home_team", "away_team", "season", "league"]].copy()


def discover_fbref_hashes() -> dict[tuple, str]:
    """Return {(home, away, date_str): fbref_hash}.

    Parses fbref fixtures.html files (1 per season) to map (home, away, date) → hash.
    """
    out = {}
    html_dir = PROJECT / "data" / "raw" / "html"
    for season_dir in html_dir.iterdir():
        if not season_dir.is_dir():
            continue
        # _epl season dirs are now parsed too — EPL fbref hashes feed the EPL bridge.
        # The (home, away, date) key is collision-free across SA/EPL (no shared names).
        fixtures_html = season_dir / "fixtures.html"
        if not fixtures_html.exists():
            continue
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(fixtures_html.read_text(encoding="utf-8", errors="replace"), "html.parser")
            for row in soup.find_all("tr"):
                # Each match row has a date cell and an href to /en/matches/{hash}/...
                date_cell = row.find("td", {"data-stat": "date"}) or row.find("th", {"data-stat": "date"})
                date_val = date_cell.get("csk") if date_cell else None
                if not date_val and date_cell:
                    date_val = date_cell.get_text(strip=True)
                if not date_val:
                    continue
                match_link = None
                for a in row.find_all("a"):
                    href = a.get("href", "")
                    m = re.match(r"/en/matches/([a-f0-9]{8})/", href)
                    if m:
                        match_link = m.group(1)
                        break
                if not match_link:
                    continue
                home_cell = row.find("td", {"data-stat": "home_team"})
                away_cell = row.find("td", {"data-stat": "away_team"})
                if not home_cell or not away_cell:
                    continue
                home = normalize_team(home_cell.get_text(strip=True))
                away = normalize_team(away_cell.get_text(strip=True))
                try:
                    date = pd.to_datetime(date_val).strftime("%Y-%m-%d")
                except Exception:
                    continue
                out[(home, away, date)] = match_link
        except Exception as e:
            log.warning("Failed parsing %s: %s", fixtures_html, e)
    log.info("FBref fixtures discovery: %d (home,away,date) → hash entries", len(out))
    return out


def discover_sofascore_ids() -> dict[tuple, str]:
    """Return {(home, away, date_str): sofascore_id}.

    Parses sofascore/fixtures_YYYY_YYYY.json files.
    """
    out = {}
    sofa_dir = PROJECT / "data" / "external" / "sofascore"
    for p in sofa_dir.glob("fixtures_*.json"):
        # EPL fixtures are now included — they carry the sofascore_id that the
        # EPL feature builders (shot xG, first-half splits, missing players) join on.
        try:
            data = json.load(open(p))
            if not isinstance(data, list):
                continue
            for event in data:
                home = event.get("homeTeam", {})
                away = event.get("awayTeam", {})
                ts = event.get("startTimestamp")
                if not ts:
                    # Look for alternative time fields
                    time = event.get("time", {})
                    if isinstance(time, dict):
                        ts = time.get("startTimestamp")
                if not ts:
                    continue
                date = pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d")
                home_name = normalize_team(home.get("name", "")) if isinstance(home, dict) else ""
                away_name = normalize_team(away.get("name", "")) if isinstance(away, dict) else ""
                sid = str(event.get("id", ""))
                if home_name and away_name and sid:
                    out[(home_name, away_name, date)] = sid
        except Exception as e:
            log.warning("Failed parsing %s: %s", p, e)
    log.info("Sofascore fixtures discovery: %d entries", len(out))
    return out


def discover_understat_ids() -> dict[tuple, str]:
    """Return {(home, away, date_str): understat_id}.

    Uses data/external/understat/matches_xg.parquet (must have been built by parse_all_understat).
    """
    out = {}
    understat_parquet = PROJECT / "data" / "external" / "understat" / "matches_xg.parquet"
    if not understat_parquet.exists():
        log.warning("Understat parquet not found — run parse_all_understat first")
        return out
    df = pd.read_parquet(understat_parquet)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for _, r in df.iterrows():
        if pd.isna(r.get("datetime")):
            continue
        # Source schema: 'h'/'a' are dicts {'id','short_title','title'}, 'id' is the understat match id.
        h = r.get("h")
        a = r.get("a")
        home = normalize_team(h.get("title", "")) if isinstance(h, dict) else ""
        away = normalize_team(a.get("title", "")) if isinstance(a, dict) else ""
        date = r["datetime"].strftime("%Y-%m-%d")
        uid = str(r.get("id", ""))
        if home and away and uid:
            out[(home, away, date)] = uid
    log.info("Understat discovery: %d entries", len(out))
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    matches = load_matches()
    log.info("Base matches (SA + EPL): %d", len(matches))

    fbref_map = discover_fbref_hashes()
    sofa_map = discover_sofascore_ids()
    under_map = discover_understat_ids()

    def lookup(mapping, home, away, date_str):
        return mapping.get((home, away, date_str))

    matches["_date_str"] = matches["match_date"].dt.strftime("%Y-%m-%d")
    matches["fbref_hash"] = matches.apply(
        lambda r: lookup(fbref_map, r["home_team"], r["away_team"], r["_date_str"]), axis=1
    )
    matches["sofascore_id"] = matches.apply(
        lambda r: lookup(sofa_map, r["home_team"], r["away_team"], r["_date_str"]), axis=1
    )
    matches["understat_id"] = matches.apply(
        lambda r: lookup(under_map, r["home_team"], r["away_team"], r["_date_str"]), axis=1
    )
    matches = matches.drop(columns=["_date_str"])

    # Stats
    log.info("Coverage:")
    log.info("  fbref_hash found: %d / %d (%.1f%%)",
             matches["fbref_hash"].notna().sum(), len(matches),
             100 * matches["fbref_hash"].notna().sum() / len(matches))
    log.info("  sofascore_id found: %d / %d (%.1f%%)",
             matches["sofascore_id"].notna().sum(), len(matches),
             100 * matches["sofascore_id"].notna().sum() / len(matches))
    log.info("  understat_id found: %d / %d (%.1f%%)",
             matches["understat_id"].notna().sum(), len(matches),
             100 * matches["understat_id"].notna().sum() / len(matches))

    matches.to_parquet(OUT_PATH, index=False)
    log.info("Saved match_id_mapping.parquet: %d rows at %s", len(matches), OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
