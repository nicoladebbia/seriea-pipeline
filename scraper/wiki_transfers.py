"""Scrape Serie A transfer rows from Wikipedia's season-transfer list pages.

Second, independent transfer source alongside Transfermarkt. Wikipedia is
complementary, not redundant: it carries the EXACT TRANSFER DATE that the
Transfermarkt transfer table does not expose (TM only exposes the summer/winter
window via a URL param), plus an independently-edited from/to club and fee. We
use it to (a) attach a real date to TM rows, (b) DERIVE the transfer window from
that date instead of guessing, and (c) raise a cross-source confidence flag
(a transfer confirmed by both sources vs one).

Verified specimen 2026-07-14 (cardinal rule — never parse an unverified source):
  Page:   en.wikipedia.org/wiki/List_of_Italian_football_transfers_summer_2026
  Table:  a single table.wikitable, header
          ['Date', 'Name', 'Moving from', 'Moving to', 'Fee'], 174 rows,
          consistent 5 cells/row. 66 rows touch a Serie A club.
  Fee:    frequently "Undisclosed[nb 1]" — footnote markers stripped; numeric
          fee parsed only when a €/£ value is present.

Wikipedia names each page by the SEASON IT PRECEDES; "summer_2026" actually holds
the 2025-26 winter + 2026 spring/summer moves. The window is derived from the
row's own date (Jun-Aug + Sep-early → summer; Dec-Feb → winter), so the page
title is irrelevant to correctness.

This is NEVER read by the model feature layer directly — it enriches the
Transfermarkt spine via scraper/transfermarkt-side merge. Display + data-quality.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

WIKI_DIR = DATA_DIR / "external" / "transfermarkt"  # co-located with TM for the merge
WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (seriea-pipeline transfer cross-check)"}

# Serie A page slugs by season. Wikipedia titles the page by the season it
# precedes; the row date (not the title) decides the window, so both the
# "summer" and "winter" list pages for a season are valid inputs.
_WIKI_PAGES = {
    "2026-2027": [
        "List_of_Italian_football_transfers_summer_2026",
        "List_of_Italian_football_transfers_winter_2025%E2%80%9326",
    ],
}

# Current (2026-27) Serie A member clubs as Wikipedia renders them (short forms).
# Used to keep only rows where a Serie A club is the origin or destination.
# VERIFIED 2026-07-14 against TM's competition page AND Wikipedia's promoted/
# relegated table: PROMOTED in = Venezia, Frosinone, Monza; RELEGATED out =
# Cremonese, Hellas Verona, Pisa. Getting this wrong silently drops the promoted
# clubs' incoming transfers (they sign mostly from Serie B / abroad).
_SERIE_A_WIKI = {
    "Milan", "AC Milan", "Inter", "Inter Milan", "Napoli", "Juventus", "Roma",
    "AS Roma", "Lazio", "Atalanta", "Fiorentina", "Bologna", "Torino", "Udinese",
    "Genoa", "Como", "Cagliari", "Parma", "Lecce", "Sassuolo",
    "Venezia", "Frosinone", "Monza",
}


def _clean_fee(text: str) -> tuple[str, float | None]:
    """Strip footnote markers; return (clean_text, eur_or_None).

    "Undisclosed[nb 1]" → ("Undisclosed", None). "€44m" → ("€44m", 44_000_000).
    Only a real €/£ magnitude yields a number; loans / free / undisclosed → None.
    """
    clean = re.sub(r"\[[^\]]*\]", "", text).strip()
    low = clean.lower()
    if not clean or "undisclosed" in low or "free" in low or "loan" in low or "swap" in low:
        return clean, None
    m = re.search(r"[€£]\s*([\d.]+)\s*(m|bn|k|million|billion)?", clean, re.I)
    if not m:
        return clean, None
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("bn", "billion"):
        val *= 1_000_000_000
    elif unit in ("m", "million"):
        val *= 1_000_000
    elif unit == "k":
        val *= 1_000
    return clean, val


def _derive_window(date: datetime | None) -> str | None:
    """Summer window = May-early Sep; winter = Dec-early Feb; else None.

    May is included in summer: pre-window agreed moves are commonly dated late
    May on Wikipedia (registration opens ~June) and belong to the summer window.
    """
    if date is None:
        return None
    m = date.month
    if 5 <= m <= 9:
        return "summer"
    if m == 12 or m <= 2:
        return "winter"
    return None


def _parse_date(text: str) -> datetime | None:
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_wiki_page(html: str) -> list[dict]:
    """Parse one Wikipedia transfer list page → list of Serie A-touching rows.

    Defensive: skips rows whose cell count or header shape doesn't match the
    verified [Date, Name, Moving from, Moving to, Fee] layout, and logs how many
    were dropped so a silent schema drift never masquerades as "no transfers".
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.select("table.wikitable")
    rows_out: list[dict] = []
    total_rows = 0
    skipped_shape = 0
    skipped_non_sa = 0

    for table in tables:
        header_cells = [th.get_text(strip=True).lower() for th in table.select("tr")[0].select("th")]
        # Only tables that look like the transfer table (Date/Name/Moving from/to)
        if not ({"date", "name"} <= set(header_cells) and any("moving" in h for h in header_cells)):
            continue
        # Wikipedia rowspan-merges the Date cell: several transfers share one date,
        # so only the FIRST row of each date group carries a <td> date (5 cells);
        # the rest inherit it and have 4 cells (name, from, to, fee). Carry the
        # last-seen date forward so those rows are kept, not dropped as "bad shape".
        last_date_txt = ""
        for tr in table.select("tr")[1:]:
            cells = tr.find_all(["td", "th"])
            total_rows += 1
            if len(cells) == 5:
                last_date_txt = cells[0].get_text(" ", strip=True)
                name, from_club, to_club, fee_raw = (
                    cells[1].get_text(" ", strip=True), cells[2].get_text(" ", strip=True),
                    cells[3].get_text(" ", strip=True), cells[4].get_text(" ", strip=True),
                )
            elif len(cells) == 4:
                # date inherited from the rowspan above
                name, from_club, to_club, fee_raw = (
                    cells[0].get_text(" ", strip=True), cells[1].get_text(" ", strip=True),
                    cells[2].get_text(" ", strip=True), cells[3].get_text(" ", strip=True),
                )
            else:
                skipped_shape += 1
                continue
            date_txt = last_date_txt
            if not name:
                skipped_shape += 1
                continue
            # keep only Serie A-touching moves
            if from_club not in _SERIE_A_WIKI and to_club not in _SERIE_A_WIKI:
                skipped_non_sa += 1
                continue
            dt = _parse_date(date_txt)
            fee_clean, fee_eur = _clean_fee(fee_raw)
            rows_out.append({
                "player_name": name,
                "from_club": from_club,
                "to_club": to_club,
                "fee_text": fee_clean,
                "fee_eur": fee_eur,
                "transfer_date": dt.date().isoformat() if dt else None,
                "window": _derive_window(dt),
                "source": "wikipedia",
            })

    log.info(
        "Wikipedia parse: %d rows seen, %d kept (Serie A), %d non-SA, %d bad-shape",
        total_rows, len(rows_out), skipped_non_sa, skipped_shape,
    )
    return rows_out


def scrape_wiki_transfers(season: str = "2026-2027") -> pd.DataFrame:
    """Fetch + parse Wikipedia transfer pages for a season → Serie A rows only.

    Writes wiki_transfers_{season}.parquet next to the TM files. Returns the df.
    Fail-soft per page: an HTTP error on one page logs and continues (the other
    page may still yield rows); a total failure returns an empty df, never raises.
    """
    pages = _WIKI_PAGES.get(season)
    if not pages:
        log.warning("No Wikipedia page configured for season %s", season)
        return pd.DataFrame()

    all_rows: list[dict] = []
    for slug in pages:
        url = f"https://en.wikipedia.org/wiki/{slug}"
        try:
            resp = requests.get(url, headers=WIKI_HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning("Wikipedia fetch failed for %s: %s", slug, e)
            continue
        all_rows.extend(_parse_wiki_page(resp.text))

    df = pd.DataFrame(all_rows)
    if not df.empty:
        # a player can appear on both the summer and winter page — keep the first
        df = df.drop_duplicates(subset=["player_name", "from_club", "to_club"], keep="first")
        WIKI_DIR.mkdir(parents=True, exist_ok=True)
        out = WIKI_DIR / f"wiki_transfers_{season.replace('-', '_')}.parquet"
        df.to_parquet(out, index=False)
        log.info("Saved %d Wikipedia transfers to %s", len(df), out)
    return df


def _norm(text: str) -> str:
    """Normalize a name for cross-source matching: strip accents, lowercase."""
    import unicodedata
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", " ", t.lower()).strip()


def enrich_transfers_with_wiki(season: str = "2026-2027") -> pd.DataFrame | None:
    """Attach Wikipedia's exact date + a cross-source confidence flag to the TM file.

    The Transfermarkt file stays the spine (it has the money/position detail); this
    joins Wikipedia's `transfer_date` onto it by normalized player name and adds:
      - transfer_date : exact date from Wikipedia where the player matched (else NaT)
      - n_sources     : 2 if the player appears in both TM and Wikipedia, else 1
      - date_window   : window DERIVED from the Wikipedia date (independent check of
                        the TM `w_s` window tag; they should agree)

    Additive and idempotent — re-running overwrites the three columns, never
    duplicates rows. Returns the enriched TM df (also written back to the TM file),
    or None if either source file is missing. Never raises on a missing match.
    """
    tm_path = WIKI_DIR / f"transfers_{season.replace('-', '_')}.parquet"
    wiki_path = WIKI_DIR / f"wiki_transfers_{season.replace('-', '_')}.parquet"
    if not tm_path.exists():
        log.warning("TM transfers file missing (%s) — cannot enrich", tm_path)
        return None
    tm = pd.read_parquet(tm_path)
    if not wiki_path.exists():
        log.warning("Wikipedia file missing (%s) — TM left un-enriched", wiki_path)
        return tm

    wiki = pd.read_parquet(wiki_path)
    # map normalized name -> (date, window) from Wikipedia (first occurrence wins)
    wiki_by_name: dict[str, tuple] = {}
    for _, r in wiki.iterrows():
        key = _norm(r["player_name"])
        if key and key not in wiki_by_name:
            wiki_by_name[key] = (r.get("transfer_date"), r.get("window"))

    dates, windows, nsrc = [], [], []
    matched = 0
    for _, r in tm.iterrows():
        hit = wiki_by_name.get(_norm(r["player_name"]))
        # A TM "End of loan" row and a Wikipedia row for the same player describe
        # DIFFERENT events (the summer return vs the original January loan-out), so
        # the Wikipedia date would be the wrong event's date. Cross-confirm the
        # player (n_sources=2) but do NOT attach a date/window to loan-return rows.
        is_loan_return = "end of loan" in str(r.get("fee_text", "")).lower()
        if hit is not None:
            matched += 1
            if is_loan_return:
                dates.append(None)
                windows.append(None)
            else:
                dates.append(hit[0])
                windows.append(hit[1])
            nsrc.append(2)
        else:
            dates.append(None)
            windows.append(None)
            nsrc.append(1)
    tm["transfer_date"] = dates
    tm["date_window"] = windows
    tm["n_sources"] = nsrc

    tm.to_parquet(tm_path, index=False)
    log.info(
        "Enriched TM transfers with Wikipedia: %d/%d rows matched (%.0f%%), "
        "%d dual-sourced", matched, len(tm), 100 * matched / max(len(tm), 1), matched,
    )
    return tm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    d = scrape_wiki_transfers("2026-2027")
    print(f"\n{len(d)} Serie A transfer rows from Wikipedia")
    if not d.empty:
        print(d.head(10).to_string())
