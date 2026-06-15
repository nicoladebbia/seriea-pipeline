"""FBref World Cup 2026 results — a Cloudflare-resilient SECOND live source.

Why this exists
---------------
The grader merges live results onto the martj42 CSV (which lags hours). The
primary live source is Sofascore, but it shares a Cloudflare estate that
periodically IP-bans this host. FBref is an INDEPENDENT source (Stats Perform /
Opta data) — when Sofascore is banned, FBref may still answer (and vice-versa).
This module fetches FBref's WC schedule page, parses the finished matches, and
writes ``fbref_results.json`` in the EXACT shape of ``sofascore_results.json`` so
``engine.load_results_with_live`` merges it with zero extra wiring.

Method (matches the rest of the repo's FBref scraping)
------------------------------------------------------
FBref is Cloudflare-fronted, so a headless ``botasaurus`` browser fetches the
HTML (the same approach as ``scripts/pipeline/_refresh_fbref_fixtures``); the
saved page is then parsed with ``pandas.read_html``. The schedule table schema
(Wk/Day/Date/Time/Home/Score/Away) is identical across competitions — the parser
is verified against the saved Serie A specimen in ``data/raw/html`` (see
``tests/test_worldcup.py::TestFbrefResults``).

Honesty / safety
----------------
- A short cooldown breaker: if the browser fetch fails (ban / timeout), this
  fails fast and writes nothing, rather than hanging the 30-min refresh.
- Score strings are Opta full-time scores ('2–1', en-dash). Extra-time/penalty
  shootout handling matches the CSV/Sofascore convention: the FT score is stored;
  90'-reconstruction for knockouts happens downstream from goalscorers, unchanged.
- STATUS: live-verified 2026-06-15 against the real WC schedule page (visible
  solver returned ~366 KB, 12 finished matches parsed with correct scores and
  canon-joinable team names). FBref is a MANUAL backup, not an unattended source:
  it needs a VISIBLE browser to pass Cloudflare (headless is detected), so it is
  NOT in the 30-min cron. The unattended fallback is ESPN's key-free scoreboard
  (``sofascore_fetch._espn_results``). Run FBref by hand only when ESPN +
  Sofascore are both down: ``python3 -m scripts.worldcup.fbref_fetch``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from io import StringIO
from typing import Any

import pandas as pd

from scripts.worldcup.engine import (
    DATA_DIR,
    atomic_write_json,
    canon_team,
)

log = logging.getLogger(__name__)

# FBref World Cup competition id is 1. The 2026 edition's schedule page:
WC_SCHEDULE_URL = (
    "https://fbref.com/en/comps/1/schedule/FIFA-World-Cup-Scores-and-Fixtures"
)
HTML_DIR = DATA_DIR.parent / "raw" / "html" / "wc2026"
SPECIMEN_HTML = HTML_DIR / "fixtures.html"
FBREF_RESULTS_JSON = DATA_DIR / "fbref_results.json"

# FBref uses an EN DASH between scores ("2–1"), occasionally a hyphen.
_SCORE_SEPS = ("–", "—", "-")

# FBref renders each team cell with a 2-letter country flag code attached:
# Home = "Mexico mx" (trailing), Away = "za South Africa" (leading). Plus a few
# spellings differ from the canonical results dataset.
_FBREF_TEAM_FIXUPS = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Korea Republic": "South Korea",   # canon_team also maps this; belt-and-braces
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
}


def _clean_team(raw: Any) -> str:
    """Strip FBref's flag code and normalise to a canon-joinable name.

    'Mexico mx' -> 'Mexico'; 'za South Africa' -> 'South Africa'. The flag code
    is a standalone lowercase 2-letter token at the start or end of the cell.
    """
    s = str(raw).strip()
    if not s:
        return s
    parts = s.split()
    # Flag code is a standalone lowercase 2-3 letter token (xx=ISO-2, or 3-letter
    # for UK home nations: sct/eng/wls/nir). Trailing on Home, leading on Away.
    def _is_code(tok: str) -> bool:
        return 2 <= len(tok) <= 3 and tok.islower() and tok.isalpha()
    if len(parts) > 1 and _is_code(parts[-1]):
        parts = parts[:-1]          # trailing code (Home cells)
    elif len(parts) > 1 and _is_code(parts[0]):
        parts = parts[1:]           # leading code (Away cells)
    name = " ".join(parts).strip()
    name = _FBREF_TEAM_FIXUPS.get(name, name)
    return canon_team(name)


def fetch_schedule_html(timeout_s: int = 150) -> str | None:
    """Fetch the WC schedule page past Cloudflare; None on any failure.

    Uses the undetected-chromedriver solver and reads the HTML DIRECTLY from the
    solved browser (``driver.page_source``) — NOT cookie-extraction, which is
    racy and short-lived. Verified to return the real WC schedule (~366 KB, 12+
    matches) where plain HTTP gets a 403 challenge.

    IMPORTANT — NOT for the unattended cron: Cloudflare reliably detects a
    HEADLESS browser, so this must run VISIBLE (a Chrome window opens for ~30 s).
    That needs a desktop session. The 30-min refresh job does NOT call this; run
    it manually (``python3 -m scripts.worldcup.fbref_fetch``) as a backup when
    BOTH ESPN and Sofascore are down. The primary live source is ESPN's key-free
    scoreboard (``sofascore_fetch._espn_results``), which is Cloudflare-free and
    runs unattended.
    """
    try:
        from scraper.cloudflare_solver import _create_driver, _wait_for_challenge
    except ImportError as exc:
        log.warning("cloudflare_solver unavailable — FBref source off: %s", exc)
        return None

    driver = None
    try:
        # Visible (headless is detected by Cloudflare). Real browser solves the JS
        # challenge; we then read the already-rendered DOM — no cookie handoff.
        driver = _create_driver(use_undetected=True, headless=False)
        driver.get(WC_SCHEDULE_URL)
        if not _wait_for_challenge(driver, timeout_s):
            log.warning("FBref: Cloudflare challenge unresolved in %ss", timeout_s)
            return None
        import time
        time.sleep(3)  # let the schedule table finish rendering
        html = driver.page_source or ""
    except Exception as exc:  # noqa: BLE001 — any browser/network error => no source
        log.warning("FBref fetch failed: %s", exc)
        return None
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:  # noqa: BLE001, S110 — quit errors are non-fatal
                pass

    if len(html) < 50_000:
        log.warning("FBref page too small (%s bytes) — likely a CF challenge",
                    len(html))
        return None
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    SPECIMEN_HTML.write_text(html, encoding="utf-8")
    log.info("FBref schedule saved: %s (%s bytes)", SPECIMEN_HTML, len(html))
    return html


def _schedule_table(html: str) -> pd.DataFrame | None:
    """Locate the FBref schedule table (Date/Home/Score/Away) in a page."""
    try:
        tables = pd.read_html(StringIO(html))
    except ValueError:
        return None
    for t in tables:
        cols = {str(c) for c in t.columns}
        if {"Home", "Score", "Away"}.issubset(cols) and "Date" in cols:
            return t
    return None


def _parse_score(raw: Any) -> tuple[int, int] | None:
    """'2–1' -> (2, 1). None if not a played, well-formed score."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    for sep in _SCORE_SEPS:
        if sep in s:
            left, _, right = s.partition(sep)
            try:
                return int(left.strip()), int(right.strip())
            except ValueError:
                return None
    return None


def parse_results(html: str) -> list[dict[str, Any]]:
    """Finished WC matches from a saved FBref schedule page.

    Output rows match ``sofascore_results.json`` exactly so the grader overlay
    merges them: {event_id, date, home, away, home_score, away_score, ...}.
    Team names are canon-mapped so they join martj42/Sofascore.
    """
    table = _schedule_table(html)
    if table is None:
        log.warning("FBref: no schedule table found in HTML")
        return []
    out: list[dict[str, Any]] = []
    for row in table.itertuples(index=False):
        d = dict(row._asdict()) if hasattr(row, "_asdict") else {}
        score = _parse_score(d.get("Score"))
        if score is None:
            continue  # not played yet / malformed
        date = d.get("Date")
        if not (d.get("Home") and d.get("Away") and date) or str(d.get("Home")) == "nan":
            continue
        home = _clean_team(d["Home"])   # strips FBref flag codes -> canon name
        away = _clean_team(d["Away"])
        if not home or not away:
            continue
        hs, as_ = score
        # FBref id: stable surrogate from (date, teams) — no event id on the page.
        eid = f"fbref:{date}:{home}:{away}"
        out.append({
            "event_id": eid,
            "date": str(date),
            "home": home,
            "away": away,
            "home_score": hs,
            "away_score": as_,
            "winner": (home if hs > as_ else away if as_ > hs else None),
            "decided_by": "FT",
            "penalties": None,
            "source": "fbref",
        })
    return out


def fetch_results() -> dict[str, Any]:
    """Fetch + parse + persist WC results from FBref. Fail-soft.

    Returns the written blob ({fetched_at, results}). On any source failure the
    EXISTING file is preserved (union semantics like the Sofascore fetcher) so a
    transient ban never erases already-collected results.
    """
    from scripts.worldcup.engine import read_json_safe

    prior = read_json_safe(FBREF_RESULTS_JSON, {})
    by_id: dict[str, dict[str, Any]] = {
        str(r["event_id"]): r
        for r in (prior.get("results", []) if isinstance(prior, dict) else [])
    }

    html = fetch_schedule_html()
    if html is None and SPECIMEN_HTML.exists():
        # ban now, but parse the last saved page so we never go backwards
        html = SPECIMEN_HTML.read_text(encoding="utf-8")
    n_new = 0
    if html:
        for rec in parse_results(html):
            if rec["event_id"] not in by_id:
                by_id[rec["event_id"]] = rec
                n_new += 1

    out = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "results": sorted(by_id.values(), key=lambda r: (r["date"], r["event_id"])),
    }
    atomic_write_json(FBREF_RESULTS_JSON, out)
    log.info("FBref results: %d stored (%d new) -> %s",
             len(by_id), n_new, FBREF_RESULTS_JSON)
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    fetch_results()


if __name__ == "__main__":
    main()
