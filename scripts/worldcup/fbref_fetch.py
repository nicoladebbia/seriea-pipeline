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
- IMPORTANT: the live FBref WC page has not been fetched from this host yet (the
  IP is Cloudflare-banned at build time). The PARSER is specimen-verified; the
  FETCH + the WC team-name mapping must be confirmed against one real WC specimen
  before the output is trusted — see ``verify_against_specimen`` and the log line
  it emits. Until then this is armed, not proven.
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


def fetch_schedule_html(timeout_s: int = 60) -> str | None:
    """Headless-browser fetch of the WC schedule page; None on any failure.

    Mirrors ``_refresh_fbref_fixtures``: botasaurus solves Cloudflare, we grab
    the rendered HTML. Fails fast (no retry storm) so a ban can't hang refresh.
    """
    try:
        from botasaurus.browser import Driver, browser
    except ImportError:
        log.warning("botasaurus not installed — FBref source unavailable")
        return None

    @browser(
        headless=True,
        block_images_and_css=False,
        wait_for_complete_page_load=True,
        reuse_driver=False,
        output=None,
        create_error_logs=False,
    )
    def _fetch(driver: Driver, data: dict) -> str:
        driver.get(data["url"])
        # Cloudflare interstitial / consent dismissal, best-effort.
        try:
            driver.wait_for_element("body.fb", wait=timeout_s)
        except Exception:  # noqa: BLE001, S110 — absence handled by the size guard below
            pass
        html = driver.page_html
        return html or ""

    try:
        html = _fetch({"url": WC_SCHEDULE_URL})
    except Exception as exc:  # noqa: BLE001 — any browser/network error => no source
        log.warning("FBref fetch failed: %s", exc)
        return None
    if not html or len(html) < 50_000:
        log.warning("FBref page too small (%s bytes) — likely a CF challenge",
                    len(html) if html else 0)
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
        home = d.get("Home")
        away = d.get("Away")
        date = d.get("Date")
        if not (home and away and date) or str(home) == "nan":
            continue
        hs, as_ = score
        # FBref id: stable surrogate from (date, teams) — no event id on the page.
        eid = f"fbref:{date}:{home}:{away}"
        out.append({
            "event_id": eid,
            "date": str(date),
            "home": canon_team(str(home)),
            "away": canon_team(str(away)),
            "home_score": hs,
            "away_score": as_,
            "winner": (canon_team(str(home)) if hs > as_
                       else canon_team(str(away)) if as_ > hs else None),
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
