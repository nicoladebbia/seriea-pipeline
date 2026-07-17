"""Download the Serie A match reports FBref has published but we don't hold.

Rebuilt 2026-07-16. This module was invoked weekly by
``scripts/pipeline/refresh_weekly_data.py`` (step 2) and by the runbook, but was
never committed — every run raised ModuleNotFoundError. It is the *fetcher* for
the five FBref parsers (``parse_all_player_stats``, ``parse_all_lineups``,
``parse_all_events``, ``parse_all_goalkeeper_stats``, ``parse_all_shots``), all
of which read only HTML already on disk.

Reachability, measured 2026-07-16 — not inherited from a memo
------------------------------------------------------------
=========================  ===========================================
Method                     Result
=========================  ===========================================
``curl_cffi`` chrome124    **403**, a 6 KB block page
botasaurus **headless**    **BLOCKED** — the Turnstile wall
                           ("Just a moment"), 46s, no scorebox
botasaurus **visible**     **FETCHED** — 6s, 426 KB, scorebox present
=========================  ===========================================

So FBref's block is real, unlike Sofascore's (whose June 403 had lifted by the
time it was re-probed). But it is a *headless* block, not a site ban: a visible
browser passes Turnstile in about six seconds. ``scrape_epl_match_reports.py``
is this module's direct sibling — same site, same wall — and its 320 cached EPL
reports were all fetched exactly this way. Its decisions are mirrored here
rather than re-litigated: ``headless=False`` by default, ``reuse_driver=True``
so Turnstile is solved once per batch, incremental skip, scroll-to-load, an 8s
rate limit, and a consecutive-failure backoff.

``scraper/cloudflare_solver.py`` is deliberately NOT used. ``reuse_driver=True``
already gives solve-once-then-reuse, and its ``cf_clearance`` cookie is IP- and
UA-bound with a lifetime in minutes-to-hours, so it could never carry a *weekly*
job anyway.

Why ``--headless`` probes instead of refusing
---------------------------------------------
``refresh_weekly_data.py:100`` calls this with ``--headless``, which the table
above shows cannot work. Rather than hardcode that verdict, the headless path
fetches **one** page and reads the result: if the wall is there it logs what to
run instead and exits 0; if Cloudflare ever relaxes, it just keeps going. A
measurement each week beats a constant asserting what was true in July.

Exit 0 on a wall is deliberate and matches the caller, whose own comment says
the Sofascore fallback covers the results either way. The job going red weekly
for a known, unfixable-from-cron condition trains everyone to ignore it.

The reused session wears out around ~100 fetches — re-run, don't re-engineer
----------------------------------------------------------------------------
Observed on the 2026-07-16 backfill of 119 matches: the first pass took 103,
then every remaining fetch returned **0 bytes** (not the 27 KB wall — the driver
itself was gone, alongside "Connection to remote host was lost"). A second run
with a fresh driver took the remaining 16, **0 failed** — including the one that
had already failed. Same URLs, different driver, so it is the session, not the
pages.

The backoff already handles this: repeated failures abort the run, and the
incremental skip means re-running resumes exactly where it stopped. No partial
file is ever written — ``is_real_report`` rejects a short page before the write.
That is why this is documented rather than fixed. A weekly run fetches ~10 new
matches, nowhere near the limit; cycling the driver mid-batch would add an
untested moving part to buy nothing the re-run does not already give. If a
future backfill of this size is needed, expect two passes.

Two landmines this module is shaped around
------------------------------------------
* **The filename stem IS the match_id.** ``parse_all_player_stats`` calls
  ``parse_match_html(html_path, season, html_path.stem)``, so reports are saved
  as ``{fbref_8hex}.html``. The sibling's ``{date}_{home}_{away}`` naming is
  correct for *its* parser and wrong for this one.
* **Serie A reports live in the hyphen dir** (``2025-2026``); the underscore dir
  (``2025_2026``) holds the season-level pages and is where ``fixtures.html`` is
  read from. The parser's own comment warns that mixing them "yields an empty
  parse that still exits 0".

Usage::

    python3 -m scripts.data.scrape_fbref_missing --season 2025-2026
    python3 -m scripts.data.scrape_fbref_missing --season 2025-2026 --dry-run
    python3 -m scripts.data.scrape_fbref_missing --season 2025-2026 --limit 1
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import RAW_HTML_DIR

log = logging.getLogger(__name__)

FBREF_BASE = "https://fbref.com"

#: Seconds between requests. The sibling's value, unchanged — FBref is not a
#: site to hurry, and the whole batch is under a minute of actual transfer.
RATE_LIMIT = 8

#: An FBref match hash: exactly 8 lowercase hex chars.
MATCH_HREF = re.compile(r"^/en/matches/([a-f0-9]{8})/")

#: A fetched page is real only if it carries the scorebox AND is not the wall.
#: Both halves matter: the Turnstile page is ~27 KB and has neither.
MIN_REAL_BYTES = 50_000
WALL_MARKER = "Just a moment"


def fixtures_path(season: str) -> Path:
    """The season-level schedule page — the UNDERSCORE dir.

    ``_refresh_fbref_fixtures.py`` writes it there (``season.replace("-", "_")``).
    """
    return RAW_HTML_DIR / season.replace("-", "_") / "fixtures.html"


def reports_dir(season: str) -> Path:
    """Where match reports live — the HYPHEN dir, matching the parser.

    ``parse_all_player_stats._season_dir`` is ``RAW_HTML_DIR / season``. Writing
    the underscore form here would parse to nothing and still exit 0.
    """
    return RAW_HTML_DIR / season


def is_real_report(html: str) -> bool:
    """True only for an actual match report, not the Turnstile wall."""
    return bool(html) and len(html) > MIN_REAL_BYTES and "scorebox" in html and WALL_MARKER not in html


def parse_fixtures(html: str) -> dict[str, str]:
    """Map ``{match_hash: absolute report URL}`` for every published Serie A match.

    Scoped to the schedule tables on purpose. The page also carries match links
    for *other competitions* — a specimen held Premier League and Argentine Liga
    Profesional hrefs — and every one of them sits outside any ``<table>``, in
    nav markup. An unscoped regex over the page returns 364 ids across three
    leagues; this returns only Serie A's, and the slug check is the second belt:
    a foreign report parsed into the Serie A parquet would be silent poison.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, str] = {}

    for table in soup.find_all("table"):
        tid = table.get("id") or ""
        if not tid.startswith("sched_"):
            continue
        for row in table.find_all("tr"):
            cell = row.find("td", {"data-stat": "match_report"})
            if not cell:
                continue
            a = cell.find("a")
            href = a.get("href", "") if a else ""
            m = MATCH_HREF.match(href)
            if not m:
                continue
            if "Serie-A" not in href:  # belt two: never a foreign competition
                continue
            out[m.group(1)] = FBREF_BASE + href

    return out


def find_missing(season: str) -> dict[str, str]:
    """Published-but-absent reports: fixtures minus what is already on disk."""
    fx = fixtures_path(season)
    if not fx.exists():
        log.error(
            "No fixtures.html for %s at %s — run _refresh_fbref_fixtures.py first",
            season, fx,
        )
        return {}

    published = parse_fixtures(fx.read_text(encoding="utf-8", errors="replace"))
    if not published:
        log.error("fixtures.html holds no Serie A match links — schema break?")
        return {}

    have = {
        p.stem for p in reports_dir(season).glob("*.html")
        if not p.name.startswith(("fixtures", "stats_"))
    } if reports_dir(season).exists() else set()

    missing = {h: u for h, u in published.items() if h not in have}
    log.info(
        "%s: %d published, %d on disk, %d missing",
        season, len(published), len(have), len(missing),
    )
    return missing


def download(
    season: str,
    missing: dict[str, str],
    headless: bool,
    limit: int | None,
    _fetch=None,
) -> dict[str, int]:
    """Fetch each missing report into ``{hash}.html``. Returns a count dict.

    ``status`` is one of ``ok`` (something was fetched), ``walled`` (Cloudflare
    turned us away on the very first page — expected when headless), or
    ``nothing_to_do``.

    ``_fetch`` replaces the browser with a callable for tests; the batch logic
    (the wall bail-out, the backoff, what counts as a real page) is worth
    exercising without paying for a Chrome launch.
    """
    counts = {"downloaded": 0, "failed": 0, "status": "nothing_to_do"}
    if not missing:
        return counts

    out_dir = reports_dir(season)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = sorted(missing.items())
    if limit:
        items = items[:limit]

    if _fetch is not None:
        fetch = _fetch
        return _run_batch(out_dir, items, fetch, counts, season)

    from botasaurus.browser import Driver, browser

    @browser(
        headless=headless,
        block_images_and_css=False,
        wait_for_complete_page_load=False,  # waiting is handled below
        reuse_driver=True,  # solve Turnstile once, reuse the session for the batch
        output=None,
        create_error_logs=False,
        raise_exception=False,
    )
    def fetch(driver: Driver, url: str) -> str:
        try:
            driver.get(url, timeout=40)
        except Exception:
            pass  # navigation error — the page may still be usable

        html = ""
        for _ in range(25):
            try:
                html = driver.page_html
            except Exception:
                time.sleep(2)
                continue
            if "scorebox" in html:
                break
            time.sleep(2)
        else:
            # Return what we actually saw, NOT "". The wall page IS the
            # diagnosis — discarding it here makes a Turnstile block
            # indistinguishable from an empty response, and the caller then
            # grinds through all 69 matches proving the same point 69 times.
            return html

        # Scroll to trip the IntersectionObserver: the passing/defense/
        # possession/misc tables are lazy and absent from an unscrolled page.
        try:
            height = driver.run_js("return document.body.scrollHeight;")
            for pos in range(0, height + 800, 800):
                driver.run_js(f"window.scrollTo(0, {pos});")
                time.sleep(0.3)
            time.sleep(2)
            driver.run_js("window.scrollTo(0, 0);")
            time.sleep(1)
        except Exception:
            pass  # scroll failure is non-fatal

        try:
            return driver.page_html
        except Exception:
            return ""

    return _run_batch(out_dir, items, fetch, counts, season)


def _run_batch(out_dir: Path, items: list[tuple[str, str]], fetch, counts: dict, season: str) -> dict:
    """Walk the missing list with whatever ``fetch`` is handed in."""
    consecutive = 0
    for i, (match_hash, url) in enumerate(items):
        try:
            html = fetch(url) or ""
        except Exception as e:
            log.error("  %s raised: %s", match_hash, e)
            html = ""

        if is_real_report(html):
            (out_dir / f"{match_hash}.html").write_text(html, encoding="utf-8")
            counts["downloaded"] += 1
            counts["status"] = "ok"
            consecutive = 0
        else:
            counts["failed"] += 1
            consecutive += 1
            walled = WALL_MARKER in html
            log.warning(
                "  %s failed (%s, %d bytes)",
                match_hash, "Turnstile wall" if walled else "empty/short", len(html),
            )
            # The first page decides the batch: Turnstile gates the whole
            # session, so a wall here means every subsequent fetch walls too.
            # Don't burn the caller's 300s budget proving it 69 more times.
            if i == 0 and walled:
                counts["status"] = "walled"
                return counts

        if counts["downloaded"] and counts["downloaded"] % 20 == 0:
            log.info("  progress: %d/%d", counts["downloaded"], len(items))

        if consecutive >= 10:
            log.error("10 consecutive failures — stopping. Re-run to retry the rest.")
            break
        if consecutive >= 3:
            time.sleep(min(60 * consecutive, 300))
        elif i < len(items) - 1:
            time.sleep(RATE_LIMIT)

    log.info("%s: %d downloaded, %d failed", season, counts["downloaded"], counts["failed"])
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--season", default="2025-2026")
    ap.add_argument(
        "--headless", action="store_true",
        help="Measured as blocked by Turnstile — the weekly caller passes it. "
             "Probes one page, then reports rather than grinding.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Report the gap, fetch nothing.")
    ap.add_argument("--limit", type=int, default=None, help="Fetch at most N (use --limit 1 to verify a parse).")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    missing = find_missing(args.season)
    if not missing:
        log.info("Nothing missing for %s.", args.season)
        return 0

    if args.dry_run:
        log.info("DRY RUN — %d would be fetched, e.g. %s", len(missing), sorted(missing)[:3])
        return 0

    counts = download(args.season, missing, headless=args.headless, limit=args.limit)

    if counts["status"] == "walled":
        log.warning(
            "Cloudflare Turnstile turned the headless browser away — as measured on "
            "2026-07-16, this is expected and is not a code fault. %d Serie A reports "
            "are still missing. To recover them, run WITHOUT --headless (a visible "
            "browser passes in ~6s):\n"
            "    python3 -m scripts.data.scrape_fbref_missing --season %s",
            len(missing), args.season,
        )

    # Always 0: a Cloudflare wall is FBref's posture, not a broken job, and the
    # caller's own comment says the Sofascore fallback carries the results.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
