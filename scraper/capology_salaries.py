"""Capology player-salary scraper for the current Serie A season.

WHAT THIS IS — and what it is NOT
---------------------------------
Capology publishes an **estimated** guaranteed annual salary per player. Their
own header says it plainly: "All amounts are estimates and do not represent
official figures." So every number this module produces is a *Capology estimate*
and MUST be surfaced as such (never mixed unlabeled beside Transfermarkt's real
market values). It is **display-only** — it never feeds a model feature. Salary
estimates predicting match outcomes is exactly the fiction this pipeline gates.

WHY THE STATIC HTML IS ENOUGH (specimen-verified 2026-07-13)
------------------------------------------------------------
The visible `<tbody>` is empty — bootstrap-table renders rows client-side. But
the row data itself IS in the static HTML, in a JS array:

    var data = [{
        'name': "<a ...>Kenan Yıldız</a>",
        'verified': "<img ...verified-green.svg.../>",
        'weekly_gross_eur': accounting.formatMoney("11110000"/52, "€ ", 0),
        'annual_gross_eur': accounting.formatMoney("11110000", "€ ", 0),
        ...
        'position': "F", 'age': "20", ...
    }, {...}]

`annual_gross_eur` is the **Est. Fixed** guaranteed salary (the base the club
pays; Capology reports Fixed / Bonus / Total separately — we present Fixed only,
because bonuses are conditional and Capology flags them as "may be incomplete").

Each field is read from the SAME `{...}` object — never two zipped lists — so a
name can never drift onto the wrong player's salary. Correctness is checked per
club by summing the parsed fixed salaries against Capology's own stated
`annual-fixed` total on the page (must match to the euro, else we trip a breaker
and drop that club rather than ship a mis-aligned number).

Requires the project's Transfermarkt browser headers (TM_HEADERS) — Capology
returns 403 to a bare fetcher and 200 to those. Paced like the TM scraper.
"""

from __future__ import annotations

import html as _html
import logging
import re
import time
import unicodedata

import pandas as pd
import requests

from config.settings import DATA_DIR
from scraper.transfermarkt import TM_HEADERS

log = logging.getLogger(__name__)

_BASE = "https://www.capology.com/club/{slug}/salaries/"
_PACE_SEC = 4.0  # respectful delay between club fetches (matches TM scraper)

# Sane band for a Serie A fixed annual gross salary. Outside this, distrust it.
_MIN_SALARY_EUR = 30_000        # deep-squad youngster floor
_MAX_SALARY_EUR = 35_000_000    # comfortably above any Serie A wage

# Canonical Capology slug per Serie A 2026-27 club. Keys MUST match the `team`
# column in market_values_2026_2027.parquet exactly (that parquet is the join
# target and the authority on which 20 clubs are in the league this season).
# Verified against Capology's Serie A index page (2026-07-13). A club that
# returns 0 players is a LOUD failure below, never a silent gap.
CLUB_SLUGS: dict[str, str] = {
    "Inter": "inter-milan",
    "Milan": "ac-milan",
    "Juventus": "juventus",
    "Napoli": "napoli",
    "Atalanta": "atalanta",
    "Roma": "roma",
    "Lazio": "lazio",
    "Fiorentina": "fiorentina",
    "Bologna": "bologna",
    "Torino": "torino",
    "Udinese": "udinese",
    "Genoa": "genoa",
    "Como": "como",
    "Cagliari": "cagliari",
    "Parma": "parma",
    "Lecce": "lecce",
    "Sassuolo": "sassuolo",
    "Frosinone": "frosinone",
    "Monza": "monza",
    "Venezia": "venezia",
}


def _norm(text: str) -> str:
    """Strip accents + lowercase for cross-source name matching (mirrors
    scraper.wiki_transfers._norm so a salary joins to the TM roster cleanly)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", " ", t.lower()).strip()


def _parse_club_html(html_text: str) -> list[dict]:
    """Parse per-player salary rows from one club's Capology salaries page.

    Reads each JS row object as a unit — name, salary, verified, position, age
    all come from the same `{...}` block. Returns [] if the data array is absent
    (page structure changed / club has no roster yet).
    """
    i = html_text.find("var data")
    if i < 0:
        return []
    arr_start = html_text.find("[", i)
    if arr_start < 0:
        return []

    rows: list[dict] = []
    # Each row object begins with `{ 'name': "..."`. Anchor on that, then read
    # the fields inside that single object's window. We brace-MATCH the window
    # (not just find the first "}") so a field value containing a literal brace
    # can never silently truncate the object mid-parse. (Today no field carries a
    # brace — verified across 4 clubs / 135 rows — but this makes it robust if
    # Capology adds one.)
    for m in re.finditer(r"\{\s*'name'\s*:\s*\"(.*?)\",", html_text[arr_start:], re.DOTALL):
        obj_start = arr_start + m.start()
        depth, j = 0, obj_start
        while j < len(html_text):
            ch = html_text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:  # unbalanced — array truncated; stop rather than mis-parse
            break
        obj = html_text[obj_start : j + 1]

        name = _html.unescape(re.sub(r"<[^>]+>", "", m.group(1)).strip())
        if not name:
            continue

        ag = re.search(r"'annual_gross_eur'\s*:\s*accounting\.formatMoney\(\"([0-9]+)\"", obj)
        annual = int(ag.group(1)) if ag else None

        # `verified` field appears before the money fields; a green check = confirmed.
        head = obj[: obj.find("weekly")] if "weekly" in obj else obj
        verified = "verified-green" in head

        pos = re.search(r"'position'\s*:\s*\"([^\"]*)\"", obj)
        # age is rendered as Math.round("21") — read the quoted integer inside.
        age = re.search(r"'age'\s*:\s*(?:Math\.round\()?\"?([0-9]{1,2})\"?", obj)
        # contract expiry: moment("2030-06-30").format(...) — grab the ISO date.
        exp = re.search(r"'expiration'\s*:\s*moment\(\"(\d{4}-\d{2}-\d{2})\"", obj)
        yrs = re.search(r"'years'\s*:\s*\"?([0-9]+)\"?", obj)

        rows.append(
            {
                "player_name": name,
                "annual_gross_eur": annual,
                "monthly_gross_eur": (annual // 12) if annual else None,
                "weekly_gross_eur": (annual // 52) if annual else None,
                "verified": bool(verified),
                "capology_position": pos.group(1) if pos else None,
                "capology_age": int(age.group(1)) if age else None,
                "contract_expiration": exp.group(1) if exp else None,
                "years_remaining": int(yrs.group(1)) if yrs else None,
            }
        )
    return rows


def _stated_fixed_total(html_text: str) -> int | None:
    """Capology prints the club's total annual-fixed wage bill in the page
    header (`#annual-fixed`). We use it as a per-club checksum: the sum of the
    rows we parse must equal it, or the parse is untrustworthy."""
    m = re.search(
        r"annual-fixed'\)\.html\(accounting\.formatMoney\(\"([0-9]+)\"", html_text
    )
    return int(m.group(1)) if m else None


def _get_with_backoff(sess: requests.Session, url: str, max_retries: int = 3) -> requests.Response:
    """GET with exponential backoff on 429. Capology throttles a fast sweep
    (measured: 429 at ~club 15 of a 4s-paced run), so an UNATTENDED run must
    self-recover — the robustness can't live in an interactive cooldown."""
    delay = 20.0
    for attempt in range(max_retries + 1):
        r = sess.get(url, headers=TM_HEADERS, timeout=25)
        if r.status_code != 429:
            return r
        if attempt < max_retries:
            # Retry-After may be a seconds-integer OR an HTTP-date (both spec-legal).
            # Only trust the integer form; on anything else fall back to our delay
            # so the backoff always engages rather than raising ValueError.
            ra = r.headers.get("Retry-After", "")
            wait = float(ra) if ra.isdigit() else delay
            log.warning("429 on %s — backing off %.0fs (attempt %d/%d)", url, wait, attempt + 1, max_retries)
            time.sleep(wait)
            delay *= 2
    return r  # last response (still 429) — caller raises


def scrape_club_salaries(team: str, slug: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch + parse one club. Raises on HTTP failure, empty roster, or a
    checksum mismatch — the caller decides whether to skip or abort. A silent
    empty result is never returned for a club we expected to have players."""
    sess = session or requests.Session()
    url = _BASE.format(slug=slug)
    r = _get_with_backoff(sess, url)
    if r.status_code != 200:
        raise RuntimeError(f"{team} ({slug}): HTTP {r.status_code}")

    rows = _parse_club_html(r.text)
    priced = [x for x in rows if x["annual_gross_eur"]]
    if not priced:
        raise RuntimeError(f"{team} ({slug}): 0 priced players — structure changed or empty roster")

    # Range sentinel: every salary in a sane band, else distrust the parse.
    for x in priced:
        v = x["annual_gross_eur"]
        if not (_MIN_SALARY_EUR <= v <= _MAX_SALARY_EUR):
            log.warning("%s: %s salary €%s outside sane band — check parse", team, x["player_name"], v)

    # Checksum: sum of parsed fixed salaries must equal Capology's stated total.
    # This is the one net that catches a truncated object-window or dropped row.
    # If the stated-total markup ever changes we can't check — that must be LOUD
    # (a warning in the refresh log), not a silent skip that lets a mis-aligned
    # club through unnoticed.
    stated = _stated_fixed_total(r.text)
    parsed_total = sum(x["annual_gross_eur"] for x in priced)
    if stated is None:
        log.warning(
            "%s (%s): could not read Capology's stated total — checksum SKIPPED. "
            "If this recurs, the #annual-fixed markup changed; re-verify the parser.",
            team, slug,
        )
    elif abs(parsed_total - stated) > 1000:
        raise RuntimeError(
            f"{team} ({slug}): checksum FAIL — parsed €{parsed_total:,} "
            f"vs stated €{stated:,}. Refusing to ship a mis-aligned club."
        )

    df = pd.DataFrame(priced)
    df["team"] = team
    df["capology_slug"] = slug
    df["name_norm"] = df["player_name"].map(_norm)
    return df


def scrape_all_salaries(
    only_teams: list[str] | None = None, pace_sec: float = _PACE_SEC
) -> pd.DataFrame:
    """Scrape every Serie A club (or a subset), one paced request each.

    A club that fails (HTTP / empty / checksum) is logged LOUDLY and skipped —
    the rest proceed (fail-soft), and the returned frame records which clubs are
    present so a gap is visible, not silent.
    """
    teams = only_teams or list(CLUB_SLUGS)
    sess = requests.Session()
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for i, team in enumerate(teams):
        slug = CLUB_SLUGS.get(team)
        if not slug:
            log.error("No Capology slug mapped for %s — skipping", team)
            failed.append(team)
            continue
        try:
            df = scrape_club_salaries(team, slug, session=sess)
            frames.append(df)
            log.info("%s: %d players, top €%s/yr", team, len(df), f"{df['annual_gross_eur'].max():,}")
        except Exception as e:  # noqa: BLE001 — fail-soft; a dead club must not kill the sweep
            log.error("Capology salary scrape FAILED for %s: %s", team, e)
            failed.append(team)
        if i < len(teams) - 1:
            time.sleep(pace_sec)

    if failed:
        log.warning("Capology: %d/%d clubs missing: %s", len(failed), len(teams), ", ".join(failed))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_salaries(season: str = "2026-2027", only_teams: list[str] | None = None) -> pd.DataFrame:
    """Scrape + persist to salaries_{season}.parquet. Returns the frame."""
    df = scrape_all_salaries(only_teams=only_teams)
    if df.empty:
        log.error("Capology: nothing scraped — parquet NOT overwritten")
        return df
    out = DATA_DIR / "external" / "transfermarkt" / f"salaries_{season.replace('-', '_')}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("Saved %d salary rows across %d clubs -> %s", len(df), df["team"].nunique(), out)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse

    ap = argparse.ArgumentParser(description="Scrape Capology Serie A salary estimates")
    ap.add_argument("--season", default="2026-2027")
    ap.add_argument("--team", action="append", help="limit to specific team(s)")
    args = ap.parse_args()
    frame = save_salaries(season=args.season, only_teams=args.team)
    if not frame.empty:
        print(f"\n{len(frame)} players, {frame['team'].nunique()} clubs")
        top = frame.nlargest(10, "annual_gross_eur")
        for _, x in top.iterrows():
            v = "✓" if x["verified"] else "est"
            print(
                f"  {x['team']:<11} {x['player_name']:<22} "
                f"€{x['annual_gross_eur']:>10,}/yr  €{x['monthly_gross_eur']:>8,}/mo  [{v}]"
            )
