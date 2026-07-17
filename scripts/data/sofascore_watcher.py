"""Refresh the Premier League table from Sofascore's HTML tournament page.

Invoked by ``com.seriea-pipeline.sofascore-watcher.plist`` as
``python3 -m scripts.data.sofascore_watcher`` every 600s. ``KeepAlive`` is
false, so this is **one tick per invocation** — it scrapes, writes, and exits.
Nothing here loops or sleeps.

Why this module was rebuilt from its own output
-----------------------------------------------
The original was never git-added and was swept after 2026-06-01, while the
plist kept invoking it. Two artifacts it left behind pin its contract, and both
are now committed as fixtures rather than left to rot in gitignored ``data/``:

* ``data/upcoming/standings_premier_league.json`` — its last write. Its
  tz-aware ``generated_at`` lands 4s after the tick-3055 state write, which is
  what proves it authored the file.
* ``data/external/sofascore/.last_refresh.json`` — the heartbeat that
  ``web/app.py:2262`` (``/api/data-freshness``) reads to drive the global
  staleness banner.

Premier League only — and why
-----------------------------
Serie A's ``standings.json`` has a writer already:
``scripts/prediction/standings_generator.py`` derives it from
``matches.parquet`` with real ``form_last5`` and real home/away splits. This
scraper cannot produce either (see below), so a watcher writing that file every
10 minutes would degrade it. The EPL file has no other writer, and the oracle
pins it exactly, so that is the whole scope.

The consequence, stated plainly: the *serie_a* file gets no 10-minute HTML
freshness for its non-dashboard consumers (``ensemble_prediction_engine``,
``web/advisor.py``, ``generate_epl_supplementary`` read the file rather than
``web/app.py:_get_standings``). It stays pipeline-fresh. The dashboard is
unaffected — it scrapes the HTML live on every request.

The one deliberate deviation from the oracle: no home/away blocks
----------------------------------------------------------------
The oracle's per-team record carries ``home: {"played": 19, "wins": 0, ...}``.
That is **not reproduced here, on purpose.**

The tournament page serves exactly one table (``type: "total"``, confirmed
against the committed specimen) — there is no home/away tab in the payload, so
``scraper.sofascore_standings`` emits all-zero split blocks. The original
watcher then filled ``played`` by halving the total, which is only correct at
the moment it was observed: every team in the oracle sits at ``played=38``, the
one point in a season where home and away are *necessarily* 19 each. Mid-season
a team on 7 played could be 4/3 or 3/4, and halving invents the answer.

Worse, any split block at all corrupts a real record. ``web/app.py:7528``
overrides the *computed* ``_split_stats(home_matches)`` values with the file's
block whenever ``if s_home:`` passes — and a dict is truthy on *presence*, not
on whether its values mean anything. So the oracle's block makes the EPL team
page render "19 played, 0 won, 0 drawn, 0 lost", and an all-zero block would
render "0 played". Omitting the keys entirely makes ``entry.get("home", {})``
at ``web/app.py:7259`` return ``{}``, the override is skipped, and the real
computed record survives.

There was no faithful-and-correct option: every reproduction of the original's
shape is buggy. Omission is the only correct one. ``form_last5: ""`` *is* kept
— it matches the oracle and no consumer reads it.

Omission is safe because every reader of a split block defaults or guards.
Enumerated 2026-07-16 over both leagues' files, not sampled:

===========================  ==========================  ====================
Site                         Access                      Absent →
===========================  ==========================  ====================
``web/app.py:7038``          ``.get("home", {})`` then   zeros → the frontend
                             ``.get("wins", 0)``         renders ``-``
``web/app.py:7527``          ``if s_home:``              override skipped —
                                                         the real record lives
``web/app.py:600``           ``if pq_row.get("home")``   guarded
``teams.html:481``           ``st.home || {}`` then      ``-``
                             ``hr.played ? … : '-'``
``teams.html:745``           ``st.home ? … : '-'``       ``-``
``web/advisor.py:3011``      position/points/W-D-L/gd    never touches splits
``web/app.py:3637``          position/points/form        never touches splits
``generate_epl_supp``        ``form_last5``              never touches splits
``web/app.py:6880``          ``entry.get("team")``       never touches splits
===========================  ==========================  ====================

No unguarded ``entry["home"]`` exists, so no KeyError path opens when this
finally writes a real table at MW≥1. The visible change is a dash where the
original showed a fabricated number.

Note ``generate_epl_supplementary.py:1515`` is a *second writer* of this same
file (a read-modify-write that replaces the ``standings`` key to add
``position``). It preserves entries verbatim, so it neither restores nor
objects to the missing splits. The two writers racing is pre-existing and out
of scope here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import DATA_DIR
from scraper.sofascore_standings import live_standings_via_html

log = logging.getLogger(__name__)

#: Serie A is deliberately absent — standings_generator.py owns standings.json.
WATCHED_LEAGUES = ("premier_league",)

STANDINGS_DIR = DATA_DIR / "upcoming"
SOFASCORE_DIR = DATA_DIR / "external" / "sofascore"
TICK_PATH = SOFASCORE_DIR / ".watcher_tick.json"
HEARTBEAT_PATH = SOFASCORE_DIR / ".last_refresh.json"

#: Split blocks the HTML cannot source. Dropping them is what keeps
#: web/app.py:7528 from overriding a real record with a fake one.
_UNSOURCEABLE = ("home", "away")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write — a reader must never catch a half-written table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def next_tick() -> int:
    """Read-increment-return the tick counter.

    The counter is bookkeeping the heartbeat carries; a corrupt or missing file
    must not take the scrape down with it, so it restarts at 1 rather than
    raising. It was at 3055 when the original was swept.
    """
    try:
        tick = int(json.loads(TICK_PATH.read_text())["tick"]) + 1
    except (OSError, ValueError, TypeError, KeyError):
        # Missing file, unreadable, not JSON, or a non-numeric tick — each is
        # recoverable, and none is worth losing the scrape over.
        log.warning("Tick state unreadable at %s — restarting the count", TICK_PATH)
        tick = 1
    _write_json(TICK_PATH, {"tick": tick, "updated_at": _utcnow()})
    return tick


def shape_payload(scraped: dict[str, Any]) -> dict[str, Any]:
    """Reshape a scraper payload into the oracle's file shape.

    Pure — the only transform is dropping the unsourceable split blocks and
    stamping the write time.
    """
    return {
        "season": scraped["season"],
        "generated_at": _utcnow(),
        "source": scraped["_source"],
        "standings": {
            name: {k: v for k, v in row.items() if k not in _UNSOURCEABLE}
            for name, row in scraped["standings"].items()
        },
    }


def refresh_standings(league: str) -> tuple[str, int]:
    """Scrape and conditionally write one league's table.

    Returns ``(status, teams_written)`` where status is one of:

    * ``written``  — a real table replaced the file.
    * ``preseason`` — the page served a kicked-off-nothing table; see below.
    * ``failed``   — the scrape returned nothing. The retry, sentinel, negative
      cache and breaker all live in ``scraper.sofascore_standings``, so a falsy
      return means "no table this tick", never "write an empty one".

    **Why MW0 is refused.** An all-zero table is a pre-season placeholder, not
    a result: Sofascore publishes next season's fixture-less table months
    early. Measured 2026-07-16, the EPL page already served the 26/27 table
    (20 rows, every ``played=0``) while ``get_current_season()`` still returned
    ``2025-2026`` — the stamp comes off the clock, not the page (a known gap,
    pinned by ``test_season_is_stamped_from_the_clock_not_the_page``). Writing
    that would have destroyed the real 38-played 25/26 table *and* mislabelled
    the empty one with the wrong season.

    The rule is deliberately about the matchweek, not the season, because the
    season stamp is the broken field and cannot arbitrate. It costs nothing: an
    empty table carries no information, so declining to write it loses none.
    Once a season actually kicks off, MW≥1 and the write proceeds — replacing
    38 with 1 is correct there, because that is a real new season.
    """
    scraped = live_standings_via_html(league)
    if not scraped or not scraped.get("standings"):
        log.warning("HTML standings %s: no table this tick — file left untouched", league)
        return ("failed", 0)

    if not scraped.get("current_matchweek", 0):
        log.info(
            "HTML standings %s: page served an unplayed table (MW0) — pre-season, "
            "file left untouched", league,
        )
        return ("preseason", 0)

    payload = shape_payload(scraped)
    path = STANDINGS_DIR / f"standings_{league}.json"
    _write_json(path, payload)
    n = len(payload["standings"])
    log.info("Wrote %d %s teams to %s", n, league, path)
    return ("written", n)


def run_tick() -> dict[str, Any]:
    """One full tick: scrape every watched league, write the heartbeat.

    The heartbeat is the contract ``/api/data-freshness`` reads
    (``completed_at``/``started_at``, ``any_failure``, ``leagues``). Fields the
    original carried for work this module does not do — ``incidents_scraped``,
    ``matches_live``, the pre/post windows — are deliberately absent rather than
    written as zeros: a zero asserts "I looked and found none".
    """
    started = _utcnow()
    tick = next_tick()

    results = {league: refresh_standings(league) for league in WATCHED_LEAGUES}
    counts = {league: n for league, (_, n) in results.items()}

    # The off-season is a legitimate result, not a failure — the same call the
    # sibling fetch_upcoming_matches makes about an empty fixture list. Calling
    # it one would light /api/data-freshness's staleness banner red all summer.
    failed = [league for league, (status, _) in results.items() if status == "failed"]

    heartbeat = {
        "started_at": started,
        "tick": tick,
        "leagues": {
            league: {"status": status, "standings_teams": n}
            for league, (status, n) in results.items()
        },
        "any_failure": bool(failed),
        "did_standings_refresh": any(s == "written" for s, _ in results.values()),
        "standings_json_teams": counts,
        "standings_html_failure": bool(failed),
        "completed_at": _utcnow(),
    }
    _write_json(HEARTBEAT_PATH, heartbeat)
    return heartbeat


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    hb = run_tick()
    # Exit 0 even on a scrape failure: a 403 is Sofascore's mood, not a broken
    # job, and launchd would otherwise mark the whole watcher failed every time
    # the Cloudflare ban this HTML path exists to survive comes back.
    log.info("Tick %d done: %s", hb["tick"], hb["standings_json_teams"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
