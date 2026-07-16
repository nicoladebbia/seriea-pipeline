"""Tests for the standings scraper extracted from web/app.py.

The code is a verbatim move (proven byte-identical under the rename map), so
these tests do not assert *new* behaviour — they pin the behaviour that was
already there and previously had no coverage at all, before a second caller
(``scripts.data.sofascore_watcher``) starts depending on it.

Two ends again, as with live_sofascore:

* **Input** — a real Sofascore tournament payload
  (``tests/fixtures/sofascore/tournament_standings_serie_a.json``, the untouched
  ``props.pageProps.standings`` subtree captured 2026-07-16).
* **Output** — ``data/upcoming/standings_premier_league.json``, which the
  original watcher wrote at tick 3055. Gitignored, so it skips on a fresh clone.

Nothing here makes a network call.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scraper import sofascore_standings as ss

FIXTURES = Path(__file__).parent / "fixtures" / "sofascore"
UPCOMING = Path(__file__).resolve().parents[1] / "data" / "upcoming"

SPECIMEN = json.loads((FIXTURES / "tournament_standings_serie_a.json").read_text())


@pytest.fixture(autouse=True)
def _clean_module_state():
    """The cache and breaker are module-level dicts shared by every caller."""
    ss._html_standings_cache.clear()
    ss._html_health.clear()
    yield
    ss._html_standings_cache.clear()
    ss._html_health.clear()


@pytest.fixture(autouse=True)
def slept(monkeypatch):
    """Record the retry backoff instead of serving it.

    Only sofascore_get_retry uses the module-level `_time` (the scraper takes
    its clock from a function-local `import time as _t`), so replacing it here
    cannot skew anything else — and it turns 15s of real sleeping into an
    assertion about the schedule.
    """
    calls: list[float] = []
    monkeypatch.setattr(ss, "_time", SimpleNamespace(sleep=calls.append))
    return calls


class _Resp:
    def __init__(self, text="", status=200):
        self.text = text
        self.status_code = status


def _page(standings) -> str:
    """Wrap a standings subtree the way the real page embeds it."""
    blob = json.dumps({"props": {"pageProps": {"standings": standings}}})
    return f'<html><body><script id="__NEXT_DATA__" type="application/json">{blob}</script></body></html>'


@pytest.fixture
def serve(monkeypatch):
    """Serve a fixed response instead of hitting Sofascore."""

    def _serve(resp):
        class _Session:
            headers: dict = {}

            def get(self, url, timeout=10):
                if isinstance(resp, Exception):
                    raise resp
                return resp

        monkeypatch.setattr("curl_cffi.requests.Session", lambda **kw: _Session())

    return _serve


# --------------------------------------------------------------------------
# The real specimen through the real parser
# --------------------------------------------------------------------------


def test_parses_the_real_tournament_payload(serve):
    serve(_Resp(_page(SPECIMEN)))
    out = ss.live_standings_via_html("serie_a")

    assert len(out["standings"]) == 20
    assert out["_source"] == "sofascore_html"
    assert out["league"] == "serie_a"

    inter = out["standings"]["Inter"]
    assert inter["gd"] == inter["gf"] - inter["ga"]
    assert inter["league"] == "serie_a"
    # HTML exposes neither form nor home/away splits; the dashboard splices
    # those from the parquet. Emitting placeholders is the contract.
    assert inter["form_last5"] == ""
    assert inter["home"]["ppg"] == 0.0


def test_success_resets_the_breaker(serve):
    ss.html_health_now("serie_a")["consecutive_failures"] = 2
    serve(_Resp(_page(SPECIMEN)))
    ss.live_standings_via_html("serie_a")
    assert ss.html_health_now("serie_a")["consecutive_failures"] == 0
    assert not ss.html_is_broken("serie_a")


# --------------------------------------------------------------------------
# Known gaps — pinned as behaviour, NOT fixed here (this is a pure move)
# --------------------------------------------------------------------------


def test_season_is_stamped_from_the_clock_not_the_page(serve):
    """Known gap #1. The specimen's own table says "Serie A 26/27", but the
    payload's season comes from get_current_season() (boundary Aug 1).

    Captured 2026-07-16, when those disagreed. This asserts the CURRENT
    behaviour so a future fix has to change the test deliberately.
    """
    assert SPECIMEN[0]["name"] == "Serie A 26/27"  # the page states the truth
    serve(_Resp(_page(SPECIMEN)))
    out = ss.live_standings_via_html("serie_a")
    assert out["season"] == ss.get_current_season()  # ...and it is ignored


def test_sentinel_does_not_catch_a_partial_table(serve):
    """Known gap #2. The sentinel asserts one team is present, so a table
    missing rows still passes. The original watcher's stored EPL file holds
    14 of 20 teams with its sentinel (Arsenal) intact — this is how.
    """
    partial = json.loads(json.dumps(SPECIMEN))
    keep = [r for r in partial[0]["rows"] if r["team"]["name"] in ("Inter", "Juventus")]
    partial[0]["rows"] = keep

    serve(_Resp(_page(partial)))
    out = ss.live_standings_via_html("serie_a")

    assert len(out["standings"]) == 2  # 2 of 20 accepted...
    assert not ss.html_is_broken("serie_a")  # ...and the breaker stayed shut


# --------------------------------------------------------------------------
# Sentinel + schema breaks
# --------------------------------------------------------------------------


def test_missing_sentinel_trips_a_schema_break(serve):
    without_inter = json.loads(json.dumps(SPECIMEN))
    without_inter[0]["rows"] = [
        r for r in without_inter[0]["rows"] if r["team"]["name"] != "Inter"
    ]

    serve(_Resp(_page(without_inter)))
    assert ss.live_standings_via_html("serie_a") == {}
    assert ss.html_health_now("serie_a")["schema_break"] is True
    assert ss.html_is_broken("serie_a")  # schema break alone is enough


@pytest.mark.parametrize(
    "payload,marker",
    [
        ("<html>no next data here</html>", "NEXT_DATA script not found"),
        (_page([]), "standings list missing in NEXT_DATA"),
        (_page([{"rows": []}]), "rows array empty"),
    ],
)
def test_schema_breaks_are_distinguished_from_transport(serve, payload, marker):
    serve(_Resp(payload))
    assert ss.live_standings_via_html("serie_a") == {}
    h = ss.html_health_now("serie_a")
    assert h["schema_break"] is True
    assert marker in h["last_error"]


def test_the_2026_06_hoisting_fallback_still_parses(serve):
    """Sofascore hoisted initialProps.* onto pageProps; the specimen confirms
    the flat path is what's live. The nested path is legacy but supported.
    """
    blob = json.dumps({"props": {"pageProps": {"initialProps": {"standings": SPECIMEN}}}})
    serve(_Resp(f'<script id="__NEXT_DATA__">{blob}</script>'))
    assert len(ss.live_standings_via_html("serie_a")["standings"]) == 20


# --------------------------------------------------------------------------
# Transport, breaker, cache
# --------------------------------------------------------------------------


def test_transport_failure_is_not_a_schema_break(serve, slept):
    """A 403 must not set schema_break — that flag means "Sofascore changed
    shape" and drives a different alert in the dashboard's health endpoint.

    A 403 is also the exact symptom of the Cloudflare IP ban this whole HTML
    path exists to survive, so it must be retried, not treated as a hard status.
    """
    serve(_Resp("", status=403))
    assert ss.live_standings_via_html("serie_a") == {}
    h = ss.html_health_now("serie_a")
    assert h["schema_break"] is False
    assert h["consecutive_failures"] == 1
    assert slept == [1, 2], "403 must back off 1s then 2s before giving up"


def test_a_hard_status_is_not_retried(serve, slept):
    """404 is not a Cloudflare mood — retrying it just wastes the tick."""
    serve(_Resp("", status=404))
    assert ss.live_standings_via_html("serie_a") == {}
    assert slept == []


def test_breaker_opens_only_after_the_threshold(serve):
    serve(_Resp("", status=500))
    for i in range(1, ss.HTML_FAILURE_THRESHOLD + 1):
        ss._html_standings_cache.clear()  # bypass the 30s negative cache
        ss.live_standings_via_html("serie_a")
        assert ss.html_is_broken("serie_a") == (i >= ss.HTML_FAILURE_THRESHOLD)


def test_unconfigured_league_never_scrapes(serve):
    def boom(**kw):  # pragma: no cover - must not run
        raise AssertionError("scraped an unknown league")

    monkey = pytest.MonkeyPatch()
    monkey.setattr("curl_cffi.requests.Session", boom)
    try:
        assert ss.live_standings_via_html("ligue_1") == {}
        assert ss.html_health_now("ligue_1")["last_error"] == "league not configured"
    finally:
        monkey.undo()


def test_success_is_cached_and_the_caller_cannot_mutate_it(serve):
    serve(_Resp(_page(SPECIMEN)))
    first = ss.live_standings_via_html("serie_a")
    first["standings"].clear()  # a caller mangling its copy...

    serve(_Resp("", status=500))  # ...and the network now failing
    second = ss.live_standings_via_html("serie_a")
    assert len(second["standings"]) == 20  # cache survived both


def test_failures_are_negative_cached_so_sofascore_is_not_hammered(serve):
    calls = {"n": 0}

    class _Session:
        headers: dict = {}

        def get(self, url, timeout=10):
            calls["n"] += 1
            return _Resp("", status=403)

    monkey = pytest.MonkeyPatch()
    monkey.setattr("curl_cffi.requests.Session", lambda **kw: _Session())
    try:
        for _ in range(5):
            assert ss.live_standings_via_html("serie_a") == {}
        # 3 in-call retries on the first attempt, then the {} marker is cached.
        assert calls["n"] == 3, f"re-hammered Sofascore: {calls['n']} requests"
    finally:
        monkey.undo()


# --------------------------------------------------------------------------
# Output oracle — the original watcher's own file
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (UPCOMING / "standings_premier_league.json").exists(),
    reason="oracle unavailable: data/ is gitignored",
)
def test_per_team_shape_matches_what_the_watcher_wrote(serve):
    """The stored EPL file was written by the (lost) watcher from this
    scraper's payload — its tz-aware generated_at lands 4s after the tick 3055
    state write. So its per-team keys are this function's output contract.
    """
    stored = json.loads((UPCOMING / "standings_premier_league.json").read_text())
    assert stored["source"] == "sofascore_html"

    serve(_Resp(_page(SPECIMEN)))
    produced = ss.live_standings_via_html("serie_a")["standings"]

    stored_team = next(iter(stored["standings"].values()))
    produced_team = next(iter(produced.values()))
    assert set(stored_team) == set(produced_team)
    assert set(stored_team["home"]) == set(produced_team["home"])
