"""The per-season squad cache froze, then served ghosts.

`scrape_squad_market_values` returned early whenever every club was already in
the cache — no TTL — so a squad file stopped updating the day it was first
completed and sat through an entire transfer window unchanged. And because the
merge was a bare append against a team map that is a historical superset, the
file accumulated clubs that had long since left the league.

Measured 2026-08-25: market_values_2026_2027.parquet held 29 clubs for a
20-club league. The 9 extras — Chievo (last in Serie A 2018-19), SPAL, Crotone,
Benevento, Brescia, Sampdoria, Salernitana, Empoli, Verona — were 256 players
rendered on /rosters as current squad members.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import pytest

from scraper import transfermarkt as tm

CURRENT = {"Inter", "Milan"}
GHOST = "Chievo"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(tm, "TM_DIR", tmp_path)
    monkeypatch.setattr(tm, "_get_league_teams",
                        lambda league: {"Inter": ("inter", 46), "Milan": ("milan", 5),
                                        GHOST: ("chievo", 100)})
    monkeypatch.setattr(tm, "_league_cache_prefix", lambda league: "")
    monkeypatch.setattr(tm.time, "sleep", lambda s: None)
    return tmp_path / "market_values_2026_2027.parquet"


def _seed(path, teams, *, age_hours=0.0):
    pd.DataFrame([{"team": t, "player_name": f"{t} Player", "market_value_eur": 1.0}
                  for t in teams]).to_parquet(path, index=False)
    if age_hours:
        old = time.time() - age_hours * 3600
        os.utime(path, (old, old))


def _serve(monkeypatch, calls):
    """Record which clubs get fetched; return one synthetic player each."""
    class _R:
        text = "<html></html>"
        def raise_for_status(self): return None

    monkeypatch.setattr(tm.requests, "get", lambda url, **k: (calls.append(url), _R())[1])
    monkeypatch.setattr(tm, "_parse_squad_page",
                        lambda html, team: [{"team": team, "player_name": f"{team} Fresh",
                                             "market_value_eur": 2.0}])


def test_a_complete_but_stale_cache_is_rescraped(cache, monkeypatch):
    """THE freeze: before the fix this returned the cache and never refetched."""
    _seed(cache, CURRENT, age_hours=tm.SQUAD_CACHE_MAX_AGE_HOURS + 1)
    calls: list[str] = []
    _serve(monkeypatch, calls)

    out = tm.scrape_squad_market_values("2026-2027", only_teams=CURRENT)
    assert len(calls) == len(CURRENT), "a stale cache must refetch every club"
    assert set(out["player_name"]) == {f"{t} Fresh" for t in CURRENT}


def test_a_complete_and_fresh_cache_is_not_rescraped(cache, monkeypatch):
    """True positive for the test above — without it, an always-rescrape
    implementation would pass and hammer Transfermarkt every run."""
    _seed(cache, CURRENT, age_hours=1)
    calls: list[str] = []
    _serve(monkeypatch, calls)

    out = tm.scrape_squad_market_values("2026-2027", only_teams=CURRENT)
    assert calls == [], "a fresh cache must not touch the network"
    assert set(out["team"]) == CURRENT


def test_a_rescrape_replaces_a_squad_instead_of_doubling_it(cache, monkeypatch):
    """The old merge was a bare concat. Safe only while re-scrapes were
    impossible; with a TTL it would double every squad on the first refresh."""
    _seed(cache, CURRENT, age_hours=tm.SQUAD_CACHE_MAX_AGE_HOURS + 1)
    _serve(monkeypatch, [])

    out = tm.scrape_squad_market_values("2026-2027", only_teams=CURRENT)
    assert len(out) == len(CURRENT), f"expected one row per club, got {len(out)}"
    assert not out.duplicated(subset=["team", "player_name"]).any()


def test_clubs_no_longer_in_the_league_are_pruned(cache, monkeypatch):
    """Chievo et al. must not survive into a 2026-2027 squad file."""
    _seed(cache, CURRENT | {GHOST}, age_hours=1)
    _serve(monkeypatch, [])

    out = tm.scrape_squad_market_values("2026-2027", only_teams=CURRENT)
    assert GHOST not in set(out["team"])
    assert set(out["team"]) == CURRENT

    on_disk = pd.read_parquet(cache)
    assert GHOST not in set(on_disk["team"]), "the prune must reach the file, not just the return"


def test_without_only_teams_the_historical_map_is_left_alone(cache, monkeypatch):
    """only_teams=None means the caller wants the full historical scope, so
    pruning would be wrong. Guards the prune from over-reaching."""
    _seed(cache, CURRENT | {GHOST}, age_hours=1)
    _serve(monkeypatch, [])

    out = tm.scrape_squad_market_values("2026-2027", only_teams=None)
    assert GHOST in set(out["team"])
