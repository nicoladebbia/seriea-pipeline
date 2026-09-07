"""The transfers cache served the same ghosts the squad cache was fixed for.

`scrape_squad_market_values` got `_prune_to_league()` on 2026-08-25 at both its
cache-return path and after its merge. `scrape_transfers` — the writer behind
the `/transfers` page — never got either, and it has the same two ingredients:
a team map that is a historical superset and an append-only `pd.concat` merge.

Measured 2026-09-07: transfers_2026_2027.parquet held 30 clubs for a 20-club
league. The 10 extras — Chievo (last in Serie A 2018-19), SPAL, Crotone,
Benevento, Brescia, Sampdoria, Salernitana, Empoli, Pisa, Verona — accounted
for 299 of 891 rows, so a third of the /transfers page described clubs that
are not in Serie A.

Both tests below fail against the unpruned version: the first because the
early `return cached_df` handed the ghost straight back, the second because
`pd.concat([cached_df, new_df])` carried it through the merge.
"""

from __future__ import annotations

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
    return tmp_path / "transfers_2026_2027.parquet"


def _seed(path, teams):
    pd.DataFrame([{"team": t, "player_name": f"{t} Player", "transfer_type": "in",
                   "window": "summer"} for t in teams]).to_parquet(path, index=False)


def test_a_complete_cache_is_pruned_before_it_is_returned(cache, monkeypatch):
    """The cache short-circuit returned every club it had ever seen."""
    _seed(cache, CURRENT | {GHOST})
    monkeypatch.setattr(tm.requests, "get",
                        lambda *a, **k: pytest.fail("should not re-scrape a complete cache"))

    df = tm.scrape_transfers(season="2026-2027", only_teams=CURRENT)

    assert set(df.team) == CURRENT, f"ghost survived the cache path: {sorted(set(df.team))}"
    # and it must be persisted -- /api/transfers reads the parquet directly, so
    # pruning only the return value leaves the ghosts on screen.
    assert set(pd.read_parquet(cache).team) == CURRENT


def test_a_rescrape_does_not_carry_a_relegated_club_through_the_merge(cache, monkeypatch):
    """Cache missing a current club -> scrape it, merge, and still drop the ghost."""
    _seed(cache, {"Inter", GHOST})

    class _Resp:
        text = "<html></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(tm.requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(tm, "_parse_transfers_page",
                        lambda html, team: ([{"team": team, "player_name": "New Signing",
                                              "transfer_type": "in"}], []))

    df = tm.scrape_transfers(season="2026-2027", only_teams=CURRENT)

    assert GHOST not in set(df.team), f"ghost survived the merge: {sorted(set(df.team))}"
    assert "Milan" in set(df.team), "the missing current club was not scraped"


def test_without_only_teams_the_historical_map_is_left_alone(cache, monkeypatch):
    """Backfilling an old season legitimately wants the historical superset."""
    _seed(cache.parent / "transfers_2018_2019.parquet", CURRENT | {GHOST})
    monkeypatch.setattr(tm.requests, "get",
                        lambda *a, **k: pytest.fail("should not re-scrape a complete cache"))

    df = tm.scrape_transfers(season="2018-2019", only_teams=None)

    assert GHOST in set(df.team), "only_teams=None must not prune"
