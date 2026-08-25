"""EPL stopped reaching matches.parquet on 2026-03-22 — not by breaking, but by
construction.

`run_matchday_update` loops serie_a → premier_league. Both leagues shared one
fixtures cache and one diff basis, so Serie A refreshed the cache, EPL found it
FRESH inside the 6h window, loaded *Serie A's* fixtures, diffed them against
*Serie A's* match ids and detected nothing. Every run. No error, no warning —
just "No new matches detected", which is what a healthy run also says.

These tests reproduce that exact state. Each asserts its own true positive so it
cannot pass vacuously.
"""

import json

import pandas as pd
import pytest

from scripts.data import matchday_updater as mu

SEASON = "2026-2027"
SA_IDS = [1, 2, 3]
EPL_INGESTED = [101]
EPL_PENDING = [102, 103]


def _fixture(fid: int, home: str, away: str) -> dict:
    return {
        "id": fid,
        "status": {"type": "finished"},
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "roundInfo": {"round": 1},
    }


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """The production steady state: Serie A fully ingested, EPL lagging."""
    monkeypatch.setattr(mu, "SOFASCORE_DIR", tmp_path)

    # Serie A owns the unsuffixed cache — this is what the loop leaves behind.
    (tmp_path / f"fixtures_{SEASON.replace('-', '_')}.json").write_text(
        json.dumps([_fixture(i, f"SA{i}", f"SA{i}b") for i in SA_IDS])
    )
    # EPL's own cache, holding matches Serie A's cache knows nothing about.
    (tmp_path / f"fixtures_{SEASON.replace('-', '_')}_premier_league.json").write_text(
        json.dumps(
            [_fixture(i, f"E{i}", f"E{i}b") for i in EPL_INGESTED + EPL_PENDING]
        )
    )
    pd.DataFrame({"match_id": SA_IDS}).to_parquet(tmp_path / "player_match_stats.parquet")
    pd.DataFrame({"match_id": EPL_INGESTED}).to_parquet(
        tmp_path / "player_match_stats_premier_league.parquet"
    )

    # Any network call means the cache logic misfired; fail loudly rather than
    # silently hitting Sofascore from a test.
    def _no_network(*a, **k):
        raise AssertionError("detection tried to refresh the cache over the network")

    monkeypatch.setattr(mu, "_refresh_fixtures_cache", _no_network)
    return tmp_path


def test_the_mechanism_can_detect_a_new_match_at_all(sandbox):
    """True positive: without this, every 'detected N' assertion below is vacuous."""
    assert len(mu.detect_new_matches(season=SEASON, league="premier_league")) > 0


def test_serie_a_still_sees_its_own_cache_as_fully_ingested(sandbox):
    """The unsuffixed file stays Serie A's — the fix must not move it."""
    assert mu.detect_new_matches(season=SEASON, league="serie_a") == []


def test_epl_detects_its_own_matches_when_serie_a_wrote_the_shared_cache(sandbox):
    """THE regression. Returned [] on every run before 2026-08-25."""
    found = {f["id"] for f in mu.detect_new_matches(season=SEASON, league="premier_league")}
    assert found == set(EPL_PENDING)


def test_epl_does_not_redetect_matches_already_in_its_own_parquet(sandbox):
    """Pins bug A: with the Serie A file as the diff basis, 101 looks new."""
    found = {f["id"] for f in mu.detect_new_matches(season=SEASON, league="premier_league")}
    assert EPL_INGESTED[0] not in found


def test_existing_ids_come_from_the_league_s_own_parquet(sandbox):
    sa = mu._get_existing_sofascore_match_ids("serie_a")
    epl = mu._get_existing_sofascore_match_ids("premier_league")
    assert sa == set(SA_IDS)
    assert epl == set(EPL_INGESTED)
    assert not (sa & epl), "league parquets hold disjoint ids"


def test_each_league_gets_its_own_fixtures_cache(sandbox):
    sa = mu._fixtures_cache_path(SEASON, "serie_a")
    epl = mu._fixtures_cache_path(SEASON, "premier_league")
    assert sa != epl
    assert sa.name == f"fixtures_{SEASON.replace('-', '_')}.json"
    assert "premier_league" in epl.name


def test_serie_a_keeps_the_unsuffixed_filenames(sandbox):
    """Matches scripts/data/scrape_sofascore.py, which writes the same files."""
    assert mu._league_suffix("serie_a") == ""
    assert mu._league_suffix("premier_league") == "_premier_league"
    assert mu._sofascore_parquet("player_match_stats", "serie_a").name == (
        "player_match_stats.parquet"
    )


def test_merge_writes_to_the_league_s_own_stat_parquets(sandbox, monkeypatch):
    """Bug C: fixing detection alone would route EPL rows into Serie A's files."""
    monkeypatch.setattr(
        mu, "extract_player_rows",
        lambda md, fx, sn: [{"match_id": 102, "player_id": 9, "season": sn}],
    )
    monkeypatch.setattr(mu, "extract_team_stats_rows", lambda md, fx, sn: [])
    monkeypatch.setattr(mu, "extract_shotmap_rows", lambda md, fx, sn: [])

    mu.merge_to_sofascore_parquets([({}, {})], SEASON, league="premier_league")

    epl = pd.read_parquet(sandbox / "player_match_stats_premier_league.parquet")
    sa = pd.read_parquet(sandbox / "player_match_stats.parquet")
    assert 102 in set(epl["match_id"]), "EPL row must land in the EPL parquet"
    assert 102 not in set(sa["match_id"]), "EPL row must not contaminate Serie A"


def test_incidents_remain_a_deliberately_shared_file(sandbox):
    """Not an oversight: incidents are keyed by globally-unique Sofascore ids and
    live in ONE merged file holding both leagues (3388 SA + 3378 EPL on disk).
    Suffixing it would strand half the history."""
    import inspect

    assert "league" not in inspect.signature(mu.fetch_and_merge_incidents).parameters


def test_a_failed_refresh_falls_back_to_the_cache_on_disk(sandbox, monkeypatch):
    """Sofascore 403s for hours-to-days. Before 2026-08-25 a stale cache plus a
    banned API meant detection returned [] — the league went blind for the whole
    ban even though usable fixtures sat on disk."""
    monkeypatch.setattr(mu, "_is_cache_stale", lambda p: True)
    monkeypatch.setattr(mu, "_refresh_fixtures_cache", lambda *a, **k: [])
    monkeypatch.setattr(mu.asyncio, "run", lambda coro: coro)

    found = {f["id"] for f in mu.detect_new_matches(season=SEASON, league="premier_league")}
    assert found == set(EPL_PENDING), "must fall back to the on-disk cache"


def test_a_failed_refresh_with_no_cache_still_returns_empty(sandbox, monkeypatch):
    """True positive for the test above: the fallback must not invent fixtures."""
    monkeypatch.setattr(mu, "_is_cache_stale", lambda p: True)
    monkeypatch.setattr(mu, "_refresh_fixtures_cache", lambda *a, **k: [])
    monkeypatch.setattr(mu.asyncio, "run", lambda coro: coro)
    (sandbox / f"fixtures_{SEASON.replace('-', '_')}_premier_league.json").unlink()

    assert mu.detect_new_matches(season=SEASON, league="premier_league") == []
