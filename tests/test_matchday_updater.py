"""What `update_matches_parquet` must get right about the ground-truth key.

This module is the live Sofascore -> matches.parquet writer. Until 2026-07-17 it
keyed every row it appended on `str(fixture["id"])` — Sofascore's fixture id —
so matches.parquet ended up holding 94 numeric keys beside 15,795 canonical
`{date}_{home}_{away}` ones. Two incompatible formats in a single id column.

The damage was invisible from inside the file: same-file joins still worked
because both sides carried the same wrong key, and nothing in this repo parses a
match_id back into its parts. It only showed up across sources — and worse, the
numeric keys silently collide with `data/external/sofascore/*.parquet`, which is
natively keyed by exactly those Sofascore ids, so a stray merge would join for
those 94 rows and for no others.

Migrating the data without pinning the writer would have reverted on the next
matchday run, which is the whole reason these tests exist.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts.data import matchday_updater as mu


def _fixture(fid: int, home: str, away: str, when: datetime | None) -> dict:
    return {
        "id": fid,
        "startTimestamp": int(when.timestamp()) if when else 0,
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "homeScore": {"current": 1},
        "awayScore": {"current": 0},
        "roundInfo": {"round": 27},
    }


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Never let a test touch the real parquet.

    A test writing its fixture into real data has already cost this project once
    (the bankroll drift that turned out to be a test leak), so the redirect is a
    fixture rather than something each test remembers to do.
    """
    p = tmp_path / "matches.parquet"
    monkeypatch.setattr(mu, "MATCHES_PARQUET", p)
    return p


KICKOFF = datetime(2026, 2, 27, 19, 45, tzinfo=timezone.utc)


def test_the_key_is_canonical_not_the_sofascore_fixture_id(isolated):
    """The regression itself: 13981687 must never land in the id column."""
    pairs = [({}, _fixture(13981687, "Parma", "Cagliari", KICKOFF))]
    mu.update_matches_parquet(pairs, season="2025-2026", league="serie_a")

    got = pd.read_parquet(isolated)["match_id"].astype(str).tolist()
    assert got == ["2026-02-27_Parma_Cagliari"]
    assert "13981687" not in got


def test_the_written_key_is_never_all_digits(isolated):
    """The shape that made this bug so hard to see.

    An 8-char Sofascore id and an 8-hex FBref hash are indistinguishable by
    shape — `13980098` is both a plausible id and valid hex — which produced
    several wrong readings while this was being diagnosed. The canonical key is
    the one format that can never be confused for either.
    """
    pairs = [({}, _fixture(13981687, "Parma", "Cagliari", KICKOFF))]
    mu.update_matches_parquet(pairs, season="2025-2026", league="serie_a")

    key = pd.read_parquet(isolated)["match_id"].astype(str).iloc[0]
    assert not key.isdigit()


def test_a_dateless_fixture_is_dropped_rather_than_keyed_on_nothing(isolated):
    """No timestamp means no canonical key: "_Parma_Cagliari" joins to nothing.

    Dropping is right — a match_date is not optional in matches.parquet — but
    the point is that it must not fall back to the fixture id.
    """
    pairs = [({}, _fixture(13981687, "Parma", "Cagliari", None))]
    added = mu.update_matches_parquet(pairs, season="2025-2026", league="serie_a")

    assert added == 0
    assert not isolated.exists() or pd.read_parquet(isolated).empty


def test_team_names_are_kept_verbatim_the_way_matches_parquet_stores_them(isolated):
    """matches.parquet's own EPL rows read '2005-08-13_Aston Villa_Bolton'.

    The space is preserved there, so it is preserved here. This repo has three
    conventions in flight — verbatim (matches.parquet), `.replace(" ", "_")`
    (_fallback_ingest_from_results below) and `.replace(" ", "-")`
    (build_match_id_mapping.canonical_id) — and they only agree for Serie A,
    whose 20 team names happen to be single words. Anything writing this column
    must match the file, not the other two.
    """
    pairs = [({}, _fixture(1, "Aston Villa", "Newcastle", KICKOFF))]
    mu.update_matches_parquet(pairs, season="2025-2026", league="premier_league")

    key = pd.read_parquet(isolated)["match_id"].astype(str).iloc[0]
    assert key == "2026-02-27_Aston Villa_Newcastle"


def test_appending_the_same_fixture_twice_does_not_duplicate_the_match(isolated):
    """Dedup is on (home, away, date, season), so it never depended on the id —
    which is what makes changing the id format safe to do at all.
    """
    pairs = [({}, _fixture(13981687, "Parma", "Cagliari", KICKOFF))]
    mu.update_matches_parquet(pairs, season="2025-2026", league="serie_a")
    mu.update_matches_parquet(pairs, season="2025-2026", league="serie_a")

    assert len(pd.read_parquet(isolated)) == 1


# --- Step 6d gate: when does player_metadata.json get rebuilt? -------------
# Regression pin. The first version of this gate was `matches_fetched or not
# path.exists()`. `matches_fetched` reads 0 on every observed run (EPL detection
# is blind to matches already in the stat parquets), so that gate fired once and
# then never again — the same shape of defect that left this file with no writer
# at all from 2026-02-17. The stale-file test below is the true positive: it
# FAILS against the fetch-only gate.


def _age(tmp_path, days):
    p = tmp_path / "player_metadata.json"
    p.write_text("{}")
    os.utime(p, (time.time() - days * 86400,) * 2)
    return p


def test_a_stale_file_is_refreshed_even_when_nothing_was_fetched(tmp_path):
    assert mu._player_meta_needs_refresh(_age(tmp_path, 30), 0) is True


def test_a_fresh_file_is_left_alone(tmp_path):
    assert mu._player_meta_needs_refresh(_age(tmp_path, 0.5), 0) is False


def test_a_missing_file_is_always_rebuilt(tmp_path):
    assert mu._player_meta_needs_refresh(tmp_path / "nope.json", 0) is True


def test_a_fetch_forces_a_refresh_even_when_the_file_is_fresh(tmp_path):
    assert mu._player_meta_needs_refresh(_age(tmp_path, 0.1), 5) is True
