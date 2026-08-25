"""The match-time missing-players cache must track the JSONs it was built from.

It used to be build-once: `_ensure_cache` returned the parquet verbatim whenever
it existed, so it froze on 2026-04-22. By 2026-08-25 it held 3,329 matches while
3,389 Serie A JSONs sat on disk -- 50 of 2025-2026 and all 10 of 2026-2027 were
missing, and 1,866 of the rows it did hold came from JSONs Sofascore had since
rewritten. The Premier League directory was never walked at all, so
features_premier_league.parquet had none of these columns.
"""

import json

import pandas as pd
import pytest

import features.missing_players as mp


def _match_json(match_id, n_injured=1, n_suspended=0):
    def side():
        out = []
        for i in range(n_injured):
            out.append({"type": "missing", "description": "Thigh Injury",
                        "player": {"id": 100 + i}})
        for i in range(n_suspended):
            out.append({"type": "missing", "description": "yellow_card_accumulation_suspension",
                        "player": {"id": 200 + i}})
        return {"missing_players": out}

    return {"match_id": str(match_id), "home_lineup": side(), "away_lineup": side()}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A miniature copy of the real layout: two sibling league dirs + a cache."""
    sa = tmp_path / "matches"
    epl = tmp_path / "matches_premier_league"
    for base in (sa, epl):
        (base / "2026-2027").mkdir(parents=True)
    # both names, so this suite runs against the pre-fix module too and fails
    # there on BEHAVIOUR rather than on a missing symbol
    monkeypatch.setattr(mp, "SOFASCORE_MATCHES_DIRS",
                        ((sa, "serie_a"), (epl, "premier_league")), raising=False)
    monkeypatch.setattr(mp, "SOFASCORE_MATCHES_DIR", sa, raising=False)
    monkeypatch.setattr(mp, "CACHE_PATH", tmp_path / "missing_players.parquet")
    return {"sa": sa / "2026-2027", "epl": epl / "2026-2027", "cache": tmp_path / "missing_players.parquet"}


def _write(dirpath, match_id, **kw):
    p = dirpath / f"{match_id}.json"
    p.write_text(json.dumps(_match_json(match_id, **kw)))
    return p


def test_a_match_played_after_the_cache_was_built_is_picked_up(tree):
    """The freeze: a cache that existed was returned verbatim, forever."""
    _write(tree["sa"], "1")
    mp._ensure_cache()  # builds the cache

    _write(tree["sa"], "2", n_injured=3)
    out = mp._ensure_cache()

    assert set(out["sofascore_id"]) == {"1", "2"}, "a newly played match never entered the cache"
    row = out[out["sofascore_id"] == "2"].iloc[0]
    assert row["home_missing_injury_count"] == 3


def test_a_rewritten_json_refreshes_its_cached_row(tree, monkeypatch):
    """Sofascore rewrites a match JSON as absences are confirmed."""
    p = _write(tree["sa"], "1", n_injured=1)
    first = mp._ensure_cache()
    assert first.iloc[0]["home_missing_injury_count"] == 1

    p.write_text(json.dumps(_match_json("1", n_injured=4)))
    import os
    st = p.stat()
    os.utime(p, (st.st_atime + 60, st.st_mtime + 60))

    out = mp._ensure_cache()
    assert len(out) == 1, "the refreshed row was appended instead of replacing"
    assert out.iloc[0]["home_missing_injury_count"] == 4, "a rewritten JSON did not refresh its row"


def test_the_premier_league_directory_is_walked(tree):
    """The EPL dir is a sibling of matches/, not a subdirectory of it."""
    _write(tree["sa"], "1")
    _write(tree["epl"], "9", n_suspended=2)

    out = mp._ensure_cache()

    assert "9" in set(out["sofascore_id"]), "the Premier League directory was never scanned"
    epl_row = out[out["sofascore_id"] == "9"].iloc[0]
    assert epl_row["league"] == "premier_league"
    assert epl_row["home_missing_suspended_count"] == 2


def test_a_second_run_parses_nothing_and_changes_nothing(tree, monkeypatch):
    """Idempotence: the watermark must survive the cache write.

    Comparing against the cache FILE's mtime would pass a single re-run and
    still skip forever any JSON rewritten between two writes -- so this asserts
    zero re-parsing, not merely an unchanged frame.
    """
    _write(tree["sa"], "1")
    _write(tree["epl"], "9")
    first = mp._ensure_cache()

    calls = []
    real = mp._parse_match_json
    monkeypatch.setattr(mp, "_parse_match_json", lambda p: (calls.append(p), real(p))[1])

    second = mp._ensure_cache()

    assert calls == [], f"re-parsed {len(calls)} unchanged JSON(s) on the second run"
    pd.testing.assert_frame_equal(
        first.sort_values("sofascore_id").reset_index(drop=True),
        second.sort_values("sofascore_id").reset_index(drop=True),
    )


def test_the_parser_still_runs_on_a_genuinely_new_file(tree, monkeypatch):
    """True positive: an implementation that parses nothing cannot pass the suite."""
    _write(tree["sa"], "1")
    mp._ensure_cache()

    calls = []
    real = mp._parse_match_json
    monkeypatch.setattr(mp, "_parse_match_json", lambda p: (calls.append(p), real(p))[1])

    _write(tree["sa"], "2")
    mp._ensure_cache()

    assert len(calls) == 1, "a new match was not parsed"


def test_history_survives_a_source_json_that_disappears(tree):
    """A pruned raw JSON must not silently delete a season of history."""
    p = _write(tree["sa"], "1")
    _write(tree["sa"], "2")
    mp._ensure_cache()

    p.unlink()
    out = mp._ensure_cache()

    assert set(out["sofascore_id"]) == {"1", "2"}, "history was dropped with its source file"
