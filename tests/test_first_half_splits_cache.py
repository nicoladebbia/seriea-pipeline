"""The first-half splits cache must track the JSONs it was built from.

Same defect as the missing-players cache: `_ensure_cache` returned the parquet
verbatim whenever it existed, so it froze on 2026-04-22 at 1,465 rows -- 22% of
the 6,775 match JSONs that carry a 1ST period today. The Premier League dir was
never walked at all, so features_premier_league.parquet had no *_fh_* columns.

Unlike missing_players these feed rolling windows, so a stale cache does not
merely leave nulls -- it changes the rolling basis. That makes the refresh a
training-set change, which is why it is pinned here.
"""

import json
import os

import pandas as pd
import pytest

import features.first_half_splits as fh


def _match_json(match_id, fh_xg=0.5, full_xg=1.0):
    def period(label, xg):
        return {
            "period": label,
            "groups": [{"statisticsItems": [
                {"key": "expectedGoals", "homeValue": xg, "awayValue": xg / 2},
                {"key": "totalShotsOnGoal", "homeValue": 7, "awayValue": 4},
                {"key": "shotsOnGoal", "homeValue": 3, "awayValue": 1},
                {"key": "cornerKicks", "homeValue": 2, "awayValue": 1},
            ]}],
        }

    return {
        "match_id": str(match_id),
        "team_stats": {"statistics": [period("1ST", fh_xg), period("ALL", full_xg)]},
    }


@pytest.fixture
def tree(tmp_path, monkeypatch):
    sa = tmp_path / "matches"
    epl = tmp_path / "matches_premier_league"
    for base in (sa, epl):
        (base / "2026-2027").mkdir(parents=True)
    # both names, so this suite runs against the pre-fix module too and fails
    # there on BEHAVIOUR rather than on a missing symbol
    monkeypatch.setattr(fh, "SOFASCORE_MATCHES_DIRS",
                        ((sa, "serie_a"), (epl, "premier_league")), raising=False)
    monkeypatch.setattr(fh, "SOFASCORE_MATCHES_DIR", sa, raising=False)
    monkeypatch.setattr(fh, "CACHE_PATH", tmp_path / "first_half_splits.parquet")
    return {"sa": sa / "2026-2027", "epl": epl / "2026-2027", "root": tmp_path}


def _write(dirpath, match_id, **kw):
    p = dirpath / f"{match_id}.json"
    p.write_text(json.dumps(_match_json(match_id, **kw)))
    return p


def test_a_match_played_after_the_cache_was_built_is_picked_up(tree):
    _write(tree["sa"], "1")
    fh._ensure_cache()

    _write(tree["sa"], "2", fh_xg=1.4)
    out = fh._ensure_cache()

    assert set(out["sofascore_id"]) == {"1", "2"}, "a newly played match never entered the cache"
    assert out[out["sofascore_id"] == "2"].iloc[0]["home_fh_xg"] == 1.4


def test_a_rewritten_json_refreshes_its_cached_row(tree):
    """Many of these JSONs only gained their per-period breakdown on a re-scrape."""
    p = _write(tree["sa"], "1", fh_xg=0.5)
    first = fh._ensure_cache()
    assert first.iloc[0]["home_fh_xg"] == 0.5

    p.write_text(json.dumps(_match_json("1", fh_xg=2.2)))
    st = p.stat()
    os.utime(p, (st.st_atime + 60, st.st_mtime + 60))

    out = fh._ensure_cache()
    assert len(out) == 1, "the refreshed row was appended instead of replacing"
    assert out.iloc[0]["home_fh_xg"] == 2.2, "a rewritten JSON did not refresh its row"


def test_the_premier_league_directory_is_walked(tree):
    _write(tree["sa"], "1")
    _write(tree["epl"], "9", fh_xg=0.9)

    out = fh._ensure_cache()

    assert "9" in set(out["sofascore_id"]), "the Premier League directory was never scanned"
    assert out[out["sofascore_id"] == "9"].iloc[0]["league"] == "premier_league"


def test_a_second_run_parses_nothing_and_changes_nothing(tree, monkeypatch):
    """Idempotence: the watermark must survive the cache write."""
    _write(tree["sa"], "1")
    _write(tree["epl"], "9")
    first = fh._ensure_cache()

    calls = []
    real = fh._parse_match_json
    monkeypatch.setattr(fh, "_parse_match_json", lambda p: (calls.append(p), real(p))[1])

    second = fh._ensure_cache()

    assert calls == [], f"re-parsed {len(calls)} unchanged JSON(s) on the second run"
    pd.testing.assert_frame_equal(
        first.sort_values("sofascore_id").reset_index(drop=True),
        second.sort_values("sofascore_id").reset_index(drop=True),
    )


def test_the_parser_still_runs_on_a_genuinely_new_file(tree, monkeypatch):
    """True positive: an implementation that parses nothing cannot pass the suite."""
    _write(tree["sa"], "1")
    fh._ensure_cache()

    calls = []
    real = fh._parse_match_json
    monkeypatch.setattr(fh, "_parse_match_json", lambda p: (calls.append(p), real(p))[1])

    _write(tree["sa"], "2")
    fh._ensure_cache()

    assert len(calls) == 1, "a new match was not parsed"


def test_the_cache_s_own_season_and_league_do_not_collide_with_the_frame(tree, monkeypatch):
    """The cache gained season/league columns; feature_df supplies its own.

    Merging both would yield season_x/season_y and silently break every
    downstream consumer that reads `season`.
    """
    _write(tree["sa"], "1")
    fh._ensure_cache()

    mapping = pd.DataFrame({"match_id": ["m1"], "sofascore_id": ["1"]})
    map_path = tree["root"] / "match_id_mapping.parquet"
    mapping.to_parquet(map_path, index=False)
    monkeypatch.setattr(fh, "MAPPING_PATH", map_path)

    feature_df = pd.DataFrame([{
        "match_id": "m1", "home_team": "Inter", "away_team": "Napoli",
        "match_date": pd.Timestamp("2026-08-24"), "season": "2026-2027", "league": "serie_a",
    }])

    out = fh.add_first_half_splits_features(feature_df)

    assert "season" in out.columns and "season_x" not in out.columns
    assert "league" in out.columns and "league_x" not in out.columns
    assert out.iloc[0]["season"] == "2026-2027"
