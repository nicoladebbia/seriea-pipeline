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


def test_referee_comes_from_espn_when_the_fixture_has_none_and_is_null_not_blank(isolated, monkeypatch):
    """Sofascore's 2026-27 fixture list names no referee (0 of 21 finished); the
    row used to carry "" and pass every coverage check as filled."""
    seen = []
    monkeypatch.setattr(mu, "_referee_from_espn",
                        lambda league, date, home, away: seen.append((league, date, home, away)) or "Davide Massa")
    pairs = [({}, _fixture(1, "Fiorentina", "Torino", KICKOFF)),
             ({}, dict(_fixture(2, "Roma", "Atalanta", KICKOFF), referee={"name": "Simone Sozza"}))]
    mu.update_matches_parquet(pairs, season="2025-2026", league="serie_a")
    df = pd.read_parquet(isolated).set_index("home_team")
    assert df.loc["Fiorentina", "referee"] == "Davide Massa"
    assert df.loc["Roma", "referee"] == "Simone Sozza"          # the fixture's own name wins, no ESPN call
    assert seen == [("serie_a", "2026-02-27", "Fiorentina", "Torino")]
    monkeypatch.setattr(mu, "_referee_from_espn", lambda *a: None)
    mu.update_matches_parquet([({}, _fixture(3, "Lecce", "Genoa", KICKOFF))], season="2025-2026", league="serie_a")
    df = pd.read_parquet(isolated).set_index("home_team")
    assert df.loc["Lecce", "referee"] is None or pd.isna(df.loc["Lecce", "referee"])


def test_backfill_referees_fills_played_rows_and_normalises_blanks(isolated, monkeypatch):
    pd.DataFrame({
        "league": ["serie_a"] * 4 + ["premier_league"],
        "season": ["2026-2027"] * 5,
        "match_date": pd.to_datetime(["2026-08-23", "2026-08-30", "2026-09-05", "2099-01-01", "2026-08-23"]),
        "home_team": ["Inter", "Roma", "Milan", "Lazio", "Arsenal"],
        "away_team": ["Torino", "Bologna", "Lecce", "Napoli", "Chelsea"],
        "home_score": [2, 1, 0, None, 1],
        "referee": ["", None, "Luca Zufferli", "", ""],
    }).to_parquet(isolated)
    names = {("Inter", "Torino"): "Davide Massa", ("Roma", "Bologna"): None}
    monkeypatch.setattr(mu, "_referee_from_espn", lambda league, date, home, away: names.get((home, away)))
    out = mu.backfill_referees(season="2026-2027", league="serie_a")
    assert out == {"league": "serie_a", "season": "2026-2027", "candidates": 2, "filled": 1, "blanked": 3}
    df = pd.read_parquet(isolated).set_index("home_team")["referee"]
    assert df["Inter"] == "Davide Massa" and df["Milan"] == "Luca Zufferli"
    assert all(pd.isna(df[t]) for t in ("Roma", "Lazio", "Arsenal"))   # unnamed, unplayed, other league
    # idempotent: the second pass has one candidate left (Roma) and nothing to blank
    assert mu.backfill_referees(season="2026-2027", league="serie_a")["blanked"] == 0


# --- ESPN heal: what the challenged Sofascore ingest left empty --------------

def _played_row(date, home, away, **extra):
    row = {"match_id": f"{date}_{home}_{away}", "season": "2026-2027", "league": "serie_a",
           "match_date": pd.Timestamp(date), "home_team": home, "away_team": away,
           "home_score": 1, "away_score": 2, "home_possession": None, "away_possession": None,
           "home_shots_total": None, "away_shots_total": None, "home_shots_on_target_total": None,
           "away_shots_on_target_total": None, "home_fouls": None, "away_fouls": None,
           "home_passing_accuracy": None, "home_passing_accuracy_count": None,
           "home_passing_accuracy_total": None, "home_yellow_cards": None, "away_yellow_cards": None,
           "home_red_cards": None, "away_red_cards": None, "home_cards": None, "away_cards": None,
           "home_ht_score": None, "away_ht_score": None, "home_xg": None, "data_source": None}
    row.update(extra)
    return row


def _fx(fid, home, away, ts):
    return {"id": fid, "startTimestamp": ts, "status": {"type": "finished"},
            "homeTeam": {"name": home}, "awayTeam": {"name": away}}


def _sofa_row(mid, kind, cls, minute, player, is_home):
    return {"match_id": mid, "incident_type": kind, "incident_class": cls, "minute": minute, "added_time": 0,
            "player_name": player, "player_id": "1", "is_home": is_home}


def test_heal_from_espn_fills_incidents_and_team_stats_only_where_sofascore_left_nothing(isolated, tmp_path, monkeypatch):
    """Genoa–Como 2026-09-04: a score-only row and zero incident rows, and the
    detector considered the match done. ESPN fills both; a match Sofascore
    served (incidents present, possession filled) is never touched; the
    second run has nothing to do and fetches nothing."""
    import scraper.sofascore_events as se
    import scripts.data.live_espn as le
    inc_path = tmp_path / "match_incidents.parquet"
    monkeypatch.setattr(mu, "INCIDENTS_PARQUET", inc_path)
    monkeypatch.setattr(se, "_INCIDENTS_PATH", inc_path)
    monkeypatch.setattr(se, "_SOFASCORE_DIR", tmp_path)
    pd.DataFrame([_sofa_row(101, "goal", "regular", 10, "A", True),
                  _sofa_row(101, "card", "yellow", 50, "B", False)]).to_parquet(inc_path, index=False)
    pd.DataFrame([_played_row("2026-09-04", "Genoa", "Como"),
                  _played_row("2026-09-05", "Inter", "Napoli", home_possession=55.0, away_possession=45.0,
                              home_yellow_cards=1, away_yellow_cards=2, home_red_cards=0, away_red_cards=0)]
                 ).to_parquet(isolated, index=False)
    ts = int(datetime(2026, 9, 4, 18, 45, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr(mu, "_load_fixtures", lambda season, league="serie_a": [
        _fx(102, "Genoa", "Como", ts), _fx(101, "Inter", "Napoli", ts + 86400),
        _fx(103, "Roma", "Atalanta", ts + 10 * 86400)])                  # 103: not kicked off yet
    fetched = []

    def fake_summary(league, date, home, away):
        fetched.append((date, home, away))
        return {"fake": (date, home, away)}
    monkeypatch.setattr(mu, "_espn_post_summary", fake_summary)
    monkeypatch.setattr(se, "incident_rows_from_espn", lambda s, fid: [
        {**_sofa_row(fid, "goal", "regular", 19, "Osmajic", True), "player_id": "espn:osmajic", "source": "espn"},
        {**_sofa_row(fid, "card", "yellow", 36, "Sow", True), "player_id": "espn:sow", "source": "espn"},
        {**_sofa_row(fid, "card", "red", 70, "Baturina", False), "player_id": "espn:baturina", "source": "espn"}])
    monkeypatch.setattr(le, "half_time_from_summary", lambda s: (1, 3))
    monkeypatch.setattr(mu, "_espn_stat_values", lambda s: {
        "home_possession": 37.6, "away_possession": 62.4, "home_shots_total": 11, "home_shots_on_target_total": 11,
        "home_fouls": 8, "home_passing_accuracy_count": 307, "home_passing_accuracy_total": 380,
        "home_passing_accuracy": 80.8})

    out = mu.heal_from_espn(season="2026-2027", league="serie_a")
    assert out["candidates"] == 1 and out["incidents_matches"] == 1 and out["incident_rows"] == 3
    assert out["stats_rows"] == 1 and out["unreachable"] == 0
    assert fetched == [("2026-09-04", "Genoa", "Como")]

    inc = pd.read_parquet(inc_path)
    assert set(inc["match_id"]) == {101, 102}
    assert inc.loc[inc["match_id"] == 102, "source"].eq("espn").all()
    assert inc.loc[inc["match_id"] == 101, "source"].isna().all()       # Sofascore rows untouched
    assert se.sofascore_covered_ids(inc_path) == {101}                  # ESPN rows are not coverage

    gt = pd.read_parquet(isolated).set_index("home_team")
    g = gt.loc["Genoa"]
    assert g["home_possession"] == 37.6 and g["home_shots_total"] == 11 and g["home_fouls"] == 8
    assert g["home_yellow_cards"] == 1 and g["home_red_cards"] == 0 and g["away_red_cards"] == 1
    assert g["home_cards"] == 1 and g["away_cards"] == 1
    assert (g["home_ht_score"], g["away_ht_score"]) == (1, 3) and g["data_source"] == "espn"
    assert pd.isna(g["home_xg"])                                        # ESPN has no xG: stays NaN, never 0
    i = gt.loc["Inter"]
    assert i["home_possession"] == 55.0 and pd.isna(i["data_source"]) and pd.isna(i["home_ht_score"])

    # idempotent: nothing left to heal, no fetch
    again = mu.heal_from_espn(season="2026-2027", league="serie_a")
    assert again["candidates"] == 0 and len(fetched) == 1

    # the day Sofascore answers, its rows replace the ESPN stand-ins
    se._save_incidents([_sofa_row(102, "goal", "regular", 19, "Milutin Osmajic", True)], set())
    inc = pd.read_parquet(inc_path)
    rows_102 = inc[inc["match_id"] == 102]
    assert len(rows_102) == 1 and rows_102["source"].isna().all()
    assert se.sofascore_covered_ids(inc_path) == {101, 102}


def test_heal_from_espn_counts_a_match_espn_cannot_serve_and_retries_it_next_run(isolated, tmp_path, monkeypatch):
    import scraper.sofascore_events as se
    inc_path = tmp_path / "match_incidents.parquet"
    monkeypatch.setattr(mu, "INCIDENTS_PARQUET", inc_path)
    monkeypatch.setattr(se, "_INCIDENTS_PATH", inc_path)
    pd.DataFrame([_played_row("2026-09-04", "Genoa", "Como")]).to_parquet(isolated, index=False)
    ts = int(datetime(2026, 9, 4, 18, 45, tzinfo=timezone.utc).timestamp())
    monkeypatch.setattr(mu, "_load_fixtures", lambda season, league="serie_a": [_fx(102, "Genoa", "Como", ts)])
    monkeypatch.setattr(mu, "_espn_post_summary", lambda *a: None)
    out = mu.heal_from_espn(season="2026-2027", league="serie_a")
    assert out == {"league": "serie_a", "season": "2026-2027", "candidates": 1, "incidents_matches": 0,
                   "incident_rows": 0, "stats_rows": 0, "unreachable": 1}
    assert not inc_path.exists() and pd.isna(pd.read_parquet(isolated)["home_possession"]).all()
