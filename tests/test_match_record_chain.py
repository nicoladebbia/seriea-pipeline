"""Finished-match record chain (scripts/data/match_record_chain.py) and the
stand-in semantics it relies on in matchday_updater / write_shot_level_xg."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from scraper import fotmob as fm
from scripts.data import match_record_chain as chain
from scripts.data import matchday_updater as mu
from scripts.data import write_shot_level_xg as wsx

FIX = Path(__file__).parent / "fixtures"
NOW = 1_800_000_000
FIO_ID, INT_ID, ROMA_ID = 16285000, 16285001, 16285005


def _fx(fid, home, away, ts, status="finished", rnd=3, hs=1, as_=2):
    return {"id": fid, "startTimestamp": ts, "status": {"type": status}, "roundInfo": {"round": rnd},
            "homeTeam": {"name": home}, "awayTeam": {"name": away},
            "homeScore": {"current": hs}, "awayScore": {"current": as_}}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Every parquet, the fixture calendar, the FotMob raw dir, the status file
    and the shot cache under tmp_path — nothing here may touch data/."""
    paths = {c: tmp_path / f"{c}.parquet" for c in chain.COMPONENTS}
    monkeypatch.setattr(chain, "component_path", lambda c, lg: paths[c])
    monkeypatch.setattr(chain, "STATUS_FILE", tmp_path / "status.json")
    monkeypatch.setattr(fm, "RAW_DIR", tmp_path / "fotmob")
    monkeypatch.setattr(fm, "_blocked_until", 0.0)
    cache = tmp_path / "sofa_cache"
    cache.mkdir()
    monkeypatch.setattr(wsx, "cache_dir", lambda season, league="serie_a": cache)
    monkeypatch.setattr(wsx, "out_path", lambda league="serie_a": paths["all_shots_with_xg"])
    fixtures = [_fx(FIO_ID, "Fiorentina", "Torino", NOW - 20 * 3600),
                _fx(INT_ID, "Inter", "Napoli", NOW - 18 * 3600, status="notstarted"),   # cache never flipped
                _fx(ROMA_ID, "Roma", "Atalanta", NOW - 16 * 3600),
                _fx(99, "Lazio", "Milan", NOW - 3600, status="notstarted"),             # kicked off 1h ago: not yet
                _fx(98, "Genoa", "Parma", NOW + 86400, status="notstarted"),
                _fx(97, "Lecce", "Como", NOW - 5 * 86400, status="postponed")]
    monkeypatch.setattr(mu, "_load_fixtures", lambda season, league="serie_a": fixtures)
    return {"paths": paths, "cache": cache, "fixtures": fixtures, "tmp": tmp_path}


def _sofa_player_rows(mid, team="Roma", n=2):
    return [{"season": "2026-2027", "match_id": mid, "date": "2026-09-05", "team": team, "player_id": 1000 + i,
             "player_name": f"Player {i}", "minutes": 90, "total_shots": i} for i in range(n)]


def test_finished_fixtures_use_status_or_the_three_hour_rule(isolated):
    got = {f["id"] for f in chain.finished_fixtures("2026-2027", "serie_a", now=NOW)}
    assert got == {FIO_ID, INT_ID, ROMA_ID}


def test_missing_components_ignores_stand_ins_and_counts_the_cached_shotmap(isolated):
    p = isolated["paths"]
    rows = _sofa_player_rows(ROMA_ID) + [{**r, "match_id": FIO_ID, "source": "fotmob"} for r in _sofa_player_rows(FIO_ID)]
    pd.DataFrame(rows).to_parquet(p["player_match_stats"], index=False)
    # Roma's shot map is cached from Sofascore but not yet in the parquet
    (isolated["cache"] / f"{ROMA_ID}.json").write_text(json.dumps(
        {"shotmap": [{"isHome": True, "player": {"id": 5, "name": "X"}, "xg": 0.1, "shotType": "miss",
                      "playerCoordinates": {"x": 10, "y": 50}}]}))
    fx = chain.finished_fixtures("2026-2027", "serie_a", now=NOW)
    miss = chain.missing_components("2026-2027", "serie_a", fx)
    assert miss[ROMA_ID] == ["match_team_stats", "shotmap_stats"]        # players sofascore, shots cached
    assert miss[FIO_ID] == list(chain.COMPONENTS)                       # a fotmob stand-in is not coverage
    assert miss[INT_ID] == list(chain.COMPONENTS)
    assert chain.rebuild_shots_from_cache("2026-2027", "serie_a", fx) == 1
    assert chain.sofascore_backed_ids(p["all_shots_with_xg"]) == {str(ROMA_ID)}


def _fake_fotmob(ctx, league, resolve, days):
    if ctx.match_id == INT_ID:
        return None, "fotmob lists no Inter v Napoli"
    md = json.loads((FIX / "fotmob_match_5749663.json").read_text())
    return fm.parse_match(md, ctx, resolve), f"fotmob-fake {ctx.match_id}"


def _fake_espn(ctx, league, resolve):
    return {"player_match_stats": [{**ctx.base(), "team": ctx.home_team, "player_id": -5, "player_name": "E",
                                    "minutes": 90, "total_shots": 1, "source": "espn"}],
            "match_team_stats": [{**ctx.base(), "period": "ALL", "team": ctx.home_team, "total_shots": 9, "source": "espn"}]}, "espn-fake"


def test_heal_missing_walks_the_chain_writes_stand_ins_and_reports(isolated, monkeypatch):
    p = isolated["paths"]
    pd.DataFrame(_sofa_player_rows(ROMA_ID)).to_parquet(p["player_match_stats"], index=False)
    monkeypatch.setitem(chain.SOURCES, "fotmob", _fake_fotmob)
    monkeypatch.setitem(chain.SOURCES, "espn", _fake_espn)
    s = chain.heal_missing("2026-2027", "serie_a", now=NOW)
    assert (s["matches_missing_before"], s["matches_filled"], s["matches_partial"], s["matches_unfilled"]) == (3, 2, 1, 0)
    by = {m["match_id"]: m for m in s["matches"]}
    assert set(by[FIO_ID]["filled"].values()) == {"fotmob"} and by[FIO_ID]["still_missing"] == []
    assert by[INT_ID]["filled"] == {"player_match_stats": "espn", "match_team_stats": "espn"}
    assert by[INT_ID]["still_missing"] == ["all_shots_with_xg", "shotmap_stats"]
    assert "fotmob lists no Inter" in " ".join(by[INT_ID]["reasons"])
    assert by[ROMA_ID]["missing_before"] == ["match_team_stats", "all_shots_with_xg", "shotmap_stats"]
    pms = pd.read_parquet(p["player_match_stats"])
    assert set(pms["source"].dropna()) == {"fotmob", "espn"} and pms["source"].isna().sum() == 2   # Sofascore rows untouched
    assert (pms.loc[pms["match_id"] == FIO_ID, "player_id"] < 0).all()
    shots = pd.read_parquet(p["all_shots_with_xg"])
    assert set(shots["match_id"]) == {str(FIO_ID), str(ROMA_ID)} and (shots["source"] == "fotmob").all()
    status = json.loads(chain.STATUS_FILE.read_text())["serie_a"]
    assert status["matches_filled"] == 2 and status["sources"]["fotmob"] in ("ok", "not needed")
    # idempotent: a second run rewrites the same stand-ins, never duplicates them
    n_before = len(pd.read_parquet(p["match_team_stats"]))
    s2 = chain.heal_missing("2026-2027", "serie_a", now=NOW)
    assert s2["matches_missing_before"] == 3 and len(pd.read_parquet(p["match_team_stats"])) == n_before


def test_heal_missing_dry_run_writes_nothing(isolated, monkeypatch):
    monkeypatch.setitem(chain.SOURCES, "fotmob", _fake_fotmob)
    monkeypatch.setitem(chain.SOURCES, "espn", _fake_espn)
    s = chain.heal_missing("2026-2027", "serie_a", now=NOW, dry_run=True)
    assert s["matches_filled"] == 2 and not isolated["paths"]["player_match_stats"].exists()


def test_resolver_maps_a_unique_folded_name_in_the_team_to_its_sofascore_id(isolated):
    p = isolated["paths"]
    rows = [{"team": "Torino", "player_name": "Giovanni Simeone", "player_id": 4242, "source": None},
            {"team": "Torino", "player_name": "Duván Zapata", "player_id": 55, "source": None},
            {"team": "Torino", "player_name": "Duvan Zapata", "player_id": 56, "source": None},   # two ids: ambiguous
            {"team": "Torino", "player_name": "Arthur Atta", "player_id": -9, "source": "fotmob"}]  # a stand-in never seeds
    pd.DataFrame(rows).to_parquet(p["player_match_stats"], index=False)
    resolve = chain.build_resolver("serie_a")
    assert resolve("Torino", "Giovanni SIMEONE", 1) == 4242
    assert resolve("Torino", "Duvan Zapata", 1) is None
    assert resolve("Fiorentina", "Giovanni Simeone", 1) is None
    assert resolve("Torino", "Arthur Atta", 1) is None


def test_sofascore_rows_replace_stand_ins_in_save_merged_and_the_detector(isolated, monkeypatch):
    p = isolated["paths"]["player_match_stats"]
    stand = [{**r, "player_id": -r["player_id"], "source": "fotmob"} for r in _sofa_player_rows(FIO_ID, "Fiorentina")]
    pd.DataFrame(stand + _sofa_player_rows(ROMA_ID)).to_parquet(p, index=False)
    monkeypatch.setattr(mu, "_sofascore_parquet", lambda base, league: p)
    assert mu._get_existing_sofascore_match_ids("serie_a") == {ROMA_ID}          # the stand-in stays "new"
    real = pd.DataFrame(_sofa_player_rows(FIO_ID, "Fiorentina", n=3)).assign(source="sofascore")
    mu._save_merged(real, p, ["match_id", "player_id", "season"], replace_stand_ins=True)
    df = pd.read_parquet(p)
    fio = df[df["match_id"] == FIO_ID]
    assert len(fio) == 3 and (fio["player_id"] > 0).all() and (fio["source"] == "sofascore").all()
    assert len(df[df["match_id"] == ROMA_ID]) == 2
    assert mu._get_existing_sofascore_match_ids("serie_a") == {ROMA_ID, FIO_ID}


def test_write_stand_in_replaces_only_its_own_source_rows(isolated):
    p = isolated["paths"]["match_team_stats"]
    chain.write_stand_in([{"match_id": FIO_ID, "period": "ALL", "team": "A", "total_shots": 1, "source": "espn"}], p, FIO_ID, "espn")
    chain.write_stand_in([{"match_id": FIO_ID, "period": "ALL", "team": "A", "total_shots": 7, "source": "fotmob"}], p, FIO_ID, "fotmob")
    chain.write_stand_in([{"match_id": FIO_ID, "period": "ALL", "team": "A", "total_shots": 8, "source": "fotmob"}], p, FIO_ID, "fotmob")
    df = pd.read_parquet(p)
    assert sorted(df["source"]) == ["espn", "fotmob"] and int(df.loc[df["source"] == "fotmob", "total_shots"].iloc[0]) == 8


def test_shot_rebuild_keeps_stand_ins_until_the_cache_holds_the_match(isolated):
    p = isolated["paths"]["all_shots_with_xg"]
    pd.DataFrame([{"season": "2026-2027", "match_id": str(FIO_ID), "xg": 0.2, "source": "fotmob"},
                  {"season": "2026-2027", "match_id": str(INT_ID), "xg": 0.3, "source": "fotmob"},
                  {"season": "2025-2026", "match_id": "1", "xg": 0.1, "source": None}]).to_parquet(p, index=False)
    (isolated["cache"] / f"{FIO_ID}.json").write_text(json.dumps(
        {"shotmap": [{"isHome": True, "player": {"id": 5, "name": "X"}, "xg": 0.9, "shotType": "goal",
                      "playerCoordinates": {"x": 10, "y": 50}}]}))
    assert wsx.rebuild_from_cache("2026-2027", "serie_a") == 1
    df = pd.read_parquet(p)
    fio = df[df["match_id"] == str(FIO_ID)]
    assert len(fio) == 1 and fio["source"].iloc[0] == "sofascore" and float(fio["xg"].iloc[0]) == 0.9
    assert (df[df["match_id"] == str(INT_ID)]["source"] == "fotmob").all()      # no cache yet: kept
    assert len(df[df["season"] == "2025-2026"]) == 1


def test_sofascore_cooldown_is_written_on_a_denial_and_skips_the_next_run(tmp_path, monkeypatch):
    monkeypatch.setattr(mu, "SOFASCORE_COOLDOWN_FILE", tmp_path / "cool.json")
    assert mu.sofascore_cooldown_remaining() == 0.0
    assert mu._looks_denied(RuntimeError("HTTP 403: challenge")) and not mu._looks_denied(RuntimeError("timeout"))
    mu.set_sofascore_cooldown("HTTP 403", minutes=30)
    assert 29 * 60 < mu.sofascore_cooldown_remaining() <= 30 * 60
    assert mu.sofascore_cooldown_remaining(now=datetime.now(UTC).timestamp() + 31 * 60) == 0.0


def test_fetch_loop_stops_at_the_first_denial(tmp_path, monkeypatch):
    import asyncio
    import sys
    import types
    monkeypatch.setattr(mu, "SOFASCORE_COOLDOWN_FILE", tmp_path / "cool.json")
    monkeypatch.setattr(mu, "RATE_LIMIT", 0)
    calls = []

    class _Api:
        async def close(self):
            pass

    async def _scrape(api, mid, season):
        calls.append(mid)
        raise RuntimeError("403 Forbidden")

    monkeypatch.setitem(sys.modules, "sofascore_wrapper", types.SimpleNamespace(api=types.SimpleNamespace(SofascoreAPI=_Api)))
    monkeypatch.setitem(sys.modules, "sofascore_wrapper.api", types.SimpleNamespace(SofascoreAPI=_Api))
    monkeypatch.setattr(mu, "scrape_match_stats", _scrape)
    fixtures = [_fx(1, "A", "B", NOW), _fx(2, "C", "D", NOW), _fx(3, "E", "F", NOW)]
    assert asyncio.run(mu.fetch_match_details(fixtures, "2026-2027")) == []
    assert calls == [1]                                   # 2 and 3 were never asked
    assert mu.sofascore_cooldown_remaining() > 0
    assert asyncio.run(mu.fetch_match_details(fixtures, "2026-2027")) == [] and calls == [1]


def test_rows_from_espn_summary_carry_the_subset_espn_has_and_minutes(isolated):
    summary = json.loads((FIX / "espn" / "summary_401874781.json").read_text())
    ctx = fm.MatchContext(season="2026-2027", match_id=ROMA_ID, date="2026-09-05", round=3,
                          home_team="Roma", away_team="Atalanta", home_score=2, away_score=1)
    out = chain.rows_from_espn_summary(summary, ctx, resolve=lambda t, n, f: 4242 if n.startswith("Gianluca") else None)
    players = out["player_match_stats"]
    assert players and all(p["source"] == "espn" and p["minutes"] > 0 for p in players)
    assert {p["team"] for p in players} == {"Roma", "Atalanta"}
    assert sum(1 for p in players if p["is_starter"]) == 22
    assert all("tackles" not in p for p in players)              # ESPN has no tackles: absent, not 0
    assert any(p["player_id"] == 4242 for p in players) and all(p["player_id"] != 0 for p in players)
    teams = out["match_team_stats"]
    assert [t["period"] for t in teams] == ["ALL", "ALL"] and teams[0]["total_shots"] is not None
