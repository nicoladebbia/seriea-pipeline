"""The Track Record page grades through the canonical grader, per league.

It used to roll its own join — raw (un-normalised) team names, exact date
only, no league filter — so it graded a different set of matches than every
other page and blended Serie A with gated EPL into one base rate. Each test
here carries its TRUE POSITIVE: the old behaviour is computed alongside and
asserted to differ, so none of them can pass vacuously.
"""
import json

import pandas as pd
import pytest

import scripts.analysis.feedback_analyzer as fa
import web.app as W


def _seed(tmp_path, monkeypatch, rows):
    """Write a matches.parquet + predictions archive and point the grader at them.

    `rows`: dicts of home/away/hs/as_/league/date plus the archive side
    (arch_home/arch_away/arch_date default to the played values, so a test
    opts in to a name variant or a moved kickoff).
    """
    played, archive = [], {}
    for r in rows:
        hs, as_ = r["hs"], r["as_"]
        played.append({
            "match_date": r["date"], "home_team": r["home"], "away_team": r["away"],
            "home_score": hs, "away_score": as_,
            "result": "H" if hs > as_ else ("A" if as_ > hs else "D"),
            "league": r["league"],
        })
        probs = r.get("probs") or {"home": 0.5, "draw": 0.3, "away": 0.2}
        ah, aa = r.get("arch_home", r["home"]), r.get("arch_away", r["away"])
        archive[f"{ah} vs {aa}_{r.get('arch_date', r['date'])}"] = {
            "home_team": ah, "away_team": aa,
            "date": r.get("arch_date", r["date"]), "match": f"{ah} vs {aa}",
            "predicted_outcome": r.get("pick", max(probs, key=probs.get).upper()),
            "probabilities": probs, "confidence": max(probs.values()),
            "home_xg": r.get("hxg", 1.4), "away_xg": r.get("axg", 1.1),
        }
    mpath, apath = tmp_path / "matches.parquet", tmp_path / "archive.json"
    pd.DataFrame(played).to_parquet(mpath)
    apath.write_text(json.dumps(archive))
    monkeypatch.setattr(fa, "MATCHES_PATH", mpath)
    monkeypatch.setattr(fa, "ARCHIVE_PATH", apath)
    return mpath, apath


def _mk(rec, league, market):
    lg = next(x for x in rec["leagues"] if x["league"] == league)
    return next(m for m in lg["markets"] if m["market"] == market)


def _row(i, league="serie_a", **kw):
    """One played fixture with a distinct team pair (so the ±3-day window,
    which the grader applies per ordered pair, can never see two candidates)."""
    base = {"home": f"H{league[:2]}{i}", "away": f"A{league[:2]}{i}",
            "date": f"2026-03-{(i % 27) + 1:02d}", "league": league,
            "hs": 2, "as_": 0}
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# 1. Leagues are graded apart, not blended
# --------------------------------------------------------------------------
def test_serie_a_and_epl_are_separate_records_not_one_blended_number(tmp_path, monkeypatch):
    rows = [_row(i, "serie_a") for i in range(12)] + [_row(i, "premier_league") for i in range(5)]
    _seed(tmp_path, monkeypatch, rows)
    rec = W._build_track_record()

    assert [lg["league"] for lg in rec["leagues"]] == ["serie_a", "premier_league"]
    assert _mk(rec, "serie_a", "1X2")["n"] == 12
    assert _mk(rec, "premier_league", "1X2")["n"] == 5
    # TRUE POSITIVE: the old page reported one market list over both leagues.
    assert all(m["n"] != len(rows) for lg in rec["leagues"] for m in lg["markets"])
    # Serie A leads — the production earner — and the back-compat top-level
    # keys are ONE league, never the blend.
    assert rec["default_league"] == "serie_a"
    assert rec["n_matches"] == 12
    sa = next(lg for lg in rec["leagues"] if lg["league"] == "serie_a")
    assert rec["markets"] == sa["markets"]


# --------------------------------------------------------------------------
# 2. The Double Chance base rate is the best DC pick available
# --------------------------------------------------------------------------
def test_double_chance_base_is_max_over_pickable_covers_not_the_tagged_max(tmp_path, monkeypatch):
    # 2 home / 4 draw / 6 away. Best fixed DC pick is X2 (draw+away) = 10/12.
    rows = ([_row(i, hs=2, as_=0) for i in range(2)]
            + [_row(10 + i, hs=1, as_=1) for i in range(4)]
            + [_row(20 + i, hs=0, as_=2) for i in range(6)])
    _seed(tmp_path, monkeypatch, rows)
    dc = _mk(W._build_track_record(), "serie_a", "Double Chance")

    assert dc["n"] == 12
    assert dc["base_rate"] == pytest.approx(10 / 12, abs=1e-4)
    # TRUE POSITIVE: the old rule tagged every draw to "1X", so its base was
    # max(H+D, A) = max(6, 6) — it would have credited 16.7pp of phantom edge.
    old_base = max(2 + 4, 6) / 12
    assert old_base == pytest.approx(0.5, abs=1e-4)
    assert dc["base_rate"] > old_base


def test_double_chance_base_is_unchanged_where_home_beats_away(tmp_path, monkeypatch):
    # The old rule was only wrong when aways outnumbered homes; pin that the
    # fix does not move the number in the case that was already right.
    rows = ([_row(i, hs=2, as_=0) for i in range(7)]
            + [_row(10 + i, hs=1, as_=1) for i in range(2)]
            + [_row(20 + i, hs=0, as_=2) for i in range(3)])
    _seed(tmp_path, monkeypatch, rows)
    dc = _mk(W._build_track_record(), "serie_a", "Double Chance")
    assert dc["base_rate"] == pytest.approx(max(7 + 2, 2 + 3) / 12, abs=1e-4)
    assert dc["base_rate"] == pytest.approx(max(7 + 2, 3) / 12, abs=1e-4)  # old == new here


# --------------------------------------------------------------------------
# 3. Normalised names and a moved kickoff still grade
# --------------------------------------------------------------------------
def test_a_name_variant_and_a_moved_kickoff_still_grade(tmp_path, monkeypatch):
    rows = [
        # archived as "Inter Milan"/"AC Milan"; played as "Inter"/"Milan"
        {"home": "Inter", "away": "Milan", "arch_home": "Inter Milan",
         "arch_away": "AC Milan", "date": "2026-03-01", "league": "serie_a",
         "hs": 2, "as_": 0},
        # kickoff moved two days after the prediction was archived
        {"home": "Roma", "away": "Lazio", "arch_date": "2026-03-05",
         "date": "2026-03-07", "league": "serie_a", "hs": 1, "as_": 1},
    ]
    _seed(tmp_path, monkeypatch, rows)
    rec = W._build_track_record()
    assert _mk(rec, "serie_a", "1X2")["n"] == 2

    # TRUE POSITIVE: the old exact-raw-name, exact-date key matched NEITHER.
    played = {f"{r['home']} vs {r['away']}_{r['date']}" for r in rows}
    archived = {f"{r.get('arch_home', r['home'])} vs {r.get('arch_away', r['away'])}"
                f"_{r.get('arch_date', r['date'])}" for r in rows}
    assert played & archived == set()


# --------------------------------------------------------------------------
# 4. The call is the archived pick, not argmax of the stored probabilities
# --------------------------------------------------------------------------
def test_the_graded_call_is_the_archived_pick_not_argmax(tmp_path, monkeypatch):
    # The real archive holds exactly this shape (Bologna v Udinese 2026-02-23):
    # stored DRAW while the rounded probabilities put home ahead by 0.004.
    probs = {"home": 0.402, "draw": 0.398, "away": 0.201}
    _seed(tmp_path, monkeypatch,
          [_row(0, hs=1, as_=1, pick="DRAW", probs=probs)])
    x = _mk(W._build_track_record(), "serie_a", "1X2")

    assert x["matches"][0]["pick"] == "DRAW"
    assert x["hits"] == 1          # the match WAS a draw
    # TRUE POSITIVE: argmax would have called HOME and scored this a miss.
    assert max(probs, key=probs.get).upper() == "HOME"


def test_an_archive_row_with_no_stored_pick_falls_back_to_argmax(tmp_path, monkeypatch):
    probs = {"home": 0.10, "draw": 0.20, "away": 0.70}
    _seed(tmp_path, monkeypatch, [_row(0, hs=0, as_=2, pick="", probs=probs)])
    x = _mk(W._build_track_record(), "serie_a", "1X2")
    assert x["matches"][0]["pick"] == "AWAY"
    assert x["hits"] == 1


# --------------------------------------------------------------------------
# 5. The xG-derived rows are named as what they are
# --------------------------------------------------------------------------
def test_the_poisson_rows_are_not_labelled_as_the_market_that_takes_money(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_row(i) for i in range(5)])
    lg = next(x for x in W._build_track_record()["leagues"] if x["league"] == "serie_a")
    by_est = {m["market"]: m["estimator"] for m in lg["markets"]}

    # TRUE POSITIVE: the page used to call these "O/U 2.5" and "BTTS" — the
    # names of the market that places real bets, for an estimator that does not.
    assert "O/U 2.5" not in by_est and "BTTS" not in by_est
    assert by_est["O/U 2.5 (Poisson from xG)"] == "poisson_xg"
    assert by_est["BTTS (Poisson from xG)"] == "poisson_xg"
    assert by_est["1X2"] == "ensemble" and by_est["Double Chance"] == "ensemble"
    for m in lg["markets"]:
        assert bool(m["note"]) == (m["estimator"] == "poisson_xg")


# --------------------------------------------------------------------------
# 6. A small league does not render as a trusted record
# --------------------------------------------------------------------------
def test_a_league_below_the_match_floor_is_not_trusted(tmp_path, monkeypatch):
    n_small = W._TR_MIN_N - 1
    rows = ([_row(i, "serie_a") for i in range(W._TR_MIN_N + 5)]
            + [_row(i, "premier_league") for i in range(n_small)])
    _seed(tmp_path, monkeypatch, rows)
    rec = W._build_track_record()

    assert all(m["trusted"] for m in
               next(lg for lg in rec["leagues"] if lg["league"] == "serie_a")["markets"])
    assert not any(m["trusted"] for m in
                   next(lg for lg in rec["leagues"] if lg["league"] == "premier_league")["markets"])
    # TRUE POSITIVE: at the old floor of 20 this sample cleared the badge.
    assert n_small >= 20


def test_hit_rate_se_shrinks_with_the_sample(tmp_path, monkeypatch):
    # Same 50% hit rate at two sample sizes: the SE has to say which is noise.
    def half(n, league):
        return ([_row(i, league, hs=2, as_=0) for i in range(n // 2)]
                + [_row(50 + i, league, hs=0, as_=2) for i in range(n // 2)])

    _seed(tmp_path, monkeypatch, half(80, "serie_a") + half(8, "premier_league"))
    rec = W._build_track_record()
    big = _mk(rec, "serie_a", "1X2")["hit_rate_se"]
    small = _mk(rec, "premier_league", "1X2")["hit_rate_se"]
    assert 0 < big < small


# --------------------------------------------------------------------------
# 7. The cache tracks BOTH of the grader's inputs
# --------------------------------------------------------------------------
def test_a_newly_graded_result_busts_the_cache_without_an_archive_write(tmp_path, monkeypatch):
    mpath, _apath = _seed(tmp_path, monkeypatch, [_row(i) for i in range(3)])
    calls = []
    monkeypatch.setattr(W, "_build_track_record",
                        lambda: calls.append(1) or {"leagues": [], "markets": []})
    monkeypatch.setattr(W, "_TRACKREC_CACHE", {"data": None, "key": ()})

    client = W.app.test_client()
    assert client.get("/api/track-record").status_code == 200
    client.get("/api/track-record")
    assert len(calls) == 1, "second identical request must be served from cache"

    # A match settles: matches.parquet moves, the archive does not.
    st = mpath.stat()
    import os
    os.utime(mpath, (st.st_atime + 120, st.st_mtime + 120))
    client.get("/api/track-record")
    # TRUE POSITIVE: keyed on the archive alone, this stayed at 1 — a graded
    # result stayed invisible until some unrelated archive write happened.
    assert len(calls) == 2


def test_every_emitted_market_has_a_base_rate_strategy(tmp_path, monkeypatch):
    """The market name is written in two places; if they ever drift, the base
    rate must blow up rather than quietly fall back to max(outcomes) — the
    formula that under-reported the Double Chance floor by 16.7pp.
    """
    _seed(tmp_path, monkeypatch, [_row(i) for i in range(4)])
    emitted = {m["market"] for lg in W._build_track_record()["leagues"]
               for m in lg["markets"]}
    assert emitted, "fixture produced no markets"
    assert emitted <= set(W._TR_STRATEGIES), emitted - set(W._TR_STRATEGIES)
