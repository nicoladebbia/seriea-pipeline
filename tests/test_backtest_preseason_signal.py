"""Guards on the pre-season backtest harness.

A backtest that leaks is worse than no backtest: it produces a confident number
that justifies shipping constants which do not work.  Every test here attacks
the replay's honesty rather than checking that it runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.analysis import backtest_preseason_signal as bt
from scripts.prediction import lineup_predictor as lp

TEAM = "Test FC"


def _league_rows(season, rounds, players, team=TEAM, starters=11):
    rows = []
    for rnd in rounds:
        for i, pid in enumerate(players):
            rows.append({"team": team, "match_id": f"{season}-{rnd}",
                         "date": f"{season[:4]}-09-{rnd:02d}", "player_id": pid,
                         "player_name": f"P{pid}", "is_starter": i < starters,
                         "minutes": 90 if i < starters else 0, "rating": 6.9,
                         "position": "M", "shirt_number": i + 1,
                         "round": rnd, "season": season})
    return rows


def _friendly_rows(season, players, team=TEAM, event=900000):
    return [{"sofascore_event_id": event + i // 20, "match_date": f"{season[:4]}-07-20",
             "season": season, "club": team, "club_id": 1, "opponent": "X",
             "is_home": True, "is_our_club": True, "club_league": "serie_a",
             "formation": "4-3-3", "player": f"P{p}", "player_id": p,
             "shirt_number": 1, "position": "M", "is_starter": True,
             "minutes_played": 90, "was_used": True}
            for i, p in enumerate(players)]


# ------------------------------------------------------------------ leakage

def test_preseason_signal_is_season_scoped(tmp_path, monkeypatch):
    """THE leakage guard.  A 2024-25 replay handed 2026-27 friendlies would be
    scored on players who had not signed yet -- it would look brilliant and
    mean nothing."""
    monkeypatch.setattr(lp, "SOFASCORE_DIR", tmp_path)
    pd.DataFrame(_friendly_rows("2024-2025", [1, 2, 3])).to_parquet(
        tmp_path / "friendlies_2024_2025.parquet", index=False)
    pd.DataFrame(_friendly_rows("2026-2027", [7, 8, 9])).to_parquet(
        tmp_path / "friendlies_2026_2027.parquet", index=False)

    old = lp.load_preseason_signal(TEAM, season="2024-2025")
    assert set(old["players"]) == {1, 2, 3}
    assert old["season"] == "2024-2025"

    # Default still means "newest", which is what production wants.
    assert set(lp.load_preseason_signal(TEAM)["players"]) == {7, 8, 9}


def test_replay_table_at_mw1_is_last_season_only():
    """At MW1 no rows of the new season exist, so the production loader's
    season.max() yields LAST season.  A replay that leaked the new season's
    completed rounds would be predicting the answer from the answer."""
    stats = pd.DataFrame(_league_rows("2024-2025", [1, 2], range(14))
                         + _league_rows("2025-2026", [1, 2, 3], range(14)))
    t = bt._replay_table(stats, TEAM, "2025-2026", 1)
    assert set(t["season"]) == {"2024-2025"}


def test_replay_table_at_mwk_excludes_the_round_being_predicted():
    stats = pd.DataFrame(_league_rows("2025-2026", [1, 2, 3, 4], range(14)))
    t = bt._replay_table(stats, TEAM, "2025-2026", 3)
    assert set(t["round"]) == {1, 2}, "round 3 is the target, never an input"
    assert 4 not in set(t["round"]), "future rounds must never appear"


def test_replay_table_is_team_scoped():
    stats = pd.DataFrame(_league_rows("2025-2026", [1, 2], range(14))
                         + _league_rows("2025-2026", [1, 2], range(14), team="Other"))
    assert set(bt._replay_table(stats, TEAM, "2025-2026", 2)["team"]) == {TEAM}


def test_prev_season_arithmetic():
    assert bt._prev_season("2025-2026") == "2024-2025"
    assert bt._prev_season("2018-2019") == "2017-2018"


# ------------------------------------------------------------------- metric

def test_hit_rate_is_exact_and_bounded():
    stats = pd.DataFrame(_league_rows("2025-2026", [1], range(14)))
    actual = bt._actual_starters(stats, TEAM, "2025-2026", 1)
    assert actual == set(range(11))
    perfect = len(set(range(11)) & actual) / bt.XI
    assert perfect == 1.0
    half = len(set(range(5, 16)) & actual) / bt.XI
    assert half == pytest.approx(6 / 11)


def test_naive_baseline_ranks_by_raw_start_count():
    """The floor arm must stay dumb.  If it ever picks up shrinkage or bonuses
    it stops being a floor and 'model beats naive' becomes unfalsifiable."""
    rows = _league_rows("2024-2025", [1, 2, 3], range(14))
    picks = bt._naive_top11(pd.DataFrame(rows))
    assert set(picks) == set(range(11))
    assert len(picks) == bt.XI


def test_naive_on_an_empty_table_returns_nothing():
    assert bt._naive_top11(pd.DataFrame()) == []


def test_promoted_club_has_no_prior_table_but_the_signal_still_predicts(tmp_path, monkeypatch):
    """The case the signal exists for: no league history at all."""
    monkeypatch.setattr(lp, "SOFASCORE_DIR", tmp_path)
    pd.DataFrame(_friendly_rows("2025-2026", list(range(20)))).to_parquet(
        tmp_path / "friendlies_2025_2026.parquet", index=False)
    empty = pd.DataFrame(columns=["team", "match_id", "date", "player_id",
                                  "player_name", "is_starter", "minutes", "rating",
                                  "position", "shirt_number", "round", "season"])
    pre = lp.load_preseason_signal(TEAM, season="2025-2026")
    assert bt._naive_top11(empty) == []
    assert len(bt._model_top11(empty, TEAM, None)) == 0
    assert len(bt._model_top11(empty, TEAM, pre)) == bt.XI


# --------------------------------------------------------------- summarising

def test_summary_counts_changed_slots_not_just_accuracy():
    """An accuracy delta with ~zero changed slots is noise.  The summary must
    carry the count so the delta can never be read without it."""
    fx = pd.DataFrame([
        {"season": "S", "team": "A", "round": 1, "club_friendlies": 4,
         "replay_rows": 10, "promoted": False, "n_changed": 0,
         "hit_naive": 0.5, "hit_off": 0.5, "hit_on": 0.5,
         "n_naive": 11, "n_off": 11, "n_on": 11},
        {"season": "S", "team": "B", "round": 1, "club_friendlies": 4,
         "replay_rows": 10, "promoted": False, "n_changed": 3,
         "hit_naive": 0.4, "hit_off": 0.5, "hit_on": 0.7,
         "n_naive": 11, "n_off": 11, "n_on": 11},
    ])
    s = bt._summarise(fx)
    assert s["player_slots_changed"] == 3
    assert s["fixtures_where_signal_changed_xi"] == 1
    assert s["delta_on_minus_off"] == pytest.approx(0.1)
    assert s["on_changed_fixtures_only"]["delta"] == pytest.approx(0.2)
    assert s["by_round"][1]["n"] == 2


def test_backtest_refuses_a_season_it_cannot_replay(monkeypatch):
    """Better a loud refusal than a number computed off a missing prior season."""
    monkeypatch.setattr(bt, "load_league_stats",
                        lambda: pd.DataFrame(_league_rows("2025-2026", [1], range(14))))
    monkeypatch.setattr(bt, "friendly_seasons", lambda: ["2025-2026"])
    out = bt.run_backtest(verbose=False)
    assert "error" in out, "no 2024-2025 table exists, so MW1 cannot be replayed"


def test_sweep_needs_a_holdout(monkeypatch):
    monkeypatch.setattr(bt, "load_league_stats",
                        lambda: pd.DataFrame(_league_rows("2025-2026", [1], range(14))))
    monkeypatch.setattr(bt, "friendly_seasons", lambda: ["2025-2026"])
    assert "error" in bt.sweep()


def test_sweep_restores_the_constants(tmp_path, monkeypatch):
    """The sweep mutates module globals.  Leaking a swept value into the live
    predictor would silently reconfigure production.

    The fixture must make the sweep ACTUALLY RUN -- three league seasons so two
    are replayable.  An earlier version of this test supplied one season, so
    sweep() returned at the holdout guard and the finally block was never
    reached: green, and proving nothing.
    """
    before = (lp.PRESEASON_ONLY_PRIOR, lp.PRESEASON_ABSENT_PENALTY,
              lp.PRESEASON_FADE_MATCHES)
    monkeypatch.setattr(lp, "SOFASCORE_DIR", tmp_path)
    monkeypatch.setattr(bt, "OUT_PATH", tmp_path / "out.json")
    for season in ("2024-2025", "2025-2026"):
        pd.DataFrame(_friendly_rows(season, list(range(14)))).to_parquet(
            tmp_path / f"friendlies_{season.replace('-', '_')}.parquet", index=False)
    stats = pd.DataFrame(sum(
        (_league_rows(s, [1, 2, 3], range(14))
         for s in ("2023-2024", "2024-2025", "2025-2026")), []))
    monkeypatch.setattr(bt, "load_league_stats", lambda: stats)
    monkeypatch.setattr(bt, "friendly_seasons", lambda: ["2024-2025", "2025-2026"])

    out = bt.sweep()
    assert "error" not in out, "fixture must let the sweep actually run"
    assert out["calibration"], "the grid must have produced cells"
    assert (lp.PRESEASON_ONLY_PRIOR, lp.PRESEASON_ABSENT_PENALTY,
            lp.PRESEASON_FADE_MATCHES) == before


def test_sweep_restores_the_constants_even_when_a_cell_raises(tmp_path, monkeypatch):
    """The restore must be in a finally, not on the happy path."""
    before = (lp.PRESEASON_ONLY_PRIOR, lp.PRESEASON_ABSENT_PENALTY,
              lp.PRESEASON_FADE_MATCHES)
    monkeypatch.setattr(bt, "friendly_seasons", lambda: ["2024-2025", "2025-2026"])
    stats = pd.DataFrame(sum(
        (_league_rows(s, [1], range(14))
         for s in ("2023-2024", "2024-2025", "2025-2026")), []))
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise RuntimeError("cell exploded")

    monkeypatch.setattr(bt, "load_league_stats", lambda: stats)
    monkeypatch.setattr(bt, "run_backtest", _boom)
    with pytest.raises(RuntimeError):
        bt.sweep()
    assert calls["n"] == 1
    assert (lp.PRESEASON_ONLY_PRIOR, lp.PRESEASON_ABSENT_PENALTY,
            lp.PRESEASON_FADE_MATCHES) == before, (
        "a crashed sweep must not leave swept constants in the live module")
