"""The features_quality sparse check must fire on a column that STOPPED being
filled, and must stay silent on one that is merely thin across nine seasons.

Why this test exists (measured 2026-09-07): the check warned on ">90% NaN
all-time" alone, and on the live parquet all 25 columns it named were the
Sofascore match-stat family -- 100% filled for Serie A in both the current and
previous season, 0% for the EPL in both, diluted to 2.5% by nine seasons of
history. A complete, unchanged column read as broken every cycle, so the
Analytics page carried a permanent yellow that trains the reader to ignore it.

The fault worth an alert is the one CLAUDE.md names as the enrichment-bug
detector -- filled historically, empty in the season being played (the
config.SEASONS lag, the frozen derived cache, the hash-vs-canonical key break).

Every fixture column here is >90% NaN all-time, asserted explicitly: that is
what makes the negative cases true rejections rather than columns that simply
never reached the check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pipeline import health_check as hc

OLD, PREV, CUR = "2023-2024", "2024-2025", "2025-2026"
# Enough history to push a fully-filled recent column past 90% NaN all-time --
# the same dilution the real nine-season parquet applies.
N_OLD, N_PREV = 4000, 380
# Deliberately unequal and Serie A-lighter, like the live frame (28 SA / 30 EPL):
# a Serie A-only column then averages just under 0.5 across the mixed current
# season, which is exactly the coin flip a league-blind threshold decided wrong.
N_CUR = {"serie_a": 28, "premier_league": 30}


def _frame(spec: dict[str, dict[tuple[str, str], float]],
           seasons: tuple[str, ...] = (OLD, PREV, CUR)) -> pd.DataFrame:
    """spec: {column: {(league, season): fill_rate}} -- absent pairs are all NaN.

    Rates are applied deterministically to the first ``rate * n`` rows so a
    fixture never depends on a seed.
    """
    sizes = {OLD: lambda lg: N_OLD, PREV: lambda lg: N_PREV, CUR: lambda lg: N_CUR[lg]}
    rows = []
    for league in ("serie_a", "premier_league"):
        for season in seasons:
            n = sizes[season](league)
            for i in range(n):
                row = {
                    "match_id": f"{season}_{league}_{i}",
                    "league": league,
                    "season": season,
                    "home_team": "A", "away_team": "B",
                    "dense": 1.0,
                }
                for col, rates in spec.items():
                    rate = rates.get((league, season), 0.0)
                    row[col] = 1.0 if i < int(round(rate * n)) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    (tmp_path / "features").mkdir(parents=True, exist_ok=True)

    def _run(df: pd.DataFrame) -> dict:
        df.to_parquet(tmp_path / "features" / "features.parquet", index=False)
        return hc.check_data_quality()["features_quality"]
    return _run


def _assert_is_a_true_candidate(df: pd.DataFrame, *cols: str) -> None:
    """Every fixture column must clear the >90%-NaN prefilter.

    Without this, a "not flagged" assertion below could pass because the column
    never reached the rule at all -- a rejection test with no true positive.
    """
    for c in cols:
        nan_rate = df[c].isna().mean()
        assert nan_rate > 0.90, f"{c} is only {nan_rate:.1%} NaN — the old rule would not have flagged it either"


def test_a_column_that_stopped_being_filled_is_flagged(run):
    df = _frame({"lost_it": {("serie_a", PREV): 1.0}})  # filled last season, gone now
    _assert_is_a_true_candidate(df, "lost_it")
    r = run(df)
    assert r["status"] == "WARNING"
    assert r["sparse_columns_regressed"] == 1
    assert any("lost_it (serie_a)" in i for i in r["issues"]), r["issues"]


def test_a_league_only_family_is_not_flagged(run):
    """The live shape: Serie A 1.00 in both seasons, EPL 0.00 in both."""
    df = _frame({"sa_only": {("serie_a", PREV): 1.0, ("serie_a", CUR): 1.0}})
    _assert_is_a_true_candidate(df, "sa_only")
    # Pin the coin flip the league-blind version lost on.
    cur = df[df.season == CUR]
    assert 0.45 < cur["sa_only"].notna().mean() < 0.50
    r = run(df)
    assert r["status"] == "OK", r["issues"]
    assert r["sparse_columns"] == 1 and r["sparse_columns_regressed"] == 0
    assert r["sparse_columns_historical"] == 1


def test_the_two_are_distinguished_in_one_frame(run):
    df = _frame({
        "lost_it": {("serie_a", PREV): 1.0},
        "sa_only": {("serie_a", PREV): 1.0, ("serie_a", CUR): 1.0},
    })
    _assert_is_a_true_candidate(df, "lost_it", "sa_only")
    r = run(df)
    assert r["sparse_columns"] == 2, "both must reach the rule"
    assert r["sparse_columns_regressed"] == 1
    assert any("lost_it" in i for i in r["issues"])
    assert not any("sa_only" in i for i in r["issues"]), r["issues"]


def test_a_column_never_filled_by_anyone_is_not_a_regression(run):
    """Dead all along is not the same as lost this season."""
    # Filled in 8% of ancient rows and nowhere since: >90% NaN all-time (so the
    # old rule flagged it) but nothing was ever lost.
    df = _frame({"never": {("serie_a", OLD): 0.08}})
    _assert_is_a_true_candidate(df, "never")
    r = run(df)
    assert r["status"] == "OK", r["issues"]
    assert r["sparse_columns_regressed"] == 0


def test_a_young_current_season_does_not_false_fire(run):
    """A league with almost no rows played yet cannot prove a column was lost."""
    df = _frame({"lost_it": {("serie_a", PREV): 1.0}})
    # Serie A has played 3 matches this season — under the row floor.
    keep = ~((df.league == "serie_a") & (df.season == CUR) & (df.index % 28 >= 3))
    df = df[keep].reset_index(drop=True)
    r = run(df)
    assert r["sparse_columns_regressed"] == 0, r["issues"]
    assert r["status"] == "OK", r["issues"]


def test_one_season_frame_says_it_could_not_evaluate(run):
    df = _frame({"sa_only": {("serie_a", PREV): 1.0}}, seasons=(PREV,))
    assert len(df) >= 100, "keep the unrelated low-row-count rule out of this assertion"
    r = run(df)
    assert r["sparse_columns_evaluated"] is False
    assert r["sparse_columns_regressed"] == 0
    assert r["status"] == "OK", r["issues"]


def test_the_live_parquet_has_no_regressed_columns():
    """The real frame, through the real path — the 25 sparse columns are all
    the SA-only family, none of them lost this season."""
    from config.settings import DATA_DIR as REAL_DATA_DIR
    if not (REAL_DATA_DIR / "features" / "features.parquet").exists():
        pytest.skip("features.parquet not present")
    r = hc.check_data_quality()["features_quality"]
    assert r["sparse_columns_evaluated"] is True
    assert r["sparse_columns_regressed"] == 0, r["issues"]
