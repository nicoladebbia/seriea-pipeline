"""The main ingest wrote scores but never the `result` column.

`update_matches_parquet` — the Sofascore path that ingests almost every row —
built each row with scores, half-time scores and 40+ team stats and simply
omitted `result`. Only `_fallback_ingest_from_results` (the results.json
recovery path) ever set it.

That is invisible until you notice what reads it. Every consumer that filters
`result.notna()` silently drops the row: `feedback_analyzer` grades against
exactly that filter, so a match with a perfectly good 3-0 on disk was treated
as never played.

Measured 2026-09-07: 141 finished matches carried scores and a null result --
premier_league 2025-26 (71) and 2026-27 (30), serie_a 2025-26 (30) and 2026-27
(10), every one of them `data_source == "sofascore"`, spanning 2026-04-10 to
2026-09-06. The whole of EPL 2026-27 was ungradable.

`result` is "H"/"A"/"D" here (11,877 existing rows, no other spelling).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.data.matchday_updater import _result_from_scores


class TestResultFromScores:
    @pytest.mark.parametrize("hs,aw,expected", [
        (3, 0, "H"), (0, 1, "A"), (2, 2, "D"), (0, 0, "D"),
        (3.0, 0.0, "H"),          # the ingest carries floats
    ])
    def test_derives_the_existing_spelling(self, hs, aw, expected):
        assert _result_from_scores(hs, aw) == expected

    @pytest.mark.parametrize("hs,aw", [
        (None, None), (1, None), (None, 1), (float("nan"), 1), (1, float("nan")),
    ])
    def test_an_unplayed_match_has_no_result(self, hs, aw):
        """A fixture with no score must stay None, not become a phantom draw."""
        assert _result_from_scores(hs, aw) is None


def test_the_spelling_matches_what_is_already_on_disk():
    """Guard against introducing a second convention ("HOME"/"1"/...)."""
    path = "data/parsed/matches.parquet"
    try:
        m = pd.read_parquet(path, columns=["result"])
    except (OSError, ValueError):
        pytest.skip("matches.parquet not available")
    spellings = set(m["result"].dropna().unique())
    assert spellings <= {"H", "A", "D"}, f"unexpected result spellings: {spellings}"
    assert {_result_from_scores(1, 0), _result_from_scores(0, 1),
            _result_from_scores(0, 0)} <= spellings


def test_no_finished_match_is_left_without_a_result():
    """The regression itself: scores present, result absent."""
    path = "data/parsed/matches.parquet"
    try:
        m = pd.read_parquet(path, columns=["home_score", "away_score", "result"])
    except (OSError, ValueError):
        pytest.skip("matches.parquet not available")
    orphan = m[m.home_score.notna() & m.away_score.notna() & m.result.isna()]
    assert orphan.empty, (
        f"{len(orphan)} finished matches have scores but no result — every "
        "consumer filtering result.notna() drops them silently")


# --- Grader join window -------------------------------------------------
#
# A prediction is archived pre-kickoff against the fixture date known at the
# time, and Serie A moves kickoffs afterwards. The grader joined on the date
# with a ±1-day fallback, so a fixture that shifted by two days stayed
# permanently ungraded: measured 2026-09-07 across the whole archive, 185
# entries land on the exact day, 10 at ±1 and 4 at ±2 — those 4 never graded.
#
# ±3 is safe because a Serie A ordered pair (home, away) plays once per season:
# ZERO archive entries had more than one candidate for the same pair even
# within ±30 days.

def _seed_grader(tmp_path, monkeypatch, pred_date: str, actual_date: str):
    import json as _json
    import pandas as pd
    from scripts.analysis import feedback_analyzer as fa

    (tmp_path / "upcoming").mkdir(parents=True, exist_ok=True)
    (tmp_path / "parsed").mkdir(parents=True, exist_ok=True)
    arc = tmp_path / "upcoming" / "predictions_archive.json"
    mp = tmp_path / "parsed" / "matches.parquet"
    arc.write_text(_json.dumps({
        f"{pred_date}_Inter_Roma": {
            "home_team": "Inter", "away_team": "Roma", "date": pred_date,
            "predicted_outcome": "HOME", "confidence": 0.5,
        }
    }))
    pd.DataFrame([{
        "match_date": actual_date, "home_team": "Inter", "away_team": "Roma",
        "home_score": 2, "away_score": 0, "result": "H", "league": "serie_a",
    }]).to_parquet(mp, index=False)
    monkeypatch.setattr(fa, "ARCHIVE_PATH", arc)
    monkeypatch.setattr(fa, "MATCHES_PATH", mp)
    return fa


def test_a_kickoff_that_moved_two_days_is_still_graded(tmp_path, monkeypatch):
    """The true positive: ±2 is outside the old ±1 window, so this test fails
    against the previous grader rather than passing vacuously."""
    from datetime import date, timedelta
    pred = date(2026, 4, 10)
    actual = pred + timedelta(days=2)
    assert abs((actual - pred).days) == 2, "the drift under test must exceed the old ±1 window"
    fa = _seed_grader(tmp_path, monkeypatch, pred.isoformat(), actual.isoformat())
    rows = fa.match_predictions_to_results(league=None)
    assert len(rows) == 1, "a two-day kickoff shift must not leave the prediction ungraded"
    assert rows[0]["actual_outcome"] == "HOME"


def test_the_exact_day_still_grades(tmp_path, monkeypatch):
    fa = _seed_grader(tmp_path, monkeypatch, "2026-04-10", "2026-04-10")
    assert len(fa.match_predictions_to_results(league=None)) == 1


def test_a_fixture_a_fortnight_away_is_not_silently_matched(tmp_path, monkeypatch):
    """The window widened to 3 days, not to 'any fixture with the same pair' —
    a date that far off is a data defect (the archive holds one 2026-05-04 that
    was really 2026-04-05, a day/month transposition) and must stay visible."""
    fa = _seed_grader(tmp_path, monkeypatch, "2026-05-04", "2026-04-05")
    assert fa.match_predictions_to_results(league=None) == []
