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
