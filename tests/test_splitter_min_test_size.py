"""A walk-forward split whose test season holds ten matches is not a fold.

Second half of the 2026-08-25 finding. `gate_folds` stopped an undersized fold
from voting on the catboost_no_odds release, but the same fold was still being
averaged by every OTHER consumer of TimeSeriesSplitter: the Optuna objective
(ml/tuning.py), the ensemble CV headline, feature selection, and
ml/calibration.py's `splits[-1]` calibration source.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.config import ValidationConfig
from ml.data import TimeSeriesSplitter


def _seasons(sizes: dict[str, int]) -> pd.Series:
    return pd.Series(
        [s for season, n in sizes.items() for s in [season] * n], name="_season"
    )


@pytest.fixture
def august_2026() -> pd.Series:
    """2017-2018 through 2025-2026 complete, then a ten-match 2026-2027 —
    exactly the shape of the live feature table on 2026-08-25."""
    sizes = {f"20{17 + i}-20{18 + i}": 760 for i in range(9)}
    sizes["2026-2027"] = 10
    assert list(sizes)[-2:] == ["2025-2026", "2026-2027"]
    return _seasons(sizes)


def test_a_ten_match_season_does_not_become_a_fold(august_2026):
    cfg = ValidationConfig()

    # True positive: without the guard this really does produce a 2026-2027
    # fold, so the assertion below is not vacuous.
    unguarded = TimeSeriesSplitter(
        ValidationConfig(min_test_matches=0)
    ).generate_splits(august_2026)
    assert ["2026-2027"] in [test for _train, test in unguarded]

    splits = TimeSeriesSplitter(cfg).generate_splits(august_2026)

    assert ["2026-2027"] not in [test for _train, test in splits]
    assert len(splits) == len(unguarded) - 1


def test_the_calibration_source_moves_off_the_ten_match_fold(august_2026):
    """ml/calibration.py takes splits[-1]; that must not be the tiny season."""
    splits = TimeSeriesSplitter(ValidationConfig()).generate_splits(august_2026)

    _train, test = splits[-1]

    assert test == ["2025-2026"]


def test_a_full_history_is_unchanged_by_the_guard():
    seasons = _seasons({f"20{17 + i}-20{18 + i}": 760 for i in range(7)})

    guarded = TimeSeriesSplitter(ValidationConfig()).generate_splits(seasons)
    unguarded = TimeSeriesSplitter(
        ValidationConfig(min_test_matches=0)
    ).generate_splits(seasons)

    assert guarded == unguarded


def test_a_partial_season_counts_once_it_is_big_enough():
    """The current season rejoins CV on its own, without a code change."""
    sizes = {f"20{17 + i}-20{18 + i}": 760 for i in range(6)}
    sizes["2026-2027"] = ValidationConfig().min_test_matches

    splits = TimeSeriesSplitter(ValidationConfig()).generate_splits(_seasons(sizes))

    assert ["2026-2027"] in [test for _train, test in splits]


def test_a_history_too_short_to_guard_still_returns_folds(caplog):
    """Never hand back an empty split list — that would abort a retrain."""
    seasons = _seasons({f"20{17 + i}-20{18 + i}": 20 for i in range(7)})

    with caplog.at_level("ERROR"):
        splits = TimeSeriesSplitter(ValidationConfig()).generate_splits(seasons)

    assert splits
    assert "min_test_matches" in caplog.text


def test_the_no_odds_trainer_keeps_its_final_fold():
    """retrain_no_odds_catboost opts out on purpose.

    --walkforward-final ships the LAST fold's model. The ten-match fold is what
    makes that model one trained through 2025-2026 and blind to the season it
    predicts; dropping it would silently cut a season from production. Its
    release gate filters undersized folds separately, via gate_folds.
    """
    import inspect

    from scripts.models import retrain_no_odds_catboost as trainer

    src = inspect.getsource(trainer.walk_forward_validate)

    assert "min_test_matches=0" in src
