#!/usr/bin/env python3
"""Retrain-gate tests — specifically the season rollover.

The gate is ``needs_retrain = completed_matchweeks > last_retrained_matchweek``.
``completed_matchweeks`` is season-scoped (it restarts at 1 every August, derived
from ``latest_season_with_results``), but the persisted counter was not: it kept
climbing across seasons and stood at 67 on 2026-05-25. ``1 > 67`` is false, and a
Serie A season only ever reaches 38 — so the job went silently dead for the whole
of 2026-27, exiting 0 and logging "already retrained".

These tests exercise the ROLLOVER, not the steady state. A suite that only ever
compares two in-season numbers stays green against the exact bug it should pin.

Run with: python3 -m pytest tests/test_weekly_retrain.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.weekly_retrain import _last_retrained_for_season  # noqa: E402


def test_rollover_into_new_season_does_not_inherit_the_old_counter():
    """The bug: May's MW67 suppressed every matchweek of the next season."""
    state = {"last_retrained_matchweek": 67, "last_retrained_season": "2025-2026"}
    assert _last_retrained_for_season(state, "2026-2027") == 0
    # ...and therefore the gate opens on matchweek 1.
    assert 1 > _last_retrained_for_season(state, "2026-2027")


def test_legacy_state_without_a_season_is_treated_as_a_previous_season():
    """State written before the field existed must not suppress the new season."""
    state = {"last_retrained_matchweek": 67, "last_retrained_at": "2026-05-25T22:33:30Z"}
    assert _last_retrained_for_season(state, "2026-2027") == 0


def test_within_the_same_season_the_counter_still_suppresses_a_repeat():
    """The gate must keep doing its real job: no retraining twice on one MW."""
    state = {"last_retrained_matchweek": 5, "last_retrained_season": "2026-2027"}
    assert _last_retrained_for_season(state, "2026-2027") == 5
    assert not (5 > _last_retrained_for_season(state, "2026-2027"))


def test_within_the_same_season_a_new_matchweek_opens_the_gate():
    state = {"last_retrained_matchweek": 5, "last_retrained_season": "2026-2027"}
    assert 6 > _last_retrained_for_season(state, "2026-2027")


def test_empty_state_retrains():
    assert _last_retrained_for_season({}, "2026-2027") == 0


# --- a dry run must not close the gate --------------------------------------
#
# `if result.get("promoted") or dry_run:` stamped last_retrained_matchweek even
# though a dry run promotes nothing (it returns promoted=False by construction).
# So previewing with --dry-run silently suppressed the real Tuesday retrain,
# which then logged "already retrained" and exited 0. Same shape as the season
# rollover above: an unintended write to the gate state kills the job forever.


def _fake_status(mw=3, season="2026-2027"):
    return {
        "season": season,
        "completed_matchweeks": mw,
        "current_matchweek": mw,
        "total_matches": mw * 10,
        "matches_in_progress": 0,
        "last_match_date": "2026-09-01",
        "days_since_last_match": 1,
        "last_retrained_matchweek": 0,
        "needs_retrain": True,
    }


def _run_auto(monkeypatch, dry_run, promoted):
    """Drive auto_retrain past every side effect, capturing only what it saves."""
    from scripts.pipeline import weekly_retrain as wr

    saved = {}
    monkeypatch.setattr(wr, "get_matchweek_status", lambda *a, **k: _fake_status())
    monkeypatch.setattr(wr, "_load_retrain_state", lambda: {})
    monkeypatch.setattr(wr, "_save_retrain_state", lambda st: saved.update(st))
    monkeypatch.setattr(wr, "_archive_current_models", lambda *a, **k: None)
    monkeypatch.setattr(wr, "_notify", lambda *a, **k: None)
    monkeypatch.setattr(wr, "_is_last_week_of_month", lambda: False)
    monkeypatch.setattr(
        wr, "quick_retrain",
        lambda dry_run=False: {"mode": "quick", "promoted": promoted,
                               **({"dry_run": True} if dry_run else {})},
    )
    for aux in ("retrain_no_odds", "retrain_xg_models", "retrain_draw_detector"):
        monkeypatch.setattr(wr, aux, lambda dry_run=False: {"promoted": False})
    wr.auto_retrain(dry_run=dry_run)
    return saved


def test_a_dry_run_does_not_stamp_the_retrain_gate(monkeypatch):
    """The whole point of --dry-run: production state is untouched."""
    assert _run_auto(monkeypatch, dry_run=True, promoted=False) == {}


def test_a_dry_run_that_somehow_reports_promoted_still_stamps_nothing(monkeypatch):
    """Belt and braces: dry_run wins over a promoted flag, never the reverse."""
    assert _run_auto(monkeypatch, dry_run=True, promoted=True) == {}


def test_a_real_promoted_retrain_does_stamp_the_gate(monkeypatch):
    """...and the guard must not break the gate's actual job."""
    saved = _run_auto(monkeypatch, dry_run=False, promoted=True)
    assert saved.get("last_retrained_matchweek") == 3
    assert saved.get("last_retrained_season") == "2026-2027"


def test_a_real_retrain_that_was_not_promoted_leaves_the_gate_open(monkeypatch):
    """A failed retrain must be retried, not marked done."""
    assert _run_auto(monkeypatch, dry_run=False, promoted=False) == {}


# --- the feature store must survive a crashed write -------------------------


def test_feature_writes_are_atomic(tmp_path, monkeypatch):
    """build_features wrote straight onto the live parquet. A crash mid-write
    leaves the production feature store truncated and the next pipeline step
    reads it without complaining. Simulate the crash and check the old file
    is still intact."""
    import pandas as pd

    from features.build import _atomic_to_parquet

    out = tmp_path / "features_serie_a.parquet"
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(out, index=False)

    boom = pd.DataFrame({"a": [9]})
    real_to_parquet = pd.DataFrame.to_parquet

    def exploding(self, path, *a, **k):
        real_to_parquet(self, path, *a, **k)
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", exploding)
    try:
        _atomic_to_parquet(boom, out)
    except OSError:
        pass
    monkeypatch.setattr(pd.DataFrame, "to_parquet", real_to_parquet)

    survived = pd.read_parquet(out)
    assert list(survived["a"]) == [1, 2, 3], "a failed write clobbered the store"


def test_atomic_write_leaves_no_tmp_file_behind(tmp_path):
    import pandas as pd

    from features.build import _atomic_to_parquet

    out = tmp_path / "features_serie_a.parquet"
    _atomic_to_parquet(pd.DataFrame({"a": [1]}), out)
    assert out.exists()
    assert list(tmp_path.glob("*.tmp")) == []


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
