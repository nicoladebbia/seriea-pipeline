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
    # auto_retrain imports this inside its body, so stubbing wr._notify does not
    # cover it — without this line every run posted the real matchweek card.
    from scripts.pipeline import notify as _notify_mod
    monkeypatch.setattr(_notify_mod, "notify_matchweek_summary", lambda **k: {})
    monkeypatch.setattr(wr, "_is_last_week_of_month", lambda: False)
    monkeypatch.setattr(
        wr, "quick_retrain",
        lambda dry_run=False: {"mode": "quick", "promoted": promoted,
                               **({"dry_run": True} if dry_run else {})},
    )
    for aux in ("retrain_no_odds", "retrain_xg_models", "retrain_draw_detector",
                "_retrain_ou_classifiers"):
        monkeypatch.setattr(wr, aux, lambda dry_run=False: {"promoted": False})
    monkeypatch.setattr(wr, "_refresh_predictions", lambda: None)
    monkeypatch.setattr(wr, "_refresh_goal_predictions", lambda: None)
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


# --- the O/U classifiers retrain whether or not the ensemble is promoted ------
#
# Until 2026-09-05 _retrain_ou_classifiers() and _refresh_predictions() lived
# inside the `if promote:` branch of quick_retrain / full_retrain. The O/U
# models are the ones behind the ONLY enabled betting markets and carry their
# own promotion gate, so an ensemble held "within tolerance" froze the money
# model for nothing — and when the ensemble WAS promoted the refresh ran before
# catboost_no_odds and the O/U classifiers had even been retrained.


def _run_auto_with_spies(monkeypatch, *, dry_run=False, ensemble_promoted=False,
                         ou_promoted=False, no_odds_promoted=False, refresh_ok=True):
    """auto_retrain with every trainer stubbed; returns (result, call log)."""
    from scripts.pipeline import weekly_retrain as wr

    calls = []
    monkeypatch.setattr(wr, "get_matchweek_status", lambda *a, **k: _fake_status())
    monkeypatch.setattr(wr, "_load_retrain_state", lambda: {})
    monkeypatch.setattr(wr, "_save_retrain_state", lambda st: None)
    monkeypatch.setattr(wr, "_archive_current_models", lambda *a, **k: None)
    monkeypatch.setattr(wr, "_notify", lambda *a, **k: None)
    # auto_retrain imports this inside its body, so stubbing wr._notify does not
    # cover it — without this line every run posted the real matchweek card.
    from scripts.pipeline import notify as _notify_mod
    monkeypatch.setattr(_notify_mod, "notify_matchweek_summary", lambda **k: {})
    monkeypatch.setattr(wr, "_is_last_week_of_month", lambda: False)
    monkeypatch.setattr(
        wr, "quick_retrain",
        lambda dry_run=False: (calls.append(("quick", dry_run)) or
                               {"mode": "quick", "promoted": ensemble_promoted,
                                **({"dry_run": True} if dry_run else {})}),
    )

    def _aux(name, promoted):
        def run(dry_run=False):
            calls.append((name, dry_run))
            return {"model": name, "promoted": promoted}
        return run

    monkeypatch.setattr(wr, "retrain_no_odds", _aux("no_odds", no_odds_promoted))
    monkeypatch.setattr(wr, "retrain_xg_models", _aux("xg", False))
    monkeypatch.setattr(wr, "retrain_draw_detector", _aux("draw_detector", False))
    monkeypatch.setattr(wr, "_retrain_ou_classifiers", _aux("over_under", ou_promoted))
    monkeypatch.setattr(wr, "_refresh_predictions",
                        lambda: (calls.append(("refresh", None)), refresh_ok)[1])
    monkeypatch.setattr(wr, "_refresh_goal_predictions",
                        lambda: calls.append(("goal_refresh", None)))
    result = wr.auto_retrain(dry_run=dry_run)
    return result, calls


def test_ou_classifiers_retrain_when_the_ensemble_is_held(monkeypatch):
    """The bug: a held ensemble skipped the money model's retrain entirely."""
    _, calls = _run_auto_with_spies(monkeypatch, ensemble_promoted=False)
    assert ("over_under", False) in calls


def test_an_ou_promotion_under_a_held_ensemble_refreshes_predictions(monkeypatch):
    """A newly promoted O/U model must reach goal_predictions.json today, not
    whenever the next morning pipeline happens to run."""
    result, calls = _run_auto_with_spies(monkeypatch, ensemble_promoted=False,
                                         ou_promoted=True)
    assert calls.count(("refresh", None)) == 1
    assert result["predictions_refreshed_for"] == ["over_under"]


def test_predictions_refresh_once_and_only_after_every_retrain_step(monkeypatch):
    """Ensemble AND O/U promoted: one refresh, after the last trainer ran."""
    result, calls = _run_auto_with_spies(monkeypatch, ensemble_promoted=True,
                                         ou_promoted=True, no_odds_promoted=True)
    assert calls.count(("refresh", None)) == 1
    names = [c[0] for c in calls]
    assert names.index("refresh") > max(names.index("over_under"),
                                        names.index("no_odds"),
                                        names.index("quick"))
    assert result["predictions_refreshed_for"] == ["ensemble", "no_odds", "over_under"]


def test_goal_predictions_are_regenerated_right_after_the_engine_refresh(monkeypatch):
    """predictions.json is the engine's file; scan_ou_market prices
    goal_predictions.json, which only pipeline Step 19 wrote. Both must move."""
    _, calls = _run_auto_with_spies(monkeypatch, ou_promoted=True)
    names = [c[0] for c in calls]
    assert names.count("goal_refresh") == 1
    assert names.index("goal_refresh") == names.index("refresh") + 1


def test_a_failed_engine_refresh_does_not_regenerate_goal_predictions_from_stale_input(monkeypatch):
    result, calls = _run_auto_with_spies(monkeypatch, ou_promoted=True, refresh_ok=False)
    assert ("goal_refresh", None) not in calls
    assert result.get("predictions_refresh_failed") is True
    assert "predictions_refreshed_for" not in result


def test_nothing_promoted_means_no_refresh(monkeypatch):
    result, calls = _run_auto_with_spies(monkeypatch)
    assert ("refresh", None) not in calls
    assert "predictions_refreshed_for" not in result


def test_a_dry_run_reaches_the_ou_trainer_as_a_dry_run_and_refreshes_nothing(monkeypatch):
    """--dry-run must preview the O/U gate too — and never write the slate."""
    _, calls = _run_auto_with_spies(monkeypatch, dry_run=True, ensemble_promoted=True,
                                    ou_promoted=True)
    assert ("over_under", True) in calls
    assert ("refresh", None) not in calls


def test_the_ensemble_promote_branches_no_longer_own_the_ou_retrain():
    """Pin the decoupling in the source so it cannot quietly move back."""
    import inspect

    from scripts.pipeline import weekly_retrain as wr

    for fn in (wr.quick_retrain, wr.full_retrain):
        src = inspect.getsource(fn)
        assert "_retrain_ou_classifiers()" not in src, fn.__name__
        assert "_refresh_predictions()" not in src, fn.__name__


def test_retrain_ou_classifiers_reports_the_trainer_verdict_per_line(monkeypatch):
    """Maps the trainer report onto the auxiliary-model contract, real and dry."""
    import scripts.models.train_over_under as tou
    from scripts.pipeline import weekly_retrain as wr

    history = []
    monkeypatch.setattr(wr, "_append_metrics_history", history.append)
    seen = {}

    def fake_train(lines, top_k, n_tune_trials, dry_run=False):
        seen["dry_run"] = dry_run
        return {"timestamp": "t", "lines": {
            "1.5": {"promoted": not dry_run, "would_promote": True, "dry_run": dry_run,
                    "promotion_reason": "better", "holdout": {}, "cv_metrics": {}},
            "2.5": {"promoted": False, "would_promote": False, "dry_run": dry_run,
                    "promotion_reason": "worse", "holdout": {}, "cv_metrics": {}},
        }}

    monkeypatch.setattr(tou, "train_over_under", fake_train)

    real = wr._retrain_ou_classifiers()
    assert seen["dry_run"] is False
    assert real == {"model": "over_under", "promoted": True,
                    "lines": {"1.5": "promoted", "2.5": "held"}}
    assert [h["promoted"] for h in history] == [True, False]
    assert not any("dry_run" in h for h in history)

    history.clear()
    dry = wr._retrain_ou_classifiers(dry_run=True)
    assert seen["dry_run"] is True
    assert dry == {"model": "over_under", "promoted": False,
                   "lines": {"1.5": "would_promote", "2.5": "would_hold"}}
    assert all(h.get("dry_run") is True for h in history)


def test_retrain_ou_classifiers_failure_is_an_error_not_a_crash(monkeypatch):
    import scripts.models.train_over_under as tou
    from scripts.pipeline import weekly_retrain as wr

    def boom(**kw):
        raise RuntimeError("no features")

    monkeypatch.setattr(tou, "train_over_under", boom)
    out = wr._retrain_ou_classifiers()
    assert out["promoted"] is False and "no features" in out["error"]


# --- a manual --quick/--full is the WHOLE sequence, from every entry point ----


def _run_forced(monkeypatch, mode, *, dry_run=False, ensemble_promoted=False,
                ou_promoted=False):
    from scripts.pipeline import weekly_retrain as wr

    calls, saved = [], {}
    monkeypatch.setattr(wr, "get_matchweek_status", lambda *a, **k: _fake_status())
    monkeypatch.setattr(wr, "_load_retrain_state", lambda: {})
    monkeypatch.setattr(wr, "_save_retrain_state", lambda st: saved.update(st))
    monkeypatch.setattr(wr, "_archive_current_models",
                        lambda *a, **k: calls.append(("archive", None)))

    def _ens(name):
        def run(dry_run=False):
            calls.append((name, dry_run))
            return {"mode": name, "promoted": ensemble_promoted,
                    **({"dry_run": True} if dry_run else {})}
        return run

    monkeypatch.setattr(wr, "quick_retrain", _ens("quick"))
    monkeypatch.setattr(wr, "full_retrain", _ens("full"))
    for aux in ("retrain_no_odds", "retrain_xg_models", "retrain_draw_detector"):
        monkeypatch.setattr(wr, aux, lambda dry_run=False: {"promoted": False})
    monkeypatch.setattr(
        wr, "_retrain_ou_classifiers",
        lambda dry_run=False: (calls.append(("over_under", dry_run)) or
                               {"model": "over_under", "promoted": ou_promoted}),
    )
    monkeypatch.setattr(wr, "_refresh_predictions",
                        lambda: (calls.append(("refresh", None)), True)[1])
    monkeypatch.setattr(wr, "_refresh_goal_predictions", lambda: None)
    result = wr.forced_retrain(mode, dry_run=dry_run)
    return result, calls, saved


def test_forced_retrain_runs_the_ou_classifiers_and_refreshes_after_a_held_ensemble(monkeypatch):
    """cli.py used to call quick_retrain() directly — after the decoupling that
    silently trained the ensemble only. The manual path must be the whole
    sequence: archive -> ensemble -> auxiliaries (O/U) -> one refresh."""
    result, calls, _ = _run_forced(monkeypatch, "quick", ou_promoted=True)
    names = [c[0] for c in calls]
    assert names[:2] == ["archive", "quick"]
    assert ("over_under", False) in calls
    assert names.index("refresh") > names.index("over_under")
    assert result["predictions_refreshed_for"] == ["over_under"]


def test_forced_full_retrain_uses_the_full_trainer(monkeypatch):
    _, calls, _ = _run_forced(monkeypatch, "full")
    assert ("full", False) in calls and ("quick", False) not in calls


def test_forced_retrain_stamps_the_gate_only_on_a_real_promotion(monkeypatch):
    _, _, saved = _run_forced(monkeypatch, "quick", ensemble_promoted=True)
    assert saved.get("last_retrained_matchweek") == 3
    _, _, saved = _run_forced(monkeypatch, "quick", ensemble_promoted=False)
    assert saved == {}
    _, calls, saved = _run_forced(monkeypatch, "quick", dry_run=True, ensemble_promoted=True)
    assert saved == {} and ("over_under", True) in calls and ("refresh", None) not in calls


def test_forced_retrain_rejects_an_unknown_mode():
    import pytest

    from scripts.pipeline import weekly_retrain as wr

    with pytest.raises(ValueError):
        wr.forced_retrain("auto")


def test_every_manual_entry_point_goes_through_forced_retrain():
    """Pin both callers in source: neither may call the ensemble trainers directly."""
    import inspect
    import re

    from scripts.pipeline import weekly_retrain as wr

    main_src = inspect.getsource(wr.main)
    assert "forced_retrain(" in main_src
    assert "quick_retrain(dry_run" not in main_src and "full_retrain(dry_run" not in main_src

    cli_src = (PROJECT_ROOT / "cli.py").read_text()
    retrain_cmd = cli_src[cli_src.index("def retrain("):cli_src.index("def rollback(")]
    assert "forced_retrain(" in retrain_cmd
    assert not re.search(r"\b(quick|full)_retrain\(", retrain_cmd), (
        "cli.py must not call the ensemble-only trainers directly"
    )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
