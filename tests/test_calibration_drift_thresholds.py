"""The 1X2 calibration-drift check must not alarm on its own estimator noise.

Two defects are pinned here, both found by measurement on 2026-08-25.

1. ECE sums ``|conf - acc|``, so per-bin sampling noise always ADDS and never
   cancels — the estimator is biased upward, and badly at small n. The check
   ran a 100-prediction window against a fixed 0.06 WARNING / 0.10 CRITICAL.
   A null simulation over the live confidence distribution put a PERFECTLY
   calibrated model at a median ECE of 0.079, tripping WARNING 76% of the time
   and CRITICAL 25% of the time. The live 0.0931 reading (p = 0.33) was
   indistinguishable from perfect calibration. The threshold was measuring the
   window size, not the model.

2. There was no staleness gate at all. The newest graded prediction was
   2026-05-19 — the check reported a three-month-old number as live drift
   every 30 minutes all summer.

The first test below is the true positive for defect 1: it constructs a model
that is perfectly calibrated BY CONSTRUCTION and asserts the old constant
would have flagged it while the new comparison does not. Without that
assertion, a check hard-wired to "OK" would pass every other test here.
"""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import scripts.pipeline.health_check as hc

# ─── defect 1: the threshold sat below the estimator's own noise floor ───

def _perfectly_calibrated_window(n=100, seed=7):
    """Confidences spread over bins, outcomes drawn as Bernoulli(confidence).

    That draw IS a perfectly calibrated model — there is no drift to find in
    it. Whatever ECE it produces is the estimator's floor, not a defect.
    """
    rng = np.random.default_rng(seed)
    probs = list(np.round(rng.uniform(0.35, 0.75, n), 4))
    hits = [int(rng.random() < q) for q in probs]
    return probs, hits


def test_a_perfectly_calibrated_model_is_not_reported_as_drift(tmp_path, monkeypatch):
    """Runs through check_calibration_drift so reverting the threshold fails here."""
    probs, hits = _perfectly_calibrated_window()
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    _write_archive(tmp_path, archived_at=_iso(3), probs=probs, hits=hits)

    ece, _ = hc._confidence_ece(probs, hits)
    # The true positive: the retired constant DOES fire on this model. If this
    # assertion ever fails, the one below proves nothing.
    assert ece > 0.06, (
        f"ECE {ece:.4f} no longer trips the old 0.06 constant, so this test "
        "no longer demonstrates the bug it exists to pin"
    )

    out = hc.check_calibration_drift()["calibration_1x2"]
    assert out["ece"] == pytest.approx(ece, abs=1e-4)
    assert out["status"] == "OK", (
        f"a model that is perfectly calibrated by construction was flagged "
        f"{out['status']} (ece={out['ece']}, null p90={out.get('null_ece_p90')})"
    )


def test_the_null_floor_at_this_window_size_exceeds_the_retired_constant():
    """Why the old threshold was unusable, stated as an assertion."""
    probs, _ = _perfectly_calibrated_window()
    median, p90, _ = hc._calibration_null_floor(probs)
    assert median > 0.06, (
        f"null median {median:.4f} — if the floor really sat under 0.06 the "
        "old fixed threshold would have been fine and this fix is unmotivated"
    )
    assert p90 > median


def test_real_miscalibration_is_still_caught():
    """The fix must not have bought quiet by making the check unable to fire."""
    n = 100
    probs = [0.9] * n
    hits = [1] * 30 + [0] * 70  # claims 90% confident, right 30% of the time
    ece, _ = hc._confidence_ece(probs, hits)
    _, _, p99 = hc._calibration_null_floor(probs)
    assert ece > p99, f"gross miscalibration (ece={ece:.4f}) slipped past p99={p99:.4f}"


def test_the_floor_shrinks_as_the_window_grows():
    """Sanity on the mechanism: the bias is a small-sample effect."""
    rng = np.random.default_rng(3)
    small = list(np.round(rng.uniform(0.35, 0.75, 100), 4))
    large = list(np.round(rng.uniform(0.35, 0.75, 1000), 4))
    assert hc._calibration_null_floor(large)[0] < hc._calibration_null_floor(small)[0]


def test_the_floor_is_deterministic_across_calls():
    """An unattended monitor must not flap between runs on identical data."""
    probs, _ = _perfectly_calibrated_window()
    assert hc._calibration_null_floor(probs) == hc._calibration_null_floor(probs)


def test_the_null_floor_and_the_scalar_ece_bin_identically():
    """Pins the two implementations together — they must not drift apart.

    The 4 entries at 0.55 land in a bin under CALIB_MIN_BIN. Both sides must
    drop it and both must divide by the FULL window, not by the bins they
    kept. If either stopped skipping the short bin, that bin contributes
    0.55 * 4 / 44 and neither result stays at zero.
    """
    probs = [1.0] * 20 + [0.0] * 20 + [0.55] * 4
    hits = [1] * 20 + [0] * 20 + [0] * 4  # the short bin is maximally wrong

    ece, bins_used = hc._confidence_ece(probs, hits)
    assert bins_used == 2
    assert ece == pytest.approx(0.0, abs=1e-12)
    # Deterministic bins => a perfectly calibrated draw has zero spread.
    assert hc._calibration_null_floor(probs, sims=200) == pytest.approx(
        (0.0, 0.0, 0.0), abs=1e-12
    )


# ─── defect 2: no staleness gate ───

def _write_archive(tmp_path, *, archived_at, n=100, conf=0.55, probs=None, hits=None):
    """Minimal archive + results that the check can join and grade.

    Pass probs/hits to control the calibration of the graded window; the
    default is a flat 0.55 at a 50% hit rate.
    """
    if probs is None:
        probs = [conf] * n
    if hits is None:
        hits = [int(i % 2 == 0) for i in range(n)]
    (tmp_path / "upcoming").mkdir(parents=True, exist_ok=True)
    (tmp_path / "parsed").mkdir(parents=True, exist_ok=True)
    arch, rows = {}, []
    for i in range(len(probs)):
        q = float(probs[i])
        home, away, date = f"H{i}", f"A{i}", "2026-04-01"
        key = f"{home} vs {away}_{date}"
        hit = bool(hits[i])
        arch[key] = {
            "archived_at": archived_at,
            "betting_probabilities": {
                "home": q, "draw": (1 - q) / 2, "away": (1 - q) / 2,
            },
            "probabilities": {
                "home": q, "draw": (1 - q) / 2, "away": (1 - q) / 2,
            },
        }
        rows.append({
            "match_date": date, "home_team": home, "away_team": away,
            "home_score": 2 if hit else 0, "away_score": 0 if hit else 2,
        })
    (tmp_path / "upcoming" / "predictions_archive.json").write_text(json.dumps(arch))
    pd.DataFrame(rows).to_parquet(tmp_path / "parsed" / "matches.parquet")


def _iso(days_ago):
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


def test_a_window_from_last_season_is_skipped_not_reported_as_live_drift(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    _write_archive(tmp_path, archived_at=_iso(98))  # the live 2026-08-25 state
    out = hc.check_calibration_drift()["calibration_1x2"]
    assert out["status"] == "SKIP"
    assert "98d old" in out["reason"]
    assert out["ece"] is None, "a stale window must not publish a drift number"


def test_a_current_window_is_still_graded(tmp_path, monkeypatch):
    """True positive for the gate: it must not skip everything."""
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    _write_archive(tmp_path, archived_at=_iso(3))
    out = hc.check_calibration_drift()["calibration_1x2"]
    assert out["status"] != "SKIP", out.get("reason")
    assert out["ece"] is not None
    assert out["newest_graded_age_days"] == pytest.approx(3, abs=0.1)


def test_the_reported_number_carries_its_own_age(tmp_path, monkeypatch):
    """A reader must be able to tell a live reading from a stale one."""
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    _write_archive(tmp_path, archived_at=_iso(3))
    out = hc.check_calibration_drift()["calibration_1x2"]
    for key in ("newest_graded_age_days", "null_ece_p90", "null_ece_p99", "ece_excess"):
        assert key in out, f"{key} missing — the ECE is uninterpretable without it"


def test_iso_age_days_reads_a_naive_stamp_as_utc():
    """predictions_archive.json writes naive stamps; the old helper raised on them."""
    stamp = (datetime.now(UTC) - timedelta(days=10)).replace(
        tzinfo=None
    ).isoformat()
    assert hc._iso_age_days(stamp) == pytest.approx(10, abs=0.01)
    assert hc._iso_age_days((datetime.now(UTC) - timedelta(days=10)).isoformat()) \
        == pytest.approx(10, abs=0.01)


def test_iso_age_days_returns_none_rather_than_raising():
    for bad in ("", None, "not-a-date", "2026-13-45"):
        assert hc._iso_age_days(bad) is None
