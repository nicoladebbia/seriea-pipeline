"""The Phase-7 correction layer is OFF by default, and why.

On 2026-08-25 it put 14 of 20 Serie A fixtures at exactly 0.98/0.01/0.01 and
made 19 of 20 predict HOME. The four component models were fine — a blend of
0.452 was pushed to 0.98 by a single additive EMA.
"""

from __future__ import annotations

import json

import pytest

from ml.correction_layer import CorrectionConfig, RollingCorrector


def test_the_rolling_path_has_no_magnitude_bound():
    """The defect itself: `correction_factor_min/max` guards only the STATIC
    path. RollingCorrector adds its EMA raw, clips to [0.01, 0.99] and
    renormalises, so any EMA large enough saturates the output.

    This test DOCUMENTS broken behaviour rather than desired behaviour. If it
    starts failing because the rolling path grew a bound, that is the fix —
    update this test and re-evaluate re-enabling the layer.
    """
    cfg = CorrectionConfig()
    rc = RollingCorrector(cfg)
    # The live bucket on 2026-08-25: +0.600 on prob_H.
    rc.buckets = {
        rc._get_bucket_key(0.452, 0.220, 0.328): {
            "n": 192, "ema": [0.600, -0.272, -0.327],
        }
    }
    h, d, a = rc.correct(0.452, 0.220, 0.328)

    assert h > 0.97, "expected the documented saturation"
    # The configured clamp would have capped the shift at 1.3x.
    assert h > 0.452 * cfg.correction_factor_max, (
        "the rolling path respected the factor bound — re-check whether the "
        "layer can now be re-enabled"
    )


def test_a_bucket_below_min_samples_is_passthrough():
    """The one guard the rolling path does have, so the fix is targeted."""
    cfg = CorrectionConfig()
    rc = RollingCorrector(cfg)
    rc.buckets = {
        rc._get_bucket_key(0.452, 0.220, 0.328): {
            "n": cfg.rolling_min_samples - 1, "ema": [0.600, -0.272, -0.327],
        }
    }
    assert rc.correct(0.452, 0.220, 0.328) == (0.452, 0.220, 0.328)


def test_shipped_predictions_are_not_saturated():
    """The user-visible contract: no fixture may ship a 0.98 headline while the
    rolling path is unbounded. Guards the regression end-to-end."""
    from config.settings import DATA_DIR

    checked = 0
    for name in ("predictions.json", "predictions_premier_league.json"):
        path = DATA_DIR / "upcoming" / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        matches = payload.get("matches") or payload.get("predictions") or []
        if not matches:
            continue
        checked += 1
        bad = [
            m.get("match")
            for m in matches
            if max((m.get("probabilities") or {}).values() or [0]) >= 0.98
        ]
        assert not bad, f"{name}: saturated predictions {bad[:5]}"
    if not checked:
        pytest.skip("no prediction files on disk")
