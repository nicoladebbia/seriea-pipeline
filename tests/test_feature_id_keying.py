"""The feature id-keying health guard must fire when the derived layer re-mints
Sofascore numeric match_ids, and must NOT fire on canonical ids — including the
shape trap where a legitimate all-digit id is a real matches.parquet key.

Why this guard exists: the 64 shot features join canonical-only
(features/shot_level_xg.py:_map_to_canonical). A re-minted numeric id silently
gives those rows ZERO shot columns instead of erroring. This guard turns that
silent degradation into a loud CRITICAL. See DATA_CATALOG features row 2.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.pipeline import health_check as hc

CUR = "2025-2026"


def _canon(n: int) -> list[str]:
    return [f"2026-05-{d:02d}_Home{d}_Away{d}" for d in range(1, n + 1)]


def _write(root, canonical_ids: list[str], feature_ids: list[str]) -> None:
    (root / "parsed").mkdir(parents=True, exist_ok=True)
    (root / "features").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "match_id": canonical_ids,
        "league": ["serie_a"] * len(canonical_ids),
        "season": [CUR] * len(canonical_ids),
    }).to_parquet(root / "parsed" / "matches.parquet", index=False)
    pd.DataFrame({
        "match_id": feature_ids,
        "season": [CUR] * len(feature_ids),
    }).to_parquet(root / "features" / f"features_serie_a.parquet", index=False)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    # check_data_quality references the module-global DATA_DIR; redirect it so the
    # test never touches real data.
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    return tmp_path


def _guard(root) -> dict:
    return hc.check_data_quality()["feature_id_keying"]


def test_all_canonical_is_ok(data_dir):
    ids = _canon(20)
    _write(data_dir, ids, ids)
    r = _guard(data_dir)
    assert r["status"] == "OK"
    assert r["per_league"]["serie_a"]["orphan_ids"] == 0
    assert r["current_season"] == CUR


def test_systemic_remint_is_critical(data_dir):
    ids = _canon(20)
    # >1 matchweek (15 of 20) reverted to Sofascore numeric keys
    feats = [str(13981600 + i) for i in range(15)] + ids[15:]
    _write(data_dir, ids, feats)
    r = _guard(data_dir)
    assert r["status"] == "CRITICAL"
    assert r["per_league"]["serie_a"]["orphan_ids"] == 15
    assert any("systemic re-mint" in i for i in r["issues"])


def test_single_transient_remint_is_warning_not_critical(data_dir):
    ids = _canon(20)
    feats = ["13981699"] + ids[1:]  # a single new-match window orphan
    _write(data_dir, ids, feats)
    r = _guard(data_dir)
    assert r["status"] == "WARNING"
    assert r["per_league"]["serie_a"]["orphan_ids"] == 1
    assert any("transient" in i for i in r["issues"])


def test_all_digit_fbref_hash_that_is_a_real_match_is_not_flagged(data_dir):
    """The shape trap: '02493616' is all-digits but if it's a genuine
    matches.parquet key it is canonical, not an orphan. Classification is by
    membership, never by .isdigit()."""
    ids = _canon(19) + ["02493616"]
    _write(data_dir, ids, ids)
    r = _guard(data_dir)
    assert r["status"] == "OK", "an all-digit id present in matches.parquet must not be shape-flagged"
    assert r["per_league"]["serie_a"]["orphan_ids"] == 0
