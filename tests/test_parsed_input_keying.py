"""The parsed-input keying guard must fire when a per-player parsed input
(player_stats / goalkeeper_stats) carries current-season match_ids that are NOT
in matches.parquet — i.e. hash-keyed inputs that silently drop out of the
canonical joins building player_impact / gk_quality / team_aggregates.

This is the guard the feature_id_keying check could NOT catch: feature-table ids
are always canonical (they come from matches), so the 2026-07-17 hash-keying of
player_stats + goalkeeper_stats left every feature-table guard GREEN while the
whole current season silently went null. Classify by membership in matches, never
by id shape (an FBref hash can be all-digits).
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.pipeline import health_check as hc

CUR = "2025-2026"


def _canon(n: int) -> list[str]:
    return [f"2026-05-{d:02d}_Home{d}_Away{d}" for d in range(1, n + 1)]


def _write(root, canonical_ids, player_ids, gk_ids=None) -> None:
    (root / "parsed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "match_id": canonical_ids,
        "season": [CUR] * len(canonical_ids),
    }).to_parquet(root / "parsed" / "matches.parquet", index=False)
    pd.DataFrame({
        "match_id": player_ids,
        "season": [CUR] * len(player_ids),
    }).to_parquet(root / "parsed" / "player_stats.parquet", index=False)
    pd.DataFrame({
        "match_id": gk_ids if gk_ids is not None else canonical_ids,
        "season": [CUR] * len(gk_ids if gk_ids is not None else canonical_ids),
    }).to_parquet(root / "parsed" / "goalkeeper_stats.parquet", index=False)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "DATA_DIR", tmp_path)
    return tmp_path


def _guard() -> dict:
    return hc.check_data_quality()["parsed_input_keying"]


def test_all_canonical_is_ok(data_dir):
    ids = _canon(20)
    _write(data_dir, ids, ids, ids)
    r = _guard()
    assert r["status"] == "OK"
    assert r["per_file"]["player_stats.parquet"]["orphan_ids"] == 0
    assert r["per_file"]["goalkeeper_stats.parquet"]["orphan_ids"] == 0
    assert r["current_season"] == CUR


def test_whole_season_hash_keyed_is_critical(data_dir):
    ids = _canon(20)
    # the exact 2026-07-17 regression: player_stats keyed by FBref hash
    hashes = [f"{i:08x}" for i in range(20)]
    _write(data_dir, ids, hashes, ids)
    r = _guard()
    assert r["status"] == "CRITICAL"
    assert r["per_file"]["player_stats.parquet"]["orphan_ids"] == 20
    assert any("player_stats.parquet" in i for i in r["issues"])


def test_all_digit_hash_not_misclassified_by_shape(data_dir):
    # an FBref hash can be all-digits; the guard must classify by matches
    # membership, not .isdigit() — these are absent from matches, so CRITICAL
    ids = _canon(20)
    numeric_hashes = [str(13981600 + i) for i in range(20)]
    _write(data_dir, ids, numeric_hashes, ids)
    r = _guard()
    assert r["status"] == "CRITICAL"
    assert r["per_file"]["player_stats.parquet"]["orphan_ids"] == 20


def test_single_orphan_is_warning(data_dir):
    ids = _canon(20)
    gk = ["deadbeef"] + ids[1:]  # one stray hash-keyed gk match
    _write(data_dir, ids, ids, gk)
    r = _guard()
    assert r["status"] == "WARNING"
    assert r["per_file"]["goalkeeper_stats.parquet"]["orphan_ids"] == 1
