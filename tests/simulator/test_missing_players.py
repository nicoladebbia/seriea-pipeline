"""Tests for Phase 0b.3 missing players features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from features.missing_players import _classify, _parse_match_json


def test_classify_injury_keywords():
    assert _classify("ACL Knee Injury", "missing")["injury"]
    assert _classify("hamstring injury", "missing")["injury"]
    assert _classify("muscle strain", "missing")["injury"]
    assert _classify("", "doubtful")["doubtful"]
    assert not _classify("", "doubtful")["injury"]


def test_classify_suspended_keywords():
    assert _classify("yellow card accumulation", "missing")["suspended"]
    assert _classify("Red card suspension", "missing")["suspended"]


def test_classify_doubtful_flag():
    assert _classify("Possible doubt", "doubtful")["doubtful"]
    assert not _classify("Possible doubt", "missing")["doubtful"]


def test_parse_match_json_counts(tmp_path):
    fake = {
        "match_id": "12345",
        "home_lineup": {
            "missing_players": [
                {"player": {"id": 1}, "type": "missing", "description": "ACL Knee Injury"},
                {"player": {"id": 2}, "type": "doubtful", "description": "Muscle strain"},
                {"player": {"id": 3}, "type": "missing", "description": "suspension"},
            ]
        },
        "away_lineup": {
            "missing_players": [
                {"player": {"id": 10}, "type": "missing", "description": "hamstring"},
            ]
        },
    }
    p = tmp_path / "12345.json"
    p.write_text(json.dumps(fake))
    rec = _parse_match_json(p)
    assert rec is not None
    assert rec["home_missing_count"] == 3
    assert rec["home_missing_injury_count"] == 2  # ACL + Muscle strain
    assert rec["home_missing_suspended_count"] == 1
    assert rec["home_missing_doubtful_count"] == 1
    assert rec["away_missing_count"] == 1
    assert rec["away_missing_injury_count"] == 1


def test_parse_match_json_handles_empty_lineup(tmp_path):
    fake = {"match_id": "999", "home_lineup": {}, "away_lineup": {}}
    p = tmp_path / "999.json"
    p.write_text(json.dumps(fake))
    rec = _parse_match_json(p)
    assert rec["home_missing_count"] == 0
    assert rec["away_missing_count"] == 0


def test_parse_match_json_returns_none_on_bad_file(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{invalid json")
    assert _parse_match_json(p) is None
