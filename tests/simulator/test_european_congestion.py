"""Tests for Phase 0b.5 European + Coppa Italia congestion."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from features.european_congestion import _count_recent, _days_since_last


def _extras():
    return {
        "TeamA": pd.DataFrame({
            "team": ["TeamA"] * 3,
            "match_date": pd.to_datetime(["2024-03-01", "2024-03-08", "2024-03-22"]),
            "source": ["european", "european", "coppa"],
        }).sort_values("match_date"),
        "TeamB": pd.DataFrame({
            "team": ["TeamB"],
            "match_date": pd.to_datetime(["2024-02-15"]),
            "source": ["coppa"],
        }).sort_values("match_date"),
    }


def test_count_recent_basic():
    extras = _extras()
    # TeamA as of 2024-03-10, 7-day window includes 2024-03-08 only
    n = _count_recent("TeamA", pd.Timestamp("2024-03-10"), 7, extras, "european")
    assert n == 1
    # 14-day window includes 2024-03-01 + 2024-03-08
    n14 = _count_recent("TeamA", pd.Timestamp("2024-03-10"), 14, extras, "european")
    assert n14 == 2


def test_count_recent_respects_source():
    extras = _extras()
    # TeamA 14-day window, coppa only → just 0 because coppa match is on 2024-03-22 (after cutoff)
    n_coppa = _count_recent("TeamA", pd.Timestamp("2024-03-10"), 30, extras, "coppa")
    assert n_coppa == 0
    # TeamA with cutoff after March 22 catches the coppa match
    n_coppa_later = _count_recent("TeamA", pd.Timestamp("2024-03-25"), 7, extras, "coppa")
    assert n_coppa_later == 1


def test_count_recent_excludes_current_match_date():
    """Match on cutoff_date itself should not count (strict <)."""
    extras = _extras()
    n = _count_recent("TeamA", pd.Timestamp("2024-03-08"), 7, extras, "european")
    # Window is [2024-03-01, 2024-03-08) — exclusive of cutoff → 2024-03-01 is inside
    assert n == 1


def test_days_since_last_returns_nan_when_no_history():
    extras = _extras()
    v = _days_since_last("TeamC", pd.Timestamp("2024-03-10"), extras, "european")
    assert pd.isna(v)


def test_days_since_last_computes_correctly():
    extras = _extras()
    v = _days_since_last("TeamA", pd.Timestamp("2024-03-15"), extras, "european")
    # Most recent european match on 2024-03-08 → 7 days
    assert v == 7.0
