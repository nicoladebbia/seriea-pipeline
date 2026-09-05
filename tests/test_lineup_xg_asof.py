"""Understat lineup proxy must describe the PREVIOUS complete season.

Joining the same season was wrong on both ends (2026-09-05): training rows
read full-season totals at matchweek 2 (lookahead), and serving read
season-cumulative stats that barely exist yet — us_squad_depth counted
players with >270 minutes, so every early-season row served 0.0 against a
training band of [17, 31].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.lineup_xg import _compute_understat_lineup_proxy


def _us_players():
    rows = []
    # 2025-2026: Genoa has 12 players with a full season (depth 12)
    for i in range(12):
        rows.append({"team": "Genoa", "season": "2025-2026",
                     "player": f"g{i}", "minutes": 2000, "xg_per_90": 0.30})
    # 2026-2027 (in progress): same players, 180 minutes, inflated rates
    for i in range(12):
        rows.append({"team": "Genoa", "season": "2026-2027",
                     "player": f"g{i}", "minutes": 180, "xg_per_90": 0.90})
    return pd.DataFrame(rows)


def _feature_df():
    return pd.DataFrame([
        {"home_team": "Genoa", "away_team": "Como", "season": "2026-2027"},
        # promoted side: no prior-season Understat rows anywhere
        {"home_team": "Frosinone", "away_team": "Genoa", "season": "2026-2027"},
    ])


def test_proxy_joins_previous_complete_season():
    out = _compute_understat_lineup_proxy(_feature_df(), _us_players())
    # Genoa's 2026-27 row carries the COMPLETE 2025-26 depth, not the
    # in-progress count (which would be 0: nobody is past 270 minutes)
    assert out.loc[0, "home_us_squad_depth"] == 12
    # and the 2025-26 rate, not the inflated 3-matchweek 0.90
    assert abs(out.loc[0, "home_us_top11_xg90_sum"] - 11 * 0.30) < 1e-9
    # away side of row 1 joins the same shifted values
    assert out.loc[1, "away_us_squad_depth"] == 12


def test_first_known_season_gets_nan_not_zero():
    out = _compute_understat_lineup_proxy(_feature_df(), _us_players())
    # Frosinone has no prior Understat season: NaN (filled with training
    # medians downstream), never a fabricated 0.0
    assert np.isnan(out.loc[1, "home_us_squad_depth"])
