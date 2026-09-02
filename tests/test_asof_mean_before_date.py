"""asof_mean_before_date must be order-independent and strictly pre-date.

The idiom it replaces — sort → shift(1) → expanding().mean() — depends on row
order: sorted team-then-date (add_strength_ratings), a mid-season row's league
average absorbed entire future seasons of alphabetically earlier teams (the
P1b lookahead behind the 23-feature strength/poisson cluster); even
date-sorted, same-day peers leaked in. These tests exercise the mutations the
helper exists to survive: shuffled row order, same-day peers, NaN fixture
rows, and multi-group isolation.
"""

import numpy as np
import pandas as pd

from features.strength import asof_mean_before_date


def _df(rows):
    return pd.DataFrame(rows, columns=["season", "match_date", "g"])


BASE = _df([
    ("s1", "2025-01-01", 2.0),
    ("s1", "2025-01-01", 4.0),   # same-day peer
    ("s1", "2025-01-08", 6.0),
    ("s1", "2025-01-15", 0.0),
    ("s2", "2025-08-01", 10.0),  # different season: isolated
])


def test_strictly_before_date_excludes_same_day_peers():
    out = asof_mean_before_date(BASE, ["season"], "g")
    # Jan 1 rows: nothing earlier -> NaN (NOT each other)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    # Jan 8: mean of both Jan 1 rows = 3.0
    assert out.iloc[2] == 3.0
    # Jan 15: mean of (2, 4, 6) = 4.0
    assert out.iloc[3] == 4.0
    # s2 opener: previous season never bleeds in
    assert np.isnan(out.iloc[4])


def test_order_independence():
    shuffled = BASE.sample(frac=1, random_state=7)
    out = asof_mean_before_date(shuffled, ["season"], "g")
    expected = asof_mean_before_date(BASE, ["season"], "g")
    # Compare by original index, not position
    for idx in BASE.index:
        a, b = out.loc[idx], expected.loc[idx]
        assert (np.isnan(a) and np.isnan(b)) or a == b


def test_nan_rows_contribute_nothing():
    df = _df([
        ("s1", "2025-01-01", 2.0),
        ("s1", "2025-01-05", np.nan),   # unplayed fixture row
        ("s1", "2025-01-10", 4.0),
    ])
    out = asof_mean_before_date(df, ["season"], "g")
    assert out.iloc[2] == 2.0  # the NaN row neither adds value nor count


def test_empty_group_cols_means_global():
    df = _df([
        ("s1", "2025-01-01", 1.0),
        ("s2", "2025-01-08", 3.0),
    ])
    out = asof_mean_before_date(df, [], "g")
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 1.0


def test_matches_old_idiom_on_date_sorted_unique_dates():
    """On strictly increasing unique dates the old shift(1)+expanding and the
    new helper must agree — the change is only about ties and order."""
    df = _df([("s1", f"2025-02-{d:02d}", float(d)) for d in range(1, 8)])
    old = df["g"].expanding().mean().shift(1)
    new = asof_mean_before_date(df, ["season"], "g")
    pd.testing.assert_series_equal(
        old.reset_index(drop=True), new.reset_index(drop=True),
        check_names=False)
