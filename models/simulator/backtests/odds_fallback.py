"""Odds fallback chain: Pinnacle closing → market avg → B365 → opening.

Never silently drop a match for missing odds. If ALL sources are missing we
raise `NoOddsAvailable`; the harness decides what to do with it (log & skip
that specific bet, but keep the match in the prediction count).

Every bet returned by `resolve_odds_binary` / `resolve_odds_multiclass` is
tagged with a `source` label so the final report can slice ROI per source —
this is how we detect selection bias (if ROI collapses on non-Pinnacle rows,
the predictor only works where sharp pricing is available).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class NoOddsAvailable(Exception):
    pass


@dataclass(frozen=True)
class BinaryOdds:
    yes: float
    no: float
    source: str  # "pinnacle_close" | "market_avg" | "b365_close" | "b365" | "opening"


@dataclass(frozen=True)
class MulticlassOdds:
    odds: tuple[float, ...]       # len == len(classes)
    classes: tuple[str, ...]
    source: str


# Priority order — higher index = lower-quality source.
_SOURCE_LABELS = ["pinnacle_close", "b365_close", "market_avg", "b365", "opening"]


def resolve_odds_binary(
    row: pd.Series,
    chain: list[tuple[str, str]],
) -> BinaryOdds:
    """Try each (col_yes, col_no) tuple in order; return first valid pair.

    Valid means: both cols exist, both non-NaN, both > 1.0.
    """
    for idx, (col_yes, col_no) in enumerate(chain):
        if col_yes in row.index and col_no in row.index:
            v_yes = row[col_yes]
            v_no = row[col_no]
            if pd.notna(v_yes) and pd.notna(v_no) and v_yes > 1.0 and v_no > 1.0:
                source = _SOURCE_LABELS[idx] if idx < len(_SOURCE_LABELS) else f"fallback_{idx}"
                return BinaryOdds(yes=float(v_yes), no=float(v_no), source=source)
    raise NoOddsAvailable(f"No valid binary odds in chain {chain}")


def resolve_odds_multiclass(
    row: pd.Series,
    chain: list[tuple[str, ...]],
    classes: tuple[str, ...],
) -> MulticlassOdds:
    """Try each tuple of N columns in order; return first valid tuple.

    `classes` is the label ordering expected by the caller (e.g. ("H","D","A")).
    Chain tuples must be aligned to that same order.
    """
    for idx, col_tuple in enumerate(chain):
        if len(col_tuple) != len(classes):
            continue
        if all(c in row.index for c in col_tuple):
            vals = [row[c] for c in col_tuple]
            if all(pd.notna(x) and x > 1.0 for x in vals):
                source = _SOURCE_LABELS[idx] if idx < len(_SOURCE_LABELS) else f"fallback_{idx}"
                return MulticlassOdds(
                    odds=tuple(float(x) for x in vals),
                    classes=classes,
                    source=source,
                )
    raise NoOddsAvailable(f"No valid multiclass odds in chain for classes {classes}")


def deoverround_binary(odds: BinaryOdds) -> tuple[float, float]:
    inv_y = 1.0 / odds.yes
    inv_n = 1.0 / odds.no
    s = inv_y + inv_n
    return inv_y / s, inv_n / s


def deoverround_multiclass(odds: MulticlassOdds) -> tuple[float, ...]:
    inv = [1.0 / x for x in odds.odds]
    s = sum(inv)
    return tuple(i / s for i in inv)
