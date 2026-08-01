"""Does the pre-season friendly signal actually improve XI prediction?

WHY THIS EXISTS
---------------
``lineup_predictor`` gained four constants when the friendly signal was wired
in -- ``PRESEASON_ONLY_PRIOR``, ``PRESEASON_PRIOR_SHRINK``,
``PRESEASON_FADE_MATCHES``, ``PRESEASON_ABSENT_PENALTY``.  All four were
JUDGEMENT, not measurement.  They survived review because the output looked
plausible (Vlahovic dropping 87 -> 47 *feels* right), and a plausible-looking
output is a hypothesis, not a verification.

This replays past matchweeks and measures whether the signal helps.

THE REPLAY (the part that must not leak)
----------------------------------------
For matchweek ``k`` of season ``S``, the predictor in production would have
seen exactly what ``_load_current_season_stats`` returns at that moment:

* ``k == 1`` -- no season-``S`` rows exist yet, so ``season.max()`` is ``S-1``
  and the predictor reasons off LAST season's completed table.  This is the
  case the whole signal exists for.
* ``k > 1``  -- ``season.max()`` is ``S``, so the predictor sees only rounds
  ``1..k-1``: a handful of matches.

and friendlies from season ``S`` only.  ``load_preseason_signal(team, season=S)``
is passed explicitly -- defaulting to the newest pre-season on disk would hand a
2024-25 replay its 2026-27 friendlies, which is look-ahead leakage that would
manufacture a good result.

THREE ARMS
----------
``naive``  top 11 by raw start COUNT in the replayed table.  No shrinkage, no
           form, no signal.  The floor.  If ``off`` does not beat this, the
           model machinery is doing nothing and no delta from the signal means
           anything.
``off``    the real predictor, ``preseason=None``.
``on``     the real predictor with the signal.

METRIC
------
XI hit-rate: of the 11 predicted, how many actually started.  Reported with
``n_changed`` -- the number of player slots where ``on`` and ``off`` disagree.
**If ``n_changed`` is tiny the accuracy delta is noise regardless of its sign**,
and the honest output is "N too small to calibrate", not a tuned constant.

Run:
    python3 -m scripts.analysis.backtest_preseason_signal
    python3 -m scripts.analysis.backtest_preseason_signal --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.prediction import lineup_predictor as lp  # noqa: E402

XI = 11
DEFAULT_ROUNDS = (1, 2, 3, 4, 5)
OUT_PATH = _PROJECT_ROOT / "data" / "analysis" / "preseason_signal_backtest.json"


def _prev_season(season: str) -> str:
    a, b = season.split("-")
    return f"{int(a) - 1}-{int(b) - 1}"


def _season_opener(stats: pd.DataFrame, team: str, season: str):
    """Date of the club's first league match that season -- the cutoff beyond
    which a 'pre-season' friendly is no longer pre-season (and, for matchweek 1,
    is outright future data).  None when unknown, which leaves the signal
    unfiltered rather than silently emptying it."""
    t = stats[(stats["team"] == team) & (stats["season"] == season)]
    if t.empty or "date" not in t.columns:
        return None
    first = pd.to_datetime(t["date"], errors="coerce").min()
    return None if pd.isna(first) else first


def load_league_stats() -> pd.DataFrame:
    """All seasons, both leagues.  The replay slices this itself."""
    frames = []
    for name in lp._PLAYER_STATS_FILES:
        p = lp.SOFASCORE_DIR / name
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ("minutes", "is_starter", "round"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date")


def friendly_seasons() -> list[str]:
    files = sorted(lp.SOFASCORE_DIR.glob("friendlies_*.parquet"))
    seasons: set[str] = set()
    for p in files:
        try:
            d = pd.read_parquet(p, columns=["season", "is_our_club"])
        except (OSError, ValueError, KeyError):
            continue
        seasons.update(d[d["is_our_club"]]["season"].dropna().unique().tolist())
    return sorted(seasons)


def _replay_table(stats: pd.DataFrame, team: str, season: str, rnd: int) -> pd.DataFrame:
    """Exactly what _load_current_season_stats would have returned then."""
    t = stats[stats["team"] == team]
    if rnd <= 1:
        return t[t["season"] == _prev_season(season)]
    return t[(t["season"] == season) & (t["round"] < rnd)]


def _actual_starters(stats: pd.DataFrame, team: str, season: str, rnd: int) -> set[int]:
    m = stats[(stats["team"] == team) & (stats["season"] == season)
              & (stats["round"] == rnd) & (stats["is_starter"] == True)]  # noqa: E712
    return set(pd.to_numeric(m["player_id"], errors="coerce").dropna().astype(int))


def _naive_top11(table: pd.DataFrame) -> list[int]:
    if table.empty:
        return []
    starts = (table[table["is_starter"] == True]  # noqa: E712
              .groupby("player_id").size().sort_values(ascending=False))
    return [int(p) for p in starts.head(XI).index]


def _model_top11(table: pd.DataFrame, team: str, preseason: dict | None) -> list[int]:
    freq = lp.get_starter_frequency(table, team, n_matches=10, preseason=preseason)
    return [int(p["player_id"]) for p in freq[:XI]]


def run_backtest(seasons: list[str] | None = None,
                 rounds: tuple[int, ...] = DEFAULT_ROUNDS,
                 verbose: bool = True,
                 write: bool = True) -> dict[str, Any]:
    """Replay the XI predictor with the signal on and off.

    write: persist the fixture rows to OUT_PATH.  `sweep()` passes False --
        it calls this once per grid cell, and every cell writing the canonical
        artefact leaves the file holding an ARBITRARY cell's fixtures, computed
        with SWEPT constants over only the calibration seasons.  Anything that
        then reads the JSON believing it holds the production-constants run
        gets a confidently wrong answer (this happened 2026-08-01).
    """
    stats = load_league_stats()
    if stats.empty:
        return {"error": "no player-stats parquet"}

    fs = friendly_seasons()
    have_league = set(stats["season"].dropna().unique())
    # Only seasons where we have BOTH the friendlies and the league result, AND
    # the previous season's table the k==1 replay needs.
    seasons = seasons or [s for s in fs
                          if s in have_league and _prev_season(s) in have_league]
    if not seasons:
        return {"error": "no season has friendlies + league results + a prior season",
                "friendly_seasons": fs, "league_seasons": sorted(have_league)}

    fixtures: list[dict[str, Any]] = []
    for season in seasons:
        clubs = sorted(stats[stats["season"] == season]["team"].dropna().unique())
        for team in clubs:
            # Season-scoping alone leaks: `_season_for` stamps ANY June-onward
            # friendly with the season starting that August, so a March friendly
            # carries the previous season's label and would reach a matchweek-1
            # replay from seven months in its future.  Cut at the club's first
            # league match -- which is also what "pre-season" literally means.
            pre = lp.load_preseason_signal(
                team, season=season, before=_season_opener(stats, team, season))
            for rnd in rounds:
                actual = _actual_starters(stats, team, season, rnd)
                if len(actual) < XI:
                    continue  # match not played / not scraped
                table = _replay_table(stats, team, season, rnd)
                arms = {
                    "naive": _naive_top11(table),
                    "off": _model_top11(table, team, None),
                    "on": _model_top11(table, team, pre or None),
                }
                row: dict[str, Any] = {
                    "season": season, "team": team, "round": rnd,
                    "club_friendlies": (pre or {}).get("club_friendlies", 0),
                    "replay_rows": int(len(table)),
                    "promoted": bool(table.empty),
                    "n_changed": len(set(arms["on"]) ^ set(arms["off"])) // 2,
                }
                for arm, picks in arms.items():
                    row[f"hit_{arm}"] = len(set(picks) & actual) / XI if picks else 0.0
                    row[f"n_{arm}"] = len(picks)
                fixtures.append(row)

    if not fixtures:
        return {"error": "no gradable fixtures", "seasons_tried": seasons}

    fx = pd.DataFrame(fixtures)
    summary = _summarise(fx)
    if verbose:
        _report(fx, summary)
    if write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(
            {"summary": summary, "fixtures": fixtures}, indent=2, default=str))
    return {"summary": summary, "fixtures": fx}


def _summarise(fx: pd.DataFrame) -> dict[str, Any]:
    changed = fx[fx["n_changed"] > 0]
    out = {
        "n_fixtures": int(len(fx)),
        "seasons": sorted(fx["season"].unique().tolist()),
        "hit_naive": round(float(fx["hit_naive"].mean()), 4),
        "hit_off": round(float(fx["hit_off"].mean()), 4),
        "hit_on": round(float(fx["hit_on"].mean()), 4),
        "delta_on_minus_off": round(float((fx["hit_on"] - fx["hit_off"]).mean()), 4),
        "model_lift_over_naive": round(float((fx["hit_off"] - fx["hit_naive"]).mean()), 4),
        "fixtures_where_signal_changed_xi": int(len(changed)),
        "player_slots_changed": int(fx["n_changed"].sum()),
        "promoted_fixtures": int(fx["promoted"].sum()),
        "by_round": {},
    }
    for rnd, grp in fx.groupby("round"):
        out["by_round"][int(rnd)] = {
            "n": int(len(grp)),
            "naive": round(float(grp["hit_naive"].mean()), 4),
            "off": round(float(grp["hit_off"].mean()), 4),
            "on": round(float(grp["hit_on"].mean()), 4),
            "delta": round(float((grp["hit_on"] - grp["hit_off"]).mean()), 4),
            "slots_changed": int(grp["n_changed"].sum()),
        }
    if len(changed):
        out["on_changed_fixtures_only"] = {
            "n": int(len(changed)),
            "off": round(float(changed["hit_off"].mean()), 4),
            "on": round(float(changed["hit_on"].mean()), 4),
            "delta": round(float((changed["hit_on"] - changed["hit_off"]).mean()), 4),
        }
    return out


def _report(fx: pd.DataFrame, s: dict[str, Any]) -> None:
    print("\n" + "=" * 74)
    print("PRE-SEASON FRIENDLY SIGNAL — XI BACKTEST")
    print("=" * 74)
    print(f"seasons {s['seasons']}   fixtures {s['n_fixtures']}   "
          f"promoted (no prior table) {s['promoted_fixtures']}")
    print(f"\n  naive (raw start count)   {s['hit_naive']:.1%}")
    print(f"  off   (model, no signal)  {s['hit_off']:.1%}   "
          f"lift over naive {s['model_lift_over_naive']:+.1%}")
    print(f"  on    (model + signal)    {s['hit_on']:.1%}   "
          f"delta {s['delta_on_minus_off']:+.1%}")
    print(f"\n  fixtures where the signal changed the XI: "
          f"{s['fixtures_where_signal_changed_xi']}/{s['n_fixtures']}"
          f"   player slots changed: {s['player_slots_changed']}")
    if s["player_slots_changed"] < 20:
        print("  !! too few changes to calibrate anything — treat any delta as noise")
    if s["model_lift_over_naive"] <= 0:
        print("  !! the model does NOT beat a raw start count — the signal delta is "
              "measured on top of a baseline that is already not working")
    print("\n  by matchweek (this is what PRESEASON_FADE_MATCHES should be tuned on):")
    print(f"    {'MW':<4}{'n':<6}{'naive':<9}{'off':<9}{'on':<9}{'delta':<9}slots")
    for rnd, r in sorted(s["by_round"].items()):
        print(f"    {rnd:<4}{r['n']:<6}{r['naive']:<9.1%}{r['off']:<9.1%}"
              f"{r['on']:<9.1%}{r['delta']:<+9.1%}{r['slots_changed']}")
    if "on_changed_fixtures_only" in s:
        c = s["on_changed_fixtures_only"]
        print(f"\n  restricted to the {c['n']} fixtures the signal actually moved:")
        print(f"    off {c['off']:.1%}  ->  on {c['on']:.1%}   ({c['delta']:+.1%})")
    print(f"\n  written to {OUT_PATH.relative_to(_PROJECT_ROOT)}")
    print("=" * 74)


def sweep(seasons: list[str] | None = None) -> dict[str, Any]:
    """Grid over the invented constants.

    Calibrate on the EARLIEST season, validate on the latest.  Reporting the
    best cell of a grid fitted on all the data would just be overfitting with
    extra steps.
    """
    stats = load_league_stats()
    fs = [s for s in friendly_seasons()
          if s in set(stats["season"].unique())
          and _prev_season(s) in set(stats["season"].unique())]
    seasons = seasons or fs
    if len(seasons) < 2:
        return {"error": "need >=2 replayable seasons to hold one out",
                "replayable": seasons}
    calib, holdout = seasons[:-1], seasons[-1:]

    grid = [(prior, penalty, fade)
            for prior in (30.0, 40.0, 50.0)
            for penalty in (-5.0, -15.0, -25.0)
            for fade in (3.0, 5.0, 8.0)]
    orig = (lp.PRESEASON_ONLY_PRIOR, lp.PRESEASON_ABSENT_PENALTY,
            lp.PRESEASON_FADE_MATCHES)
    results = []
    try:
        for prior, penalty, fade in grid:
            lp.PRESEASON_ONLY_PRIOR = prior
            lp.PRESEASON_ABSENT_PENALTY = penalty
            lp.PRESEASON_FADE_MATCHES = fade
            r = run_backtest(calib, verbose=False, write=False)
            if "summary" not in r:
                continue
            results.append({"prior": prior, "penalty": penalty, "fade": fade,
                            **{k: r["summary"][k] for k in
                               ("hit_on", "hit_off", "delta_on_minus_off",
                                "player_slots_changed")}})
        results.sort(key=lambda d: -d["hit_on"])
        best = results[0] if results else None
        val = None
        if best:
            lp.PRESEASON_ONLY_PRIOR = best["prior"]
            lp.PRESEASON_ABSENT_PENALTY = best["penalty"]
            lp.PRESEASON_FADE_MATCHES = best["fade"]
            v = run_backtest(holdout, verbose=False, write=False)
            val = v.get("summary")
    finally:
        (lp.PRESEASON_ONLY_PRIOR, lp.PRESEASON_ABSENT_PENALTY,
         lp.PRESEASON_FADE_MATCHES) = orig

    print("\n" + "=" * 74)
    print(f"CONSTANT SWEEP — calibrate {calib}, hold out {holdout}")
    print("=" * 74)
    print(f"  {'prior':<8}{'penalty':<10}{'fade':<8}{'on':<9}{'off':<9}{'delta':<9}slots")
    for r in results[:10]:
        print(f"  {r['prior']:<8.0f}{r['penalty']:<10.0f}{r['fade']:<8.0f}"
              f"{r['hit_on']:<9.1%}{r['hit_off']:<9.1%}"
              f"{r['delta_on_minus_off']:<+9.1%}{r['player_slots_changed']}")
    if val:
        print(f"\n  HOLDOUT {holdout} with the best cell "
              f"(prior={best['prior']:.0f}, penalty={best['penalty']:.0f}, "
              f"fade={best['fade']:.0f}):")
        print(f"    off {val['hit_off']:.1%}  ->  on {val['hit_on']:.1%}   "
              f"({val['delta_on_minus_off']:+.1%}), "
              f"{val['player_slots_changed']} slots changed")
        print("    A cell that wins in calibration and loses here is NOISE. "
              "Ship nothing unless the holdout agrees.")
    print("=" * 74)
    return {"calibration": results, "holdout": val, "best": best}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", default=None,
                    help="comma-separated, e.g. 2024-2025,2025-2026")
    ap.add_argument("--rounds", default="1,2,3,4,5")
    ap.add_argument("--sweep", action="store_true",
                    help="grid over the invented constants, with a holdout season")
    args = ap.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",")] if args.seasons else None
    rounds = tuple(int(r) for r in args.rounds.split(","))

    if args.sweep:
        sweep(seasons)
        return 0
    res = run_backtest(seasons, rounds)
    if "error" in res:
        print(f"cannot backtest: {res}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
