"""Pre-match XI quality features (Phase 3a — 2026-04-24).

Builds features describing the announced starting XI's quality based ONLY
on each starter's performance in matches prior to the current one.

Leakage spec (MANDATORY):
    - For match M on date D: each player's rolling window is computed ONLY
      from rows where match_date < D. The current match M is never
      included in the feature computation for M itself.
    - `is_starter=True` in current-match row is used only for IDENTITY
      (who the 11 starters are). Player ID is known ~1 hour before kickoff
      when lineups publish — this is pre-match information.
    - If a starter has fewer than MIN_PRIOR_MATCHES matches of history,
      we fall back to NaN for that player's contribution (do not impute 0).
      CatBoost handles NaN natively.

Feature outputs per match (6 columns):
    home_xi_avg_rating_last5       — mean of starters' last-5-match Sofascore rating
    away_xi_avg_rating_last5
    home_xi_xg_per90_sum           — sum of starters' last-10-match xG/90
    away_xi_xg_per90_sum
    home_xi_minutes_continuity     — % of starters who also started last team match
    away_xi_minutes_continuity

Routing: accepts `league` to pick player_match_stats.parquet (SA) vs
player_match_stats_premier_league.parquet (EPL). Same pattern as lineup_xg.py.

Version history:
    1.0 — 2026-04-24 initial Phase 3a.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

MIN_PRIOR_MATCHES_PER_PLAYER = 3   # fewer prior matches → NaN for that player's stat
RATING_WINDOW = 5                   # last-5-match rating average
XG_WINDOW = 10                      # last-10-match xG/90 (longer, xg/90 is noisier)


def _load_player_match_stats(league: str | None) -> pd.DataFrame | None:
    base = DATA_DIR / "external" / "sofascore"
    if league == "premier_league":
        path = base / "player_match_stats_premier_league.parquet"
    else:
        path = base / "player_match_stats.parquet"
    if not path.exists():
        log.warning("xi_quality: no Sofascore player data at %s", path)
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # Clip rating to sane bounds (Sofascore is 0-10 but sometimes has outliers)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(0.0, 10.0)
    df["xg"] = pd.to_numeric(df["xg"], errors="coerce").clip(0.0, 3.0)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").clip(0.0, 120.0)
    log.info("xi_quality: loaded %d player-match rows for league=%s",
             len(df), league or "default")
    return df


def _compute_player_history(stats: pd.DataFrame) -> dict:
    """Pre-index per-player chronological history.

    Returns a dict mapping player_id → sorted list of (date, rating, xg, minutes,
    is_starter) tuples. Used for fast "give me everything before date D" lookups.
    """
    by_player: dict[int, list] = {}
    stats_sorted = stats.sort_values(["player_id", "date"])
    for pid, grp in stats_sorted.groupby("player_id", sort=False):
        recs = list(zip(
            grp["date"].tolist(),
            grp["rating"].tolist(),
            grp["xg"].tolist(),
            grp["minutes"].tolist(),
            grp["is_starter"].tolist(),
        ))
        by_player[int(pid)] = recs
    return by_player


def _starter_prior_stats(pid: int, as_of: any, history: dict,
                        n_window: int, field_idx: int) -> float | None:
    """Return mean of `field_idx`-th column across the player's last n matches
    strictly before `as_of`. Returns None if < MIN_PRIOR_MATCHES_PER_PLAYER.
    """
    recs = history.get(pid, [])
    if not recs:
        return None
    # recs is sorted by date ascending; find all strictly before as_of
    prior = [r for r in recs if r[0] < as_of]
    if len(prior) < MIN_PRIOR_MATCHES_PER_PLAYER:
        return None
    # Last n of those
    window = prior[-n_window:] if len(prior) >= n_window else prior
    vals = [r[field_idx] for r in window if pd.notna(r[field_idx])]
    if not vals:
        return None
    return float(np.mean(vals))


def _xi_features_one_match(starters_pids: list,
                           match_date: any,
                           history: dict) -> tuple[float | None, float | None]:
    """Return (avg_rating_last5, xg_per90_sum) for a given XI.

    `avg_rating_last5`: mean over players whose rating history is sufficient.
    `xg_per90_sum`: sum over players whose xg history is sufficient, scaled
    to xg per 90 using the player's own mean minutes across the window.
    """
    ratings = []
    xg_per90s = []
    for pid in starters_pids:
        r = _starter_prior_stats(pid, match_date, history, RATING_WINDOW, field_idx=1)
        if r is not None:
            ratings.append(r)
        # xG/90: use last 10-match xg + minutes, compute (sum_xg / sum_min) * 90
        recs = history.get(pid, [])
        prior = [rec for rec in recs if rec[0] < match_date]
        if len(prior) < MIN_PRIOR_MATCHES_PER_PLAYER:
            continue
        window = prior[-XG_WINDOW:]
        sum_xg = sum((rec[2] or 0.0) for rec in window if pd.notna(rec[2]))
        sum_min = sum((rec[3] or 0.0) for rec in window if pd.notna(rec[3]))
        if sum_min > 90 * MIN_PRIOR_MATCHES_PER_PLAYER:
            xg_per90s.append(sum_xg / sum_min * 90.0)
    avg_rating = float(np.mean(ratings)) if ratings else None
    sum_xg90 = float(np.sum(xg_per90s)) if xg_per90s else None
    return avg_rating, sum_xg90


def add_xi_quality_features(feature_df: pd.DataFrame,
                            league: str | None = None) -> pd.DataFrame:
    """Emit the 6 pre-match XI quality columns into `feature_df`.

    `feature_df` is the match-level table. We join on (date, team) using
    the Sofascore player-match rows; the current-match row contributes
    *only* the identity of starters (pre-match info), not any stats.
    """
    df = feature_df.copy()
    cols_before = len(df.columns)

    stats = _load_player_match_stats(league)
    if stats is None or len(stats) == 0:
        log.info("xi_quality: no data, skipping")
        return df

    # Need join key in feature_df
    if "match_date" not in df.columns or "home_team" not in df.columns:
        log.warning("xi_quality: feature_df missing match_date/home_team, skipping")
        return df

    # Pre-index player history for fast prior-match lookups
    history = _compute_player_history(stats)

    # Pre-index starters per (date, team): the identity of the XI
    starters = stats[stats["is_starter"] == True][  # noqa: E712
        ["date", "team", "player_id", "minutes"]
    ].copy()
    starter_map: dict[tuple, list] = {}
    minutes_map: dict[tuple, dict] = {}
    for (d, t), grp in starters.groupby(["date", "team"], sort=False):
        starter_map[(d, t)] = grp["player_id"].astype(int).tolist()
        minutes_map[(d, t)] = dict(zip(grp["player_id"].astype(int),
                                       grp["minutes"]))

    # Prior-match starters per team for continuity: list of (date, starters_set)
    # sorted by date ascending, used to find "last match prior to D".
    team_match_starters: dict[str, list] = {}
    for team, grp in starters.sort_values("date").groupby("team", sort=False):
        per_team = []
        for d, sub in grp.groupby("date", sort=True):
            per_team.append((d, set(sub["player_id"].astype(int).tolist())))
        team_match_starters[team] = per_team

    def _continuity(team: str, match_date: any, current_starters: list) -> float | None:
        hist = team_match_starters.get(team, [])
        # Most recent prior (date, starters_set) strictly before match_date
        prev = None
        for d, s in reversed(hist):
            if d < match_date:
                prev = s
                break
        if prev is None:
            return None
        cur = set(int(p) for p in current_starters)
        if not cur:
            return None
        return len(cur & prev) / len(cur)

    # Prepare output arrays
    n = len(df)
    out = {
        "home_xi_avg_rating_last5": [np.nan] * n,
        "away_xi_avg_rating_last5": [np.nan] * n,
        "home_xi_xg_per90_sum": [np.nan] * n,
        "away_xi_xg_per90_sum": [np.nan] * n,
        "home_xi_minutes_continuity": [np.nan] * n,
        "away_xi_minutes_continuity": [np.nan] * n,
    }

    dates = pd.to_datetime(df["match_date"], errors="coerce").dt.date.tolist()
    home_teams = df["home_team"].tolist()
    away_teams = df["away_team"].tolist()

    hits = 0
    for i in range(n):
        d = dates[i]
        ht = home_teams[i]
        at = away_teams[i]
        if d is None or pd.isna(d):
            continue
        home_pids = starter_map.get((d, ht))
        away_pids = starter_map.get((d, at))
        if home_pids:
            r, x = _xi_features_one_match(home_pids, d, history)
            out["home_xi_avg_rating_last5"][i] = r if r is not None else np.nan
            out["home_xi_xg_per90_sum"][i] = x if x is not None else np.nan
            out["home_xi_minutes_continuity"][i] = (
                _continuity(ht, d, home_pids) or np.nan
            )
            hits += 1
        if away_pids:
            r, x = _xi_features_one_match(away_pids, d, history)
            out["away_xi_avg_rating_last5"][i] = r if r is not None else np.nan
            out["away_xi_xg_per90_sum"][i] = x if x is not None else np.nan
            out["away_xi_minutes_continuity"][i] = (
                _continuity(at, d, away_pids) or np.nan
            )

    for k, v in out.items():
        df[k] = v

    new_cols = len(df.columns) - cols_before
    log.info("xi_quality: added %d columns, matched %d of %d match-team rows",
             new_cols, hits, n)
    return df
