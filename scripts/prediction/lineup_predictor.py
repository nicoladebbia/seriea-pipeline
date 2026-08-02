#!/usr/bin/env python3
"""Lineup and formation prediction engine.

Predicts the most likely starting XI and formation for each team in upcoming
matches. Uses Sofascore historical match data (player_match_stats.parquet,
match JSONs) to determine:

1. Most likely formation (weighted by recency)
2. Predicted starting XI based on start frequency & minutes
3. Suspension risk (yellow card accumulation — 5 yellows = 1-match ban in Serie A)
4. Injury/absence info from latest match JSON missing_players
5. Replacement logic when a regular starter is unavailable

Output: data/upcoming/lineup_predictions.json
"""

import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import DATA_DIR
from config.team_names import normalize_team

# Serie A suspension thresholds
YELLOW_SUSPENSION_THRESHOLD = 5  # 5 yellows = 1-match ban
YELLOW_RESET_AFTER = None  # No mid-season reset in Serie A 2025-2026

SOFASCORE_DIR = DATA_DIR / ".." / "data" / "external" / "sofascore"
MATCH_JSON_DIR = SOFASCORE_DIR / "matches"

# Position mapping: Sofascore position codes → generic labels (fallback only)
POSITION_MAP = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}

# Formation → tactical position labels per slot group.
# Within each group, labels are ordered by display position (R→L for defense/midfield).
# Key = full formation string, Value = dict of position group → ordered labels.
FORMATION_TACTICAL_SLOTS = {
    # === 3-back ===
    "3-5-2":   {"G": ["GK"], "D": ["RCB", "CB", "LCB"], "M": ["RWB", "RCM", "CM", "LCM", "LWB"], "F": ["RS", "LS"]},
    "3-4-3":   {"G": ["GK"], "D": ["RCB", "CB", "LCB"], "M": ["RWB", "RCM", "LCM", "LWB"], "F": ["RW", "ST", "LW"]},
    "3-4-2-1": {"G": ["GK"], "D": ["RCB", "CB", "LCB"], "M": ["RWB", "RCM", "LCM", "LWB", "RAM", "LAM"], "F": ["ST"]},
    "3-4-1-2": {"G": ["GK"], "D": ["RCB", "CB", "LCB"], "M": ["RWB", "CM", "CM", "LWB", "CAM"], "F": ["RS", "LS"]},
    "3-1-4-2": {"G": ["GK"], "D": ["RCB", "CB", "LCB"], "M": ["CDM", "RM", "RCM", "LCM", "LM"], "F": ["RS", "LS"]},
    "3-5-1-1": {"G": ["GK"], "D": ["RCB", "CB", "LCB"], "M": ["RWB", "RCM", "CM", "LCM", "LWB", "SS"], "F": ["ST"]},
    # === 4-back ===
    "4-3-3":   {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["RCM", "CM", "LCM"], "F": ["RW", "ST", "LW"]},
    "4-4-2":   {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["RM", "RCM", "LCM", "LM"], "F": ["RS", "LS"]},
    "4-2-3-1": {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["RDM", "LDM", "RAM", "CAM", "LAM"], "F": ["ST"]},
    "4-3-1-2": {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["RCM", "CM", "LCM", "CAM"], "F": ["RS", "LS"]},
    "4-1-4-1": {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["CDM", "RM", "RCM", "LCM", "LM"], "F": ["ST"]},
    "4-5-1":   {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["RM", "RCM", "CM", "LCM", "LM"], "F": ["ST"]},
    "4-4-1-1": {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["RM", "RCM", "LCM", "LM", "SS"], "F": ["ST"]},
    "4-1-3-2": {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["CDM", "RCM", "CM", "LCM"], "F": ["RS", "LS"]},
    "4-3-2-1": {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["RCM", "CM", "LCM", "RAM", "LAM"], "F": ["ST"]},
    "4-2-4-0": {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["RDM", "LDM", "RW", "RAM", "LAM", "LW"], "F": []},
    "4-1-2-3": {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["CDM", "RCM", "LCM"], "F": ["RW", "ST", "LW"]},
    "4-2-2-2": {"G": ["GK"], "D": ["RB", "RCB", "LCB", "LB"], "M": ["RDM", "LDM", "RAM", "LAM"], "F": ["RS", "LS"]},
    # === 5-back ===
    "5-3-2":   {"G": ["GK"], "D": ["RWB", "RCB", "CB", "LCB", "LWB"], "M": ["RCM", "CM", "LCM"], "F": ["RS", "LS"]},
    "5-4-1":   {"G": ["GK"], "D": ["RWB", "RCB", "CB", "LCB", "LWB"], "M": ["RM", "RCM", "LCM", "LM"], "F": ["ST"]},
    "5-2-3":   {"G": ["GK"], "D": ["RWB", "RCB", "CB", "LCB", "LWB"], "M": ["RCM", "LCM"], "F": ["RW", "ST", "LW"]},
}


def _get_tactical_labels(formation: str, slots: dict) -> dict:
    """Map formation string + slot counts to tactical position labels.

    Uses FORMATION_TACTICAL_SLOTS for known formations. Falls back to
    generic positional labels derived from slot counts for unknown ones.
    """
    known = FORMATION_TACTICAL_SLOTS.get(formation)
    if known:
        return known

    labels = {"G": ["GK"]}

    n_def = slots.get("D", 4)
    if n_def == 2:
        labels["D"] = ["RCB", "LCB"]
    elif n_def == 3:
        labels["D"] = ["RCB", "CB", "LCB"]
    elif n_def == 4:
        labels["D"] = ["RB", "RCB", "LCB", "LB"]
    elif n_def == 5:
        labels["D"] = ["RWB", "RCB", "CB", "LCB", "LWB"]
    else:
        labels["D"] = [f"DEF{i+1}" for i in range(n_def)]

    n_mid = slots.get("M", 3)
    if n_mid == 1:
        labels["M"] = ["CM"]
    elif n_mid == 2:
        labels["M"] = ["RCM", "LCM"]
    elif n_mid == 3:
        labels["M"] = ["RCM", "CM", "LCM"]
    elif n_mid == 4:
        labels["M"] = ["RM", "RCM", "LCM", "LM"]
    elif n_mid == 5:
        labels["M"] = ["RM", "RCM", "CM", "LCM", "LM"]
    elif n_mid == 6:
        labels["M"] = ["RDM", "LDM", "RM", "RCM", "LCM", "LM"]
    else:
        labels["M"] = [f"MID{i+1}" for i in range(n_mid)]

    n_fwd = slots.get("F", 3)
    if n_fwd == 0:
        labels["F"] = []
    elif n_fwd == 1:
        labels["F"] = ["ST"]
    elif n_fwd == 2:
        labels["F"] = ["RS", "LS"]
    elif n_fwd == 3:
        labels["F"] = ["RW", "ST", "LW"]
    else:
        labels["F"] = [f"FWD{i+1}" for i in range(n_fwd)]

    return labels


#: Both league variants of the Sofascore player-stats parquet.  Reading only the
#: first is failure mode #1 in the project's "EPL data missing where SA has it"
#: catalogue: 18 Premier League clubs had pre-season friendly data and no XI
#: prediction path at all, purely because this loader opened one file.
#: Safe to concatenate: the two files share ZERO team names (verified
#: 2026-08-01, 32 SA vs 33 EPL clubs, empty intersection), and every lookup
#: downstream keys on `team`.  `player_id` DOES overlap (229 players appear in
#: both, correctly -- it is a global Sofascore id), which is why nothing may key
#: on player_id alone across leagues.
_PLAYER_STATS_FILES = ("player_match_stats.parquet",
                       "player_match_stats_premier_league.parquet")


def _load_current_season_stats() -> pd.DataFrame:
    """Load player match stats for the current season, ALL tracked leagues.

    Each league is filtered to its OWN latest season before concatenating --
    a global `season.max()` would silently drop a league that is a season
    behind (mid-transition, or a source that lags).
    """
    frames = []
    for name in _PLAYER_STATS_FILES:
        path = SOFASCORE_DIR / name
        if not path.exists():
            continue
        part = pd.read_parquet(path)
        if part.empty:
            continue
        if "season" in part.columns:
            part = part[part["season"] == part["season"].max()]
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Ensure proper types
    for col in ["minutes", "is_starter", "round"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date")


def _friendlies_frame(season: str | None = None) -> pd.DataFrame:
    """Every scraped club friendly for ONE pre-season, our clubs only.

    Shared by `load_preseason_signal` and `preseason_club_names` so the two can
    never disagree about which pre-season is current or which rows count as
    ours -- a club routed in off one answer and handed nothing by the other
    would be worse than not routing it at all.

    Returns an empty frame (never raises) when there is no data: a fresh
    checkout, a scraper that has not run yet, and a corrupt parquet must all
    degrade to "no signal", which every caller already handles.
    """
    files = sorted(SOFASCORE_DIR.glob("friendlies_*.parquet"))
    if not files:
        return pd.DataFrame()
    try:
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    except (OSError, ValueError):
        return pd.DataFrame()
    if df.empty or "is_our_club" not in df.columns:
        return pd.DataFrame()
    df = df[df["season"] == (season or df["season"].max())]
    return df[df["is_our_club"]]


def preseason_club_names(season: str | None = None) -> set[str]:
    """Clubs with at least one scraped friendly in that pre-season.

    Used to route a club that has no league history at all -- see
    `_resolve_team_name`.
    """
    df = _friendlies_frame(season)
    if df.empty or "club" not in df.columns:
        return set()
    return {str(c) for c in df["club"].dropna().unique()}


def _resolve_team_name(team: str, league_teams: "set[str] | Iterable[str]",
                       preseason_clubs: "set[str] | Iterable[str] | None" = None,
                       ) -> tuple[str | None, str | None]:
    """Map a prediction's club name onto the name our data files use.

    Returns `(resolved_name, source)` where source is "league", "preseason" or
    None.  The ladder, strictest first: exact league name, normalised league
    name, substring league name (last resort), then -- only when the club has
    no league presence whatsoever -- exact or normalised PRE-SEASON name.

    The pre-season rung exists because `_load_current_season_stats` filters to
    `season.max()`, which at matchweek 1 is still LAST season: a newly promoted
    club has zero rows there, resolved to None on every league rung, and was
    dropped with a warning -- no XI at all.  Measured 2026-08-02 against the
    live files, all six promoted clubs across both leagues were being dropped
    while holding 25-29 friendly players and a real formation on disk, and
    `predict_team_lineup` returns a full 11-man XI when handed one of them.
    Those clubs are exactly what the pre-season signal was built for: with no
    league history, friendlies are the ONLY evidence of who is at the club.

    Substring matching is deliberately NOT offered on the pre-season rung.  On
    the league rung a bad match degrades an XI that would otherwise be built
    from real data; here it would invent an entire lineup out of another
    squad's players, and every club plays friendlies, so the map it searches
    contains the whole league.
    """
    league_teams = set(league_teams)
    if team in league_teams:
        return team, "league"
    norm_to_raw = {normalize_team(t): t for t in league_teams}
    if normalize_team(team) in norm_to_raw:
        return norm_to_raw[normalize_team(team)], "league"
    for t in league_teams:
        if team.lower() in t.lower() or t.lower() in team.lower():
            return t, "league"

    pre = set(preseason_clubs or ())
    if team in pre:
        return team, "preseason"
    pre_norm = {normalize_team(c): c for c in pre}
    if normalize_team(team) in pre_norm:
        return pre_norm[normalize_team(team)], "preseason"
    return None, None


def load_preseason_signal(team: str, season: str | None = None,
                          before: "str | pd.Timestamp | None" = None) -> dict:
    """Pre-season friendly participation for one club.

    Args:
        season: which pre-season to read.  Defaults to the newest on disk,
            which is what production wants.  The backtest passes an explicit
            past season to replay a historical matchweek -- without it, a
            2024-25 replay would be handed 2026-27 friendlies, which is
            look-ahead leakage and would fake a good result.
        before: drop friendlies played on/after this date.  Season-scoping
            ALONE is not leak-free: `sofascore_friendlies._season_for` files
            anything from June onward under the season that starts in August,
            so a friendly played in **March 2025** is stamped `2024-2025` and
            would reach a replay of 2024-25 matchweek 1 -- seven months of
            look-ahead.  Production leaves this None (there is no future to
            leak); the backtest passes the club's first league fixture of the
            season, which is also what "PRE-season" means literally.

    Why this exists: `_load_current_season_stats` filters to `season.max()`, so
    between May and the first league matchweek the predictor is reasoning off
    LAST season's completed table.  Measured 2026-08-01 across 17 Serie A clubs:
    194 players who featured in a pre-season friendly have ZERO rows in that
    table (summer arrivals, promoted youth) and therefore cannot appear in a
    predicted XI at all, while 72 players with >=5/10 starts last season
    appeared in no friendly whatsoever (departed, sold, long-term injured) and
    are still ranked as nailed-on starters.

    Friendlies are a weak signal about ABILITY -- managers rotate freely and the
    opposition is uneven -- but a strong one about AVAILABILITY, which is
    exactly what the XI predictor gets wrong at this point in the calendar.

    Returns {"players": {player_id: {...}}, "club_friendlies": int}.  An empty
    dict means "no signal", and every caller must degrade to prior behaviour.
    """
    df = _friendlies_frame(season)
    if df.empty:
        return {}
    df = df[df["club"] == team]
    if before is not None and "match_date" in df.columns:
        df = df[pd.to_datetime(df["match_date"], errors="coerce")
                < pd.to_datetime(before)]
    if df.empty:
        return {}

    club_friendlies = int(df["sofascore_event_id"].nunique())
    # Shapes trialled in pre-season, most recent first. For a promoted club this
    # is the only formation evidence that exists, and the alternative is a
    # hardcoded 4-3-3.
    formations = [
        str(f) for f in df.sort_values("match_date", ascending=False)
        .drop_duplicates("sofascore_event_id")["formation"].dropna().tolist()
    ] if "formation" in df.columns and "match_date" in df.columns else []
    players: dict[int, dict] = {}
    for pid, grp in df.groupby("player_id"):
        # `appearances` counts SQUAD PRESENCE, not minutes: an unused sub is
        # still evidence the player is at the club, which is the whole point of
        # the absent-from-pre-season signal below.
        appearances = int(len(grp))
        starts = int(grp["is_starter"].sum())
        players[int(pid)] = {
            "starts": starts,
            "appearances": appearances,
            "minutes": int(pd.to_numeric(grp["minutes_played"], errors="coerce").fillna(0).sum()),
            "start_pct": round(starts / appearances * 100, 1) if appearances else 0.0,
            "name": grp["player"].iloc[0],
            "position": grp["position"].mode().iloc[0] if len(grp["position"].mode()) else "?",
            "shirt_number": grp["shirt_number"].mode().iloc[0] if len(grp["shirt_number"].mode()) else None,
        }
    return {"players": players, "club_friendlies": club_friendlies,
            "season": str(df["season"].iloc[0]), "formations": formations}


def _load_season_incidents() -> pd.DataFrame:
    """Load all match incidents (cards, goals, subs) for the current season."""
    path = SOFASCORE_DIR / "match_incidents.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df


def _get_match_json(match_id: int) -> dict:
    """Load a single match JSON file."""
    season = "2025-2026"
    path = MATCH_JSON_DIR / season / f"{match_id}.json"
    if not path.exists():
        # Try other season dirs
        for d in MATCH_JSON_DIR.iterdir():
            p = d / f"{match_id}.json"
            if p.exists():
                path = p
                break
        else:
            return {}
    with open(path) as f:
        return json.load(f)


def _get_latest_match_ids(stats_df: pd.DataFrame, n: int = 1) -> dict:
    """Get the latest n match_ids for each team.

    Returns: {team: [match_id1, match_id2, ...]} ordered most recent first.
    When n=1, returns {team: match_id} for backward compatibility.
    """
    if stats_df.empty:
        return {}
    result = {}
    for team, grp in stats_df.groupby("team"):
        match_dates = grp.groupby("match_id")["date"].first().sort_values(ascending=False)
        ids = [int(mid) for mid in match_dates.head(n).index]
        result[team] = ids[0] if n == 1 else ids
    return result


def get_formation_patterns(stats_df: pd.DataFrame, team: str,
                           n_matches: int = 10) -> list:
    """Get formation history for a team from match JSONs.

    Returns list of dicts: [{formation, date, match_id, opponent, round}, ...]
    sorted by date descending (most recent first).
    """
    if stats_df.empty:
        return []

    team_matches = stats_df[stats_df["team"] == team]
    if team_matches.empty:
        return []

    # Get unique matches for this team, sorted by date desc
    match_info = (
        team_matches.groupby("match_id")
        .first()[["date", "home_team", "away_team", "is_home", "round"]]
        .reset_index()
        .sort_values("date", ascending=False)
        .head(n_matches)
    )

    formations = []
    for _, row in match_info.iterrows():
        mid = int(row["match_id"])
        match_data = _get_match_json(mid)
        if not match_data:
            continue

        is_home = bool(row["is_home"])
        side = "home_lineup" if is_home else "away_lineup"
        lineup = match_data.get(side, {})
        formation = lineup.get("formation")
        if formation:
            opponent = row["away_team"] if is_home else row["home_team"]
            formations.append({
                "formation": formation,
                "date": str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10],
                "match_id": mid,
                "opponent": opponent,
                "round": int(row["round"]) if pd.notna(row["round"]) else None,
                "is_home": is_home,
            })

    return formations


def predict_formation(formation_history: list) -> dict:
    """Predict most likely formation based on recent history.

    Uses recency-weighted voting: more recent matches count more.
    Returns: {predicted: str, confidence: float, alternatives: [{formation, pct}]}
    """
    if not formation_history:
        # Must carry the SAME keys as the populated return: predict_team_lineup
        # reads formation_pred["last_used"] unconditionally, so a club with no
        # league history (a promoted side at MW1) raised KeyError and took the
        # whole lineup step down with it.
        return {"predicted": "4-3-3", "confidence": 0.0, "alternatives": [],
                "based_on_matches": 0, "last_used": None}

    # Weight by recency: most recent match gets weight N, oldest gets 1
    n = len(formation_history)
    weighted_counts = Counter()
    for i, fh in enumerate(formation_history):
        weight = n - i  # Most recent = highest weight
        weighted_counts[fh["formation"]] += weight

    total_weight = sum(weighted_counts.values())
    # Sort by weight desc
    ranked = weighted_counts.most_common()

    predicted = ranked[0][0]
    confidence = ranked[0][1] / total_weight

    alternatives = []
    for formation, weight in ranked:
        alternatives.append({
            "formation": formation,
            "pct": round(weight / total_weight * 100, 1),
            "count": sum(1 for fh in formation_history if fh["formation"] == formation),
        })

    return {
        "predicted": predicted,
        "confidence": round(confidence, 3),
        "alternatives": alternatives,
        "based_on_matches": n,
        "last_used": formation_history[0]["formation"] if formation_history else None,
    }


def _compute_form_scores(stats_df: pd.DataFrame, team: str,
                         form_window: int = 3) -> dict:
    """Compute form scores for all players on a team.

    Compares each player's average rating over the last `form_window` matches
    to their full-season average. The deviation is normalized by the player's
    own standard deviation, producing a z-score-like metric:

      form_z = (recent_avg - season_avg) / max(season_std, 0.3)

    This is bounded to [-2, +2] and scaled to a small percentage adjustment
    (max ±3%) that only matters for contested positions.

    Why z-score instead of raw diff: a 0.3 diff means different things for a
    consistent player (std=0.2, very unusual) vs a volatile one (std=0.7,
    normal variance). The z-score normalizes for this.

    Returns: {player_id: {"form_z": float, "form_trend": str,
              "recent_avg": float, "season_avg": float, "form_bonus": float}}
    """
    team_data = stats_df[stats_df["team"] == team].copy()
    if team_data.empty or "rating" not in team_data.columns:
        return {}

    # Get sorted match dates for recency ordering
    match_dates = (
        team_data.groupby("match_id")["date"]
        .first()
        .sort_values(ascending=False)
    )
    recent_match_ids = set(match_dates.head(form_window).index)

    result = {}
    for pid, grp in team_data.groupby("player_id"):
        ratings = grp["rating"].dropna()
        if len(ratings) < 3:
            continue  # Not enough data for meaningful form

        season_avg = ratings.mean()
        season_std = ratings.std()
        if pd.isna(season_std) or season_std < 0.3:
            season_std = 0.3  # Floor to avoid dividing by near-zero

        # Recent ratings (last form_window matches this player appeared in)
        recent = grp[grp["match_id"].isin(recent_match_ids)]["rating"].dropna()
        if len(recent) == 0:
            continue

        recent_avg = recent.mean()
        form_z = (recent_avg - season_avg) / season_std
        form_z = max(-2.0, min(2.0, form_z))  # Clamp to [-2, +2]

        # Form bonus: scale z-score to ±3% adjustment
        # Only significant form shifts matter: |z| < 0.5 → negligible bonus
        FORM_MAX_BONUS = 3.0  # max percentage points
        form_bonus = round(form_z * (FORM_MAX_BONUS / 2.0), 1)

        # Trend label for display
        if form_z >= 1.0:
            trend = "hot"
        elif form_z >= 0.5:
            trend = "good"
        elif form_z <= -1.0:
            trend = "cold"
        elif form_z <= -0.5:
            trend = "poor"
        else:
            trend = "stable"

        result[int(pid)] = {
            "form_z": round(form_z, 2),
            "form_trend": trend,
            "recent_avg": round(recent_avg, 2),
            "season_avg": round(season_avg, 2),
            "form_bonus": form_bonus,
        }

    return result


def _compute_recent_stats(stats_df: pd.DataFrame, team: str,
                          window: int = 5) -> dict:
    """Compute recent performance stats for all players on a team.

    Aggregates key metrics from the last `window` matches each player appeared
    in. Returns per-player dicts with totals and per-90 rates useful for
    dashboard display.

    Returns: {player_id: {"goals": int, "assists": int, "xg": float,
              "key_passes": float, "shots": float, "minutes": int}}
    """
    team_data = stats_df[stats_df["team"] == team].copy()
    if team_data.empty:
        return {}

    result = {}
    for pid, grp in team_data.groupby("player_id"):
        recent = grp.sort_values("date", ascending=False).head(window)
        mins = recent["minutes"].sum()
        result[int(pid)] = {
            "goals": int(recent["goals"].sum()) if "goals" in recent.columns else 0,
            "assists": int(recent["assists"].sum()) if "assists" in recent.columns else 0,
            "xg": round(float(recent["xg"].sum()), 2) if "xg" in recent.columns else 0,
            "key_passes": round(float(recent["key_passes"].mean()), 1) if "key_passes" in recent.columns else 0,
            "shots_per_game": round(float(recent["total_shots"].mean()), 1) if "total_shots" in recent.columns else 0,
            "minutes": int(mins),
            "matches": len(recent),
        }
    return result


#: Pre-season wiring constants.  All three effects are inert when `preseason`
#: is None, so the historical code path is bit-for-bit unchanged.
#: A friendly is worth less than a league match (free rotation, uneven
#: opposition), so it informs the PRIOR RATE but does not raise PRIOR_STRENGTH:
#: a 10-appearance regular still keeps ~83% observed weight.
PRESEASON_ONLY_PRIOR = 40.0      # unproven player: below a coin flip, not at it
PRESEASON_PRIOR_SHRINK = 2.0     # friendly start-rate is itself shrunk toward 50
PRESEASON_FADE_MATCHES = 5.0     # league matches that fully retire the signal


def _preseason_entries(preseason: dict | None, known_ids: set, total_matches: int,
                       fade: float) -> list:
    """Candidate entries for players who exist ONLY in pre-season.

    Summer arrivals and promoted youth have no rows in the league table, so the
    frequency loop never sees them and they cannot be picked no matter how
    obviously they are starting. Same Bayesian form as the league path, with
    zero league appearances and a deliberately pessimistic prior -- friendly
    minutes are evidence of presence, not of a guaranteed shirt.

    Shared by both callers on purpose: a PROMOTED club has no league rows at
    all, so `get_starter_frequency` returned an empty pool for it while holding
    25-28 friendly players. Venezia, Frosinone and Monza all hit that path.
    """
    if not preseason or fade <= 0:
        return []

    out = []
    for pid, ps in preseason.get("players", {}).items():
        if int(pid) in known_ids or ps["appearances"] <= 0:
            continue
        PRIOR_STRENGTH = 2
        shrunk = (
            (ps["start_pct"] * ps["appearances"] + PRESEASON_ONLY_PRIOR * PRIOR_STRENGTH)
            / (ps["appearances"] + PRIOR_STRENGTH)
        )
        shirt = ps.get("shirt_number")
        out.append({
            "player_id": int(pid),
            "name": ps["name"],
            "position": ps["position"],
            "position_label": POSITION_MAP.get(ps["position"], ps["position"]),
            "shirt_number": int(shirt) if pd.notna(shirt) else None,
            "starts": ps["starts"],
            "appearances": 0,          # zero LEAGUE appearances, by definition
            "total_matches": total_matches,
            # Scaled by fade: once league matches exist, having played none of
            # them is itself evidence against starting, so a player known only
            # from July must not stay parked at 80%.
            "start_pct": round(max(0, min(100, shrunk * fade)), 1),
            "total_minutes": 0,
            "avg_minutes": 0.0,
            "avg_rating": 0,
            "is_new_signing": True,
            "start_streak": 0,
            "preseason_only": True,
            "preseason_starts": ps["starts"],
            "preseason_appearances": ps["appearances"],
            "preseason_minutes": ps["minutes"],
        })
    return out
PRESEASON_ABSENT_PENALTY = -15.0  # league regular who featured in zero friendlies
PRESEASON_ABSENT_MIN_MATCHES = 3  # below this, absence carries no information

# --- XI scoring components -------------------------------------------------
# Promoted from function-locals 2026-08-01 so they can be ABLATED by the
# backtest.  Values are unchanged; this refactor is behaviour-identical and was
# verified by re-running the replay and getting byte-identical hit rates.
#
# ⚠️ All of these were tuned on MID-SEASON matchweeks, where "the most recent
# match" is last week.  At matchweek 1 the most recent match is the FINAL match
# of LAST season -- routinely a dead rubber with a rotated XI -- and the replay
# measures the model scoring 47.6% there against 48.8% for raw start-counts.
# See scripts/analysis/backtest_preseason_signal.py --ablate.
RECENCY_DECAY = 0.85          # exponential weight per match of age
SHRINK_PRIOR_STRENGTH = 2     # virtual matches at SHRINK_PRIOR_RATE
SHRINK_PRIOR_RATE = 50.0      # uninformed prior when there is no friendly signal
LAST_MATCH_STARTED_BONUS = 25.0    # started the most recent match
LAST_MATCH_BENCHED_PENALTY = -5.0  # in the squad, not selected
LAST_MATCH_ABSENT_PENALTY = -10.0  # not in the squad at all

# Manager inertia is REGIME-DEPENDENT, and this is the fix for the matchweek-1
# regression. Measured by --ablate over 2024-25 + 2025-26:
#
#   component disabled        MW1     MW3     MW4     MW5
#   nothing (baseline)       48.3%   81.5%   76.6%   77.0%
#   no last-match bonus      49.5%   78.4%   71.2%   70.7%
#   -- naive floor --        48.6%   78.3%   71.4%   72.8%
#
# The bonus is simultaneously the model's MOST valuable component in-season
# (removing it collapses MW3-5 to the naive floor) and the ONLY component whose
# removal helps at MW1 -- where it drags the model BELOW raw start-counts.
#
# Why: "he started the most recent match" is strong evidence when that match was
# last week. At matchweek 1 the most recent match is the final round of a
# FINISHED season -- routinely a dead rubber with a rotated XI, played before a
# transfer window, by players who may no longer be at the club.
#
# So scale the bonus rather than delete it, and only when the table predates the
# window. 0.0 = ignore inertia entirely at MW1.
#
# HONEST SCOPE -- re-measured over EIGHT seasons (2018-19..2025-26, n=316 MW1
# fixtures; the `off` arm never touches friendlies, so it is not limited to the
# two backfilled pre-seasons):
#
#     naive floor   50.00%
#     off BEFORE    48.68%   <- the defect generalises across all 8 seasons
#     off AFTER     49.74%   <- +1.06pp, paired t=2.08 (82 better, 69 worse)
#
# This is a PARTIAL repair, not a solution. The gate recovers about a point and
# the direction is corroborated independently by the ablation, but the model is
# STILL below raw start-counts at matchweek 1. On the 2-season sample it looked
# like it cleared the floor (49.5 vs 48.6); the 8-season sample says otherwise.
# Do not cite the 2-season figure. MW1 remains an open problem.
LAST_MATCH_STALE_SCALE = 0.0


def get_starter_frequency(stats_df: pd.DataFrame, team: str,
                          n_matches: int = 10,
                          preseason: dict | None = None,
                          target_season: str | None = None) -> list:
    """Get starter frequency for each player in the team.

    target_season: the season being PREDICTED.  Supplying it is what lets the
        manager-inertia bonus know whether "the most recent match" was last week
        or the final round of a season that has since ended.  See
        LAST_MATCH_STALE_SCALE.  When None, falls back to the pre-season-based
        detection, which only works when `preseason` is also supplied.

    Uses recency-weighted start percentage: more recent matches count more.
    Decay factor 0.85 means the most recent match has ~5x the weight of
    the 10th-most-recent match. This ensures players on current hot streaks
    (like starting the last 5 straight) rank above players with the same
    raw count but more spread-out starts.

    Form adjustment: players with significantly above/below-average recent
    ratings get a small start_pct bonus/penalty (max ±3%). This only matters
    for contested positions where the margin between starter and bench is thin.

    Returns list of player dicts sorted by start_pct desc.
    """
    # A club with NO league history at all -- a promoted side, or an empty table
    # before the first scrape -- still has friendlies, and they are then the only
    # evidence in existence. Returning [] here silently produced an empty XI for
    # Venezia, Frosinone and Monza while 25-28 friendly players sat unused.
    team_data = stats_df[stats_df["team"] == team].copy() if not stats_df.empty else stats_df
    if stats_df.empty or team_data.empty:
        entries = _preseason_entries(preseason, set(), 0, 1.0)
        entries.sort(key=lambda p: (p["start_pct"], p["avg_minutes"]), reverse=True)
        return entries

    # Compute form scores (recent rating vs season average)
    form_scores = _compute_form_scores(stats_df, team)

    # Compute recent performance stats (goals, assists, xG, etc.)
    recent_stats = _compute_recent_stats(stats_df, team)

    # Get the last n_matches match_ids for this team, ordered most recent first
    match_dates = (
        team_data.groupby("match_id")["date"]
        .first()
        .sort_values(ascending=False)
        .head(n_matches)
    )
    match_ids = match_dates.index.tolist()  # Most recent first
    most_recent_match_id = match_ids[0] if match_ids else None
    recent = team_data[team_data["match_id"].isin(match_ids)]
    total_matches = len(match_ids)

    # How much should the pre-season signal still count?
    #
    # It must EXPIRE once the league season is under way, or it inverts into the
    # very error it was built to fix: simulated at 8 matchweeks played, the
    # un-faded signal injected 32 players at 80% who had not played a league
    # minute all season, and permanently docked every regular who missed July.
    #
    # Match COUNT alone cannot decide this -- during pre-season the 10-match
    # window is full of LAST season's matches, so any "fade by matches played"
    # rule would silence the signal exactly when it is the only evidence there
    # is. The discriminator is whether the league table has caught up to the
    # friendlies' own season. If it has, the season has started and every league
    # match played is direct evidence that supersedes a July friendly.
    _fade = 1.0
    # FAIL-SAFE DEFAULT. Assume the table is CURRENT unless we can POSITIVELY
    # establish it is not. This flag now gates manager inertia, which the
    # ablation showed is the model's most valuable in-season component -- so
    # defaulting to "predates" would disable it for every club with no friendly
    # data, and for the ENTIRE league if the friendlies parquet ever goes
    # missing, silently collapsing MW3-5 to the naive floor (-3 to -6pp).
    # `target_season` is authoritative and needs no preseason dict; the
    # preseason comparison is the fallback for callers that supply neither.
    _table_predates_window = False
    _table_season = (str(team_data["season"].iloc[0])
                     if "season" in team_data.columns and len(team_data) else None)
    if _table_season is not None:
        if target_season is not None:
            _table_predates_window = _table_season != str(target_season)
        elif preseason and preseason.get("season"):
            _table_predates_window = _table_season != str(preseason["season"])
    if preseason and _table_season is not None:
        if _table_season == str(preseason.get("season")):
            # The table is from the SAME season as the friendlies, so the most
            # recent league match was played after the transfer window, not
            # before it.  (This block no longer sets `_table_predates_window`:
            # the positive detection above already yields False in exactly this
            # case, and re-setting it here silently OVERRODE an explicit
            # `target_season`, which is authoritative.)
            _fade = max(0.0, 1.0 - total_matches / PRESEASON_FADE_MATCHES)

    # Build match recency rank: 0 = most recent, 1 = 2nd most recent, ...
    match_recency = {mid: rank for rank, mid in enumerate(match_ids)}

    # Recency weights: exponential decay (RECENCY_DECAY^rank)
    # Per-player max_weight only counts matches where the player was in the squad,
    # so players absent due to injury aren't penalized for those matches.
    DECAY = RECENCY_DECAY

    players = []
    for (pid, pname), grp in recent.groupby(["player_id", "player_name"]):
        starts = int(grp["is_starter"].sum())
        appearances = len(grp)
        total_mins = int(grp["minutes"].sum())
        avg_mins = round(grp["minutes"].mean(), 1)
        _raw_rating = grp["rating"].mean() if "rating" in grp.columns else 0
        avg_rating = round(_raw_rating, 2) if pd.notna(_raw_rating) else 0
        position = grp["position"].mode().iloc[0] if len(grp["position"].mode()) > 0 else "?"
        shirt = grp["shirt_number"].mode().iloc[0] if "shirt_number" in grp.columns and len(grp["shirt_number"].mode()) > 0 else None

        # Per-player max_weight: only matches this player was in the squad
        player_match_ids = set(grp["match_id"])
        player_max_weight = sum(
            DECAY ** match_recency[mid]
            for mid in player_match_ids
            if mid in match_recency
        )

        # Recency-weighted start percentage
        weighted_starts = 0.0
        for _, row in grp.iterrows():
            if row["is_starter"]:
                rank = match_recency.get(row["match_id"], total_matches)
                weighted_starts += DECAY ** rank
        raw_weighted_pct = weighted_starts / player_max_weight * 100 if player_max_weight > 0 else 0

        # Bayesian shrinkage: blend observed rate toward 50% prior based on
        # sample size. With few appearances, we're less confident → pull toward
        # 50%. With many appearances, the observed rate dominates.
        # k = strength of prior (equivalent to k "virtual" matches at 50%)
        # k=2: 10-app player barely affected (~92%), 2-app player corrected (83%)
        PRIOR_STRENGTH = SHRINK_PRIOR_STRENGTH  # virtual matches of uncertainty
        PRIOR_RATE = SHRINK_PRIOR_RATE
        # Effect (a): an INFORMED prior. When we watched this player through
        # pre-season, "what we believed before seeing league data" is his
        # friendly start rate, not a coin flip. Strength is deliberately NOT
        # raised — see PRESEASON_ONLY_PRIOR note above.
        # The informed prior is ITSELF an estimate off a small sample, so it is
        # shrunk toward 50 by its own friendly count. Without this, one
        # unused-sub appearance in a club's only friendly drove a player's
        # prior to 0% and cost him 23 points (measured on Milan/Odogu).
        _ps = (preseason or {}).get("players", {}).get(int(pid))
        if _ps and _ps["appearances"] > 0 and _fade > 0:
            _fr_n = _ps["appearances"]
            _informed = ((_ps["start_pct"] * _fr_n + 50.0 * PRESEASON_PRIOR_SHRINK)
                         / (_fr_n + PRESEASON_PRIOR_SHRINK))
            PRIOR_RATE = 50.0 + (_informed - 50.0) * _fade
        shrunk_pct = (raw_weighted_pct * appearances + PRIOR_RATE * PRIOR_STRENGTH) / (appearances + PRIOR_STRENGTH)

        # Recent-start momentum: consecutive starts from the most recent match.
        sorted_by_date = grp.sort_values("date", ascending=False)
        sorted_starts = sorted_by_date["is_starter"].values
        start_streak = 0
        for s in sorted_starts:
            if s:
                start_streak += 1
            else:
                break

        # Last-match-started bonus: managers show strong inertia — a player
        # who started the previous match has ~75% chance of starting the next.
        # The "repeat last XI" baseline achieves 8.2/11 by just re-selecting
        # whoever started last time. This bonus ensures the model respects
        # that signal rather than diluting it across the 10-match window.
        #
        # Three tiers based on player's status in the team's most recent match:
        #   Started → +25% (strong anchor toward repeating last XI)
        #   Benched → -5%  (in squad but not selected; moderate signal)
        #   Absent  → -10% (not in squad at all; strong negative signal)
        #
        # Calibrated via parametric backtest over 81 team-matches:
        #   +8/-3/0  → 8.04/11 (original)
        #   +25/-5/-10 → 8.32/11 (selected — beats repeat baseline 8.2)
        #   +30/-10/-15 → 8.37/11 (marginal gain, too aggressive)
        #
        # Graduated absent penalty (-3 for 1-match, -10 for 2+) was tested
        # and performed slightly worse (8.20 vs 8.23 on 5-match backtest).
        # The 10-match weighted average already handles tactical rest: a
        # regular with 9/10 starts minus -10 penalty still ranks ~75%.
        last_match_bonus = 0.0
        if most_recent_match_id is not None and len(sorted_by_date) > 0:
            player_last_match_id = sorted_by_date.iloc[0]["match_id"]
            player_started_last = bool(sorted_by_date.iloc[0]["is_starter"])
            if player_last_match_id == most_recent_match_id:
                # Player was in the squad for the most recent match
                if player_started_last:
                    last_match_bonus = LAST_MATCH_STARTED_BONUS
                else:
                    last_match_bonus = LAST_MATCH_BENCHED_PENALTY   # benched
            else:
                # Player's last appearance was NOT the most recent match
                # (absent from squad — injury, rest, dropped, etc.)
                last_match_bonus = LAST_MATCH_ABSENT_PENALTY

        # Manager inertia does not survive a summer. See LAST_MATCH_STALE_SCALE.
        if _table_predates_window:
            last_match_bonus *= LAST_MATCH_STALE_SCALE

        # Form adjustment: apply a small bonus/penalty from recent rating trends.
        # Max ±3% — only matters for contested positions where margins are thin.
        form_data = form_scores.get(int(pid), {})
        form_bonus = form_data.get("form_bonus", 0)
        # Effect (c): a last-season regular who featured in ZERO friendlies
        # while his club played several has most likely left, been sold, or is
        # long-term injured. Squad presence counts, so an unused sub is safe.
        preseason_bonus = 0.0
        if (preseason
                and _fade > 0
                and preseason.get("club_friendlies", 0) >= PRESEASON_ABSENT_MIN_MATCHES
                and int(pid) not in preseason.get("players", {})):
            preseason_bonus = PRESEASON_ABSENT_PENALTY * _fade
            # Drop the manager-inertia bonus as well. "+25 because he started
            # the last league match" is an argument about squad continuity, and
            # it does not survive a transfer window: without this, a 10/10
            # regular sat at shrunk 91.7 + 25 = clipped 100, so the penalty was
            # invisible for precisely the players a summer sale matters most for.
            # Only drop the inertia bonus when the "last match" predates the
            # transfer window. Once the new season is under way, "he started
            # last week" is current-season proof and must survive: scaling it
            # by the fade made the MW1 penalty (-32) HARSHER than pre-season
            # (-23), penalising a player for missing July right after he
            # started the opening fixture.
            if _table_predates_window:
                last_match_bonus = 0.0

        adjusted_pct = shrunk_pct + form_bonus + last_match_bonus + preseason_bonus
        weighted_pct = round(max(0, min(100, adjusted_pct)), 1)

        # Detect new signings: player appeared in last n_matches but not before
        all_player_matches = team_data[team_data["player_id"] == pid]
        oldest_in_window = match_dates.iloc[-1] if len(match_dates) > 0 else None
        is_new_signing = False
        if oldest_in_window is not None:
            pre_window = all_player_matches[all_player_matches["date"] < oldest_in_window]
            if len(pre_window) == 0 and appearances <= 5:
                is_new_signing = True

        player_entry = {
            "player_id": int(pid),
            "name": pname,
            "position": position,
            "position_label": POSITION_MAP.get(position, position),
            "shirt_number": int(shirt) if pd.notna(shirt) else None,
            "starts": starts,
            "appearances": appearances,
            "total_matches": total_matches,
            "start_pct": weighted_pct,
            "total_minutes": total_mins,
            "avg_minutes": avg_mins,
            "avg_rating": avg_rating,
            "is_new_signing": is_new_signing,
            "start_streak": start_streak,
        }
        # Add form data if available
        if form_data:
            player_entry["form_trend"] = form_data.get("form_trend", "stable")
            player_entry["form_z"] = form_data.get("form_z", 0)
            player_entry["recent_rating"] = form_data.get("recent_avg", avg_rating)
        # Add recent performance stats
        perf = recent_stats.get(int(pid), {})
        if perf:
            player_entry["recent_goals"] = perf["goals"]
            player_entry["recent_assists"] = perf["assists"]
            player_entry["recent_xg"] = perf["xg"]
            player_entry["recent_key_passes"] = perf["key_passes"]
            player_entry["recent_shots_pg"] = perf["shots_per_game"]
        players.append(player_entry)

    # Effect (b): players who exist ONLY in pre-season. Summer arrivals and
    # promoted youth have no rows in last season's league table, so the loop
    # above never sees them and they cannot be picked no matter how obviously
    # they are starting. Same Bayesian form, with zero league appearances and a
    # deliberately pessimistic prior — friendly minutes are evidence of
    # presence, not of a guaranteed shirt.
    players.extend(_preseason_entries(
        preseason, {p["player_id"] for p in players}, total_matches, _fade))

    players.sort(key=lambda p: (p["start_pct"], p["avg_minutes"]), reverse=True)
    return players


def compute_suspensions(incidents_df: pd.DataFrame, stats_df: pd.DataFrame,
                        team: str) -> dict:
    """Compute suspensions from yellow card accumulation AND red cards.

    Serie A rules:
    - 5 yellow cards = 1-match ban
    - Direct red card = 1-match ban (minimum)
    - Second yellow (yellowRed) = 1-match ban + counts toward yellow tally

    Returns: {player_id: {name, yellows, suspended, suspension_reason, risk_level, ...}}
    """
    if incidents_df.empty or stats_df.empty:
        return {}

    # Get all match_ids for this team in current season
    team_matches = stats_df[stats_df["team"] == team]
    team_match_ids = set(team_matches["match_id"].unique())
    max_round = int(team_matches["round"].max()) if len(team_matches) > 0 else 0

    # Build match_id -> is_home mapping and match_id -> round mapping
    match_sides = {}
    match_rounds = {}
    for mid, grp in team_matches.groupby("match_id"):
        row = grp.iloc[0]
        match_sides[mid] = bool(row["is_home"])
        match_rounds[mid] = int(row["round"])

    # Get ALL cards (yellow, yellowRed, red) for this team
    all_cards = incidents_df[
        (incidents_df["match_id"].isin(team_match_ids))
        & (incidents_df["incident_type"] == "card")
        & (incidents_df["card_type"].isin(["yellow", "yellowRed", "red"]))
    ].copy()

    # Filter to only this team's cards (using is_home flag)
    team_cards = []
    for _, card in all_cards.iterrows():
        mid = card["match_id"]
        if mid in match_sides:
            card_is_home = bool(card["is_home"])
            team_is_home = match_sides[mid]
            if card_is_home == team_is_home:
                team_cards.append(card)

    if not team_cards:
        return {}

    cards_df = pd.DataFrame(team_cards)

    result = {}
    for pid, grp in cards_df.groupby("player_id"):
        pid_int = int(pid) if str(pid).isdigit() else pid
        name = grp["player_name"].iloc[0]

        # Separate card types
        yellow_cards = grp[grp["card_type"] == "yellow"]
        yellow_red_cards = grp[grp["card_type"] == "yellowRed"]
        direct_red_cards = grp[grp["card_type"] == "red"]

        # Yellow count: yellows + yellowReds (second yellow counts toward tally)
        yellow_count = len(yellow_cards) + len(yellow_red_cards)

        # Determine rounds for each card type
        def get_rounds(card_grp):
            rounds = []
            for _, row in card_grp.iterrows():
                r = match_rounds.get(row["match_id"])
                if r is not None:
                    rounds.append(r)
            return sorted(rounds)

        yellow_rounds = get_rounds(pd.concat([yellow_cards, yellow_red_cards]) if len(yellow_red_cards) > 0 else yellow_cards)
        red_rounds = get_rounds(direct_red_cards)
        yellow_red_rounds = get_rounds(yellow_red_cards)

        last_yellow_round = max(yellow_rounds) if yellow_rounds else None

        # --- Suspension logic ---
        suspended = False
        suspension_reason = None

        # 1. Check direct red card in latest round
        if red_rounds and red_rounds[-1] == max_round:
            suspended = True
            suspension_reason = f"Direct red card (round {red_rounds[-1]})"

        # 2. Check second yellow (yellowRed) in latest round
        if yellow_red_rounds and yellow_red_rounds[-1] == max_round:
            suspended = True
            suspension_reason = f"Second yellow / red card (round {yellow_red_rounds[-1]})"

        # 3. Check yellow accumulation (5 yellows)
        if yellow_count >= YELLOW_SUSPENSION_THRESHOLD and yellow_count % YELLOW_SUSPENSION_THRESHOLD == 0:
            # Check if suspension was already served: did they play after the 5th yellow?
            threshold_round = yellow_rounds[YELLOW_SUSPENSION_THRESHOLD - 1] if len(yellow_rounds) >= YELLOW_SUSPENSION_THRESHOLD else None
            if threshold_round is not None:
                # Check if player appeared after the threshold round
                player_matches = team_matches[team_matches["player_id"] == pid_int]
                if len(player_matches) > 0:
                    player_rounds = set(player_matches["round"].astype(int))
                    played_after = any(r > threshold_round for r in player_rounds)
                    if not played_after:
                        suspended = True
                        suspension_reason = f"Yellow card accumulation ({yellow_count} yellows)"

        # Risk level
        risk_level = "none"
        remainder = yellow_count % YELLOW_SUSPENSION_THRESHOLD
        if suspended:
            risk_level = "suspended"
        elif remainder == YELLOW_SUSPENSION_THRESHOLD - 1:
            risk_level = "high"  # One yellow away from ban
        elif remainder == YELLOW_SUSPENSION_THRESHOLD - 2:
            risk_level = "medium"
        elif yellow_count >= YELLOW_SUSPENSION_THRESHOLD:
            risk_level = "served"

        result[str(pid_int)] = {
            "name": name,
            "yellows": yellow_count,
            "reds": len(direct_red_cards),
            "yellow_reds": len(yellow_red_cards),
            "suspended": suspended,
            "suspension_reason": suspension_reason,
            "risk_level": risk_level,
            "last_yellow_round": last_yellow_round,
            "last_red_round": red_rounds[-1] if red_rounds else None,
            "yellow_rounds": yellow_rounds,
        }

    return result


def get_missing_players(stats_df: pd.DataFrame, team: str) -> list:
    """Get missing/injured/suspended players by aggregating the last 3 match JSONs.

    Aggregating across multiple matches catches:
    - Persistent injuries (listed in 2+ matches → high confidence still out)
    - New injuries (listed in latest match only → might be fresh)
    - Recovered players (listed in older match but not the latest → likely back)

    The most recent match's data wins for conflicting info. Players listed as
    missing only in older matches (not the latest) are filtered unless the
    expected return date is still in the future.

    Filters out players whose expected return date has passed by >2 days
    (grace period for match-day-only updates).

    Returns list of {name, position, reason, description, expected_return}.
    """
    recent_ids = _get_latest_match_ids(stats_df, n=3)
    match_ids = recent_ids.get(team, [])
    if not match_ids:
        return []

    today = datetime.now().date()
    reason_map = {1: "Injury", 2: "Suspension", 3: "National Team",
                  4: "Personal", 13: "Suspension", 5: "Other"}

    # Collect missing players from each match, keyed by player_id
    # Most recent match = index 0 (highest priority)
    seen_players = {}  # player_id -> {entry, match_index, appearances}

    for match_idx, match_id in enumerate(match_ids):
        match_data = _get_match_json(match_id)
        if not match_data:
            continue

        team_matches = stats_df[
            (stats_df["team"] == team) & (stats_df["match_id"] == match_id)
        ]
        if team_matches.empty:
            continue

        is_home = bool(team_matches.iloc[0]["is_home"])
        side = "home_lineup" if is_home else "away_lineup"
        lineup = match_data.get(side, {})
        missing = lineup.get("missing_players", [])

        for mp in missing:
            player = mp.get("player", {})
            pid = player.get("id")
            if not pid:
                continue

            expected_return = mp.get("expectedEndDate")
            reason_code = mp.get("reason", 0)

            entry = {
                "name": player.get("name", "Unknown"),
                "short_name": player.get("shortName", ""),
                "position": POSITION_MAP.get(player.get("position", ""), player.get("position", "")),
                "reason": reason_map.get(reason_code, f"Code {reason_code}"),
                "reason_code": reason_code,
                "description": mp.get("description", ""),
                "type": mp.get("type", "missing"),
                "expected_return": expected_return,
                "player_id": pid,
            }

            if pid not in seen_players:
                seen_players[pid] = {
                    "entry": entry,
                    "first_match_idx": match_idx,
                    "appearances": 1,
                    "in_latest": match_idx == 0,
                }
            else:
                seen_players[pid]["appearances"] += 1
                # Most recent data wins (lower index = more recent)
                if match_idx < seen_players[pid]["first_match_idx"]:
                    seen_players[pid]["entry"] = entry
                    seen_players[pid]["first_match_idx"] = match_idx
                if match_idx == 0:
                    seen_players[pid]["in_latest"] = True

    # Filter and build result
    result = []
    for pid, info in seen_players.items():
        entry = info["entry"]
        expected_return = entry.get("expected_return")

        # If player was NOT in latest match's missing list, they may have recovered.
        # Only keep if expected return date is still in the future.
        if not info["in_latest"]:
            if expected_return:
                try:
                    ret_date = datetime.fromisoformat(expected_return.replace("Z", "+00:00")).date()
                    if ret_date < today:
                        continue  # Not in latest missing list AND return date passed → recovered
                except (ValueError, TypeError):
                    pass
            else:
                continue  # Not in latest missing list, no return date → likely recovered

        # Filter stale injuries (return date well past)
        if expected_return:
            try:
                ret_date = datetime.fromisoformat(expected_return.replace("Z", "+00:00")).date()
                days_past = (today - ret_date).days
                if days_past > 2:
                    desc_lower = (entry.get("description", "") or "").lower()
                    long_term_keywords = ["acl", "ligament", "cruciate", "meniscus", "achilles", "fracture"]
                    if not any(kw in desc_lower for kw in long_term_keywords):
                        continue  # Likely recovered, skip
            except (ValueError, TypeError):
                pass

        # Enrich with persistence info
        entry["missing_in_matches"] = info["appearances"]
        result.append(entry)

    return result


# Role families: positions that are "close" to each other on the pitch.
# Used to score how well a player fits a slot they haven't played exactly.
_ROLE_FAMILIES = {
    # Centre-back family
    "RCB": "CB", "CB": "CB", "LCB": "CB",
    # Fullback family
    "RB": "FB", "LB": "FB",
    # Wing-back family
    "RWB": "WB", "LWB": "WB",
    # Central midfield family
    "RCM": "CM", "CM": "CM", "LCM": "CM", "CDM": "CM", "RDM": "CM", "LDM": "CM",
    # Attacking midfield family
    "CAM": "AM", "RAM": "AM", "LAM": "AM", "SS": "AM",
    # Wide midfield family
    "RM": "WM", "LM": "WM",
    # Winger family
    "RW": "W", "LW": "W",
    # Striker family
    "ST": "ST", "RS": "ST", "LS": "ST",
    # Goalkeeper
    "GK": "GK",
}


def _same_role_family(label_a: str, label_b: str) -> bool:
    """Check if two tactical labels belong to the same role family."""
    return _ROLE_FAMILIES.get(label_a, label_a) == _ROLE_FAMILIES.get(label_b, label_b)


def _assign_players_to_slots(players: list, slot_labels: list,
                              position_map: dict) -> list:
    """Assign players to formation slots using historical position data.

    Uses a greedy algorithm: for each slot (in order), pick the unassigned
    player with the highest score for that specific slot. Score = number of
    times the player has played that exact position + related positions.

    Falls back to start_pct ordering if no position history is available.

    Returns: list of (player, slot_label) tuples in slot order.
    """
    if len(players) != len(slot_labels):
        # Mismatch — fall back to sequential
        return list(zip(players, slot_labels[:len(players)]))

    n = len(players)

    # Build score matrix: score[i][j] = how well player i fits slot j
    scores = []
    for player in players:
        pid = player["player_id"]
        pos_data = position_map.get(pid, {})
        counts = pos_data.get("position_counts", {})
        row = []
        for slot_label in slot_labels:
            exact = counts.get(slot_label, 0)
            related = sum(
                cnt for pos, cnt in counts.items()
                if _same_role_family(slot_label.upper(), pos.upper()) and pos != slot_label
            )
            # Score: exact matches weighted 10x, related 1x, start_pct as tiebreaker
            score = exact * 10 + related + player["start_pct"] / 1000
            row.append(score)
        scores.append(row)

    # Greedy assignment: iterate slots, pick best unassigned player
    assigned_players = set()
    result = [None] * n  # (player, slot_label) indexed by slot position

    # Process slots that have the most decisive data first (highest max score)
    slot_order = sorted(range(n), key=lambda j: max(scores[i][j] for i in range(n)), reverse=True)

    for j in slot_order:
        best_i = None
        best_score = -1
        for i in range(n):
            if i in assigned_players:
                continue
            if scores[i][j] > best_score:
                best_score = scores[i][j]
                best_i = i
        if best_i is not None:
            result[j] = (players[best_i], slot_labels[j])
            assigned_players.add(best_i)

    # Sanity check — fill any None entries (shouldn't happen)
    unassigned = [i for i in range(n) if i not in assigned_players]
    empty_slots = [j for j in range(n) if result[j] is None]
    for i, j in zip(unassigned, empty_slots):
        result[j] = (players[i], slot_labels[j])

    return result


def predict_starting_xi(starter_freq: list, formation: str,
                        missing_players: list, yellow_cards: dict,
                        player_position_map: dict = None) -> dict:
    """Predict the starting XI given formation, availability, and history.

    Logic:
    1. Parse formation into positional slots (e.g., 4-3-3 → 1 GK, 4 DEF, 3 MID, 3 FWD)
    2. Fill each slot with the highest-frequency starter for that position
    3. Use player_position_map (from Sofascore match history) to assign players
       to the correct tactical slot (e.g., RCB vs LCB) based on where they
       actually play, not just by start_pct ordering
    4. If a regular starter is suspended/injured, promote the next-best
    5. Flag uncertainties

    Args:
        player_position_map: Dict mapping player_id → {primary_position, position_counts, ...}
            from player_positions.extract_positions_from_matches(). If None, falls back to
            sequential slot assignment.

    Returns: {
        starters: [{name, position, position_label, shirt_number, start_pct, status}],
        bench: [{name, position, position_label, start_pct}],
        unavailable: [{name, reason}],
        changes_from_last: [{out, in, reason}]
    }
    """
    if player_position_map is None:
        player_position_map = {}
    # Parse formation: e.g., "3-5-2" → [3, 5, 2]
    try:
        parts = [int(x) for x in formation.split("-")]
    except (ValueError, AttributeError):
        parts = [4, 3, 3]  # fallback

    # Map formation parts to position categories
    # Always: 1 GK + formation parts distributed as DEF, MID, FWD
    slots = {"G": 1}
    if len(parts) == 3:
        slots["D"] = parts[0]
        slots["M"] = parts[1]
        slots["F"] = parts[2]
    elif len(parts) == 4:
        # e.g., 4-1-4-1 or 3-4-2-1
        slots["D"] = parts[0]
        slots["M"] = parts[1] + parts[2]
        slots["F"] = parts[3]
    elif len(parts) == 5:
        slots["D"] = parts[0]
        slots["M"] = parts[1] + parts[2] + parts[3]
        slots["F"] = parts[4]
    else:
        slots["D"] = 4
        slots["M"] = 3
        slots["F"] = 3

    # Build unavailable set — merge injury data + yellow card suspensions,
    # deduplicating by player name
    unavailable_names = set()
    unavailable_ids = set()
    unavailable_by_name = {}  # name_lower -> entry (for dedup)

    # Build a player_id -> name/position lookup from starter_freq
    id_to_info = {}
    for p in starter_freq:
        id_to_info[p["player_id"]] = {
            "name": p["name"],
            "position_label": p["position_label"],
            "start_pct": p["start_pct"],
        }

    for mp in missing_players:
        name_lower = mp["name"].lower()
        unavailable_names.add(name_lower)
        if mp.get("player_id"):
            unavailable_ids.add(mp["player_id"])
        unavailable_by_name[name_lower] = {
            "name": mp["name"],
            "short_name": mp.get("short_name", ""),
            "position": mp["position"],
            "reason": mp.get("reason", "Unknown"),
            "reason_code": mp.get("reason_code"),
            "description": mp.get("description", ""),
            "type": mp.get("type", "missing"),
            "expected_return": mp.get("expected_return"),
            "player_id": mp.get("player_id"),
        }

    # Check for suspended players from card tracking (yellows + reds)
    for pid, yc in yellow_cards.items():
        if yc["suspended"]:
            name_lower = yc["name"].lower()
            susp_desc = yc.get("suspension_reason") or f"Yellow card accumulation ({yc['yellows']} yellows)"
            if name_lower in unavailable_by_name:
                # Already listed (e.g., from match JSON missing_players with reason=2)
                # Enrich with suspension detail
                existing = unavailable_by_name[name_lower]
                existing["description"] = existing["description"] or susp_desc
                existing["type"] = "suspended"
                existing["yellows"] = yc["yellows"]
                if yc.get("reds"):
                    existing["reds"] = yc["reds"]
            else:
                # Not in match JSON missing_players — add as suspended
                unavailable_names.add(name_lower)
                # Try to resolve position from starter_freq
                info = id_to_info.get(int(pid), {}) if str(pid).isdigit() else {}
                entry = {
                    "name": info.get("name", yc["name"]),
                    "short_name": "",
                    "position": info.get("position_label", "?"),
                    "reason": "Suspension",
                    "reason_code": 2 if not yc.get("reds") else 13,
                    "description": susp_desc,
                    "type": "suspended",
                    "expected_return": None,
                    "player_id": int(pid) if str(pid).isdigit() else None,
                    "yellows": yc["yellows"],
                    "start_pct": info.get("start_pct"),
                }
                if yc.get("reds"):
                    entry["reds"] = yc["reds"]
                unavailable_by_name[name_lower] = entry

    unavailable_list = [v for v in unavailable_by_name.values() if v["name"]]

    # --- Tactical position reclassification ---
    # Sofascore labels players as G/D/M/F, but many players' ACTUAL tactical role
    # doesn't match. E.g., Conceição (F) plays RAM, Koopmeiners (M) plays LCB.
    # Use player_position_map to reclassify based on where they actually play.
    # This is critical for formations with attacking midfield (RAM/LAM/CAM) where
    # forwards compete for too few FWD slots while MID slots go unfilled.
    TACTICAL_TO_GROUP = {
        # Goal
        "GK": "G",
        # Defense
        "RB": "D", "LB": "D", "CB": "D", "RCB": "D", "LCB": "D",
        "RWB": "D", "LWB": "D",
        # Midfield (includes defensive, central, and attacking mid)
        "CDM": "M", "RDM": "M", "LDM": "M",
        "CM": "M", "RCM": "M", "LCM": "M",
        "RM": "M", "LM": "M",
        "CAM": "M", "RAM": "M", "LAM": "M",
        "SS": "M",  # second striker / trequartista plays behind ST
        # Attack
        "ST": "F", "RS": "F", "LS": "F",
        "RW": "F", "LW": "F",
    }

    # Which tactical slots does this formation actually have?
    formation_tactical = _get_tactical_labels(formation, slots)
    formation_mid_slots = set(formation_tactical.get("M", []))
    formation_fwd_slots = set(formation_tactical.get("F", []))
    # Attacking mid slots: RAM, LAM, CAM, SS within mid
    attacking_mid_slots = formation_mid_slots & {"RAM", "LAM", "CAM", "SS"}
    # Defensive positions within the formation
    formation_def_slots = set(formation_tactical.get("D", []))

    def _effective_position(player, sofascore_pos):
        """Determine effective formation group using tactical history.

        Conservative reclassification: only move F→M when a forward primarily
        plays attacking midfield (RAM/LAM/CAM/SS) and the formation has those
        slots. Other cross-group moves (M→D, D→M) cause more harm than good
        because wing-back/defensive-mid roles are ambiguous and already handled
        by the formation slot system.

        Requires ≥5 appearances AND ≥50% of total apps in the role.
        """
        pid = player["player_id"]
        pos_data = player_position_map.get(pid, {})
        counts = pos_data.get("position_counts", {})
        if not counts:
            return sofascore_pos

        primary = pos_data.get("primary_position", "")
        primary_group = TACTICAL_TO_GROUP.get(primary, sofascore_pos)
        primary_count = counts.get(primary, 0)

        # Only reclassify with strong evidence
        total_apps = sum(counts.values())
        if primary_count < 5 or primary_count < total_apps * 0.5:
            return sofascore_pos

        # F → M: forward who mostly plays attacking mid (RAM/LAM/CAM/SS)
        # Only reclassify if the formation HAS attacking mid slots
        if sofascore_pos == "F" and primary_group == "M" and attacking_mid_slots:
            return "M"

        return sofascore_pos

    # Separate available players by effective position
    available_by_pos = defaultdict(list)
    for p in starter_freq:
        name_lower = p["name"].lower()
        pid = p["player_id"]
        if name_lower in unavailable_names or pid in unavailable_ids:
            # Mark the unavailable player's position
            for u in unavailable_list:
                if u["name"].lower() == name_lower:
                    u["position"] = p["position_label"]
                    u["start_pct"] = p["start_pct"]
            continue
        effective_pos = _effective_position(p, p["position"]) if player_position_map else p["position"]
        p["_effective_position"] = effective_pos
        available_by_pos[effective_pos].append(p)

    # Get tactical position labels for this formation
    tactical_labels = _get_tactical_labels(formation, slots)

    # Fill starting XI
    starters = []
    bench = []
    changes = []

    # Sort candidates within each position group by start_pct.
    # GK: Bayesian shrinkage + last-match bonus now correctly handle backup GKs
    # (low appearances → shrunk toward 50%, benched → -5 penalty). Use start_pct
    # as primary like outfield players, with appearances as tiebreaker only.
    for pos in available_by_pos:
        if pos == "G":
            available_by_pos[pos].sort(
                key=lambda c: (c["start_pct"], c["appearances"], c["avg_minutes"]),
                reverse=True,
            )
        else:
            # Primary: start_pct, tiebreaker: start_streak (recent momentum),
            # then avg_minutes. Streak only matters when start_pct is close.
            available_by_pos[pos].sort(
                key=lambda c: (c["start_pct"], c.get("start_streak", 0), c["avg_minutes"]),
                reverse=True,
            )

    # Track who's been selected to avoid duplicates
    selected_ids = set()

    def _pick(candidates, count, min_pct=10.0):
        """Pick top `count` from candidates not already selected.

        Players below `min_pct` start_pct are deferred — left as a deficit
        to be filled from adjacent positions rather than selecting a player
        who essentially never starts. Pass min_pct=0 to disable.
        """
        picked = []
        deferred = []
        rest = []
        for c in candidates:
            if c["player_id"] in selected_ids:
                continue
            if len(picked) < count:
                if c["start_pct"] >= min_pct:
                    picked.append(c)
                    selected_ids.add(c["player_id"])
                else:
                    deferred.append(c)
            else:
                rest.append(c)
        # Put deferred candidates back into rest for potential adjacency fill
        rest = deferred + rest
        return picked, rest

    def _slot_score(player, slot_label):
        """Score how well a player fits a specific tactical slot.

        Uses position_counts from historical match data. Higher = better fit.
        Returns (exact_match_count, related_match_count, start_pct) for sorting.
        """
        pid = player["player_id"]
        pos_data = player_position_map.get(pid, {})
        counts = pos_data.get("position_counts", {})

        # Exact match: how many times this player has played this exact slot
        exact = counts.get(slot_label, 0)

        # Related match: similar positions (e.g., RCB counts for CB, LCB)
        # Group labels by role similarity
        related = 0
        label_upper = slot_label.upper()
        for played_pos, cnt in counts.items():
            pu = played_pos.upper()
            # Same if they share the base role (CB family, CM family, etc.)
            if _same_role_family(label_upper, pu):
                related += cnt

        return (exact, related, player["start_pct"])

    # Global display order counter — ensures correct sort after fill
    display_order = 0

    # Fill each position slot; if not enough at a position, borrow from adjacent
    deficits = {}
    remaining_by_pos = {}

    for pos, count in slots.items():
        candidates = available_by_pos.get(pos, [])
        selected, remaining = _pick(candidates, count)

        pos_labels = tactical_labels.get(pos, [])

        # Smart slot assignment: match players to their best-fitting slot
        # using historical position data
        if player_position_map and len(selected) > 1 and len(pos_labels) == len(selected):
            assigned = _assign_players_to_slots(selected, pos_labels, player_position_map)
        else:
            # Fallback: sequential assignment (first player → first slot)
            assigned = list(zip(selected, pos_labels[:len(selected)]))
            # Fill any remaining with generic labels
            for i in range(len(pos_labels), len(selected)):
                assigned.append((selected[i], POSITION_MAP.get(pos, pos)))

        for player, tact_label in assigned:
            status = "likely"
            if player["start_pct"] >= 80:
                status = "certain"
            elif player["start_pct"] >= 50:
                status = "likely"
            elif player["start_pct"] >= 30:
                status = "rotation"
            else:
                status = "unlikely"

            entry = {
                "name": player["name"],
                "position": player["position"],
                "position_label": tact_label,
                "shirt_number": player["shirt_number"],
                "start_pct": player["start_pct"],
                "avg_rating": player["avg_rating"],
                "avg_minutes": player["avg_minutes"],
                "status": status,
                "_display_order": display_order,
                "_formation_slot": pos,
            }
            if player.get("is_new_signing"):
                entry["is_new_signing"] = True
            # A player with NO league history at all is picked purely off
            # pre-season minutes. Carry that through so the UI can say so
            # rather than presenting him with the same confidence as a regular.
            if player.get("preseason_only"):
                entry["preseason_only"] = True
                entry["preseason_starts"] = player.get("preseason_starts", 0)
                entry["preseason_appearances"] = player.get("preseason_appearances", 0)
            if player.get("form_trend"):
                entry["form_trend"] = player["form_trend"]
                entry["form_z"] = player.get("form_z", 0)
                entry["recent_rating"] = player.get("recent_rating", player["avg_rating"])
            # Performance stats
            for stat_key in ("recent_goals", "recent_assists", "recent_xg",
                             "recent_key_passes", "recent_shots_pg", "start_streak"):
                if player.get(stat_key) is not None:
                    entry[stat_key] = player[stat_key]
            starters.append(entry)
            display_order += 1

        remaining_by_pos[pos] = remaining
        if len(selected) < count:
            deficits[pos] = count - len(selected)

    # Handle deficits: borrow from adjacent positions
    # FWD deficit → borrow from MID; MID deficit → borrow from DEF or FWD;
    # DEF deficit → borrow from MID
    adjacency = {"F": ["M"], "M": ["F", "D"], "D": ["M"], "G": []}
    slot_counters = {pos: sum(1 for s in starters if s.get("_formation_slot") == pos)
                     for pos in slots}

    def _fill_deficit(pos, deficit, label="fill-in"):
        """Try to fill deficit slots from adjacent positions, then fall back."""
        nonlocal display_order
        for adj_pos in adjacency.get(pos, []):
            if deficit <= 0:
                break
            extras = remaining_by_pos.get(adj_pos, [])
            borrowed, remaining_by_pos[adj_pos] = _pick(extras, deficit, min_pct=10.0)
            pos_labels = tactical_labels.get(pos, [])
            for b in borrowed:
                idx = slot_counters.get(pos, 0)
                tact_label = pos_labels[idx] if idx < len(pos_labels) else POSITION_MAP.get(pos, pos)
                slot_counters[pos] = idx + 1

                starters.append({
                    "name": b["name"],
                    "position": b["position"],
                    "position_label": tact_label,
                    "shirt_number": b["shirt_number"],
                    "start_pct": b["start_pct"],
                    "avg_rating": b["avg_rating"],
                    "avg_minutes": b["avg_minutes"],
                    "status": label,
                    "_display_order": display_order,
                    "_formation_slot": pos,
                })
                display_order += 1
                deficit -= 1
        return deficit

    for pos, deficit in deficits.items():
        deficit = _fill_deficit(pos, deficit, label="fill-in")

    # Second pass: if still unfilled slots, accept ANY remaining player (no min_pct)
    # Recalculate deficits
    remaining_deficits = {}
    for pos, count in slots.items():
        filled = sum(1 for s in starters if s.get("_formation_slot") == pos)
        if filled < count:
            remaining_deficits[pos] = count - filled

    if remaining_deficits:
        for pos, deficit in remaining_deficits.items():
            # Try own position group first (was deferred earlier)
            own_remaining = remaining_by_pos.get(pos, [])
            fallback, remaining_by_pos[pos] = _pick(own_remaining, deficit, min_pct=0)
            pos_labels = tactical_labels.get(pos, [])
            for fb in fallback:
                idx = slot_counters.get(pos, 0)
                tact_label = pos_labels[idx] if idx < len(pos_labels) else POSITION_MAP.get(pos, pos)
                slot_counters[pos] = idx + 1
                starters.append({
                    "name": fb["name"],
                    "position": fb["position"],
                    "position_label": tact_label,
                    "shirt_number": fb["shirt_number"],
                    "start_pct": fb["start_pct"],
                    "avg_rating": fb["avg_rating"],
                    "avg_minutes": fb["avg_minutes"],
                    "status": "fill-in",
                    "_display_order": display_order,
                    "_formation_slot": pos,
                })
                display_order += 1
                deficit -= 1
            # Still not enough? Try adjacent with no threshold
            if deficit > 0:
                _fill_deficit(pos, deficit, label="fill-in")

    # Quality swap: replace weak starters with strong unused players from
    # adjacent positions. E.g., if a MID slot is filled by a 35% player but
    # an 80% FWD is sitting unused, swap them. Only swap when the gap is large
    # (>25%) and the starter is weak (<40%), to avoid false positives.
    SWAP_THRESHOLD = 40.0   # only consider swapping starters below this %
    SWAP_MARGIN = 25.0      # replacement must be this many % points better
    swapped = True
    while swapped:
        swapped = False
        for i, s in enumerate(starters):
            if s["start_pct"] >= SWAP_THRESHOLD:
                continue
            s_pos = s.get("_formation_slot", "")
            # Check adjacent positions for better candidates
            for adj_pos in adjacency.get(s_pos, []):
                candidates = remaining_by_pos.get(adj_pos, [])
                for j, cand in enumerate(candidates):
                    if cand["player_id"] in selected_ids:
                        continue
                    if cand["start_pct"] > s["start_pct"] + SWAP_MARGIN:
                        # Swap: demote starter to remaining, promote candidate
                        pos_labels = tactical_labels.get(s_pos, [])
                        idx = slot_counters.get(s_pos, 0)
                        tact_label = s.get("position_label", POSITION_MAP.get(s_pos, s_pos))

                        new_starter = {
                            "name": cand["name"],
                            "position": cand["position"],
                            "position_label": tact_label,
                            "shirt_number": cand["shirt_number"],
                            "start_pct": cand["start_pct"],
                            "avg_rating": cand["avg_rating"],
                            "avg_minutes": cand["avg_minutes"],
                            "status": "quality-swap",
                            "_display_order": s.get("_display_order", display_order),
                            "_formation_slot": s_pos,
                        }
                        if cand.get("form_trend"):
                            new_starter["form_trend"] = cand["form_trend"]
                            new_starter["form_z"] = cand.get("form_z", 0)
                            new_starter["recent_rating"] = cand.get("recent_rating", cand["avg_rating"])
                        for stat_key in ("recent_goals", "recent_assists", "recent_xg",
                                         "recent_key_passes", "recent_shots_pg", "start_streak"):
                            if cand.get(stat_key) is not None:
                                new_starter[stat_key] = cand[stat_key]

                        # Return old starter to its position group
                        old_pid = None
                        for sf in starter_freq:
                            if sf["name"] == s["name"]:
                                old_pid = sf["player_id"]
                                remaining_by_pos.setdefault(s_pos, []).append(sf)
                                selected_ids.discard(old_pid)
                                break

                        selected_ids.add(cand["player_id"])
                        candidates.pop(j)
                        starters[i] = new_starter
                        swapped = True
                        break
                if swapped:
                    break

    # Build bench from remaining players across all positions
    # Use player_position_map for tactical labels (e.g., RCB instead of DEF)
    for pos in ["G", "D", "M", "F"]:
        for r in remaining_by_pos.get(pos, []):
            if r["player_id"] not in selected_ids:
                # Try to get tactical position from historical data
                pos_label = r["position_label"]  # fallback: generic DEF/MID/FWD
                if player_position_map:
                    pos_data = player_position_map.get(r["player_id"], {})
                    primary = pos_data.get("primary_position")
                    if primary:
                        pos_label = primary
                bench_entry = {
                    "name": r["name"],
                    "position": r["position"],
                    "position_label": pos_label,
                    "shirt_number": r["shirt_number"],
                    "start_pct": r["start_pct"],
                    "avg_rating": r["avg_rating"],
                }
                if r.get("form_trend"):
                    bench_entry["form_trend"] = r["form_trend"]
                    bench_entry["form_z"] = r.get("form_z", 0)
                    bench_entry["recent_rating"] = r.get("recent_rating", r["avg_rating"])
                for stat_key in ("recent_goals", "recent_assists", "recent_xg",
                                 "recent_key_passes", "recent_shots_pg", "start_streak"):
                    if r.get(stat_key) is not None:
                        bench_entry[stat_key] = r[stat_key]
                bench.append(bench_entry)

    # Sort bench by start_pct descending so truncation keeps the most
    # likely alternatives regardless of position group
    bench.sort(key=lambda b: b["start_pct"], reverse=True)

    # Close-call indicators: flag positions where starter and top bench
    # player are within 15 percentage points of each other
    for s in starters:
        pos = s.get("position")  # G, D, M, F
        s_pct = s["start_pct"]
        # Find best bench player at same position
        best_bench_pct = 0
        best_bench_name = None
        for b in bench:
            if b["position"] == pos:
                if b["start_pct"] > best_bench_pct:
                    best_bench_pct = b["start_pct"]
                    best_bench_name = b["name"]
        margin = round(s_pct - best_bench_pct, 1)
        if best_bench_name and margin < 15:
            s["contested"] = True
            s["margin"] = margin
            s["rival"] = best_bench_name

    # Detect changes: compare predicted XI to last match starters
    # (A starter who usually starts but is now unavailable → replacement flagged)
    for u in unavailable_list:
        if (u.get("start_pct") or 0) >= 50:
            # This was a regular starter — find their replacement
            pos = u.get("position", "")
            pos_code = next((k for k, v in POSITION_MAP.items() if v == pos), "")
            replacement = None
            for b in bench:
                if b["position"] == pos_code:
                    replacement = b["name"]
                    break
            changes.append({
                "out": u["name"],
                "in": replacement or "Unknown",
                "reason": u.get("description") or u.get("reason", "Unavailable"),
            })

    # Sort starters by formation display order (GK → DEF → MID → FWD, in slot order)
    starters.sort(key=lambda s: s.get("_display_order", 99))

    # Strip internal fields from output
    for s in starters:
        s.pop("_display_order", None)
        s.pop("_formation_slot", None)

    return {
        "starters": starters,
        "bench": bench[:7],  # Top 7 bench players
        "unavailable": unavailable_list,
        "changes_from_last": changes,
        "formation": formation,
        "total_available": sum(len(v) for v in available_by_pos.values()),
    }


def get_fitness_concerns(stats_df: pd.DataFrame, team: str) -> list:
    """Detect fitness concerns from the latest match.

    Flags starters who were subbed off unusually early (before 60 min), which
    could indicate injury, illness, or tactical issues. A halftime sub (45 min)
    is particularly suspicious.

    Also detects starters who played significantly fewer minutes than their
    recent average, suggesting managed fitness.

    Returns list of {player_id, name, minutes_played, concern_type, note}.
    """
    team_data = stats_df[stats_df["team"] == team].copy()
    if team_data.empty:
        return []

    # Get latest match
    latest_date = team_data["date"].max()
    latest_match = team_data[team_data["date"] == latest_date]
    latest_match_id = latest_match["match_id"].iloc[0]

    # Get starters who played < 60 minutes
    starters = latest_match[latest_match["is_starter"] == True]
    concerns = []

    for _, row in starters.iterrows():
        mins = int(row["minutes"])
        if mins >= 65:
            continue  # Normal full-ish match

        concern_type = "early_sub"
        if mins <= 46:
            note = f"Subbed at halftime ({mins}min) — possible injury or illness"
            concern_type = "halftime_sub"
        else:
            note = f"Subbed early ({mins}min) — fitness or tactical"

        concerns.append({
            "player_id": int(row["player_id"]),
            "name": row["player_name"],
            "position": row["position"],
            "minutes_played": mins,
            "concern_type": concern_type,
            "note": note,
            "match_id": int(latest_match_id),
            "match_date": str(latest_date),
        })

    return concerns


def predict_team_lineup(stats_df: pd.DataFrame, incidents_df: pd.DataFrame,
                        team: str, player_position_map: dict = None) -> dict:
    """Full lineup prediction for a single team.

    Combines formation prediction, starter frequency, yellow card tracking,
    missing player info, fitness concerns, and starting XI prediction.

    Args:
        player_position_map: Dict mapping player_id → detailed position info
            from player_positions.extract_positions_from_matches(). Enables
            tactical slot assignment (CB vs RB vs LWB).
    """
    # 1. Formation history & prediction
    preseason = load_preseason_signal(team)
    formation_history = get_formation_patterns(stats_df, team, n_matches=10)
    if not formation_history and preseason.get("formations"):
        # Promoted club: no league history at all, so the shapes trialled in
        # pre-season beat falling back to a hardcoded 4-3-3.
        formation_history = [{"formation": f} for f in preseason["formations"]]
    formation_pred = predict_formation(formation_history)

    # 2. Starter frequency
    starter_freq = get_starter_frequency(stats_df, team, n_matches=10,
                                         preseason=preseason)

    # 3. Suspension tracking (yellow accumulation + red cards)
    yellow_cards = compute_suspensions(incidents_df, stats_df, team)

    # 4. Missing players (injuries/suspensions from last 3 matches)
    missing_players = get_missing_players(stats_df, team)

    # 5. Fitness concerns (early subs in latest match)
    fitness_concerns = get_fitness_concerns(stats_df, team)

    # 6. Predict starting XI
    predicted_formation = formation_pred["predicted"]
    xi_prediction = predict_starting_xi(
        starter_freq, predicted_formation, missing_players, yellow_cards,
        player_position_map=player_position_map,
    )

    # 7. Annotate starters with fitness concerns
    concern_ids = {c["player_id"] for c in fitness_concerns}
    concern_map = {c["player_id"]: c for c in fitness_concerns}
    for starter in xi_prediction["starters"]:
        pid = starter.get("player_id")
        if not pid:
            # Try to match by name from starter_freq
            for sf in starter_freq:
                if sf["name"] == starter["name"]:
                    pid = sf["player_id"]
                    break
        if pid and pid in concern_ids:
            starter["fitness_concern"] = concern_map[pid]["note"]

    # 8. Suspension risk players (4 yellows = next yellow means ban)
    suspension_risk = []
    for pid, yc in yellow_cards.items():
        if yc["risk_level"] == "high":
            suspension_risk.append({
                "name": yc["name"],
                "yellows": yc["yellows"],
                "last_yellow_round": yc["last_yellow_round"],
            })

    return {
        "team": team,
        "formation": {
            "predicted": formation_pred["predicted"],
            "confidence": formation_pred["confidence"],
            "alternatives": formation_pred["alternatives"],
            "last_used": formation_pred["last_used"],
            "history": formation_history[:5],  # Last 5 for display
        },
        "predicted_xi": xi_prediction["starters"],
        "bench": xi_prediction["bench"],
        "unavailable": xi_prediction["unavailable"],
        "changes": xi_prediction["changes_from_last"],
        "fitness_concerns": fitness_concerns,
        "suspension_risk": suspension_risk,
        "yellow_cards": {
            pid: {
                "name": yc["name"],
                "yellows": yc["yellows"],
                "reds": yc.get("reds", 0),
                "risk_level": yc["risk_level"],
            }
            for pid, yc in yellow_cards.items()
            if yc["yellows"] >= 3 or yc.get("reds", 0) > 0
        },
    }


def generate_lineup_predictions() -> dict:
    """Main entry point: generate lineup predictions for all upcoming matches.

    Loads predictions.json for the match list, then generates lineup predictions
    for both teams in each match.
    """
    pred_path = DATA_DIR / "upcoming" / "predictions.json"
    if not pred_path.exists():
        print("No predictions.json found")
        return {}

    with open(pred_path) as f:
        pred_data = json.load(f)

    predictions = pred_data.get("predictions", [])
    if not predictions:
        print("No upcoming predictions")
        return {}

    # Load data sources
    print("Loading player match stats...")
    stats_df = _load_current_season_stats()
    if stats_df.empty:
        print("No player match stats available")
        return {}

    print("Loading match incidents...")
    incidents_df = _load_season_incidents()

    # Build detailed player position map from match history
    print("Building player position map from match history...")
    from scripts.prediction.player_positions import build_player_position_map
    player_position_map = build_player_position_map()

    # Get unique teams from predictions
    teams = set()
    for p in predictions:
        teams.add(p.get("home_team", ""))
        teams.add(p.get("away_team", ""))
    teams.discard("")

    # Map prediction team names to Sofascore team names.  Both should now use
    # canonical names via normalize_team(), but as a safety net the ladder in
    # `_resolve_team_name` also normalises and partial-matches -- and falls
    # back to the pre-season friendlies for a club with no league history at
    # all, which is every promoted club at matchweek 1.
    sofascore_teams = set(stats_df["team"].unique())
    preseason_clubs = preseason_club_names()
    team_map = {}
    team_source = {}
    for t in teams:
        resolved, source = _resolve_team_name(t, sofascore_teams, preseason_clubs)
        if resolved:
            team_map[t] = resolved
            team_source[t] = source

    print(f"Generating lineup predictions for {len(teams)} teams...")

    # Generate predictions per team (cache to avoid recomputing)
    team_predictions = {}
    for team in sorted(teams):
        ss_team = team_map.get(team)
        if not ss_team:
            print(f"  Warning: No Sofascore data for {team}")
            continue
        if team_source.get(team) == "preseason":
            print(f"  Processing {team} (Sofascore: {ss_team}) — "
                  "no league history, pre-season friendlies only...")
        else:
            print(f"  Processing {team} (Sofascore: {ss_team})...")
        team_predictions[team] = predict_team_lineup(
            stats_df, incidents_df, ss_team,
            player_position_map=player_position_map,
        )
        # Fix team name back to canonical
        team_predictions[team]["team"] = team

    # Build match-level predictions
    match_predictions = {}
    for pred in predictions:
        home = pred.get("home_team", "")
        away = pred.get("away_team", "")
        match_key = pred.get("match", f"{home} vs {away}")

        match_predictions[match_key] = {
            "match": match_key,
            "home_team": home,
            "away_team": away,
            "home_lineup": team_predictions.get(home, {}),
            "away_lineup": team_predictions.get(away, {}),
        }

    output = {
        "generated_at": datetime.now().isoformat(),
        "match_count": len(match_predictions),
        "team_count": len(team_predictions),
        "matches": match_predictions,
        "teams": team_predictions,
    }

    out_path = DATA_DIR / "upcoming" / "lineup_predictions.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Archive a copy so predictions aren't lost when re-generated.
    # Each archive is keyed by generation date. After the matches are played,
    # evaluate_past_predictions() compares these against actual Sofascore data.
    archive_dir = DATA_DIR / "lineup_history"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(archive_dir / archive_name, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Generated lineup predictions for {len(match_predictions)} matches → {out_path}")
    print(f"Archived → {archive_dir / archive_name}")
    return output


def evaluate_past_predictions(verbose: bool = True) -> dict:
    """Compare archived lineup predictions against actual lineups from Sofascore.

    Reads archived predictions from data/lineup_history/ and checks each
    predicted XI against the actual starters in player_match_stats.parquet.
    Produces per-match and per-team accuracy reports, and stores results
    back into data/lineup_history/evaluation.json for tracking over time.

    Returns dict with overall accuracy and per-team breakdown.
    """
    archive_dir = DATA_DIR / "lineup_history"
    eval_path = archive_dir / "evaluation.json"

    # Load existing evaluations (avoid re-evaluating already-scored matches)
    existing_eval = {}
    if eval_path.exists():
        with open(eval_path) as f:
            existing_eval = json.load(f)
    already_scored = set(existing_eval.get("scored_keys", []))

    # Load actual starters from Sofascore -- BOTH leagues.  Grading only the
    # Serie A file would silently score every EPL prediction as a miss (no
    # actual starters found), which reads as a model failure rather than a
    # missing input.  All seasons are kept here on purpose: this grades archived
    # predictions, which may be older than the current season.
    frames = []
    for name in _PLAYER_STATS_FILES:
        p = DATA_DIR / "external" / "sofascore" / name
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        print(f"No player-stats parquet ({', '.join(_PLAYER_STATS_FILES)}) — cannot evaluate")
        return {}

    stats_df = pd.concat(frames, ignore_index=True)
    stats_df["team"] = stats_df["team"].apply(normalize_team)

    # Build actual starters lookup: (date, team) → set of player names
    actual_starters = {}
    for (mid, team), grp in stats_df[stats_df["is_starter"] == True].groupby(["match_id", "team"]):
        date_val = grp["date"].iloc[0]
        date_str = str(date_val)[:10]
        actual_starters[(date_str, team)] = set(grp["player_name"].values)

    # Scan all archived prediction files
    if not archive_dir.exists():
        print("No lineup_history directory — generate predictions first")
        return {}

    archive_files = sorted(archive_dir.glob("predictions_*.json"))
    if not archive_files:
        print("No archived predictions found")
        return {}

    new_results = []
    for archive_file in archive_files:
        with open(archive_file) as f:
            pred_data = json.load(f)

        generated_at = pred_data.get("generated_at", "")

        for match_key, match_pred in pred_data.get("matches", {}).items():
            for side in ("home", "away"):
                lineup_data = match_pred.get(f"{side}_lineup", {})
                predicted_xi = lineup_data.get("predicted_xi", [])
                if not predicted_xi:
                    continue

                team = match_pred.get(f"{side}_team", "")
                if not team:
                    continue

                # Find the actual match date from predictions.json context
                # Match key format: "TeamA vs TeamB"
                # We need to find this match in actual data by team + approximate date
                pred_names = {p["name"] for p in predicted_xi}

                # Search for the actual match within a reasonable date window
                # Use the archive timestamp as approximate match date
                best_match = None
                best_date = None
                for (date_str, act_team), starters in actual_starters.items():
                    if act_team != team:
                        continue
                    # Check if this match was already scored
                    score_key = f"{date_str}:{team}"
                    if score_key in already_scored:
                        continue

                    # Skip if the match date is before the prediction was generated
                    if generated_at and date_str < generated_at[:10]:
                        continue

                    # This could be the match — check if date is reasonable
                    # (within 14 days of prediction)
                    if generated_at:
                        from datetime import timedelta
                        gen_date = datetime.fromisoformat(generated_at[:10])
                        match_date = datetime.fromisoformat(date_str)
                        if match_date - gen_date > timedelta(days=14):
                            continue
                        if match_date < gen_date:
                            continue

                    if len(starters) >= 11:
                        best_match = starters
                        best_date = date_str
                        break  # Take the first valid match

                if best_match is None:
                    continue

                score_key = f"{best_date}:{team}"
                if score_key in already_scored:
                    continue

                overlap = len(pred_names & best_match)
                missed = best_match - pred_names
                wrong = pred_names - best_match

                result = {
                    "date": best_date,
                    "team": team,
                    "match": match_key,
                    "overlap": overlap,
                    "predicted_count": len(pred_names),
                    "actual_count": len(best_match),
                    "missed": sorted(missed),
                    "wrong": sorted(wrong),
                    "generated_at": generated_at,
                    "archive_file": archive_file.name,
                }
                new_results.append(result)
                already_scored.add(score_key)

                if verbose:
                    status = "✓" if overlap >= 9 else "~" if overlap >= 7 else "✗"
                    print(f"  {status} {best_date} {team:>15s}: {overlap}/11"
                          f"{'  missed: ' + ', '.join(sorted(missed)) if missed else ''}")

    # Merge with existing results
    all_results = existing_eval.get("results", []) + new_results

    # Compute summary stats
    if all_results:
        overlaps = [r["overlap"] for r in all_results]
        overall_avg = round(sum(overlaps) / len(overlaps), 2)

        # Per-team averages
        from collections import defaultdict
        team_overlaps = defaultdict(list)
        for r in all_results:
            team_overlaps[r["team"]].append(r["overlap"])
        team_avgs = {
            team: round(sum(ovs) / len(ovs), 1)
            for team, ovs in sorted(team_overlaps.items())
        }
    else:
        overall_avg = 0
        team_avgs = {}

    evaluation = {
        "last_evaluated": datetime.now().isoformat(),
        "total_evaluations": len(all_results),
        "new_evaluations": len(new_results),
        "overall_avg_overlap": overall_avg,
        "team_averages": team_avgs,
        "scored_keys": sorted(already_scored),
        "results": all_results,
    }

    # Save
    with open(eval_path, "w") as f:
        json.dump(evaluation, f, indent=2, default=str)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Lineup Prediction Accuracy: {overall_avg}/11 "
              f"({len(all_results)} evaluations)")
        if team_avgs:
            print(f"\nPer-team:")
            for team, avg in sorted(team_avgs.items(), key=lambda x: x[1]):
                print(f"  {team:>15s}: {avg}/11")

    return evaluation


def assess_lineup_impact(
    predicted_names: list[str],
    confirmed_names: list[str],
    team: str,
    player_xg_map: dict = None,
) -> dict:
    """Compare predicted vs confirmed lineup and assess the impact.

    Called when real lineups drop ~1h before kickoff. Returns a concise
    impact assessment: which players changed, whether the confirmed XI
    is stronger/weaker, and whether the match prediction should be trusted.

    Args:
        predicted_names: List of 11 predicted starter names.
        confirmed_names: List of 11 confirmed starter names.
        team: Team name (for context).
        player_xg_map: Optional dict player_name → xG per 90. Used to
            compute squad strength delta. If None, only reports overlap.

    Returns:
        Dict with overlap, changes, strength_delta, and verdict.
    """
    # Fuzzy name matching for cross-source comparison
    import difflib
    import unicodedata

    def _norm(name):
        nfkd = unicodedata.normalize("NFKD", name)
        stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
        return " ".join(stripped.lower().split()).replace(".", "")

    pred_norm = {_norm(n): n for n in predicted_names}
    conf_norm = {_norm(n): n for n in confirmed_names}

    # Match confirmed names to predicted names
    matched_pred = set()
    matched_conf = set()
    name_matches = {}  # confirmed_name → predicted_name

    for cn, c_orig in conf_norm.items():
        # Exact normalized match
        if cn in pred_norm:
            matched_pred.add(cn)
            matched_conf.add(cn)
            name_matches[c_orig] = pred_norm[cn]
            continue
        # Fuzzy match
        best_ratio = 0
        best_pn = None
        for pn, p_orig in pred_norm.items():
            if pn in matched_pred:
                continue
            ratio = difflib.SequenceMatcher(None, cn, pn).ratio()
            # Also try last-name matching
            c_last = cn.split()[-1] if cn.split() else ""
            p_last = pn.split()[-1] if pn.split() else ""
            last_ratio = difflib.SequenceMatcher(None, c_last, p_last).ratio()
            effective = max(ratio, last_ratio * 0.9)
            if effective > best_ratio:
                best_ratio = effective
                best_pn = pn
        if best_pn and best_ratio >= 0.75:
            matched_pred.add(best_pn)
            matched_conf.add(cn)
            name_matches[c_orig] = pred_norm[best_pn]

    overlap = len(matched_conf)

    # Players in confirmed but NOT predicted (surprises)
    surprise_in = [conf_norm[cn] for cn in conf_norm if cn not in matched_conf]
    # Players predicted but NOT confirmed (dropped/rested)
    surprise_out = [pred_norm[pn] for pn in pred_norm if pn not in matched_pred]

    # Compute squad strength delta if xG data available
    strength_delta = None
    pred_xg = 0.0
    conf_xg = 0.0
    key_changes = []

    if player_xg_map:
        xg_norm = {_norm(k): v for k, v in player_xg_map.items()}

        for name in predicted_names:
            pred_xg += xg_norm.get(_norm(name), 0)
        for name in confirmed_names:
            conf_xg += xg_norm.get(_norm(name), 0)

        if pred_xg > 0:
            strength_delta = round((conf_xg - pred_xg) / pred_xg * 100, 1)

        # Identify key changes (players with highest xG impact)
        for name in surprise_out:
            xg = xg_norm.get(_norm(name), 0)
            if xg > 0.2:  # notable xG contributor
                key_changes.append({
                    "player": name,
                    "direction": "out",
                    "xg_per90": round(xg, 2),
                })
        for name in surprise_in:
            xg = xg_norm.get(_norm(name), 0)
            if xg > 0.2:
                key_changes.append({
                    "player": name,
                    "direction": "in",
                    "xg_per90": round(xg, 2),
                })
        key_changes.sort(key=lambda c: c["xg_per90"], reverse=True)

    # Verdict
    changes = 11 - overlap
    if changes == 0:
        verdict = "as_predicted"
        verdict_text = "Lineup matches prediction exactly"
    elif changes <= 2 and (strength_delta is None or abs(strength_delta) < 5):
        verdict = "minor_changes"
        verdict_text = f"{changes} change{'s' if changes > 1 else ''}, minimal impact"
    elif strength_delta is not None and strength_delta < -10:
        verdict = "weaker"
        verdict_text = f"{changes} changes, significantly weaker ({strength_delta:+.0f}%)"
    elif strength_delta is not None and strength_delta > 10:
        verdict = "stronger"
        verdict_text = f"{changes} changes, stronger than expected ({strength_delta:+.0f}%)"
    elif changes >= 4:
        verdict = "major_rotation"
        verdict_text = f"{changes} changes — heavy rotation, prediction less reliable"
    else:
        verdict = "moderate_changes"
        verdict_text = f"{changes} changes"
        if strength_delta is not None:
            verdict_text += f" ({strength_delta:+.0f}% strength)"

    return {
        "team": team,
        "overlap": overlap,
        "changes": changes,
        "surprise_in": surprise_in,
        "surprise_out": surprise_out,
        "predicted_xi_xg90": round(pred_xg, 2),
        "confirmed_xi_xg90": round(conf_xg, 2),
        "strength_delta_pct": strength_delta,
        "key_changes": key_changes,
        "verdict": verdict,
        "verdict_text": verdict_text,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lineup prediction and evaluation")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate past predictions against actual lineups")
    parser.add_argument("--predict", action="store_true",
                        help="Generate lineup predictions for upcoming matches")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_past_predictions()
    elif args.predict:
        generate_lineup_predictions()
    else:
        # Default: generate predictions
        generate_lineup_predictions()
