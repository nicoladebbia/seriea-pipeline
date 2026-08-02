"""Transfer impact analysis module.

Goes beyond simple transfer counts to measure ACTUAL impact:
- Pre vs post transfer window performance comparison
- New signing integration curves (position-specific)
- Squad disruption from key departures
- January window feature generation

Data sources:
- scraper/transfermarkt.py → data/external/transfermarkt/
- data/parsed/matches.parquet → pre/post performance
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

# Position-specific integration curves
# Tuple = effectiveness at game 1, 3, 5, 7+
#
# ⚠️ MEASURED WORTHLESS 2026-08-01 — do not tune these, and do not add more.
# These 32 numbers were invented, never validated, and exist only to produce
# `{home,away}_signing_integration` via compute_integration_curve().  Two
# independent measurements:
#   (1) no resolution — over 7,980 matches the output takes THREE distinct
#       values (1.00 / 0.30 / 0.65) with 93.3% of rows at 1.00, so the
#       position-specificity never actually surfaces;
#   (2) no signal — in all three live Serie A models it is rank 62/68 at
#       0.0-0.1% importance (CatBoost 0.101, LightGBM 2.0, XGBoost 0.000).
# The columns are now excluded from training in features/build.py
# (get_ml_feature_columns).  DELETE this dict, compute_integration_curve, and
# the `signing_integration` output once no live model still lists the columns in
# its feature_names — i.e. after the next full retrain.
#
# Its sibling compute_squad_disruption is NOT affected: same module, same window
# logic, but ranks 10/68, 24/68 and 19/68 in those same models.  Keep it.
INTEGRATION_CURVES = {
    "GK": (0.50, 0.80, 0.95, 1.0),   # GK integrates fastest
    "CB": (0.40, 0.70, 0.90, 1.0),
    "DF": (0.40, 0.70, 0.90, 1.0),
    "FB": (0.35, 0.65, 0.85, 1.0),
    "LB": (0.35, 0.65, 0.85, 1.0),
    "RB": (0.35, 0.65, 0.85, 1.0),
    "DM": (0.30, 0.60, 0.85, 1.0),
    "CM": (0.30, 0.60, 0.85, 1.0),
    "MF": (0.30, 0.60, 0.85, 1.0),
    "AM": (0.30, 0.55, 0.80, 1.0),
    "LW": (0.30, 0.55, 0.80, 1.0),
    "RW": (0.30, 0.55, 0.80, 1.0),
    "FW": (0.30, 0.55, 0.80, 1.0),    # FW takes longest
    "CF": (0.30, 0.55, 0.80, 1.0),
}


def analyze_transfer_impact(
    team: str,
    transfer_date: str,
    matches_df: pd.DataFrame,
    window_days: int = 30,
) -> dict:
    """Compare team performance pre vs post transfer window.

    Args:
        team: Team name
        transfer_date: ISO date string (e.g., "2026-01-15")
        matches_df: DataFrame with match data (must have 'date', team cols, scores)
        window_days: Days before/after to compare

    Returns:
        Dict with performance diff fields and confidence indicator.
    """
    try:
        td = pd.Timestamp(transfer_date)
    except Exception:
        return {"impact": 0.0, "confidence": "none", "error": "invalid_date"}

    # Normalize date column
    if "match_date" in matches_df.columns:
        date_col = "match_date"
    elif "date" in matches_df.columns:
        date_col = "date"
    else:
        return {"impact": 0.0, "confidence": "none", "error": "no_date_column"}

    df = matches_df.copy()
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Get team's matches
    team_mask = (
        (df.get("home_team", pd.Series(dtype=str)).str.lower() == team.lower()) |
        (df.get("away_team", pd.Series(dtype=str)).str.lower() == team.lower())
    )
    team_df = df[team_mask].copy()

    if team_df.empty:
        return {"impact": 0.0, "confidence": "none", "error": "no_matches"}

    # Pre-window matches
    pre_start = td - timedelta(days=window_days)
    pre = team_df[(team_df["_date"] >= pre_start) & (team_df["_date"] < td)]

    # Post-window matches
    post_end = td + timedelta(days=window_days)
    post = team_df[(team_df["_date"] > td) & (team_df["_date"] <= post_end)]

    if len(pre) < 3 or len(post) < 3:
        return {"impact": 0.0, "confidence": "low", "reason": "insufficient_data"}

    # Compute metrics for each window
    def compute_metrics(subset, team_name):
        goals = []
        xg = []
        points = []
        clean_sheets = 0

        for _, row in subset.iterrows():
            is_home = str(row.get("home_team", "")).lower() == team_name.lower()
            if is_home:
                g = pd.to_numeric(row.get("home_score"), errors="coerce")
                gc = pd.to_numeric(row.get("away_score"), errors="coerce")
                x = pd.to_numeric(row.get("home_xg"), errors="coerce")
            else:
                g = pd.to_numeric(row.get("away_score"), errors="coerce")
                gc = pd.to_numeric(row.get("home_score"), errors="coerce")
                x = pd.to_numeric(row.get("away_xg"), errors="coerce")

            if pd.notna(g):
                goals.append(g)
            if pd.notna(x):
                xg.append(x)
            if pd.notna(g) and pd.notna(gc):
                if g > gc:
                    points.append(3)
                elif g == gc:
                    points.append(1)
                else:
                    points.append(0)
                if gc == 0:
                    clean_sheets += 1

        n = max(len(goals), 1)
        return {
            "goals_per_game": sum(goals) / n if goals else 0,
            "xg_per_game": sum(xg) / max(len(xg), 1) if xg else 0,
            "points_per_game": sum(points) / max(len(points), 1) if points else 0,
            "clean_sheet_pct": clean_sheets / n if goals else 0,
            "matches": n,
        }

    pre_metrics = compute_metrics(pre, team)
    post_metrics = compute_metrics(post, team)

    return {
        "impact": round(post_metrics["points_per_game"] - pre_metrics["points_per_game"], 3),
        "goals_diff": round(post_metrics["goals_per_game"] - pre_metrics["goals_per_game"], 3),
        "xg_diff": round(post_metrics["xg_per_game"] - pre_metrics["xg_per_game"], 3),
        "points_diff": round(post_metrics["points_per_game"] - pre_metrics["points_per_game"], 3),
        "clean_sheet_diff": round(
            post_metrics["clean_sheet_pct"] - pre_metrics["clean_sheet_pct"], 3
        ),
        "pre_matches": pre_metrics["matches"],
        "post_matches": post_metrics["matches"],
        "confidence": "high" if len(pre) >= 5 and len(post) >= 5 else "medium",
    }


def compute_integration_curve(games_since_signing: int, position: str = "MF") -> float:
    """New signing effectiveness over time (position-specific).

    GK integrates fastest (0.50 at game 1), FW takes longest (0.30 at game 1).
    """
    if games_since_signing <= 0:
        return 0.30

    curve = INTEGRATION_CURVES.get(position, INTEGRATION_CURVES["MF"])

    if games_since_signing <= 2:
        return curve[0]
    elif games_since_signing <= 4:
        return curve[1]
    elif games_since_signing <= 6:
        return curve[2]
    else:
        return curve[3]


def compute_squad_disruption(
    departures_df: pd.DataFrame,
    team: str,
    season: str,
) -> float:
    """Measure disruption from key player departures.

    Args:
        departures_df: Transfer DataFrame with transfer_type, team, minutes columns
        team: Team name
        season: Season string

    Returns:
        disruption score 0-1
    """
    if departures_df.empty:
        return 0.0

    team_col = "team" if "team" in departures_df.columns else "squad"
    team_departures = departures_df[
        (departures_df[team_col].str.lower() == team.lower()) &
        (departures_df.get("transfer_type", pd.Series(dtype=str)) == "out")
    ]

    if team_departures.empty:
        return 0.0

    disruption = 0.0
    for _, dep in team_departures.iterrows():
        minutes = pd.to_numeric(dep.get("minutes_played", 0), errors="coerce")
        if pd.isna(minutes):
            minutes = 0

        if minutes > 1500:
            disruption += 0.25  # Key player (starter level)
        elif minutes > 500:
            disruption += 0.10  # Regular rotation player
        else:
            disruption += 0.02  # Minor departure

    return min(disruption, 1.0)


def compute_january_window_features(
    team: str,
    transfers_df: pd.DataFrame,
    matches_df: pd.DataFrame = None,
) -> dict:
    """Compute January window impact features for a team.

    Returns dict with:
    - jan_arrivals: count of January signings
    - jan_spend: January spending (EUR)
    - squad_disruption: key departure penalty (0-1)
    - signing_integration: average integration level of new signings (0.3-1.0)
    """
    if transfers_df.empty:
        return {
            "jan_arrivals": 0,
            "jan_spend": 0,
            "squad_disruption": 0.0,
            "signing_integration": 1.0,
        }

    team_col = "team" if "team" in transfers_df.columns else "squad"

    # Filter to this team's transfers
    team_transfers = transfers_df[
        transfers_df[team_col].str.lower() == team.lower()
    ].copy()

    if team_transfers.empty:
        return {
            "jan_arrivals": 0,
            "jan_spend": 0,
            "squad_disruption": 0.0,
            "signing_integration": 1.0,
        }

    # Filter to the WINTER (January) window via the scraped `window` tag. The
    # transfers parquet has NO date column, so the old `else: assume all January`
    # fallback ingested the WHOLE season as January — a within-row temporal leak
    # (a September match got January signings). Filter on `window` instead:
    # keep only winter. Untagged legacy rows default to summer → excluded (0), so
    # a pre-backfill file yields no jan_* signal rather than re-leaking the season.
    if "window" in team_transfers.columns:
        winter_mask = team_transfers["window"].fillna("summer") == "winter"
        jan_transfers = team_transfers[winter_mask]
    else:
        # no window tag (file predates the backfill) → no winter signal, no leak
        jan_transfers = team_transfers.iloc[0:0]

    # phantom "End of loan" OUTs for bought-back loanees → drop from disruption
    phantom_outs = _loan_to_permanent_outs(jan_transfers)

    # Arrivals
    type_col = "transfer_type" if "transfer_type" in jan_transfers.columns else "direction"
    arrivals = jan_transfers[jan_transfers.get(type_col, pd.Series(dtype=str)) == "in"]
    departures = jan_transfers[jan_transfers.get(type_col, pd.Series(dtype=str)) == "out"]

    # Spend
    fee_col = "fee_eur" if "fee_eur" in arrivals.columns else "fee"
    spend = pd.to_numeric(arrivals.get(fee_col, pd.Series(dtype=float)), errors="coerce").sum()
    if pd.isna(spend):
        spend = 0

    # Squad disruption from departures
    disruption = 0.0
    for _, dep in departures.iterrows():
        # a bought-back loanee's "End of loan" OUT is a phantom departure — skip
        dep_cat, _ = _transfer_materiality(dep.get("fee_text"), bool(dep.get("is_loan")))
        if dep_cat == "loan_return" and _normalize_name(dep.get("player_name", "")) in phantom_outs:
            continue
        minutes = pd.to_numeric(dep.get("minutes_played", 0), errors="coerce")
        if pd.isna(minutes):
            minutes = 0
        if minutes > 1500:
            disruption += 0.25
        elif minutes > 500:
            disruption += 0.10
        else:
            disruption += 0.02
    disruption = min(disruption, 1.0)

    # Average integration level of new signings
    # Estimate games since signing based on current date vs Jan window
    if matches_df is not None and not arrivals.empty:
        # Count team matches since Jan 15 (mid-window estimate)
        jan_mid = pd.Timestamp(f"{datetime.now().year}-01-15")
        if "match_date" in matches_df.columns:
            m_dates = pd.to_datetime(matches_df["match_date"], errors="coerce")
            team_mask = (
                (matches_df.get("home_team", pd.Series(dtype=str)).str.lower() == team.lower()) |
                (matches_df.get("away_team", pd.Series(dtype=str)).str.lower() == team.lower())
            )
            games_since = ((m_dates > jan_mid) & team_mask).sum()
        else:
            games_since = 3  # Default estimate

        integration_scores = []
        for _, arr in arrivals.iterrows():
            pos = arr.get("position", "MF")
            integration_scores.append(compute_integration_curve(games_since, pos))

        avg_integration = sum(integration_scores) / max(len(integration_scores), 1)
    else:
        avg_integration = 0.65  # Mid-range default

    return {
        "jan_arrivals": len(arrivals),
        "jan_spend": int(spend),
        "squad_disruption": round(disruption, 3),
        "signing_integration": round(avg_integration, 3),
    }


def _build_squad_value_lookup(
    tm_dir: Path, file_prefix: str = ""
) -> dict[tuple[str, str], dict]:
    """Build a lookup of squad market value features per (team, season).

    Uses market_values_YYYY_YYYY.parquet files from Transfermarkt.
    Each file has: team, player_name, position, age, market_value_eur, nationality.

    Args:
        tm_dir: Directory containing Transfermarkt parquet files
        file_prefix: League prefix for files (e.g. "premier_league_"). Empty for Serie A.

    Returns dict mapping (team_lower, season) -> {squad_value, squad_avg_value,
    squad_avg_age, squad_depth}.
    """
    lookup: dict[tuple[str, str], dict] = {}

    glob_pattern = f"{file_prefix}market_values_*.parquet"
    for path in sorted(tm_dir.glob(glob_pattern)):
        if "backup" in path.name:
            continue
        # Extract season from filename: market_values_2024_2025.parquet -> 2024-2025
        # or premier_league_market_values_2024_2025.parquet -> 2024-2025
        stem = path.stem
        if file_prefix:
            stem = stem.replace(file_prefix, "", 1)
        parts = stem.replace("market_values_", "").split("_")
        if len(parts) != 2:
            continue
        season = f"{parts[0]}-{parts[1]}"

        try:
            mv = pd.read_parquet(path)
        except Exception:
            continue

        if mv.empty or "team" not in mv.columns or "market_value_eur" not in mv.columns:
            continue

        mv["market_value_eur"] = pd.to_numeric(mv["market_value_eur"], errors="coerce")
        mv["age"] = pd.to_numeric(mv.get("age", pd.Series(dtype=float)), errors="coerce")

        # Filter to realistic ages (source data has noise: ages 1, 99, etc.)
        mv.loc[(mv["age"] < 16) | (mv["age"] > 42), "age"] = np.nan

        # Filter to players with non-zero market value (removes duplicate/noise rows)
        mv_valid = mv[mv["market_value_eur"] > 0].copy()

        for team, grp in mv_valid.groupby("team"):
            vals = grp["market_value_eur"].dropna()
            ages = grp["age"].dropna()
            team_key = team.lower().strip()

            lookup[(team_key, season)] = {
                "squad_value": int(vals.sum()) if len(vals) > 0 else 0,
                "squad_avg_value": int(vals.mean()) if len(vals) > 0 else 0,
                "squad_avg_age": round(float(ages.mean()), 1) if len(ages) > 0 else 0.0,
                "squad_depth": len(grp),
            }

    log.info("Built squad value lookup: %d team-seasons", len(lookup))
    return lookup


# --- Net squad delta (2026-27 window feature) -------------------------------

# Materiality of a transfer to the squad for the upcoming season. End-of-loan
# returns are re-integrations, not fresh squad changes, so they count at a
# reduced weight (and only when the player isn't ALSO leaving this window — see
# the double-count guard in compute_net_squad_delta).
_LOAN_RETURN_WEIGHT = 0.3   # user decision 2026-07-14: include returns, reduced
_MATERIAL_WEIGHT = 1.0      # paid, free, and fresh loan moves


def _transfer_materiality(fee_text: str, is_loan: bool) -> tuple[str, float]:
    """Classify a transfer row → (category, materiality_weight).

    Categories: loan_return (a player coming back from / going out on the
    *return* leg), loan_move (a fresh loan for the season), paid, free.
    Only loan_return is discounted; every other move is a real squad change.
    """
    ft = str(fee_text or "").strip().lower()
    if "end of loan" in ft:
        return "loan_return", _LOAN_RETURN_WEIGHT
    if is_loan or "loan" in ft:
        return "loan_move", _MATERIAL_WEIGHT
    return ("free" if ft in ("-", "", "free transfer", "?", "nan") else "paid",
            _MATERIAL_WEIGHT)


def _loan_to_permanent_outs(team_transfers: pd.DataFrame) -> set[str]:
    """Normalized names of players whose OUT row is a PHANTOM 'End of loan'.

    A player loaned in season N-1 then bought permanently in the summer gets BOTH
    an "End of loan" OUT row (returning to the parent club) AND a real paid/free
    IN row (the permanent purchase). The OUT is a phantom departure — the club
    KEPT the player. This returns those names so both the net_squad_delta and the
    squad_disruption departure loops can drop them (a bought-back loanee is an
    arrival, not a departure).

    Only the OUT that is specifically an "End of loan" AND paired with a
    non-loan-return IN for the same player is dropped — a genuine sale still
    counts, and a player who leaves on a fresh loan still counts.
    """
    tc = "transfer_type" if "transfer_type" in team_transfers.columns else "direction"
    if tc not in team_transfers.columns:
        return set()
    # players with a REAL (non-return) arrival this club/window
    real_in: set[str] = set()
    end_of_loan_out: set[str] = set()
    for _, row in team_transfers.iterrows():
        cat, _ = _transfer_materiality(row.get("fee_text"), bool(row.get("is_loan")))
        name = _normalize_name(row.get("player_name", ""))
        if row.get(tc) == "in" and cat != "loan_return":
            real_in.add(name)
        elif row.get(tc) == "out" and cat == "loan_return":
            end_of_loan_out.add(name)
    return end_of_loan_out & real_in


def _player_importance_lookup(
    season: str, minutes_path: Path | None = None
) -> dict[str, float]:
    """Per-player on-pitch importance for a season, keyed by normalized name.

    importance = minutes_share (share of a full season of minutes) blended with
    normalized average rating — "how heavy the player was in the team". Used to
    weight departures by how central the player actually was, not just his fee.
    Returns {} if the source is unavailable (feature degrades to value-only).
    """
    path = minutes_path or (
        DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
    )
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(
            path, columns=["season", "player_name", "minutes", "rating"]
        )
    except Exception as e:  # noqa: BLE001 — importance is optional; degrade to value-only
        log.warning("player importance parquet unreadable (%s)", e)
        return {}
    df = df[df["season"].astype(str) == str(season)]
    if df.empty:
        return {}
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    agg = df.groupby("player_name").agg(
        total_minutes=("minutes", "sum"), avg_rating=("rating", "mean")
    )
    if agg.empty:
        return {}
    # minutes_share: relative to the busiest player in the league that season
    max_min = agg["total_minutes"].max() or 1.0
    minutes_share = (agg["total_minutes"] / max_min).clip(0, 1)
    # rating normalized to [0,1] over the observed 6.0–8.0 band (SA typical)
    rating_norm = ((agg["avg_rating"] - 6.0) / 2.0).clip(0, 1).fillna(0.3)
    importance = (0.6 * minutes_share + 0.4 * rating_norm)
    return {
        _normalize_name(name): float(v)
        for name, v in importance.items()
    }


def _normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation for cross-source name joins."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace(".", " ").split())


def compute_net_squad_delta(
    season: str = "2026-2027",
    league: str = "serie_a",
    tm_dir: Path | None = None,
) -> dict[str, dict]:
    """Per-club net squad delta from the season's confirmed transfers.

    weight(player) blends market value (talent) with last-season on-pitch
    importance (centrality). net_delta = Σ weight(material arrivals) −
    Σ weight(material departures), with end-of-loan returns discounted and a
    guard against double-counting a loanee who returns AND then leaves.

    Confirmed transfers ONLY — rumors never reach this path.

    Returns {team_lower: {net_squad_delta, arrivals_weight, departures_weight,
    material_in, material_out}}.
    """
    tm_dir = tm_dir or (DATA_DIR / "external" / "transfermarkt")
    prefix = "" if league == "serie_a" else f"{league}_"
    tpath = tm_dir / f"{prefix}transfers_{season.replace('-', '_')}.parquet"
    if not tpath.exists():
        log.warning("No transfers parquet at %s — net_squad_delta empty", tpath)
        return {}
    tf = pd.read_parquet(tpath)
    if tf.empty or "team" not in tf.columns:
        return {}

    # Leak-free by construction: exclude WINTER-window transfers. This is a
    # PRE-SEASON squad delta — the net talent present at match 1, applied to every
    # match of the season. A summer signing is legitimately on the roster at match
    # 38; a January signing must NOT retroactively inflate that season's August
    # matches (training leak). Written as exclude-winter WITH A DEFAULT, never
    # `== "summer"`: legacy files scraped before the window tag existed have no
    # `window` column, and `.get("summer")` defaulting to "summer" keeps all their
    # rows rather than silently zeroing the whole season.
    if "window" in tf.columns:
        tf = tf[tf["window"].fillna("summer") != "winter"]
        if tf.empty:
            return {}

    # last completed season's on-pitch importance (2026-27 window ⇒ 2025-2026)
    start = int(season.split("-")[0])
    prev_season = f"{start - 1}-{start}"
    importance = _player_importance_lookup(prev_season)

    # market-value lookup (talent proxy). MUST be the season-matched file, not
    # the latest one: using the newest valuations for a historical season leaks
    # future information (a 2019 transfer weighted by 2026 prices) and fabricates
    # every historical delta — a walk-forward violation. Prefer the season's own
    # file; if it is missing (older seasons), fall back to the nearest-PRIOR
    # season's file, never a future one. For the live 2026-27 season this resolves
    # to market_values_2026_2027 exactly, so the live signal is unaffected.
    mv_lookup: dict[str, float] = {}
    season_key = season.replace("-", "_")
    exact = tm_dir / f"{prefix}market_values_{season_key}.parquet"
    if exact.exists():
        mv_path: Path | None = exact
    else:
        # nearest-prior: largest start-year <= this season's start year
        candidates = []
        for p in tm_dir.glob(f"{prefix}market_values_*.parquet"):
            m = re.search(r"market_values_(\d{4})_\d{4}", p.name)
            if m and int(m.group(1)) <= start:
                candidates.append((int(m.group(1)), p))
        mv_path = max(candidates)[1] if candidates else None
    if mv_path is not None:
        try:
            mv = pd.read_parquet(mv_path)
            mv["market_value_eur"] = pd.to_numeric(
                mv.get("market_value_eur"), errors="coerce"
            )
            vmax = mv["market_value_eur"].max() or 1.0
            for _, r in mv.iterrows():
                v = r["market_value_eur"]
                if pd.notna(v) and v > 0:
                    mv_lookup[_normalize_name(r["player_name"])] = float(v / vmax)
        except Exception as e:  # noqa: BLE001 — value-only fallback is acceptable
            log.warning("market-value lookup failed (%s) — using importance-only weights", e)

    def player_weight(name: str) -> float:
        key = _normalize_name(name)
        imp = importance.get(key)               # 0..1 centrality (may be None)
        val = mv_lookup.get(key)                # 0..1 talent (may be None)
        if imp is not None and val is not None:
            return 0.5 * imp + 0.5 * val
        if val is not None:                     # new signing: value-only
            return val
        if imp is not None:                     # in our data but no TM value
            return imp
        return 0.15                             # unknown player: small floor

    # double-count guard: a player who both returns from loan AND leaves this
    # window is only counted on the departure side.
    out_names = {
        _normalize_name(n)
        for n in tf.loc[tf["transfer_type"] == "out", "player_name"]
    }

    result: dict[str, dict] = {}
    for team, grp in tf.groupby("team"):
        arrivals_w = departures_w = 0.0
        n_in = n_out = 0
        # phantom "End of loan" OUTs for players the club actually bought → drop
        # them from the departure side (a bought-back loanee is not a departure).
        phantom_outs = _loan_to_permanent_outs(grp)
        for _, row in grp.iterrows():
            cat, mat = _transfer_materiality(
                row.get("fee_text"), bool(row.get("is_loan"))
            )
            w = player_weight(row["player_name"]) * mat
            direction = row["transfer_type"]
            name = _normalize_name(row["player_name"])
            if direction == "in":
                # skip a loan-return that is also leaving this window
                if cat == "loan_return" and name in out_names:
                    continue
                arrivals_w += w
                n_in += 1
            elif direction == "out":
                # skip a phantom "End of loan" OUT for a bought-back loanee
                if cat == "loan_return" and name in phantom_outs:
                    continue
                departures_w += w
                n_out += 1
        result[team.lower().strip()] = {
            "net_squad_delta": round(arrivals_w - departures_w, 4),
            "arrivals_weight": round(arrivals_w, 4),
            "departures_weight": round(departures_w, 4),
            "material_in": n_in,
            "material_out": n_out,
        }
    log.info("Computed net_squad_delta for %d clubs (%s)", len(result), season)
    return result


def add_transfer_impact_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add transfer impact features to the match-level DataFrame.

    Adds 16 columns:
    - home_jan_arrivals / away_jan_arrivals
    - home_jan_spend / away_jan_spend
    - home_squad_disruption / away_squad_disruption
    - home_signing_integration / away_signing_integration
    - home_squad_value / away_squad_value
    - home_squad_avg_age / away_squad_avg_age
    - squad_value_diff / squad_value_ratio
    """
    df = feature_df.copy()

    # Initialize with defaults
    for prefix in ("home", "away"):
        df[f"{prefix}_jan_arrivals"] = 0
        df[f"{prefix}_jan_spend"] = 0
        df[f"{prefix}_squad_disruption"] = 0.0
        df[f"{prefix}_signing_integration"] = 1.0
        df[f"{prefix}_squad_value"] = np.nan
        df[f"{prefix}_squad_avg_age"] = np.nan
        df[f"{prefix}_net_squad_delta"] = 0.0

    # Load transfers
    tm_dir = DATA_DIR / "external" / "transfermarkt"
    if not tm_dir.exists():
        log.info("No Transfermarkt data available — transfer features set to defaults")
        return df

    # Determine league prefix for file lookup (Serie A = no prefix for backward compat)
    league_key = ""
    if "league" in df.columns:
        leagues = df["league"].dropna().unique()
        if len(leagues) == 1:
            league_key = str(leagues[0])
    file_prefix = "" if league_key in ("serie_a", "") else f"{league_key}_"

    # ── Squad market value features ──────────────────────────────────
    squad_lookup = _build_squad_value_lookup(tm_dir, file_prefix=file_prefix)
    if squad_lookup:
        for idx, row in df.iterrows():
            season = row.get("season", "")
            for prefix in ("home", "away"):
                team = str(row.get(f"{prefix}_team", "")).lower().strip()
                sv = squad_lookup.get((team, season))
                if sv:
                    df.at[idx, f"{prefix}_squad_value"] = sv["squad_value"]
                    df.at[idx, f"{prefix}_squad_avg_age"] = sv["squad_avg_age"]

        # Relative squad value features (vectorized)
        h_val = pd.to_numeric(df["home_squad_value"], errors="coerce")
        a_val = pd.to_numeric(df["away_squad_value"], errors="coerce")
        df["squad_value_diff"] = h_val - a_val
        df["squad_value_ratio"] = h_val / a_val.replace(0, np.nan)

        n_sv = df["home_squad_value"].notna().sum()
        log.info("Squad value features: %d/%d matches with data (%.0f%%)",
                 n_sv, len(df), 100 * n_sv / max(len(df), 1))

    # ── January transfer window features ─────────────────────────────
    # Load matches for integration calculation
    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    matches_df = pd.read_parquet(matches_path) if matches_path.exists() else None

    # Process per season
    seasons = df["season"].unique() if "season" in df.columns else []
    transfer_cache: dict[tuple[str, str], dict] = {}

    for season in seasons:
        tr_path = tm_dir / f"{file_prefix}transfers_{season.replace('-', '_')}.parquet"
        if not tr_path.exists():
            continue

        try:
            transfers = pd.read_parquet(tr_path)
        except Exception:
            continue

        season_mask = df["season"] == season
        teams = set(
            df.loc[season_mask, "home_team"].dropna().unique().tolist() +
            df.loc[season_mask, "away_team"].dropna().unique().tolist()
        )

        for team in teams:
            cache_key = (team, season)
            if cache_key not in transfer_cache:
                transfer_cache[cache_key] = compute_january_window_features(
                    team, transfers, matches_df
                )

    # Apply cached features
    for idx, row in df.iterrows():
        season = row.get("season", "")
        for prefix in ("home", "away"):
            team = row.get(f"{prefix}_team", "")
            features = transfer_cache.get((team, season), {})
            if features:
                df.at[idx, f"{prefix}_jan_arrivals"] = features.get("jan_arrivals", 0)
                df.at[idx, f"{prefix}_jan_spend"] = features.get("jan_spend", 0)
                df.at[idx, f"{prefix}_squad_disruption"] = features.get("squad_disruption", 0)
                df.at[idx, f"{prefix}_signing_integration"] = features.get(
                    "signing_integration", 1.0
                )

    n_with_data = (df["home_jan_arrivals"] + df["away_jan_arrivals"] > 0).sum()
    log.info(f"Added transfer impact features ({n_with_data} matches with transfer data, "
             f"{len(squad_lookup)} team-seasons with squad value)")

    # ── Net squad delta (materiality- and importance-weighted) ───────────
    # Per-club net talent gained/lost from that season's confirmed transfers,
    # applied to every match of the season. Confirmed transfers only.
    league_arg = league_key if league_key in ("serie_a", "premier_league") else "serie_a"
    delta_cache: dict[str, dict[str, dict]] = {}
    for season in seasons:
        try:
            delta_cache[season] = compute_net_squad_delta(
                season=season, league=league_arg, tm_dir=tm_dir
            )
        except Exception as e:  # noqa: BLE001 — feature must never break the pipeline
            log.warning("net_squad_delta failed for %s: %s", season, e)
            delta_cache[season] = {}

    if any(delta_cache.values()):
        for idx, row in df.iterrows():
            season = row.get("season", "")
            per_club = delta_cache.get(season, {})
            if not per_club:
                continue
            for prefix in ("home", "away"):
                team = str(row.get(f"{prefix}_team", "")).lower().strip()
                d = per_club.get(team)
                if d:
                    df.at[idx, f"{prefix}_net_squad_delta"] = d["net_squad_delta"]
        df["net_squad_delta_diff"] = (
            df["home_net_squad_delta"] - df["away_net_squad_delta"]
        )
        n_delta = (df["home_net_squad_delta"] != 0).sum()
        log.info("Net squad delta: %d/%d matches with home-side data", n_delta, len(df))
    else:
        df["net_squad_delta_diff"] = 0.0

    return df


def validate_january_transfer_predictions() -> dict:
    """Compare predicted transfer impact vs actual for validation.

    Loads January transfers and compares pre/post team performance.
    Returns correlation and bias metrics.
    """
    tm_dir = DATA_DIR / "external" / "transfermarkt"
    matches_path = DATA_DIR / "parsed" / "matches.parquet"

    if not matches_path.exists():
        return {"error": "No matches data for validation"}

    matches_df = pd.read_parquet(matches_path)

    # Find most recent season with transfer data
    results = []
    for season in ["2024-2025", "2023-2024"]:
        tr_path = tm_dir / f"transfers_{season.replace('-', '_')}.parquet"
        if not tr_path.exists():
            continue

        transfers = pd.read_parquet(tr_path)
        if transfers.empty:
            continue

        # Get teams that made January signings
        team_col = "team" if "team" in transfers.columns else "squad"
        teams = transfers[team_col].unique()

        year = int(season.split("-")[1])
        jan_date = f"{year}-01-15"

        for team in teams:
            impact = analyze_transfer_impact(team, jan_date, matches_df)
            if impact.get("confidence") in ("high", "medium"):
                results.append({
                    "team": team,
                    "season": season,
                    **impact,
                })

    if not results:
        return {"error": "No sufficient data for validation", "results": []}

    # Compute summary statistics
    impacts = [r.get("impact", 0) for r in results]
    return {
        "total_analyzed": len(results),
        "avg_impact": round(np.mean(impacts), 3) if impacts else 0,
        "positive_impact_pct": round(sum(1 for i in impacts if i > 0) / max(len(impacts), 1), 3),
        "results": results,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing transfer impact analysis...")
    print("=" * 60)

    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    if matches_path.exists():
        matches = pd.read_parquet(matches_path)
        print(f"Loaded {len(matches)} matches")

        impact = analyze_transfer_impact("Napoli", "2025-01-15", matches)
        print(f"\nNapoli January 2025 impact: {impact}")

    print(f"\nIntegration curve (FW, game 1): {compute_integration_curve(1, 'FW')}")
    print(f"Integration curve (FW, game 5): {compute_integration_curve(5, 'FW')}")
    print(f"Integration curve (GK, game 1): {compute_integration_curve(1, 'GK')}")

    print("\n" + "=" * 60)
