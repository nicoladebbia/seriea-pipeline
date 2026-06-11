"""World Cup 2026 player-availability layer (team news -> prediction input).

Answers, per fixture inside the news horizon: who is expected to start
("people think he plays" — recent competitive starts), who IS starting
(Sofascore confirmed XI ~1h pre-kickoff), who is out or doubtful (Sofascore
missing-player list with reasons), and how much that team news should move
the model lambdas.

The lambda adjustment is mechanical, not fitted (same honesty class as
players.py's LINEUP_* constants): absence impact = market-value share of the
expected XI that is unavailable, position-weighted (a missing striker hits
the team's own lambda; a missing keeper/centre-back bumps the opponent's),
scaled by ALPHA and clamped to FACTOR_BOUNDS. generate_predictions.py applies
it to the MODEL lambdas BEFORE the market blend — bookmakers already price
team news, so adjusting pre-blend improves the model leg without double
counting the market leg.

--study replays the construction on the scraped 12-month window (competitive
internationals only — friendlies rotate too much to carry signal) and writes
the absence-share vs goals-shortfall regression to availability_study.json.
That is directional evidence for ALPHA's sign and scale, NOT a fitted
parameter: n is small and Transfermarkt values are end-of-window. Read the
study file, never quote numbers from this docstring.

Run:  python3 -m scripts.worldcup.availability            (writes player_availability.json)
      python3 -m scripts.worldcup.availability --study    (writes availability_study.json)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from scripts.worldcup.engine import DATA_DIR, canon_team
from scripts.worldcup.players import (
    CONFIRMED_LINEUPS_JSON,
    RECENT_WINDOW,
    SOFA_PARQUET,
    SQUADS_JSON,
    TM_JSON,
    load_sofa,
    load_squads,
    norm_sorted,
)
from scripts.worldcup.simulate import load_fixtures
from scripts.worldcup.sofascore_fetch import sofa_canon

AVAILABILITY_JSON = DATA_DIR / "player_availability.json"
STUDY_JSON = DATA_DIR / "availability_study.json"

# How far ahead a fixture gets an expected-XI / team-news entry. Sofascore
# only exposes missing-player lists alongside lineups (~48h out); beyond
# that the entry is a projected XI with zero lambda impact.
HORIZON_DAYS = 7

# Position weight on each lambda side. A forward's absence is ~pure attack;
# a keeper's ~pure defense; midfielders split. Mechanical priors, not fitted.
ATTACK_W = {"G": 0.0, "D": 0.2, "M": 0.6, "F": 1.0}
DEFENSE_W = {"G": 1.0, "D": 1.0, "M": 0.4, "F": 0.1}

# Lambda sensitivity per unit of position-weighted value share out, and the
# hard clamp on any single factor. A typical one-starter absence (~10-15%
# share) moves lambda 3-5%, in line with how books move on a confirmed star
# absence. The --study regression (availability_study.json) confirms the
# SIGN but its point estimate is noisy and rotation-diluted (squad absences
# in qualifiers include resting, which a WC opener does not) — so ALPHA sits
# below the mechanical prior (0.45) and above the diluted estimate (~0.14).
ALPHA = 0.35
FACTOR_BOUNDS = (0.85, 1.15)

# Presence deficit per status: a confirmed-bench expected starter still
# plays ~LINEUP_BENCH minutes, an out player plays none, a doubtful one is
# a coin flip. Mirrors players.py's mechanical minute expectations.
DEFICIT_OUT = 1.0
DEFICIT_DOUBTFUL = 0.5
DEFICIT_BENCHED = 0.65


def pos_letter(raw: str) -> str:
    """Map any source's position spelling to G/D/M/F ('' if unknown).

    squads.json uses GK/DF/MF/FW, Transfermarkt 'Goalkeeper'/'Centre-Back'/...,
    Sofascore single letters. Defenders/midfield/forward keywords first so
    'Defensive Midfield' lands on M, not D.
    """
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    if s[0] == "g" or "keeper" in s:
        return "G"
    if "midfield" in s or s in {"m", "mf", "cm", "dm", "am"}:
        return "M"
    if "back" in s or "defen" in s or s in {"d", "df", "cb", "lb", "rb"}:
        return "D"
    if "wing" in s or "striker" in s or "forward" in s or s in {"f", "fw", "cf", "st"}:
        return "F"
    return ""


def akey(name: str) -> str:
    """Availability-layer name key: norm_sorted with hyphens treated as
    spaces — Transfermarkt and Sofascore disagree on 'Min-jae'/'Minjae' and
    on East-Asian name order; token-sorting the dehyphenated form matches
    both. Used for EVERY cross-source join in this module."""
    return norm_sorted(name.replace("-", " "))


def tm_value_index(tm_path: Any = TM_JSON) -> dict[str, dict[str, float]]:
    """{canon team -> {akey player -> value_eur}} from the scrape."""
    if not tm_path.exists():
        return {}
    raw = json.loads(tm_path.read_text())
    out: dict[str, dict[str, float]] = {}
    for team, blob in raw.items():
        if team.startswith("_") or not isinstance(blob, dict):
            continue
        vals: dict[str, float] = {}
        for p in blob.get("players", []):
            v = p.get("value_eur")
            if v:
                vals[akey(str(p["name"]))] = float(v)
        # TM uses FIFA spellings (Korea Republic, ...) — sofa_canon covers them
        out[sofa_canon(team)] = vals
    return out


def expected_xi(
    team_df: pd.DataFrame, before: pd.Timestamp | None = None
) -> list[dict[str, Any]]:
    """Projected XI: top-11 by starts (then minutes) over the team's last
    RECENT_WINDOW matches — the 'people expect him to play' list. Returns
    [{key, name, position, starts, recent_matches}] sorted strongest first."""
    sub = team_df if before is None else team_df[team_df["date"] < before]
    if sub.empty:
        return []
    recent_dates = sorted(sub["date"].unique())[-RECENT_WINDOW:]
    recent = sub[sub["date"].isin(recent_dates)].copy()
    recent["_akey"] = recent["player_name"].map(akey)
    n_matches = len(recent_dates)
    rows: list[dict[str, Any]] = []
    for key, grp in recent.groupby("_akey"):
        rows.append(
            {
                "key": str(key),
                "name": str(grp["player_name"].iloc[-1]),
                "position": pos_letter(str(grp["position"].iloc[-1])),
                "starts": int(grp["started"].sum()),
                "minutes": float(grp["minutes"].fillna(0).sum()),
                "recent_matches": n_matches,
            }
        )
    rows.sort(key=lambda r: (-r["starts"], -r["minutes"]))
    xi = rows[:11]
    for r in xi:
        r.pop("minutes", None)
    return xi


def absence_impact(
    xi: list[dict[str, Any]],
    deficits: dict[str, float],
    values: dict[str, float],
) -> dict[str, Any]:
    """Position-weighted value share of the expected XI that is unavailable,
    and the lambda factors it implies.

    deficits: {player key -> presence deficit in [0,1]} for XI members only
    (absences outside the expected XI are news, not lambda input — they were
    not expected to start, so zero impact is the conservative call).
    """
    if not xi:
        return {
            "attack_share_out": 0.0,
            "defense_share_out": 0.0,
            "lambda_factor_self": 1.0,
            "lambda_factor_opp": 1.0,
        }
    # Unmatched-value fallback: median of the matched XI values, so a missing
    # Transfermarkt row neither zeroes a star nor inflates a reserve.
    matched = [values[p["key"]] for p in xi if p["key"] in values]
    fallback = sorted(matched)[len(matched) // 2] if matched else 1.0
    atk_total = def_total = atk_out = def_out = 0.0
    for p in xi:
        v = values.get(p["key"], fallback)
        aw, dw = ATTACK_W.get(p["position"], 0.5), DEFENSE_W.get(p["position"], 0.5)
        atk_total += aw * v
        def_total += dw * v
        d = deficits.get(p["key"], 0.0)
        if d > 0:
            atk_out += d * aw * v
            def_out += d * dw * v
    attack_share = atk_out / atk_total if atk_total > 0 else 0.0
    defense_share = def_out / def_total if def_total > 0 else 0.0
    lo, hi = FACTOR_BOUNDS
    return {
        "attack_share_out": round(attack_share, 4),
        "defense_share_out": round(defense_share, 4),
        "lambda_factor_self": round(max(lo, 1.0 - ALPHA * attack_share), 4),
        "lambda_factor_opp": round(min(hi, 1.0 + ALPHA * defense_share), 4),
    }


def _side_report(
    team: str,
    team_df: pd.DataFrame,
    lineup_side: dict[str, Any] | None,
    lineup_confirmed: bool,
    values: dict[str, float],
    squad: list[dict[str, Any]],
) -> dict[str, Any]:
    """One team's availability block for one fixture."""
    xi = expected_xi(team_df)
    source = "projected"
    if not xi and squad:  # no scraped minutes at all — caps-order fallback
        ranked = sorted(squad, key=lambda p: -int(p.get("caps", 0) or 0))[:11]
        xi = [
            {
                "key": akey(str(p["name"])),
                "name": str(p["name"]),
                "position": pos_letter(str(p.get("position", ""))),
                "starts": 0,
                "recent_matches": 0,
            }
            for p in ranked
        ]
        source = "caps_fallback"

    deficits: dict[str, float] = {}
    out_list: list[dict[str, Any]] = []
    doubtful_list: list[dict[str, Any]] = []
    xi_keys = {p["key"] for p in xi}
    based_on = "none"

    for m in (lineup_side or {}).get("missing", []):
        key = akey(str(m["name"]))
        entry = {
            "name": str(m["name"]),
            "position": pos_letter(str(m.get("position", ""))),
            "reason": str(m.get("reason", "Other")),
            "value_eur": values.get(key),
            "in_expected_xi": key in xi_keys,
        }
        if str(m.get("type", "missing")) == "doubtful":
            doubtful_list.append(entry)
            deficits[key] = max(deficits.get(key, 0.0), DEFICIT_DOUBTFUL)
        else:
            out_list.append(entry)
            deficits[key] = DEFICIT_OUT
        based_on = "missing_list"

    if lineup_confirmed and lineup_side:
        # Confirmed XI is FACT: expected starters not in it are absent
        # (benched ~ partial deficit, out of squad ~ full), and the displayed
        # XI flips from projection to the published one.
        starters = {akey(n) for n in lineup_side.get("starters", [])}
        bench = {akey(n) for n in lineup_side.get("bench", [])}
        for p in xi:
            if p["key"] in starters:
                deficits.pop(p["key"], None)
            elif p["key"] in bench:
                # Confirmed teamsheet is FACT: a benched player is AVAILABLE,
                # so it overrides any earlier stale missing-list OUT marking
                # (mirrors the starters branch above).
                deficits[p["key"]] = DEFICIT_BENCHED
            else:
                deficits[p["key"]] = DEFICIT_OUT
        impact = absence_impact(xi, deficits, values)
        name_by_key = {p["key"]: p for p in xi}
        xi_display = [
            {
                "key": akey(n),
                "name": n,
                "position": name_by_key.get(akey(n), {}).get("position", ""),
                "starts": name_by_key.get(akey(n), {}).get("starts", 0),
                "recent_matches": name_by_key.get(akey(n), {}).get(
                    "recent_matches", 0
                ),
            }
            for n in lineup_side.get("starters", [])
        ]
        based_on = "confirmed_vs_expected"
        source = "confirmed"
    else:
        impact = absence_impact(xi, deficits, values)
        xi_display = xi

    for p in xi_display:
        p["value_eur"] = values.get(p["key"])
        p.pop("key", None)
    return {
        "team": team,
        "lineup_status": source,
        "xi": xi_display,
        "out": out_list,
        "doubtful": doubtful_list,
        "impact": {**impact, "based_on": based_on},
    }


def build(horizon_days: int = HORIZON_DAYS) -> dict[str, Any]:
    """Write player_availability.json for fixtures inside the horizon."""
    fixtures = load_fixtures()
    sofa = load_sofa() if SOFA_PARQUET.exists() else pd.DataFrame()
    if not sofa.empty:
        # Parquet keeps raw Sofascore names (Czechia, Korea Republic, ...)
        sofa["team"] = sofa["team"].map(sofa_canon)
    squads = load_squads() if SQUADS_JSON.exists() else {}
    values = tm_value_index()
    lineups: dict[str, Any] = {}
    if CONFIRMED_LINEUPS_JSON.exists():
        lineups = json.loads(CONFIRMED_LINEUPS_JSON.read_text())

    squad_by_canon = {canon_team(t): s for t, s in squads.items()}
    now = datetime.now(UTC)
    matches: dict[str, Any] = {}
    for f in fixtures:
        try:
            ko = datetime.fromisoformat(str(f["date_utc"]).replace("Z", "+00:00"))
        except ValueError:
            continue
        if not (now - timedelta(hours=3) <= ko <= now + timedelta(days=horizon_days)):
            continue
        lineup = lineups.get(str(f["match_number"])) or {}
        confirmed = bool(lineup.get("confirmed"))
        sides: dict[str, Any] = {}
        for side in ("home", "away"):
            team = canon_team(str(f[side]))
            if not sofa.empty and team in set(sofa["team"].unique()):
                team_df = sofa[sofa["team"] == team]
            else:
                team_df = pd.DataFrame(
                    columns=["date", "norm_name_sorted", "player_name", "position",
                             "started", "minutes"]
                )
            sides[side] = _side_report(
                team,
                team_df,
                lineup.get(side),
                confirmed,
                values.get(team, {}),
                squad_by_canon.get(team, []),
            )
        matches[str(f["match_number"])] = sides

    payload = {
        "generated_at": now.isoformat(),
        "alpha": ALPHA,
        "factor_bounds": list(FACTOR_BOUNDS),
        "horizon_days": horizon_days,
        "matches": matches,
    }
    AVAILABILITY_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    n_news = sum(
        1
        for m in matches.values()
        for s in m.values()
        if s["out"] or s["doubtful"] or s["impact"]["based_on"] != "none"
    )
    print(
        f"availability: {len(matches)} fixtures in horizon, "
        f"{n_news} sides with team news -> {AVAILABILITY_JSON}"
    )
    return payload


def study() -> dict[str, Any]:
    """Directional check of the absence signal on the scraped window.

    Competitive internationals only. For each team-match with >=3 prior
    matches: expected XI from prior matches; absent = expected starters with
    NO row that day (the parquet carries the full ~23-man matchday squad, so
    no row = out of squad); x = attack-weighted value share absent,
    y = goals minus the team's own mean goals in its other competitive
    matches (leave-one-out). OLS slope < 0 supports the adjustment's sign.
    Small n and end-of-window values — evidence, not a fit.
    """
    sofa = load_sofa()
    sofa["team"] = sofa["team"].map(sofa_canon)
    values = tm_value_index()
    comp = sofa[sofa["tournament"] != "Int. Friendly Games"]
    xs: list[float] = []
    ys: list[float] = []
    for team, tdf_all in comp.groupby("team"):
        tdf_full = sofa[sofa["team"] == team]  # priors may include friendlies
        team_vals = values.get(str(team), {})
        match_dates = sorted(tdf_all["date"].unique())
        goals_by_date = {
            d: float(tdf_all[tdf_all["date"] == d]["our_score"].iloc[0])
            for d in match_dates
        }
        for d in match_dates:
            prior = tdf_full[tdf_full["date"] < d]
            if prior["date"].nunique() < 3:
                continue
            xi = expected_xi(tdf_full, before=pd.Timestamp(d))
            if len(xi) < 11:
                continue
            day_keys = set(tdf_all[tdf_all["date"] == d]["player_name"].map(akey))
            deficits = {
                p["key"]: DEFICIT_OUT for p in xi if p["key"] not in day_keys
            }
            impact = absence_impact(xi, deficits, team_vals)
            others = [g for dd, g in goals_by_date.items() if dd != d]
            if not others:
                continue
            xs.append(impact["attack_share_out"])
            ys.append(goals_by_date[d] - sum(others) / len(others))
    n = len(xs)
    result: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_team_matches": n,
        "scope": "competitive internationals in the 12-month scrape window",
        "alpha_in_code": ALPHA,
    }
    if n >= 30:
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
        slope = sxy / sxx if sxx > 0 else 0.0
        resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys, strict=True)]
        se = (
            math.sqrt((sum(r * r for r in resid) / max(n - 2, 1)) / sxx)
            if sxx > 0
            else float("inf")
        )
        result.update(
            {
                "slope_goals_per_unit_share": round(slope, 3),
                "slope_se": round(se, 3),
                "mean_share_when_nonzero": round(
                    sum(x for x in xs if x > 0) / max(sum(1 for x in xs if x > 0), 1), 4
                ),
                "n_nonzero_share": sum(1 for x in xs if x > 0),
                "supports_sign": bool(slope < 0),
                "note": (
                    f"ALPHA*lambda implies ~{-ALPHA * 1.4:.2f} goals per unit "
                    "attack share; slope materially positive would argue for "
                    "ALPHA=0."
                ),
            }
        )
    else:
        result["note"] = "insufficient sample — adjustment stays mechanical"
    STUDY_JSON.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    args = parser.parse_args()
    if args.study:
        study()
    else:
        build(horizon_days=args.horizon_days)


if __name__ == "__main__":
    main()
