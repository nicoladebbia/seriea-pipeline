"""AI Advisor Blueprint — Claude-powered Serie A betting intelligence.

Provides:
- SSE streaming chat endpoint with tool use loop
- 8 data tools that inject pre-computed narratives
- Zero-cost greeting from local data files
- Prompt caching for 90% discount on repeated system prompt
- Cost tracking to data/api_usage.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, jsonify, render_template, request

log = logging.getLogger(__name__)

advisor_bp = Blueprint("advisor", __name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from config.settings import DATA_DIR, UPCOMING_DIR, BETTING_DIR, LIVE_DIR
from scripts.utils.json_utils import load_json_safe
USAGE_FILE = DATA_DIR / "api_usage.json"
# Project root. Six call sites (Sofascore player stats, match incidents, matches,
# features) build paths as `_BASE / "data" / ...`; a past "deduplicate path
# constants" commit dropped its definition, silently breaking those lookups
# (they fail-soft in try/except, so the per-match ratings/team-sheet degraded
# without surfacing). DATA_DIR is <root>/data, so _BASE is its parent.
_BASE = DATA_DIR.parent

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _get_bankroll() -> dict:
    """Bankroll view from ledger.get_metrics() — the one computation.

    Key names kept for existing callers. bankroll_growth_pct and
    roi_on_stake_pct are SEPARATE named numbers — never conflate them.
    """
    try:
        from scripts.betting.ledger import get_metrics
        m = get_metrics(include_alerts=False)
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        return {}
    mb, rec = m["bankroll"], m["record"]
    return {
        "current_bankroll": mb["current"],
        "initial_bankroll": mb["initial"],
        "peak_bankroll": mb["peak"],
        "lowest_bankroll": mb["lowest"],
        "available": mb["available"],
        "pending_stakes": mb["pending_stakes"],
        "drawdown_pct": mb["drawdown_pct"],
        "bankroll_growth_pct": mb["bankroll_growth_pct"],
        "roi_on_stake_pct": m["roi"]["all_time_pct"],
        "daily_pnl": m["periods"]["today"]["pnl"],
        "total_bets": rec["settled_n"],
        "total_wins": rec["won"],
        "current_streak": m["streak"]["streak_decisive"],
        "last_updated": m["meta"]["computed_at"],
        "source": "ledger.get_metrics()",
    }


def _resolve_team(query: str) -> str | None:
    """Fuzzy-resolve user input to canonical team name (Serie A + EPL)."""
    try:
        from config.team_names import (
            normalize_team, SERIE_A_2026_27, PREMIER_LEAGUE_2026_27,
        )
    except ImportError:
        return query

    all_teams = set(SERIE_A_2026_27) | set(PREMIER_LEAGUE_2026_27)

    # Exact match via normalize_team
    canonical = normalize_team(query)
    if canonical in all_teams:
        return canonical

    # Substring match
    q = query.lower().strip()
    for team in all_teams:
        if q in team.lower():
            return team
    # Common nicknames
    nicknames = {
        # Serie A
        "juve": "Juventus", "inter": "Inter", "milan": "Milan",
        "roma": "Roma", "lazio": "Lazio", "napoli": "Napoli",
        "viola": "Fiorentina", "toro": "Torino", "samp": "Sampdoria",
        "ata": "Atalanta", "dea": "Atalanta", "grifone": "Genoa",
        # EPL
        "city": "Man City", "united": "Man United", "spurs": "Tottenham",
        "toffees": "Everton", "gunners": "Arsenal", "reds": "Liverpool",
        "blues": "Chelsea", "hammers": "West Ham", "magpies": "Newcastle",
        "saints": "Southampton", "foxes": "Leicester", "bees": "Brentford",
        "seagulls": "Brighton", "cherries": "Bournemouth", "eagles": "Crystal Palace",
        "forest": "Nott'ham Forest", "cottagers": "Fulham", "villa": "Aston Villa",
        "lfc": "Liverpool", "mufc": "Man United", "mcfc": "Man City",
        "cfc": "Chelsea", "afc": "Arsenal", "thfc": "Tottenham",
    }
    if q in nicknames and nicknames[q] in all_teams:
        return nicknames[q]
    return None


def _find_match(home: str, away: str, predictions: list) -> dict | None:
    """Find a match prediction by home/away team names."""
    h = _resolve_team(home)
    a = _resolve_team(away)
    if not h or not a:
        return None
    for p in predictions:
        if p.get("home_team") == h and p.get("away_team") == a:
            return p
    return None


# ---------------------------------------------------------------------------
# Tool handlers — inject pre-computed narratives
# ---------------------------------------------------------------------------

def _tool_get_match_prediction(args: dict) -> str:
    home = args.get("home_team", "")
    away = args.get("away_team", "")

    # Fallback: if Claude passed a single "match" string, parse it
    if not home and not away:
        match_str = args.get("match", "")
        if " vs " in match_str:
            parts = match_str.split(" vs ", 1)
            home, away = parts[0].strip(), parts[1].strip()
        elif " - " in match_str:
            parts = match_str.split(" - ", 1)
            home, away = parts[0].strip(), parts[1].strip()

    preds_data = load_json_safe(UPCOMING_DIR / "predictions.json")
    predictions = preds_data.get("predictions", [])
    for p in predictions:
        p.setdefault("league", "serie_a")

    # Merge predictions from all active leagues (e.g. EPL)
    for extra_league in ["premier_league"]:
        extra_path = UPCOMING_DIR / f"predictions_{extra_league}.json"
        extra_raw = load_json_safe(extra_path)
        if extra_raw:
            extra_list = extra_raw.get("predictions", []) if isinstance(extra_raw, dict) else extra_raw
            for p in extra_list:
                p.setdefault("league", extra_league)
            predictions.extend(extra_list)

    match = _find_match(home, away, predictions)

    # If not found, try reversed (user might say "Pisa vs Como" when it's "Como vs Pisa")
    if not match and home and away:
        match = _find_match(away, home, predictions)

    if not match:
        # List available matches to help Claude self-correct
        available = [p.get("match", "") for p in predictions[:10]]
        return json.dumps({
            "error": f"No prediction found for {home} vs {away}.",
            "available_matches": available,
            "hint": "Try using the exact match name from the available list. Format is 'Home vs Away'."
        })

    # Compute clear ensemble-average probabilities for Claude
    comp = match.get("component_predictions", {})
    ens_probs = []
    for method_data in comp.values():
        if isinstance(method_data, dict) and "prob_H" in method_data:
            h, d, a = method_data["prob_H"], method_data["prob_D"], method_data["prob_A"]
            if h > 0 and d > 0:
                ens_probs.append((h, d, a))
    if ens_probs:
        avg_h = round(sum(p[0] for p in ens_probs) / len(ens_probs), 3)
        avg_d = round(sum(p[1] for p in ens_probs) / len(ens_probs), 3)
        avg_a = round(sum(p[2] for p in ens_probs) / len(ens_probs), 3)
    else:
        avg_h = avg_d = avg_a = 0

    result = {
        "match": match.get("match"),
        "home_team": match.get("home_team"),
        "away_team": match.get("away_team"),
        "predicted_outcome": match.get("predicted_outcome"),
        # Clear, unambiguous probabilities — Claude MUST use these exact numbers
        "home_win_probability": avg_h,
        "draw_probability": avg_d,
        "away_win_probability": avg_a,
        "_note": f"Probabilities: {match.get('home_team')} {avg_h*100:.1f}%, Draw {avg_d*100:.1f}%, {match.get('away_team')} {avg_a*100:.1f}%. Quote these EXACT numbers.",
        "probabilities": match.get("probabilities"),
        "confidence_level": match.get("confidence_level"),
        "home_xg": match.get("home_xg"),
        "away_xg": match.get("away_xg"),
        "component_methods": comp,
        "strategy": match.get("strategy"),
    }

    # Inject betting info
    slip = load_json_safe(UPCOMING_DIR / "unified_bet_slip.json")
    h_canon = _resolve_team(home)
    a_canon = _resolve_team(away)
    match_bets = [
        b for b in slip.get("selected_bets", [])
        if h_canon and a_canon and h_canon in b.get("match", "") and a_canon in b.get("match", "")
    ]
    if match_bets:
        result["value_bets"] = [
            {"market": b["market"], "selection": b["selection"],
             "odds": b.get("best_odds"), "edge_pct": b.get("edge_pct"),
             "bookmaker": b.get("best_bookmaker")}
            for b in match_bets
        ]

    # Inject player analysis narrative
    pa = load_json_safe(UPCOMING_DIR / "player_analysis.json")
    for m in pa.get("matches", []):
        if m.get("home_team") == h_canon and m.get("away_team") == a_canon:
            result["player_analysis"] = m.get("analysis_summary", "")
            result["home_strength"] = m.get("home_strength")
            result["away_strength"] = m.get("away_strength")
            result["key_factors"] = m.get("key_factors", [])
            break

    # Inject sentiment narrative
    sa = load_json_safe(UPCOMING_DIR / "sentiment_analysis.json")
    for m in sa.get("matches", []):
        if m.get("home_team") == h_canon and m.get("away_team") == a_canon:
            result["sentiment"] = m.get("analysis", m.get("summary", ""))
            break

    # Inject goal predictions (detailed)
    goals = load_json_safe(UPCOMING_DIR / "goal_predictions.json")
    match_str = f"{h_canon} vs {a_canon}"
    if isinstance(goals, dict):
        for key, m in (goals.get("predictions", {}).items() if isinstance(goals.get("predictions"), dict) else enumerate(goals.get("predictions", []))):
            m_data = m if isinstance(m, dict) else {}
            if m_data.get("home_team") == h_canon or match_str in str(m_data.get("match", "")):
                result["goal_predictions"] = {
                    "expected_home_goals": m_data.get("expected_home_goals"),
                    "expected_away_goals": m_data.get("expected_away_goals"),
                    "expected_total": m_data.get("expected_total_goals"),
                    "over_1_5": m_data.get("over_1_5"),
                    "over_2_5": m_data.get("over_2_5"),
                    "over_3_5": m_data.get("over_3_5"),
                    "factors": m_data.get("factors", []),
                }
                break

    # Inject BTTS predictions
    btts = load_json_safe(UPCOMING_DIR / "btts_predictions.json", default=[])
    if isinstance(btts, list):
        for m in btts:
            if h_canon in m.get("match", "") and a_canon in m.get("match", ""):
                result["btts"] = {
                    "btts_yes": m.get("btts_yes"),
                    "btts_no": m.get("btts_no"),
                    "expected_home_goals": m.get("expected_home_goals"),
                    "expected_away_goals": m.get("expected_away_goals"),
                }
                break

    # Inject corners predictions
    corners = load_json_safe(UPCOMING_DIR / "corners_predictions.json", default=[])
    if isinstance(corners, list):
        for m in corners:
            if h_canon in m.get("match", "") and a_canon in m.get("match", ""):
                result["corners"] = {
                    "expected_corners": m.get("expected_corners"),
                    "expected_home": m.get("expected_home_corners"),
                    "expected_away": m.get("expected_away_corners"),
                    "over_8_5": m.get("over_8_5"),
                    "over_9_5": m.get("over_9_5"),
                    "over_10_5": m.get("over_10_5"),
                }
                break

    # Inject cards predictions
    cards = load_json_safe(UPCOMING_DIR / "cards_predictions.json", default=[])
    if isinstance(cards, list):
        for m in cards:
            if h_canon in m.get("match", "") and a_canon in m.get("match", ""):
                result["cards"] = {
                    "expected_cards": m.get("expected_cards"),
                    "expected_home": m.get("expected_home_cards"),
                    "expected_away": m.get("expected_away_cards"),
                    "over_3_5": m.get("over_3_5"),
                    "over_4_5": m.get("over_4_5"),
                }
                break

    # Inject margin predictions
    margins = load_json_safe(UPCOMING_DIR / "margin_predictions.json")
    if isinstance(margins, dict):
        for key, m in (margins.get("predictions", {}).items() if isinstance(margins.get("predictions"), dict) else enumerate(margins.get("predictions", []))):
            m_data = m if isinstance(m, dict) else {}
            if m_data.get("home_team") == h_canon or match_str in str(m_data.get("match", "")):
                result["margin"] = {
                    "expected_margin": m_data.get("expected_margin"),
                    "home_rating": m_data.get("home_rating"),
                    "away_rating": m_data.get("away_rating"),
                    "handicap_probs": m_data.get("handicap_probs"),
                    "factors": m_data.get("factors", []),
                }
                break

    # Inject lineups (confirmed or predicted)
    confirmed = load_json_safe(UPCOMING_DIR / "confirmed_lineups.json")
    conf_matches = confirmed.get("matches", {})
    lineup_found = False
    for key, lu in conf_matches.items():
        if h_canon in key and a_canon in key:
            home_lu = lu.get("home_lineup", [])
            away_lu = lu.get("away_lineup", [])
            result["lineups"] = {
                "source": "confirmed",
                "home_formation": lu.get("home_formation"),
                "home_lineup": home_lu[:11] if isinstance(home_lu, list) else home_lu,
                "away_formation": lu.get("away_formation"),
                "away_lineup": away_lu[:11] if isinstance(away_lu, list) else away_lu,
                "home_bench": lu.get("home_bench", []),
                "away_bench": lu.get("away_bench", []),
            }
            lineup_found = True
            break

    if not lineup_found:
        predicted = load_json_safe(UPCOMING_DIR / "lineup_predictions.json")
        for key, lu in predicted.get("matches", {}).items():
            if h_canon in key and a_canon in key:
                home_data = lu.get("home_lineup", {})
                away_data = lu.get("away_lineup", {})
                if isinstance(home_data, dict):
                    result["lineups"] = {
                        "source": "predicted",
                        "home_formation": home_data.get("formation"),
                        "home_lineup": home_data.get("predicted_xi", []),
                        "home_unavailable": home_data.get("unavailable", []),
                        "away_formation": away_data.get("formation") if isinstance(away_data, dict) else None,
                        "away_lineup": away_data.get("predicted_xi", []) if isinstance(away_data, dict) else away_data,
                        "away_unavailable": away_data.get("unavailable", []) if isinstance(away_data, dict) else [],
                    }
                elif isinstance(home_data, list):
                    result["lineups"] = {
                        "source": "predicted",
                        "home_lineup": home_data[:11],
                        "away_lineup": away_data[:11] if isinstance(away_data, list) else away_data,
                    }
                break

    # Inject bookmaker analysis (sharp vs soft consensus)
    bk = load_json_safe(UPCOMING_DIR / "bookmaker_analysis.json")
    bk_matches = bk.get("matches", [])
    if isinstance(bk_matches, list):
        for m in bk_matches:
            if h_canon in m.get("match", "") and a_canon in m.get("match", ""):
                result["bookmaker_analysis"] = {
                    "sharp_consensus": m.get("sharp_consensus"),
                    "soft_consensus": m.get("soft_consensus"),
                    "divergence": m.get("divergence"),
                    "sharp_direction": m.get("sharp_direction"),
                    "outliers": m.get("outliers"),
                }
                break

    return json.dumps(result, default=str)


def _tool_get_team_detail(args: dict) -> str:
    team_q = args.get("team", "")
    team = _resolve_team(team_q)
    if not team:
        return json.dumps({"error": f"Team '{team_q}' not found."})

    result: dict[str, Any] = {"team": team}

    # Standings
    standings = load_json_safe(UPCOMING_DIR / "standings.json")
    team_standing = standings.get("standings", {}).get(team)
    if team_standing:
        result["standings"] = team_standing

    # Form
    form_data = load_json_safe(UPCOMING_DIR / "current_form.json")
    team_form = form_data.get("teams", {}).get(team)
    if team_form:
        result["form"] = team_form

    # Top scorers from parquet
    try:
        import pandas as pd
        df = pd.read_parquet(DATA_DIR / "parsed" / "player_stats.parquet")
        current = df[df["season"] == "2025-2026"] if "season" in df.columns else df[df["match_date"] >= "2025-08-01"]
        team_players = current[current["team"] == team]
        if not team_players.empty:
            agg = team_players.groupby("player").agg(
                goals=("goals", "sum"),
                assists=("assists", "sum"),
                minutes=("minutes", "sum"),
                matches=("player", "count"),
            )
            top_scorers = agg.sort_values("goals", ascending=False).head(5)
            result["top_scorers"] = [
                {"player": name, "goals": int(row["goals"]), "assists": int(row["assists"]),
                 "minutes": int(row["minutes"]), "matches": int(row["matches"])}
                for name, row in top_scorers.iterrows()
            ]
            top_assists = agg.sort_values("assists", ascending=False).head(3)
            result["top_assisters"] = [
                {"player": name, "assists": int(row["assists"]), "goals": int(row["goals"])}
                for name, row in top_assists.iterrows()
            ]
    except Exception as e:
        log.warning("Team top scorers lookup failed: %s", e)

    # Recent results for this team
    results_data = load_json_safe(UPCOMING_DIR / "results.json")
    results_all = results_data.get("results", {})
    team_results = []
    if isinstance(results_all, dict):
        for key, r in results_all.items():
            if r.get("home_team") == team or r.get("away_team") == team:
                team_results.append({
                    "match": r.get("match"),
                    "score": f"{r.get('home_score')}-{r.get('away_score')}",
                    "result": r.get("result"),
                    "date": (r.get("commence_time", "") or "")[:10],
                })
    if team_results:
        result["recent_results"] = team_results

    return json.dumps(result, default=str)


def _normalize_name(name: str) -> str:
    """Strip accents and normalize for fuzzy player name matching."""
    import unicodedata
    # Manual mapping for chars that NFKD doesn't decompose (Turkish, etc.)
    _EXTRA = str.maketrans("ıİşŞğĞçÇöÖüÜ", "iIssgGcCoOuU")
    nfkd = unicodedata.normalize("NFKD", name.translate(_EXTRA))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def _player_name_match(series, query: str):
    """Match player names with accent-insensitive, multi-word search."""
    import pandas as pd
    query_norm = _normalize_name(query)
    words = query_norm.split()

    # Normalize the entire series once
    normalized = series.apply(lambda x: _normalize_name(str(x)) if pd.notna(x) else "")

    # All words must appear in the normalized name
    mask = pd.Series(True, index=series.index)
    for word in words:
        mask = mask & normalized.str.contains(word, na=False)
    return mask


def played_bracket_context(res, national_team, next_match_date) -> str:
    """Describe who the winner of the team's next match would face, asserting ONLY
    resolved results. If the other same-round tie is unplayed, name it as unresolved
    ('the France/Spain winner') — never state an outcome the results file hasn't
    recorded. Guards against the user being ahead of our data (they may know Spain
    won before international_results.csv does)."""
    import pandas as pd
    try:
        nd = pd.to_datetime(next_match_date)
        # other matches within ±1 day of this fixture = same knockout round
        same_round = res[
            (res["date"] >= nd - pd.Timedelta(days=1))
            & (res["date"] <= nd + pd.Timedelta(days=1))
        ]
        others = same_round[
            (same_round["home_team"] != national_team)
            & (same_round["away_team"] != national_team)
            & same_round["home_team"].notna()
        ]
        if others.empty:
            return ""
        o = others.iloc[0]
        h, a = o["home_team"], o["away_team"]
        if pd.notna(o.get("home_score")) and pd.notna(o.get("away_score")):
            # resolved — name the actual winner
            hs, as_ = int(o["home_score"]), int(o["away_score"])
            winner = h if hs > as_ else (a if as_ > hs else None)
            if winner:
                return (
                    f"The other same-round tie ({h} vs {a}) is RESOLVED: {winner} won "
                    f"({hs}-{as_}), so the winner of this match would face {winner} next."
                )
            return f"The other same-round tie {h} vs {a} finished level ({hs}-{as_}, decided on/after)."
        # unresolved — do NOT assert a winner even if the user claims one
        return (
            f"The other same-round tie ({h} vs {a}) has NOT been played/recorded in our "
            f"data yet, so the winner of this match would face the {h}/{a} winner — do not "
            f"state which of them advances unless our results file shows the score."
        )
    except Exception:  # noqa: BLE001
        return ""


def _tool_get_player_stats(args: dict) -> str:
    player_q = args.get("player", "").lower().strip()
    team_q = args.get("team", "").strip()
    if not player_q:
        return json.dumps({"error": "Please provide a player name."})

    result: dict[str, Any] = {"query": args.get("player")}

    # 1. Season stats from player_stats.parquet (141 columns, per-match)
    try:
        import pandas as pd
        df = pd.read_parquet(DATA_DIR / "parsed" / "player_stats.parquet")
        # Current season
        current = df[df["season"] == "2025-2026"] if "season" in df.columns else df[df["match_date"] >= "2025-08-01"]

        # Fuzzy search by player name (accent-insensitive, multi-word)
        mask = _player_name_match(current["player"], player_q)
        if team_q:
            team_resolved = _resolve_team(team_q)
            if team_resolved:
                mask = mask & (current["team"] == team_resolved)

        player_rows = current[mask]

        if not player_rows.empty:
            # Get canonical player name (most common match)
            player_name = player_rows["player"].mode().iloc[0]
            team_name = player_rows["team"].mode().iloc[0]
            player_rows = player_rows[player_rows["player"] == player_name]

            total_matches = len(player_rows)
            total_minutes = player_rows["minutes"].sum()
            per90 = total_minutes / 90 if total_minutes > 0 else 1

            season_stats = {
                "player": player_name,
                "team": team_name,
                "position": player_rows["position"].mode().iloc[0] if "position" in player_rows.columns and not player_rows["position"].isna().all() else "?",
                "matches": total_matches,
                "minutes": int(total_minutes),
                "goals": int(player_rows["goals"].sum()),
                "assists": int(player_rows["assists"].sum()),
                # FBref xG in the parsed parquet is often broken (sums to 0.0 even
                # for a striker with goals+shots). A 0.0 next to real shots is a
                # DATA GAP, not reality — null it so the model uses Understat's real
                # xG (surfaced in understat_history) instead of quoting a false 0.0.
                "xg": (
                    round(player_rows["xg"].sum(), 2)
                    if "xg" in player_rows.columns and player_rows["xg"].sum() > 0
                    else None
                ),
                "xa": (
                    round(player_rows["xg_assist"].sum(), 2)
                    if "xg_assist" in player_rows.columns and player_rows["xg_assist"].sum() > 0
                    else None
                ),
                "shots": int(player_rows["shots"].sum()) if "shots" in player_rows.columns else None,
                "shots_on_target": int(player_rows["shots_on_target"].sum()) if "shots_on_target" in player_rows.columns else None,
                "yellow_cards": int(player_rows["cards_yellow"].sum()) if "cards_yellow" in player_rows.columns else None,
                "red_cards": int(player_rows["cards_red"].sum()) if "cards_red" in player_rows.columns else None,
                "tackles_won": int(player_rows["tackles_won"].sum()) if "tackles_won" in player_rows.columns else None,
                "interceptions": int(player_rows["interceptions"].sum()) if "interceptions" in player_rows.columns else None,
                "goals_per_90": round(player_rows["goals"].sum() / per90, 2),
                "assists_per_90": round(player_rows["assists"].sum() / per90, 2),
            }
            result["season_stats"] = season_stats

            # Recent form (last 5 matches)
            recent = player_rows.sort_values("match_date", ascending=False).head(5)
            result["recent_form"] = [
                {
                    "date": str(r["match_date"])[:10],
                    "minutes": int(r["minutes"]),
                    "goals": int(r["goals"]),
                    "assists": int(r["assists"]),
                    "shots": int(r.get("shots", 0)),
                    "xg": round(r.get("xg", 0), 2),
                }
                for _, r in recent.iterrows()
            ]

            # Compare to teammates — is this the best player?
            team_players = current[current["team"] == team_name]
            team_agg = team_players.groupby("player").agg(
                goals=("goals", "sum"),
                assists=("assists", "sum"),
                minutes=("minutes", "sum"),
            ).sort_values("goals", ascending=False)
            rank_goals = list(team_agg.index).index(player_name) + 1 if player_name in team_agg.index else None
            team_agg_assists = team_agg.sort_values("assists", ascending=False)
            rank_assists = list(team_agg_assists.index).index(player_name) + 1 if player_name in team_agg_assists.index else None
            result["team_ranking"] = {
                "goals_rank": rank_goals,
                "assists_rank": rank_assists,
                "total_players": len(team_agg),
                "team_top_scorer": team_agg.index[0],
                "team_top_scorer_goals": int(team_agg.iloc[0]["goals"]),
            }
    except Exception as e:
        log.warning("Player parquet lookup failed: %s", e)

    # 2. Per-match detailed performance from Sofascore (80 columns, includes today)
    try:
        import pandas as pd
        sof_path = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
        if sof_path.exists():
            sof = pd.read_parquet(sof_path)
            sof_mask = _player_name_match(sof["player_name"], player_q)
            if team_q:
                team_resolved = _resolve_team(team_q)
                if team_resolved:
                    sof_mask = sof_mask & (sof["team"] == team_resolved)
            sof_rows = sof[sof_mask]
            if not sof_rows.empty:
                player_name_sof = sof_rows["player_name"].mode().iloc[0]
                sof_rows = sof_rows[sof_rows["player_name"] == player_name_sof]

                # Last 5 detailed match performances
                recent_sof = sof_rows.sort_values("date", ascending=False).head(5)
                # Pre-render recent form as markdown table to prevent hallucination
                form_header = "| Date | Opponent | Min | Rating | G | A | xG | Shots | SoT | KeyP |"
                form_sep = "|------|----------|-----|--------|---|---|-----|-------|-----|------|"
                form_rows = [form_header, form_sep]
                for _, r in recent_sof.iterrows():
                    rating = round(float(r.get("rating", 0)), 1) if pd.notna(r.get("rating")) else "-"
                    xg = round(float(r.get("xg", 0)), 2) if pd.notna(r.get("xg")) else "-"
                    venue = "H" if r.get("is_home") else "A"
                    form_rows.append(
                        f"| {str(r.get('date', ''))[:10]} | {r.get('opponent', '')} ({venue}) "
                        f"| {int(r.get('minutes', 0))} | **{rating}** "
                        f"| {int(r.get('goals', 0))} | {int(r.get('assists', 0))} "
                        f"| {xg} | {int(r.get('total_shots', 0))} "
                        f"| {int(r.get('shots_on_target', 0))} | {int(r.get('key_passes', 0))} |"
                    )
                latest_rating = round(float(recent_sof.iloc[0].get("rating", 0)), 1) if pd.notna(recent_sof.iloc[0].get("rating")) else "?"
                result["match_performances"] = (
                    f"EXACT DATA — DO NOT MODIFY THESE NUMBERS.\n"
                    f"Latest match rating: {latest_rating}\n\n"
                    + "\n".join(form_rows)
                )

                # Season-aggregate advanced Sofascore stats (2025-26 only). sof_rows
                # spans ALL seasons (2017→2026), so scope to the current season before
                # summing or a 30-year-old's career totals would leak in. These are the
                # role-defining numbers the tool didn't surface: passing volume, defensive
                # work rate, duels/aerials, creativity (big chances), and GK saves — all
                # 100%-filled columns, verified. Per-90 where volume matters.
                try:
                    cur_sof = sof_rows[sof_rows["date"] >= "2025-08-01"] if "date" in sof_rows.columns else sof_rows
                    if not cur_sof.empty:
                        smin = float(cur_sof["minutes"].sum())
                        p90 = smin / 90 if smin > 0 else 1

                        def _ssum(col):
                            return float(cur_sof[col].sum()) if col in cur_sof.columns else 0.0

                        def _savg(col):
                            return round(float(cur_sof[col].mean()), 2) if col in cur_sof.columns and cur_sof[col].notna().any() else None

                        acc_p = _ssum("accurate_passes")
                        tot_p = _ssum("total_passes")
                        d_won = _ssum("duels_won")
                        d_lost = _ssum("duels_lost")
                        acc_lb = _ssum("accurate_long_balls")
                        tot_lb = _ssum("total_long_balls")
                        adv = {
                            "matches": int(len(cur_sof)),
                            "minutes": int(smin),
                            "avg_rating": _savg("rating"),
                            # passing / involvement
                            "accurate_passes_total": int(acc_p),
                            "passes_per_90": round(acc_p / p90, 1),
                            "pass_accuracy_pct": round(100 * acc_p / tot_p, 1) if tot_p > 0 else None,
                            "long_ball_accuracy_pct": round(100 * acc_lb / tot_lb, 1) if tot_lb > 0 else None,
                            "touches_per_90": round(_ssum("touches") / p90, 1),
                            "ball_recoveries_per_90": round(_ssum("ball_recoveries") / p90, 2),
                            "possession_lost_per_90": round(_ssum("possession_lost") / p90, 1),
                            # creativity
                            "big_chances_created": int(_ssum("big_chances_created")),
                            "big_chances_missed": int(_ssum("big_chances_missed")),
                            "key_passes_per_90": round(_ssum("key_passes") / p90, 2),
                            # defensive work rate
                            "tackles": int(_ssum("tackles")),
                            "interceptions": int(_ssum("interceptions")),
                            "clearances": int(_ssum("clearances")),
                            "blocks": int(_ssum("blocks")),
                            "tackles_plus_int_per_90": round((_ssum("tackles") + _ssum("interceptions")) / p90, 2),
                            "errors_leading_to_goal": int(_ssum("error_to_goal")),
                            # duels
                            "duels_won": int(d_won),
                            "duel_win_pct": round(100 * d_won / (d_won + d_lost), 1) if (d_won + d_lost) > 0 else None,
                            "aerials_won": int(_ssum("aerial_won")),
                            # discipline
                            "fouls": int(_ssum("fouls")),
                            "was_fouled": int(_ssum("was_fouled")),
                        }
                        # Goalkeeper-only: saves (populated only for keepers — ~6% of rows).
                        saves = _ssum("saves")
                        if saves > 0:
                            adv["saves"] = int(saves)
                            adv["saves_per_90"] = round(saves / p90, 2)
                        adv["_note"] = (
                            "Season 2025-26 Sofascore aggregates. big_chances_missed vs "
                            "goals = finishing waste; tackles_plus_int_per_90 = defensive "
                            "work rate; pass_accuracy_pct + touches = involvement/security; "
                            "duel_win_pct = physical dominance; errors_leading_to_goal is a "
                            "hard red flag. Exact numbers — do not modify."
                        )
                        result["season_advanced_stats"] = adv
                except Exception as e:  # noqa: BLE001 — advanced block is enrichment; never blocks
                    log.warning("Advanced Sofascore aggregate failed: %s", e)

                # Most recent match: add full team sheet + match events
                latest = recent_sof.iloc[0]
                latest_match_id = latest.get("match_id")
                latest_team = latest.get("team", "")
                latest_date = str(latest.get("date", ""))[:10]

                if latest_match_id is not None:
                    # Full team sheet for the match
                    teammates = sof[
                        (sof["match_id"] == latest_match_id) &
                        (sof["team"] == latest_team)
                    ].sort_values("is_starter", ascending=False)

                    team_sheet = []
                    for _, tm in teammates.sort_values("rating", ascending=False).iterrows():
                        team_sheet.append({
                            "player": tm.get("player_name", ""),
                            "position": tm.get("position", ""),
                            "minutes": int(tm.get("minutes", 0)),
                            "starter": bool(tm.get("is_starter", False)),
                            "rating": round(float(tm.get("rating", 0)), 1) if pd.notna(tm.get("rating")) else None,
                            "goals": int(tm.get("goals", 0)),
                            "assists": int(tm.get("assists", 0)),
                            "xg": round(float(tm.get("xg", 0)), 2) if pd.notna(tm.get("xg")) else None,
                            "shots": int(tm.get("total_shots", 0)),
                            "key_passes": int(tm.get("key_passes", 0)),
                        })
                    # Don't send raw team_sheet JSON — only send pre-rendered table
                    # to prevent Haiku from fabricating different numbers

                    # Pre-render markdown table — Claude MUST copy verbatim
                    header = "| Player | Pos | Min | Rating | G | A | Shots | KeyP |"
                    sep = "|--------|-----|-----|--------|---|---|-------|------|"
                    rows = [header, sep]
                    for p in team_sheet:
                        r = p["rating"] if p["rating"] else "-"
                        rows.append(f"| {p['player']} | {p['position']} | {p['minutes']} | **{r}** | {p['goals']} | {p['assists']} | {p['shots']} | {p['key_passes']} |")
                    best = team_sheet[0] if team_sheet else None
                    best_note = f"Best rated: {best['player']} ({best['rating']})" if best else ""
                    result["latest_match_team_sheet"] = (
                        f"VERIFIED SOFASCORE DATA — COPY THIS TABLE EXACTLY, DO NOT CHANGE ANY NUMBERS:\n"
                        f"{best_note}\n\n"
                        + "\n".join(rows)
                    )

                    # Match events (goals, cards, subs)
                    inc_path = DATA_DIR / "external" / "sofascore" / "match_incidents.parquet"
                    if inc_path.exists():
                        try:
                            inc = pd.read_parquet(inc_path)
                            match_events = inc[(inc["match_id"] == latest_match_id)
                                               & (inc["incident_type"] != "var_checked")].sort_values("minute")  # backfill marker rows are not events
                            events = []
                            for _, ev in match_events.iterrows():
                                evt = {
                                    "minute": int(ev.get("minute", 0)),
                                    "type": ev.get("incident_type", ""),
                                    "detail": ev.get("incident_class", ""),
                                }
                                if ev.get("incident_type") == "goal":
                                    evt["scorer"] = ev.get("player_name", "")
                                    evt["assist"] = ev.get("assist_player", "")
                                    evt["goal_type"] = ev.get("goal_type", ev.get("incident_class", ""))
                                elif ev.get("incident_type") == "substitution":
                                    evt["player_in"] = ev.get("player_in_name", "")
                                    # Infer player_out from starters who left at this minute
                                    sub_min = int(ev.get("minute", 0))
                                    possible_out = teammates[
                                        (teammates["is_starter"] == True) &
                                        (teammates["minutes"] <= sub_min + 2) &
                                        (teammates["minutes"] >= sub_min - 2) &
                                        (teammates["minutes"] < 90)
                                    ]
                                    if not possible_out.empty:
                                        evt["player_out"] = possible_out.iloc[0].get("player_name", "")
                                elif ev.get("incident_type") == "card":
                                    evt["player"] = ev.get("player_name", "")
                                    evt["card"] = ev.get("card_type", ev.get("incident_class", ""))
                                events.append(evt)
                            if events:
                                result["latest_match_events"] = events
                        except Exception:
                            pass

                    # Player's usual position vs this match position
                    if "season_stats" in result or not sof_rows.empty:
                        usual_pos = sof_rows["position"].mode().iloc[0] if not sof_rows["position"].isna().all() else None
                        match_pos = latest.get("position", "")
                        if usual_pos and match_pos and usual_pos != match_pos:
                            result["position_change"] = {
                                "usual": usual_pos,
                                "today": match_pos,
                                "note": f"Played as {match_pos} instead of usual {usual_pos}",
                            }

                # If no season stats from FBref, compute from Sofascore
                if "season_stats" not in result:
                    current_sof = sof_rows[sof_rows["date"] >= "2025-08-01"]
                    if not current_sof.empty:
                        total_min = current_sof["minutes"].sum()
                        per90 = total_min / 90 if total_min > 0 else 1
                        result["season_stats"] = {
                            "player": player_name_sof,
                            "team": current_sof["team"].mode().iloc[0],
                            "position": current_sof["position"].mode().iloc[0] if not current_sof["position"].isna().all() else "?",
                            "matches": len(current_sof),
                            "minutes": int(total_min),
                            "starts": int(current_sof["is_starter"].sum()),
                            "goals": int(current_sof["goals"].sum()),
                            "assists": int(current_sof["assists"].sum()),
                            "xg": round(current_sof["xg"].sum(), 2) if "xg" in current_sof.columns else None,
                            "xa": round(current_sof["xa"].sum(), 2) if "xa" in current_sof.columns else None,
                            "shots": int(current_sof["total_shots"].sum()),
                            "shots_on_target": int(current_sof["shots_on_target"].sum()),
                            "key_passes": int(current_sof["key_passes"].sum()),
                            "avg_rating": round(current_sof["rating"].mean(), 2) if not current_sof["rating"].isna().all() else None,
                            "goals_per_90": round(current_sof["goals"].sum() / per90, 2),
                            "assists_per_90": round(current_sof["assists"].sum() / per90, 2),
                            "xg_per_90": round(current_sof["xg"].sum() / per90, 2) if "xg" in current_sof.columns else None,
                            "shots_per_90": round(current_sof["total_shots"].sum() / per90, 2),
                        }

                # --- Availability / injury detection ---
                try:
                    from datetime import date as dt_date, timedelta
                    player_team = sof_rows["team"].mode().iloc[0]
                    current_season_sof = sof_rows[sof_rows["date"] >= "2025-08-01"]
                    if not current_season_sof.empty:
                        # All team match dates this season
                        team_season = sof[(pd.to_datetime(sof["date"]) >= pd.Timestamp("2025-08-01")) & (sof["team"] == player_team)]
                        all_team_dates = sorted(pd.to_datetime(team_season["date"]).dt.date.unique())
                        player_dates = set(pd.to_datetime(current_season_sof["date"]).dt.date.unique())

                        team_total = len(all_team_dates)
                        player_total = len(player_dates)
                        missed_dates = sorted([d for d in all_team_dates if d not in player_dates])
                        matches_missed = len(missed_dates)

                        availability = {
                            "team_matches_played": team_total,
                            "player_appearances": player_total,
                            "matches_missed": matches_missed,
                            "availability_pct": round(100 * player_total / team_total, 1) if team_total else 100,
                        }

                        # Find longest consecutive absence
                        if missed_dates:
                            streaks = []
                            streak_start = missed_dates[0]
                            prev = missed_dates[0]
                            for i in range(1, len(missed_dates)):
                                if (missed_dates[i] - prev).days <= 21:  # within ~2 matchweeks
                                    prev = missed_dates[i]
                                else:
                                    streaks.append((streak_start, prev, len([d for d in missed_dates if streak_start <= d <= prev])))
                                    streak_start = missed_dates[i]
                                    prev = missed_dates[i]
                            streaks.append((streak_start, prev, len([d for d in missed_dates if streak_start <= d <= prev])))

                            longest = max(streaks, key=lambda s: s[2])
                            if longest[2] >= 3:  # 3+ consecutive missed matches = significant absence
                                availability["longest_absence"] = {
                                    "from": str(longest[0]),
                                    "to": str(longest[1]),
                                    "matches_missed": longest[2],
                                    "duration_days": (longest[1] - longest[0]).days,
                                }

                            # Current status: is the player currently absent?
                            today = dt_date.today()
                            last_played = max(player_dates)
                            last_team_match = max(all_team_dates)

                            if last_played < last_team_match:
                                # Missed recent matches
                                recent_missed = len([d for d in missed_dates if d > last_played])
                                availability["current_status"] = "absent"
                                availability["last_played"] = str(last_played)
                                availability["consecutive_missed"] = recent_missed
                                availability["days_since_last_match"] = (today - last_played).days
                            else:
                                # Played the most recent match
                                if matches_missed >= 3:
                                    availability["current_status"] = "returning"
                                    availability["last_played"] = str(last_played)
                                    # Find when the absence ended
                                    pre_return_gap = [s for s in streaks if s[1] >= last_played - timedelta(days=30)]
                                    if pre_return_gap:
                                        gap = max(pre_return_gap, key=lambda s: s[2])
                                        availability["returned_after"] = {
                                            "absent_from": str(gap[0]),
                                            "absent_to": str(gap[1]),
                                            "matches_missed": gap[2],
                                        }
                                else:
                                    availability["current_status"] = "fit"

                        # --- Ground the absence REASON from the injury feed ---
                        # The block above detects WHEN a player was absent (gap in
                        # match dates) but not WHY. The Transfermarkt injury snapshots
                        # (data/external/injuries/) carry injury_type + expected_return.
                        # Attach the documented reason ONLY when a snapshot's window
                        # overlaps the detected absence — never infer/invent a cause.
                        try:
                            absence = availability.get("longest_absence") or availability.get("returned_after")
                            if absence:
                                from datetime import date as _dt_date
                                a_from = pd.to_datetime(absence.get("from") or absence.get("absent_from")).date()
                                a_to = pd.to_datetime(absence.get("to") or absence.get("absent_to")).date()
                                inj_dir = _BASE / "data" / "external" / "injuries"
                                reason = None
                                if inj_dir.exists():
                                    # snapshots whose scrape date falls in/near the window
                                    for snap in sorted(inj_dir.glob("injuries_*.parquet")):
                                        # parse date from filename injuries_YYYY-MM-DD[...].parquet
                                        import re as _re
                                        m = _re.search(r"(\d{4}-\d{2}-\d{2})", snap.name)
                                        if not m:
                                            continue
                                        snap_date = pd.to_datetime(m.group(1)).date()
                                        # only snapshots inside the absence window (± a few days)
                                        if not (a_from - timedelta(days=7) <= snap_date <= a_to + timedelta(days=7)):
                                            continue
                                        idf = pd.read_parquet(snap)
                                        ncol = next((c for c in idf.columns if c.lower() in ("player_name", "name", "player")), None)
                                        if ncol is None:
                                            continue
                                        # tight match: player name AND same team (guards
                                        # "Lautaro Valenti"/Parma vs "Lautaro Martínez"/Inter)
                                        cand = idf[_player_name_match(idf[ncol], player_q)]
                                        if "team" in idf.columns and player_team:
                                            cand = cand[cand["team"].astype(str).apply(
                                                lambda t: _normalize_name(str(t)) == _normalize_name(str(player_team))
                                                or _normalize_name(str(player_team)) in _normalize_name(str(t))
                                            )]
                                        cand = cand[cand.get("is_currently_out", True) == True] if "is_currently_out" in cand.columns else cand
                                        if not cand.empty:
                                            row = cand.iloc[0]
                                            itype = str(row.get("injury_type", "") or "").strip()
                                            exp_ret = row.get("expected_return")
                                            if itype:
                                                reason = {
                                                    "injury_type": itype,
                                                    "source": str(row.get("source", "transfermarkt")),
                                                    "recorded_on": str(snap_date),
                                                }
                                                if pd.notna(exp_ret) and str(exp_ret).strip():
                                                    reason["expected_return_at_time"] = str(exp_ret)[:10]
                                                break  # first overlapping snapshot with a type wins
                                if reason:
                                    availability["absence_reason"] = reason
                                else:
                                    availability["absence_reason"] = "NOT_IN_DATA — the injury feed has no record covering this absence window; state the absence but do NOT invent a cause (do not say 'injury', 'suspension', 'rested' unless this field carries it)."
                        except Exception as _ie:  # noqa: BLE001
                            log.warning("Absence-reason lookup failed: %s", _ie)

                        result["availability"] = availability
                except Exception as e:
                    log.warning("Availability analysis failed: %s", e)

    except Exception as e:
        log.warning("Sofascore player lookup failed: %s", e)

    # 3. Upcoming match props from player_props.json
    props = load_json_safe(UPCOMING_DIR / "player_props.json")
    for match_key, match_data in props.get("matches", {}).items():
        if isinstance(match_data, dict):
            for p in match_data.get("players", []):
                if isinstance(p, dict) and all(
                    w in _normalize_name(p.get("name", "")) for w in _normalize_name(player_q).split()
                ):
                    result["upcoming_props"] = {
                        "match": match_key,
                        "name": p.get("name"),
                        "xg_per_90": p.get("xg_per_90"),
                        "xa_per_90": p.get("xa_per_90"),
                        "anytime_goal_prob": p.get("anytime_goal_prob"),
                        "anytime_fair_odds": p.get("anytime_fair_odds"),
                        "assist_prob": p.get("assist_prob"),
                        "shots_expected": p.get("shots_expected"),
                        "to_be_carded_prob": p.get("to_be_carded_prob"),
                    }
                    break
        if "upcoming_props" in result:
            break

    # 4. Market value + contract + salary estimate (2026-27 squad data). Lets the
    #    assistant answer "he earns X/month, has N years left, and last season
    #    played…" in one breath. Salary is a Capology ESTIMATE (fixed gross, excl.
    #    bonuses) — labeled as such so the model never states it as official.
    try:
        import pandas as pd
        tm_dir = DATA_DIR / "external" / "transfermarkt"
        mv_path = tm_dir / "market_values_2026_2027.parquet"
        if mv_path.exists():
            mv = pd.read_parquet(mv_path)
            mmask = _player_name_match(mv["player_name"], player_q)
            if team_q:
                tr = _resolve_team(team_q)
                if tr and "team" in mv.columns:
                    mmask = mmask & (mv["team"] == tr)
            hit = mv[mmask]
            if not hit.empty:
                row = hit.iloc[0]
                market = {
                    "team": row.get("team"),
                    "market_value_eur": int(row["market_value_eur"]) if pd.notna(row.get("market_value_eur")) else None,
                    "age": int(row["age"]) if pd.notna(row.get("age")) else None,
                    "contract_until": str(row["contract_until"])[:10] if pd.notna(row.get("contract_until")) else None,
                    "nationality": row.get("nationality"),
                }
                # Capology salary estimate, matched by (team, fuzzy name).
                sal_path = tm_dir / "salaries_2026_2027.parquet"
                if sal_path.exists():
                    sal = pd.read_parquet(sal_path)
                    smask = _player_name_match(sal["player_name"], player_q)
                    if pd.notna(row.get("team")) and "team" in sal.columns:
                        smask = smask & (sal["team"] == row.get("team"))
                    shit = sal[smask]
                    if not shit.empty:
                        s = shit.iloc[0]
                        ann = int(s["annual_gross_eur"]) if pd.notna(s.get("annual_gross_eur")) else None
                        market["salary_estimate"] = {
                            "source": "Capology estimate (fixed gross, excludes bonuses — NOT an official figure)",
                            "verified_by_capology": bool(s.get("verified")),
                            "annual_gross_eur": ann,
                            "monthly_gross_eur": int(s["monthly_gross_eur"]) if pd.notna(s.get("monthly_gross_eur")) else (ann // 12 if ann else None),
                            "weekly_gross_eur": int(s["weekly_gross_eur"]) if pd.notna(s.get("weekly_gross_eur")) else (ann // 52 if ann else None),
                            "contract_years_remaining": int(s["years_remaining"]) if pd.notna(s.get("years_remaining")) else None,
                            "contract_expiration": str(s["contract_expiration"])[:10] if pd.notna(s.get("contract_expiration")) else None,
                        }
                if "salary_estimate" not in market:
                    market["salary_note"] = (
                        "No Capology wage estimate on file for this player — "
                        "state that it's unavailable; do NOT invent a salary."
                    )
                result["market_and_salary"] = market
            else:
                # No current-squad row. The player may have LEFT the league (contract
                # expired / transferred out). Check the transfers feed so we tell the
                # user the truth ("now a free agent") instead of returning silent
                # zeros that invite a fabricated wage/contract.
                dep = None
                try:
                    tf_path = tm_dir / "transfers_2026_2027.parquet"
                    if tf_path.exists():
                        tf = pd.read_parquet(tf_path)
                        tmask = _player_name_match(tf["player_name"], player_q)
                        outs = tf[tmask & (tf["transfer_type"] == "out")] if "transfer_type" in tf.columns else tf[tmask]
                        if not outs.empty:
                            o = outs.iloc[0]
                            dep = {
                                "from_club": o.get("from_club"),
                                "to_club": o.get("to_club"),
                                "fee_text": o.get("fee_text"),
                            }
                except Exception:  # noqa: BLE001
                    dep = None
                result["market_and_salary"] = {
                    "status": "NOT_IN_CURRENT_SQUAD",
                    "note": (
                        "This player is NOT in any 2026-27 Serie A squad. Do NOT "
                        "estimate their current wage, contract, or club — we have "
                        "no current data. If a departure is shown below, report it."
                    ),
                    "departure": dep,
                }
    except Exception as e:  # noqa: BLE001 — market/salary is enrichment; never blocks the stats answer
        log.warning("Market/salary lookup failed: %s", e)

    # 5. Understat multi-season history + xG over/under-performance. This is the
    #    single most insightful GROUNDED analytical signal for a forward: goals
    #    vs expected goals tells you if they're clinical or wasteful, and the
    #    per-season rows give real "last season vs the season before" context.
    #    Understat covers ONLY Serie A + Premier League — a Ligue 1 / Bundesliga
    #    stint (e.g. David at Lille) simply won't appear, and that's fine: we
    #    return what we have and never invent the gap.
    try:
        import pandas as pd
        us_path = DATA_DIR / "parsed" / "understat_players.parquet"
        if us_path.exists():
            us = pd.read_parquet(us_path)
            umask = _player_name_match(us["player"], player_q)
            uhit = us[umask]
            if not uhit.empty:
                # canonical name = most common exact match; keep only that player
                uname = uhit["player"].mode().iloc[0]
                uhit = uhit[uhit["player"] == uname].sort_values("season")
                seasons = []
                for _, r in uhit.iterrows():
                    g = float(r.get("goals", 0) or 0)
                    xg = float(r.get("xg", 0) or 0)
                    a = float(r.get("assists", 0) or 0)
                    xa = float(r.get("xa", 0) or 0)
                    npg = float(r.get("np_goals", 0) or 0)
                    npxg = float(r.get("np_xg", 0) or 0)
                    seasons.append({
                        "season": r.get("season"),
                        "league": r.get("league"),
                        "team": r.get("team"),
                        "matches": int(r.get("matches", 0) or 0),
                        "minutes": int(r.get("minutes", 0) or 0),
                        "goals": int(g),
                        "xg": round(xg, 1),
                        # + = clinical (scored more than chances warranted), − = wasteful
                        "goals_minus_xg": round(g - xg, 1),
                        # NON-PENALTY goals vs xG — strips penalties, the truer finishing
                        # signal for a penalty-taker (a striker padded by spot-kicks looks
                        # clinical on raw xG but np_goals_minus_np_xg exposes open-play form)
                        "np_goals": int(npg),
                        "np_xg": round(npxg, 1),
                        "np_goals_minus_np_xg": round(npg - npxg, 1),
                        "assists": int(a),
                        "xa": round(xa, 1),
                        "shots": int(r.get("shots", 0) or 0),
                        "key_passes": int(r.get("key_passes", 0) or 0),
                        # buildup involvement: total xG of possessions the player was in
                        # (chain) / in but not the shot or assist (buildup) — deep playmakers
                        "xg_chain": round(float(r.get("xg_chain", 0) or 0), 1),
                        "xg_buildup": round(float(r.get("xg_buildup", 0) or 0), 1),
                    })
                result["understat_history"] = {
                    "note": (
                        "Understat per-season data (Serie A + Premier League only). "
                        "goals_minus_xg > 0 = clinical, < 0 = wasteful/unlucky. "
                        "np_goals_minus_np_xg = the same but PENALTY-STRIPPED (truer "
                        "open-play finishing — use it for penalty-takers). xg_chain / "
                        "xg_buildup = involvement in the buildup of chances (high = deep "
                        "playmaker even with few goals). Use verbatim — never invent "
                        "seasons or leagues not listed here."
                    ),
                    "player": uname,
                    "seasons": seasons,
                }

                # ── Reconcile the two current-season goal counts ──────────────────────
                # season_stats (FBref) and understat_history (Understat) are INDEPENDENT
                # scrapes of the SAME season with different match coverage, so their goal
                # totals routinely disagree by 1-2 (e.g. Lautaro FBref 16 vs Understat 17).
                # Without this, the model quotes BOTH silently in adjacent sentences and
                # contradicts itself. Hand it ONE reconciled headline number + the delta,
                # and forbid citing both as if separate facts.
                try:
                    cur = next(
                        (s for s in seasons if str(s.get("season", "")).startswith("2025")),
                        None,
                    )
                    ss = result.get("season_stats")
                    if cur and isinstance(ss, dict) and ss.get("goals") is not None:
                        us_g, fb_g = int(cur["goals"]), int(ss["goals"])
                        if us_g != fb_g:
                            result["stat_reconciliation"] = (
                                f"GOAL-COUNT SOURCE CONFLICT (2025-26): Understat says "
                                f"{us_g} goals, the club-stats feed (FBref/Sofascore) says "
                                f"{fb_g} — two independent scrapes with slightly different "
                                f"match coverage. Report ONE number: use {us_g} (Understat, "
                                f"the same source as the xG figures) as the headline goal "
                                f"total so goals and xG are consistent, and if precision "
                                f"matters say 'roughly {min(us_g, fb_g)}-{max(us_g, fb_g)} "
                                f"depending on source'. Do NOT state {us_g} in one sentence "
                                f"and {fb_g} in another as if both are separately true — "
                                f"that reads as a contradiction. The advanced-stats block "
                                f"may show a THIRD count ({ss.get('goals')} etc.); it is the "
                                f"same conflict, not a new fact."
                            )
                except Exception as _re:  # noqa: BLE001
                    log.warning("Goal-count reconciliation failed: %s", _re)
    except Exception as e:  # noqa: BLE001 — Understat is enrichment; never blocks the answer
        log.warning("Understat history lookup failed: %s", e)

    # 6. World Cup 2026 tournament performance. Just played (Jun 11–Jul 19) — a
    #    player's form/fitness/minutes there directly colours the new club season
    #    (deep run + heavy minutes = fatigue risk; strong WC = confidence/value).
    #    Filter to the ACTUAL tournament ("FIFA World Cup, Group X" / "…, Knockout")
    #    — NOT the "World Cup Qual…" rows, which are a different competition.
    try:
        import pandas as pd
        wc_path = DATA_DIR / "worldcup" / "sofascore_intl_player_stats.parquet"
        if wc_path.exists():
            wc = pd.read_parquet(wc_path)
            if "tournament" in wc.columns:
                is_wc = wc["tournament"].astype(str).str.startswith("FIFA World Cup, ")
                is_qual = wc["tournament"].astype(str).str.contains("Qual", case=False, na=False)
                wc = wc[is_wc & ~is_qual]
                if "date" in wc.columns:
                    wc = wc[pd.to_datetime(wc["date"], utc=True, errors="coerce") >= "2026-06-01"]
            wmask = _player_name_match(wc["player_name"], player_q)
            whit = wc[wmask]
            if not whit.empty:
                # canonical name (guards the "David" collision — 11 different Davids)
                wname = whit["player_name"].mode().iloc[0]
                whit = whit[whit["player_name"] == wname]
                wmin = int(whit["minutes"].sum())
                wmp = int(whit["match_id"].nunique()) if "match_id" in whit.columns else len(whit)
                # NOTE: parquet goal-sum is only for the matches it captured. The AUTHORITATIVE
                # goal count comes from the event log below — the parquet undercounts any player
                # who scored in an uncaptured (Sofascore-banned) knockout match.
                wgoals_parquet = int(whit["goals"].sum())
                national_team = whit["team"].mode().iloc[0] if "team" in whit.columns else None
                ratings = whit["rating"].dropna()
                wavg = round(float(ratings.mean()), 1) if not ratings.empty else None

                # ── Layered authoritative sourcing (per source's ONE job) ──────────────
                # The parquet (Sofascore) lags the live bracket because the provider is
                # IP-banned mid-tournament (measured: breaker open, 0 rows appended). So:
                #   • GOALS      → international_goalscorers.csv (event log = ground truth)
                #   • COUNT/DATES/BRACKET → international_results.csv (fresh, ESPN/CSV path)
                #   • MINUTES/RATINGS/SHOTS per match → parquet only (nothing else has them)
                # Every fact is read live and asserted ONLY when the source confirms it —
                # never infer a score or a bracket outcome the results file hasn't recorded.
                wc_goals_true = wgoals_parquet          # fallback if event log unreadable
                wc_goal_source = "parquet (may undercount)"
                wc_stale_note = ""
                try:
                    last_in_parquet = pd.to_datetime(
                        whit["date"], utc=True, errors="coerce"
                    ).max()
                    wc_dir = DATA_DIR / "worldcup"

                    # 1) TRUE goal count from the per-goal event log
                    gc_path = wc_dir / "international_goalscorers.csv"
                    if gc_path.exists():
                        gc = pd.read_csv(gc_path)
                        gc["date"] = pd.to_datetime(gc["date"], errors="coerce")
                        gc = gc[gc["date"] >= "2026-06-01"]
                        gmask = _player_name_match(gc["scorer"].astype(str), player_q)
                        pl_goals = gc[gmask]
                        if national_team and "team" in gc.columns:
                            pl_goals = pl_goals[
                                pl_goals["team"].astype(str).apply(
                                    lambda t: _normalize_name(str(t)) == _normalize_name(str(national_team))
                                )
                                | pl_goals["team"].isna()
                            ]
                        # only count if we actually located this scorer's rows
                        if not pl_goals.empty or gmask.any():
                            wc_goals_true = int(len(pl_goals))
                            wc_goal_source = "event log (international_goalscorers.csv)"

                    # 2) TRUE match count + schedule + bracket from the results file
                    res_path = wc_dir / "international_results.csv"
                    if national_team and res_path.exists():
                        res = pd.read_csv(res_path)
                        res["date"] = pd.to_datetime(res["date"], utc=True, errors="coerce")
                        res = res[
                            res["tournament"].astype(str).str.contains("World Cup", na=False)
                            & (res["date"] >= "2026-06-01")
                        ]
                        team_res = res[
                            (res["home_team"] == national_team)
                            | (res["away_team"] == national_team)
                        ]
                        played = team_res.dropna(subset=["home_score", "away_score"])
                        sched = team_res[team_res["home_score"].isna()]
                        n_played = len(played)
                        last_played = played["date"].max() if not played.empty else None
                        next_sched = sched["date"].min() if not sched.empty else None
                        parts = []
                        missing = n_played - wmp
                        if missing > 0 or (
                            last_played is not None
                            and pd.notna(last_in_parquet)
                            and last_played > last_in_parquet
                        ):
                            parts.append(
                                f"DATA NOTE: {national_team} have actually PLAYED {n_played} WC "
                                f"matches (through {str(last_played)[:10]}), but the per-match "
                                f"detail table below only captured {wmp} of them (the stats "
                                f"provider was blocked for the newer knockout rounds). Use "
                                f"{wc_goals_true} as the goal count (from the goal event log), "
                                f"NOT the sum of the partial table. The per-match minutes/ratings "
                                f"cover only the captured matches."
                            )
                        if next_sched is not None and pd.notna(next_sched):
                            nrow = sched.sort_values("date").iloc[0]
                            opp = nrow["away_team"] if nrow["home_team"] == national_team else nrow["home_team"]
                            # bracket context: name what's still unresolved, assert only resolved
                            other = played_bracket_context(res, national_team, next_sched)
                            parts.append(
                                f"{national_team}'s NEXT WC match is still to be played: vs {opp} "
                                f"on {str(next_sched)[:10]} — the tournament is LIVE, not finished. "
                                f"{other}"
                            )
                        if parts:
                            wc_stale_note = " ".join(parts) + " "
                except Exception as fe:  # noqa: BLE001
                    log.warning("WC freshness/goal cross-check failed: %s", fe)
                # Pre-render as a markdown table (same anti-hallucination pattern as
                # match_performances) so the model copies exact numbers verbatim and
                # never mistakes a nested JSON array for "truncated" data.
                wc_header = "| Date | Opponent | Start | Min | G | Shots | Rating |"
                wc_sep = "|------|----------|-------|-----|---|-------|--------|"
                wc_lines = [wc_header, wc_sep]
                for _, r in whit.sort_values("date", ascending=False).iterrows():
                    rt = round(float(r["rating"]), 1) if pd.notna(r.get("rating")) else "-"
                    st = "Y" if r.get("started") else "sub"
                    wc_lines.append(
                        f"| {str(r.get('date',''))[:10]} | {r.get('opponent','')} | {st} "
                        f"| {int(r.get('minutes',0) or 0)} | {int(r.get('goals',0) or 0)} "
                        f"| {int(r.get('shots',0) or 0)} | **{rt}** |"
                    )
                result["world_cup_2026"] = (
                    f"FIFA WORLD CUP 2026 PERFORMANCE (this field IS present and populated — "
                    f"the player DID feature; report it, never say it's missing). "
                    f"{wc_stale_note}"
                    f"{national_team}: {wc_goals_true} goals [source: {wc_goal_source}] — this "
                    f"is the goal count to report. Per-match detail captured for {wmp} match(es) "
                    f"below (~{wmin} min, avg rating {wavg} across captured matches only). He JUST "
                    f"played this tournament (Jun-Jul 2026) — weigh fatigue (heavy minutes/deep "
                    f"run) vs form against his club season. The table shows minutes/ratings/shots "
                    f"for the matches the provider captured; do NOT invent matches beyond it, and "
                    f"do NOT re-sum its goals column (use the goal count above):\n\n"
                    + "\n".join(wc_lines)
                )
    except Exception as e:  # noqa: BLE001 — WC block is enrichment; never blocks the answer
        log.warning("World Cup lookup failed: %s", e)

    if (
        "season_stats" not in result
        and "upcoming_props" not in result
        and "market_and_salary" not in result
        and "understat_history" not in result
        and "world_cup_2026" not in result
    ):
        return json.dumps({"error": f"No data found for player '{args.get('player')}'. Try the full name (e.g., 'Lautaro Martinez')."})

    return json.dumps(result, default=str)


def _tool_get_h2h(args: dict) -> str:
    t1 = _resolve_team(args.get("team1", ""))
    t2 = _resolve_team(args.get("team2", ""))
    if not t1 or not t2:
        return json.dumps({"error": "Could not resolve team names."})

    h2h_data = load_json_safe(UPCOMING_DIR / "h2h_upcoming.json")
    h2h_list = h2h_data.get("h2h", [])
    if isinstance(h2h_list, list):
        for entry in h2h_list:
            teams = entry.get("teams", entry.get("match", ""))
            if isinstance(teams, str) and t1 in teams and t2 in teams:
                return json.dumps({"teams": [t1, t2], "h2h": entry}, default=str)

    return json.dumps({"teams": [t1, t2], "h2h": None, "note": "No H2H data available."})


def _tool_get_value_bets(args: dict) -> str:
    from datetime import datetime as _dt

    slip = load_json_safe(UPCOMING_DIR / "unified_bet_slip.json")
    bets = slip.get("selected_bets", [])
    summary = slip.get("summary", {})
    today = _dt.now().strftime("%Y-%m-%d")

    # Separate today's bets from future bets
    today_bets = []
    future_bets = []
    for b in bets:
        bet_date = b.get("date", "")
        entry = {
            "match": b.get("match"),
            "date": bet_date,
            "market": b.get("market"),
            "selection": b.get("selection"),
            "odds": b.get("best_odds"),
            "edge_pct": b.get("edge_pct"),
            "bookmaker": b.get("best_bookmaker"),
            "kelly_stake": b.get("kelly_stake_pct", b.get("stake_pct")),
            "model_prob": b.get("model_prob"),
        }
        if bet_date == today:
            today_bets.append(entry)
        else:
            future_bets.append(entry)

    result = {
        "generated_at": slip.get("generated_at"),
        "today": today,
        "today_bets": today_bets,
        "today_count": len(today_bets),
        "future_bets": future_bets[:5],
        "future_count": len(future_bets),
        "total_bets": len(bets),
        "summary": summary,
        # Keep "bets" for backward compat but add _note
        "bets": today_bets if today_bets else [entry for entry in (today_bets + future_bets)[:8]],
        "_note": f"Today is {today}. Only recommend bets from today_bets for immediate action. future_bets are for upcoming matchdays.",
    }

    # Add handicap bets — mark which are today
    hcap = load_json_safe(UPCOMING_DIR / "handicap_bets.json")
    hcap_bets = hcap.get("recommended", [])
    if hcap_bets:
        result["handicap_bets"] = [
            {
                "match": b.get("match"),
                "date": b.get("date", ""),
                "is_today": b.get("date", "") == today,
                "bet": b.get("bet"),
                "italian_format": b.get("italian_format"),
                "odds": b.get("odds"),
                "our_probability": b.get("our_probability"),
                "value_pct": b.get("value_pct"),
                "stake_pct": b.get("stake_pct"),
            }
            for b in hcap_bets[:10]
        ]

    # Add over/under bets — mark which are today
    ou = load_json_safe(UPCOMING_DIR / "over_under_bets.json")
    ou_bets = ou.get("recommended", [])
    if ou_bets:
        result["over_under_bets"] = [
            {
                "match": b.get("match"),
                "date": b.get("date", ""),
                "is_today": b.get("date", "") == today,
                "bet": b.get("bet"),
                "italian_format": b.get("italian_format"),
                "odds": b.get("odds"),
                "our_probability": b.get("our_probability"),
                "value_pct": b.get("value_pct"),
                "stake_pct": b.get("stake_pct"),
            }
            for b in ou_bets[:10]
        ]

    return json.dumps(result, default=str)


def _tool_get_match_players(args: dict) -> str:
    """Get full rated lineups for both teams in a specific match."""
    home_q = args.get("home", "").strip()
    away_q = args.get("away", "").strip()
    date_q = args.get("date", "").strip()  # optional YYYY-MM-DD

    if not home_q and not away_q:
        return json.dumps({"error": "Provide at least one team name (home or away)."})

    try:
        import pandas as pd
        sof_path = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"
        if not sof_path.exists():
            return json.dumps({"error": "Player match stats not available."})

        sof = pd.read_parquet(sof_path)
        sof["date"] = pd.to_datetime(sof["date"])

        # Resolve team names
        home_team = _resolve_team(home_q) if home_q else None
        away_team = _resolve_team(away_q) if away_q else None

        # Find the match — filter by team(s) and optionally date
        mask = pd.Series(True, index=sof.index)
        if home_team:
            mask = mask & (sof["team"] == home_team) & (sof["is_home"] == True)
        if away_team:
            mask = mask & (sof["team"] == away_team) & (sof["is_home"] == False) if home_team else \
                   mask & (sof["team"] == away_team)
        if date_q:
            mask = mask & (sof["date"].dt.strftime("%Y-%m-%d") == date_q)

        candidates = sof[mask]
        if candidates.empty:
            # Try without home/away constraint
            mask2 = pd.Series(False, index=sof.index)
            if home_team:
                mask2 = mask2 | (sof["team"] == home_team)
            if away_team:
                mask2 = mask2 | (sof["team"] == away_team)
            if date_q:
                mask2 = mask2 & (sof["date"].dt.strftime("%Y-%m-%d") == date_q)
            candidates = sof[mask2]

        if candidates.empty:
            return json.dumps({"error": f"No match found for {home_q} vs {away_q}" + (f" on {date_q}" if date_q else "")})

        # Get the most recent match
        latest_date = candidates["date"].max()
        candidates = candidates[candidates["date"] == latest_date]

        # Find the match_id — if both teams provided, find the shared match
        if home_team and away_team:
            home_matches = set(candidates[candidates["team"] == home_team]["match_id"].unique())
            away_matches = set(candidates[candidates["team"] == away_team]["match_id"].unique())
            shared = home_matches & away_matches
            if shared:
                match_id = shared.pop()
            elif home_matches:
                match_id = home_matches.pop()
            else:
                match_id = candidates["match_id"].iloc[0]
        else:
            match_id = candidates["match_id"].iloc[0]

        # Get all players for this match
        match_data = sof[sof["match_id"] == match_id]
        if match_data.empty:
            return json.dumps({"error": "Match found but no player data."})

        sample = match_data.iloc[0]
        match_date = str(sample["date"])[:10]
        home_score = int(sample.get("home_score", 0))
        away_score = int(sample.get("away_score", 0))

        # Identify home and away teams from the data
        teams_in_match = match_data["team"].unique().tolist()
        home_t = match_data[match_data["is_home"] == True]["team"].iloc[0] if not match_data[match_data["is_home"] == True].empty else teams_in_match[0]
        away_t = [t for t in teams_in_match if t != home_t][0] if len(teams_in_match) > 1 else "?"

        result: dict[str, Any] = {
            "match": f"{home_t} vs {away_t}",
            "date": match_date,
            "score": f"{home_score}-{away_score}",
        }

        # Build pre-rendered team sheets for both teams
        for team_name, label in [(home_t, "home_team_sheet"), (away_t, "away_team_sheet")]:
            team_players = match_data[match_data["team"] == team_name].sort_values("rating", ascending=False)
            if team_players.empty:
                continue

            header = f"### {team_name} Ratings"
            rows = [
                "| Player | Pos | Min | Rating | G | A | Shots | SoT | KeyP | Touches |",
                "|--------|-----|-----|--------|---|---|-------|-----|------|---------|",
            ]
            best_player = None
            best_rating = 0
            for _, p in team_players.iterrows():
                rating = round(float(p.get("rating", 0)), 1) if pd.notna(p.get("rating")) else "-"
                if isinstance(rating, float) and rating > best_rating:
                    best_rating = rating
                    best_player = p.get("player_name", "")
                rows.append(
                    f"| {p.get('player_name', '')} | {p.get('position', '')} "
                    f"| {int(p.get('minutes', 0))} | **{rating}** "
                    f"| {int(p.get('goals', 0))} | {int(p.get('assists', 0))} "
                    f"| {int(p.get('total_shots', 0))} | {int(p.get('shots_on_target', 0))} "
                    f"| {int(p.get('key_passes', 0))} | {int(p.get('touches', 0))} |"
                )
            result[label] = (
                f"VERIFIED SOFASCORE DATA — COPY EXACTLY.\n"
                f"Best rated: {best_player} ({best_rating})\n\n"
                f"{header}\n" + "\n".join(rows)
            )

        # Match events
        inc_path = DATA_DIR / "external" / "sofascore" / "match_incidents.parquet"
        if inc_path.exists():
            try:
                inc = pd.read_parquet(inc_path)
                match_events = inc[(inc["match_id"] == match_id)
                                   & (inc["incident_type"] != "var_checked")].sort_values("minute")  # backfill marker rows are not events
                events = []
                for _, ev in match_events.iterrows():
                    evt = {"minute": int(ev.get("minute", 0)), "type": ev.get("incident_type", "")}
                    if ev.get("incident_type") == "goal":
                        evt["scorer"] = ev.get("player_name", "")
                        evt["assist"] = ev.get("assist_player", "")
                        evt["goal_type"] = ev.get("goal_type", ev.get("incident_class", ""))
                    elif ev.get("incident_type") == "substitution":
                        evt["player_in"] = ev.get("player_in_name", "")
                        evt["player_out"] = ev.get("player_name", "")
                    elif ev.get("incident_type") == "card":
                        evt["player"] = ev.get("player_name", "")
                        evt["card"] = ev.get("card_type", ev.get("incident_class", ""))
                    events.append(evt)
                if events:
                    result["match_events"] = events
            except Exception:
                pass

        return json.dumps(result, default=str)

    except Exception as e:
        log.warning("get_match_players failed: %s", e)
        return json.dumps({"error": str(e)})


def _tool_get_live_matches(args: dict) -> str:
    """Get live match data from today's matchday file."""
    today = datetime.now().strftime("%Y-%m-%d")
    matchday = load_json_safe(LIVE_DIR / f"{today}.json")

    if not matchday or not matchday.get("matches"):
        # Try yesterday (late matches may still be in yesterday's file)
        from datetime import timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        matchday = load_json_safe(LIVE_DIR / f"{yesterday}.json")
        if not matchday or not matchday.get("matches"):
            return json.dumps({"status": "No live match data available for today."})

    matches = matchday.get("matches", {})
    result = []
    for key, m in matches.items():
        status = m.get("status", "unknown")
        match_info = {
            "match": key,
            "home_team": m.get("home_team"),
            "away_team": m.get("away_team"),
            "status": status,
            "score": m.get("final_score") or (m.get("snapshots", [{}])[-1].get("score") if m.get("snapshots") else None),
        }
        # Add current minute estimate from latest snapshot
        if m.get("snapshots"):
            latest = m["snapshots"][-1]
            match_info["minute"] = latest.get("min")
            match_info["current_odds"] = latest.get("avg_odds")
        # Add pre-match odds
        if m.get("pre_match_odds"):
            match_info["pre_match_odds"] = m["pre_match_odds"]
        # Add key events (goals, red cards)
        events = m.get("live_events", [])
        key_events = [e for e in events if e.get("type") in ("goal", "card") and e.get("card_type", "") != "yellow"]
        if key_events:
            match_info["key_events"] = key_events[:10]
        # Add live stats summary
        if m.get("live_stats"):
            stats = m["live_stats"]
            match_info["stats"] = {
                "possession": stats.get("possession"),
                "xg": stats.get("xg"),
                "shots": stats.get("shots"),
                "corners": stats.get("corners"),
            }
        # Add bet tracking
        bet_tracking = matchday.get("bet_tracking", [])
        match_bets = [b for b in bet_tracking if key in b.get("match", "")]
        if match_bets:
            match_info["bets"] = [
                {"market": b.get("market"), "selection": b.get("selection"),
                 "odds": b.get("placed_odds"), "status": b.get("status")}
                for b in match_bets
            ]
        result.append(match_info)

    return json.dumps({"date": matchday.get("date", today), "matches": result, "polls": matchday.get("polls", 0)}, default=str)


def _tool_get_results(args: dict) -> str:
    """Get recent match results and settled bets."""
    date_filter = args.get("date", "")  # optional YYYY-MM-DD filter

    # Match results
    results_data = load_json_safe(UPCOMING_DIR / "results.json")
    results = results_data.get("results", {})
    if isinstance(results, dict):
        results_list = list(results.values())
    elif isinstance(results, list):
        results_list = results
    else:
        results_list = []

    # Filter by date if provided
    if date_filter:
        results_list = [
            r for r in results_list
            if date_filter in (r.get("commence_time", "") or r.get("date", ""))
        ]

    formatted = []
    for r in results_list:
        formatted.append({
            "match": r.get("match", f"{r.get('home_team', '?')} vs {r.get('away_team', '?')}"),
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "score": f"{r.get('home_score', '?')}-{r.get('away_score', '?')}",
            "result": r.get("result"),
            "date": (r.get("commence_time", "") or r.get("date", ""))[:10],
            "total_goals": r.get("total_goals"),
            "btts": r.get("btts"),
        })

    # Settled bets for those matches
    history = load_json_safe(BETTING_DIR / "history.json")
    settled = history.get("settled_bets", []) if isinstance(history, dict) else history if isinstance(history, list) else []

    # Filter settled bets to matching dates
    match_names = {r.get("match", "") for r in formatted}
    relevant_bets = []
    for b in settled:
        bet_match = b.get("match", "")
        bet_date = (b.get("date", "") or b.get("settled_at", ""))[:10]
        if (date_filter and date_filter in bet_date) or bet_match in match_names:
            relevant_bets.append({
                "match": bet_match,
                "market": b.get("market"),
                "selection": b.get("selection"),
                "odds": b.get("odds"),
                "stake": b.get("stake"),
                "status": b.get("status"),
                "profit": b.get("profit", b.get("profit_loss", 0)),
            })

    return json.dumps({
        "fetched_at": results_data.get("fetched_at"),
        "total_results": len(formatted),
        "results": formatted,
        "settled_bets": relevant_bets[-20:],  # last 20 relevant
    }, default=str)


def _tool_get_bankroll_status(args: dict) -> str:
    try:
        from scripts.betting.ledger import get_metrics
        m = get_metrics()
    except (ImportError, OSError, ValueError, KeyError, TypeError) as e:
        return json.dumps({"error": f"ledger unavailable: {e}"})

    mb, rec, roi, streak = m["bankroll"], m["record"], m["roi"], m["streak"]
    result: dict[str, Any] = {
        "current_bankroll": mb["current"],
        "initial_bankroll": mb["initial"],
        "peak_bankroll": mb["peak"],
        "lowest_bankroll": mb["lowest"],
        "available_after_pending": mb["available"],
        "pending_bets": mb["pending_n"],
        "pending_stakes": mb["pending_stakes"],
        "drawdown_pct": mb["drawdown_pct"],
        "max_drawdown_pct": mb["max_drawdown_pct"],
        "bankroll_growth_pct": mb["bankroll_growth_pct"],
        "roi_pct": roi["all_time_pct"],  # ROI on stake — NOT bankroll growth
        "rolling_roi_pct": roi["rolling_pct"],
        "rolling_n": roi["rolling_n"],
        "daily_pnl": m["periods"]["today"]["pnl"],
        "last_betting_day": m["periods"]["last_betting_day"],
        "total_bets": rec["settled_n"],
        "total_wins": rec["won"],
        "win_rate_decisive": rec["win_rate_decisive"],
        "current_streak": streak["streak_decisive"],
        "last_updated": m["meta"]["computed_at"],
        "alerts": m["alerts"],
    }

    # P&L history (recent daily snapshots) — a display log, not a metric source
    pnl = load_json_safe(BETTING_DIR / "pnl_history.json", default=[])
    if isinstance(pnl, list) and pnl:
        result["pnl_history"] = [
            {
                "date": p.get("date"),
                "settled": p.get("settled_this_run"),
                "profit": p.get("profit_this_run"),
                "total_bets": p.get("total_bets"),
                "bankroll": p.get("bankroll_after", p.get("current_bankroll")),
            }
            for p in pnl[-10:]  # last 10 snapshots
        ]

    # CLV — per-bet clv_pct from the payload (the one CLV definition)
    if m["clv"]["n"]:
        result["clv"] = {
            "avg_clv_pct": m["clv"]["avg_pct"],
            "total_tracked": m["clv"]["n"],
            "positive_rate_pct": m["clv"]["positive_rate"],
        }

    # Win rate by market — from the payload
    result["market_breakdown"] = {
        k: {
            "wins": g["won"],
            "losses": g["lost"],
            "total": g["won"] + g["lost"],
            "profit": g["profit"],
            "win_rate": round(g["won"] / max(g["won"] + g["lost"], 1) * 100, 1),
            "roi_pct": g["roi_pct"],
        }
        for k, g in roi["by_market"].items()
    }

    return json.dumps(result, default=str)


def _tool_get_betting_performance(args: dict) -> str:
    """Personal betting track record — win rates, streaks, calibration, patterns."""
    days = args.get("days", 30)
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400

    # Load data sources
    history = load_json_safe(BETTING_DIR / "history.json")
    journal = load_json_safe(BETTING_DIR / "bet_journal.json")
    pnl = load_json_safe(BETTING_DIR / "pnl_history.json", default=[])
    perf_dash = load_json_safe(DATA_DIR / "performance_dashboard.json")

    # Settled bets — normalise to a list
    settled: list[dict] = []
    if isinstance(history, dict):
        settled = history.get("settled_bets", [])
    elif isinstance(history, list):
        settled = history

    # Journal bets (richer data with confidence/edge)
    journal_bets: list[dict] = []
    if isinstance(journal, dict):
        raw = journal.get("bets", journal.get("settled", []))
        # bets may be a dict keyed by bet_id — extract values
        if isinstance(raw, dict):
            journal_bets = list(raw.values())
        elif isinstance(raw, list):
            journal_bets = raw
    elif isinstance(journal, list):
        journal_bets = journal

    # Build a lookup from journal for extra fields (confidence, edge)
    journal_lookup: dict[str, dict] = {}
    for jb in journal_bets:
        key = f"{jb.get('match', jb.get('home', '') + ' v ' + jb.get('away', ''))}|{jb.get('market', '')}|{jb.get('selection', '')}"
        journal_lookup[key] = jb

    # Filter by time window if bets have timestamps
    def _in_window(bet: dict) -> bool:
        ts = bet.get("placed_at") or bet.get("settled_at") or bet.get("date", "")
        if isinstance(ts, (int, float)):
            return ts >= cutoff
        if isinstance(ts, str) and ts:
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(ts.replace("Z", "+00:00"))
                return dt.timestamp() >= cutoff
            except (ValueError, TypeError):
                pass
        return True  # include if no timestamp

    filtered = [b for b in settled if _in_window(b)]
    if not filtered:
        filtered = settled  # fall back to all if nothing in window

    result: dict[str, Any] = {"period_days": days, "total_bets": len(filtered)}

    # --- recent_form: last 10 bets ---
    recent = filtered[-10:] if len(filtered) > 10 else filtered
    result["recent_form"] = [
        {
            "match": b.get("match", f"{b.get('home', '?')} v {b.get('away', '?')}"),
            "market": b.get("market", "?"),
            "selection": b.get("selection", "?"),
            "odds": b.get("odds"),
            "status": b.get("status", "?"),
            "profit": round(b.get("profit", b.get("profit_loss", 0)), 2),
        }
        for b in reversed(recent)
    ]

    # --- market_breakdown ---
    market_stats: dict[str, dict] = {}
    for b in filtered:
        mkt = b.get("market", "unknown")
        ms = market_stats.setdefault(mkt, {"wins": 0, "losses": 0, "profit": 0.0, "count": 0})
        ms["count"] += 1
        status = (b.get("status") or "").lower()
        if status in ("won", "win"):
            ms["wins"] += 1
        elif status in ("lost", "loss"):
            ms["losses"] += 1
        ms["profit"] += b.get("profit", b.get("profit_loss", 0))

    result["market_breakdown"] = {
        k: {
            **v,
            "profit": round(v["profit"], 2),
            "win_rate": round(v["wins"] / max(v["wins"] + v["losses"], 1) * 100, 1),
        }
        for k, v in market_stats.items()
    }

    # --- confidence_calibration ---
    conf_stats: dict[str, dict] = {}
    for b in filtered:
        key = f"{b.get('match', '')}|{b.get('market', '')}|{b.get('selection', '')}"
        jb = journal_lookup.get(key, {})
        conf = jb.get("confidence") or b.get("confidence") or "UNKNOWN"
        cs = conf_stats.setdefault(conf, {"wins": 0, "losses": 0, "profit": 0.0, "count": 0})
        cs["count"] += 1
        status = (b.get("status") or "").lower()
        if status in ("won", "win"):
            cs["wins"] += 1
        elif status in ("lost", "loss"):
            cs["losses"] += 1
        cs["profit"] += b.get("profit", b.get("profit_loss", 0))

    result["confidence_calibration"] = {
        k: {
            **v,
            "profit": round(v["profit"], 2),
            "win_rate": round(v["wins"] / max(v["wins"] + v["losses"], 1) * 100, 1),
        }
        for k, v in conf_stats.items()
    }

    # --- edge_calibration ---
    edge_buckets: dict[str, dict] = {"3-6%": {"wins": 0, "losses": 0, "profit": 0.0, "count": 0},
                                      "6-10%": {"wins": 0, "losses": 0, "profit": 0.0, "count": 0},
                                      "10%+": {"wins": 0, "losses": 0, "profit": 0.0, "count": 0}}
    for b in filtered:
        key = f"{b.get('match', '')}|{b.get('market', '')}|{b.get('selection', '')}"
        jb = journal_lookup.get(key, {})
        edge = jb.get("edge") or b.get("edge") or b.get("edge_pct")
        if edge is None:
            continue
        try:
            edge_val = float(str(edge).replace("%", ""))
        except (ValueError, TypeError):
            continue
        if edge_val < 3:
            continue
        elif edge_val < 6:
            bucket = "3-6%"
        elif edge_val < 10:
            bucket = "6-10%"
        else:
            bucket = "10%+"
        eb = edge_buckets[bucket]
        eb["count"] += 1
        status = (b.get("status") or "").lower()
        if status in ("won", "win"):
            eb["wins"] += 1
        elif status in ("lost", "loss"):
            eb["losses"] += 1
        eb["profit"] += b.get("profit", b.get("profit_loss", 0))

    result["edge_calibration"] = {
        k: {
            **v,
            "profit": round(v["profit"], 2),
            "win_rate": round(v["wins"] / max(v["wins"] + v["losses"], 1) * 100, 1),
        }
        for k, v in edge_buckets.items()
        if v["count"] > 0
    }

    # --- streak ---
    current_streak = 0
    streak_type = None
    best_win_streak = 0
    worst_loss_streak = 0
    running_win = 0
    running_loss = 0
    for b in filtered:
        status = (b.get("status") or "").lower()
        if status in ("won", "win"):
            running_win += 1
            running_loss = 0
            best_win_streak = max(best_win_streak, running_win)
        elif status in ("lost", "loss"):
            running_loss += 1
            running_win = 0
            worst_loss_streak = max(worst_loss_streak, running_loss)
        else:
            continue
    # Current streak from the end
    current_streak = 0
    streak_type = None
    for b in reversed(filtered):
        status = (b.get("status") or "").lower()
        if status in ("won", "win"):
            if streak_type is None:
                streak_type = "win"
            if streak_type == "win":
                current_streak += 1
            else:
                break
        elif status in ("lost", "loss"):
            if streak_type is None:
                streak_type = "loss"
            if streak_type == "loss":
                current_streak += 1
            else:
                break
        else:
            break

    result["streak"] = {
        "current": f"{current_streak} {'wins' if streak_type == 'win' else 'losses'}" if streak_type else "N/A",
        "best_win_streak": best_win_streak,
        "worst_loss_streak": worst_loss_streak,
    }

    # --- trend: last 7 days P&L from pnl_history ---
    if isinstance(pnl, list) and pnl:
        result["trend_7d"] = [
            {
                "date": p.get("date"),
                "profit": p.get("profit_this_run", p.get("profit", 0)),
                "bankroll": p.get("bankroll_after", p.get("current_bankroll")),
            }
            for p in pnl[-7:]
        ]

    # --- best_pattern / worst_pattern: market + confidence combo ---
    combo_stats: dict[str, dict] = {}
    for b in filtered:
        mkt = b.get("market", "unknown")
        key_lookup = f"{b.get('match', '')}|{mkt}|{b.get('selection', '')}"
        jb = journal_lookup.get(key_lookup, {})
        conf = jb.get("confidence") or b.get("confidence") or "UNKNOWN"
        combo_key = f"{mkt} / {conf}"
        cs = combo_stats.setdefault(combo_key, {"wins": 0, "losses": 0, "profit": 0.0, "count": 0})
        cs["count"] += 1
        status = (b.get("status") or "").lower()
        if status in ("won", "win"):
            cs["wins"] += 1
        elif status in ("lost", "loss"):
            cs["losses"] += 1
        cs["profit"] += b.get("profit", b.get("profit_loss", 0))

    # Only consider combos with at least 3 bets
    qualified = {k: v for k, v in combo_stats.items() if v["count"] >= 3}
    if qualified:
        best_key = max(qualified, key=lambda k: qualified[k]["profit"])
        worst_key = min(qualified, key=lambda k: qualified[k]["profit"])
        bv = qualified[best_key]
        wv = qualified[worst_key]
        result["best_pattern"] = {
            "combo": best_key,
            "count": bv["count"],
            "win_rate": round(bv["wins"] / max(bv["wins"] + bv["losses"], 1) * 100, 1),
            "profit": round(bv["profit"], 2),
        }
        result["worst_pattern"] = {
            "combo": worst_key,
            "count": wv["count"],
            "win_rate": round(wv["wins"] / max(wv["wins"] + wv["losses"], 1) * 100, 1),
            "profit": round(wv["profit"], 2),
        }
    elif combo_stats:
        # Less than 3 bets per combo — use all combos
        best_key = max(combo_stats, key=lambda k: combo_stats[k]["profit"])
        worst_key = min(combo_stats, key=lambda k: combo_stats[k]["profit"])
        for label, k in [("best_pattern", best_key), ("worst_pattern", worst_key)]:
            v = combo_stats[k]
            result[label] = {
                "combo": k,
                "count": v["count"],
                "win_rate": round(v["wins"] / max(v["wins"] + v["losses"], 1) * 100, 1),
                "profit": round(v["profit"], 2),
                "note": "small sample size",
            }

    # --- accuracy from performance dashboard ---
    if isinstance(perf_dash, dict):
        accuracy = {}
        for key in ("overall_accuracy", "accuracy_by_market", "accuracy_by_confidence", "roi"):
            if key in perf_dash:
                accuracy[key] = perf_dash[key]
        if accuracy:
            result["dashboard_accuracy"] = accuracy

    return json.dumps(result, default=str)


def _tool_get_match_context(args: dict) -> str:
    home = args.get("home_team", "")
    away = args.get("away_team", "")
    h = _resolve_team(home)
    a = _resolve_team(away)
    if not h or not a:
        return json.dumps({"error": f"Could not resolve teams: {home}, {away}"})

    result: dict[str, Any] = {"home_team": h, "away_team": a}

    # Referee
    refs = load_json_safe(UPCOMING_DIR / "referees.json")
    if isinstance(refs, dict):
        for match_key, ref_info in refs.items():
            if h in match_key and a in match_key:
                result["referee"] = ref_info
                break
    elif isinstance(refs, list):
        for r in refs:
            if r.get("home_team") == h and r.get("away_team") == a:
                result["referee"] = r
                break

    # Weather
    weather = load_json_safe(UPCOMING_DIR / "weather.json")
    if isinstance(weather, dict):
        for match_key, w_info in weather.items():
            if h in str(match_key) and a in str(match_key):
                result["weather"] = w_info
                break
    elif isinstance(weather, list):
        for w in weather:
            if w.get("home_team") == h and w.get("away_team") == a:
                result["weather"] = w
                break

    # Odds movement
    odds_mv = load_json_safe(UPCOMING_DIR / "odds_movement.json")
    if isinstance(odds_mv, dict):
        for match_key, mv_info in odds_mv.items():
            if h in str(match_key) and a in str(match_key):
                result["odds_movement"] = mv_info
                break
    elif isinstance(odds_mv, list):
        for mv in odds_mv:
            if mv.get("home_team") == h and mv.get("away_team") == a:
                result["odds_movement"] = mv
                break

    # Cross-market signals narrative
    cms = load_json_safe(UPCOMING_DIR / "cross_market_signals.json")
    cms_matches = cms.get("matches", {})
    if isinstance(cms_matches, dict):
        for key, m in cms_matches.items():
            if isinstance(m, dict) and h in key and a in key:
                result["cross_market_signals"] = m
                break
    elif isinstance(cms_matches, list):
        for m in cms_matches:
            if isinstance(m, dict) and m.get("home_team") == h and m.get("away_team") == a:
                result["cross_market_signals"] = m
                break

    # Full odds from all bookmakers
    odds_full = load_json_safe(UPCOMING_DIR / "odds_full.json")
    for match_key, odds_data in odds_full.get("matches", {}).items():
        if h in match_key and a in match_key:
            # Compact: just best odds per outcome + bookmaker list
            bookmakers = odds_data.get("bookmakers", odds_data.get("odds", {}))
            if isinstance(bookmakers, dict):
                result["odds_all_bookmakers"] = bookmakers
            elif isinstance(bookmakers, list):
                result["odds_all_bookmakers"] = bookmakers[:10]  # cap to avoid token bloat
            break

    return json.dumps(result, default=str)


def _tool_get_match_scorers(args: dict) -> str:
    """Get ranked goalscorer candidates for a match with probabilities and season stats."""
    home = args.get("home_team", "")
    away = args.get("away_team", "")
    h = _resolve_team(home)
    a = _resolve_team(away)
    if not h or not a:
        return json.dumps({"error": f"Could not resolve teams: {home}, {away}"})

    result: dict[str, Any] = {"home_team": h, "away_team": a, "scorers": []}

    # 1. Get anytime goal probabilities from player_props.json
    props = load_json_safe(UPCOMING_DIR / "player_props.json")
    props_by_name: dict[str, dict] = {}
    for match_key, match_data in props.get("matches", {}).items():
        if h in match_key and a in match_key:
            for p in match_data.get("players", []):
                props_by_name[p.get("name", "")] = {
                    "anytime_goal_prob": p.get("anytime_goal_prob"),
                    "anytime_fair_odds": p.get("anytime_fair_odds"),
                    "two_plus_prob": p.get("two_plus_prob"),
                    "first_goal_prob": p.get("first_goal_prob"),
                    "assist_prob": p.get("assist_prob"),
                    "shots_expected": p.get("shots_expected"),
                    "sot_expected": p.get("sot_expected"),
                    "xg_per_90": p.get("xg_per_90"),
                    "xa_per_90": p.get("xa_per_90"),
                    "to_be_carded_prob": p.get("to_be_carded_prob"),
                    "team": p.get("team"),
                }
            break

    # 2. Get season stats from parquet for both squads
    season_stats: dict[str, dict] = {}
    try:
        import pandas as pd
        df = pd.read_parquet(DATA_DIR / "parsed" / "player_stats.parquet")
        current = df[df["season"] == "2025-2026"] if "season" in df.columns else df[df["match_date"] >= "2025-08-01"]
        for team in [h, a]:
            team_df = current[current["team"] == team]
            agg = team_df.groupby("player").agg(
                goals=("goals", "sum"),
                assists=("assists", "sum"),
                xg=("xg", "sum"),
                minutes=("minutes", "sum"),
                matches=("player", "count"),
                shots=("shots", "sum"),
                shots_on_target=("shots_on_target", "sum"),
            )
            for name, row in agg.iterrows():
                mins = row["minutes"]
                per90 = mins / 90 if mins > 0 else 1
                season_stats[name] = {
                    "team": team,
                    "goals": int(row["goals"]),
                    "assists": int(row["assists"]),
                    "xg": round(row["xg"], 2) if row["xg"] == row["xg"] else 0,
                    "minutes": int(mins),
                    "matches": int(row["matches"]),
                    "goals_per_90": round(row["goals"] / per90, 2),
                    "shots_per_90": round(row["shots"] / per90, 1) if row["shots"] == row["shots"] else 0,
                }
                # Recent form: last 3 matches goals
                player_recent = team_df[team_df["player"] == name].sort_values("match_date", ascending=False).head(3)
                season_stats[name]["last_3_goals"] = int(player_recent["goals"].sum())
                season_stats[name]["last_3_shots"] = int(player_recent["shots"].sum()) if "shots" in player_recent else 0
    except Exception as e:
        log.warning("Match scorers parquet lookup failed: %s", e)

    # 3. Merge and rank
    all_players: dict[str, dict] = {}

    # Add players from props (they're the ones expected to play)
    for name, p_data in props_by_name.items():
        all_players[name] = {**p_data, **(season_stats.get(name, {}))}

    # Add top season scorers who might not be in props
    for name, s_data in season_stats.items():
        if name not in all_players and s_data.get("goals", 0) >= 3:
            all_players[name] = s_data

    # Sort by anytime goal probability (if available), then by season goals/90
    scorers = sorted(
        [{"name": k, **v} for k, v in all_players.items()],
        key=lambda x: (x.get("anytime_goal_prob") or 0, x.get("goals_per_90", 0)),
        reverse=True,
    )

    # Return top scorers per team
    home_scorers = [s for s in scorers if s.get("team") == h][:8]
    away_scorers = [s for s in scorers if s.get("team") == a][:8]

    result["home_scorers"] = home_scorers
    result["away_scorers"] = away_scorers
    result["top_scorer_overall"] = scorers[0]["name"] if scorers else None

    # Player prop odds if available
    prop_odds = load_json_safe(UPCOMING_DIR / "player_prop_odds.json")
    for match_key, odds_data in prop_odds.get("matches", {}).items():
        if h in match_key and a in match_key:
            result["bookmaker_goalscorer_odds"] = odds_data.get("anytime_goalscorer", odds_data.get("odds", {}))
            break

    return json.dumps(result, default=str)


def _tool_place_bet(args: dict) -> str:
    """Place a bet: find it in the unified bet slip and record to bet journal."""
    from scripts.betting.bet_journal import add_bet, get_pending_bets

    action = args.get("action", "place").lower()
    match_query = args.get("match", "").strip()
    market_query = args.get("market", "").strip()
    selection_query = args.get("selection", "").strip()
    custom_odds = args.get("odds")
    custom_stake = args.get("stake")

    # --- place_all: bulk place all selected bets ---
    if action == "place_all":
        slip = load_json_safe(UPCOMING_DIR / "unified_bet_slip.json")
        bets = slip.get("selected_bets", [])
        if not bets:
            return json.dumps({"error": "No value bets in the current bet slip."})

        placed = []
        skipped = []
        for b in bets:
            bet_data = {
                "match": b.get("match", ""),
                "date": b.get("date", ""),
                "market": b.get("market", ""),
                "selection": b.get("selection", ""),
                "model_prob": b.get("model_prob"),
                "sharp_implied_prob": b.get("sharp_implied_prob"),
                "edge_pct": b.get("edge_pct"),
                "odds": b.get("best_odds"),
                "bookmaker": b.get("best_bookmaker"),
                "avg_odds": b.get("avg_odds"),
                "pinnacle_odds": b.get("pinnacle_odds"),
                "stake": b.get("stake_amount"),
                "confidence": b.get("confidence_tier"),
                "placed_at": datetime.now().isoformat(),
                "pipeline_status": "advisor_placed",
            }
            bet_id = add_bet(bet_data)
            if bet_id:
                placed.append({
                    "bet_id": bet_id,
                    "match": b.get("match"),
                    "market": b.get("market"),
                    "selection": b.get("selection"),
                    "odds": b.get("best_odds"),
                    "stake": b.get("stake_amount"),
                    "edge_pct": b.get("edge_pct"),
                })
            else:
                skipped.append({"match": b.get("match"), "market": b.get("market"),
                                "reason": "Edge cap exceeded or invalid"})

        return json.dumps({
            "action": "place_all",
            "placed": placed,
            "skipped": skipped,
            "total_placed": len(placed),
            "total_skipped": len(skipped),
        }, default=str)

    # --- Single bet placement ---
    if not match_query:
        return json.dumps({"error": "Please specify which match to bet on."})

    # Resolve team name from query
    resolved = _resolve_team(match_query)

    # Search the unified bet slip for matching bets
    slip = load_json_safe(UPCOMING_DIR / "unified_bet_slip.json")
    all_bets = slip.get("selected_bets", [])

    # Also check handicap + O/U specialty bets
    hcap = load_json_safe(UPCOMING_DIR / "handicap_bets.json")
    ou = load_json_safe(UPCOMING_DIR / "over_under_bets.json")

    # Find bets matching the team/match query
    q_lower = match_query.lower()
    candidates = []
    for b in all_bets:
        match_name = (b.get("match", "") or "").lower()
        if resolved and resolved.lower() in match_name:
            candidates.append(b)
        elif q_lower in match_name:
            candidates.append(b)

    if not candidates:
        # No match in bet slip — maybe user wants a custom bet
        return json.dumps({
            "error": f"No recommended bet found matching '{match_query}'. "
                     "Available bets are in the current value bet slip. "
                     "Call get_value_bets to see what's available, or ask the user for specific match, market, odds, and stake to place a custom bet.",
            "suggestion": "call get_value_bets first",
        })

    # Filter by market if specified
    if market_query:
        mq = market_query.lower().replace("/", "").replace(" ", "")
        filtered = []
        for b in candidates:
            bm = (b.get("market", "") or "").lower().replace("/", "").replace(" ", "")
            if mq in bm or bm in mq:
                filtered.append(b)
        if filtered:
            candidates = filtered

    # Filter by selection if specified
    if selection_query:
        sq = selection_query.lower()
        filtered = [b for b in candidates if sq in (b.get("selection", "") or "").lower()]
        if filtered:
            candidates = filtered

    # If still multiple, return all for Claude to present
    if len(candidates) > 1 and not market_query:
        return json.dumps({
            "status": "multiple_matches",
            "message": f"Found {len(candidates)} bets for '{match_query}'. Specify the market (e.g., 'O/U 1.5', 'DC', '1X2') to narrow down.",
            "available": [
                {
                    "match": b.get("match"),
                    "market": b.get("market"),
                    "selection": b.get("selection"),
                    "odds": b.get("best_odds"),
                    "edge_pct": b.get("edge_pct"),
                    "stake": b.get("stake_amount"),
                }
                for b in candidates
            ],
        }, default=str)

    # Place the bet (first match or only match)
    b = candidates[0]
    bet_data = {
        "match": b.get("match", ""),
        "date": b.get("date", ""),
        "market": b.get("market", ""),
        "selection": b.get("selection", ""),
        "model_prob": b.get("model_prob"),
        "sharp_implied_prob": b.get("sharp_implied_prob"),
        "edge_pct": b.get("edge_pct"),
        "odds": custom_odds or b.get("best_odds"),
        "bookmaker": b.get("best_bookmaker"),
        "avg_odds": b.get("avg_odds"),
        "pinnacle_odds": b.get("pinnacle_odds"),
        "stake": custom_stake or b.get("stake_amount"),
        "confidence": b.get("confidence_tier"),
        "placed_at": datetime.now().isoformat(),
        "pipeline_status": "advisor_placed",
    }
    bet_id = add_bet(bet_data)
    if not bet_id:
        return json.dumps({"error": "Bet rejected — edge cap exceeded or invalid data."})

    return json.dumps({
        "action": "placed",
        "bet_id": bet_id,
        "match": b.get("match"),
        "date": b.get("date"),
        "market": b.get("market"),
        "selection": b.get("selection"),
        "odds": bet_data["odds"],
        "stake": bet_data["stake"],
        "edge_pct": b.get("edge_pct"),
        "bookmaker": b.get("best_bookmaker"),
        "confidence": b.get("confidence_tier"),
        "model_prob": b.get("model_prob"),
    }, default=str)


def _tool_manage_bets(args: dict) -> str:
    """List, cancel, or update pending bets in the journal."""
    from scripts.betting.bet_journal import (
        _load_journal, _save_journal, get_pending_bets, get_journal_stats
    )

    action = args.get("action", "list").lower()

    # --- List pending bets ---
    if action == "list":
        pending = get_pending_bets()
        stats = get_journal_stats()
        return json.dumps({
            "action": "list",
            "pending_bets": [
                {
                    "bet_id": b.get("bet_id"),
                    "match": b.get("match"),
                    "date": b.get("date"),
                    "market": b.get("market"),
                    "selection": b.get("selection"),
                    "odds": b.get("odds"),
                    "stake": b.get("stake"),
                    "edge_pct": b.get("edge_pct"),
                    "confidence": b.get("confidence"),
                    "placed_at": b.get("placed_at"),
                    "status": b.get("status"),
                }
                for b in pending
            ],
            "total_pending": len(pending),
            "journal_stats": {
                "total_bets": stats.get("total_bets"),
                "pending": stats.get("pending"),
                "settled": stats.get("settled"),
                "total_profit": stats.get("total_profit"),
                "roi_pct": stats.get("roi_pct"),
            },
        }, default=str)

    # --- Cancel a pending bet ---
    if action == "cancel":
        match_query = args.get("match", "").strip().lower()
        market_query = args.get("market", "").strip().lower()
        bet_id_query = args.get("bet_id", "").strip()

        if not match_query and not bet_id_query:
            return json.dumps({"error": "Specify which bet to cancel (match name or bet_id)."})

        journal = _load_journal()
        cancelled = []

        for bid, bet in journal["bets"].items():
            if bet.get("status") not in ("pending", "superseded"):
                continue

            # Match by bet_id
            if bet_id_query and bet_id_query in bid:
                bet["status"] = "void"
                bet["settled_at"] = datetime.now().isoformat()
                bet["profit"] = 0
                cancelled.append({"bet_id": bid, "match": bet.get("match"),
                                   "market": bet.get("market")})
                continue

            # Match by team name + optional market
            match_name = (bet.get("match", "") or "").lower()
            resolved = _resolve_team(match_query)
            team_match = (resolved and resolved.lower() in match_name) or match_query in match_name

            if team_match:
                if market_query:
                    bm = (bet.get("market", "") or "").lower().replace("/", "").replace(" ", "")
                    mq = market_query.replace("/", "").replace(" ", "")
                    if mq not in bm and bm not in mq:
                        continue
                bet["status"] = "void"
                bet["settled_at"] = datetime.now().isoformat()
                bet["profit"] = 0
                cancelled.append({"bet_id": bid, "match": bet.get("match"),
                                   "market": bet.get("market"),
                                   "selection": bet.get("selection")})

        if cancelled:
            _save_journal(journal)

        return json.dumps({
            "action": "cancel",
            "cancelled": cancelled,
            "total_cancelled": len(cancelled),
            "message": f"Cancelled {len(cancelled)} bet(s)." if cancelled else "No matching pending bets found.",
        }, default=str)

    # --- Update stake/odds on a pending bet ---
    if action == "update":
        match_query = args.get("match", "").strip().lower()
        new_odds = args.get("odds")
        new_stake = args.get("stake")

        if not match_query:
            return json.dumps({"error": "Specify which bet to update (match name)."})
        if not new_odds and not new_stake:
            return json.dumps({"error": "Specify what to update: odds and/or stake."})

        journal = _load_journal()
        updated = []

        for bid, bet in journal["bets"].items():
            if bet.get("status") not in ("pending", "superseded"):
                continue

            match_name = (bet.get("match", "") or "").lower()
            resolved = _resolve_team(match_query)
            if not ((resolved and resolved.lower() in match_name) or match_query in match_name):
                continue

            if new_odds is not None:
                bet["odds"] = float(new_odds)
            if new_stake is not None:
                bet["stake"] = float(new_stake)
            bet["updated_at"] = datetime.now().isoformat()
            updated.append({
                "bet_id": bid, "match": bet.get("match"),
                "market": bet.get("market"), "selection": bet.get("selection"),
                "odds": bet.get("odds"), "stake": bet.get("stake"),
            })

        if updated:
            _save_journal(journal)

        return json.dumps({
            "action": "update",
            "updated": updated,
            "total_updated": len(updated),
            "message": f"Updated {len(updated)} bet(s)." if updated else "No matching pending bets found.",
        }, default=str)

    return json.dumps({"error": f"Unknown action '{action}'. Use: place, place_all, list, cancel, update."})


def _tool_settle_bets(args: dict) -> str:
    """Trigger bet settlement: fetch results, settle bets, update bankroll."""
    import requests as _requests
    try:
        # Use the existing /api/settle endpoint (runs in background thread)
        resp = _requests.post("http://127.0.0.1:5001/api/settle", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if not data.get("ok"):
                return json.dumps({"status": "blocked", "reason": data.get("message", "Settlement already running")})

            # Poll for completion (max 60 seconds)
            import time as _time
            for _ in range(30):
                _time.sleep(2)
                try:
                    status_resp = _requests.get("http://127.0.0.1:5001/api/settle/status", timeout=5)
                    status = status_resp.json()
                    if status.get("status") in ("done", "error"):
                        return json.dumps(status, default=str)
                except Exception:
                    continue

            return json.dumps({"status": "timeout", "message": "Settlement still running. Check back shortly."})
        else:
            return json.dumps({"error": f"Settle endpoint returned {resp.status_code}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to trigger settlement: {e}"})


def _tool_query_history(args: dict) -> str:
    """Query historical match data for patterns and trends."""
    import pandas as pd

    team_query = args.get("team", "").strip()
    venue = args.get("venue", "all").lower()
    opponent_query = args.get("opponent", "").strip()
    seasons = args.get("seasons", 3)
    situation = args.get("situation", "").lower().strip()
    referee_query = args.get("referee", "").strip()
    market = args.get("market", "").lower().strip()

    if not team_query:
        return json.dumps({"error": "Please specify a team to query."})

    team = _resolve_team(team_query)
    if not team:
        return json.dumps({"error": f"Unknown team: '{team_query}'"})

    opponent = _resolve_team(opponent_query) if opponent_query else None

    # Load matches
    matches_path = DATA_DIR / "parsed" / "matches.parquet"
    if not matches_path.exists():
        return json.dumps({"error": "matches.parquet not found."})

    try:
        df = pd.read_parquet(matches_path)
    except Exception as e:
        return json.dumps({"error": f"Failed to load matches: {e}"})

    # Filter by team and venue
    if venue == "home":
        mask = df["home_team"] == team
    elif venue == "away":
        mask = df["away_team"] == team
    else:
        mask = (df["home_team"] == team) | (df["away_team"] == team)

    df = df[mask].copy()

    if df.empty:
        return json.dumps({"error": f"No matches found for {team}."})

    # Filter by opponent
    if opponent:
        df = df[(df["home_team"] == opponent) | (df["away_team"] == opponent)]
        if df.empty:
            return json.dumps({"error": f"No matches found for {team} vs {opponent}."})

    # Filter by seasons (last N)
    if isinstance(seasons, int) and seasons > 0:
        all_seasons = sorted(df["season"].dropna().unique())
        recent = all_seasons[-seasons:]
        df = df[df["season"].isin(recent)]

    # Filter by referee
    if referee_query:
        ref_lower = referee_query.lower()
        df = df[df["referee"].fillna("").str.lower().str.contains(ref_lower, na=False)]
        if df.empty:
            return json.dumps({"error": f"No matches with referee matching '{referee_query}'."})

    # Situational filters — need features.parquet
    if situation:
        features_path = DATA_DIR / "features" / "features.parquet"
        if features_path.exists():
            try:
                feat_cols = ["match_id"]
                if situation in ("derby", "derbies"):
                    feat_cols.append("is_derby")
                elif situation in ("midweek",):
                    feat_cols.append("is_midweek")
                elif situation in ("short_rest", "short rest"):
                    feat_cols.extend(["home_short_rest", "away_short_rest"])
                elif situation in ("long_rest", "long rest"):
                    feat_cols.extend(["home_rest_days", "away_rest_days"])

                df_feat = pd.read_parquet(features_path, columns=feat_cols)
                df = df.merge(df_feat, on="match_id", how="inner")

                if situation in ("derby", "derbies"):
                    df = df[df["is_derby"] == 1]
                elif situation == "midweek":
                    df = df[df["is_midweek"] == 1]
                elif situation in ("short_rest", "short rest"):
                    if venue == "home":
                        df = df[df["home_short_rest"] == 1]
                    elif venue == "away":
                        df = df[df["away_short_rest"] == 1]
                    else:
                        df = df[(df["home_short_rest"] == 1) | (df["away_short_rest"] == 1)]
                elif situation in ("long_rest", "long rest"):
                    if venue == "home":
                        df = df[df["home_rest_days"] >= 7]
                    elif venue == "away":
                        df = df[df["away_rest_days"] >= 7]
                    else:
                        df = df[(df["home_rest_days"] >= 7) | (df["away_rest_days"] >= 7)]
            except Exception as e:
                log.warning("Situational filter failed: %s", e)

        if df.empty:
            return json.dumps({"error": f"No {situation} matches found for {team}."})

    # Ensure we have scores
    df = df.dropna(subset=["home_score", "away_score"])
    if df.empty:
        return json.dumps({"error": f"No completed matches found for {team} with given filters."})

    # --- Compute stats ---
    # Normalize: compute team's goals scored/conceded regardless of home/away
    is_home = df["home_team"] == team
    goals_scored = pd.Series(
        [r["home_score"] if h else r["away_score"] for _, r, h in
         zip(range(len(df)), df.to_dict("records"), is_home)],
        dtype=float,
    )
    goals_conceded = pd.Series(
        [r["away_score"] if h else r["home_score"] for _, r, h in
         zip(range(len(df)), df.to_dict("records"), is_home)],
        dtype=float,
    )
    total_goals = goals_scored + goals_conceded

    wins = (goals_scored > goals_conceded).sum()
    draws = (goals_scored == goals_conceded).sum()
    losses = (goals_scored < goals_conceded).sum()
    n = len(df)

    result = {
        "team": team,
        "filters": {
            "venue": venue,
            "opponent": opponent,
            "seasons": seasons,
            "situation": situation or None,
            "referee": referee_query or None,
        },
        "matches": n,
        "record": {
            "wins": int(wins),
            "draws": int(draws),
            "losses": int(losses),
            "win_rate": round(wins / n * 100, 1),
            "draw_rate": round(draws / n * 100, 1),
            "loss_rate": round(losses / n * 100, 1),
            "points_per_game": round((wins * 3 + draws) / n, 2),
        },
        "goals": {
            "avg_scored": round(goals_scored.mean(), 2),
            "avg_conceded": round(goals_conceded.mean(), 2),
            "avg_total": round(total_goals.mean(), 2),
            "total_scored": int(goals_scored.sum()),
            "total_conceded": int(goals_conceded.sum()),
            "clean_sheets": int((goals_conceded == 0).sum()),
            "clean_sheet_rate": round((goals_conceded == 0).sum() / n * 100, 1),
            "failed_to_score": int((goals_scored == 0).sum()),
            "failed_to_score_rate": round((goals_scored == 0).sum() / n * 100, 1),
        },
        "markets": {
            "over_2.5_rate": round((total_goals > 2.5).sum() / n * 100, 1),
            "under_2.5_rate": round((total_goals < 2.5).sum() / n * 100, 1),
            "over_1.5_rate": round((total_goals > 1.5).sum() / n * 100, 1),
            "over_3.5_rate": round((total_goals > 3.5).sum() / n * 100, 1),
            "btts_rate": round(((goals_scored > 0) & (goals_conceded > 0)).sum() / n * 100, 1),
            "btts_no_rate": round(((goals_scored == 0) | (goals_conceded == 0)).sum() / n * 100, 1),
        },
        "scoring_patterns": {
            "scored_first_rate": None,  # would need event data
            "avg_ht_goals": None,
        },
    }

    # Half-time stats if available
    if "home_ht_score" in df.columns and "away_ht_score" in df.columns:
        ht_df = df.dropna(subset=["home_ht_score", "away_ht_score"])
        if len(ht_df) > 0:
            ht_scored = pd.Series(
                [r["home_ht_score"] if h else r["away_ht_score"] for _, r, h in
                 zip(range(len(ht_df)), ht_df.to_dict("records"),
                     ht_df["home_team"] == team)],
                dtype=float,
            )
            ht_conceded = pd.Series(
                [r["away_ht_score"] if h else r["home_ht_score"] for _, r, h in
                 zip(range(len(ht_df)), ht_df.to_dict("records"),
                     ht_df["home_team"] == team)],
                dtype=float,
            )
            ht_total = ht_scored + ht_conceded
            result["scoring_patterns"]["avg_ht_goals"] = round(ht_total.mean(), 2)
            result["scoring_patterns"]["ht_over_0.5_rate"] = round(
                (ht_total > 0.5).sum() / len(ht_df) * 100, 1)
            result["scoring_patterns"]["ht_over_1.5_rate"] = round(
                (ht_total > 1.5).sum() / len(ht_df) * 100, 1)

    # Match stats if available
    stat_cols = {
        "avg_possession": ("home_possession", "away_possession"),
        "avg_shots_on_target": ("home_shots_on_target", "away_shots_on_target"),
        "avg_corners": ("home_corners", "away_corners"),
        "avg_fouls": ("home_fouls", "away_fouls"),
        "avg_xg": ("home_xg", "away_xg"),
    }
    match_stats = {}
    for stat_name, (home_col, away_col) in stat_cols.items():
        if home_col in df.columns and away_col in df.columns:
            vals = pd.Series(
                [r[home_col] if h else r[away_col] for _, r, h in
                 zip(range(len(df)), df.to_dict("records"), is_home)],
                dtype=float,
            ).dropna()
            if len(vals) > 0:
                match_stats[stat_name] = round(vals.mean(), 2)
    if match_stats:
        result["match_stats"] = match_stats

    # Recent form (last 5 matches in the filtered set)
    recent = df.sort_values("match_date", ascending=False).head(5)
    result["recent_matches"] = []
    for _, row in recent.iterrows():
        is_h = row["home_team"] == team
        gs = row["home_score"] if is_h else row["away_score"]
        gc = row["away_score"] if is_h else row["home_score"]
        opp = row["away_team"] if is_h else row["home_team"]
        res = "W" if gs > gc else ("D" if gs == gc else "L")
        result["recent_matches"].append({
            "date": str(row.get("match_date", ""))[:10],
            "opponent": opp,
            "venue": "H" if is_h else "A",
            "score": f"{int(gs)}-{int(gc)}",
            "result": res,
        })

    # Specific market deep dive if requested
    if market:
        if "over" in market or "under" in market or "goal" in market:
            # Goals distribution
            dist = {}
            for g in range(6):
                dist[f"exactly_{g}_goals"] = round((total_goals == g).sum() / n * 100, 1)
            dist["6+_goals"] = round((total_goals >= 6).sum() / n * 100, 1)
            result["goals_distribution"] = dist
        elif "btts" in market:
            # BTTS detail by venue
            result["btts_detail"] = {
                "btts_yes_matches": int(((goals_scored > 0) & (goals_conceded > 0)).sum()),
                "btts_no_matches": int(((goals_scored == 0) | (goals_conceded == 0)).sum()),
            }
        elif "corner" in market and "avg_corners" in match_stats:
            # Corner lines
            if "home_corners" in df.columns and "away_corners" in df.columns:
                team_corners = pd.Series(
                    [r["home_corners"] if h else r["away_corners"] for _, r, h in
                     zip(range(len(df)), df.to_dict("records"), is_home)],
                    dtype=float,
                ).dropna()
                if len(team_corners) > 0:
                    result["corner_detail"] = {
                        "avg_team_corners": round(team_corners.mean(), 2),
                        "over_4.5_team_corners": round((team_corners > 4.5).sum() / len(team_corners) * 100, 1),
                        "over_5.5_team_corners": round((team_corners > 5.5).sum() / len(team_corners) * 100, 1),
                    }

    return json.dumps(result, default=str)


def _tool_build_parlay(args: dict) -> str:
    """Build a parlay (accumulator) from value bet selections."""
    from scripts.betting.bet_journal import _load_journal, _save_journal

    matches_input = args.get("matches", [])
    auto_best = args.get("auto_best", 0)
    custom_stake = args.get("stake")
    place = args.get("place", False)

    slip = load_json_safe(UPCOMING_DIR / "unified_bet_slip.json")
    all_bets = slip.get("selected_bets", [])

    if not all_bets:
        return json.dumps({"error": "No value bets in the current bet slip. Run the pipeline first."})

    # --- Auto-select best N bets by edge ---
    if auto_best and auto_best > 0:
        # Sort by edge descending, pick top N, max 1 per match to avoid correlation
        sorted_bets = sorted(all_bets, key=lambda b: b.get("edge_pct", 0), reverse=True)
        seen_matches = set()
        selected = []
        for b in sorted_bets:
            match_name = b.get("match", "")
            if match_name in seen_matches:
                continue  # skip correlated legs
            seen_matches.add(match_name)
            selected.append(b)
            if len(selected) >= auto_best:
                break
    else:
        # --- Manual selection: match user-provided matches to slip ---
        if not matches_input:
            return json.dumps({
                "error": "Specify matches for the parlay. Either provide a list of team names, "
                         "or use auto_best=N to auto-pick the top N bets by edge.",
                "available_bets": [
                    {"match": b.get("match"), "market": b.get("market"),
                     "selection": b.get("selection"), "odds": b.get("best_odds"),
                     "edge_pct": b.get("edge_pct")}
                    for b in all_bets
                ],
            }, default=str)

        selected = []
        not_found = []
        for query in matches_input:
            q_lower = query.lower().strip()
            resolved = _resolve_team(q_lower)
            found = None
            for b in all_bets:
                match_name = (b.get("match", "") or "").lower()
                if (resolved and resolved.lower() in match_name) or q_lower in match_name:
                    found = b
                    break
            if found:
                selected.append(found)
            else:
                not_found.append(query)

        if not_found:
            return json.dumps({
                "error": f"Could not find bets for: {', '.join(not_found)}",
                "found": len(selected),
                "not_found": not_found,
                "available_bets": [
                    {"match": b.get("match"), "market": b.get("market"),
                     "selection": b.get("selection")}
                    for b in all_bets
                ],
            }, default=str)

    if len(selected) < 2:
        return json.dumps({"error": "A parlay needs at least 2 legs. Only found 1 matching bet."})

    # --- Calculate parlay math ---
    combined_odds = 1.0
    combined_prob = 1.0
    legs = []
    match_names_seen = set()
    has_correlation = False

    for b in selected:
        odds = b.get("best_odds", 1.0)
        prob = b.get("model_prob", 0.5)
        match_name = b.get("match", "")

        if match_name in match_names_seen:
            has_correlation = True
        match_names_seen.add(match_name)

        combined_odds *= odds
        combined_prob *= prob

        legs.append({
            "match": match_name,
            "date": b.get("date"),
            "market": b.get("market"),
            "selection": b.get("selection"),
            "odds": odds,
            "model_prob": prob,
            "edge_pct": b.get("edge_pct"),
            "bookmaker": b.get("best_bookmaker"),
            "confidence": b.get("confidence_tier"),
        })

    combined_odds = round(combined_odds, 2)
    combined_prob = round(combined_prob, 4)
    implied_prob = round(1.0 / combined_odds, 4) if combined_odds > 0 else 0
    ev = round(combined_prob * combined_odds - 1, 4)
    edge_pct = round((combined_prob - implied_prob) / implied_prob * 100, 2) if implied_prob > 0 else 0

    # Kelly stake for parlay
    b_val = combined_odds - 1
    kelly_raw = (b_val * combined_prob - (1 - combined_prob)) / b_val if b_val > 0 else 0
    kelly_fraction = 0.10  # 10% Kelly for parlays (conservative)
    kelly_adj = max(kelly_raw * kelly_fraction, 0)

    bankroll = _get_bankroll()
    balance = bankroll.get("current_bankroll", 1000)
    kelly_stake = round(kelly_adj * balance, 2)

    # Cap stake at 2% bankroll for parlays
    max_stake = round(balance * 0.02, 2)
    suggested_stake = min(kelly_stake, max_stake) if kelly_stake > 0 else 0
    actual_stake = custom_stake if custom_stake else suggested_stake

    potential_return = round(actual_stake * combined_odds, 2)
    potential_profit = round(potential_return - actual_stake, 2)

    result = {
        "parlay": {
            "legs": legs,
            "num_legs": len(legs),
            "combined_odds": combined_odds,
            "combined_probability": combined_prob,
            "implied_probability": implied_prob,
            "ev": ev,
            "edge_pct": edge_pct,
            "kelly_raw": round(kelly_raw, 4),
            "kelly_adjusted": round(kelly_adj, 4),
            "suggested_stake": suggested_stake,
            "actual_stake": actual_stake,
            "potential_return": potential_return,
            "potential_profit": potential_profit,
            "bankroll": balance,
        },
        "warnings": [],
    }

    if has_correlation:
        result["warnings"].append(
            "CORRELATED LEGS: Multiple bets from the same match. "
            "Combined probability assumes independence — actual probability may differ."
        )
    if combined_prob < 0.10:
        result["warnings"].append(
            f"Low combined probability ({combined_prob:.1%}). "
            "Parlays with many legs are high-variance lottery tickets."
        )
    if ev < 0:
        result["warnings"].append(
            f"NEGATIVE EV ({ev:+.4f}). This parlay is -EV — you're expected to lose money long-term."
        )

    # --- Place the parlay in the journal if requested ---
    if place and actual_stake > 0:
        journal = _load_journal()
        leg_ids = [f"{l['date']}_{l['match']}_{l['market']}_{l['selection']}"
                   .replace(" ", "_") for l in legs]
        parlay_id = "PARLAY_" + "_".join(sorted(set(
            l["match"].split(" vs ")[0].replace(" ", "") for l in legs
        ))) + f"_{legs[0]['date']}"

        entry = {
            "bet_id": parlay_id,
            "match": " + ".join(l["match"] for l in legs),
            "date": min(l["date"] for l in legs if l.get("date")),
            "market": "PARLAY",
            "selection": f"{len(legs)}-leg parlay",
            "model_prob": combined_prob,
            "sharp_implied_prob": implied_prob,
            "edge_pct": edge_pct,
            "odds": combined_odds,
            "bookmaker": "Multi",
            "avg_odds": None,
            "pinnacle_odds": None,
            "stake": actual_stake,
            "confidence": "PARLAY",
            "factors": [f"{l['match']}: {l['selection']} @{l['odds']}" for l in legs],
            "status": "pending",
            "result_score": None,
            "profit": None,
            "closing_odds": None,
            "clv_pct": None,
            "placed_at": datetime.now().isoformat(),
            "settled_at": None,
            "pipeline_status": "advisor_parlay",
            "legs": legs,
        }
        journal["bets"][parlay_id] = entry
        _save_journal(journal)
        result["placed"] = True
        result["parlay_id"] = parlay_id
    else:
        result["placed"] = False
        result["note"] = "Parlay calculated but NOT placed. Say 'place it' or set place=true to record it."

    return json.dumps(result, default=str)


# Tool dispatch table
TOOL_HANDLERS = {
    "get_match_prediction": _tool_get_match_prediction,
    "get_team_detail": _tool_get_team_detail,
    "get_player_stats": _tool_get_player_stats,
    "get_h2h": _tool_get_h2h,
    "get_value_bets": _tool_get_value_bets,
    "get_live_matches": _tool_get_live_matches,
    "get_bankroll_status": _tool_get_bankroll_status,
    "get_betting_performance": _tool_get_betting_performance,
    "get_match_context": _tool_get_match_context,
    "get_results": _tool_get_results,
    "get_match_scorers": _tool_get_match_scorers,
    "get_match_players": _tool_get_match_players,
    "settle_bets": _tool_settle_bets,
    "place_bet": _tool_place_bet,
    "manage_bets": _tool_manage_bets,
    "build_parlay": _tool_build_parlay,
    "query_history": _tool_query_history,
}


# ---------------------------------------------------------------------------
# Tool definitions for Claude
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_match_prediction",
        "description": "Get the full prediction for a specific match including probabilities, value bets, player analysis narrative, and sentiment. Use this when asked about a specific match.",
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {"type": "string", "description": "Home team name"},
                "away_team": {"type": "string", "description": "Away team name"},
            },
            "required": ["home_team", "away_team"],
        },
    },
    {
        "name": "get_team_detail",
        "description": "Get team standings, form, home/away splits. Use when asked about a specific team's performance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "description": "Team name"},
            },
            "required": ["team"],
        },
    },
    {
        "name": "get_player_stats",
        "description": "Get comprehensive player info: ON-PITCH STATS + WAGE/CONTRACT/MARKET VALUE. Returns season totals, per-90 rates, recent form (last 5 matches with Sofascore ratings), team ranking, upcoming props, the full team sheet from the player's most recent match, match events, and availability/injury detection — AND the player's 2026-27 market value, contract length (years remaining / expiry), and SALARY (yearly/monthly/weekly), AND Understat MULTI-SEASON history with xG over/under-performance (goals_minus_xg: positive = clinical finisher, negative = wasteful/unlucky — e.g. Kean 8 goals on 15.4 xG = badly underperforming; plus non-penalty np_xg and buildup involvement xg_chain/xg_buildup), AND season advanced Sofascore stats (pass accuracy, touches, tackles+interceptions per 90, duel win %, big chances created/missed, errors leading to goals, GK saves) to profile the player's ROLE and physical/defensive/creative contribution beyond goals, AND their FIFA WORLD CUP 2026 performance (matches, minutes, goals, ratings — just played Jun-Jul 2026, so it shapes fitness/form entering the new club season). NOTE: salary is a Capology ESTIMATE of FIXED gross pay (excludes bonuses), NOT an official figure — present it as an estimate, never as fact. Understat covers only Serie A + Premier League, so a Ligue 1/Bundesliga season may be absent — never invent missing seasons. Use this tool for ANY player question: performance, form, 'tell me about X', 'how much does X earn', 'how long is X's contract', 'is he any good', or a scouting/analysis take. Fuzzy name matching supported.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player": {"type": "string", "description": "Player name (partial ok, e.g. 'lautaro' or 'pulisic')"},
                "team": {"type": "string", "description": "Optional team name to disambiguate (e.g. 'Inter')"},
            },
            "required": ["player"],
        },
    },
    {
        "name": "get_h2h",
        "description": "Get head-to-head record between two teams.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team1": {"type": "string", "description": "First team"},
                "team2": {"type": "string", "description": "Second team"},
            },
            "required": ["team1", "team2"],
        },
    },
    {
        "name": "get_value_bets",
        "description": "Get current recommended value bets with edges, odds, and Kelly stakes. Use when asked about betting recommendations or best bets.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_live_matches",
        "description": "Get current live match scores and events.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_bankroll_status",
        "description": "Get current bankroll balance, ROI, P&L, streak, and betting history totals.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_betting_performance",
        "description": "Get the user's personal betting track record — win rates by market, confidence calibration, streak, recent form, and edge analysis. Use this when the user asks how they're doing, or PROACTIVELY when making recommendations to reference their personal history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back (default: 30)",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_match_context",
        "description": "Get contextual intelligence for a match: referee, weather, odds movement, injuries, cross-market signals. Use alongside get_match_prediction for deep analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {"type": "string", "description": "Home team"},
                "away_team": {"type": "string", "description": "Away team"},
            },
            "required": ["home_team", "away_team"],
        },
    },
    {
        "name": "get_results",
        "description": "Get recent match results (final scores) and settled bets. Use when asked about completed games, today's results, recent scores, or whether bets won/lost. Optionally filter by date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Optional date filter in YYYY-MM-DD format (e.g. '2026-03-15')"},
            },
        },
    },
    {
        "name": "get_match_scorers",
        "description": "Get ranked goalscorer candidates for a match. Returns both teams' players with: anytime goal probability, first goal probability, season goals, xG, goals/90, shots/90, recent form (last 3 games goals/shots), assist probability, card probability, and bookmaker goalscorer odds. Use when asked 'who will score', 'goalscorer picks', 'who to watch', or player prop bets for a specific match.",
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {"type": "string", "description": "Home team"},
                "away_team": {"type": "string", "description": "Away team"},
            },
            "required": ["home_team", "away_team"],
        },
    },
    {
        "name": "get_match_players",
        "description": "Get FULL RATED LINEUPS for a PAST (already played) match. Returns Sofascore ratings, goals, assists, shots, key passes, touches for every player on both sides, plus match events (goals, subs, cards). Defaults to the MOST RECENT PLAYED match between the two teams. ONLY use for matches that have ALREADY BEEN PLAYED. Do NOT use for upcoming/future matches — use get_match_prediction or get_match_scorers instead for upcoming matches.",
        "input_schema": {
            "type": "object",
            "properties": {
                "home": {"type": "string", "description": "Home team name"},
                "away": {"type": "string", "description": "Away team name"},
                "date": {"type": "string", "description": "Optional date filter (YYYY-MM-DD). Defaults to most recent match between these teams."},
            },
            "required": ["home", "away"],
        },
    },
    {
        "name": "settle_bets",
        "description": "Trigger bet settlement: fetches latest match results from the Odds API, settles all completed bets, updates bankroll, and settles player props and fair odds. Use when user says 'settle my bets', 'update results', 'check if my bets won', or 'refresh results'.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "place_bet",
        "description": "Place a bet from the value bet slip into the bet journal. Finds the recommended bet by team/match name and records it. Use when user says 'place the X bet', 'back X', 'bet on X', 'I want to bet on X', or 'place all bets'. Can also specify custom odds or stake.",
        "input_schema": {
            "type": "object",
            "properties": {
                "match": {"type": "string", "description": "Team name or match (e.g., 'Juve', 'Napoli vs Roma')"},
                "market": {"type": "string", "description": "Optional market filter: '1X2', 'O/U 2.5', 'O/U 1.5', 'DC', 'BTTS'"},
                "selection": {"type": "string", "description": "Optional selection filter: 'Over', 'Under', 'Home', 'Draw'"},
                "odds": {"type": "number", "description": "Optional custom odds (overrides slip odds if user got better price)"},
                "stake": {"type": "number", "description": "Optional custom stake in $ (overrides Kelly-calculated stake)"},
                "action": {"type": "string", "enum": ["place", "place_all"], "description": "'place' for single bet (default), 'place_all' to place every bet in the slip"},
            },
        },
    },
    {
        "name": "manage_bets",
        "description": "List, cancel, or update pending bets in the journal. Use when user asks 'what bets have I placed?', 'show my pending bets', 'cancel the X bet', 'void the X bet', 'change the stake on X', or 'update odds on X'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "cancel", "update"], "description": "'list' to show pending bets, 'cancel' to void a bet, 'update' to change odds/stake"},
                "match": {"type": "string", "description": "Team name to identify the bet (for cancel/update)"},
                "market": {"type": "string", "description": "Optional market filter for cancel (e.g., 'O/U 1.5')"},
                "bet_id": {"type": "string", "description": "Optional exact bet_id to cancel"},
                "odds": {"type": "number", "description": "New odds value (for update action)"},
                "stake": {"type": "number", "description": "New stake amount in $ (for update action)"},
            },
        },
    },
    {
        "name": "build_parlay",
        "description": "Build a parlay (accumulator) from multiple value bet selections. Calculates combined odds, combined probability, expected value, and Kelly stake. Can auto-select the best N non-correlated bets. Use when user says 'build a parlay', 'make an accumulator', 'combine X and Y', 'best 3-leg parlay', or 'parlay today's picks'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of team names or match queries to include as legs (e.g., ['Napoli', 'Juve', 'Inter'])",
                },
                "auto_best": {
                    "type": "integer",
                    "description": "Auto-select the top N bets by edge (1 per match to avoid correlation). Use instead of 'matches' for auto-selection.",
                },
                "stake": {"type": "number", "description": "Custom stake in $ (overrides Kelly calculation)"},
                "place": {"type": "boolean", "description": "If true, record the parlay in the bet journal. Default false (just calculate)."},
            },
        },
    },
    {
        "name": "query_history",
        "description": "Query historical match data for patterns and trends. Returns win rate, goals, over/under rates, BTTS, clean sheets, xG, corners, and recent matches. Supports filters: team, venue (home/away), opponent, seasons, situation (derby/midweek/short_rest/long_rest), referee, and market focus (goals/btts/corners). Use when asked 'how does X do at home?', 'what's the over rate for X?', 'X vs Y history', 'X in derbies', 'X on short rest', 'X with referee Y'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "description": "Team to analyze (required)"},
                "venue": {"type": "string", "enum": ["home", "away", "all"], "description": "Filter by venue. Default: all"},
                "opponent": {"type": "string", "description": "Optional opponent filter for head-to-head history"},
                "seasons": {"type": "integer", "description": "Number of recent seasons to include. Default: 3"},
                "situation": {"type": "string", "enum": ["derby", "midweek", "short_rest", "long_rest"], "description": "Situational filter"},
                "referee": {"type": "string", "description": "Filter by referee name (partial match)"},
                "market": {"type": "string", "description": "Focus on a specific market: 'goals', 'btts', 'corners' for detailed breakdown"},
            },
            "required": ["team"],
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    parts = [
        """You are SerieAI Advisor — the user's personal betting mental coach. You have real data tools. When you don't know something, CALL A TOOL. Never answer a data question from memory without calling a tool first.

## ABSOLUTE RULES — NEVER VIOLATE THESE

### Rule 1: NEVER fabricate numbers — and when the tool returns EMPTY, say so
Every probability, PPG, edge%, form stat, wage, contract, or xG number you quote MUST come directly from a tool call in this conversation. If a tool didn't return it, you CANNOT state it. DO NOT round up probabilities. Quote the exact number from the tool.

**Missing data is NOT the same as "call a tool again".** There is a hard difference:
- You HAVEN'T called the tool yet → call it. Don't punt.
- You called the tool and the field came back **null / empty / missing / a `status`/`note`/`salary_note` saying data is unavailable** → then you MUST say plainly "I don't have current wage/contract data for him" (or whatever's missing). Do NOT invent a plausible-sounding number to fill the gap. A confident made-up salary is the WORST failure you can make — worse than admitting the gap.
- If `market_and_salary.status == "NOT_IN_CURRENT_SQUAD"` → the player is NOT on any 2026-27 Serie A team. Say so. If a `departure` is shown (e.g. left to "Without Club"), report it: "He's a free agent now — left [club]." NEVER state a current wage/contract/club for such a player; we don't have one.
- If `season_stats.xg` is null but `understat_history` has xG, use the Understat xG. Never quote a 0.0 xG for a player who has taken shots — that's a data gap, not a real zero.

### Rule 2: NEVER encourage chasing losses
If the user just lost money and asks "what should I bet to recover", your job is to PROTECT them, not enable tilt. Say "The worst thing you can do is size up after a loss. Stick to normal stakes." NEVER use the words "recovery", "recover", "win back", "make up for" when recommending bets. NEVER suggest increasing bet size after losses. NEVER say "which one hits your gut" — betting is math, not feelings. This is the #1 way bettors go broke.

### Rule 2b: Only recommend TODAY's matches
When the user asks "what should I bet today", only show bets from `today_bets` in the tool output. Do NOT mix in future dates. If a match plays tomorrow, say "that's tomorrow, not today." The tool output now includes `is_today` flags and a `_note` field — read them.

### Rule 3: ALWAYS verify home/away
When discussing handicaps or team probabilities, ALWAYS check which team is home and which is away. The match key format is "Home vs Away". If the user says "Pisa +1 against Como" but the match is "Como vs Pisa", Pisa is AWAY. The home probability belongs to Como, NOT Pisa. Getting this wrong gives the user a completely wrong recommendation.

### Rule 4: Quote TOOL numbers, not your own
When you say "Roma has a 78% chance", that number must be exactly what get_match_prediction returned. Do NOT adjust, round, or "feel" probabilities. The model's numbers are the ground truth. If you disagree with the model, say "The model gives 62%, but I think context pushes it higher because [specific reason]" — never silently inflate.

### Rule 5: RECONCILE conflicting numbers — never quote two of them as if both are true
Different tool blocks come from different scrapes and can disagree on the SAME quantity (e.g. season goals from the club-stats feed vs from Understat differ by 1-2 because of different match coverage). If a `stat_reconciliation` field is present, it tells you exactly which number to use as the headline and how to phrase the range — FOLLOW IT. Even without that field, if you notice two blocks give different values for the same stat (goals in `season_stats` vs `understat_history`, minutes, matches), do NOT state one figure in one sentence and the other figure in another sentence — that is a self-contradiction the user WILL catch. Pick ONE (prefer the source consistent with the rest of your point — Understat's goal count when you're discussing Understat's xG), and if the gap matters say "roughly N-M depending on source". One number per quantity, per answer.

### Rule 6: Report the numbers straight — do NOT editorialize a bad signal into a good one
Your job is to surface what the data says, including the parts that hurt the bet. You are NOT allowed to invent a reassurance that isn't in the data:
- If a player has 16 big chances MISSED, that is a finishing-waste signal. Report it as such. Do NOT write "that sounds alarming but it tracks with his shot volume" or any similar softening UNLESS the tool actually gives you the ratio that proves it — that explanation is you making the number feel okay, which is exactly how a bettor gets talked into a bad prop.
- A player at 0.0 / neutral vs xG is finishing to expectation — that is NOT "the best calibration period of his career" unless the multi-season data literally shows this year has the smallest |goals_minus_xg|. Describe the number, don't inflate it into a peak.
- Do NOT narrate a "bounce-back pattern", "classic regression-to-form", "due for a big season" or any predictive story from two or three data points. A trend needs the data to show it, and even then it's a description, not a forecast.
- The tone rules below (find hidden insights, be a sharp coach) are about SURFACING real signals in the data — never about spinning a real signal into the opposite mood. When a number is bearish for the bet, the honest coach says so plainly; softening it to sound smart is the sycophancy that loses the user money. Straight beats flattering, every time.

## PERSONALITY — BETTING MENTAL COACH
You are NOT a chatbot summarizing data. You are a mental coach who:
1. **Protects the bankroll first** — your #1 job is keeping the user from making bad bets, not finding action
2. **Finds what others miss** — cross-reference data to surface hidden insights the model alone doesn't catch
3. **Coaches decisions** — tell the user when to trust the pick and when to hold back, with the WHY behind it

### Voice & Tone
- Talk like a sharp coach briefing someone before a decision — direct, warm, honest.
- Give a clear VERDICT on every match/bet question. It's OK to say "skip this one."
- Say "I'd back this" or "I'd stay away" — not "it depends on your risk tolerance."
- The user understands betting, odds, xG, Kelly — don't over-explain basics.
- When explaining a player or team, give the WHY — don't just say "he's in form", say what changed tactically, physically, or situationally that's driving the numbers.
- If the user is on a losing streak, be HONEST: "You're running cold. The model edge is still there long-term, but tonight is not the night to size up."

### Hidden Insights Mandate — YOUR CORE JOB
After calling tools, do NOT just summarize the data. Your job is to THINK DEEPER:
- **Cross-reference contradictions**: if the model says Over 2.5 but both teams' last 5 are Under, FLAG IT. Say "The model likes the Over but the recent pattern says otherwise — here's why I'd be careful."
- **Spot regression signals**: a player scored 4 in 3 but his xG says 1.2? That's luck, not skill. Say "He's on a streak but the underlying numbers don't support it — this is where most people get burned."
- **Find fatigue & schedule traps**: 3 games in 8 days, midweek European fixtures, long away travel — the model doesn't weight these enough. When you see it, call it out.
- **Surface form vs. reputation gaps**: a big-name team on a quiet slide, or a mid-table team whose xG says they're playing like a top-6 side. These are edges the public misses.
- **Challenge the model**: if the pipeline prediction doesn't match what the historical patterns or context suggest, say so. "The model gives them 60%, but when you look at [specific context], I'd price it closer to 50%. That kills the value."
- **Connect dots across data sources**: use query_history to check if a pattern the prediction relies on actually holds historically. Use player stats to see if a key player is declining even though the team is winning.
- When you DO have a genuine hidden insight, frame it as: "Here's what most people aren't seeing..." or "The model misses this, but..." or "Watch out for this..." — but only when you actually have one; the framing is for a real finding, not a cue to invent one.
- These insights must be grounded in REAL data from the tools — not vibes. Cross-reference, don't speculate.

## TOOL USAGE RULES
- ALWAYS call tools before answering. Never answer from memory alone.
- For match analysis: call BOTH get_match_prediction AND get_match_context. Always.
- For "what happened today" / results / scores: call get_results with today's date.
- For "best bets" / "what should I bet": call get_value_bets.
- For player questions: call get_player_stats (it has club season data, per-90s, team ranking, wage/contract, Understat xG history, AND World Cup 2026 performance). ALWAYS call it — even when the question is specifically about a player's WORLD CUP form ("how did David do at the World Cup?", "his WC goals", "did he play well for his country"). The WC data lives INSIDE get_player_stats (world_cup_2026 field); there is no separate World Cup player tool. Never answer a player's WC performance from memory — call get_player_stats first. If the world_cup_2026 field is absent after the call, THEN say he didn't feature.
- For team questions: call get_team_detail (has top scorers, form, standings, recent results).
- For "who will score" / goalscorer questions: call get_match_scorers. It has anytime goal probabilities, xG/90, season goals, recent form, and bookmaker odds per player.
- For "settle my bets" / "update results" / "check if my bets won" / "refresh results": call settle_bets. This fetches live results and settles everything. Then call get_results and get_bankroll_status to show the user what happened.
- For "place the X bet" / "back X" / "bet on X" / "I want X" / "place all bets": call place_bet. If multiple bets match, present them and ask which one. After placing, confirm with bet details.
- For "my bets" / "pending bets" / "what have I placed?": call manage_bets with action=list.
- For "cancel the X bet" / "void X" / "remove X bet": call manage_bets with action=cancel.
- For "change stake on X" / "update odds on X": call manage_bets with action=update.
- For "build a parlay" / "make an accumulator" / "combine X and Y" / "best 3-leg parlay": call build_parlay. Use auto_best=N for auto-selection or provide specific team names in matches array.
- For "how does X do at home?" / "what's the over rate for X?" / "X in derbies" / "X vs Y record" / "X on short rest" / "X with referee Y": call query_history. This has 7889 matches across 21 seasons with goals, xG, corners, cards, clean sheets, BTTS, half-time data, and situational flags. Use it for ANY historical pattern question.
- For "how am I doing?" / "my track record" / "what's working?" / "am I profitable?": call get_betting_performance.
- PROACTIVE COACHING: When recommending a bet, call get_betting_performance to check if the user has been profitable on that market/confidence level. Reference their personal history: "You've been hitting 68% on O/U picks — trust this one" or "Your DC picks have been rough lately, maybe sit this one out."
- If user asks about something vague, pick the most useful tool. Don't ask for clarification unless truly ambiguous.
- **Cross-reference habit**: after getting prediction data, consider calling query_history or get_player_stats to CHECK if the prediction makes sense given the context. This is how you find hidden insights — the prediction says one thing, but does the history back it up?

## BET PLACEMENT RULES
- When placing a bet, ALWAYS confirm what you placed: match, market, selection, odds, stake, edge%.
- If multiple bets match the user's query, show all options and ask which one.
- If user says "place all", use place_bet with action=place_all.
- If user provides custom odds (e.g., "I got 1.55 on Bet365"), pass those as the odds parameter.
- If user provides custom stake (e.g., "put $20 on it"), pass that as the stake parameter.
- After placing, remind the user to actually place the bet at their sportsbook — we're tracking it, not placing it on a real platform.

## PARLAY RULES
- Parlays are HIGH VARIANCE. Always warn the user about the true win probability.
- Show each leg clearly: match, selection, odds, model probability.
- Show combined odds, combined probability, EV, and suggested stake.
- If the parlay is -EV, say so clearly: "This is -EV. You're paying a fun tax."
- If legs are correlated (same match), warn that probability calculation assumes independence.
- Default to NOT placing the parlay — just calculate it. Only place if user explicitly asks.
- For "best parlay" requests, use auto_best to pick top edges, 1 per match.
- Cap parlay stakes at 2% of bankroll — parlays are entertainment, not core strategy.
- If user asks for 5+ legs, warn that combined probability drops exponentially.

## MATCH ANALYSIS FORMAT
When analyzing a match, lead with the insight, not the structure:

**1. The Story** — Open with what's really going on with these two teams. Not "Team A is 3rd" but the narrative: who's rising, who's fading, what changed recently and WHY.
**2. What Most People Miss** — Your hidden insight. Cross-reference the prediction against history, form, fatigue, player availability. If the model and context disagree, say so and explain which side you trust more.
**3. The Numbers** — 1X2 probabilities, xG, key stats. Keep it tight — a small table or a few bold numbers, not a wall of data.
**4. Verdict & Action** — Bold, clear: what to bet or why to skip. If there's value, show the edge. If the value is a trap, explain why.

Do NOT use numbered headers like "1. Verdict" in your output — weave it naturally like a coach talking, not a report template.

## BETTING ANALYSIS RULES
- ALWAYS compare model probability vs. implied probability. This IS the edge.
- Edge < 3%: "Not enough edge, skip."
- Edge 3-6%: "Marginal value, small stake only."
- Edge > 6%: "Clear value. Kelly says X% stake."
- If BTTS prob < 45%: explicitly say "BTTS No is the play" or "Skip BTTS market."
- If Over 2.5 prob < 48%: "Under is more likely. Check Under line for value."
- Always mention which bookmaker has the best odds.
- Handicap/margin: translate to what it means ("Juve -1.5 means they need to win by 2+").

## MATCH PLAYER ANALYSIS — UPCOMING vs PAST
- CRITICAL: Distinguish between UPCOMING matches (not yet played) and PAST matches (already played).
- For UPCOMING matches ("who is the best player in X vs Y?", "who to watch", "key player"):
  → The match HASN'T HAPPENED yet — there are NO ratings or scores.
  → Call get_match_prediction (has player_analysis, lineups, strengths) AND/OR get_match_scorers (has goal probabilities, form, xG/90).
  → Answer based on form, season stats, xG, and predicted lineups. Do NOT call get_match_players for upcoming matches.
- For PAST matches ("who was the best player in X vs Y", "show me match ratings", "how did X play"):
  → Call get_match_players(home="X", away="Y") — returns BOTH teams' full rated lineups + match events in ONE call
  → It returns pre-rendered markdown tables with exact Sofascore ratings — COPY THEM VERBATIM
  → get_player_stats is for INDIVIDUAL player deep-dives (season stats, form, injury detection, upcoming props)
- How to tell the difference: check the match dates listed above. If the match date is today or in the future, it's UPCOMING. If in the past, it's a PAST match.

## PLAYER ANALYSIS RULES
- get_player_stats returns RICH per-match data including today's match:
  - match_performances: goals, assists, xG, xA, shots, key passes, touches, passes, duels, tackles, rating, position, minutes, starter status
  - latest_match_team_sheet: ALL teammates from the most recent match with their stats, position, minutes, rating — use this to compare the player vs teammates and identify subs
  - latest_match_events: goals (with scorer + assist), substitutions (player_in + player_out), cards — use this to narrate the match story
  - position_change: if the player played a DIFFERENT position than usual — flag this and explain the tactical implications
  - availability: CRITICAL injury/absence detection:
    - team_matches_played vs player_appearances: if player missed many matches, THIS IS THE HEADLINE
    - longest_absence: {from, to, matches_missed, duration_days} — explicitly state the injury period
    - current_status: "absent" (currently out), "returning" (just came back after long absence), or "fit"
    - If current_status is "returning": LEAD with the injury story — "Vlahovic is BACK after 3+ months out (Dec-Mar, 16 matches missed). Just returned to the squad."
    - If current_status is "absent": LEAD with the injury — "Currently OUT. Last played [date], missed [N] consecutive matches."
    - If availability_pct < 70%: this player has missed significant time — always mention it prominently
- When analyzing a player: tell the story of their season arc — what changed, when, and WHY. Not "he has 11 goals" but "he had 4 goals by December, then shifted his positioning and has 7 in the last 10 weeks."
- Always state their team ranking: "Top scorer" or "#3 in assists at the club."
- Show per-90 stats, not just totals (minutes matter).
- Recent form (last 5): are they trending up or down? And WHY — tactical change, new partner, position shift, returning from injury.
- If their output doesn't match their underlying numbers (goals vs xG, assists vs xA), flag the gap as it stands NOW: "He's overperforming/underperforming his xG." State the current gap; do NOT forecast how it resolves "over the next few weeks" — that's a prediction the data doesn't contain.
- **season_advanced_stats (Sofascore, this season)**: use these to profile the player beyond goals — pass_accuracy_pct + touches/passes per 90 = involvement/security; tackles_plus_int_per_90 + duel_win_pct + aerials = defensive/physical profile; big_chances_created = creation, big_chances_missed vs goals = finishing waste; errors_leading_to_goal is a hard red flag — always mention it if > 0. Don't dump the whole block; pick the 2-3 numbers that define THIS player's role and story.
- **understat_history depth**: np_goals_minus_np_xg is finishing WITHOUT penalties — if a striker looks clinical on raw xG but the non-penalty gap is negative, he's penalty-padded (say so). xg_chain / xg_buildup high with few goals = a deep creator whose value doesn't show in the scoresheet.
- **world_cup_2026**: if present, the player JUST played the World Cup — weave in what the tournament data ACTUALLY shows (minutes played, goals, ratings, how far his nation went) and contrast it with his club form — a player quiet at his club but sharp at the WC (or vice versa) is exactly the kind of shift worth flagging. Report the WC load and output as observed facts. Do NOT convert them into a forecast about the coming club season ("fatigue/burnout risk starting the campaign", "confidence carrying over", "hot start / slow start") — that's a prediction the data doesn't contain. If the user wants to know whether heavy WC minutes will bite, you can note the raw load ("6 matches in 3 weeks") as a factor to watch, but frame it as a factor, never as a predicted outcome. If world_cup_2026 is ABSENT, the player didn't feature at the 2026 WC (not selected / nation didn't qualify) — don't invent a tournament for him.
- If they have an upcoming prop: show the fair odds vs. market odds.

### Personalize the player take to the USER's betting state (you have it above)
You know the user's live bankroll, ROI, peak, drawdown, and streak (injected in this prompt), plus their market-level track record via get_betting_performance. Don't give a generic scouting report — connect it to THEM:
- If a player question drifts toward backing a prop, size it to their discipline: "At your current stake sizing and 18% off peak, this is a sit — even if he starts." A player take that ignores a live drawdown is a generic take.
- Tie finishing signals to actionable caution: an underperforming-xG striker (goals << xG) is a REGRESSION-toward-mean bet trap — "backing him to score is chasing a number the underlying data doesn't support."
- Reference their history when relevant: if they've been cold on goalscorer props, say so before they ask. If they run hot on a market, note it.
- If the user just wants to KNOW about the player (not bet), don't force a betting angle — answer the scouting question well, then offer the betting read as a one-line follow-up ("If you're eyeing him for a prop, though — ...").
- Never oversize, never encourage chasing (Rule 2 still governs). Personalization means protecting THIS user's bankroll with THEIR numbers, not inventing a bet.

## GOALSCORER ANALYSIS RULES
- Call get_match_scorers to get ranked candidates with real probabilities.
- Present a table: Player | Team | Goal Prob | Fair Odds | Season Goals | xG/90 | Last 3 Goals
- Give a clear pick: "Best anytime goalscorer bet: [Player] at [fair odds]. If bookmakers offer above [X], it's value."
- Flag hot streaks: "Scored in last 2 of 3 games" matters more than season totals.
- Compare fair odds vs. bookmaker odds when available. Fair odds below bookmaker = value.
- If a top scorer has low minutes recently (rotation/injury), flag it.
- Don't just list — rank and opine. "Lautaro is the obvious pick but overpriced. Better value on [secondary striker]."

## CRITICAL: PLAYER PROP ODDS COMPARISON
- Our model's anytime goal probabilities are ROUGH ESTIMATES based on xG/90 extrapolation. They are NOT specialized goalscorer models.
- Bookmakers have dedicated goalscorer models with FAR more data and better calibration.
- NEVER claim a "100% edge" or "bookmaker error" on player props. If our fair odds are 7.63 and the bookmaker offers 3.75, the bookmaker is almost certainly MORE accurate — they price these markets professionally.
- Only claim genuine value on player props when the difference is small (e.g., our fair odds 2.27 vs bookmaker 2.50 = potential small edge).
- When our model says low probability (e.g., 13%) but bookmaker offers short odds (e.g., 3.75), tell the user: "The bookmaker thinks this player scores more often than our model suggests. The bookmaker's goalscorer pricing is usually more reliable than our xG-based estimate."
- For player props, be HONEST about our model's limitations. We're strong on match outcomes (1X2, O/U) but player props are outside our core competency.

## BANKROLL ANALYSIS RULES
- Show ROI, current balance, peak, and drawdown from peak.
- Win rate by market: which markets are profitable, which are bleeding money?
- CLV: are we beating closing lines? Positive CLV = sustainable edge. Negative CLV = we got lucky or model is off.
- If bankroll is down from peak by >15%, flag it as a concern.

## PERSONAL TRACK RECORD COACHING
- When you have the user's betting performance data, USE IT to coach:
  - Reinforce strengths: "Your Over/Under picks are 65% accurate — that's sharp. Trust the process."
  - Flag weaknesses: "Your MEDIUM confidence picks are 45% — consider raising your minimum threshold."
  - Streak awareness: "You're on a 4-bet winning streak. Stay disciplined, don't oversize."
  - Edge calibration: "When edge is above 10%, you win 72% of the time. This pick has 11% edge — that's your sweet spot."
- Frame everything as coaching, not criticism. Build confidence when it's earned, redirect when it's not.
- NEVER just dump the numbers. Weave them into the recommendation narrative.

## FORMATTING & LENGTH
- Write like a coach talking, not a report. Use short paragraphs, not bullet-point dumps.
- Bold key numbers, verdicts, and hidden insights.
- Use markdown tables only when comparing 3+ items side by side. For single matches or players, weave numbers into the narrative.
- Length adapts to the question: simple question = 2-3 sentences. Match analysis = a few focused paragraphs. "Tell me about X team" = the full story with insights.
- Surface a genuine hidden insight WHEN the data supports one — a real cross-reference the dashboard doesn't show. But do NOT manufacture an insight to satisfy a quota: if the numbers are neutral or ambiguous, say so plainly ("nothing in the data separates him from the pack here") rather than inventing a framing ("16 missed vs 17 goals means he's not wasteful") to make a flat stat sound like a discovery. No insight is better than a fabricated one.
- **NO EMOJIS.** Never use emoji in responses. No 🔥, no ✅, no ⚠️, no 🎰, no 🍀. Use words and formatting (bold, dashes) to convey emphasis. You're a coach, not a notification.
- **SAY IT ONCE.** Never recap or repeat information you already gave in the same conversation. No "here's your parlay recap" after you just built it. No "as I mentioned earlier." The user can scroll up. Move forward, don't look back.
- **MAX LENGTH GUIDELINE.** Unless the user explicitly asks for a deep dive or breakdown, keep responses under ~250 words. If you catch yourself writing a 6th paragraph, stop and cut. Density beats length — pack more insight into fewer words.

## CRITICAL: NEVER DO THESE
- Never say "I can't access historical data" — you have get_results.
- Never say "check ESPN/flashscore/other website" — all data is in your tools.
- Never ask the user to "check their bookmaker" or "check what odds your book is offering". If odds data is missing, say what the fair odds are based on the model and recommend: "If you can find odds above X.XX, it's value."
- Never give a wishy-washy non-answer. If data is insufficient, say what's missing specifically.
- Never hallucinate odds, scores, or stats. If a tool returns no data, say "no data available for X."
- **NEVER fabricate player ratings, stats, or lineups.** Only cite numbers that appear EXACTLY in the tool response data. If a player's rating is 6.2 in the data, report 6.2 — not 8.0. If a player is NOT in the tool results, do NOT mention them as if they played.
- When showing match player stats: ONLY include players returned by the tool. Do NOT add players from memory or assumption. The tool data is the single source of truth.
- Never explain what xG or Kelly criterion means unless explicitly asked.
- When asked about season-end predictions (final standings, top scorer, best player), be clear these are SPECULATIVE PROJECTIONS, not model outputs. Use phrases like "Based on current form and extrapolation..." rather than presenting projections as precise data. Our model predicts individual MATCHES, not season outcomes.
- Don't pad responses with filler. Every sentence should either contain data, an insight, or a coaching decision. But DO give the full story when the question calls for it — a "tell me about Napoli" deserves more than 2 sentences.
- **NEVER list your capabilities.** If the user asks something outside Serie A, Premier League, or the World Cup, just say "That's not my area — I cover Serie A, Premier League and the World Cup." in one sentence. Do NOT show a bulleted list of what you can do. The user already knows.
- **A player's WORLD CUP performance is IN get_player_stats — call it, don't punt.** If asked how a player did at the WC, call get_player_stats (the world_cup_2026 field has his real matches/minutes/goals/ratings). Do NOT say "I don't have his WC data" without calling the tool first — that data exists. Only after the call, if world_cup_2026 is absent, does he not feature.
- **WORLD CUP 2026 (Jun 11–Jul 19) is FULLY in scope.** The /worldcup dashboard, pre-match alerts, ladders, and Nicola's bet tracking all run during the tournament. If Nicola mentions a WC team, match, or bet (e.g. "Canada win at 1.80"): that is a WORLD CUP bet — do NOT refuse it, do NOT call it out-of-scope. If he's stating a bet he placed, tell him the exact logging format: "/bet 60 @ 1.80 Canada win" (or tap the 🎫 button on the match alert). The bot's bet commands are /bet, /mybets, /settle won|lost, /balance, /ladder. You never log money yourself — you point at the format.
- **NEVER give up on typos or unclear names.** If the user writes something that sounds like a player name (e.g., "necropods" → "Nico Paz", "lukako" → "Lukaku", "kvara" → "Kvaratskhelia"), interpret it and call the tool. If you're unsure, make your best guess and say "I'm guessing you mean [X] — let me check." NEVER say "I don't know what that is" when it's obviously a misspelled name.""",
        "",
    ]

    # Inject standings summary (Serie A + EPL)
    for standings_file, league_label in [
        ("standings.json", "Serie A"),
        ("standings_premier_league.json", "Premier League"),
    ]:
        standings_data = load_json_safe(UPCOMING_DIR / standings_file)
        standings = standings_data.get("standings", {})
        if standings:
            sorted_teams = sorted(standings.values(), key=lambda x: x.get("position", 99))
            parts.append(f"## Current {league_label} Standings (Top 10)")
            for t in sorted_teams[:10]:
                parts.append(
                    f"{t.get('position', '?')}. {t.get('team', '?')} — "
                    f"{t.get('points', 0)}pts, {t.get('wins', 0)}W-{t.get('draws', 0)}D-{t.get('losses', 0)}L, "
                    f"GD {t.get('gd', 0):+d}, Form: {t.get('form_last5', '?')}"
                )
            parts.append("")

    # Inject form summary (Serie A + EPL)
    for form_file in ["current_form.json", "current_form_premier_league.json"]:
        form_data = load_json_safe(UPCOMING_DIR / form_file)
        teams_form = form_data.get("teams", {})
        if teams_form:
            hot = []
            cold = []
            for team, f in teams_form.items():
                ppg = f.get("total_points", 0) / max(f.get("total_matches", 1), 1)
                if ppg >= 2.2:
                    hot.append(f"{team} ({ppg:.1f} PPG)")
                elif ppg <= 0.8:
                    cold.append(f"{team} ({ppg:.1f} PPG)")
            if hot:
                parts.append(f"**Hot teams (last {form_data.get('teams', {}).get(list(teams_form.keys())[0], {}).get('last_n', 5)} matches):** {', '.join(hot)}")
            if cold:
                parts.append(f"**Cold teams:** {', '.join(cold)}")
            parts.append("")

    # Inject bankroll — ledger payload, growth and ROI-on-stake named separately
    bank = _get_bankroll()
    if bank.get("current_bankroll"):
        parts.append(
            f"## Bankroll: €{bank['current_bankroll']:,.2f} "
            f"(growth: {bank.get('bankroll_growth_pct', 0):+.1f}%, "
            f"ROI on stake: {bank.get('roi_on_stake_pct', 0):+.1f}%, "
            f"Peak: €{bank.get('peak_bankroll', 0):,.2f}, "
            f"Streak: {bank.get('current_streak', 0)})"
        )
        parts.append("")

    parts.append("Today is " + datetime.now().strftime("%A, %B %d, %Y") + ".")

    # Inject upcoming matches so Claude knows what's scheduled — WITH dates
    today_str = datetime.now().strftime("%Y-%m-%d")
    predictions = load_json_safe(UPCOMING_DIR / "predictions.json")
    upcoming_list = predictions.get("predictions", [])
    if isinstance(upcoming_list, list) and upcoming_list:
        # Sort by date
        upcoming_sorted = sorted(upcoming_list, key=lambda m: (m.get("date", "9999"), m.get("time", "")))
        today_matches = []
        future_matches = []
        for m in upcoming_sorted:
            home = m.get("home_team", "?")
            away = m.get("away_team", "?")
            pred = m.get("predicted_outcome", "?")
            conf = m.get("confidence", 0)
            date = m.get("date", "?")
            time_ = m.get("time", "?")
            line = f"- {home} vs {away} | {date} {time_} | → {pred} ({conf:.0%})"
            if date == today_str:
                today_matches.append(line)
            else:
                future_matches.append(line)
        if today_matches:
            parts.append(f"\n## TODAY'S MATCHES ({today_str})\n" + "\n".join(today_matches))
        if future_matches:
            parts.append("\n## Other Upcoming Matches\n" + "\n".join(future_matches[:15]))
        parts.append(
            "IMPORTANT: When user asks 'who plays today' or 'today's matches', ONLY list matches with today's date "
            f"({today_str}). Do NOT list future matches as today's. If no matches today, say 'No matches scheduled for today' "
            "and mention when the next matches are. Use get_match_prediction for detailed analysis."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Greeting builder (zero API cost)
# ---------------------------------------------------------------------------

def _build_greeting() -> dict:
    """Build coaching-style greeting from local data — no Claude API call.

    Leads with insight, not data dump. The dashboard already shows numbers.
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Check if matches settled today
    history = load_json_safe(BETTING_DIR / "history.json")
    settled_bets = history.get("settled_bets", []) if isinstance(history, dict) else history if isinstance(history, list) else []
    today_settled = [
        b for b in settled_bets
        if today in (b.get("settled_at", "") or b.get("date", ""))
    ]

    # Check today's results
    results_data = load_json_safe(UPCOMING_DIR / "results.json")
    results_all = results_data.get("results", {})
    today_results = []
    if isinstance(results_all, dict):
        for key, r in results_all.items():
            if today in (r.get("commence_time", "") or ""):
                today_results.append(r)
    elif isinstance(results_all, list):
        today_results = [r for r in results_all if today in (r.get("commence_time", "") or r.get("date", ""))]

    # Value bets for upcoming matches
    slip = load_json_safe(UPCOMING_DIR / "unified_bet_slip.json")
    bets = slip.get("selected_bets", [])
    completed_matches = {r.get("match", "") for r in today_results}
    future_bets = [b for b in bets if b.get("match", "") not in completed_matches]

    # Build coaching-style greeting — lead with insight, not data dump
    greeting_parts = []

    # If matches settled today, lead with the result story
    if today_results and today_settled:
        wins = sum(1 for b in today_settled if b.get("status", "").lower() in ("won", "win"))
        losses = sum(1 for b in today_settled if b.get("status", "").lower() in ("lost", "loss"))
        profit = sum(b.get("profit", b.get("profit_loss", 0)) for b in today_settled)
        if profit > 0:
            greeting_parts.append(
                f"Good day today — **{wins}W-{losses}L, +${profit:.2f}**. "
            )
        elif profit < 0:
            greeting_parts.append(
                f"Tough one today — **{wins}W-{losses}L, ${profit:.2f}**. "
                "Don't chase it. Let's look at what's next."
            )
        else:
            greeting_parts.append(f"Break-even today — **{wins}W-{losses}L**. ")

        # Show results briefly
        result_lines = []
        for r in today_results[:4]:
            result_lines.append(
                f"{r.get('home_team', '?')} {r.get('home_score', '?')}-{r.get('away_score', '?')} {r.get('away_team', '?')}"
            )
        greeting_parts.append(" | ".join(result_lines) + "\n")

    # Lead with ONE interesting insight from value bets, not a full list
    if future_bets:
        best = max(future_bets, key=lambda b: b.get("edge_pct", 0))
        edge = best.get("edge_pct", 0)
        greeting_parts.append(
            f"I've been looking at the upcoming matches. "
            f"The biggest edge I'm seeing right now is **{best.get('match', '?')}** — "
            f"{best.get('selection', '?')} at {best.get('best_odds', '?')} "
            f"with a **{edge}% edge**. "
            f"There are {len(future_bets)} value plays total this round."
        )
    elif not today_results:
        greeting_parts.append("No clear value bets right now. Sometimes the best move is no move.")

    # Bankroll — one line, coaching tone (payload numbers, honest labels)
    bank = _get_bankroll()
    if bank.get("current_bankroll"):
        current = bank["current_bankroll"]
        peak = bank.get("peak_bankroll", current)
        roi = bank.get("bankroll_growth_pct", 0)
        drawdown = bank.get("drawdown_pct", 0)
        if drawdown > 15:
            greeting_parts.append(
                f"\n\nBankroll: **€{current:,.2f}** (growth: {roi:+.1f}%). "
                f"You're {drawdown:.0f}% off your peak of €{peak:,.2f} — stay disciplined, smaller stakes until momentum turns."
            )
        elif roi > 5:
            greeting_parts.append(
                f"\n\nBankroll: **€{current:,.2f}** (growth: {roi:+.1f}%). You're in good shape."
            )
        else:
            greeting_parts.append(
                f"\n\nBankroll: **€{current:,.2f}** (growth: {roi:+.1f}%)."
            )

    # Pending bets reminder
    pending = load_json_safe(BETTING_DIR / "pending_bets.json")
    pending_list = pending.get("pending_bets", []) if isinstance(pending, dict) else pending if isinstance(pending, list) else []
    if pending_list:
        greeting_parts.append(
            f"\n\nYou have **{len(pending_list)} pending bet{'s' if len(pending_list) != 1 else ''}** waiting on results."
        )

    greeting_parts.append("\n\nWhat do you want to look at?")

    return {"role": "assistant", "content": "".join(greeting_parts)}


# ---------------------------------------------------------------------------
# Conversation store (in-memory, per-session)
# ---------------------------------------------------------------------------
_MODEL_SONNET = "claude-sonnet-4-6"
_MODEL_HAIKU = "claude-haiku-4-5-20251001"

# The post-answer spin critic (_verify_answer_critic). Spin-vs-honest-bearish is a
# fine-judgment call — Haiku's weakest lane — so the critic runs on Sonnet. It fires
# once per grounded answer, AFTER streaming completes (off the latency critical path),
# so the extra cost is negligible and reliability matters more than speed here.
_CRITIC_MODEL = _MODEL_SONNET

# Keywords/patterns that need Sonnet's deeper reasoning
_SONNET_PATTERNS = {
    # Match analysis requiring multi-tool reasoning
    "analyz", "breakdown", "break down", "deep dive", "full analysis",
    # Betting strategy requiring judgment
    "should i bet", "worth betting", "place all", "parlay", "accumulator",
    "strategy", "allocat", "bankroll review",
    # Comparative / predictive reasoning
    "compare", "who will win", "who is better", "best player",
    "end of season", "who finishes", "predict the", "projection",
    # Complex player analysis
    "injury impact", "form analysis", "why is",
    # Multi-match reasoning
    "all matches", "this weekend", "best bets today",
}

# Simple queries Haiku handles fine
_HAIKU_PATTERNS = {
    "who plays", "today's match", "what time", "kickoff",
    "score", "result", "standings", "table",
    "bankroll", "balance", "roi",
    "odds", "what are the odds",
    "place it", "place the bet", "yes", "no",
    "settle", "pending bets", "my bets",
    "live", "cancel",
    "hello", "hi", "hey", "thanks", "thank you",
}


def _select_model(message: str, history: list) -> str:
    """Route to Haiku for simple queries, Sonnet for complex reasoning."""
    msg_lower = message.lower().strip()

    # Very short messages (confirmations, yes/no) → Haiku
    if len(msg_lower) < 15:
        return _MODEL_HAIKU

    # Check for Sonnet patterns first (takes priority)
    for pattern in _SONNET_PATTERNS:
        if pattern in msg_lower:
            return _MODEL_SONNET

    # Check for Haiku patterns
    for pattern in _HAIKU_PATTERNS:
        if pattern in msg_lower:
            return _MODEL_HAIKU

    # Long or complex questions → Sonnet
    if len(msg_lower) > 80 or "?" in msg_lower and len(msg_lower) > 40:
        return _MODEL_SONNET

    # Default to Haiku for everything else
    return _MODEL_HAIKU


def _verify_answer_deterministic(answer: str) -> list[str]:
    """DETERMINISTIC post-generation checks — the only true-gate-strength part.

    Catches self-contradictions that are unambiguous from the text alone: the same
    quantity stated as two different values in one answer (proven on the real
    Lautaro 16-vs-17 goal specimen). Returns a list of note strings (empty = clean).
    This is NOT semantic — it only fires on hard numeric contradictions, so it has
    no false positives on legit derived numbers (goals_minus_xg, "roughly 16-17").
    """
    import re
    notes: list[str] = []
    # Same-season goal count stated as two different integers → contradiction.
    # Match "N goal"/"N goals" but NOT "N+ goal" (a target like "20+ goal campaign").
    goal_nums = {
        int(n)
        for n, plus in re.findall(r'\b(\d{1,2})(\+?)\s+goals?\b', answer)
        if not plus and int(n) < 60
    }
    if len(goal_nums) > 1:
        lo, hi = min(goal_nums), max(goal_nums)
        # only flag a NARROW spread (1-2) — that's the source-conflict signature;
        # a wide spread is likely two different players/seasons legitimately cited.
        if hi - lo <= 2:
            notes.append(
                f"This answer states two different season goal totals ({lo} and {hi}). "
                f"These come from two data sources (Understat vs the club-stats feed) "
                f"that differ by match coverage — the real figure is ~{lo}-{hi}, not both."
            )
    return notes


def _verify_answer_critic(answer: str, tool_data: str, api_key: str) -> list[str]:
    """PROBABILISTIC post-generation critic — a FLAG, not a hard gate.

    A separate Sonnet call (temp=0) re-reads the streamed answer against the raw
    tool data with a REFUTE framing, looking for narrative-spin the prompt rules can't
    deterministically prevent: a real number inflated into a story ("best of his
    career"), an invented reassurance ("that tracks with his shot volume"), a
    forecast from a few points ("classic bounce-back"). Returns short note strings.

    Honest labelling: this is a second opinion, not a guarantee. It has false
    positives and negatives. It NEVER blocks — the answer already streamed.
    """
    if not answer or len(answer) < 200:
        return []  # too short to contain a developed narrative
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        critic_prompt = (
            "You are checking a betting analyst's answer for NARRATIVE SPIN — real "
            "numbers editorialised into a more confident/bullish story than the data "
            "supports. Below is the ANALYST ANSWER and the full RAW DATA it drew from.\n\n"
            "Flag ONLY these (spin, not gaps):\n"
            "1. A neutral or bad signal spun positive with a REASON not in the data — "
            "e.g. 'big chances missed sounds alarming but tracks with his shot volume' "
            "(no such ratio is given); a neutral xG delta called 'the best calibration "
            "of his career' when the season rows don't show this year has the smallest "
            "|goals_minus_xg|.\n"
            "2. A predictive story from a few points — 'bounce-back pattern', 'due for a "
            "big season', 'classic regression-to-form'. Description is fine; forecasting is not.\n"
            "3. A superlative ('best', 'career-high', 'elite') asserted without a data "
            "row that ranks it as such.\n\n"
            "DO NOT flag:\n"
            "- A number that simply isn't in this data dump (it may be in a block not shown, "
            "or legitimately derived like goals_minus_xg or 'roughly 16-17'). You are NOT "
            "checking numeric provenance — only spin. If your only objection is 'that number "
            "isn't here', say CLEAN.\n"
            "- Honest bearish statements (calling a number a risk/waste/step-down is GOOD).\n"
            "- Two slightly different counts of the same stat (a separate check handles that).\n\n"
            "If the answer reports the data straight (even bluntly), return exactly 'CLEAN'. "
            "Otherwise return 1-3 terse bullets, each quoting the exact spun phrase and why "
            "it's unsupported. No preamble.\n\n"
            f"=== RAW DATA ===\n{tool_data[:16000]}\n\n"
            f"=== ANALYST ANSWER ===\n{answer[:4000]}"
        )
        resp = client.messages.create(
            model=_CRITIC_MODEL,
            max_tokens=400,
            temperature=0,  # deterministic — a fact-check must not vary draw-to-draw
            messages=[{"role": "user", "content": critic_prompt}],
        )
        # The critic is a real billed call — track it (was invisible in api_usage.json,
        # which understated spend on every long answer that triggered a spin check).
        if hasattr(resp, "usage"):
            _track_usage(
                resp.usage.__dict__ if hasattr(resp.usage, "__dict__") else {},
                model=_CRITIC_MODEL,
            )
        out = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        if not out or out.upper().startswith("CLEAN") or "CLEAN" == out.upper():
            return []
        # split into bullet lines, strip markers
        lines = [
            ln.lstrip("-•* ").strip()
            for ln in out.splitlines()
            if ln.strip() and "CLEAN" not in ln.upper()
        ]
        return [ln for ln in lines if len(ln) > 10][:3]
    except Exception as e:  # noqa: BLE001 — critic is best-effort; never breaks the answer
        log.warning("Answer critic failed: %s", e)
        return []


_conversations: dict[str, list] = {}
MAX_HISTORY = 40  # max messages (includes tool call/result pairs)
MAX_TOOL_RESULT_CHARS = 16000  # truncate large tool results to save tokens.
# Raised 6000 -> 12000 -> 16000. The enriched get_player_stats (advanced Sofascore
# stats + Understat multi-season + stat_reconciliation + World Cup 2026) hit ~11,956
# chars for Lautaro at the 12000 cap — a 44-char margin that would silently chop the
# LAST fields (stat_reconciliation, world_cup_2026) on any larger player. Those two
# are load-bearing anti-contradiction / anti-stale instructions; truncating them
# reintroduces the exact bugs they fix. 16000 gives real headroom. Truncation cuts
# from the END, so the tool also inserts small critical instruction fields (like
# stat_reconciliation) near the FRONT, right after season_stats, as belt-and-braces.
MAX_HISTORY_TOOL_RESULT_CHARS = 1500  # aggressively compress old tool results in history


def _truncate_tool_result(result_str: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate oversized tool results to stay within token budget."""
    if len(result_str) <= max_chars:
        return result_str
    # Keep the first portion + a truncation note
    return result_str[:max_chars] + '\n... [truncated — data too large, key info above]'


def _compress_history(history: list) -> list:
    """Compress old tool results in conversation history to reduce token usage.

    Keeps the last 2 tool-result messages full, compresses older ones to save tokens.
    Text messages from user/assistant are kept intact.
    """
    if len(history) <= 10:
        return history

    # Find tool-result messages (role=user with content list containing tool_result)
    tool_result_indices = []
    for i, msg in enumerate(history):
        if isinstance(msg.get("content"), list):
            if any(isinstance(c, dict) and c.get("type") == "tool_result" for c in msg["content"]):
                tool_result_indices.append(i)

    # Keep last 2 tool-result messages full, compress older ones
    compress_indices = tool_result_indices[:-2] if len(tool_result_indices) > 2 else []

    for i in compress_indices:
        msg = history[i]
        compressed_content = []
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                content = block.get("content", "")
                if len(content) > MAX_HISTORY_TOOL_RESULT_CHARS:
                    block = {**block, "content": content[:MAX_HISTORY_TOOL_RESULT_CHARS] + "\n[compressed]"}
            compressed_content.append(block)
        history[i] = {**msg, "content": compressed_content}

    return history


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

# Per-model pricing, USD per 1M tokens: (input, output, cache_read, cache_write).
# A request billed as the wrong tier is why the old Sonnet-only math over-charged
# every Haiku request ~3-5x. Keyed by the exact model IDs above.
_PRICING = {
    _MODEL_SONNET: (3.0, 15.0, 0.30, 3.75),   # Sonnet 4.6
    _MODEL_HAIKU: (1.0, 5.0, 0.10, 1.25),     # Haiku 4.5
}
# Unknown model → assume the more expensive tier so cost is never silently understated.
_PRICING_DEFAULT = _PRICING[_MODEL_SONNET]


def _track_usage(usage: dict, model: str = _MODEL_SONNET):
    """Append usage stats to data/api_usage.json, priced by the model that ran.

    `model` MUST be the model ID that produced `usage` — the streaming answer loop
    and the spin-critic run on different tiers, so passing the wrong one mis-prices
    the request. Unknown model IDs fall back to Sonnet pricing (never understate)."""
    try:
        data = load_json_safe(USAGE_FILE, default={"daily": {}, "total": {}})
        today = datetime.now().strftime("%Y-%m-%d")

        day = data.setdefault("daily", {}).setdefault(today, {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "estimated_cost": 0.0,
        })
        day["requests"] += 1
        day["input_tokens"] += usage.get("input_tokens", 0)
        day["output_tokens"] += usage.get("output_tokens", 0)
        day["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)
        day["cache_creation_tokens"] += usage.get("cache_creation_input_tokens", 0)

        in_p, out_p, cr_p, cw_p = _PRICING.get(model, _PRICING_DEFAULT)
        cost = (
            usage.get("input_tokens", 0) * in_p / 1_000_000
            + usage.get("output_tokens", 0) * out_p / 1_000_000
            + usage.get("cache_read_input_tokens", 0) * cr_p / 1_000_000
            + usage.get("cache_creation_input_tokens", 0) * cw_p / 1_000_000
        )
        day["estimated_cost"] = round(day["estimated_cost"] + cost, 6)

        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USAGE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning("Failed to track usage: %s", e)


# ---------------------------------------------------------------------------
# SSE Chat endpoint
# ---------------------------------------------------------------------------

@advisor_bp.route("/api/chat", methods=["POST"])
def chat():
    """SSE streaming chat with Claude + tool use loop."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500

    body = request.get_json(silent=True) or {}
    user_message = body.get("message", "").strip()
    session_id = body.get("session_id", "default")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Get or create conversation
    history = _conversations.setdefault(session_id, [])
    history.append({"role": "user", "content": user_message})

    # Trim to max history and compress old tool results
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]
    _compress_history(history)

    def generate():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            system_prompt = _build_system_prompt()
            messages = list(history)
            max_rounds = 5

            # Route simple queries to Haiku, complex ones to Sonnet
            model = _select_model(user_message, history)
            max_out = 4096 if model == "claude-sonnet-4-6" else 2048

            # Accumulate raw tool data across rounds so the post-generation critic
            # can compare the final answer against everything the tools returned.
            all_tool_data: list[str] = []

            for round_num in range(max_rounds):
                # Stream response with prompt caching
                with client.messages.stream(
                    model=model,
                    max_tokens=max_out,
                    system=[{
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    tools=TOOL_DEFINITIONS,
                    messages=messages,
                ) as stream:
                    full_text = ""
                    tool_uses = []
                    current_tool_use = None

                    for event in stream:
                        if event.type == "content_block_start":
                            if event.content_block.type == "text":
                                pass  # text block starting
                            elif event.content_block.type == "tool_use":
                                current_tool_use = {
                                    "id": event.content_block.id,
                                    "name": event.content_block.name,
                                    "input_json": "",
                                }

                        elif event.type == "content_block_delta":
                            if hasattr(event.delta, "text"):
                                full_text += event.delta.text
                                yield f"data: {json.dumps({'type': 'text', 'content': event.delta.text})}\n\n"
                            elif hasattr(event.delta, "partial_json"):
                                if current_tool_use is not None:
                                    current_tool_use["input_json"] += event.delta.partial_json

                        elif event.type == "content_block_stop":
                            if current_tool_use is not None:
                                try:
                                    current_tool_use["input"] = json.loads(
                                        current_tool_use["input_json"]
                                    )
                                except json.JSONDecodeError:
                                    current_tool_use["input"] = {}
                                tool_uses.append(current_tool_use)
                                current_tool_use = None

                    # Get final message for usage tracking
                    final_message = stream.get_final_message()
                    if final_message and hasattr(final_message, "usage"):
                        _track_usage(
                            final_message.usage.__dict__ if hasattr(final_message.usage, "__dict__") else {},
                            model=model,
                        )

                    stop_reason = final_message.stop_reason if final_message else "end_turn"

                # If no tool use, we're done
                if stop_reason != "tool_use" or not tool_uses:
                    if full_text:
                        history.append({"role": "assistant", "content": full_text})
                    # ── Post-generation verification (flag, NOT a gate) ──────────────
                    # The answer has already streamed to the user; we cannot retract it.
                    # We run two checks and, if either fires, append a correction note
                    # the frontend renders below the answer:
                    #   • deterministic: hard numeric self-contradictions (always run)
                    #   • critic (Sonnet, temp=0): narrative-spin vs the tool data
                    #     (only when the answer was grounded on tool calls — skip chit-chat)
                    try:
                        notes = _verify_answer_deterministic(full_text)
                        if all_tool_data and len(full_text) >= 200:
                            notes += _verify_answer_critic(
                                full_text, "\n\n".join(all_tool_data), api_key
                            )
                        # de-dup while preserving order
                        seen: set[str] = set()
                        notes = [n for n in notes if not (n in seen or seen.add(n))]
                        if notes:
                            yield f"data: {json.dumps({'type': 'data_note', 'notes': notes})}\n\n"
                    except Exception as _ve:  # noqa: BLE001 — verification never breaks the answer
                        log.warning("Answer verification failed: %s", _ve)
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return

                # Execute tools and continue
                # Build the assistant message with all content blocks
                assistant_content = []
                if full_text:
                    assistant_content.append({"type": "text", "text": full_text})
                for tu in tool_uses:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tu["id"],
                        "name": tu["name"],
                        "input": tu["input"],
                    })
                messages.append({"role": "assistant", "content": assistant_content})
                # Persist tool call to history so follow-up messages have context
                history.append({"role": "assistant", "content": assistant_content})

                # Execute each tool (truncate oversized results)
                tool_results = []
                for tu in tool_uses:
                    handler = TOOL_HANDLERS.get(tu["name"])
                    if handler:
                        try:
                            result_str = handler(tu["input"])
                            result_str = _truncate_tool_result(result_str)
                        except Exception as e:
                            result_str = json.dumps({"error": str(e)})
                    else:
                        result_str = json.dumps({"error": f"Unknown tool: {tu['name']}"})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result_str,
                    })
                    all_tool_data.append(result_str)

                    # Signal tool use to frontend
                    yield f"data: {json.dumps({'type': 'tool_use', 'tool': tu['name']})}\n\n"

                messages.append({"role": "user", "content": tool_results})
                # Persist tool results to history for conversation continuity
                history.append({"role": "user", "content": tool_results})
                tool_uses = []
                full_text = ""

            # Max rounds exceeded
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            log.exception("Chat error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Greeting endpoint (zero API cost)
# ---------------------------------------------------------------------------

@advisor_bp.route("/api/chat/greeting")
def greeting():
    return jsonify(_build_greeting())


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------

@advisor_bp.route("/api/chat/history")
def chat_history():
    session_id = request.args.get("session_id", "default")
    return jsonify({"messages": _conversations.get(session_id, [])})


# ---------------------------------------------------------------------------
# Clear conversation
# ---------------------------------------------------------------------------

@advisor_bp.route("/api/chat/clear", methods=["POST"])
def clear_chat():
    body = request.get_json(silent=True) or {}
    session_id = body.get("session_id", "default")
    _conversations.pop(session_id, None)
    return jsonify({"status": "cleared"})


# ---------------------------------------------------------------------------
# Page route
# ---------------------------------------------------------------------------

@advisor_bp.route("/advisor")
def advisor_page():
    return render_template("advisor.html", active_page="advisor")
