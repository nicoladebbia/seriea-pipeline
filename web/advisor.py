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
_BASE = Path(__file__).parent.parent
DATA_DIR = _BASE / "data"
UPCOMING_DIR = DATA_DIR / "upcoming"
BETTING_DIR = DATA_DIR / "betting"
BANKROLL_DIR = DATA_DIR / "bankroll"
LIVE_DIR = DATA_DIR / "live"
USAGE_FILE = DATA_DIR / "api_usage.json"

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default=None):
    if default is None:
        default = {}
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        log.warning("Failed to load %s: %s", path.name, e)
    return default


def _get_bankroll() -> dict:
    """Get bankroll from the freshest source.

    Two files exist:
    - bankroll/state.json (updated by pipeline/manual)
    - betting/bankroll.json (updated by settle flow)
    We merge both, preferring the one with the higher 'updated_at' or file mtime.
    """
    state = _load_json(BANKROLL_DIR / "state.json")
    live = _load_json(BETTING_DIR / "bankroll.json")

    # Determine which is fresher by file mtime
    state_path = BANKROLL_DIR / "state.json"
    live_path = BETTING_DIR / "bankroll.json"
    state_mtime = state_path.stat().st_mtime if state_path.exists() else 0
    live_mtime = live_path.stat().st_mtime if live_path.exists() else 0

    # Build unified view
    if live_mtime > state_mtime and live.get("current_balance"):
        return {
            "current_bankroll": live.get("current_balance"),
            "initial_bankroll": live.get("initial_balance", state.get("initial_bankroll", 1000)),
            "peak_bankroll": max(live.get("peak_balance", 0), state.get("peak_bankroll", 0)),
            "daily_pnl": state.get("daily_pnl", 0),
            "total_bets": state.get("total_bets", 0),
            "total_wins": state.get("total_wins", 0),
            "current_streak": state.get("current_streak", 0),
            "last_updated": live.get("updated_at", ""),
            "source": "betting/bankroll.json",
        }
    else:
        return {
            "current_bankroll": state.get("current_bankroll", live.get("current_balance")),
            "initial_bankroll": state.get("initial_bankroll", 1000),
            "peak_bankroll": max(state.get("peak_bankroll", 0), live.get("peak_balance", 0)),
            "daily_pnl": state.get("daily_pnl", 0),
            "total_bets": state.get("total_bets", 0),
            "total_wins": state.get("total_wins", 0),
            "current_streak": state.get("current_streak", 0),
            "last_updated": state.get("last_updated", ""),
            "source": "bankroll/state.json",
        }


def _resolve_team(query: str) -> str | None:
    """Fuzzy-resolve user input to canonical team name."""
    try:
        from config.team_names import normalize_team, SERIE_A_2025_26
    except ImportError:
        return query

    # Exact match via normalize_team
    canonical = normalize_team(query)
    if canonical in SERIE_A_2025_26:
        return canonical

    # Substring match
    q = query.lower().strip()
    for team in SERIE_A_2025_26:
        if q in team.lower():
            return team
    # Common nicknames
    nicknames = {
        "juve": "Juventus", "inter": "Inter", "milan": "Milan",
        "roma": "Roma", "lazio": "Lazio", "napoli": "Napoli",
        "viola": "Fiorentina", "toro": "Torino", "samp": "Sampdoria",
        "ata": "Atalanta", "dea": "Atalanta", "grifone": "Genoa",
    }
    if q in nicknames and nicknames[q] in SERIE_A_2025_26:
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

    preds_data = _load_json(UPCOMING_DIR / "predictions.json")
    predictions = preds_data.get("predictions", [])
    match = _find_match(home, away, predictions)
    if not match:
        return json.dumps({"error": f"No prediction found for {home} vs {away}. Check team names."})

    result = {
        "match": match.get("match"),
        "home_team": match.get("home_team"),
        "away_team": match.get("away_team"),
        "predicted_outcome": match.get("predicted_outcome"),
        "probabilities": match.get("probabilities"),
        "confidence": match.get("confidence"),
        "strategy": match.get("strategy"),
    }

    # Inject betting info
    slip = _load_json(UPCOMING_DIR / "unified_bet_slip.json")
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
    pa = _load_json(UPCOMING_DIR / "player_analysis.json")
    for m in pa.get("matches", []):
        if m.get("home_team") == h_canon and m.get("away_team") == a_canon:
            result["player_analysis"] = m.get("analysis_summary", "")
            result["home_strength"] = m.get("home_strength")
            result["away_strength"] = m.get("away_strength")
            result["key_factors"] = m.get("key_factors", [])
            break

    # Inject sentiment narrative
    sa = _load_json(UPCOMING_DIR / "sentiment_analysis.json")
    for m in sa.get("matches", []):
        if m.get("home_team") == h_canon and m.get("away_team") == a_canon:
            result["sentiment"] = m.get("analysis", m.get("summary", ""))
            break

    # Inject goal predictions (detailed)
    goals = _load_json(UPCOMING_DIR / "goal_predictions.json")
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
    btts = _load_json(UPCOMING_DIR / "btts_predictions.json", default=[])
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
    corners = _load_json(UPCOMING_DIR / "corners_predictions.json", default=[])
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
    cards = _load_json(UPCOMING_DIR / "cards_predictions.json", default=[])
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
    margins = _load_json(UPCOMING_DIR / "margin_predictions.json")
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
    confirmed = _load_json(UPCOMING_DIR / "confirmed_lineups.json")
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
        predicted = _load_json(UPCOMING_DIR / "lineup_predictions.json")
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
    bk = _load_json(UPCOMING_DIR / "bookmaker_analysis.json")
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
    standings = _load_json(UPCOMING_DIR / "standings.json")
    team_standing = standings.get("standings", {}).get(team)
    if team_standing:
        result["standings"] = team_standing

    # Form
    form_data = _load_json(UPCOMING_DIR / "current_form.json")
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
    results_data = _load_json(UPCOMING_DIR / "results.json")
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
                "xg": round(player_rows["xg"].sum(), 2) if "xg" in player_rows.columns else None,
                "xa": round(player_rows["xg_assist"].sum(), 2) if "xg_assist" in player_rows.columns else None,
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
        sof_path = _BASE / "data" / "external" / "sofascore" / "player_match_stats.parquet"
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
                    f"⚠️ EXACT DATA — DO NOT MODIFY THESE NUMBERS.\n"
                    f"Latest match rating: {latest_rating}\n\n"
                    + "\n".join(form_rows)
                )

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
                        f"⚠️ VERIFIED SOFASCORE DATA — COPY THIS TABLE EXACTLY, DO NOT CHANGE ANY NUMBERS:\n"
                        f"{best_note}\n\n"
                        + "\n".join(rows)
                    )

                    # Match events (goals, cards, subs)
                    inc_path = _BASE / "data" / "external" / "sofascore" / "match_incidents.parquet"
                    if inc_path.exists():
                        try:
                            inc = pd.read_parquet(inc_path)
                            match_events = inc[inc["match_id"] == latest_match_id].sort_values("minute")
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

                        result["availability"] = availability
                except Exception as e:
                    log.warning("Availability analysis failed: %s", e)

    except Exception as e:
        log.warning("Sofascore player lookup failed: %s", e)

    # 3. Upcoming match props from player_props.json
    props = _load_json(UPCOMING_DIR / "player_props.json")
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

    if "season_stats" not in result and "upcoming_props" not in result:
        return json.dumps({"error": f"No data found for player '{args.get('player')}'. Try the full name (e.g., 'Lautaro Martinez')."})

    return json.dumps(result, default=str)


def _tool_get_h2h(args: dict) -> str:
    t1 = _resolve_team(args.get("team1", ""))
    t2 = _resolve_team(args.get("team2", ""))
    if not t1 or not t2:
        return json.dumps({"error": "Could not resolve team names."})

    h2h_data = _load_json(UPCOMING_DIR / "h2h_upcoming.json")
    h2h_list = h2h_data.get("h2h", [])
    if isinstance(h2h_list, list):
        for entry in h2h_list:
            teams = entry.get("teams", entry.get("match", ""))
            if isinstance(teams, str) and t1 in teams and t2 in teams:
                return json.dumps({"teams": [t1, t2], "h2h": entry}, default=str)

    return json.dumps({"teams": [t1, t2], "h2h": None, "note": "No H2H data available."})


def _tool_get_value_bets(args: dict) -> str:
    slip = _load_json(UPCOMING_DIR / "unified_bet_slip.json")
    bets = slip.get("selected_bets", [])
    summary = slip.get("summary", {})

    result = {
        "generated_at": slip.get("generated_at"),
        "total_bets": len(bets),
        "summary": summary,
        "bets": [
            {
                "match": b.get("match"),
                "market": b.get("market"),
                "selection": b.get("selection"),
                "odds": b.get("best_odds"),
                "edge_pct": b.get("edge_pct"),
                "bookmaker": b.get("best_bookmaker"),
                "kelly_stake": b.get("kelly_stake_pct", b.get("stake_pct")),
                "model_prob": b.get("model_prob"),
            }
            for b in bets
        ],
    }

    # Add handicap bets
    hcap = _load_json(UPCOMING_DIR / "handicap_bets.json")
    hcap_bets = hcap.get("recommended", [])
    if hcap_bets:
        result["handicap_bets"] = [
            {
                "match": b.get("match"),
                "bet": b.get("bet"),
                "italian_format": b.get("italian_format"),
                "odds": b.get("odds"),
                "our_probability": b.get("our_probability"),
                "value_pct": b.get("value_pct"),
                "stake_pct": b.get("stake_pct"),
            }
            for b in hcap_bets[:10]
        ]

    # Add over/under bets
    ou = _load_json(UPCOMING_DIR / "over_under_bets.json")
    ou_bets = ou.get("recommended", [])
    if ou_bets:
        result["over_under_bets"] = [
            {
                "match": b.get("match"),
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
        sof_path = _BASE / "data" / "external" / "sofascore" / "player_match_stats.parquet"
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
                f"⚠️ VERIFIED SOFASCORE DATA — COPY EXACTLY.\n"
                f"Best rated: {best_player} ({best_rating})\n\n"
                f"{header}\n" + "\n".join(rows)
            )

        # Match events
        inc_path = _BASE / "data" / "external" / "sofascore" / "match_incidents.parquet"
        if inc_path.exists():
            try:
                inc = pd.read_parquet(inc_path)
                match_events = inc[inc["match_id"] == match_id].sort_values("minute")
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
    live = _load_json(LIVE_DIR / "scores.json", default=[])
    if not live:
        live = _load_json(LIVE_DIR / "live_state.json", default={})
        if isinstance(live, dict):
            live = live.get("matches", [])
    if not live:
        return json.dumps({"status": "No live matches currently."})
    return json.dumps({"live_matches": live}, default=str)


def _tool_get_results(args: dict) -> str:
    """Get recent match results and settled bets."""
    date_filter = args.get("date", "")  # optional YYYY-MM-DD filter

    # Match results
    results_data = _load_json(UPCOMING_DIR / "results.json")
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
    history = _load_json(BETTING_DIR / "history.json")
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
    state = _get_bankroll()
    history = _load_json(BETTING_DIR / "history.json")

    result: dict[str, Any] = {
        "current_bankroll": state.get("current_bankroll"),
        "initial_bankroll": state.get("initial_bankroll"),
        "peak_bankroll": state.get("peak_bankroll"),
        "daily_pnl": state.get("daily_pnl"),
        "total_bets": state.get("total_bets"),
        "total_wins": state.get("total_wins"),
        "current_streak": state.get("current_streak"),
        "last_updated": state.get("last_updated"),
    }

    # ROI calculation
    initial = state.get("initial_bankroll", 1000)
    current = state.get("current_bankroll", initial)
    if initial > 0:
        result["roi_pct"] = round((current - initial) / initial * 100, 2)

    # Recent results from history
    totals = history.get("totals", {}) if isinstance(history, dict) else {}
    if totals:
        result["history_totals"] = totals

    # P&L history (recent daily snapshots)
    pnl = _load_json(BETTING_DIR / "pnl_history.json", default=[])
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

    # CLV tracking (Closing Line Value — are we beating closing odds?)
    clv = _load_json(BETTING_DIR / "clv_history.json")
    if isinstance(clv, dict) and clv.get("running_clv") is not None:
        result["clv"] = {
            "running_clv_pct": clv.get("running_clv_pct"),
            "total_tracked": clv.get("total_bets_tracked"),
            "positive_clv_count": clv.get("positive_clv_count"),
            "negative_clv_count": clv.get("negative_clv_count"),
        }

    # Win rate by market
    settled = history.get("settled_bets", []) if isinstance(history, dict) else history if isinstance(history, list) else []
    if settled:
        market_stats: dict[str, dict] = {}
        for b in settled:
            mkt = b.get("market", "unknown")
            ms = market_stats.setdefault(mkt, {"wins": 0, "losses": 0, "profit": 0.0})
            if b.get("status", "").lower() in ("won", "win"):
                ms["wins"] += 1
            elif b.get("status", "").lower() in ("lost", "loss"):
                ms["losses"] += 1
            ms["profit"] += b.get("profit", b.get("profit_loss", 0))
        result["market_breakdown"] = {
            k: {**v, "profit": round(v["profit"], 2), "win_rate": round(v["wins"] / max(v["wins"] + v["losses"], 1) * 100, 1)}
            for k, v in market_stats.items()
        }

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
    refs = _load_json(UPCOMING_DIR / "referees.json")
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
    weather = _load_json(UPCOMING_DIR / "weather.json")
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
    odds_mv = _load_json(UPCOMING_DIR / "odds_movement.json")
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
    cms = _load_json(UPCOMING_DIR / "cross_market_signals.json")
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
    odds_full = _load_json(UPCOMING_DIR / "odds_full.json")
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
    props = _load_json(UPCOMING_DIR / "player_props.json")
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
    prop_odds = _load_json(UPCOMING_DIR / "player_prop_odds.json")
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
        slip = _load_json(UPCOMING_DIR / "unified_bet_slip.json")
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
    slip = _load_json(UPCOMING_DIR / "unified_bet_slip.json")
    all_bets = slip.get("selected_bets", [])

    # Also check handicap + O/U specialty bets
    hcap = _load_json(UPCOMING_DIR / "handicap_bets.json")
    ou = _load_json(UPCOMING_DIR / "over_under_bets.json")

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
    matches_path = _BASE / "data" / "parsed" / "matches.parquet"
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
        features_path = _BASE / "data" / "features" / "features.parquet"
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

    slip = _load_json(UPCOMING_DIR / "unified_bet_slip.json")
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
    balance = bankroll.get("balance", bankroll.get("current_balance", 1000))
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
        "description": "Get comprehensive player statistics including PER-MATCH RATINGS and FULL TEAM SHEET. Returns: season totals, per-90 rates, recent form (last 5 matches with Sofascore ratings), team ranking, upcoming props, PLUS the full team sheet from the player's most recent match (all teammates with ratings, goals, assists, minutes). Also returns match events (goals, subs, cards) and availability/injury detection. Use for player analysis AND for match analysis (query any player from the match to get the full team sheet). Fuzzy name matching supported.",
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
        "description": "Get FULL RATED LINEUPS for both teams in a specific match. Returns Sofascore ratings, goals, assists, shots, key passes, touches for every player on both sides, plus match events (goals, subs, cards). Defaults to the MOST RECENT match between the two teams if no date given. Use this when asked 'who was the best player in X vs Y', 'show me ratings', 'match analysis', 'full analysis of X vs Y', or ANY question about player performances in a past match. ALWAYS call this tool when a user asks about a match between two teams — do NOT ask for clarification about dates.",
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
        """You are SerieAI Advisor — an opinionated Serie A betting analyst. You have real data tools. NEVER guess, speculate, or say "I don't have access". If you don't know, call a tool.

## PERSONALITY
- Talk like a sharp bettor, not a chatbot. Be direct, concise, opinionated.
- Give a clear VERDICT on every match/bet question. Never sit on the fence.
- Say "I'd bet this" or "I'd skip this" — not "it depends on your risk tolerance."
- Use numbers to back every claim. No vague statements like "they're in good form."
- The user understands betting, odds, xG, Kelly — don't over-explain basics.

## TOOL USAGE RULES
- ALWAYS call tools before answering. Never answer from memory alone.
- For match analysis: call BOTH get_match_prediction AND get_match_context. Always.
- For "what happened today" / results / scores: call get_results with today's date.
- For "best bets" / "what should I bet": call get_value_bets.
- For player questions: call get_player_stats (it has full season data, per-90s, team ranking).
- For team questions: call get_team_detail (has top scorers, form, standings, recent results).
- For "who will score" / goalscorer questions: call get_match_scorers. It has anytime goal probabilities, xG/90, season goals, recent form, and bookmaker odds per player.
- For "settle my bets" / "update results" / "check if my bets won" / "refresh results": call settle_bets. This fetches live results and settles everything. Then call get_results and get_bankroll_status to show the user what happened.
- For "place the X bet" / "back X" / "bet on X" / "I want X" / "place all bets": call place_bet. If multiple bets match, present them and ask which one. After placing, confirm with bet details.
- For "my bets" / "pending bets" / "what have I placed?": call manage_bets with action=list.
- For "cancel the X bet" / "void X" / "remove X bet": call manage_bets with action=cancel.
- For "change stake on X" / "update odds on X": call manage_bets with action=update.
- For "build a parlay" / "make an accumulator" / "combine X and Y" / "best 3-leg parlay": call build_parlay. Use auto_best=N for auto-selection or provide specific team names in matches array.
- For "how does X do at home?" / "what's the over rate for X?" / "X in derbies" / "X vs Y record" / "X on short rest" / "X with referee Y": call query_history. This has 7889 matches across 21 seasons with goals, xG, corners, cards, clean sheets, BTTS, half-time data, and situational flags. Use it for ANY historical pattern question.
- If user asks about something vague, pick the most useful tool. Don't ask for clarification unless truly ambiguous.

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
When analyzing a match, structure your response as:

**1. Verdict** — One sentence: who wins and why. Bold it.
**2. Prediction** — 1X2 probabilities, predicted outcome, confidence level.
**3. Key Numbers** — Table with: expected goals (home/away), BTTS prob, over/under lines, expected corners, expected cards.
**4. Lineups** — Formation + key players. Flag injuries/absences from unavailable list.
**5. Value Bets** — If our model found edge: show market, selection, odds, edge%, Kelly stake. If no edge: say "No value at current odds."
**6. Sharp Money** — What are sharp bookmakers pricing vs. soft books? Any divergence = smart money signal.
**7. Verdict Reasoning** — 2-3 sentences connecting form, H2H, home/away splits, and tactical matchup.

## BETTING ANALYSIS RULES
- ALWAYS compare model probability vs. implied probability. This IS the edge.
- Edge < 3%: "Not enough edge, skip."
- Edge 3-6%: "Marginal value, small stake only."
- Edge > 6%: "Clear value. Kelly says X% stake."
- If BTTS prob < 45%: explicitly say "BTTS No is the play" or "Skip BTTS market."
- If Over 2.5 prob < 48%: "Under is more likely. Check Under line for value."
- Always mention which bookmaker has the best odds.
- Handicap/margin: translate to what it means ("Juve -1.5 means they need to win by 2+").

## MATCH PLAYER ANALYSIS — USE get_match_players
- When asked "who was the best player in X vs Y", "show me match ratings", "match analysis X vs Y":
  → Call get_match_players(home="X", away="Y") — returns BOTH teams' full rated lineups + match events in ONE call
  → It returns pre-rendered markdown tables with exact Sofascore ratings — COPY THEM VERBATIM
  → NEVER use get_player_stats for match-level analysis — use get_match_players instead
  → get_player_stats is for INDIVIDUAL player deep-dives (season stats, form, injury detection, upcoming props)

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
- When analyzing a player's match: tell the STORY. Who scored, who got subbed in/out, how the player compared to teammates, was he the best/worst rated, did he play out of position.
- Always state their team ranking: "Top scorer" or "#3 in assists at the club."
- Show per-90 stats, not just totals (minutes matter).
- Recent form (last 5): are they trending up or down?
- If they have an upcoming prop: show the fair odds vs. market odds.

## GOALSCORER ANALYSIS RULES
- Call get_match_scorers to get ranked candidates with real probabilities.
- Present a table: Player | Team | Goal Prob | Fair Odds | Season Goals | xG/90 | Last 3 Goals
- Give a clear pick: "Best anytime goalscorer bet: [Player] at [fair odds]. If bookmakers offer above [X], it's value."
- Flag hot streaks: "Scored in last 2 of 3 games" matters more than season totals.
- Compare fair odds vs. bookmaker odds when available. Fair odds below bookmaker = value.
- If a top scorer has low minutes recently (rotation/injury), flag it.
- Don't just list — rank and opine. "Lautaro is the obvious pick but overpriced. Better value on [secondary striker]."

## BANKROLL ANALYSIS RULES
- Show ROI, current balance, peak, and drawdown from peak.
- Win rate by market: which markets are profitable, which are bleeding money?
- CLV: are we beating closing lines? Positive CLV = sustainable edge. Negative CLV = we got lucky or model is off.
- If bankroll is down from peak by >15%, flag it as a concern.

## FORMATTING
- Use markdown tables for multi-row comparisons (odds, probabilities, player stats).
- Bold key numbers and verdicts.
- Keep responses tight. No filler paragraphs. Every sentence should contain data or an opinion.
- Use | tables for structured data, bullet points for narrative.

## CRITICAL: NEVER DO THESE
- Never say "I can't access historical data" — you have get_results.
- Never say "check ESPN/flashscore/other website" — all data is in your tools.
- Never give a wishy-washy non-answer. If data is insufficient, say what's missing specifically.
- Never hallucinate odds, scores, or stats. If a tool returns no data, say "no data available for X."
- **NEVER fabricate player ratings, stats, or lineups.** Only cite numbers that appear EXACTLY in the tool response data. If a player's rating is 6.2 in the data, report 6.2 — not 8.0. If a player is NOT in the tool results, do NOT mention them as if they played.
- When showing match player stats: ONLY include players returned by the tool. Do NOT add players from memory or assumption. The tool data is the single source of truth.
- Never explain what xG or Kelly criterion means unless explicitly asked.""",
        "",
    ]

    # Inject standings summary
    standings_data = _load_json(UPCOMING_DIR / "standings.json")
    standings = standings_data.get("standings", {})
    if standings:
        sorted_teams = sorted(standings.values(), key=lambda x: x.get("position", 99))
        parts.append("## Current Serie A Standings (Top 10)")
        for t in sorted_teams[:10]:
            parts.append(
                f"{t.get('position', '?')}. {t.get('team', '?')} — "
                f"{t.get('points', 0)}pts, {t.get('wins', 0)}W-{t.get('draws', 0)}D-{t.get('losses', 0)}L, "
                f"GD {t.get('gd', 0):+d}, Form: {t.get('form_last5', '?')}"
            )
        parts.append("")

    # Inject form summary
    form_data = _load_json(UPCOMING_DIR / "current_form.json")
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

    # Inject bankroll
    bank = _load_json(BANKROLL_DIR / "state.json")
    if bank.get("current_bankroll"):
        initial = bank.get("initial_bankroll", 1000)
        current = bank.get("current_bankroll", initial)
        roi = (current - initial) / initial * 100 if initial > 0 else 0
        parts.append(
            f"## Bankroll: ${current:,.2f} (ROI: {roi:+.1f}%, "
            f"Peak: ${bank.get('peak_bankroll', current):,.2f}, "
            f"Streak: {bank.get('current_streak', 0)})"
        )
        parts.append("")

    parts.append("Today is " + datetime.now().strftime("%A, %B %d, %Y") + ".")

    # Inject upcoming matches so Claude knows what's scheduled
    predictions = _load_json(UPCOMING_DIR / "predictions.json")
    upcoming_list = predictions.get("predictions", [])
    if isinstance(upcoming_list, list) and upcoming_list:
        match_lines = []
        for m in upcoming_list[:12]:
            home = m.get("home_team", "?")
            away = m.get("away_team", "?")
            pred = m.get("predicted_outcome", "?")
            conf = m.get("confidence", 0)
            match_lines.append(f"- {home} vs {away} → {pred} ({conf:.0%})")
        parts.append("\n## Upcoming Matches\n" + "\n".join(match_lines))
        parts.append("When a user asks about 'today match', 'upcoming match', 'this weekend', or 'next match', refer to these fixtures. Use get_match_prediction for detailed analysis.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Greeting builder (zero API cost)
# ---------------------------------------------------------------------------

def _build_greeting() -> dict:
    """Build smart greeting from local data — no Claude API call.

    Logic:
    - If matches settled today → show results + P&L + bankroll
    - Otherwise → show value bets + upcoming matches + bankroll
    """
    sections = []
    today = datetime.now().strftime("%Y-%m-%d")

    # Check if matches settled today
    history = _load_json(BETTING_DIR / "history.json")
    settled_bets = history.get("settled_bets", []) if isinstance(history, dict) else history if isinstance(history, list) else []
    today_settled = [
        b for b in settled_bets
        if today in (b.get("settled_at", "") or b.get("date", ""))
    ]

    # Check today's results
    results_data = _load_json(UPCOMING_DIR / "results.json")
    results_all = results_data.get("results", {})
    today_results = []
    if isinstance(results_all, dict):
        for key, r in results_all.items():
            if today in (r.get("commence_time", "") or ""):
                today_results.append(r)
    elif isinstance(results_all, list):
        today_results = [r for r in results_all if today in (r.get("commence_time", "") or r.get("date", ""))]

    if today_results:
        # MATCHES PLAYED TODAY — show results
        result_lines = []
        for r in today_results:
            result_lines.append(
                f"  - **{r.get('home_team', '?')} {r.get('home_score', '?')}-{r.get('away_score', '?')} {r.get('away_team', '?')}**"
            )
        sections.append("### Today's Results\n" + "\n".join(result_lines))

        # Today's betting P&L
        if today_settled:
            wins = sum(1 for b in today_settled if b.get("status", "").lower() in ("won", "win"))
            losses = sum(1 for b in today_settled if b.get("status", "").lower() in ("lost", "loss"))
            profit = sum(b.get("profit", b.get("profit_loss", 0)) for b in today_settled)
            sections.append(
                f"### Today's Bets: {wins}W-{losses}L | P&L: **${profit:+.2f}**"
            )
            # Show individual results
            bet_lines = []
            for b in today_settled[:8]:
                status_icon = "+" if b.get("status", "").lower() in ("won", "win") else "-"
                p = b.get("profit", b.get("profit_loss", 0))
                bet_lines.append(
                    f"  - [{status_icon}] {b.get('match', '?')} — {b.get('selection', '?')} "
                    f"({b.get('market', '?')}) ${p:+.2f}"
                )
            sections.append("\n".join(bet_lines))

    # Always show value bets for upcoming matches (if any exist)
    slip = _load_json(UPCOMING_DIR / "unified_bet_slip.json")
    bets = slip.get("selected_bets", [])
    # Filter out bets for already-completed matches
    completed_matches = {r.get("match", "") for r in today_results}
    future_bets = [b for b in bets if b.get("match", "") not in completed_matches]
    if future_bets:
        bet_lines = []
        for b in future_bets[:5]:
            edge = b.get("edge_pct", 0)
            odds = b.get("best_odds", "?")
            bet_lines.append(
                f"  - **{b.get('match', '?')}** — {b.get('selection', '?')} "
                f"({b.get('market', '?')}) @ {odds} | Edge: {edge}%"
            )
        sections.append("### Value Bets (Upcoming)\n" + "\n".join(bet_lines))
    elif not today_results:
        sections.append("*No value bets currently available.*")

    # Upcoming matches (not yet played)
    preds = _load_json(UPCOMING_DIR / "predictions.json")
    predictions = preds.get("predictions", [])
    future_matches = [
        p for p in predictions
        if f"{p.get('home_team', '?')} vs {p.get('away_team', '?')}" not in completed_matches
    ]
    if future_matches:
        match_lines = []
        for p in future_matches[:6]:
            conf = p.get("confidence", {})
            conf_label = conf.get("label", "?") if isinstance(conf, dict) else str(conf)
            match_lines.append(
                f"  - {p.get('home_team', '?')} vs {p.get('away_team', '?')} — "
                f"Prediction: **{p.get('predicted_outcome', '?')}** ({conf_label})"
            )
        sections.append("### Upcoming Matches\n" + "\n".join(match_lines))

    # Bankroll (always, from freshest source)
    bank = _get_bankroll()
    if bank.get("current_bankroll"):
        initial = bank.get("initial_bankroll", 1000)
        current = bank.get("current_bankroll", initial)
        roi = (current - initial) / initial * 100 if initial > 0 else 0
        sections.append(
            f"### Bankroll\n"
            f"  Balance: **${current:,.2f}** | ROI: **{roi:+.1f}%** | "
            f"Peak: ${bank.get('peak_bankroll', current):,.2f}"
        )

    greeting = "Welcome back. Here's your current Serie A intelligence:\n\n" + "\n\n".join(sections)
    greeting += "\n\n---\n*Ask me anything — match analysis, player stats, betting strategy, or bankroll review.*"

    return {"role": "assistant", "content": greeting}


# ---------------------------------------------------------------------------
# Conversation store (in-memory, per-session)
# ---------------------------------------------------------------------------
_conversations: dict[str, list] = {}
MAX_HISTORY = 60  # includes tool call/result pairs now


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------

def _track_usage(usage: dict):
    """Append usage stats to data/api_usage.json."""
    try:
        data = _load_json(USAGE_FILE, default={"daily": {}, "total": {}})
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

        # Haiku 4.5 pricing: input $0.80/M, output $4/M, cache read $0.08/M, cache write $1/M
        cost = (
            usage.get("input_tokens", 0) * 0.80 / 1_000_000
            + usage.get("output_tokens", 0) * 4.0 / 1_000_000
            + usage.get("cache_read_input_tokens", 0) * 0.08 / 1_000_000
            + usage.get("cache_creation_input_tokens", 0) * 1.0 / 1_000_000
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

    # Trim to max history
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    def generate():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            system_prompt = _build_system_prompt()
            messages = list(history)
            max_rounds = 5

            for round_num in range(max_rounds):
                # Stream response with prompt caching
                with client.messages.stream(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=2048,
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
                        _track_usage(final_message.usage.__dict__ if hasattr(final_message.usage, "__dict__") else {})

                    stop_reason = final_message.stop_reason if final_message else "end_turn"

                # If no tool use, we're done
                if stop_reason != "tool_use" or not tool_uses:
                    if full_text:
                        history.append({"role": "assistant", "content": full_text})
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

                # Execute each tool
                tool_results = []
                for tu in tool_uses:
                    handler = TOOL_HANDLERS.get(tu["name"])
                    if handler:
                        try:
                            result_str = handler(tu["input"])
                        except Exception as e:
                            result_str = json.dumps({"error": str(e)})
                    else:
                        result_str = json.dumps({"error": f"Unknown tool: {tu['name']}"})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu["id"],
                        "content": result_str,
                    })

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
