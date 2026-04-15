"""Generate a unified match report consolidating all available intelligence.

Reads ~19 separate JSON files from data/upcoming/ and merges them into a single
unified_report.json with one entry per match. This gives the dashboard and Claude
a single file to read for complete match intelligence.

Usage:
    python3 -m scripts.prediction.generate_unified_report

Input files (all optional — missing data is gracefully skipped):
    predictions_unified.json    — Ensemble probabilities + betting recs
    unified_bet_slip.json       — Selected value bets + stakes
    player_analysis.json        — Player-level analysis
    current_form.json           — Team form + matchup data
    h2h_upcoming.json           — Head-to-head history
    market_intelligence.json    — Market analysis + line movements
    weather.json                — Weather conditions
    referees.json               — Referee assignments
    lineup_predictions.json     — Expected lineups + player xG
    cross_market_signals.json   — Cross-market correlation signals
    bookmaker_analysis.json     — Bookmaker consensus + sharp vs soft
    goal_predictions.json       — Poisson xG predictions
    margin_predictions.json     — Score margin predictions
    extended_markets.json       — O/U, AH, DC, DNB, BTTS odds
    odds_full.json              — Full odds data (1X2 + extras)
    sentiment_analysis.json     — Sentiment data
    standings.json              — Current league table
    btts_predictions.json       — BTTS predictions
    cards_predictions.json      — Cards predictions
    corners_predictions.json    — Corners predictions

Output:
    data/upcoming/unified_report.json
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.utils.json_utils import load_json_safe

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "upcoming"


def _normalize_match_key(key: str) -> str:
    """Normalize match key for fuzzy matching: lowercase, strip whitespace."""
    return key.strip().lower()


def _build_match_index_from_list(items: list, key_field: str = "match") -> Dict[str, dict]:
    """Build {normalized_match_key: item} from a list of dicts."""
    index = {}
    for item in items:
        if isinstance(item, dict):
            mk = item.get(key_field, "")
            if mk:
                index[_normalize_match_key(mk)] = item
    return index


def _build_match_index_from_dict(d: dict) -> Dict[str, Any]:
    """Build {normalized_match_key: value} from a dict keyed by match names."""
    return {_normalize_match_key(k): v for k, v in d.items()}


def _extract_matches(data: Any, list_key: str, key_field: str = "match") -> Dict[str, Any]:
    """Extract match-indexed data from various JSON structures.

    Handles both:
      - {"matches": [{"match": "A vs B", ...}, ...]}  (list of dicts)
      - {"matches": {"A vs B": {...}, ...}}            (dict keyed by match)
    """
    if data is None:
        return {}

    items = data.get(list_key, data) if isinstance(data, dict) else data

    if isinstance(items, list):
        return _build_match_index_from_list(items, key_field)
    elif isinstance(items, dict):
        # Filter out metadata keys (timestamps, summaries, etc.)
        match_items = {}
        for k, v in items.items():
            if isinstance(v, dict) and " vs " in k:
                match_items[_normalize_match_key(k)] = v
            elif isinstance(v, (dict, list)) and " vs " in k:
                match_items[_normalize_match_key(k)] = v
        return match_items if match_items else _build_match_index_from_dict(items)
    return {}


def _extract_bets_for_match(bets: list, match_key: str) -> List[dict]:
    """Get all bets for a specific match from the bet slip."""
    norm = _normalize_match_key(match_key)
    return [b for b in bets if _normalize_match_key(b.get("match", "")) == norm]


def generate_unified_report(data_dir: Optional[Path] = None) -> dict:
    """Generate the unified report by consolidating all data files.

    Returns the report dict and writes it to unified_report.json.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    log.info("Generating unified match report from %s", data_dir)

    # ── Load all source files ──────────────────────────────────────
    # Try current pipeline output first, fall back to legacy name
    predictions = load_json_safe(data_dir / "predictions.json", default=None)
    if not predictions:
        predictions = load_json_safe(data_dir / "predictions_unified.json", default=None)
    bet_slip = load_json_safe(data_dir / "unified_bet_slip.json", default=None)
    player_analysis = load_json_safe(data_dir / "player_analysis.json", default=None)
    current_form = load_json_safe(data_dir / "current_form.json", default=None)
    h2h = load_json_safe(data_dir / "h2h_upcoming.json", default=None)
    market_intel = load_json_safe(data_dir / "market_intelligence.json", default=None)
    weather = load_json_safe(data_dir / "weather.json", default=None)
    referees = load_json_safe(data_dir / "referees.json", default=None)
    lineups = load_json_safe(data_dir / "lineup_predictions.json", default=None)
    cross_market = load_json_safe(data_dir / "cross_market_signals.json", default=None)
    bookmaker = load_json_safe(data_dir / "bookmaker_analysis.json", default=None)
    goals = load_json_safe(data_dir / "goal_predictions.json", default=None)
    margins = load_json_safe(data_dir / "margin_predictions.json", default=None)
    extended = load_json_safe(data_dir / "extended_markets.json", default=None)
    odds_full = load_json_safe(data_dir / "odds_full.json", default=None)
    sentiment = load_json_safe(data_dir / "sentiment_analysis.json", default=None)
    standings_data = load_json_safe(data_dir / "standings.json", default=None)
    btts = load_json_safe(data_dir / "btts_predictions.json", default=None)
    cards = load_json_safe(data_dir / "cards_predictions.json", default=None)
    corners = load_json_safe(data_dir / "corners_predictions.json", default=None)

    # ── Build match indexes ────────────────────────────────────────
    pred_index = _extract_matches(predictions, "predictions")
    player_index = _extract_matches(player_analysis, "matches")
    form_index = _extract_matches(current_form, "matchups")
    h2h_index = _extract_matches(h2h, "h2h")
    intel_index = _extract_matches(market_intel, "matches")
    weather_index = _extract_matches(weather, "matches")
    lineup_index = _extract_matches(lineups, "matches")
    cross_index = _extract_matches(cross_market, "matches")
    book_index = _extract_matches(bookmaker, "matches")
    goal_index = _extract_matches(goals, "predictions")
    margin_index = _extract_matches(margins, "predictions")
    ext_index = _extract_matches(extended, "matches")
    odds_index = _extract_matches(odds_full, "matches")
    sent_index = _extract_matches(sentiment, "matches")
    btts_index = _extract_matches(btts, "predictions") if isinstance(btts, dict) else _build_match_index_from_list(btts or [])
    cards_index = _extract_matches(cards, "predictions") if isinstance(cards, dict) else _build_match_index_from_list(cards or [])
    corners_index = _extract_matches(corners, "predictions") if isinstance(corners, dict) else _build_match_index_from_list(corners or [])

    # Referee data is keyed by match name directly
    ref_index = {}
    if isinstance(referees, dict):
        ref_index = {_normalize_match_key(k): v for k, v in referees.items() if " vs " in k}

    # Bet slip: get all selected bets
    all_bets = []
    if isinstance(bet_slip, dict):
        all_bets = bet_slip.get("selected_bets", [])

    # Standings as a lookup
    standings_table = {}
    if isinstance(standings_data, dict):
        for entry in standings_data.get("standings", []):
            if isinstance(entry, dict):
                team = entry.get("team", "")
                if team:
                    standings_table[team.lower()] = entry

    # ── Collect all unique match keys ──────────────────────────────
    all_match_keys = set()
    for idx in [pred_index, player_index, form_index, h2h_index, intel_index,
                weather_index, lineup_index, cross_index, book_index, goal_index,
                margin_index, ext_index, odds_index, sent_index, btts_index,
                cards_index, corners_index, ref_index]:
        all_match_keys.update(idx.keys())

    log.info("Found %d unique matches across all data files", len(all_match_keys))

    # ── Build unified report per match ─────────────────────────────
    matches_report = {}
    sources_used = set()

    for norm_key in sorted(all_match_keys):
        match = {}

        # Get display name from prediction or any source
        display_name = None
        for idx in [pred_index, goal_index, player_index, ext_index, odds_index]:
            entry = idx.get(norm_key)
            if isinstance(entry, dict):
                display_name = entry.get("match", entry.get("match_name"))
                if display_name:
                    break
        if not display_name:
            # Reconstruct from normalized key
            display_name = " vs ".join(w.title() for w in norm_key.split(" vs "))

        match["match"] = display_name

        # ─ Prediction ─
        pred = pred_index.get(norm_key)
        if pred and isinstance(pred, dict):
            sources_used.add("predictions")
            match["prediction"] = {
                "outcome": pred.get("prediction", {}).get("outcome"),
                "probabilities": pred.get("prediction", {}).get("probabilities"),
                "confidence": pred.get("prediction", {}).get("confidence"),
                "methods_used": pred.get("prediction", {}).get("methods_used"),
            }
            match["date"] = pred.get("date")
            match["time"] = pred.get("time")
            match["venue"] = pred.get("venue")

            # Value analysis
            va = pred.get("value_analysis", {})
            if va:
                match["value_analysis"] = {
                    "has_value": va.get("has_value"),
                    "best_bet": va.get("best_bet"),
                    "edge": va.get("edge"),
                }

            # Draw analysis
            da = pred.get("draw_analysis")
            if da:
                match["draw_analysis"] = da

        # ─ Betting recommendations ─
        match_bets = _extract_bets_for_match(all_bets, display_name)
        if match_bets:
            sources_used.add("betting")
            match["bets"] = [{
                "market": b.get("market"),
                "selection": b.get("selection"),
                "odds": b.get("best_odds"),
                "bookmaker": b.get("best_bookmaker"),
                "edge_pct": b.get("edge_pct"),
                "stake": b.get("stake_amount"),
                "confidence": b.get("confidence_tier"),
                "expected_profit": b.get("expected_profit"),
            } for b in match_bets]

        # ─ Goal predictions ─
        gp = goal_index.get(norm_key)
        if gp and isinstance(gp, dict):
            sources_used.add("goals")
            match["goals"] = {
                "expected_home": gp.get("expected_home_goals"),
                "expected_away": gp.get("expected_away_goals"),
                "expected_total": gp.get("expected_total_goals"),
                "over_2_5_prob": gp.get("over_2_5"),
                "under_2_5_prob": gp.get("under_2_5"),
                "btts_yes_prob": gp.get("btts_yes"),
            }

        # ─ Player analysis ─
        pa = player_index.get(norm_key)
        if pa and isinstance(pa, dict):
            sources_used.add("players")
            match["players"] = pa

        # ─ Current form ─
        form = form_index.get(norm_key)
        if form and isinstance(form, dict):
            sources_used.add("form")
            match["form"] = form

        # ─ H2H ─
        h2h_data = h2h_index.get(norm_key)
        if h2h_data:
            sources_used.add("h2h")
            match["h2h"] = h2h_data

        # ─ Weather ─
        wx = weather_index.get(norm_key)
        if wx:
            sources_used.add("weather")
            match["weather"] = wx

        # ─ Referee ─
        ref = ref_index.get(norm_key)
        if ref:
            sources_used.add("referee")
            match["referee"] = ref

        # ─ Lineups ─
        lu = lineup_index.get(norm_key)
        if lu:
            sources_used.add("lineups")
            match["lineups"] = lu

        # ─ Market intelligence ─
        mi = intel_index.get(norm_key)
        if mi:
            sources_used.add("market_intel")
            match["market_intelligence"] = mi

        # ─ Cross-market signals ─
        cm = cross_index.get(norm_key)
        if cm:
            sources_used.add("cross_market")
            match["cross_market_signals"] = cm

        # ─ Bookmaker analysis ─
        ba = book_index.get(norm_key)
        if ba:
            sources_used.add("bookmaker")
            match["bookmaker_analysis"] = ba

        # ─ Odds ─
        od = odds_index.get(norm_key)
        if od:
            sources_used.add("odds")
            match["odds"] = od

        # ─ Extended markets ─
        em = ext_index.get(norm_key)
        if em:
            sources_used.add("extended_markets")
            match["extended_markets"] = em

        # ─ Margin predictions ─
        mp = margin_index.get(norm_key)
        if mp and isinstance(mp, dict):
            sources_used.add("margins")
            match["margin_prediction"] = {
                "most_likely_score": mp.get("most_likely_score"),
                "predicted_margin": mp.get("predicted_margin"),
                "home_clean_sheet": mp.get("home_clean_sheet_prob"),
                "away_clean_sheet": mp.get("away_clean_sheet_prob"),
            }

        # ─ Sentiment ─
        sa = sent_index.get(norm_key)
        if sa:
            sources_used.add("sentiment")
            match["sentiment"] = sa

        # ─ BTTS ─
        bt = btts_index.get(norm_key)
        if bt and isinstance(bt, dict):
            sources_used.add("btts")
            match["btts"] = {
                "btts_yes": bt.get("btts_yes"),
                "btts_no": bt.get("btts_no"),
            }

        # ─ Cards ─
        cd = cards_index.get(norm_key)
        if cd and isinstance(cd, dict):
            sources_used.add("cards")
            match["cards"] = {
                "expected_total": cd.get("expected_cards"),
                "expected_home": cd.get("expected_home_cards"),
                "expected_away": cd.get("expected_away_cards"),
            }

        # ─ Corners ─
        co = corners_index.get(norm_key)
        if co and isinstance(co, dict):
            sources_used.add("corners")
            match["corners"] = {
                "expected_total": co.get("expected_corners"),
                "expected_home": co.get("expected_home_corners"),
                "expected_away": co.get("expected_away_corners"),
            }

        # ─ Standings positions ─
        parts = display_name.split(" vs ")
        if len(parts) == 2:
            home_pos = standings_table.get(parts[0].strip().lower(), {})
            away_pos = standings_table.get(parts[1].strip().lower(), {})
            if home_pos or away_pos:
                sources_used.add("standings")
                match["standings"] = {
                    "home": home_pos if home_pos else None,
                    "away": away_pos if away_pos else None,
                }

        # Track data completeness
        sections = [k for k in match.keys() if k not in ("match", "date", "time", "venue")]
        match["_data_sources"] = len(sections)
        match["_sections_available"] = sections

        matches_report[display_name] = match

    # ── Build final report ─────────────────────────────────────────
    report = {
        "generated_at": datetime.now().isoformat(),
        "match_count": len(matches_report),
        "data_sources_found": sorted(sources_used),
        "data_sources_count": len(sources_used),
        "standings": standings_data.get("standings") if standings_data else None,
        "bankroll": bet_slip.get("bankroll") if bet_slip else None,
        "betting_summary": bet_slip.get("summary") if bet_slip else None,
        "matches": matches_report,
    }

    # Write output
    out_path = data_dir / "unified_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Unified report written to %s (%d matches, %d data sources)",
             out_path, len(matches_report), len(sources_used))

    # Print summary
    print(f"\n{'='*60}")
    print(f"UNIFIED MATCH REPORT")
    print(f"{'='*60}")
    print(f"Matches: {len(matches_report)}")
    print(f"Data sources: {len(sources_used)} ({', '.join(sorted(sources_used))})")
    print(f"Output: {out_path}")
    print()

    for name, m in matches_report.items():
        n_sources = m.get("_data_sources", 0)
        sections = m.get("_sections_available", [])
        has_bets = "bets" in sections
        pred = m.get("prediction", {})
        outcome = pred.get("outcome", "?")
        conf = pred.get("confidence", 0)
        probs = pred.get("probabilities", {})

        bet_str = ""
        if has_bets:
            for b in m["bets"]:
                bet_str += f" | {b['market']} {b['selection']} @{b['odds']} +{b['edge_pct']:.0f}%"

        prob_str = ""
        if probs:
            prob_str = f"H={probs.get('home',0):.0%} D={probs.get('draw',0):.0%} A={probs.get('away',0):.0%}"

        print(f"  {name}")
        print(f"    {outcome} ({prob_str}) | {n_sources} data sections{bet_str}")

    print(f"\n{'='*60}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_unified_report()
