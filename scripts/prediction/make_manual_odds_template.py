"""Track 2 — Manual odds JSON template generator.

Reads data/upcoming/simulator_shadow_log.json (predictions) and writes a
blank-but-structured JSON file at data/upcoming/manual_odds_log.json.
User opens this file in any editor Saturday morning and fills in the odds
they see at their bookmaker (Scommesse via VPN, etc.).

The template includes the same market labels the shadow predictor emits,
so the ROI computer can trivially join them. ALL FIELDS ARE OPTIONAL —
leave any blank and that market simply gets skipped in the ROI calculation.

Usage:
    python3 scripts/prediction/make_manual_odds_template.py
    python3 scripts/prediction/make_manual_odds_template.py --preserve-existing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SHADOW_LOG_PATH = PROJECT_ROOT / "data" / "upcoming" / "simulator_shadow_log.json"
MANUAL_ODDS_PATH = PROJECT_ROOT / "data" / "upcoming" / "manual_odds_log.json"


# Market labels that benefit from manual odds entry.
# Each entry is (market_key, description, notes_for_user).
MARKETS_TO_REQUEST = [
    # Basic
    ("odds_1", "Home win (1)", "e.g. 1.33"),
    ("odds_X", "Draw (X)", "e.g. 4.75"),
    ("odds_2", "Away win (2)", "e.g. 10.0"),
    ("odds_over_0_5", "Over 0.5 goals", ""),
    ("odds_under_0_5", "Under 0.5 goals", ""),
    ("odds_over_1_5", "Over 1.5 goals", "usually ~1.25-1.50"),
    ("odds_under_1_5", "Under 1.5 goals", ""),
    ("odds_over_2_5", "Over 2.5 goals", "e.g. 1.85"),
    ("odds_under_2_5", "Under 2.5 goals", ""),
    ("odds_over_3_5", "Over 3.5 goals", ""),
    ("odds_under_3_5", "Under 3.5 goals", ""),
    ("odds_over_4_5", "Over 4.5 goals", ""),
    ("odds_under_4_5", "Under 4.5 goals", ""),
    # BTTS
    ("odds_btts_yes", "Both teams to score — YES", "e.g. 2.25"),
    ("odds_btts_no", "Both teams to score — NO", ""),
    # Clean sheets
    ("odds_home_clean_sheet_yes", "Home clean sheet — YES", ""),
    ("odds_away_clean_sheet_yes", "Away clean sheet — YES", ""),
    # Double chance
    ("odds_dc_1X", "Double chance 1X (home or draw)", ""),
    ("odds_dc_X2", "Double chance X2 (draw or away)", ""),
    ("odds_dc_12", "Double chance 12 (home or away)", ""),
    # DNB
    ("odds_dnb_home", "Draw No Bet — home", ""),
    ("odds_dnb_away", "Draw No Bet — away", ""),
    # Corners (from Scommesse: these are the primary shadow-test targets!)
    ("odds_corners_over_7_5", "Corners over 7.5", ""),
    ("odds_corners_under_7_5", "Corners under 7.5", ""),
    ("odds_corners_over_8_5", "Corners over 8.5", ""),
    ("odds_corners_under_8_5", "Corners under 8.5", ""),
    ("odds_corners_over_9_5", "Corners over 9.5", ""),
    ("odds_corners_under_9_5", "Corners under 9.5", ""),
    ("odds_corners_over_10_5", "Corners over 10.5", ""),
    ("odds_corners_under_10_5", "Corners under 10.5", ""),
    ("odds_corners_over_11_5", "Corners over 11.5", ""),
    ("odds_corners_under_11_5", "Corners under 11.5", ""),
    # Team corners
    ("odds_home_corners_over_4_5", "Home corners over 4.5", ""),
    ("odds_home_corners_under_4_5", "Home corners under 4.5", ""),
    ("odds_away_corners_over_4_5", "Away corners over 4.5", ""),
    ("odds_away_corners_under_4_5", "Away corners under 4.5", ""),
    # First-half corners (you mentioned this)
    ("odds_fh_corners_1", "FH corners: home more (1)", ""),
    ("odds_fh_corners_X", "FH corners: equal (X)", ""),
    ("odds_fh_corners_2", "FH corners: away more (2)", ""),
    # Cards
    ("odds_cards_over_2_5", "Cards over 2.5", ""),
    ("odds_cards_under_2_5", "Cards under 2.5", ""),
    ("odds_cards_over_3_5", "Cards over 3.5", ""),
    ("odds_cards_under_3_5", "Cards under 3.5", ""),
    ("odds_cards_over_4_5", "Cards over 4.5", ""),
    ("odds_cards_under_4_5", "Cards under 4.5", ""),
    ("odds_cards_over_5_5", "Cards over 5.5", ""),
    ("odds_cards_under_5_5", "Cards under 5.5", ""),
    # Asian Handicap
    ("odds_AH_home_-2.5", "AH home -2.5", ""),
    ("odds_AH_home_-1.5", "AH home -1.5", ""),
    ("odds_AH_home_-0.5", "AH home -0.5", ""),
    ("odds_AH_home_+0.5", "AH home +0.5", ""),
    ("odds_AH_home_+1.5", "AH home +1.5", ""),
    # Shots
    ("odds_shots_over_22_5", "Total shots over 22.5", ""),
    ("odds_shots_under_22_5", "Total shots under 22.5", ""),
    ("odds_shots_over_24_5", "Total shots over 24.5", ""),
    ("odds_shots_under_24_5", "Total shots under 24.5", ""),
    # SOT
    ("odds_sot_over_7_5", "Shots on target over 7.5", ""),
    ("odds_sot_under_7_5", "Shots on target under 7.5", ""),
    ("odds_sot_over_8_5", "Shots on target over 8.5", ""),
    ("odds_sot_under_8_5", "Shots on target under 8.5", ""),
]


def build_template_for_fixture(shadow_fixture: dict) -> dict:
    """One entry per upcoming match, blank odds fields + predicted probs."""
    entry = {
        "match_key": shadow_fixture["match_key"],
        "home_team": shadow_fixture["home_team"],
        "away_team": shadow_fixture["away_team"],
        "match_date": shadow_fixture["match_date"],
        "kickoff_utc": shadow_fixture.get("kickoff_utc", ""),
        "source": "",  # e.g. "scommesse", "bet365", etc.
        "odds_captured_at": "",  # ISO timestamp when user noted the odds
        "odds": {m[0]: None for m in MARKETS_TO_REQUEST},
        "player_prop_odds": {
            # Free-form: user adds {player_name: {market_type: odds}}
            # Predictor will try to match by name against top_scorers.
            # Example:
            #   "Scott McTominay": {"anytime_scorer_yes": 2.5, "shots_over_1_5": 3.2}
        },
        "_predicted_probs_for_reference": {
            # Snapshot of the simulator's probabilities so you can compare at-a-glance
            "1X2": shadow_fixture["goal_markets"].get("1X2"),
            "over_2_5": shadow_fixture["goal_markets"].get("over_2_5"),
            "btts": shadow_fixture["goal_markets"].get("btts"),
            "expected_corners": shadow_fixture["expected"].get("corners_total"),
            "expected_cards": shadow_fixture["expected"].get("cards_total"),
            "top_scorers": [
                {"name": p["player_name"], "prob": p["anytime_scorer_prob"]}
                for p in shadow_fixture.get("top_scorers", [])[:5]
            ],
        },
        "_market_descriptions": {m[0]: m[1] + (f" ({m[2]})" if m[2] else "") for m in MARKETS_TO_REQUEST},
    }
    return entry


def run(shadow_log_path: Path, output_path: Path, preserve_existing: bool = False) -> dict:
    log.info("Loading shadow predictions from %s", shadow_log_path)
    with open(shadow_log_path) as f:
        shadow = json.load(f)
    fixtures = shadow.get("fixtures", [])

    # If preserving existing, load and only overwrite fixtures that aren't yet filled
    existing_data: dict[str, dict] = {}
    if preserve_existing and output_path.exists():
        try:
            prior = json.loads(output_path.read_text())
            for entry in prior.get("fixtures", []):
                existing_data[entry.get("match_key", "")] = entry
            log.info("Preserving odds from %d existing entries", len(existing_data))
        except Exception as e:
            log.warning("Failed to read existing file: %s", e)

    fixture_entries = []
    for fx in fixtures:
        mk = fx["match_key"]
        if mk in existing_data and any(v for v in existing_data[mk].get("odds", {}).values()):
            # User has filled in odds for this fixture — keep as-is
            fixture_entries.append(existing_data[mk])
        else:
            fixture_entries.append(build_template_for_fixture(fx))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_shadow_run_id": shadow.get("run_id"),
        "n_fixtures": len(fixture_entries),
        "instructions": (
            "Fill in ONLY the odds you check at your bookmaker. Leave others as null. "
            "Numeric fields are decimal odds (1.85 = implied ~54% prob). "
            "For player props, add entries under player_prop_odds keyed by player name. "
            "Delete this 'instructions' field when done if you want a cleaner file."
        ),
        "fixtures": fixture_entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("Wrote template for %d fixtures to %s", len(fixture_entries), output_path)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shadow-log", default=str(SHADOW_LOG_PATH))
    ap.add_argument("--output", default=str(MANUAL_ODDS_PATH))
    ap.add_argument("--preserve-existing", action="store_true",
                    help="Keep entries where user already filled in odds")
    args = ap.parse_args()

    payload = run(Path(args.shadow_log), Path(args.output), args.preserve_existing)
    print(f"\nTemplate written for {payload['n_fixtures']} fixtures")
    print(f"File: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Open the file in any editor")
    print(f"  2. Fill in the odds you see at your bookmaker (leave others null)")
    print(f"  3. Save")
    print(f"  4. Run: python3 scripts/prediction/compute_shadow_roi.py")
    print(f"\nEach fixture shows _predicted_probs_for_reference so you can see ")
    print(f"simulator's probabilities alongside the blank odds fields.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
