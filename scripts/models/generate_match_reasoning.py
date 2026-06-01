"""One-sentence per-match reasoning, derived from existing prediction bundles.

NOT an LLM call — a deterministic template that picks the strongest signal
from the match's prediction outputs and writes one sentence explaining it.

Reads:
    data/upcoming/predictions.json
    data/upcoming/corners_predictions.json
    data/upcoming/cards_predictions.json
    data/upcoming/scorers_predictions.json (optional)

Writes:
    data/upcoming/match_reasoning.json — one entry per match with `reasoning` key.

Run:
    python3 -m scripts.models.generate_match_reasoning
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

UPCOMING_DIR = DATA_DIR / "upcoming"

# Base rates from training-time data (used to flag "X is unusually high/low")
SA_CARDS_OVER_3_5_BASE = 0.64
SA_CARDS_OVER_4_5_BASE = 0.40
SA_CORNERS_OVER_9_5_BASE = 0.50
SA_CORNERS_OVER_10_5_BASE = 0.35
SCORER_HIGH_THRESHOLD = 0.30  # individual goal prob "high" cutoff


def _load_optional(path: Path, default):
    if not path.exists():
        return default
    try:
        d = json.loads(path.read_text())
        if isinstance(d, dict) and "predictions" in d:
            return d["predictions"]
        return d
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return default


def _index(preds, *keys) -> dict:
    """Index prediction list by composite key (e.g., 'match' + 'date')."""
    out: dict = {}
    for p in preds:
        if not isinstance(p, dict):
            continue
        composite = tuple(p.get(k, "") for k in keys)
        out[composite] = p
    return out


def _build_reasoning(
    match_pred: dict,
    corners_pred: dict | None,
    cards_pred: dict | None,
    scorers_pred: dict | None,
) -> str:
    """Pick the strongest signal across the bundle and return one sentence."""
    home = match_pred.get("home_team", "")
    away = match_pred.get("away_team", "")
    probs = match_pred.get("probabilities", {})
    p_h = probs.get("home", 0)
    p_d = probs.get("draw", 0)
    p_a = probs.get("away", 0)
    max_p = max(p_h, p_d, p_a)

    # Signal 1: high-confidence 1X2 outcome
    if max_p >= 0.55:
        side = "home win" if p_h == max_p else ("draw" if p_d == max_p else "away win")
        favored = home if p_h == max_p else (away if p_a == max_p else "draw")
        # Compare to market, if available
        edge = match_pred.get("market_edge")
        market_part = ""
        if isinstance(edge, (int, float)):
            if edge > 0.05:
                market_part = f"; +{edge*100:.0f}% edge over market"
            elif edge < -0.05:
                market_part = f"; market priced tighter ({edge*100:+.0f}%)"
        return (
            f"Model leans {side} with {max_p:.0%} probability"
            f" ({favored} favored){market_part}."
        )

    # Signal 2: cards market deviates strongly from base rate
    if cards_pred:
        c45 = cards_pred.get("over_4_5")
        if isinstance(c45, (int, float)):
            dev = c45 - SA_CARDS_OVER_4_5_BASE
            if abs(dev) >= 0.12:
                direction = "high-card" if dev > 0 else "low-card"
                return (
                    f"Bookings angle: model gives {c45:.0%} for over 4.5 cards"
                    f" ({direction} matchup vs {SA_CARDS_OVER_4_5_BASE:.0%} base);"
                    f" expected {cards_pred.get('expected_cards', '?')} total."
                )

    # Signal 3: corners deviate from base rate
    if corners_pred:
        c105 = corners_pred.get("over_10_5")
        if isinstance(c105, (int, float)):
            dev = c105 - SA_CORNERS_OVER_10_5_BASE
            if abs(dev) >= 0.10:
                direction = "high-corner" if dev > 0 else "low-corner"
                return (
                    f"Corners angle: {c105:.0%} for over 10.5"
                    f" ({direction} matchup); expected"
                    f" {corners_pred.get('expected_corners', '?')} total."
                )

    # Signal 4: standout individual scorer
    if scorers_pred:
        all_top = (scorers_pred.get("home_top_scorers") or []) + (
            scorers_pred.get("away_top_scorers") or [])
        all_top.sort(key=lambda x: x.get("goal_prob", 0), reverse=True)
        if all_top:
            best = all_top[0]
            if best.get("goal_prob", 0) >= SCORER_HIGH_THRESHOLD:
                return (
                    f"Player angle: {best['player']} ({best['position']})"
                    f" projects {best['goal_prob']:.0%} to score —"
                    f" the standout pick across both XIs."
                )

    # Signal 5: tight match
    return (
        f"Tight match: 1X2 split"
        f" {p_h:.0%}/{p_d:.0%}/{p_a:.0%} (H/D/A); model has no clear edge."
    )


def generate_reasoning() -> list[dict]:
    matches_pred = _load_optional(UPCOMING_DIR / "predictions.json", [])
    if not matches_pred:
        log.warning("No predictions.json — cannot generate reasoning")
        return []

    corners_idx = _index(
        _load_optional(UPCOMING_DIR / "corners_predictions.json", []),
        "match", "date",
    )
    cards_idx = _index(
        _load_optional(UPCOMING_DIR / "cards_predictions.json", []),
        "match", "date",
    )
    # scorers_predictions.json may not carry a date (lineup-source lacks it).
    # Index by match-name only — works even when scorer file is from a
    # different matchweek snapshot.
    scorers_raw = _load_optional(UPCOMING_DIR / "scorers_predictions.json", [])
    scorers_idx_by_match = {
        p.get("match"): p
        for p in scorers_raw
        if isinstance(p, dict) and p.get("match")
    }

    out = []
    for p in matches_pred:
        match_name = p.get("match")
        date = p.get("date", "")
        if not match_name:
            continue
        key = (match_name, date)
        sentence = _build_reasoning(
            p,
            corners_idx.get(key),
            cards_idx.get(key),
            scorers_idx_by_match.get(match_name),
        )
        out.append({
            "match": match_name,
            "date": date,
            "reasoning": sentence,
        })
        log.info("  %s — %s", match_name, sentence)

    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    UPCOMING_DIR.mkdir(parents=True, exist_ok=True)
    out = generate_reasoning()
    if not out:
        log.warning("No reasoning generated.")
        return 0
    path = UPCOMING_DIR / "match_reasoning.json"
    path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": out,
    }, indent=2))
    log.info("Wrote %s (%d entries)", path, len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
