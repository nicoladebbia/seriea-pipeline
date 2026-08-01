"""Betfair exchange odds -> comparison_odds.json adapter (data only, NO betting).

Fills the one missing link between the Betfair fetcher (scripts/data/betfair_feed.py)
and the source-agnostic edge-comparison UI (odds_comparison.compare_match, read by
/api/projections and /api/value-bets). Betfair is a value-DISCOVERY source here —
it is deliberately NOT wired into the production betting engine (betting_unified.py),
which relies on Pinnacle sharp/soft divergence that an exchange cannot provide.

MARKET_ODDS (1x2) only — betfair_feed fetches only MATCH_ODDS, and only who-wins
markets carry skill in this project. No O/U / BTTS synthesis.

VERIFICATION STATUS (mirror betfair_feed.py's own tripwire):
  The MECHANICAL mapping (runner->outcome, back-price->price, null handling) is
  unit-tested against betfair_feed.py's fixed output contract. The NAME JOIN
  (Betfair event/runner strings -> our short-form match keys) is BUILT but
  UNVERIFIED until `python3 -m scripts.data.betfair_feed --probe` returns a real
  Betfair response once (needs an app key on disk — Nicola-side). Until the probe
  passes, do not trust this output for real betting decisions.

Everything degrades to a clean skip: no input file, empty markets, unparseable
snapshot -> log one line, write nothing (or an empty book), exit 0.

Run:  python3 -m scripts.betting.betfair_to_comparison
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config.team_names import normalize_team

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
IN_PATH = BASE_DIR / "data" / "betting" / "betfair_odds.json"
OUT_PATH = BASE_DIR / "data" / "upcoming" / "comparison_odds.json"

# Betfair labels the draw runner "The Draw"; normalize_team passes it through
# unchanged, so we match on the normalized value.
_DRAW_LABEL = normalize_team("The Draw")


def _split_event_teams(event_name: str) -> tuple[str, str] | None:
    """'Inter v AC Milan' -> ('Inter', 'AC Milan'). None if not a two-team 'X v Y'.

    Betfair events are 'Home v Away' (single ' v '). We keep the RAW halves here;
    normalization happens where we compare, so the caller controls it.
    """
    if not event_name:
        return None
    # Betfair uses ' v ' as separator; be tolerant of surrounding whitespace.
    for sep in (" v ", " vs ", " V "):
        if sep in event_name:
            left, right = event_name.split(sep, 1)
            left, right = left.strip(), right.strip()
            if left and right:
                return left, right
    return None


def _classify_runners(
    runners: list[dict], home_raw: str, away_raw: str
) -> tuple[dict[str, float], list[str]]:
    """Map Betfair runners -> {home/draw/away: back_price}. Returns (mapped, unclassified).

    Home/away are decided by normalized-name equality against the event halves —
    NOT by runner order (order is not guaranteed). 'unclassified' collects any
    runner we could not place, so the caller can log+count it (never silent).
    """
    home_norm = normalize_team(home_raw)
    away_norm = normalize_team(away_raw)
    mapped: dict[str, float] = {}
    unclassified: list[str] = []

    for r in runners:
        name = (r.get("name") or "").strip()
        back = r.get("back")
        if back is None:
            # No available-to-back price (thin market / suspended) — skip this
            # outcome, but record it so a whole-market gap is visible.
            unclassified.append(f"{name!r} (no back price)")
            continue
        norm = normalize_team(name)
        if norm == _DRAW_LABEL or name.lower() in ("the draw", "draw"):
            mapped["draw"] = back
        elif norm == home_norm:
            mapped["home"] = back
        elif norm == away_norm:
            mapped["away"] = back
        else:
            unclassified.append(f"{name!r} (norm={norm!r}; event={home_norm!r} v {away_norm!r})")
    return mapped, unclassified


def build_comparison_odds(
    in_path: Path = IN_PATH, out_path: Path = OUT_PATH, write: bool = True
) -> dict:
    """Read betfair_odds.json, emit comparison_odds.json. Returns the built dict.

    Uses the LATEST snapshot per market (snapshots[-1]) — the toward-close price is
    the sharpest and the one you'd bet into. Never raises: any structural problem
    degrades to a clean skip with a logged reason.
    """
    try:
        store = json.loads(in_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.info("betfair-adapter: no readable %s (%s) — nothing to convert", in_path.name, e)
        return {"book": "Betfair", "odds": {}}

    markets = store.get("markets", {}) if isinstance(store, dict) else {}
    if not markets:
        log.info("betfair-adapter: %s has no markets — clean skip", in_path.name)
        return {"book": "Betfair", "odds": {}}

    odds: dict[str, dict] = {}
    n_markets = 0
    n_written = 0
    n_no_event = 0
    n_incomplete = 0
    unclassified_total = 0

    for mid, m in markets.items():
        n_markets += 1
        snaps = m.get("snapshots") or []
        if not snaps:
            continue
        latest = snaps[-1]
        teams = _split_event_teams(m.get("event") or "")
        if not teams:
            n_no_event += 1
            log.warning(
                "betfair-adapter: market %s event=%r not parseable as 'X v Y' — skipped",
                mid, m.get("event"),
            )
            continue
        home_raw, away_raw = teams
        mapped, unclassified = _classify_runners(latest.get("runners") or [], home_raw, away_raw)
        if unclassified:
            unclassified_total += len(unclassified)
            log.warning(
                "betfair-adapter: market %s (%s) unclassified runners: %s",
                mid, m.get("event"), "; ".join(unclassified),
            )
        # Require the full 1x2 triple — a partial market can't be de-vigged cleanly
        # and a silent partial is exactly the failure mode we refuse to ship.
        if not all(k in mapped for k in ("home", "draw", "away")):
            n_incomplete += 1
            log.warning(
                "betfair-adapter: market %s (%s) incomplete 1x2 %s — skipped (not written)",
                mid, m.get("event"), sorted(mapped),
            )
            continue

        match_key = f"{normalize_team(home_raw)} vs {normalize_team(away_raw)}"
        if match_key in odds:
            log.warning("betfair-adapter: duplicate match key %r (market %s) — keeping first",
                        match_key, mid)
            continue
        odds[match_key] = {"1x2": mapped}
        n_written += 1

    result = {"book": "Betfair", "odds": odds}

    # Loud accounting — a caller (or a human reading logs) sees exactly what happened,
    # never a silent 'wrote 0'. Mirrors the sofascore_fetch health-line fix.
    log.info(
        "betfair-adapter: %d markets in -> %d written | skipped: %d bad-event, "
        "%d incomplete-1x2 | %d unclassified runners total",
        n_markets, n_written, n_no_event, n_incomplete, unclassified_total,
    )
    if n_markets and n_written == 0:
        log.warning(
            "betfair-adapter: SAW %d markets but WROTE 0 — likely a name-join gap "
            "(Betfair strings not mapped by config.team_names). This is the "
            "unverified-until-probe risk; check the unclassified warnings above.",
            n_markets,
        )

    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=1))
        tmp.replace(out_path)
        log.info("betfair-adapter: wrote %s (%d matches)", out_path.name, n_written)

    return result


def main() -> None:
    build_comparison_odds()


if __name__ == "__main__":
    main()
