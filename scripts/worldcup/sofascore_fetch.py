"""Sofascore fetchers for WC2026 via the www.sofascore.com/api/v1 proxy.

Discovery (2026-06-10): api.sofascore.com is TCP-unreachable from this
network, but www.sofascore.com/api/v1 serves the identical API with no
Cloudflare challenge — including upcoming-event ODDS and lineups. This module
is the production home for that access pattern (the original bulk scrape was
a one-shot per repo policy; this stays because the tournament needs daily
refreshes: odds, confirmed lineups, new played matches).

CLI:
  python3 -m scripts.worldcup.sofascore_fetch --odds
      Resolve all known fixtures to Sofascore event ids and write
      data/worldcup/market_odds.json (1X2 decimal odds + de-vigged implied
      probabilities). Display-only market comparison — NOT blended into the
      model (no validated blend weight exists yet).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from scripts.worldcup.engine import atomic_write_json, canon_team, read_json_safe


def sofa_canon(name: str) -> str:
    """canon_team with Sofascore's spelling quirks pre-normalized:
    'Bosnia & Herzegovina' (ampersand) and curly apostrophes (’)."""
    return canon_team(name.replace("’", "'").replace(" & ", " and "))

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "worldcup"
FIXTURES_JSON = DATA_DIR / "fixtures.json"
MARKET_ODDS_JSON = DATA_DIR / "market_odds.json"
SOFA_TEAM_DIR = DATA_DIR / "sofascore_intl"

BASE = "https://www.sofascore.com/api/v1"
DELAY_RANGE = (2.0, 4.5)


def _session() -> Any:
    from curl_cffi import requests as creq

    return creq.Session(impersonate="chrome")


_CONSEC_FAILURES = 0
_BREAKER_LIMIT = 8


class SourceDownError(RuntimeError):
    """Raised when the host has failed _BREAKER_LIMIT consecutive requests —
    callers save partials and stop instead of crawling through a dead network."""


def get_json(session: Any, path: str) -> dict[str, Any] | None:
    """Polite GET with one retry; None on failure (caller decides severity).
    Trips SourceDownError after _BREAKER_LIMIT consecutive failures."""
    global _CONSEC_FAILURES
    if _CONSEC_FAILURES >= _BREAKER_LIMIT:
        raise SourceDownError(
            f"{_BREAKER_LIMIT} consecutive Sofascore failures — breaker open"
        )
    for attempt in (1, 2):
        try:
            r = session.get(f"{BASE}/{path}", timeout=20)
            if r.status_code == 200:
                data = dict(r.json())  # parse BEFORE resetting: poisoned-200
                _CONSEC_FAILURES = 0   # challenge pages must count as failures
                return data
            if r.status_code == 404:
                # resource not published yet (pre-kickoff lineups) — the host
                # is healthy; never count toward the breaker (sibling scraper
                # learned this lesson: sofascore_events.py)
                _CONSEC_FAILURES = 0
                return None
            if r.status_code in (403, 429):
                time.sleep(10.0 * attempt)
                continue
            _CONSEC_FAILURES += 1
            return None
        except Exception:  # noqa: BLE001 — transport errors vary by backend; retry-once then None
            time.sleep(3.0 * attempt)
    _CONSEC_FAILURES += 1
    return None


def _sleep() -> None:
    # politeness jitter, not crypto
    time.sleep(random.uniform(*DELAY_RANGE))  # noqa: S311


def team_ids_from_scrape() -> dict[str, int]:
    """Team display name -> Sofascore team id, from the per-team scrape JSONs."""
    out: dict[str, int] = {}
    fixtures = json.loads(FIXTURES_JSON.read_text())
    display_names = {
        str(t)
        for f in fixtures
        if f["stage"] == "group"
        for t in (f["home"], f["away"])
    }
    by_ascii = {p.stem: p for p in SOFA_TEAM_DIR.glob("*.json")}
    for name in display_names:
        ascii_key = (
            name.replace("ô", "o").replace("ç", "c").replace("ü", "u")
            .replace("'", "_").replace(" ", "_").replace("é", "e")
        )
        path = by_ascii.get(ascii_key) or by_ascii.get(name.replace(" ", "_"))
        if path is None:
            continue
        blob = json.loads(path.read_text())
        tid = blob.get("team_id") or blob.get("meta", {}).get("team_id")
        if tid:
            out[name] = int(tid)
    return out


def fractional_to_decimal(value: str) -> float:
    """Sofascore fractionalValue ('21/50') -> decimal odds (1.42)."""
    return float(1 + Fraction(value))


def parse_1x2(odds_payload: dict[str, Any]) -> dict[str, float] | None:
    """Extract the full-time 1X2 market as decimal odds {home, draw, away}."""
    for market in odds_payload.get("markets", []):
        if market.get("marketName") == "Full time" and not market.get("isLive"):
            decs: dict[str, float] = {}
            key_map = {"1": "home", "X": "draw", "2": "away"}
            for choice in market.get("choices", []):
                side = key_map.get(str(choice.get("name", "")))
                if side and choice.get("fractionalValue"):
                    decs[side] = round(
                        fractional_to_decimal(choice["fractionalValue"]), 3
                    )
            if len(decs) == 3:
                return decs
    return None


def devig(decimal_odds: dict[str, float]) -> dict[str, float]:
    """Implied probabilities, overround removed proportionally."""
    raw = {k: 1.0 / v for k, v in decimal_odds.items()}
    total = sum(raw.values())
    return {k: round(v / total, 4) for k, v in raw.items()}


def fetch_market_odds() -> dict[str, Any]:
    """Resolve fixtures -> Sofascore events -> 1X2 odds. Incremental-merge:
    keeps previously fetched matches when a refresh run misses them."""
    fixtures = json.loads(FIXTURES_JSON.read_text())
    team_ids = team_ids_from_scrape()
    session = _session()

    # fixture lookup: (canon_home, canon_away) -> (match_number, date).
    # Group-stage pairings are unique, so the date is a sanity check (±1 day
    # tolerated — late local kickoffs cross UTC midnight between sources),
    # not part of the join key.
    fix_key: dict[tuple[str, str], tuple[int, str]] = {}
    for f in fixtures:
        fix_key[(canon_team(str(f["home"])), canon_team(str(f["away"])))] = (
            int(f["match_number"]),
            str(f["date_utc"])[:10],
        )

    events = _collect_wc_events(session, team_ids)

    out: dict[str, Any] = dict(read_json_safe(MARKET_ODDS_JSON, {}))  # type: ignore[arg-type]
    matched = 0
    breaker_tripped = False
    for eid, e in sorted(events.items()):
        date = datetime.fromtimestamp(int(e["startTimestamp"]), tz=UTC)
        key = (
            sofa_canon(str(e["homeTeam"]["name"])),
            sofa_canon(str(e["awayTeam"]["name"])),
        )
        hit = fix_key.get(key)
        if hit is None:
            continue
        match_number, fix_date = hit
        gap_days = abs(
            (date.date() - datetime.strptime(fix_date, "%Y-%m-%d").date()).days
        )
        if gap_days > 1:
            continue  # same pairing, wrong occasion — not a group fixture
        try:
            payload = get_json(session, f"event/{eid}/odds/1/all")
        except SourceDownError as exc:
            print(f"breaker: {exc} — saving {matched} odds fetched so far")
            breaker_tripped = True
            break
        _sleep()
        if payload is None:
            continue
        decs = parse_1x2(payload)
        if decs is None:
            continue
        out[str(match_number)] = {
            "event_id": eid,
            "odds": decs,
            "implied": devig(decs),
            "fetched_at": datetime.now(UTC).isoformat(),
        }
        matched += 1

    atomic_write_json(MARKET_ODDS_JSON, out)
    print(
        f"market odds: {matched} fetched this run, {len(out)} total "
        f"-> {MARKET_ODDS_JSON}"
    )
    if breaker_tripped and matched == 0:
        raise SystemExit(1)  # refresh.py keys last_odds_fetch off the rc
    return out


CONFIRMED_LINEUPS_JSON = DATA_DIR / "confirmed_lineups.json"

# Sofascore missing-player reason codes (same map the Serie A
# lineup_predictor uses); type "doubtful" arrives alongside "missing".
MISSING_REASON_MAP = {1: "Injury", 2: "Suspension", 3: "National team",
                      4: "Personal", 5: "Other", 13: "Suspension"}

SOFA_PARQUET = DATA_DIR / "sofascore_intl_player_stats.parquet"

# Historical closing odds for the backtest tournaments (blend validation).
HISTORICAL_ODDS_JSON = DATA_DIR / "historical_odds.json"


def fetch_played_stats() -> int:
    """Append per-player stats for newly PLAYED matches (WC + any other
    internationals) to the scrape parquet. This is what feeds (a) the
    mid-tournament re-gating of the shots-floor market (3+ real WC matches
    per team) and (b) starter-dampening freshness. Idempotent: existing
    (team, match_id) pairs are skipped. Returns number of rows appended."""
    import pandas as pd

    df = pd.read_parquet(SOFA_PARQUET)
    seen = set(zip(df["team"], df["match_id"], strict=True))
    team_ids = team_ids_from_scrape()
    session = _session()

    new_rows: list[dict[str, Any]] = []
    now_iso = datetime.now(UTC).isoformat()
    breaker_open = False
    for team, tid in sorted(team_ids.items()):
        if breaker_open:
            break
        try:
            payload = get_json(session, f"team/{tid}/events/last/0")
        except SourceDownError as exc:
            print(f"breaker: {exc} — appending {len(new_rows)} rows collected so far")
            break
        _sleep()
        if payload is None:
            continue
        for e in payload.get("events", []):
            eid = int(e["id"])
            if (team, eid) in seen:
                continue
            if str(e.get("status", {}).get("type")) != "finished":
                continue
            is_home = int(e["homeTeam"].get("id", -1)) == tid
            try:
                lineups = get_json(session, f"event/{eid}/lineups")
            except SourceDownError:
                breaker_open = True
                print(f"breaker open — appending {len(new_rows)} rows collected so far")
                break
            _sleep()
            if lineups is None:
                continue
            side = "home" if is_home else "away"
            date = datetime.fromtimestamp(int(e["startTimestamp"]), tz=UTC)
            our_score = e.get("homeScore" if is_home else "awayScore", {}).get("current")
            opp_score = e.get("awayScore" if is_home else "homeScore", {}).get("current")
            for p in lineups.get(side, {}).get("players", []):
                stats = p.get("statistics") or {}
                minutes = stats.get("minutesPlayed", 0) or 0
                new_rows.append(
                    {
                        "team": team,
                        "team_id": tid,
                        "match_id": eid,
                        "date": date.isoformat(),  # '+00:00' form, matches existing rows
                        "tournament": str(e.get("tournament", {}).get("name", "")),
                        "opponent": str(
                            e["awayTeam" if is_home else "homeTeam"]["name"]
                        ),
                        "home_or_away": side,
                        "our_score": our_score,
                        "opp_score": opp_score,
                        "player_name": str(p["player"]["name"]),
                        "norm_name": normalize_simple(str(p["player"]["name"])),
                        "norm_name_sorted": norm_sorted_simple(
                            str(p["player"]["name"])
                        ),
                        "player_id": int(p["player"].get("id", 0)),
                        "position": str(p["player"].get("position", "")),
                        "started": (not p.get("substitute", True)) and minutes > 0,
                        "minutes": minutes,
                        "shots": (stats.get("totalShots", 0) or 0)
                        if minutes > 0 else None,
                        "shots_on_target": (
                            stats.get("onTargetScoringAttempt", 0) or 0
                        ) if minutes > 0 else None,
                        "goals": (stats.get("goals", 0) or 0) if minutes > 0 else None,
                        "rating": stats.get("rating"),
                        "scraped_at": now_iso,
                        "source_url": f"{BASE}/event/{eid}/lineups",
                    }
                )
            seen.add((team, eid))

    if new_rows:
        import pandas as pd

        updated = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_parquet(SOFA_PARQUET, index=False)
    print(f"played-stats: {len(new_rows)} player-match rows appended")
    return len(new_rows)


def normalize_simple(name: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", name.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def norm_sorted_simple(name: str) -> str:
    return " ".join(sorted(normalize_simple(name).split()))


def _collect_wc_events(
    session: Any, team_ids: dict[str, int]
) -> dict[int, dict[str, Any]]:
    """Upcoming WC events across all teams (each fixture found twice; deduped).
    Returns whatever was collected before a breaker trip."""
    events: dict[int, dict[str, Any]] = {}
    try:
        for _name, tid in sorted(team_ids.items()):
            payload = get_json(session, f"team/{tid}/events/next/0")
            _sleep()
            if payload is None:
                continue
            for e in payload.get("events", []):
                if "World Cup" in str(e.get("tournament", {}).get("name", "")):
                    events[int(e["id"])] = e
    except SourceDownError as exc:
        print(f"breaker: {exc} — proceeding with {len(events)} events collected")
    return events


def fetch_confirmed_lineups(horizon_hours: float = 48.0) -> dict[str, Any]:
    """Confirmed starting XIs for fixtures kicking off within the horizon.
    Sofascore publishes lineups ~1h pre-kickoff (confirmed=true); run this on
    match days and regenerate the player predictions afterwards."""
    fixtures = json.loads(FIXTURES_JSON.read_text())
    team_ids = team_ids_from_scrape()
    session = _session()
    fix_key: dict[tuple[str, str], int] = {
        (canon_team(str(f["home"])), canon_team(str(f["away"]))): int(
            f["match_number"]
        )
        for f in fixtures
    }

    out: dict[str, Any] = dict(read_json_safe(CONFIRMED_LINEUPS_JSON, {}))  # type: ignore[arg-type]

    now = datetime.now(UTC).timestamp()
    confirmed_count = 0
    for eid, e in sorted(_collect_wc_events(session, team_ids).items()):
        ts = int(e["startTimestamp"])
        if not (now - 3 * 3600 <= ts <= now + horizon_hours * 3600):
            continue
        key = (
            sofa_canon(str(e["homeTeam"]["name"])),
            sofa_canon(str(e["awayTeam"]["name"])),
        )
        match_number = fix_key.get(key)
        if match_number is None:
            continue
        try:
            payload = get_json(session, f"event/{eid}/lineups")
        except SourceDownError as exc:
            print(f"breaker: {exc} — saving partial lineups")
            break
        _sleep()
        if payload is None:
            continue
        entry: dict[str, Any] = {
            "event_id": eid,
            "confirmed": bool(payload.get("confirmed", False)),
            "fetched_at": datetime.now(UTC).isoformat(),
        }
        for side in ("home", "away"):
            side_blob = payload.get(side, {})
            entry[side] = {
                "starters": [
                    str(p["player"]["name"])
                    for p in side_blob.get("players", [])
                    if not p.get("substitute", True)
                ],
                "bench": [
                    str(p["player"]["name"])
                    for p in side_blob.get("players", [])
                    if p.get("substitute", False)
                ],
                "missing": [
                    {
                        "name": str(m["player"]["name"]),
                        "position": str(m["player"].get("position", "")),
                        "type": str(m.get("type", "missing")),
                        "reason_code": int(m.get("reason", 0) or 0),
                        "reason": MISSING_REASON_MAP.get(
                            int(m.get("reason", 0) or 0), "Other"
                        ),
                    }
                    for m in side_blob.get("missingPlayers", [])
                    if m.get("player")
                ],
            }
        out[str(match_number)] = entry
        if entry["confirmed"]:
            confirmed_count += 1

    atomic_write_json(CONFIRMED_LINEUPS_JSON, out)
    print(
        f"lineups: {len(out)} fixtures tracked, {confirmed_count} confirmed "
        f"-> {CONFIRMED_LINEUPS_JSON}"
    )
    return out

# (label, unique-tournament id, season year string, group window [start, end])
HISTORICAL_TOURNAMENTS = [
    ("World Cup 2018", 16, "2018", "2018-06-14", "2018-06-28"),
    ("Euro 2020", 1, "2020", "2021-06-11", "2021-06-23"),
    ("World Cup 2022", 16, "2022", "2022-11-20", "2022-12-02"),
    ("Euro 2024", 1, "2024", "2024-06-14", "2024-06-26"),
    ("Copa América 2024", 133, "2024", "2024-06-20", "2024-07-02"),
]


def fetch_historical_odds() -> dict[str, Any]:
    """Closing 1X2 odds for the group-stage windows of the five backtest
    tournaments. Output rows keyed for the backtest join:
    (canon_home, canon_away, date). Incremental: completed tournaments skip."""
    session = _session()
    out: dict[str, Any] = {}
    if HISTORICAL_ODDS_JSON.exists():
        out = json.loads(HISTORICAL_ODDS_JSON.read_text())

    for label, ut_id, year, start, end in HISTORICAL_TOURNAMENTS:
        if label in out and out[label].get("complete"):
            continue
        try:
            seasons = get_json(session, f"unique-tournament/{ut_id}/seasons")
        except SourceDownError as exc:
            print(f"breaker: {exc} — stopping historical fetch")
            break
        _sleep()
        if seasons is None:
            continue
        season_id = None
        for x in seasons.get("seasons", []):
            if str(x.get("year")) in (year, str(int(year) + 1)):
                season_id = int(x["id"])
                break
        if season_id is None:
            out[label] = {"complete": False, "error": "season not found", "rows": []}
            continue

        # Page through the season's events; collect those in the group window.
        events: list[dict[str, Any]] = []
        for page in range(6):
            try:
                payload = get_json(
                    session,
                    f"unique-tournament/{ut_id}/season/{season_id}/events/last/{page}",
                )
            except SourceDownError as exc:
                print(f"breaker: {exc} — stopping pagination for {label}")
                break
            _sleep()
            if payload is None or not payload.get("events"):
                break
            events.extend(payload["events"])
            if not payload.get("hasNextPage"):
                break

        rows: list[dict[str, Any]] = []
        misses = 0
        for e in events:
            date = datetime.fromtimestamp(int(e["startTimestamp"]), tz=UTC)
            day = date.strftime("%Y-%m-%d")
            if not (start <= day <= end):
                continue
            try:
                payload = get_json(session, f"event/{int(e['id'])}/odds/1/all")
            except SourceDownError as exc:
                print(f"breaker: {exc} — saving partial rows for {label}")
                break
            _sleep()
            decs = parse_1x2(payload) if payload else None
            if decs is None:
                misses += 1
                continue
            rows.append(
                {
                    "home": sofa_canon(str(e["homeTeam"]["name"])),
                    "away": sofa_canon(str(e["awayTeam"]["name"])),
                    "date": day,
                    "odds": decs,
                    "implied": devig(decs),
                }
            )
        out[label] = {
            "complete": True,
            "fetched_at": datetime.now(UTC).isoformat(),
            "n_rows": len(rows),
            "n_missing_odds": misses,
            "rows": rows,
        }
        HISTORICAL_ODDS_JSON.write_text(json.dumps(out, indent=2))
        print(f"{label}: {len(rows)} matches with odds, {misses} without")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odds", action="store_true")
    parser.add_argument("--historical-odds", action="store_true")
    parser.add_argument("--lineups", action="store_true")
    parser.add_argument("--refresh-stats", action="store_true")
    args = parser.parse_args()
    if args.odds:
        fetch_market_odds()
    elif args.historical_odds:
        fetch_historical_odds()
    elif args.lineups:
        fetch_confirmed_lineups()
    elif args.refresh_stats:
        fetch_played_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
