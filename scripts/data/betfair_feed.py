"""Betfair Exchange odds feed (Italy wallet) — Serie A match odds + CLV benchmark.

WHY: the exchange close is the sharpest public line — a better CLV benchmark
than any bookmaker close — and exchange prices (~2% commission) beat bookie
margin when we eventually place there. Account registered 2026-06-03;
this module is the data layer only. NO BETTING here.

STATUS — built 2026-06-11 against the documented API-NG contract but NEVER
run against the live API (no app key on disk yet). The contract is NOT
verified until `--probe` succeeds once — that's step 6 of AUGUST_RUNBOOK.md.
Do not wire consumers to its output before the probe passes.

Auth model (manual, Nicola-side):
  config/api_keys.json -> {"betfair": {"app_key": "...", "session_token": "..."}}
  Session tokens expire (~20min idle / 24h max on the .it jurisdiction); the
  keep-alive endpoint extends them. Interactive login (identitysso.betfair.it)
  is done by a human; this module only consumes the resulting token and says
  loudly when it's missing or expired.

Everything degrades to a clean skip — no credentials, no network, API change:
log one line, write nothing, exit 0. The pipeline must never block on this.

Run:  python3 -m scripts.data.betfair_feed --probe   # one auth+contract check
      python3 -m scripts.data.betfair_feed --fetch   # Serie A MATCH_ODDS snapshot
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KEYS_PATH = BASE_DIR / "config" / "api_keys.json"
OUT_PATH = BASE_DIR / "data" / "betting" / "betfair_odds.json"

API_URL = "https://api.betfair.com/exchange/betting/rest/v1.0"
KEEPALIVE_URL = "https://identitysso.betfair.it/api/keepAlive"

SERIE_A_COMPETITION_ID = "81"   # Betfair competition id for Serie A — verify in --probe
EVENT_TYPE_SOCCER = "1"
TIMEOUT = 15


def _load_creds() -> dict | None:
    """Betfair creds from api_keys.json, or None (clean skip) when absent."""
    try:
        keys = json.loads(KEYS_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        log.info("betfair: no readable %s — skipping", KEYS_PATH.name)
        return None
    bf = keys.get("betfair") or {}
    if not bf.get("app_key") or not bf.get("session_token"):
        log.info("betfair: app_key/session_token not configured — skipping "
                 "(see AUGUST_RUNBOOK.md step 6)")
        return None
    return bf


def _headers(creds: dict) -> dict:
    return {
        "X-Application": creds["app_key"],
        "X-Authentication": creds["session_token"],
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _post(creds: dict, method: str, body: dict) -> Any | None:
    """One REST call; None on any failure (callers skip, never raise)."""
    try:
        r = requests.post(f"{API_URL}/{method}/", headers=_headers(creds),
                          json=body, timeout=TIMEOUT)
        if r.status_code == 400 and "INVALID_SESSION" in r.text:
            log.warning("betfair: session token expired — re-login needed")
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning("betfair %s failed: %s", method, e)
        return None


def probe() -> bool:
    """One cheap authenticated call. THE contract verification for August."""
    creds = _load_creds()
    if not creds:
        return False
    res = _post(creds, "listEventTypes", {"filter": {}})
    if res is None:
        return False
    soccer = [e for e in res if e.get("eventType", {}).get("id") == EVENT_TYPE_SOCCER]
    log.info("betfair probe OK — %d event types, soccer present: %s",
             len(res), bool(soccer))
    comps = _post(creds, "listCompetitions",
                  {"filter": {"eventTypeIds": [EVENT_TYPE_SOCCER],
                              "marketCountries": ["IT"]}})
    if comps:
        for c in comps:
            comp = c.get("competition", {})
            if "serie a" in comp.get("name", "").lower():
                log.info("betfair: Serie A competition id = %s (module assumes %s)",
                         comp.get("id"), SERIE_A_COMPETITION_ID)
    return True


def fetch_match_odds() -> int:
    """Snapshot best back/lay for every Serie A MATCH_ODDS market.

    Writes data/betting/betfair_odds.json (merged by market id, so repeated
    runs build toward-the-close history per match). Returns markets written.
    """
    creds = _load_creds()
    if not creds:
        return 0
    cat = _post(creds, "listMarketCatalogue", {
        "filter": {"eventTypeIds": [EVENT_TYPE_SOCCER],
                   "competitionIds": [SERIE_A_COMPETITION_ID],
                   "marketTypeCodes": ["MATCH_ODDS"]},
        "maxResults": "100",
        "marketProjection": ["EVENT", "RUNNER_DESCRIPTION", "MARKET_START_TIME"],
    })
    if not cat:
        log.info("betfair: no MATCH_ODDS markets returned (off-season or auth issue)")
        return 0

    market_ids = [m["marketId"] for m in cat]
    books = _post(creds, "listMarketBook", {
        "marketIds": market_ids[:40],   # listMarketBook caps weight per call
        "priceProjection": {"priceData": ["EX_BEST_OFFERS"]},
    }) or []
    book_by_id = {b["marketId"]: b for b in books}

    now = datetime.now(UTC).isoformat()
    try:
        store = json.loads(OUT_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        store = {"markets": {}, "note": "exchange snapshots; key = betfair marketId"}

    n = 0
    for m in cat:
        mid = m["marketId"]
        book = book_by_id.get(mid)
        if not book:
            continue
        runners_meta = {r["selectionId"]: r.get("runnerName", "?")
                        for r in m.get("runners", [])}
        runners = []
        for r in book.get("runners", []):
            ex = r.get("ex", {})
            back = (ex.get("availableToBack") or [{}])[0].get("price")
            lay = (ex.get("availableToLay") or [{}])[0].get("price")
            runners.append({
                "name": runners_meta.get(r.get("selectionId"), "?"),
                "back": back, "lay": lay,
                "implied_mid": round(2 / (back + lay), 4) if back and lay else None,
            })
        entry = store["markets"].setdefault(mid, {
            "event": m.get("event", {}).get("name"),
            "kickoff": m.get("marketStartTime"),
            "snapshots": [],
        })
        entry["snapshots"].append({"at": now, "runners": runners,
                                   "total_matched": book.get("totalMatched")})
        # keep first 20 + always the latest 20 — open + close matter, middle less
        if len(entry["snapshots"]) > 40:
            entry["snapshots"] = entry["snapshots"][:20] + entry["snapshots"][-20:]
        n += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, indent=1))
    tmp.replace(OUT_PATH)
    log.info("betfair: wrote %d markets -> %s", n, OUT_PATH.name)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Betfair exchange odds feed (data only)")
    ap.add_argument("--probe", action="store_true", help="auth + contract check")
    ap.add_argument("--fetch", action="store_true", help="snapshot Serie A match odds")
    args = ap.parse_args()
    if args.probe:
        sys.exit(0 if probe() else 1)
    if args.fetch:
        fetch_match_odds()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
