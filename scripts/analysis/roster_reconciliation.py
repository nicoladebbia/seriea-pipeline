"""Reconcile our scraped 2026-27 transfers against Transfermarkt's own rosters.

For each of the 20 Serie A 2026-27 clubs this fetches TWO squad snapshots from
Transfermarkt — ``saison_id/2025`` (the 25/26 rosa) and ``saison_id/2026`` (the
26/27 rosa) — and set-diffs them by player id:

    reality_in  = in 26/27 rosa, NOT in 25/26 rosa   → TM's real arrivals
    reality_out = in 25/26 rosa, NOT in 26/27 rosa   → TM's real departures

That set-diff is the ground truth (as-of-today) of who actually moved. It is
then compared to our ``transfers_2026_2027.parquet`` IN/OUT list to flag:

    missing_signing    reality says arrived, our scrape has no IN row
    missing_departure  reality says left,   our scrape has no OUT row
    phantom_in         our scrape has an IN row reality's rosa-diff doesn't show
    phantom_departure  our scrape has an OUT row reality's rosa-diff doesn't show

IMPORTANT — what this verifies and what it does NOT:
  * This checks that OUR SCRAPE faithfully mirrors TM's own rosa + transfer
    pages. TM is the realistic-rosa source this project already uses, so it is
    the right check — but it verifies "we match TM", not "TM matches reality".
  * The window is OPEN (mid-summer). Both TM pages are moving targets: a deal on
    the transfers page but not yet reflected in the rosa (or vice-versa) reads as
    a "discrepancy" that is really TM mid-update. Every flag is as-of-today;
    small mismatches are expected and are not necessarily bugs.

Set-diff (not a single-rosa "who has no transfer record") is deliberate: a rosa
is ~25-38 players but only a handful moved this window — the retained core and
youth have no transfer record, so a single-rosa check would flag the whole squad
as noise. The 25-vs-26 diff isolates exactly the players who actually moved.

Paced (2s/request) and ban-aware: if a fetch returns zero players, TM likely
IP-banned us — stop rather than burn the live cron's IP.

Run: python3 -m scripts.analysis.roster_reconciliation
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from features.transfer_impact_analysis import _normalize_name
from scraper.transfermarkt import (
    SERIE_A_TEAMS_TM,
    TM_BASE,
    TM_DIR,
    TM_HEADERS,
    current_league_teams,
)

# saison_id the TWO rosters key to (verified via page <title>):
#   2025 -> "Detailed squad 25/26"   (last season, the OUT baseline)
#   2026 -> "Detailed squad 26/27"   (this season, the current rosa)
PREV_SEASON_ID = 2025
CURR_SEASON_ID = 2026
TRANSFERS_FILE = TM_DIR / "transfers_2026_2027.parquet"
OUT_JSON = Path("data/analysis/roster_reconciliation_2026_2027.json")
REQUEST_GAP_S = 2.0

# player-profile anchor: /profil/spieler/{id}"> Name </a>  (verified specimen)
_PLAYER_RE = re.compile(r'/profil/spieler/(\d+)\">\s*([^<\n][^<]*?)\s*</a>')


def _log(msg: str) -> None:
    print(f"[{datetime.now(UTC).isoformat(timespec='seconds')}] {msg}", flush=True)


def fetch_rosa(slug: str, verein_id: int, season_id: int) -> dict[str, str]:
    """Return {player_id: name} for a club's squad at a given saison_id.

    Raises RuntimeError on an empty parse (likely IP-ban / schema break) so the
    caller can stop rather than silently treat a ban as an all-players-left diff.
    """
    url = f"{TM_BASE}/{slug}/kader/verein/{verein_id}/saison_id/{season_id}/plus/1"
    resp = requests.get(url, headers=TM_HEADERS, timeout=20)
    resp.raise_for_status()
    seen: dict[str, str] = {}
    for pid, name in _PLAYER_RE.findall(resp.text):
        name = name.strip()
        if pid not in seen and name:
            seen[pid] = name
    if not seen:
        raise RuntimeError(f"0 players parsed for {slug} saison_id={season_id}")
    return seen


def _our_moves(transfers: pd.DataFrame, team: str) -> tuple[set[str], set[str]]:
    """(normalized IN names, normalized OUT names) from our scrape for a team."""
    tf = transfers[transfers["team"] == team]
    ins = {_normalize_name(n) for n in tf.loc[tf["transfer_type"] == "in", "player_name"]}
    outs = {_normalize_name(n) for n in tf.loc[tf["transfer_type"] == "out", "player_name"]}
    return ins, outs


def reconcile_club(team: str, slug: str, verein_id: int, transfers: pd.DataFrame) -> dict:
    prev = fetch_rosa(slug, verein_id, PREV_SEASON_ID)
    time.sleep(REQUEST_GAP_S)
    curr = fetch_rosa(slug, verein_id, CURR_SEASON_ID)

    prev_ids, curr_ids = set(prev), set(curr)
    # reality's NET roster change, by player id (same source both sides → id-stable)
    reality_in = {curr[i] for i in (curr_ids - prev_ids)}
    reality_out = {prev[i] for i in (prev_ids - curr_ids)}
    ri_norm = {_normalize_name(n): n for n in reality_in}
    ro_norm = {_normalize_name(n): n for n in reality_out}
    curr_norm = {_normalize_name(n) for n in curr.values()}

    our_in, our_out = _our_moves(transfers, team)

    # --- HIGH confidence: the flags the user asked for --------------------------
    # (1) missing signing: TM's roster gained a player our IN list doesn't have.
    missing_signing = sorted(ri_norm[k] for k in ri_norm.keys() - our_in)
    # (2) phantom exit: we marked a player OUT but he is STILL in the 26/27 kader
    #     (bad OUT row, or loan-out/return churn double-listing). This is the
    #     user's "player-still-on-roster-but-we-said-out" flag, anchored on the
    #     CURRENT rosa (not the net set-diff).
    phantom_exit = sorted(k for k in our_out & curr_norm)

    # --- MEDIUM/LOW confidence: informational, not headline flags ---------------
    # our IN not in the current squad — often a signed-then-loaned-out player TM
    # legitimately omits from the kader. Surface, don't alarm.
    in_not_in_squad = sorted(our_in - curr_norm)
    # rosa lost a player our OUT list doesn't have — mostly Primavera promotion /
    # youth-list churn, not a missed first-team transfer.
    missing_departure = sorted(ro_norm[k] for k in ro_norm.keys() - our_out)

    return {
        "team": team,
        "rosa_prev_25_26": len(prev),
        "rosa_curr_26_27": len(curr),
        "reality_arrivals": len(reality_in),
        "reality_departures": len(reality_out),
        "our_in": len(our_in),
        "our_out": len(our_out),
        # high-confidence flags (what the user asked for)
        "missing_signing": missing_signing,
        "phantom_exit": phantom_exit,
        # informational
        "in_not_in_squad": in_not_in_squad,
        "missing_departure_youth_churn": missing_departure,
        "flags": len(missing_signing) + len(phantom_exit),
    }


def main() -> int:
    if not TRANSFERS_FILE.exists():
        _log(f"FATAL: {TRANSFERS_FILE} missing — scrape 2026-27 transfers first.")
        return 1
    transfers = pd.read_parquet(TRANSFERS_FILE)

    # the 20 clubs actually in Serie A 2026-27 (from the live TM competition page,
    # falling back to whatever teams our transfers file covers)
    live = current_league_teams("2026-2027", "serie_a")
    clubs = sorted(live) if live else sorted(transfers["team"].unique())
    _log(f"=== roster reconciliation start: {len(clubs)} clubs ===")

    results, banned = [], False
    for team in clubs:
        entry = SERIE_A_TEAMS_TM.get(team)
        if entry is None:
            _log(f"{team}: no verein id in SERIE_A_TEAMS_TM — skipped")
            continue
        slug, vid = entry
        try:
            r = reconcile_club(team, slug, vid, transfers)
        except RuntimeError as e:  # empty parse → ban/schema break, stop the sweep
            _log(f"{team}: {e} — likely IP-ban, stopping sweep.")
            banned = True
            break
        except Exception as e:  # noqa: BLE001 — one bad club must not lose the rest
            _log(f"{team}: FAILED {type(e).__name__}: {e}")
            results.append({"team": team, "error": f"{type(e).__name__}: {e}"})
            time.sleep(REQUEST_GAP_S)
            continue
        _log(
            f"{team}: rosa {r['rosa_prev_25_26']}→{r['rosa_curr_26_27']} | "
            f"reality in/out {r['reality_arrivals']}/{r['reality_departures']} | "
            f"ours {r['our_in']}/{r['our_out']} | "
            f"missing_sig {len(r['missing_signing'])} phantom_exit {len(r['phantom_exit'])}"
        )
        results.append(r)
        time.sleep(REQUEST_GAP_S)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "as_of_note": "window open (mid-summer); flags are as-of-today, TM-vs-TM",
        "prev_season_id": PREV_SEASON_ID,
        "curr_season_id": CURR_SEASON_ID,
        "banned_midway": banned,
        "clubs_checked": len([r for r in results if "error" not in r]),
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _log(f"wrote {OUT_JSON}")

    graded = [r for r in results if "error" not in r]
    total_flags = sum(r["flags"] for r in graded)
    _log(f"=== done: {len(graded)} clubs, {total_flags} total flags ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
