"""Detect and log what actually changed between two transfer/squad refreshes.

The daily/twice-daily refresh overwrites the parquets; on its own it never says
WHAT moved. This module diffs the freshly-scraped squad against the previous
snapshot and emits a change record for each real delta, so the dashboard can show
a "recent changes" feed and nothing is silently overwritten.

Tracked change types (per the 2026-07-14 scope decision):
  - signing        a player now in the squad who wasn't before
  - departure      a player who was in the squad and is gone
  - value_change   market value moved by more than VALUE_EPS
  - contract_change contract-until date changed (extension / new deal)

Rumors are deliberately NOT tracked (noisy, display-only, never model-fed).

Snapshot + changelog live under data/external/transfermarkt/:
  squad_snapshot_{season}.json   the previous squad state (per club → players)
  transfer_changes_{season}.json append-only log of change records (newest first)

Free + self-contained: reads only local parquets the scrapers already write, no
paid API, no external call. Safe to run every refresh — first run seeds the
snapshot and logs nothing (no phantom "everyone signed" on a cold start).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

TM_DIR = DATA_DIR / "external" / "transfermarkt"

# A market value must move more than this (EUR) to count as a change — filters
# out tiny re-estimates. €500k is below one materiality step for a squad tracker.
VALUE_EPS = 500_000
# Cap the changelog so it can't grow without bound across a months-long window.
MAX_LOG = 2_000


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", " ", s.lower()).strip()


def _load_current_squads(season: str) -> dict[str, dict[str, dict]]:
    """Read market_values_{season}.parquet → {club: {norm_name: player_dict}}."""
    sfx = season.replace("-", "_")
    path = TM_DIR / f"market_values_{sfx}.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    squads: dict[str, dict[str, dict]] = {}
    for _, r in df.iterrows():
        club = str(r.get("team"))
        key = _norm(r.get("player_name"))
        if not key:
            continue
        squads.setdefault(club, {})[key] = {
            "name": r.get("player_name"),
            "position": r.get("position"),
            "value": None if pd.isna(r.get("market_value_eur")) else float(r["market_value_eur"]),
            "contract_until": r.get("contract_until") if pd.notna(r.get("contract_until")) else None,
        }
    return squads


def _snapshot_path(season: str) -> Path:
    return TM_DIR / f"squad_snapshot_{season.replace('-', '_')}.json"


def _changelog_path(season: str) -> Path:
    return TM_DIR / f"transfer_changes_{season.replace('-', '_')}.json"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        log.warning("could not read %s — treating as empty", path.name)
        return default


def detect_changes(season: str = "2026-2027") -> list[dict]:
    """Diff the current squads against the saved snapshot; log + return changes.

    First run (no snapshot) seeds the snapshot and returns [] — a cold start must
    not report the whole league as new signings. Every later run reports only the
    real deltas since the previous snapshot, appends them to the changelog
    (newest first, capped), and saves the new snapshot.
    """
    current = _load_current_squads(season)
    if not current:
        log.warning("no squad data for %s — nothing to diff", season)
        return []

    snap_path = _snapshot_path(season)
    previous = _load_json(snap_path, None)
    now = datetime.now(UTC).isoformat()

    # Cold start: seed and report nothing.
    if previous is None:
        snap_path.write_text(json.dumps(current, ensure_ascii=False))
        log.info("seeded squad snapshot for %s (%d clubs) — no changes on first run",
                 season, len(current))
        return []

    changes: list[dict] = []
    for club, players in current.items():
        prev_players = previous.get(club, {})
        prev_keys = set(prev_players)
        cur_keys = set(players)

        for key in cur_keys - prev_keys:
            p = players[key]
            changes.append({
                "type": "signing", "club": club, "player": p["name"],
                "position": p["position"], "value": p["value"],
                "detail": f"joined {club}", "at": now,
            })
        for key in prev_keys - cur_keys:
            p = prev_players[key]
            changes.append({
                "type": "departure", "club": club, "player": p.get("name"),
                "position": p.get("position"), "value": p.get("value"),
                "detail": f"left {club}", "at": now,
            })
        for key in cur_keys & prev_keys:
            cur_p, prev_p = players[key], prev_players[key]
            cv, pv = cur_p.get("value"), prev_p.get("value")
            if cv is not None and pv is not None and abs(cv - pv) > VALUE_EPS:
                direction = "up" if cv > pv else "down"
                changes.append({
                    "type": "value_change", "club": club, "player": cur_p["name"],
                    "position": cur_p["position"], "value": cv, "prev_value": pv,
                    "detail": f"value {direction} €{pv/1e6:.1f}m → €{cv/1e6:.1f}m",
                    "at": now,
                })
            cc, pc = cur_p.get("contract_until"), prev_p.get("contract_until")
            if cc and pc and cc != pc:
                changes.append({
                    "type": "contract_change", "club": club, "player": cur_p["name"],
                    "position": cur_p["position"], "value": cur_p.get("value"),
                    "contract_until": cc, "prev_contract_until": pc,
                    "detail": f"contract {pc} → {cc}", "at": now,
                })

    # Persist: prepend new changes (newest first), cap, save new snapshot.
    if changes:
        existing = _load_json(_changelog_path(season), [])
        combined = (changes + existing)[:MAX_LOG]
        _changelog_path(season).write_text(json.dumps(combined, ensure_ascii=False, indent=1))
        log.info("logged %d transfer changes for %s", len(changes), season)
    snap_path.write_text(json.dumps(current, ensure_ascii=False))
    return changes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ch = detect_changes("2026-2027")
    print(f"\n{len(ch)} changes detected")
    for c in ch[:25]:
        print(f"  [{c['type']:15s}] {c.get('club'):12s} {c.get('player')} — {c.get('detail')}")
