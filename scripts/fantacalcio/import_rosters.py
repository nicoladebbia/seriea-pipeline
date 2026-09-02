"""Import EVERY team's roster from a Leghe Fantacalcio league export.

The league xlsx ("Rose" export: one 3-column block per team — name, costo,
spacer — with a trailing "totale" footer row) is the only source of truth for
who owns whom, and it changes whenever the league trades or repairs
(svincoli/scambi). Re-run this after dropping a fresh export to refresh:

  * ``league_rosters.json`` — all 10 squads, board-id-matched, for trade
    analysis ("can we do switches any time").
  * ``my_team.json`` — MY_TEAM's squad in the exact shape the tracker and the
    XI advisor read ({"roster": [{"id", "paid"}]}).

Names are matched against the auction board (fantacalcio.it listone ids — the
same id space the voti pages use, so a matched id is scoreable forever).
An unmatched name is REPORTED, never silently dropped: it usually means a
post-listone arrival (paid lesson: Gonzalez N. returned to Juventus after the
listone snapshot; his pid was recovered from the round-2 voti page and patched
into the board + patched listone).

Usage:
    python3 -m scripts.fantacalcio.import_rosters [path/to/export.xlsx]
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.fantacalcio.namematch import norm

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "data" / "fantacalcio" / "auction_board.json"
DEFAULT_XLSX = ROOT / "data" / "fantacalcio" / "league_rosters_source.xlsx"
LEAGUE_OUT = ROOT / "data" / "fantacalcio" / "league_rosters.json"
MY_OUT = ROOT / "data" / "fantacalcio" / "my_team.json"

MY_TEAM = "Whisky Palermo"
BUDGET = 500


def parse_league_xlsx(path: Path) -> dict[str, list[tuple[str, int]]]:
    """{team_name: [(player_name, paid), ...]} — 'totale' footers dropped."""
    df = pd.read_excel(path, header=None)
    teams: dict[str, list[tuple[str, int]]] = {}
    for c in range(0, df.shape[1], 3):
        tname = df.iloc[0, c]
        if pd.isna(tname) or not str(tname).strip():
            continue
        rows = []
        for r in range(1, df.shape[0]):
            nome = df.iloc[r, c]
            if pd.isna(nome) or str(nome).strip().lower() == "totale":
                continue
            cost = df.iloc[r, c + 1]
            rows.append((str(nome).strip(), int(cost) if pd.notna(cost) else 0))
        if rows:
            teams[str(tname).strip()] = rows
    return teams


def import_rosters(xlsx: Path = DEFAULT_XLSX) -> dict:
    teams = parse_league_xlsx(xlsx)
    board = json.loads(BOARD.read_text())
    by_norm: dict[str, list[dict]] = {}
    for p in board["players"]:
        by_norm.setdefault(norm(p["nome"]), []).append(p)

    out: dict = {"generated_at": datetime.now(UTC).isoformat(),
                 "source_file": str(xlsx), "my_team": MY_TEAM, "teams": {}}
    problems: list[str] = []
    for tname, rows in teams.items():
        roster, unmatched = [], []
        for nome, paid in rows:
            cand = by_norm.get(norm(nome), [])
            if len(cand) == 1:
                p = cand[0]
                roster.append({"id": int(p["id"]), "nome": p["nome"],
                               "R": p["R"], "team": p["team"], "paid": paid})
            else:
                unmatched.append({"nome": nome, "paid": paid})
                problems.append(f"{tname}: {nome!r} "
                                f"({'ambiguous' if cand else 'not in board'})")
        out["teams"][tname] = {"spent": sum(r[1] for r in rows),
                               "n": len(rows), "roster": roster,
                               "unmatched": unmatched}

    if MY_TEAM not in out["teams"]:
        raise SystemExit(f"MY_TEAM {MY_TEAM!r} not in export "
                         f"(teams: {sorted(out['teams'])})")

    LEAGUE_OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    mine = out["teams"][MY_TEAM]
    MY_OUT.write_text(json.dumps({
        "team_name": MY_TEAM,
        "budget": BUDGET,
        "imported_at": out["generated_at"],
        "roster": [{"id": r["id"], "paid": r["paid"]} for r in mine["roster"]],
        "unmatched": mine["unmatched"],
    }, indent=1, ensure_ascii=False))

    print(f"imported {len(out['teams'])} teams -> {LEAGUE_OUT.name}; "
          f"{MY_TEAM}: {len(mine['roster'])} players -> {MY_OUT.name}")
    for line in problems:
        print(f"  UNMATCHED {line} — resolve the pid (voti page once he plays) "
              f"and patch the board, or the player can never be scored")
    return out


if __name__ == "__main__":
    import_rosters(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX)
