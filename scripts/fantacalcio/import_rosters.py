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


CAL_FILES = [ROOT / "data" / "fantacalcio" / "calendar_coppa_del_nonno.xlsx",
             ROOT / "data" / "fantacalcio" / "calendar_hunger_games.xlsx"]
SCHEDULE_OUT = ROOT / "data" / "fantacalcio" / "league_schedule.json"
_HDR_LEGA = __import__("re").compile(r"(\d+)ª Giornata lega")
_HDR_SA = __import__("re").compile(r"(\d+)ª Giornata serie a")


def parse_calendar_xlsx(path: Path, known_teams: set[str]) -> tuple[str, list]:
    """One Leghe calendar export -> (competition name, rounds).

    Layout (specimen-verified 2026-09-02): two league-rounds per row-block,
    side by side; header cells "Nª Giornata lega" / "Mª Giornata serie a";
    fixture rows are [girone?] home 0.0 0 away "-"; gironi exports add a
    per-row girone letter and "Riposa X" rows. Team tokens are validated
    against the roster import, NEVER positional — the score cells fill in as
    rounds play, and a positional parse would break on the first real score.
    """
    df = pd.read_excel(path, header=None)
    comp = str(df.iloc[0, 0]).replace("Calendario", "").strip()
    headers = []          # (row, col, league_round, sa_round)
    for r in range(df.shape[0]):
        row_hdrs = []
        for c in range(df.shape[1]):
            v = df.iloc[r, c]
            m = _HDR_LEGA.search(str(v)) if pd.notna(v) else None
            if m:
                row_hdrs.append((c, int(m.group(1))))
        for i, (c, lr) in enumerate(row_hdrs):
            sa = None
            for c2 in range(c + 1, df.shape[1]):
                v = df.iloc[r, c2]
                m = _HDR_SA.search(str(v)) if pd.notna(v) else None
                if m:
                    sa = int(m.group(1))
                    break
            end_c = row_hdrs[i + 1][0] if i + 1 < len(row_hdrs) else df.shape[1]
            headers.append((r, c, end_c, lr, sa))
    hdr_rows = sorted({h[0] for h in headers})
    rounds = {}
    for r0, c0, c1, lr, sa in headers:
        nxt = next((hr for hr in hdr_rows if hr > r0), df.shape[0])
        fixtures, rests = [], []
        for r in range(r0 + 1, nxt):
            toks = [str(v).strip() for v in df.iloc[r, c0:c1] if pd.notna(v)]
            if not toks:
                continue
            girone = toks[0] if toks and len(toks[0]) == 1 and toks[0].isalpha() \
                else None
            rest = next((t for t in toks if t.startswith("Riposa ")), None)
            if rest:
                rests.append({"girone": girone,
                              "team": rest.removeprefix("Riposa ")})
                continue
            names = [t for t in toks if t in known_teams]
            if len(names) == 2:
                fixtures.append({"girone": girone, "home": names[0],
                                 "away": names[1]})
        if fixtures:
            rounds[lr] = {"league_round": lr, "sa_round": sa,
                          "fixtures": fixtures, "rests": rests}
    out = [rounds[k] for k in sorted(rounds)]
    if len(out) < 8 or any(len(rd["fixtures"]) < 2 or rd["sa_round"] is None
                           for rd in out):
        raise SystemExit(f"{path.name}: parse looks broken "
                         f"({len(out)} rounds) — schema changed?")
    return comp, out


def import_calendars(paths: list[Path] = CAL_FILES) -> dict:
    league = json.loads(LEAGUE_OUT.read_text())
    known = set(league["teams"])
    out = {"generated_at": datetime.now(UTC).isoformat(), "competitions": {}}
    for path in paths:
        comp, rounds = parse_calendar_xlsx(path, known)
        fmt = "gironi" if any(f["girone"] for rd in rounds
                              for f in rd["fixtures"]) else "campionato"
        out["competitions"][comp] = {"format": fmt, "source_file": str(path),
                                     "rounds": rounds}
        print(f"{comp}: {len(rounds)} rounds ({fmt}), sa "
              f"{rounds[0]['sa_round']}..{rounds[-1]['sa_round']}")
    SCHEDULE_OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    if "--calendars" in sys.argv:
        extra = [Path(a) for a in sys.argv[2:] if not a.startswith("-")]
        import_calendars(extra or CAL_FILES)
    else:
        import_rosters(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX)
