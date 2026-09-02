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
import re
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


QUOTAZIONI_URL = "https://www.fantacalcio.it/quotazioni-fantacalcio"
MERCATO_TTL_H = 7 * 24.0     # window is shut most of the year; weekly is plenty
_QROW = __import__("re").compile(
    r'data-filter-keywords="([^"]+)"[^>]*data-filter-role-classic="([pdca])"'
    r'.*?/serie-a/squadre/([^/"]+)/[^/"]+/(\d+)"', __import__("re").S)


def parse_quotazioni(html: str) -> dict[int, dict] | None:
    """Live fantacalcio.it listone -> {pid: {nome, R, team}}. None on break.

    The quotazioni page is the primary source for WHO IS IN SERIE A right now:
    a player who left after our auction snapshot is simply absent, an arrival
    is present. Same pid space as the voti pages and the board, so no name
    matching. Sentinels: >=400 rows, 20 club slugs, Svilar present.
    """
    rows = html.split('<tr class="player-row"')[1:]
    out: dict[int, dict] = {}
    clubs: set[str] = set()
    for row in rows:
        m = _QROW.search(row)
        if not m:
            continue
        nome, role, team_slug, pid = m.groups()
        nome = __import__("html").unescape(nome)
        team = team_slug.replace("-", " ").title()
        clubs.add(team)
        out[int(pid)] = {"nome": nome.strip(), "R": role.upper(), "team": team}
    if len(out) < 400 or len(clubs) < 20 \
            or not any(v["nome"] == "Svilar" for v in out.values()):
        return None
    return out


def sync_mercato(force: bool = False) -> list[str] | None:
    """Reconcile the board against the LIVE listone: arrivals, club moves,
    placeholder-pid adoption, status verification. Returns the change log
    (possibly empty), or None when the fetch/parse failed or the last sync is
    fresh (TTL-gated — the tracker calls this every run; the mercato only
    matters weekly, and in January).

    What this deliberately does NOT do: mark departures. Measured 2026-09-02:
    the quotazioni page KEEPS the rows of players who left Serie A (Di
    Gregorio still listed under JUV a week after moving to Bournemouth), so
    presence there proves nothing about being gone, and absence is the only
    signal it gives — used here solely for orphan reporting. DEPARTED comes
    from the board builder's wiki-transfers pass and is never touched here.

    Never destroys: a failed fetch changes nothing, and rows are only ever
    annotated (status/team/note) — auction economics (fvm, qt, prices) are
    frozen at auction day on purpose.
    """
    board = json.loads(BOARD.read_text())
    if not force:
        try:
            age_h = (datetime.now(UTC) - datetime.fromisoformat(
                board["mercato_synced_at"])).total_seconds() / 3600
            if age_h < MERCATO_TTL_H:
                return None
        except (KeyError, ValueError):
            pass
    try:
        from curl_cffi import requests as rq
        r = rq.get(QUOTAZIONI_URL, impersonate="chrome124", timeout=30)
        live = parse_quotazioni(r.text) if r.status_code == 200 else None
    except Exception:
        live = None
    if live is None:
        print("sync_mercato: quotazioni fetch/parse failed — board untouched")
        return None

    changes = _apply_live(board, live)
    board["mercato_synced_at"] = datetime.now(UTC).isoformat()
    tmp = BOARD.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(board, indent=1, ensure_ascii=False))
    tmp.replace(BOARD)
    print(f"sync_mercato: {len(changes)} changes")
    for c in changes:
        print(f"  {c}")
    return changes


def _apply_live(board: dict, live: dict[int, dict]) -> list[str]:
    """Pure apply step of sync_mercato — see its docstring for the contract."""
    today = datetime.now(UTC).date().isoformat()
    changes: list[str] = []
    seen = set()
    for p in board["players"]:
        pid = int(p["id"])
        seen.add(pid)
        lv = live.get(pid)
        if lv is None or p.get("status") == "DEPARTED":
            continue    # departures are the wiki-transfers pass's call, not ours
        if p.get("status") in (None, "UNVERIFIED", "TEAM_MISMATCH"):
            p["status"] = "OK"
            changes.append(f"VERIFIED  {p['R']} {p['nome']} (live listone)")
        if lv["team"] != p["team"]:
            p.setdefault("team_listone", p["team"])
            changes.append(f"MOVED     {p['R']} {p['nome']} "
                           f"{p['team']} -> {lv['team']}")
            p["team"] = lv["team"]
    # Placeholder adoption: the auction build stamped post-listone arrivals
    # with synthetic 99xxx ids. Those ids exist in NO other artifact (voti,
    # probabili), so the player could never be scored. When the live listone
    # supplies the real pid, correct the placeholder row IN PLACE — it holds
    # the auction priors (fvm, proj_min, mv_hat) the bare arrival row lacks.
    # Match by first surname token, and only when unique both ways.
    def _tok(nome: str) -> str:
        return norm(nome).split()[0]
    placeholders = [p for p in board["players"] if int(p["id"]) >= 99000]
    for pid, lv in live.items():
        if pid in seen:
            continue
        cand = [f for f in placeholders if _tok(f["nome"]) == _tok(lv["nome"])]
        if len(cand) == 1:
            f = cand[0]
            placeholders.remove(f)
            changes.append(f"PID-FIX   {lv['R']} {lv['nome']} "
                           f"{f['id']} -> {pid} ({lv['team']})")
            if lv["team"] != f["team"]:
                f.setdefault("team_listone", f["team"])
            f.update(id=pid, nome=lv["nome"], R=lv["R"], team=lv["team"],
                     status="OK",
                     note=f"pid corrected via live listone {today}")
            continue
        board["players"].append({
            "id": pid, "nome": lv["nome"], "R": lv["R"], "team": lv["team"],
            "status": "OK", "note": f"post-listone arrival {today}",
            "new": True})
        changes.append(f"ARRIVED   {lv['R']} {lv['nome']} ({lv['team']})")
    for f in placeholders:
        changes.append(f"ORPHANED  placeholder {f['id']} {f['nome']} "
                       f"({f['team']}) — no live-listone partner, left as-is")
    return changes


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
                # Score cells (specimen-verified on the UNPLAYED shape only:
                # home | 0.0 | 0 | away | "-"). Mapping assumption for played
                # rows — fp_home, fp_away between the teams, goals result
                # after the away team — MUST be re-verified against the first
                # settled giornata before trusting standings built on it;
                # build_standings only counts fixtures whose score matches
                # N-N, so a wrong guess yields an empty table, not a wrong one.
                hi, ai = toks.index(names[0]), toks.index(names[1])
                mid = [t for t in toks[hi + 1:ai]
                       if re.fullmatch(r"\d+(\.\d+)?", t)]
                after = next((t for t in toks[ai + 1:]
                              if re.fullmatch(r"\d+-\d+", t)), None)
                fixtures.append({"girone": girone, "home": names[0],
                                 "away": names[1],
                                 "fp_home": float(mid[0]) if mid else None,
                                 "fp_away": float(mid[1]) if len(mid) > 1 else None,
                                 "score": after})
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
    if "--sync-mercato" in sys.argv:
        sync_mercato(force=True)
        import_rosters()
    elif "--calendars" in sys.argv:
        extra = [Path(a) for a in sys.argv[2:] if not a.startswith("-")]
        import_calendars(extra or CAL_FILES)
    else:
        import_rosters(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XLSX)
