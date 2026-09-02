"""Trade scanner + incoming-offer evaluator for the fantacalcio league.

The complementary-surplus analysis done by hand for the Carnesecchi probe,
as a standing function: a window opens when MY depth at a role would START
for a rival (their poverty) while THEIR depth would upgrade MY starting
line. Both sides must gain or the trade never happens — pairs are ranked by
my gain, with the rival's gain shown as the sales pitch.

Currency is the same one the svincolati radar and the XI advisor rank by:
``p_play x level`` (probabili-informed chance of getting a voto, times the
shrunk live fantamedia). No fixtures, no congestion — trade value is a
season-long question, not a matchweek one.

Offer evaluator: ``evaluate_offer(give, get)`` simulates the swap on my
roster and scores team strength as the best XI over the league's legal
modules, plus a bench-depth guard (the league needs 3P/8D/8C/6A — a trade
that breaks the roster shape is flagged, not scored).

CLI:
    python3 -m scripts.fantacalcio.trades                    # scan windows
    python3 -m scripts.fantacalcio.trades --offer \\
        --give "Douvikas" --get "Kean, Martinez L."          # evaluate
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.fantacalcio.namematch import norm
from scripts.fantacalcio.tracker import MODULES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "fantacalcio" / "trades.json"

# A candidate pair is a window only when both sides gain visibly.
MIN_GAIN = 0.05
MAX_PAIRS = 10
MAX_PER_RIVAL = 3
# Legal roster shape (league rules: 3 P, 8 D, 8 C, 6 A).
ROSTER_SHAPE = {"P": 3, "D": 8, "C": 8, "A": 6}


def _slots(module: tuple[int, int, int]) -> dict[str, int]:
    d, c, a = module
    return {"P": 1, "D": d, "C": c, "A": a}


def _score(r: dict) -> float:
    return r["p_play"] * r["level"]


def team_strength(rows: list[dict]) -> tuple[float, str]:
    """Best XI total (p_play x level currency) over the legal modules."""
    by_role = {ro: sorted((r for r in rows if r["R"] == ro),
                          key=_score, reverse=True) for ro in "PDCA"}
    best, best_m = 0.0, None
    for m in MODULES:
        sl = _slots(m)
        if any(len(by_role[ro]) < n for ro, n in sl.items()):
            continue
        tot = sum(_score(r) for ro, n in sl.items() for r in by_role[ro][:n])
        if tot > best:
            best, best_m = tot, f"{m[0]}-{m[1]}-{m[2]}"
    return best, best_m


def starter_lines(rows: list[dict], module: tuple[int, int, int]) -> dict[str, float]:
    """Per role, the score of the WEAKEST starter under the given module —
    the bar a newcomer must clear to enter the XI."""
    out = {}
    for ro, n in _slots(module).items():
        pool = sorted((_score(r) for r in rows if r["R"] == ro), reverse=True)
        out[ro] = pool[n - 1] if len(pool) >= n else 0.0
    return out


def scan_windows(squads: dict[str, list[dict]], my_name: str) -> list[dict]:
    """Best trade window per (rival, my role -> their role).

    Two passes: a cheap starter-line prefilter (both raw gains >= MIN_GAIN),
    then the EXACT team-strength delta on both sides — my_gain that ignores
    what the XI loses by giving a starter away overrates every swap (caught
    by test before shipping). A window is kept only when I gain visibly and
    the rival at least doesn't lose — the reason they say yes.
    """
    my_rows = squads[my_name]
    my_before, my_mod = team_strength(my_rows)
    if my_mod is None:
        return []
    my_line = starter_lines(my_rows, tuple(int(x) for x in my_mod.split("-")))
    best: dict[tuple, dict] = {}
    for rival, rows in squads.items():
        if rival == my_name:
            continue
        r_before, r_mod = team_strength(rows)
        if r_mod is None:
            continue
        r_line = starter_lines(rows, tuple(int(x) for x in r_mod.split("-")))
        for mine in my_rows:
            if _score(mine) - r_line[mine["R"]] < MIN_GAIN:
                continue
            for theirs in rows:
                if _score(theirs) - my_line[theirs["R"]] < MIN_GAIN:
                    continue
                my_after, _ = team_strength(
                    [r for r in my_rows if r["id"] != mine["id"]] + [theirs])
                my_net = my_after - my_before
                if my_net < MIN_GAIN:
                    continue
                r_after, _ = team_strength(
                    [r for r in rows if r["id"] != theirs["id"]] + [mine])
                their_net = r_after - r_before
                if their_net < 0.02:
                    continue
                key = (rival, mine["R"], theirs["R"])
                cand = {"rival": rival,
                        "give": mine["nome"], "give_R": mine["R"],
                        "give_score": round(_score(mine), 2),
                        "get": theirs["nome"], "get_R": theirs["R"],
                        "get_score": round(_score(theirs), 2),
                        "my_gain": round(my_net, 2),
                        "their_gain": round(their_net, 2)}
                if key not in best or cand["my_gain"] > best[key]["my_gain"]:
                    best[key] = cand
    pairs = sorted(best.values(), key=lambda p: -p["my_gain"])
    kept, per_rival = [], {}
    for p in pairs:
        if per_rival.get(p["rival"], 0) >= MAX_PER_RIVAL:
            continue
        per_rival[p["rival"]] = per_rival.get(p["rival"], 0) + 1
        kept.append(p)
        if len(kept) >= MAX_PAIRS:
            break
    return kept


def evaluate_offer(give: list[dict], get: list[dict],
                   my_rows: list[dict]) -> dict:
    """Score an incoming offer: my roster minus `give` plus `get`.

    Verdict on the best-XI delta: ACCETTA above +0.15, RIFIUTA below -0.05,
    TRATTA in between (a fair swap worth negotiating on price).
    """
    give_ids = {r["id"] for r in give}
    after = [r for r in my_rows if r["id"] not in give_ids] + get
    counts = {ro: sum(1 for r in after if r["R"] == ro) for ro in "PDCA"}
    shape_ok = counts == ROSTER_SHAPE
    before_s, before_m = team_strength(my_rows)
    after_s, after_m = team_strength(after)
    delta = after_s - before_s
    verdict = "ACCETTA" if delta > 0.15 else \
        "RIFIUTA" if delta < -0.05 else "TRATTA"
    if not shape_ok:
        verdict = "ROSA ILLEGALE"
    return {"delta": round(delta, 2), "verdict": verdict,
            "before": round(before_s, 2), "after": round(after_s, 2),
            "module_before": before_m, "module_after": after_m,
            "shape_ok": shape_ok, "counts": counts,
            "give": [f"{r['R']} {r['nome']} ({round(_score(r), 2)})" for r in give],
            "get": [f"{r['R']} {r['nome']} ({round(_score(r), 2)})" for r in get]}


# ---------------------------------------------------------------------------
# IO wrappers
# ---------------------------------------------------------------------------

def _squads() -> tuple[dict[str, list[dict]], str]:
    from scripts.fantacalcio.probabili import fetch_probabili, status_by_pid
    from scripts.fantacalcio.xi_advisor import BOARD, _history, _rival_roster
    board = json.loads(BOARD.read_text())
    by_id = {int(p["id"]): p for p in board["players"]}
    hist = _history()
    prob = status_by_pid(fetch_probabili())
    league = json.loads(
        (ROOT / "data" / "fantacalcio" / "league_rosters.json").read_text())
    squads = {name: _rival_roster(entry, by_id, hist, prob)
              for name, entry in league["teams"].items()}
    return squads, league["my_team"]


def build_trades() -> dict:
    squads, me = _squads()
    windows = scan_windows(squads, me)
    strengths = sorted(
        ((name, *team_strength(rows)) for name, rows in squads.items()),
        key=lambda t: -t[1])
    out = {"generated_at": datetime.now(UTC).isoformat(),
           "my_team": me, "windows": windows,
           "strengths": [{"team": n, "strength": round(s, 2), "module": m,
                          "me": n == me} for n, s, m in strengths]}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def _resolve(names: list[str], pool: list[dict], label: str) -> list[dict]:
    rows = []
    for raw in names:
        cand = [r for r in pool if norm(r["nome"]) == norm(raw)]
        if not cand:
            cand = [r for r in pool
                    if norm(r["nome"]).split()[0] == norm(raw).split()[0]]
        if len(cand) != 1:
            raise SystemExit(
                f"{label} {raw!r}: {'ambiguous' if cand else 'not found'} "
                f"({[c['nome'] for c in cand][:5]})")
        rows.append(cand[0])
    return rows


def main() -> None:
    if "--offer" in sys.argv:
        def _arg(flag: str) -> list[str]:
            i = sys.argv.index(flag)
            return [x.strip() for x in sys.argv[i + 1].split(",") if x.strip()]
        squads, me = _squads()
        everyone = [r for rows in squads.values() for r in rows]
        give = _resolve(_arg("--give"), squads[me], "give")
        get = _resolve(_arg("--get"), everyone, "get")
        res = evaluate_offer(give, get, squads[me])
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return
    out = build_trades()
    print(f"{len(out['windows'])} trade windows")
    for w in out["windows"]:
        print(f"  {w['rival']:<22} {w['give_R']} {w['give']} ({w['give_score']})"
              f" -> {w['get_R']} {w['get']} ({w['get_score']})"
              f"  io {w['my_gain']:+.2f} / loro {w['their_gain']:+.2f}")


if __name__ == "__main__":
    main()
