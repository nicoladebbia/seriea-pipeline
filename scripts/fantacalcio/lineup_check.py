"""Official-lineups consumer for the fantacalcio advisor (T-60).

The betting pipeline already fetches confirmed XIs into
``data/upcoming/confirmed_lineups.json`` (scheduler ``lineup_fetch`` stage,
T-55, sofascore-sourced, retried until confirmed). This module is the
fantacalcio side: when official lineups exist for a match of the coming
giornata and the league DEADLINE (the round's first kickoff) has not passed,
rebuild the XI advice with ground-truth p_play overrides and push a
last-minute diff. p_play sources are labeled official_xi / official_bench /
official_out so pred_ledger's per-source calibration covers them like every
probabilistic tier.

Matching is name-based (the feed has full names, the board has surnames):
accent-folded token-suffix match scoped to the club, uniqueness required on
BOTH sides, ambiguity fails open — the same ladder discipline as
xi_advisor._avail_lookup. ``official_out`` (in the club, in neither list) is
only inferred when the club's board rows matched the feed well enough that a
missing name means exclusion, not a rename.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIRMED = ROOT / "data" / "upcoming" / "confirmed_lineups.json"
BOARD = ROOT / "data" / "fantacalcio" / "auction_board.json"
STATE = ROOT / "data" / "fantacalcio" / "xi_notify_state.json"
ADVICE = ROOT / "data" / "fantacalcio" / "xi_advice.json"

P_OFFICIAL_XI = 0.97
P_OFFICIAL_BENCH = 0.15
P_OFFICIAL_OUT = 0.03
OUT_COVERAGE_MIN = 0.7   # below this, a missing name is a rename risk, not an exclusion
MAX_FEED_AGE_H = 3.0


def _tokens(name: str) -> tuple[str, ...]:
    from scripts.fantacalcio.xi_advisor import _deaccent
    return tuple(_deaccent(name).replace("-", " ").split())


def _split_initials(toks: tuple[str, ...]) -> tuple[tuple[str, ...], list[str]]:
    """The listone writes 'Esposito F.' / 'Gabriel T.' — surname plus
    first-name initial(s). Split them off so the surname can suffix-match the
    feed's full name and the initial can disambiguate the first name."""
    core = list(toks)
    inits = []
    while core and len(core[-1]) <= 2 and core[-1].endswith("."):
        inits.append(core.pop()[0])
    return tuple(core), inits


def _match_one(board_toks: tuple[str, ...],
               sofa_toks: list[tuple[str, ...]]) -> int | None:
    """Index of the unique sofa name whose token tail equals the board name
    (initials checked against the leading tokens), falling back to unique
    last-token match. None on miss or ambiguity."""
    core, inits = _split_initials(board_toks)
    if not core:
        return None
    n = len(core)
    hits = [i for i, st in enumerate(sofa_toks)
            if len(st) >= n and st[-n:] == core]
    if not hits:
        hits = [i for i, st in enumerate(sofa_toks)
                if st and st[-1] == core[-1]]
    if inits:
        hits = [i for i in hits
                if all(any(t and t[0] == ini
                           for t in sofa_toks[i][:len(sofa_toks[i]) - n])
                       for ini in inits)]
    return hits[0] if len(hits) == 1 else None


def _official_overrides(confirmed: dict, players: list[dict]) -> dict[int, dict]:
    """pid -> {p_play, src, club} from the confirmed-lineups feed. Pure."""
    from config.team_names import normalize_team as NT
    clubs: dict[str, dict[str, list[str]]] = {}
    for mk, m in (confirmed.get("matches") or {}).items():
        parts = mk.split(" vs ")
        if len(parts) != 2:
            continue
        for side, club in (("home", parts[0]), ("away", parts[1])):
            xi = m.get(f"{side}_lineup") or []
            bench = m.get(f"{side}_bench") or []
            if isinstance(xi, list) and len(xi) >= 7:
                clubs[NT(club)] = {"xi": list(xi), "bench": list(bench)}

    out: dict[int, dict] = {}
    for club, lists in clubs.items():
        rows = [p for p in players
                if NT(p.get("team", "")) == club and not p.get("departed")]
        if not rows:
            continue
        sofa_names = lists["xi"] + lists["bench"]
        sofa_toks = [_tokens(n) for n in sofa_names]
        n_xi = len(lists["xi"])
        # first pass: candidate target per board row
        cand: dict[int, int] = {}
        for p in rows:
            toks = _tokens(str(p.get("nome", "")))
            if not toks:
                continue
            hit = _match_one(toks, sofa_toks)
            if hit is not None:
                cand[int(p["id"])] = hit
        # both-side uniqueness: a sofa name claimed twice identifies nobody
        claims: dict[int, int] = {}
        for idx in cand.values():
            claims[idx] = claims.get(idx, 0) + 1
        matched = {pid: idx for pid, idx in cand.items() if claims[idx] == 1}
        for pid, idx in matched.items():
            if idx < n_xi:
                out[pid] = {"p_play": P_OFFICIAL_XI, "src": "official_xi",
                            "club": club}
            else:
                out[pid] = {"p_play": P_OFFICIAL_BENCH, "src": "official_bench",
                            "club": club}
        # exclusion inference, only when the club matched well enough that a
        # missing board name means "not in the 23", not "spelled differently"
        if rows and len(matched) / len(rows) >= OUT_COVERAGE_MIN:
            for p in rows:
                pid = int(p["id"])
                if pid not in matched and pid not in out:
                    out[pid] = {"p_play": P_OFFICIAL_OUT, "src": "official_out",
                                "club": club}
    return out


def run_official_lineup_check(now_ts: float | None = None) -> str:
    """Rebuild advice with official lineups and push a diff. Returns a status
    line for the scheduler log; never raises past its own guard rails."""
    import time
    now_ts = now_ts if now_ts is not None else time.time()
    try:
        confirmed = json.loads(CONFIRMED.read_text())
    except (OSError, ValueError):
        return "no confirmed_lineups.json"
    from scripts.fantacalcio.probabili import feed_age_h
    age = feed_age_h(confirmed)
    if age is None or age > MAX_FEED_AGE_H:
        return f"lineups feed stale ({age if age is None else round(age, 1)}h)"
    try:
        players = json.loads(BOARD.read_text())["players"]
    except (OSError, ValueError, KeyError) as e:
        return f"board unreadable: {e}"
    overrides = _official_overrides(confirmed, players)
    if not overrides:
        return "no board players in confirmed lineups"

    from scripts.fantacalcio.xi_advisor import build_advice
    adv = build_advice(official=overrides)
    fk = adv.get("first_kickoff")
    if not fk or now_ts >= float(fk):
        return "deadline passed (formation locked)"
    if not adv.get("xi"):
        return "no XI buildable"
    ADVICE.write_text(json.dumps(adv, indent=1, ensure_ascii=False))

    try:
        state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        state = {}
    rnd = adv.get("round")
    if state.get("round") != rnd or "advice" not in state:
        return "baseline push not sent yet (tracker owns the first push)"

    from scripts.fantacalcio.tracker import _advice_diff
    cur = {"module": adv.get("module"),
           "xi": sorted(x["nome"] for x in adv["xi"]),
           "bench": [x["nome"] for x in adv["bench"]],
           "vs": (state.get("advice") or {}).get("vs")}
    sig = json.dumps(cur, sort_keys=True, ensure_ascii=False)
    if state.get("official_sig") == sig:
        return "unchanged since last official check"
    diff = _advice_diff(state.get("advice", {}), cur)
    n_xi = sum(1 for o in overrides.values() if o["src"] == "official_xi")
    if not diff:
        state["official_sig"] = sig
        STATE.write_text(json.dumps(state))
        return f"officials in ({n_xi} confirmed titolari), advice unchanged"

    from scripts.fantacalcio.tracker import _SCHIERA_BTN, _feed_age_line
    from scripts.pipeline.notify import notify
    fl = _feed_age_line(adv.get("feed_ages"))
    notify(f"Giornata {rnd} — FORMAZIONI UFFICIALI, cambia la formazione\n"
           f"{diff}" + (f"\n{fl}" if fl else ""),
           title="Fantacalcio XI — ufficiali", level="warning",
           category="system",
           tg_html=(f"<b>🚨 Giornata {rnd} — formazioni UFFICIALI</b>\n{diff}"
                    + (f"\n<i>{fl}</i>" if fl else "")),
           tg_reply_markup=_SCHIERA_BTN)
    state.update({"advice": cur, "official_sig": sig})
    STATE.write_text(json.dumps(state))
    return "pushed official-lineup update"
