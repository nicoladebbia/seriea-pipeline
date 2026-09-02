"""Score my fantacalcio team, round by round, against the published voti.

Two numbers per round, and the difference between them is the point:

  * `settable` -- the XI a manager could actually have fielded. The module and the eleven
    are chosen from the PROJECTION, before kickoff, knowing nothing about who played.
    Starters who turn out to have no voto are then replaced by the ordered bench, at most
    three, same role, exactly as Leghe does it.
  * `hindsight` -- the best eleven the roster could have produced knowing every result.
    Unattainable by construction. It is reported only as the ceiling `settable` is
    measured against; quoting it as "my score" would be a lie.

Reporting only the hindsight number is the obvious way to manufacture a flattering total,
which is why both are computed and the page labels them.

The modificatore di difesa is scored on the RAW VOTO of the keeper and the best three
fielded defenders -- not the fantavoto -- using the same table and the same voto d'ufficio
the auction board priced with. A module with three defenders forfeits it entirely, so the
module search compares four-defender and three-defender shapes on total points including
the modifier rather than assuming the modifier always wins.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.fantacalcio.live_scores import fetch_round, played_rounds

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "data" / "fantacalcio" / "auction_board.json"
TEAM = ROOT / "data" / "fantacalcio" / "my_team.json"
OUT = ROOT / "data" / "fantacalcio" / "tracker.json"

SEASON = "2026-27"
# In-season level updating. Measured on all 38 rounds of 2025-26 (10,787 player-rounds):
# round-to-round "form" does NOT exist -- lag-1..3 autocorrelation of a player's residual
# fantavoto is -0.04 to -0.06, slightly mean-REVERTING -- so there is deliberately no
# last-game term here. What is real is the LEVEL: a player's first-10-round mean predicts
# his next ten at r=0.496, and Spearman-Brown turns that into the shrinkage constant
# below: after ~10 observed rounds the season's evidence deserves the same weight as the
# auction prior. live_level = (K*prior + n*observed_mean) / (K + n).
LEVEL_K = 10.0
MAX_SUBS = 3
ROUNDS = 38
# What a four-defender module has to be worth for the module search to prefer it over a
# stronger three-defender eleven. Seeded from the ~47/season the auction board measured.
#
# Measured on a full 38-round replay: this behaves as a THRESHOLD, not a dial. At 0 the
# search takes a three-defender module in 38 of 38 rounds; at anything from ~1.24 upward it
# takes four in 38 of 38, and the season total is identical (2954.0) at 1.24, 2.0 and 6.0.
# So the exact value does not matter, only that it clears the bar -- and it is conservative:
# the modifier actually paid 71 points over those 38 rounds (1.87/round) against the
# 1.24/round assumed here. Fielding the fourth defender was worth +31 points on the season.
MOD_EV = 47.0 / ROUNDS
# (D, C, A). One keeper always. These are the modules Leghe accepts; the four- and
# five-defender shapes are the ones that can earn the modifier.
MODULES = [(3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1), (5, 3, 2), (5, 4, 1)]


def _modifier(gk_voto: float | None, d_votos: list[float],
              table: list, office: list[float]) -> float:
    """Modifier points for one round. Missing defenders take the voto d'ufficio."""
    if gk_voto is None:
        return 0.0
    have = sorted([v for v in d_votos if v is not None], reverse=True)[:3]
    for i in range(3 - len(have)):
        have.append(office[min(i, len(office) - 1)])
    avg = (gk_voto + sum(have)) / 4.0
    pts = 0.0
    for threshold, value in table:
        if avg >= threshold:
            pts = float(value)
    return pts


def _pick(roster: list[dict], votes: dict, table: list, office: list[float],
          key: str) -> dict:
    """Best legal lineup under `key` ordering, with substitutions applied.

    `key` is what the eleven is CHOSEN on -- "proj" for the settable lineup (no knowledge
    of the round) and "actual" for the hindsight ceiling. Scoring is always on the actual
    fantavoti; only the selection differs.
    """
    by_role = {r: [] for r in "PDCA"}
    for p in roster:
        by_role.get(p["R"], []).append(p)

    def score_of(p):
        v = votes.get(p["id"])
        return (v or {}).get("fantavoto")

    def order(pool):
        if key == "proj":
            return sorted(pool, key=lambda p: -p.get("proj", 0.0))
        # Hindsight: a player with no voto is worth nothing and sorts last.
        return sorted(pool, key=lambda p: (-(score_of(p) if score_of(p) is not None
                                             else -99), -p.get("proj", 0.0)))

    best = None
    for nd, nc, na in MODULES:
        need = {"P": 1, "D": nd, "C": nc, "A": na}
        if any(len(by_role[r]) < n for r, n in need.items()):
            continue
        xi, bench, subs = [], [], []
        for r, n in need.items():
            ranked = order(by_role[r])
            xi += [dict(p, slot="start") for p in ranked[:n]]
            bench += ranked[n:]
        # Substitutions, capped at three like the real rules. Leghe enters the bench in
        # PANCHINA ORDER, not in role order: when more than three starters are missing it
        # is the bench ranking that decides which three come on, and iterating the XI
        # P-D-C-A instead would quietly always favour defenders. That would score better
        # under the modifier and would not be the rule, so the pairings are built first
        # and then taken in bench order.
        ranked_bench = order(bench)
        pairs = []
        for i, p in enumerate(xi):
            if score_of(p) is not None:
                continue
            for b in ranked_bench:
                if b["R"] == p["R"] and score_of(b) is not None \
                        and not any(b is q for _, q in pairs):
                    pairs.append((i, b))
                    break
        pairs.sort(key=lambda t: ranked_bench.index(t[1]))
        for i, sub in pairs[:MAX_SUBS]:
            out = xi[i]
            bench.remove(sub)
            xi[i] = dict(sub, slot="sub", replaced=out["nome"])
            subs.append({"out": out["nome"], "in": sub["nome"]})
        gk_v = next(((votes.get(p["id"]) or {}).get("voto") for p in xi
                     if p["R"] == "P"), None)
        d_v = [(votes.get(p["id"]) or {}).get("voto") for p in xi if p["R"] == "D"]
        # A three-defender module cannot earn the modifier at all.
        mod = _modifier(gk_v, d_v, table, office) if nd >= 4 else 0.0
        # A fielded player who never got a voto and could not be replaced scores nothing.
        base = sum(score_of(p) or 0.0 for p in xi)
        total = base + mod
        row = {"module": f"{nd}-{nc}-{na}", "base": round(base, 2),
               "modifier": round(mod, 2), "total": round(total, 2),
               "subs": subs, "xi": [
                   {"id": p["id"], "nome": p["nome"], "R": p["R"], "team": p["team"],
                    "slot": p.get("slot"), "replaced": p.get("replaced"),
                    "voto": (votes.get(p["id"]) or {}).get("voto"),
                    "fantavoto": score_of(p),
                    "bonus": (votes.get(p["id"]) or {}).get("bonus"),
                    "cards": (votes.get(p["id"]) or {}).get("cards")}
                   for p in xi]}
        # The settable lineup is chosen before kickoff, so it is ranked on the PROJECTED
        # total, not the realised one -- otherwise the module choice peeks at the result.
        # `proj` is a SEASON total, so it has to be divided down to a round before the
        # per-round modifier bonus is added: comparing a season-scale sum against a 1.25
        # round-scale bonus makes the four-defender comparison inert, which is the whole
        # decision the modifier is supposed to drive.
        rank_on = (sum(p.get("proj", 0.0) for p in xi) / ROUNDS
                   + (MOD_EV if nd >= 4 else 0.0)) if key == "proj" else total
        if best is None or rank_on > best[0]:
            best = (rank_on, row)
    return best[1] if best else {"module": None, "base": 0.0, "modifier": 0.0,
                                 "total": 0.0, "subs": [], "xi": []}


def build(season: str = SEASON, refresh: bool = False) -> dict:
    board = json.loads(BOARD.read_text())
    table = [tuple(x) for x in board["settings"]["mod_table"]]
    office = board["settings"]["mod_office"]
    by_id = {int(p["id"]): p for p in board["players"]}

    saved = json.loads(TEAM.read_text()) if TEAM.exists() else {}
    if saved.get("roster"):
        team, source = saved, "saved"
    else:
        # An EMPTY saved roster is not a team. The board page mirrors its state on every
        # change, so before the auction it writes `[]` -- and treating that as "saved"
        # would score an empty squad and silently report zero instead of falling back.
        # Before the auction there is no won squad, so the page still has something to
        # show: the board's own optimal 25. Labelled, never passed off as the real team.
        team = {"budget": board["settings"]["budget"],
                "roster": [{"id": int(s["id"]), "paid": int(s.get("cost", 0))}
                           for s in board["squad"]]}
        source = "plan"

    roster = []
    for r in team.get("roster", []):
        p = by_id.get(int(r["id"]))
        if not p:
            continue
        roster.append({"id": int(p["id"]), "nome": p["nome"], "R": p["R"],
                       "team": p["team"], "paid": int(r.get("paid", 0)),
                       "mv_hat": p.get("mv_hat"),
                       # `.get(..., 0.0)` is not enough: a blind player carries an
                       # explicit null, which is present and therefore not defaulted.
                       "proj": float(p.get("season_points") or 0.0)})

    rounds, totals = [], {"settable": 0.0, "hindsight": 0.0}
    per_player: dict[int, dict] = {}
    for rnd in played_rounds(season, refresh=refresh):
        df = fetch_round(season, rnd)
        if df is None:
            continue
        votes = {int(r.pid): {"voto": None if r.voto is None or r.voto != r.voto
                              else float(r.voto),
                              "fantavoto": None if r.fantavoto is None
                              or r.fantavoto != r.fantavoto else float(r.fantavoto),
                              "bonus": float(r.bonus), "cards": float(r.cards)}
                 for r in df.itertuples() if bool(r.played)}
        settable = _pick(roster, votes, table, office, "proj")
        hindsight = _pick(roster, votes, table, office, "actual")
        totals["settable"] += settable["total"]
        totals["hindsight"] += hindsight["total"]
        for e in settable["xi"]:
            d = per_player.setdefault(e["id"], {"nome": e["nome"], "R": e["R"],
                                                "team": e["team"], "starts": 0,
                                                "points": 0.0, "rounds": []})
            d["starts"] += 1
            d["points"] += e["fantavoto"] or 0.0
            d["rounds"].append({"round": rnd, "fantavoto": e["fantavoto"],
                                "voto": e["voto"]})
        rounds.append({"round": rnd, "settable": settable, "hindsight": hindsight})

    for pid, d in per_player.items():
        d["points"] = round(d["points"], 2)
        p = next((x for x in roster if x["id"] == pid), None)
        # Prior per-round level: the board's projected season points spread over 38, on
        # the fantavoto scale (6.0 base + projected bonus above replacement per round).
        prior = 6.0 + (p["proj"] / 38.0 if p else 0.0)
        obs = [r["fantavoto"] for r in d["rounds"] if r["fantavoto"] is not None]
        n = len(obs)
        live = (LEVEL_K * prior + sum(obs)) / (LEVEL_K + n) if n else prior
        d["prior_level"] = round(prior, 2)
        d["live_level"] = round(live, 2)
        d["delta"] = round(live - prior, 2)
        d["n_rounds"] = n
        # Raw-voto level, same shrinkage: the XI advisor's modificatore estimate runs on
        # votos, not fantavotos. Prior = the board's shrunk media voto when known.
        vprior = float(p.get("mv_hat") or 6.0) if p else 6.0
        vobs = [r["voto"] for r in d["rounds"] if r.get("voto") is not None]
        d["live_voto"] = round((LEVEL_K * vprior + sum(vobs))
                               / (LEVEL_K + len(vobs)), 2)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "season": season, "source": source,
        "roster_n": len(roster), "spent": sum(p["paid"] for p in roster),
        "rounds_played": len(rounds),
        "totals": {k: round(v, 2) for k, v in totals.items()},
        "average": {k: round(v / len(rounds), 2) if rounds else 0.0
                    for k, v in totals.items()},
        "rounds": rounds,
        "players": sorted(per_player.values(), key=lambda d: -d["points"]),
    }


# Serie A discipline ladder: a one-match ban lands at the 5th, 10th and 15th
# yellow; a red is an automatic ban (length set by the giudice sportivo — we
# assume 1 match, the modal outcome). Counted from OUR voti parquets, so it is
# the LEAGUE ladder only (Coppa Italia runs its own, which we cannot see).
# One-tap target for the push buttons. No deep link can PRE-FILL a
# formation (probed 2026-09-02: the schiera page is session-gated,
# no public API) — this just lands you on the league, one tap from
# Gestione formazioni.
LEAGUE_URL = "https://leghe.fantacalcio.it/us-fantacalcio-serie-a"
_SCHIERA_BTN = {"inline_keyboard":
                [[{"text": "Schiera su Leghe \u2192", "url": LEAGUE_URL}]]}

BAN_STEPS = frozenset({5, 10, 15})


def _discipline_from_frames(frames: list) -> dict[int, dict]:
    """[(round, df with pid/cards)] -> pid: {yellows, diffidato, banned_next, why}.

    banned_next is true only when the trigger fell in the LATEST played round —
    once the following round is played the flag self-clears, so a served ban
    never haunts later advice.
    """
    yellows: dict[int, int] = {}
    trigger: dict[int, str] = {}
    last_rnd = max((r for r, _ in frames), default=0)
    for rnd, df in sorted(frames, key=lambda x: x[0]):
        for row in df.itertuples():
            if row.pid is None or not row.played:
                continue
            pid = int(row.pid)
            c = float(row.cards or 0.0)
            if c == -0.5:
                yellows[pid] = yellows.get(pid, 0) + 1
                if rnd == last_rnd and yellows[pid] in BAN_STEPS:
                    trigger[pid] = f"squalificato ({yellows[pid]}° giallo)"
            elif c <= -1.0 and rnd == last_rnd:
                trigger[pid] = "squalificato (espulsione)"
    out = {}
    for pid, y in yellows.items():
        out[pid] = {"yellows": y, "diffidato": (y % 5) == 4,
                    "banned_next": pid in trigger, "why": trigger.get(pid)}
    for pid, why in trigger.items():
        out.setdefault(pid, {"yellows": 0, "diffidato": False,
                             "banned_next": True, "why": why})
    return out


def discipline_status(season: str = SEASON) -> dict[int, dict]:
    """League-wide card ledger from the round parquets on disk (no network)."""
    import pandas as pd
    frames = []
    voti_dir = ROOT / "data" / "fantacalcio" / "voti"
    tag = f"round_{season.replace('-', '_')}_"
    for f in sorted(voti_dir.glob(f"{tag}*.parquet")):
        try:
            rnd = int(f.stem.rsplit("_", 1)[1])
            frames.append((rnd, pd.read_parquet(
                f, columns=["pid", "cards", "played"])))
        except (OSError, ValueError):
            continue
    return _discipline_from_frames(frames)


FINAL_WINDOW_H = 6.0


def _push_phase(state: dict, rnd: int | None, kick: float | None,
                now: float) -> str | None:
    """Which push this run owes: 'first' (once, <=48h out), 'final' (once,
    inside the last FINAL_WINDOW_H before kickoff — the job cadence includes
    early-afternoon/evening runs so one lands there), or None."""
    if not rnd or not kick or now >= kick:
        return None
    if state.get("round") != rnd:
        return "first" if kick - now <= 48 * 3600 else None
    if state.get("final_checked") or kick - now > FINAL_WINDOW_H * 3600:
        return None
    return "final"


def _advice_diff(prev: dict, cur: dict) -> str | None:
    """Human-readable delta between the pushed advice and the current one.
    None when nothing a manager acts on has changed."""
    lines = []
    if prev.get("module") != cur.get("module"):
        lines.append(f"Modulo: {prev.get('module')} → {cur.get('module')}")
    p_xi, c_xi = set(prev.get("xi", [])), set(cur.get("xi", []))
    if p_xi != c_xi:
        ins = ", ".join(sorted(c_xi - p_xi))
        outs = ", ".join(sorted(p_xi - c_xi))
        lines.append(f"Dentro: {ins} — Fuori: {outs}")
    elif prev.get("bench", []) != cur.get("bench", []):
        lines.append("Panchina riordinata: "
                     + ", ".join(cur.get("bench", [])))
    if prev.get("vs") is not None and prev.get("vs") != cur.get("vs"):
        for comp, rec in (cur.get("vs") or {}).items():
            if (prev["vs"] or {}).get(comp) == rec:
                continue
            extra = (f" (dentro {', '.join(rec['in'])} — "
                     f"fuori {', '.join(rec['out'])})") if rec.get("in") else ""
            lines.append(f"Contro {rec['opp']} ({comp}): "
                         f"{rec['module']}{extra}")
    return "\n".join(lines) if lines else None


def _vs_block(riv: dict | None) -> tuple[str | None, str | None, dict]:
    """Per-competition opponent forecast + the formation to play AGAINST it.

    Reads the rival matrix (rivals.json payload): the opponent's predicted
    module (observed repertoire when the ledger has screenshots, else
    estimated), my P(win), and the risk-tilted alternative XI when it beats
    the base by ALT_MIN_GAIN against that specific opponent. Collapses
    duplicate lines when both competitions meet the same opponent with the
    same recommendation. The sig dict rides the notify latch so a flipped
    recommendation (e.g. a screenshot taught us the rival's real module)
    fires the last-hour diff push.
    """
    if not riv:
        return None, None, {}
    rows = {r["team"]: r for r in riv.get("rivals", [])}
    me = riv.get("me") or {}
    sig, groups = {}, {}
    for nx in riv.get("next_opponents", []):
        opp = nx.get("opponent")
        r = rows.get(opp) if opp else None
        if not r or r.get("module") is None or r.get("p_win") is None:
            continue
        alt = r.get("alt")
        rec = {"opp": opp,
               "module": (alt or {}).get("module") or me.get("module"),
               "in": (alt or {}).get("in") or [],
               "out": (alt or {}).get("out") or []}
        sig[nx["competition"]] = rec
        key = (opp, rec["module"], tuple(rec["in"]), tuple(rec["out"]))
        g = groups.setdefault(key, {"comps": [], "row": r, "alt": alt})
        g["comps"].append(nx["competition"])
    if not sig:
        return None, None, {}
    txt, tg = [], []
    for (opp, _mod, _ins, _outs), g in groups.items():
        r, alt = g["row"], g["alt"]
        comps = " + ".join(g["comps"])
        src = "visto" if r.get("module_src") == "osservato" else "stima"
        txt.append(f"{comps}: vs {opp} — previsto {r['module']} ({src}, "
                   f"exp {r['total']}) · P(vittoria) {r['p_win']:.0%}")
        tg.append(f"🆚 <b>{comps}</b>: {opp} — previsto <b>{r['module']}</b> "
                  f"({src}, exp {r['total']}) · P(vittoria) "
                  f"<b>{r['p_win']:.0%}</b>")
        if alt:
            ins, outs = ", ".join(alt["in"]), ", ".join(alt["out"])
            txt.append(f"→ contro di loro gioca {alt['module']}: dentro {ins}"
                       f" — fuori {outs} (P(vittoria) {alt['p_win']:.0%})")
            tg.append(f"   → <b>gioca {alt['module']}</b>: dentro {ins} — "
                      f"fuori {outs} (P(vittoria) {alt['p_win']:.0%})")
        else:
            txt.append("→ la formazione base è già la migliore contro di loro")
            tg.append("   → la base sopra è già la migliore contro di loro")
    return "\n".join(txt), "\n".join(tg), sig


def _standings_from_schedule(schedule: dict) -> dict:
    """League tables from the calendar exports' score cells — the only real
    H2H source (the Leghe page 404s anonymously, probed 2026-09-02).

    A fixture counts ONLY when its `score` cell matches N-N, so the unplayed
    "-" shape (and any wrong guess about how Leghe writes played rows) yields
    an empty table, never a wrong one. 3/1/0 points, fantapunti as tiebreak.
    """
    import re as _re
    comps: dict = {}
    for comp, cd in (schedule.get("competitions") or {}).items():
        tables: dict[str, dict] = {}
        played = 0
        for rd in cd.get("rounds", []):
            any_played = False
            for f in rd.get("fixtures", []):
                sc = str(f.get("score") or "")
                if not _re.fullmatch(r"\d+-\d+", sc):
                    continue
                gh, ga = (int(x) for x in sc.split("-"))
                t = tables.setdefault(f.get("girone") or "", {})
                for name in (f["home"], f["away"]):
                    t.setdefault(name, {"team": name, "g": 0, "w": 0, "d": 0,
                                        "l": 0, "gf": 0, "gs": 0, "fp": 0.0,
                                        "pts": 0})
                h, a = t[f["home"]], t[f["away"]]
                h["g"] += 1
                a["g"] += 1
                h["gf"] += gh
                h["gs"] += ga
                a["gf"] += ga
                a["gs"] += gh
                h["fp"] += float(f.get("fp_home") or 0.0)
                a["fp"] += float(f.get("fp_away") or 0.0)
                if gh > ga:
                    h["w"], a["l"] = h["w"] + 1, a["l"] + 1
                    h["pts"] += 3
                elif gh < ga:
                    a["w"], h["l"] = a["w"] + 1, h["l"] + 1
                    a["pts"] += 3
                else:
                    h["d"], a["d"] = h["d"] + 1, a["d"] + 1
                    h["pts"] += 1
                    a["pts"] += 1
                any_played = True
            played += any_played
        comps[comp] = {
            "format": cd.get("format"), "rounds_played": played,
            "tables": {g: sorted(rows.values(),
                                 key=lambda r: (-r["pts"], -r["fp"]))
                       for g, rows in sorted(tables.items())}}
    return comps


def _new_risk_alerts(items: list[dict], state: dict,
                     now_iso: str, hits) -> list[str]:
    """Lines to push for headlines that are (player, category)-new.

    `hits(item)` yields (player, category) pairs (news.risk_hits — title-
    bound). `state` maps "nome|categoria" -> last alerted ISO time; a
    signature re-alerts only after RISK_REALERT_D days. Mutates state.
    """
    lines = []
    for it in items:
        for nome, cat in hits(it):
            key = f"{nome}|{cat}"
            last = state.get(key, "")
            if last and (datetime.fromisoformat(now_iso)
                         - datetime.fromisoformat(last)).days < RISK_REALERT_D:
                continue
            state[key] = now_iso
            lines.append(f"{nome} — {cat}: {it['title'][:90]}")
    return lines


RISK_REALERT_D = 7


def _round_digest(state: dict) -> dict | None:
    """Digest for the newest settled giornata, once (latched in `state`).

    "Cosa avrei fatto": the advised XI's score, the hindsight best, what the
    bench cost, plus the ledger's predicted-vs-actual once reconciled. All
    from artifacts already on disk — no network.
    """
    try:
        data = json.loads(OUT.read_text())
    except (OSError, ValueError):
        return None
    played = int(data.get("rounds_played") or 0)
    if played <= int(state.get("digest_round") or 0) or not data.get("rounds"):
        return None
    rd = data["rounds"][-1]
    rnd = rd["round"]
    st, hd = rd.get("settable") or {}, rd.get("hindsight") or {}
    regret = (hd.get("total") or 0.0) - (st.get("total") or 0.0)
    st_names = {x["nome"] for x in st.get("xi", [])}
    missed = max((x for x in hd.get("xi", []) if x["nome"] not in st_names),
                 key=lambda x: x.get("fantavoto") or 0.0, default=None)
    lines = [f"XI consigliato: {st.get('total')} ({st.get('module')})",
             f"Col senno di poi: {hd.get('total')} ({hd.get('module')})",
             f"Panchina costata: {regret:+.1f}"]
    if missed:
        lines.append(f"Rimpianto: {missed['nome']} "
                     f"({missed.get('fantavoto')}) fuori dall'XI")
    try:
        led = json.loads((ROOT / "data" / "fantacalcio"
                          / "pred_ledger.json").read_text())
        rec = (led.get("rounds") or {}).get(str(rnd))
        if rec and rec.get("actual_total") is not None:
            lines.append(f"Previsto {rec.get('predicted_total')} -> "
                         f"reale {rec.get('actual_total')}")
    except (OSError, ValueError):
        pass
    lines.append("Riesporta i calendari di lega per aggiornare le classifiche")
    state["digest_round"] = played
    text = f"Giornata {rnd} — bilancio\n" + "\n".join(lines)
    return {"title": f"Fantacalcio — giornata {rnd}", "text": text,
            "tg_html": f"<b>📊 Giornata {rnd} — bilancio</b>\n"
                       + "\n".join(f"• {ln}" for ln in lines)}


def _push_xi_advice() -> None:
    """Rebuild the weekly XI advice; push once per giornata, then one
    last-hours re-check that fires ONLY if the advice changed.

    The first push goes out when the round's first kickoff is within 48h; the
    final check runs inside FINAL_WINDOW_H of kickoff and sends a diff
    ("Dentro X — Fuori Y") when probabili/injury news moved the XI, staying
    silent otherwise — the fire-only-on-change rule every notify call site
    owes since the 2026-08-27 Telegram cleanup.
    """
    from datetime import UTC, datetime

    from scripts.fantacalcio.xi_advisor import build_advice

    # Mercato identity sync FIRST (weekly TTL inside): placeholder pids,
    # club moves, listone arrivals. When it changed the board, re-import the
    # rosters so every squad points at the corrected pids. Best-effort.
    try:
        from scripts.fantacalcio.import_rosters import import_rosters, sync_mercato
        if sync_mercato():
            import_rosters()
    except Exception as e:
        print(f"mercato sync failed (advice unaffected): {e}")

    # Freshen the news accumulator on the same twice-daily cadence. Best-effort:
    # a feed outage must never block the XI advice. Probabili needs no explicit
    # refresh here -- build_advice fetches it through a 6h-TTL cache and the
    # tracker runs are 12h apart, so every run gets a fresh page anyway.
    try:
        from scripts.fantacalcio.news import fetch_news, roster_for_news
        fetch_news(roster_for_news())
    except Exception as e:
        print(f"news refresh failed (advice unaffected): {e}")

    adv = build_advice()
    (ROOT / "data" / "fantacalcio" / "xi_advice.json").write_text(
        json.dumps(adv, indent=1, ensure_ascii=False))
    # Freeze the forecast pre-kickoff and reconcile any settled round — the
    # predicted-vs-actual loop that will refit the p_play/exp constants.
    try:
        from scripts.fantacalcio.pred_ledger import reconcile, snapshot
        st = snapshot(adv)
        rec = reconcile()
        print(f"pred ledger: snapshot={st}"
              + (f" reconciled={rec}" if rec else ""))
    except Exception as e:
        print(f"pred ledger update failed (advice unaffected): {e}")
    # Rival matrix: expected score + win prob vs every league team, so the two
    # competitions (Coppa Del Nonno / Hunger Games) are both covered whoever
    # the weekend's opponent is.
    riv = None
    try:
        from scripts.fantacalcio.xi_advisor import build_rivals
        riv = build_rivals(adv)
        (ROOT / "data" / "fantacalcio" / "rivals.json").write_text(
            json.dumps(riv, indent=1, ensure_ascii=False))
    except Exception as e:
        print(f"rivals build failed (advice unaffected): {e}")
    # Daily press-pulse update: scores today's headlines per club and labels
    # parked ones once results land — the layer that "learns each time".
    try:
        from scripts.fantacalcio.team_pulse import refresh as pulse_refresh
        st = pulse_refresh(next_round=adv.get("round"))
        print(f"team pulse: labeled {st.get('labeled_this_run', 0)}, "
              f"pending {len(st.get('pending', []))}")
    except Exception as e:
        print(f"team pulse failed (advice unaffected): {e}")
    # Free-agent radar: the unowned half of the listone, re-ranked every run.
    try:
        from scripts.fantacalcio.xi_advisor import build_svincolati
        sv = build_svincolati()
        (ROOT / "data" / "fantacalcio" / "svincolati.json").write_text(
            json.dumps(sv, indent=1, ensure_ascii=False))
    except Exception as e:
        print(f"svincolati scan failed (advice unaffected): {e}")
    try:
        schedule = json.loads((ROOT / "data" / "fantacalcio"
                               / "league_schedule.json").read_text())
        (ROOT / "data" / "fantacalcio" / "league_standings.json").write_text(
            json.dumps({"generated_at": datetime.now(UTC).isoformat(),
                        "competitions": _standings_from_schedule(schedule)},
                       indent=1, ensure_ascii=False))
    except Exception as e:
        print(f"standings build failed (advice unaffected): {e}")
    try:
        from scripts.fantacalcio.trades import build_trades
        tr = build_trades()
        print(f"trade scan: {len(tr['windows'])} windows")
    except Exception as e:
        print(f"trade scan failed (advice unaffected): {e}")
    state_path = ROOT / "data" / "fantacalcio" / "xi_notify_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    # Roster risk alerts: headlines already matched to my players that hit
    # the risk lexicon, pushed once per (player, category) per week.
    try:
        from scripts.fantacalcio.news import risk_hits
        items = json.loads((ROOT / "data" / "fantacalcio"
                            / "news.json").read_text()).get("items", [])
        rstate = state.setdefault("risk_alerts", {})
        lines = _new_risk_alerts(items, rstate,
                                 datetime.now(UTC).isoformat(), risk_hits)
        if lines:
            from scripts.pipeline.notify import notify
            notify("Notizie rosa — rischi:\n" + "\n".join(lines),
                   title="Fantacalcio — rischi rosa", level="warning",
                   category="alert",
                   tg_html="<b>⚠️ Rischi rosa</b>\n"
                           + "\n".join(f"• {ln}" for ln in lines))
            state_path.write_text(json.dumps(state, indent=1,
                                             ensure_ascii=False))
    except Exception as e:
        print(f"risk alerts failed (advice unaffected): {e}")
    # Post-round digest: once, when a new giornata lands in the tracker.
    try:
        digest = _round_digest(state)
        if digest:
            from scripts.pipeline.notify import notify
            notify(digest["text"], title=digest["title"], level="info",
                   category="alert", tg_html=digest["tg_html"])
            state_path.write_text(json.dumps(state, indent=1,
                                             ensure_ascii=False))
    except Exception as e:
        print(f"round digest failed (advice unaffected): {e}")
    rnd, kick = adv.get("round"), adv.get("first_kickoff")
    if not adv.get("xi"):
        return
    phase = _push_phase(state, rnd, kick, datetime.now(UTC).timestamp())
    if phase is None:
        return
    vs_txt, vs_tg, vs_sig = _vs_block(riv)
    cur = {"module": adv.get("module"),
           "xi": sorted(x["nome"] for x in adv["xi"]),
           "bench": [x["nome"] for x in adv["bench"]],
           "vs": vs_sig}

    if phase == "final":
        diff = _advice_diff(state.get("advice", {}), cur)
        state.update({"final_checked": True, "advice": cur})
        if diff:
            try:
                from scripts.pipeline.notify import notify
                notify(f"Giornata {rnd} — cambi dell'ultima ora\n{diff}",
                       title="Fantacalcio XI — aggiornamento", level="warning",
                       category="system",
                       tg_html=(f"<b>🔁 Giornata {rnd} — cambi dell'ultima ora"
                                f"</b>\n{diff}"
                                + (f"\n\n{vs_tg}" if vs_tg else "")),
                       tg_reply_markup=_SCHIERA_BTN)
            except Exception as e:
                print(f"XI final-check notify failed: {e}")
        state_path.write_text(json.dumps(state))
        return

    role_order = {"P": 0, "D": 1, "C": 2, "A": 3}
    xi = sorted(adv["xi"], key=lambda x: role_order[x["R"]])
    lines = [f"{x['R']} {x['nome']} ({x['team']} "
             f"{'vs' if x['home'] else '@'} {x['opp']})" for x in xi]
    bench = [f"{x['R']} {x['nome']}" for x in adv["bench"]]
    inj = [f"{x['nome']}: {x.get('inj') or x.get('why')}"
           for x in adv["unavailable"]]
    diffid = [x["nome"] for x in adv["xi"] + adv["bench"] if x.get("diffidato")]
    msg = (f"Giornata {rnd} — modulo {adv['module']} "
           f"(exp {adv['total']}, mod +{adv['modifier']})\n"
           + (f"{vs_txt}\n\n" if vs_txt else "")
           + "\n".join(lines)
           + "\nPanchina (in quest'ordine): " + ", ".join(bench)
           + (("\nOut: " + "; ".join(inj)) if inj else "")
           + (("\nDiffidati (4 gialli): " + ", ".join(diffid)) if diffid else ""))
    tg = (f"<b>⚽ Formazione giornata {rnd}</b> — <b>{adv['module']}</b> "
          f"(exp {adv['total']}, mod +{adv['modifier']})\n"
          + (f"{vs_tg}\n\n" if vs_tg else "")
          + "\n".join(lines)
          + "\n\n<b>Panchina</b> (ordine sub): " + ", ".join(bench)
          + (("\n<b>Out:</b> " + "; ".join(inj)) if inj else "")
          + (("\n<b>⚠ Diffidati:</b> " + ", ".join(diffid)) if diffid else ""))
    try:
        from scripts.pipeline.notify import notify
        tg += ("\n\n📸 Un'ora prima: se vedi la formazione avversaria su "
               "Leghe, mandami uno screenshot qui — la leggo e ti confermo "
               "modulo e XI.")
        notify(msg, title="Fantacalcio XI", level="info",
               category="system", tg_html=tg, tg_reply_markup=_SCHIERA_BTN)
        state_path.write_text(json.dumps(
            {"round": rnd, "sent_at": datetime.now(UTC).isoformat(),
             "final_checked": False, "advice": cur}))
    except Exception as e:  # advice on disk is the deliverable; push is best-effort
        print(f"XI notify failed (advice still written): {e}")


def main() -> None:
    import sys
    data = build(refresh="--refresh" in sys.argv)
    OUT.write_text(json.dumps(data, indent=1))
    print(f"{OUT.relative_to(ROOT)}  source={data['source']} "
          f"rounds={data['rounds_played']} "
          f"settable={data['totals']['settable']} "
          f"hindsight={data['totals']['hindsight']}")
    try:
        _push_xi_advice()
    except Exception as e:
        print(f"XI advice refresh failed (tracker output unaffected): {e}")


if __name__ == "__main__":
    main()
