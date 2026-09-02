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
from datetime import datetime, timezone
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
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season, "source": source,
        "roster_n": len(roster), "spent": sum(p["paid"] for p in roster),
        "rounds_played": len(rounds),
        "totals": {k: round(v, 2) for k, v in totals.items()},
        "average": {k: round(v / len(rounds), 2) if rounds else 0.0
                    for k, v in totals.items()},
        "rounds": rounds,
        "players": sorted(per_player.values(), key=lambda d: -d["points"]),
    }


def _push_xi_advice() -> None:
    """Rebuild the weekly XI advice and push it ONCE per giornata.

    Runs on the twice-daily tracker job. Fires only when the round's first
    kickoff is within 48h AND this round has not been announced yet (state in
    xi_notify_state.json) — the fire-only-on-change rule every notify call
    site owes since the 2026-08-27 Telegram cleanup.
    """
    from datetime import UTC, datetime

    from scripts.fantacalcio.xi_advisor import build_advice

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
    state_path = ROOT / "data" / "fantacalcio" / "xi_notify_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    rnd, kick = adv.get("round"), adv.get("first_kickoff")
    if not rnd or not adv.get("xi") or state.get("round") == rnd:
        return
    if not kick or kick - datetime.now(UTC).timestamp() > 48 * 3600:
        return

    role_order = {"P": 0, "D": 1, "C": 2, "A": 3}
    xi = sorted(adv["xi"], key=lambda x: role_order[x["R"]])
    lines = [f"{x['R']} {x['nome']} ({x['team']} "
             f"{'vs' if x['home'] else '@'} {x['opp']})" for x in xi]
    bench = [f"{x['R']} {x['nome']}" for x in adv["bench"]]
    inj = [f"{x['nome']}: {x.get('inj') or x.get('why')}"
           for x in adv["unavailable"]]
    msg = (f"Giornata {rnd} — modulo {adv['module']} "
           f"(exp {adv['total']}, mod +{adv['modifier']})\n"
           + "\n".join(lines)
           + "\nPanchina (in quest'ordine): " + ", ".join(bench)
           + (("\nOut: " + "; ".join(inj)) if inj else ""))
    tg = (f"<b>⚽ Formazione giornata {rnd}</b> — <b>{adv['module']}</b> "
          f"(exp {adv['total']}, mod +{adv['modifier']})\n"
          + "\n".join(lines)
          + "\n\n<b>Panchina</b> (ordine sub): " + ", ".join(bench)
          + (("\n<b>Out:</b> " + "; ".join(inj)) if inj else ""))
    try:
        from scripts.pipeline.notify import notify
        notify(msg, title="Fantacalcio XI", level="info",
               category="system", tg_html=tg)
        state_path.write_text(json.dumps({"round": rnd,
                                          "sent_at": datetime.now(UTC).isoformat()}))
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
