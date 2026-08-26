"""Weekly XI advisor: who to field next giornata, from live levels and the fixture.

Expected fantavoto for the round = the player's live level (tracker) plus two fixture
terms, both MEASURED on 9,815 joined player-rounds of 2025-26 rather than assumed:

  * home advantage, per role:      A +0.10  C +0.13  D +0.16  P +0.15  fantavoto
  * Elo edge, per 100 points:      A +0.085 C +0.039 D +0.034 P +0.104

That expectation is CONDITIONAL on playing, so the slot ranking multiplies it by
titolarita -- p(gets a voto this round). Prior = proj_min / (38 * E[min|appearance]
for the role), because a starter subbed at 75' still collects a voto: dividing by a
full 90 would call Orsolini a 55% starter. E[min|appearance] is MEASURED on 11,928
appearance rows of 2025-26 (G 88.7, D 68.4, M 63.4, F 50.8), capped at 0.95, and
shrunk toward the observed appearance rate as rounds accumulate (same K=10 the level
uses). Without any of this the first run fielded Milan's 134-minute backup keeper over
Svilar because their conditional levels tied; a 4% starter cannot outrank an 88% one.
The pooled-by-role denominator overstates p for iron men and understates it for cameo
subs -- tolerable in a prior that observed appearances replace within ~10 rounds.

There is deliberately NO form term -- lag-1 autocorrelation of a player's residual
fantavoto is -0.05 (slightly mean-reverting), so "he was great last week" would make the
advice worse. The raw voto barely moves with the fixture (+0.02/100 Elo), so the
modificatore estimate runs on the live voto level with only that small slope.

The module search maximises the sum of p_play * expected fantavoto PLUS the expected
modificatore (computed on conditional votos -- an approximation that ignores no-show
office votes). The total therefore excludes auto-sub recovery: it is the floor of what
the fielded XI earns, not the ceiling after substitutions.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config.team_names import normalize_team as NT
from scripts.fantacalcio.tracker import MODULES, _modifier

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "data" / "fantacalcio" / "auction_board.json"
TRACKER = ROOT / "data" / "fantacalcio" / "tracker.json"
# Season derived, never written down -- a "fixtures_2026_2027" literal here is next
# August's silent breakage (third documented instance of that trap in this repo).
def _fixtures_path() -> Path:
    from scripts.utils.match_timing import _sofascore_fixture_files
    return next(p for p, lg in _sofascore_fixture_files() if lg == "serie_a")
FEATURES = ROOT / "data" / "features" / "features_serie_a.parquet"

# Measured 2026-08-26 on 2025-26 (see module docstring). Do not tune by hand.
HOME_ADJ = {"A": 0.103, "C": 0.132, "D": 0.159, "P": 0.146}
ELO_SLOPE = {"A": 0.085, "C": 0.039, "D": 0.034, "P": 0.104}   # per 100 Elo
VOTO_ELO_SLOPE = 0.02                                          # per 100 Elo
PLAY_K = 10.0        # rounds of prior weight on titolarita, mirrors LEVEL_K
# E[minutes | appearance] by role, measured on 11,928 rows of 2025-26 player_match_stats
# (positions G/D/M/F mapped to P/D/C/A). Denominator for p(voto) from projected minutes.
APP_MIN = {"P": 88.7, "D": 68.4, "C": 63.4, "A": 50.8}


def _next_fixtures() -> tuple[dict, int | None]:
    """Earliest future fixture per canonical team name, and the round it belongs to."""
    try:
        raw = json.loads(_fixtures_path().read_text())
    except (OSError, ValueError, StopIteration):
        return {}, None
    now = datetime.now(UTC).timestamp()
    fut = [x for x in raw if (x.get("startTimestamp") or 0) > now]
    if not fut:
        return {}, None
    rnd = min((x.get("roundInfo") or {}).get("round", 99) for x in fut)
    out: dict = {}
    for x in fut:
        if (x.get("roundInfo") or {}).get("round") != rnd:
            continue
        h = NT((x.get("homeTeam") or {}).get("name", "")) or ""
        a = NT((x.get("awayTeam") or {}).get("name", "")) or ""
        ts = x.get("startTimestamp")
        if h:
            out[h] = {"opp": a, "home": 1, "ts": ts}
        if a:
            out[a] = {"opp": h, "home": 0, "ts": ts}
    return out, rnd


def _current_elo() -> dict:
    f = pd.read_parquet(FEATURES, columns=["match_date", "home_team", "away_team",
                                           "home_elo", "away_elo"])
    f = f[f.home_elo.notna()].sort_values("match_date")
    elo: dict = {}
    for r in f.itertuples():
        elo[r.home_team] = float(r.home_elo)
        elo[r.away_team] = float(r.away_elo)
    return elo


def _out_ids(board_players: list, fixtures: dict) -> dict:
    """id -> injury note, for players out for THIS round.

    A dated return before kickoff clears the flag; anything else currently out sits out.
    Four days before a round, "out, no return date" means he misses it -- for a lineup
    call that is the conservative and correct read, unlike the season projection where
    inventing a duration would compound.
    """
    out = {}
    for p in board_players:
        note = p.get("inj_note")
        if not note:
            continue
        fx = fixtures.get(p["team"])
        kick = datetime.fromtimestamp(fx["ts"], tz=UTC) if fx and fx.get("ts") else None
        ret = None
        if "out until ~" in note:
            try:
                ret = datetime.fromisoformat(note.split("out until ~")[1][:10]) \
                    .replace(tzinfo=UTC)
            except ValueError:
                ret = None
        if ret is not None and kick is not None and ret <= kick:
            continue                      # back in time for this one
        out[int(p["id"])] = note
    return out


def advise(roster: list, fixtures: dict, elo: dict, out: dict) -> dict:
    """Pure core: pick module + XI by p_play * expected fantavoto. Testable without disk.

    Roster rows without a p_play field count as certain starters (p=1.0) so the
    conditional ranking is unchanged for callers that never model titolarita.
    """
    cand = []
    for p in roster:
        fx = fixtures.get(p["team"])
        note = out.get(p["id"])
        if fx is None:
            cand.append({**p, "exp": None, "exp_slot": None,
                         "why": "no fixture this round", "inj": note})
            continue
        d_elo = (elo.get(p["team"], 1450.0) - elo.get(fx["opp"], 1450.0)) / 100.0
        adj = HOME_ADJ[p["R"]] * fx["home"] + ELO_SLOPE[p["R"]] * d_elo
        pp = float(p.get("p_play", 1.0))
        exp = None if note else p["level"] + adj
        cand.append({**p,
                     "exp": None if exp is None else round(exp, 2),
                     "exp_slot": None if exp is None else round(pp * exp, 2),
                     "p_play": round(pp, 2),
                     "exp_voto": round(p["voto"] + VOTO_ELO_SLOPE * d_elo, 2),
                     "fix_adj": round(adj, 2), "opp": fx["opp"], "home": fx["home"],
                     "inj": note})
    by_role = {r: sorted([c for c in cand if c["R"] == r and c["exp_slot"] is not None],
                         key=lambda x: -x["exp_slot"]) for r in "PDCA"}
    best = None
    for nd, nc, na in MODULES:
        need = {"P": 1, "D": nd, "C": nc, "A": na}
        if any(len(by_role[r]) < n for r, n in need.items()):
            continue
        xi = [x for r, n in need.items() for x in by_role[r][:n]]
        gk_v = next((x["exp_voto"] for x in xi if x["R"] == "P"), None)
        d_v = [x["exp_voto"] for x in xi if x["R"] == "D"]
        mod = _modifier(gk_v, d_v, [(6.0, 1), (6.5, 3), (7.0, 6)],
                        [5.0, 4.5, 4.5]) if nd >= 4 else 0.0
        total = sum(x["exp_slot"] for x in xi) + mod
        if best is None or total > best["total"]:
            bench = sorted([c for c in cand if c not in xi and c["exp_slot"] is not None],
                           key=lambda x: -x["exp_slot"])
            best = {"module": f"{nd}-{nc}-{na}", "total": round(total, 2),
                    "modifier": round(mod, 2), "xi": xi, "bench": bench,
                    "unavailable": [c for c in cand if c["exp_slot"] is None]}
    return best or {"module": None, "total": 0.0, "modifier": 0.0, "xi": [],
                    "bench": [], "unavailable": cand}


def build_advice() -> dict:
    board = json.loads(BOARD.read_text())
    by_id = {int(p["id"]): p for p in board["players"]}
    fixtures, rnd = _next_fixtures()
    elo = _current_elo()
    out = _out_ids(board["players"], fixtures)

    team_file = ROOT / "data" / "fantacalcio" / "my_team.json"
    saved = {}
    try:
        saved = json.loads(team_file.read_text())
    except (OSError, ValueError):
        pass
    ids = [int(r["id"]) for r in saved.get("roster", [])] \
        or [int(s["id"]) for s in board["squad"]]
    source = "saved" if saved.get("roster") else "plan"

    # Live levels + appearance counts from the tracker artifact; board priors otherwise.
    # Same formulas the tracker uses -- one definition of "level".
    live: dict = {}
    rounds_elapsed = 0
    try:
        t = json.loads(TRACKER.read_text())
        rounds_elapsed = int(t.get("rounds_played") or 0)
        for p in t.get("players", []):
            live[p["nome"]] = p
    except (OSError, ValueError):
        pass

    roster_src = []
    for pid in ids:
        p = by_id.get(pid)
        if not p:
            continue
        lv = live.get(p["nome"], {})
        prior = 6.0 + float(p.get("season_points") or 0.0) / 38.0
        pp_prior = min(max(
            float(p.get("proj_min") or 0.0) / (38.0 * APP_MIN[p["R"]]), 0.02), 0.95)
        n_seen = int(lv.get("n_rounds", 0))
        p_play = (PLAY_K * pp_prior + n_seen) / (PLAY_K + rounds_elapsed)
        roster_src.append({"id": pid, "nome": p["nome"], "R": p["R"], "team": p["team"],
                           "level": float(lv.get("live_level", prior)),
                           "voto": float(lv.get("live_voto")
                                         or p.get("mv_hat") or 6.0),
                           "p_play": min(max(p_play, 0.02), 0.95),
                           "n_rounds": n_seen})

    adv = advise(roster_src, fixtures, elo, out)
    kicks = [fixtures[t_]["ts"] for t_ in fixtures if fixtures[t_].get("ts")]
    return {"generated_at": datetime.now(UTC).isoformat(),
            "round": rnd, "source": source,
            "first_kickoff": min(kicks) if kicks else None,
            **adv}


def main() -> None:
    out = build_advice()
    print(json.dumps({k: out[k] for k in ("round", "module", "total", "modifier")},
                     indent=1))
    for x in out["xi"]:
        print(f"  {x['R']} {x['nome']:16s} exp={x['exp']:5.2f} st={x['p_play']:.0%} "
              f"slot={x['exp_slot']:5.2f} "
              f"({'home' if x['home'] else 'away'} vs {x['opp']}, adj {x['fix_adj']:+.2f})")


if __name__ == "__main__":
    main()
