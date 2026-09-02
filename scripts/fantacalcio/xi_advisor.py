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
from scripts.fantacalcio.probabili import fetch_probabili, p_play_override, status_by_pid
from scripts.fantacalcio.tracker import LEVEL_K, MODULES, _modifier

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "data" / "fantacalcio" / "auction_board.json"
VOTI_DIR = ROOT / "data" / "fantacalcio" / "voti"
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
# Per-player fantavoto sd is a PERSISTENT trait (split-half r=0.467, n=419
# players of 2025-26), so an observed sd earns weight against the role mean at
# K = n(1-r)/r ~ 17. Role means measured on the same 11,373 played rows.
SD_ROLE = {"A": 1.69, "C": 1.19, "D": 0.93, "P": 1.55}
SD_K = 17.0


_HISTORY: dict | None = None


def _history() -> dict:
    """One read of every round parquet: per-pid voti history, both seasons.

    Returns {"sd": pid->shrunk fantavoto sd, "live": pid->{n, fv, voto} for the
    CURRENT season, "rounds_elapsed": current-season rounds on disk}. The round
    parquets are league-wide (every Serie A player), which is what makes rival
    squads scoreable with the same formulas as mine.
    """
    global _HISTORY
    if _HISTORY is not None:
        return _HISTORY
    import pandas as pd
    cur_tag = "round_2026_27"
    frames, cur_frames = [], []
    for f in sorted(VOTI_DIR.glob("round_*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["pid", "role", "voto",
                                            "fantavoto", "played"])
        except (OSError, ValueError):
            continue
        d = d[d.played & d.fantavoto.notna() & d.pid.notna()]
        frames.append(d)
        if f.name.startswith(cur_tag):
            cur_frames.append(d)
    sd: dict[int, float] = {}
    live: dict[int, dict] = {}
    if frames:
        alld = pd.concat(frames)
        g = alld.groupby(["pid", "role"]).fantavoto.agg(["count", "std"]).reset_index()
        for r in g.itertuples():
            role = str(r.role).upper()
            prior = SD_ROLE.get(role, 1.3)
            obs = float(r.std) if pd.notna(r.std) else prior
            n = max(int(r.count) - 1, 0)   # sd has n-1 df; one game says nothing
            sd[int(r.pid)] = round((SD_K * prior + n * obs) / (SD_K + n), 3)
    if cur_frames:
        curd = pd.concat(cur_frames)
        for pid, grp in curd.groupby("pid"):
            live[int(pid)] = {"n": len(grp), "fv": float(grp.fantavoto.mean()),
                              "voto": float(grp.voto.mean())}
    _HISTORY = {"sd": sd, "live": live,
                "rounds_elapsed": len(list(VOTI_DIR.glob(f"{cur_tag}_*.parquet")))}
    return _HISTORY


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


# League bench is 9 ORDERED slots -- 1 keeper, 3 defenders, 3 midfielders,
# 2 strikers, in that order (league rule, Nicola 2026-09-02). Auto-subs promote
# within the role group top-down, so slot order inside a role = exp_slot order.
BENCH_SLOTS = (("P", 1), ("D", 3), ("C", 3), ("A", 2))


def _bench_split(cand: list, xi: list) -> tuple[list, list]:
    """(bench, tribuna): the 9 league slots filled per role by exp_slot, then
    everyone else who could still play. Unavailable players belong to neither."""
    rest = [c for c in cand if c not in xi and c["exp_slot"] is not None]
    bench = []
    for role, n in BENCH_SLOTS:
        pool = sorted([c for c in rest if c["R"] == role],
                      key=lambda x: -x["exp_slot"])[:n]
        bench.extend(pool)
        rest = [c for c in rest if c not in pool]
    return bench, rest


def advise(roster: list, fixtures: dict, elo: dict, out: dict,
           risk_lambda: float = 0.0) -> dict:
    """Pure core: pick module + XI by p_play * expected fantavoto. Testable without disk.

    Roster rows without a p_play field count as certain starters (p=1.0) so the
    conditional ranking is unchanged for callers that never model titolarita.

    risk_lambda tilts SELECTION (never the reported numbers) by lambda * p_play
    * sd: positive prefers volatile players — the underdog play, since only
    variance can carry a weaker XI past a stronger opponent — negative prefers
    steady ones. Bench order stays pure exp_slot: entry order maximizes the
    expected recovery, whoever the opponent is.
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
    def key(c: dict) -> float:
        return c["exp_slot"] + risk_lambda * c["p_play"] * (c.get("sd") or 0.0)
    by_role = {r: sorted([c for c in cand if c["R"] == r and c["exp_slot"] is not None],
                         key=lambda x: -key(x)) for r in "PDCA"}
    best, best_obj = None, None
    for nd, nc, na in MODULES:
        need = {"P": 1, "D": nd, "C": nc, "A": na}
        if any(len(by_role[r]) < n for r, n in need.items()):
            continue
        xi = [x for r, n in need.items() for x in by_role[r][:n]]
        gk_v = next((x["exp_voto"] for x in xi if x["R"] == "P"), None)
        d_v = [x["exp_voto"] for x in xi if x["R"] == "D"]
        # League table verified from the Opzioni di Lega screen 2026-09-02:
        # tiers 6.0->1, 6.5->3, 7.0->6, 7.5->9 (caps at 9; .25 steps repeat).
        mod = _modifier(gk_v, d_v, [(6.0, 1), (6.5, 3), (7.0, 6), (7.5, 9)],
                        [5.0, 4.5, 4.5]) if nd >= 4 else 0.0
        total = sum(x["exp_slot"] for x in xi) + mod
        obj = sum(key(x) for x in xi) + mod
        if best_obj is None or obj > best_obj:
            bench, tribuna = _bench_split(cand, xi)
            best_obj = obj
            best = {"module": f"{nd}-{nc}-{na}", "total": round(total, 2),
                    "xi_sd": round(sum(x.get("sd", 0.0) ** 2 for x in xi) ** 0.5, 2),
                    "modifier": round(mod, 2), "xi": xi, "bench": bench,
                    "tribuna": tribuna,
                    "unavailable": [c for c in cand if c["exp_slot"] is None]}
    return best or {"module": None, "total": 0.0, "modifier": 0.0, "xi": [],
                    "bench": [], "tribuna": [], "unavailable": cand}


def _my_roster(by_id: dict, prob_by_pid: dict,
               fallback_ids: list[int]) -> tuple[list, str]:
    """My roster rows, enriched exactly as build_advice always did (+ sd)."""
    team_file = ROOT / "data" / "fantacalcio" / "my_team.json"
    saved = {}
    try:
        saved = json.loads(team_file.read_text())
    except (OSError, ValueError):
        pass
    ids = [int(r["id"]) for r in saved.get("roster", [])] or fallback_ids
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
    sds = _history()["sd"]

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
        p_play = min(max(p_play, 0.02), 0.95)
        p_play, pp_src = p_play_override(pid, p_play, prob_by_pid)
        roster_src.append({"id": pid, "nome": p["nome"], "R": p["R"], "team": p["team"],
                           "level": float(lv.get("live_level", prior)),
                           "voto": float(lv.get("live_voto")
                                         or p.get("mv_hat") or 6.0),
                           "p_play": p_play, "p_play_src": pp_src,
                           "sd": sds.get(pid) or SD_ROLE[p["R"]],
                           "n_rounds": n_seen})
    return roster_src, source


def build_advice() -> dict:
    board = json.loads(BOARD.read_text())
    by_id = {int(p["id"]): p for p in board["players"]}
    fixtures, rnd = _next_fixtures()
    elo = _current_elo()
    out = _out_ids(board["players"], fixtures)

    # Probable lineups beat the appearance-rate model when the page lists the
    # player (fetch is cached 6h and falls back to the last good cache on failure).
    prob_by_pid = status_by_pid(fetch_probabili())
    roster_src, source = _my_roster(
        by_id, prob_by_pid, [int(sq["id"]) for sq in board["squad"]])

    adv = advise(roster_src, fixtures, elo, out)
    kicks = [fixtures[t_]["ts"] for t_ in fixtures if fixtures[t_].get("ts")]
    return {"generated_at": datetime.now(UTC).isoformat(),
            "round": rnd, "source": source,
            "first_kickoff": min(kicks) if kicks else None,
            **adv}


def _p_win(mu_a: float, sd_a: float, mu_b: float, sd_b: float) -> float:
    """P(A outscores B) under a normal approximation of the two XI totals.

    Team sd = sqrt(sum of per-player fantavoto variances), conditioned on the
    XI actually playing; modifier variance and auto-sub recovery are ignored.
    Good enough for guidance, not for staking."""
    from math import erf, sqrt
    spread = sqrt(sd_a ** 2 + sd_b ** 2) or 1.0
    return 0.5 * (1.0 + erf((mu_a - mu_b) / spread / sqrt(2.0)))


def _rival_roster(entry: dict, by_id: dict, hist: dict, prob_by_pid: dict) -> list:
    """A rival squad enriched with the SAME formulas as mine — levels from the
    league-wide voti parquets (keyed by pid), p_play prior from projected
    minutes with the probabili override, sd from history."""
    rounds_elapsed = hist["rounds_elapsed"]
    rows = []
    for r in entry.get("roster", []):
        pid = int(r["id"])
        p = by_id.get(pid)
        if not p:
            continue
        lv = hist["live"].get(pid)
        prior = 6.0 + float(p.get("season_points") or 0.0) / 38.0
        mv_prior = float(p.get("mv_hat") or 6.0)
        n_seen = int(lv["n"]) if lv else 0
        level = (LEVEL_K * prior + n_seen * lv["fv"]) / (LEVEL_K + n_seen) \
            if lv else prior
        voto = (LEVEL_K * mv_prior + n_seen * lv["voto"]) / (LEVEL_K + n_seen) \
            if lv else mv_prior
        pp_prior = min(max(
            float(p.get("proj_min") or 0.0) / (38.0 * APP_MIN[p["R"]]), 0.02), 0.95)
        p_play = (PLAY_K * pp_prior + n_seen) / (PLAY_K + rounds_elapsed) \
            if rounds_elapsed else pp_prior
        p_play = min(max(p_play, 0.02), 0.95)
        p_play, pp_src = p_play_override(pid, p_play, prob_by_pid)
        rows.append({"id": pid, "nome": p["nome"], "R": p["R"], "team": p["team"],
                     "level": level, "voto": voto,
                     "p_play": p_play, "p_play_src": pp_src,
                     "sd": hist["sd"].get(pid) or SD_ROLE[p["R"]],
                     "n_rounds": n_seen})
    return rows


# Selection tilts tried against each opponent. Positive = chase variance
# (underdog), negative = buy stability (favourite). An alternative XI is
# reported only when it moves P(win) by at least ALT_MIN_GAIN.
RISK_LAMBDAS = (-0.3, -0.15, 0.15, 0.3)
ALT_MIN_GAIN = 0.01


def build_rivals(adv: dict | None = None) -> dict:
    """Expected score + my win probability vs EVERY league team this round.

    Calendar-free by design: with two competitions (Coppa Del Nonno groups,
    Hunger Games) the opponent differs per weekend — a full rival matrix
    covers both without knowing either schedule. Rival totals include their
    defense modifier: they run through the same module search mine does."""
    board = json.loads(BOARD.read_text())
    by_id = {int(p["id"]): p for p in board["players"]}
    fixtures, rnd = _next_fixtures()
    elo = _current_elo()
    out = _out_ids(board["players"], fixtures)
    hist = _history()
    prob_by_pid = status_by_pid(fetch_probabili())

    league = json.loads(
        (ROOT / "data" / "fantacalcio" / "league_rosters.json").read_text())
    my_name = league.get("my_team")

    my_roster, _src = _my_roster(
        by_id, prob_by_pid, [int(sq["id"]) for sq in board["squad"]])
    base = advise(my_roster, fixtures, elo, out)
    base_names = {x["nome"] for x in base["xi"]}

    # Tilted candidates are opponent-independent; compute the 4 once, price
    # them per opponent.
    tilted = []
    for lam in RISK_LAMBDAS:
        alt = advise(my_roster, fixtures, elo, out, risk_lambda=lam)
        if {x["nome"] for x in alt["xi"]} != base_names:
            tilted.append((lam, alt))

    rivals = []
    for tname, entry in league.get("teams", {}).items():
        if tname == my_name:
            continue
        rows = _rival_roster(entry, by_id, hist, prob_by_pid)
        radv = advise(rows, fixtures, elo, out)
        n_missing = (len(entry.get("unmatched", []))
                     + len(entry.get("roster", [])) - len(rows))
        if not radv.get("xi"):
            rivals.append({"team": tname, "module": None, "total": None,
                           "sd": None, "p_win": None, "n_missing": n_missing,
                           "alt": None})
            continue
        p0 = _p_win(base["total"], base["xi_sd"], radv["total"], radv["xi_sd"])
        best_alt = None
        for lam, alt in tilted:
            pa = _p_win(alt["total"], alt["xi_sd"], radv["total"], radv["xi_sd"])
            if pa >= p0 + ALT_MIN_GAIN and (best_alt is None
                                            or pa > best_alt["p_win"]):
                alt_names = {x["nome"] for x in alt["xi"]}
                best_alt = {"lambda": lam, "module": alt["module"],
                            "total": alt["total"], "sd": alt["xi_sd"],
                            "p_win": round(pa, 3),
                            "in": sorted(alt_names - base_names),
                            "out": sorted(base_names - alt_names)}
        rivals.append({"team": tname, "module": radv["module"],
                       "total": radv["total"], "sd": radv["xi_sd"],
                       "p_win": round(p0, 3), "n_missing": n_missing,
                       "alt": best_alt})
    rivals.sort(key=lambda r: (r["p_win"] is None, r["p_win"]))
    return {"generated_at": datetime.now(UTC).isoformat(), "round": rnd,
            "me": {"team": my_name, "module": base["module"],
                   "total": base["total"], "sd": base["xi_sd"]},
            "rivals": rivals}


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
