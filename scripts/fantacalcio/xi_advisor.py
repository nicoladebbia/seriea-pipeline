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
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from config.team_names import normalize_team as NT
from scripts.fantacalcio.probabili import (
    BALLOT_CLAMP,
    P_DOUBT,
    P_NEWS_CAP,
    P_OUT,
    P_SUSPENDED,
    feed_age_h,
    fetch_indisponibili,
    fetch_probabili,
    fetch_rigoristi,
    fetch_sosfanta,
    p_play_override,
    rigoristi_by_pid,
    status_by_pid,
)
from scripts.fantacalcio.tracker import (
    LEVEL_K,
    MODULES,
    PORTA_INVIOLATA,
    SEASON,
    _modifier,
    discipline_status,
)

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
VOTO_SD_ROLE = {"A": 0.62, "C": 0.55, "D": 0.50, "P": 0.60}  # base-voto spread priors
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
    # Derived from SEASON, never a literal — a hardcoded season tag is the
    # annual-fuse trap this repo keeps paying for (see CLAUDE.md).
    cur_tag = "round_" + SEASON.replace("-", "_")
    frames, cur_frames = [], []
    by_tag: dict[str, list] = {}
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
        else:
            by_tag.setdefault(f.name[:len(cur_tag)], []).append(d)
    sd: dict[int, float] = {}
    voto_sd: dict[int, float] = {}
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
        gv = alld.groupby(["pid", "role"]).voto.agg(["count", "std"]).reset_index()
        for r in gv.itertuples():
            role = str(r.role).upper()
            prior = VOTO_SD_ROLE.get(role, 0.5)
            obs = float(r.std) if pd.notna(r.std) else prior
            n = max(int(r.count) - 1, 0)
            voto_sd[int(r.pid)] = round((SD_K * prior + n * obs) / (SD_K + n), 3)
    if cur_frames:
        curd = pd.concat(cur_frames)
        for pid, grp in curd.groupby("pid"):
            live[int(pid)] = {"n": len(grp), "fv": float(grp.fantavoto.mean()),
                              "voto": float(grp.voto.mean())}
    # Latest completed season: the REAL prior for players the auction model
    # never scored (arrivals, failed matches) — measured fantamedia beats a
    # bare 6.0 every time it exists.
    prev: dict[int, dict] = {}
    prev_tag = max((t for t in by_tag if t < cur_tag), default=None)
    if prev_tag:
        prevd = pd.concat(by_tag[prev_tag])
        for pid, grp in prevd.groupby("pid"):
            prev[int(pid)] = {"n": len(grp), "fv": float(grp.fantavoto.mean()),
                              "voto": float(grp.voto.mean())}
    _HISTORY = {"sd": sd, "voto_sd": voto_sd, "live": live, "prev": prev,
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
# within the role group top-down and SKIP a bench player without a voto
# (standard Leghe; max 3 subs total -- both still to be confirmed from the
# Opzioni screen), so slot order inside a role is by exp, not exp_slot.
BENCH_SLOTS = (("P", 1), ("D", 3), ("C", 3), ("A", 2))


def _role_recovery(chain: list) -> float:
    """Expected sub value of an ORDERED role chain when one XI slot of that
    role needs a replacement: p1*v1 + (1-p1)*(p2*v2 + (1-p2)*p3*v3) -- the
    auto-sub takes the first listed player who got a voto and skips the rest."""
    r, miss = 0.0, 1.0
    for c in chain:
        p = float(c.get("p_play", 1.0))
        r += miss * p * float(c.get("exp") or 0.0)
        miss *= 1.0 - p
    return r


def _best_bench_for_role(pool: list, n: int) -> list:
    """The n spares whose ORDERED chain maximizes recovery value. Order
    within a set is by exp desc (exchange-optimal under skip-no-voto subs:
    swapping adjacent (i, j) changes recovery by p_i*p_j*(v_i - v_j)); the
    SET is brute-forced -- pools are tiny on a 25-man squad. This is where a
    low-p_play star earns a slot: absent he costs nothing, present he is the
    best sub. Tie-break (all p==1 leaves only v1 in play) prefers the deeper
    chain by exp_slot sum."""
    from itertools import combinations
    if len(pool) <= n:
        return sorted(pool, key=lambda x: -(x["exp"] or 0.0))
    best, best_key = None, None
    for combo in combinations(pool, n):
        chain = sorted(combo, key=lambda x: -(x["exp"] or 0.0))
        k = (_role_recovery(chain), sum(c["exp_slot"] for c in chain))
        if best_key is None or k > best_key:
            best, best_key = chain, k
    return list(best)


def _bench_recovery_ev(xi: list, bench: list) -> float:
    """First-order expected points the ordered bench recovers: expected
    same-role XI absences x that role chain's recovery value. Understates
    chain depletion at 2+ same-role absences and ignores the 3-sub cap --
    both second-order at real starter p_play levels."""
    ev = 0.0
    for role in "PDCA":
        chain = [b for b in bench if b["R"] == role]
        if not chain:
            continue
        absences = sum(1.0 - float(x.get("p_play", 1.0))
                       for x in xi if x["R"] == role)
        ev += absences * _role_recovery(chain)
    return ev


def _bench_split(cand: list, xi: list) -> tuple[list, list]:
    """(bench, tribuna): the 9 league slots filled per role, then everyone
    else who could still play. Unavailable players belong to neither.
    Per role the SET+ORDER maximize recovery value -- see _best_bench_for_role."""
    rest = [c for c in cand if c not in xi and c["exp_slot"] is not None]
    bench = []
    for role, n in BENCH_SLOTS:
        pool = _best_bench_for_role([c for c in rest if c["R"] == role], n)
        bench.extend(pool)
        rest = [c for c in rest if c not in pool]
    return bench, rest


def advise(roster: list, fixtures: dict, elo: dict, out: dict,
           risk_lambda: float = 0.0,
           modules: list[tuple[int, int, int]] | None = None) -> dict:
    """Pure core: pick module + XI by p_play * expected fantavoto. Testable without disk.

    Roster rows without a p_play field count as certain starters (p=1.0) so the
    conditional ranking is unchanged for callers that never model titolarita.

    risk_lambda tilts SELECTION (never the reported numbers) by lambda * p_play
    * sd: positive prefers volatile players — the underdog play, since only
    variance can carry a weaker XI past a stronger opponent — negative prefers
    steady ones. Bench SELECTION stays pure exp_slot; entry ORDER within a
    role is by exp -- see _bench_split.
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
        tilt = (p["scorer_edge"] if p.get("scorer_edge") is not None
                else float(p.get("rig_bonus") or 0.0))
        exp = None if note else p["level"] + adj + tilt
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
    for nd, nc, na in (modules or MODULES):
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
        # Bench recovery is part of the objective: a module that strands the
        # spare quality in one role leaves real expected points on the table.
        # "total" stays XI+modifier (ledger/backtest continuity); exp_total
        # is the honest expected score including auto-sub recovery.
        bench, tribuna = _bench_split(cand, xi)
        bench_ev = _bench_recovery_ev(xi, bench)
        obj = sum(key(x) for x in xi) + mod + bench_ev
        if best_obj is None or obj > best_obj:
            best_obj = obj
            best = {"module": f"{nd}-{nc}-{na}", "total": round(total, 2),
                    "bench_ev": round(bench_ev, 2),
                    "exp_total": round(total + bench_ev, 2),
                    "xi_sd": round(sum(x.get("sd", 0.0) ** 2 for x in xi) ** 0.5, 2),
                    "modifier": round(mod, 2), "xi": xi, "bench": bench,
                    "tribuna": tribuna,
                    "unavailable": [c for c in cand if c["exp_slot"] is None]}
    if best and best.get("xi"):
        # Porta inviolata expectation at TOTAL level only — the published
        # fantavoto the ledger reconciles player exp against does NOT carry
        # the league bonus, so it must never enter a player's exp.
        gk = next((x for x in best["xi"] if x["R"] == "P"), None)
        p_cs = _p_clean_sheet(gk.get("team") if gk else None, fixtures)
        cs_ev = (PORTA_INVIOLATA * p_cs * float(gk.get("p_play") or 1.0)
                 if gk else 0.0)
        best["p_cs"] = round(p_cs, 3)
        best["total"] = round(best["total"] + cs_ev, 2)
        best["exp_total"] = round(best["exp_total"] + cs_ev, 2)
    return best or {"module": None, "total": 0.0, "bench_ev": 0.0,
                    "exp_total": 0.0, "modifier": 0.0, "xi": [],
                    "bench": [], "tribuna": [], "unavailable": cand}


def _board_priors(p: dict, prev: dict | None) -> tuple[float, float, float]:
    """(level, voto, p_play) priors for a board row, most-real-first.

    1. The auction model's projection (season_points / mv_hat / proj_min) —
       fit on two seasons of real votes and minutes.
    2. Else the player's ACTUAL latest-season record, shrunk toward the 6.0
       role mean by LEVEL_K — arrivals and failed auction matches (19 players
       measured 2026-09-02, e.g. David 6.33 over 30 games) were riding a
       bare 6.0 before this.
    3. Else 6.0 / 2% — the honest nothing-known floor.
    """
    sp = p.get("season_points")
    if sp:
        level = 6.0 + float(sp) / 38.0
    elif prev:
        level = (LEVEL_K * 6.0 + prev["n"] * prev["fv"]) / (LEVEL_K + prev["n"])
    else:
        level = 6.0
    if p.get("mv_hat"):
        voto = float(p["mv_hat"])
    elif prev:
        voto = (LEVEL_K * 6.0 + prev["n"] * prev["voto"]) / (LEVEL_K + prev["n"])
    else:
        voto = 6.0
    pm = float(p.get("proj_min") or 0.0)
    if pm:
        pp = pm / (38.0 * APP_MIN[p["R"]])
    elif prev:
        pp = prev["n"] / 38.0
    else:
        pp = 0.02
    return level, voto, min(max(pp, 0.02), 0.95)


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
    hist = _history()
    sds = hist["sd"]

    roster_src = []
    for pid in ids:
        p = by_id.get(pid)
        if not p:
            continue
        lv = live.get(p["nome"], {})
        prior, mv_prior, pp_prior = _board_priors(
            p, (hist.get("prev") or {}).get(pid))
        n_seen = int(lv.get("n_rounds", 0))
        p_play = (PLAY_K * pp_prior + n_seen) / (PLAY_K + rounds_elapsed)
        p_play = min(max(p_play, 0.02), 0.95)
        p_play, pp_src = p_play_override(pid, p_play, prob_by_pid)
        row = {"id": pid, "nome": p["nome"], "R": p["R"], "team": p["team"],
               "level": float(lv.get("live_level") or prior),
               "level_src": "live" if lv.get("live_level") else "prior",
               "voto": float(lv.get("live_voto") or mv_prior),
               "p_play": p_play, "p_play_src": pp_src,
               "sd": sds.get(pid) or SD_ROLE[p["R"]],
               "voto_sd": hist.get("voto_sd", {}).get(pid) or VOTO_SD_ROLE[p["R"]],
               "n_rounds": n_seen}
        if p.get("status") == "DEPARTED":
            row.update(p_play=0.02, p_play_src="departed", departed=True)
        roster_src.append(row)
    return roster_src, source


def _apply_official(rows: list[dict], official: dict[int, dict]) -> None:
    """Confirmed lineups are ground truth: applied LAST, over every
    probabilistic tier (probabili, titolarita, indisponibili, news). The
    p_play_src labels feed pred_ledger's per-source calibration."""
    for r in rows:
        o = official.get(int(r.get("id") or 0))
        if o:
            r["p_play"] = o["p_play"]
            r["p_play_src"] = o["src"]


ADVICE = ROOT / "data" / "fantacalcio" / "xi_advice.json"
# Hysteresis on the recommended MODULE: once a round's advice has shown one,
# a new best must beat it by this many exp_total fantapunti to displace it.
# Near-tied modules otherwise flip on every tenth-point input nudge — paid
# 2026-09-03: 3-5-2 was shown and SAVED on the league site, then the board
# flipped to 3-4-3 over a 0.15-point bench-recovery edge. HEURISTIC margin;
# the ledger's per-round module record is the refit path.
STICKY_MODULE_MARGIN = 1.0


def _sticky_module(adv: dict, roster: list, fixtures: dict, elo: dict,
                   out: dict, rnd: int | None) -> dict:
    """Keep the previously shown module unless the new best REALLY beats it.

    Re-evaluates the held module under CURRENT inputs (officials included),
    so a hold is never stale data — only a stable label. A new round, an
    infeasible previous module, or a gap >= STICKY_MODULE_MARGIN switches."""
    try:
        prev = json.loads(ADVICE.read_text())
    except (OSError, ValueError):
        return adv
    pm = prev.get("module")
    if not pm or prev.get("round") != rnd or pm == adv.get("module"):
        return adv
    try:
        shape = tuple(int(x) for x in pm.split("-"))
    except ValueError:
        return adv
    if len(shape) != 3:
        return adv
    held = advise(roster, fixtures, elo, out, modules=[shape])
    if not held.get("xi"):
        return adv
    gap = float(adv.get("exp_total") or 0.0) - float(held.get("exp_total") or 0.0)
    if gap >= STICKY_MODULE_MARGIN:
        return adv
    held["module_held"] = {"over": adv.get("module"),
                           "gain_forgone": round(max(gap, 0.0), 2)}
    return held


def build_advice(fresh: bool = False,
                 official: dict[int, dict] | None = None) -> dict:
    board = json.loads(BOARD.read_text())
    by_id = {int(p["id"]): p for p in board["players"]}
    fixtures, rnd = _next_fixtures()
    elo = _apply_market_elo(_current_elo(), fixtures)
    out = _out_ids(board["players"], fixtures)

    # Probable lineups beat the appearance-rate model when the page lists the
    # player (fetch is cached 6h and falls back to the last good cache on failure).
    prob_data = fetch_probabili(refresh=fresh)
    prob_by_pid = status_by_pid(prob_data)
    avail = fetch_indisponibili(refresh=fresh)
    sf = fetch_sosfanta(refresh=fresh)
    rig = rigoristi_by_pid(fetch_rigoristi(refresh=fresh))
    roster_src, source = _my_roster(
        by_id, prob_by_pid, [int(sq["id"]) for sq in board["squad"]])
    congestion = _club_congestion(fixtures)
    for row in roster_src:
        c = congestion.get(row["team"])
        if c:
            row["rest_d"] = c["rest_d"]
            row["congested"] = c["congested"]
            row["last_comp"] = c["last_comp"]
    _apply_discipline(roster_src, out, discipline_status())
    _apply_rigoristi(roster_src, rig)
    _apply_scorer(roster_src, _scorer_by_pid())
    _apply_availability(roster_src, avail, _news_caps(), sf=sf)
    if official:
        _apply_official(roster_src, official)

    adv = _sticky_module(advise(roster_src, fixtures, elo, out),
                         roster_src, fixtures, elo, out, rnd)
    kicks = [fixtures[t_]["ts"] for t_ in fixtures if fixtures[t_].get("ts")]
    return {"generated_at": datetime.now(UTC).isoformat(),
            "round": rnd, "source": source,
            "first_kickoff": min(kicks) if kicks else None,
            "feed_ages": {"probabili_h": feed_age_h(prob_data),
                          "indisponibili_h": feed_age_h(avail),
                          "sosfanta_h": feed_age_h(sf),
                          "scorer_h": _scorer_age_h()},
            **adv}


CONGESTION_CACHE = ROOT / "data" / "fantacalcio" / "club_congestion.json"
CONGESTION_TTL_H = 12.0
SHORT_REST_D = 4.0


def _congestion_from_events(events: list[dict], next_ts: float) -> dict | None:
    """Latest finished match before next_ts -> rest context. Pure, testable."""
    past = [e for e in events
            if (e.get("status") or {}).get("type") == "finished"
            and (e.get("startTimestamp") or 0) < next_ts]
    if not past:
        return None
    last = max(past, key=lambda e: e["startTimestamp"])
    rest_d = round((next_ts - last["startTimestamp"]) / 86400.0, 1)
    return {"last_ts": last["startTimestamp"],
            "last_comp": (last.get("tournament") or {}).get("name"),
            "rest_d": rest_d, "congested": rest_d <= SHORT_REST_D}


def _club_congestion(fixtures: dict) -> dict:
    """Serie A club -> rest context before ITS next fixture, ALL competitions.

    Sofascore team/events/last sees Coppa Italia and Europe, which the Serie A
    parquets cannot (probed live 2026-09-02: Sassuolo's Wednesday Coppa win is
    there). Measured on 2025-26 the fantavoto cost of short rest is -0.06 +-
    0.08 — statistically ZERO — so this is CONTEXT ONLY, never a coefficient:
    the real channel is rotation, and probabili already prices that. Cached
    12h, stale-on-failure, 3-strike breaker (ban discipline)."""
    cached = None
    try:
        cached = json.loads(CONGESTION_CACHE.read_text())
    except (OSError, ValueError):
        pass
    if cached:
        try:
            age_h = (datetime.now(UTC) - datetime.fromisoformat(
                cached["fetched_at"])).total_seconds() / 3600
            if age_h < CONGESTION_TTL_H:
                return cached.get("clubs", {})
        except (KeyError, ValueError):
            pass
    # team name -> sofascore id from the fixtures file we already parse
    try:
        raw = json.loads(_fixtures_path().read_text())
    except (OSError, ValueError, StopIteration):
        return (cached or {}).get("clubs", {})
    ids: dict[str, int] = {}
    for x in raw:
        for side in ("homeTeam", "awayTeam"):
            t = x.get(side) or {}
            name = NT(t.get("name", "")) or ""
            if name and t.get("id"):
                ids[name] = int(t["id"])
    clubs: dict[str, dict] = {}
    failures = 0
    try:
        import time

        from curl_cffi import requests as rq
        for name, fx in fixtures.items():
            tid, next_ts = ids.get(name), fx.get("ts")
            if not tid or not next_ts:
                continue
            if failures >= 3:      # breaker: do not grind a banned endpoint
                break
            try:
                r = rq.get(f"https://api.sofascore.com/api/v1/team/{tid}"
                           f"/events/last/0", impersonate="chrome124", timeout=15)
                if r.status_code != 200:
                    failures += 1
                    continue
                info = _congestion_from_events(r.json().get("events", []), next_ts)
                if info:
                    clubs[name] = info
                failures = 0
                time.sleep(0.4)
            except Exception:
                failures += 1
    except ImportError:
        pass
    if clubs:
        CONGESTION_CACHE.write_text(json.dumps(
            {"fetched_at": datetime.now(UTC).isoformat(), "clubs": clubs},
            indent=1, ensure_ascii=False))
        return clubs
    return (cached or {}).get("clubs", {})


def _deaccent(t: str) -> str:
    import unicodedata
    n = unicodedata.normalize("NFD", t)
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def _fold(nome: str) -> str:
    """Accent-folded surname key: the indisponibili page writes 'Kon\xe8'
    where the listone writes 'Kon\xe9' — one diacritic, zero matches."""
    from scripts.fantacalcio.news import _surname
    return _deaccent(_surname(nome))


def _avail_lookup(items: list[dict], rows: list[dict]) -> dict[int, dict]:
    """row-index -> injured-list item, one club's worth.

    Three tiers, each requiring a UNIQUE match on both sides before the next
    is tried: exact folded name (the page is the listone's own publisher, so
    'Kristensen T.' style names usually match verbatim — and an initial
    disambiguates two same-surname teammates), folded surname, folded last
    token (catches page 'Anguissa' vs listone 'Zambo Anguissa'). Departed
    rows are invisible: a ghost teammate must not block his replacement.
    """
    live = [(i, r) for i, r in enumerate(rows) if not r.get("departed")]
    out: dict[int, dict] = {}
    used: set[int] = set()
    for keyer in (lambda n: _deaccent(" ".join(n.split())),
                  _fold,
                  lambda n: _fold(n).split()[-1]):
        page: dict[str, list[int]] = {}
        for j, it in enumerate(items):
            if j not in used:
                page.setdefault(keyer(it["nome"]), []).append(j)
        side: dict[str, list[int]] = {}
        for i, r in live:
            if i not in out:
                side.setdefault(keyer(r["nome"]), []).append(i)
        for k, idxs in side.items():
            js = page.get(k)
            if js and len(js) == 1 and len(idxs) == 1:
                out[idxs[0]] = items[js[0]]
                used.add(js[0])
    return out


# p_play_src labels that mean "a probabili page listed this player" — the
# availability tiers below only annotate these, never override them (the
# lineup pages are fresher than the injury page; a listed player's injury
# rides along as avail_note). "titolarita" was MISSING from this set from
# its introduction until 2026-09-03, which silently inverted the hierarchy
# for every pct-listed player (393 of 479 listings) once the pct bar
# replaced the flat P_STARTER tier.
_LISTED_SRCS = ("probabili", "ballottaggio", "titolarita",
                "titolarita2", "ballottaggio2", "sosfanta")


def _apply_sosfanta(rows: list, sf: dict | None) -> None:
    """Second probabili source (SosFanta) — mutates p_play in place, BEFORE
    the availability tiers so _LISTED_SRCS sees the combined labels.

    The page carries its own per-player titolarita pct but no pids, so the
    join is by name within the club via the same 3-tier unique matcher the
    indisponibili page uses. Combination rule (each outcome its own ledger
    bucket, so per-source calibration judges every arm separately):
      fantacalcio pct  + sosfanta -> mean, "titolarita2"
      fantacalcio ballot + sosfanta -> mean, "ballottaggio2"
      flat P_STARTER/P_RESERVE or model prior + sosfanta -> sosfanta pct
        alone ("sosfanta") — one measurement beats a constant.
    Unmatched rows are untouched."""
    if not sf:
        return
    lo, hi = BALLOT_CLAMP
    club_rows: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(rows):
        club_rows.setdefault(r.get("team") or "", []).append((i, r))
    for team, t in (sf.get("teams") or {}).items():
        pairs = club_rows.get(team)
        players = t.get("players") or {}
        if not pairs or not players:
            continue
        items = [{"nome": n, "pct": v} for n, v in players.items()]
        sub = [r for _, r in pairs]
        for j, it in _avail_lookup(items, sub).items():
            r = pairs[j][1]
            p_sf = min(max(it["pct"] / 100.0, lo), hi)
            src = r.get("p_play_src")
            if src == "titolarita":
                r.update(p_play=round((r["p_play"] + p_sf) / 2, 4),
                         p_play_src="titolarita2")
            elif src == "ballottaggio":
                r.update(p_play=round((r["p_play"] + p_sf) / 2, 4),
                         p_play_src="ballottaggio2")
            else:
                r.update(p_play=p_sf, p_play_src="sosfanta")


# ── anytime-scorer market tilt (T-60) ──────────────────────────────────
# The per-event player_goal_scorer_anytime market (probe-verified live
# 2026-09-03: eu region, 2 books, full names, 1 credit/event) prices THIS
# match's goal expectation for ~45 players — lineup news, matchup and pens
# included. Only "Yes" is quoted, so the overround of a 45-outcome market
# is unobservable and a flat vig divisor is a guess (the first live build
# proved it: every edge came out positive). The formulation is therefore
# SHARE vs SHARE — vig cancels in the ratio:
#   s_i  = λ_raw_i / Σ_club λ_raw          (market share of club goals)
#   ŝ_i  = rate_shrunk_i / Σ_club rate     (his historical share)
#   Δexp = W · 3 · λ_club · (s_i − ŝ_i), capped — zero-sum within the club,
# so team strength stays priced by the market-Elo blend and ONLY the
# within-team allocation moves. λ_club comes from the totals+h2h markets
# already on disk (_market_team_lambdas: de-vig, Poisson-solve the total,
# bisect the home/away split — all measured, no invented constants).
# DECLARED HEURISTICS, refit path = pred_ledger rows carry both λs:
#   SCORER_W   — half-weight hedge (market λ and the level's own bonus
#                content overlap), same convention as RIGORISTA_BONUS.
#   SCORER_CAP — thin 2-book markets misprice; bound the damage.
#   SCORER_K   — own rate shrunk toward his market-implied rate with K=10:
#                a 2-match arrival cannot out-argue the market, a 38-match
#                veteran can. λ_own uses understat goals-per-appearance,
#                both seasons pooled.
# Priced player with a rig_bonus: the market λ already contains his
# penalties, so the tilt REPLACES the bonus (rank kept for the ledger).
SCORER_W = 0.5
SCORER_CAP = 0.45
SCORER_K = 10.0
SCORER_RAW = ROOT / "data" / "fantacalcio" / "scorer_odds_raw.json"
SCORER_EDGES = ROOT / "data" / "fantacalcio" / "scorer_edges.json"


def _lam_from_prices(prices: list[float]) -> tuple[float, float] | None:
    """(p_raw, lambda_raw) from the quoted prices, median across books.
    RAW implied probability — the vig is never divided out here, it cancels
    later in the within-club share."""
    import math
    import statistics
    ps = [x for x in prices if x and x > 1.01]
    if not ps:
        return None
    p = min(max(1.0 / statistics.median(ps), 0.005), 0.9)
    return p, -math.log(1.0 - p)


def _pois_cdf(k: int, lam: float) -> float:
    import math
    t, term = 0.0, math.exp(-lam)
    for i in range(k + 1):
        t += term
        term *= lam / (i + 1)
    return t


def _market_team_lambdas(home: str, away: str,
                         odds: dict | None = None) -> tuple[float, float] | None:
    """(λ_home, λ_away) implied by the totals + h2h markets on disk.

    λ_total: de-vig the half-integer totals line closest to 2.5, solve
    P_Pois(N ≤ ⌊line⌋) = p_under by bisection. Split: bisect the difference
    so the independent-Poisson P(H>A)/(P(H>A)+P(A>H)) matches the de-vigged
    h2h ratio. None when either market is missing — callers fail open."""
    if odds is None:
        try:
            odds = json.loads((ROOT / "data" / "upcoming"
                               / "odds_full.json").read_text()).get("matches", {})
        except (OSError, ValueError):
            return None
    m = odds.get(f"{home} vs {away}") or {}
    h2h = m.get("h2h") or {}
    try:
        inv = [1.0 / float(h2h[k]) for k in ("home", "draw", "away")]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    r_target = inv[0] / (inv[0] + inv[2])
    cands = [t for t in (m.get("totals") or [])
             if t.get("line") and float(t["line"]) % 1 == 0.5
             and t.get("over") and t.get("under")]
    if not cands:
        return None
    tot = min(cands, key=lambda t: abs(float(t["line"]) - 2.5))
    io, iu = 1.0 / float(tot["over"]), 1.0 / float(tot["under"])
    p_under = iu / (io + iu)
    k_floor = int(float(tot["line"]))
    lo, hi = 0.05, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _pois_cdf(k_floor, mid) > p_under:
            lo = mid
        else:
            hi = mid
    lam_tot = (lo + hi) / 2.0

    import math

    def _ratio(d: float) -> float:
        lh, la = (lam_tot + d) / 2.0, (lam_tot - d) / 2.0
        ph = [math.exp(-lh) * lh ** i / math.factorial(i) for i in range(13)]
        pa = [math.exp(-la) * la ** i / math.factorial(i) for i in range(13)]
        w = sum(ph[i] * pa[j] for i in range(13) for j in range(i))
        l_ = sum(ph[i] * pa[j] for i in range(13) for j in range(i + 1, 13))
        return w / (w + l_) if (w + l_) > 0 else 0.5
    lo_d, hi_d = -lam_tot + 0.02, lam_tot - 0.02
    for _ in range(50):
        mid = (lo_d + hi_d) / 2.0
        if _ratio(mid) < r_target:
            lo_d = mid
        else:
            hi_d = mid
    d = (lo_d + hi_d) / 2.0
    return (lam_tot + d) / 2.0, (lam_tot - d) / 2.0


def build_scorer_edges() -> dict | None:
    """scorer_odds_raw.json -> scorer_edges.json (pid-keyed, advisor-ready).

    All fragile work happens HERE, once per fetch, inspectable in the
    artifact: market full name -> board pid and -> understat rate, both via
    the same 3-tier folded-name ladder (_avail_lookup) scoped to the event's
    two clubs. Ambiguity fails open — an unmatched price is dropped, never
    guessed."""
    try:
        raw = json.loads(SCORER_RAW.read_text())
    except (OSError, ValueError):
        return None
    board = json.loads(BOARD.read_text())
    brows_by_team: dict[str, list[dict]] = {}
    for pl in board["players"]:
        if pl.get("status") != "DEPARTED":
            brows_by_team.setdefault(NT(pl["team"]), []).append(pl)
    us = pd.read_parquet(ROOT / "data" / "parsed" / "understat_players.parquet",
                         columns=["league", "season", "team", "player",
                                  "matches", "goals"])
    us = us[(us.league == "ITA-Serie A")
            & (us.season.isin(_us_seasons()))].copy()
    us["tnorm"] = [NT(t) for t in us.team]
    by_pid: dict = {}
    matches_out = []
    for eid, ev in (raw.get("events") or {}).items():
        prices = ev.get("prices") or {}
        if not prices:
            continue
        names = list(prices)
        items = [{"nome": n} for n in names]
        matched = 0
        lams = _market_team_lambdas(NT(ev.get("home") or ""),
                                    NT(ev.get("away") or ""))
        for side, club in enumerate((ev.get("home"), ev.get("away"))):
            club = NT(club or "")
            brows = brows_by_team.get(club) or []
            if not brows or lams is None:
                continue
            lam_club = lams[side]
            found = _avail_lookup(items, brows)      # board-idx -> item
            urows_src = us[us.tnorm == club]
            grp = urows_src.groupby("player", as_index=False).agg(
                matches=("matches", "sum"), goals=("goals", "sum"))
            urows = [{"nome": r.player, "matches": int(r.matches),
                      "goals": int(r.goals)} for r in grp.itertuples()]
            ufound = {}                              # market name -> u row
            if urows:
                for j, it in _avail_lookup(items, urows).items():
                    ufound[it["nome"]] = urows[j]
            # pass 1: raw market lambdas + own rates for this club's matches
            side_rows = []
            for j, it in found.items():
                pl = brows[j]
                lam = _lam_from_prices(prices[it["nome"]])
                if lam is None:
                    continue
                p_raw, lam_raw = lam
                u = ufound.get(it["nome"])
                n_app = int(u["matches"]) if u else 0
                rate = (u["goals"] / u["matches"]) if u and u["matches"] else 0.0
                side_rows.append((pl, it["nome"], p_raw, lam_raw, n_app, rate))
            s_lam = sum(r[3] for r in side_rows)
            if not side_rows or s_lam <= 0:
                continue
            # pass 2: shares. Own rate shrunk toward the player's OWN
            # market-implied rate (K=SCORER_K), then renormalized — the
            # edge is zero-sum within the club by construction.
            shrunk = []
            for _pl, _nome, _p, lam_raw, n_app, rate in side_rows:
                implied = lam_club * lam_raw / s_lam
                shrunk.append((SCORER_K * implied + n_app * rate)
                              / (SCORER_K + n_app))
            s_shr = sum(shrunk) or 1.0
            for (pl, nome, p_raw, lam_raw, n_app, _r), rs in zip(side_rows,
                                                                 shrunk):
                s_i = lam_raw / s_lam
                sh_i = rs / s_shr
                lam_mkt = lam_club * s_i
                lam_own = lam_club * sh_i
                edge = SCORER_W * 3.0 * (lam_mkt - lam_own)
                edge = min(max(edge, -SCORER_CAP), SCORER_CAP)
                by_pid[int(pl["id"])] = {
                    "nome": pl["nome"], "team": pl["team"],
                    "p_raw": round(p_raw, 4),
                    "lam_mkt": round(lam_mkt, 4),
                    "lam_own": round(lam_own, 4), "n_app": n_app,
                    "n_books": len(prices[nome]),
                    "edge": round(edge, 3),
                }
                matched += 1
        matches_out.append({"event": eid, "home": ev.get("home"),
                            "away": ev.get("away"),
                            "commence": ev.get("commence"),
                            "priced": len(prices), "matched": matched})
    out = {"built_at": datetime.now(UTC).isoformat(),
           "matches": matches_out,
           "by_pid": {str(k): v for k, v in by_pid.items()}}
    SCORER_EDGES.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def _us_seasons() -> list[str]:
    """Current + previous season tags in understat format, derived from
    SEASON ('2026-27' -> ['2025-2026', '2026-2027']) — never a literal."""
    y1 = int("20" + SEASON.split("-")[0][-2:]) if len(SEASON.split("-")[0]) == 2 \
        else int(SEASON.split("-")[0])
    return [f"{y1 - 1}-{y1}", f"{y1}-{y1 + 1}"]


def _scorer_by_pid() -> dict[int, dict]:
    try:
        d = json.loads(SCORER_EDGES.read_text())
        return {int(k): v for k, v in (d.get("by_pid") or {}).items()}
    except (OSError, ValueError):
        return {}


def _scorer_age_h() -> float | None:
    try:
        d = json.loads(SCORER_EDGES.read_text())
        return (datetime.now(UTC)
                - datetime.fromisoformat(d["built_at"])).total_seconds() / 3600
    except (OSError, ValueError, KeyError):
        return None


def _apply_scorer(rows: list, sco_by_pid: dict[int, dict] | None) -> None:
    """pid-exact join; sets the market tilt and drops rig_bonus on priced
    rows (the market λ already contains penalties). No-op without edges."""
    for r in rows:
        e = (sco_by_pid or {}).get(int(r.get("id") or 0))
        if e:
            r["scorer_edge"] = e["edge"]
            r["lam_mkt"] = e["lam_mkt"]
            r["lam_own"] = e["lam_own"]
            r.pop("rig_bonus", None)


# Penalty-taker premium, DECLARED HEURISTIC with a refit path (per the
# real-data directive): league priors ~0.24 pens/team-match x (0.78 conv
# x +3 fv - 0.22 miss x 3) = +0.40 fv/match for an ever-present rank-1
# taker. Rank 1 is HALVED because an incumbent taker's level already
# prices his own past penalty income (double-count); rank 2 takes only
# the minutes rank 1 is absent. pred_ledger rows carry the rank, so ~5
# graded rounds fit the residual premium properly.
RIGORISTA_BONUS = {1: 0.20, 2: 0.06}


def _apply_rigoristi(rows: list, rig_by_pid: dict[int, int] | None) -> None:
    """pid-exact join (the page links carry pids); sets the rank flag and
    the exp bump advise() consumes. No-op without the feed."""
    for r in rows:
        rank = (rig_by_pid or {}).get(int(r.get("id") or 0))
        if rank:
            r["rigorista"] = rank
            b = RIGORISTA_BONUS.get(rank)
            if b:
                r["rig_bonus"] = b


def _apply_availability(rows: list, avail: dict | None,
                        news_caps: dict[str, str] | None = None,
                        sf: dict | None = None) -> None:
    """Availability tiers BELOW a probabili listing; mutates p_play in place.

    Hierarchy (pid-exact fresh beats name-matched):
      1. probabili/ballottaggio listing — p_play untouched; an injured-list
         row OR a news risk hit rides along as avail_note (the late-breaking
         conflict a manager should eyeball before kickoff).
      2. indisponibili page (matched by _avail_lookup within the club):
         squalificato -> P_SUSPENDED, infortunato -> P_OUT,
         infortunato_dubbio -> min(p, P_DOUBT).
      3. title-bound news risk hit (infortunio/squalifica, never mercato —
         transfers are the wiki pass's job): min(p, P_NEWS_CAP). Caps only,
         never zeroes — headlines lie.
    Every tier labels p_play_src, so pred_ledger's per-source calibration
    grades each one against who actually got a voto.
    """
    _apply_sosfanta(rows, sf)
    by_club: dict[str, dict[int, dict]] = {}
    club_rows: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(rows):
        club_rows.setdefault(r.get("team") or "", []).append((i, r))
    for team, items in ((avail or {}).get("teams") or {}).items():
        pairs = club_rows.get(team)
        if not items or not pairs:
            continue
        sub = [r for _, r in pairs]
        found = _avail_lookup(items, sub)
        by_club[team] = {pairs[j][0]: it for j, it in found.items()}
    hits = {i: it for d in by_club.values() for i, it in d.items()}
    for i, r in enumerate(rows):
        if r.get("departed"):
            continue
        hit = hits.get(i)
        cat = (news_caps or {}).get(r["nome"])
        if r.get("p_play_src") in _LISTED_SRCS:
            if hit:                      # listed AND on the injured list
                r["avail_note"] = f"{hit['status']}: {hit['note'][:140]}"
            elif cat:                    # listed, but headlines disagree
                r["avail_note"] = f"news: {cat}"
            continue
        if hit:
            if hit["status"] == "squalificato":
                r.update(p_play=P_SUSPENDED, p_play_src="squalificato_sito")
            elif hit["status"] == "infortunato_dubbio":
                r.update(p_play=min(r["p_play"], P_DOUBT),
                         p_play_src="infortunio_dubbio")
            else:
                r.update(p_play=P_OUT, p_play_src="infortunio_sito")
            r["avail_note"] = hit["note"][:140]
            continue
        if cat:
            r.update(p_play=min(r["p_play"], P_NEWS_CAP),
                     p_play_src="news_risk")
            r["avail_note"] = f"news: {cat}"


def _news_caps() -> dict[str, str]:
    """nome -> risk category from the accumulated news items, title-bound,
    availability categories only (mercato moves are handled by the board)."""
    try:
        from scripts.fantacalcio.news import risk_hits
        items = json.loads((ROOT / "data" / "fantacalcio"
                            / "news.json").read_text()).get("items", [])
    except (OSError, ValueError):
        return {}
    caps: dict[str, str] = {}
    for it in items:
        for nome, cat in risk_hits(it):
            if cat in ("infortunio", "squalifica"):
                caps[nome] = cat
    return caps


def _apply_discipline(rows: list, out: dict, disc: dict) -> None:
    """Bans become unavailability; yellow counts ride along for display.

    The ban targets exactly the round being advised (trigger measured in the
    latest PLAYED round), matching how advice always looks one round ahead."""
    for row in rows:
        d = disc.get(int(row["id"]))
        if not d:
            continue
        row["yellows"] = d["yellows"]
        row["diffidato"] = d["diffidato"]
        if d["banned_next"] and int(row["id"]) not in out:
            out[int(row["id"])] = d["why"]


def _p_win(mu_a: float, sd_a: float, mu_b: float, sd_b: float) -> float:
    """P(A outscores B) under a normal approximation of the two XI totals.

    Team sd = sqrt(sum of per-player fantavoto variances), conditioned on the
    XI actually playing; modifier variance and auto-sub recovery are ignored.
    Good enough for guidance, not for staking."""
    from math import erf, sqrt
    spread = sqrt(sd_a ** 2 + sd_b ** 2) or 1.0
    return 0.5 * (1.0 + erf((mu_a - mu_b) / spread / sqrt(2.0)))


# ── Monte Carlo H2H ─────────────────────────────────────────────────────
# Goal thresholds + modifier table: BOTH verified against the league's
# live Opzioni page (2026-09-03, in-browser read): 1° gol 66 then +6 each
# (114 = 9°), all four correttivi OFF (limita vittoria, limita pareggio,
# autogol, ammonito senza voto) so the pure ladder below is exact; difesa
# modifier ON, best-3+GK, its 0.25-bands collapse to MOD_TABLE; every
# other modifier OFF. pred_ledger.verify_goal_ladder still cross-checks
# from real score cells: it additionally guards the calendar importer's
# played-row cell mapping, which no options page can verify.
GOAL_BASE, GOAL_STEP = 66.0, 6.0
MC_N = 4000
# P(a Serie A side concedes 0): measured on data/parsed/matches.parquet,
# seasons 2019-20 onward, n=5,360 team-matches (0.2632, 2026-09-03). The
# market path below overrides this whenever the fixture is priced.
CS_BASE_RATE = 0.263
# Shared per-club shock loading: same-club players move together (a blowout
# showers bonuses on everyone). HEURISTIC pending refit from the ledger —
# the measured path is per-club fv correlation across the voti parquets.
MC_CLUB_BETA = 0.35
MOD_TABLE = ((6.0, 1), (6.5, 3), (7.0, 6), (7.5, 9))
MOD_OFFICE = (5.0, 4.5, 4.5)


def _fp_to_goals(fp):
    """Vector fp → goals under the threshold ladder."""
    import numpy as np
    fp = np.asarray(fp, dtype=float)
    return np.where(fp < GOAL_BASE, 0.0,
                    np.floor((fp - GOAL_BASE) / GOAL_STEP) + 1.0)


def _p_clean_sheet(team: str | None, fixtures: dict) -> float:
    """P(the club keeps a clean sheet): exp(-lam_opponent) from the market
    team-lambda solver when the fixture is priced, else the measured league
    base rate. Feeds the porta inviolata (+1 GK) league bonus."""
    import math
    fx = fixtures.get(team) if team else None
    if fx and fx.get("opp"):
        try:
            home, away = (team, fx["opp"]) if fx.get("home") else (fx["opp"], team)
            lams = _market_team_lambdas(home, away)
            if lams:
                lam_opp = lams[1] if fx.get("home") else lams[0]
                return float(math.exp(-lam_opp))
        except (OSError, ValueError, KeyError):
            pass
    return CS_BASE_RATE


def _side_totals(adv: dict, zc: dict, rng, n: int):
    """Simulated fantapunti totals (length-n vector) for one advise() result.

    Per player: correlated normal shock (club-shared component MC_CLUB_BETA),
    fv and voto move on the SAME standardized draw; plays with prob p_play;
    an absent XI slot takes the first same-role bench player (listed order)
    who played. The Leghe 3-sub global cap is NOT enforced — at starter-level
    p_play the fourth simultaneous absence is rare enough to ignore, and the
    error is conservative (slightly optimistic bench recovery on both sides).
    The defense modifier is computed per-draw from the simulated votes of the
    FIELDED defenders (subs included), top-3 + GK — so tier-edge risk prices
    correctly instead of thresholding an expectation (Jensen).
    """
    import numpy as np
    xi = adv.get("xi") or []
    bench = adv.get("bench") or []
    rows = xi + bench
    if not xi:
        return np.zeros(n)
    m = len(rows)
    def _f(r, key, default):
        v = r.get(key)
        return default if v is None else float(v)   # 0.0 is a real value

    exp = np.array([_f(r, "exp", 6.0) for r in rows])
    sd = np.array([_f(r, "sd", SD_ROLE.get(r.get("R"), 1.3)) for r in rows])
    ev = np.array([_f(r, "exp_voto", _f(r, "voto", 6.0)) for r in rows])
    vsd = np.array([_f(r, "voto_sd", VOTO_SD_ROLE.get(r.get("R"), 0.5))
                    for r in rows])
    pp = np.clip(np.array([float(r.get("p_play", 1.0)) for r in rows]), 0, 1)

    eps = rng.standard_normal((n, m))
    z = np.empty((n, m))
    b2 = (1.0 - MC_CLUB_BETA ** 2) ** 0.5
    for j, r in enumerate(rows):
        club = zc.get(r.get("team"))
        z[:, j] = (MC_CLUB_BETA * club + b2 * eps[:, j]
                   if club is not None else eps[:, j])
    fv = exp + sd * z
    voto = ev + vsd * z
    played = rng.random((n, m)) < pp

    n_xi = len(xi)
    fielded_fv = np.where(played[:, :n_xi], fv[:, :n_xi], 0.0)
    total = fielded_fv.sum(axis=1)

    # role-wise bench chains, listed order
    d_votos = []      # per-draw votes of fielded defenders (for the modifier)
    gk_voto = np.full(n, np.nan)
    for role in "PDCA":
        xi_idx = [j for j in range(n_xi) if rows[j].get("R") == role]
        ch_idx = [n_xi + j for j in range(len(bench))
                  if bench[j].get("R") == role]
        if not xi_idx:
            continue
        missing = (~played[:, xi_idx]).sum(axis=1).astype(float)
        used_before = np.zeros(n)
        for j in ch_idx:
            use = played[:, j] & (used_before < missing)
            total += np.where(use, fv[:, j], 0.0)
            if role == "P":
                gk_voto = np.where(use, voto[:, j], gk_voto)
            elif role == "D":
                d_votos.append(np.where(use, voto[:, j], -np.inf))
            used_before += use.astype(float)
        if role == "P":
            j0 = xi_idx[0]
            gk_voto = np.where(played[:, j0], voto[:, j0], gk_voto)
        elif role == "D":
            for j in xi_idx:
                d_votos.append(np.where(played[:, j], voto[:, j], -np.inf))

    try:
        nd = int(str(adv.get("module") or "0").split("-")[0])
    except ValueError:
        nd = 0
    if nd >= 4 and d_votos:
        dm = np.sort(np.vstack(d_votos), axis=0)[::-1][:3]   # top-3 fielded
        for i in range(3):
            fill = MOD_OFFICE[min(i, len(MOD_OFFICE) - 1)]
            dm[i] = np.where(np.isinf(dm[i]), fill, dm[i])
        gk = np.where(np.isnan(gk_voto), np.nan, gk_voto)
        avg = (gk + dm.sum(axis=0)) / 4.0
        mod = np.zeros(n)
        for threshold, value in MOD_TABLE:
            mod = np.where(avg >= threshold, float(value), mod)
        total += np.where(np.isnan(avg), 0.0, mod)
    # Porta inviolata: +1 when the fielded GK's club concedes 0. The
    # clean-sheet indicator rides the GK club's SHARED shock, so clean
    # sheets co-move with defender votes (and the modifier) instead of
    # being independent noise; the marginal rate is exactly p_cs.
    p_cs = float(adv.get("p_cs") or 0.0)
    if 0.0 < p_cs < 1.0:
        from statistics import NormalDist
        gk_row = next((r for r in xi if r.get("R") == "P"), None)
        club = zc.get(gk_row.get("team")) if gk_row else None
        cs = (club > NormalDist().inv_cdf(1.0 - p_cs)) if club is not None             else (rng.random(n) < p_cs)
        total += PORTA_INVIOLATA * np.where(~np.isnan(gk_voto) & cs, 1.0, 0.0)
    elif p_cs >= 1.0:
        total += PORTA_INVIOLATA * np.where(~np.isnan(gk_voto), 1.0, 0.0)
    return total


def h2h_mc(mine: dict, theirs: dict, n: int = MC_N,
           seed: int | None = None) -> dict:
    """Simulated head-to-head between two advise() results.

    Returns P(win)/P(draw)/P(loss) through the Leghe goal thresholds — the
    draw band the closed-form Φ ignores — plus expected league points
    (3·W + D) and the simulated total moments. Club shocks are shared
    ACROSS the two sides: a Juve player on both rosters cancels variance,
    which the independence approximation could not see.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    clubs = {r.get("team")
             for adv in (mine, theirs)
             for r in (adv.get("xi") or []) + (adv.get("bench") or [])}
    zc = {c: rng.standard_normal(n) for c in clubs if c}
    fa = _side_totals(mine, zc, rng, n)
    fb = _side_totals(theirs, zc, rng, n)
    ga, gb = _fp_to_goals(fa), _fp_to_goals(fb)
    p_win = float((ga > gb).mean())
    p_draw = float((ga == gb).mean())
    return {"p_win": round(p_win, 3), "p_draw": round(p_draw, 3),
            "p_loss": round(1.0 - p_win - p_draw, 3),
            "e_pts": round(3 * p_win + p_draw, 2),
            "my_mu": round(float(fa.mean()), 2),
            "my_sd": round(float(fa.std()), 2),
            "opp_mu": round(float(fb.mean()), 2),
            "opp_sd": round(float(fb.std()), 2)}


def _mc_seed(*parts) -> int:
    """Stable seed from round + team names: reruns of the same matchup give
    the same probabilities (no display jitter between tracker runs)."""
    import zlib
    return zlib.crc32("|".join(str(x) for x in parts).encode()) & 0x7FFFFFFF


# Market → Elo-equivalent blend. The per-role vote slopes (ELO_SLOPE) were
# FIT in ΔElo units; converting the de-vigged 1X2 into an Elo-equivalent
# delta lets sharper market information flow through those measured slopes
# into every player's exp — no invented coefficient. Weights are HEURISTIC
# with a refit path (ledger grading by round).
MARKET_ELO_W = 0.7        # market share of the blended delta where odds exist
HOME_ELO_EDGE = 65.0      # market prices home advantage; the Elo table keeps
                          # it separate (HOME_ADJ), so strip it before mixing


def _apply_market_elo(elo: dict, fixtures: dict) -> dict:
    """Per fixture pair, shift the two ratings so their delta becomes
    w·Δ_market + (1−w)·Δ_elo (pair mean preserved). Teams without a priced
    match keep their table rating; any failure returns the table untouched."""
    import math
    try:
        odds = json.loads((ROOT / "data" / "upcoming"
                           / "odds_full.json").read_text()).get("matches", {})
    except (OSError, ValueError):
        return elo
    adj = dict(elo)
    done = set()
    for team, fx in fixtures.items():
        opp = fx.get("opp")
        if not opp or team in done or opp in done:
            continue
        home, away = (team, opp) if fx.get("home") else (opp, team)
        m = odds.get(f"{home} vs {away}")
        h2h = (m or {}).get("h2h") or {}
        try:
            inv = [1.0 / float(h2h[k]) for k in ("home", "draw", "away")]
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        tot = sum(inv)
        if tot <= 0:
            continue
        p_home = (inv[0] + 0.5 * inv[1]) / tot        # draw-half score expectancy
        p_home = min(max(p_home, 0.01), 0.99)
        d_mkt = -400.0 * math.log10(1.0 / p_home - 1.0) - HOME_ELO_EDGE
        r_h, r_a = elo.get(home, 1450.0), elo.get(away, 1450.0)
        d_blend = MARKET_ELO_W * d_mkt + (1.0 - MARKET_ELO_W) * (r_h - r_a)
        mean = (r_h + r_a) / 2.0
        adj[home] = mean + d_blend / 2.0
        adj[away] = mean - d_blend / 2.0
        done.update((team, opp))
    return adj


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
        prior, mv_prior, pp_prior = _board_priors(
            p, (hist.get("prev") or {}).get(pid))
        n_seen = int(lv["n"]) if lv else 0
        level = (LEVEL_K * prior + n_seen * lv["fv"]) / (LEVEL_K + n_seen) \
            if lv else prior
        voto = (LEVEL_K * mv_prior + n_seen * lv["voto"]) / (LEVEL_K + n_seen) \
            if lv else mv_prior
        p_play = (PLAY_K * pp_prior + n_seen) / (PLAY_K + rounds_elapsed) \
            if rounds_elapsed else pp_prior
        p_play = min(max(p_play, 0.02), 0.95)
        p_play, pp_src = p_play_override(pid, p_play, prob_by_pid)
        row = {"id": pid, "nome": p["nome"], "R": p["R"], "team": p["team"],
               "level": level, "voto": voto,
               "p_play": p_play, "p_play_src": pp_src,
               "sd": hist["sd"].get(pid) or SD_ROLE[p["R"]],
               "voto_sd": hist.get("voto_sd", {}).get(pid) or VOTO_SD_ROLE[p["R"]],
               "n_rounds": n_seen}
        if p.get("status") == "DEPARTED":
            row.update(p_play=0.02, p_play_src="departed", departed=True)
        rows.append(row)
    return rows


# Selection tilts tried against each opponent. Positive = chase variance
# (underdog), negative = buy stability (favourite). An alternative XI is
# reported only when it moves P(win) by at least ALT_MIN_GAIN.
RISK_LAMBDAS = (-0.3, -0.15, 0.15, 0.3)
ALT_MIN_GAIN = 0.01


FIELDED = ROOT / "data" / "fantacalcio" / "rival_modules.json"


def record_fielded(team: str, module: str, rnd: int,
                   path: Path = FIELDED) -> dict:
    """Record the module a league team ACTUALLY fielded in a round.

    The learning input for the rival matrix: observed behaviour beats the
    optimal-module assumption (a rival who always plays 4-4-2, or forgets
    lineups, is weaker than his best XI). Fed from Leghe screenshots (the
    Telegram bot asks the vision reply to end with 'Modulo avversario
    osservato: <squadra> <modulo>') or by hand via --fielded.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    data.setdefault(team, {})[str(rnd)] = module
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return data


def _observed_modules(team: str, path: Path = FIELDED) -> list[tuple]:
    """This team's observed-module repertoire, most recent first."""
    try:
        data = json.loads(path.read_text()).get(team) or {}
    except (OSError, ValueError):
        return []
    mods = []
    for rnd in sorted(data, key=int, reverse=True):
        try:
            t = tuple(int(x) for x in str(data[rnd]).split("-"))
        except ValueError:
            continue
        if len(t) == 3 and t not in mods:
            mods.append(t)
    return mods


def build_rivals(adv: dict | None = None) -> dict:
    """Expected score + my win probability vs EVERY league team this round.

    Calendar-free by design: with two competitions (Coppa Del Nonno groups,
    Hunger Games) the opponent differs per weekend — a full rival matrix
    covers both without knowing either schedule. Rival totals include their
    defense modifier: they run through the same module search mine does."""
    board = json.loads(BOARD.read_text())
    by_id = {int(p["id"]): p for p in board["players"]}
    fixtures, rnd = _next_fixtures()
    elo = _apply_market_elo(_current_elo(), fixtures)
    out = _out_ids(board["players"], fixtures)
    hist = _history()
    prob_by_pid = status_by_pid(fetch_probabili())
    avail = fetch_indisponibili()
    sf = fetch_sosfanta()
    rig = rigoristi_by_pid(fetch_rigoristi())

    league = json.loads(
        (ROOT / "data" / "fantacalcio" / "league_rosters.json").read_text())
    my_name = league.get("my_team")
    congestion = _club_congestion(fixtures)

    # Calendars (import_rosters --calendars): who I actually face this round,
    # per competition, and every future meeting per rival.
    schedule = {}
    try:
        schedule = json.loads((ROOT / "data" / "fantacalcio"
                               / "league_schedule.json").read_text())
    except (OSError, ValueError):
        pass
    next_opps: list[dict] = []
    meetings: dict[str, list] = {}
    for comp, cdata in (schedule.get("competitions") or {}).items():
        for rd in cdata.get("rounds", []):
            mine = next((f for f in rd["fixtures"]
                         if my_name in (f["home"], f["away"])), None)
            resting = any(r["team"] == my_name for r in rd.get("rests", []))
            if rd["sa_round"] == rnd:
                if mine:
                    opp = mine["away"] if mine["home"] == my_name else mine["home"]
                    next_opps.append({"competition": comp, "opponent": opp,
                                      "league_round": rd["league_round"],
                                      "home": mine["home"] == my_name})
                elif resting:
                    next_opps.append({"competition": comp, "opponent": None,
                                      "league_round": rd["league_round"],
                                      "home": None})
            if mine and rd["sa_round"] is not None and rnd is not None \
                    and rd["sa_round"] >= rnd:
                opp = mine["away"] if mine["home"] == my_name else mine["home"]
                meetings.setdefault(opp, []).append(
                    {"competition": comp, "league_round": rd["league_round"],
                     "sa_round": rd["sa_round"]})

    my_roster, _src = _my_roster(
        by_id, prob_by_pid, [int(sq["id"]) for sq in board["squad"]])
    disc = discipline_status()
    for row in my_roster:
        c = congestion.get(row["team"])
        if c:
            row["rest_d"] = c["rest_d"]
            row["congested"] = c["congested"]
            row["last_comp"] = c["last_comp"]
    _apply_discipline(my_roster, out, disc)
    _apply_rigoristi(my_roster, rig)
    _apply_scorer(my_roster, _scorer_by_pid())
    _apply_availability(my_roster, avail, _news_caps(), sf=sf)
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
        for row in rows:
            c = congestion.get(row["team"])
            if c:
                row["rest_d"] = c["rest_d"]
                row["congested"] = c["congested"]
                row["last_comp"] = c["last_comp"]
        _apply_discipline(rows, out, disc)
        _apply_rigoristi(rows, rig)
        _apply_scorer(rows, _scorer_by_pid())
        _apply_availability(rows, avail, sf=sf)
        obs = _observed_modules(tname)
        radv = advise(rows, fixtures, elo, out, modules=obs or None)
        module_src = "osservato" if obs and radv.get("xi") else "stimato"
        if obs and not radv.get("xi"):    # repertoire infeasible (bans etc.)
            radv = advise(rows, fixtures, elo, out)
        n_missing = (len(entry.get("unmatched", []))
                     + len(entry.get("roster", [])) - len(rows))
        if not radv.get("xi"):
            rivals.append({"team": tname, "module": None, "total": None,
                           "sd": None, "p_win": None, "n_missing": n_missing,
                           "alt": None})
            continue
        mc0 = h2h_mc(base, radv, seed=_mc_seed(rnd, my_name, tname))
        p0 = mc0["p_win"]
        best_alt = None
        for lam, alt in tilted:
            mca = h2h_mc(alt, radv, seed=_mc_seed(rnd, my_name, tname, lam))
            # decision on expected league points (3W+D): with a real draw
            # band, a tilt can raise P(win) while giving away more E[pts]
            if mca["e_pts"] >= mc0["e_pts"] + 3 * ALT_MIN_GAIN                     and (best_alt is None or mca["e_pts"] > best_alt["e_pts"]):
                alt_names = {x["nome"] for x in alt["xi"]}
                best_alt = {"lambda": lam, "module": alt["module"],
                            "total": alt["total"], "sd": alt["xi_sd"],
                            "p_win": mca["p_win"], "p_draw": mca["p_draw"],
                            "e_pts": mca["e_pts"],
                            "in": sorted(alt_names - base_names),
                            "out": sorted(base_names - alt_names)}
        rivals.append({"team": tname, "module": radv["module"],
                       "module_src": module_src,
                       "total": radv["total"],
                       "exp_total": radv.get("exp_total"),
                       "sd": radv["xi_sd"],
                       "p_win": p0, "p_draw": mc0["p_draw"],
                       "p_loss": mc0["p_loss"], "e_pts": mc0["e_pts"],
                       "n_missing": n_missing,
                       "meetings": meetings.get(tname, []),
                       "xi": [{"id": x.get("id"), "nome": x["nome"],
                               "R": x["R"], "team": x["team"],
                               "exp": x["exp"], "p_play": x["p_play"],
                               "p_play_src": x.get("p_play_src"),
                               "congested": x.get("congested", False)}
                              for x in radv["xi"]],
                       "alt": best_alt})
    rivals.sort(key=lambda r: (r["p_win"] is None, r["p_win"]))
    return {"generated_at": datetime.now(UTC).isoformat(), "round": rnd,
            "next_opponents": next_opps,
            "me": {"team": my_name, "module": base["module"],
                   "total": base["total"],
                   "exp_total": base.get("exp_total"), "sd": base["xi_sd"],
                   "congested": sorted({x["nome"] for x in base["xi"]
                                        if x.get("congested")})},
            "rivals": rivals}


def score_observed_xi(team: str, player_names: list[str],
                      module: str | None = None,
                      bench_names: list[str] | None = None) -> dict:
    """Value an opponent's ACTUALLY-FIELDED XI (names read from a Leghe
    screenshot) with the same levels/p_play machinery as the rival matrix,
    and price my current board against it.

    Restricting the rival's roster to the observed players and forcing the
    observed module makes advise() score exactly the fielded XI — exp_total,
    sd, defense modifier included — instead of our guess of their best XI.
    Returns their numbers, my P(win) vs THIS lineup, and the tilted
    alternative of mine that beats the base against it, if any.
    """
    from scripts.fantacalcio.lineup_check import _match_one, _tokens

    board = json.loads(BOARD.read_text())
    by_id = {int(p["id"]): p for p in board["players"]}
    fixtures, rnd = _next_fixtures()
    elo = _apply_market_elo(_current_elo(), fixtures)
    out = _out_ids(board["players"], fixtures)
    hist = _history()
    prob_by_pid = status_by_pid(fetch_probabili())
    avail = fetch_indisponibili()
    sf = fetch_sosfanta()
    rig = rigoristi_by_pid(fetch_rigoristi())
    congestion = _club_congestion(fixtures)
    disc = discipline_status()

    league = json.loads(
        (ROOT / "data" / "fantacalcio" / "league_rosters.json").read_text())
    entry = league.get("teams", {}).get(team)
    if entry is None:
        low = team.casefold()
        hits = [t for t in league.get("teams", {})
                if low in t.casefold() or t.casefold() in low]
        if len(hits) == 1:
            team, entry = hits[0], league["teams"][hits[0]]
        else:
            return {"error": f"squadra '{team}' non trovata in lega",
                    "teams": sorted(league.get("teams", {}))}

    rows = _rival_roster(entry, by_id, hist, prob_by_pid)
    for row in rows:
        c = congestion.get(row["team"])
        if c:
            row["rest_d"] = c["rest_d"]
            row["congested"] = c["congested"]
            row["last_comp"] = c["last_comp"]
    _apply_discipline(rows, out, disc)
    _apply_rigoristi(rows, rig)
    _apply_scorer(rows, _scorer_by_pid())
    _apply_availability(rows, avail, sf=sf)

    def _norm(s: str) -> str:
        import unicodedata
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return "".join(c for c in s.casefold() if c.isalnum())

    # Leghe screenshots carry the SAME listone names as our roster, so an
    # exact (accent/punct-insensitive) match covers ~all; the token matcher
    # is the fallback for OCR quirks — it alone misses initial-form names
    # ("Keita M.") when BOTH sides carry the initial.
    by_norm = {_norm(r["nome"]): i for i, r in enumerate(rows)}
    roster_toks = [_tokens(r["nome"]) for r in rows]
    matched, unmatched = [], []
    seen: set[int] = set()
    for nm in player_names:
        idx = by_norm.get(_norm(nm))
        if idx is None:
            idx = _match_one(_tokens(nm), roster_toks)
        if idx is None or idx in seen:
            unmatched.append(nm)
        else:
            seen.add(idx)
            matched.append(rows[idx])

    mod_tuple = None
    if module:
        digs = [int(c) for c in str(module) if c.isdigit()]
        if len(digs) == 3 and sum(digs) == 10:
            mod_tuple = tuple(digs)
    if mod_tuple is None:
        counts = {"D": 0, "C": 0, "A": 0}
        for r in matched:
            if r["R"] in counts:
                counts[r["R"]] += 1
        if sum(counts.values()) == 10:
            mod_tuple = (counts["D"], counts["C"], counts["A"])

    obs_bench = []
    for nm in bench_names or []:
        idx = by_norm.get(_norm(nm))
        if idx is None:
            idx = _match_one(_tokens(nm), roster_toks)
        if idx is not None and idx not in seen:
            seen.add(idx)
            obs_bench.append(rows[idx])

    radv = advise(matched, fixtures, elo, out,
                  modules=[mod_tuple] if mod_tuple else None)
    if not radv.get("xi"):
        radv = advise(matched, fixtures, elo, out)
    if not radv.get("xi"):
        return {"error": "XI avversario non valutabile "
                         f"({len(matched)}/11 nomi abbinati)",
                "unmatched": unmatched}
    if obs_bench:
        # _role_recovery reads row["exp"], which raw roster rows lack —
        # apply the same fixture terms advise() uses, keeping screenshot
        # order (the real auto-sub order).
        enriched = []
        for pl in obs_bench:
            fx = fixtures.get(pl["team"])
            if fx is None or out.get(pl["id"]):
                continue
            d_elo = (elo.get(pl["team"], 1450.0)
                     - elo.get(fx["opp"], 1450.0)) / 100.0
            adj = HOME_ADJ[pl["R"]] * fx["home"] + ELO_SLOPE[pl["R"]] * d_elo
            enriched.append({**pl, "exp": round(pl["level"] + adj, 2),
                             "p_play": round(float(pl.get("p_play", 1.0)), 2)})
        bev = _bench_recovery_ev(radv["xi"], enriched)
        radv["bench_ev"] = round(bev, 2)
        radv["exp_total"] = round(radv["total"] + bev, 2)
    else:
        # No bench visible: compare XI-vs-XI so neither side gets a
        # phantom bench advantage.
        radv["exp_total"] = radv["total"]

    # My side — same rebuild as build_rivals, priced vs THIS actual lineup.
    my_roster, _src = _my_roster(
        by_id, prob_by_pid, [int(sq["id"]) for sq in board["squad"]])
    for row in my_roster:
        c = congestion.get(row["team"])
        if c:
            row["rest_d"] = c["rest_d"]
            row["congested"] = c["congested"]
            row["last_comp"] = c["last_comp"]
    _apply_discipline(my_roster, out, disc)
    _apply_rigoristi(my_roster, rig)
    _apply_scorer(my_roster, _scorer_by_pid())
    _apply_availability(my_roster, avail, _news_caps(), sf=sf)
    base = advise(my_roster, fixtures, elo, out)
    base_names = {x["nome"] for x in base["xi"]}
    # Like-for-like benches: when their bench is invisible, mine is muted
    # too so neither side carries a phantom recovery advantage in the sim.
    their = dict(radv) if obs_bench else {**radv, "bench": []}
    my_base = base if obs_bench else {**base, "bench": []}
    mc0 = h2h_mc(my_base, their, seed=_mc_seed(rnd, team))
    p0 = mc0["p_win"]
    best_alt = None
    for lam in RISK_LAMBDAS:
        alt = advise(my_roster, fixtures, elo, out, risk_lambda=lam)
        alt_names = {x["nome"] for x in alt["xi"]}
        if alt_names == base_names:
            continue
        my_alt = alt if obs_bench else {**alt, "bench": []}
        mca = h2h_mc(my_alt, their, seed=_mc_seed(rnd, team, lam))
        if mca["e_pts"] >= mc0["e_pts"] + 3 * ALT_MIN_GAIN                 and (best_alt is None or mca["e_pts"] > best_alt["e_pts"]):
            best_alt = {"module": alt["module"], "total": alt["total"],
                        "p_win": mca["p_win"], "p_draw": mca["p_draw"],
                        "e_pts": mca["e_pts"],
                        "in": sorted(alt_names - base_names),
                        "out": sorted(base_names - alt_names)}
    return {
        "round": rnd, "team": team,
        "opponent": {"module": radv["module"], "total": radv["total"],
                     "exp_total": radv.get("exp_total"), "sd": radv["xi_sd"],
                     "bench_ev": radv.get("bench_ev"),
                     "bench_seen": len(obs_bench),
                     "n_matched": len(matched), "unmatched": unmatched,
                     "xi": [{"nome": x["nome"], "R": x["R"],
                             "exp": x["exp"], "p_play": x["p_play"]}
                            for x in radv["xi"]]},
        "me": {"module": base["module"], "total": base["total"],
               "exp_total": base.get("exp_total"), "sd": base["xi_sd"],
               "xi": sorted(base_names)},
        "p_win": p0, "p_draw": mc0["p_draw"], "p_loss": mc0["p_loss"],
        "e_pts": mc0["e_pts"],
        "alt": best_alt,
    }


def build_svincolati(top_n: int = 8) -> dict:
    """Unowned listone players worth watching, ranked by p_play * level.

    The other half of the listone nobody in the league owns: enriched with the
    SAME machinery as any roster (live levels from the voti, probabili
    titolarità, discipline), so a nobody who just became a starter surfaces
    the day the probabili say so. `upgrade_over` names my weakest same-role
    player he'd beat on level — the pickup-worth-a-svincolo signal."""
    board = json.loads(BOARD.read_text())
    by_id = {int(p["id"]): p for p in board["players"]}
    league = json.loads(
        (ROOT / "data" / "fantacalcio" / "league_rosters.json").read_text())
    owned = {int(r["id"]) for t in league["teams"].values()
             for r in t.get("roster", [])}
    fixtures, rnd = _next_fixtures()
    hist = _history()
    prob_by_pid = status_by_pid(fetch_probabili())
    avail = fetch_indisponibili()
    sf = fetch_sosfanta()
    rig = rigoristi_by_pid(fetch_rigoristi())
    # DEPARTED = left Serie A (board status from the wiki-transfers pass) —
    # a free agent you cannot field is not a pickup (Di Gregorio lesson,
    # 2026-09-02: the radar led with a keeper already at Bournemouth).
    free_ids = [pid for pid in by_id if pid not in owned
                and by_id[pid].get("status") != "DEPARTED"]
    rows = _rival_roster({"roster": [{"id": i} for i in free_ids]},
                         by_id, hist, prob_by_pid)
    out = _out_ids(board["players"], fixtures)
    disc = discipline_status()
    _apply_discipline(rows, out, disc)
    _apply_rigoristi(rows, rig)
    _apply_scorer(rows, _scorer_by_pid())
    _apply_availability(rows, avail, sf=sf)

    # my weakest per role (by level), for the upgrade comparison
    team = json.loads((ROOT / "data" / "fantacalcio" / "my_team.json").read_text())
    my_rows = _rival_roster({"roster": team["roster"]}, by_id, hist, prob_by_pid)
    # Compare on p_play * level (the ranking currency): a raw-level compare
    # would let a 15% third keeper's unplayed 6.00 prior "beat" a measured
    # starter — the prior-vs-observed apples-to-oranges this repo keeps paying
    # for. Same basis both sides, no mixed currencies.
    my_worst = {}
    for ro in "PDCA":
        mine = [r for r in my_rows if r["R"] == ro]
        if mine:
            my_worst[ro] = min(mine, key=lambda r: r["p_play"] * r["level"])

    picks: dict[str, list] = {}
    for ro in "PDCA":
        pool = [r for r in rows if r["R"] == ro and int(r["id"]) not in out
                and r["p_play"] >= 0.10]
        pool.sort(key=lambda r: -(r["p_play"] * r["level"]))
        sel = []
        for r in pool[:top_n]:
            w = my_worst.get(ro)
            sel.append({"id": r["id"], "nome": r["nome"], "team": r["team"],
                        "level": round(r["level"], 2),
                        "p_play": round(r["p_play"], 2),
                        "p_play_src": r["p_play_src"],
                        "score": round(r["p_play"] * r["level"], 2),
                        "diffidato": r.get("diffidato", False),
                        "upgrade_over": w["nome"]
                        if w and r["p_play"] * r["level"]
                        > w["p_play"] * w["level"] + 0.1 else None})
        picks[ro] = sel
    return {"generated_at": datetime.now(UTC).isoformat(), "round": rnd,
            "n_free": len(free_ids), "picks": picks}


def main() -> None:
    if "--fielded" in sys.argv:
        i = sys.argv.index("--fielded")
        rnd = int(sys.argv[sys.argv.index("--round") + 1]) \
            if "--round" in sys.argv else _next_fixtures()[1]
        for pair in sys.argv[i + 1].split(","):
            team, mod = (x.strip() for x in pair.split("="))
            record_fielded(team, mod, rnd)
            print(f"registrato: {team} {mod} (giornata {rnd})")
        return
    out = build_advice()
    print(json.dumps({k: out[k] for k in ("round", "module", "total", "modifier")},
                     indent=1))
    for x in out["xi"]:
        print(f"  {x['R']} {x['nome']:16s} exp={x['exp']:5.2f} st={x['p_play']:.0%} "
              f"slot={x['exp_slot']:5.2f} "
              f"({'home' if x['home'] else 'away'} vs {x['opp']}, adj {x['fix_adj']:+.2f})")


if __name__ == "__main__":
    main()
