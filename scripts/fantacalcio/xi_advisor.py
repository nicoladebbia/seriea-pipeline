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
    P_DOUBT,
    P_NEWS_CAP,
    P_OUT,
    P_SUSPENDED,
    fetch_indisponibili,
    fetch_probabili,
    p_play_override,
    status_by_pid,
)
from scripts.fantacalcio.tracker import LEVEL_K, MODULES, SEASON, _modifier, discipline_status

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
    _HISTORY = {"sd": sd, "live": live, "prev": prev,
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
           risk_lambda: float = 0.0,
           modules: list[tuple[int, int, int]] | None = None) -> dict:
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
               "n_rounds": n_seen}
        if p.get("status") == "DEPARTED":
            row.update(p_play=0.02, p_play_src="departed", departed=True)
        roster_src.append(row)
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
    avail = fetch_indisponibili()
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
    _apply_availability(roster_src, avail, _news_caps())

    adv = advise(roster_src, fixtures, elo, out)
    kicks = [fixtures[t_]["ts"] for t_ in fixtures if fixtures[t_].get("ts")]
    return {"generated_at": datetime.now(UTC).isoformat(),
            "round": rnd, "source": source,
            "first_kickoff": min(kicks) if kicks else None,
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


def _apply_availability(rows: list, avail: dict | None,
                        news_caps: dict[str, str] | None = None) -> None:
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
        if r.get("p_play_src") in ("probabili", "ballottaggio"):
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
    elo = _current_elo()
    out = _out_ids(board["players"], fixtures)
    hist = _history()
    prob_by_pid = status_by_pid(fetch_probabili())
    avail = fetch_indisponibili()

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
    _apply_availability(my_roster, avail, _news_caps())
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
        _apply_availability(rows, avail)
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
                       "module_src": module_src,
                       "total": radv["total"], "sd": radv["xi_sd"],
                       "p_win": round(p0, 3), "n_missing": n_missing,
                       "meetings": meetings.get(tname, []),
                       "xi": [{"nome": x["nome"], "R": x["R"], "team": x["team"],
                               "exp": x["exp"], "p_play": x["p_play"],
                               "p_play_src": x.get("p_play_src"),
                               "congested": x.get("congested", False)}
                              for x in radv["xi"]],
                       "alt": best_alt})
    rivals.sort(key=lambda r: (r["p_win"] is None, r["p_win"]))
    return {"generated_at": datetime.now(UTC).isoformat(), "round": rnd,
            "next_opponents": next_opps,
            "me": {"team": my_name, "module": base["module"],
                   "total": base["total"], "sd": base["xi_sd"],
                   "congested": sorted({x["nome"] for x in base["xi"]
                                        if x.get("congested")})},
            "rivals": rivals}


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
    _apply_availability(rows, avail)

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
