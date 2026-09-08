"""Market promotion gate: a paper market earns real stakes by its settled record.

Nicola, 2026-09-05: the system should bet 3-4 lines a match (player shots,
first-half angles, goalscorers) with real money. The constraint he set five
days earlier still holds: a market earns real stakes only after a settled
paper record. Measured facts behind the gate: props at real money ran -54%
ROI (2026-08-31), anytime scorer has no skill over the base rate (2026-06),
and the picks paper journal held ZERO settled bets when this was written.
So the gate, not a person, decides — and every market starts on paper.

Flow
  picks.build_picks (T-30)  -> paper entry in picks_journal (flat PAPER_STAKE)
                            -> if the market is PROMOTED: a real entry in
                               bet_journal too (Kelly-sized, capped), linked by
                               extra.picks_ref
  picks.settle_picks        -> grades the paper entry, settles the linked real
                               entry with the SAME outcome (same grader, same
                               voids), then evaluate_promotions() re-reads the
                               records and rewrites market_promotion.json
  results_fetcher.settle_bets skips real entries carrying picks_ref: its
  full-time grader defaults an unknown market to "lost".

The bar (PROMOTION_BAR) is per market key (player_shots_on_target,
double_chance_h1, ...): >= 50 settled paper bets, ROI > 0, z-score of the
per-bet return >= 1.0, and CLV > 0 when >= 20 real closing prices exist
(props have no Pinnacle; CLV comes from the last pre-kickoff refresh, so it
is often unmeasurable and then not required). Demotion: a promoted market
with >= 30 settled REAL bets and real ROI < -10% or z < -1 goes back to
paper; re-promotion needs a fresh 50 paper bets placed after the demotion.

Incumbents (INCUMBENT_MARKETS). O/U Over 1.5 / 2.5 bet real money before this
gate existed and were never asked to pass it. Measured on the real journal
the day after the bar was set (2026-09-06): O/U 1.5 Over n=47, ROI +5.3%,
z +0.59, CLV +2.4%; O/U 2.5 Over n=44, ROI -1.8%, z -0.10, CLV +4.7%. Neither
clears the bar it imposes on every prop, and the record ends 2026-05-17 on a
model that has since been refit (real bets since go-live: 0). So the gate
scores them against the SAME bar on their REAL record, keeps a separate
since-go-live record for the model actually betting, and says so on the
/record card. It does not switch them off: betting_unified owns the market
config and that is Nicola's call, taken knowing the number.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev

from config.settings import DATA_DIR

log = logging.getLogger(__name__)

STATE_PATH = DATA_DIR / "betting" / "market_promotion.json"

PROMOTION_BAR = {"min_settled": 50, "min_roi_pct": 0.0, "min_z": 2.5,
                 "min_clv_pct": 0.0, "min_clv_n": 20}
DEMOTION_BAR = {"min_real_bets": 30, "max_roi_pct": -10.0, "max_z": -1.0}

# Real stake on a promoted market: the O/U Kelly (0.15) halved and a lower
# cap, until the market has its own real record. Edge over-claim is the
# measured failure mode on props (+9.9% claimed -> +1.8% realised elsewhere).
PROMOTED_KELLY_SCALE = 0.5
PROMOTED_MAX_STAKE_PCT = 1.5
PROMOTED_MIN_STAKE_PCT = 0.2
PIPELINE_STATUS = "pick:promoted"

# Markets that bet real money WITHOUT passing the bar — the incumbents when
# the gate was written (2026-09-05). key -> (journal market, selection side).
# Scored against PROMOTION_BAR on their real record; never promoted (already
# real) and never demoted here (betting_unified owns the switch).
INCUMBENT_MARKETS = {"ou_over_1_5": ("O/U 1.5", "Over"), "ou_over_2_5": ("O/U 2.5", "Over")}
# Serie A go-live. The O/U model has been refit since the incumbent record was
# built, so the since-go-live record is the one on the model actually betting.
INCUMBENT_LIVE_FROM = "2026-08-27T00:00:00+00:00"
# Stake ladder for an incumbent (2026-09-06, closing the Kelly 0.15 question):
# the engine stakes it at PROMOTED_KELLY_SCALE x Kelly and x cap — what a
# freshly promoted market gets — until its since-go-live real record clears
# INCUMBENT_FULL_STAKE_BAR. The record decides, not a hand-set fraction.
#
# THE QUALITY LEG IS CLV, NOT RETURN (Nicola's call, 2026-09-08). Both measure
# the same claim — "these bets are priced better than the market" — but the
# return needs the coin to land and CLV does not, so their sample efficiency is
# not comparable. Measured that day on the live journal, same bets:
#
#   market          ROI z   n for ROI z>=2.5   CLV z   n for CLV z>=2.5
#   O/U 1.5 Over    +0.68        ~648          +9.98          ~3
#   O/U 2.5 Over    -0.10        never        +16.11          ~1
#
# A return-z leg at n=30 needed a +28.1% ROI run to clear; simulated on the real
# odds mix it fired 7.4% of the time, and it got WORSE with volume, because the
# only way a thin book clears it is luck. That is a gate that never opens, which
# is not a strict gate — it is a broken one.
#
# ROI > 0 stays as a FLOOR, deliberately. CLV says the price was good; it does
# not say the market made money, and full stake on a market that is beating the
# close while losing is not a trade anyone wants. Both must hold.
#
# The CLV legs are REQUIRED here, not conditional. PROMOTION_BAR waives CLV
# below min_clv_n because a queueing paper market may have no closing prices at
# all; for this ladder CLV IS the evidence, so too few prices BLOCKS. Fail
# closed — see full_stake_misses.
INCUMBENT_FULL_STAKE_MIN_N = DEMOTION_BAR["min_real_bets"]
INCUMBENT_FULL_STAKE_BAR = {
    "min_settled": INCUMBENT_FULL_STAKE_MIN_N,
    "min_roi_pct": PROMOTION_BAR["min_roi_pct"],   # the floor, not the quality leg
    # The quality leg reads the sharp line's MOVEMENT between our entry and the
    # close, not beat-the-close. Beat-the-close is ~80% the same-moment spread
    # between our best-of-N price and the one sharp book the close is read from
    # (2026-09-08: 0 of 48 and 0 of 38 O/U rows negative) — a restatement of the
    # engine's own entry edge, so gating on it is the engine agreeing with
    # itself, and its near-zero variance sends the t-statistic to +16 on 39
    # bets. Movement is the half that says whether the market later came to our
    # side. It needs the SAME book at both ends (`clv_move_pct`), so a row whose
    # closing price came from a market mean or an untagged capture is excluded.
    "min_clv_move_pct": PROMOTION_BAR["min_clv_pct"],
    "min_clv_move_n": PROMOTION_BAR["min_clv_n"],
    "min_clv_move_z": PROMOTION_BAR["min_z"],      # the same 2.5, read off movement
}

# Market key -> what the bet is, for the /record card
MARKET_NAMES_IT = {
    "ou_over_1_5": "Over 1.5 (motore)", "ou_over_2_5": "Over 2.5 (motore)",
    "player_shots": "Tiri totali giocatore", "player_shots_on_target": "Tiri in porta giocatore",
    "player_goal_scorer_anytime": "Marcatore", "player_assists": "Assist giocatore",
    "h2h_h1": "1° tempo 1x2", "totals_h1": "1° tempo under/over", "btts_h1": "Goal 1° tempo",
    "double_chance_h1": "Doppia chance 1° tempo", "halftime_fulltime": "Primo tempo / Finale",
    "correct_score": "Risultato esatto", "h2h": "1x2 finale", "totals": "Under/over",
    "double_chance": "Doppia chance", "btts": "Goal",
}


# ---------------------------------------------------------------------------
# Null simulation — what the bar does to a market with NO edge
# ---------------------------------------------------------------------------
def null_simulation(*, n_markets: int = 14, max_n: int = 300, sims: int = 4000,
                    odds: float = 2.0, edge: float = 0.0, seed: int = 0,
                    bar: dict = PROMOTION_BAR, demotion: dict = DEMOTION_BAR,
                    real_n: int = 120) -> dict:
    """Monte Carlo of the bar AS IT IS EVALUATED: after every settlement, from
    `min_settled` on, promote the first time ROI > 0 and z >= min_z both hold
    (the CLV leg is left out — paper CLV exists only where a closing price
    does). `edge` is the true expected return per unit stake (0.0 = fair
    price; a bookmaker margin is a negative edge, e.g. -0.05). Returns, per
    market: the single-look pass rate at exactly `min_settled` bets, the
    first-crossing pass rate anywhere in [min_settled, max_n], the median n
    at promotion, and — for a promoted market that then takes real stakes at
    the same edge — the share demoted within `real_n` real bets under the
    demotion bar. `n_markets` markets share the null, so the any-market
    figure is 1 - (1 - p)^n_markets."""
    import numpy as np
    if max_n < bar["min_settled"] or real_n < demotion["min_real_bets"]:
        raise ValueError("max_n / real_n must reach the bars' minimum counts")
    rng = np.random.default_rng(seed)
    win_p = (1.0 + edge) / odds
    win = rng.random((sims, max_n)) < win_p
    r = np.where(win, odds - 1.0, -1.0)            # unit return per bet
    n = np.arange(1, max_n + 1)
    cs, cs2 = np.cumsum(r, axis=1), np.cumsum(r * r, axis=1)
    mu = cs / n
    var = np.maximum(cs2 / n - mu * mu, 0.0)       # pstdev, as market_record
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, mu / (sd / np.sqrt(n)), 0.0)
    ok = (n >= bar["min_settled"]) & (mu * 100 > bar["min_roi_pct"]) & (z >= bar["min_z"])
    k = bar["min_settled"] - 1
    single = float(ok[:, k].mean())
    first = ok.any(axis=1)
    first_rate = float(first.mean())
    n_at = np.where(first, ok.argmax(axis=1) + 1, 0)
    median_n = float(np.median(n_at[first])) if first.any() else None
    # real leg: a promoted null market keeps real stakes until the demotion bar trips
    rw = rng.random((sims, real_n)) < win_p
    rr = np.where(rw, odds - 1.0, -1.0)
    rn = np.arange(1, real_n + 1)
    rmu = np.cumsum(rr, axis=1) / rn
    rvar = np.maximum(np.cumsum(rr * rr, axis=1) / rn - rmu * rmu, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rz = np.where(rvar > 0, rmu / (np.sqrt(rvar) / np.sqrt(rn)), 0.0)
    dem = (rn >= demotion["min_real_bets"]) & ((rmu * 100 < demotion["max_roi_pct"]) | (rz < demotion["max_z"]))
    demoted_rate = float(dem.any(axis=1).mean())
    return {"odds": odds, "edge": edge, "n_markets": n_markets, "max_n": max_n, "sims": sims,
            "single_look_at_min": round(single, 4),
            "first_crossing_by_max_n": round(first_rate, 4),
            "median_n_at_promotion": median_n,
            "any_of_n_markets_promoted": round(1 - (1 - first_rate) ** n_markets, 4),
            "expected_false_promotions": round(first_rate * n_markets, 2),
            "demoted_within_real_n": round(demoted_rate, 4), "real_n": real_n,
            "bar": dict(bar), "demotion_bar": dict(demotion)}


def _null_sim_report(args: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="market_promotion --null-sim")
    ap.add_argument("--markets", type=int, default=len(MARKET_NAMES_IT))
    ap.add_argument("--max-n", type=int, default=300)
    ap.add_argument("--sims", type=int, default=4000)
    ap.add_argument("--odds", type=float, nargs="+", default=[1.5, 2.0, 4.0])
    ap.add_argument("--edge", type=float, nargs="+", default=[0.0, -0.05, 0.05])
    a = ap.parse_args(args)
    print(f"bar={PROMOTION_BAR} demotion={DEMOTION_BAR} markets={a.markets} max_n={a.max_n} sims={a.sims}")
    print(f"{'odds':>5} {'edge':>6} {'single@min':>10} {'first-cross':>11} {'median n':>8} "
          f"{'any-of-K':>8} {'E[false]':>8} {'demoted':>8}")
    for o in a.odds:
        for e in a.edge:
            r = null_simulation(n_markets=a.markets, max_n=a.max_n, sims=a.sims, odds=o, edge=e)
            print(f"{o:>5.2f} {e:>+6.2f} {r['single_look_at_min']:>10.3f} {r['first_crossing_by_max_n']:>11.3f} "
                  f"{str(r['median_n_at_promotion']):>8} {r['any_of_n_markets_promoted']:>8.3f} "
                  f"{r['expected_false_promotions']:>8.2f} {r['demoted_within_real_n']:>8.3f}")
    return 0


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def _unit_returns(bets: list[dict]) -> list[float]:
    out = []
    for b in bets:
        stake = float(b.get("stake") or 0)
        if stake > 0 and b.get("status") in ("won", "lost", "push"):
            out.append(float(b.get("profit") or 0) / stake)
    return out


def market_record(bets: list[dict], since: str | None = None) -> dict[str, dict]:
    """Per market key: n settled (won/lost/push; voids excluded), won, ROI,
    z-score of the per-bet return (mean / (std / sqrt n)), mean CLV and its n.
    `since` (ISO) keeps only bets placed at or after it — the fresh record a
    demoted market must build."""
    by: dict[str, dict] = {}
    for b in bets:
        if since and str(b.get("placed_at") or "") < since:
            continue
        if b.get("status") not in ("won", "lost", "push"):
            continue
        m = by.setdefault(b.get("market") or "?", {"bets": [], "clv": [], "move": []})
        m["bets"].append(b)
        if b.get("clv_pct") is not None:
            m["clv"].append(float(b["clv_pct"]))
        if b.get("clv_move_pct") is not None:
            m["move"].append(float(b["clv_move_pct"]))
    out: dict[str, dict] = {}
    for mk, m in by.items():
        r = _unit_returns(m["bets"])
        n = len(r)
        mu = mean(r) if r else 0.0
        sd = pstdev(r) if n > 1 else 0.0
        z = (mu / (sd / n ** 0.5)) if n > 1 and sd > 0 else 0.0
        # Beat-the-close and the sharp line's MOVEMENT both get the return's
        # t-statistic, and the ladder reads the second one. Measured 2026-09-08
        # by decomposing the live journal: of O/U 1.5 Over's +2.61% CLV, +2.07%
        # is the same-moment spread between our best-of-N price and Pinnacle
        # (0 of 48 negative — we always shop) and -0.04% is the line actually
        # moving. The spread term is the engine's own entry edge restated, so
        # its z grows without bound in n (+9.98 at n=48, +16.11 at n=39) and a
        # gate on it would open for any market where we shop books. The
        # movement term has real negative mass (12/48, 8/38) and z -0.30 / +2.96.
        nc, sdc = len(m["clv"]), (pstdev(m["clv"]) if len(m["clv"]) > 1 else 0.0)
        clv_z = (mean(m["clv"]) / (sdc / nc ** 0.5)) if nc > 1 and sdc > 0 else 0.0
        nm, sdm = len(m["move"]), (pstdev(m["move"]) if len(m["move"]) > 1 else 0.0)
        move_z = (mean(m["move"]) / (sdm / nm ** 0.5)) if nm > 1 and sdm > 0 else 0.0
        out[mk] = {"n": n, "won": sum(b.get("status") == "won" for b in m["bets"]),
                   "roi_pct": round(mu * 100, 1), "z": round(z, 2),
                   "profit": round(sum(float(b.get("profit") or 0) for b in m["bets"]), 2),
                   "mean_clv_pct": round(mean(m["clv"]), 2) if m["clv"] else None,
                   "n_clv": nc, "clv_z": round(clv_z, 2),
                   "mean_clv_move_pct": round(mean(m["move"]), 2) if m["move"] else None,
                   "n_clv_move": nm, "clv_move_z": round(move_z, 2)}
    return out


def passes_bar(rec: dict, bar: dict = PROMOTION_BAR) -> tuple[bool, str]:
    """(passes, reason). The reason names the FIRST unmet condition so the card
    can say how far a market is from real money."""
    if rec["n"] < bar["min_settled"]:
        return False, f"{rec['n']}/{bar['min_settled']} settled"
    if rec["roi_pct"] <= bar["min_roi_pct"]:
        return False, f"ROI {rec['roi_pct']:+.1f}% (needs > {bar['min_roi_pct']:.0f}%)"
    if rec["z"] < bar["min_z"]:
        return False, f"z {rec['z']:.2f} (needs >= {bar['min_z']:.1f})"
    if rec["n_clv"] >= bar["min_clv_n"] and (rec["mean_clv_pct"] or 0) <= bar["min_clv_pct"]:
        return False, f"CLV {rec['mean_clv_pct']:+.2f}% on {rec['n_clv']} (needs > 0)"
    return True, "bar cleared"


def bar_misses(rec: dict, bar: dict = PROMOTION_BAR) -> list[str]:
    """EVERY unmet condition, for a record that is not queueing for the bar
    but being measured against it (the incumbents)."""
    out = []
    if rec["n"] < bar["min_settled"]:
        out.append(f"{rec['n']}/{bar['min_settled']} settled")
    if rec["roi_pct"] <= bar["min_roi_pct"]:
        out.append(f"ROI {rec['roi_pct']:+.1f}%")
    if rec["z"] < bar["min_z"]:
        out.append(f"z {rec['z']:.2f} < {bar['min_z']:.1f}")
    if rec["n_clv"] >= bar["min_clv_n"] and (rec["mean_clv_pct"] or 0) <= bar["min_clv_pct"]:
        out.append(f"CLV {rec['mean_clv_pct']:+.2f}%")
    return out


def full_stake_misses(rec: dict, bar: dict = INCUMBENT_FULL_STAKE_BAR) -> list[str]:
    """EVERY unmet condition for an incumbent's FULL stake — a different gate
    from `bar_misses`, which is admission to real money.

    Two deliberate differences. The quality leg is the t-statistic of the sharp
    line's MOVEMENT, not the return's and not beat-the-close's (see
    INCUMBENT_FULL_STAKE_BAR for why). And the movement legs are REQUIRED rather
    than waived below their n: a market with no same-book closing prices has not
    produced the evidence this gate reads, so it stays on the half stake instead
    of passing by absence."""
    out = []
    if rec["n"] < bar["min_settled"]:
        out.append(f"{rec['n']}/{bar['min_settled']} settled")
    if rec["roi_pct"] <= bar["min_roi_pct"]:
        out.append(f"ROI {rec['roi_pct']:+.1f}%")
    if rec.get("n_clv_move", 0) < bar["min_clv_move_n"]:
        # entry AND close from the SAME named sharp book — one end alone says
        # nothing about movement
        out.append(f"{rec.get('n_clv_move', 0)}/{bar['min_clv_move_n']} same-book entry+close")
    else:
        if (rec.get("mean_clv_move_pct") or 0) <= bar["min_clv_move_pct"]:
            out.append(f"line move {rec['mean_clv_move_pct']:+.2f}%")
        if (rec.get("clv_move_z") or 0) < bar["min_clv_move_z"]:
            out.append(f"line-move z {rec.get('clv_move_z') or 0:.2f} < {bar['min_clv_move_z']:.1f}")
    return out


_EMPTY_REC = {"n": 0, "won": 0, "roi_pct": 0.0, "z": 0.0, "profit": 0.0,
              "mean_clv_pct": None, "n_clv": 0, "clv_z": 0.0,
              "mean_clv_move_pct": None, "n_clv_move": 0, "clv_move_z": 0.0}


def incumbent_records(real_settled: list[dict], *, live_from: str = INCUMBENT_LIVE_FROM,
                      incumbents: dict = INCUMBENT_MARKETS) -> dict[str, dict]:
    """The real-money record of each incumbent market (engine bets: no `extra`,
    not a promoted mirror), scored against PROMOTION_BAR exactly as a paper
    market would be, plus the same record restricted to bets placed since
    `live_from`. `bar_passed` / `distance` say whether the incumbent would
    clear the bar it imposes on the props; `would_demote` applies the
    demotion bar to the real record. Neither changes where money goes."""
    out: dict[str, dict] = {}
    for key, (market, side) in incumbents.items():
        bets = [dict(b, market=key) for b in real_settled
                if b.get("market") == market and str(b.get("selection") or "").startswith(side)
                and not b.get("extra") and b.get("pipeline_status") != PIPELINE_STATUS]
        rec = market_record(bets).get(key) or dict(_EMPTY_REC)
        since = market_record(bets, since=live_from).get(key) or dict(_EMPTY_REC)
        misses = bar_misses(rec)
        ok, why = (not misses), ("bar cleared" if not misses else "; ".join(misses))
        demote, dwhy = should_demote(rec)
        dates = sorted(str(b.get("placed_at") or "")[:10] for b in bets if b.get("placed_at"))
        scale, scale_why = _stake_scale_from_since(since)
        out[key] = {"status": "incumbent", "market": market, "side": side, "real": rec,
                    "real_since_live": since, "live_from": live_from,
                    "record_span": [dates[0], dates[-1]] if dates else None,
                    "bar_passed": ok, "distance": why, "would_demote": demote, "demotion_reason": dwhy,
                    "stake_scale": scale, "stake_reason": scale_why}
    return out


def _stake_scale_from_since(since: dict) -> tuple[float, str]:
    """(multiplier, reason) for an incumbent, from its since-go-live record.
    Full stake needs that record to clear INCUMBENT_FULL_STAKE_BAR: enough
    settled bets, ROI > 0 as a floor, and a sharp-line-movement record that is
    positive and significant. Anything short stays on the half a freshly promoted market
    gets. The demotion-bar reason
    is kept separate from the short-of-the-bar one because they are different
    sizes of bad and the card says which."""
    n = since.get("n", 0)
    if n < INCUMBENT_FULL_STAKE_MIN_N:
        return PROMOTED_KELLY_SCALE, f"since go-live {n}/{INCUMBENT_FULL_STAKE_MIN_N} settled"
    demote, why = should_demote(since)
    if demote:
        return PROMOTED_KELLY_SCALE, f"since go-live record at the demotion bar: {why}"
    misses = full_stake_misses(since)
    if misses:
        return PROMOTED_KELLY_SCALE, f"since go-live n={n} short of the bar: {'; '.join(misses)}"
    return 1.0, (f"since go-live n={n} ROI {since['roi_pct']:+.1f}% "
                 f"line move {since.get('mean_clv_move_pct') or 0:+.2f}% "
                 f"z {since.get('clv_move_z') or 0:+.2f} — bar cleared")


def incumbent_stake_scale(market: str, selection: str, state: dict | None = None) -> tuple[float, str]:
    """(Kelly-and-cap multiplier, reason) for an engine bet. 1.0 for anything
    that is not an incumbent; for an incumbent, what its evaluated record says
    (`stake_scale`). No evaluated record yet = PROMOTED_KELLY_SCALE: fail
    closed, the same half stake a promoted market starts on."""
    key = next((k for k, (m, side) in INCUMBENT_MARKETS.items()
                if m == market and str(selection or "").startswith(side)), None)
    if key is None:
        return 1.0, ""
    st = state if state is not None else load_state()
    row = (st.get("incumbents") or {}).get(key)
    if not row or row.get("stake_scale") is None:
        return PROMOTED_KELLY_SCALE, "incumbent record not evaluated yet"
    return float(row["stake_scale"]), str(row.get("stake_reason") or "")


def should_demote(real_rec: dict | None, bar: dict = DEMOTION_BAR) -> tuple[bool, str]:
    if not real_rec or real_rec["n"] < bar["min_real_bets"]:
        return False, ""
    if real_rec["roi_pct"] < bar["max_roi_pct"]:
        return True, f"real ROI {real_rec['roi_pct']:+.1f}% on {real_rec['n']} bets"
    if real_rec["z"] < bar["max_z"]:
        return True, f"real z {real_rec['z']:.2f} on {real_rec['n']} bets"
    return False, ""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def load_state(path: Path | None = None) -> dict:
    p = path or STATE_PATH
    try:
        return json.loads(p.read_text()) if p.exists() else {"markets": {}}
    except (OSError, ValueError):
        return {"markets": {}}


def is_promoted(market_key: str, state: dict | None = None) -> bool:
    st = state if state is not None else load_state()
    return (st.get("markets") or {}).get(market_key, {}).get("status") == "promoted"


def evaluate_promotions(paper_settled: list[dict] | None = None, real_settled: list[dict] | None = None,
                        *, real_all: list[dict] | None = None, now: datetime | None = None,
                        path: Path | None = None, write: bool = True) -> dict:
    """Re-read both journals and rewrite the state. Idempotent. Every market
    seen in the paper journal gets a row; a promotion or demotion is a state
    transition with its record snapshot and reason, and is logged at WARNING
    because it changes where real money goes. `real_settled` is the promoted
    mirrors (demotion leg); `real_all` is the whole real journal, from which
    the incumbents' record is scored (state["incumbents"])."""
    from scripts.betting.bet_journal import get_settled_bets
    from scripts.betting.picks import PICKS_JOURNAL_PATH
    now = now or datetime.now(UTC)
    if paper_settled is None:
        paper_settled = get_settled_bets(journal_path=PICKS_JOURNAL_PATH)
    if real_settled is None or real_all is None:
        everything = get_settled_bets() if real_all is None else real_all
        if real_all is None:
            real_all = everything
        if real_settled is None:
            real_settled = [b for b in everything if b.get("pipeline_status") == PIPELINE_STATUS]
    state = load_state(path)
    # Transitions collected here and pushed AFTER the state write: a failed
    # notification must never cost us the state change it describes.
    transitions: list[dict] = []
    prev_scale = {k: (r or {}).get("stake_scale")
                  for k, r in (state.get("incumbents") or {}).items()}
    state["incumbents"] = incumbent_records(real_all)
    for key, row in state["incumbents"].items():
        # Compared against the PERSISTED scale, so a steady state never re-pushes
        # however often this runs. A market sitting on the demotion boundary CAN
        # oscillate 1.0 -> 0.5 -> 1.0 across settlements and get a card each time:
        # that is accepted deliberately, because every one of those flips halves
        # or doubles the money on the next slip. A stake change is never noise.
        was, now_scale = prev_scale.get(key), row.get("stake_scale")
        if was is not None and now_scale is not None and was != now_scale:
            transitions.append({
                "kind": "stake_up" if now_scale > was else "stake_down",
                "market": key, "reason": row.get("stake_reason"),
                "n": row["real_since_live"]["n"],
                "roi_pct": row["real_since_live"]["roi_pct"],
                "z": row["real_since_live"]["z"],
            })
        if not row["bar_passed"]:
            log.info("Incumbent %s bets real money without clearing the bar: %s (real n=%d, since go-live n=%d)",
                     key, row["distance"], row["real"]["n"], row["real_since_live"]["n"])
    markets: dict = state.setdefault("markets", {})
    real_by = market_record(real_settled)
    seen = {b.get("market") for b in paper_settled if b.get("market")}
    for mk in sorted(seen | set(markets)):
        row = markets.setdefault(mk, {"status": "paper", "since": now.isoformat(), "record_from": None})
        paper_rec = market_record(paper_settled, since=row.get("record_from")).get(mk) or \
            dict(_EMPTY_REC)
        row["paper"] = paper_rec
        row["real"] = real_by.get(mk)
        if row["status"] == "promoted":
            demote, why = should_demote(row["real"])
            if demote:
                row.update({"status": "paper", "since": now.isoformat(), "record_from": now.isoformat(),
                            "reason": f"demoted: {why}", "demoted_at": now.isoformat()})
                # the paper count restarts here: the record that promoted it is spent
                row["paper"] = dict(_EMPTY_REC)
                row["distance"] = passes_bar(row["paper"])[1]
                log.warning("Market %s DEMOTED to paper: %s", mk, why)
                transitions.append({"kind": "demoted", "market": mk, "reason": why,
                                    "n": (row.get("real") or {}).get("n"),
                                    "roi_pct": (row.get("real") or {}).get("roi_pct"),
                                    "z": (row.get("real") or {}).get("z")})
            else:
                row["distance"] = "promoted"
            continue
        ok, why = passes_bar(paper_rec)
        row["distance"] = why
        if ok:
            row.update({"status": "promoted", "since": now.isoformat(), "reason": f"promoted: {why}",
                        "promoted_at": now.isoformat(), "snapshot": dict(paper_rec)})
            log.warning("Market %s PROMOTED to real stakes on %d paper bets (ROI %+.1f%%, z %.2f)",
                        mk, paper_rec["n"], paper_rec["roi_pct"], paper_rec["z"])
            transitions.append({"kind": "promoted", "market": mk, "reason": why,
                                "n": paper_rec["n"], "roi_pct": paper_rec["roi_pct"],
                                "z": paper_rec["z"]})
    state["updated_at"] = now.isoformat()
    state["bar"] = PROMOTION_BAR
    state["demotion_bar"] = DEMOTION_BAR
    if write:
        from config.settings import atomic_write_json
        atomic_write_json(path or STATE_PATH, state)
        # After the write, and only on a real write: a promotion/demotion is
        # where real money starts or stops flowing. The gate is the product;
        # until 2026-09-07 it announced itself nowhere.
        if transitions:
            try:
                from scripts.pipeline.notify import notify_market_promotion
                notify_market_promotion(transitions)
            except Exception as e:
                log.error("Market-gate notification could not be sent: %s", e)
    return state


# ---------------------------------------------------------------------------
# Real stake on a promoted market
# ---------------------------------------------------------------------------
def promoted_stake(model_prob: float, odds: float, bankroll: float, *, kelly_fraction: float | None = None) -> float:
    """Kelly at the O/U fraction x PROMOTED_KELLY_SCALE, capped at
    PROMOTED_MAX_STAKE_PCT of bankroll; 0 below PROMOTED_MIN_STAKE_PCT."""
    from scripts.betting.betting_unified import BettingConfig, calculate_kelly
    kf = kelly_fraction if kelly_fraction is not None else BettingConfig().kelly_fraction
    if not bankroll or bankroll <= 0 or not odds or odds <= 1.0 or not (0 < model_prob < 1):
        return 0.0
    pct = min(calculate_kelly(model_prob, odds, fraction=kf * PROMOTED_KELLY_SCALE) * 100, PROMOTED_MAX_STAKE_PCT)
    if pct < PROMOTED_MIN_STAKE_PCT:
        return 0.0
    return round(bankroll * pct / 100, 2)


def journal_promoted(paper_entry: dict, picks_bet_id: str, *, bankroll: float | None = None) -> str | None:
    """Mirror a paper pick into the REAL journal at a Kelly stake. Returns the
    real bet id, or None when the stake rounds to nothing."""
    from scripts.betting.bet_journal import add_bet
    if bankroll is None:
        from scripts.betting.bankroll_loader import get_effective_bankroll
        bankroll = get_effective_bankroll()
    stake = promoted_stake(float(paper_entry.get("model_prob") or 0), float(paper_entry.get("odds") or 0), bankroll)
    if stake <= 0:
        return None
    entry = {k: v for k, v in paper_entry.items() if k not in ("stake", "pipeline_status", "extra")}
    entry["stake"] = stake
    entry["pipeline_status"] = PIPELINE_STATUS
    entry["extra"] = {**(paper_entry.get("extra") or {}), "picks_ref": picks_bet_id, "paper_stake": paper_entry.get("stake")}
    bet_id = add_bet(entry)
    log.warning("PROMOTED real bet %s: %s %s %s @ %s stake EUR %.2f (paper %s)", bet_id, entry.get("match"),
                entry.get("market"), entry.get("selection"), entry.get("odds"), stake, picks_bet_id)
    return bet_id


def settle_linked(picks_bet_id: str, outcome: str, *, result_score: str | None = None,
                  match_kickoff_at: str | None = None, closing_odds: float | None = None) -> int:
    """Settle every pending real entry linked to a paper pick with the pick's
    outcome; profit from the real entry's own stake. Returns how many."""
    from scripts.betting.bet_journal import get_pending_bets, settle_bet
    n = 0
    for b in get_pending_bets():
        if (b.get("extra") or {}).get("picks_ref") != picks_bet_id:
            continue
        stake = float(b.get("stake") or 0)
        odds = float(b.get("odds") or 0)
        profit = {"won": round(stake * (odds - 1), 2), "lost": -stake}.get(outcome, 0.0)
        if settle_bet(b.get("bet_id", ""), outcome, result_score=result_score, profit=profit,
                      match_kickoff_at=match_kickoff_at, closing_odds=closing_odds):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------
def record_card(state: dict | None = None, *, html: bool = True) -> str:
    """Per market: paper n / ROI / CLV, distance to the bar, real record when
    promoted. Italian, one line per market, promoted first."""
    st = state if state is not None else load_state()
    rows = (st.get("markets") or {})
    incumbents = (st.get("incumbents") or {})
    b = ("<b>", "</b>") if html else ("", "")
    i = ("<i>", "</i>") if html else ("", "")
    if not rows and not incumbents:
        return (f"{b[0]}Record mercati{b[1]}\nNessuna scelta ancora liquidata: ogni mercato è carta finché "
                f"non ha {PROMOTION_BAR['min_settled']} scelte liquidate con ROI > 0.")
    order = {"promoted": 0, "paper": 1}
    lines = [f"{b[0]}Record mercati{b[1]} · soglia {PROMOTION_BAR['min_settled']} carta, ROI > 0, z ≥ {PROMOTION_BAR['min_z']:.1f}"]
    for mk, r in incumbents.items():
        rr = r.get("real") or {}
        sl = r.get("real_since_live") or {}
        # Both halves, because they say different things: beat-the-close is
        # mostly our shopping, the line move is the market coming to our side.
        clv = (f" · CLV {rr['mean_clv_pct']:+.1f}%" if rr.get("mean_clv_pct") is not None else "")
        clv += (f" · linea {rr['mean_clv_move_pct']:+.2f}% (z {rr.get('clv_move_z') or 0:+.1f})"
                if rr.get("mean_clv_move_pct") is not None else "")
        bar = "barra superata" if r.get("bar_passed") else f"barra NON superata: {r.get('distance', '')}"
        stake = ("puntata piena" if (r.get("stake_scale") or 1.0) >= 1.0
                 else f"puntata ×{r.get('stake_scale')} finché {INCUMBENT_FULL_STAKE_MIN_N} vere dal go-live "
                      f"hanno ROI > 0 e movimento linea z ≥ {INCUMBENT_FULL_STAKE_BAR['min_clv_move_z']:.1f}")
        lines.append(f"🏦 {b[0]}{MARKET_NAMES_IT.get(mk, mk)}{b[1]} vera n={rr.get('n', 0)} ROI {rr.get('roi_pct', 0):+.0f}% "
                     f"z {rr.get('z', 0):+.2f}{clv} · {i[0]}{bar}{i[1]} · dal go-live n={sl.get('n', 0)} · {stake}")
    if not rows:
        lines.append(f"📝 {i[0]}nessuna scelta carta ancora liquidata{i[1]}")
    for mk, r in sorted(rows.items(), key=lambda kv: (order.get(kv[1].get("status"), 2), -(kv[1].get("paper") or {}).get("n", 0))):
        p = r.get("paper") or {}
        name = MARKET_NAMES_IT.get(mk, mk)
        clv = f" · CLV {p['mean_clv_pct']:+.1f}%" if p.get("mean_clv_pct") is not None else ""
        if r.get("status") == "promoted":
            rr = r.get("real") or {}
            real = (f" · vera n={rr['n']} ROI {rr['roi_pct']:+.0f}%" if rr else " · vera: nessuna ancora")
            lines.append(f"💰 {b[0]}{name}{b[1]} carta n={p.get('n', 0)} ROI {p.get('roi_pct', 0):+.0f}%{clv}{real}")
        else:
            lines.append(f"📝 {name} n={p.get('n', 0)} ROI {p.get('roi_pct', 0):+.0f}%{clv} · {i[0]}{r.get('distance', '')}{i[1]}")
    lines.append(f"{i[0]}💰 = puntata vera (Kelly dimezzato, max {PROMOTED_MAX_STAKE_PCT:.1f}%) · 📝 = carta €10 · "
                 f"un mercato torna carta con ≥{DEMOTION_BAR['min_real_bets']} vere sotto {DEMOTION_BAR['max_roi_pct']:.0f}%"
                 + (" · 🏦 = titolare: punta vero da prima della barra, senza averla passata; "
                    "misurato sullo stesso metro, a metà puntata finché il record dal go-live non lo regge" if incumbents else "") + i[1])
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--null-sim" in sys.argv:
        raise SystemExit(_null_sim_report([a for a in sys.argv[1:] if a != "--null-sim"]))
    print(json.dumps(evaluate_promotions(write=False), indent=1, default=str))
