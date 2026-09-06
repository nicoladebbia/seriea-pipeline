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

PROMOTION_BAR = {"min_settled": 50, "min_roi_pct": 0.0, "min_z": 1.0,
                 "min_clv_pct": 0.0, "min_clv_n": 20}
DEMOTION_BAR = {"min_real_bets": 30, "max_roi_pct": -10.0, "max_z": -1.0}

# Real stake on a promoted market: the O/U Kelly (0.15) halved and a lower
# cap, until the market has its own real record. Edge over-claim is the
# measured failure mode on props (+9.9% claimed -> +1.8% realised elsewhere).
PROMOTED_KELLY_SCALE = 0.5
PROMOTED_MAX_STAKE_PCT = 1.5
PROMOTED_MIN_STAKE_PCT = 0.2
PIPELINE_STATUS = "pick:promoted"

# Market key -> what the bet is, for the /record card
MARKET_NAMES_IT = {
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
        m = by.setdefault(b.get("market") or "?", {"bets": [], "clv": []})
        m["bets"].append(b)
        if b.get("clv_pct") is not None:
            m["clv"].append(float(b["clv_pct"]))
    out: dict[str, dict] = {}
    for mk, m in by.items():
        r = _unit_returns(m["bets"])
        n = len(r)
        mu = mean(r) if r else 0.0
        sd = pstdev(r) if n > 1 else 0.0
        z = (mu / (sd / n ** 0.5)) if n > 1 and sd > 0 else 0.0
        out[mk] = {"n": n, "won": sum(b.get("status") == "won" for b in m["bets"]),
                   "roi_pct": round(mu * 100, 1), "z": round(z, 2),
                   "profit": round(sum(float(b.get("profit") or 0) for b in m["bets"]), 2),
                   "mean_clv_pct": round(mean(m["clv"]), 2) if m["clv"] else None,
                   "n_clv": len(m["clv"])}
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
                        *, now: datetime | None = None, path: Path | None = None, write: bool = True) -> dict:
    """Re-read both journals and rewrite the state. Idempotent. Every market
    seen in the paper journal gets a row; a promotion or demotion is a state
    transition with its record snapshot and reason, and is logged at WARNING
    because it changes where real money goes."""
    from scripts.betting.bet_journal import get_settled_bets
    from scripts.betting.picks import PICKS_JOURNAL_PATH
    now = now or datetime.now(UTC)
    if paper_settled is None:
        paper_settled = get_settled_bets(journal_path=PICKS_JOURNAL_PATH)
    if real_settled is None:
        real_settled = [b for b in get_settled_bets() if b.get("pipeline_status") == PIPELINE_STATUS]
    state = load_state(path)
    markets: dict = state.setdefault("markets", {})
    real_by = market_record(real_settled)
    seen = {b.get("market") for b in paper_settled if b.get("market")}
    for mk in sorted(seen | set(markets)):
        row = markets.setdefault(mk, {"status": "paper", "since": now.isoformat(), "record_from": None})
        paper_rec = market_record(paper_settled, since=row.get("record_from")).get(mk) or \
            {"n": 0, "won": 0, "roi_pct": 0.0, "z": 0.0, "profit": 0.0, "mean_clv_pct": None, "n_clv": 0}
        row["paper"] = paper_rec
        row["real"] = real_by.get(mk)
        if row["status"] == "promoted":
            demote, why = should_demote(row["real"])
            if demote:
                row.update({"status": "paper", "since": now.isoformat(), "record_from": now.isoformat(),
                            "reason": f"demoted: {why}", "demoted_at": now.isoformat()})
                # the paper count restarts here: the record that promoted it is spent
                row["paper"] = {"n": 0, "won": 0, "roi_pct": 0.0, "z": 0.0, "profit": 0.0,
                                "mean_clv_pct": None, "n_clv": 0}
                row["distance"] = passes_bar(row["paper"])[1]
                log.warning("Market %s DEMOTED to paper: %s", mk, why)
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
    state["updated_at"] = now.isoformat()
    state["bar"] = PROMOTION_BAR
    state["demotion_bar"] = DEMOTION_BAR
    if write:
        from config.settings import atomic_write_json
        atomic_write_json(path or STATE_PATH, state)
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
    b = ("<b>", "</b>") if html else ("", "")
    i = ("<i>", "</i>") if html else ("", "")
    if not rows:
        return (f"{b[0]}Record mercati{b[1]}\nNessuna scelta ancora liquidata: ogni mercato è carta finché "
                f"non ha {PROMOTION_BAR['min_settled']} scelte liquidate con ROI > 0.")
    order = {"promoted": 0, "paper": 1}
    lines = [f"{b[0]}Record mercati{b[1]} · soglia {PROMOTION_BAR['min_settled']} carta, ROI > 0, z ≥ {PROMOTION_BAR['min_z']:.0f}"]
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
                 f"un mercato torna carta con ≥{DEMOTION_BAR['min_real_bets']} vere sotto {DEMOTION_BAR['max_roi_pct']:.0f}%{i[1]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--null-sim" in sys.argv:
        raise SystemExit(_null_sim_report([a for a in sys.argv[1:] if a != "--null-sim"]))
    print(json.dumps(evaluate_promotions(write=False), indent=1, default=str))
