"""In-play paper engine — fair prices from the goal-process simulator conditioned
on the score and the minute, against the in-play market the live monitor
already stores, one paper pick per state change. Money never moves here.

Why it exists (measured 2026-09-05 on the Roma–Atalanta price path): the
Odds API in-play feed lags the pitch by minutes, Pinnacle does not reprice
in-play through it, and the in-play overround is ~7% against 5% pre-match.
"Atalanta scored, back Roma at 10" only pays if P(Roma win | 0-1, 81') beats
what the soft books think — the simulator said 3.4% (fair 29.0) against a
market 10.0. So the question is a model question, and it is answered on the
snapshots on disk before any euro moves.

What the engine does
- ``baseline_for_entry``: simulator inputs implied by the PRE-MATCH market
  (pre_match_odds, else the first ≤10' 0-0 snapshot) via
  ``goal_process.market_profile`` — the same information the books had.
  Model xG is deliberately not used here: the backtest asks whether the
  CONDITIONING beats the books' repricing, not whether our xG does.
- ``fair_for_snapshot``: 1X2 + every totals line the snapshot quotes, from
  ``goal_process.simulate_from_state``, red cards through the measured
  ``red_mult`` in the profile.
- ``edges``: fair − de-vigged average market, per selection.
- ``best_pick`` after a state change: the first priced snapshot after the
  score changed, best 1X2 edge that clears the snapshot's own overround plus
  the simulator's 95% Monte Carlo interval (no hand-set edge / probability /
  minute thresholds), one pick per selection per match; an edge above the
  journal cap is counted but never journaled (the same 12% cap the pick
  engine uses). The fair price is first shrunk toward the market by the
  walk-forward weight the backtest fitted (``shrink.w_latest``), if any.
- Journal: ``data/betting/inplay_journal.json`` through ``bet_journal.add_bet``
  (flat PAPER_STAKE); settled at full time from the final score; CLV against
  the NEXT priced snapshot (the price a human could actually have taken,
  given the feed lag).
- ``backtest``: the whole engine over ``data/live/*.json`` →
  ``data/models/inplay/backtest.json``. The gate is skill vs the in-play
  market's own probabilities (1 − Brier_fair / Brier_market) and the paper
  ROI at the NEXT-snapshot price, not at the price we saw.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config.settings import DATA_DIR
from scripts.models import goal_process as gp

log = logging.getLogger(__name__)

INPLAY_JOURNAL_PATH = DATA_DIR / "betting" / "inplay_journal.json"
BACKTEST_PATH = DATA_DIR / "models" / "inplay" / "backtest.json"
PING_STATE_KEY = "inplay_pings"          # pipeline_state: "on" | "off" (default — the backtest fails the gate)
PING_MODES = ("on", "off")

# Picks are 1X2 ONLY. The totals lines in a snapshot are the pre-match lines
# carried along by the feed, not live prices: the first backtest "found"
# Over 0.5 @ 2.32 in the 84th minute at 1-0 (fair 1.00). Totals fair prices
# are still computed for the card, never picked.
PICK_MARKETS = ("inplay_1x2",)
# No hand-set thresholds (2026-09-05). A pick must clear two things the data
# itself provides: the book's margin on THAT snapshot (its overround, the
# price of betting into it) and the simulator's own Monte Carlo uncertainty
# on the fair probability (a 95% interval from the number of paths). Late
# states need no minute cut-off: as the fair price goes to 0/1 the edge
# vanishes and the interval does the rest. The 12% journal cap is the
# system-wide rule shared with every other paper market.
MAX_EDGE_JOURNAL = 12.0  # bet_journal.MAX_EDGE_PCT — above it the pick is counted, not journaled
MC_Z = 1.96              # 95% Monte Carlo interval on a simulated probability
N_SIMS_LIVE = 6000
N_SIMS_BACKTEST = 3000
BASELINE_MAX_MINUTE = 10  # a 0-0 snapshot this early stands in for the pre-match line
SEL_1X2 = {"home": "Home", "draw": "Draw", "away": "Away"}
KEY_1X2 = {v: k for k, v in SEL_1X2.items()}


# ------------------------------------------------------------------ helpers
def devig(prices: dict) -> dict:
    """Proportional de-vig of a {selection: decimal odds} book."""
    inv = {k: 1.0 / float(v) for k, v in prices.items() if v and float(v) > 1.0}
    tot = sum(inv.values())
    return {k: v / tot for k, v in inv.items()} if tot > 0 else {}


def _priced(snaps: list) -> list:
    return [s for s in snaps if (s.get("avg_odds") or {}).get("home")]


def _totals_book(snap: dict) -> dict:
    """{line: {"over": odds, "under": odds}} from a snapshot's totals block."""
    t = snap.get("totals") or {}
    out: dict = {}
    for line, pair in (t.get("all_lines") or {}).items():
        try:
            ln = float(line)
        except (TypeError, ValueError):
            continue
        if pair.get("over") and pair.get("under"):
            out[ln] = {"over": float(pair["over"]), "under": float(pair["under"])}
    if not out and t.get("over_avg") and t.get("under_avg") and t.get("line") is not None:
        out[float(t["line"])] = {"over": float(t["over_avg"]), "under": float(t["under_avg"])}
    return out


def _p_over_2_5_from(entry: dict, first: dict | None) -> tuple[float | None, str]:
    pre = (entry.get("pre_match_odds") or {}).get("totals") or {}
    if pre.get("over") and pre.get("under") and float(pre.get("line") or 0) == 2.5:
        return devig({"over": pre["over"], "under": pre["under"]}).get("over"), "pre_match_odds"
    if first:
        book = _totals_book(first)
        if 2.5 in book:
            return devig(book[2.5]).get("over"), "first_snapshot"
    return None, "none"


def _teams(entry: dict, mk: str = "") -> tuple[str, str]:
    home = entry.get("home_team") or (mk.split(" vs ")[0] if mk else "")
    away = entry.get("away_team") or (mk.split(" vs ")[1] if " vs " in mk else "")
    return home, away


PROFILE_LEAGUE = "serie_a"  # the only league with a fitted goal-process profile (EPL has none)


def entry_league(entry: dict, mk: str = "") -> str:
    """The live entry's league: the stamp the monitor wrote, else inferred
    from the team names. Until 2026-09-05 entries carried no league and EPL
    matches were priced with the Serie A hazard, red-card multipliers and
    calibration — 45 of the 156 stored matches."""
    lg = entry.get("league")
    if lg:
        return str(lg)
    from config.leagues import infer_league
    home, away = _teams(entry, mk)
    return infer_league(home, away)


def reds_at(entry: dict, minute: int) -> tuple[int, int]:
    """Red cards (home, away) shown by ``minute`` in the entry's live events
    (Sofascore and ESPN both store ``type: card`` + ``card_type``)."""
    h = a = 0
    for ev in entry.get("live_events") or []:
        if ev.get("type") != "card" or ev.get("card_type") not in ("red", "yellowRed"):
            continue
        if int(ev.get("minute") or 0) > minute:
            continue
        if ev.get("is_home"):
            h += 1
        else:
            a += 1
    return h, a


def _closing_line(entry: dict, mk: str) -> dict:
    """The last pre-kickoff odds snapshot on disk (live_monitor owns the store)."""
    try:
        from scripts.data.live_monitor import _closing_line_from_snapshots
        home, away = _teams(entry, mk)
        return _closing_line_from_snapshots(home, away, entry.get("commence_time") or "")
    except Exception:  # noqa: BLE001
        return {}


def _archived_xg(entry: dict, mk: str) -> dict | None:
    """Our own pre-kickoff xG for the match from predictions_archive.json, if archived."""
    try:
        from config.team_names import normalize_team
        with open(DATA_DIR / "upcoming" / "predictions_archive.json") as f:
            arch = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    home, away = _teams(entry, mk)
    key = ((entry.get("commence_time") or "")[:10], normalize_team(home), normalize_team(away))
    for row in (arch.values() if isinstance(arch, dict) else arch):
        if (row.get("date"), normalize_team(row.get("home_team") or ""), normalize_team(row.get("away_team") or "")) == key:
            if row.get("home_xg") and row.get("away_xg"):
                return {"xg_h": float(row["home_xg"]), "xg_a": float(row["away_xg"])}
    return None


def baseline_for_entry(entry: dict, prof: dict | None = None, n: int = N_SIMS_LIVE, mk: str = "",
                       baseline: str = "market") -> dict | None:
    """Simulator inputs for the match; cached on the entry.

    baseline="market": xG split + k implied by the pre-match market
    (pre_match_odds, else the closing odds snapshot on disk, else the first
    ≤10' 0-0 snapshot). baseline="model": our archived pre-kickoff xG with k
    solved on the market's P(over 2.5) — the backtest's second variant."""
    cached = entry.get("inplay_baseline")
    if cached and cached.get("xg_h") is not None and cached.get("baseline", "market") == baseline:
        return cached
    snaps = _priced(entry.get("snapshots") or [])
    first = snaps[0] if snaps else None
    pre = entry.get("pre_match_odds") or {}
    if not (pre.get("home") and pre.get("draw") and pre.get("away")):
        closing = _closing_line(entry, mk)
        if closing.get("home"):
            pre = {**pre, **closing}
    if pre.get("home") and pre.get("draw") and pre.get("away"):
        h2h, h2h_source = devig({"home": pre["home"], "draw": pre["draw"], "away": pre["away"]}), pre.get("source") or "pre_match_odds"
    elif first and int(first.get("min") or 0) <= BASELINE_MAX_MINUTE and list(first.get("score") or [0, 0]) == [0, 0]:
        h2h, h2h_source = devig(first["avg_odds"]), "first_snapshot"
    else:
        return None
    if len(h2h) != 3:
        return None
    p_over, totals_source = _p_over_2_5_from(entry, first)
    prof = prof or gp.load_profile()
    if baseline == "model":
        xg = _archived_xg(entry, mk)
        if not xg:
            return None
        sim = gp.simulate(xg["xg_h"], xg["xg_a"], prof, n=n, p_over_2_5=p_over)
        base = {"xg_h": xg["xg_h"], "xg_a": xg["xg_a"], "k": sim["calibration_k"], "saturated": sim["calibration_saturated"]}
    else:
        base = gp.market_profile(h2h, p_over, prof, n=n)
    base.update({"baseline": baseline, "h2h_source": h2h_source, "totals_source": totals_source,
                 "p_over_2_5": (round(p_over, 4) if p_over is not None else None),
                 "h2h": {k: round(v, 4) for k, v in h2h.items()}})
    entry["inplay_baseline"] = base
    return base


def fair_for_snapshot(base: dict, snap: dict, prof: dict | None = None, n: int = N_SIMS_LIVE,
                      seed: int = 0, red: tuple[int, int] = (0, 0)) -> dict:
    """Fair probabilities for the state the snapshot describes (score, minute, red cards)."""
    minute = int(snap.get("min") or 0)
    score = tuple(int(x) for x in (snap.get("score") or [0, 0]))
    paths = gp.simulate_from_state(base["xg_h"], base["xg_a"], minute, score, prof, k=base["k"], n=n, seed=seed, red=red,
                                   added_time=int(snap.get("added_time") or 0))
    pr = gp.market_probs(paths)
    total = paths["home_final"] + paths["away_final"]
    totals = {}
    for line in _totals_book(snap):
        over = float((total > line).mean())
        totals[str(line)] = {"over": round(over, 4), "under": round(1.0 - over, 4)}
    return {"minute": minute, "score": list(score), "red": list(red),
            "1x2": {"home": round(pr["home_win"], 4), "draw": round(pr["draw"], 4), "away": round(pr["away_win"], 4)},
            "totals": totals, "red_cards_modelled": bool(paths.get("red_cards_modelled"))}


def shrink_weight(fair_probs: np.ndarray, market_probs: np.ndarray, outcomes: np.ndarray) -> float | None:
    """The weight w in w·fair + (1−w)·market that minimises the 1X2 Brier on
    the rows given (fair, market: n×3 probabilities; outcomes: n×3 one-hot).
    Fitted walk-forward in the backtest — never on the matchday it is applied
    to — and stored for the live engine. None below the sample floor."""
    if len(outcomes) < gp.N_GATE:
        return None
    best_w, best_b = 1.0, None
    for w in np.arange(0.0, 1.0001, 0.02):
        blend = w * fair_probs + (1 - w) * market_probs
        b = float(((blend - outcomes) ** 2).sum(axis=1).mean())
        if best_b is None or b < best_b:
            best_w, best_b = float(w), b
    return round(best_w, 2)


def live_shrink_weight() -> float | None:
    """The latest walk-forward weight the backtest wrote; None until it exists."""
    try:
        with open(BACKTEST_PATH) as f:
            return (json.load(f).get("shrink") or {}).get("w_latest")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def edges(snap: dict, fair: dict, w: float | None = None) -> list[dict]:
    """fair − de-vigged market, one row per selection, best first. With w the
    fair price is shrunk toward the market first (w=1 is the raw simulator)."""
    rows: list[dict] = []
    mkt = devig(snap.get("avg_odds") or {})
    for sel in ("home", "draw", "away"):
        if sel in mkt:
            f = fair["1x2"][sel] if w is None else w * fair["1x2"][sel] + (1 - w) * mkt[sel]
            rows.append({"market": "inplay_1x2", "selection": SEL_1X2[sel], "key": sel,
                         "fair": round(f, 4), "fair_raw": fair["1x2"][sel], "market_prob": round(mkt[sel], 4),
                         "odds": float(snap["avg_odds"][sel]), "edge": round(f - mkt[sel], 4), "shrink_w": w})
    for line, pair in _totals_book(snap).items():
        fl = fair["totals"].get(str(line))
        if not fl:
            continue
        dv = devig(pair)
        if len(dv) != 2:
            continue  # one side at 1.00 or missing: not a market
        for side in ("over", "under"):
            rows.append({"market": "inplay_ou", "selection": f"{side.title()} {line}", "key": f"{side}_{line}",
                         "fair": fl[side], "market_prob": round(dv[side], 4), "odds": pair[side],
                         "edge": round(fl[side] - dv[side], 4), "line": line, "side": side})
    rows.sort(key=lambda r: -r["edge"])
    return rows


def overround(snap: dict) -> float | None:
    """The book's margin on this snapshot's 1X2: sum of implied − 1."""
    a = snap.get("avg_odds") or {}
    try:
        return sum(1.0 / float(a[k]) for k in ("home", "draw", "away")) - 1.0
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def mc_se(p: float, n: int) -> float:
    """Standard error of a simulated probability from n paths."""
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / max(int(n), 1)))


def best_pick(rows: list[dict], snap: dict, n_sims: int) -> dict | None:
    """Best 1X2 row whose edge clears the snapshot's own margin plus the
    simulator's 95% Monte Carlo interval. Nothing here is a tuned number."""
    margin = overround(snap)
    if margin is None:
        return None
    for r in rows:
        if r["market"] not in PICK_MARKETS:
            continue
        floor = margin + MC_Z * mc_se(r["fair"], n_sims)
        if r["edge"] >= floor:
            return {**r, "edge_pct": round(r["edge"] * 100, 2), "floor": round(floor, 4), "margin": round(margin, 4),
                    "over_cap": r["edge"] * 100 > MAX_EDGE_JOURNAL}
    return None


def state_changed(prev: dict | None, snap: dict) -> bool:
    """A goal (or a correction) between two priced snapshots."""
    if prev is None:
        return False
    return list(prev.get("score") or [0, 0]) != list(snap.get("score") or [0, 0])


# ------------------------------------------------------------------ live hook
def ping_mode() -> str:
    try:
        from scripts.utils.json_utils import load_json_safe
        state = load_json_safe(DATA_DIR / "pipeline_state.json", {}) or {}
        return "on" if state.get(PING_STATE_KEY) == "on" else "off"
    except Exception:  # noqa: BLE001 - a broken state file must not stop the tick
        return "off"


def journal_pick(mk: str, entry: dict, snap: dict, pick: dict, *, journal_path: Path | None = None) -> str | None:
    """Flat-stake paper entry. Selection = the outcome only, so the journal's
    own dedup (date+match+market+selection) blocks a second pick on the same
    outcome in the same match."""
    if pick.get("over_cap"):
        return None
    from scripts.betting.bet_journal import add_bet
    from scripts.betting.betting_unified import PAPER_STAKE
    date = (entry.get("commence_time") or snap.get("ts") or "")[:10]
    base = entry.get("inplay_baseline") or {}
    return add_bet({
        "match": mk, "date": date, "league": entry.get("league"),
        "market": pick["market"], "selection": pick["selection"],
        "model_prob": pick["fair"], "sharp_implied_prob": None,
        "edge_pct": pick["edge_pct"], "odds": pick["odds"], "bookmaker": "avg_inplay",
        "stake": PAPER_STAKE, "placed_at": snap.get("ts") or datetime.now(timezone.utc).isoformat(),
        "pipeline_status": "inplay:paper",
        "extra": {"minute": snap.get("min"), "score": snap.get("score"), "market_prob": pick["market_prob"],
                  "side": pick.get("side"), "line": pick.get("line"), "snapshot_ts": snap.get("ts"),
                  "baseline": {k: base.get(k) for k in ("xg_h", "xg_a", "k", "h2h_source")},
                  "red": list(reds_at(entry, int(snap.get("min") or 0))),
                  "red_cards_modelled": bool((gp.load_profile().get("red_mult") or {}).get("short"))},
    }, journal_path=journal_path or INPLAY_JOURNAL_PATH)


def on_snapshot(mk: str, entry: dict, snap: dict, *, prof: dict | None = None,
                journal_path: Path | None = None, notify_fn=None, n: int = N_SIMS_LIVE) -> dict | None:
    """Price one freshly appended snapshot; journal + ping a paper pick after a
    state change. Writes ``fair`` / ``fair_totals`` / ``best_edge`` onto the
    snapshot (the card reads them). Returns the pick record or None."""
    if not (snap.get("avg_odds") or {}).get("home"):
        return None
    league = entry_league(entry, mk)
    if league != PROFILE_LEAGUE:
        entry.setdefault("inplay_note", f"no goal-process profile for {league}")
        return None
    base = baseline_for_entry(entry, prof, n=n, mk=mk)
    if not base:
        entry.setdefault("inplay_note", "no pre-match baseline")
        return None
    red = reds_at(entry, int(snap.get("min") or 0))
    fair = fair_for_snapshot(base, snap, prof, n=n, red=red)
    w = live_shrink_weight()
    rows = edges(snap, fair, w)
    snap["fair"] = {r["key"]: r["fair"] for r in rows if r["market"] == "inplay_1x2"} or fair["1x2"]
    snap["fair_raw"] = fair["1x2"]
    snap["fair_totals"] = fair["totals"]
    snap["fair_red"] = list(red)
    snap["shrink_w"] = w
    snap["best_edge"] = rows[0] if rows else None
    prev = None
    for s in reversed(_priced(entry.get("snapshots") or [])):
        if s is not snap and s.get("ts") != snap.get("ts"):
            prev = s
            break
    if not state_changed(prev, snap):
        return None
    pick = best_pick(rows, snap, n)
    if not pick:
        return None
    taken = entry.setdefault("inplay_picks", [])
    if any(p.get("selection") == pick["selection"] for p in taken):
        return None
    rec = {"selection": pick["selection"], "market": pick["market"], "odds": pick["odds"], "fair": pick["fair"],
           "fair_raw": pick.get("fair_raw"), "shrink_w": w, "floor": pick.get("floor"), "margin": pick.get("margin"),
           "market_prob": pick["market_prob"], "edge_pct": pick["edge_pct"], "minute": snap.get("min"),
           "score": snap.get("score"), "ts": snap.get("ts"), "over_cap": pick["over_cap"], "status": "pending",
           "side": pick.get("side"), "line": pick.get("line")}
    try:
        rec["bet_id"] = journal_pick(mk, entry, snap, pick, journal_path=journal_path)
    except Exception as exc:  # noqa: BLE001 - the journal must not sink the poll
        log.warning("in-play journal failed for %s: %s", mk, exc)
        rec["bet_id"] = None
    taken.append(rec)
    if notify_fn is not None and ping_mode() == "on":
        try:
            notify_fn(mk, entry, rec)
        except Exception as exc:  # noqa: BLE001
            log.warning("in-play ping failed for %s: %s", mk, exc)
    return rec


def send_pick_ping(mk: str, entry: dict, rec: dict) -> None:
    from scripts.pipeline.notify import notify
    score = rec.get("score") or [0, 0]
    cap = " (edge above the 12% cap: counted, not journaled)" if rec.get("over_cap") else ""
    msg = (f"PAPER in-play: {rec['selection']} @ {rec['odds']:.2f} — {mk} {score[0]}-{score[1]} at {rec.get('minute')}'\n"
           f"fair {rec['fair'] * 100:.0f}% vs market {rec['market_prob'] * 100:.0f}% (+{rec['edge_pct']:.1f}%){cap}\n"
           f"Paper only. Books reprice before you can act; the record decides if this ever earns a stake.")
    notify(msg, title=f"IN-PLAY (paper) {mk}", level="info", category="live")


# ------------------------------------------------------------------ grading
def _grade(pick: dict, final: tuple[int, int]) -> str:
    h, a = final
    sel = pick.get("selection") or ""
    if pick.get("market") == "inplay_1x2":
        won = {"Home": h > a, "Draw": h == a, "Away": a > h}.get(sel)
        return "won" if won else "lost"
    ex = pick.get("extra") or pick
    side, line = ex.get("side"), ex.get("line")
    if side is None or line is None:
        return "void"
    total = h + a
    if total == line:
        return "push"
    return "won" if ((side == "over") == (total > line)) else "lost"


def _price_in(snap: dict, pick: dict) -> float | None:
    if pick.get("market") == "inplay_1x2":
        return float((snap.get("avg_odds") or {}).get(KEY_1X2.get(pick.get("selection"), "")) or 0) or None
    ex = pick.get("extra") or pick
    book = _totals_book(snap).get(float(ex.get("line") or 0))
    return float(book[ex["side"]]) if book and ex.get("side") in book else None


def _next_price(entry: dict, placed_ts: str, pick: dict) -> float | None:
    """The first priced snapshot AFTER the pick — the price a human could have taken."""
    for s in _priced(entry.get("snapshots") or []):
        if (s.get("ts") or "") <= (placed_ts or ""):
            continue
        return _price_in(s, pick)
    return None


def settle_for_matchday(matchday: dict, *, journal_path: Path | None = None) -> int:
    """Settle every pending in-play paper pick whose match is completed."""
    from scripts.betting.bet_journal import _load_journal, settle_bet
    path = journal_path or INPLAY_JOURNAL_PATH
    journal = _load_journal(path)
    settled = 0
    for bet_id, bet in list((journal.get("bets") or {}).items()):
        if bet.get("status") != "pending":
            continue
        entry = (matchday.get("matches") or {}).get(bet.get("match"))
        if not entry or entry.get("status") != "completed" or not entry.get("final_score"):
            continue
        final = tuple(int(x) for x in entry["final_score"])
        status = _grade(bet, final)
        odds, stake = float(bet.get("odds") or 0), float(bet.get("stake") or 0)
        profit = {"won": stake * (odds - 1), "lost": -stake}.get(status, 0.0)
        closing = _next_price(entry, bet.get("placed_at") or "", bet)
        if settle_bet(bet_id, status, result_score=f"{final[0]}-{final[1]}", profit=round(profit, 2),
                      match_kickoff_at=entry.get("commence_time"), journal_path=path, closing_odds=closing):
            settled += 1
            for rec in entry.get("inplay_picks") or []:
                if rec.get("bet_id") == bet_id:
                    rec["status"] = status
                    rec["closing_odds"] = closing
    return settled


# ------------------------------------------------------------------ backtest
def _bucket(minute: int) -> str:
    return "0-30" if minute <= 30 else "31-60" if minute <= 60 else "61-85" if minute <= 85 else "86+"


def _summarise(rows: list) -> dict:
    rows = [r for r in rows if not r["over_cap"]]
    out: dict = {"n": len(rows)}
    if not rows:
        return out
    for basis in ("odds", "next_odds"):
        usable = [r for r in rows if r.get(basis)]
        pnl = sum((r[basis] - 1) if r["status"] == "won" else (-1 if r["status"] == "lost" else 0) for r in usable)
        out[basis] = {"n": len(usable), "roi_pct": round(100 * pnl / len(usable), 1) if usable else None,
                      "hit_rate": round(sum(r["status"] == "won" for r in usable) / len(usable), 3) if usable else None}
    clv = [(1 / r["next_odds"] - 1 / r["odds"]) * 100 for r in rows if r.get("next_odds")]
    out["clv_vs_next_snapshot_pct"] = round(float(np.mean(clv)), 2) if clv else None
    out["avg_edge_pct"] = round(100 * float(np.mean([r["edge"] for r in rows])), 2)
    out["by_market"] = {m: sum(r["market"] == m for r in rows) for m in ("inplay_1x2", "inplay_ou")}
    return out


def _skill(f: list, m: list) -> float | None:
    return round(1 - float(np.mean(f)) / float(np.mean(m)), 4) if f and m else None


def backtest(files: list[str] | None = None, n: int = N_SIMS_BACKTEST, write: bool = True,
             baseline: str = "market") -> dict:
    """The engine over every stored matchday. Two pick rules (after a state
    change; any snapshot), each graded at the price we SAW and at the NEXT
    snapshot's price (the feed lag), plus 1X2 skill vs the in-play market on
    every priced snapshot."""
    prof = gp.load_profile()
    files = files or sorted(glob.glob(str(DATA_DIR / "live" / "20[0-9][0-9]-[0-9][0-9]-[0-9][0-9].json")))
    brier: dict = {"fair": [], "market": []}
    by_match: dict = {}      # match -> (sum fair brier, sum market brier) for the cluster bootstrap
    by_min: dict = {}
    picks: dict = {"state_change": [], "any_snapshot": []}
    n_matches = n_snaps = n_no_baseline = n_other_league = 0
    raw_brier_with_w: list = []  # raw fair Brier on the rows the blend was scored on
    # walk-forward shrinkage: rows seen on EARLIER matchdays fit the weight for this one
    hist_f: list = []
    hist_m: list = []
    hist_o: list = []
    w_trail: list = []
    blend_brier: list = []
    for path in sorted(files):
        w = shrink_weight(np.array(hist_f), np.array(hist_m), np.array(hist_o)) if hist_o else None
        w_trail.append({"file": path[-15:-5], "w": w, "rows_before": len(hist_o)})
        day_f: list = []
        day_m: list = []
        day_o: list = []
        try:
            with open(path) as f:
                day = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for mk, entry in (day.get("matches") or {}).items():
            entry = dict(entry)
            entry.pop("inplay_baseline", None)
            snaps = _priced(entry.get("snapshots") or [])
            final = entry.get("final_score")
            if not final and entry.get("status") == "completed" and entry.get("snapshots"):
                final = entry["snapshots"][-1].get("score")
            if not snaps or not final:
                continue
            if entry_league(entry, mk) != PROFILE_LEAGUE:
                n_other_league += 1
                continue
            base = baseline_for_entry(entry, prof, n=n, mk=mk, baseline=baseline)
            if not base:
                n_no_baseline += 1
                continue
            n_matches += 1
            final = tuple(int(x) for x in final)
            outcome = "home" if final[0] > final[1] else "away" if final[1] > final[0] else "draw"
            prev = None
            seen: dict = {"state_change": set(), "any_snapshot": set()}
            for i, snap in enumerate(snaps):
                minute = int(snap.get("min") or 0)
                if minute >= 90 and list(snap.get("score") or []) == list(final):
                    prev = snap
                    continue  # the whistle: nothing left to price
                fair = fair_for_snapshot(base, snap, prof, n=n, seed=i, red=reds_at(entry, minute))
                rows = edges(snap, fair, w)
                mkt = devig(snap["avg_odds"])
                if len(mkt) == 3:
                    n_snaps += 1
                    bf = sum((fair["1x2"][k] - (1.0 if k == outcome else 0.0)) ** 2 for k in mkt)
                    bm = sum((mkt[k] - (1.0 if k == outcome else 0.0)) ** 2 for k in mkt)
                    onehot = [1.0 if k == outcome else 0.0 for k in ("home", "draw", "away")]
                    day_f.append([fair["1x2"][k] for k in ("home", "draw", "away")])
                    day_m.append([mkt[k] for k in ("home", "draw", "away")])
                    day_o.append(onehot)
                    if w is not None:
                        blend = [w * fair["1x2"][k] + (1 - w) * mkt[k] for k in ("home", "draw", "away")]
                        blend_brier.append((sum((b - o) ** 2 for b, o in zip(blend, onehot)), bm))
                        raw_brier_with_w.append(bf)
                    brier["fair"].append(bf)
                    brier["market"].append(bm)
                    agg = by_match.setdefault(f"{path}:{mk}", [0.0, 0.0])
                    agg[0] += bf
                    agg[1] += bm
                    b = by_min.setdefault(_bucket(minute), {"fair": [], "market": []})
                    b["fair"].append(bf)
                    b["market"].append(bm)
                nxt = snaps[i + 1] if i + 1 < len(snaps) else None
                for rule in ("state_change", "any_snapshot"):
                    if rule == "state_change" and not state_changed(prev, snap):
                        continue
                    pick = best_pick(rows, snap, n)
                    if not pick or pick["selection"] in seen[rule]:
                        continue
                    seen[rule].add(pick["selection"])
                    picks[rule].append({"match": mk, "date": path[-15:-5], "minute": minute, "score": snap.get("score"),
                                        "selection": pick["selection"], "market": pick["market"], "odds": pick["odds"],
                                        "next_odds": _price_in(nxt, pick) if nxt else None,
                                        "fair": pick["fair"], "fair_raw": pick.get("fair_raw"), "shrink_w": w,
                                        "market_prob": pick["market_prob"], "edge": pick["edge"], "floor": pick.get("floor"),
                                        "over_cap": pick["over_cap"], "status": _grade(pick, final)})
                prev = snap
        hist_f += day_f
        hist_m += day_m
        hist_o += day_o

    # Cluster bootstrap by match: snapshots of one match are not independent.
    ci = None
    if len(by_match) >= 10:
        rng = np.random.default_rng(0)
        pairs = np.array(list(by_match.values()))
        draws = []
        for _ in range(2000):
            idx = rng.integers(0, len(pairs), len(pairs))
            f, m = pairs[idx, 0].sum(), pairs[idx, 1].sum()
            draws.append(1 - f / m if m > 0 else 0.0)
        ci = [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]
    pick_f = [x for k, v in by_min.items() if k != "86+" for x in v["fair"]]
    pick_m = [x for k, v in by_min.items() if k != "86+" for x in v["market"]]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": len(files), "matches": n_matches, "matches_without_baseline": n_no_baseline,
        "matches_other_league_skipped": n_other_league, "league": PROFILE_LEAGUE, "priced_snapshots": n_snaps,
        "skill_vs_inplay_market_1x2": _skill(brier["fair"], brier["market"]),
        "skill_ci95_by_match_bootstrap": ci,
        "skill_pickable_window_le85": _skill(pick_f, pick_m),
        "shrink": {"w_latest": (shrink_weight(np.array(hist_f), np.array(hist_m), np.array(hist_o)) if hist_o else None),
                   "trail": w_trail[-6:],
                   "skill_blend_vs_market_walk_forward": (_skill([b for b, _ in blend_brier], [m for _, m in blend_brier])
                                                          if blend_brier else None),
                   "skill_raw_vs_market_same_rows": (_skill(raw_brier_with_w, [m for _, m in blend_brier])
                                                     if blend_brier else None),
                   "rows_scored_with_a_weight": len(blend_brier)},
        "brier": {"fair": round(float(np.mean(brier["fair"])), 4) if brier["fair"] else None,
                  "market": round(float(np.mean(brier["market"])), 4) if brier["market"] else None},
        "skill_by_minute": {k: _skill(v["fair"], v["market"]) for k, v in sorted(by_min.items())},
        "snapshots_by_minute": {k: len(v["fair"]) for k, v in sorted(by_min.items())},
        "picks": {rule: _summarise(rows) for rule, rows in picks.items()},
        "picks_over_cap": {rule: sum(r["over_cap"] for r in rows) for rule, rows in picks.items()},
        "rule": {"edge_floor": "snapshot overround + 1.96 x Monte Carlo SE of the fair probability (no hand-set numbers)",
                 "minute_cutoff": None, "baseline": baseline, "red_cards_modelled": bool((prof.get("red_mult") or {}).get("short")),
                 "red_mult": prof.get("red_mult"),
                 "pick_markets": list(PICK_MARKETS), "totals_feed": "pre-match lines carried in-play — never picked"},
        "gate": {"skill_min": gp.SKILL_GATE, "n_min": gp.N_GATE},
    }
    result["passes_gate"] = bool(result["skill_vs_inplay_market_1x2"] is not None
                                 and result["skill_vs_inplay_market_1x2"] >= gp.SKILL_GATE and n_snaps >= gp.N_GATE)
    result["sample_picks"] = {rule: rows[:40] for rule, rows in picks.items()}
    if write and baseline == "market":
        # the market variant is the gate; the model variant rides along as a sub-report
        try:
            model = backtest(files, n=n, write=False, baseline="model")
            model.pop("sample_picks", None)
            result["model_variant"] = model
        except Exception as exc:  # noqa: BLE001
            result["model_variant"] = {"error": str(exc)}
        BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BACKTEST_PATH, "w") as f:
            json.dump(result, f, indent=2, default=str)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="in-play paper engine")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--n", type=int, default=N_SIMS_BACKTEST)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.backtest:
        r = backtest(n=args.n)
        r.pop("sample_picks", None)
        print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
