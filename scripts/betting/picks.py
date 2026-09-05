"""Pick engine: one line for EVERY upcoming Serie A match, every market priced.

Decision (2026-09-05, Nicola): "Pick every match, real money only on VALUE".

Three labels, one rule each:
  VALUE   the betting engine's own verdict (a selected bet in the unified slip,
          or a morning candidate that commits at T-30). Read from the slip,
          never recomputed here — shrinkage, bands and vetoes live in
          betting_unified.py.
  LEAN    the best positive-edge angle across every market the system prices
          (ensemble 1x2, O/U blend, Poisson artifact, goal-process simulator,
          player floors incl. goalscorer/assists) against a REAL bookmaker
          price, with 0 < edge <= MAX_EDGE_PCT. Paper-journaled at a flat
          stake in its own journal (PICKS_JOURNAL_PATH) on the T-30 timing, so
          each market builds the CLV/ROI record it needs to earn real stakes
          the way O/U 1.5 did. Never money.
  NO_EDGE the most probable priced outcome, shown with its price and why it is
          not a bet (market already prices it, or the only edges are above the
          overconfidence cap).

Prices come from three artifacts: odds_full.json (h2h, totals), odds_extra_
markets.json (btts, double chance, DNB, alt totals) and pick_markets_raw.json
(per-event first-half, HT/FT, correct score and the eight player-prop markets;
scripts/data/odds_fetcher.fetch_pick_markets). Model rows come from
web/match_markets.build_match_markets — the same list the prediction page
shows, so a pick here is always a row the reader can see there.

Grading: settle_picks() grades what the data can grade — full-time markets from
the settlement results, first-half markets from data/parsed/goal_timeline.parquet,
player props from player_match_stats.parquet — and leaves the rest pending
with one summary line, never a warning per bet.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from config.settings import DATA_DIR
from config.settings import UPCOMING_DIR as UPCOMING

log = logging.getLogger(__name__)

PICKS_FILE = UPCOMING / "picks.json"
PICKS_JOURNAL_PATH = DATA_DIR / "betting" / "picks_journal.json"
PICK_MARKETS_RAW = UPCOMING / "pick_markets_raw.json"
GOAL_TIMELINE = DATA_DIR / "parsed" / "goal_timeline.parquet"
PMS_PATH = DATA_DIR / "external" / "sofascore" / "player_match_stats.parquet"

LABEL_VALUE, LABEL_LEAN, LABEL_NO_EDGE = "VALUE", "LEAN", "NO_EDGE"
MIN_ODDS = 1.10            # below this the price carries no information
OVERCONFIDENCE_CAP = 12.0  # = bet_journal.MAX_EDGE_PCT; >10% edge ran 38% WR live
PICK_JOURNAL_WINDOW_H = 3.0  # journal a LEAN only inside the T-30 timing window
MIN_STAT_ROWS_FOR_DNP = 22   # a match's player stats are "in" when both XIs (≥22 rows) are on disk
MIN_PROB_PCT = 20.0        # a "+9% edge" on a 3% event is inside the model's own error
_TIER_RANK = {"A": 0, "B": 1, "C": 2}
# "Insolite": every priced family outside the mainstream match markets. Shown
# per match in their own slot so a player prop or a first-half angle is visible
# even when a plain 1x2 / totals row wins the headline.
_MAIN_BET_TYPES = {"1x2 finale", "Under/over", "Doppia chance", "Goal"}

# ---------------------------------------------------------------------------
# Model row -> price key. A row with no entry here is shown but never priced
# (Vince o quasi, Minuti, Multi goal, rare events: no market on this feed).
# ---------------------------------------------------------------------------
_H2H_SEL = {"1": "home", "X": "draw", "2": "away"}
_PLAYER_BET_TYPES = {
    "Tiri totali del giocatore": "player_shots",
    "Tiri in porta": "player_shots_on_target",
    "Giocatore marcatore": "player_goal_scorer_anytime",
    "Assist giocatore": "player_assists",
}


def _deaccent(s: str) -> str:
    n = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def _tokens(name: str) -> tuple[str, ...]:
    return tuple(t for t in _deaccent(name).replace("-", " ").replace(".", " ").split() if t)


def _ou_parts(selection: str) -> tuple[str, float] | None:
    parts = (selection or "").split()
    if len(parts) != 2 or parts[0] not in ("Over", "Under"):
        return None
    try:
        return parts[0].lower(), float(parts[1])
    except ValueError:
        return None


def price_key_for_row(row: dict) -> tuple | None:
    """The price-book key a model row is priced against, or None."""
    bt, sel = row.get("bet_type"), row.get("selection")
    if bt == "1x2 finale" and sel in _H2H_SEL:
        return ("h2h", _H2H_SEL[sel], None, None)
    if bt == "1° tempo 1x2" and sel in _H2H_SEL:
        return ("h2h_h1", _H2H_SEL[sel], None, None)
    if bt == "Under/over":
        ou = _ou_parts(sel)
        return ("totals", ou[0], ou[1], None) if ou else None
    if bt == "1° tempo under/over":
        ou = _ou_parts(sel)
        return ("totals_h1", ou[0], ou[1], None) if ou else None
    if bt == "Doppia chance" and sel in ("1X", "X2", "12"):
        return ("double_chance", sel, None, None)
    if bt == "Goal" and sel in ("Sì", "No"):
        return ("btts", "yes" if sel == "Sì" else "no", None, None)
    if bt == "Goal 1° tempo" and sel in ("Sì", "No"):
        return ("btts_h1", "yes" if sel == "Sì" else "no", None, None)
    if bt == "Doppia chance 1° tempo" and sel in ("1X", "X2", "12"):
        return ("double_chance_h1", sel, None, None)
    if bt == "Risultato esatto":
        return ("correct_score", sel, None, None)
    if bt == "Primo tempo / Finale":
        return ("halftime_fulltime", sel, None, None)
    mk = _PLAYER_BET_TYPES.get(bt)
    if mk and row.get("player"):
        player = _deaccent(row["player"])
        if mk in ("player_goal_scorer_anytime",):
            return (mk, "yes", None, player)
        if mk == "player_assists":
            return (mk, "over", 0.5, player)
        ou = _ou_parts(sel)
        return (mk, ou[0], ou[1], player) if ou else None
    return None


# ---------------------------------------------------------------------------
# Price book
# ---------------------------------------------------------------------------
def _entry(prices: list[tuple[float, str]]) -> dict | None:
    prices = [(float(o), b) for o, b in prices if o and float(o) > 1.0]
    if not prices:
        return None
    best = max(prices)
    return {"odds": round(best[0], 3), "book": best[1], "avg": round(mean(o for o, _ in prices), 3),
            "n_books": len(prices)}


def _match_player(sofa_name: str, feed_names: list[str]) -> str | None:
    """Feed name for a Sofascore name: exact (accent-folded), then the
    Sofascore tokens as a subset of the feed tokens ('Gonçalo Ramos' ->
    'Goncalo Matias Ramos'), then unique surname + first initial. None on miss
    or ambiguity — a wrong player is worse than no price."""
    st = _tokens(sofa_name)
    if not st:
        return None
    ft = {f: _tokens(f) for f in feed_names}
    exact = [f for f, t in ft.items() if t == st]
    if len(exact) == 1:
        return exact[0]
    sub = [f for f, t in ft.items() if set(st) <= set(t)]
    if len(sub) == 1:
        return sub[0]
    loose = [f for f, t in ft.items() if t and t[-1] == st[-1] and t[0][:1] == st[0][:1]]
    return loose[0] if len(loose) == 1 else None


def build_price_book(odds_match: dict | None, extra_match: dict | None,
                     pick_event: dict | None) -> dict[tuple, dict]:
    """Every real price for one match keyed (market, selection, point, player)."""
    book: dict[tuple, dict] = {}
    om = odds_match or {}
    h2h = om.get("h2h") or {}
    n = int(h2h.get("bookmakers_count") or 0)
    for sel in ("home", "draw", "away"):
        o = h2h.get(f"best_{sel}")
        if o and n:
            book[("h2h", sel, None, None)] = {"odds": float(o), "book": "best of market",
                                              "avg": h2h.get(sel), "n_books": n}
    for t in om.get("totals") or []:
        try:
            line = float(t.get("line"))
        except (TypeError, ValueError):
            continue
        nb = int(t.get("bookmakers_count") or 0)
        for side in ("over", "under"):
            o = t.get(f"best_{side}")
            if o and nb:
                book[("totals", side, line, None)] = {"odds": float(o), "book": "best of market",
                                                      "avg": t.get(side), "n_books": nb}
    xm = extra_match or {}
    for line_s, t in (xm.get("alternate_totals") or {}).items():
        try:
            line = float(line_s)
        except (TypeError, ValueError):
            continue
        nb = int(t.get("bookmakers_count") or 0)
        for side in ("over", "under"):
            o = t.get(f"best_{side}")
            if o and nb and ("totals", side, line, None) not in book:
                book[("totals", side, line, None)] = {"odds": float(o), "book": "best of market",
                                                      "avg": t.get(side), "n_books": nb}
    bt = xm.get("btts") or {}
    for sel in ("yes", "no"):
        o = bt.get(f"best_{sel}")
        if o:
            book[("btts", sel, None, None)] = {"odds": float(o), "book": "best of market",
                                               "avg": bt.get(sel), "n_books": int(bt.get("bookmakers_count") or 1)}
    for sel, d in (xm.get("double_chance") or {}).items():
        o = (d or {}).get("best")
        if o:
            book[("double_chance", sel, None, None)] = {"odds": float(o), "book": "best of market",
                                                        "avg": d.get("avg"), "n_books": int(d.get("bookmakers_count") or 1)}
    if pick_event:
        home_raw, away_raw = pick_event.get("home_raw", ""), pick_event.get("away_raw", "")
        side_of = {home_raw: "home", away_raw: "away", "Draw": "draw"}
        letter_of = {home_raw: "H", away_raw: "A", "Draw": "D"}
        # specimen 2026-09-05 (Roma vs Atalanta): 'AS Roma or Draw', 'Atalanta BC
        # or Draw', 'AS Roma or Atalanta BC' — the bookmaker's own team strings
        dc_of = {f"{home_raw} or Draw": "1X", f"{away_raw} or Draw": "X2",
                 f"{home_raw} or {away_raw}": "12", f"{away_raw} or {home_raw}": "12"}
        raw: dict[tuple, list[tuple[float, str]]] = {}
        for bm in pick_event.get("bookmakers") or []:
            title = bm.get("title", "?")
            for m in bm.get("markets") or []:
                mk = m.get("key")
                for o in m.get("outcomes") or []:
                    name, desc, point, price = o.get("name"), o.get("description"), o.get("point"), o.get("price")
                    key = None
                    if mk == "h2h_h1" and name in side_of:
                        key = (mk, side_of[name], None, None)
                    elif mk == "totals_h1" and name in ("Over", "Under") and point is not None:
                        key = (mk, name.lower(), float(point), None)
                    elif mk == "btts_h1" and name in ("Yes", "No"):
                        key = (mk, name.lower(), None, None)
                    elif mk == "double_chance_h1" and name in dc_of:
                        key = (mk, dc_of[name], None, None)
                    elif mk == "halftime_fulltime" and name and "/" in name:
                        a, b = name.split("/", 1)
                        if a in letter_of and b in letter_of:
                            key = (mk, f"{letter_of[a]}/{letter_of[b]}", None, None)
                    elif mk == "correct_score" and name and "|" in name:
                        goals = {}
                        for part in name.split("|"):
                            team, _, g = part.rpartition(":")
                            if team in (home_raw, away_raw) and g.isdigit():
                                goals[team] = int(g)
                        if len(goals) == 2:
                            key = (mk, f"{goals[home_raw]}-{goals[away_raw]}", None, None)
                    elif mk and mk.startswith("player_") and desc:
                        if name == "Yes":
                            key = (mk, "yes", None, _deaccent(desc))
                        elif name in ("Over", "Under") and point is not None:
                            key = (mk, name.lower(), float(point), _deaccent(desc))
                    if key is not None:
                        raw.setdefault(key, []).append((price, title))
        for key, prices in raw.items():
            e = _entry(prices)
            if e:
                book[key] = e
    return book


def _feed_players(book: dict[tuple, dict]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = {}
    for mk, _sel, _pt, player in book:
        if player:
            out.setdefault(mk, set()).add(player)
    return {mk: sorted(v) for mk, v in out.items()}


# ---------------------------------------------------------------------------
# Candidates and labels
# ---------------------------------------------------------------------------
def annotate_rows(rows: list[dict], book: dict[tuple, dict]) -> int:
    """Attach the real price (odds, book, n_books, implied, edge) IN PLACE to
    every row that has one; returns how many were priced. Player names are
    matched per market against the feed's own list."""
    feed = _feed_players(book)
    n = 0
    for r in rows:
        key = price_key_for_row(r)
        if key is None:
            continue
        mk, sel, pt, player = key
        if player:
            feed_name = _match_player(r["player"], feed.get(mk) or [])
            if feed_name is None:
                continue
            key = (mk, sel, pt, feed_name)
        e = book.get(key)
        if not e or e["odds"] < MIN_ODDS:
            continue
        p = (r.get("probability_pct") or 0) / 100.0
        if p <= 0:
            continue
        r.update({"market_key": mk, "odds": e["odds"], "book": e["book"], "n_books": e["n_books"],
                  "implied_pct": round(100.0 / e["odds"], 1),
                  "edge_pct": round((p * e["odds"] - 1) * 100, 1)})
        n += 1
    return n


def price_rows(rows: list[dict], book: dict[tuple, dict]) -> list[dict]:
    """Copies of the rows that carry a price (see annotate_rows)."""
    rows = [dict(r) for r in rows]
    annotate_rows(rows, book)
    return [r for r in rows if r.get("odds") is not None]


def match_price_book(match_key: str, league: str = "serie_a") -> dict[tuple, dict]:
    """The price book for one match from the three odds artifacts on disk."""
    odds_file = "odds_full.json" if league == "serie_a" else f"odds_full_{league}.json"
    odds = (_read(UPCOMING / odds_file, {}) or {}).get("matches", {})
    extra = (_read(UPCOMING / "odds_extra_markets.json", {}) or {}).get("matches", {})
    ev = _pick_event_for(_read(PICK_MARKETS_RAW, {}), match_key)
    return build_price_book(odds.get(match_key), extra.get(match_key), ev)


def attach_prices(match_key: str, payload: dict, league: str = "serie_a") -> dict:
    """For the prediction page: price every market/player row of a
    build_match_markets payload in place and attach this match's line from
    PICKS_FILE (label, pick, reason) as payload["pick"]."""
    book = match_price_book(match_key, league)
    # copies: simulator rows come from goal_process._SERVED_CACHE and are shared
    # across requests — pricing must never write into them
    payload["markets"] = [dict(r) for r in payload.get("markets") or []]
    payload["players"] = [dict(r) for r in payload.get("players") or []]
    n = annotate_rows(payload["markets"], book) + annotate_rows(payload["players"], book)
    payload["n_priced"] = n
    picks = _read(PICKS_FILE, {}) or {}
    line = next((p for p in picks.get("picks") or [] if p.get("match") == match_key), None)
    payload["pick"] = None if line is None else {
        k: line.get(k) for k in ("label", "stage", "pick", "lean", "reason", "alternatives", "most_probable",
                                 "exotic", "exotic_fallback", "n_exotic_positive",
                                 "n_priced", "n_positive", "n_overconfident", "n_longshot_edges",
                                 "journaled_bet_id", "prices_fetched_at")}
    if payload["pick"] is not None:
        payload["pick"]["generated_at"] = picks.get("generated_at")
    return payload


def _sel_text(c: dict) -> str:
    return f"{c['player']} {c['selection']}" if c.get("player") else c["selection"]


def rank_candidates(cands: list[dict], band: tuple[float, float]) -> list[dict]:
    """Credible first: inside the band, then tier (A measured > B label > C
    base rate), then a multi-book price over a single book, then edge.
    Long shots (p < MIN_PROB_PCT) and edges above the cap are flagged and
    sink: with 40+ priced rows per match the largest edge is a max over
    noise, so the headline must be the most MEASURED angle, not the biggest."""
    lo, _hi = band
    for c in cands:
        c["overconfident"] = c["edge_pct"] > OVERCONFIDENCE_CAP
        c["longshot"] = (c.get("probability_pct") or 0) < MIN_PROB_PCT
        c["thin"] = (c.get("n_books") or 0) < 2
        c["in_band"] = lo <= c["edge_pct"] <= OVERCONFIDENCE_CAP and not c["longshot"]
    return sorted(cands, key=lambda c: (c["overconfident"] or c["longshot"], not c["in_band"],
                                        _TIER_RANK.get(c.get("tier"), 3), c["thin"], -c["edge_pct"]))


def _engine_verdict(match_key: str, slip: dict, candidates: dict) -> dict | None:
    for b in slip.get("selected_bets") or []:
        if isinstance(b, dict) and b.get("match") == match_key:
            return {"stage": "selected", **b}
    for c in (candidates.get("candidates") if isinstance(candidates, dict) else None) or []:
        if isinstance(c, dict) and c.get("match") == match_key:
            return {"stage": "candidate", **c}
    return None


def _engine_note(match_key: str, c: dict, slip: dict) -> str | None:
    """When the LEAN is a bet the real engine priced and rejected, say so with
    the engine's reason: its edge (shrunk, Pinnacle de-vigged, per-line band)
    is the one that counts for money; this one is the raw gap."""
    if c.get("bet_type") != "Under/over":
        return None
    for nm in slip.get("near_misses") or []:
        if not isinstance(nm, dict) or nm.get("match") != match_key:
            continue
        if str(nm.get("selection", "")).lower() == str(c.get("selection", "")).lower():
            band = f" (band {nm.get('min_edge')}-{nm.get('max_edge')}%)" if nm.get("min_edge") is not None else ""
            e = nm.get("edge_pct")
            at = f" at {e:+.1f}%" if isinstance(e, int | float) else ""
            return f"engine rejected it: {nm.get('reason')}{at}{band}"
    return None


def _pick_view(c: dict) -> dict:
    return {k: c.get(k) for k in ("group", "bet_type", "selection", "player", "team", "tier", "source",
                                  "probability_pct", "odds", "book", "n_books", "implied_pct", "edge_pct",
                                  "in_band", "thin", "market_key", "engine_note",
                                  "lineup", "xi_status", "start_pct", "start_prob")}


def build_match_pick(match_key: str, rows: list[dict], book: dict[tuple, dict], *,
                     slip: dict, candidates: dict, band: tuple[float, float]) -> dict:
    """The line for one match: label, headline pick, three alternatives, counts."""
    priced = rank_candidates(price_rows(rows, book), band)
    verdict = _engine_verdict(match_key, slip, candidates)
    positive = [c for c in priced if c["edge_pct"] > 0 and not c["overconfident"] and not c["longshot"]]
    over = [c for c in priced if c["overconfident"] and not c["longshot"]]
    longshots = [c for c in priced if c["longshot"] and c["edge_pct"] > 0]
    exotic = [c for c in positive if c.get("bet_type") not in _MAIN_BET_TYPES]
    out: dict[str, Any] = {"match": match_key, "n_rows": len(rows), "n_priced": len(priced),
                           "n_positive": len(positive), "n_overconfident": len(over),
                           "n_longshot_edges": len(longshots),
                           "alternatives": [_pick_view(c) for c in positive[:6]],
                           # positive-edge angles outside 1x2 / totals / DC (player props,
                           # first half, HT/FT, first team to score, exact score ...)
                           "exotic": [_pick_view(c) for c in exotic[:3]],
                           "n_exotic_positive": len(exotic)}
    if not exotic:
        # nothing outside the main markets beats its price: show the most
        # probable priced player prop (else any exotic row) with its price, so
        # the reader sees the market is ahead of the model there
        pool = [c for c in priced if c.get("bet_type") not in _MAIN_BET_TYPES and not c["overconfident"]]
        players_first = [c for c in pool if c.get("player")] or pool
        if players_first:
            out["exotic_fallback"] = _pick_view(max(players_first, key=lambda c: c.get("probability_pct") or 0))
    for c in positive[:1]:
        note = _engine_note(match_key, c, slip)
        if note:
            c["engine_note"] = note
    if verdict is not None:
        out["label"] = LABEL_VALUE
        out["stage"] = verdict["stage"]
        out["pick"] = {"bet_type": verdict.get("market"), "selection": verdict.get("selection"),
                       "odds": verdict.get("best_odds") or verdict.get("odds"),
                       "book": verdict.get("best_bookmaker") or verdict.get("bookmaker"),
                       "edge_pct": verdict.get("edge_pct"), "probability_pct": None
                       if verdict.get("model_prob") is None else round(float(verdict["model_prob"]) * 100, 1),
                       "stake": verdict.get("stake_amount") or verdict.get("stake"), "tier": "engine"}
        out["reason"] = ("the betting engine's own selection: real stake, committed at T-30"
                         if verdict["stage"] == "selected" else
                         "morning candidate: the engine commits it at T-30 if the price holds")
        # a LEAN can ride alongside the real bet as the paper record, but never
        # the same bet twice (the real journal already holds it)
        vsel = str(verdict.get("selection") or "").lower()
        other = [c for c in positive
                 if not (c.get("bet_type") == "Under/over" and str(c.get("selection", "")).lower() == vsel)]
        out["lean"] = _pick_view(other[0]) if other else None
        return out
    if positive:
        best = positive[0]
        out["label"] = LABEL_LEAN
        out["pick"] = _pick_view(best)
        out["reason"] = (f"model {best['probability_pct']}% vs market {best['implied_pct']}% "
                         f"({best['book']}, {best['n_books']} book{'s' if best['n_books'] != 1 else ''}); "
                         + ("edge inside the credible band, paper stake" if best["in_band"] else
                            f"edge below the {band[0]:.0f}% band, paper only")
                         + (f"; {best['engine_note']}" if best.get("engine_note") else ""))
        return out
    out["label"] = LABEL_NO_EDGE
    out["pick"] = None
    if priced:
        top = max(priced, key=lambda c: c.get("probability_pct") or 0)
        out["most_probable"] = _pick_view(top)
        if over and not positive:
            out["reason"] = (f"only edges above the {OVERCONFIDENCE_CAP:.0f}% cap (best "
                             f"{over[0]['edge_pct']:+.1f}%): model overconfidence, not value")
        else:
            out["reason"] = (f"most probable is {_sel_text(top)} at {top['probability_pct']}%, "
                             f"but the market prices it at {top['implied_pct']}% ({top['odds']}): no edge")
    else:
        out["reason"] = "no market price for any row (per-event odds not fetched yet or no lineup)"
    return out


# ---------------------------------------------------------------------------
# Slate
# ---------------------------------------------------------------------------
def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _pick_event_for(store: dict, match_key: str) -> dict | None:
    for ev in (store.get("events") or {}).values():
        if f"{ev.get('home')} vs {ev.get('away')}" == match_key:
            return ev
    return None


def _kickoff(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def build_picks(league: str = "serie_a", *, journal: bool = False,
                now: datetime | None = None) -> dict:
    """Build the slate (every upcoming match of `league`), write PICKS_FILE,
    and — on the T-30 path only (`journal=True`) — paper-journal each LEAN
    whose kickoff is inside PICK_JOURNAL_WINDOW_H."""
    from scripts.betting.betting_unified import BettingConfig
    from web.match_markets import assemble_market_inputs, build_match_markets

    cfg = BettingConfig()
    band = (cfg.min_edge_pct, cfg.max_edge_pct)
    now = now or datetime.now(UTC)
    pred_file = "predictions.json" if league == "serie_a" else f"predictions_{league}.json"
    preds_raw = _read(UPCOMING / pred_file, {})
    preds = preds_raw.get("predictions", []) if isinstance(preds_raw, dict) else preds_raw
    odds_file = "odds_full.json" if league == "serie_a" else f"odds_full_{league}.json"
    odds = (_read(UPCOMING / odds_file, {}) or {}).get("matches", {})
    extra = (_read(UPCOMING / "odds_extra_markets.json", {}) or {}).get("matches", {})
    store = _read(PICK_MARKETS_RAW, {})
    slip = _read(UPCOMING / "unified_bet_slip.json", {}) or {}
    candidates = _read(UPCOMING / "betting_candidates.json", {}) or {}
    today = now.strftime("%Y-%m-%d")
    try:
        from scripts.betting.market_promotion import load_state
        promo_state = load_state()   # which paper markets have earned real stakes
    except Exception:  # noqa: BLE001
        promo_state = {"markets": {}}

    picks, n_journaled = [], 0
    for pred in preds:
        if not isinstance(pred, dict) or pred.get("league", "serie_a") != league:
            continue
        if (pred.get("date") or "9999")[:10] < today:
            continue
        match_key = pred.get("match", "")
        inputs = assemble_market_inputs(match_key, pred=pred, load_json=_read, league=league)
        payload = build_match_markets(match_key, engine_bet=None, **inputs)
        # match rows AND player rows: the first slate priced only "markets" and
        # never saw a player prop (caught by the exotic fallback, 2026-09-05)
        rows = (payload.get("markets") or []) + (payload.get("players") or [])
        om = odds.get(match_key) or {}
        ev = _pick_event_for(store, match_key)
        book = build_price_book(om, extra.get(match_key), ev)
        line = build_match_pick(match_key, rows, book, slip=slip, candidates=candidates, band=band)
        ko = _kickoff(om.get("commence_time") or (ev or {}).get("commence"))
        if ko and ko < now:
            # kicked off: the /picks card listed Roma–Atalanta 40 minutes into
            # the match (2026-09-05) because the date filter is day-resolution
            continue
        xi_states = {r.get("lineup") for r in (payload.get("players") or []) if r.get("lineup")}
        line["lineup_state"] = ("confirmed" if xi_states == {"confirmed"} else
                                "predicted" if "predicted" in xi_states else
                                "recent" if xi_states else None)
        line.update({"date": pred.get("date"), "kickoff_utc": ko.isoformat() if ko else None,
                     "league": league, "home_team": pred.get("home_team"), "away_team": pred.get("away_team"),
                     "prices_fetched_at": (ev or {}).get("fetched_at")})
        lean = line.get("pick") if line["label"] == LABEL_LEAN else line.get("lean")
        if journal and ko and timedelta(0) <= (ko - now) <= timedelta(hours=PICK_JOURNAL_WINDOW_H):
            # the headline LEAN and the best exotic angle both build a record;
            # add_bet dedups when they are the same bet
            ids = []
            real_ids = []
            for cand in (lean, (line.get("exotic") or [None])[0]):
                if cand:
                    bet_id = journal_lean(match_key, pred.get("date") or today, cand, league, placed_at=now)
                    if bet_id and bet_id not in ids:
                        ids.append(bet_id)
                        # a market that cleared the promotion bar is ALSO a real
                        # bet, Kelly-sized, linked to this paper entry
                        real_id = _mirror_if_promoted(bet_id, promo_state)
                        if real_id:
                            real_ids.append(real_id)
            if ids:
                n_journaled += len(ids)
                line["journaled_bet_id"] = ids[0]
                line["journaled_bet_ids"] = ids
            if real_ids:
                line["real_bet_ids"] = real_ids
        picks.append(line)

    order = {LABEL_VALUE: 0, LABEL_LEAN: 1, LABEL_NO_EDGE: 2}
    picks.sort(key=lambda p: (p.get("date") or "", order.get(p["label"], 3),
                              -((p.get("pick") or {}).get("edge_pct") or 0)))
    out = {"generated_at": now.isoformat(), "league": league, "band": list(band),
           "overconfidence_cap": OVERCONFIDENCE_CAP,
           "counts": {lab: sum(1 for p in picks if p["label"] == lab) for lab in order},
           "n_journaled": n_journaled, "picks": picks}
    from config.settings import atomic_write_json
    atomic_write_json(PICKS_FILE, out)
    log.info("Picks: %d matches (%s), %d LEAN paper-journaled", len(picks),
             ", ".join(f"{v} {k}" for k, v in out["counts"].items()), n_journaled)
    return out


def _mirror_if_promoted(picks_bet_id: str, state: dict | None) -> str | None:
    """Real-journal mirror of a paper pick whose market is promoted
    (scripts/betting/market_promotion.py). Reads the paper entry back so the
    real one carries exactly what was journaled. Never raises: a failure here
    must not stop the paper record."""
    try:
        from scripts.betting.bet_journal import _load_journal
        from scripts.betting.market_promotion import is_promoted, journal_promoted
        entry = (_load_journal(PICKS_JOURNAL_PATH).get("bets") or {}).get(picks_bet_id)
        if not entry or entry.get("status") != "pending" or not is_promoted(entry.get("market") or "", state):
            return None
        return journal_promoted(entry, picks_bet_id)
    except Exception as e:  # noqa: BLE001
        log.warning("promoted mirror failed for %s: %s", picks_bet_id, e)
        return None


def journal_lean(match_key: str, date: str, lean: dict, league: str, *, placed_at: datetime) -> str:
    """Flat-stake paper entry in PICKS_JOURNAL_PATH. Selection embeds the
    player so two players' 'Over 0.5' never collide on the journal's
    match+selection dedup."""
    from scripts.betting.bet_journal import add_bet
    from scripts.betting.betting_unified import PAPER_STAKE
    ou = _ou_parts(lean.get("selection") or "") if lean.get("market_key") != "player_assists" else ("over", 0.5)
    return add_bet({
        "match": match_key, "date": date, "league": league,
        "market": lean["market_key"], "selection": _sel_text(lean),
        "bet_type": lean.get("bet_type"), "player": lean.get("player"), "team": lean.get("team"),
        "model_prob": round((lean.get("probability_pct") or 0) / 100.0, 4),
        # implied_pct is 1/odds of the PLACED price: written as sharp_implied_prob
        # it made every paper CLV exactly 0.0 (a fake measurement). CLV for a
        # pick comes from the closing price the grader passes at settle time.
        "sharp_implied_prob": None,
        "edge_pct": lean.get("edge_pct"), "odds": lean.get("odds"), "bookmaker": lean.get("book"),
        "stake": PAPER_STAKE, "confidence": lean.get("tier"), "placed_at": placed_at.isoformat(),
        "pipeline_status": "pick:lean",
        "extra": {"bet_type": lean.get("bet_type"), "player": lean.get("player"), "team": lean.get("team"),
                  "source": lean.get("source"), "tier": lean.get("tier"),
                  "side": (ou[0] if ou else None), "line": (ou[1] if ou else None),
                  # the XI basis at journal time, so the paper record can be split
                  # confirmed vs predicted before any market earns real stakes
                  "lineup": lean.get("lineup"), "start_pct": lean.get("start_pct")},
    }, journal_path=PICKS_JOURNAL_PATH)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
def _grade(bet: dict, res: dict | None, h1: tuple[int, int] | None,
           player_row: dict | None) -> str | None:
    """'won' / 'lost' / 'push' / None (not gradable yet). `res` carries the
    full-time score, `h1` the first-half score, `player_row` the player's
    counts for the match. Side/line come from the entry's `extra`."""
    mk = bet.get("market") or ""
    sel = bet.get("selection") or ""
    ex = bet.get("extra") or {}
    ft = None if res is None else (int(res["home_score"]), int(res["away_score"]))

    def _ou(total: float) -> str | None:
        side, line = ex.get("side"), ex.get("line")
        if side not in ("over", "under") or line is None:
            return None
        if total == line:
            return "push"
        return "won" if ((total > line) == (side == "over")) else "lost"

    def _wdl(score: tuple[int, int]) -> str:
        return "H" if score[0] > score[1] else "A" if score[1] > score[0] else "D"

    sym = {"1": "H", "X": "D", "2": "A"}
    if mk == "h2h" and ft:
        return "won" if _wdl(ft) == sym.get(sel) else "lost"
    if mk == "totals" and ft:
        return _ou(ft[0] + ft[1])
    if mk == "double_chance" and ft:
        return "won" if _wdl(ft) in {"1X": "HD", "X2": "DA", "12": "HA"}.get(sel, "") else "lost"
    if mk == "btts" and ft:
        return "won" if (ft[0] > 0 and ft[1] > 0) == (sel == "Sì") else "lost"
    if mk == "correct_score" and ft:
        return "won" if sel == f"{ft[0]}-{ft[1]}" else "lost"
    if mk == "h2h_h1" and h1:
        return "won" if _wdl(h1) == sym.get(sel) else "lost"
    if mk == "totals_h1" and h1:
        return _ou(h1[0] + h1[1])
    if mk == "btts_h1" and h1:
        return "won" if (h1[0] > 0 and h1[1] > 0) == (sel == "Sì") else "lost"
    if mk == "double_chance_h1" and h1:
        return "won" if _wdl(h1) in {"1X": "HD", "X2": "DA", "12": "HA"}.get(sel, "") else "lost"
    if mk == "halftime_fulltime" and h1 and ft:
        return "won" if sel == f"{_wdl(h1)}/{_wdl(ft)}" else "lost"
    if mk.startswith("player_") and player_row is not None:
        col = {"player_shots": "total_shots", "player_shots_on_target": "shots_on_target",
               "player_goal_scorer_anytime": "goals", "player_assists": "assists"}.get(mk)
        if col is None:
            return None
        if player_row.get("_did_not_play"):
            # the match's stats are in and the player has no row (or 0 minutes):
            # he never entered — books void a player prop, so does the paper record
            return "void"
        count = int(player_row.get(col) or 0)
        if mk == "player_goal_scorer_anytime":
            return "won" if count >= 1 else "lost"
        return _ou(count)
    return None


def _first_half_scores() -> dict[str, tuple[int, int]]:
    if not GOAL_TIMELINE.exists():
        return {}
    import pandas as pd
    t = pd.read_parquet(GOAL_TIMELINE, columns=["match_id", "side", "half"])
    h1 = t[t["half"] == 1]
    out: dict[str, tuple[int, int]] = {}
    for mid, g in h1.groupby("match_id"):
        out[str(mid)] = (int((g["side"] == "home").sum()), int((g["side"] == "away").sum()))
    return out


def _timeline_match_ids() -> set[str]:
    if not GOAL_TIMELINE.exists():
        return set()
    import pandas as pd
    return set(pd.read_parquet(GOAL_TIMELINE, columns=["match_id"])["match_id"].astype(str))


def settle_picks(results: dict[str, dict] | None = None) -> dict:
    """Grade pending picks. Full-time markets need `results` (auto_settle's
    completed dict); first-half markets need the goal timeline to hold the
    match; player props need the player's row in player_match_stats. Anything
    else stays pending and is counted once."""
    from scripts.betting.bet_journal import get_pending_bets, settle_bet
    pending = get_pending_bets(journal_path=PICKS_JOURNAL_PATH)
    summary = {"settled": 0, "won": 0, "push": 0, "voided": 0, "pending": len(pending),
               "ungradable": 0}
    if not pending:
        return summary
    results = results or {}
    h1_scores = _first_half_scores()
    tl_ids = set(h1_scores) | _timeline_match_ids()
    pms = None
    for bet in pending:
        match = bet.get("match", "")
        home, _, away = match.partition(" vs ")
        res = results.get(match)
        canon = f"{bet.get('date')}_{home}_{away}"
        h1 = h1_scores.get(canon) if canon in tl_ids else None
        if canon in tl_ids and h1 is None:
            h1 = (0, 0)  # the timeline holds the match with no first-half goal
        player_row = None
        player = (bet.get("extra") or {}).get("player")
        if player and PMS_PATH.exists():
            if pms is None:
                import pandas as pd
                pms = pd.read_parquet(PMS_PATH, columns=["date", "team", "player_name", "goals", "assists",
                                                         "total_shots", "shots_on_target", "minutes"])
            sub = pms[(pms["date"].astype(str) == str(bet.get("date"))) & (pms["team"].isin([home, away]))]
            hit = sub[sub["player_name"].map(_deaccent) == _deaccent(player)]
            if len(hit) == 1:
                player_row = hit.iloc[0].to_dict()
                if not float(player_row.get("minutes") or 0):
                    player_row["_did_not_play"] = True
            elif len(hit) == 0 and len(sub) >= MIN_STAT_ROWS_FOR_DNP:
                # both squads' stats are on disk and he is not among them:
                # benched and never used (Pašalić, 2026-09-05) -> void, not pending
                player_row = {"_did_not_play": True}
        status = (res or {}).get("status", "").lower() if res else ""
        if status in ("postponed", "cancelled", "suspended", "walkover"):
            outcome = "void"
        else:
            outcome = _grade(bet, res if res and res.get("home_score") is not None else None, h1, player_row)
        if outcome is None:
            summary["ungradable"] += 1
            continue
        stake = float(bet.get("stake") or 0)
        odds = float(bet.get("odds") or 0)
        profit = {"won": round(stake * (odds - 1), 2), "lost": -stake}.get(outcome, 0.0)
        score = None if res is None or res.get("home_score") is None else f"{int(res['home_score'])}-{int(res['away_score'])}"
        kickoff = (res or {}).get("commence_time") or None
        closing = closing_price_for(bet)
        if settle_bet(bet.get("bet_id", ""), outcome, result_score=score, profit=profit,
                      match_kickoff_at=kickoff, journal_path=PICKS_JOURNAL_PATH, closing_odds=closing):
            summary["settled"] += 1
            summary["pending"] -= 1
            if outcome in ("won", "push"):
                summary[outcome] += 1
            elif outcome == "void":
                summary["voided"] += 1
            try:
                from scripts.betting.market_promotion import settle_linked
                summary["real_settled"] = summary.get("real_settled", 0) + settle_linked(
                    bet.get("bet_id", ""), outcome, result_score=score, match_kickoff_at=kickoff, closing_odds=closing)
            except Exception as e:  # noqa: BLE001 - the paper record must settle even if the mirror fails
                log.warning("linked real settle failed for %s: %s", bet.get("bet_id"), e)
    if summary["settled"] or summary["ungradable"]:
        log.info("Picks settle: %(settled)d settled (%(won)d W, %(push)d P, %(voided)d V), "
                 "%(pending)d pending, %(ungradable)d not gradable yet", summary)
    if summary["settled"]:
        try:
            from scripts.betting.market_promotion import evaluate_promotions
            evaluate_promotions()
        except Exception as e:  # noqa: BLE001
            log.warning("promotion evaluation failed: %s", e)
    return summary


def closing_price_for(bet: dict) -> float | None:
    """The last price the feed held for a journaled pick: best across books in
    pick_markets_raw.json (refreshed every 45 min per event down to kickoff),
    or the O/U / h2h artifact for the match markets. None when the feed no
    longer holds the event or the key is not priced — then no CLV is claimed."""
    try:
        ex = bet.get("extra") or {}
        mk = bet.get("market") or ""
        sel_row = {"bet_type": ex.get("bet_type"), "selection": (bet.get("selection") or "").replace(f"{ex.get('player')} ", "", 1)
                   if ex.get("player") else bet.get("selection"), "player": ex.get("player")}
        key = price_key_for_row(sel_row)
        if key is None:
            return None
        match_key = bet.get("match") or ""
        ev = _pick_event_for(_read(PICK_MARKETS_RAW, {}), match_key)
        odds_all = (_read(UPCOMING / "odds_full.json", {}) or {}).get("matches", {})
        extra_all = (_read(UPCOMING / "odds_extra_markets.json", {}) or {}).get("matches", {})
        book = build_price_book(odds_all.get(match_key), extra_all.get(match_key), ev)
        if key[3]:  # player key: the feed's own spelling
            feed = _feed_players(book).get(mk) or []
            name = _match_player(key[3], feed)
            if name is None:
                return None
            key = (key[0], key[1], key[2], name)
        e = book.get(key)
        return float(e["odds"]) if e else None
    except Exception as e:  # noqa: BLE001
        log.debug("closing price lookup failed: %s", e)
        return None


def picks_record(league: str | None = None) -> dict:
    """Per-market paper record: the bar a market must clear to earn stakes."""
    from scripts.betting.bet_journal import get_pending_bets, get_settled_bets
    settled = get_settled_bets(journal_path=PICKS_JOURNAL_PATH)
    pending = get_pending_bets(journal_path=PICKS_JOURNAL_PATH, include_superseded=False)
    if league:
        settled = [b for b in settled if b.get("league") == league]
        pending = [b for b in pending if b.get("league") == league]
    by_market: dict[str, dict] = {}
    for b in settled:
        m = by_market.setdefault(b.get("market") or "?", {"n": 0, "won": 0, "profit": 0.0, "clv": []})
        m["n"] += 1
        m["won"] += b.get("status") == "won"
        m["profit"] += float(b.get("profit") or 0)
        if b.get("clv_pct") is not None:
            m["clv"].append(float(b["clv_pct"]))
    for m in by_market.values():
        stake = 10.0 * m["n"]
        m["roi_pct"] = round(m["profit"] / stake * 100, 1) if stake else None
        m["mean_clv_pct"] = round(mean(m["clv"]), 2) if m["clv"] else None
        m["n_clv"] = len(m.pop("clv"))
        m["profit"] = round(m["profit"], 2)
    return {"n_settled": len(settled), "n_pending": len(pending), "by_market": by_market}


if __name__ == "__main__":  # pragma: no cover
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--journal", action="store_true", help="paper-journal LEAN picks inside the T-30 window")
    ap.add_argument("--settle", action="store_true", help="grade pending picks from the parquets (no results dict)")
    ap.add_argument("--record", action="store_true", help="print the per-market paper record")
    ap.add_argument("--league", default="serie_a")
    a = ap.parse_args()
    if a.settle:
        print(json.dumps(settle_picks(), indent=1))
    elif a.record:
        print(json.dumps(picks_record(a.league), indent=1))
    else:
        out = build_picks(a.league, journal=a.journal)
        for p in out["picks"]:
            pk = p.get("pick") or p.get("most_probable") or {}
            print(f"{p['date']} {p['label']:<7} {p['match']:<26} {pk.get('bet_type') or '-'} "
                  f"{(pk.get('player') + ' ') if pk.get('player') else ''}{pk.get('selection') or ''} "
                  f"@ {pk.get('odds') or '-'} edge {pk.get('edge_pct') if pk.get('edge_pct') is not None else '-'} "
                  f"[{pk.get('tier') or '-'}]  {p['reason']}")
