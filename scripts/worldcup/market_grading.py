"""Per-market grading for played World Cup matches.

Given an archived pre-kickoff snapshot (home_xg / away_xg / 1X2 probabilities) and
the real result, this reconstructs the full tipster market menu — the SAME numbers
the /worldcup page shows as insight — and grades each market against reality:

  * which pick the model would have made (highest-probability side),
  * whether that pick HIT,
  * how DISTANT it was from being right (goals off, score off, margin off),
  * the AI-suggested STAKE (fractional Kelly vs the Sofascore-proxy / fair odds) and
    the PAYOUT had it hit.

Two honesty tiers:
  * "settled" markets resolve from the final 90' score alone (1X2, O/U, BTTS, exact
    score, handicap, margin, multigoal, odd/even, win-to-nil …).
  * "needs goal timing" markets (HT/FT, first/second half, both-halves) resolve ONLY
    when half-time can be reconstructed from complete scorer-minute coverage; otherwise
    they are returned ungraded rather than guessed.

Edge honesty mirrors the rest of the page: the who-wins family (1X2 / DC / handicap)
inherits the backtested edge; goal-quantity props are flagged ``edge="display"`` so the
suggested stake on them is shown as illustrative, never as a real recommendation.

No new model. Everything rides ``scripts.betting.extended_markets`` (calibrated Poisson),
exactly like the live page. Pure functions, no I/O — the API layer feeds in results.
"""

from __future__ import annotations

import math
from typing import Any

from scripts.betting import extended_markets as em

# --------------------------------------------------------------------------- #
# Stake model
# --------------------------------------------------------------------------- #
# Notional bankroll the "AI suggested stake" is sized against. This is a display
# bankroll for the record page, independent of Nicola's real my_bankroll.json.
NOTIONAL_BANKROLL = 100.0
# Fractional Kelly — quarter-Kelly is the house convention for this project's
# staking (full Kelly is too swingy for a 53-55%% edge). Only ever applied to the
# who-wins family; goal props get a flat token stake flagged display-only.
KELLY_FRACTION = 0.25
# Token stake (units of bankroll) shown for display-only goal props so the
# "what could I have won" column is populated without implying a real bet.
DISPLAY_TOKEN_STAKE = 1.0
# Min edge over the priced odds before the Kelly sizer suggests anything.
MIN_EDGE = 0.02

# Families whose pick carries the backtested who-wins edge (everything else is
# display-grade on internationals — see model_metadata.json / page disclaimers).
EDGE_FAMILIES = {"1x2", "double_chance", "european_handicap"}


def _kelly_stake(prob: float, dec_odds: float) -> float:
    """Quarter-Kelly stake in CURRENCY off the notional bankroll.

    Returns 0 when there is no edge at the offered odds. ``dec_odds`` is the
    price the bet would be struck at (proxy market odds if known, else fair).
    """
    if dec_odds <= 1.0 or prob <= 0.0:
        return 0.0
    b = dec_odds - 1.0
    edge = prob * dec_odds - 1.0  # expected value per unit staked
    if edge <= MIN_EDGE:
        return 0.0
    kelly = (prob * b - (1.0 - prob)) / b  # full-Kelly fraction
    if kelly <= 0:
        return 0.0
    return round(NOTIONAL_BANKROLL * KELLY_FRACTION * kelly, 2)


def _suggestion(
    prob: float, fair_odds: float, priced_odds: float | None, is_edge: bool
) -> dict[str, Any]:
    """Stake + payout-if-hit for one market pick.

    Edge families get a real quarter-Kelly stake at the best available price.
    Display-only props get a flat token stake so the payout column is populated
    but clearly illustrative (``kind="display"``).
    """
    price = priced_odds if (priced_odds and priced_odds > 1.0) else fair_odds
    has_price = bool(price and price < 99)
    if is_edge:
        stake = _kelly_stake(prob, price)
        kind = "kelly" if stake > 0 else "no_edge"
    else:
        stake = DISPLAY_TOKEN_STAKE
        kind = "display"
    # Payout always reflects the actual recommended stake (0 when no edge).
    payout = round(stake * price, 2) if has_price else 0.0
    profit = round(payout - stake, 2)
    # Reference return on a flat 1-unit bet, so the "what it pays" column is
    # never empty even on no-edge edge markets — purely illustrative.
    unit_payout = round(price, 2) if has_price else None
    return {
        "stake": stake,
        "price": round(price, 2) if has_price else None,
        "priced_from": "market" if (priced_odds and priced_odds > 1.0) else "fair",
        "payout_if_hit": payout,
        "profit_if_hit": profit,
        "unit_payout": unit_payout,  # 1-unit reference return at this price
        "kind": kind,
    }


def _top(items: list[dict], key: str = "prob") -> dict | None:
    """Highest-probability entry of a market's option list."""
    if not items:
        return None
    return max(items, key=lambda x: x.get(key, 0.0))


def _top_kv(d: dict) -> tuple[str, dict] | None:
    """Highest-probability (key, value) of a {option: {prob, fair_odds}} dict."""
    opts = [(k, v) for k, v in d.items() if isinstance(v, dict) and "prob" in v]
    if not opts:
        return None
    return max(opts, key=lambda kv: kv[1].get("prob", 0.0))


# --------------------------------------------------------------------------- #
# Grading primitives
# --------------------------------------------------------------------------- #
def _btts_yes(hxg: float, axg: float) -> float:
    y = (1 - math.exp(-hxg)) * (1 - math.exp(-axg))
    return max(0.0, min(1.0, y))


def _market_row(
    family: str,
    label: str,
    pick: str,
    prob: float,
    fair_odds: float,
    hit: bool | None,
    actual: str,
    distance: float | None,
    distance_txt: str,
    priced_odds: float | None,
) -> dict[str, Any]:
    """Assemble one graded market row in the shape the UI consumes."""
    is_edge = family in EDGE_FAMILIES
    row: dict[str, Any] = {
        "family": family,
        "label": label,            # human market name, e.g. "Over/Under 2.5"
        "pick": pick,              # the model's selection, e.g. "Over 2.5"
        "prob": round(prob, 4),
        "fair_odds": round(fair_odds, 2) if fair_odds and fair_odds < 99 else None,
        "edge": "model" if is_edge else "display",
        "hit": hit,                # True / False / None(ungraded)
        "actual": actual,          # what reality settled to, human text
        "distance": distance,      # numeric distance-to-correct (None = binary/NA)
        "distance_txt": distance_txt,
    }
    row.update(_suggestion(prob, fair_odds, priced_odds, is_edge))
    return row


def grade_match_markets(
    snapshot: dict[str, Any],
    home_goals: int,
    away_goals: int,
    ht: tuple[int, int] | None = None,
    market_odds: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Grade the full market menu for one played match.

    Parameters
    ----------
    snapshot
        Archived pre-kickoff snapshot — needs ``home_xg``, ``away_xg`` and a
        ``probabilities`` dict (home/draw/away). This is what we froze BEFORE
        kickoff, so the grade is honest.
    home_goals, away_goals
        Final (90') score — already reconstructed honestly by the caller.
    ht
        Half-time score if known (from complete scorer coverage), else None.
    market_odds
        Optional ``{"odds": {...}, "implied": {...}}`` Sofascore-proxy block for
        this match (1X2 prices) so edge stakes are struck at the real price.

    Returns a list of graded-market rows (see ``_market_row``), display order
    roughly strongest→weakest family.
    """
    hxg = snapshot.get("home_xg")
    axg = snapshot.get("away_xg")
    if hxg is None or axg is None:
        return []
    try:
        hxg, axg = float(hxg), float(axg)
    except (TypeError, ValueError):
        return []
    if hxg <= 0 or axg <= 0:
        return []

    probs_raw = snapshot.get("probabilities") or {}
    p1x2 = {
        "home": float(probs_raw.get("home", probs_raw.get("H", 0.0))),
        "draw": float(probs_raw.get("draw", probs_raw.get("D", 0.0))),
        "away": float(probs_raw.get("away", probs_raw.get("A", 0.0))),
    }

    hg, ag = int(home_goals), int(away_goals)
    total = hg + ag
    margin = hg - ag
    outcome = "home" if hg > ag else "draw" if hg == ag else "away"
    odds_blk = (market_odds or {}).get("odds", {}) if isinstance(market_odds, dict) else {}

    rows: list[dict[str, Any]] = []

    # --- 1X2 (the headline edge market) -----------------------------------
    pick_1x2 = max(("home", "draw", "away"), key=lambda k: p1x2[k])
    label_map = {
        "home": snapshot.get("home_team", "Home"),
        "draw": "Draw",
        "away": snapshot.get("away_team", "Away"),
    }
    # market price for the picked side, if we have it
    mk_1x2 = None
    if isinstance(odds_blk, dict):
        mk_1x2 = odds_blk.get({"home": "home", "draw": "draw", "away": "away"}[pick_1x2])
    rows.append(_market_row(
        "1x2", "Match result (1X2)", str(label_map[pick_1x2]),
        p1x2[pick_1x2], round(1 / p1x2[pick_1x2], 2) if p1x2[pick_1x2] > 0 else 99,
        pick_1x2 == outcome,
        f"{label_map[outcome]} ({hg}-{ag})",
        None, "exact result — hit or miss",
        mk_1x2,
    ))

    # --- Double chance ----------------------------------------------------
    dc = em.compute_double_chance(p1x2)
    dc_cover = {"1X": ("home", "draw"), "X2": ("draw", "away"), "12": ("home", "away")}
    dc_top = _top_kv(dc)
    if dc_top:
        k, v = dc_top
        dc_lbls = {"1X": "Home or Draw", "X2": "Draw or Away", "12": "Home or Away"}
        rows.append(_market_row(
            "double_chance", "Double chance", dc_lbls.get(k, k),
            v["prob"], v.get("fair_odds", 99), outcome in dc_cover[k],
            f"{label_map[outcome]} won" if outcome != "draw" else "Draw",
            None, "covers two outcomes — hit or miss", None,
        ))

    # --- Over/Under total goals (every standard line) ----------------------
    for line in (0.5, 1.5, 2.5, 3.5, 4.5):
        k = int(line)
        over_p = 1 - em.poisson_cdf(k, hxg + axg)
        under_p = em.poisson_cdf(k, hxg + axg)
        over = over_p >= under_p
        prob = over_p if over else under_p
        pick = f"Over {line}" if over else f"Under {line}"
        hit = (total > line) if over else (total < line)
        side_now = "Over" if total > line else "Under"
        # distance = how many goals would need to change to flip the line.
        # line is .5 so it never pushes; e.g. line 2.5, total 4 → 2 goals clear.
        gap = int(round(abs(total - line) + 0.0001))  # 1.5,0.5 -> 1; 2.5->2 ...
        rows.append(_market_row(
            "over_under", f"Total goals O/U {line}", pick,
            prob, round(1 / prob, 2) if prob > 0 else 99, hit,
            f"{total} goals → {side_now} {line}",
            float(gap),
            f"{gap} goal(s) {'clear of' if gap > 0 else 'from'} the line",
            None,
        ))

    # --- BTTS -------------------------------------------------------------
    by = _btts_yes(hxg, axg)
    btts_yes_pick = by >= 0.5
    btts_actual_yes = hg > 0 and ag > 0
    rows.append(_market_row(
        "btts", "Both teams to score", "Yes" if btts_yes_pick else "No",
        by if btts_yes_pick else 1 - by,
        round(1 / (by if btts_yes_pick else 1 - by), 2) if (by if btts_yes_pick else 1 - by) > 0 else 99,
        btts_yes_pick == btts_actual_yes,
        "Both scored" if btts_actual_yes else f"{'Only one' if total else 'Neither'} scored",
        None, "yes/no — hit or miss", None,
    ))

    # --- Exact score (top pick from the grid) ------------------------------
    es = em.compute_exact_score_top10(hxg, axg)
    es_top = _top(es)
    if es_top:
        eh, ea = es_top["home_goals"], es_top["away_goals"]
        es_dist = abs(eh - hg) + abs(ea - ag)
        rows.append(_market_row(
            "exact_score", "Correct score", es_top["score"],
            es_top["prob"], es_top.get("fair_odds", 99), (eh == hg and ea == ag),
            f"{hg}-{ag}",
            float(es_dist),
            "exact hit" if es_dist == 0 else f"{es_dist} goal(s) off the real score",
            None,
        ))

    # --- Winning margin ---------------------------------------------------
    wm = em.compute_winning_margin(hxg, axg)
    wm_top = _top_kv(wm)
    if wm_top:
        k, v = wm_top
        wm_pred = {
            "draw": 0, "home_by_1": 1, "home_by_2": 2, "home_by_3_plus": 3,
            "away_by_1": -1, "away_by_2": -2, "away_by_3_plus": -3,
        }
        wm_lbls = {
            "draw": "Draw", "home_by_1": "Home by 1", "home_by_2": "Home by 2",
            "home_by_3_plus": "Home by 3+", "away_by_1": "Away by 1",
            "away_by_2": "Away by 2", "away_by_3_plus": "Away by 3+",
        }
        pred_m = wm_pred[k]
        if k.endswith("3_plus"):
            hit = (margin >= 3) if "home" in k else (margin <= -3)
        else:
            hit = margin == pred_m
        # distance in goals between predicted margin band and actual margin
        if k.endswith("3_plus"):
            ref = 3 if "home" in k else -3
            wm_dist = max(0, (ref - margin) if "home" in k else (margin - ref))
        else:
            wm_dist = abs(margin - pred_m)
        rows.append(_market_row(
            "winning_margin", "Winning margin", wm_lbls.get(k, k),
            v["prob"], v.get("fair_odds", 99), hit,
            f"margin {margin:+d}", float(wm_dist),
            "exact margin" if wm_dist == 0 else f"{wm_dist} goal(s) off the margin band",
            None,
        ))

    # --- European handicap (home -1, the classic favourite line) ----------
    eh_all = em.compute_european_handicap(hxg, axg)
    # pick the most informative single line: home -1 if home favoured else away -1
    fav_home = p1x2["home"] >= p1x2["away"]
    hc_key = "home_-1"
    hc = eh_all.get(hc_key)
    if isinstance(hc, dict):
        hc_top = _top_kv(hc)
        if hc_top:
            side, v = hc_top
            adj_margin = margin - 1  # home -1
            adj_outcome = "home" if adj_margin > 0 else "draw" if adj_margin == 0 else "away"
            hc_lbls = {"home": "Home -1", "draw": "Tie (-1)", "away": "Away +1"}
            rows.append(_market_row(
                "european_handicap", "Handicap (Home -1)", hc_lbls.get(side, side),
                v["prob"], v.get("fair_odds", 99), side == adj_outcome,
                f"home -1 → {adj_outcome} ({hg-1}-{ag})",
                float(abs(adj_margin)) if side != adj_outcome else 0.0,
                "covered" if side == adj_outcome else f"{abs(adj_margin)} goal(s) short of cover",
                None,
            ))

    # --- Odd / Even total -------------------------------------------------
    oe = em.compute_odd_even(hxg, axg)
    oe_top = _top_kv(oe)
    if oe_top:
        k, v = oe_top
        actual_oe = "odd" if total % 2 == 1 else "even"
        rows.append(_market_row(
            "odd_even", "Total goals odd/even", k.capitalize(),
            v["prob"], v.get("fair_odds", 99), k == actual_oe,
            f"{total} is {actual_oe}", None, "odd/even — hit or miss", None,
        ))

    # --- Multigoal range --------------------------------------------------
    mg = em.compute_multi_goal(hxg, axg)
    mg_top = _top(mg)
    if mg_top:
        rng = mg_top["range"]
        lo, hi = _parse_range(rng)
        hit = (total >= lo) and (hi is None or total <= hi)
        if hit:
            mg_dist = 0.0
        elif total < lo:
            mg_dist = float(lo - total)
        else:
            mg_dist = float(total - hi)  # type: ignore[operator]
        rows.append(_market_row(
            "multi_goal", "Goal range", f"{rng} goals",
            mg_top["prob"], mg_top.get("fair_odds", 99), hit,
            f"{total} goals", mg_dist,
            "in range" if hit else f"{int(mg_dist)} goal(s) outside the range",
            None,
        ))

    # --- Win to nil -------------------------------------------------------
    wtn = em.compute_win_to_nil(hxg, axg, p1x2)
    # show only the favourite's win-to-nil as the representative pick
    wtn_key = "home_win_to_nil" if fav_home else "away_win_to_nil"
    wtn_v = wtn.get(wtn_key)
    if isinstance(wtn_v, dict):
        if fav_home:
            actual_wtn = hg > ag and ag == 0
            team = label_map["home"]
        else:
            actual_wtn = ag > hg and hg == 0
            team = label_map["away"]
        rows.append(_market_row(
            "win_to_nil", "Win to nil", f"{team} win to nil",
            wtn_v["prob"], wtn_v.get("fair_odds", 99), actual_wtn,
            "won to nil" if actual_wtn else "did not win to nil",
            None, "win + clean sheet — hit or miss", None,
        ))

    # --- Goal-timing tier (needs HT) --------------------------------------
    if ht is not None:
        rows.extend(_grade_timing_markets(snapshot, hxg, axg, hg, ag, ht))
    else:
        rows.extend(_timing_markets_ungraded(hxg, axg))

    return rows


def _parse_range(rng: str) -> tuple[int, int | None]:
    """'2-4' -> (2,4); '3+' -> (3,None)."""
    if rng.endswith("+"):
        return int(rng[:-1]), None
    lo, hi = rng.split("-")
    return int(lo), int(hi)


# --------------------------------------------------------------------------- #
# Goal-timing markets (half-time dependent)
# --------------------------------------------------------------------------- #
def _grade_timing_markets(
    snapshot: dict, hxg: float, axg: float,
    hg: int, ag: int, ht: tuple[int, int],
) -> list[dict[str, Any]]:
    """HT/FT, first-half result, both-halves — graded from a known HT score."""
    hh, ah = int(ht[0]), int(ht[1])
    sh_h, sh_a = hg - hh, ag - ah  # second-half goals
    rows: list[dict[str, Any]] = []

    # First-half result (1X2 of the half) — nested under result_1x2
    fh = em.compute_first_half(hxg, axg)
    fh_1x2 = fh.get("result_1x2") if isinstance(fh, dict) else None
    fh_top = _top_kv(fh_1x2) if isinstance(fh_1x2, dict) else None
    if fh_top:
        k, v = fh_top
        ht_outcome = "home" if hh > ah else "draw" if hh == ah else "away"
        fh_lbls = {"home": "Home leads HT", "draw": "Level at HT", "away": "Away leads HT"}
        ht_res_lbl = {"home": "home led", "draw": "level", "away": "away led"}[ht_outcome]
        rows.append(_market_row(
            "first_half", "Half-time result", fh_lbls.get(k, k),
            v["prob"], v.get("fair_odds", 99), k == ht_outcome,
            f"HT {hh}-{ah} ({ht_res_lbl})", None,
            "half-time result — hit or miss", None,
        ))

    # Goal in both halves (any team scores in each half)
    gbh = em.compute_goal_in_both_halves(hxg, axg)
    if isinstance(gbh, dict) and "goal_both_halves_yes" in gbh:
        yes_v = gbh["goal_both_halves_yes"]
        no_v = gbh.get("goal_both_halves_no", {"prob": 0, "fair_odds": 99})
        pick_yes = yes_v.get("prob", 0) >= no_v.get("prob", 0)
        v = yes_v if pick_yes else no_v
        actual_both = ((hh + ah) > 0) and ((sh_h + sh_a) > 0)
        rows.append(_market_row(
            "goal_both_halves", "Goal in both halves", "Yes" if pick_yes else "No",
            v.get("prob", 0), v.get("fair_odds", 99), pick_yes == actual_both,
            "goals in both halves" if actual_both else "not both halves",
            None, "yes/no — hit or miss", None,
        ))

    return rows


def _timing_markets_ungraded(hxg: float, axg: float) -> list[dict[str, Any]]:
    """Same timing markets, returned ungraded when HT can't be reconstructed."""
    rows: list[dict[str, Any]] = []
    fh = em.compute_first_half(hxg, axg)
    fh_1x2 = fh.get("result_1x2") if isinstance(fh, dict) else None
    fh_top = _top_kv(fh_1x2) if isinstance(fh_1x2, dict) else None
    if fh_top:
        k, v = fh_top
        fh_lbls = {"home": "Home leads HT", "draw": "Level at HT", "away": "Away leads HT"}
        rows.append(_market_row(
            "first_half", "Half-time result", fh_lbls.get(k, k),
            v["prob"], v.get("fair_odds", 99), None,
            "needs goal timing", None,
            "half-time score not in the scorer data yet — ungraded", None,
        ))
    return rows
