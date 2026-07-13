"""World Cup best-combo accumulators: build, archive, grade.

Build — the three combo tiers served by /api/worldcup (web.app imports
build_best_combos): safe (top double-chance legs), favorites (top straight
1X2 legs), value (model beats the Sofascore-implied price). Legs come ONLY
from the backtest-gated who-wins family; goal-quantity props are noise on
internationals and never enter a combo.

Archive — pre-kickoff ticket snapshots in combos_archive.json, keyed by
tier + first-leg kickoff. Protocol mirrors predictions_archive: LAST write
BEFORE the first leg kicks off wins (the graded ticket is the closing
deployment), immutable after — the writer refuses post-kickoff writes AND
grading independently drops tickets stamped at/after their first kickoff.
If a tier's leg set shifts to a different first kickoff between refreshes,
that is a new ticket key; the old one freezes at its last shown state.

Grade — tickets where every leg's match has an honest 90' result (group
matches as-is; knockouts via grading.py reconstruction, skipped when scorer
coverage is incomplete). Reports per-tier hit rate vs the expected rate
(mean combined probability — the calibration check) and ROI at the archived
quoted odds for priced tickets.

Run: python3 -m scripts.worldcup.combos   (wired into scripts.worldcup.refresh)
Writes data/worldcup/combos_archive.json + combo_record.json.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from scripts.worldcup.engine import (
    DATA_DIR,
    atomic_write_json,
    read_json_safe,
)

PREDICTIONS_JSON = DATA_DIR / "predictions.json"
MARKET_ODDS_JSON = DATA_DIR / "market_odds.json"
COMBO_ARCHIVE_JSON = DATA_DIR / "combos_archive.json"
COMBO_RECORD_JSON = DATA_DIR / "combo_record.json"

# Value-leg gates: ≥2pp model-vs-implied edge (calibration noise floor),
# positive EV at the quoted price, and no lottery legs in a recommended bet.
VALUE_EDGE_MIN = 0.02
VALUE_PROB_MIN = 0.20

_PICK_OUTCOME = {"1": "home", "X": "draw", "2": "away"}
_DC_COVER = {"1X": ("home", "draw"), "X2": ("draw", "away"), "12": ("home", "away")}


def _poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam). Closed form, no scipy."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    import math
    return math.exp(-lam) * lam**k / math.factorial(k)


def _over_prob(lam_total: float, line: float) -> float:
    """P(goals > line) for a single Poisson stream of rate ``lam_total``.
    Lines are .5 so there are no pushes: sum the PMF up to floor(line)."""
    floor = int(line)  # 1.5 -> 1, 0.5 -> 0, 2.5 -> 2
    p_under = sum(_poisson_pmf(k, lam_total) for k in range(floor + 1))
    return max(0.0, min(1.0, 1.0 - p_under))


# Goal-quantity markets for the FUN combos, each derived from the engine's
# own Poisson lambdas (home_xg / away_xg in predictions.json). These are the
# DNA of Nicola's played slip (Over 1.5, team-total Over) — NOT backtest-gated,
# goal props are noise on internationals, so every leg carries edge="none".
_FUN_MARKETS = (
    ("O15",  "Over 1.5",         lambda lh, la: _over_prob(lh + la, 1.5)),
    ("O25",  "Over 2.5",         lambda lh, la: _over_prob(lh + la, 2.5)),
    ("H_O05", "Home team Over 0.5", lambda lh, la: _over_prob(lh, 0.5)),
    ("A_O05", "Away team Over 0.5", lambda lh, la: _over_prob(la, 0.5)),
    ("H_O15", "Home team Over 1.5", lambda lh, la: _over_prob(lh, 1.5)),
)

FUN_PROB_MIN = 0.50          # never stack a goal leg the model rates a coin-flip or worse
FUN_BIG_TARGET = (10.0, 20.0)  # combined fair-odds band for the "big odds" ticket


def build_fun_combos(preds: list, now: datetime | None = None,
                     window_hours: float = 48.0, max_legs: int = 8) -> dict:
    """Goal-quantity accumulators in the style Nicola actually plays (Over 1.5 /
    team-total Over), built from the engine's Poisson lambdas.

    Returns two tickets: 'fun_safe' stacks the highest-probability Over legs;
    'fun_big' stacks legs to land in the FUN_BIG_TARGET combined-odds band.
    Goal props do NOT pass the WC backtest (skill ≤ 0 on internationals), so
    every leg and ticket is stamped edge='none' — entertainment, not +EV. The
    who-wins build_best_combos remains the only edge-bearing path.
    """
    now = now or datetime.now(UTC)
    horizon = now + timedelta(hours=window_hours)

    # Two independent pools so the two tickets draw GENUINELY different legs:
    #  safe_cands  — the highest-probability Over leg per match (Over 0.5-ish)
    #  big_cands   — the Over leg per match priced nearest FUN_BIG_LEG_PRICE
    #                (Over 1.5 / Over 2.5 — the legs that build a real payout,
    #                exactly the DNA of Nicola's played 14.68 slip).
    # One leg per match in EACH pool keeps every ticket's product-of-probs valid.
    FUN_BIG_LEG_PRICE = 1.55  # ideal per-leg fair odds for the big ticket
    safe_cands: list[dict] = []
    big_cands: list[dict] = []
    n_slate = 0
    for p in preds:
        try:
            ko = datetime.fromisoformat(p.get("kickoff_utc") or "")
        except (TypeError, ValueError):
            continue
        if ko.tzinfo is None or not (now < ko <= horizon):
            continue
        lh = p.get("home_xg")
        la = p.get("away_xg")
        if lh is None or la is None:
            continue
        lh, la = float(lh), float(la)
        n_slate += 1
        home, away = p.get("home_team", "?"), p.get("away_team", "?")
        base = {
            "match": p.get("match") or f"{home} vs {away}",
            "home_team": home, "away_team": away,
            "stage": p.get("stage", "group"),
            "date": p.get("date", ""), "time": p.get("time", ""),
            "kickoff_utc": p.get("kickoff_utc"),
            "match_number": p.get("match_number"),
        }
        cands = []
        for code, label, fn in _FUN_MARKETS:
            prob = fn(lh, la)
            if prob < FUN_PROB_MIN:
                continue
            disp = label if code in ("O15", "O25") else (
                label.replace("Home team", home).replace("Away team", away))
            cands.append({
                **base, "market": "Goals", "pick": code, "pick_label": disp,
                "prob": round(prob, 4),
                "fair_odds": round(1 / prob, 2) if prob > 0.001 else 99,
                "market_odds": None, "edge": "none",
            })
        if cands:
            safe_cands.append(max(cands, key=lambda x: x["prob"]))
            # Highest-paying leg that still clears the floor — nearest the
            # target price, tie-broken toward longer odds for a bigger payout.
            big_cands.append(min(
                cands, key=lambda x: (abs(x["fair_odds"] - FUN_BIG_LEG_PRICE),
                                      -x["fair_odds"])))

    def assemble(key: str, title: str, note: str, chosen: list[dict]) -> dict | None:
        if len(chosen) < 2:
            return None
        chosen = sorted(chosen, key=lambda x: (x["date"], x["time"]))
        prob = 1.0
        for leg in chosen:
            prob *= leg["prob"]
        return {
            "key": key, "title": title, "note": note, "edge": "none",
            "legs": chosen,
            "combined": {"prob": round(prob, 4),
                         "fair_odds": round(1 / prob, 2) if prob > 0.001 else 99},
        }

    # Safest: the most-probable Over legs, up to max_legs.
    safe_legs = sorted(safe_cands, key=lambda x: x["prob"], reverse=True)[:max_legs]

    # Big odds: stack the higher-paying legs (already chosen per match nearest
    # ~1.55) longest-first, building the product until combined fair odds enter
    # FUN_BIG_TARGET, then stop. fair_odds = 1/prob grows as legs are added
    # (prob shrinks), so the band test runs on the accumulated product.
    big_legs: list[dict] = []
    running = 1.0  # running combined probability
    for leg in sorted(big_cands, key=lambda x: x["fair_odds"], reverse=True):
        if 1 / running >= FUN_BIG_TARGET[0]:
            break  # already in the band — don't dilute the payout further
        nxt = running * leg["prob"]
        if 1 / nxt > FUN_BIG_TARGET[1] and big_legs:
            break  # this leg would overshoot the top of the band
        big_legs.append(leg)
        running = nxt
        if len(big_legs) >= max_legs:
            break

    combos = [
        assemble("fun_safe", "🎲 Fun — safest Overs",
                 "Highest-probability goal legs (Over 1.5 / team Over), model "
                 "Poisson. NO model edge — goal props are noise on "
                 "internationals; entertainment only.", safe_legs),
        assemble("fun_big", "🎲 Fun — big odds",
                 f"Stacked toward {FUN_BIG_TARGET[0]:.0f}-{FUN_BIG_TARGET[1]:.0f}x "
                 "like a played multipla. NO model edge — entertainment only.",
                 big_legs),
    ]
    return {"window_hours": window_hours, "n_matches": n_slate,
            "edge": "none",
            "combos": [c for c in combos if c]}


def build_best_combos(preds: list, market: dict, now: datetime | None = None,
                      window_hours: float = 48.0, max_legs: int = 3) -> dict:
    """The three accumulator tiers from the next `window_hours` of matches.

    One leg per match, matches independent, so the combined probability is
    the plain product. The value pool scans ALL six who-wins framings per
    match (3×1X2 + 3×DC) — the model's biggest market disagreements are
    often NOT on the favorite — gated by VALUE_EDGE_MIN/VALUE_PROB_MIN and
    ranked by per-leg EV at the quoted price (maximizing the product of
    (1+EV) across legs). Deployed probs are already shrunk 0.4 toward this
    same market, so a surviving gap is genuine model disagreement.
    """
    now = now or datetime.now(UTC)
    horizon = now + timedelta(hours=window_hours)

    def make_leg(base: dict, market_name: str, pick: str, pick_label: str,
                 prob: float, mkt_odds: float | None, mkt_implied: float | None) -> dict:
        return {
            **base,
            "market": market_name, "pick": pick, "pick_label": pick_label,
            "prob": round(prob, 4),
            "fair_odds": round(1 / prob, 2) if prob > 0.001 else 99,
            "market_odds": round(mkt_odds, 2) if mkt_odds else None,
            "edge": round(prob - mkt_implied, 4) if mkt_implied else None,
            "ev": round(prob * mkt_odds - 1, 4) if mkt_odds else None,
        }

    dc_legs: list[dict] = []
    fav_legs: list[dict] = []
    value_pool: list[dict] = []
    n_slate = 0
    for p in preds:
        try:
            ko = datetime.fromisoformat(p.get("kickoff_utc") or "")
        except (TypeError, ValueError):
            continue
        if ko.tzinfo is None or not (now < ko <= horizon):
            continue
        probs = p.get("probabilities") or {}
        h = float(probs.get("home", 0.0))
        d = float(probs.get("draw", 0.0))
        a = float(probs.get("away", 0.0))
        if h + d + a < 0.9:  # malformed entry — skip, never crash the page
            continue
        n_slate += 1
        home, away = p.get("home_team", "?"), p.get("away_team", "?")
        mk = market.get(str(p.get("match_number"))) if isinstance(market, dict) else None
        odds = (mk or {}).get("odds") or {}
        implied = (mk or {}).get("implied") or {}
        base = {
            "match": p.get("match") or f"{home} vs {away}",
            "home_team": home, "away_team": away,
            "stage": p.get("stage", "group"),
            "date": p.get("date", ""), "time": p.get("time", ""),
            "kickoff_utc": p.get("kickoff_utc"),
            "match_number": p.get("match_number"),
        }

        # All six who-wins framings for this match.
        out_prob = {"home": h, "draw": d, "away": a}
        one_x_two = [
            make_leg(base, "1X2", {"home": "1", "draw": "X", "away": "2"}[k],
                     {"home": home, "draw": "Draw", "away": away}[k],
                     out_prob[k], odds.get(k), implied.get(k))
            for k in ("home", "draw", "away")
        ]
        # DC market odds derived from the quoted 1X2 prices
        # (1/(1/oA+1/oB), margin carried over — slightly conservative EV).
        double_chance = []
        for code, keys, label in (("1X", ("home", "draw"), f"{home} or draw"),
                                  ("X2", ("draw", "away"), f"Draw or {away}"),
                                  ("12", ("home", "away"), f"{home} or {away}")):
            prob = out_prob[keys[0]] + out_prob[keys[1]]
            o1, o2 = odds.get(keys[0]), odds.get(keys[1])
            dc_odds = 1 / (1 / o1 + 1 / o2) if o1 and o2 else None
            dc_imp = (implied.get(keys[0], 0) + implied.get(keys[1], 0)) or None
            double_chance.append(
                make_leg(base, "Double Chance", code, label, prob, dc_odds, dc_imp))

        fav_legs.append(max(one_x_two, key=lambda x: x["prob"]))
        dc_legs.append(max(double_chance, key=lambda x: x["prob"]))
        value_cands = [
            x for x in one_x_two + double_chance
            if x["edge"] is not None and x["edge"] >= VALUE_EDGE_MIN
            and x["ev"] is not None and x["ev"] > 0
            and x["prob"] >= VALUE_PROB_MIN
        ]
        if value_cands:
            value_pool.append(max(value_cands, key=lambda x: x["ev"]))

    def combo(key: str, title: str, note: str, legs: list[dict]) -> dict | None:
        if len(legs) < 2:
            return None
        legs = sorted(legs, key=lambda x: (x["date"], x["time"]))
        prob = 1.0
        for leg in legs:
            prob *= leg["prob"]
        out = {"key": key, "title": title, "note": note, "legs": legs,
               "combined": {"prob": round(prob, 4),
                            "fair_odds": round(1 / prob, 2) if prob > 0.001 else 99}}
        if all(leg["market_odds"] for leg in legs):
            mo = 1.0
            for leg in legs:
                mo *= leg["market_odds"]
            out["combined"]["market_odds"] = round(mo, 2)
            out["combined"]["ev"] = round(prob * mo - 1, 4)
        return out

    def top(legs: list[dict], sort_key: Callable[[dict], Any]) -> list[dict]:
        return sorted(legs, key=sort_key, reverse=True)[:max_legs]

    combos = [
        combo("safe", "Safe — double chance",
              "Highest combined probability; modest payout.",
              top(dc_legs, lambda x: x["prob"])),
        combo("favorites", "Favorites — straight 1X2",
              "Top model favorites, straight up.",
              top(fav_legs, lambda x: x["prob"])),
        combo("value", "Value — model vs market",
              "Best framing per match (any 1X2 or DC pick, not just the "
              "favorite) where the blended model beats the Sofascore-implied "
              "price by ≥2pp; ranked by per-leg EV at quoted odds (proxy, "
              "not a bookmaker).",
              top(value_pool, lambda x: x["ev"])),
    ]
    return {"window_hours": window_hours, "n_matches": n_slate,
            "combos": [c for c in combos if c]}


def merge_combo_archive(best: dict, now_iso: str) -> int:
    """Pre-kickoff ticket snapshots; last write before first leg wins."""
    archive: dict[str, Any] = dict(
        read_json_safe(COMBO_ARCHIVE_JSON, {}, quarantine=True)  # type: ignore[arg-type]
    )
    now = datetime.fromisoformat(now_iso)
    written = 0
    for c in best.get("combos", []):
        legs = c.get("legs") or []
        if len(legs) < 2 or any(not leg.get("kickoff_utc") for leg in legs):
            continue
        first_ko = min((str(leg["kickoff_utc"]) for leg in legs),
                       key=datetime.fromisoformat)
        if now >= datetime.fromisoformat(first_ko):
            continue  # ticket already in play — snapshot frozen forever
        key = f"{c['key']}|{first_ko}"
        existing = archive.get(key, {})
        archive[key] = {
            "tier": c["key"], "title": c["title"],
            "legs": legs, "combined": c["combined"],
            "first_kickoff_utc": first_ko,
            "first_archived_at": existing.get("first_archived_at", now_iso),
            "archived_at": now_iso,
        }
        written += 1
    atomic_write_json(COMBO_ARCHIVE_JSON, archive)
    return written


def _leg_hit(market_name: str, pick: str, outcome: str) -> bool:
    if market_name == "1X2":
        return _PICK_OUTCOME.get(pick) == outcome
    return outcome in _DC_COVER.get(pick, ())


def _result_resolver() -> Callable[[str, str, str, str], str | None]:
    """Who-wins outcome lookup, shared with the track record.

    Combo legs are 1X2 / double-chance (who-wins only), so knockouts resolve on
    who ADVANCED (:func:`grading.resolve_knockout`) — same semantics as the
    record — not the 90' score. A penalty tie with no known winner returns None
    (leg stays ungraded rather than settled a draw).
    """
    from scripts.worldcup.engine import load_results_with_live
    from scripts.worldcup.grading import (
        _find_result,
        _load_advance_winners,
        resolve_knockout,
    )

    df = load_results_with_live()  # CSV + live Sofascore overlay (same-night settling)
    advance_winners = _load_advance_winners()

    def resolve(home: str, away: str, date: str, stage: str) -> str | None:
        res = _find_result(df, home, away, date)
        if res is None:
            return None
        hs, as_, _result_date = res
        if stage != "group":
            resolved = resolve_knockout(home, away, hs, as_, advance_winners)
            return resolved[0] if resolved else None
        return "home" if hs > as_ else "draw" if hs == as_ else "away"

    return resolve


def build_combo_record(
    resolve: Callable[[str, str, str, str], str | None] | None = None,
) -> dict:
    """Grade fully-settled archived tickets; write combo_record.json."""
    archive: dict[str, Any] = dict(read_json_safe(COMBO_ARCHIVE_JSON, {}))  # type: ignore[arg-type]
    tickets: list[dict] = []
    if archive and resolve is None:
        resolve = _result_resolver()
    for key, t in sorted(archive.items()):
        first_ko = t.get("first_kickoff_utc")
        archived = t.get("archived_at")
        # Honesty guard: never grade a ticket stamped at/after its first kickoff.
        if not first_ko or not archived or datetime.fromisoformat(
            str(archived)
        ) >= datetime.fromisoformat(str(first_ko)):
            continue
        legs_out: list[dict] = []
        for leg in t.get("legs", []):
            outcome = resolve(  # type: ignore[misc]
                str(leg["home_team"]), str(leg["away_team"]),
                str(leg["date"]), str(leg.get("stage", "group")),
            )
            if outcome is None:
                break  # a leg still unplayed/unresolvable — ticket pending
            legs_out.append({**leg, "outcome": outcome,
                             "hit": _leg_hit(leg["market"], leg["pick"], outcome)})
        if len(legs_out) != len(t.get("legs", [])):
            continue
        cm = t.get("combined") or {}
        hit = all(leg["hit"] for leg in legs_out)
        entry: dict[str, Any] = {
            "ticket": key, "tier": t.get("tier"), "title": t.get("title"),
            "first_kickoff_utc": first_ko, "n_legs": len(legs_out),
            "prob": cm.get("prob"), "market_odds": cm.get("market_odds"),
            "hit": hit, "legs_hit": sum(leg["hit"] for leg in legs_out),
            "legs": legs_out,
        }
        if cm.get("market_odds"):
            entry["profit"] = round(
                float(cm["market_odds"]) - 1.0 if hit else -1.0, 4)
        tickets.append(entry)

    tiers: dict[str, dict] = {}
    for tier in ("safe", "favorites", "value"):
        rows = [t for t in tickets if t["tier"] == tier]
        if not rows:
            continue
        priced = [t for t in rows if "profit" in t]
        n = len(rows)
        probs = [float(t["prob"]) for t in rows if t.get("prob") is not None]
        tiers[tier] = {
            "n": n,
            "hits": sum(1 for t in rows if t["hit"]),
            "hit_rate": round(sum(1 for t in rows if t["hit"]) / n, 4),
            # calibration check: mean combined prob = the rate we PROMISED
            "expected_hit_rate": round(sum(probs) / len(probs), 4) if probs else None,
            "n_priced": len(priced),
            "profit": round(sum(t["profit"] for t in priced), 4) if priced else None,
            "roi": round(sum(t["profit"] for t in priced) / len(priced), 4)
            if priced else None,
        }
    record = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_graded": len(tickets),
        "tiers": tiers,
        "tickets": tickets,
    }
    atomic_write_json(COMBO_RECORD_JSON, record)
    return record


def main() -> None:
    now_iso = datetime.now(UTC).isoformat()
    doc = read_json_safe(PREDICTIONS_JSON, {})
    preds = doc.get("predictions", []) if isinstance(doc, dict) else []
    market: dict[str, Any] = dict(read_json_safe(MARKET_ODDS_JSON, {}))  # type: ignore[arg-type]
    best = build_best_combos(preds, market)
    written = merge_combo_archive(best, now_iso)
    record = build_combo_record()
    print(f"combos: archived {written} tickets, graded {record['n_graded']}")


if __name__ == "__main__":
    main()
