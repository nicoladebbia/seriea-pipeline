"""Best-bets engine — per game, the strongest plays from TRUSTED markets only.

Takes a full projection (every market predicted) and surfaces:
  - best single: highest confidence-above-base-rate pick from a backtest-trusted market
  - best combo: independent trusted picks combined (joint probability)
  - all candidates ranked, with noise-market picks shown but flagged not-recommended

"Best" = confidence × trust, NOT raw %. A 92% "Over 0.5" (base 95%) has NEGATIVE
lift and is never recommended; a 60% 1X2-home (base 40%) has +20 lift and ranks high.
Only markets the 2026-06-03 harsh walk-forward backtest validated as "trust" (or
"ensemble_only" for O/U) are recommended. Trust verdicts + base rates live in
config/comparative_markets.json (market_trust), editable when re-backtested.
"""
from __future__ import annotations

from config import comparative_markets as _cfg


def _candidates_from_projection(proj: dict, comparative: list | None,
                                total_cards: dict | None, total_fouls: dict | None) -> list:
    """Flatten a projection into [{market, pick, prob, market_key}] candidates."""
    out = []
    probs = proj.get("probabilities", {})

    # 1X2
    for side, key in (("home", "1x2_home"), ("draw", "1x2_draw"), ("away", "1x2_away")):
        p = probs.get(side)
        if p is not None:
            label = {"home": proj.get("home_team"), "draw": "Draw", "away": proj.get("away_team")}[side]
            out.append({"market": "1X2", "pick": label, "prob": p, "key": key})

    # O/U 2.5 (ensemble_only trust)
    mg = {r["range"]: r["prob"] for r in proj.get("multi_goal", [])}
    if "3+" in mg:
        out.append({"market": "Over/Under 2.5", "pick": "Over 2.5", "prob": mg["3+"], "key": "ou_2.5"})

    # BTTS
    btts = proj.get("btts", {})
    if btts.get("yes", {}).get("prob") is not None:
        y = btts["yes"]["prob"]
        out.append({"market": "BTTS", "pick": "Goal" if y >= 0.5 else "No Goal",
                    "prob": max(y, 1 - y), "key": "btts"})

    # referee total cards / fouls — pick the line where confidence is highest.
    # base_side tells the ranker whether the pick is the Over or Under side, so lift
    # is computed against the CORRECT base rate (config base_rate is the OVER rate).
    for tot, key_pref, lbl in ((total_cards, "ref_cards", "Cartellini"),
                               (total_fouls, "ref_fouls", "Falli")):
        if not tot:
            continue
        for ln, v in tot.get("lines", {}).items():
            over, under = v.get("over", 0), v.get("under", 0)
            is_over = over >= under
            out.append({"market": f"{lbl} totali", "pick": f"{'Over' if is_over else 'Under'} {ln}",
                        "prob": max(over, under), "key": f"{key_pref}_o{ln}",
                        "base_side": "over" if is_over else "under"})

    # comparative who-makes-more
    for c in (comparative or []):
        stat = c.get("stat", "")
        keymap = {"corners": "corners_more", "fouls": "compare_fouls", "yellow_cards": "compare_cards"}
        key = keymap.get(stat)
        if not key:
            continue
        opts = [(c.get("home_more", 0), f"{proj.get('home_team')} +"),
                (c.get("tie", 0), "Pari"),
                (c.get("away_more", 0), f"{proj.get('away_team')} +")]
        best = max(opts, key=lambda x: x[0])
        out.append({"market": c.get("label", stat), "pick": best[1], "prob": best[0], "key": key})

    return out


def rank_bets(proj: dict, comparative=None, total_cards=None, total_fouls=None) -> dict:
    """Return best single, best combo, and ranked candidates for one match."""
    trust_cfg = _cfg.market_trust()
    bb = _cfg.best_bets_cfg()
    min_lift = bb.get("min_lift", 0.06)
    min_conf = bb.get("min_confidence", 0.55)
    max_legs = bb.get("combo_max_legs", 3)
    min_joint = bb.get("combo_min_joint", 0.30)

    cands = _candidates_from_projection(proj, comparative, total_cards, total_fouls)
    ranked = []
    for c in cands:
        tc = trust_cfg.get(c["key"], {})
        trust = tc.get("trust", "noise")
        base = tc.get("base_rate", 0.5)
        # the config base_rate is for the "primary" side (home / over). If the pick
        # is the OTHER side (under), the correct base rate is 1 - base.
        if c.get("base_side") == "under":
            base = 1 - base
        lift = c["prob"] - base
        recommended = (trust in ("trust", "ensemble_only")
                       and c["prob"] >= min_conf and lift >= min_lift)
        ranked.append({**c, "trust": trust, "base_rate": round(base, 3),
                       "lift": round(lift, 4), "recommended": bool(recommended)})
    # sort: recommended first, then by lift
    ranked.sort(key=lambda x: (x["recommended"], x["lift"]), reverse=True)

    rec = [r for r in ranked if r["recommended"]]
    best_single = rec[0] if rec else None

    # best combo: independent recommended legs (distinct market families), joint prob
    combo = []
    seen_families = set()
    for r in rec:
        fam = r["key"].split("_")[0]  # 1x2 / ref / corners / ou
        if fam in seen_families:
            continue
        seen_families.add(fam)
        combo.append(r)
        if len(combo) >= max_legs:
            break
    joint = 1.0
    for r in combo:
        joint *= r["prob"]
    best_combo = None
    if len(combo) >= 2 and joint >= min_joint:
        best_combo = {"legs": combo, "joint_prob": round(joint, 4),
                      "fair_odds": round(1 / joint, 2) if joint > 0.01 else 99}

    return {
        "best_single": best_single,
        "best_combo": best_combo,
        "all_ranked": ranked,
        "n_recommended": len(rec),
    }
