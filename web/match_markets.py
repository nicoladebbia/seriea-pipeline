"""One match, every market the system can price, each with its honesty tier.

Assembly only: nothing here computes a probability. Each row is read from an
artifact an existing engine wrote, and carries the engine that wrote it and
the skill tier that engine earned on held-out data:

  A  skill above base rate demonstrated (1X2 ensemble; player floor lines)
  B  priceable, expect about base rate (Poisson-derived goal/card markets,
     the O/U blend on lines other than the bet ones)
  C  base rate only (red card, penalty, anytime scorer)

Markets the design (.plans/every-market-design.md) assigns to engines that do
not exist yet are listed under `not_built`, never served as numbers. Anything
an artifact serves as a per-match CONSTANT (team corners: 5.0 / 4.8 on every
row of extended_markets.json) is listed under `excluded` with the reason.

Italian bet names are kept exactly as the bookmaker lists them.
"""
from __future__ import annotations

from typing import Any

TIER_A, TIER_B, TIER_C = "A", "B", "C"

# Player floor markets: validated leak-free vs base rate (player_predictions.validate,
# 2026-06-04 / 06-11). goalscorer is served by the same engine but measured no-skill.
_PLAYER_MARKET_IT = {
    "shots_o05": ("Tiri totali del giocatore", "Over 0.5"),
    "shots_o15": ("Tiri totali del giocatore", "Over 1.5"),
    "shots_o25": ("Tiri totali del giocatore", "Over 2.5"),
    "sot_o05": ("Tiri in porta", "Over 0.5"),
    "sot_o15": ("Tiri in porta", "Over 1.5"),
    "fouls_o05": ("Falli commessi giocatore", "Over 0.5"),
    "fouls_o15": ("Falli commessi giocatore", "Over 1.5"),
    "fouled_o05": ("Falli subiti giocatore", "Over 0.5"),
    "tackles_o05": ("Contrasti giocatore", "Over 0.5"),
    "tackles_o15": ("Contrasti giocatore", "Over 1.5"),
    "tackles_o25": ("Contrasti giocatore", "Over 2.5"),
    "passes_o195": ("Passaggi riusciti giocatore", "Over 19.5"),
    "passes_o295": ("Passaggi riusciti giocatore", "Over 29.5"),
    "passes_o395": ("Passaggi riusciti giocatore", "Over 39.5"),
    "duels_o25": ("Duelli vinti giocatore", "Over 2.5"),
    "duels_o45": ("Duelli vinti giocatore", "Over 4.5"),
    "intercepts_o05": ("Intercetti giocatore", "Over 0.5"),
    "intercepts_o15": ("Intercetti giocatore", "Over 1.5"),
    "goalscorer": ("Giocatore marcatore", "Sì"),
}

NOT_BUILT = [
    {"group": "Principali", "bet_type": "Estro finale", "engine": "goal-process simulator",
     "note": "[FILL: bookmaker definition]"},
    {"group": "Minuti", "bet_type": "Minuti xy (other intervals)", "engine": "goal-process simulator",
     "note": "only 0-15', 76-90', 2H stoppage and the 15' lead are served; other intervals are one label away"},
    {"group": "Corner", "bet_type": "Corner per intervallo", "engine": "goal-process simulator",
     "note": "[PLACEHOLDER: corner minutes are not in the catalog]"},
    {"group": "Giocatori", "bet_type": "Primo marcatore / Doppietta più / Assist", "engine": "player event engine",
     "note": "scorer share × goal paths; scorer markets measured no-skill, served as tier C when built"},
    {"group": "Speciali match", "bet_type": "Rigore VAR / Espulsione VAR", "engine": "rare-event base-rate table",
     "note": "match_incidents.parquet carries no VAR incident type (goal, card, substitution only)"},
    {"group": "O tutte", "bet_type": "O tutte", "engine": "-", "note": "[FILL: undefined in the brief]"},
]


def _pct(p: Any) -> float | None:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return round(v * 100.0, 1)


def _p(v: Any) -> float | None:
    """Probability of an artifact cell. Cells are {"prob": x, "fair_odds": y};
    siblings like booking_points["expected"] are bare floats and must not be
    read as cells (live 500 on 2026-09-05)."""
    return v.get("prob") if isinstance(v, dict) else None


def _row(group: str, bet_type: str, selection: str, prob: Any, tier: str,
         source: str, **extra) -> dict[str, Any] | None:
    pct = _pct(prob)
    if pct is None:
        return None
    r = {"group": group, "bet_type": bet_type, "selection": selection,
         "probability_pct": pct, "tier": tier, "source": source}
    r.update(extra)
    return r


def _reasoning(pred: dict, goal_pred: dict) -> list[str]:
    out: list[str] = []
    hx, ax = pred.get("home_xg"), pred.get("away_xg")
    if hx is not None and ax is not None:
        out.append(f"xG {hx} v {ax}")
    if goal_pred.get("expected_total_goals") is not None:
        out.append(f"expected total goals {goal_pred['expected_total_goals']}")
    for side, key in (("home", "home_factors"), ("away", "away_factors"), ("", "neutral_factors")):
        for f in pred.get(key) or []:
            out.append(f"{side + ': ' if side else ''}{f}")
    mk = pred.get("market_implied") or {}
    if mk.get("home") is not None:
        out.append(f"market implied H/D/A {_pct(mk.get('home'))}/{_pct(mk.get('draw'))}/{_pct(mk.get('away'))}% ({mk.get('source', '')})")
    if pred.get("confidence_level"):
        out.append(f"confidence {pred['confidence_level']}")
    if pred.get("methods_used"):
        out.append("methods " + ", ".join(pred["methods_used"]))
    return out


def build_match_markets(match_key: str, *, pred: dict | None, goal_pred: dict | None,
                        ext: dict | None, btts: dict | None, engine_bet: dict | None,
                        players: dict | None, kickoff_utc: str | None = None,
                        league: str | None = None, sim: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble the per-match market list. Every input is an already-loaded dict
    (or None when the artifact has no row for this match). `sim` is the
    goal-process simulator's row list (scripts/models/goal_process.served_rows):
    where it prices a bet the independent-Poisson artifact also prices, the
    simulator row REPLACES the artifact row (its tier comes from a walk-forward
    backtest; the artifact's tier B is a label, not a measurement)."""
    pred = pred or {}
    goal_pred = goal_pred or {}
    ext = ext or {}
    btts = btts or {}
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    missing: list[str] = []

    def add(r):
        if r:
            rows.append(r)

    # ---- Principali: 1x2 finale (ensemble, tier A) ----------------------------
    probs = pred.get("probabilities") or {}
    if probs:
        for sel, key in (("1", "home"), ("X", "draw"), ("2", "away")):
            add(_row("Principali", "1x2 finale", sel, probs.get(key), TIER_A, "ensemble"))
    else:
        missing.append("predictions.json row")

    # ---- Under/over 0.5 … 4.5 (O/U blend: the money model on 1.5 / 2.5) -------
    have_ou = False
    for line in ("0.5", "1.5", "2.5", "3.5", "4.5"):
        p_over = goal_pred.get(f"over_{line.replace('.', '_')}")
        if p_over is None:
            continue
        have_ou = True
        add(_row("Under/over", "Under/over", f"Over {line}", p_over, TIER_B, "ou_blend",
                 bet_line=line in ("1.5", "2.5")))
        add(_row("Under/over", "Under/over", f"Under {line}", 1 - float(p_over), TIER_B, "ou_blend",
                 bet_line=False))
    if not sim:
        for line in ("5.5", "6.5"):
            excluded.append({"bet_type": "Under/over", "selection": f"Over/Under {line}",
                             "reason": "goal-process simulator rows unavailable for this match"})
    if not have_ou:
        missing.append("goal_predictions.json row")

    # ---- Poisson-derived (extended_markets.json, tier B) ----------------------
    dc = ext.get("double_chance") or {}
    for sel in ("1X", "X2", "12"):
        add(_row("Doppia chance", "Doppia chance", sel, (dc.get(sel) or {}).get("prob"), TIER_B, "poisson"))
    for mg in ext.get("multi_goal") or []:
        add(_row("Multi goal", "Multi goal", mg.get("range", "?"), mg.get("prob"), TIER_B, "poisson"))
    wm = ext.get("winning_margin") or {}
    for sel, key in (("0", "draw"), ("1 casa", "home_by_1"), ("2 casa", "home_by_2"), ("3+ casa", "home_by_3_plus"),
                     ("1 ospite", "away_by_1"), ("2 ospite", "away_by_2"), ("3+ ospite", "away_by_3_plus")):
        add(_row("Somma gol", "Margine di vittoria", sel, (wm.get(key) or {}).get("prob"), TIER_B, "poisson"))
    for es in (ext.get("exact_score") or [])[:10]:
        add(_row("Risultati", "Risultato esatto", es.get("score", "?"), es.get("prob"), TIER_B, "poisson"))
    tt = ext.get("team_totals") or {}
    for side, label in (("home", "Casa"), ("away", "Ospite")):
        for sel, v in (tt.get(side) or {}).items():
            add(_row("Casa/ospite", f"Gol {label}", sel.replace("_", " ").capitalize(), _p(v), TIER_B, "poisson"))
    wtn = ext.get("win_to_nil") or {}
    for sel, key in (("Casa vince senza subire", "home_win_to_nil"), ("Ospite vince senza subire", "away_win_to_nil"),
                     ("Casa porta inviolata", "home_clean_sheet"), ("Ospite porta inviolata", "away_clean_sheet")):
        add(_row("Casa/ospite", "Vince a zero / Porta inviolata", sel, (wtn.get(key) or {}).get("prob"), TIER_B, "poisson"))
    if btts.get("btts_yes") is not None:
        add(_row("Goal", "Goal", "Sì", btts.get("btts_yes"), TIER_B, "btts_model"))
        add(_row("Goal", "Goal", "No", btts.get("btts_no", 1 - float(btts["btts_yes"])), TIER_B, "btts_model"))
    gbh = ext.get("goal_both_halves") or {}
    for sel, key in (("Sì", "goal_both_halves_yes"), ("No", "goal_both_halves_no")):
        add(_row("Goal", "Gol in entrambi i tempi", sel, (gbh.get(key) or {}).get("prob"), TIER_B, "poisson"))
    oe = ext.get("odd_even") or {}
    for sel, key in (("Pari", "even"), ("Dispari", "odd")):
        add(_row("Goal", "Pari/Dispari", sel, (oe.get(key) or {}).get("prob"), TIER_B, "poisson"))
    tsf = ext.get("team_to_score_first") or {}
    for sel, key in (("Casa", "home"), ("Ospite", "away"), ("Nessuno", "no_goal")):
        add(_row("Goal", "Prima squadra a segnare", sel, (tsf.get(key) or {}).get("prob"), TIER_B, "poisson"))
    fh = ext.get("first_half") or {}
    for sel, key in (("1", "home"), ("X", "draw"), ("2", "away")):
        add(_row("Tempi", "1° tempo 1x2", sel, ((fh.get("result_1x2") or {}).get(key) or {}).get("prob"), TIER_B, "poisson_1h"))
    for sel, v in (fh.get("over_under") or {}).items():
        add(_row("Tempi", "1° tempo under/over", sel.replace("_", " ").capitalize(), _p(v), TIER_B, "poisson_1h"))
    sh = ext.get("second_half_result") or {}
    for sel, key in (("1", "home"), ("X", "draw"), ("2", "away")):
        add(_row("Tempi", "2° tempo 1x2", sel, (sh.get(key) or {}).get("prob"), TIER_B, "poisson"))
    for c in ext.get("htft") or []:
        add(_row("Tempi", "Primo tempo / Finale", c.get("combo", "?"), c.get("prob"), TIER_B, "poisson"))

    # ---- Sanzioni (cards: skill ≤ 0 measured 2026-05-06 → tier B, base-rate grade) --
    tc = ext.get("team_cards") or {}
    for side, label in (("home", "Casa"), ("away", "Ospite")):
        for k, v in tc.items():
            if k.startswith(f"{side}_over_"):
                add(_row("Sanzioni", f"Cartellini {label}", "Over " + k.split("_over_")[1].replace("_", "."),
                         _p(v), TIER_B, "cards_poisson"))
    for k, v in (ext.get("booking_points") or {}).items():
        add(_row("Sanzioni", "Punti cartellini", k.replace("_", " ").capitalize(), _p(v), TIER_B, "cards_poisson"))
    cbh = ext.get("cards_by_half") or {}
    for half, label in (("first_half", "1° tempo"), ("second_half", "2° tempo")):
        for k, v in (cbh.get(half) or {}).items():
            if k.startswith("over_"):
                add(_row("Sanzioni", f"Cartellini {label}", "Over " + k[5:].replace("_", "."), _p(v), TIER_B, "cards_poisson"))

    # ---- Corner: constant in the artifact → excluded, never served -----------
    tcorn = ext.get("team_corners") or {}
    if tcorn:
        excluded.append({"bet_type": "Corner", "selection": "tutti",
                         "reason": f"extended_markets.json serves a per-match CONSTANT (home {tcorn.get('home_expected')}, away {tcorn.get('away_expected')} on every row); corners model skill ≤ 0 (2026-05-06)"})

    # ---- Speciali match (tier C: base rate) ------------------------------------
    rc = ext.get("red_card") or {}
    for sel, key in (("Sì", "yes"), ("No", "no")):
        add(_row("Speciali match", "Espulsione", sel, (rc.get(key) or {}).get("prob"), TIER_C, "base_rate"))
    pen = ext.get("penalty_in_match") or {}
    for sel, key in (("Sì", "penalty_yes"), ("No", "penalty_no")):
        add(_row("Speciali match", "Rigore", sel, (pen.get(key) or {}).get("prob"), TIER_C, "base_rate"))

    # ---- Giocatori (floor engine: tier A; goalscorer tier C) -------------------
    player_rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        for pl in ((players or {}).get(f"{side}_players") or []):
            for key, m in (pl.get("markets") or {}).items():
                it = _PLAYER_MARKET_IT.get(key)
                if not it or m.get("prob") is None:
                    continue
                tier = TIER_C if key == "goalscorer" else TIER_A
                player_rows.append({
                    "group": "Giocatori", "bet_type": it[0], "selection": it[1],
                    "player": pl.get("player_name"), "team": pl.get("team") or (pred.get(f"{side}_team") or side),
                    "position": pl.get("position"), "proj_minutes": pl.get("proj_minutes"),
                    "probability_pct": _pct(m["prob"]), "tier": tier, "source": "player_floors",
                    "interval": "90'",
                    "contribution_pct": None,  # [PLACEHOLDER: per-interval split, design step 3]
                })
    if players is None:
        missing.append("player floors (no lineup or engine)" if (league or pred.get("league", "serie_a")) == "serie_a"
                       else "player floors (engine is Serie A only)")

    # ---- Goal-process simulator: replaces its Poisson twins, adds the rest ----
    if sim:
        twins = {(r["bet_type"], r["selection"]) for r in sim}
        rows = [r for r in rows if (r["bet_type"], r["selection"]) not in twins]
        rows.extend(sim)
    else:
        missing.append("goal-process simulator rows (Vince o quasi, Minuti, 2° tempo under/over)"
                       + ("" if (league or pred.get("league", "serie_a")) == "serie_a" else " (profile and gate are Serie A only)"))

    n_tier = {t: sum(1 for r in rows if r["tier"] == t) for t in (TIER_A, TIER_B, TIER_C)}
    n_tier[TIER_A] += sum(1 for r in player_rows if r["tier"] == TIER_A)
    n_tier[TIER_C] += sum(1 for r in player_rows if r["tier"] == TIER_C)
    return {
        "match": match_key,
        "league": league or pred.get("league"),
        "kickoff_utc": kickoff_utc,
        "engine_bet": engine_bet or {"status": "none"},
        "reasoning": _reasoning(pred, goal_pred),
        "tiers": {"A": "skill above base rate on held-out data",
                  "B": "priceable; expect about base rate",
                  "C": "base rate only"},
        "counts": n_tier,
        "markets": rows,
        "players": player_rows,
        "excluded": excluded,
        "not_built": NOT_BUILT,
        "missing": missing,
    }
