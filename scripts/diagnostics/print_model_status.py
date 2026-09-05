#!/usr/bin/env python3
"""Print live model performance from production metadata.

Two models, in order of importance:

1. O/U classifiers — data/models/universal/over_under/ou_*_catboost_metadata.json.
   These price the ONLY enabled betting markets (O/U Over, Alt O/U). PRIMARY.
2. 1X2 CatBoost — data/models/universal/catboost_no_odds_metadata.json.
   Feeds the dashboard, Telegram and the fantacalcio card. Bets nothing. SECONDARY.

Never edit numbers in markdown — run this script.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:      # run as a script: make `scripts.*` importable
    sys.path.insert(0, str(PROJECT_ROOT))
METADATA = PROJECT_ROOT / "data" / "models" / "universal" / "catboost_no_odds_metadata.json"
OU_DIR = PROJECT_ROOT / "data" / "models" / "universal" / "over_under"
JOURNAL = PROJECT_ROOT / "data" / "betting" / "bet_journal.json"

REALISTIC_CEILING = 0.55   # Pinnacle closing line / academic SOTA (1X2)
LEGACY_CAL_GATE = 0.03     # the old fixed O/U calibration bar — informational now
BET_LINES = (1.5, 2.5)     # lines the weekly O/U retrain maintains
RECENT_DAYS = 30


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def naive_baselines(base_rate: float) -> tuple[float, float]:
    """(log-loss, Brier) of always predicting the base rate."""
    b = min(max(float(base_rate), 1e-6), 1 - 1e-6)
    ll = -(b * math.log(b) + (1 - b) * math.log(1 - b))
    return round(ll, 4), round(b * (1 - b), 4)


def ou_line_status(meta: dict) -> dict:
    """Normalise one O/U metadata file into the numbers the report prints."""
    cv = meta.get("cv_metrics") or {}
    ev = meta.get("eval_metrics") or {}
    base = float(meta.get("base_rate") or cv.get("base_rate") or 0.5)
    naive_ll, naive_brier = naive_baselines(base)
    promo = meta.get("promotion")
    hold = (promo or {}).get("holdout") or {}
    line = float(meta.get("line", 0))
    return {
        "line": line,
        "bet": line in BET_LINES,
        "trained_at": str(meta.get("trained_at") or "unknown")[:10],
        "n_features": meta.get("n_features"),
        "n_rows": meta.get("n_training_rows"),
        "base_rate": round(base, 4),
        "cv": {
            "log_loss": cv.get("overall_log_loss"),
            "naive_log_loss": naive_ll,
            "brier": cv.get("overall_brier"),
            "naive_brier": naive_brier,
            "calibration_gap": cv.get("overall_calibration_gap"),
            "accuracy": cv.get("overall_accuracy"),
        },
        "holdout": {
            "n": hold.get("n"),
            "dates": hold.get("dates"),
            "log_loss": ev.get("log_loss"),
            "naive_log_loss": hold.get("naive_log_loss"),
            "brier": ev.get("brier"),
            "calibration_gap": ev.get("calibration_gap"),
            "incumbent": hold.get("incumbent"),
        },
        "promotion": (
            {"promoted": bool(promo.get("promoted")), "reason": promo.get("reason"),
             "decided_at": str(promo.get("decided_at") or "")[:10]}
            if promo else None
        ),
        "legacy_gates": meta.get("quality_gates") or {},
    }


def journal_edge(journal: dict, prefix: str = "O/U", now: datetime | None = None) -> dict:
    """Realised edge per market from the immutable bet journal."""
    now = now or datetime.now(UTC)
    bets = journal.get("bets") or {}
    rows = bets.values() if isinstance(bets, dict) else bets
    out: dict[str, dict] = {}
    recent = 0
    for b in rows:
        market = str(b.get("market") or "")
        if not market.startswith(prefix):
            continue
        placed = str(b.get("placed_at") or "")[:10]
        try:
            if placed and now - datetime.strptime(placed, "%Y-%m-%d").replace(tzinfo=UTC) \
                    <= timedelta(days=RECENT_DAYS):
                recent += 1
        except ValueError:
            pass
        if b.get("status") not in ("won", "lost", "push"):
            continue
        r = out.setdefault(market, {"n": 0, "stake": 0.0, "profit": 0.0,
                                    "clv": [], "last_placed": ""})
        r["n"] += 1
        r["stake"] += float(b.get("stake") or 0)
        r["profit"] += float(b.get("profit") or 0)
        if b.get("clv_pct") is not None:
            r["clv"].append(float(b["clv_pct"]))
        r["last_placed"] = max(r["last_placed"], placed)
    for r in out.values():
        r["roi"] = round(r["profit"] / r["stake"] * 100, 1) if r["stake"] else 0.0
        r["mean_clv"] = round(sum(r["clv"]) / len(r["clv"]), 2) if r["clv"] else None
        r["clv_pos_share"] = (round(sum(c > 0 for c in r["clv"]) / len(r["clv"]), 2)
                              if r["clv"] else None)
        del r["clv"]
    return {"markets": out, "recent_placed": recent}


def enabled_markets() -> list[str] | None:
    """Read the live market rules; None if the betting module can't be imported."""
    try:
        from scripts.betting.betting_unified import BettingConfig
        rules = BettingConfig().market_rules
        return sorted(k for k, v in rules.items() if v.get("enabled"))
    except Exception:  # noqa: BLE001 — diagnostics must not crash on a betting import
        return None


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _f(x, fmt=".4f", missing="   n/a"):
    return format(x, fmt) if isinstance(x, int | float) else missing


def render_ou(lines: list[dict], edge: dict | None, markets: list[str] | None) -> list[str]:
    o: list[str] = []
    o.append("  Enabled betting markets: "
             + (", ".join(markets) if markets else "(BettingConfig unavailable)"))
    o.append("")
    o.append("  O/U CLASSIFIERS — PRIMARY (these price the bets)")
    o.append("  " + "-" * 60)
    if not lines:
        o.append("    WARNING: no O/U metadata found — the money model is unreadable")
    for s in sorted(lines, key=lambda x: x["line"]):
        tag = "[BET]" if s["bet"] else "[not bet]"
        o.append(f"  Line {s['line']:<4} trained {s['trained_at']}   "
                 f"{s['n_features']} features   {s['n_rows']} rows   {tag}")
        h = s["holdout"]
        if h["log_loss"] is not None:
            span = "→".join(h["dates"]) if h.get("dates") else "newest 15%"
            n = f"n={h['n']}, " if h.get("n") else ""
            o.append(f"    Holdout ({n}{span}):")
            delta = (f"   ({h['log_loss'] - h['naive_log_loss']:+.4f})"
                     if isinstance(h["naive_log_loss"], int | float) else "")
            o.append(f"      Log-loss   : {_f(h['log_loss'])}   vs naive "
                     f"{_f(h['naive_log_loss'])}{delta}")
            o.append(f"      Brier      : {_f(h['brier'])}")
            if h["calibration_gap"] is not None:
                verdict = "PASS" if h["calibration_gap"] < LEGACY_CAL_GATE else "FAIL"
                o.append(f"      Cal gap    : {_f(h['calibration_gap'])}   "
                         f"(legacy fixed gate {LEGACY_CAL_GATE}: {verdict})")
        c = s["cv"]
        acc = f"  acc {c['accuracy']:.1%}" if isinstance(c["accuracy"], int | float) else ""
        o.append(f"    Walk-forward CV: ll {_f(c['log_loss'])} (naive {_f(c['naive_log_loss'])})"
                 f"  brier {_f(c['brier'])} (naive {_f(c['naive_brier'])})"
                 f"  cal gap {_f(c['calibration_gap'])}{acc}")
        p = s["promotion"]
        if p:
            o.append(f"    Promotion  : {'PROMOTED' if p['promoted'] else 'REFUSED'} "
                     f"{p['decided_at']} — {p['reason']}")
            inc = h.get("incumbent")
            if inc:
                o.append(f"                 incumbent on the same holdout: ll {_f(inc.get('log_loss'))}"
                         f"  cal gap {_f(inc.get('calibration_gap'))}")
        else:
            o.append("    Promotion  : pre-gate metadata — this model was saved UNCONDITIONALLY;"
                     " the next retrain gates it")
        o.append("")
    o.append("  Realised edge — settled O/U bets (bet_journal.json)")
    o.append("  " + "-" * 60)
    if edge is None:
        o.append("    journal unreadable")
    else:
        mk = edge["markets"]
        if not mk:
            o.append("    no settled O/U bets")
        else:
            o.append(f"    {'market':<10}{'settled':>8}{'ROI':>9}{'mean CLV':>10}{'CLV>0':>7}   last placed")
            for name, r in sorted(mk.items()):
                clv = f"{r['mean_clv']:+.2f}%" if r["mean_clv"] is not None else "n/a"
                pos = f"{r['clv_pos_share']:.0%}" if r["clv_pos_share"] is not None else "n/a"
                o.append(f"    {name:<10}{r['n']:>8}{r['roi']:>+8.1f}%{clv:>10}{pos:>7}   {r['last_placed']}")
        o.append(f"    O/U bets placed in the last {RECENT_DAYS} days: {edge['recent_placed']}")
    o.append("")
    return o


def render_1x2(meta: dict) -> list[str]:
    cv = meta.get("cv_summary", {})
    metrics = meta.get("metrics", {})
    o: list[str] = []
    o.append("  1X2 CATBOOST — SECONDARY (dashboard / Telegram / fantacalcio; bets nothing)")
    o.append("  " + "-" * 60)
    o.append(f"  Variant {meta.get('variant', 'unknown')}   {meta.get('n_features', '?')} features"
             f"   saved {str(meta.get('saved_at', 'unknown'))[:10]}")
    last3_acc = cv.get("last3_accuracy")
    if last3_acc is not None:
        gate = "PASS" if last3_acc >= REALISTIC_CEILING else "below ceiling"
        o.append(f"    Walk-forward last 3 folds: acc {last3_acc:.2%} ({gate} vs "
                 f"{REALISTIC_CEILING:.0%} realistic ceiling)"
                 f"   ll {_f(cv.get('last3_logloss'))}   brier {_f(cv.get('last3_brier'))}")
    if cv.get("all_folds_accuracy") is not None:
        o.append(f"    Walk-forward all folds:    acc {cv['all_folds_accuracy']:.2%}"
                 f"   ll {_f(cv.get('all_folds_logloss'))}")
    held = []
    if metrics.get("accuracy") is not None:
        held.append(f"acc {metrics['accuracy']:.2%}")
    if metrics.get("ece") is not None:
        held.append(f"ECE {metrics['ece']:.4f}")
    if metrics.get("kelly_roi") is not None:
        held.append(f"Kelly ROI {metrics['kelly_roi']:+.2f}% (noisy)")
    if held:
        o.append("    Held-out fold:             " + "   ".join(held))
    o.append("")
    return o


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main() -> int:
    ou_metas = [m for m in (_load_json(p) for p in sorted(OU_DIR.glob("ou_*_catboost_metadata.json")))
                if m] if OU_DIR.exists() else []
    ou_lines = [ou_line_status(m) for m in ou_metas]
    journal = _load_json(JOURNAL)
    edge = journal_edge(journal) if journal else None
    meta_1x2 = _load_json(METADATA)

    if not ou_lines and meta_1x2 is None:
        print(f"ERROR: neither {OU_DIR} nor {METADATA} is readable. Has anything been trained?",
              file=sys.stderr)
        return 1

    print()
    print("=" * 64)
    print("  MODEL STATUS — read from production metadata")
    print("=" * 64)
    for line in render_ou(ou_lines, edge, enabled_markets()):
        print(line)
    if meta_1x2 is None:
        print(f"  WARNING: {METADATA} not found — 1X2 section skipped")
        print()
    else:
        for line in render_1x2(meta_1x2):
            print(line)
    print("=" * 64)
    print("  Reality check: the O/U section is what places bets. 1X2 accuracy is")
    print("  capped at 53-55%; anything above ~56% is leakage or fiction. Trust")
    print("  this output, not docs.")
    print("=" * 64)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
