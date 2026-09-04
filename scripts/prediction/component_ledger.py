"""Per-component prediction ledger — the hub's calibration flywheel.

The ensemble's weights were frozen guesses (ENSEMBLE_WEIGHTS, hardcoded since
Phase 6) and nothing graded the components: the ML leg served skewed features
for months (P1, 2026-08-31) with no instrument to notice. This module closes
the loop, the same shape as the fantacalcio pred_ledger:

- snapshot(): upsert the CURRENT predictions.json per-component 1X2 probs per
  match, refusing writes once that match's kickoff has passed — the stored
  row is the last pre-kickoff forecast, ex-ante by construction.
- settle(): join finished matches (matches.parquet) and grade every component
  per row: multiclass Brier, log-loss, pick-correct.
- summary(): rolling per-component health; rot_alarm() notifies (change-gated)
  when a component's recent Brier degrades against its own trailing mean.
- refit_weights(): once >= REFIT_N settled rows hold ALL core components,
  optimize the mix weights by log-loss with shrinkage toward the current
  weights, gate on a time-ordered holdout, and write the override file
  data/models/ensemble_weights.json ONLY on a pass. The engine reads that
  override at init (fail-soft) — no silent weight changes, full provenance.

Every hook into live paths is fail-soft: a ledger failure must never block
predictions or bet commit.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "predictions" / "component_ledger.json"
PREDICTIONS = ROOT / "data" / "upcoming" / "predictions.json"
MATCHES = ROOT / "data" / "parsed" / "matches.parquet"
WEIGHTS_OVERRIDE = ROOT / "data" / "models" / "ensemble_weights.json"

CORE = ("ml", "market", "xg", "player_xg", "factor")   # 1X2 components
ROLL_RECENT = 20      # rot alarm window
ROLL_BASE = 100       # trailing baseline window
ROT_BRIER_DELTA = 0.04
REFIT_N = 100         # settled rows with all core components before refit
REFIT_HOLDOUT = 0.2   # newest slice, time-ordered
REFIT_SHRINK = 0.5    # pull fitted weights halfway toward current
DRIFT_ALERT_N = 15    # serving features outside the training band per row


def _load() -> dict:
    try:
        return json.loads(LEDGER.read_text())
    except (OSError, ValueError):
        return {"matches": {}, "alarm_state": {}}


def _save(led: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    led["updated_at"] = datetime.now(UTC).isoformat()
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(led, ensure_ascii=False))
    tmp.replace(LEDGER)


def _nt(name: str) -> str:
    from config.team_names import normalize_team
    return normalize_team(name or "") or (name or "")


def _kickoff_by_pair() -> dict:
    """(home, away) -> kickoff ts from the Sofascore fixtures file."""
    try:
        from scripts.utils.match_timing import _sofascore_fixture_files
        path = next(p for p, lg in _sofascore_fixture_files() if lg == "serie_a")
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError, StopIteration, ImportError):
        return {}
    out = {}
    for x in raw:
        h = _nt((x.get("homeTeam") or {}).get("name", ""))
        a = _nt((x.get("awayTeam") or {}).get("name", ""))
        ts = x.get("startTimestamp")
        if h and a and ts:
            out[(h, a)] = float(ts)
    return out


def snapshot(now_ts: float | None = None) -> dict:
    """Upsert pre-kickoff component forecasts from predictions.json.

    Per-match lock: a row refreshes freely until ITS kickoff, then freezes
    (frozen_at stamped once, later writes refused) — same discipline as the
    fantacalcio ledger, but per match instead of per round.
    """
    now = datetime.now(UTC).timestamp() if now_ts is None else now_ts
    try:
        pj = json.loads(PREDICTIONS.read_text())
    except (OSError, ValueError):
        return {"stored": 0, "frozen": 0, "skipped": 0}
    kicks = _kickoff_by_pair()
    led = _load()
    stored = frozen = skipped = 0
    for r in pj.get("predictions", []):
        if r.get("league") not in (None, "serie_a"):
            continue
        home, away = _nt(r.get("home_team")), _nt(r.get("away_team"))
        date = str(r.get("date") or "")[:10]
        if not home or not away or not date:
            continue
        key = f"{date}_{home}_{away}"
        kick = kicks.get((home, away))
        entry = led["matches"].get(key)
        lock = (entry or {}).get("kickoff_ts") or kick
        if lock is None and date < datetime.now(UTC).strftime("%Y-%m-%d"):
            # kickoff unknown and the date has passed: never take a post-hoc
            # forecast — ex-ante or nothing.
            continue
        if lock is not None and now >= float(lock):
            if entry and not entry.get("frozen_at"):
                entry["frozen_at"] = datetime.now(UTC).isoformat()
                frozen += 1
            else:
                skipped += 1
            continue
        comp = r.get("component_predictions") or {}
        comps = {}
        for name in CORE:
            c = comp.get(name)
            if isinstance(c, dict) and c.get("prob_H") is not None:
                comps[name] = {k: round(float(c[k]), 4)
                               for k in ("prob_H", "prob_D", "prob_A")}
        if not comps:
            continue
        bp = r.get("betting_probabilities") or r.get("probabilities") or {}
        led["matches"][key] = {
            "date": date, "home": home, "away": away,
            "kickoff_ts": kick,
            "snapshot_at": datetime.now(UTC).isoformat(),
            "frozen_at": (entry or {}).get("frozen_at"),
            "components": comps,
            "weights_applied": r.get("weights_applied"),
            "ensemble": {k: round(float(v), 4) for k, v in bp.items()
                         if isinstance(v, (int, float))},
            "ml_reasons": r.get("ml_reasons"),
            "ml_drift": r.get("ml_drift"),
            "settled_at": (entry or {}).get("settled_at"),
            "outcome": (entry or {}).get("outcome"),
            "grades": (entry or {}).get("grades"),
        }
        stored += 1
    if stored or frozen:
        _save(led)
    return {"stored": stored, "frozen": frozen, "skipped": skipped}


def _grade(probs: dict, outcome: str) -> dict:
    """Multiclass Brier + log-loss + pick correctness for one forecast."""
    y = {"H": 0.0, "D": 0.0, "A": 0.0}
    y[outcome] = 1.0
    p = {"H": float(probs.get("prob_H") or 0.0),
         "D": float(probs.get("prob_D") or 0.0),
         "A": float(probs.get("prob_A") or 0.0)}
    brier = sum((p[k] - y[k]) ** 2 for k in y)
    ll = -math.log(max(p[outcome], 1e-6))
    pick = max(p, key=lambda k: p[k])
    return {"brier": round(brier, 4), "log_loss": round(ll, 4),
            "correct": int(pick == outcome)}


def settle() -> int:
    """Grade every frozen-or-past-kickoff row whose result is available."""
    import pandas as pd
    led = _load()
    todo = {k: e for k, e in led["matches"].items() if not e.get("settled_at")}
    if not todo:
        return 0
    m = pd.read_parquet(MATCHES,
                        columns=["match_date", "home_team", "away_team",
                                 "result", "league"])
    m = m[(m.league == "serie_a") & m.result.notna()]
    res = {}
    for r in m.itertuples():
        res[(str(r.match_date)[:10], _nt(r.home_team), _nt(r.away_team))] = \
            str(r.result).upper()[:1]
    done = 0
    for key, e in todo.items():
        out = res.get((e["date"], e["home"], e["away"]))
        if out not in ("H", "D", "A"):
            continue
        e["outcome"] = out
        e["grades"] = {name: _grade(p, out)
                       for name, p in (e.get("components") or {}).items()}
        ens = e.get("ensemble") or {}
        if ens.get("home") is not None:
            e["grades"]["ensemble"] = _grade(
                {"prob_H": ens.get("home"), "prob_D": ens.get("draw"),
                 "prob_A": ens.get("away")}, out)
        e["settled_at"] = datetime.now(UTC).isoformat()
        done += 1
    if done:
        _save(led)
    return done


def _settled_rows(led: dict) -> list[dict]:
    rows = [e for e in led["matches"].values() if e.get("settled_at")]
    rows.sort(key=lambda e: (e["date"], e["home"]))
    return rows


def summary() -> dict:
    """Rolling per-component health: recent vs trailing Brier, LL, accuracy."""
    led = _load()
    rows = _settled_rows(led)
    names = sorted({n for e in rows for n in (e.get("grades") or {})})
    out = {"n_settled": len(rows), "components": {}}
    for name in names:
        seq = [e["grades"][name] for e in rows if name in (e.get("grades") or {})]
        if not seq:
            continue
        recent = seq[-ROLL_RECENT:]
        base = seq[-ROLL_BASE:]
        out["components"][name] = {
            "n": len(seq),
            "brier_recent": round(sum(g["brier"] for g in recent) / len(recent), 4),
            "brier_base": round(sum(g["brier"] for g in base) / len(base), 4),
            "log_loss_base": round(sum(g["log_loss"] for g in base) / len(base), 4),
            "accuracy_base": round(sum(g["correct"] for g in base) / len(base), 4),
        }
    return out


def rot_alarm() -> str | None:
    """Change-gated component-rot alert: recent Brier >> its own baseline.

    Needs a full recent window AND a baseline at least twice that long, so a
    cold ledger can't alarm on noise. Fires once per (component, degraded)
    state transition — the alarm_state signature clears on recovery.
    """
    led = _load()
    rows = _settled_rows(led)
    lines = []
    state = led.setdefault("alarm_state", {})
    changed = False
    for name in sorted({n for e in rows for n in (e.get("grades") or {})}):
        seq = [e["grades"][name] for e in rows if name in (e.get("grades") or {})]
        if len(seq) < max(ROLL_RECENT * 2, 40):
            continue
        recent = sum(g["brier"] for g in seq[-ROLL_RECENT:]) / ROLL_RECENT
        base = sum(g["brier"] for g in seq[-ROLL_BASE:]) / len(seq[-ROLL_BASE:])
        degraded = recent - base > ROT_BRIER_DELTA
        if degraded and state.get(name) != "degraded":
            lines.append(f"{name}: Brier {recent:.3f} recente vs {base:.3f} "
                         f"base (+{recent - base:.3f}) — possibile rot")
            state[name] = "degraded"
            changed = True
        elif not degraded and state.get(name) == "degraded":
            state[name] = "ok"
            changed = True
    if changed:
        _save(led)
    return "\n".join(lines) if lines else None


def drift_alarm() -> str | None:
    """Change-gated serving-skew alert: an upcoming row whose ML feature
    vector has >= DRIFT_ALERT_N features outside the training quantile band
    (the P1 signature — cache-served rows were skewed for months, silently).
    Fires on the none->degraded transition, clears when no row is skewed."""
    led = _load()
    bad = []
    for e in led["matches"].values():
        d = e.get("ml_drift") or {}
        if not e.get("settled_at") and d.get("n_out", 0) >= DRIFT_ALERT_N:
            bad.append(f"{e['home']}–{e['away']}: {d['n_out']}/{d.get('n_checked', '?')} "
                       f"features fuori banda training ({', '.join(d.get('out', [])[:4])}…)")
    state = led.setdefault("alarm_state", {})
    fired = None
    if bad and state.get("ml_drift") != "degraded":
        state["ml_drift"] = "degraded"
        fired = "\n".join(bad[:6])
        _save(led)
    elif not bad and state.get("ml_drift") == "degraded":
        state["ml_drift"] = "ok"
        _save(led)
    return fired


def _mix_ll(rows: list[dict], w: dict) -> float:
    tot = 0.0
    for e in rows:
        p = {"H": 0.0, "D": 0.0, "A": 0.0}
        for name, wt in w.items():
            c = e["components"].get(name)
            for k, f in (("H", "prob_H"), ("D", "prob_D"), ("A", "prob_A")):
                p[k] += wt * float(c[f])
        s = sum(p.values()) or 1.0
        tot += -math.log(max(p[e["outcome"]] / s, 1e-6))
    return tot / len(rows)


def refit_weights(current: dict | None = None) -> dict:
    """Evidence-based weight refit, deployment-gated. Returns a report dict.

    Coordinate-descent on the simplex over settled rows holding ALL core
    components, shrunk toward the current weights, judged on the newest
    REFIT_HOLDOUT slice. Writes the override ONLY when the holdout log-loss
    beats the current weights'. Below the data floor: report-only.
    """
    led = _load()
    rows = [e for e in _settled_rows(led)
            if e.get("outcome") and all(n in (e.get("components") or {})
                                        for n in CORE)]
    if current is None:
        # Baseline = what production actually runs: a previously deployed
        # override if one exists, else the hardcoded engine constants.
        try:
            prev = json.loads(WEIGHTS_OVERRIDE.read_text()).get("weights")
            current = prev if prev and set(prev) == set(CORE) else None
        except (OSError, ValueError):
            current = None
    cur = dict(current or {"factor": 0.035, "xg": 0.124, "ml": 0.605,
                           "player_xg": 0.032, "market": 0.205})
    if len(rows) < REFIT_N:
        return {"status": "below-floor", "n": len(rows), "floor": REFIT_N}
    cut = int(len(rows) * (1 - REFIT_HOLDOUT))
    train, hold = rows[:cut], rows[cut:]
    w = dict(cur)
    for _ in range(200):
        improved = False
        for name in CORE:
            for step in (0.05, -0.05, 0.02, -0.02):
                trial = dict(w)
                trial[name] = min(max(trial[name] + step, 0.0), 0.95)
                s = sum(trial.values()) or 1.0
                trial = {k: v / s for k, v in trial.items()}
                if _mix_ll(train, trial) < _mix_ll(train, w) - 1e-6:
                    w = trial
                    improved = True
        if not improved:
            break
    w = {k: round(cur[k] + REFIT_SHRINK * (w[k] - cur[k]), 4) for k in CORE}
    s = sum(w.values())
    w = {k: round(v / s, 4) for k, v in w.items()}
    ll_new, ll_cur = _mix_ll(hold, w), _mix_ll(hold, cur)
    report = {"status": "gated-fail", "n": len(rows),
              "holdout_ll_new": round(ll_new, 4),
              "holdout_ll_current": round(ll_cur, 4),
              "weights_fitted": w, "weights_current": cur}
    if ll_new < ll_cur:
        WEIGHTS_OVERRIDE.write_text(json.dumps({
            "weights": w, "fitted_at": datetime.now(UTC).isoformat(),
            "n_settled": len(rows),
            "holdout_ll_new": round(ll_new, 4),
            "holdout_ll_current": round(ll_cur, 4),
            "provenance": "component_ledger.refit_weights",
        }, indent=1))
        report["status"] = "deployed"
    return report


def run(now_ts: float | None = None) -> str:
    """snapshot + settle + rot alarm + (floor-gated) refit — one line out."""
    snap = snapshot(now_ts)
    n = settle()
    drift = drift_alarm()
    if drift:
        try:
            from scripts.pipeline.notify import notify
            notify("ML feature drift\n" + drift, title="Ensemble — ML drift",
                   level="alert", category="alert",
                   tg_html="<b>⚠️ ML serving drift</b>\n" + drift)
        except Exception:
            pass
    alarm = rot_alarm()
    if alarm:
        try:
            from scripts.pipeline.notify import notify
            notify("Component rot\n" + alarm, title="Ensemble — component rot",
                   level="alert", category="alert",
                   tg_html="<b>⚠️ Ensemble component rot</b>\n" + alarm)
        except Exception:
            pass
    refit = refit_weights()
    if refit.get("status") == "deployed":
        try:
            from scripts.pipeline.notify import notify
            notify(f"Ensemble weights refit deployed: {refit['weights_fitted']}",
                   title="Ensemble — weights refit", level="alert",
                   category="alert")
        except Exception:
            pass
    return (f"component ledger: stored={snap['stored']} frozen={snap['frozen']} "
            f"settled={n} refit={refit.get('status')}")


if __name__ == "__main__":
    print(run())
    print(json.dumps(summary(), indent=1))
