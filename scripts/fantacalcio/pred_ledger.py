"""Prediction ledger: freeze each giornata's forecast, reconcile it with the voti.

The advisor predicts, the tracker scores — nothing until now compared the two.
This module closes the loop so the constants can be FITTED instead of guessed:

  * ``snapshot(adv)`` stores the advice for the coming round — every roster
    player with his predicted p_play (+source), conditional fantavoto ``exp``
    and slot (xi/bench/tribuna/out). It overwrites freely while the round's
    first kickoff is in the future and REFUSES afterwards: the stored row is
    ex-ante by construction, or it does not exist. A round we never
    snapshotted pre-kickoff stays an honest hole, never a backfill.
  * ``reconcile()`` joins frozen rounds against the published voti parquet
    (RECONCILE_GRACE_DAYS after first kickoff, so a half-graded weekend can't
    poison the actuals) and stores per-player errors + round metrics.
  * ``summary()`` aggregates the calibration signal the heuristics wait for:
    per p_play source (probabili / ballottaggio / model), predicted play rate
    vs realized; per round, MAE / bias of the fantavoto forecast.

A player we declared OUT is scored as a p_play=0 prediction — if he plays,
that is exactly the miss the Brier term must record.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "fantacalcio" / "pred_ledger.json"
SCHEDULE = ROOT / "data" / "fantacalcio" / "league_schedule.json"
ROSTERS = ROOT / "data" / "fantacalcio" / "league_rosters.json"
VOTI_DIR = ROOT / "data" / "fantacalcio" / "voti"
SEASON = "2026-27"
RECONCILE_GRACE_DAYS = 4.0


def _load() -> dict:
    try:
        return json.loads(LEDGER.read_text())
    except (OSError, ValueError):
        return {"season": SEASON, "rounds": {}}


def _save(led: dict) -> None:
    led["updated_at"] = datetime.now(UTC).isoformat()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=1, ensure_ascii=False))


def _round_parquet(rnd: int) -> Path:
    return VOTI_DIR / f"round_{SEASON.replace('-', '_')}_{rnd:02d}.parquet"


def _row(p: dict, slot: str) -> dict:
    # An OUT player is a "won't play" call; own that as p=0 rather than keeping
    # the roster's model prior, which nobody acted on.
    pp = 0.0 if slot == "out" else p.get("p_play")
    return {"id": p.get("id"), "nome": p["nome"], "R": p["R"], "team": p["team"],
            "slot": slot, "opp": p.get("opp"), "home": p.get("home"),
            "p_play": pp, "p_play_src": p.get("p_play_src"),
            # rank on the rigoristi page — the refit handle for the
            # RIGORISTA_BONUS heuristic once ~5 rounds are graded
            "rigorista": p.get("rigorista"),
            # anytime-scorer market tilt inputs — the refit handles for
            # SCORER_W / SCORER_K once graded vs realized goals
            "scorer_edge": p.get("scorer_edge"),
            "lam_mkt": p.get("lam_mkt"), "lam_own": p.get("lam_own"),
            # flywheel write-back actually applied to this forecast, so a
            # graded round can audit its own correction
            "exp_bias": p.get("exp_bias"),
            # rate-prior fields (player_rates): graded vs realized goals /
            # assists once voti land — the refit handles for RATE_CAP
            "p_goal": p.get("p_goal"), "p_assist": p.get("p_assist"),
            "rate_tilt": p.get("rate_tilt"),
            "exp": p.get("exp"), "exp_voto": p.get("exp_voto")}


def _h2h_forecasts(riv: dict | None) -> list[dict]:
    """Per-competition opponent + my predicted P(win), lifted from the rival
    matrix at snapshot time. Ex-ante like everything else in this file."""
    out = []
    for nx in ((riv or {}).get("next_opponents") or []):
        opp = nx.get("opponent")
        if not opp:
            continue
        r = next((r for r in riv.get("rivals", [])
                  if r.get("team") == opp), None)
        if not r or r.get("p_win") is None:
            continue
        me = riv.get("me") or {}
        row = {"competition": nx.get("competition"), "opponent": opp,
               "p_win": r["p_win"],
               "my_exp": me.get("exp_total") or me.get("total"),
               "opp_exp": r.get("exp_total") or r.get("total")}
        # MC triple (2026-09-03): draw band + expected league points, so
        # grading can calibrate W/D/L instead of a binary win prob.
        for k in ("p_draw", "p_loss", "e_pts"):
            if r.get(k) is not None:
                row[k] = r[k]
        out.append(row)
    return out


def snapshot(adv: dict, riv: dict | None = None,
             now_ts: float | None = None) -> str:
    """Store/refresh the coming round's forecast. Returns what happened."""
    rnd, kick = adv.get("round"), adv.get("first_kickoff")
    if not rnd or not kick or not adv.get("xi"):
        return "skipped-no-round"
    now = datetime.now(UTC).timestamp() if now_ts is None else now_ts
    led = _load()
    key = str(rnd)
    entry = led["rounds"].get(key)
    # The lock is the round's ORIGINAL first kickoff, held by the stored
    # entry. The advice's first_kickoff rolls forward to the next remaining
    # fixture once matches start, so comparing against `kick` alone let a
    # post-lock rebuild OVERWRITE the frozen-worthy forecast with slot=out
    # rows and a Saturday lock (paid 2026-09-04 17:34: the 3-5-2 snapshot
    # was clobbered by 4-4-2 junk and restored from the state backup).
    lock = float((entry or {}).get("first_kickoff") or kick)
    if now >= lock:
        # Freeze: mark the last pre-kickoff snapshot, never write a new one.
        if entry and not entry.get("frozen_at"):
            entry["frozen_at"] = datetime.now(UTC).isoformat()
            _save(led)
            return "frozen"
        return "skipped-post-kickoff"
    players = ([_row(p, "xi") for p in adv.get("xi", [])]
               + [_row(p, "bench") for p in adv.get("bench", [])]
               + [_row(p, "tribuna") for p in adv.get("tribuna", [])]
               + [_row(p, "out") for p in adv.get("unavailable", [])])
    led["rounds"][key] = {
        "snapshot_at": datetime.now(UTC).isoformat(),
        "first_kickoff": kick, "frozen_at": None, "reconciled_at": None,
        "module": adv.get("module"), "predicted_total": adv.get("total"),
        "predicted_exp_total": adv.get("exp_total"),
        "modifier": adv.get("modifier"), "players": players,
    }
    h2h = _h2h_forecasts(riv)
    if h2h:
        led["rounds"][key]["h2h"] = h2h
    _save(led)
    return "updated"


EXP_BIAS_N0 = 60.0   # pseudo-observations: at n=60 graded player-rounds the
EXP_BIAS_CAP = 0.5   # correction is half the measured bias; hard cap ±0.5 fv


def exp_bias() -> dict[str, dict]:
    """Per-role forecast bias (actual − predicted fantavoto) across every
    reconciled round, with the shrunk correction the advisor writes back.

    This is the calibration flywheel's closing arc: reconcile() grades each
    frozen forecast, this aggregates the signed errors, and _apply_exp_bias
    in xi_advisor feeds the correction into the next round's exp. Shrinkage
    n/(n+N0) means it self-arms smoothly — ~5% of the bias after one round,
    half at 60 graded player-rounds — with no cliff and no manual refit.
    """
    agg: dict[str, list[float]] = {}
    for e in _load()["rounds"].values():
        if not e.get("reconciled_at"):
            continue
        for p in e.get("players", []):
            if p.get("err_fv") is not None and p.get("R"):
                agg.setdefault(p["R"], []).append(float(p["err_fv"]))
    out: dict[str, dict] = {}
    for role, errs in agg.items():
        n = len(errs)
        b = sum(errs) / n
        corr = max(-EXP_BIAS_CAP,
                   min(EXP_BIAS_CAP, n / (n + EXP_BIAS_N0) * b))
        out[role] = {"n": n, "bias": round(b, 3), "corr": round(corr, 3)}
    return out


def frozen_entry(rnd: int | None) -> dict | None:
    """The stored, frozen forecast for a round still in play — or None.

    frozen_at is only ever stamped once the round's first kickoff has passed,
    so its presence alone means "locked": advisory rebuilds must serve this
    forecast instead of re-optimizing against the remaining fixtures — which
    drops already-kicked-off teams as "no fixture this round" and churns the
    module on the dashboard mid-round (the Douvikas 2026-09-04 complaint).
    """
    if rnd is None:
        return None
    entry = _load()["rounds"].get(str(rnd))
    return entry if entry and entry.get("frozen_at") else None


def reconcile(now_ts: float | None = None) -> list[int]:
    """Join every ripe, unreconciled round against its voti parquet."""
    now = datetime.now(UTC).timestamp() if now_ts is None else now_ts
    led = _load()
    done: list[int] = []
    for key, entry in led["rounds"].items():
        if entry.get("reconciled_at"):
            continue
        if now - entry["first_kickoff"] < RECONCILE_GRACE_DAYS * 86400:
            continue
        pq = _round_parquet(int(key))
        if not pq.exists():
            continue
        voti = pd.read_parquet(pq)
        by_pid = {int(r.pid): r for r in voti.itertuples() if pd.notna(r.pid)}
        errs, briers = [], []
        for p in entry["players"]:
            r = by_pid.get(int(p["id"])) if p.get("id") is not None else None
            played = bool(r.played) if r is not None else False
            p["actual_played"] = played
            p["actual_voto"] = float(r.voto) if played and pd.notna(r.voto) else None
            p["actual_fantavoto"] = (float(r.fantavoto)
                                     if played and pd.notna(r.fantavoto) else None)
            if p.get("p_play") is not None:
                p["play_brier"] = round((float(p["p_play"]) - played) ** 2, 4)
                briers.append(p["play_brier"])
            if played and p.get("exp") is not None and p["actual_fantavoto"] is not None:
                p["err_fv"] = round(p["actual_fantavoto"] - p["exp"], 2)
                errs.append(p["err_fv"])
        entry["metrics"] = {
            "n_played": sum(1 for p in entry["players"] if p.get("actual_played")),
            "n_scored": len(errs),
            "mae_fv": round(sum(abs(e) for e in errs) / len(errs), 3) if errs else None,
            "bias_fv": round(sum(errs) / len(errs), 3) if errs else None,
            "play_brier": round(sum(briers) / len(briers), 4) if briers else None,
        }
        entry["reconciled_at"] = datetime.now(UTC).isoformat()
        done.append(int(key))
    if done:
        _save(led)
    return done


def _h2h_result(fx: dict, my_name: str) -> dict | None:
    """W/D/L + fantapunti from one played score cell, from MY side. Pure;
    an unplayed "-" cell (or a fixture not mine) grades nothing."""
    import re
    sc = str(fx.get("score") or "")
    if not re.fullmatch(r"\d+-\d+", sc):
        return None
    gh, ga = (int(x) for x in sc.split("-"))
    if fx.get("home") == my_name:
        mine, theirs = gh, ga
        fp_m, fp_o = fx.get("fp_home"), fx.get("fp_away")
    elif fx.get("away") == my_name:
        mine, theirs = ga, gh
        fp_m, fp_o = fx.get("fp_away"), fx.get("fp_home")
    else:
        return None
    return {"result": "W" if mine > theirs else ("L" if mine < theirs else "D"),
            "goals": f"{mine}-{theirs}", "fp_mine": fp_m, "fp_opp": fp_o}


def reconcile_h2h() -> list[str]:
    """Grade stored H2H forecasts against the calendar score cells. The cells
    exist only after a fresh calendar export is re-dropped, so this lags by
    design and re-runs harmlessly until they appear. Over a season this is
    the Brier record of the P(win) matrix."""
    led = _load()
    try:
        schedule = json.loads(SCHEDULE.read_text())
        my_name = json.loads(ROSTERS.read_text()).get("my_team")
    except (OSError, ValueError):
        return []
    if not my_name:
        return []
    graded: list[str] = []
    for key, entry in led["rounds"].items():
        for h in entry.get("h2h", []):
            if h.get("result"):
                continue
            cd = (schedule.get("competitions") or {}).get(h["competition"]) or {}
            rd = next((r for r in cd.get("rounds", [])
                       if r.get("sa_round") == int(key)), None)
            if not rd:
                continue
            mine = next((f for f in rd.get("fixtures", [])
                         if my_name in (f.get("home"), f.get("away"))), None)
            res = _h2h_result(mine, my_name) if mine else None
            if res:
                h.update(res)
                h["graded_at"] = datetime.now(UTC).isoformat()
                graded.append(f"{key}:{h['competition']}")
    if graded:
        _save(led)
    return graded


def _ladder_observations() -> list[tuple[float, int, bool]]:
    """(fantapunti, goals, fanta_home) for every played score cell, both
    competitions. A side under 20 fp is a forfeit/rest artifact, not a real
    lineup — its pair would poison the fit, so it is dropped per side."""
    import re
    try:
        schedule = json.loads(SCHEDULE.read_text())
    except (OSError, ValueError):
        return []
    seen: set[tuple] = set()
    obs: list[tuple[float, int, bool]] = []
    for cd in (schedule.get("competitions") or {}).values():
        for rd in cd.get("rounds", []):
            for fx in rd.get("fixtures", []):
                sc = str(fx.get("score") or "")
                if not re.fullmatch(r"\d+-\d+", sc):
                    continue
                gh, ga = (int(x) for x in sc.split("-"))
                for team, fp, g, home in (
                        (fx.get("home"), fx.get("fp_home"), gh, True),
                        (fx.get("away"), fx.get("fp_away"), ga, False)):
                    if fp is None or float(fp) < 20.0:
                        continue
                    key = (team, rd.get("sa_round"), home, float(fp), g)
                    if key in seen:
                        continue
                    seen.add(key)
                    obs.append((float(fp), g, home))
    return obs


def verify_goal_ladder() -> str | None:
    """Solve the goal ladder (base, step, home fp bonus) from the played
    score cells and check the configured GOAL_BASE/GOAL_STEP against it.

    The MC's thresholds are an assumption until the first settled giornata
    (so is the importer's played-row cell mapping — a layout that fits NO
    ladder is that bug, not a rules change). Verdict persists in the ledger
    under ``goal_ladder``; the returned alert string is non-None only when
    the verdict CHANGES — the tracker runs several times a day and a
    standing mismatch must not re-alert every cycle.
    """
    from scripts.fantacalcio.xi_advisor import GOAL_BASE, GOAL_STEP

    obs = _ladder_observations()
    if not obs:
        return None
    bases = [50.0 + 0.5 * i for i in range(61)]      # 50 .. 80
    steps = [2.0 + 0.5 * i for i in range(13)]       # 2 .. 8
    advs = (0.0, 1.0, 2.0, 3.0)
    eps = 1e-9

    def fits(base: float, step: float, adv: float) -> bool:
        for fp, g, home in obs:
            eff = fp + (adv if home else 0.0)
            pred = 0 if eff < base else int((eff - base) / step + eps) + 1
            if pred != g:
                return False
        return True

    feasible = [(b, s, a) for b in bases for s in steps for a in advs
                if fits(b, s, a)]
    conf_ok = fits(GOAL_BASE, GOAL_STEP, 0.0)
    zero_adv = [(b, s) for b, s, a in feasible if a == 0.0]
    unique_step = len({s for _, s in zero_adv}) == 1 if zero_adv else False

    led = _load()
    prev = led.get("goal_ladder") or {}
    verdict = {
        "checked_at": datetime.now(UTC).isoformat(),
        "n_obs": len(obs),
        "configured": [GOAL_BASE, GOAL_STEP],
        "configured_ok": conf_ok,
        "n_feasible": len(feasible),
        "base_range": [min(b for b, _, _ in feasible),
                       max(b for b, _, _ in feasible)] if feasible else None,
        "step_range": [min(s for _, s, _ in feasible),
                       max(s for _, s, _ in feasible)] if feasible else None,
        "unique_step_at_zero_adv": unique_step,
    }
    changed = (prev.get("configured_ok") != conf_ok
               or (not conf_ok and len(obs) > int(prev.get("n_obs") or 0))
               or (not prev and conf_ok))
    led["goal_ladder"] = verdict
    _save(led)
    if not changed:
        return None
    if not feasible:
        return (f"NESSUNA scala gol coerente con {len(obs)} celle giocate — "
                f"probabile mapping celle rotto in import_rosters (fp/gol "
                f"scambiati?), NON un cambio regole. MC inaffidabile.")
    if not conf_ok:
        return (f"Scala gol SMENTITA dai risultati: configurata "
                f"{GOAL_BASE:g}+{GOAL_STEP:g}, ma {len(obs)} celle giocate "
                f"ammettono base {verdict['base_range']} step "
                f"{verdict['step_range']}. Correggere GOAL_BASE/GOAL_STEP "
                f"in xi_advisor.")
    prec = ("UNICA" if unique_step and len(zero_adv) <= 3
            else f"coerente ({len(feasible)} alternative aperte)")
    return (f"Scala gol {GOAL_BASE:g}+{GOAL_STEP:g} verificata sui "
            f"risultati reali ({len(obs)} celle): {prec}.")


def summary() -> dict:
    """Cross-round calibration: the numbers that will refit the constants."""
    led = _load()
    rounds = []
    src: dict[str, dict] = {}
    for key in sorted(led["rounds"], key=int):
        e = led["rounds"][key]
        rounds.append({"round": int(key), "frozen": bool(e.get("frozen_at")),
                       "reconciled": bool(e.get("reconciled_at")),
                       "module": e.get("module"),
                       "predicted_total": e.get("predicted_total"),
                       **(e.get("metrics") or {})})
        if not e.get("reconciled_at"):
            continue
        for p in e["players"]:
            if p.get("p_play") is None or "actual_played" not in p:
                continue
            k = p.get("p_play_src") or ("out" if p["slot"] == "out" else "model")
            a = src.setdefault(k, {"n": 0, "p_sum": 0.0, "played": 0})
            a["n"] += 1
            a["p_sum"] += float(p["p_play"])
            a["played"] += int(p["actual_played"])
    calib = {k: {"n": a["n"],
                 "predicted_rate": round(a["p_sum"] / a["n"], 3),
                 "realized_rate": round(a["played"] / a["n"], 3)}
             for k, a in src.items() if a["n"]}
    return {"season": led.get("season"), "rounds": rounds, "calibration": calib,
            "exp_bias": exp_bias(),
            "updated_at": led.get("updated_at")}


if __name__ == "__main__":
    try:
        adv = json.loads((ROOT / "data" / "fantacalcio" / "xi_advice.json").read_text())
        print("snapshot:", snapshot(adv))
    except (OSError, ValueError) as e:
        print(f"no advice to snapshot: {e}")
    print("reconciled:", reconcile())
    print(json.dumps(summary(), indent=1, ensure_ascii=False))
