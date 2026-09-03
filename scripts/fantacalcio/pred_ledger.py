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
    if now >= kick:
        # Freeze: mark the last pre-kickoff snapshot, never write a new one.
        entry = led["rounds"].get(key)
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
            "updated_at": led.get("updated_at")}


if __name__ == "__main__":
    try:
        adv = json.loads((ROOT / "data" / "fantacalcio" / "xi_advice.json").read_text())
        print("snapshot:", snapshot(adv))
    except (OSError, ValueError) as e:
        print(f"no advice to snapshot: {e}")
    print("reconciled:", reconcile())
    print(json.dumps(summary(), indent=1, ensure_ascii=False))
