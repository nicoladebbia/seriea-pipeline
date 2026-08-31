"""Fair Odds Historical Tracking System.

Records model fair odds (1/probability) before each match and tracks accuracy
against actual outcomes. Builds a persistent ledger for long-term calibration
and model quality monitoring.

Two main operations:
1. record_predictions() — called before matches, saves model probabilities
2. settle_predictions() — called after matches, records actual outcomes

The ledger lives at data/betting/fair_odds_ledger.json.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from scripts.utils.ledger import load_json_ledger, save_json_ledger

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent.parent
BETTING_DIR = BASE / "data" / "betting"
LEDGER_PATH = BETTING_DIR / "fair_odds_ledger.json"
SUMMARY_PATH = BETTING_DIR / "fair_odds_summary.json"


def record_predictions(predictions: List[Dict]) -> Dict:
    """Record model fair odds for upcoming matches — ONE row per fixture.

    Keyed on (match, match date). A fixture re-predicted before kickoff has
    its row updated in place (latest pre-kickoff probabilities win); settled
    rows are never touched. Until 2026-08-31 the incoming key used the match
    date while the stored key used the prediction date, so nothing ever
    matched and every dashboard hit appended a fresh copy — 3,404 rows for 80
    fixtures (one fixture 104x), poisoning every accuracy statistic.

    Args:
        predictions: List of dicts with keys:
            match, home_team, away_team, date, commence_time,
            probabilities (home/draw/away), predicted_outcome,
            confidence_level, odds.h2h (home/draw/away).

    Returns:
        dict with counts of new and updated rows.
    """
    ledger = load_json_ledger(LEDGER_PATH)
    by_key = {(r.get("match"), str(r.get("date") or "")[:10]): r for r in ledger}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    new_count = 0
    updated = 0
    for p in predictions:
        match = p.get("match") or f"{p.get('home_team', '')} vs {p.get('away_team', '')}"
        date = p.get("date") or (
            p["commence_time"][:10] if p.get("commence_time") else ""
        )
        date = str(date)[:10]

        probs = p.get("probabilities", {}) or {}
        h2h = (p.get("odds", {}) or {}).get("h2h", {}) or {}

        # Fair odds = 1/probability (no margin). Use `or 0` not `default=0`
        # because probs.get("home", 0) returns None if the key is present
        # with explicit-None value, then None > 0.01 raises TypeError.
        fair_home = round(1 / probs["home"], 3) if (probs.get("home") or 0) > 0.01 else None
        fair_draw = round(1 / probs["draw"], 3) if (probs.get("draw") or 0) > 0.01 else None
        fair_away = round(1 / probs["away"], 3) if (probs.get("away") or 0) > 0.01 else None

        fields = {
            "prediction_date": today,
            "commence_time": p.get("commence_time", ""),
            # Model probabilities
            "prob_home": round(probs.get("home") or 0, 4),
            "prob_draw": round(probs.get("draw") or 0, 4),
            "prob_away": round(probs.get("away") or 0, 4),
            # Fair odds (1/prob)
            "fair_home": fair_home,
            "fair_draw": fair_draw,
            "fair_away": fair_away,
            # Market odds for comparison
            "market_home": h2h.get("home"),
            "market_draw": h2h.get("draw"),
            "market_away": h2h.get("away"),
            # Prediction metadata
            "predicted_outcome": p.get("predicted_outcome", ""),
            "confidence_level": p.get("confidence_level", ""),
        }

        existing = by_key.get((match, date))
        if existing is not None:
            if existing.get("settled"):
                continue
            if any(existing.get(k) != v for k, v in fields.items()):
                existing.update(fields)
                updated += 1
            continue

        record = {
            "match": match,
            "home_team": p.get("home_team", ""),
            "away_team": p.get("away_team", ""),
            "date": date,
            **fields,
            # Outcome (filled later by settle_predictions)
            "actual_outcome": None,
            "actual_score": None,
            "settled": False,
        }
        ledger.append(record)
        by_key[(match, date)] = record
        new_count += 1

    if new_count or updated:
        save_json_ledger(LEDGER_PATH, ledger)
        log.info("Fair odds ledger: %d new, %d updated", new_count, updated)

    return {"recorded": new_count, "updated": updated}


def _parse_score(score) -> Optional[tuple]:
    """Normalise a score to (home, away) ints.

    Accepts "2-1" / "2:1" strings, [2, 1] sequences and {"home", "away"}
    dicts. Until 2026-08-31 the journal's "2-1" STRING was indexed like a
    list — score[0]='2', score[1]='-' — so every journal result compared a
    digit against a dash and settled as HOME (a 1-1 draw became "HOME").
    """
    if score is None:
        return None
    if isinstance(score, dict):
        h, a = score.get("home"), score.get("away")
    elif isinstance(score, str):
        import re
        m = re.match(r"\s*(\d+)\s*[-:\u2013]\s*(\d+)", score)
        if not m:
            return None
        h, a = m.group(1), m.group(2)
    elif isinstance(score, (list, tuple)) and len(score) >= 2:
        h, a = score[0], score[1]
    else:
        return None
    try:
        return int(h), int(a)
    except (TypeError, ValueError):
        return None


def _outcome(h: int, a: int) -> str:
    return "HOME" if h > a else "AWAY" if a > h else "DRAW"


def _closest_result(candidates: List[Dict], record_date: str,
                    tolerance_days: int) -> Optional[Dict]:
    """Pick the candidate whose date is within `tolerance_days` of the record."""
    try:
        rd = datetime.strptime(record_date[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return None
    best, best_gap = None, None
    for c in candidates:
        try:
            cd = datetime.strptime(str(c.get("date", ""))[:10], "%Y-%m-%d")
        except (TypeError, ValueError):
            continue
        gap = abs((cd - rd).days)
        if gap <= tolerance_days and (best_gap is None or gap < best_gap):
            best, best_gap = c, gap
    return best


def settle_predictions(results: Dict = None, tolerance_days: int = 3) -> Dict:
    """Settle prediction records with actual outcomes — DATE-AWARE.

    A record is settled only by a result dated within `tolerance_days` of its
    own match date, and never before its match date has arrived. Until
    2026-08-31 the join was by bare fixture name, so "Roma vs Atalanta" on
    2026-09-05 was settled with January's Roma vs Atalanta (86 future
    fixtures "settled", the 20 most recent all wrong -> recent_20_accuracy 5%).

    Args:
        results: Optional {match_key: [{"date": "YYYY-MM-DD", "score": ...}]}.
            The legacy {match_key: {"score": [h, a]}} shape is still accepted
            and applied date-agnostically (explicit caller input).
            If None, reads ground truth + journal + live files.

    Returns:
        dict with count settled.
    """
    ledger = load_json_ledger(LEDGER_PATH)
    unsettled = [r for r in ledger if not r.get("settled")]
    if not unsettled:
        return {"settled": 0}

    if results is None:
        results = _load_results()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    settled_count = 0
    for record in unsettled:
        rec_date = str(record.get("date") or "")[:10]
        if rec_date and rec_date > today:
            continue  # fixture not played yet — a same-name result is a different game
        cands = results.get(record["match"])
        if not cands:
            continue
        if isinstance(cands, dict):  # legacy undated shape
            cands = [{"date": rec_date, "score": cands.get("score") or cands.get("final_score")}]
        chosen = _closest_result(cands, rec_date, tolerance_days)
        if not chosen:
            continue
        parsed = _parse_score(chosen.get("score"))
        if not parsed:
            continue
        h, a = parsed
        actual = _outcome(h, a)
        record["actual_outcome"] = actual
        record["actual_score"] = [h, a]
        record["settled"] = True
        record["settled_at"] = datetime.now(timezone.utc).isoformat()
        record["prediction_correct"] = record.get("predicted_outcome") == actual
        settled_count += 1

    if settled_count:
        save_json_ledger(LEDGER_PATH, ledger)
        _update_summary(ledger)
        log.info("Settled %d fair odds records", settled_count)

    return {"settled": settled_count}


def get_summary() -> Dict:
    """Get the fair odds tracking summary."""
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            return json.load(f)
    return {}


def _update_summary(ledger: List[Dict]):
    """Update the aggregated summary from the full ledger."""
    settled = [r for r in ledger if r.get("settled")]
    if not settled:
        return

    total = len(settled)
    correct = len([r for r in settled if r.get("prediction_correct")])

    # Calibration by probability bucket
    # For each outcome (H/D/A), bucket predictions by their probability
    calibration = {}
    for side, prob_key, outcome in [
        ("home", "prob_home", "HOME"),
        ("draw", "prob_draw", "DRAW"),
        ("away", "prob_away", "AWAY"),
    ]:
        buckets = {}
        for r in settled:
            prob = r.get(prob_key, 0)
            bucket = f"{int(prob * 10) * 10}-{int(prob * 10) * 10 + 10}%"
            if bucket not in buckets:
                buckets[bucket] = {"count": 0, "actual": 0, "avg_prob": 0}
            buckets[bucket]["count"] += 1
            buckets[bucket]["avg_prob"] += prob
            if r.get("actual_outcome") == outcome:
                buckets[bucket]["actual"] += 1

        for b, stats in buckets.items():
            stats["avg_prob"] = round(stats["avg_prob"] / stats["count"] * 100, 1)
            stats["actual_rate"] = round(stats["actual"] / stats["count"] * 100, 1)
            stats["cal_error"] = round(stats["actual_rate"] - stats["avg_prob"], 1)

        calibration[side] = buckets

    # Model vs market accuracy
    model_correct = len([r for r in settled if r.get("prediction_correct")])
    market_correct = 0
    for r in settled:
        mh, md, ma = r.get("market_home"), r.get("market_draw"), r.get("market_away")
        if mh and md and ma:
            # Market favorite = lowest odds
            min_odds = min(mh, md, ma)
            if min_odds == mh:
                market_pred = "HOME"
            elif min_odds == ma:
                market_pred = "AWAY"
            else:
                market_pred = "DRAW"
            if market_pred == r.get("actual_outcome"):
                market_correct += 1

    matches_with_market = len([
        r for r in settled
        if r.get("market_home") and r.get("market_draw") and r.get("market_away")
    ])

    # Fair odds value tracking: if model said fair odds < market odds (value),
    # did that outcome hit more often?
    value_bets = {"total": 0, "correct": 0, "avg_edge": 0}
    for r in settled:
        # Check if predicted outcome had value (fair odds < market odds)
        pred = r.get("predicted_outcome")
        if pred == "HOME":
            fair, market = r.get("fair_home"), r.get("market_home")
        elif pred == "AWAY":
            fair, market = r.get("fair_away"), r.get("market_away")
        elif pred == "DRAW":
            fair, market = r.get("fair_draw"), r.get("market_draw")
        else:
            continue
        if fair and market and fair < market:
            value_bets["total"] += 1
            edge = (market / fair - 1) * 100
            value_bets["avg_edge"] += edge
            if r.get("prediction_correct"):
                value_bets["correct"] += 1

    if value_bets["total"] > 0:
        value_bets["hit_rate"] = round(
            value_bets["correct"] / value_bets["total"] * 100, 1
        )
        value_bets["avg_edge"] = round(
            value_bets["avg_edge"] / value_bets["total"], 1
        )
    else:
        value_bets["hit_rate"] = 0
        value_bets["avg_edge"] = 0

    # Recent form (last 20 predictions)
    recent = sorted(settled, key=lambda r: r.get("settled_at") or "", reverse=True)[:20]
    recent_correct = len([r for r in recent if r.get("prediction_correct")])

    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_predictions": total,
        "total_correct": correct,
        "accuracy_pct": round(correct / total * 100, 1),
        "model_accuracy": round(model_correct / total * 100, 1),
        "market_accuracy": round(
            market_correct / matches_with_market * 100, 1
        ) if matches_with_market > 0 else None,
        "matches_with_market_odds": matches_with_market,
        "recent_20_accuracy": round(recent_correct / len(recent) * 100, 1) if recent else 0,
        "value_bets": value_bets,
        "calibration": calibration,
        "unsettled_count": len([r for r in ledger if not r.get("settled")]),
    }

    BETTING_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)


def _load_results() -> Dict[str, List[Dict]]:
    """Dated results index: {"Home vs Away": [{"date", "score"}, ...]}.

    Sources, most authoritative first: data/parsed/matches.parquet (canonical
    ground truth, both leagues), the bet journal, and data/live/<date>.json.
    Every entry carries the date it belongs to so settlement can refuse a
    same-name fixture from another round.
    """
    from collections import defaultdict
    results: Dict[str, List[Dict]] = defaultdict(list)

    # 1. Ground truth parquet — complete and dated
    mp = BASE / "data" / "parsed" / "matches.parquet"
    if mp.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(mp, columns=["home_team", "away_team", "match_date",
                                              "home_score", "away_score"])
            df = df.dropna(subset=["home_score", "away_score"])
            for r in df.itertuples(index=False):
                results[f"{r.home_team} vs {r.away_team}"].append({
                    "date": str(r.match_date)[:10],
                    "score": (int(r.home_score), int(r.away_score)),
                })
        except Exception as e:
            log.debug("matches.parquet unavailable for fair-odds settlement: %s", e)

    # 2. Bet journal (settled bets carry result_score + the match date)
    journal_path = BETTING_DIR / "bet_journal.json"
    if journal_path.exists():
        try:
            with open(journal_path) as f:
                journal = json.load(f)
            bets = journal.get("bets", journal)
            if isinstance(bets, dict):
                bets = list(bets.values())
            for entry in bets:
                if not isinstance(entry, dict) or not entry.get("match"):
                    continue
                score = entry.get("result_score") or entry.get("actual_score")
                if score:
                    results[entry["match"]].append({
                        "date": str(entry.get("date") or "")[:10],
                        "score": score,
                    })
        except (json.JSONDecodeError, IOError):
            pass

    # 3. Live match files — date is the filename
    live_dir = BASE / "data" / "live"
    if live_dir.exists():
        for fpath in sorted(live_dir.glob("20*.json"), reverse=True)[:14]:
            try:
                with open(fpath) as f:
                    data = json.load(f)
                for mk, mdata in data.get("matches", {}).items():
                    if mdata.get("status") == "completed":
                        score = mdata.get("final_score") or mdata.get("score")
                        if score:
                            results[mk].append({"date": fpath.stem[:10], "score": score})
            except (json.JSONDecodeError, IOError):
                pass

    return dict(results)


def rebuild_ledger() -> Dict:
    """Collapse duplicates to one row per (match, date) and re-settle from scratch.

    Keeps the latest prediction per fixture, wipes every settlement field (the
    pre-2026-08-31 ledger settled future fixtures against same-name results
    and parsed "1-1" as HOME), then runs the date-aware settlement. Idempotent.
    """
    ledger = load_json_ledger(LEDGER_PATH)
    best: Dict = {}
    for r in ledger:
        key = (r.get("match"), str(r.get("date") or "")[:10])
        cur = best.get(key)
        if cur is None or str(r.get("prediction_date") or "") >= str(cur.get("prediction_date") or ""):
            best[key] = r
    rows = []
    for r in best.values():
        r = dict(r)
        r.update({"actual_outcome": None, "actual_score": None,
                  "settled": False, "prediction_correct": None})
        r.pop("settled_at", None)
        rows.append(r)
    rows.sort(key=lambda r: (str(r.get("date") or ""), str(r.get("match") or "")))
    save_json_ledger(LEDGER_PATH, rows)
    res = settle_predictions()
    _update_summary(load_json_ledger(LEDGER_PATH))
    log.info("Fair odds ledger rebuilt: %d rows -> %d unique fixtures, %d settled",
             len(ledger), len(rows), res.get("settled", 0))
    return {"before": len(ledger), "after": len(rows), "settled": res.get("settled", 0)}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Fair odds ledger maintenance")
    ap.add_argument("--settle", action="store_true", help="settle unsettled records")
    ap.add_argument("--rebuild", action="store_true", help="dedup + re-settle the whole ledger")
    args = ap.parse_args()
    if args.rebuild:
        print(rebuild_ledger())
    elif args.settle:
        print(settle_predictions())
    else:
        ap.print_help()
