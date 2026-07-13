"""Live Track Record grading for the World Cup predictions.

Joins the pre-kickoff snapshots (predictions_archive.json — last write
BEFORE kickoff wins, immutable after) against played results and scores:

- 1X2 pick hit rate (argmax of the deployed probabilities)
- multiclass Brier of the deployed system vs the frozen market vs base rate
  (model-vs-market compared LIKE-FOR-LIKE on the same odds-covered subset)

KNOCKOUT SEMANTICS: results.csv stores extra-time-inclusive scores, but the
predictions are 90-minute probabilities. For non-group snapshots the
90-minute outcome is reconstructed: a match in shootouts.csv was a draw at
90'; otherwise goals with minute <= 90 from goalscorers.csv rebuild the
score, and a match whose scorer rows don't fully cover the final score is
SKIPPED rather than graded wrong (the csv is a sample).

Honesty guards: snapshots stamped at-or-after their kickoff are never graded
(a regenerated archive cannot mint 'predictions' for played matches), and
the base-rate comparator pool is frozen at pre-tournament data.

Run: python3 -m scripts.worldcup.grading
Writes data/worldcup/track_record.json, served at /api/worldcup/record.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.worldcup.engine import (
    DATA_DIR,
    atomic_write_json,
    canon_team,
    load_results_with_live,
    read_json_safe,
)

ARCHIVE_JSON = DATA_DIR / "predictions_archive.json"
TRACK_RECORD_JSON = DATA_DIR / "track_record.json"
GOALSCORERS_CSV = DATA_DIR / "international_goalscorers.csv"
SHOOTOUTS_CSV = DATA_DIR / "international_shootouts.csv"
# Live overlays carry the penalty-shootout winner (decided_by/penalties/winner)
# that the martj42 CSV lacks — needed to resolve a knockout level at full time.
LIVE_RESULT_JSONS = ("sofascore_results.json", "fbref_results.json")

# Comparator pool frozen pre-tournament: graded matches must never enter
# their own baseline.
BASE_POOL_CUTOFF = "2026-06-01"


def _load_advance_winners() -> dict[frozenset[str], str]:
    """{unordered canon team-pair -> canon winner} for ties decided beyond 90'.

    The results CSV stores a knockout that went to penalties as its level
    pre-shootout score (e.g. 1-1), losing who actually advanced. The live
    Sofascore/FBref overlays carry ``winner`` + ``decided_by`` for those, so a
    knockout level at full time can still be graded on the REAL outcome (who
    went through) instead of being scored a draw. Keyed by unordered pair to
    survive neutral-venue home/away swaps.
    """
    winners: dict[frozenset[str], str] = {}
    for src in LIVE_RESULT_JSONS:
        blob = read_json_safe(DATA_DIR / src, {})
        rows = blob.get("results", []) if isinstance(blob, dict) else []
        for r in rows:
            w = r.get("winner")
            if not w:
                continue
            try:
                pair = frozenset((canon_team(str(r["home"])), canon_team(str(r["away"]))))
            except (KeyError, TypeError):
                continue
            # Sofascore is listed first and is richer; don't let FBref overwrite it.
            winners.setdefault(pair, canon_team(str(w)))
    return winners


def resolve_knockout(
    home: str,
    away: str,
    hs: int,
    as_: int,
    advance_winners: dict[frozenset[str], str],
) -> tuple[str, str] | None:
    """Resolve a played knockout to ``(outcome, display_score)`` on the ACTUAL
    result — who advanced — not the 90-minute score.

    Single source of truth shared by the track record, the per-match drill-down
    and the combo grader so the three can't drift on a knockout's outcome.

    - decided in normal/extra time (``hs != as_``): the score stands.
    - level at full time: went to penalties; the martj42 CSV lost the winner, so
      it's read from the live overlay (:func:`_load_advance_winners`). ``outcome``
      is the advancer; the in-play score is kept (penalties aren't goals) and the
      display is annotated. No winner anywhere → ``None`` (refuse to grade a
      knockout a draw — nobody draws a knockout).

    ``outcome`` is what 1X2 grades against; the returned score is the in-play
    (90'+ET) score that goal-quantity markets keep grading against.
    """
    if hs != as_:
        return ("home" if hs > as_ else "away", f"{hs}-{as_}")
    pair = frozenset((canon_team(home), canon_team(away)))
    winner = advance_winners.get(pair)
    if winner is None:
        return None
    outcome = "home" if winner == canon_team(home) else "away"
    return (outcome, f"{hs}-{as_} (pens)")


def _base_rates(df: pd.DataFrame) -> tuple[float, float, float]:
    pool = df[
        (df["date"] >= pd.Timestamp("2010-01-01"))
        & (df["date"] < pd.Timestamp(BASE_POOL_CUTOFF))
        & df["neutral"]
        & ~df["tournament"].str.contains("Friendly", case=False)
    ]
    o = np.sign(pool["home_score"] - pool["away_score"])
    return (
        float((o == 1).mean()),
        float((o == 0).mean()),
        float((o == -1).mean()),
    )


def _find_result(
    df: pd.DataFrame, home: str, away: str, date: str
) -> tuple[int, int, pd.Timestamp] | None:
    """FT score + result date, by canon names + date (±1 day for tz skew).

    Orientation-agnostic: the results CSV stores each neutral-venue match in one
    fixed home/away order, but a pre-kickoff snapshot may have been archived with
    the teams swapped (the fixture feed and the CSV don't always agree on which
    side is "home" at a neutral venue). We try the queried orientation first; if
    it misses, we try the swap and **flip the score back into the queried frame**
    so the returned ``(home_score, away_score)`` always means "queried-home's
    goals, queried-away's goals". Every caller's ``outcome = "home" if hs>as_``
    is then correct in the snapshot's own frame — no backwards grading.
    """
    target = pd.Timestamp(date)
    lo, hi = target - pd.Timedelta(days=1), target + pd.Timedelta(days=1)
    chome, caway = canon_team(home), canon_team(away)
    in_window = (df["date"] >= lo) & (df["date"] <= hi)
    ordered = df[in_window & (df["home_team"] == chome) & (df["away_team"] == caway)]
    if not ordered.empty:
        row = ordered.iloc[0]
        return int(row["home_score"]), int(row["away_score"]), row["date"]
    swapped = df[in_window & (df["home_team"] == caway) & (df["away_team"] == chome)]
    if not swapped.empty:
        row = swapped.iloc[0]
        # flip: CSV home==queried-away, so queried-home's goals are away_score
        return int(row["away_score"]), int(row["home_score"]), row["date"]
    return None


def _ninety_minute_score(
    scorers: pd.DataFrame,
    shootouts: pd.DataFrame,
    home_canon: str,
    away_canon: str,
    result_date: pd.Timestamp,
    ft: tuple[int, int],
) -> tuple[int, int] | None:
    """90' score for a knockout match. None = cannot reconstruct honestly."""
    if not shootouts.empty:
        pens = shootouts[
            (shootouts["home_team"] == home_canon)
            & (shootouts["away_team"] == away_canon)
            & (pd.to_datetime(shootouts["date"]) == result_date)
        ]
        if not pens.empty:
            return (0, 0)  # went to penalties => level at 90' (score irrelevant)
    # team col = beneficiary, so own-goal rows count toward the credited
    # team's score — include them.
    rows = scorers[
        (pd.to_datetime(scorers["date"]) == result_date)
        & (scorers["home_team"] == home_canon)
        & (scorers["away_team"] == away_canon)
    ] if not scorers.empty else pd.DataFrame()
    if len(rows) != ft[0] + ft[1]:
        return None  # scorer coverage incomplete — refuse to grade wrong
    in_90 = rows[rows["minute"].fillna(0) <= 90]
    h90 = int((in_90["team"] == home_canon).sum())
    a90 = int((in_90["team"] == away_canon).sum())
    return (h90, a90)


def reconstruct_halftime(
    scorers: pd.DataFrame,
    home_canon: str,
    away_canon: str,
    result_date: pd.Timestamp,
    ft: tuple[int, int],
) -> tuple[int, int] | None:
    """Half-time (<=45') score from scorer minutes; None if coverage incomplete.

    Mirrors ``_ninety_minute_score``'s honesty guard: if the number of scorer
    rows doesn't match the full-time goal count, coverage is incomplete and we
    refuse to fabricate a half-time split. Own goals are credited to the
    beneficiary team (the ``team`` column already encodes that).
    """
    if scorers.empty:
        return None
    rows = scorers[
        (pd.to_datetime(scorers["date"]) == result_date)
        & (scorers["home_team"] == home_canon)
        & (scorers["away_team"] == away_canon)
    ]
    if len(rows) != ft[0] + ft[1]:
        return None  # incomplete scorer coverage — cannot split halves honestly
    first_half = rows[rows["minute"].fillna(999) <= 45]
    hh = int((first_half["team"] == home_canon).sum())
    ah = int((first_half["team"] == away_canon).sum())
    return (hh, ah)


def _brier(probs: tuple[float, float, float], y: tuple[float, float, float]) -> float:
    return float(sum((p - t) ** 2 for p, t in zip(probs, y, strict=True)))


def build_track_record() -> dict[str, Any]:
    archive = dict(read_json_safe(ARCHIVE_JSON, {}))  # type: ignore[arg-type]
    if not archive:
        record = {"n_graded": 0, "matches": []}
        atomic_write_json(TRACK_RECORD_JSON, record)
        return record
    df = load_results_with_live()  # CSV + live Sofascore overlay (same-night grading)
    base = _base_rates(df)
    advance_winners = _load_advance_winners()

    matches: list[dict[str, Any]] = []
    skipped_ko = 0
    for snap in archive.values():
        home = snap.get("home_team")
        away = snap.get("away_team")
        date = snap.get("date")
        if not (home and away and date):
            continue
        # Honesty guard: never grade a snapshot stamped at/after kickoff.
        kickoff = snap.get("kickoff_utc")
        archived = snap.get("archived_at")
        if kickoff and archived:
            if datetime.fromisoformat(str(archived)) >= datetime.fromisoformat(
                str(kickoff)
            ):
                continue
        result = _find_result(df, str(home), str(away), str(date))
        if result is None:
            continue
        hs, as_, _result_date = result
        if snap.get("stage") != "group":
            resolved = resolve_knockout(str(home), str(away), hs, as_, advance_winners)
            if resolved is None:
                skipped_ko += 1  # penalty winner unknown — can't grade honestly
                continue
            outcome, score_str = resolved
        else:
            outcome = "home" if hs > as_ else "draw" if hs == as_ else "away"
            score_str = f"{hs}-{as_}"
        y = (
            1.0 if outcome == "home" else 0.0,
            1.0 if outcome == "draw" else 0.0,
            1.0 if outcome == "away" else 0.0,
        )
        probs = snap["probabilities"]
        p = (float(probs["home"]), float(probs["draw"]), float(probs["away"]))
        pick = max(("home", "draw", "away"), key=lambda k: float(probs[k]))
        entry: dict[str, Any] = {
            "match": f"{home} vs {away}",
            "date": date,
            "stage": snap.get("stage", "group"),
            "score": score_str,
            "outcome": outcome,
            "pick": pick,
            "pick_prob": round(float(probs[pick]), 4),
            "hit": pick == outcome,
            "brier_system": round(_brier(p, y), 4),
            "brier_base": round(_brier(base, y), 4),
        }
        mi = snap.get("market_implied")
        if mi:
            m = (float(mi["home"]), float(mi["draw"]), float(mi["away"]))
            entry["brier_market"] = round(_brier(m, y), 4)
        pm = snap.get("probabilities_pure_model")
        if pm:
            q = (float(pm["home"]), float(pm["draw"]), float(pm["away"]))
            entry["brier_pure_model"] = round(_brier(q, y), 4)
        matches.append(entry)

    matches.sort(key=lambda m: str(m["date"]))
    n = len(matches)
    with_market = [m for m in matches if "brier_market" in m]

    def avg(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [r[key] for r in rows if key in r]
        return round(float(np.mean(vals)), 4) if vals else None

    record = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_graded": n,
        "n_skipped_ko_coverage": skipped_ko,
        "hits": sum(1 for m in matches if m["hit"]),
        "hit_rate": round(sum(1 for m in matches if m["hit"]) / n, 4) if n else None,
        "brier_system": avg(matches, "brier_system"),
        "brier_base": avg(matches, "brier_base"),
        # LIKE-FOR-LIKE: system vs market on the same odds-covered subset.
        "n_with_market": len(with_market),
        "brier_system_market_subset": avg(with_market, "brier_system"),
        "brier_market": avg(with_market, "brier_market"),
        "brier_pure_model": avg(matches, "brier_pure_model"),
        "matches": matches,
    }
    atomic_write_json(TRACK_RECORD_JSON, record)
    return record


def main() -> None:
    r = build_track_record()
    if not r["n_graded"]:
        print("track record: 0 matches graded yet (results not in csv or none played)")
        return
    mk = (
        f" | vs market (n={r['n_with_market']}): system "
        f"{r['brier_system_market_subset']} vs {r['brier_market']}"
        if r.get("brier_market") is not None
        else ""
    )
    print(
        f"track record: {r['hits']}/{r['n_graded']} picks "
        f"({r['hit_rate']:.0%}) | Brier system {r['brier_system']} "
        f"vs base {r['brier_base']}{mk}"
    )
    for m in r["matches"][-5:]:
        print(
            f"  {m['date']} {m['match']} {m['score']} -> {m['outcome']} | "
            f"pick {m['pick']} {m['pick_prob']:.0%} {'HIT' if m['hit'] else 'miss'}"
        )


if __name__ == "__main__":
    main()
