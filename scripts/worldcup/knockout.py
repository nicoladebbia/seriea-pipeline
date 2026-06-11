"""Knockout bracket auto-fill: resolve fixtures.json slot labels from REAL results.

Replaces the manual flow ("edit fixtures.json by hand as rounds resolve").
Each refresh-loop run, AFTER the results CSV re-download and BEFORE
generate_predictions:

- a group whose 6 matches all appear in results.csv is ranked with the exact
  2026 rules (simulate._rank_group_2026: points → head-to-head with
  reapplication → overall GD/goals → Elo proxy), and its 1X/2X slots fill;
- when ALL 12 groups are final, the 8 best thirds are ranked (points, GD,
  goals, Elo proxy) and placed via the official FIFA Annex C allocation
  table (format_spec.json, all 495 combinations);
- a W## slot fills once match ## has a result — level FT scores (results.csv
  is ET-inclusive) fall through to shootouts.csv for the winner.

Safety: a slot fills ONLY from complete data (partial groups never rank);
fills are idempotent (the original label is preserved in slot_home/slot_away
and a filled side is never touched again); fixtures.json is written
atomically and only when something changed. Team names stay in FIFA display
space (what squads.json/the UI join on); canonization happens only at
results/Elo lookup, mirroring the simulator.

Run: python3 -m scripts.worldcup.knockout   (wired into scripts.worldcup.refresh)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from scripts.worldcup.engine import (
    DATA_DIR,
    atomic_write_json,
    canon_team,
    read_json_safe,
)
from scripts.worldcup.simulate import (
    FIXTURES_JSON,
    SlotRef,
    _rank_group_2026,
    load_format_spec,
    load_third_alloc,
    parse_slot,
)

SHOOTOUTS_CSV = DATA_DIR / "international_shootouts.csv"

ResultFn = Callable[[str, str, str], "tuple[int, int] | None"]


def _is_slot(label: str) -> SlotRef | None:
    """SlotRef if the label is a bracket slot, None if it's a real team."""
    try:
        return parse_slot(label)
    except ValueError:
        return None


def group_tables(group_fixtures: list[dict[str, Any]],
                 result_of: ResultFn) -> dict[str, dict[str, Any]]:
    """Per group: standings accumulators from REAL results (display names).

    A group is complete only when every one of its matches has a result —
    partial groups must never rank (a 5/6 table can invert on the last game).
    """
    tables: dict[str, dict[str, Any]] = {}
    for f in group_fixtures:
        letter = f["group"]
        t = tables.setdefault(letter, {
            "teams": [], "pts": {}, "gd": {}, "gf": {},
            "h2h": {}, "n_played": 0, "n_fixtures": 0,
        })
        t["n_fixtures"] += 1
        for team in (f["home"], f["away"]):
            if team not in t["teams"]:
                t["teams"].append(team)
                t["pts"][team] = t["gd"][team] = t["gf"][team] = 0
        res = result_of(f["home"], f["away"], str(f.get("date_utc", ""))[:10])
        if res is None:
            continue
        hs, as_ = res
        h, a = f["home"], f["away"]
        t["n_played"] += 1
        t["gf"][h] += hs
        t["gf"][a] += as_
        t["gd"][h] += hs - as_
        t["gd"][a] += as_ - hs
        if hs > as_:
            t["pts"][h] += 3
        elif hs < as_:
            t["pts"][a] += 3
        else:
            t["pts"][h] += 1
            t["pts"][a] += 1
        t["h2h"][(h, a)] = (hs, as_)
        t["h2h"][(a, h)] = (as_, hs)
    for t in tables.values():
        t["complete"] = t["n_played"] == t["n_fixtures"] and t["n_fixtures"] > 0
    return tables


def rank_complete_groups(tables: dict[str, dict[str, Any]],
                         elo: dict[str, float]) -> dict[str, list[str]]:
    """Final ranking for every complete group, exact 2026 rules."""
    import numpy as np

    # Seeded rng: its jitter only breaks ties still level after Elo — a dead
    # branch on real floats — but determinism keeps reruns idempotent.
    rng = np.random.default_rng(2026)
    return {
        letter: _rank_group_2026(t["teams"], t["pts"], t["gd"], t["gf"],
                                 t["h2h"], elo, rng)
        for letter, t in sorted(tables.items()) if t["complete"]
    }


def best_third_letters(rankings: dict[str, list[str]],
                       tables: dict[str, dict[str, Any]],
                       elo: dict[str, float]) -> list[str]:
    """The 8 qualified third-placed groups (2026: points, GD, goals, Elo
    proxy for the FIFA-ranking step; conduct points unmodelable)."""
    stats = {}
    for letter, ranked in rankings.items():
        third = ranked[2]
        t = tables[letter]
        stats[letter] = (t["pts"][third], t["gd"][third], t["gf"][third],
                         elo[third])
    return sorted(stats, key=lambda k: stats[k], reverse=True)[:8]


def make_match_resolver(fixture_by_number: dict[int, dict[str, Any]],
                        result_of: ResultFn,
                        shootout_winner: Callable[[str, str, str], str | None],
                        ) -> Callable[[int], tuple[str, str] | None]:
    """W##/L## resolver — (winner, loser) of a feeder match, or None until
    the match is filled, played, and decided."""

    def decided_of(match_number: int) -> tuple[str, str] | None:
        f = fixture_by_number.get(match_number)
        if f is None:
            return None
        home, away = str(f["home"]), str(f["away"])
        if _is_slot(home) or _is_slot(away):
            return None  # feeder match itself not filled yet
        date = str(f.get("date_utc", ""))[:10]
        res = result_of(home, away, date)
        if res is None:
            return None
        hs, as_ = res
        if hs != as_:  # results.csv scores are ET-inclusive — a lead decides
            return (home, away) if hs > as_ else (away, home)
        w_canon = shootout_winner(home, away, date)
        if w_canon is None:
            return None  # level + no shootout row yet — refuse to guess
        rev = {canon_team(home): home, canon_team(away): away}
        winner = rev.get(w_canon)
        if winner is None:
            return None
        return (winner, away if winner == home else home)

    return decided_of


def resolve_slots(fixtures: list[dict[str, Any]],
                  rankings: dict[str, list[str]],
                  third_by_match: dict[int, str] | None,
                  decided_of: Callable[[int], tuple[str, str] | None],
                  ) -> list[dict[str, Any]]:
    """Fill every resolvable slot in place; return the change list."""
    changes: list[dict[str, Any]] = []
    for f in fixtures:
        if f.get("stage") == "group":
            continue
        for side in ("home", "away"):
            if f.get(f"slot_{side}"):
                continue  # already filled on a previous run — frozen
            label = str(f[side])
            ref = _is_slot(label)
            if ref is None:
                continue  # already a real team name
            team: str | None = None
            if ref.kind == "rank" and ref.groups:
                ranked = rankings.get(ref.groups[0])
                if ranked:
                    team = ranked[(ref.rank or 1) - 1]
            elif ref.kind == "third" and third_by_match is not None:
                letter = third_by_match.get(int(f["match_number"]))
                if letter and letter in (ref.groups or ()):
                    team = rankings[letter][2]
                elif letter:
                    # Annex C says a group this slot can't host — config bug,
                    # surface it loudly instead of silently mis-seeding.
                    changes.append({"match_number": f["match_number"],
                                    "side": side, "slot": label,
                                    "error": f"Annex C letter {letter} not in {label}"})
            elif ref.kind in ("winner", "loser") and ref.match_number:
                decided = decided_of(int(ref.match_number))
                if decided:
                    team = decided[0] if ref.kind == "winner" else decided[1]
            if team:
                f[f"slot_{side}"] = label
                f[side] = team
                changes.append({"match_number": f["match_number"],
                                "side": side, "slot": label, "team": team})
    return changes


def _real_result_lookup() -> ResultFn:
    from scripts.worldcup.engine import load_results
    from scripts.worldcup.grading import _find_result

    df = load_results()

    def result_of(home: str, away: str, date: str) -> tuple[int, int] | None:
        res = _find_result(df, home, away, date)
        return None if res is None else (res[0], res[1])

    return result_of


def _real_shootout_lookup() -> Callable[[str, str, str], str | None]:
    import pandas as pd

    df = pd.read_csv(SHOOTOUTS_CSV) if SHOOTOUTS_CSV.exists() else None

    def shootout_winner(home: str, away: str, date: str) -> str | None:
        if df is None or df.empty:
            return None
        target = pd.Timestamp(date)
        dates = pd.to_datetime(df["date"])
        rows = df[(df["home_team"] == canon_team(home))
                  & (df["away_team"] == canon_team(away))
                  & (dates >= target - pd.Timedelta(days=1))
                  & (dates <= target + pd.Timedelta(days=1))]
        return None if rows.empty else str(rows.iloc[0]["winner"])

    return shootout_winner


def fill_knockout_slots(write: bool = True) -> dict[str, Any]:
    """Resolve every slot the data allows; atomic write only on change."""
    fixtures: list[dict[str, Any]] = list(read_json_safe(FIXTURES_JSON, []))  # type: ignore[arg-type]
    ko = [f for f in fixtures if f.get("stage") != "group"]
    open_slots = [f for f in ko
                  for side in ("home", "away")
                  if not f.get(f"slot_{side}") and _is_slot(str(f[side]))]
    if not fixtures or not open_slots:
        return {"changes": [], "groups_complete": [], "open_slots": 0,
                "note": "nothing to fill"}

    result_of = _real_result_lookup()
    group_fx = [f for f in fixtures if f.get("stage") == "group"]
    tables = group_tables(group_fx, result_of)
    complete = sorted(letter for letter, t in tables.items() if t["complete"])

    rankings: dict[str, list[str]] = {}
    third_by_match: dict[int, str] | None = None
    if complete:
        # Engine build (Elo tiebreak proxy) only when a group can actually rank.
        from scripts.worldcup.engine import WorldCupEngine

        engine = WorldCupEngine.build()
        elo = {t: engine.elo(canon_team(t))
               for tab in tables.values() for t in tab["teams"]}
        rankings = rank_complete_groups(tables, elo)
        if len(rankings) == 12:
            best8 = best_third_letters(rankings, tables, elo)
            third_by_match = load_third_alloc(load_format_spec()).get(frozenset(best8))

    decided_of = make_match_resolver({int(f["match_number"]): f for f in fixtures},
                                     result_of, _real_shootout_lookup())
    changes = resolve_slots(fixtures, rankings, third_by_match, decided_of)
    filled = [c for c in changes if "team" in c]
    if filled and write:
        atomic_write_json(FIXTURES_JSON, fixtures)
    return {
        "changes": changes,
        "groups_complete": complete,
        "open_slots": len(open_slots) - len(filled),
    }


def main() -> None:
    report = fill_knockout_slots()
    for c in report["changes"]:
        if "team" in c:
            print(f"match {c['match_number']}: {c['side']} {c['slot']} -> {c['team']}")
        else:
            print(f"match {c['match_number']}: ERROR {c.get('error')}")
    print(f"knockout: groups complete {report['groups_complete'] or 'none'}, "
          f"{len([c for c in report['changes'] if 'team' in c])} slots filled, "
          f"{report['open_slots']} still open")


if __name__ == "__main__":
    main()
