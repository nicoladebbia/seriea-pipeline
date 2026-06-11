"""Monte Carlo simulation of the 2026 World Cup (48 teams, 12 groups).

Consumes data/worldcup/fixtures.json + format_spec.json and a built
WorldCupEngine. Group ranking implements the 2026 regulations (Article 13):
head-to-head points/GD/goals among tied teams FIRST (with reapplication),
then overall GD/goals — the first World Cup since 1970 not to lead with
overall goal difference. Team-conduct points are unmodelable pre-tournament;
the FIFA-ranking criterion is proxied by Elo (documented approximation).

Third-place allocation uses FIFA's exact Annex C table (all 495 combinations
of 8-from-12 advancing groups) from format_spec.json, with a backtracking
matching as a defensive fallback.

Knockouts: 90' Poisson, extra time at 1/3 rate, penalties as a coin flip
(penalty skill is ~noise; documented). The third-place playoff is not
simulated — it affects no advancement statistic.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.worldcup.engine import WorldCupEngine, canon_team

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "worldcup"
FIXTURES_JSON = DATA_DIR / "fixtures.json"
FORMAT_SPEC_JSON = DATA_DIR / "format_spec.json"

EXTRA_TIME_FACTOR = 1.0 / 3.0  # 30 min of play at the 90-min scoring rate

_MEX_CITIES = {"mexico city", "guadalajara", "monterrey"}
_CAN_CITIES = {"toronto", "vancouver"}
_US_CITIES = {
    "atlanta", "boston", "dallas", "houston", "kansas city", "los angeles",
    "miami", "new york/new jersey", "philadelphia", "san francisco bay area",
    "seattle",
}

KO_ROUND_ORDER = [
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "final",
]

STAGE_COUNTER_KEY = {
    "round_of_16": "reach_r16",
    "quarter_final": "reach_qf",
    "semi_final": "reach_sf",
    "final": "reach_final",
}


def country_of_city(city: str) -> str:
    """Canonical host-country name for a FIFA host-city designation.

    Raises on unknown cities rather than silently defaulting — a feed-side
    city rename must fail loudly, not strip a host's advantage unnoticed.
    """
    c = city.strip().lower()
    if c in _MEX_CITIES:
        return "Mexico"
    if c in _CAN_CITIES:
        return "Canada"
    if c in _US_CITIES:
        return "United States"
    raise ValueError(f"Unknown 2026 host city: {city!r} — update simulate.py city sets")


def load_fixtures(path: Path = FIXTURES_JSON) -> list[dict[str, Any]]:
    with open(path) as f:
        fixtures: list[dict[str, Any]] = json.load(f)
    return fixtures


def load_format_spec(path: Path = FORMAT_SPEC_JSON) -> dict[str, Any]:
    with open(path) as f:
        spec: dict[str, Any] = json.load(f)
    return spec


@dataclass
class SlotRef:
    """A parsed bracket slot: group rank ('1A'), third-place pool ('3ABCDF'),
    or match-winner reference ('W74')."""

    kind: str  # 'rank' | 'third' | 'winner'
    rank: int = 0
    groups: tuple[str, ...] = ()
    match_number: int = 0


_RANK_RE = re.compile(r"^([12])([A-L])$")
_WINNER_RE = re.compile(r"^W(\d+)$|^Winner\s*(?:of\s*)?(?:Match\s*)?(\d+)$", re.I)


def parse_slot(label: str) -> SlotRef:
    """Parse official bracket slot labels in their common spellings."""
    s = label.strip()
    m = _RANK_RE.match(s)
    if m:
        return SlotRef(kind="rank", rank=int(m.group(1)), groups=(m.group(2),))
    m = _WINNER_RE.match(s)
    if m:
        return SlotRef(kind="winner", match_number=int(m.group(1) or m.group(2)))
    groups = tuple(re.findall(r"[A-L]", s.upper().replace("3RD", "")))
    if s.upper().lstrip().startswith("3") and groups:
        return SlotRef(kind="third", rank=3, groups=groups)
    raise ValueError(f"Unparseable bracket slot label: {label!r}")


@dataclass
class SimResult:
    n_sims: int
    team_stats: dict[str, dict[str, float]]
    r32_matchup_probs: dict[int, list[tuple[str, str, float]]]


def _rank_group_2026(
    teams: list[str],
    pts: dict[str, int],
    gd: dict[str, int],
    gf: dict[str, int],
    h2h: dict[tuple[str, str], tuple[int, int]],
    elo: dict[str, float],
    rng: np.random.Generator,
) -> list[str]:
    """2026 group ranking: points, then head-to-head (pts/GD/goals) among the
    tied teams with reapplication, then overall GD, overall goals, then Elo
    as the FIFA-ranking proxy (conduct points unmodelable), then random."""

    def h2h_triple(tied: list[str]) -> dict[str, tuple[int, int, int]]:
        h_pts = {t: 0 for t in tied}
        h_gd = {t: 0 for t in tied}
        h_gf = {t: 0 for t in tied}
        for a in tied:
            for b in tied:
                if a != b and (a, b) in h2h:
                    ga, gb = h2h[(a, b)]
                    h_gf[a] += ga
                    h_gd[a] += ga - gb
                    if ga > gb:
                        h_pts[a] += 3
                    elif ga == gb:
                        h_pts[a] += 1
        return {t: (h_pts[t], h_gd[t], h_gf[t]) for t in tied}

    def order_tied(tied: list[str]) -> list[str]:
        if len(tied) == 1:
            return tied
        triple = h2h_triple(tied)
        distinct = sorted({triple[t] for t in tied}, reverse=True)
        if len(distinct) > 1:
            ordered: list[str] = []
            for key in distinct:
                subset = [t for t in tied if triple[t] == key]
                # Reapplication: h2h criteria re-run on the still-level subset.
                ordered.extend(order_tied(subset) if len(subset) < len(tied) else subset)
            return ordered
        # Fully level on head-to-head: overall GD, overall goals, Elo, random.
        jitter = {t: rng.random() for t in tied}
        return sorted(
            tied, key=lambda t: (gd[t], gf[t], elo[t], jitter[t]), reverse=True
        )

    result: list[str] = []
    for points in sorted({pts[t] for t in teams}, reverse=True):
        cluster = [t for t in teams if pts[t] == points]
        result.extend(order_tied(cluster))
    return result


def _assign_thirds_fallback(
    third_slots: list[tuple[int, tuple[str, ...]]],
    qualified_thirds: dict[str, str],
) -> dict[int, str]:
    """Backtracking perfect matching (slot -> team), defensive fallback only."""
    slots = sorted(
        third_slots,
        key=lambda s: len([g for g in s[1] if g in qualified_thirds]),
    )
    assignment: dict[int, str] = {}
    used: set[str] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(slots):
            return True
        match_number, allowed = slots[idx]
        for g in allowed:
            if g in qualified_thirds and g not in used:
                used.add(g)
                assignment[match_number] = qualified_thirds[g]
                if backtrack(idx + 1):
                    return True
                used.discard(g)
                del assignment[match_number]
        return False

    if not backtrack(0):
        # No constrained matching: assign in slot order, ignoring pools.
        ordered = list(qualified_thirds.values())
        return {mn: ordered[i] for i, (mn, _) in enumerate(slots)}
    return assignment


class TournamentSimulator:
    def __init__(
        self,
        engine: WorldCupEngine,
        fixtures: list[dict[str, Any]],
        spec: dict[str, Any],
        rng: np.random.Generator | None = None,
        lambda_overrides: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        """lambda_overrides: match_number -> (lam_home, lam_away), used to
        keep the simulation coherent with market-tilted group-game lambdas
        (knockouts have no odds and stay pure model)."""
        self.engine = engine
        self.spec = spec
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.lambda_overrides = lambda_overrides or {}

        self.group_fixtures = [f for f in fixtures if f["stage"] == "group"]
        self.groups: dict[str, list[str]] = defaultdict(list)
        for f in self.group_fixtures:
            for t in (f["home"], f["away"]):
                if t not in self.groups[f["group"]]:
                    self.groups[f["group"]].append(t)

        all_teams = [t for group in self.groups.values() for t in group]
        self.team_elo = {t: engine.elo(canon_team(t)) for t in all_teams}

        # Precompute lambdas per group game (host advantage only in own country).
        self.group_games: list[dict[str, Any]] = []
        for f in self.group_fixtures:
            override = self.lambda_overrides.get(int(f["match_number"]))
            if override is not None:
                lam_h, lam_a = override
            else:
                venue_country = country_of_city(f.get("city", ""))
                lam_h, lam_a = engine.lambdas(
                    canon_team(f["home"]),
                    canon_team(f["away"]),
                    home_at_home=(canon_team(f["home"]) == venue_country),
                    away_at_home=(canon_team(f["away"]) == venue_country),
                )
            self.group_games.append({**f, "lam_home": lam_h, "lam_away": lam_a})

        self.r32_pairings: list[dict[str, Any]] = spec["r32_pairings"]
        self.fixture_by_number = {f["match_number"]: f for f in fixtures}
        self.ko_progression: dict[str, list[dict[str, Any]]] = {
            stage: [f for f in fixtures if f["stage"] == stage]
            for stage in KO_ROUND_ORDER[1:]
        }

        # FIFA Annex C: exact third-place allocation table.
        alloc_spec = spec["third_place_allocation"]
        self.third_alloc: dict[frozenset[str], dict[int, str]] = {}
        slot_to_mn: dict[str, int] = alloc_spec["slot_to_match_number"]
        for combo in alloc_spec["combinations"]:
            key = frozenset(combo["advancing_third_place_groups"])
            self.third_alloc[key] = {
                slot_to_mn[slot]: group.lstrip("3")
                for slot, group in combo["allocation"].items()
            }

    # ---------------------------------------------------------------- sim --

    def _sim_group_stage(
        self, scores: dict[int, tuple[int, int]]
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        """Returns (group -> ranked teams, group -> third-placed team)."""
        rankings: dict[str, list[str]] = {}
        thirds: dict[str, str] = {}
        for letter, teams in self.groups.items():
            pts = {t: 0 for t in teams}
            gd = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            h2h: dict[tuple[str, str], tuple[int, int]] = {}
            for g in self.group_games:
                if g["group"] != letter:
                    continue
                hs, as_ = scores[g["match_number"]]
                h, a = g["home"], g["away"]
                gf[h] += hs
                gf[a] += as_
                gd[h] += hs - as_
                gd[a] += as_ - hs
                if hs > as_:
                    pts[h] += 3
                elif hs < as_:
                    pts[a] += 3
                else:
                    pts[h] += 1
                    pts[a] += 1
                h2h[(h, a)] = (hs, as_)
                h2h[(a, h)] = (as_, hs)
            ranked = _rank_group_2026(
                teams, pts, gd, gf, h2h, self.team_elo, self.rng
            )
            rankings[letter] = ranked
            thirds[letter] = ranked[2]
        return rankings, thirds

    def _pick_best_thirds(
        self,
        thirds: dict[str, str],
        scores: dict[int, tuple[int, int]],
    ) -> dict[str, str]:
        """Rank the 12 third-placed teams (2026 rules: points, GD, goals,
        conduct [unmodeled], FIFA ranking [Elo proxy]); return best 8."""
        stats: dict[str, tuple[int, int, int, float, float]] = {}
        for letter, team in thirds.items():
            pts = gd = gf = 0
            for g in self.group_games:
                if g["group"] != letter:
                    continue
                hs, as_ = scores[g["match_number"]]
                if g["home"] == team:
                    f_, a_ = hs, as_
                elif g["away"] == team:
                    f_, a_ = as_, hs
                else:
                    continue
                gf += f_
                gd += f_ - a_
                if f_ > a_:
                    pts += 3
                elif f_ == a_:
                    pts += 1
            stats[letter] = (
                pts,
                gd,
                gf,
                self.team_elo[team],
                float(self.rng.random()),
            )
        best8 = sorted(stats, key=lambda k: stats[k], reverse=True)[:8]
        return {g: thirds[g] for g in best8}

    def _sim_knockout_match(self, home: str, away: str, city: str) -> str:
        venue_country = country_of_city(city)
        lam_h, lam_a = self.engine.lambdas(
            canon_team(home),
            canon_team(away),
            home_at_home=(canon_team(home) == venue_country),
            away_at_home=(canon_team(away) == venue_country),
        )
        hs = int(self.rng.poisson(lam_h))
        as_ = int(self.rng.poisson(lam_a))
        if hs != as_:
            return home if hs > as_ else away
        et_h = int(self.rng.poisson(lam_h * EXTRA_TIME_FACTOR))
        et_a = int(self.rng.poisson(lam_a * EXTRA_TIME_FACTOR))
        if et_h != et_a:
            return home if et_h > et_a else away
        return home if self.rng.random() < 0.5 else away

    def run(self, n_sims: int = 10000) -> SimResult:
        teams = [t for group in self.groups.values() for t in group]
        counters: dict[str, Counter[str]] = {
            key: Counter()
            for key in (
                "group_winner",
                "group_runner_up",
                "third_qualified",
                "reach_r32",
                "reach_r16",
                "reach_qf",
                "reach_sf",
                "reach_final",
                "champion",
            )
        }
        r32_counts: dict[int, Counter[tuple[str, str]]] = defaultdict(Counter)

        # Pre-sample all group-game scores, vectorized across sims.
        sampled: dict[int, np.ndarray] = {}
        for g in self.group_games:
            sampled[g["match_number"]] = np.column_stack(
                [
                    self.rng.poisson(g["lam_home"], size=n_sims),
                    self.rng.poisson(g["lam_away"], size=n_sims),
                ]
            )

        for s in range(n_sims):
            scores = {
                mn: (int(arr[s, 0]), int(arr[s, 1])) for mn, arr in sampled.items()
            }
            rankings, thirds = self._sim_group_stage(scores)
            qualified_thirds = self._pick_best_thirds(thirds, scores)

            for ranked in rankings.values():
                counters["group_winner"][ranked[0]] += 1
                counters["group_runner_up"][ranked[1]] += 1
            for team in qualified_thirds.values():
                counters["third_qualified"][team] += 1

            # Resolve R32 entrants: rank slots directly, thirds via Annex C.
            alloc = self.third_alloc.get(frozenset(qualified_thirds))
            r32_entrants: dict[int, dict[str, str]] = {}
            third_slots: list[tuple[int, tuple[str, ...]]] = []
            for pairing in self.r32_pairings:
                mn = pairing["match_number"]
                entry: dict[str, str] = {}
                for side in ("home", "away"):
                    ref = parse_slot(pairing[f"{side}_slot"])
                    if ref.kind == "rank":
                        entry[side] = rankings[ref.groups[0]][ref.rank - 1]
                    elif ref.kind == "third":
                        third_slots.append((mn, ref.groups))
                        entry[side] = ""
                    else:
                        raise ValueError(
                            f"R32 slot cannot be a match-winner ref: {pairing}"
                        )
                r32_entrants[mn] = entry

            assignment: dict[int, str]
            if alloc is not None:
                assignment = {
                    mn: qualified_thirds[group] for mn, group in alloc.items()
                }
            else:  # defensive: Annex C covers all 495 combinations
                assignment = _assign_thirds_fallback(third_slots, qualified_thirds)
            for mn, team in assignment.items():
                side = "home" if r32_entrants[mn]["home"] == "" else "away"
                r32_entrants[mn][side] = team

            # Play the knockout rounds.
            winners: dict[int, str] = {}
            for mn, entry in sorted(r32_entrants.items()):
                counters["reach_r32"][entry["home"]] += 1
                counters["reach_r32"][entry["away"]] += 1
                pair = (entry["home"], entry["away"])
                lo, hi = sorted(pair)
                r32_counts[mn][(lo, hi)] += 1
                city = self.fixture_by_number.get(mn, {}).get("city", "")
                winners[mn] = self._sim_knockout_match(*pair, city)

            for stage in KO_ROUND_ORDER[1:]:
                for f in self.ko_progression[stage]:
                    refs = (parse_slot(f["home"]), parse_slot(f["away"]))
                    if any(r.kind != "winner" for r in refs):
                        raise ValueError(
                            f"Knockout fixture {f['match_number']} has "
                            f"non-winner slots: {f['home']!r} vs {f['away']!r}"
                        )
                    th = winners[refs[0].match_number]
                    ta = winners[refs[1].match_number]
                    key = STAGE_COUNTER_KEY[stage]
                    counters[key][th] += 1
                    counters[key][ta] += 1
                    winners[f["match_number"]] = self._sim_knockout_match(
                        th, ta, f.get("city", "")
                    )
                    if stage == "final":
                        counters["champion"][winners[f["match_number"]]] += 1

        team_stats = {
            t: {key: counters[key][t] / n_sims for key in counters} for t in teams
        }
        r32_matchup_probs = {
            mn: [
                (a, b, count / n_sims)
                for (a, b), count in counter.most_common(5)
            ]
            for mn, counter in r32_counts.items()
        }
        return SimResult(
            n_sims=n_sims,
            team_stats=team_stats,
            r32_matchup_probs=r32_matchup_probs,
        )
