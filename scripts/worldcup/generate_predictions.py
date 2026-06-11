"""Generate the World Cup 2026 prediction artifacts for the dashboard.

Run: python3 -m scripts.worldcup.generate_predictions [--sims N]

Writes to data/worldcup/:
- predictions.json   — per-match lambdas + 1X2 for every fixture with known
                       teams (all 72 group games now; knockouts as they fill
                       in when fixtures.json is updated with real names)
- simulation.json    — Monte Carlo advancement/champion probabilities
- predictions_archive.json — Track-Record-compatible pre-kickoff snapshots
                       (first write per match wins; never overwritten)

Markets are derived at serve time by web/app.py from home_xg/away_xg via
scripts.betting.extended_markets — same pattern as /projections.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from typing import Any

from scripts.worldcup.engine import (
    WorldCupEngine,
    canon_team,
    one_x_two,
    tilt_lambdas,
)
from scripts.worldcup.simulate import (
    DATA_DIR,
    TournamentSimulator,
    country_of_city,
    load_fixtures,
    load_format_spec,
)

PREDICTIONS_JSON = DATA_DIR / "predictions.json"
SIMULATION_JSON = DATA_DIR / "simulation.json"
ARCHIVE_JSON = DATA_DIR / "predictions_archive.json"
MARKET_ODDS_JSON = DATA_DIR / "market_odds.json"
MODEL_METADATA_JSON = DATA_DIR / "model_metadata.json"
AVAILABILITY_JSON = DATA_DIR / "player_availability.json"


def _apply_availability(
    lam_h: float, lam_a: float, av_match: dict[str, Any] | None
) -> tuple[float, float, bool, dict[str, Any] | None]:
    """Fold team-news lambda factors (scripts.worldcup.availability) into the
    MODEL lambdas. Called BEFORE the market blend: the market already prices
    absences, so only the model leg needs the correction — adjusting after
    the blend would double count. No-op when there is no team news."""
    if not av_match:
        return lam_h, lam_a, False, None
    fh = av_match.get("home", {}).get("impact", {})
    fa = av_match.get("away", {}).get("impact", {})
    h_self = float(fh.get("lambda_factor_self", 1.0))
    h_opp = float(fh.get("lambda_factor_opp", 1.0))
    a_self = float(fa.get("lambda_factor_self", 1.0))
    a_opp = float(fa.get("lambda_factor_opp", 1.0))
    # Own attack absences shrink own lambda; opponent's defensive absences
    # grow it (factor_opp travels to the OTHER team's lambda).
    new_h = lam_h * h_self * a_opp
    new_a = lam_a * a_self * h_opp
    adjusted = abs(new_h - lam_h) > 1e-9 or abs(new_a - lam_a) > 1e-9
    summary = None
    if adjusted:
        summary = {
            "home_factor": round(new_h / lam_h, 4),
            "away_factor": round(new_a / lam_a, 4),
            "home_out": [p["name"] for p in av_match.get("home", {}).get("out", [])],
            "away_out": [p["name"] for p in av_match.get("away", {}).get("out", [])],
            "home_doubtful": [
                p["name"] for p in av_match.get("home", {}).get("doubtful", [])
            ],
            "away_doubtful": [
                p["name"] for p in av_match.get("away", {}).get("doubtful", [])
            ],
        }
    return new_h, new_a, adjusted, summary


def _blend_config() -> float | None:
    """Model weight for the market blend, ONLY if its gate passed (the live
    source is model_metadata.json's market_blend section). None = no blend."""
    if not MODEL_METADATA_JSON.exists():
        return None
    blend = json.loads(MODEL_METADATA_JSON.read_text()).get("market_blend") or {}
    if blend.get("gate", {}).get("passed"):
        return float(blend["weight_model"])
    return None

ADVANCE_KEYS = ("group_winner", "group_runner_up", "third_qualified")


def build_match_predictions(
    engine: WorldCupEngine, fixtures: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One prediction per fixture whose two teams are known real teams.

    When the validated market blend is active and odds exist for a match,
    lambdas are tilted so the score grid reproduces the blended 1X2 — every
    derived market stays coherent with the market-informed view."""
    weight_model = _blend_config()
    market: dict[str, Any] = {}
    if weight_model is not None and MARKET_ODDS_JSON.exists():
        market = json.loads(MARKET_ODDS_JSON.read_text())
    availability: dict[str, Any] = {}
    if AVAILABILITY_JSON.exists():
        availability = json.loads(AVAILABILITY_JSON.read_text()).get("matches", {})

    preds: list[dict[str, Any]] = []
    for f in fixtures:
        home, away = str(f["home"]), str(f["away"])
        if canon_team(home) not in engine.ratings or canon_team(away) not in engine.ratings:
            continue  # knockout slot labels ('1A', 'W74') until teams are known
        venue_country = country_of_city(str(f.get("city", "")))
        lam_h, lam_a = engine.lambdas(
            canon_team(home),
            canon_team(away),
            home_at_home=(canon_team(home) == venue_country),
            away_at_home=(canon_team(away) == venue_country),
        )
        lam_h, lam_a, availability_adjusted, av_summary = _apply_availability(
            lam_h, lam_a, availability.get(str(f["match_number"]))
        )
        p_home, p_draw, p_away = one_x_two(lam_h, lam_a)
        # The pure-model leg (post-availability, pre-blend) — archived so the
        # track record can grade model and market independently.
        pure_model = {
            "home": round(p_home, 4),
            "draw": round(p_draw, 4),
            "away": round(p_away, 4),
        }

        market_informed = False
        mk = market.get(str(f["match_number"]))
        if weight_model is not None and isinstance(mk, dict) and mk.get("implied"):
            mi = mk["implied"]
            target = (
                weight_model * p_home + (1 - weight_model) * float(mi["home"]),
                weight_model * p_draw + (1 - weight_model) * float(mi["draw"]),
                weight_model * p_away + (1 - weight_model) * float(mi["away"]),
            )
            lam_h, lam_a = tilt_lambdas(lam_h, lam_a, target)
            p_home, p_draw, p_away = one_x_two(lam_h, lam_a)
            market_informed = True
        date_utc = str(f["date_utc"])
        outcome = max(
            (("home", p_home), ("draw", p_draw), ("away", p_away)),
            key=lambda kv: kv[1],
        )[0]
        preds.append(
            {
                "match_number": f["match_number"],
                "match": f"{home} vs {away}",
                "home_team": home,
                "away_team": away,
                "date": date_utc[:10],
                "time": date_utc[11:16],
                "stage": f["stage"],
                "group": f.get("group"),
                "venue": f.get("venue", ""),
                "city": f.get("city", ""),
                "home_xg": round(lam_h, 3),
                "away_xg": round(lam_a, 3),
                "probabilities": {
                    "home": round(p_home, 4),
                    "draw": round(p_draw, 4),
                    "away": round(p_away, 4),
                },
                "predicted_outcome": outcome,
                "elo_home": round(engine.elo(canon_team(home)), 1),
                "elo_away": round(engine.elo(canon_team(away)), 1),
                "market_informed": market_informed,
                "availability_adjusted": availability_adjusted,
                **({"availability_impact": av_summary} if av_summary else {}),
                "probabilities_pure_model": pure_model,
                "kickoff_utc": f"{date_utc[:10]}T{date_utc[11:16]}:00+00:00",
            }
        )
    return preds


def _merge_archive(preds: list[dict[str, Any]], now_iso: str) -> int:
    """Pre-kickoff snapshots in the Track Record join shape.

    Protocol: LAST write BEFORE kickoff wins — every refresh updates a
    snapshot until its match kicks off, so the graded probabilities are the
    actual closing deployment (odds movement, lineups, team news included).
    From kickoff onward the snapshot is immutable: post-kickoff writes are
    refused here AND grading independently drops any snapshot stamped after
    kickoff (defense in depth — a deleted-and-regenerated archive can never
    mint gradeable 'predictions' for played matches)."""
    from scripts.worldcup.engine import atomic_write_json, read_json_safe

    market: dict[str, Any] = {}
    if MARKET_ODDS_JSON.exists():
        market = dict(read_json_safe(MARKET_ODDS_JSON, {}))  # type: ignore[arg-type]
    archive: dict[str, Any] = dict(
        read_json_safe(ARCHIVE_JSON, {}, quarantine=True)  # type: ignore[arg-type]
    )
    now = datetime.now(UTC)
    written = 0
    for p in preds:
        key = f"{p['match']}_{p['date']}"
        kickoff = datetime.fromisoformat(str(p["kickoff_utc"]))
        if now >= kickoff:
            continue  # match underway or played — snapshot frozen forever
        mk = market.get(str(p["match_number"]), {})
        existing = archive.get(key, {})
        archive[key] = {
            "match_number": p["match_number"],
            "home_team": p["home_team"],
            "away_team": p["away_team"],
            "date": p["date"],
            "kickoff_utc": p["kickoff_utc"],
            "home_xg": p["home_xg"],
            "away_xg": p["away_xg"],
            "probabilities": p["probabilities"],
            "probabilities_pure_model": p.get("probabilities_pure_model"),
            # keep the last known market view if this run's fetch missed it
            "market_implied": mk.get("implied") or existing.get("market_implied"),
            "market_informed": p.get("market_informed", False),
            "stage": p["stage"],
            "first_archived_at": existing.get("first_archived_at", now_iso),
            "archived_at": now_iso,
        }
        written += 1
    atomic_write_json(ARCHIVE_JSON, archive)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sims", type=int, default=10000)
    args = parser.parse_args()

    engine = WorldCupEngine.build()
    fixtures = load_fixtures()
    spec = load_format_spec()

    group_teams = sorted(
        {
            str(t)
            for f in fixtures
            if f["stage"] == "group"
            for t in (f["home"], f["away"])
        }
    )
    missing = [t for t in group_teams if canon_team(t) not in engine.ratings]
    if missing:
        raise SystemExit(f"Unmapped team names (fix FIXTURE_TO_CANON): {missing}")

    now_iso = datetime.now(UTC).isoformat()
    preds = build_match_predictions(engine, fixtures)
    PREDICTIONS_JSON.write_text(
        json.dumps(
            {
                "generated_at": now_iso,
                "fitted_through": engine.fitted_through,
                "count": len(preds),
                "predictions": preds,
            },
            indent=2,
        )
    )

    overrides = {
        int(p["match_number"]): (float(p["home_xg"]), float(p["away_xg"]))
        for p in preds
        if p.get("market_informed") or p.get("availability_adjusted")
    }
    sim = TournamentSimulator(engine, fixtures, spec, lambda_overrides=overrides).run(
        args.sims
    )
    groups: dict[str, list[str]] = {}
    for f in fixtures:
        if f["stage"] == "group":
            groups.setdefault(f["group"], [])
            for t in (str(f["home"]), str(f["away"])):
                if t not in groups[f["group"]]:
                    groups[f["group"]].append(t)

    groups_out: dict[str, list[dict[str, Any]]] = {}
    for letter in sorted(groups):
        rows: list[dict[str, Any]] = []
        for t in groups[letter]:
            s = sim.team_stats[t]
            advance = sum(s[k] for k in ADVANCE_KEYS)
            rows.append(
                {
                    "team": t,
                    "win_group": round(s["group_winner"], 4),
                    "runner_up": round(s["group_runner_up"], 4),
                    "third_qualified": round(s["third_qualified"], 4),
                    "advance": round(advance, 4),
                }
            )
        rows.sort(key=lambda r: -float(r["advance"]))
        groups_out[letter] = rows

    teams_out = [
        {
            "team": t,
            **{k: round(v, 4) for k, v in s.items()},
        }
        for t, s in sorted(
            sim.team_stats.items(),
            key=lambda kv: (-kv[1]["champion"], -kv[1]["reach_final"]),
        )
    ]

    r32_out = []
    for mn in sorted(sim.r32_matchup_probs):
        fixture = next((f for f in fixtures if f["match_number"] == mn), {})
        r32_out.append(
            {
                "match_number": mn,
                "date": str(fixture.get("date_utc", ""))[:10],
                "city": fixture.get("city", ""),
                "slots": f"{fixture.get('home', '')} vs {fixture.get('away', '')}",
                "most_likely": [
                    {"teams": f"{a} vs {b}", "prob": round(p, 4)}
                    for a, b, p in sim.r32_matchup_probs[mn]
                ],
            }
        )

    SIMULATION_JSON.write_text(
        json.dumps(
            {
                "generated_at": now_iso,
                "n_sims": sim.n_sims,
                "fitted_through": engine.fitted_through,
                "teams": teams_out,
                "groups": groups_out,
                "r32_most_likely": r32_out,
            },
            indent=2,
        )
    )

    added = _merge_archive(preds, now_iso)

    print(f"predictions: {len(preds)} matches -> {PREDICTIONS_JSON}")
    print(f"simulation : {sim.n_sims} sims -> {SIMULATION_JSON}")
    print(f"archive    : {added} new snapshots -> {ARCHIVE_JSON}")
    print("\nTop 10 champion probabilities:")
    for row in teams_out[:10]:
        print(
            f"  {row['team']:<16} champion {row['champion']:.3f}  "
            f"final {row['reach_final']:.3f}  SF {row['reach_sf']:.3f}"
        )


if __name__ == "__main__":
    main()
