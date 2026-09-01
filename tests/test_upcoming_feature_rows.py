"""Pipeline-built feature rows for upcoming fixtures (P1, 2026-08-31).

The ML classifiers used to be served from the engine's team cache — the
pre-match state of each team's PREVIOUS game — which reproduced 8/126 training
features exactly. These tests pin the new path: fixtures shaped like
matches.parquet rows (canonical match_id, NaN scores, next matchweek), no
production step-cache writes from ad-hoc builds, the FeatureBuilder preferring a
pre-built row and never overwriting its values, and the ML weight scaled down
whenever a match still falls back to the cache.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import build as fb_mod
from features.build import (
    FeaturePipeline,
    FeaturePlugin,
    FeatureState,
    _fixture_frame,
    upcoming_features_path,
)


def _historical():
    return pd.DataFrame({
        "match_id": ["2026-08-23_Inter_Torino", "2026-08-30_Milan_Lecce"],
        "home_team": ["Inter", "Milan"],
        "away_team": ["Torino", "Lecce"],
        "match_date": pd.to_datetime(["2026-08-23", "2026-08-30"]),
        "season": ["2026-2027", "2026-2027"],
        "matchweek": [1, 2],
        "home_score": [5.0, 1.0],
        "away_score": [0.0, 1.0],
        "league": ["serie_a", "serie_a"],
        "league_name": ["Serie A", "Serie A"],
        "referee": ["Doveri", "Massa"],
    })


def test_fixture_frame_shapes_fixtures_like_played_rows():
    fixtures = pd.DataFrame({
        "home_team": ["Genoa"], "away_team": ["Como"],
        "match_date": pd.to_datetime(["2026-09-04"]), "season": ["2026-2027"],
    })
    combined = _fixture_frame(fixtures, _historical(), "serie_a")

    assert len(combined) == 3
    fx = combined[combined["home_team"] == "Genoa"].iloc[0]
    # canonical key — without it every downstream merge joined NaN to NaN and
    # a 10-fixture build exploded to 1,008,000 rows
    assert fx["match_id"] == "2026-09-04_Genoa_Como"
    assert np.isnan(fx["home_score"]) and np.isnan(fx["away_score"])
    assert fx["league"] == "serie_a" and fx["league_name"] == "Serie A"
    # ppg_pace = league_points / matchweek — the fixture is the NEXT round
    assert fx["matchweek"] == 3
    # every historical column exists on the fixture row (NaN where unknown)
    assert pd.isna(fx["referee"])
    assert list(combined["match_date"]) == sorted(combined["match_date"])


def test_fixture_frame_new_season_starts_at_matchweek_one():
    fixtures = pd.DataFrame({
        "home_team": ["Genoa"], "away_team": ["Como"],
        "match_date": pd.to_datetime(["2027-08-21"]), "season": ["2027-2028"],
    })
    combined = _fixture_frame(fixtures, _historical(), "serie_a")
    assert combined.iloc[-1]["matchweek"] == 1


class _StubPlugin(FeaturePlugin):
    name = "stub_step"
    version = "1.0"

    def apply(self, state: FeatureState) -> FeatureState:
        state.feature_df = state.matches.assign(stub=1.0)
        return state


def _run_pipeline(tmp_path, write_cache):
    cache_dir = tmp_path / f"cache_{int(write_cache)}"
    pipe = FeaturePipeline(cache_dir=cache_dir, league="testleague")
    pipe.register(_StubPlugin())
    state = FeatureState(matches=_historical())
    pipe.build(state, use_cache=False, write_cache=write_cache)
    return sorted(p.name for p in (cache_dir / "testleague").glob("*")) if (cache_dir / "testleague").exists() else []


def test_write_cache_false_never_touches_the_step_cache(tmp_path):
    # the upcoming-fixture build runs on a non-production frame; it used to
    # overwrite the production step cache mid-build (2026-08-31)
    assert _run_pipeline(tmp_path, write_cache=False) == []
    # the guard is specific: the default path still caches
    assert any(n.startswith("stub_step") for n in _run_pipeline(tmp_path, write_cache=True))


def test_upcoming_features_path_is_per_league():
    assert upcoming_features_path("serie_a").name == "upcoming_features_serie_a.parquet"
    assert upcoming_features_path("premier_league").name != upcoming_features_path("serie_a").name


# ---------------------------------------------------------------- engine side

from scripts.prediction import ensemble_prediction_engine as eng  # noqa: E402


def _prebuilt_parquet(tmp_path):
    rows = pd.DataFrame({
        "home_team": ["Genoa", "Roma"], "away_team": ["Como", "Lazio"],
        "match_date": pd.to_datetime(["2026-09-04", "2026-09-05"]),
        "season": ["2026-2027"] * 2, "match_id": ["a", "b"],
        "home_elo": [1480.0, 1610.0], "away_elo": [1520.0, np.nan],
        "home_roll_5_points_mean": [1.2, 2.0],
        "home_score": [np.nan, np.nan],
    })
    p = tmp_path / "upcoming_features_serie_a.parquet"
    rows.to_parquet(p, index=False)
    return p


def test_feature_builder_prefers_prebuilt_row_and_drops_nan(tmp_path, monkeypatch):
    p = _prebuilt_parquet(tmp_path)
    monkeypatch.setattr(fb_mod, "upcoming_features_path", lambda league: p)
    fb = eng.FeatureBuilder(league="serie_a")
    fb._load_prebuilt()

    row = fb._prebuilt_row("Genoa", "Como", "2026-09-04")
    assert row is not None
    assert row["home_elo"] == 1480.0 and row["home_roll_5_points_mean"] == 1.2
    # identity / target columns never leak into the feature row
    assert "match_id" not in row and "home_team" not in row and "home_score" not in row
    # NaN pipeline values are left to the engine's own fillers (referee,
    # weather, lineups) or to the model's imputation — never served as NaN keys
    roma = fb._prebuilt_row("Roma", "Lazio", pd.Timestamp("2026-09-05T18:45:00Z"))
    assert "away_elo" not in roma and roma["home_elo"] == 1610.0
    # wrong date, swapped sides, unknown pair -> cache fallback
    assert fb._prebuilt_row("Genoa", "Como", "2026-09-05") is None
    assert fb._prebuilt_row("Como", "Genoa", "2026-09-04") is None
    assert fb._prebuilt_row("Genoa", "Como", None) is None


def test_feature_builder_without_prebuilt_file_falls_back_quietly(tmp_path, monkeypatch):
    monkeypatch.setattr(fb_mod, "upcoming_features_path", lambda league: tmp_path / "missing.parquet")
    fb = eng.FeatureBuilder(league="serie_a")
    fb._load_prebuilt()
    assert fb._prebuilt == {} and fb.last_feature_source == "cache"
    assert fb._prebuilt_row("Genoa", "Como", "2026-09-04") is None


def _predictor(scale):
    ep = eng.EnsemblePredictor.__new__(eng.EnsemblePredictor)
    ep.weights = dict(eng.ENSEMBLE_WEIGHTS)
    ep._ml_weight_scale = scale
    return ep


_FIVE = {m: {"home": 0.4, "draw": 0.3, "away": 0.3} for m in ("factor", "xg", "ml", "player_xg", "market")}


def test_ml_weight_is_scaled_only_on_cache_fallback():
    base = _predictor(1.0)._get_effective_weights(_FIVE)
    scaled = _predictor(eng.ML_CACHE_FALLBACK_SCALE)._get_effective_weights(_FIVE)

    assert abs(sum(base.values()) - 1.0) < 1e-9 and abs(sum(scaled.values()) - 1.0) < 1e-9
    assert scaled["ml"] < base["ml"]
    # ml is cut by exactly the scale factor RELATIVE to every other method
    for m in ("factor", "xg", "player_xg", "market"):
        assert scaled["ml"] / scaled[m] == pytest.approx(eng.ML_CACHE_FALLBACK_SCALE * base["ml"] / base[m])
    # a predictor that never set the flag behaves like the pipeline path
    ep = eng.EnsemblePredictor.__new__(eng.EnsemblePredictor)
    ep.weights = dict(eng.ENSEMBLE_WEIGHTS)
    assert ep._get_effective_weights(_FIVE) == base


def test_weight_scale_leaves_ensembles_without_ml_untouched():
    three = {m: {"home": 0.4, "draw": 0.3, "away": 0.3} for m in ("factor", "xg", "market")}
    assert _predictor(0.5)._get_effective_weights(three) == _predictor(1.0)._get_effective_weights(three)


# ------------------------------------------------ fill-only invariant (real data)

_FEATURES_PARQUET = fb_mod.DATA_DIR / "features" / "features_serie_a.parquet"


@pytest.mark.skipif(not _FEATURES_PARQUET.exists(), reason="features parquet not present")
def test_pipeline_values_survive_build_match_features(monkeypatch, tmp_path):
    """A pre-built row's training-formula values must reach the model untouched:
    the engine's mirrors (derived, interaction, team cache) only FILL gaps."""
    fb = eng.FeatureBuilder(league="serie_a")
    monkeypatch.setattr(fb_mod, "upcoming_features_path", lambda league: tmp_path / "none.parquet")
    assert fb.load_historical()
    played = pd.read_parquet(_FEATURES_PARQUET)
    played = played[played["season"] == played["season"].max()].dropna(subset=["home_elo", "away_elo"])
    row = played.iloc[-1]
    home, away, date = row["home_team"], row["away_team"], pd.Timestamp(row["match_date"])
    fb._prebuilt = {(home, away, str(date.date())): row}

    out = fb.build_match_features(home, away, match_date=date)
    assert fb.last_feature_source == "pipeline"
    out = out.iloc[0]
    owned = [c for c in row.index if pd.notna(row[c]) and c in out.index and (
        c.startswith(("home_roll_", "away_roll_", "home_elo", "away_elo", "elo_diff",
                      "home_attack", "away_attack", "home_defense", "away_defense",
                      "attack_strength_diff", "defense_strength_diff", "matchup_competitiveness"))
    ) and isinstance(row[c], int | float | np.integer | np.floating)]
    assert len(owned) >= 30
    changed = [c for c in owned if not np.isclose(float(out[c]), float(row[c]), equal_nan=True)]
    assert changed == [], f"engine overwrote pipeline values: {changed[:10]}"

    # and the same fixture WITHOUT a pre-built row reports the cache path
    fb._prebuilt = {}
    fb.build_match_features(home, away, match_date=date)
    assert fb.last_feature_source == "cache"


def test_build_upcoming_features_returns_only_the_fixture_rows(monkeypatch):
    """Exercises the whole function with the 58-step pipeline stubbed out —
    the tail that selects fixture rows out of the combined frame once referenced
    a name that no longer existed (caught by ruff after a green suite)."""
    calls = {}

    def fake_pipeline(matches, season, use_cache, league, write_cache):
        calls.update(use_cache=use_cache, write_cache=write_cache, n=len(matches))
        return matches.assign(home_elo=1500.0)

    monkeypatch.setattr(fb_mod, "_build_features_for_matches", fake_pipeline)
    fixtures = pd.DataFrame({
        "home_team": ["Genoa", "Roma"], "away_team": ["Como", "Lazio"],
        "match_date": ["2026-09-04", "2026-09-05"], "season": ["2026-2027"] * 2,
    })
    out = fb_mod.build_upcoming_features(fixtures, league="serie_a", historical=_historical())

    assert calls == {"use_cache": False, "write_cache": False, "n": 4}
    assert len(out) == 2
    assert sorted(out["match_id"]) == ["2026-09-04_Genoa_Como", "2026-09-05_Roma_Lazio"]
    assert out["home_score"].isna().all() and (out["home_elo"] == 1500.0).all()
