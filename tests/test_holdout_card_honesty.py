"""The per-league model card must say how big its hold-out was.

`train_optimized` holds out the newest season and writes that season's metrics
into the model card via `save_model`. In August the newest season is a handful
of matches — measured 2026-08-25, Serie A held out 2026-2027 with ten — and an
accuracy over ten matches has a standard error of 0.158. Written bare, that
number is indistinguishable in the card from the same number measured over a
full 380-match season, which defeats this repo's "trust the metadata, never the
markdown" rule at its root.

Scope note, deliberately narrow: `train_optimized` loads its features from disk
and runs Optuna, so there is no cheap injection point for an end-to-end test.
What is pinned here is the CARD CONTRACT (a real round-trip through save_model)
plus a source assertion that the overfit check is gated. The training loop
itself is not exercised.
"""

import json

import pytest

import ml.persistence as persistence
from ml.evaluation import MIN_GATE_TEST_MATCHES
from ml.persistence import save_model


class _FakeModel:
    """Minimal stand-in for a booster: save() writes a file, that is all."""

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("stub")

    def get_feature_importance(self):
        return {"home_elo": 1.0}


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    # The module-level name is what save_model resolves, so patch it there —
    # patching config.settings.MODELS_DIR would not redirect this write.
    monkeypatch.setattr(persistence, "MODELS_DIR", tmp_path)
    return tmp_path


def _card(accuracy: float, n_holdout: int, season: str) -> dict:
    """A card exactly as train_optimized now builds it."""
    return {
        "accuracy": accuracy,
        "log_loss": 1.02,
        "holdout_season": season,
        "n_holdout": n_holdout,
        "holdout_reliable": n_holdout >= MIN_GATE_TEST_MATCHES,
    }


def _read_card(models_dir, variant: str) -> dict:
    with open(models_dir / variant / "catboost_metadata.json") as f:
        return json.load(f)["metrics"]


def test_a_ten_match_holdout_is_marked_unreliable_in_the_card(models_dir):
    save_model(
        _FakeModel(), "serie_a", "catboost", ["home_elo"],
        _card(0.400, 10, "2026-2027"),
    )
    card = _read_card(models_dir, "serie_a")

    assert card["n_holdout"] == 10
    assert card["holdout_reliable"] is False
    assert card["holdout_season"] == "2026-2027"


def test_a_full_season_holdout_is_marked_reliable(models_dir):
    save_model(
        _FakeModel(), "premier_league", "catboost", ["home_elo"],
        _card(0.400, 380, "2025-2026"),
    )
    card = _read_card(models_dir, "premier_league")

    assert card["n_holdout"] == 380
    assert card["holdout_reliable"] is True


def test_without_the_sample_size_the_two_cards_are_indistinguishable(models_dir):
    """The true positive: this is the defect the extra keys exist to fix.

    Both runs score 0.400 — one over ten matches, one over a season. On the
    float-only metrics compute_metrics returns, nothing tells them apart.
    """
    tiny = _card(0.400, 10, "2026-2027")
    full = _card(0.400, 380, "2025-2026")

    float_only = {"accuracy", "log_loss"}
    assert {k: tiny[k] for k in float_only} == {k: full[k] for k in float_only}, (
        "precondition broken: the two cards must agree on the plain metrics, "
        "otherwise this test is not exercising the ambiguity it claims to"
    )
    assert tiny["holdout_reliable"] != full["holdout_reliable"]


def test_the_extra_keys_do_not_break_an_existing_card_reader(models_dir):
    """ensemble_prediction_engine reads feature_names out of this same file."""
    save_model(
        _FakeModel(), "premier_league", "catboost", ["home_elo", "away_elo"],
        _card(0.400, 10, "2026-2027"),
    )
    with open(models_dir / "premier_league" / "catboost_metadata.json") as f:
        meta = json.load(f)

    assert meta.get("feature_names", []) == ["home_elo", "away_elo"]
    assert meta["model_type"] == "catboost"


def test_the_overfit_check_is_gated_on_a_measurable_holdout():
    """Source pin, not a behaviour test — see the module docstring.

    The OVERFIT FLAG compares hold-out log-loss against CV log-loss with a
    0.02 tolerance. Against ten matches that comparison is noise either way:
    it will both fire spuriously and stay silent on real overfitting.
    """
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "ml" / "training.py"
    ).read_text()

    assert "holdout_is_measurable = n_holdout >= MIN_GATE_TEST_MATCHES" in src
    assert (
        'if holdout_is_measurable and cv_key in results and "log_loss" in results[cv_key]:'
        in src
    ), "the overfit check must not run against an undersized hold-out"


def test_the_trainer_actually_puts_the_sample_size_on_the_card_it_saves():
    """Source pin closing the loop the round-trip tests above cannot reach.

    Tests 1-3 build the card themselves, so they prove save_model PERSISTS the
    keys, not that train_optimized SETS them. This asserts the link.
    """
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1] / "ml" / "training.py"
    ).read_text()

    for key in ("holdout_season", "n_holdout", "holdout_reliable"):
        assert f'"{key}": ' in src, f"train_optimized must stamp {key} on the card"
    assert (
        "save_model(model, variant, mt, selected_feats, holdout_card)" in src
    ), "the labelled card, not the bare float metrics, must reach save_model"
