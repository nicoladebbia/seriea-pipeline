"""Tests for the player floor engine (scripts.betting.player_predictions).

Synthetic frames only — no parquet reads, so the suite stays fast and runs
without the 100k-row data file. Covers the leak-freedom invariant, the
count-tail math (Poisson + negative binomial) against scipy references,
dispersion fitting, and the market-prob contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scripts.betting.player_predictions import (
    _RATE_COLS,
    TARGETS,
    _count_tail,
    _player_market_probs,
    build_player_features,
    compute_position_base_rates,
    fit_dispersion,
    predict_player_markets,
)

BASE_DIR = Path(__file__).resolve().parent.parent


def _synthetic_pms(rows: list[dict]) -> pd.DataFrame:
    """Minimal frame with every column build_player_features touches."""
    defaults = {c: 0.0 for c in _RATE_COLS}
    out = []
    for i, r in enumerate(rows):
        row = {
            "player_id": 1,
            "player_name": "Test Player",
            "team": "Inter",
            "opponent": "Milan",
            "position": "M",
            "is_home": True,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * i),
            "minutes": 90.0,
            **defaults,
        }
        row.update(r)
        out.append(row)
    return pd.DataFrame(out)


# ── leak-freedom ─────────────────────────────────────────────────────────────

class TestLeakFreedom:
    def test_prior_excludes_current_match(self):
        """The prior on row i must be the mean of rows < i only."""
        pms = _synthetic_pms([
            {"total_shots": 2.0},
            {"total_shots": 4.0},
            {"total_shots": 0.0},
            {"total_shots": 6.0},
        ])
        feat = build_player_features(pms)
        pri = feat["total_shots_p90_prior"].tolist()
        assert np.isnan(pri[0])                       # no history yet
        assert pri[1] == pytest.approx(2.0)           # mean of [2]
        assert pri[2] == pytest.approx(3.0)           # mean of [2, 4]
        assert pri[3] == pytest.approx(2.0)           # mean of [2, 4, 0]

    def test_sub_60_minute_matches_excluded_from_rate(self):
        pms = _synthetic_pms([
            {"total_shots": 2.0, "minutes": 90.0},
            {"total_shots": 9.0, "minutes": 30.0},    # cameo — must not enter rate
            {"total_shots": 2.0, "minutes": 90.0},
        ])
        feat = build_player_features(pms)
        assert feat["total_shots_p90_prior"].iloc[2] == pytest.approx(2.0)
        # but prior_n counts only 60+ matches; the cameo isn't one
        assert feat["prior_n"].iloc[2] == 1

    def test_passes_and_tackles_priors_built(self):
        """New rate columns ride the same leak-free construction."""
        pms = _synthetic_pms([
            {"accurate_passes": 30.0, "tackles": 2.0},
            {"accurate_passes": 50.0, "tackles": 4.0},
        ])
        feat = build_player_features(pms)
        assert feat["accurate_passes_p90_prior"].iloc[1] == pytest.approx(30.0)
        assert feat["tackles_p90_prior"].iloc[1] == pytest.approx(2.0)


# ── count-tail math ──────────────────────────────────────────────────────────

class TestCountTail:
    @pytest.mark.parametrize("lam,k", [(0.5, 1), (1.2, 2), (3.0, 1), (28.0, 30)])
    def test_poisson_matches_scipy(self, lam, k):
        ours = _count_tail(lam, k)
        ref = float(stats.poisson.sf(k - 1, lam))
        assert ours == pytest.approx(ref, abs=1e-9)

    @pytest.mark.parametrize("lam,k,r", [(28.0, 30, 8.0), (28.0, 40, 8.0), (45.0, 40, 10.0)])
    def test_nbinom_matches_scipy(self, lam, k, r):
        ours = _count_tail(lam, k, dist="nbinom", r=r)
        p = r / (r + lam)
        ref = float(stats.nbinom.sf(k - 1, r, p))
        assert ours == pytest.approx(ref, abs=1e-9)

    def test_nbinom_has_fatter_upper_tail_than_poisson(self):
        """Over-dispersion must raise P(N >= k) for k well above the mean."""
        lam, k = 28.0, 40
        assert _count_tail(lam, k, dist="nbinom", r=8.0) > _count_tail(lam, k)

    def test_zero_lambda(self):
        assert _count_tail(0.0, 1) == 0.0
        assert _count_tail(0.0, 1, dist="nbinom", r=8.0) == 0.0

    def test_nbinom_without_r_falls_back_to_poisson(self):
        lam, k = 28.0, 30
        assert _count_tail(lam, k, dist="nbinom", r=None) == pytest.approx(
            _count_tail(lam, k))


# ── dispersion fit ───────────────────────────────────────────────────────────

class TestDispersionFit:
    def test_recovers_known_r(self):
        rng = np.random.default_rng(7)
        r_true, mean = 8.0, 28.0
        p = r_true / (r_true + mean)
        rows = []
        for pid in range(120):
            counts = rng.negative_binomial(r_true, p, size=60)
            for j, c in enumerate(counts):
                rows.append({
                    "player_id": pid, "minutes": 90.0, "position": "M",
                    "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=j),
                    "accurate_passes": float(c),
                })
        pms = pd.DataFrame(rows)
        for c in _RATE_COLS:
            if c not in pms.columns:
                pms[c] = 0.0
        disp = fit_dispersion(pms)
        assert "accurate_passes" in disp
        assert 5.0 < disp["accurate_passes"] < 13.0   # MoM is noisy; loose band

    def test_poisson_like_data_yields_large_r(self):
        """Near-Poisson counts (var≈mean) must NOT get a small r."""
        rng = np.random.default_rng(3)
        rows = []
        for pid in range(80):
            counts = rng.poisson(1.0, size=60)
            for j, c in enumerate(counts):
                rows.append({
                    "player_id": pid, "minutes": 90.0, "position": "D",
                    "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=j),
                    "accurate_passes": float(c),
                })
        pms = pd.DataFrame(rows)
        for c in _RATE_COLS:
            if c not in pms.columns:
                pms[c] = 0.0
        disp = fit_dispersion(pms)
        # var-mean ~ 0 → r explodes → capped; anything >= cap floor is fine
        assert disp["accurate_passes"] >= 20.0


# ── market prob contract ─────────────────────────────────────────────────────

class TestMarketProbs:
    def _probs(self, priors=None, prior_n=20, minutes=90.0):
        base = {k: {"M": 0.5, "_overall": 0.5} for k in TARGETS}
        return _player_market_probs(
            priors or {}, minutes, "M", prior_n, base, calib=None,
            disp={"accurate_passes": 8.0},
        )

    def test_all_targets_present_and_bounded(self):
        priors = {c: 1.0 for c in _RATE_COLS}
        priors["accurate_passes"] = 45.0
        mk = self._probs(priors)
        assert set(mk) == set(TARGETS)
        for v in mk.values():
            assert 0.0 <= v["prob"] <= 0.99

    def test_passes_monotone_in_line(self):
        priors = {c: 0.0 for c in _RATE_COLS}
        priors["accurate_passes"] = 35.0
        mk = self._probs(priors)
        assert mk["passes_o195"]["prob"] > mk["passes_o295"]["prob"] > mk["passes_o395"]["prob"]

    def test_tackles_monotone_in_line(self):
        priors = {c: 0.0 for c in _RATE_COLS}
        priors["tackles"] = 2.5
        mk = self._probs(priors)
        assert mk["tackles_o05"]["prob"] > mk["tackles_o15"]["prob"] > mk["tackles_o25"]["prob"]

    def test_position_fallback_when_thin_history(self):
        mk = self._probs(priors={}, prior_n=2)
        assert all(v["source"] == "position_base" for v in mk.values())

    def test_poss_factor_scales_passes_only(self):
        """Possession factor must move passes probs and leave tackles alone."""
        pms = _synthetic_pms(
            [{"accurate_passes": 30.0, "tackles": 2.0, "total_shots": 1.0}] * 12)
        feat = build_player_features(pms)
        kw = dict(
            player_name="Test Player", player_id=1, team="Inter", opponent="Milan",
            position="M", is_home=True, pms=feat,
            base_rates=compute_position_base_rates(feat), calib={}, disp={},
        )
        lo = predict_player_markets(**kw, poss_factor=0.8)
        hi = predict_player_markets(**kw, poss_factor=1.25)
        assert hi["markets"]["passes_o295"]["prob"] > lo["markets"]["passes_o295"]["prob"]
        assert hi["markets"]["tackles_o15"]["prob"] == lo["markets"]["tackles_o15"]["prob"]

    def test_base_rates_cover_new_targets(self):
        pms = _synthetic_pms([
            {"accurate_passes": 30.0, "tackles": 1.0},
            {"accurate_passes": 40.0, "tackles": 2.0},
        ])
        feat = build_player_features(pms)
        rates = compute_position_base_rates(feat)
        for key in ("passes_o195", "passes_o295", "passes_o395",
                    "tackles_o05", "tackles_o15", "tackles_o25"):
            assert key in rates and "_overall" in rates[key]


# ── cross-module display contract ────────────────────────────────────────────

class TestDisplayContract:
    """app.py's display/headline lists and the template MK_ORDER must only
    reference real TARGETS keys, and headline ⊆ display. Parsed from source
    text to avoid importing the Flask app in unit tests."""

    def _extract(self, text: str, name: str) -> list[str]:
        m = re.search(rf"{name}\s*=\s*\[(.*?)\]", text, re.S)
        assert m, f"{name} not found"
        return re.findall(r"[\"']([a-z0-9_]+)[\"']", m.group(1))

    def test_app_lists_are_valid_subsets(self):
        src = (BASE_DIR / "web" / "app.py").read_text()
        display = self._extract(src, "_FLOOR_DISPLAY_MARKETS")
        headline = self._extract(src, "_FLOOR_HEADLINE_MARKETS")
        assert set(display) <= set(TARGETS)
        assert set(headline) <= set(display)
        # the near-universal floors must NOT be headline-eligible
        assert {"passes_o195", "passes_o295", "tackles_o05"}.isdisjoint(headline)

    def test_template_mk_order_covers_display(self):
        src = (BASE_DIR / "web" / "templates" / "projections.html").read_text()
        app_src = (BASE_DIR / "web" / "app.py").read_text()
        mk_order = self._extract(src, "const MK_ORDER")
        display = self._extract(app_src, "_FLOOR_DISPLAY_MARKETS")
        assert set(display) <= set(mk_order), "template would silently drop markets"


# ---------------------------------------------------------------------------
# Team-name canonicalisation
#
# The engine used a hand-typed six-entry TEAM_MAP covering Serie A only. Names
# arrive from the Sofascore parquets, but every lookup below is keyed by the
# CANONICAL name the fixture side uses, so a club the literal did not know
# became a key nobody queries — "Liverpool FC" sat beside "Liverpool" as a
# separate club and Coventry had no entry at all.

def _empty_pms() -> pd.DataFrame:
    """Only the columns _get_possession's player-context merge touches."""
    return pd.DataFrame(
        {"player_id": pd.Series(dtype=int), "match_id": pd.Series(dtype=str),
         "team": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")}
    )


def _mts(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"match_id": f"m{i}", "team": t, "possession": p,
             "date": d, "period": "ALL"}
            for i, (t, p, d) in enumerate(rows)
        ]
    )


def test_aliases_collapse_into_one_club_not_two(tmp_path, monkeypatch):
    """"Liverpool FC" and "Liverpool" are the SAME club, so one prior.

    Splitting them is not a cosmetic issue: the fixture side only ever asks for
    "Liverpool", so every row filed under the alias was excluded from the prior
    it should have fed.
    """
    from scripts.betting import player_predictions as pp

    frame = _mts(
        [("Liverpool", 60.0, "2025-01-01"), ("Liverpool", 62.0, "2025-01-08"),
         ("Liverpool FC", 50.0, "2025-01-15"), ("Liverpool FC", 52.0, "2025-01-22")]
    )
    path = tmp_path / "match_team_stats_premier_league.parquet"
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(pp, "SS_DIR", tmp_path)
    monkeypatch.setattr(pp, "_POSS_CACHE", {"key": None, "team": None, "player": None})

    team_prior, _ = pp._get_possession(_empty_pms(), "premier_league")

    assert "Liverpool FC" not in team_prior, "the alias survived as its own club"
    # Mean of the three matches before the last: (60 + 62 + 50) / 3.
    assert team_prior["Liverpool"] == pytest.approx((60.0 + 62.0 + 50.0) / 3)


def test_a_promoted_club_is_not_invisible(tmp_path, monkeypatch):
    """Coventry arrives from Sofascore as "Coventry City"; the fixture side
    calls it "Coventry". The literal knew neither, so it got no prior at all."""
    from scripts.betting import player_predictions as pp

    frame = _mts(
        [("Coventry City", 44.0, "2026-08-21"), ("Coventry City", 46.0, "2026-08-28")]
    )
    path = tmp_path / "match_team_stats_premier_league.parquet"
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(pp, "SS_DIR", tmp_path)
    monkeypatch.setattr(pp, "_POSS_CACHE", {"key": None, "team": None, "player": None})

    team_prior, _ = pp._get_possession(_empty_pms(), "premier_league")

    assert "Coventry" in team_prior, "promoted club still keyed by its raw name"
    assert team_prior["Coventry"] == pytest.approx(44.0)


def test_the_serie_a_names_the_literal_handled_still_resolve():
    """The replacement must be a superset, not a swap."""
    from config.team_names import normalize_team_safe

    for raw, canonical in (
        ("Hellas Verona", "Verona"), ("ChievoVerona", "Chievo"),
        ("AC Milan", "Milan"), ("Inter Milan", "Inter"),
        ("Internazionale", "Inter"), ("Parma Calcio 1913", "Parma"),
    ):
        assert normalize_team_safe(raw) == canonical


def test_distinct_clubs_do_not_fuse():
    """The mirror risk: a fuzzy normaliser merging two REAL clubs into one key.

    That would pool two teams' possession priors — the same failure as the
    Liverpool split, in the opposite direction. Suffix-stripping is exactly
    what would fuse shared-city and shared-stem names, so those are the cases
    worth asserting.
    """
    from config.team_names import normalize_team_safe

    for a, b in (
        ("Man City", "Man United"),
        ("Manchester City", "Manchester United"),
        ("Sheffield United", "Sheffield Wednesday"),
        ("Hellas Verona", "Chievo"),
        ("Inter", "Milan"),
        ("AC Milan", "Inter Milan"),
        ("Nottingham Forest", "Norwich"),
    ):
        assert normalize_team_safe(a) != normalize_team_safe(b), f"{a} fused with {b}"


def test_canon_maps_every_row_the_same_way_as_the_normaliser():
    """The unique-value lookup must not change behaviour, only cost."""
    from config.team_names import normalize_team_safe
    from scripts.betting.player_predictions import _canon

    raw = pd.Series(["Liverpool FC", "Liverpool", "AC Milan", None, "Coventry City",
                     "Liverpool FC", "Hellas Verona"])
    got = _canon(raw)
    assert got.isna().tolist() == raw.isna().tolist()
    for r, g in zip(raw.dropna(), got.dropna()):
        assert g == normalize_team_safe(r)
