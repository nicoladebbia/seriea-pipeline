"""Staking has ONE definition (config.settings) — reliability phase 9, 2026-09-06.

Before this, twelve files carried their own kelly literal. Production staked at
0.15 while five modules claimed 0.10 "synced with production", and the parlay
generator (0.05) disagreed with the advisor's parlay card (0.10). None of it
errored: a stale literal is a number that looks like a decision.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

from config import settings
from config.settings import KELLY_FRACTION, LEAGUE_KELLY_FRACTIONS, MAX_STAKE_PCT, PARLAY_KELLY_FRACTION

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("scripts", "web", "features", "ml", "config", "scraper")
# A lowercase kelly literal anywhere but the definition is the drift this test exists to stop.
LITERAL = re.compile(r"^\s*(?:self\.)?(?:parlay_)?kelly_fraction\s*(?::\s*float\s*)?=\s*0\.\d+", re.M)


def test_the_production_engine_reads_the_one_definition():
    from scripts.betting import betting_unified as bu
    cfg = bu.BettingConfig()
    assert cfg.kelly_fraction == KELLY_FRACTION
    assert cfg.max_stake_pct == MAX_STAKE_PCT
    # the per-league scaler divides by cfg.kelly_fraction: serie_a must equal it or every SA stake rescales
    assert bu._LEAGUE_KELLY_DEFAULTS == LEAGUE_KELLY_FRACTIONS
    assert LEAGUE_KELLY_FRACTIONS["serie_a"] == KELLY_FRACTION
    assert cfg.market_rules["O/U_Over"]["kelly_fraction"] == KELLY_FRACTION
    assert inspect.signature(bu._load_league_kelly_fraction).parameters["default"].default == KELLY_FRACTION


def test_every_module_that_claimed_to_be_synced_actually_is():
    from features.value_betting import ValueBettingPipeline, get_value_pipeline
    from scripts.analysis.backtest_unified import BacktestConfig
    from scripts.prediction.predict_unified import PredictionConfig

    assert inspect.signature(ValueBettingPipeline.__init__).parameters["kelly_fraction"].default == KELLY_FRACTION
    assert inspect.signature(get_value_pipeline).parameters["kelly_fraction"].default == KELLY_FRACTION
    assert BacktestConfig().kelly_fraction == KELLY_FRACTION
    assert PredictionConfig().kelly_fraction == KELLY_FRACTION


def test_parlays_stake_and_report_the_same_fraction():
    from scripts.betting import parlay_generator as pg
    assert pg.KELLY_FRACTION == PARLAY_KELLY_FRACTION
    src = inspect.getsource(pg)
    assert "parlay_kelly_fraction = PARLAY_KELLY_FRACTION" in src
    import web.advisor as adv
    assert "kelly_fraction = PARLAY_KELLY_FRACTION" in inspect.getsource(adv)


def test_no_kelly_literal_survives_outside_the_definition():
    """The mutation this guards: someone re-types `kelly_fraction = 0.10` in a
    module. The definition file is the only place a literal may live."""
    offenders = []
    for d in SCAN_DIRS:
        for py in (ROOT / d).rglob("*.py"):
            if py == Path(settings.__file__) or "tests" in py.parts:
                continue
            text = py.read_text(errors="replace")
            for m in LITERAL.finditer(text):
                offenders.append(f"{py.relative_to(ROOT)}: {m.group(0).strip()}")
    assert not offenders, "kelly literal outside config.settings:\n" + "\n".join(offenders)
