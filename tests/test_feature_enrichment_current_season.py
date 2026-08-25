"""The per-season enrichment steps must reach the season being played.

config.SEASONS is hand-maintained and lags the calendar. On 2026-08-25 it still
ended at "2025-2026" while get_current_season() returned "2026-2027", so every
full feature rebuild skipped the in-progress season: 14 squad-value columns and
58 odds/disagreement columns sat null for the current season while the model
carried them as live features.

These tests pin the enrichment to the seasons actually present in the frame, so
the rollover can never go missing again.
"""

import pandas as pd
import pytest

from features.build import _add_market_data

CURRENT = "2026-2027"  # deliberately NOT in config.SEASONS at the time of writing
KNOWN = "2025-2026"

VALUE_COLS = [
    "home_squad_size",
    "away_squad_size",
    "home_squad_total_value",
    "home_avg_player_value",
]


def _matches(*seasons):
    rows = []
    for s in seasons:
        rows.append(
            {
                "match_id": f"{s}-1",
                "season": s,
                "league": "serie_a",
                "home_team": "Inter",
                "away_team": "Napoli",
                "match_date": pd.Timestamp("2026-08-24"),
            }
        )
    return pd.DataFrame(rows)


def _market_values():
    rows = []
    for team, base in (("Inter", 30_000_000.0), ("Napoli", 20_000_000.0)):
        for i in range(3):
            rows.append(
                {
                    "team": team,
                    "player_name": f"{team} p{i}",
                    "market_value_eur": base + i,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def tm_dir(tmp_path, monkeypatch):
    """Redirect DATA_DIR so the test never reads or writes the real tree."""
    import config.settings as settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    d = tmp_path / "external" / "transfermarkt"
    d.mkdir(parents=True)
    return d


def _write_mv(tm_dir, season):
    _market_values().to_parquet(
        tm_dir / f"market_values_{season.replace('-', '_')}.parquet", index=False
    )


def test_the_in_progress_season_is_enriched_even_though_config_has_not_rolled_over(tm_dir):
    """The bug: a season missing from config.SEASONS was skipped entirely."""
    _write_mv(tm_dir, CURRENT)

    out = _add_market_data(_matches(CURRENT), None)

    row = out[out["season"] == CURRENT].iloc[0]
    for col in VALUE_COLS:
        assert col in out.columns, f"{col} never computed for the current season"
        assert pd.notna(row[col]), f"{col} is null for the season being played"
    assert row["home_squad_size"] == 3


def test_a_season_config_already_knows_is_still_enriched(tm_dir):
    """True positive: the mechanism fires, so a no-op cannot satisfy this suite."""
    _write_mv(tm_dir, KNOWN)

    out = _add_market_data(_matches(KNOWN), None)

    row = out[out["season"] == KNOWN].iloc[0]
    assert pd.notna(row["home_squad_size"])
    assert row["home_squad_size"] == 3


def test_both_seasons_are_enriched_in_one_pass(tm_dir):
    """A rebuild spanning the rollover must not drop the newer half."""
    _write_mv(tm_dir, KNOWN)
    _write_mv(tm_dir, CURRENT)

    out = _add_market_data(_matches(KNOWN, CURRENT), None)

    filled = out.set_index("season")["home_squad_size"]
    assert pd.notna(filled[KNOWN])
    assert pd.notna(filled[CURRENT])


def test_an_explicit_season_argument_still_only_touches_that_season(tm_dir):
    """Per-season rebuilds must stay surgical -- no cross-season contamination."""
    _write_mv(tm_dir, KNOWN)
    _write_mv(tm_dir, CURRENT)

    out = _add_market_data(_matches(KNOWN, CURRENT), KNOWN)

    filled = out.set_index("season")["home_squad_size"]
    assert pd.notna(filled[KNOWN])
    assert pd.isna(filled[CURRENT]), "an explicit season leaked into another season"


def test_a_season_with_no_market_file_is_a_harmless_no_op(tm_dir):
    """Widening the season list must not raise on a season with no data."""
    _write_mv(tm_dir, KNOWN)

    out = _add_market_data(_matches(KNOWN, CURRENT), None)

    filled = out.set_index("season")["home_squad_size"]
    assert pd.notna(filled[KNOWN])
    assert pd.isna(filled[CURRENT])
