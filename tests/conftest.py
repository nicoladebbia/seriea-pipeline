"""Shared fixtures for the Serie A pipeline test suite."""

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Markers registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: integration tests (may touch disk/data)")
    config.addinivalue_line("markers", "slow: tests that take >30 seconds")


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def features_df():
    """Load a session-scoped sample of features.parquet (first 300 rows).

    Skips the entire test if the file does not exist on disk.
    """
    from config.settings import DATA_DIR

    path = DATA_DIR / "features" / "features.parquet"
    if not path.exists():
        pytest.skip("features.parquet not found -- data not built yet")

    df = pd.read_parquet(path)
    df = df.sort_values("match_date").reset_index(drop=True)
    # Return first 300 rows for speed; enough for meaningful tests
    return df.head(300).copy()


@pytest.fixture(scope="session")
def full_features_df():
    """Load the full features.parquet (session-scoped, read once).

    Skips if file does not exist.
    """
    from config.settings import DATA_DIR

    path = DATA_DIR / "features" / "features.parquet"
    if not path.exists():
        pytest.skip("features.parquet not found -- data not built yet")

    df = pd.read_parquet(path)
    return df.sort_values("match_date").reset_index(drop=True)


@pytest.fixture
def sample_matches_df():
    """Create a minimal synthetic matches DataFrame for unit-level tests."""
    np.random.seed(42)
    n = 50
    teams = ["Inter", "Milan", "Juventus", "Roma", "Napoli",
             "Lazio", "Atalanta", "Fiorentina", "Torino", "Bologna"]
    home_teams = np.random.choice(teams, size=n)
    away_teams = np.random.choice(teams, size=n)
    # Ensure no team plays itself
    for i in range(n):
        while away_teams[i] == home_teams[i]:
            away_teams[i] = np.random.choice(teams)

    df = pd.DataFrame({
        "match_id": [f"match_{i}" for i in range(n)],
        "season": ["2024-2025"] * n,
        "match_date": pd.date_range("2024-08-18", periods=n, freq="7D"),
        "home_team": home_teams,
        "away_team": away_teams,
        "home_score": np.random.randint(0, 4, size=n),
        "away_score": np.random.randint(0, 4, size=n),
    })
    results = []
    for _, row in df.iterrows():
        if row["home_score"] > row["away_score"]:
            results.append("H")
        elif row["home_score"] == row["away_score"]:
            results.append("D")
        else:
            results.append("A")
    df["result"] = results
    return df


@pytest.fixture
def sample_prediction():
    """A single prediction dict matching the pipeline's output format."""
    return {
        "match": "Inter vs Milan",
        "home_team": "Inter",
        "away_team": "Milan",
        "date": "2026-02-09",
        "predicted_outcome": "Home Win",
        "probabilities": {
            "home": 0.48,
            "draw": 0.27,
            "away": 0.25,
        },
        "confidence": "HIGH",
        "factors": {
            "home_factors": ["home_favorite", "big_stadium"],
            "away_factors": [],
            "neutral_factors": ["derby"],
        },
    }


@pytest.fixture
def sample_odds():
    """Sample odds dict for a single match (multi-bookmaker)."""
    return {
        "Inter vs Milan": {
            "bookmakers": [
                {
                    "bookmaker": "Pinnacle",
                    "home": 2.10, "draw": 3.30, "away": 3.80,
                },
                {
                    "bookmaker": "Bet365",
                    "home": 2.15, "draw": 3.40, "away": 3.90,
                },
                {
                    "bookmaker": "Unibet",
                    "home": 2.12, "draw": 3.35, "away": 3.85,
                },
            ],
        },
    }


@pytest.fixture(autouse=True)
def _promotion_state_isolated(tmp_path, monkeypatch):
    """market_promotion.json is rewritten after every settle; a test that
    settles picks must never write the real one (2026-07-13 ledger drift was
    exactly this class of leak, on bankroll.json)."""
    from scripts.betting import market_promotion as _mp
    monkeypatch.setattr(_mp, "STATE_PATH", tmp_path / "market_promotion.json")
